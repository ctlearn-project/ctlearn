import pytest
import numpy as np
import keras
import torch
import torch.nn as nn

from ctlearn.core.keras.model import KerasResNet, KerasSingleCNN
from ctlearn.core.pytorch.model import PyTorchResNet, PyTorchSingleCNN

rng = np.random.default_rng(42)

@pytest.fixture
def common_config():
    return {
        "tasks": ["type", "energy", "cameradirection"],
        "input_shape_keras": (110, 110, 2),    # (H, W, C)
        "input_shape_pytorch": (2, 110, 110),  # (C, H, W)
        "kwargs": {
            "head_layers": {
                "type": [64, 2],
                "energy": [64, 1],
                "cameradirection": [64, 2],
            },
            "head_activation_function": {
                "type": "relu",
                "energy": "relu",
                "cameradirection": "tanh",
            },
        },
    }

@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize(
    "attention",
    [
        {"mechanism": None},
        {"mechanism": "Channel-SE", "reduction_ratio": 8},
        {"mechanism": "Spatial-SE"},
        {"mechanism": "Dual-SE", "reduction_ratio": 32},
    ],
)
def test_SingleCNN_model_structure_parity(common_config, batchnorm, attention):
    """Verify that Keras and PyTorch models have matching layer counts and weight shapes."""
    tasks = common_config["tasks"]
    kwargs = common_config["kwargs"].copy()
    kwargs["architecture"] = [
        {"filters": 32, "kernel_size": 2, "number": 1},
        {"filters": 64, "kernel_size": 3, "number": 4},
        {"filters": 128, "kernel_size": 2, "number": 2},
        {"filters": 128, "kernel_size": 3, "number": 1},
    ]
    kwargs["batchnorm"] = batchnorm
    kwargs["attention_mechanism"] = attention["mechanism"]
    if "reduction_ratio" in attention:
        kwargs["attention_reduction_ratio"] = attention["reduction_ratio"]

    for task in tasks:
        keras_wrapper = KerasSingleCNN(
            input_shape=common_config["input_shape_keras"],
            tasks=[task],
            **kwargs
        )
        torch_wrapper = PyTorchSingleCNN(
            input_shape=common_config["input_shape_pytorch"],
            tasks=[task],
            **kwargs
        )
        # Collect all the layers from the Keras-based model 
        keras_layers = [
            l.get_weights()[0].shape 
            for l in keras_wrapper.backbone_model.layers 
            if isinstance(l, (keras.layers.Conv2D, keras.layers.Dense, keras.layers.BatchNormalization))
        ]
        # Collect also the dense layers from the head
        keras_layers.extend([
                l.get_weights()[0].shape 
                for l in keras_wrapper.model.layers
                if isinstance(l, keras.layers.Dense) and task in l.name
            ]
        )
        # Collect all the layers from the PyTorch-based model 
        # Note different shapes (PyTorch format: Out, In, H, W -> Keras: H, W, In, Out)
        torch_layers = []
        for m in torch_wrapper.backbone_model.modules():
            if isinstance(m, torch.nn.Conv2d):
                # PyTorch format: (Out, In, H, W) -> Keras format: (H, W, In, Out)
                torch_layers.append(
                    (m.weight.shape[2], m.weight.shape[3], m.weight.shape[1], m.weight.shape[0])
                )
            elif isinstance(m, torch.nn.Linear):
                torch_layers.append((m.weight.shape[1], m.weight.shape[0]))
            elif isinstance(m, torch.nn.BatchNorm2d):
                # BatchNorm parameters are 1D (gamma/beta/running stats)
                torch_layers.append((m.weight.shape[0],))
        # Collect also the linear layers from the head
        internal_key = torch_wrapper.logits_head._task_mapping[task]
        p_head_module = torch_wrapper.logits_head.heads[internal_key]
        torch_layers.extend([(m.weight.shape[1], m.weight.shape[0]) for m in p_head_module if isinstance(m, torch.nn.Linear)])
        # Assert structural length and individual weight shape alignment
        assert len(keras_layers) == len(torch_layers), (
            f"Layer count mismatch: Keras has {len(keras_layers)}, PyTorch has {len(torch_layers)}"
        )
        for idx, (k_shape, p_shape) in enumerate(zip(keras_layers, torch_layers)):
            assert k_shape == p_shape, (
                f"Shape mismatch at layer {idx}: Keras shape {k_shape} vs PyTorch mapped shape {p_shape}"
            )


@pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
@pytest.mark.parametrize(
    "first_layers",
    [
        {"init_layer": None, "init_max_pool": None},
        {"init_layer": {'filters': 8, 'kernel_size': 7, 'strides': 2}, "init_max_pool": {'size': 3, 'strides': 2}},
    ],
)
@pytest.mark.parametrize(
    "attention",
    [
        {"mechanism": None},
        {"mechanism": "Channel-SE", "reduction_ratio": 8},
        {"mechanism": "Spatial-SE"},
        {"mechanism": "Dual-SE", "reduction_ratio": 32},
    ],
)
def test_ResNet_model_structure_parity(common_config, block_type, first_layers, attention):
    """Verify that Keras and PyTorch models have matching layer counts and weight shapes."""
    tasks = common_config["tasks"]
    kwargs = common_config["kwargs"].copy()
    kwargs["init_layer"] = first_layers["init_layer"]
    kwargs["init_max_pool"] = first_layers["init_max_pool"]
    kwargs["residual_block_type"] = block_type
    kwargs["architecture"] = [
        {"filters": 16, "blocks": 2},
        {"filters": 48, "blocks": 3},
        {"filters": 48, "blocks": 4},
        {"filters": 96, "blocks": 2},
    ]
    kwargs["attention_mechanism"] = attention["mechanism"]
    if "reduction_ratio" in attention:
        kwargs["attention_reduction_ratio"] = attention["reduction_ratio"]

    for task in tasks:
        keras_wrapper = KerasResNet(
            input_shape=common_config["input_shape_keras"],
            tasks=[task],
            **kwargs
        )
        torch_wrapper = PyTorchResNet(
            input_shape=common_config["input_shape_pytorch"],
            tasks=[task],
            **kwargs
        )
        # 1. Collect Keras backbone weight shapes (Conv2D / Dense)
        keras_layers = [
            l.get_weights()[0].shape
            for l in keras_wrapper.backbone_model.layers
            if hasattr(l, "weights") and l.weights and isinstance(l, (keras.layers.Conv2D, keras.layers.Dense))
        ]

        # 2. Collect PyTorch backbone weight shapes in strict sequential block order
        torch_layers = []
        # Catch initial standalone stem conv if present
        if hasattr(torch_wrapper.backbone_model, "init_layer") and isinstance(torch_wrapper.backbone_model.init_layer, nn.Conv2d):
            c = torch_wrapper.backbone_model.init_layer
            torch_layers.append((c.kernel_size[0], c.kernel_size[1], c.in_channels, c.out_channels))
        # Iterate through stages and blocks sequentially
        for module in torch_wrapper.backbone_model.modules():
            # Detect both BasicBlock and BottleneckBlock
            if hasattr(module, "conv1") and hasattr(module, "conv2"):
                # conv1
                c1 = module.conv1
                torch_layers.append((c1.kernel_size[0], c1.kernel_size[1], c1.in_channels, c1.out_channels))
                # conv2
                c2 = module.conv2
                torch_layers.append((c2.kernel_size[0], c2.kernel_size[1], c2.in_channels, c2.out_channels))
                # conv3 (Bottleneck only)
                if hasattr(module, "conv3"):
                    c3 = module.conv3
                    torch_layers.append((c3.kernel_size[0], c3.kernel_size[1], c3.in_channels, c3.out_channels))
                # Attention layer (if present)
                if hasattr(module, "attn_layer") and module.attn_layer is not None:
                    # Inspect linear/conv layers inside the SE block (e.g. Channel-SE / Dual-SE / Spatial-SE)
                    for attn_submodule in module.attn_layer.modules():
                        if isinstance(attn_submodule, nn.Linear):
                            torch_layers.append((attn_submodule.weight.shape[1], attn_submodule.weight.shape[0]))
                        elif isinstance(attn_submodule, nn.Conv2d):
                            torch_layers.append((
                                attn_submodule.kernel_size[0],
                                attn_submodule.kernel_size[1],
                                attn_submodule.in_channels,
                                attn_submodule.out_channels,
                            ))
                # shortcut projection (if present and not Identity)
                if isinstance(module.shortcut, nn.Conv2d):
                    sc = module.shortcut
                    torch_layers.append((sc.kernel_size[0], sc.kernel_size[1], sc.in_channels, sc.out_channels))
            # Catch initial stem Conv2d (if present as direct child of backbone_model)
            elif isinstance(module, nn.Conv2d) and module in torch_wrapper.backbone_model.children():
                torch_layers.insert(0, (module.kernel_size[0], module.kernel_size[1], module.in_channels, module.out_channels))

        # 3. Append task heads
        keras_layers.extend([
            l.get_weights()[0].shape
            for l in keras_wrapper.model.layers
            if isinstance(l, keras.layers.Dense) and task in l.name
        ])

        internal_key = torch_wrapper.logits_head._task_mapping[task]
        p_head_module = torch_wrapper.logits_head.heads[internal_key]
        torch_layers.extend([
            (m.weight.shape[1], m.weight.shape[0])
            for m in p_head_module
            if isinstance(m, nn.Linear)
        ])
        # Assert structural length and individual weight shape alignment
        assert len(keras_layers) == len(torch_layers), (
            f"Layer count mismatch: Keras has {len(keras_layers)}, PyTorch has {len(torch_layers)}"
        )
        for idx, (k_shape, p_shape) in enumerate(zip(keras_layers, torch_layers)):
            assert k_shape == p_shape, (
                f"Shape mismatch at layer {idx}: Keras shape {k_shape} vs PyTorch mapped shape {p_shape}"
            )


'''
TODO: Do this correctly with more time
def _copy_head_weights(keras_wrapper, torch_wrapper, tasks):
    """Shared helper to copy Dense -> Linear head weights for mapped tasks."""
    for task in tasks:
        k_dense_layers = [
            l for l in keras_wrapper.model.layers
            if isinstance(l, keras.layers.Dense) and task in l.name
        ]

        internal_key = torch_wrapper.logits_head._task_mapping[task]
        p_head_module = torch_wrapper.logits_head.heads[internal_key]
        p_linear_modules = [m for m in p_head_module if isinstance(m, torch.nn.Linear)]

        for k_layer, p_module in zip(k_dense_layers, p_linear_modules):
            weights = k_layer.get_weights()
            if not weights:
                continue
            w = np.transpose(weights[0], (1, 0))  # Keras (In, Out) -> PyTorch (Out, In)
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()

def _copy_weights_single_cnn(keras_wrapper, torch_wrapper, tasks):
    """Transfers weights between KerasSingleCNN and PyTorchSingleCNN sequentially."""
    k_backbone = keras_wrapper.backbone_model
    p_backbone = torch_wrapper.backbone_model

    # 1. Include Dense / Linear layers alongside Conv and BN
    def extract_keras_layers(model):
        found = []
        for l in model.layers:
            # Recurse into nested sub-models / custom layers if present
            if hasattr(l, "layers") and not isinstance(l, (keras.layers.Conv2D, keras.layers.Dense, keras.layers.BatchNormalization)):
                found.extend(extract_keras_layers(l))
            elif isinstance(l, (keras.layers.Conv2D, keras.layers.Dense, keras.layers.BatchNormalization)) and len(l.weights) > 0:
                found.append(l)
        return found

    k_layers = extract_keras_layers(k_backbone)
    
    # 2. Match corresponding PyTorch parameter-bearing modules
    p_modules = [
        m for m in p_backbone.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))
    ]

    # Guard against architecture/sequence mismatch
    if len(k_layers) != len(p_modules):
        raise ValueError(
            f"Layer count mismatch! Keras found {len(k_layers)} parameter layers, "
            f"PyTorch found {len(p_modules)}. Check if Linear/Dense or SE modules are missing."
        )

    for k_layer, p_module in zip(k_layers, p_modules):
        weights = k_layer.get_weights()
        if not weights:
            continue
        
        # --- Handle Conv2D / Conv2d ---
        if isinstance(k_layer, keras.layers.Conv2D):
            assert isinstance(p_module, torch.nn.Conv2d), f"Type mismatch: {type(k_layer)} vs {type(p_module)}"
            # Keras Conv2D: (H, W, In, Out) -> PyTorch Conv2d: (Out, In, H, W)
            w = np.transpose(weights[0], (3, 2, 0, 1))
            
            assert p_module.weight.shape == w.shape, f"Shape mismatch in Conv2D: Keras transposed {w.shape} vs PyTorch {p_module.weight.shape}"
            p_module.weight.data = torch.from_numpy(w).float()
            
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()
            elif p_module.bias is not None:
                p_module.bias.data.zero_()

        # --- Handle Dense / Linear (Crucial for Channel-SE & Dual-SE!) ---
        elif isinstance(k_layer, keras.layers.Dense):
            assert isinstance(p_module, torch.nn.Linear), f"Type mismatch: {type(k_layer)} vs {type(p_module)}"
            # Keras Dense: (In, Out) -> PyTorch Linear: (Out, In)
            w = np.transpose(weights[0], (1, 0))
            
            assert p_module.weight.shape == w.shape, f"Shape mismatch in Dense: Keras transposed {w.shape} vs PyTorch {p_module.weight.shape}"
            p_module.weight.data = torch.from_numpy(w).float()
            
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()
            elif p_module.bias is not None:
                p_module.bias.data.zero_()

        # --- Handle Batch Normalization ---
        elif isinstance(k_layer, keras.layers.BatchNormalization):
            assert isinstance(p_module, torch.nn.BatchNorm2d), f"Type mismatch: {type(k_layer)} vs {type(p_module)}"
            # Keras BN weights order: [gamma (scale), beta (bias), moving_mean, moving_variance]
            p_module.weight.data = torch.from_numpy(weights[0]).float()
            p_module.bias.data = torch.from_numpy(weights[1]).float()
            p_module.running_mean.data = torch.from_numpy(weights[2]).float()
            p_module.running_var.data = torch.from_numpy(weights[3]).float()

    _copy_head_weights(keras_wrapper, torch_wrapper, tasks)

def _copy_weights_resnet(keras_wrapper, torch_wrapper, tasks):
    """Transfers weights between KerasResNet and PyTorchResNet block by block,

    including BatchNorm, Dense (SE), and Conv layers.
    """
    k_backbone = keras_wrapper.backbone_model
    p_backbone = torch_wrapper.backbone_model

    # 1. Transfer Initial Layers (Conv + BN)
    k_init_layers = [
        l for l in k_backbone.layers 
        if isinstance(l, (keras.layers.Conv2D, keras.layers.BatchNormalization))
        and not re.search(r"conv\d+_block\d+", l.name)
    ]
    p_init_modules = [
        m for m in p_backbone.children()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))
    ]

    for k_l, p_m in zip(k_init_layers, p_init_modules):
        _transfer_single_layer_weights(k_l, p_m)

    # 2. Collect PyTorch Residual Blocks
    p_res_blocks = [
        m for m in p_backbone.modules()
        if isinstance(m, (BasicBlock, BottleneckBlock))
    ]

    # 3. Group Keras layers by block key
    block_pattern = re.compile(r"(conv\d+_block\d+)")
    keras_blocks_dict = {}
    for layer in k_backbone.layers:
        match = block_pattern.search(layer.name)
        if match:
            block_key = match.group(1)
            keras_blocks_dict.setdefault(block_key, []).append(layer)

    def _sort_key(key_str):
        numbers = re.findall(r"\d+", key_str)
        return int(numbers[0]), int(numbers[1])

    sorted_keras_keys = sorted(keras_blocks_dict.keys(), key=_sort_key)

    # 4. Sequential Intra-Block Transfer (Handles Conv, BN, and Dense / SE layers)
    for block_key, p_block in zip(sorted_keras_keys, p_res_blocks):
        k_block_layers = [
            l for l in keras_blocks_dict[block_key] 
            if len(l.weights) > 0
        ]
        p_block_modules = [
            m for m in p_block.modules() 
            if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear))
        ]

        if len(k_block_layers) != len(p_block_modules):
            raise ValueError(
                f"Block {block_key} mismatch! Keras has {len(k_block_layers)} weighted layers, "
                f"PyTorch block has {len(p_block_modules)} modules."
            )

        for k_layer, p_module in zip(k_block_layers, p_block_modules):
            _transfer_single_layer_weights(k_layer, p_module)

    # 5. Transfer Task Head Weights
    _copy_head_weights(keras_wrapper, torch_wrapper, tasks)


def _transfer_single_layer_weights(k_layer, p_module):
    """Helper to copy parameter weights from a Keras layer to a PyTorch module."""
    weights = k_layer.get_weights()
    if not weights:
        return

    if isinstance(k_layer, keras.layers.Conv2D):
        assert isinstance(p_module, torch.nn.Conv2d)
        w = np.transpose(weights[0], (3, 2, 0, 1))
        p_module.weight.data = torch.from_numpy(w).float()
        if len(weights) > 1 and p_module.bias is not None:
            p_module.bias.data = torch.from_numpy(weights[1]).float()

    elif isinstance(k_layer, keras.layers.Dense):
        if isinstance(p_module, torch.nn.Linear):
            # Dense -> Linear
            w = np.transpose(weights[0], (1, 0))  # (in_features, out_features) -> (out_features, in_features)
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()
                
        elif isinstance(p_module, torch.nn.Conv2d):
            # Dense -> 1x1 Conv2d (SE Block representation)
            # Keras Dense weight shape: (in_features, out_features)
            # PyTorch Conv2d weight shape: (out_channels, in_channels, 1, 1)
            w = np.transpose(weights[0], (1, 0))[:, :, None, None]
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()
        else:
            raise TypeError(f"Unsupported PyTorch module type {type(p_module)} for Keras Dense layer.")

    elif isinstance(k_layer, keras.layers.BatchNormalization):
        assert isinstance(p_module, torch.nn.BatchNorm2d)
        p_module.weight.data = torch.from_numpy(weights[0]).float()      # gamma
        p_module.bias.data = torch.from_numpy(weights[1]).float()        # beta
        p_module.running_mean.data = torch.from_numpy(weights[2]).float() # mean
        p_module.running_var.data = torch.from_numpy(weights[3]).float()  # variance


@pytest.mark.parametrize("batchnorm", [True, False])
@pytest.mark.parametrize(
    "attention",
    [
        {"mechanism": None},
        {"mechanism": "Channel-SE", "reduction_ratio": 8},
        {"mechanism": "Spatial-SE"},
        {"mechanism": "Dual-SE", "reduction_ratio": 32},
    ],
)
def test_numerical_outputs_with_aligned_weights(common_config, batchnorm, attention):
    """Verify that models yield similar predictions once weights are copied."""
    tasks = common_config["tasks"]
    kwargs = common_config["kwargs"].copy()
    kwargs["architecture"] = [
        {"filters": 32, "kernel_size": 2, "number": 1},
        {"filters": 64, "kernel_size": 3, "number": 4},
        {"filters": 128, "kernel_size": 2, "number": 2},
        {"filters": 128, "kernel_size": 3, "number": 1},
    ]
    kwargs["batchnorm"] = batchnorm
    kwargs["attention_mechanism"] = attention["mechanism"]
    if "reduction_ratio" in attention:
        kwargs["attention_reduction_ratio"] = attention["reduction_ratio"]

    for task in tasks:
        keras_wrapper = KerasSingleCNN(
            input_shape=common_config["input_shape_keras"],
            tasks=[task],
            **kwargs
        )
        torch_wrapper = PyTorchSingleCNN(
            input_shape=common_config["input_shape_pytorch"],
            tasks=[task],
            **kwargs
        )
        _copy_weights_single_cnn(keras_wrapper, torch_wrapper, [task])

        # Prepare identical input data
        x_keras = rng.standard_normal(size=(2, *common_config["input_shape_keras"])).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))
        # Evaluate models in eval mode
        keras_preds = keras_wrapper.model(x_keras, training=False)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_preds = torch_wrapper.model(x_torch)

        # Extract predicted value array for the task
        if isinstance(keras_preds, dict):
            k_tensor = keras_preds[task]
        else:
            k_tensor = keras_preds

        k_val = k_tensor.numpy() if hasattr(k_tensor, "numpy") else np.array(k_tensor)

        # Handle PyTorch prediction output
        if isinstance(torch_preds, dict):
            p_val = torch_preds[task].cpu().numpy()
        else:
            p_val = torch_preds.cpu().numpy()

        # Compare outputs within tight numerical tolerance
        np.testing.assert_allclose(
            k_val,
            p_val,
            atol=0.025,
            err_msg=f"Value divergence detected in task '{task}'",
        )


@pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
@pytest.mark.parametrize(
    "first_layers",
    [
        {"init_layer": None, "init_max_pool": None},
        {"init_layer": {'filters': 8, 'kernel_size': 7, 'strides': 2}, "init_max_pool": {'size': 3, 'strides': 2}},
    ],
)
@pytest.mark.parametrize(
    "attention",
    [
        {"mechanism": None},
        {"mechanism": "Channel-SE", "reduction_ratio": 8},
        {"mechanism": "Spatial-SE"},
        {"mechanism": "Dual-SE", "reduction_ratio": 32},
    ],
)
def test_ResNet_model_structure_parity(common_config, block_type, first_layers, attention):
    """Verify that Keras and PyTorch models have matching layer counts and weight shapes."""
    tasks = common_config["tasks"]
    kwargs = common_config["kwargs"].copy()
    kwargs["init_layer"] = first_layers["init_layer"]
    kwargs["init_max_pool"] = first_layers["init_max_pool"]
    kwargs["residual_block_type"] = block_type
    kwargs["architecture"] = [
        {"filters": 16, "blocks": 2},
        {"filters": 48, "blocks": 3},
        {"filters": 48, "blocks": 4},
        {"filters": 96, "blocks": 2},
    ]
    kwargs["attention_mechanism"] = attention["mechanism"]
    if "reduction_ratio" in attention:
        kwargs["attention_reduction_ratio"] = attention["reduction_ratio"]
    absolute_tolerance = {
            "basic": 0.1,
            "bottleneck": 0.05,
        }

    for task in tasks:
        keras_wrapper = KerasResNet(
            input_shape=common_config["input_shape_keras"],
            tasks=[task],
            **kwargs
        )
        torch_wrapper = PyTorchResNet(
            input_shape=common_config["input_shape_pytorch"],
            tasks=[task],
            **kwargs
        )

        # Copy weights for ResNet
        _copy_weights_resnet(keras_wrapper, torch_wrapper, [task])

        x_keras = rng.standard_normal(size=(2, *common_config["input_shape_keras"])).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))

        keras_preds = keras_wrapper.model(x_keras, training=False)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_preds = torch_wrapper.model(x_torch)

        k_val = keras_preds[task].numpy() if hasattr(keras_preds[task], "numpy") else np.array(keras_preds[task])
        p_val = torch_preds[task].cpu().numpy()

        np.testing.assert_allclose(
            k_val,
            p_val,
            atol=absolute_tolerance[block_type],
            err_msg=f"Value divergence detected in ResNet ({block_type}) for task '{task}'",
        )
'''