import re
import keras
import numpy as np
import pytest
import torch
import torch.nn as nn

from ctlearn.core.keras.model import KerasResNet, KerasSingleCNN
from ctlearn.core.pytorch.model import (
    BasicBlock,
    BottleneckBlock,
    PyTorchResNet,
    PyTorchSingleCNN,
)

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

    def extract_keras_layers(model_or_layer):
        layers = []
        if hasattr(model_or_layer, "layers"):
            for l in model_or_layer.layers:
                layers.extend(extract_keras_layers(l))
        elif isinstance(model_or_layer, (keras.layers.Conv2D, keras.layers.BatchNormalization)) and len(model_or_layer.weights) > 0:
            layers.append(model_or_layer)
        return layers

    k_layers = extract_keras_layers(k_backbone)
    p_modules = [
        m for m in p_backbone.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))
    ]

    for k_layer, p_module in zip(k_layers, p_modules):
        weights = k_layer.get_weights()
        if isinstance(k_layer, keras.layers.Conv2D):
            w = np.transpose(weights[0], (3, 2, 0, 1))
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1 and p_module.bias is not None:
                p_module.bias.data = torch.from_numpy(weights[1]).float()
        elif isinstance(k_layer, keras.layers.BatchNormalization):
            p_module.weight.data = torch.from_numpy(weights[0]).float()
            p_module.bias.data = torch.from_numpy(weights[1]).float()
            p_module.running_mean.data = torch.from_numpy(weights[2]).float()
            p_module.running_var.data = torch.from_numpy(weights[3]).float()

    _copy_head_weights(keras_wrapper, torch_wrapper, tasks)


def _copy_weights_resnet(keras_wrapper, torch_wrapper, tasks):
    """Transfers weights between KerasResNet and PyTorchResNet block by block."""
    k_backbone = keras_wrapper.backbone_model
    p_backbone = torch_wrapper.backbone_model

    # 1. Transfer Initial Conv Layer only
    k_init_conv = [l for l in k_backbone.layers if isinstance(l, keras.layers.Conv2D)][0]
    p_init_conv = [m for m in p_backbone.modules() if isinstance(m, torch.nn.Conv2d)][0]

    w_init = np.transpose(k_init_conv.get_weights()[0], (3, 2, 0, 1))
    p_init_conv.weight.data = torch.from_numpy(w_init).float()
    if len(k_init_conv.get_weights()) > 1 and p_init_conv.bias is not None:
        p_init_conv.bias.data = torch.from_numpy(k_init_conv.get_weights()[1]).float()

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

    # 4. Transfer Block Weights accurately (Conv layers only)
    for block_key, p_block in zip(sorted_keras_keys, p_res_blocks):
        k_block_layers = keras_blocks_dict[block_key]
        for k_layer in k_block_layers:
            weights = k_layer.get_weights()
            if not weights:
                continue

            if isinstance(k_layer, keras.layers.Conv2D):
                w = np.transpose(weights[0], (3, 2, 0, 1))
                target_module = None
                if "_0_conv" in k_layer.name:
                    target_module = getattr(p_block, "shortcut", None)
                elif "_1_conv" in k_layer.name:
                    target_module = getattr(p_block, "conv1", None)
                elif "_2_conv" in k_layer.name:
                    target_module = getattr(p_block, "conv2", None)
                elif "_3_conv" in k_layer.name:
                    target_module = getattr(p_block, "conv3", None)

                if target_module is not None and not isinstance(target_module, torch.nn.Identity):
                    target_module.weight.data = torch.from_numpy(w).float()
                    if len(weights) > 1 and target_module.bias is not None:
                        target_module.bias.data = torch.from_numpy(weights[1]).float()

    # 5. Transfer Task Head Weights
    _copy_head_weights(keras_wrapper, torch_wrapper, tasks)

class TestSingleCNNParity:
    """Tests verifying output shapes and output parity for SingleCNN."""
    

    def test_numerical_outputs_with_aligned_weights(self, common_config):
        """Verify that models yield identical numerical predictions once weights are copied."""
        tasks = common_config["tasks"]
        kwargs = common_config["kwargs"]

        keras_wrapper = KerasSingleCNN(
            input_shape=common_config["input_shape_keras"],
            tasks=tasks,
            **kwargs
        )
        torch_wrapper = PyTorchSingleCNN(
            input_shape=common_config["input_shape_pytorch"],
            tasks=tasks,
            **kwargs
        )

        _copy_weights_single_cnn(keras_wrapper, torch_wrapper, tasks)

        # Prepare identical input data
        x_keras = rng.standard_normal(size=(2, *common_config["input_shape_keras"])).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))

        # Evaluate models in eval mode
        keras_preds = keras_wrapper.model(x_keras, training=False)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_preds = torch_wrapper.model(x_torch)

        # Compare outputs within tight numerical tolerance
        for task in tasks:
            k_val = keras_preds[task].numpy() if hasattr(keras_preds[task], "numpy") else np.array(keras_preds[task])
            p_val = torch_preds[task].cpu().numpy()
            np.testing.assert_allclose(
                k_val,
                p_val,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Value divergence detected in task '{task}'",
            )


class TestResNetParity:
    """Tests verifying output shapes and structural parity for ResNet."""

    @pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
    def test_model_layer_structure_parity(self, common_config, block_type):
        """Verify that Keras and PyTorch models have matching layer counts and weight shapes."""
        tasks = common_config["tasks"]
        kwargs = common_config["kwargs"].copy()
        kwargs["residual_block_type"] = block_type
 
        keras_wrapper = KerasResNet(
            input_shape=common_config["input_shape_keras"],
            tasks=tasks,
            **kwargs
        )
        torch_wrapper = PyTorchResNet(
            input_shape=common_config["input_shape_pytorch"],
            tasks=tasks,
            **kwargs
        )

        # 1. Collect Keras Conv2D weights shapes
        keras_layers = [
            l.get_weights()[0].shape 
            for l in keras_wrapper.backbone_model.layers 
            if isinstance(l, keras.layers.Conv2D)
        ]

        # 2. Collect PyTorch Conv2d weights shapes (PyTorch format: Out, In, H, W -> Keras: H, W, In, Out)
        torch_layers = [
            (m.weight.shape[2], m.weight.shape[3], m.weight.shape[1], m.weight.shape[0])
            for m in torch_wrapper.backbone_model.modules() 
            if isinstance(m, torch.nn.Conv2d)
        ]

        # 3. Assert structural weight shape alignment
        for idx, (k_shape, p_shape) in enumerate(zip(keras_layers, torch_layers)):
            assert k_shape == p_shape, (
                f"Shape mismatch at Conv layer {idx}: Keras shape {k_shape} vs PyTorch mapped shape {p_shape}"
            )


    @pytest.mark.parametrize("block_type", ["bottleneck"])
    def test_resnet_numerical_outputs_with_aligned_weights(self, common_config, block_type):
        """Verify numerical output parity for both Basic and Bottleneck ResNets after weight copying."""
        tasks = common_config["tasks"]
        kwargs = common_config["kwargs"].copy()
        kwargs["residual_block_type"] = block_type
        #kwargs["init_layer"] = {"filters": 16, "kernel_size": 3, "strides": 1}
        #kwargs["init_max_pool"] = {"size": 2, "strides": 2}
        absolute_tolerance = {
            "basic": 0.1,
            "bottleneck": 0.05,
        }

        keras_wrapper = KerasResNet(
            input_shape=common_config["input_shape_keras"],
            tasks=tasks,
            **kwargs
        )
        torch_wrapper = PyTorchResNet(
            input_shape=common_config["input_shape_pytorch"],
            tasks=tasks,
            **kwargs
        )

        # Copy weights for ResNet
        _copy_weights_resnet(keras_wrapper, torch_wrapper, tasks)

        x_keras = rng.standard_normal(size=(2, *common_config["input_shape_keras"])).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))

        keras_preds = keras_wrapper.model(x_keras, training=False)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_preds = torch_wrapper.model(x_torch)

        for task in tasks:
            k_val = keras_preds[task].numpy() if hasattr(keras_preds[task], "numpy") else np.array(keras_preds[task])
            p_val = torch_preds[task].cpu().numpy()

            np.testing.assert_allclose(
                k_val,
                p_val,
                atol=absolute_tolerance[block_type],
                err_msg=f"Value divergence detected in ResNet ({block_type}) for task '{task}'",
            )
'''