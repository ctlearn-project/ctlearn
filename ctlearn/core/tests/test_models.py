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
