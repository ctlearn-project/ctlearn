import numpy as np
import pytest
import torch
import keras

# Import your implementations (adjust module paths to match your project layout)
from ctlearn.core.keras.model import (
    KerasSingleCNN,
    KerasResNet,
)
from ctlearn.core.pytorch.model import (
    PyTorchSingleCNN,
    PyTorchResNet,
)


@pytest.fixture
def common_config():
    """Provides common configuration settings for both backends."""
    return {
        "tasks": ["type", "energy"],
        "input_shape_keras": (32, 32, 3),  # (H, W, C)
        "input_shape_pytorch": (3, 32, 32),  # (C, H, W)
        "kwargs": {
            "head_layers": {
                "type": [64, 2],
                "energy": [64, 1],
            },
            "head_activation_function": {
                "type": "relu",
                "energy": "relu",
            },
            "attention_mechanism": None,  # Keep attention off for baseline structural tests
        },
    }


def copy_weights_single_cnn(keras_model, pytorch_model):
    """
    Transfers weights from KerasSingleCNN to PyTorchSingleCNN layer by layer
    to ensure numerical parity tests are exact.
    """
    k_layers = [
        layer
        for layer in keras_model.layers
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense, keras.layers.BatchNormalization))
    ]
    p_layers = [
        m
        for m in pytorch_model.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))
    ]

    for k_layer, p_module in zip(k_layers, p_layers):
        weights = k_layer.get_weights()
        if isinstance(k_layer, keras.layers.Conv2D):
            # Keras Conv2D weights shape: (H, W, In, Out)
            # PyTorch Conv2d weights shape: (Out, In, H, W)
            w = np.transpose(weights[0], (3, 2, 0, 1))
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1:  # Bias
                p_module.bias.data = torch.from_numpy(weights[1]).float()

        elif isinstance(k_layer, keras.layers.Dense):
            # Keras Dense weights shape: (In, Out)
            # PyTorch Linear weights shape: (Out, In)
            w = np.transpose(weights[0], (1, 0))
            p_module.weight.data = torch.from_numpy(w).float()
            if len(weights) > 1:  # Bias
                p_module.bias.data = torch.from_numpy(weights[1]).float()

        elif isinstance(k_layer, keras.layers.BatchNormalization):
            # Keras: [gamma, beta, mean, variance]
            p_module.weight.data = torch.from_numpy(weights[0]).float()  # gamma
            p_module.bias.data = torch.from_numpy(weights[1]).float()    # beta
            p_module.running_mean.data = torch.from_numpy(weights[2]).float()
            p_module.running_var.data = torch.from_numpy(weights[3]).float()


class TestSingleCNNParity:
    """Tests verifying output shapes and output parity for SingleCNN."""

    def test_output_shapes(self, common_config):
        """Verify that both models produce identical output shapes for a multi-task scenario."""
        tasks = common_config["tasks"]
        kwargs = common_config["kwargs"]

        # 1. Instantiate Keras SingleCNN
        keras_wrapper = KerasSingleCNN(
            input_shape=common_config["input_shape_keras"],
            tasks=tasks,
            **kwargs
        )
        
        # 2. Instantiate PyTorch SingleCNN
        torch_wrapper = PyTorchSingleCNN(
            input_shape=common_config["input_shape_pytorch"],
            tasks=tasks,
            **kwargs
        )

        batch_size = 4
        
        # Dummy data creation
        x_keras = np.random.randn(batch_size, *common_config["input_shape_keras"]).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))  # (B, H, W, C) -> (B, C, H, W)

        # Forward passes
        keras_out = keras_wrapper.model(x_keras)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_out = torch_wrapper.model(x_torch)

        # Assert shape equality across tasks
        for task in tasks:
            k_shape = tuple(keras_out[task].shape)
            p_shape = tuple(torch_out[task].shape)
            assert k_shape == p_shape, f"Shape mismatch for task '{task}': Keras {k_shape} vs PyTorch {p_shape}"

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

        # Transfer weights from Keras -> PyTorch
        copy_weights_single_cnn(keras_wrapper.model, torch_wrapper.model)

        # Prepare identical input data
        np.random.seed(42)
        x_keras = np.random.randn(2, *common_config["input_shape_keras"]).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))

        # Evaluate models
        keras_preds = keras_wrapper.model(x_keras)
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
                err_msg=f"Value divergence detected in output task '{task}'",
            )


class TestResNetParity:
    """Tests verifying output shapes and structural parity for ResNet."""

    @pytest.mark.parametrize("block_type", ["basic", "bottleneck"])
    def test_resnet_output_shapes(self, common_config, block_type):
        """Ensure both Basic and Bottleneck ResNets compute matching output tensor dimensions."""
        tasks = common_config["tasks"]
        kwargs = common_config["kwargs"].copy()
        kwargs["residual_block_type"] = block_type
        kwargs["init_layer"] = {"filters": 16, "kernel_size": 3, "strides": 1}
        kwargs["init_max_pool"] = {"size": 2, "strides": 2}

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

        batch_size = 2
        x_keras = np.random.randn(batch_size, *common_config["input_shape_keras"]).astype(np.float32)
        x_torch = torch.from_numpy(np.transpose(x_keras, (0, 3, 1, 2)))

        keras_out = keras_wrapper.model(x_keras)
        torch_wrapper.model.eval()
        with torch.no_grad():
            torch_out = torch_wrapper.model(x_torch)

        for task in tasks:
            assert tuple(keras_out[task].shape) == tuple(torch_out[task].shape)