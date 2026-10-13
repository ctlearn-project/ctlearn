import numpy as np
import pytest
import pytorch_hexagdly as phgly
import torch
import torch.nn as nn

from ctlearn.core.pytorch.model import PyTorchSingleCNN, PyTorchResNet
from ctlearn.tools.utils import (
    get_lst1_subarray_description,
    validate_conv_backend,
    model_conv_backend,
)

HEX_LAYERS = (phgly.Conv2d, phgly.MaxPool2d, phgly.AvgPool2d)
SQUARE_LAYERS = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)


def _lst1_input_shape(mapper_name, n_channels=2):
    from dl1_data_handler.image_mapper import ImageMapper

    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    mapper = ImageMapper.from_name(mapper_name, geometry=geometry, subarray=subarray)
    # PyTorch models take (C, H, W)
    return mapper, (n_channels, mapper.image_shape, mapper.image_shape)


def _predict(model, input_shape, seed):
    batch = torch.from_numpy(
        np.random.default_rng(seed).uniform(size=(2, *input_shape)).astype(np.float32)
    )
    model.model.eval()
    with torch.no_grad():
        logits, _ = model.model(batch)
    return logits


def _layer_types(model):
    return [type(m) for m in model.backbone_model.modules()]


class TestSingleCNNConvBackend:
    """PyTorchSingleCNN's conv_backend trait."""

    def test_hexagdly_backend_builds_and_predicts_single_task(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[
                {"filters": 4, "kernel_size": 1, "number": 1},
                {"filters": 8, "kernel_size": 1, "number": 1},
            ],
            attention_mechanism=None,
        )
        types = _layer_types(model)
        assert phgly.Conv2d in types and phgly.MaxPool2d in types
        assert not any(t in SQUARE_LAYERS for t in types)
        assert _predict(model, input_shape, seed=0).shape == (2, 2)

    def test_hexagdly_backend_multi_task_with_batchnorm_and_bottleneck(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type", "energy"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            batchnorm=True,
            bottleneck_filters=6,
            attention_mechanism=None,
            head_layers={"type": [8, 2], "energy": [8, 1]},
        )
        outputs = _predict(model, input_shape, seed=1)
        assert outputs["type"].shape == (2, 2)
        assert outputs["energy"].shape == (2, 1)
        # The bottleneck uses the backend's conv class, as on the Keras side.
        convs = [m for m in model.backbone_model.modules() if isinstance(m, phgly.Conv2d)]
        assert convs[-1].out_channels == 6

    def test_hexagdly_backend_supports_average_pooling(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            pooling_type="average",
            attention_mechanism=None,
        )
        assert phgly.AvgPool2d in _layer_types(model)
        assert _predict(model, input_shape, seed=6).shape == (2, 2)

    def test_square_backend_still_builds_and_predicts(self):
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            attention_mechanism=None,
        )
        assert not any(t in HEX_LAYERS for t in _layer_types(model))
        assert _predict(model, input_shape, seed=2).shape == (2, 2)


class TestResNetConvBackend:
    """PyTorchResNet's conv_backend trait: both residual_block_type variants."""

    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_builds_and_predicts(self, residual_block_type):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}, {"filters": 8, "blocks": 1}],
            attention_mechanism=None,
        )
        # Every conv and pool of the stem and the residual blocks, shortcuts
        # included, is a hex layer.
        types = _layer_types(model)
        assert phgly.Conv2d in types
        assert not any(t in SQUARE_LAYERS for t in types)
        assert _predict(model, input_shape, seed=3).shape == (2, 2)

    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_with_attention(self, residual_block_type):
        """Attention on the hex path. The ratio stays below the tiny filter
        count, or the squeeze-excite bottleneck floors to zero units."""
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}],
            attention_reduction_ratio=2,
        )
        assert _predict(model, input_shape, seed=4).shape == (2, 2)

    def test_hexagdly_backend_backpropagates(self):
        """A backward pass through the stride-2 residual additions reaches
        every hex conv's weights."""
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "blocks": 1}, {"filters": 8, "blocks": 1}],
            attention_mechanism=None,
        )
        logits, _ = model.model(torch.rand(2, *input_shape))
        nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        convs = [m for m in model.model.modules() if isinstance(m, phgly.Conv2d)]
        assert convs
        for conv in convs:
            grads = [p.grad for p in conv.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert any(g.abs().sum() > 0 for g in grads)

    def test_square_backend_still_builds_and_predicts(self):
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )
        assert not any(t in HEX_LAYERS for t in _layer_types(model))
        assert _predict(model, input_shape, seed=5).shape == (2, 2)


class TestModelConvBackend:
    """ctlearn.tools.utils.model_conv_backend on PyTorch models, as used by
    the PyTorch training tool and the prediction tools."""

    def test_detects_hexagdly_backend(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "hexagdly"
        # The LST1 prediction tool checks the backbone on its own.
        assert model_conv_backend(model.model.backbone) == "hexagdly"

    def test_detects_square_backend(self):
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = PyTorchSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "square"

    def test_detects_hexagdly_backend_through_data_parallel(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )
        assert model_conv_backend(nn.DataParallel(model.model)) == "hexagdly"

    def test_detects_hexagdly_backend_after_save_load_round_trip(self, tmp_path):
        """The PyTorch tools save the whole module with torch.save and load it
        back with torch.load, so the restored layers have to identify the
        backend, and the restored model has to predict the same."""
        mapper, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )
        path = tmp_path / "hexagdly_model.pt"
        torch.save(model.model, path)

        restored = torch.load(path, weights_only=False)
        assert model_conv_backend(restored) == "hexagdly"
        assert validate_conv_backend({"LSTCam": mapper}, model_conv_backend(restored)) is True

        batch = torch.rand(2, *input_shape)
        model.model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(batch)[0], model.model(batch)[0])

    def test_hexagdly_model_with_square_mapper_is_rejected(self):
        """What the PyTorch training tool now checks: a hex model fed by a
        square mapper is an error rather than a silent mismatch."""
        mapper, _ = _lst1_input_shape("BilinearMapper")
        _, hex_shape = _lst1_input_shape("HexagdlyMapper")
        model = PyTorchSingleCNN(
            input_shape=hex_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, model_conv_backend(model.model))
