import keras
import keras_hexagdly as hgly
import numpy as np
import pytest

from ctlearn.core.keras.model import KerasSingleCNN, KerasResNet
from ctlearn.tools.utils import (
    get_lst1_subarray_description,
    validate_conv_backend,
    model_conv_backend,
)


def _lst1_input_shape(mapper_name, n_channels=2):
    from dl1_data_handler.image_mapper import ImageMapper

    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    mapper = ImageMapper.from_name(mapper_name, geometry=geometry, subarray=subarray)
    return mapper, (mapper.image_shape, mapper.image_shape, n_channels)


class TestSingleCNNConvBackend:
    """SingleCNN's conv_backend trait, replacing the old standalone HexCNN class."""

    def test_default_backend_is_square(self):
        assert KerasSingleCNN.class_traits()["conv_backend"].default_value == "square"

    def test_hexagdly_backend_builds_and_predicts_single_task(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[
                {"filters": 4, "kernel_size": 1, "number": 1},
                {"filters": 8, "kernel_size": 1, "number": 1},
            ],
            attention_mechanism=None,
        )

        rng = np.random.default_rng(0)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)

        # Single-task 'type' output: (batch, 2) softmax logits.
        assert output.shape == (2, 2)
        np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), rtol=1e-4)

    def test_hexagdly_backend_multi_task_with_batchnorm_and_bottleneck(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type", "energy"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            batchnorm=True,
            bottleneck_filters=6,
            attention_mechanism=None,
            head_layers={"type": [8, 2], "energy": [8, 1]},
        )

        rng = np.random.default_rng(1)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        outputs = model.model.predict(batch, verbose=0)

        assert outputs["type"].shape == (2, 2)
        assert outputs["energy"].shape == (2, 1)

    def test_square_backend_still_builds_and_predicts(self):
        """Sanity check that merging HexCNN's logic into SingleCNN didn't
        regress the pre-existing square conv path."""
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            attention_mechanism=None,
        )

        rng = np.random.default_rng(2)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)

    def test_hexagdly_backend_supports_average_pooling(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            pooling_type="average",
            attention_mechanism=None,
        )
        assert any(isinstance(layer, hgly.AvgPool2d) for layer in model.backbone_model.layers)

        rng = np.random.default_rng(6)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        assert model.model.predict(batch, verbose=0).shape == (2, 2)

    def test_hexagdly_convs_apply_the_activation_themselves(self):
        """The ReLU is passed to hgly.Conv2d as for keras.layers.Conv2D, not
        added as a separate layer."""
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        convs = [layer for layer in model.backbone_model.layers if isinstance(layer, hgly.Conv2d)]
        assert convs
        assert all(layer.activation is keras.activations.relu for layer in convs)
        assert not any(
            isinstance(layer, (keras.layers.ReLU, keras.layers.Activation))
            for layer in model.backbone_model.layers
        )


class TestResNetConvBackend:
    """ResNet's conv_backend trait: both residual_block_type variants."""

    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_builds_and_predicts(self, residual_block_type):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}, {"filters": 8, "blocks": 1}],
            attention_mechanism=None,
        )

        rng = np.random.default_rng(3)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)
        np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), rtol=1e-4)

    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_with_attention(self, residual_block_type):
        """Attention on the hex path.

        The squeeze-excite block sizes its bottleneck as
        ``filters // reduction_ratio``, and the basic block applies attention
        at ``filters`` rather than the bottleneck block's ``4 * filters``, so
        the ratio has to stay below this deliberately tiny filter count or
        that division floors to a zero-unit Dense.
        """
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}],
            attention_reduction_ratio=2,
        )

        rng = np.random.default_rng(4)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)

    def test_square_backend_still_builds_and_predicts(self):
        """Sanity check that the conv_backend branching didn't regress the
        pre-existing square ResNet path."""
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = KerasResNet(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )

        rng = np.random.default_rng(5)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)


class TestValidateConvBackend:
    """ctlearn.utils.validate_conv_backend -- the mapper<->model conv
    backend consistency check requested in review."""

    def test_matched_hexagdly_pairing_passes(self):
        mapper, _ = _lst1_input_shape("HexagdlyMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "hexagdly") is True

    def test_matched_square_pairing_passes(self):
        mapper, _ = _lst1_input_shape("BilinearMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "square") is True

    def test_hexagdly_mapper_with_square_backend_raises(self):
        mapper, _ = _lst1_input_shape("HexagdlyMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "square")

    def test_square_mapper_with_hexagdly_backend_raises(self):
        mapper, _ = _lst1_input_shape("BilinearMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "hexagdly")

    @pytest.mark.parametrize("conv_backend", ["hexagdly", "square"])
    def test_mixed_mappers_raise_for_either_backend(self, conv_backend):
        """image_mapper_type is a TelescopeParameter, so telescope types can
        be given different mappers. No single conv_backend can serve a mix of
        hex-addressed and square-mapped inputs, so both are rejected."""
        hex_mapper, _ = _lst1_input_shape("HexagdlyMapper")
        square_mapper, _ = _lst1_input_shape("BilinearMapper")
        with pytest.raises(ValueError, match="[Mm]ixed"):
            validate_conv_backend(
                {"LSTCam": hex_mapper, "CHEC": square_mapper}, conv_backend
            )


class TestModelConvBackend:
    """ctlearn.utils.model_conv_backend -- detects the conv backend of a
    (possibly loaded-from-disk) keras.Model by inspecting its layers, used
    at prediction time where no conv_backend trait is available."""

    def test_detects_hexagdly_backend(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "hexagdly"

    def test_detects_square_backend(self):
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "square"

    def test_detects_hexagdly_backend_after_save_load_round_trip(self, tmp_path):
        """The prediction tools call this on a model restored by
        keras.saving.load_model, and LoadedModel keeps conv_backend at its
        default whatever it wraps -- so the restored layers, not the trait,
        have to be what identifies the backend.
        """
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            attention_mechanism=None,
        )
        path = tmp_path / "hexagdly_model.keras"
        model.model.save(path)

        restored = keras.saving.load_model(path)
        assert model_conv_backend(restored) == "hexagdly"

        mapper, _ = _lst1_input_shape("HexagdlyMapper")
        assert (
            validate_conv_backend({"LSTCam": mapper}, model_conv_backend(restored))
            is True
        )
