import numpy as np
import pytest

from ctlearn.core.model import SingleCNN, ResNet
from ctlearn.utils import (
    get_lst1_subarray_description,
    validate_conv_backend,
    model_conv_backend,
)

try:
    from dl1_data_handler.image_mapper import HexagdlyMapper  # noqa: F401

    HEXAGDLY_MAPPER_AVAILABLE = True
except ImportError:
    HEXAGDLY_MAPPER_AVAILABLE = False

requires_hexagdly_mapper = pytest.mark.skipif(
    not HEXAGDLY_MAPPER_AVAILABLE,
    reason=(
        "dl1_data_handler.image_mapper.HexagdlyMapper not available -- needs "
        "dl1-data-handler with hex mapper support merged/released "
        "(cta-observatory/dl1-data-handler PR pending)"
    ),
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
        assert SingleCNN.class_traits()["conv_backend"].default_value == "square"

    @requires_hexagdly_mapper
    def test_hexagdly_backend_builds_and_predicts_single_task(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = SingleCNN(
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

    @requires_hexagdly_mapper
    def test_hexagdly_backend_multi_task_with_batchnorm_and_bottleneck(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = SingleCNN(
            input_shape=input_shape,
            tasks=["type", "energy"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            batchnorm=True,
            bottleneck_filters=6,
            # attention_mechanism left at None: SingleCNN._build_backbone has
            # a pre-existing, out-of-scope bug (reads self.attention["ratio"]
            # instead of "reduction_ratio") unrelated to conv_backend --
            # see ResNet's tests below for attention + hexagdly coverage,
            # since ResNet's residual blocks already use the correct key.
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
        model = SingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            attention_mechanism=None,
        )

        rng = np.random.default_rng(2)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)

    @requires_hexagdly_mapper
    def test_hexagdly_backend_rejects_average_pooling(self):
        """keras_hexagdly has no hexagonal average-pool layer."""
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        with pytest.raises(ValueError, match="average"):
            SingleCNN(
                input_shape=input_shape,
                tasks=["type"],
                conv_backend="hexagdly",
                pooling_type="average",
                attention_mechanism=None,
            )


class TestResNetConvBackend:
    """ResNet's conv_backend trait: both residual_block_type variants."""

    @requires_hexagdly_mapper
    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_builds_and_predicts(self, residual_block_type):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = ResNet(
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

    @requires_hexagdly_mapper
    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_with_attention(self, residual_block_type):
        """ResNet's residual blocks already use the correct
        'reduction_ratio' key (unlike SingleCNN's pre-existing, out-of-scope
        bug), so attention should work out of the box on the hex path too."""
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = ResNet(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}],
        )

        rng = np.random.default_rng(4)
        batch = rng.uniform(size=(2, *input_shape)).astype(np.float32)
        output = model.model.predict(batch, verbose=0)
        assert output.shape == (2, 2)

    def test_square_backend_still_builds_and_predicts(self):
        """Sanity check that the conv_backend branching didn't regress the
        pre-existing square ResNet path."""
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = ResNet(
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

    @requires_hexagdly_mapper
    def test_matched_hexagdly_pairing_passes(self):
        mapper, _ = _lst1_input_shape("HexagdlyMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "hexagdly") is True

    def test_matched_square_pairing_passes(self):
        mapper, _ = _lst1_input_shape("BilinearMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "square") is True

    @requires_hexagdly_mapper
    def test_hexagdly_mapper_with_square_backend_raises(self):
        mapper, _ = _lst1_input_shape("HexagdlyMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "square")

    def test_square_mapper_with_hexagdly_backend_raises(self):
        mapper, _ = _lst1_input_shape("BilinearMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "hexagdly")


class TestModelConvBackend:
    """ctlearn.utils.model_conv_backend -- detects the conv backend of a
    (possibly loaded-from-disk) keras.Model by inspecting its layers, used
    at prediction time where no conv_backend trait is available."""

    @requires_hexagdly_mapper
    def test_detects_hexagdly_backend(self):
        _, input_shape = _lst1_input_shape("HexagdlyMapper")
        model = SingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "hexagdly"

    def test_detects_square_backend(self):
        _, input_shape = _lst1_input_shape("BilinearMapper")
        model = SingleCNN(
            input_shape=input_shape,
            tasks=["type"],
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "square"
