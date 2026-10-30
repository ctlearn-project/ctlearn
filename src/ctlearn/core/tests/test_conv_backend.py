"""Tests for the ``conv_backend`` trait, run against both frameworks.

Each test is written once and parametrized over ``framework``; the few places
where the two frameworks genuinely differ (input shape order, how a model is
run, saved and reloaded) live in the ``FRAMEWORKS`` adapters below, and the
handful of checks that only exist on one side are named accordingly.
"""

import re
import warnings

import keras
import keras_hexagdly as hgly
import numpy as np
import pytest
import pytorch_hexagdly as phgly
import torch
import torch.nn as nn

from ctlearn.core.keras.model import KerasResNet, KerasSingleCNN
from ctlearn.core.pytorch.model import PyTorchResNet, PyTorchSingleCNN
from ctlearn.tools.utils import (
    get_lst1_subarray_description,
    model_conv_backend,
    validate_conv_backend,
)


def _image_mapper(mapper_name):
    from dl1_data_handler.image_mapper import ImageMapper

    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    return ImageMapper.from_name(mapper_name, geometry=geometry, subarray=subarray)


class _Keras:
    single_cnn = KerasSingleCNN
    res_net = KerasResNet
    hex_layers = (hgly.Conv2d, hgly.MaxPool2d, hgly.AvgPool2d)
    square_layers = (
        keras.layers.Conv2D,
        keras.layers.MaxPool2D,
        keras.layers.AveragePooling2D,
    )
    hex_conv = hgly.Conv2d
    hex_avg_pool = hgly.AvgPool2d
    hex_pools = (hgly.MaxPool2d, hgly.AvgPool2d)
    model_suffix = "keras"

    @staticmethod
    def input_shape(mapper, n_channels=2):
        return (mapper.image_shape, mapper.image_shape, n_channels)

    @staticmethod
    def layers(model):
        return list(model.backbone_model.layers)

    @staticmethod
    def predict(model, input_shape, seed):
        batch = np.random.default_rng(seed).uniform(size=(2, *input_shape))
        return model.model.predict(batch.astype(np.float32), verbose=0)

    @staticmethod
    def save_and_load(model, path):
        model.model.save(path)
        return keras.saving.load_model(path)


class _PyTorch:
    single_cnn = PyTorchSingleCNN
    res_net = PyTorchResNet
    hex_layers = (phgly.Conv2d, phgly.MaxPool2d, phgly.AvgPool2d)
    square_layers = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)
    hex_conv = phgly.Conv2d
    hex_avg_pool = phgly.AvgPool2d
    hex_pools = (phgly.MaxPool2d, phgly.AvgPool2d)
    model_suffix = "pt"

    @staticmethod
    def input_shape(mapper, n_channels=2):
        return (n_channels, mapper.image_shape, mapper.image_shape)

    @staticmethod
    def layers(model):
        return list(model.backbone_model.modules())

    @staticmethod
    def predict(model, input_shape, seed):
        batch = np.random.default_rng(seed).uniform(size=(2, *input_shape))
        model.model.eval()
        with torch.no_grad():
            # The PyTorch pipeline returns (logits, backbone features).
            logits, _ = model.model(torch.from_numpy(batch.astype(np.float32)))
        if isinstance(logits, dict):
            return {task: out.numpy() for task, out in logits.items()}
        return logits.numpy()

    @staticmethod
    def save_and_load(model, path):
        torch.save(model.model, path)
        return torch.load(path, weights_only=False)


FRAMEWORKS = {"Keras": _Keras, "PyTorch": _PyTorch}


class TestSingleCNNConvBackend:
    """SingleCNN's conv_backend trait, replacing the old standalone HexCNN class."""

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_default_backend_is_square(self, framework):
        single_cnn = FRAMEWORKS[framework].single_cnn
        assert single_cnn.class_traits()["conv_backend"].default_value == "square"

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_hexagdly_backend_builds_and_predicts_single_task(self, framework):
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("HexagdlyMapper"))
        model = fw.single_cnn(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[
                {"filters": 4, "kernel_size": 1, "number": 1},
                {"filters": 8, "kernel_size": 1, "number": 1},
            ],
            attention_mechanism=None,
        )
        layers = fw.layers(model)
        assert any(isinstance(layer, fw.hex_conv) for layer in layers)
        assert not any(isinstance(layer, fw.square_layers) for layer in layers)

        output = fw.predict(model, input_shape, seed=0)
        assert output.shape == (2, 2)
        if framework == "Keras":
            # The Keras model ends in a Softmax layer; the PyTorch head returns
            # raw logits, since CrossEntropyLoss expects them.
            np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), rtol=1e-4)

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_hexagdly_backend_multi_task_with_batchnorm_and_bottleneck(self, framework):
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("HexagdlyMapper"))
        model = fw.single_cnn(
            input_shape=input_shape,
            tasks=["type", "energy"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            batchnorm=True,
            bottleneck_filters=6,
            attention_mechanism=None,
            head_layers={"type": [8, 2], "energy": [8, 1]},
        )
        outputs = fw.predict(model, input_shape, seed=1)
        assert outputs["type"].shape == (2, 2)
        assert outputs["energy"].shape == (2, 1)

        # The bottleneck is built with the backend's conv class too, so the one
        # architecture conv and the bottleneck are both hexagonal.
        hex_convs = [
            layer for layer in fw.layers(model) if isinstance(layer, fw.hex_conv)
        ]
        assert len(hex_convs) == 2

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_hexagdly_backend_supports_average_pooling(self, framework):
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("HexagdlyMapper"))
        model = fw.single_cnn(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            pooling_type="average",
            attention_mechanism=None,
        )
        assert any(isinstance(layer, fw.hex_avg_pool) for layer in fw.layers(model))
        assert fw.predict(model, input_shape, seed=6).shape == (2, 2)

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_square_backend_still_builds_and_predicts(self, framework):
        """Sanity check that merging HexCNN's logic into SingleCNN didn't
        regress the pre-existing square conv path."""
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("BilinearMapper"))
        model = fw.single_cnn(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            attention_mechanism=None,
        )
        assert not any(isinstance(layer, fw.hex_layers) for layer in fw.layers(model))
        assert fw.predict(model, input_shape, seed=2).shape == (2, 2)

    def test_keras_hexagdly_convs_apply_the_activation_themselves(self):
        """The ReLU is passed to hgly.Conv2d as for keras.layers.Conv2D, not
        added as a separate layer. PyTorch has no equivalent: there the
        activations are separate nn.ReLU modules for both backends.
        """
        mapper = _image_mapper("HexagdlyMapper")
        model = KerasSingleCNN(
            input_shape=_Keras.input_shape(mapper),
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        convs = [
            layer for layer in _Keras.layers(model) if isinstance(layer, hgly.Conv2d)
        ]
        assert convs
        assert all(layer.activation is keras.activations.relu for layer in convs)
        assert not any(
            isinstance(layer, (keras.layers.ReLU, keras.layers.Activation))
            for layer in _Keras.layers(model)
        )


class TestResNetConvBackend:
    """ResNet's conv_backend trait: both residual_block_type variants."""

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_builds_and_predicts(self, framework, residual_block_type):
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("HexagdlyMapper"))
        model = fw.res_net(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}, {"filters": 8, "blocks": 1}],
            attention_mechanism=None,
        )
        # Every conv of the stem and of the residual blocks, shortcuts
        # included, is a hexagonal one.
        layers = fw.layers(model)
        assert any(isinstance(layer, fw.hex_conv) for layer in layers)
        assert not any(isinstance(layer, fw.square_layers) for layer in layers)

        output = fw.predict(model, input_shape, seed=3)
        assert output.shape == (2, 2)
        if framework == "Keras":
            np.testing.assert_allclose(output.sum(axis=-1), np.ones(2), rtol=1e-4)

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_hexagdly_backend_with_attention(self, framework, residual_block_type):
        """Attention on the hex path.

        The squeeze-excite block sizes its bottleneck as
        ``filters // reduction_ratio``, and the basic block applies attention
        at ``filters`` rather than the bottleneck block's ``4 * filters``, so
        the ratio has to stay below this deliberately tiny filter count or
        that division floors to a zero-unit Dense.
        """
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("HexagdlyMapper"))
        model = fw.res_net(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}],
            attention_reduction_ratio=2,
        )
        assert fw.predict(model, input_shape, seed=4).shape == (2, 2)

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_square_backend_still_builds_and_predicts(self, framework):
        """Sanity check that the conv_backend branching didn't regress the
        pre-existing square ResNet path."""
        fw = FRAMEWORKS[framework]
        input_shape = fw.input_shape(_image_mapper("BilinearMapper"))
        model = fw.res_net(
            input_shape=input_shape,
            tasks=["type"],
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )
        assert not any(isinstance(layer, fw.hex_layers) for layer in fw.layers(model))
        assert fw.predict(model, input_shape, seed=5).shape == (2, 2)

    def test_pytorch_hexagdly_backend_backpropagates(self):
        """A backward pass through the stride-2 residual additions reaches
        every hex conv's weights. Keras covers this through model.fit rather
        than an explicit backward pass.
        """
        mapper = _image_mapper("HexagdlyMapper")
        input_shape = _PyTorch.input_shape(mapper)
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


class TestKernelSizeMeaning:
    """``kernel_size`` means the same thing for both backends: a square-kernel
    width. A hex layer reads it that way (``kernel_size_unit="pixels"``) rather
    than as a number of hexagonal rings, so one configuration describes
    comparable models on a square and on a hex grid (issue #302).

    The hexagonal layers' ``hexbase_size`` is the ring count they actually use:
    3 pixels -> 1 ring (the 7-cell hexagon), 1 pixel -> 0 rings (a plain 1x1).
    """

    @staticmethod
    def _hex_layers(fw, model, kinds):
        return [layer for layer in fw.layers(model) if isinstance(layer, kinds)]

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("kernel_size, rings", [(1, 0), (3, 1), (5, 2)])
    def test_single_cnn_conv_reads_kernel_size_as_a_width(
        self, framework, kernel_size, rings
    ):
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": kernel_size, "number": 1}],
            attention_mechanism=None,
        )
        (conv,) = self._hex_layers(fw, model, fw.hex_conv)
        assert conv.kernel_size_unit == "pixels"
        assert conv.hexbase_size == rings

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_single_cnn_bottleneck_is_a_plain_1x1(self, framework):
        """The bottleneck is built with ``kernel_size=1``: a pure per-cell
        channel projection on the square grid, and now on the hex grid too."""
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            bottleneck_filters=6,
            attention_mechanism=None,
        )
        arch_conv, bottleneck = self._hex_layers(fw, model, fw.hex_conv)
        assert arch_conv.hexbase_size == 1
        assert bottleneck.hexbase_size == 0

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("residual_block_type", ["bottleneck", "basic"])
    def test_resnet_hardcoded_kernel_sizes_follow_the_same_meaning(
        self, framework, residual_block_type
    ):
        """The ResNet blocks hardcode ``kernel_size`` 3 (spatial convs) and 1
        (shortcuts, projections), with no way to configure them. Read as widths
        they are a 7-cell hexagon and a plain 1x1, as in a square ResNet."""
        fw = FRAMEWORKS[framework]
        model = fw.res_net(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            residual_block_type=residual_block_type,
            architecture=[{"filters": 4, "blocks": 1}, {"filters": 8, "blocks": 1}],
            attention_mechanism=None,
        )
        convs = self._hex_layers(fw, model, fw.hex_conv)
        assert {conv.kernel_size_unit for conv in convs} == {"pixels"}
        assert {(conv.kernel_size, conv.hexbase_size) for conv in convs} == {
            (1, 0),
            (3, 1),
        }

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("pooling_type", ["max", "average"])
    def test_pooling_size_three_is_the_seven_cell_hexagon(
        self, framework, pooling_type
    ):
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            pooling_type=pooling_type,
            pooling_parameters={"size": 3, "strides": 2},
            attention_mechanism=None,
        )
        pools = self._hex_layers(fw, model, fw.hex_pools)
        assert pools
        assert {
            (pool.kernel_size, pool.hexbase_size, pool.kernel_size_unit)
            for pool in pools
        } == {(3, 1, "pixels")}

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_square_backend_is_untouched(self, framework):
        """The square layers are not wrapped or replaced."""
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("BilinearMapper")),
            tasks=["type"],
            architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
            attention_mechanism=None,
        )
        assert model._conv_layer is fw.square_layers[0]
        assert not any(isinstance(layer, fw.hex_layers) for layer in fw.layers(model))

    # -- even sizes: no exact hexagonal equivalent, rounded up with a warning --------
    # A hexagonal window is centred on a cell, so it only exists for odd sizes
    # (1 cell, then 7 for size 3, 19 for size 5, ...). An even size is rounded up
    # to the next odd one, so 2 gives the same window as 3 (and 1 is the identity).
    # The warning comes from keras-hexagdly / pytorch-hexagdly (>=0.6.1 / >=0.4.1).

    @staticmethod
    def _rounded_sizes(record):
        """The (size, rounded size) pairs announced by the hexagonal layers."""
        return sorted(
            {
                (int(m.group(1)), int(m.group(2)))
                for w in record
                if (
                    m := re.search(
                        r"the size (\d+) is rounded up to (\d+)", str(w.message)
                    )
                )
            }
        )

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    @pytest.mark.parametrize("pooling_type", ["max", "average"])
    def test_even_pooling_size_is_rounded_up_and_warns(self, framework, pooling_type):
        fw = FRAMEWORKS[framework]
        with pytest.warns(UserWarning, match="no even size") as record:
            model = fw.single_cnn(
                input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
                tasks=["type"],
                conv_backend="hexagdly",
                architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
                pooling_type=pooling_type,
                pooling_parameters={"size": 2, "strides": 2},
                attention_mechanism=None,
            )
        pools = self._hex_layers(fw, model, fw.hex_pools)
        assert pools
        # size 2 gives the 7-cell window of size 3, not the identity of size 1
        assert {(pool.kernel_size, pool.hexbase_size) for pool in pools} == {(2, 1)}
        assert self._rounded_sizes(record) == [(2, 3)]

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_default_configuration_warns_about_its_pooling_size(self, framework):
        """CTLearn's default pooling is ``size: 2``."""
        fw = FRAMEWORKS[framework]
        with pytest.warns(UserWarning, match="no even size") as record:
            fw.single_cnn(
                input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
                tasks=["type"],
                conv_backend="hexagdly",
                attention_mechanism=None,
            )
        assert self._rounded_sizes(record) == [(2, 3)]

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_even_kernel_size_is_rounded_up_and_warns(self, framework):
        fw = FRAMEWORKS[framework]
        with pytest.warns(UserWarning, match="no even size") as record:
            model = fw.single_cnn(
                input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
                tasks=["type"],
                conv_backend="hexagdly",
                architecture=[{"filters": 4, "kernel_size": 4, "number": 1}],
                pooling_parameters={"size": 3, "strides": 2},
                attention_mechanism=None,
            )
        (conv,) = self._hex_layers(fw, model, fw.hex_conv)
        assert conv.hexbase_size == 2  # the 19-cell window of size 5
        assert self._rounded_sizes(record) == [(4, 5)]

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_resnet_stem_even_sizes_warn(self, framework):
        """Only the stem is configurable: the residual blocks hardcode 1 and 3."""
        fw = FRAMEWORKS[framework]
        with pytest.warns(UserWarning, match="no even size") as record:
            fw.res_net(
                input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
                tasks=["type"],
                conv_backend="hexagdly",
                init_layer={"filters": 4, "kernel_size": 4, "strides": 2},
                init_max_pool={"size": 2, "strides": 2},
                architecture=[{"filters": 4, "blocks": 1}],
                attention_mechanism=None,
            )
        assert self._rounded_sizes(record) == [(2, 3), (4, 5)]

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_odd_sizes_do_not_warn(self, framework):
        fw = FRAMEWORKS[framework]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*no even size.*")
            fw.single_cnn(
                input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
                tasks=["type"],
                conv_backend="hexagdly",
                architecture=[{"filters": 4, "kernel_size": 3, "number": 1}],
                pooling_parameters={"size": 3, "strides": 2},
                attention_mechanism=None,
            )

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_square_backend_does_not_warn_about_even_sizes(self, framework):
        """Even sizes are fine on a square grid (the default pooling is 2x2)."""
        fw = FRAMEWORKS[framework]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*no even size.*")
            fw.single_cnn(
                input_shape=fw.input_shape(_image_mapper("BilinearMapper")),
                tasks=["type"],
                architecture=[{"filters": 4, "kernel_size": 4, "number": 1}],
                attention_mechanism=None,
            )


class TestValidateConvBackend:
    """ctlearn.tools.utils.validate_conv_backend -- the mapper<->model conv
    backend consistency check requested in review. Framework-independent."""

    def test_matched_hexagdly_pairing_passes(self):
        mapper = _image_mapper("HexagdlyMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "hexagdly") is True

    def test_matched_square_pairing_passes(self):
        mapper = _image_mapper("BilinearMapper")
        assert validate_conv_backend({"LSTCam": mapper}, "square") is True

    def test_hexagdly_mapper_with_square_backend_raises(self):
        mapper = _image_mapper("HexagdlyMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "square")

    def test_square_mapper_with_hexagdly_backend_raises(self):
        mapper = _image_mapper("BilinearMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend({"LSTCam": mapper}, "hexagdly")

    @pytest.mark.parametrize("conv_backend", ["hexagdly", "square"])
    def test_mixed_mappers_raise_for_either_backend(self, conv_backend):
        """image_mapper_type is a TelescopeParameter, so telescope types can
        be given different mappers. No single conv_backend can serve a mix of
        hex-addressed and square-mapped inputs, so both are rejected."""
        hex_mapper = _image_mapper("HexagdlyMapper")
        square_mapper = _image_mapper("BilinearMapper")
        with pytest.raises(ValueError, match="[Mm]ixed"):
            validate_conv_backend(
                {"LSTCam": hex_mapper, "CHEC": square_mapper}, conv_backend
            )


class TestModelConvBackend:
    """ctlearn.tools.utils.model_conv_backend -- detects the conv backend of a
    (possibly loaded-from-disk) model by inspecting its layers, used by the
    training tools and at prediction time where no conv_backend trait is
    available."""

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_detects_hexagdly_backend(self, framework):
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "hexagdly"
        # predict_LST1 inspects the backbone on its own, not the full model.
        assert model_conv_backend(model.backbone_model) == "hexagdly"

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_detects_square_backend(self, framework):
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("BilinearMapper")),
            tasks=["type"],
            attention_mechanism=None,
        )
        assert model_conv_backend(model.model) == "square"

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_hexagdly_model_with_square_mapper_is_rejected(self, framework):
        """What the training tools check: a hex model fed by a square mapper
        is an error rather than a silent mismatch."""
        fw = FRAMEWORKS[framework]
        model = fw.single_cnn(
            input_shape=fw.input_shape(_image_mapper("HexagdlyMapper")),
            tasks=["type"],
            conv_backend="hexagdly",
            attention_mechanism=None,
        )
        square_mapper = _image_mapper("BilinearMapper")
        with pytest.raises(ValueError, match="conv_backend"):
            validate_conv_backend(
                {"LSTCam": square_mapper}, model_conv_backend(model.model)
            )

    @pytest.mark.parametrize("framework", ["Keras", "PyTorch"])
    def test_detects_hexagdly_backend_after_save_load_round_trip(
        self, framework, tmp_path
    ):
        """The tools load models from disk rather than building them through
        CTLearnModel.from_name, and LoadedModel keeps conv_backend at its
        default whatever it wraps -- so the restored layers, not the trait,
        have to be what identifies the backend.
        """
        fw = FRAMEWORKS[framework]
        mapper = _image_mapper("HexagdlyMapper")
        input_shape = fw.input_shape(mapper)
        model = fw.single_cnn(
            input_shape=input_shape,
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "kernel_size": 1, "number": 1}],
            attention_mechanism=None,
        )
        restored = fw.save_and_load(
            model, tmp_path / f"hexagdly_model.{fw.model_suffix}"
        )
        assert model_conv_backend(restored) == "hexagdly"
        assert (
            validate_conv_backend({"LSTCam": mapper}, model_conv_backend(restored))
            is True
        )

    def test_pytorch_detects_hexagdly_backend_through_data_parallel(self):
        """The PyTorch training tool wraps the model in nn.DataParallel on
        multi-GPU machines."""
        mapper = _image_mapper("HexagdlyMapper")
        model = PyTorchResNet(
            input_shape=_PyTorch.input_shape(mapper),
            tasks=["type"],
            conv_backend="hexagdly",
            architecture=[{"filters": 4, "blocks": 1}],
            attention_mechanism=None,
        )
        assert model_conv_backend(nn.DataParallel(model.model)) == "hexagdly"
