import importlib

import pytest

import ctlearn.core as core


@pytest.mark.parametrize(
    ("name", "module_name"),
    [
        ("CTLearnModel", "ctlearn.core.model"),
        ("SingleCNN", "ctlearn.core.model"),
        ("ResNet", "ctlearn.core.model"),
        ("LoadedModel", "ctlearn.core.model"),
        ("build_fully_connect_keras_head", "ctlearn.core.keras.model"),
        ("KerasSingleCNN", "ctlearn.core.keras.model"),
        ("KerasResNet", "ctlearn.core.keras.model"),
        ("KerasLoadedModel", "ctlearn.core.keras.model"),
        ("dual_squeeze_excite_block", "ctlearn.core.keras.attention"),
        ("channel_squeeze_excite_block", "ctlearn.core.keras.attention"),
        ("spatial_squeeze_excite_block", "ctlearn.core.keras.attention"),
        ("KerasSequence", "ctlearn.core.keras.sequence"),
        ("BasicBlock", "ctlearn.core.pytorch.model"),
        ("BottleneckBlock", "ctlearn.core.pytorch.model"),
        ("MultiFullyConnectedHead", "ctlearn.core.pytorch.model"),
        ("build_fully_connect_pytorch_head", "ctlearn.core.pytorch.model"),
        ("PyTorchSingleCNN", "ctlearn.core.pytorch.model"),
        ("PyTorchResNet", "ctlearn.core.pytorch.model"),
        ("PyTorchLoadedModel", "ctlearn.core.pytorch.model"),
        ("DualSqueezeExciteBlock", "ctlearn.core.pytorch.attention"),
        ("ChannelSqueezeExciteBlock", "ctlearn.core.pytorch.attention"),
        ("SpatialSqueezeExciteBlock", "ctlearn.core.pytorch.attention"),
        ("PyTorchDataset", "ctlearn.core.pytorch.dataset"),
    ],
)
def test_lazy_core_exports(name, module_name):
    implementation = importlib.import_module(module_name)

    assert getattr(core, name) is getattr(implementation, name)


def test_unknown_core_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="unknown_core_export"):
        core.unknown_core_export
