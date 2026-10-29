"""
ctlearn core functionalities
"""

from importlib import import_module

__all__ = [
    "ConvBackend",
    "CTLearnModel",
    "SingleCNN",
    "ResNet",
    "LoadedModel",
    "build_fully_connect_keras_head",
    "KerasConvBackend",
    "KerasSingleCNN",
    "KerasResNet",
    "KerasLoadedModel",
    "dual_squeeze_excite_block",
    "channel_squeeze_excite_block",
    "spatial_squeeze_excite_block",
    "KerasSequence",
    "BasicBlock",
    "BottleneckBlock",
    "MultiFullyConnectedHead",
    "build_fully_connect_pytorch_head",
    "PyTorchConvBackend",
    "PyTorchSingleCNN",
    "PyTorchResNet",
    "PyTorchLoadedModel",
    "DualSqueezeExciteBlock",
    "ChannelSqueezeExciteBlock",
    "SpatialSqueezeExciteBlock",
    "PyTorchDataset",
]

_LAZY_MODULES = {
    **dict.fromkeys(
        ("CTLearnModel", "SingleCNN", "ResNet", "LoadedModel"),
        ".model",
    ),
    **dict.fromkeys(
        (
            "build_fully_connect_keras_head",
            "KerasConvBackend",
            "KerasSingleCNN",
            "KerasResNet",
            "KerasLoadedModel",
        ),
        ".keras.model",
    ),
    **dict.fromkeys(
        (
            "dual_squeeze_excite_block",
            "channel_squeeze_excite_block",
            "spatial_squeeze_excite_block",
        ),
        ".keras.attention",
    ),
    "KerasSequence": ".keras.sequence",
    **dict.fromkeys(
        (
            "BasicBlock",
            "BottleneckBlock",
            "MultiFullyConnectedHead",
            "build_fully_connect_pytorch_head",
            "PyTorchConvBackend",
            "PyTorchSingleCNN",
            "PyTorchResNet",
            "PyTorchLoadedModel",
        ),
        ".pytorch.model",
    ),
    **dict.fromkeys(
        (
            "DualSqueezeExciteBlock",
            "ChannelSqueezeExciteBlock",
            "SpatialSqueezeExciteBlock",
        ),
        ".pytorch.attention",
    ),
    "PyTorchDataset": ".pytorch.dataset",
}


def __getattr__(name):
    """
    Dynamically retrieve and return the requested attribute from the module.
    """
    try:
        module_name = _LAZY_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    return getattr(module, name)
