"""
ctlearn core functionalities
"""

__all__ = [
    "CTLearnModel",
    "SingleCNN",
    "ResNet",
    "LoadedModel",
    "build_fully_connect_keras_head",
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
    "PyTorchSingleCNN",
    "PyTorchResNet",
    "PyTorchLoadedModel",
    "DualSqueezeExciteBlock",
    "ChannelSqueezeExciteBlock",
    "SpatialSqueezeExciteBlock",
    "PyTorchDataset",
]

def __getattr__(name):
    if name in ("CTLearnModel", "SingleCNN", "ResNet", "LoadedModel"):
        from .model import CTLearnModel, SingleCNN, ResNet, LoadedModel
        return locals()[name]
    if name in ("build_fully_connect_keras_head", "KerasSingleCNN", "KerasResNet", "KerasLoadedModel"):
        from .keras.model import build_fully_connect_keras_head, KerasSingleCNN, KerasResNet, KerasLoadedModel
        return locals()[name]
    if name in ("dual_squeeze_excite_block", "channel_squeeze_excite_block", "spatial_squeeze_excite_block"):
        from .keras.attention import dual_squeeze_excite_block, channel_squeeze_excite_block, spatial_squeeze_excite_block
        return locals()[name]
    if name == "KerasSequence":
        from .keras.sequence import KerasSequence
        return KerasSequence
    if name in ("BasicBlock", "BottleneckBlock", "MultiFullyConnectedHead", "build_fully_connect_pytorch_head", "PyTorchSingleCNN", "PyTorchResNet", "PyTorchLoadedModel"):
        from .pytorch.model import BasicBlock, BottleneckBlock, MultiFullyConnectedHead, build_fully_connect_pytorch_head, PyTorchSingleCNN, PyTorchResNet, PyTorchLoadedModel
        return locals()[name]
    if name in ("DualSqueezeExciteBlock", "ChannelSqueezeExciteBlock", "SpatialSqueezeExciteBlock"):
        from .pytorch.attention import DualSqueezeExciteBlock, ChannelSqueezeExciteBlock, SpatialSqueezeExciteBlock
        return locals()[name]
    if name == "PyTorchDataset":
        from .pytorch.dataset import PyTorchDataset
        return PyTorchDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")