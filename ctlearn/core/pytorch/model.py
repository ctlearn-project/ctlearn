"""
CTLearn PyTorch Model Registry

This module defines the ctapipe Component wrapper classes for PyTorch models,
allowing them to be registered, configured, and instantiated dynamically using
ctapipe's Component system, matching the Keras model design.
"""

from ctapipe.core import Component
from ctapipe.core.traits import Unicode, List, Float, Bool, Int
from ctlearn.core.ctlearn_enum import Task
import torch

class CTLearnPyTorchModel(Component):
    """
    Base class for PyTorch models in CTLearn.
    Acts as a ctapipe Component wrapper for torch.nn.Module models.
    """
    model_name = Unicode(help="Name of the model architecture").tag(config=True)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.model = None


class ThinResNet(CTLearnPyTorchModel):
    """
    Component wrapper for ThinResNet PyTorch model.
    """
    num_blocks = List(
        trait=Int(),
        default_value=[3, 4, 6, 3],
        help="Number of blocks per stage in ResNet",
    ).tag(config=True)

    dropout = Float(
        default_value=0.1,
        help="Dropout probability",
    ).tag(config=True)

    use_bn = Bool(
        default_value=False,
        help="Whether to use Batch Normalization",
    ).tag(config=True)

    def __init__(self, task="type", num_inputs=1, num_outputs=2, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        from ctlearn.core.pytorch.nets.models.ThinResNet.ThinResNet import ThinResNet as PTThinResNet
        self.model = PTThinResNet(
            task=task,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            use_bn=self.use_bn,
        )


class DoubleBBEfficientNet(CTLearnPyTorchModel):
    """
    Component wrapper for DoubleBBEfficientNet PyTorch model.
    """
    model_variant = Unicode(
        default_value="efficientnet-b3",
        help="Variant of EfficientNet backbone (e.g. efficientnet-b0 to b7)",
    ).tag(config=True)

    def __init__(self, task="type", num_inputs=2, num_outputs=2, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        from ctlearn.core.pytorch.nets.models.DoubleBBEfficientNet.DoubleBBEfficientNet import DoubleBBEfficientNet as PTDoubleBBEfficientNet
        device_str = "cuda"
        if parent is not None and hasattr(parent, "device_str"):
            device_str = parent.device_str
        self.model = PTDoubleBBEfficientNet(
            task=task,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            model_variant=self.model_variant,
            device_str=device_str,
        )


class ThinResNet_DBB(CTLearnPyTorchModel):
    """
    Component wrapper for ThinResNet_DBB (Dual Backbone ThinResNet) PyTorch model.
    """
    num_blocks = List(
        trait=Int(),
        default_value=[3, 4, 6, 3],
        help="Number of blocks per stage in ResNet",
    ).tag(config=True)

    dropout = Float(
        default_value=0.1,
        help="Dropout probability",
    ).tag(config=True)

    use_bn = Bool(
        default_value=False,
        help="Whether to use Batch Normalization",
    ).tag(config=True)

    def __init__(self, task="type", num_inputs=2, num_outputs=3, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        from ctlearn.core.pytorch.nets.models.ThinResNet_DBB.ThinResNet_DBB import ThinResNet_DBB as PTThinResNet_DBB
        self.model = PTThinResNet_DBB(
            task=task,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            use_bn=self.use_bn,
        )
