"""
This module defines the ``CTLearnModel`` classes, which holds the basic functionality for creating a Keras model to be used in CTLearn.
"""

from abc import abstractmethod
import keras

from ctapipe.core import Component
from ctapipe.core.traits import Bool, Int, CaselessStrEnum, List, Dict, Unicode, Path
from ctlearn.utils import validate_trait_dict

__all__ = [
    "CTLearnModel",
    "SingleCNN",
    "ResNet",
    "LoadedModel",
]


class CTLearnModel(Component):
    """
    Base component for creating a Keras model to be used in CTLearn.

    This class defines the basic functionality for creating a Keras model to be used in CTLearn.
    It provides the necessary methods to build the backbone of the model and the fully connected head
    for the specified tasks.
    """

    init_padding = Int(
        default_value=0,
        allow_none=False,
        min=0,
        help="Initial padding to apply to the input data.",
    ).tag(config=True)

    head_layers = Dict(
        default_value={
            "type": [512, 256, 2],
            "energy": [512, 256, 1],
            "cameradirection": [512, 256, 3],
            "skydirection": [512, 256, 3],
        },
        allow_none=False,
        help=(
            "Dictionary containing the number of neurons in the fully connected head for each "
            "task ('type', 'energy', 'cameradirection', 'skydirection'). Note: The number of neurons in the last layer "
            "must match the number of classes or the number of reconstructed values."
        ),
    ).tag(config=True)

    head_activation_function = Dict(
        default_value={
            "type": "relu",
            "energy": "relu",
            "cameradirection": "tanh",
            "skydirection": "tanh",
        },
        allow_none=False,
        help=(
            "Dictionary containing the activation function for the fully connected head for each "
            "task ('type', 'energy', 'cameradirection', 'skydirection'). Note: The default activation functions "
            "are 'relu' for 'type' and 'energy' tasks, and 'tanh' for 'cameradirection' and 'skydirection' tasks. "
            "The 'type' task uses 'softmax' as the final activation function."
        ),
    ).tag(config=True)

    attention_mechanism = CaselessStrEnum(
        ["Dual-SE", "Channel-SE", "Spatial-SE"],
        default_value="Dual-SE",
        allow_none=True,
        help="Type of squeeze and excitation attention mechanism to use.",
    ).tag(config=True)

    attention_reduction_ratio = Int(
        default_value=16,
        allow_none=True,
        min=1,
        help="Reduction ratio for the squeeze and excitation attention mechanism.",
    ).tag(config=True)

    def __init__(
        self,
        config=None,
        parent=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        config : traitlets.loader.Config
            Configuration specified by config file or cmdline arguments.
            Used to set traitlet values.
            This is mutually exclusive with passing a ``parent``.
        parent : ctapipe.core.Component or ctapipe.core.Tool
            Parent of this component in the configuration hierarchy,
            this is mutually exclusive with passing ``config``
        """
        super().__init__(config=config, parent=parent, **kwargs)

        # Define the squeeze and excitation attention mechanism
        self.attention = None
        if self.attention_mechanism is not None:
            self.attention = {
                "mechanism": self.attention_mechanism,
                "reduction_ratio": self.attention_reduction_ratio,
            }


    @abstractmethod
    def _build_backbone(self, input_shape):
        """
        Build the backbone of the CTLearn model.

        Function to build the backbone of the CTLearn model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the backbone of the CTLearn model.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """
        pass


class SingleCNN(CTLearnModel):
    """
    ``SingleCNN`` is the base class of a simple convolutional neural network model.

    This class extends the functionality of ``CTLearnModel`` by implementing
    methods to build a simple convolutional neural network model.
    """

    name = Unicode(
        "SingleCNN",
        help="Name of the model backbone.",
    ).tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 32, "kernel_size": 3, "number": 1},
            {"filters": 32, "kernel_size": 3, "number": 1},
            {"filters": 64, "kernel_size": 3, "number": 1},
            {"filters": 128, "kernel_size": 3, "number": 1},
        ],
        allow_none=False,
        help=(
            "List of dicts containing the number of filters, kernel sizes and number of repetition. "
            "E.g. ``[{'filters': 12, 'kernel_size': 3, 'number': 1}, ...]``."
        ),
    ).tag(config=True)

    pooling_type = CaselessStrEnum(
        ["max", "average"],
        default_value="max",
        allow_none=True,
        help="Type of pooling to apply to the convolutional layers with ``pooling_parameters``.",
    ).tag(config=True)

    pooling_parameters = Dict(
        default_value={"size": 2, "strides": 2},
        allow_none=True,
        help=(
            "Parameters for the max or average pooling layers. "
            "E.g. ``{'size': 2, 'strides': 2}``."
        ),
    ).tag(config=True)

    batchnorm = Bool(
        default_value=False,
        allow_none=False,
        help="Apply batch normalization to the convolutional layers.",
    ).tag(config=True)

    bottleneck_filters = Int(
        default_value=None,
        allow_none=True,
        help="Number of filters in the bottleneck layer.",
    ).tag(config=True)

    def __init__(
        self,
        tasks,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parent=parent,
            **kwargs,
        )

        # Validate the architecture trait
        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "kernel_size", "number"])
        # Validate the pooling parameters trait
        validate_trait_dict(self.pooling_parameters, ["size", "strides"])
        # Validate the head trait with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"


class ResNet(CTLearnModel):
    """
    ``ResNet`` is a residual neural network model.

    This class extends the functionality of ``CTLearnModel`` by implementing
    methods to build a residual neural network model.
    """

    name = Unicode(
        "ThinResNet",
        help="Name of the model backbone.",
    ).tag(config=True)

    init_layer = Dict(
        default_value=None,
        allow_none=True,
        help=(
            "Parameters for the first convolutional layer. "
            "E.g. ``{'filters': 64, 'kernel_size': 7, 'strides': 2}``."
        ),
    ).tag(config=True)

    init_max_pool = Dict(
        default_value=None,
        allow_none=True,
        help=(
            "Parameters for the first max pooling layer. "
            "E.g. ``{'size': 3, 'strides': 2}``."
        ),
    ).tag(config=True)

    residual_block_type = CaselessStrEnum(
        ["basic", "bottleneck"],
        default_value="bottleneck",
        allow_none=False,
        help="Type of residual block to use.",
    ).tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 48, "blocks": 2},
            {"filters": 96, "blocks": 3},
            {"filters": 128, "blocks": 3},
            {"filters": 256, "blocks": 3},
        ],
        allow_none=False,
        help=(
            "List of dicts containing the number of filters and residual blocks. "
            "E.g. ``[{'filters': 12, 'blocks': 2}, ...]``."
        ),
    ).tag(config=True)

    def __init__(
        self,
        tasks,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parent=parent,
            **kwargs,
        )

        # Validate the architecture trait
        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "blocks"])
        # Validate the initial layers trait
        if self.init_layer is not None:
            validate_trait_dict(self.init_layer, ["filters", "kernel_size", "strides"])
        if self.init_max_pool is not None:
            validate_trait_dict(self.init_max_pool, ["size", "strides"])
        # Validate the head traits with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"


class LoadedModel(CTLearnModel):
    """
    ``LoadedModel`` is a pre-trained Keras model.

    This class extends the functionality of ``CTLearnModel`` by implementing
    methods to load a pre-trained Keras model. The model can be used as a backbone
    for the CTLearn model.
    """

    load_model_from = Path(
        default_value=None,
        help="Path to a Keras model file (Keras3) or directory Keras2)",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=True,
    ).tag(config=True)

    overwrite_head = Bool(
        default_value=False,
        allow_none=False,
        help="Set to overwrite the fully connected head from the loaded model.",
    ).tag(config=True)

    trainable_backbone = Bool(
        default_value=True,
        allow_none=False,
        help="Set to set the backbone model to be trainable.",
    ).tag(config=True)

    def __init__(
        self,
        tasks,
        config=None,
        parent=None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            parent=parent,
            **kwargs,
        )
        # Load the fully connected head from the loaded model or build a new one
        if self.overwrite_head:
            backbone_output = self.backbone_model(self.input_layer)
            # Validate the head trait with the provided tasks
            validate_trait_dict(self.head_layers, tasks)