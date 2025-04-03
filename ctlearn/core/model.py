"""
This module defines the ``CTLearnModel`` classes, which holds the basic functionality for creating a Keras model to be used in CTLearn.
"""

from abc import abstractmethod
import keras
import tensorflow as tf
from ctapipe.core import Component
from ctapipe.core.traits import Bool, Int, CaselessStrEnum, List, Dict, Unicode, Path, Tuple
from ctlearn.core.attention import (
    dual_squeeze_excite_block,
    channel_squeeze_excite_block,
    spatial_squeeze_excite_block,
)
from ctlearn.core.indexed import IndexedMaxPool2D, IndexedAveragePool2D, IndexedConv2D, IndexedConv3D
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Conv3D
from ctlearn.utils import validate_trait_dict
from traitlets.config import Config
import numpy as np
__all__ = [
    "build_fully_connect_head",
    "CTLearnModel",
    "SingleCNN",
    "ResNet",
    "LoadedModel",
]


def build_fully_connect_head(inputs, layers, activation_function, tasks):
    """
    Build the fully connected head for the CTLearn model.

    Function to build the fully connected head of the CTLearn model using the specified parameters.

    Parameters
    ----------
    inputs : keras.layers.Layer
        Keras layer of the model.
    layers : dict
        Dictionary containing the number of neurons (as value) in the fully connected head for each task (as key).
    activation_function : dict
        Dictionary containing the activation function (as value) for the fully connected head for each task (as key).
    tasks : list
        List of tasks to build the head for.

    Returns
    -------
    logits : dict
        Dictionary containing the logits for each task.
    """
    logits = {}
    for task in tasks:
        x = inputs
        for i, units in enumerate(layers[task]):
            if i != len(layers[task]) - 1:
                x = keras.layers.Dense(
                    units=units,
                    activation=activation_function[task],
                    name=f"fc_{task}_{i+1}",
                )(x)
            else:
                x = keras.layers.Dense(units=units, name=task)(x)
        logits[task] = keras.layers.Softmax()(x) if task == "type" else x
         
    # Temp fix till keras support class weights for multiple outputs or I wrote custom loss
    # https://github.com/keras-team/keras/issues/11735
    if len(tasks) == 1 and tasks[0] == "type":
        logits = logits[tasks[0]]
    return logits


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
            "cameradirection": [512, 256, 2],
            "skydirection": [512, 256, 2],
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
    ``SingleCNN`` is a simple convolutional neural network model.

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
        input_shape,
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

        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"

        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        # Validate the head trait with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Build the fully connected head depending on the tasks
        self.logits = build_fully_connect_head(
            backbone_output, self.head_layers, self.head_activation_function, tasks
        )

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")


    def _build_backbone(self, input_shape):
        """
        Build the SingleCNN model backbone.

        Function to build the backbone of the SingleCNN model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the backbone of the SingleCNN model.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """
        # Define the input layer from the input shape
        network_input = keras.Input(input_shape, name="input")
        # Get model arcihtecture parameters for the backbone
        filters_list = [layer["filters"] for layer in self.architecture]
        kernel_sizes = [layer["kernel_size"] for layer in self.architecture]
        numbers_list = [layer["number"] for layer in self.architecture]

        x = network_input
        if self.batchnorm:
            x = keras.layers.BatchNormalization(momentum=0.99)(x)
        total_layers = len(filters_list)
        for i, (filters, kernel_size, number) in enumerate(
            zip(filters_list, kernel_sizes, numbers_list)
        ):
            for nr in range(number):
                x = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation="relu",
                    name=f"{self.backbone_name}_conv_{i+1}_{nr+1}",
                )(x)
            if self.pooling_type is not None and i < total_layers - 1:
                if self.pooling_type == "max":
                    x = keras.layers.MaxPool2D(
                        pool_size=self.pooling_parameters["size"],
                        strides=self.pooling_parameters["strides"],
                        name=f"{self.backbone_name}_pool_{i+1}",
                    )(x)
                elif self.pooling_type == "average":
                    x = keras.layers.AveragePooling2D(
                        pool_size=self.pooling_parameters["size"],
                        strides=self.pooling_parameters["strides"],
                        name=f"{self.backbone_name}_pool_{i+1}",
                    )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        # bottleneck layer
        if self.bottleneck_filters is not None:
            x = keras.layers.Conv2D(
                filters=self.bottleneck_filters,
                kernel_size=1,
                padding="same",
                activation="relu",
                name=f"{self.backbone_name}_bottleneck",
            )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        # Attention mechanism
        if self.attention is not None:
            if self.attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(
                    x, self.attention["reduction_ratio"], name=f"{self.backbone_name}_dse"
                )
            elif self.attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(
                    x, self.attention["reduction_ratio"], name=f"{self.backbone_name}_cse"
                )
            elif self.attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=f"{self.backbone_name}_sse")

        # Apply global average pooling as the final layer of the backbone
        network_output = keras.layers.GlobalAveragePooling2D(
            name=self.backbone_name + "_global_avgpool"
        )(x)
        # Create the backbone model
        backbone_model = keras.Model(
            network_input, network_output, name=self.backbone_name
        )
        return backbone_model, network_input


class SingleCNNIndexed2D(CTLearnModel):
    """
    ``SingleCNN`` is a simple convolutional neural network model.

    This class extends the functionality of ``CTLearnModel`` by implementing
    methods to build a simple convolutional neural network model.
    """

    name = Unicode(
        "SingleCNNIndexed2D",
        help="Name of the model backbone.",
    ).tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 32, "number": 1},
            {"filters": 32, "number": 1},
            {"filters": 64, "number": 1},
            {"filters": 128, "number": 1},
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
        default_value={"size": (7,1), "strides": (1,1)},
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

    temporal_kernel_size = Int(
        default_value=3,
        allow_none=False,
        help="Temporal dimension for the convolutional kernel.",
    ).tag(config=True)

    def __init__(
        self,
        input_shape,
        indices,
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
            validate_trait_dict(layer, ["filters", "number"])
        # Validate the pooling parameters trait
        validate_trait_dict(self.pooling_parameters, ["size", "strides"])

        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"

        self.indices, self.mask = self.prepare_mask(indices)

        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        # Validate the head trait with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Build the fully connected head depending on the tasks
        self.logits = build_fully_connect_head(
            backbone_output, self.head_layers, self.head_activation_function, tasks
        )

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")


    def prepare_mask(self, neighbor_indices):
        """
        Prepares the mask from the neighbor array (shape: (L, K)).

        Parameters:
        neighbor_indices: array (numpy or similar) with shape (L, K), where L is the number of pixels 
                            and K is the maximum number of neighbors. -1 is used to indicate the absence of a neighbor.

        Returns:
        indices: tensor of shape (L, K) unchanged (using -1 in the masking logic).
        mask: tensor of shape (L, K) with 1.0 where the index is valid and 0.0 where it is -1.
        """
        indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
        mask = tf.cast(tf.not_equal(indices, -1), tf.float32)
        indices_for_gather = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), indices)
        indices = tf.expand_dims(indices, axis=0)  # (1, L, K)
        mask = tf.expand_dims(mask, axis=0)                     # (1, L, K)
        mask = tf.expand_dims(mask, axis=-1)   # (1, L, K, 1)

        return indices, mask

    def prepare_convolution(self, x):
        batch_size = tf.shape(x)[0]
        if not hasattr(self, "tiled_indices") or not hasattr(self, "tiled_mask"):
            self.tiled_indices = tf.tile(self.indices, [batch_size, 1, 1])  # shape (batch, L, K)
            self.tiled_mask = tf.tile(self.mask, [batch_size, 1, 1, 1])  # shape (batch, L, K, 1)

        neighbor_feats = tf.gather(x, self.tiled_indices, batch_dims=1) # shape (batch, L, K, T)
        neighbor_feats = neighbor_feats * tf.cast(tf.expand_dims(self.tiled_mask, axis=-1), dtype=neighbor_feats.dtype) #put to 0 the positions where there is no pixel
        return neighbor_feats


    def _build_backbone(self, input_shape):
        """
        Build the SingleCNN model backbone.

        Function to build the backbone of the SingleCNN model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the backbone of the SingleCNN model.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """
        # Define the input layer from the input shape
        network_input = keras.Input(input_shape, name="input")
        # Get model arcihtecture parameters for the backbone
        filters_list = [layer["filters"] for layer in self.architecture]
        numbers_list = [layer["number"] for layer in self.architecture]
        x =  tf.expand_dims(network_input, axis=-1)

        if self.batchnorm:
            x = keras.layers.BatchNormalization(momentum=0.99)(x)
        total_layers = len(filters_list)

        for i, (filters, number) in enumerate(
            zip(filters_list, numbers_list)
        ):
            for nr in range(number):
                x = self.prepare_convolution(x)
                x = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(7, self.temporal_kernel_size),
                    padding="valid",
                    activation="relu",
                    name=f"{self.backbone_name}_conv_{i+1}_{nr+1}",
                )(x)
                x = tf.squeeze(x, axis=2)

            if self.pooling_type is not None and i < total_layers - 1:
                if self.pooling_type == "max":
                    x = self.prepare_convolution(x)
                    x = keras.layers.MaxPool3D(
                        pool_size=(1, self.pooling_parameters["size"][0], self.pooling_parameters["size"][1]),
                        strides=(1, self.pooling_parameters["strides"][0], self.pooling_parameters["strides"][1]),
                        padding='valid',
                        name=f"{self.backbone_name}_pool_{i+1}",
                    )(x)
                elif self.pooling_type == "average":
                    x = self.prepare_convolution(x)
                    x = keras.layers.AveragePooling3D(
                        pool_size=(1, self.pooling_parameters["size"][0], self.pooling_parameters["size"][1]),
                        strides=(1, self.pooling_parameters["strides"][0], self.pooling_parameters["strides"][1]),
                        padding='valid',
                        name=f"{self.backbone_name}_pool_{i+1}",
                    )(x)
                x = tf.squeeze(x, axis=2)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        # bottleneck layer
        if self.bottleneck_filters is not None:

            x = keras.layers.Conv2D(
                filters=self.bottleneck_filters,
                kernel_size=1,
                padding="valid",
                activation="relu",
                name=f"{self.backbone_name}_bottleneck",
            )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        # # Attention mechanism
        # if self.attention is not None:
        #     if self.attention["mechanism"] == "Dual-SE":
        #         x = dual_squeeze_excite_block(
        #             x, self.attention["reduction_ratio"], name=f"{self.backbone_name}_dse"
        #         )
        #     elif self.attention["mechanism"] == "Channel-SE":
        #         x = channel_squeeze_excite_block(
        #             x, self.attention["reduction_ratio"], name=f"{self.backbone_name}_cse"
        #         )
        #     elif self.attention["mechanism"] == "Spatial-SE":
        #         x = spatial_squeeze_excite_block(x, name=f"{self.backbone_name}_sse")

        # Apply global average pooling as the final layer of the backbone
        network_output = keras.layers.GlobalAveragePooling2D(
            name=self.backbone_name + "_global_avgpool"
        )(x)
        # Create the backbone model
        backbone_model = keras.Model(
            network_input, network_output, name=self.backbone_name
        )
        return backbone_model, network_input

class ResNetIndexed2D(CTLearnModel):
    """
    ``ResNet`` is a residual neural network model.

    This class extends the functionality of ``CTLearnModel`` by implementing
    methods to build a residual neural network model.
    """

    name = Unicode(
        "ThinResNetIndexed2D",
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

    temporal_kernel_size = Int(
        default_value=3,
        allow_none=False,
        help="Temporal dimension for the convolutional kernel.",
    ).tag(config=True)

    def __init__(
        self,
        input_shape,
        tasks,
        indices,
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
            validate_trait_dict(self.init_layer, ["filters", "strides"])
        if self.init_max_pool is not None:
            validate_trait_dict(self.init_max_pool, ["size", "strides"])

        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"


        self.indices, self.mask = self.prepare_mask(indices)

        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        # Validate the head traits with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Build the fully connected head depending on the tasks
        self.logits = build_fully_connect_head(
            backbone_output, self.head_layers, self.head_activation_function, tasks
        )

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    def prepare_mask(self, neighbor_indices):
        """
        Prepares the mask from the neighbor array (shape: (L, K)).

        Parameters:
        neighbor_indices: array (numpy or similar) with shape (L, K), where L is the number of pixels 
                            and K is the maximum number of neighbors. -1 is used to indicate the absence of a neighbor.

        Returns:
        indices: tensor of shape (L, K) unchanged (using -1 in the masking logic).
        mask: tensor of shape (L, K) with 1.0 where the index is valid and 0.0 where it is -1.
        """
        indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
        mask = tf.cast(tf.not_equal(indices, -1), tf.float32)
        indices_for_gather = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), indices)
        indices = tf.expand_dims(indices, axis=0)  # (1, L, K)
        mask = tf.expand_dims(mask, axis=0)                     # (1, L, K)
        mask = tf.expand_dims(mask, axis=-1)   # (1, L, K, 1)

        return indices, mask

    def prepare_convolution(self, x):
        batch_size = tf.shape(x)[0]
        if not hasattr(self, "tiled_indices") or not hasattr(self, "tiled_mask"):
            self.tiled_indices = tf.tile(self.indices, [batch_size, 1, 1])  # shape (batch, L, K)
            self.tiled_mask = tf.tile(self.mask, [batch_size, 1, 1, 1])  # shape (batch, L, K, 1)

        neighbor_feats = tf.gather(x, self.tiled_indices, batch_dims=1) # shape (batch, L, K, T)
        neighbor_feats = neighbor_feats * tf.cast(tf.expand_dims(self.tiled_mask, axis=-1), dtype=neighbor_feats.dtype) #put to 0 the positions where there is no pixel
        return neighbor_feats


    def _build_backbone(self, input_shape):
        """
        Build the ResNet model backbone.

        Function to build the backbone of the ResNet model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the ResNet backbone.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """

        input_shape = input_shape + (1,)
        print(f"new input shape is {input_shape}")
        # Define the input layer from the input shape
        x = keras.Input(shape=input_shape, name="input")

        
        # Apply initial convolutional layer if specified
        # if self.init_layer is not None:
        #     x = self.prepare_convolution(x)
        #     x = keras.layers.Conv2D(
        #         filters=self.init_layer["filters"],
        #         kernel_size=(7, self.temporal_kernel_size),
        #         padding="valid",
        #         name=self.backbone_name + "_conv1_conv",
        #     )(x)
        #     x = tf.squeeze(x, axis=2)
            
        # # Apply max pooling if specified
        # if self.init_max_pool is not None:
        #     x = self.prepare_convolution(x)
        #     x = keras.layers.MaxPool2D(
        #         pool_size=(1, self.init_max_pool["size"][0], self.init_max_pool["size"][1]),
        #         strides=(1, self.init_max_pool["strides"][0], self.init_max_pool["strides"][1]),
        #         padding='valid',
        #         name=f"{self.backbone_name}_pool_{i+1}",
        #     )(x)          
        # Build the residual blocks
        engine_output = self._stacked_res_blocks(
            x,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention,
            name=self.backbone_name,
        )
        # Apply global average pooling as the final layer of the backbone

        network_output = keras.layers.GlobalAveragePooling2D(name=self.backbone_name + "_global_avgpool")(engine_output)
        # Create the backbone model
        backbone_model = keras.Model(
            x, network_output, name=self.backbone_name
        )
        return backbone_model, x

    def _stacked_res_blocks(
        self, inputs, architecture, residual_block_type, attention, name=None
    ):
        """
        Build a stack of residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block consists of a series of convolutional layers with skip connections.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input Keras layer to the residual blocks.
        architecture : list of dict
            List of dictionaries containing the architecture of the ResNet model, which includes:
            - Number of filters for the convolutional layers in the residual blocks.
            - Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block to use. Options are 'basic' or 'bottleneck'.
        attention : dict
            Dictionary containing the configuration parameters for the attention mechanism.
        name : str, optional
            Label for the model.

        Returns
        -------
        x : keras.layers.Layer
            Output Keras layer after passing through the stack of residual blocks.
        """

        # Get hyperparameters for the model architecture
        filters_list = [layer["filters"] for layer in architecture]
        blocks_list = [layer["blocks"] for layer in architecture]
        # Build the ResNet model
        x = self._stack_fn(
            inputs,
            filters_list[0],
            blocks_list[0],
            residual_block_type,
            stride=1,
            attention=attention,
            name=name + "_conv2",
        )
        for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
            x = self._stack_fn(
                x,
                filters,
                blocks,
                residual_block_type,
                attention=attention,
                name=name + "_conv" + str(i + 3),
            )
        return x

    def _stack_fn(
        self,
        inputs,
        filters,
        blocks,
        residual_block_type,
        stride=1,
        attention=None,
        name=None,
    ):
        """
        Stack residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block can be of different types (e.g., basic or bottleneck) and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual blocks.
        filters : int
            Number of filters for the bottleneck layer in a block.
        blocks : int
            Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block ('basic' or 'bottleneck').
        stride : int, optional
            Stride for the first layer in the first block. Default is 2.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Label for the stack. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor for the stacked blocks.
        """

        res_blocks = {
            "basic": self._basic_residual_block,
            "bottleneck": self._bottleneck_residual_block,
        }

        x = res_blocks[residual_block_type](
            inputs,
            filters,
            stride=stride,
            attention=attention,
            name=name + "_block1",
        )
        for i in range(2, blocks + 1):
            x = res_blocks[residual_block_type](
                x,
                filters,
                conv_shortcut=False,
                attention=attention,
                name=name + "_block" + str(i),
            )

        return x

    def _basic_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a basic residual block for the CTLearn model.

        This function constructs a basic residual block, which is a fundamental building block
        of ResNet architectures. The block consists of two convolutional layers with an optional
        convolutional shortcut, and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor after applying the residual block.
        """

        if conv_shortcut:

            shortcut = keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1,3),
                strides=stride,
                padding="same",
                name=name + "_0_conv",
            )(inputs)

        else:
            shortcut = inputs

        x = self.prepare_convolution(inputs)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(7, 2),
            padding="valid",
            activation="relu",
            name=name + "_1_conv",
        )(x)
        x = tf.squeeze(x, axis=2)

        x = self.prepare_convolution(x)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(7, 2),
            padding="valid",
            activation="relu",
            name=name + "_2_conv",
        )(x)
        x = tf.squeeze(x, axis=2)
        

        # # Attention mechanism
        # if attention is not None:
        #     if attention["mechanism"] == "Dual-SE":
        #         x = dual_squeeze_excite_block(
        #             x, attention["reduction_ratio"], name=name + "_dse"
        #         )
        #     elif attention["mechanism"] == "Channel-SE":
        #         x = channel_squeeze_excite_block(
        #             x, attention["reduction_ratio"], name=name + "_cse"
        #         )
        #     elif attention["mechanism"] == "Spatial-SE":
        #         x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x

    def _bottleneck_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a bottleneck residual block for the CTLearn model.

        This function constructs a bottleneck residual block, which is a fundamental building block of
        ResNet architectures. The block consists of three convolutional layers: a 1x1 convolution to reduce
        dimensionality, a 3x3 convolution for main computation, and another 1x1 convolution to restore dimensionality.
        It also includes an optional shortcut connection and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        output : keras.layers.Layer
            Output layer of the residual block.
        """

        # if conv_shortcut:
        shortcut = keras.layers.Conv2D(
            filters=4 * filters,
            kernel_size=(1,3),
            strides=(1,1),
            padding="valid",
            name=name + "_0_conv",
        )(inputs)
        # else:
        #     shortcut = inputs

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1,1),
            strides=stride,
            activation="relu",
            name=name + "_1_conv",
        )(inputs)

        x = self.prepare_convolution(x)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=(7, self.temporal_kernel_size),
            padding="valid",
            activation="relu",
            name=name + "_2_conv",
        )(x)
        x = tf.squeeze(x, axis=2)

        x = keras.layers.Conv2D(
            filters=4 * filters, kernel_size=(1,1), name=name + "_3_conv"
        )(x)

        # # Attention mechanism
        # if attention is not None:
        #     if attention["mechanism"] == "Dual-SE":
        #         x = dual_squeeze_excite_block(
        #             x, attention["reduction_ratio"], name=name + "_dse"
        #         )
        #     elif attention["mechanism"] == "Channel-SE":
        #         x = channel_squeeze_excite_block(
        #             x, attention["reduction_ratio"], name=name + "_cse"
        #         )
        #     elif attention["mechanism"] == "Spatial-SE":
        #         x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x







class SingleCNNIndexed(CTLearnModel):
    """
    SingleCNNIndexed is a simple CNN network that uses indexed convolutions and pooling.

    The expected input shape is (batch, L, channels), where L is the total number of pixels
    (e.g., 7987), and "indices" is a neighbor array of shape (L, K) (e.g., (7987,6)).
    """
    name = Unicode("SingleCNNIndexed", help="Name of the model backbone.").tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 32, "number": 1},
            {"filters": 32, "number": 1},
            {"filters": 64, "number": 1},
            {"filters": 128, "number": 1},
        ],
        allow_none=False,
        help="List of dictionaries containing the number of filters and number of repetitions."
    ).tag(config=True)

    pooling_type = CaselessStrEnum(
        ["max", "average"],
        default_value="max",
        allow_none=True,
        help="Type of pooling to apply."
    ).tag(config=True)

    pooling_parameters = Dict(
        default_value={"size": 2, "strides": 2},
        allow_none=True,
        help="Parameters for the pooling layers."
    ).tag(config=True)

    bottleneck_filters = Int(
        default_value=None,
        allow_none=True,
        help="Number of filters in the bottleneck layer.",
    ).tag(config=True)

    batchnorm = Bool(
        default_value=False,
        allow_none=False,
        help="Whether to apply batch normalization in convolutional layers."
    ).tag(config=True)

    def __init__(self, input_shape, tasks, indices, config=None, parent=None, **kwargs):
        """
        Args:
          input_shape: tuple, e.g., (7987, sequence_length)
          tasks: list of tasks (used to build the FC head)
          indices: neighbor array of shape (L, K), e.g., (7987, 6)
        """
        super().__init__(config=config, parent=parent, **kwargs)

        # Validate architecture
        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "number"])
        if self.pooling_parameters is not None:
            validate_trait_dict(self.pooling_parameters, ["size", "strides"])

        self.backbone_name = self.name + "_block"
        self.backbone_model, self.input_layer = self._build_backbone(input_shape, indices)
        backbone_output = self.backbone_model(self.input_layer)

        # Validate and build the FC head using build_fully_connect_head (defined in CTLearnModel)
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        self.logits = build_fully_connect_head(backbone_output, self.head_layers, self.head_activation_function, tasks)

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    def _build_backbone(self, input_shape, indices):
        """
        Builds the backbone using indexed layers.
        Assumes that input_shape is (L, channels), where L is the total number of pixels.
        """
        network_input = keras.Input(shape=input_shape, name="input")
        x = network_input

        x = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
        x = Conv1D(filters=64, kernel_size=3, padding="same", activation="relu")(x)

        if self.batchnorm:
            x = keras.layers.BatchNormalization()(x)

        filters_list = [layer["filters"] for layer in self.architecture]
        numbers_list = [layer["number"] for layer in self.architecture]
        total_layers = len(filters_list)

        for i, (filters, number) in enumerate(zip(filters_list, numbers_list)):
            for nr in range(number):
                x = IndexedConv2D(
                    out_channels=filters,
                    neighbor_indices=indices,
                    use_bias=True
                )(x)
                x = keras.layers.ReLU()(x)
            if self.pooling_type is not None and i < total_layers - 1:
                if self.pooling_type == "max":
                    x = IndexedMaxPool2D(neighbor_indices=indices)(x)
                elif self.pooling_type == "average":
                    x = IndexedAveragePool2D(neighbor_indices=indices)(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization()(x)

        if self.bottleneck_filters is not None:
            x = IndexedConv2D(
                filters=self.bottleneck_filters,
                neighbor_indices=indices,
                activation="relu",
                name=f"{self.backbone_name}_bottleneck",
            )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        if self.attention is not None:
            # if self.attention["mechanism"] == "Dual-SE":
            #     x = indexed_dual_squeeze_excite_block(x, indices, self.attention["reduction_ratio"],
            #                                           name=f"{self.backbone_name}_dse")
            # elif self.attention["mechanism"] == "Channel-SE":
            #     x = indexed_channel_squeeze_excite_block(x, indices, self.attention["reduction_ratio"],
            #                                              name=f"{self.backbone_name}_cse")
            # elif self.attention["mechanism"] == "Spatial-SE":
            #     x = indexed_spatial_squeeze_excite_block(x, indices, name=f"{self.backbone_name}_sse")
            x = indexed_attention_block(x, indices, ratio=4, name=f"{self.backbone_name}_attn")

        # Global average pooling to reduce L dimension to a single vector per sample
        # x = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
        x = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
        backbone_model = keras.Model(network_input, x, name=self.backbone_name)
        return backbone_model, network_input




class SingleCNNIndexed3D(CTLearnModel):
    """
    SingleCNNIndexed3D is a CNN that uses a custom 3D indexed convolution.
    
    The expected input shape is (batch, L, T, channels), where:
      - L is the number of pixels,
      - T is the number of time samples, and
      - channels is the number of features per pixel.
    
    The layer uses a neighbor index array (of shape (L, K)) to gather each pixel’s
    spatial neighbors and applies a kernel that slides over a temporal window.
    """
    name = Unicode("SingleCNNIndexed3D", help="Name of the model backbone.").tag(config=True)
    
    # Define the convolutional architecture as a list of dicts.
    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 32, "number": 1},
            {"filters": 32, "number": 1},
            {"filters": 64, "number": 1},
            {"filters": 128, "number": 1},
        ],
        allow_none=False,
        help="List of dictionaries containing the number of filters and number of repetitions."
    ).tag(config=True)
    temporal_kernel_size = Int(
        default_value=3, 
        allow_none=False,
        help="Temporal kernel size for the convolutional layers."
    ).tag(config=True)
    pooling_type = CaselessStrEnum(
        ["max", "average"],
        default_value="max",
        allow_none=True,
        help="Type of pooling to apply."
    ).tag(config=True)
    
    pooling_parameters = Dict(
        default_value={"size": 2, "strides": 2},
        allow_none=True,
        help="Parameters for the pooling layers."
    ).tag(config=True)
    
    bottleneck_filters = Int(
        default_value=None,
        allow_none=True,
        help="Number of filters in the bottleneck layer."
    ).tag(config=True)
    
    batchnorm = Bool(
        default_value=False,
        allow_none=False,
        help="Whether to apply batch normalization in convolutional layers."
    ).tag(config=True)
    
    def __init__(self, input_shape, tasks, indices, config=None, parent=None, **kwargs):
        """
        Args:
          input_shape: tuple, e.g. (L, T, channels)
          tasks: list of tasks (used to build the FC head)
          indices: neighbor array of shape (L, K) (e.g., (7987,7) if including the pixel itself)
          temporal_kernel_size: int, the temporal extent of the kernel (e.g., 3)
        """

        super().__init__(config=config, parent=parent, **kwargs)
        
        # Validate architecture.
        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "number"])
        if self.pooling_parameters is not None:
            validate_trait_dict(self.pooling_parameters, ["size", "strides"])
        
        self.backbone_name = self.name + "_block"
        self.backbone_model, self.input_layer = self._build_backbone(input_shape, indices)
        backbone_output = self.backbone_model(self.input_layer)
        
        # Validate and build the FC head using build_fully_connect_head (defined in CTLearnModel).
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        self.logits = build_fully_connect_head(backbone_output, self.head_layers, self.head_activation_function, tasks)
        
        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")
    
    def _build_backbone(self, input_shape, indices):
        """
        Builds the backbone using the 3D indexed convolution.
        
        Assumes input_shape is (L, T, channels).
        """
        network_input = keras.Input(shape=input_shape, name="input")  # shape: (L, T)
        x = tf.expand_dims(network_input, axis=-1)  # now shape: (L, T, 1)
        
        # Optionally, you could apply some initial temporal processing via Conv1D on the T dimension.
        # Here, we directly pass x to the indexed convolution layers.
        
        filters_list = [layer["filters"] for layer in self.architecture]
        numbers_list = [layer["number"] for layer in self.architecture]
        total_layers = len(filters_list)
        
        for i, (filters, number) in enumerate(zip(filters_list, numbers_list)):
            for _ in range(number):
                # Each IndexedConv3D layer operates on the full (L, T, channels) input.
                x = IndexedConv3D(
                    out_channels=filters,
                    temporal_kernel_size=self.temporal_kernel_size,
                    indices=indices,
                    use_bias=True
                )(x)
                x = keras.layers.ReLU()(x)
            # Optional pooling: if you have a corresponding 3D pooling (or you could pool only spatially).
            # Here, we use GlobalAveragePooling2D later, so you might skip intermediate pooling.
            if self.batchnorm:
                x = keras.layers.BatchNormalization()(x)
        
        if self.bottleneck_filters is not None:
            x = IndexedConv3D(
                out_channels=self.bottleneck_filters,
                temporal_kernel_size=self.temporal_kernel_size,
                indices=indices,
                use_bias=True,
            )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)
        

        # if self.attention is not None:
        #     if self.attention["mechanism"] == "Dual-SE":
        #         x = indexed_dual_squeeze_excite_block_3d(x, indices, self.attention["reduction_ratio"],
        #                                               name=f"{self.backbone_name}_dse")
        #     elif self.attention["mechanism"] == "Channel-SE":
        #         x = indexed_channel_squeeze_excite_block_3d(x, indices, self.attention["reduction_ratio"],
        #                                                  name=f"{self.backbone_name}_cse")
        #     elif self.attention["mechanism"] == "Spatial-SE":
        #         x = indexed_spatial_squeeze_excite_block_3d(x, indices, name=f"{self.backbone_name}_sse")



        # Global pooling: since x now has shape (batch, L, T_new, filters),
        # we use GlobalAveragePooling2D to aggregate over both L and the new time dimension.
        # x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

        # como los cambios son muy pequeños probamos con max
        x = keras.layers.GlobalMaxPooling2D(name="global_max_pool")(x)
        
        backbone_model = keras.Model(network_input, x, name=self.backbone_name)
        return backbone_model, network_input



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

    use_3d_conv = Bool(
        default_value=False,
        help="If True, use 3D convolutions (expects input shape (H, W, T) and adds a channel dimension)."
    ).tag(config=True)

    def __init__(
        self,
        input_shape,
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

        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"

        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        # Validate the head traits with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Build the fully connected head depending on the tasks
        self.logits = build_fully_connect_head(
            backbone_output, self.head_layers, self.head_activation_function, tasks
        )

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    def _build_backbone(self, input_shape):
        """
        Build the ResNet model backbone.

        Function to build the backbone of the ResNet model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the ResNet backbone.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """
        if self.use_3d_conv:
            input_shape = input_shape + (1,)
            print(f"new input shape is {input_shape}")
            # Define the input layer from the input shape
        network_input = keras.Input(shape=input_shape, name="input")
        # Apply initial padding if specified
        if self.init_padding > 0:
            if self.use_3d_conv:
                network_input = keras.layers.ZeroPadding3D(
                    padding=self.init_padding,
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_padding",
                )(network_input)
            else:
                network_input = keras.layers.ZeroPadding2D(
                    padding=self.init_padding,
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_padding",
                )(network_input)
        # Apply initial convolutional layer if specified
        if self.init_layer is not None:
            if self.use_3d_conv:
                network_input = keras.layers.Conv3D(
                    filters=self.init_layer["filters"],
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_conv1_conv",
                )(network_input)
            else:
                network_input = keras.layers.Conv2D(
                    filters=self.init_layer["filters"],
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_conv1_conv",
                )(network_input)
        # Apply max pooling if specified
        if self.init_max_pool is not None:
            if self.use_3d_conv:
                network_input = keras.layers.MaxPool3D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool",
                )(network_input)
            else:
                network_input = keras.layers.MaxPool2D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool",
                )(network_input)               
        # Build the residual blocks
        engine_output = self._stacked_res_blocks(
            network_input,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention,
            name=self.backbone_name,
        )
        # Apply global average pooling as the final layer of the backbone
        if self.use_3d_conv:
            network_output = keras.layers.GlobalAveragePooling3D(name=self.backbone_name + "_global_avgpool")(engine_output)
        else:
            network_output = keras.layers.GlobalAveragePooling2D(name=self.backbone_name + "_global_avgpool")(engine_output)
        # Create the backbone model
        backbone_model = keras.Model(
            network_input, network_output, name=self.backbone_name
        )
        return backbone_model, network_input

    def _stacked_res_blocks(
        self, inputs, architecture, residual_block_type, attention, name=None
    ):
        """
        Build a stack of residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block consists of a series of convolutional layers with skip connections.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input Keras layer to the residual blocks.
        architecture : list of dict
            List of dictionaries containing the architecture of the ResNet model, which includes:
            - Number of filters for the convolutional layers in the residual blocks.
            - Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block to use. Options are 'basic' or 'bottleneck'.
        attention : dict
            Dictionary containing the configuration parameters for the attention mechanism.
        name : str, optional
            Label for the model.

        Returns
        -------
        x : keras.layers.Layer
            Output Keras layer after passing through the stack of residual blocks.
        """

        # Get hyperparameters for the model architecture
        filters_list = [layer["filters"] for layer in architecture]
        blocks_list = [layer["blocks"] for layer in architecture]
        # Build the ResNet model
        x = self._stack_fn(
            inputs,
            filters_list[0],
            blocks_list[0],
            residual_block_type,
            stride=1,
            attention=attention,
            name=name + "_conv2",
        )
        for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
            x = self._stack_fn(
                x,
                filters,
                blocks,
                residual_block_type,
                attention=attention,
                name=name + "_conv" + str(i + 3),
            )
        return x

    def _stack_fn(
        self,
        inputs,
        filters,
        blocks,
        residual_block_type,
        stride=2,
        attention=None,
        name=None,
    ):
        """
        Stack residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block can be of different types (e.g., basic or bottleneck) and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual blocks.
        filters : int
            Number of filters for the bottleneck layer in a block.
        blocks : int
            Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block ('basic' or 'bottleneck').
        stride : int, optional
            Stride for the first layer in the first block. Default is 2.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Label for the stack. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor for the stacked blocks.
        """

        res_blocks = {
            "basic": self._basic_residual_block,
            "bottleneck": self._bottleneck_residual_block,
        }

        x = res_blocks[residual_block_type](
            inputs,
            filters,
            stride=stride,
            attention=attention,
            name=name + "_block1",
        )
        for i in range(2, blocks + 1):
            x = res_blocks[residual_block_type](
                x,
                filters,
                conv_shortcut=False,
                attention=attention,
                name=name + "_block" + str(i),
            )

        return x

    def _basic_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a basic residual block for the CTLearn model.

        This function constructs a basic residual block, which is a fundamental building block
        of ResNet architectures. The block consists of two convolutional layers with an optional
        convolutional shortcut, and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor after applying the residual block.
        """

        if conv_shortcut:
            if self.use_3d_conv:
                shortcut = keras.layers.Conv3D(
                    filters=filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    name=name + "_0_conv",
                )(inputs)
            else:
                shortcut = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    name=name + "_0_conv",
                )(inputs)
        else:
            shortcut = inputs

        if self.use_3d_conv:
            x = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                activation="relu",
                name=name + "_1_conv",
            )(inputs)
            x = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=name + "_2_conv",
            )(x)
        else:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                activation="relu",
                name=name + "_1_conv",
            )(inputs)
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=name + "_2_conv",
            )(x)

        # Attention mechanism
        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_dse"
                )
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_cse"
                )
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x

    def _bottleneck_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a bottleneck residual block for the CTLearn model.

        This function constructs a bottleneck residual block, which is a fundamental building block of
        ResNet architectures. The block consists of three convolutional layers: a 1x1 convolution to reduce
        dimensionality, a 3x3 convolution for main computation, and another 1x1 convolution to restore dimensionality.
        It also includes an optional shortcut connection and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        output : keras.layers.Layer
            Output layer of the residual block.
        """

        if conv_shortcut:
            shortcut = keras.layers.Conv2D(
                filters=4 * filters,
                kernel_size=1,
                strides=stride,
                name=name + "_0_conv",
            )(inputs)
        else:
            shortcut = inputs

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            activation="relu",
            name=name + "_1_conv",
        )(inputs)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=name + "_2_conv",
        )(x)
        x = keras.layers.Conv2D(
            filters=4 * filters, kernel_size=1, name=name + "_3_conv"
        )(x)

        # Attention mechanism
        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_dse"
                )
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_cse"
                )
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x







# IndexedResNet class using indexed layers (for flattened data) and attention.
class IndexedResNet(CTLearnModel):
    """
    IndexedResNet is a residual neural network model using indexed convolutions.

    This class extends CTLearnModel by implementing a ResNet-like model that works on
    flattened pixel data (batch, L, channels), where L is the number of pixels.
    The neighbor index array (of shape (L, K)) is used by the IndexedConv2D and pooling layers.
    """
    name = Unicode(
        "IndexedResNet",
        help="Name of the model backbone.",
    ).tag(config=True)

    # Initial layer configuration (kept for compatibility, though not used by IndexedConv2D)
    init_layer = Dict(
        default_value={'filters': 64, 'kernel_size': 7, 'strides': 2},
        allow_none=True,
        help="Parameters for the first convolutional layer.",
    ).tag(config=True)

    init_max_pool = Dict(
        default_value={'size': 3, 'strides': 2},
        allow_none=True,
        help="Parameters for the first max pooling layer.",
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
            "List of dictionaries defining the number of filters and residual blocks per stage. "
            "Example: [{'filters': 12, 'blocks': 2}, ...]."
        ),
    ).tag(config=True)

    def __init__(self, input_shape, tasks, indices, config=None, parent=None, **kwargs):
        """
        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch, L, channels), e.g., (7987, sequence_length).
        tasks : list
            List of tasks used to build the fully connected head.
        indices : array
            Neighbor index array of shape (L, K) (e.g., (7987, 6)).
        """
        super().__init__(config=config, parent=parent, **kwargs)

        # Validate architecture configuration
        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "blocks"])
        if self.init_layer is not None:
            validate_trait_dict(self.init_layer, ["filters", "kernel_size", "strides"])
        if self.init_max_pool is not None:
            validate_trait_dict(self.init_max_pool, ["size", "strides"])

        self.backbone_name = self.name + "_backbone"

        # Build the backbone using indexed layers
        self.backbone_model, self.input_layer = self._build_backbone(input_shape, indices)
        backbone_output = self.backbone_model(self.input_layer)

        # Validate head configuration and build the fully connected head
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        self.logits = build_fully_connect_head(backbone_output, self.head_layers, self.head_activation_function, tasks)

        self.model = keras.Model(self.input_layer, self.logits, name="IndexedResNet_model")

    def _build_backbone(self, input_shape, indices):
        """
        Builds the backbone of the IndexedResNet model.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (L, channels).
        indices : array
            Neighbor index array of shape (L, K).

        Returns
        -------
        backbone_model : keras.Model
            Model representing the backbone.
        network_input : keras.Input
            Input layer for the backbone.
        """
        # For indexed data, the input is already flattened.
        network_input = keras.Input(shape=input_shape, name="input")
        x = network_input

        # Optionally apply an initial indexed convolution (if defined)
        if self.init_layer is not None:
            x = IndexedConv2D(
                out_channels=self.init_layer["filters"],
                neighbor_indices=indices,
                use_bias=True,
                name=self.backbone_name + "_conv1"
            )(x)
            x = keras.layers.BatchNormalization(name=self.backbone_name + "_conv1_bn")(x)
            x = keras.layers.ReLU(name=self.backbone_name + "_conv1_relu")(x)
        # Optionally apply initial indexed max pooling
        if self.init_max_pool is not None:
            x = IndexedMaxPool2D(neighbor_indices=indices, name=self.backbone_name + "_pool1")(x)

        # Build stacked residual blocks with indexed convolutions.
        x = self._stacked_res_blocks(x, self.architecture, self.residual_block_type, self.attention, indices, name=self.backbone_name)

        # Global average pooling to reduce L (number of pixels) to a single vector per sample.
        x = keras.layers.GlobalAveragePooling1D(name=self.backbone_name + "_global_avgpool")(x)
        backbone_model = keras.Model(network_input, x, name=self.backbone_name)
        return backbone_model, network_input

    def _stacked_res_blocks(self, inputs, architecture, residual_block_type, attention, indices, name=None):
        """
        Stacks residual blocks for the IndexedResNet model.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual blocks.
        architecture : list of dict
            List of dictionaries defining the number of filters and residual blocks per stage.
        residual_block_type : str
            Type of residual block ("basic" or "bottleneck").
        attention : dict
            Configuration for the attention mechanism.
        indices : array
            Neighbor index array of shape (L, K).
        name : str, optional
            Name prefix for the blocks.

        Returns
        -------
        keras.layers.Layer
            Output tensor after stacking the residual blocks.
        """
        x = inputs
        for i, layer_params in enumerate(architecture):
            filters = layer_params["filters"]
            blocks = layer_params["blocks"]
            x = self._stack_fn(x, filters, blocks, residual_block_type, attention, indices, name=name + f"_stage{i+1}")
        return x

    def _stack_fn(self, inputs, filters, blocks, residual_block_type, attention, indices, stride=2, name=None):
        """
        Creates a stack of residual blocks.
        
        The first block in the stack applies downsampling (stride=stride) and the rest use stride=1.
        
        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor.
        filters : int
            Number of filters.
        blocks : int
            Number of residual blocks to stack.
        residual_block_type : str
            "basic" or "bottleneck".
        attention : dict
            Attention configuration.
        indices : array
            Neighbor index array (L, K).
        stride : int, optional
            Stride for the first block. Default is 2.
        name : str, optional
            Name for the block stack.

        Returns
        -------
        keras.layers.Layer
            Output tensor.
        """
        # Use bottleneck residual blocks (as in the original ResNet) for this example.
        x = self._bottleneck_residual_block(inputs, filters, stride=stride, attention=attention, indices=indices, name=name + "_block1")
        for i in range(1, blocks):
            x = self._bottleneck_residual_block(x, filters, stride=1, conv_shortcut=False, attention=attention, indices=indices, name=name + f"_block{i+1}")
        return x

    def _bottleneck_residual_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, indices=None, name=None):
        """
        Builds a bottleneck residual block using IndexedConv2D with attention.

        Parameters
        ----------
        x : keras.layers.Layer
            Input tensor.
        filters : int
            Number of filters for the bottleneck.
        kernel_size : int, optional
            (Not used for indexed convolutions) Default is 3.
        stride : int, optional
            Stride for the convolution. Default is 1.
        conv_shortcut : bool, optional
            Whether to apply a shortcut convolution. Default is True.
        attention : dict, optional
            Attention configuration.
        indices : array
            Neighbor index array (L, K).
        name : str, optional
            Name for the residual block.

        Returns
        -------
        keras.layers.Layer
            Output tensor of the bottleneck block.
        """
        shortcut = x
        if conv_shortcut or (stride > 1 or int(x.shape[-1]) != filters * 4):
            shortcut = IndexedConv2D(
                out_channels=filters * 4,
                neighbor_indices=indices,
                use_bias=True,
                name=name + "_0_conv"
            )(x)
            shortcut = keras.layers.BatchNormalization(name=name + "_0_bn")(shortcut)

        # 1x1 convolution to reduce dimensions
        y = IndexedConv2D(
            out_channels=filters,
            neighbor_indices=indices,
            use_bias=True,
            name=name + "_1_conv"
        )(x)
        y = keras.layers.BatchNormalization(name=name + "_1_bn")(y)
        y = keras.layers.ReLU(name=name + "_1_relu")(y)

        # 3x3 indexed convolution
        y = IndexedConv2D(
            out_channels=filters,
            neighbor_indices=indices,
            use_bias=True,
            name=name + "_2_conv"
        )(y)
        y = keras.layers.BatchNormalization(name=name + "_2_bn")(y)
        y = keras.layers.ReLU(name=name + "_2_relu")(y)

        # 1x1 convolution to restore dimensions
        y = IndexedConv2D(
            out_channels=filters * 4,
            neighbor_indices=indices,
            use_bias=True,
            name=name + "_3_conv"
        )(y)
        y = keras.layers.BatchNormalization(name=name + "_3_bn")(y)

        # # Apply attention if configured
        # if attention is not None:
        #     if attention["mechanism"] == "Dual-SE":
        #         y = indexed_dual_squeeze_excite_block(y, indices, ratio=attention["reduction_ratio"], name=name + "_dse")
        #     elif attention["mechanism"] == "Channel-SE":
        #         y = indexed_channel_squeeze_excite_block(y, indices, ratio=attention["reduction_ratio"], name=name + "_cse")
        #     elif attention["mechanism"] == "Spatial-SE":
        #         y = indexed_spatial_squeeze_excite_block(y, indices, name=name + "_sse")

        y = keras.layers.Add(name=name + "_add")([shortcut, y])
        y = keras.layers.ReLU(name=name + "_out")(y)
        return y


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

    use_3d_conv = Bool(
        default_value=False,
        help="If True, use 3D convolutions (expects input shape (H, W, T) and adds a channel dimension)."
    ).tag(config=True)

    def __init__(
        self,
        input_shape,
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

        # Construct the name of the backbone model by appending "_block" to the model name
        self.backbone_name = self.name + "_block"

        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        # Validate the head traits with the provided tasks
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        # Build the fully connected head depending on the tasks
        self.logits = build_fully_connect_head(
            backbone_output, self.head_layers, self.head_activation_function, tasks
        )

        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    def _build_backbone(self, input_shape):
        """
        Build the ResNet model backbone.

        Function to build the backbone of the ResNet model using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the ResNet backbone.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """
        if self.use_3d_conv:
            input_shape = input_shape + (1,)
            print(f"new input shape is {input_shape}")
            # Define the input layer from the input shape
        network_input = keras.Input(shape=input_shape, name="input")
        # Apply initial padding if specified
        if self.init_padding > 0:
            if self.use_3d_conv:
                network_input = keras.layers.ZeroPadding3D(
                    padding=self.init_padding,
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_padding",
                )(network_input)
            else:
                network_input = keras.layers.ZeroPadding2D(
                    padding=self.init_padding,
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_padding",
                )(network_input)
        # Apply initial convolutional layer if specified
        if self.init_layer is not None:
            if self.use_3d_conv:
                network_input = keras.layers.Conv3D(
                    filters=self.init_layer["filters"],
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_conv1_conv",
                )(network_input)
            else:
                network_input = keras.layers.Conv2D(
                    filters=self.init_layer["filters"],
                    kernel_size=self.init_layer["kernel_size"],
                    strides=self.init_layer["strides"],
                    name=self.backbone_name + "_conv1_conv",
                )(network_input)
        # Apply max pooling if specified
        if self.init_max_pool is not None:
            if self.use_3d_conv:
                network_input = keras.layers.MaxPool3D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool",
                )(network_input)
            else:
                network_input = keras.layers.MaxPool2D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool",
                )(network_input)               
        # Build the residual blocks
        engine_output = self._stacked_res_blocks(
            network_input,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention,
            name=self.backbone_name,
        )
        # Apply global average pooling as the final layer of the backbone
        if self.use_3d_conv:
            network_output = keras.layers.GlobalAveragePooling3D(name=self.backbone_name + "_global_avgpool")(engine_output)
        else:
            network_output = keras.layers.GlobalAveragePooling2D(name=self.backbone_name + "_global_avgpool")(engine_output)
        # Create the backbone model
        backbone_model = keras.Model(
            network_input, network_output, name=self.backbone_name
        )
        return backbone_model, network_input

    def _stacked_res_blocks(
        self, inputs, architecture, residual_block_type, attention, name=None
    ):
        """
        Build a stack of residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block consists of a series of convolutional layers with skip connections.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input Keras layer to the residual blocks.
        architecture : list of dict
            List of dictionaries containing the architecture of the ResNet model, which includes:
            - Number of filters for the convolutional layers in the residual blocks.
            - Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block to use. Options are 'basic' or 'bottleneck'.
        attention : dict
            Dictionary containing the configuration parameters for the attention mechanism.
        name : str, optional
            Label for the model.

        Returns
        -------
        x : keras.layers.Layer
            Output Keras layer after passing through the stack of residual blocks.
        """

        # Get hyperparameters for the model architecture
        filters_list = [layer["filters"] for layer in architecture]
        blocks_list = [layer["blocks"] for layer in architecture]
        # Build the ResNet model
        x = self._stack_fn(
            inputs,
            filters_list[0],
            blocks_list[0],
            residual_block_type,
            stride=1,
            attention=attention,
            name=name + "_conv2",
        )
        for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
            x = self._stack_fn(
                x,
                filters,
                blocks,
                residual_block_type,
                attention=attention,
                name=name + "_conv" + str(i + 3),
            )
        return x

    def _stack_fn(
        self,
        inputs,
        filters,
        blocks,
        residual_block_type,
        stride=2,
        attention=None,
        name=None,
    ):
        """
        Stack residual blocks for the CTLearn model.

        This function constructs a stack of residual blocks, which are used to build the backbone of the CTLearn model.
        Each residual block can be of different types (e.g., basic or bottleneck) and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual blocks.
        filters : int
            Number of filters for the bottleneck layer in a block.
        blocks : int
            Number of residual blocks to stack.
        residual_block_type : str
            Type of residual block ('basic' or 'bottleneck').
        stride : int, optional
            Stride for the first layer in the first block. Default is 2.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Label for the stack. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor for the stacked blocks.
        """

        res_blocks = {
            "basic": self._basic_residual_block,
            "bottleneck": self._bottleneck_residual_block,
        }

        x = res_blocks[residual_block_type](
            inputs,
            filters,
            stride=stride,
            attention=attention,
            name=name + "_block1",
        )
        for i in range(2, blocks + 1):
            x = res_blocks[residual_block_type](
                x,
                filters,
                conv_shortcut=False,
                attention=attention,
                name=name + "_block" + str(i),
            )

        return x

    def _basic_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a basic residual block for the CTLearn model.

        This function constructs a basic residual block, which is a fundamental building block
        of ResNet architectures. The block consists of two convolutional layers with an optional
        convolutional shortcut, and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        keras.layers.Layer
            Output tensor after applying the residual block.
        """

        if conv_shortcut:
            if self.use_3d_conv:
                shortcut = keras.layers.Conv3D(
                    filters=filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    name=name + "_0_conv",
                )(inputs)
            else:
                shortcut = keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    name=name + "_0_conv",
                )(inputs)
        else:
            shortcut = inputs

        if self.use_3d_conv:
            x = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                activation="relu",
                name=name + "_1_conv",
            )(inputs)
            x = keras.layers.Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=name + "_2_conv",
            )(x)
        else:
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                activation="relu",
                name=name + "_1_conv",
            )(inputs)
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=name + "_2_conv",
            )(x)

        # Attention mechanism
        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_dse"
                )
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_cse"
                )
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x

    def _bottleneck_residual_block(
        self,
        inputs,
        filters,
        kernel_size=3,
        stride=1,
        conv_shortcut=True,
        attention=None,
        name=None,
    ):
        """
        Build a bottleneck residual block for the CTLearn model.

        This function constructs a bottleneck residual block, which is a fundamental building block of
        ResNet architectures. The block consists of three convolutional layers: a 1x1 convolution to reduce
        dimensionality, a 3x3 convolution for main computation, and another 1x1 convolution to restore dimensionality.
        It also includes an optional shortcut connection and can include attention mechanisms.

        Parameters
        ----------
        inputs : keras.layers.Layer
            Input tensor to the residual block.
        filters : int
            Number of filters for the convolutional layers.
        kernel_size : int, optional
            Size of the convolutional kernel. Default is 3.
        stride : int, optional
            Stride for the convolutional layers. Default is 1.
        conv_shortcut : bool, optional
            Whether to use a convolutional layer for the shortcut connection. Default is True.
        attention : dict, optional
            Configuration parameters for the attention mechanism. Default is None.
        name : str, optional
            Name for the residual block. Default is None.

        Returns
        -------
        output : keras.layers.Layer
            Output layer of the residual block.
        """

        if conv_shortcut:
            shortcut = keras.layers.Conv2D(
                filters=4 * filters,
                kernel_size=1,
                strides=stride,
                name=name + "_0_conv",
            )(inputs)
        else:
            shortcut = inputs

        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride,
            activation="relu",
            name=name + "_1_conv",
        )(inputs)
        x = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=name + "_2_conv",
        )(x)
        x = keras.layers.Conv2D(
            filters=4 * filters, kernel_size=1, name=name + "_3_conv"
        )(x)

        # Attention mechanism
        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_dse"
                )
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(
                    x, attention["reduction_ratio"], name=name + "_cse"
                )
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)

        return x


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
        input_shape,
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

        # Load the model from the specified path
        self.model = keras.saving.load_model(self.load_model_from)
        # Build the ResNet model backbone
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        # Load the fully connected head from the loaded model or build a new one
        if self.overwrite_head:
            backbone_output = self.backbone_model(self.input_layer)
            # Validate the head trait with the provided tasks
            validate_trait_dict(self.head_layers, tasks)
            # Build the fully connected head depending on the tasks
            self.logits = build_fully_connect_head(
                backbone_output, self.head_layers, self.head_activation_function, tasks
            )
            self.model = keras.Model(
                self.input_layer, self.logits, name="CTLearn_model"
            )

    def _build_backbone(self, input_shape):
        """
        Build the LoadedModel backbone.

        Function to build the backbone of the LoadedModel using the specified parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the LoadedModel backbone.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """

        # Define the input layer from the input shape
        network_input = keras.Input(shape=input_shape, name="input")
        # Set the backbone model to be trainable or not
        for layer in self.model.layers:
            if layer.name.endswith("_block"):
                backbone_layer = self.model.get_layer(layer.name)
                self.backbone_name = backbone_layer.name
                backbone_layer.trainable = self.trainable_backbone
        network_output = backbone_layer(network_input)
        # Create the backbone model
        backbone_model = keras.Model(
            network_input, network_output, name=self.backbone_name
        )
        return backbone_model, network_input
