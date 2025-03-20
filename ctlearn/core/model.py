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

class IndexedConv2D(keras.layers.Layer):
    """
    Convolución indexada en TensorFlow que opera solo en píxeles con suficientes vecinos.

    Parámetros:
      - out_channels: número de filtros de salida.
      - neighbor_indices: array (N, K) con los índices de los vecinos.
      - use_bias: si se usa sesgo.
    """
    def __init__(self, out_channels, neighbor_indices, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.neighbor_indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
        self.num_neighbors = self.neighbor_indices.shape[-1]  # K (vecinos por píxel)
        self.use_bias = use_bias

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.num_neighbors, in_channels, self.out_channels),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        in_channels = inputs.shape[-1]

        # Aplanamos la imagen: (batch, N, in_channels), donde N = H*W
        flat_inputs = tf.reshape(inputs, (batch_size, -1, in_channels))

        # Filtrar los píxeles con suficientes vecinos
        valid_mask = tf.reduce_all(self.neighbor_indices >= 0, axis=1)  # Solo píxeles con K vecinos válidos
        valid_indices = tf.boolean_mask(self.neighbor_indices, valid_mask)  # Filtrar índices válidos
        flat_valid_inputs = tf.boolean_mask(flat_inputs, valid_mask, axis=1)  # Filtrar solo píxeles válidos

        # Obtener características de vecinos: (batch, N_valid, num_neighbors, in_channels)
        neighbor_features = tf.gather(flat_valid_inputs, valid_indices, batch_dims=1)

        # Aplicar convolución indexada
        out = tf.einsum('bnki,nio->bno', neighbor_features, self.kernel)  # (batch, N_valid, out_channels)

        if self.use_bias:
            out = out + self.bias  # Agregar sesgo

        # Reconstruir salida (manteniendo solo los píxeles válidos)
        out_full = tf.zeros((batch_size, H * W, self.out_channels), dtype=out.dtype)
        out_full = tf.tensor_scatter_nd_update(out_full, tf.where(valid_mask)[:, None], out)

        return tf.reshape(out_full, (batch_size, H, W, self.out_channels))

# ================= Modelo ResNet con Convolución Indexada =================
class ThinResNetIndexed(CTLearnModel):
    """
    ThinResNetIndexed es una versión de ResNet que utiliza convolución indexada en TensorFlow.
    
    Se espera que se proporcione un tensor de índices (neighbor_indices) que defina, para cada píxel "válido"
    (es decir, que tenga la cantidad completa de vecinos), los índices de sus vecinos.
    
    La salida de la capa IndexedConv2D será de forma (batch, H_valid, W_valid, out_channels), 
    donde H_valid y W_valid corresponden al área sin bordes (similar a una convolución "valid").
    """
    name = Unicode("ThinResNetIndexed", help="Nombre del backbone del modelo.").tag(config=True)

    init_layer = Dict(
        default_value=None,
        allow_none=True,
        help="Parámetros para la primera capa de convolución indexada. Ejemplo: {'filters': 64, 'kernel_size': 7, 'strides': 2}."
    ).tag(config=True)

    init_max_pool = Dict(
        default_value=None,
        allow_none=True,
        help="Parámetros para la primera capa de max pooling. Ejemplo: {'size': 3, 'strides': 2}."
    ).tag(config=True)

    residual_block_type = CaselessStrEnum(
        ["basic", "bottleneck"],
        default_value="bottleneck",
        allow_none=False,
        help="Tipo de bloque residual a utilizar.",
    ).tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 48, "blocks": 2, "kernel_size": 3},
            {"filters": 96, "blocks": 3, "kernel_size": 3},
            {"filters": 128, "blocks": 3, "kernel_size": 3},
            {"filters": 256, "blocks": 3, "kernel_size": 3},
        ],
        allow_none=False,
        help="Lista de diccionarios con número de filtros, bloques residuales y kernel sizes."
    ).tag(config=True)

    use_3d_conv = Bool(
        default_value=False,
        help="Si es True, se usan convoluciones 3D (se espera input shape (H, W, T) y se añade dimensión de canal)."
    ).tag(config=True)

    # Se asume que head_layers y head_activation_function vienen definidas en CTLearnModel

    def __init__(self, input_shape, tasks, neighbor_indices, output_shape, config=None, parent=None, **kwargs):
        """
        Parámetros:
          - input_shape: tupla (H, W, channels) o (H, W, T) para modo 3D.
          - tasks: lista de tareas.
          - neighbor_indices: tensor o array numpy de forma (N_valid, K) con los índices de vecinos para cada píxel válido.
          - output_shape: tupla (H_valid, W_valid) para la salida de las IndexedConv2D.
        """
        super().__init__(config=config, parent=parent, **kwargs)

        for layer in self.architecture:
            validate_trait_dict(layer, ["filters", "blocks", "kernel_size"])
        if self.init_layer is not None:
            validate_trait_dict(self.init_layer, ["filters", "kernel_size", "strides"])
        if self.init_max_pool is not None:
            validate_trait_dict(self.init_max_pool, ["size", "strides"])

        self.neighbor_indices = neighbor_indices  # Tensor de índices de vecinos
        self.output_shape = output_shape  # (H_valid, W_valid)

        self.backbone_name = self.name + "_block"
        self.backbone_model, self.input_layer = self._build_backbone(input_shape)
        backbone_output = self.backbone_model(self.input_layer)
        validate_trait_dict(self.head_layers, tasks)
        validate_trait_dict(self.head_activation_function, tasks)
        self.logits = build_fully_connect_head(backbone_output, self.head_layers, self.head_activation_function, tasks)
        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    def _build_backbone(self, input_shape):
        """
        Construye el backbone de ResNet usando IndexedConv2D en modo "valid".
        """
        # Si se usa 3D conv, se añade una dimensión de canal
        if self.use_3d_conv:
            input_shape = input_shape + (1,)
            print(f"Nuevo input shape para 3D: {input_shape}")
        network_input = keras.Input(shape=input_shape, name="input")
        x = network_input

        # Capa inicial: si se define init_layer, se aplica IndexedConv2D
        if self.init_layer is not None:
            # NOTA: En IndexedConv2D no se utiliza kernel_size, pues la vecindad se define por neighbor_indices.
            x = IndexedConv2D(
                out_channels=self.init_layer["filters"],
                neighbor_indices=self.neighbor_indices,
                output_shape=self.output_shape,
                name=self.backbone_name + "_conv1_conv"
            )(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
        # Si se define init_max_pool, se aplica pooling estándar 2D (o 3D si se usa)
        if self.init_max_pool is not None:
            if self.use_3d_conv:
                x = keras.layers.MaxPool3D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool"
                )(x)
            else:
                x = keras.layers.MaxPool2D(
                    pool_size=self.init_max_pool["size"],
                    strides=self.init_max_pool["strides"],
                    name=self.backbone_name + "_pool1_pool"
                )(x)
        # Bloques residuales con IndexedConv2D
        engine_output = self._stacked_res_blocks(
            x,
            architecture=self.architecture,
            residual_block_type=self.residual_block_type,
            attention=self.attention,
            name=self.backbone_name,
        )
        # Global Average Pooling
        if self.use_3d_conv:
            network_output = keras.layers.GlobalAveragePooling3D(name=self.backbone_name + "_global_avgpool")(engine_output)
        else:
            network_output = keras.layers.GlobalAveragePooling2D(name=self.backbone_name + "_global_avgpool")(engine_output)
        backbone_model = keras.Model(network_input, network_output, name=self.backbone_name)
        return backbone_model, network_input

    def _stacked_res_blocks(self, inputs, architecture, residual_block_type, attention, name=None):
        filters_list = [layer["filters"] for layer in architecture]
        blocks_list = [layer["blocks"] for layer in architecture]

        x = self._stack_fn(
            inputs,
            filters_list[0],
            blocks_list[0],
            residual_block_type,
            stride=1,
            attention=attention,
            kernel_size=architecture[0]["kernel_size"],
            name=name + "_conv2",
        )
        for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
            x = self._stack_fn(
                x,
                filters,
                blocks,
                residual_block_type,
                attention=attention,
                kernel_size=architecture[i+1]["kernel_size"],
                name=name + "_conv" + str(i + 3),
            )
        return x

    def _stack_fn(self, inputs, filters, blocks, residual_block_type, stride=2, attention=None, kernel_size=3, name=None):
        res_blocks = {
            "basic": self._basic_residual_block,
            "bottleneck": self._bottleneck_residual_block,
        }

        x = res_blocks[residual_block_type](
            inputs,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            attention=attention,
            name=name + "_block1",
        )
        for i in range(2, blocks + 1):
            x = res_blocks[residual_block_type](
                x,
                filters,
                kernel_size=kernel_size,
                conv_shortcut=False,
                attention=attention,
                name=name + "_block" + str(i),
            )
        return x

    def _basic_residual_block(self, inputs, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, name=None):
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
            x = IndexedConv2D(
                out_channels=filters,
                neighbor_indices=self.neighbor_indices,
                output_shape=self.output_shape,
                name=name + "_1_conv"
            )(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = IndexedConv2D(
                out_channels=filters,
                neighbor_indices=self.neighbor_indices,
                output_shape=self.output_shape,
                name=name + "_2_conv"
            )(x)
            x = keras.layers.BatchNormalization()(x)
        else:
            x = IndexedConv2D(
                out_channels=filters,
                neighbor_indices=self.neighbor_indices,
                output_shape=self.output_shape,
                name=name + "_1_conv"
            )(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
            x = IndexedConv2D(
                out_channels=filters,
                neighbor_indices=self.neighbor_indices,
                output_shape=self.output_shape,
                name=name + "_2_conv"
            )(x)
            x = keras.layers.BatchNormalization()(x)

        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(x, attention["reduction_ratio"], name=name + "_dse")
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(x, attention["reduction_ratio"], name=name + "_cse")
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)
        return x

    def _bottleneck_residual_block(self, inputs, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, name=None):
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
            filters=4 * filters,
            kernel_size=1,
            name=name + "_3_conv",
        )(x)

        if attention is not None:
            if attention["mechanism"] == "Dual-SE":
                x = dual_squeeze_excite_block(x, attention["reduction_ratio"], name=name + "_dse")
            elif attention["mechanism"] == "Channel-SE":
                x = channel_squeeze_excite_block(x, attention["reduction_ratio"], name=name + "_cse")
            elif attention["mechanism"] == "Spatial-SE":
                x = spatial_squeeze_excite_block(x, name=name + "_sse")

        x = keras.layers.Add(name=name + "_add")([shortcut, x])
        x = keras.layers.ReLU(name=name + "_out")(x)
        return x


class RawWaveSingleCNN(CTLearnModel):
    """
    RawWaveSingleCNN is a CTLearn-compatible model for raw waveform input.
    
    This model builds a convolutional backbone (using conv_block) and a fully connected head 
    (via build_fully_connect_head) for type reconstruction only.
    
    When instantiated via CTLearnModel.from_name, the factory should pass:
    
      - input_shape (e.g. self.training_loader.input_shape)
      - tasks (e.g. self.reco_tasks, which for this model should be ["type"])
      - parent (the calling component)
    
    The defaults mimic your old CTLearn configuration.
    """

    name = Unicode("RawWaveSingleCNN", help="Name of the model backbone.").tag(config=True)

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

    # Default head configuration for type reconstruction.
    head_layers = Dict(
        default_value={"type": [512, 256, 2]},
        allow_none=False,
        help="Fully connected head layers for type reconstruction."
    ).tag(config=True)

    head_activation_function = Dict(
        default_value={"type": "relu"},
        allow_none=False,
        help="Activation function for the head. The 'type' task uses softmax as final activation."
    ).tag(config=True)

    # Default initial padding.
    init_padding = Int(
        default_value=0,
        allow_none=False,
        help="Initial padding to apply to the input data."
    ).tag(config=True)

    use_3d_conv = Bool(
        default_value=False,
        help="If True, perform 3D convolutions (expects input shape (H, W, T) and adds a channel dimension)."
    ).tag(config=True)

    def __init__(self, input_shape, tasks, config=None, parent=None, **kwargs):
        # If a parent is provided, do not pass an independent config.
        if parent is not None:
            # Use the parent's config.
            config = None  
        else:
            # If no config is provided, create a default Config.
            if config is None:
                config = Config()
        super().__init__(config=config, parent=parent, **kwargs)
        self.input_shape = input_shape
        self.tasks = tasks  # For this model, tasks should be ["type"]

        # Store configuration (if needed)
        if parent is not None:
            self.config = parent.config
        else:
            self.config = config

        # Use defaults for initial layers if not provided in config.
        self.init_layer = self.config.get("init_layer", {"filters": 16, "kernel_size": 3, "strides": 1})
        self.init_max_pool = self.config.get("init_max_pool", {"size": 2, "strides": 2})

        # Build the backbone.
        self.backbone_name = self.name + "_block"
        self.backbone_model, self.input_layer = self._build_backbone(self.input_shape)
        backbone_output = self.backbone_model(self.input_layer)

        # (Optional) Validate head traits...
        validate_trait_dict(self.head_layers, self.tasks)
        validate_trait_dict(self.head_activation_function, self.tasks)

        # Build the fully connected head for "type" reconstruction.
        self.logits = build_fully_connect_head(
            backbone_output,
            layers={"type": self.head_layers["type"]},
            activation_function={"type": self.head_activation_function["type"]},
            tasks=self.tasks,
        )
        self.model = keras.Model(self.input_layer, self.logits, name="CTLearn_model")

    @staticmethod
    def conv_block(inputs, params, name="cnn_block"):
        bn_momentum = params.get("batchnorm_decay", 0.99)
        filters_list = [layer["filters"] for layer in params["basic"]["conv_block"]["layers"]]
        kernel_sizes = [layer["kernel_size"] for layer in params["basic"]["conv_block"]["layers"]]
        numbers_list = [layer.get("number", 1) for layer in params["basic"]["conv_block"]["layers"]]
        max_pool = params["basic"]["conv_block"]["max_pool"]
        bottleneck_filters = params["basic"]["conv_block"]["bottleneck"]
        batchnorm = params["basic"]["conv_block"].get("batchnorm", False)
        waveform3D = len(inputs.get_shape().as_list()) == 5
        print(f"3 d convolutions is {waveform3D}")

        x = inputs
        if batchnorm:
            x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)

        for i, (filters, kernel_size, number) in enumerate(zip(filters_list, kernel_sizes, numbers_list)):
            for nr in range(number):
                if waveform3D:
                    x = keras.layers.Conv3D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding="same",
                        activation=tf.nn.relu,
                        name=f"{name}_conv_{i+1}_{nr+1}",
                    )(x)
                else:
                    x = keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=kernel_size,
                        padding="same",
                        activation=tf.nn.relu,
                        name=f"{name}_conv_{i+1}_{nr+1}",
                    )(x)
            # Check the spatial dimension before pooling.
            current_shape = keras.backend.int_shape(x)
            height = current_shape[1]  # assuming NHWC
            width = current_shape[2]
            if max_pool and height is not None and width is not None and height >= max_pool["size"] and width >= max_pool["size"]:
                if waveform3D:
                    x = keras.layers.MaxPool3D(
                        pool_size=max_pool["size"],
                        strides=max_pool["strides"],
                        name=f"{name}_pool_{i+1}",
                    )(x)
                else:
                    x = keras.layers.MaxPool2D(
                        pool_size=max_pool["size"],
                        strides=max_pool["strides"],
                        name=f"{name}_pool_{i+1}",
                    )(x)
            if batchnorm:
                x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)

        if bottleneck_filters:
            if waveform3D:
                x = keras.layers.Conv3D(
                    filters=bottleneck_filters,
                    kernel_size=1,
                    padding="same",
                    activation=tf.nn.relu,
                    name=f"{name}_bottleneck",
                )(x)
            else:
                x = keras.layers.Conv2D(
                    filters=bottleneck_filters,
                    kernel_size=1,
                    padding="same",
                    activation=tf.nn.relu,
                    name=f"{name}_bottleneck",
                )(x)
            if batchnorm:
                x = keras.layers.BatchNormalization(momentum=bn_momentum)(x)
        return x


    def _build_backbone(self, input_shape):
        """
        Build the convolutional backbone for raw waveform input.
        
        Parameters
        ----------
        input_shape : tuple
            Shape of the input data, e.g. (28, 28, 10).
        
        Returns
        -------
        backbone_model : keras.Model
            Model representing the convolutional backbone.
        network_input : keras.Input
            Input layer for the backbone.
        """
        if self.use_3d_conv:
            # Append a channel dimension: new shape is (H, W, T, 1)
            input_shape = input_shape + (1,)
        network_input = keras.Input(shape=input_shape, name="input")
        x = network_input

        # Apply initial padding if configured.
        if self.init_padding:
            x = keras.layers.ZeroPadding2D(padding=self.init_padding, name="init_padding")(x)

        # Optional initial convolution layer.
        if self.init_layer:
            x = keras.layers.Conv2D(
                filters=self.init_layer["filters"],
                kernel_size=self.init_layer["kernel_size"],
                strides=self.init_layer["strides"],
                padding="same",
                activation="relu",
                name="init_conv"
            )(x)

        # Optional initial max pooling.
        if self.init_max_pool:
            x = keras.layers.MaxPool2D(
                pool_size=self.init_max_pool["size"],
                strides=self.init_max_pool["strides"],
                name="init_pool"
            )(x)

        # Build the main backbone using conv_block from CTLearn.
        x = RawWaveSingleCNN.conv_block(x, self.config, name="raw_wave_conv_block")
        x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
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
