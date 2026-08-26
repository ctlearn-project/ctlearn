"""
This module defines ``HexCNN``, a ``CTLearnModel`` backbone built on
``keras_hexagdly``'s hexagonal convolution and pooling layers, for use on
camera images produced by ``ctlearn.core.hexagdly_mapper.HexagdlyMapper``.
"""

import keras
import keras_hexagdly as hgly

from ctapipe.core.traits import Bool, CaselessStrEnum, Dict, Int, List, Unicode

from ctlearn.core.model import CTLearnModel, build_fully_connect_head
from ctlearn.core.attention import (
    dual_squeeze_excite_block,
    channel_squeeze_excite_block,
    spatial_squeeze_excite_block,
)
from ctlearn.utils import validate_trait_dict

__all__ = ["HexCNN"]


class HexCNN(CTLearnModel):
    """``HexCNN`` is a convolutional neural network model built on hexagonal
    convolutions.

    This mirrors ``ctlearn.core.model.SingleCNN``, but its convolutional and
    pooling layers operate directly on the hexagonal pixel grid via
    ``keras_hexagdly.layers.Conv2d``/``MaxPool2d`` instead of resampling onto
    an axis-aligned square grid first. It is meant to be paired with
    ``ctlearn.core.hexagdly_mapper.HexagdlyMapper`` as the ``image_mapper_type``.

    Only max pooling is available, since ``keras_hexagdly`` does not currently
    provide an average pooling layer. The bottleneck layer (a 1x1 convolution)
    is implemented with a plain ``keras.layers.Conv2D``: a 1x1 convolution is
    purely a per-cell channel projection with no spatial mixing, so it is
    geometry-agnostic and behaves identically on a hex-addressed grid.
    """

    name = Unicode(
        "HexCNN",
        help="Name of the model backbone.",
    ).tag(config=True)

    architecture = List(
        trait=Dict(),
        default_value=[
            {"filters": 32, "kernel_size": 1, "number": 1},
            {"filters": 32, "kernel_size": 1, "number": 1},
            {"filters": 64, "kernel_size": 1, "number": 1},
            {"filters": 128, "kernel_size": 1, "number": 1},
        ],
        allow_none=False,
        help=(
            "List of dicts containing the number of filters, hex kernel sizes "
            "(hex neighbourhood radius) and number of repetitions. "
            "E.g. ``[{'filters': 12, 'kernel_size': 1, 'number': 1}, ...]``."
        ),
    ).tag(config=True)

    pooling_type = CaselessStrEnum(
        ["max"],
        default_value="max",
        allow_none=True,
        help=(
            "Type of pooling to apply to the convolutional layers with "
            "``pooling_parameters``. Only 'max' is available -- "
            "``keras_hexagdly`` does not provide a hexagonal average pooling layer."
        ),
    ).tag(config=True)

    pooling_parameters = Dict(
        default_value={"size": 1, "strides": 2},
        allow_none=True,
        help=(
            "Parameters for the hexagonal max pooling layers, mapped onto "
            "``keras_hexagdly.layers.MaxPool2d(kernel_size=size, stride=strides)``. "
            "E.g. ``{'size': 1, 'strides': 2}``."
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

        # Build the HexCNN model backbone
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
        Build the HexCNN model backbone.

        Function to build the backbone of the HexCNN model using the specified
        parameters. ``input_shape`` is expected to come from an ``ImageMapper``
        that produces a hex-addressed grid, e.g.
        ``ctlearn.core.hexagdly_mapper.HexagdlyMapper``.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (height, width, channels).

        Returns
        -------
        backbone_model : keras.Model
            Keras model object representing the backbone of the HexCNN model.
        network_input : keras.Input
            Keras input layer object for the backbone model.
        """

        # Define the input layer from the input shape
        network_input = keras.Input(shape=input_shape)
        # Get model architecture parameters for the backbone
        filters_list = [layer["filters"] for layer in self.architecture]
        kernel_sizes = [layer["kernel_size"] for layer in self.architecture]
        numbers_list = [layer["number"] for layer in self.architecture]

        x = network_input
        if self.batchnorm:
            x = keras.layers.BatchNormalization(momentum=0.99)(x)

        for i, (filters, kernel_size, number) in enumerate(
            zip(filters_list, kernel_sizes, numbers_list)
        ):
            for nr in range(number):
                x = hgly.Conv2d(
                    filters,
                    kernel_size=kernel_size,
                    name=f"{self.backbone_name}_conv_{i+1}_{nr+1}",
                )(x)
                x = keras.layers.ReLU(
                    name=f"{self.backbone_name}_conv_{i+1}_{nr+1}_relu"
                )(x)
            if self.pooling_type is not None:
                x = hgly.MaxPool2d(
                    kernel_size=self.pooling_parameters["size"],
                    stride=self.pooling_parameters["strides"],
                    name=f"{self.backbone_name}_pool_{i+1}",
                )(x)
            if self.batchnorm:
                x = keras.layers.BatchNormalization(momentum=0.99)(x)

        # bottleneck layer -- a 1x1 conv is purely a per-cell channel
        # projection, so a plain square Conv2D is used (see class docstring).
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
