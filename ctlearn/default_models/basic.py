import tensorflow as tf
from ctlearn.default_models.attention import (
    squeeze_excite_block,
    channel_squeeze_excite_block,
    spatial_squeeze_excite_block,
)


def conv_block(inputs, params, name="cnn_block"):

    # Get standard hyperparameters
    bn_momentum = params.get("batchnorm_decay", 0.99)
    # Get custom hyperparameters
    filters_list = [
        layer["filters"] for layer in params["basic"]["conv_block"]["layers"]
    ]
    kernel_sizes = [
        layer["kernel_size"] for layer in params["basic"]["conv_block"]["layers"]
    ]
    numbers_list = [
        layer.get("number", 1) for layer in params["basic"]["conv_block"]["layers"]
    ]
    max_pool = params["basic"]["conv_block"]["max_pool"]
    bottleneck_filters = params["basic"]["conv_block"]["bottleneck"]
    batchnorm = params["basic"]["conv_block"].get("batchnorm", False)
    attention = params.get("attention")

    x = inputs
    if batchnorm:
        x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    for i, (filters, kernel_size, number) in enumerate(
        zip(filters_list, kernel_sizes, numbers_list)
    ):
        for nr in range(number):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                name=f"{name}_conv_{i+1}_{nr+1}",
            )(x)
            x = tf.keras.layers.ReLU(name=f"{name}_conv_{i+1}_{nr+1}_relu")(x)
        if max_pool:
            x = tf.keras.layers.MaxPool2D(
                pool_size=max_pool["size"],
                strides=max_pool["strides"],
                name=f"{name}_pool_{i+1}",
            )(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # bottleneck layer
    if bottleneck_filters:
        x = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_bottleneck",
        )(x)
        x = tf.keras.layers.ReLU(name=f"{name}_bottleneck_relu")(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # Attention mechanism
    if attention is not None:
        if attention["mechanism"] == "Squeeze-and-Excitation":
            x = squeeze_excite_block(x, attention["ratio"], name=f"{name}_se")
        elif attention["mechanism"] == "Channel-Squeeze-and-Excitation":
            x = channel_squeeze_excite_block(x, attention["ratio"], name=f"{name}_cse")
        elif attention["mechanism"] == "Spatial-Squeeze-and-Excitation":
            x = spatial_squeeze_excite_block(x, name=f"{name}_sse")

    return x


def fully_connect(inputs, layers=None, params=None, expected_logits_dimension=None, name=None):

    if layers is None:
        layers = params["basic"]["fully_connect"]["layers"]
        name = params["basic"]["fully_connect"].get("name", "default")

    if expected_logits_dimension:
        layers.append(expected_logits_dimension)

    x = inputs
    for i, units in enumerate(layers):
        if i != len(layers) - 1:
            x = tf.keras.layers.Dense(units=units, name="fc_{}_{}".format(name, i + 1))(
                x
            )
            x = tf.keras.layers.ReLU(name="fc_{}_{}_relu".format(name, i + 1))(x)
        else:
            x = tf.keras.layers.Dense(units=units, name=name)(x)

    return x


def conv_head(inputs, params):

    # Get standard hyperparameters
    bn_momentum = params.get("batchnorm_decay", 0.99)

    # Get custom hyperparameters
    filters_list = [
        layer["filters"] for layer in params["basic"]["conv_head"]["layers"]
    ]
    kernel_sizes = [
        layer["kernel_size"] for layer in params["basic"]["conv_head"]["layers"]
    ]
    final_avg_pool = params["basic"]["conv_head"].get("final_avg_pool", True)
    batchnorm = params["basic"]["conv_head"].get("batchnorm", False)

    x = inputs

    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            name="conv_{}".format(i + 1),
        )(x)
        x = tf.keras.layers.ReLU(name="conv_{}_relu".format(i + 1))(x)

        if batchnorm:
            x = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(x)

    # Average over remaining width and length
    if final_avg_pool:
        x = tf.keras.layers.AveragePooling2D(
            pool_size=x.get_shape().as_list()[1], strides=1, name="global_avg_pool"
        )(x)

    flat = tf.keras.layers.Flatten()(x)

    return flat
