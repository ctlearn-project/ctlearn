import tensorflow as tf
from ctlearn.default_models.attention import (
    squeeze_excite_block,
    channel_squeeze_excite_block,
    spatial_squeeze_excite_block,
)


def stacked_res_blocks(inputs, params, name=None):
    """Function to set up a ResNet.
    Arguments:
      input_shape: input tensor shape.
      params: config parameters for the ResNet engine.
    Returns:
      Output tensor for the ResNet architecture.
    """
    # Get custom hyperparameters
    residual_block = params["resnet"]["stacked_res_blocks"].get(
        "residual_block", "bottleneck"
    )
    filters_list = [
        layer["filters"]
        for layer in params["resnet"]["stacked_res_blocks"]["architecture"]
    ]
    blocks_list = [
        layer["blocks"]
        for layer in params["resnet"]["stacked_res_blocks"]["architecture"]
    ]
    attention = params.get("attention", None)

    x = stack_fn(
        inputs,
        filters_list[0],
        blocks_list[0],
        residual_block,
        stride=1,
        attention=attention,
        name=name + "_conv2",
    )
    for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
        x = stack_fn(
            x,
            filters,
            blocks,
            residual_block,
            attention=attention,
            name=name + "_conv" + str(i + 3),
        )
    return x


def stack_fn(
    inputs, filters, blocks, residual_block, stride=2, attention=None, name=None
):
    """Function to stack residual blocks.
    Arguments:
      inputs: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      residual_block: string, type of residual block.
      stride: default 2, stride of the first layer in the first block.
      attention: config parameters for the attention mechanism.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """

    res_blocks = {
        "basic": basic_residual_block,
        "bottleneck": bottleneck_residual_block,
    }

    x = res_blocks[residual_block](
        inputs, filters, stride=stride, attention=attention, name=name + "_block1"
    )
    for i in range(2, blocks + 1):
        x = res_blocks[residual_block](
            x,
            filters,
            conv_shortcut=False,
            attention=attention,
            name=name + "_block" + str(i),
        )

    return x


def basic_residual_block(
    inputs,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    attention=None,
    name=None,
):
    """Function to build a basic residual block.
    Arguments:
      inputs: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      attention: config parameters for the attention mechanism.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=1, strides=stride, name=name + "_0_conv"
        )(inputs)
    else:
        shortcut = inputs

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        activation=tf.nn.relu,
        name=name + "_1_conv",
    )(inputs)
    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu,
        name=name + "_2_conv",
    )(x)

    # Attention mechanism
    if attention is not None:
        if attention["mechanism"] == "Squeeze-and-Excitation":
            x = squeeze_excite_block(x, attention["ratio"], name=name + "_se")
        elif attention["mechanism"] == "Channel-Squeeze-and-Excitation":
            x = channel_squeeze_excite_block(x, attention["ratio"], name=name + "_cse")
        elif attention["mechanism"] == "Spatial-Squeeze-and-Excitation":
            x = spatial_squeeze_excite_block(x, name=name + "_sse")

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.ReLU(name=name + "_out")(x)

    return x


def bottleneck_residual_block(
    inputs,
    filters,
    kernel_size=3,
    stride=1,
    conv_shortcut=True,
    attention=None,
    name=None,
):
    """Function to build a bottleneck residual block.
    Arguments:
      inputs: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      squeeze_excite_ratio: default 0, squeeze excite ratio for the
          attention mechanism.
      name: string, block label.
    Returns:
      Output tensor for the stacked blocks.
    """

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
            filters=4 * filters, kernel_size=1, strides=stride, name=name + "_0_conv"
        )(inputs)
    else:
        shortcut = inputs

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=1,
        strides=stride,
        activation=tf.nn.relu,
        name=name + "_1_conv",
    )(inputs)

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu,
        name=name + "_2_conv",
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=4 * filters, kernel_size=1, name=name + "_3_conv"
    )(x)

    # Attention mechanism
    if attention is not None:
        if attention["mechanism"] == "Squeeze-and-Excitation":
            x = squeeze_excite_block(x, attention["ratio"], name=name + "_se")
        elif attention["mechanism"] == "Channel-Squeeze-and-Excitation":
            x = channel_squeeze_excite_block(x, attention["ratio"], name=name + "_cse")
        elif attention["mechanism"] == "Spatial-Squeeze-and-Excitation":
            x = spatial_squeeze_excite_block(x, name=name + "_sse")

    x = tf.keras.layers.Add(name=name + "_add")([shortcut, x])
    x = tf.keras.layers.ReLU(name=name + "_out")(x)

    return x
