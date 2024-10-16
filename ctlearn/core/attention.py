import keras

__all__ = [
    "dual_squeeze_excite_block",
    "channel_squeeze_excite_block",
    "spatial_squeeze_excite_block",
]

def dual_squeeze_excite_block(inputs, ratio=16, trainable=True, name=None):
    """A channel & spatial (dual) squeeze-excite block.
    Arguments:
      inputs: input tensor.
      ratio: number of output filters
      trainable: boolean, trainable weights.
      name: string, spatial squeeze-excite block label.
    returns:
      Output tensor for the squeeze-excite block.
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = channel_squeeze_excite_block(
        inputs=inputs,
        ratio=ratio,
        trainable=trainable,
        name=name + "_cse",
    )
    sse = spatial_squeeze_excite_block(
        inputs=inputs, trainable=trainable, name=name + "_sse"
    )

    return keras.layers.Add(name=name + "_add")([cse, sse])


def channel_squeeze_excite_block(
    inputs, ratio=4, trainable=True, name=None
):
    """A channel-wise squeeze-excite block.
    Arguments:
      inputs: input tensor.
      ratio: number of output filters.
      trainable: boolean, trainable weights.
      name: string, channel squeeze-excite block label.
    returns:
      Output tensor for the channel squeeze-excite block.
    """

    filters = inputs.get_shape().as_list()[-1]

    cse = keras.layers.GlobalAveragePooling2D(
        keepdims=True, name=name + "_avgpool"
    )(inputs)

    cse = keras.layers.Dense(
        units=filters // ratio,
        activation="relu",
        trainable=trainable,
        name=name + "_1_dense",
    )(cse)
    cse = keras.layers.Dense(
        units=filters, activation="sigmoid", trainable=trainable, name=name + "_2_dense"
    )(cse)

    return keras.layers.Multiply(name=name + "_mult")([inputs, cse])


def spatial_squeeze_excite_block(inputs, trainable=True, name=None):
    """A spatial squeeze-excite block.
    Arguments:
      inputs: input tensor.
      trainable: boolean, trainable weights.
      name: string, spatial squeeze-excite block label.
    returns:
      Output tensor for the spatial squeeze-excite block.
    """

    sse = keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        trainable=trainable,
        name=name + "_spatial_conv",
    )(inputs)

    return keras.layers.Multiply(name=name + "_mult")([inputs, sse])
