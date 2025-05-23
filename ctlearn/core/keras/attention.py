"""
This module defines the squeeze-excite blocks for channel-wise and/or spatial-wise attention mechanisms.
"""

import keras

__all__ = [
    "dual_squeeze_excite_block",
    "channel_squeeze_excite_block",
    "spatial_squeeze_excite_block",
]

def dual_squeeze_excite_block(inputs, ratio=16, name=None):
    """
    A channel & spatial (dual) squeeze-excite block.

    This function creates a dual squeeze-excite block that combines both channel-wise and spatial-wise
    squeeze-excite mechanisms. The channel squeeze-excite block focuses on recalibrating the importance
    of each channel, while the spatial squeeze-excite block focuses on recalibrating the importance of
    each spatial location.

    Parameters
    ----------
    inputs : keras.layers.Layer
        Input tensor to the squeeze-excite block.
    ratio : int
        Reduction ratio for the channel squeeze-excite block. Default is 16.
    name : str, optional
        Name for the squeeze-excite block. Default is None.

    Returns
    -------
    keras.layers.Layer
        Output tensor for the squeeze-excite block.

    References
    ----------
    - [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    - [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = channel_squeeze_excite_block(
        inputs=inputs,
        ratio=ratio,
        name=name + "_cse",
    )
    sse = spatial_squeeze_excite_block(
        inputs=inputs, name=name + "_sse"
    )

    return keras.layers.Add(name=name + "_add")([cse, sse])


def channel_squeeze_excite_block(inputs, ratio=4, name=None):
    """
    A channel-wise squeeze-excite block.

    This function creates a channel-wise squeeze-excite block that recalibrates the importance
    of each channel by using global average pooling followed by two dense layers.

    Parameters
    ----------
    inputs : keras.layers.Layer
        Input tensor to the squeeze-excite block.
    ratio : int
        Reduction ratio for the squeeze-excite block. Default is 4.
    name : str, optional
        Name for the squeeze-excite block. Default is None.

    Returns
    -------
    keras.layers.Layer
        Output tensor for the channel squeeze-excite block.
    """

    # Temp fix for supporting keras2 & keras3
    if int(keras.__version__.split(".")[0]) >= 3:
        filters = inputs.shape[-1]
    else:
        filters = inputs.get_shape().as_list()[-1]
    cse = keras.layers.GlobalAveragePooling2D(
        keepdims=True, name=name + "_avgpool"
    )(inputs)

    cse = keras.layers.Dense(
        units=filters // ratio,
        activation="relu",
        name=name + "_1_dense",
    )(cse)
    cse = keras.layers.Dense(
        units=filters, activation="sigmoid", name=name + "_2_dense"
    )(cse)

    return keras.layers.Multiply(name=name + "_mult")([inputs, cse])


def spatial_squeeze_excite_block(inputs, name=None):
    """
    A spatial squeeze-excite block.

    This function creates a spatial squeeze-excite block that recalibrates the importance
    of each spatial location by using a convolutional layer with a sigmoid activation.

    Parameters
    ----------
    inputs : keras.layers.Layer
        Input tensor to the squeeze-excite block.
    name : str, optional
        Name for the squeeze-excite block. Default is None.

    Returns
    -------
    keras.layers.Layer
        Output tensor for the spatial squeeze-excite block.
    """

    sse = keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        name=name + "_spatial_conv",
    )(inputs)

    return keras.layers.Multiply(name=name + "_mult")([inputs, sse])
