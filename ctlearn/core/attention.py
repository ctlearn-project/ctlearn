"""
This module defines the squeeze-excite blocks for channel-wise and/or spatial-wise attention mechanisms.
"""
import tensorflow as tf
import keras

__all__ = [
    "dual_squeeze_excite_block",
    "channel_squeeze_excite_block",
    "spatial_squeeze_excite_block",
    "temporal_attention_block",
    "dual_squeeze_excite_block_indexed2d",
    "spatial_squeeze_excite_block_indexed2d",
    "triple_squeeze_excite_block_indexed3d",
    "spatial_squeeze_excite_block_indexed3d",

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

def dual_squeeze_excite_block_indexed2d(inputs, ratio=16, name=None):
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

    cse = channel_squeeze_excite_block_indexed2d(
        inputs=inputs,
        ratio=ratio,
        name=name + "_cse",
    )
    sse = spatial_squeeze_excite_block_indexed2d(
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

    # Global Average Pooling over (height, width) and channels
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

def channel_squeeze_excite_block_indexed2d(inputs, ratio=4, name=None):
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

    # Global Average Pooling over pixels and channels
    cse = keras.layers.GlobalAveragePooling1D(
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


def spatial_squeeze_excite_block_indexed2d(inputs, name=None):
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

    sse = keras.layers.Conv1D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        name=name + "_spatial_conv",
    )(inputs)

    return keras.layers.Multiply(name=name + "_mult")([inputs, sse])


def temporal_attention_block(x, name=None):
    # x: (B, P, T', C)
    att = keras.layers.Dense(1, name=(name+"_logits"))(x)  
    att = keras.layers.Softmax(axis=2, name=(name+"_softmax"))(att)
    ctx = tf.reduce_sum(x * att, axis=2, keepdims=True)     # → (B, P, 1, C)
    return ctx

def spatial_squeeze_excite_block_indexed3d(inputs, name=None):
    # inputs: (B, P, T, C)
    # Pool time and channels → use Conv1D over pixels (P)
    shape = tf.shape(inputs)
    B, P, T, C = shape[0], shape[1], shape[2], shape[3]

    # Reshape to (B, P, T*C) so Conv1D sees pixels as sequence
    x = tf.reshape(inputs, [B, P, T * C])

    # Apply Conv1D across pixels (P)
    sse = keras.layers.Conv1D(
        filters=1,
        kernel_size=1,
        activation="sigmoid",
        name=name + "_spatial_conv"
    )(x)  # → (B, P, 1)

    # Broadcast to (B, P, T, C)
    sse = tf.reshape(sse, [B, P, 1, 1])
    out = inputs * sse  # Elementwise broadcast multiply
    return out

def triple_squeeze_excite_block_indexed3d(inputs, ratio=16, name=None):
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
    sse = spatial_squeeze_excite_block_indexed3d(
        inputs=inputs, name=name + "_sse"
    )

    tse = temporal_attention_block(x=inputs, name=name + "_tse")

    return keras.layers.Add(name=name + "_add")([cse, sse, tse])