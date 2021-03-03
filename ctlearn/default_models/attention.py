import tensorflow as tf

def squeeze_excite_block(inputs, ratio=16, name=None):
    """ A channel & spatial squeeze-excite block.
    Arguments:
      inputs: input tensor.
      ratio: number of output filters
      name: string, spatial squeeze-excite block label.
    returns:
      Output tensor for the squeeze-excite block.
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    """

    cse = channel_squeeze_excite_block(inputs=inputs, ratio=ratio, name=name + '_cse')
    sse = spatial_squeeze_excite_block(inputs=inputs, name=name + '_sse')

    output = tf.math.add_n([cse, sse], name=name + '_add')
    return output

def channel_squeeze_excite_block(inputs, ratio=4, name=None):
    """ A channel-wise squeeze-excite block.
    Arguments:
      inputs: input tensor.
      ratio: number of output filters.
      name: string, channel squeeze-excite block label.
    returns:
      Output tensor for the channel squeeze-excite block.
    """

    filters = inputs.get_shape().as_list()[-1]

    cse = tf.reduce_mean(inputs, axis=[1,2], keepdims=True, name=name + '_avgpool')
    cse = tf.layers.dense(cse, units=tf.math.divide(filters,ratio), activation='relu', name=name + '_1_dense')
    cse = tf.layers.dense(cse, units=filters, activation='sigmoid', name=name + '_2_dense')

    output = tf.math.multiply(inputs, cse, name=name + '_mult')
    return output

def spatial_squeeze_excite_block(inputs, name=None):
    """ A spatial squeeze-excite block.
    Arguments:
      inputs: input tensor.
      name: string, spatial squeeze-excite block label.
    returns:
      Output tensor for the spatial squeeze-excite block.
    """

    sse = tf.layers.conv2d(inputs, filters=1, kernel_size=1,
              activation=tf.nn.sigmoid, name=name + '_spatial_conv')

    output = tf.math.multiply(inputs, sse, name=name + '_mult')
    return output
