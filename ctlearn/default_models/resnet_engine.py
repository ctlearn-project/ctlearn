import tensorflow as tf
from ctlearn.default_models.attention import squeeze_excite_block, channel_squeeze_excite_block, spatial_squeeze_excite_block

def stacked_res_blocks(inputs, params, reuse=None, trainable=True):
    # Get custom hyperparameters
    residual_block = params['resnet_engine']['stacked_res_blocks'].get('residual_block', 'bottleneck')
    filters_list = [layer['filters'] for layer in
            params['resnet_engine']['stacked_res_blocks']['architecture']]
    blocks_list = [layer['blocks'] for layer in
            params['resnet_engine']['stacked_res_blocks']['architecture']]
    attention = params.get('attention', None)

    x = stack_fn(inputs, filters_list[0], blocks_list[0], residual_block, stride=1, attention=attention, reuse=reuse, trainable=trainable, name='conv2')
    for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
       x = stack_fn(x, filters, blocks, residual_block, attention=attention, reuse=reuse, trainable=trainable, name='conv' + str(i+3))
    return x

def stack_fn(inputs, filters, blocks, residual_block, stride=2, attention=None, reuse=None, trainable=True, name=None):
    """A set of stacked residual blocks.
    Arguments:
      inputs: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      residual_block: string, type of residual block.
      stride: default 2, stride of the first layer in the first block.
      attention: default None, squeeze excite ratio for the
          attention mechanism.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """

    res_blocks = {
        'basic': basic_residual_block,
        'bottleneck': bottleneck_residual_block
    }
    x = res_blocks[residual_block](inputs, filters, stride=stride, attention=attention, reuse=reuse, trainable=trainable, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = res_blocks[residual_block](x, filters, conv_shortcut=False, attention=attention, reuse=reuse, trainable=trainable, name=name + '_block' + str(i))
    return x

def basic_residual_block(inputs, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, reuse=None, trainable=True, name=None):
    """A basic residual block.
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
      Output tensor for the residual block.
    """

    with tf.variable_scope("Basic_res_block", reuse=reuse):

        if conv_shortcut:
            shortcut = tf.layers.conv2d(inputs, filters=filters, kernel_size=1,
                           strides=stride, reuse=reuse, trainable=trainable, name=name + '_0_conv')
        else:
            shortcut = inputs

        x = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                strides=stride, padding='same', activation=tf.nn.relu, reuse=reuse, trainable=trainable, name=name + '_1_conv')
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                padding='same', activation=tf.nn.relu, reuse=reuse, trainable=trainable, name=name + '_2_conv')

        # Attention mechanism
        if attention is not None:
            if attention['mechanism'] == 'Squeeze-and-Excitation':
                x = squeeze_excite_block(x, attention['ratio'], reuse=reuse, trainable=trainable, name=name + '_se')
            elif attention['mechanism'] == 'Channel-Squeeze-and-Excitation':
                x = channel_squeeze_excite_block(x, attention['ratio'], reuse=reuse, trainable=trainable, name=name + '_cse')
            elif attention['mechanism'] == 'Spatial-Squeeze-and-Excitation':
                x = spatial_squeeze_excite_block(x, reuse=reuse, trainable=trainable, name=name + '_sse')

        x = tf.math.add_n([shortcut, x], name=name + '_add')
        x = tf.nn.relu(x, name=name + '_out')
        return x

def bottleneck_residual_block(inputs, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, reuse=None, trainable=True, name=None):
    """A bottleneck residual block.
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
      Output tensor for the residual block.
    """

    with tf.variable_scope("Bottleneck_res_block", reuse=reuse):

        if conv_shortcut:
            shortcut = tf.layers.conv2d(inputs, filters=4*filters, kernel_size=1,
                           strides=stride, reuse=reuse, trainable=trainable, name=name + '_0_conv')
        else:
            shortcut = inputs

        x = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=stride,
                activation=tf.nn.relu, reuse=reuse, trainable=trainable, name=name + '_1_conv')

        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                padding='same', activation=tf.nn.relu, reuse=reuse, trainable=trainable, name=name + '_2_conv')
        x = tf.layers.conv2d(x, filters=4*filters, kernel_size=1,
                reuse=reuse, trainable=trainable, name=name + '_3_conv')

        # Attention mechanism
        if attention is not None:
            if attention['mechanism'] == 'Squeeze-and-Excitation':
                x = squeeze_excite_block(x, attention['ratio'], reuse=reuse, trainable=trainable, name=name + '_se')
            elif attention['mechanism'] == 'Channel-Squeeze-and-Excitation':
                x = channel_squeeze_excite_block(x, attention['ratio'], reuse=reuse, trainable=trainable, name=name + '_cse')
            elif attention['mechanism'] == 'Spatial-Squeeze-and-Excitation':
                x = spatial_squeeze_excite_block(x, reuse=reuse, trainable=trainable, name=name + '_sse')

        x = tf.math.add_n([shortcut, x], name=name + '_add')
        x = tf.nn.relu(x, name=name + '_out')
        return x

def squeeze_excite_block(inputs, ratio=16, reuse=None, trainable=True, name=None):
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

    cse = channel_squeeze_excite_block(inputs=inputs, ratio=ratio, reuse=reuse, trainable=trainable, name=name + '_cse')
    sse = spatial_squeeze_excite_block(inputs=inputs, reuse=reuse, trainable=trainable, name=name + '_sse')

    output = tf.math.add_n([cse, sse], name=name + '_add')
    return output

def channel_squeeze_excite_block(inputs, ratio=4, reuse=None, trainable=True, name=None):
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
    cse = tf.layers.dense(cse, units=tf.math.divide(filters,ratio), activation='relu', reuse=reuse, trainable=trainable, name=name + '_1_dense')
    cse = tf.layers.dense(cse, units=filters, activation='sigmoid', reuse=reuse, trainable=trainable, name=name + '_2_dense')

    output = tf.math.multiply(inputs, cse, name=name + '_mult')
    return output

def spatial_squeeze_excite_block(inputs, reuse=None, trainable=True, name=None):
    """ A spatial squeeze-excite block.
    Arguments:
      inputs: input tensor.
      name: string, spatial squeeze-excite block label.
    returns:
      Output tensor for the spatial squeeze-excite block.
    """

    sse = tf.layers.conv2d(inputs, filters=1, kernel_size=1,
              activation=tf.nn.sigmoid, reuse=reuse, trainable=trainable, name=name + '_spatial_conv')

    output = tf.math.multiply(inputs, sse, name=name + '_mult')
    return output
