import tensorflow as tf

def conv_block(inputs, training, params, reuse=None):

    with tf.variable_scope("Basic_conv_block", reuse=reuse):

        # Get standard hyperparameters
        bn_momentum = params.get('batchnorm_decay', 0.99)
        # Get custom hyperparameters
        filters_list = [layer['filters'] for layer in
                params['basic']['conv_block']['layers']]
        kernel_sizes = [layer['kernel_size'] for layer in
                params['basic']['conv_block']['layers']]
        max_pool = params['basic']['conv_block']['max_pool']
        bottleneck_filters = params['basic']['conv_block']['bottleneck']
        batchnorm = params['basic']['conv_block'].get('batchnorm', False)
        attention = params['basic'].get('attention', None)

        x = inputs
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
            x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                    activation=tf.nn.relu, padding="same", reuse=reuse,
                    name="conv_{}".format(i+1))
            if max_pool:
                x = tf.layers.max_pooling2d(x, pool_size=max_pool['size'],
                        strides=max_pool['strides'], name="pool_{}".format(i+1))
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

        # bottleneck layer
        if bottleneck_filters:
            x = tf.layers.conv2d(x, filters=bottleneck_filters,
                    kernel_size=1, activation=tf.nn.relu, padding="same",
                    reuse=reuse, name="bottleneck")
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

        # Attention mechanism
        if attention is not None:
            if attention['mechanism'] == 'Squeeze-and-Excitation':
                x = squeeze_excite_block(x, attention['ratio'], name='se')
            elif attention['mechanism'] == 'Channel-Squeeze-and-Excitation':
                x = channel_squeeze_excite_block(x, attention['ratio'], name='cse')
            elif attention['mechanism'] == 'Spatial-Squeeze-and-Excitation':
                x = spatial_squeeze_excite_block(x, name='sse')

        return x

def fc_head(inputs, tasks_dict, expected_logits_dimension):

    layers = tasks_dict['fc_head']

    if layers[-1] != expected_logits_dimension:
        print("Warning:fc_head: Last logit unit '{}' of the fc_head array differs from the expected_logits_dimension '{}'. The expected logits dimension '{}' will be appended.".format(layers[-1], expected_logits_dimension))
        layers.append(expected_logits_dimension)

    x = inputs
    activation=tf.nn.relu
    for i, units in enumerate(layers):
        if i == len(layers)-1:
            activation=None
        x = tf.layers.dense(x, units=units, activation=activation,
                name="fc_{}_{}".format(tasks_dict['name'], i+1))
    return x

def conv_head(inputs, training, params):

    # Get standard hyperparameters
    bn_momentum = params.get('batchnorm_decay', 0.99)

    # Get custom hyperparameters
    filters_list = [layer['filters'] for layer in
            params['basic']['conv_head']['layers']]
    kernel_sizes = [layer['kernel_size'] for layer in
            params['basic']['conv_head']['layers']]
    final_avg_pool = params['basic']['conv_head'].get('final_avg_pool', True)
    batchnorm = params['basic']['conv_head'].get('batchnorm', False)

    x = inputs

    for i, (filters, kernel_size) in enumerate(zip(filters_list, kernel_sizes)):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                activation=tf.nn.relu, padding="same",
                name="conv_{}".format(i+1))
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

    # Average over remaining width and length
    if final_avg_pool:
        x = tf.layers.average_pooling2d(x,
                pool_size=x.get_shape().as_list()[1],
                strides=1, name="global_avg_pool")

    flat = tf.layers.flatten(x)

    return flat


def stacked_res_blocks(inputs, params):
    # Get custom hyperparameters
    filters_list = [layer['filters'] for layer in
            params['basic']['stacked_res_blocks']['architecture']]
    blocks_list = [layer['blocks'] for layer in
            params['basic']['stacked_res_blocks']['architecture']]
    attention = params['basic'].get('attention', None)

    x = stack_fn(inputs, filters_list[0], blocks_list[0], stride=1, attention=attention, name='conv2')
    for i, (filters, blocks) in enumerate(zip(filters_list[1:], blocks_list[1:])):
       x = stack_fn(x, filters, blocks, attention=attention, name='conv' + str(i+3))
    return x

def stack_fn(inputs, filters, blocks, stride=2, attention=None, name=None):
    """A set of stacked residual blocks.
    Arguments:
      inputs: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride: default 2, stride of the first layer in the first block.
      attention: default None, squeeze excite ratio for the
          attention mechanism.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = res_block(inputs, filters, stride=stride, attention=attention, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = res_block(x, filters, conv_shortcut=False, attention=attention, name=name + '_block' + str(i))
    return x

def res_block(inputs, filters, kernel_size=3, stride=1, conv_shortcut=True, attention=None, name=None):
    """A residual block.
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

    with tf.variable_scope("Basic_res_block"):
        if conv_shortcut:
            shortcut = tf.layers.conv2d(inputs, filters=4*filters, kernel_size=1,
                           strides=stride, name=name + '_0_conv')
        else:
            shortcut = inputs

        x = tf.layers.conv2d(inputs, filters=filters, kernel_size=1, strides=stride,
                activation=tf.nn.relu, name=name + '_1_conv')

        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                padding='same', activation=tf.nn.relu, name=name + '_2_conv')
        x = tf.layers.conv2d(x, filters=4*filters, kernel_size=1,
                name=name + '_3_conv')

        # Attention mechanism
        if attention is not None:
            if attention['mechanism'] == 'Squeeze-and-Excitation':
                x = squeeze_excite_block(x, attention['ratio'], name=name + '_se')
            elif attention['mechanism'] == 'Channel-Squeeze-and-Excitation':
                x = channel_squeeze_excite_block(x, attention['ratio'], name=name + '_cse')
            elif attention['mechanism'] == 'Spatial-Squeeze-and-Excitation':
                x = spatial_squeeze_excite_block(x, name=name + '_sse')

        x = tf.math.add_n([shortcut, x], name=name + '_add')
        x = tf.nn.relu(x, name=name + '_out')
        return x

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
