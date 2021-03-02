import tensorflow as tf
from ctlearn.default_models.attention import squeeze_excite_block, channel_squeeze_excite_block, spatial_squeeze_excite_block

def conv_block(inputs, training, params, reuse=None):

    with tf.variable_scope("Basic_conv_block", reuse=reuse):

        # Get standard hyperparameters
        bn_momentum = params.get('batchnorm_decay', 0.99)
        # Get custom hyperparameters
        filters_list = [layer['filters'] for layer in
                params['basic']['conv_block']['layers']]
        kernel_sizes = [layer['kernel_size'] for layer in
                params['basic']['conv_block']['layers']]
        numbers_list = [layer.get('number', 1) for layer in
                        params['basic']['conv_block']['layers']]
        max_pool = params['basic']['conv_block']['max_pool']
        bottleneck_filters = params['basic']['conv_block']['bottleneck']
        batchnorm = params['basic']['conv_block'].get('batchnorm', False)
        attention = params.get('attention')

        x = inputs
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                                              training=training)

        for i, (filters, kernel_size, number) in enumerate(
                zip(filters_list, kernel_sizes, numbers_list)):
            for nr in range(number):
                x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                                     activation=tf.nn.relu, padding="same", reuse=reuse,
                                     name="conv_{}_{}".format(i + 1, nr + 1))
            if max_pool:
                x = tf.layers.max_pooling2d(x, pool_size=max_pool['size'],
                                            strides=max_pool['strides'], name="pool_{}".format(i + 1))
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
