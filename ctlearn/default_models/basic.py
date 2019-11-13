import tensorflow as tf

from tensorflow.layers import conv2d,max_pooling2d
from indexedconv.engine.tf.masked import masked_conv2d,masked_avgpool2d
from indexedconv.engine.tf.masked import indexed_conv2d,indexed_avgpool2d

def conv_block(inputs, training, params, reuse=None):

    with tf.variable_scope("Basic_conv_block", reuse=reuse):

        # Get standard hyperparameters
        bn_momentum = params.get('batchnorm_decay', 0.99)
        # Get custom hyperparameters
        filters_list = [layer['filters'] for layer in
                params['basic']['conv_block']['layers']]
        hexagonal_convolution = params['basic']['conv_block'].get('hexagonal_convolution', False)
        indexed_convolution = params['basic']['conv_block'].get('indexed_convolution', False)
        kernel_sizes = [layer['kernel_size'] for layer in
                params['basic']['conv_block']['layers']]
        pool = params['basic']['conv_block']['pool']
        bottleneck_filters = params['basic']['conv_block']['bottleneck']
        batchnorm = params['basic']['conv_block'].get('batchnorm', False)
        
        x = inputs
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

        for i, (filters, kernel_size) in enumerate(
                zip(filters_list, kernel_sizes)):
                
            if indexed_convolution:
                x = indexed_conv2d(x, filters=filters, kernel_size=kernel_size,
                    activation=tf.nn.relu, padding="same",
                    name="indexed_conv2d_{}".format(i+1), reuse=reuse)
            elif hexagonal_convolution:
                x = masked_conv2d(x, filters=filters, kernel_size=kernel_size,
                    activation=tf.nn.relu, padding="same",
                    name="masked_conv2d_{}".format(i+1), reuse=reuse)
            else:
                x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                    activation=tf.nn.relu, padding="same",
                    name="conv2d_{}".format(i+1), reuse=reuse)
            if pool:
                if indexed_convolution:
                    x = indexed_avgpool2d(x, pool_size=pool['size'],
                    strides=pool['strides'], name="avg_pool_{}".format(i+1))
                elif hexagonal_convolution:
                    x = masked_avgpool2d(x, pool_size=pool['size'],
                        strides=pool['strides'], name="avg_pool_{}".format(i+1))
                else:
                    x = tf.layers.max_pooling2d(x, pool_size=pool['size'],
                        strides=pool['strides'], name="max_pool_{}".format(i+1))
            
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

        # bottleneck layer
        if bottleneck_filters:
            if indexed_convolution:
                x = indexed_conv2d(x, filters=bottleneck_filters,
                    kernel_size=1, activation=tf.nn.relu, padding="same",
                    reuse=reuse, name="bottleneck")
            elif hexagonal_convolution:
                x = masked_conv2d(x, filters=bottleneck_filters,
                    kernel_size=1, activation=tf.nn.relu, padding="same",
                    reuse=reuse, name="bottleneck")
            else:
                x = tf.layers.conv2d(x, filters=bottleneck_filters,
                    kernel_size=1, activation=tf.nn.relu, padding="same",
                    reuse=reuse, name="bottleneck")
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

        return x

def fc_head(inputs, training, params):

    # Get standard hyperparameters
    bn_momentum = params['basic'].get('batchnorm_decay', 0.99)
    
    # Get custom hyperparameters
    layers = params['basic']['fc_head']['layers']
    batchnorm = params['basic']['fc_head'].get('batchnorm', False)

    x = tf.layers.flatten(inputs)

    for i, units in enumerate(layers):
        x = tf.layers.dense(x, units=units, activation=tf.nn.relu,
                name="fc_{}".format(i+1))
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

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
