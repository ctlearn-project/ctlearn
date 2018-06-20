import tensorflow as tf

BASIC_FC_HEAD_LAYERS = [1024,512,256,128,64]

BASIC_CONV_HEAD_LAYERS = [(64,3),(128,3),(256,3)]

def basic_conv_block(inputs, training, params=None, reuse=None):

    with tf.variable_scope("Basic_conv_block", reuse=reuse):

        if params is None: params = {}
        # Get standard hyperparameters
        bn_momentum = float(params.get('batchnormdecay', 0.99))
        # Get custom hyperparameters
        filters_list = [int(f) for f in
                params.get('basicconvblockfilters').split('|')]
        kernels = [int(k) for k in
                params.get('basicconvblockkernels').split('|')]
        max_pool = bool(params.get('basicconvblockmaxpool', True))
        if max_pool:
            max_pool_size = int(params.get('basicconvblockmaxpoolsize'))
            max_pool_strides = int(params.get('basicconvblockmaxpoolstrides'))
        bottleneck = bool(params.get('basicconvblockbottleneck', False))
        if bottleneck:
            bottleneck_filters = int(
                    params.get('basicconvblockbottleneckfilters'))
        batchnorm = bool(params.get('basicconvblockbatchnorm'))
        
        x = inputs
        if batchnorm:
            x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                    training=training)

        for i, (filters, kernel_size) in enumerate(zip(filters_list, kernels)):
            x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size,
                    activation=tf.nn.relu, padding="same", reuse=reuse,
                    name="conv_{}".format(i+1))
            if max_pool:
                x = tf.layers.max_pooling2d(x, pool_size=max_pool_size,
                        strides=max_pool_strides, name="pool_{}".format(i+1))
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

        # bottleneck layer
        if bottleneck:
            x = tf.layers.conv2d(x, filters=bottleneck_filters,
                    kernel_size=1, activation=tf.nn.relu, padding="same",
                    reuse=reuse, name="bottleneck")
            if batchnorm:
                x = tf.layers.batch_normalization(x, momentum=bn_momentum,
                        training=training)

    return x

def basic_head_fc(inputs, params=None, training=True):

    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)

    x = tf.layers.flatten(inputs)

    for i, num_units in enumerate(BASIC_FC_HEAD_LAYERS):
        x = tf.layers.dense(x, units=num_units, activation=tf.nn.relu, name="fc_{}".format(i+1))
        x = tf.layers.batch_normalization(x, training=training)

    logits = tf.layers.dense(x, units=num_classes, name="logits")

    return logits

def basic_head_conv(inputs, params=None, training=True):

    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)

    x = inputs

    for i, (filters, kernel_size) in enumerate(BASIC_CONV_HEAD_LAYERS):
        x = tf.layers.conv2d(x,filters=filters,kernel_size=kernel_size,activation=tf.nn.relu,padding="same",name="conv_{}".format(i+1))
        x = tf.layers.batch_normalization(x, training=training)

    pool = tf.layers.average_pooling2d(x, pool_size=x.get_shape().as_list()[1], strides=1, name="global_avg_pool")
    flatten = tf.layers.flatten(pool)

    logits = tf.layers.dense(flatten, units=num_classes, name="logits")

    return logits


