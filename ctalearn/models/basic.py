import tensorflow as tf

BASIC_CONV_MODEL_LAYERS = [(32,3),(64,3),(128,3),(256,3)]
BASIC_CONV_MODEL_BOTTLENECK = 8

BASIC_FC_HEAD_LAYERS = [1024,512]

BASIC_CONV_HEAD_LAYERS = [(64,3),(128,3),(256,3)]

def basic_conv_block(inputs, params=None, is_training=True, reuse=None):

    with tf.variable_scope("Basic_conv_block"):

        x = tf.layers.batch_normalization(inputs, training=is_training)

        for i, (filters, kernel_size) in enumerate(BASIC_CONV_MODEL_LAYERS):
            x = tf.layers.conv2d(x,filters=filters,kernel_size=kernel_size,activation=tf.nn.relu,padding="same",reuse=reuse,name="conv_{}".format(i+1))
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name="pool_{}".format(i+1))
            x = tf.layers.batch_normalization(x, training=is_training)

        # global max pool

        # bottleneck layer
        x = tf.layers.conv2d(x,filters=BASIC_CONV_MODEL_BOTTLENECK,kernel_size=1,activation=tf.nn.relu,padding="same",reuse=reuse,name="bottleneck")
        x = tf.layers.batch_normalization(x, training=is_training)

        return x

def basic_head_fc(inputs, params=None, is_training=True):

    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)

    x = tf.layers.flatten(inputs)

    for i, num_units in enumerate(BASIC_FC_HEAD_LAYERS):
        x = tf.layers.dense(x, units=num_units, activation=tf.nn.relu, name="fc_{}".format(i+1))

    logits = tf.layers.dense(x, units=num_classes, name="logits")

    return logits

def basic_head_conv(inputs, params=None, is_training=True):

    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)

    x = inputs

    for i, (filters, kernel_size) in enumerate(BASIC_CONV_HEAD_LAYERS):
        x = tf.layers.conv2d(x,filters=filters,kernel_size=kernel_size,activation=tf.nn.relu,padding="same",name="conv_{}".format(i+1))
        x = tf.layers.batch_normalization(x, training=is_training)

    pool = tf.layers.average_pooling2d(x, pool_size=x.get_shape().as_list()[1], strides=1, name="global_avg_pool")
    flatten = tf.layers.flatten(pool)

    logits = tf.layers.dense(flatten, units=num_classes, name="logits")

    return logits


