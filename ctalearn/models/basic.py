import tensorflow as tf

BASIC_CONV_MODEL_LAYERS = [(32,3),(64,3),(128,3)]

BASIC_HEAD_LAYERS = [1024]

def basic_conv_block(inputs, params=None, is_training=True, reuse=None):

    with tf.variable_scope("Basic_conv_block"):

        x = inputs

        for i, (filters, kernel_size) in enumerate(BASIC_CONV_MODEL_LAYERS):
            x = tf.layers.conv2d(x,filters=filters,kernel_size=kernel_size,activation=tf.nn.relu,padding="same",reuse=reuse,name="conv_{}".format(i+1))
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name="pool_{}".format(i+1))
            
        return x

def basic_head_feature_vector(inputs, params=None, is_training=True):
    
    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)

    x = inputs

    for i, num_units in enumerate(BASIC_HEAD_LAYERS):
        x = tf.layers.dense(x, units=num_units, activation=tf.nn.relu,name="fc_{}".format(i+1))

    logits = tf.layers.dense(x, units=num_classes, name="logits")

    return logits


