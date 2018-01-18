import tensorflow as tf

#for use with train_datasets
def alexnet_block(inputs, triggers, params=None, is_training=True, reuse=None):

    with tf.variable_scope("Conv_block"):

        #conv1
        conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=96,
                kernel_size=[11, 11],
                strides=2, # changed from strides=4 for small image sizes
                padding="valid",
                activation=tf.nn.relu,
                name="conv1",
                reuse=reuse,
                kernel_initializer = tf.zeros_initializer())

        #local response normalization ???

        #pool1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], 
                strides=2)

        #conv2
        conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=256,
                kernel_size=[5, 5],
                padding="valid",
                activation=tf.nn.relu,
                name="conv2",
                reuse=reuse)

        #normalization ????

        #pool2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], 
                strides=2)

        #conv3
        conv3 = tf.layers.conv2d(
                inputs=pool2,
                filters=384,
                kernel_size=[3,3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv3",
                reuse=reuse)

        #conv4
        conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=384,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv4",
                reuse=reuse)

        #conv5
        conv5 = tf.layers.conv2d(
                inputs=conv4,
                filters=256,
                kernel_size=[3, 3],
                padding="valid",
                activation=tf.nn.relu,
                name="conv5",
                reuse=reuse)

        #pool5
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], 
                strides=2)

        #flatten output of pool5 layer to get feature vector of shape 
        # (num_batch,1024)
        output = tf.multiply(tf.layers.Flatten(pool5), tf.expand_dims(trig_values, 1))

        return output

def alexnet_head(inputs, params=None, is_training=True):
    
    # Get hyperparameters
    if not params:
        params = {}
    dropout_keep_prob = params.get('dropout_keep_prob', 0.5)
    num_classes = params.get('num_gamma_hadron_classes', 2)
    
    #fc6
    fc6 = tf.layers.dense(inputs=inputs, units=4096, activation=tf.nn.relu,
    name="fc6") 
    dropout6 = tf.layers.dropout(inputs=fc6, rate=dropout_keep_prob, 
    training=is_training)

    #fc7
    fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,
    name="fc7")        
    dropout7 = tf.layers.dropout(inputs=fc7, rate=dropout_keep_prob, 
    training=is_training)        

    #fc8
    fc8 = tf.layers.dense(inputs=dropout7, units=num_classes, name="fc8")

    return fc8

