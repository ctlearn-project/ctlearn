import tensorflow as tf
import numpy as np

NUM_CLASSES = 2

IMAGE_WIDTH = 240
IMAGE_LENGTH = 240
IMAGE_DEPTH = 3

NUM_TEL = 15

#for use with train_datasets
def alexnet_base_cnn_v2(input_features,number):

    #def reuse_on(): reuse = True
    #def reuse_off(): reuse = False
    #tf.cond(tf.equal(number,tf.constant(0)),reuse_on,reuse_off)
    #shared weights
    #if tf.equal(number,tf.constant(0)):
        #reuse = False
    #else:
        #reuse = True

    reuse = False
    with tf.variable_scope("Conv_block_T" + str(number)):
        #input
        input_layer = tf.reshape(input_features, [-1, IMAGE_WIDTH, IMAGE_LENGTH, IMAGE_DEPTH],name="input")

        #conv1
        conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=96,
                kernel_size=[11, 11],
                strides=4,
                padding="valid",
                activation=tf.nn.relu,
                name="conv1",
                reuse=reuse,
                kernel_initializer = tf.zeros_initializer())

        #local response normalization ???

        #pool1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

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
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

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
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

        #flatten output of pool5 layer to get feature vector
        #reshape shape = 1024
        dim = np.prod(pool5.get_shape().as_list()[1:])
        reshape = tf.reshape(pool5, [-1, dim])

    return reshape

#for use with train_datasets
def custom_multi_input_v2(tel_data,labels):
   
    tel_data_transposed = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])

    feature_vectors = []
    
    """
    i = tf.constant(0)
    def body(i):
        global TEL_COUNTER
        print(TEL_COUNTER)
        with tf.control_dependencies(None):
            feature_vectors.append(alexnet_base_cnn_v1(tf.gather(tel_data_transposed,i),TEL_COUNTER))
        TEL_COUNTER = TEL_COUNTER + 1
        print(TEL_COUNTER)
        return tf.add(i, 1)

    def while_condition(i):
        return tf.less(i, num_tels)

    tf.while_loop(while_condition, body, [i])
    """

    for i in range(NUM_TEL):
        feature_vectors.append(alexnet_base_cnn_v2(tf.gather(tel_data_transposed,i),i))

    with tf.variable_scope("Classifier"):

        #combine the feature vectors into a tensor of shape [batch size,num_tels*num_features]
        combined_feature_tensor = tf.concat(feature_vectors, 1)

        #fc6
        fc6 = tf.layers.dense(inputs=combined_feature_tensor, units=4096, activation=tf.nn.relu,name="fc6") 
        dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=True)

        #fc7
        fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,name="fc7")        
        dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=True)        

        #fc8
        fc8 = tf.layers.dense(inputs=dropout7, units=NUM_CLASSES,name="fc8")

    with tf.variable_scope("Outputs"):

        # Calculate Loss (for both TRAIN and EVAL modes) 
        if NUM_CLASSES == 2:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=onehot_labels, logits=fc8)
        else:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=fc8)

        #outputs
        if NUM_CLASSES == 2:
            predictions = {
                    "classes": tf.argmax(input=fc8,axis=1),
                    "probabilities": [tf.sigmoid(fc8), 1-tf.sigmoid(fc8)]
                    }
        else:
            predictions = {        
                    "classes": tf.argmax(input=fc8, axis=1),
                    "probabilities": tf.nn.softmax(fc8, name="softmax_tensor")
                    }

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictions['classes'],tf.int8),labels), tf.float32))

    return loss,accuracy,fc8,predictions


def alexnet_base_cnn_v1(input_features,number):

    #shared weights
    if number == 0:
        reuse = None
    else:
        reuse = True

    with tf.variable_scope("Conv_block"):
        #input
        input_layer = tf.reshape(input_features, [-1, IMAGE_WIDTH, IMAGE_LENGTH, IMAGE_DEPTH],name="input")

        #conv1
        conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=96,
                kernel_size=[11, 11],
                strides=4,
                padding="valid",
                activation=tf.nn.relu,
                name="conv1",
                reuse=reuse)

        #local response normalization ???

        #pool1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

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
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

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
        pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

        #flatten output of pool5 layer to get feature vector
        #reshape shape = 1024
        dim = np.prod(pool5.get_shape().as_list()[1:])
        reshape = tf.reshape(pool5, [-1, dim])

    return reshape

def custom_multi_input(tel_data,labels):
   
    num_tels = tel_data.get_shape().as_list()[1]
    tel_data_transposed = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])

    feature_vectors = []
    for i in range(num_tels):
        feature_vectors.append(alexnet_base_cnn_v1(tf.gather(tel_data_transposed,i),i))

    with tf.variable_scope("Classifier"):

        #combine the feature vectors into a tensor of shape [batch size,num_tels*num_features]
        combined_feature_tensor = tf.concat(feature_vectors, 1)

        #fc6
        fc6 = tf.layers.dense(inputs=combined_feature_tensor, units=4096, activation=tf.nn.relu,name="fc6") 
        dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=True)

        #fc7
        fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,name="fc7")        
        dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=True)        

        #fc8
        fc8 = tf.layers.dense(inputs=dropout7, units=NUM_CLASSES,name="fc8")

    with tf.variable_scope("Outputs"):

        # Calculate Loss (for both TRAIN and EVAL modes) 
        if NUM_CLASSES == 2:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=onehot_labels, logits=fc8)
        else:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=fc8)

        #outputs
        if NUM_CLASSES == 2:
            predictions = {
                    "classes": tf.argmax(input=fc8,axis=1),
                    "probabilities": [tf.sigmoid(fc8), 1-tf.sigmoid(fc8)]
                    }
        else:
            predictions = {        
                    "classes": tf.argmax(input=fc8, axis=1),
                    "probabilities": tf.nn.softmax(fc8, name="softmax_tensor")
                    }

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictions['classes'],tf.int8),labels), tf.float32))

    return loss,accuracy,fc8,predictions


