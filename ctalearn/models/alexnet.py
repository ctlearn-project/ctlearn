"""
Based on ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky, Sutskever, Hinton 2012)
"""

import tensorflow as tf


def alexnet_block(inputs, params={}, is_training=True, reuse=None):

    with tf.variable_scope("AlexNet_block"):

        # Initial conv layer stride decreased from 4 to 2 due to smaller input size
        conv1 = tf.layers.conv2d(inputs,filters=96,kernel_size=[11, 11],strides=2,activation=tf.nn.relu,name="conv1",reuse=reuse)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=[3, 3], strides=2,name="pool1")
        
        conv2 = tf.layers.conv2d(pool1,filters=256,kernel_size=[5, 5],activation=tf.nn.relu,name="conv2",reuse=reuse)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=2,name="pool2")

        conv3 = tf.layers.conv2d(pool2,filters=384,kernel_size=[3,3],activation=tf.nn.relu,name="conv3",reuse=reuse)
        conv4 = tf.layers.conv2d(conv3,filters=384,kernel_size=[3, 3],activation=tf.nn.relu,name="conv4",reuse=reuse)
        conv5 = tf.layers.conv2d(conv4,filters=256,kernel_size=[3, 3],activation=tf.nn.relu,name="conv5",reuse=reuse)
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2,name="pool5")

        return pool5

def alexnet_head_feature_vector(inputs, params={}, is_training=True):
    
    # Get hyperparameters
    dropout_rate = params.get('dropout_rate', 0.5)
    num_classes = params.get('num_classes', 2)

    fc6 = tf.layers.dense(inputs_flattened, units=4096, activation=tf.nn.relu, name="fc6") 
    dropout6 = tf.layers.dropout(fc6, rate=dropout_rate, training=is_training)

    fc7 = tf.layers.dense(dropout6, units=4096, activation=tf.nn.relu, name="fc7")        
    dropout7 = tf.layers.dropout(fc7, rate=dropout_keep_prob, training=is_training)        

    logits = tf.layers.dense(dropout7, units=num_classes, name="logits")

    return logits

# Identical to the original Alexnet fully connected layer section but with the
# fully connected layers replaced by additional convolutional layers
# Based on example from https://github.com/tensorflow/models/blob/master/research/slim/nets/alexnet.py
def alexnet_head_feature_map(inputs, params={}, is_training=True):
    
    # Get hyperparameters
    dropout_rate = params.get('dropout_rate', 0.5)
    num_classes = params.get('num_classes', 2)

    conv6 = tf.layers.conv2d(inputs, filters=4096, kernel_size=[5, 5], activation=tf.nn.relu, name="conv6")
    dropout6 = tf.layers.dropout(conv6, rate=dropout_rate, training=is_training)

    conv7 = tf.layers.conv2d(dropout6, filters=4096, kernel_size=[1,1], activation=tf.nn.relu, name="conv7")        
    dropout7 = tf.layers.dropout(conv7, rate=dropout_keep_prob, training=is_training)        

    logits = tf.layers.dense(dropout7, units=num_classes, name="logits")

    return logits


