"""
See paper at https://arxiv.org/abs/1608.06993
"""

import numpy as np
import tensorflow as tf

NUM_CLASSES = 2

GROWTH_RATE = 36
NUM_DENSE_BLOCKS = 3
NUM_LAYERS_PER_BLOCK = [6,12,24]

"""
Combined densenet convolution layer (BN,ReLU,Conv)
"""
def densenet_conv_layer(inputs,kernel_size,num_filters,dropout_rate=0.2,training=True):
    
    output = tf.layers.batch_normalization(inputs,training=training)
    output = tf.nn.relu(output)
    output = tf.layers.conv2d(output,filters=num_filters,kernel_size=kernel_size)
    output = tf.layers.dropout(output,rate=dropout_rate,training=training)

    return output

"""
densenet block of multiple convolution and bottleneck layers
"""
def densenet_dense_block(inputs,k,num_layers,training):
    
    output = inputs
    for i in range(num_layers):
        with tf.variable_scope("layer_{}".format(i+1)):
            #bottleneck layer (1x1 conv)
            with tf.variable_scope("bottleneck"):
                output = densenet_conv_layer(output,kernel_size=1,num_filters=4*k,training=training)
            #conv layer (3x3 filter)
            with tf.variable_scope("conv"):
                output = densenet_conv_layer(output,kernel_size=3,num_filters=k,training=training)
            #concatenate input and output feature maps
            output = tf.concat([inputs,output],axis=3)

    return output

"""
densenet transition layer
theta is compression factor from original paper. Reduce number of feature maps by a factor theta
"""
def densenet_transition_layer(inputs,theta=0.5,training):

    input_num_filters = int(inputs.get_shape()[-1])

    #1x1 convolution (compress number of filters by factor theta) followed by average pooling (size 2x2, stride 2)
    output = densenet_conv_layer(inputs,kernel_size=1,num_filters=int(theta*input_num_filters),training=training)
    output = tf.layers.AveragePooling2D(output,pool_size=2,strides=2)

    return output

"""
Densenet CNN (based on densenet-bc imagenet model)
With a 120x120 input, returns a 15x15 output
"""
def densenet_block(inputs, k=GROWTH_RATE,num_dense_blocks=NUM_DENSE_BLOCKS,triggers=None, params=None, is_training=True, reuse=None):

    with tf.variable_scope("DenseNet_block",reuse=reuse):
        with tf.variable_scope("initial_conv"):
            output = tf.layers.conv2d(inputs,kernel_size=3,strides=2,num_filters=k)
        output = tf.layers.MaxPooling2D(output,pool_size=3,strides=1)

        for i in range(num_dense_blocks):
            with tf.variable_scope("block_{}".format(i+1)):
                output = densenet_dense_block(output,k,num_layers=NUM_LAYERS_PER_BLOCK[i],training=is_training)
                if i != num_dense_blocks-1:
                    output = densenet_transition_layer(output,training=is_training)

    if triggers is not None:
        # Drop out all outputs if the telescope was not triggered
        # Reshape triggers from [BATCH_SIZE] to [BATCH_SIZE, WIDTH, HEIGHT, 
        # NUM_CHANNELS]
        triggers = tf.reshape(triggers, [-1, 1, 1, 1])
        triggers = tf.tile(triggers, tf.concat([[1], tf.shape(output)[1:]], 0))
        output = tf.multiply(output, triggers)
 
    return output
    
