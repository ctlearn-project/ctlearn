"""
Based on Densely Connected Convolutional Networks (Huang et. al., 2016)
"""

import numpy as np
import tensorflow as tf
from ctalearn.models.variable_input_model import trigger_dropout

NUM_CLASSES = 2

GROWTH_RATE = 32

# Based on Densenet-121 with the final dense block omitted
NUM_DENSE_BLOCKS = 3
NUM_LAYERS_PER_BLOCK = [6,12,24]

"""
Combined densenet convolution layer (BN,ReLU,Conv) + Dropout
"""
def densenet_conv_layer(inputs,kernel_size,num_filters,dropout_rate=0.2,training=True):
    
    output = tf.layers.batch_normalization(inputs,training=training,fused=True)
    output = tf.nn.relu(output)
    output = tf.layers.conv2d(output,filters=num_filters,kernel_size=kernel_size,padding='same')
    output = tf.layers.dropout(output,rate=dropout_rate,training=training)

    return output

"""
densenet block of multiple convolution and bottleneck layers
"""
def densenet_dense_block(inputs,k,num_layers,training):
    
    output = inputs
    for i in range(num_layers):
        with tf.variable_scope("layer_{}".format(i+1)):
            with tf.variable_scope("bottleneck"):
                output = densenet_conv_layer(output,kernel_size=1,num_filters=4*k,training=training)
            with tf.variable_scope("conv"):
                output = densenet_conv_layer(output,kernel_size=3,num_filters=k,training=training)
            #concatenate input and output feature maps
            output = tf.concat([inputs,output],axis=3)

    return output

"""
densenet transition layer
theta is compression factor from original paper. num_input_feature_maps * theta = num_output_feature_maps
"""
def densenet_transition_layer(inputs,training,theta=0.5):

    input_num_filters = int(inputs.get_shape()[-1])
    #1x1 convolution (compress number of filters by factor theta) followed by average pooling
    with tf.variable_scope("bottleneck"):
        output = densenet_conv_layer(inputs,kernel_size=1,num_filters=int(theta*input_num_filters),training=training)
    output = tf.layers.average_pooling2d(output,pool_size=2,strides=2)

    return output

"""
Densenet CNN (kernel sizes, pool sizes, strides based on densenet-bc imagenet model)
With a 120x120 input, returns
"""
def densenet_block(inputs, k=GROWTH_RATE,num_dense_blocks=NUM_DENSE_BLOCKS,triggers=None, params=None, is_training=True, reuse=None):

    with tf.variable_scope("DenseNet_block",reuse=reuse):
        with tf.variable_scope("initial_conv"):
            output = tf.layers.conv2d(inputs,kernel_size=7,strides=1,filters=k,padding='same') 
        output = tf.layers.max_pooling2d(output,pool_size=3,strides=2,padding='same')

        for i in range(num_dense_blocks):
            with tf.variable_scope("block_{}".format(i+1)):
                output = densenet_dense_block(output,k,num_layers=NUM_LAYERS_PER_BLOCK[i],training=is_training)
                # After every dense block except the last one
                if i != num_dense_blocks-1:
                    with tf.variable_scope("transition_after_block_{}".format(i+1)):
                        output = densenet_transition_layer(output,training=is_training)

    # Trigger dropout (mask outputs with binary trigger values)
    if triggers is not None:
        output = trigger_dropout(output,triggers)

    return output
    
