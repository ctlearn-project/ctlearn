from collections import namedtuple
import numpy as np

import tensorflow as tf
from tensorflow.contrib.layers import flatten

slim = tf.contrib.slim

NUM_CLASSES = 2

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# Specifies the MobileNet body for the single telescope blocks
# This is a custom MobileNet design. It is designed for 120x120 image input
# and produces 15x15 output. The number of layers is set so that every pixel
# in the final layer has input derived from the entire image. This is the
# single telescope component. The final layers should be stacked to produce
# a 15x15x(64*NUM_TEL + NUM_AUX_PARAMS*NUM_TEL) input layer for the array
# level network.
BLOCK_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=8),
    DepthSepConv(kernel=[3, 3], stride=1, depth=16),
    DepthSepConv(kernel=[3, 3], stride=2, depth=16),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64)
]

# Specifies the MobileNet body for the array level network
# Custom MobileNet array level network. The input should be stacked MobileNet
# block final layers plus additional layers for auxiliary input. The 
# classification head should be an Avg Pool layer followed by a classifier 
# with 1024 inputs for whatever output is desired.
HEAD_CONV_DEFS = [
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

# Define a MobileNet body
# scope is a scope or name
# inputs is the input layer tensor
# conv_defs is a list of ConvDef named tuples
# reuse should be None or True
def mobilenet_base(scope, inputs, conv_defs, is_training=True, reuse=None):
    end_points = {}
    with tf.variable_scope(scope, inputs, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training, 
                    decay=0.95):
                net = inputs
                for i, conv_def in enumerate(conv_defs):
                    end_point_base = 'Conv2d_%d' % i

                    if isinstance(conv_def, Conv):
                        end_point = end_point_base
                        net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                stride=conv_def.stride,
                                normalizer_fn=slim.batch_norm,
                                scope=end_point)
                        end_points[end_point] = net
                    elif isinstance(conv_def, DepthSepConv):
                        end_point = end_point_base + '_depthwise'

                        # By passing filters=None separable_conv2d produces 
                        # only a depthwise convolution layer
                        net = slim.separable_conv2d(net, None, conv_def.kernel,
                                depth_multiplier=1,
                                stride=conv_def.stride,
                                normalizer_fn=slim.batch_norm,
                                scope=end_point)

                        end_points[end_point] = net

                        end_point = end_point_base + '_pointwise'

                        net = slim.conv2d(net, conv_def.depth, [1, 1],
                                          stride=1,
                                          normalizer_fn=slim.batch_norm,
                                          scope=end_point)

                        end_points[end_point] = net
                    else:
                        raise ValueError('Unknown convolution type %s for '
                                'layer %d' % (conv_def.ltype, i))
    return net, end_points

def mobilenet_block(inputs, telescope_index, trig_values, is_training=True):
    # Set all telescopes after the first to share weights
    if telescope_index == 0:
        reuse = None
    else:
        reuse = True

    net, end_points = mobilenet_base("MobileNetBlock", inputs, BLOCK_CONV_DEFS, 
            is_training, reuse)
    
    # Drop out all outputs if the telescope was not triggered
    end_point = "Trigger_multiplier"
    # Reshape trig_values from [BATCH_SIZE] to [BATCH_SIZE, WIDTH, HEIGHT, 
    # NUM_CHANNELS]
    trig_values = tf.reshape(trig_values, [-1, 1, 1, 1])
    trig_values = tf.tile(trig_values, tf.concat([[1], tf.shape(net)[1:]], 0))
    net = tf.multiply(net, trig_values)
    end_points[end_point] = net
    
    # For compatibility with variable_input_model, do not return
    # end_points for now
    return net#, end_points

def mobilenet_head(inputs, dropout_keep_prob=0.9, num_classes=2, 
        is_training=True):
    # Define the network
    net, end_points = mobilenet_base("MobileNetHead", inputs, HEAD_CONV_DEFS, 
            is_training=is_training)
    
    with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [15, 15], padding='VALID', 
                scope='AvgPool_1a')
        end_points['AvgPool_1a'] = net
        # 1 x 1 x 1024
        net = slim.dropout(net, keep_prob=dropout_keep_prob, 
                is_training=is_training, scope='Dropout_1b')
        # Essentially a fully connected layer
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        # Reshape from [BATCH_SIZE, 1, 1, num_classes] to 
        # [BATCH_SIZE, num_classes]
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Logits'] = logits
    return logits#, end_points


