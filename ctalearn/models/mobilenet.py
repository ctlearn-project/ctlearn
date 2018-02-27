from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.slim as slim

# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

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
# Define the factor by which the input image sizes will be reduced. Combined
# with the processed image size, this allows the network head to account for
# having smaller images (i.e. from cropping) when pooling.
IMAGE_SIZE_REDUCTION = 8

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
def mobilenet_base(scope, inputs, conv_defs, batch_norm_decay=0.95, 
        is_training=True, reuse=None):
    end_points = {}
    with tf.variable_scope(scope, inputs, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training, 
                    decay=batch_norm_decay):
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

def mobilenet_block(inputs, params=None, is_training=True, reuse=None):
    
    # Get hyperparameters
    if params is None: params = {}
    batch_norm_decay = params.get('batch_norm_decay', 0.95)

    net, end_points = mobilenet_base("MobileNetBlock", inputs, BLOCK_CONV_DEFS,
            batch_norm_decay, is_training, reuse)

    # For compatibility with variable_input_model, do not return
    # end_points for now
    return net#, end_point

def mobilenet_head(inputs, params=None, is_training=True):

    # Get hyperparameters
    if params is None: params = {}
    dropout_keep_prob = params.get('dropout_keep_prob', 0.9)
    num_classes = params.get('num_gamma_hadron_classes', 2)
    try: 
        telescope_type = params['processed_telescope_types'][0]
        image_width, image_length, image_depth = (
                params['processed_image_shapes'][telescope_type])
    except KeyError:
        image_width, image_length = 120, 120
    if (image_width % IMAGE_SIZE_REDUCTION) != 0 or (image_length % 
            IMAGE_SIZE_REDUCTION) != 0:
        raise ValueError("Image dimensions not a multiple of {}".format(
            IMAGE_SIZE_REDUCTION))
    pool_width = int(image_width / IMAGE_SIZE_REDUCTION)
    pool_length = int(image_length / IMAGE_SIZE_REDUCTION)

    # Define the network
    net, end_points = mobilenet_base("MobileNetHead", inputs, HEAD_CONV_DEFS, 
            is_training=is_training)
    
    with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, [pool_width, pool_length], padding='VALID', 
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


