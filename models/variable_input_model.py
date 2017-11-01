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
def mobilenet_body(scope, inputs, conv_defs, is_training=True, reuse=None):
    end_points = {}
    with tf.variable_scope(scope, inputs, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                padding='SAME'):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                for i, conv_def in enumerate(BLOCK_CONV_DEFS):
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

    # Define the network
    net, end_points = mobilenet_body("MobileNetBlock", inputs, 
            BLOCK_CONV_DEFS, is_training, reuse)
    
    # Drop out all outputs if the telescope was not triggered
    end_point = "Trigger_multiplier"        
    net = tf.multiply(flatten(net), tf.expand_dims(trig_values, 1))
    end_points[end_point] = net
    
    # For compatibility with variable_input_model, do not return end_points
    # for now
    return net#, end_points

def mobilenet_head(inputs, dropout_keep_prob=0.9, num_classes=2, 
        is_training=True):
    # Define the network
    net, end_points = mobilenet_body("MobileNetHead", inputs, HEAD_CONV_DEFS, 
            is_training)
    
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

#for use with train_datasets
def alexnet_block(input_features, number, trig_values):

    #shared weights
    if number == 0:
        reuse = None
    else:
        reuse = True

    with tf.variable_scope("Conv_block"):
        #conv1
        conv1 = tf.layers.conv2d(
                inputs=input_features,
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

        # Flatten output of pool5 layer to get feature vector
        output = tf.multiply(flatten(pool5), tf.expand_dims(trig_values, 1))

    return output

def alexnet_head(inputs, dropout_keep_prob=0.5, num_classes=2, 
        is_training=True):
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

#for use with train_datasets
def variable_input_model(tel_data, labels, trig_list, tel_pos_tensor, num_tel,
        image_width, image_length, image_depth, is_training):
 
    # Reshape inputs into proper dimensions
    tel_data = tf.reshape(tel_data, [-1, num_tel, image_width, image_length, 
        image_depth])
    trig_list = tf.reshape(trig_list, [-1, num_tel])
    # TODO: move number of aux inputs (2) to be defined as a constant
    tel_pos_tensor = tf.reshape(tel_pos_tensor, [num_tel, 2])
    
    # Split data by telescope by switching the batch and telescope dimensions
    # leaving width, length, and channel depth unchanged
    tel_data_by_telescope = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])

    # Define the network being used. Each CNN block analyzes a single
    # telescope. The outputs are stacked, with the outputs for non-triggering
    # telescopes zeroed out (effectively, those channels are dropped out).
    # Unlike standard dropout, this zeroing-out procedure is performed both at
    # training and at test time since it encodes meaningful aspects of the
    # data.
    # The array-level processing in then performed by the network head. The
    # logits are returned and fed into a classifier.
    cnn_block = alexnet_block
    network_head = alexnet_head

    # Process the input for each telescope
    telescope_outputs = []
    for i in range(num_tel):
        telescope_features = cnn_block(tf.gather(tel_data_by_telescope, i), i,
                tf.gather(trig_list, i, axis=1))
        telescope_outputs.append(telescope_features)

    with tf.variable_scope("NetworkHead"):
        # Process the single telescope data into array-level input
        array_inputs = []
        for i, telescope_features in enumerate(telescope_outputs):
            # Flatten output features to get feature vectors
            telescope_features = flatten(telescope_features)
            # Get the telescope x and y position and if it triggered
            telescope_position = tel_pos_tensor[i, :]
            telescope_position = tf.tile(tf.expand_dims(telescope_position, 0),
                    [tf.shape(telescope_features)[0], 1])
            telescope_trigger = tf.expand_dims(trig_list[:, i], 1)
            # Insert auxiliary input into each feature vector
            telescope_features = tf.concat([telescope_features, 
                telescope_position, telescope_trigger], 1)
            array_inputs.append(telescope_features)
        array_features = tf.stack(array_inputs, axis=1)
        # Process the combined array features
        logits = network_head(array_features, num_classes=NUM_CLASSES, 
                is_training=is_training)

    with tf.variable_scope("Outputs"):

        # Calculate Loss (for both TRAIN and EVAL modes) 
        if NUM_CLASSES == 2:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=onehot_labels, logits=logits)
        else:
            onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        #outputs
        if NUM_CLASSES == 2:
            predictions = {
                    "classes": tf.argmax(input=logits,axis=1),
                    "probabilities": [tf.sigmoid(logits), 1-tf.sigmoid(logits)]
                    }
        else:
            predictions = {        
                    "classes": tf.argmax(input=logits, axis=1),
                    "probabilities": tf.nn.softmax(logits, 
                        name="softmax_tensor")
                    }

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(predictions['classes'],tf.int8),labels), tf.float32))

    return loss, accuracy, logits, predictions
