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

# CONV_DEFS specifies the MobileNet body
# Modified from standard MobileNet to account for small image size and to fit
# in memory
CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=8),
    DepthSepConv(kernel=[3, 3], stride=1, depth=16),
    DepthSepConv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128)
]

def mobilenet_block(inputs, telescope_index, trig_values):
    # Set all telescopes after the first to share weights
    if telescope_index == 0:
        reuse = None
    else:
        reuse = True

    end_points = {}
    with tf.variable_scope("MobilenetV1", [inputs, trig_values], reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                padding='SAME'):
            net = inputs
            for i, conv_def in enumerate(CONV_DEFS):
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

                    # By passing filters=None separable_conv2d produces only
                    # a depthwise convolution layer
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
                    raise ValueError('Unknown convolution type %s for layer %d'
                            % (conv_def.ltype, i))
            end_point = "Trigger_multiplier"
            
            # Drop out all outputs if the telescope was not triggered
            net = tf.multiply(flatten(net), tf.expand_dims(tf.to_float(trig_values), 1))
            end_points[end_point] = net
            
            # For compatibility with variable_input_model, do not return
            # end_points for now
            return net#, end_points

#for use with train_datasets
def alexnet_block(input_features, number, trig_values):

    #shared weights
    if number == 0:
        reuse = None
    else:
        reuse = True

    with tf.variable_scope("Conv_block"):

        input_tensor = tf.reshape(input_features,[-1,IMAGE_WIDTH,IMAGE_LENGTH,IMAGE_DEPTH],name="input")
       
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

        #flatten output of pool5 layer to get feature vector of shape (num_batch,1024)
        output = tf.multiply(flatten(pool5), tf.expand_dims(trig_values, 1))
    
    return output

#for use with train_datasets
def variable_input_model(tel_data, labels, trig_list, tel_pos_tensor, num_tel,
        image_width, image_length, image_depth, training):
  
    #from batch,tel,width,length,channels to tel,batch,width,length,channels
    tel_data = tf.reshape(tel_data, [-1, num_tel, image_width, image_length, 
        image_depth])
    tel_data_transposed = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])
    trig_list_transposed = tf.transpose(trig_list, perm=[1,0])

    trig_list = tf.reshape(trig_list, [-1, num_tel])

    feature_vectors = []

    cnn_block = mobilenet_block
    for i in range(num_tel):
        telescope_features = cnn_block(tf.gather(tel_data_transposed, i,name="input_"+str(i)), i,
                tf.gather(trig_list, i, axis=1,name="trig_list_"+str(i)))
        ## Flatten output features to get feature vector
        feature_vectors.append(flatten(telescope_features))

    with tf.variable_scope("Classifier"):

        #combine the feature vectors + trigger info + tel_x + tel_y into a tensor of shape [batch size,num_tels,num_features+3]
        combined_feature_tensor = tf.stack(feature_vectors, axis=1)
        
        batch_size = tf.shape(combined_feature_tensor)[0]
        #tel_pos_tensor is initially shape = (num_tel,2)
        #convert to shape = (1,num_tel)
        tel_pos_tensor_batch = tf.expand_dims(tel_pos_tensor,0)
        #convert to shape = (batch_size,num_tel,2) by tiling along 1st dimension
        tel_pos_tensor_batch =  tf.tile(tel_pos_tensor_batch, tf.stack([batch_size,1,1])) 
        #trig_list initially shape (batch_size,num_tel)
        trig_list.set_shape([None,num_tel])
        #convert to shape = (batch_size,num_tel,1)
        #concatenate all along dimension 2 (1024+2+1)
        combined_feature_tensor = tf.concat([combined_feature_tensor,tf.to_float(tf.expand_dims(trig_list,-1)),tel_pos_tensor_batch],2)
       
        #fc6
        fc6 = tf.layers.dense(inputs=combined_feature_tensor, units=4096, 
                activation=tf.nn.relu, name="fc6") 
        dropout6 = tf.layers.dropout(inputs=fc6, rate=0.5, training=training)

        #fc7
        fc7 = tf.layers.dense(inputs=dropout6, units=4096, activation=tf.nn.relu,name="fc7")        
        dropout7 = tf.layers.dropout(inputs=fc7, rate=0.5, training=training)        

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
