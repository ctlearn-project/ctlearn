import importlib
import sys

import tensorflow as tf

def res_net_model(features, model_params, example_description, training):

    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['res_net']['network']['module'])
    network = getattr(network_module,
                      model_params['res_net']['network']['function'])

    with tf.variable_scope("Network"):

       # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
       #x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
       x = tf.layers.conv2d(telescope_data, filters=64, kernel_size=7,
                    strides=1, name='conv1_conv')
       #x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
       x = tf.layers.max_pooling2d(x, 3, strides=2, name='pool1_pool')

       output = network(telescope_data, params=model_params)
       output = tf.reduce_mean(output, axis=[1,2], name='global_avgpool')

    if model_params['res_net']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['res_net']['pretrained_weights'],{'Network/':'Network/'})

    return output
