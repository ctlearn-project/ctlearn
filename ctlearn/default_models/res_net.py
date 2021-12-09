import importlib
import sys

import tensorflow as tf

def res_net_model(features, model_params, example_description, training):

    # Reshape inputs into proper dimensions
    merge_telescopes = model_params['res_net'].get('merge_telescopes', False)
    if merge_telescopes:
        # Reshape inputs into proper dimensions
        for (name, f), d in zip(features.items(), example_description):
            if name.endswith('images'):
                telescope_data = tf.reshape(f, [-1, d['shape'][1], d['shape'][2], d['shape'][0]*d['shape'][3]])
    else:
        for (name, f), d in zip(features.items(), example_description):
            if name == 'image':
                telescope_data = tf.reshape(f, [-1, *d['shape']])

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['res_net']['network']['module'])
    network = getattr(network_module,
                      model_params['res_net']['network']['function'])

    with tf.variable_scope("Network"):

       x = telescope_data
       # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
       #x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
       init_layer = model_params['res_net'].get('init_layer', False)
       if init_layer:
           x = tf.layers.conv2d(x, filters=init_layer['filters'], kernel_size=init_layer['kernel_size'],
                    strides=init_layer['strides'], name='conv1_conv')
       #x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
       init_max_pool = model_params['res_net'].get('init_max_pool', False)
       if init_max_pool:
           x = tf.layers.max_pooling2d(x, init_max_pool['size'], strides=init_max_pool['strides'], name='pool1_pool')

       output = network(x, params=model_params)
       output = tf.reduce_mean(output, axis=[1,2], name='global_avgpool')

    if model_params['res_net']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['res_net']['pretrained_weights'],{'Network/':'Network/'})

    return output
