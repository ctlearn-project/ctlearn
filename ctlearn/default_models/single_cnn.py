import importlib
import sys

import tensorflow as tf

def single_cnn_model(features, model_params, example_description, training):

    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_cnn']['network']['module'])
    network = getattr(network_module,
                      model_params['single_cnn']['network']['function'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=model_params, training=training)
        output = tf.reduce_mean(output, axis=[1,2], name='global_avgpool')

    if model_params['single_cnn']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['single_cnn']['pretrained_weights'],{'Network/':'Network/'})

    return output
