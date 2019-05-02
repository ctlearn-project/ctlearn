import importlib
import sys

import tensorflow as tf

def single_tel_model(features, model_params, example_description, training):
    
    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])
    num_classes = len(model_params['classification']['classes'])

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_tel']['network']['module'])
    network = getattr(network_module,
                      model_params['single_tel']['network']['function'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=model_params, training=training)

    if model_params['single_tel']['pretrained_weights']:
        tf.contrib.framework.init_from_checkpoint(model_params['single_tel']['pretrained_weights'],{'Network/':'Network/'})

    output_flattened = tf.layers.flatten(output)

    logits = tf.layers.dense(output_flattened, units=num_classes)

    return logits
