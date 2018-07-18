import importlib
import sys

import tensorflow as tf

def single_tel_model(features, params, training):
    
    # Reshape inputs into proper dimensions
    num_telescope_types = len(params['selected_telescope_types']) 
    if num_telescope_types != 1:
        raise ValueError('Must use a single telescope type for single telescope model. Number used: {}'.format(num_telescope_types))
    telescope_type = params['selected_telescope_types'][0]
    image_width, image_length, image_depth = params['image_shapes'][telescope_type]
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data,[-1,image_width,image_length,image_depth], name="telescope_images")

    # Load neural network model
    sys.path.append(params['model_directory'])
    network_module = importlib.import_module(params['single_tel']['network']['module'])
    network = getattr(network_module, params['single_tel']['network']['function'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=params, training=training)

    if params['single_tel']['pretrained_weights']:
        tf.contrib.framework.init_from_checkpoint(params['single_tel']['pretrained_weights'],{'Network/':'Network/'})

    output_flattened = tf.layers.flatten(output)

    logits = tf.layers.dense(output_flattened,units=num_gamma_hadron_classes)

    return logits
