import importlib
import sys

import tensorflow as tf

def single_tel_model(features, params, is_training):
    
    # Reshape inputs into proper dimensions
    num_telescope_types = len(params['processed_telescope_types']) 
    if num_telescope_types != 1:
        raise ValueError('Must use a single telescope type for single telescope model. Number used: {}'.format(num_telescope_types))
    telescope_type = params['processed_telescope_types'][0]
    image_width, image_length, image_depth = params['processed_image_shapes'][telescope_type]
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data,[-1,image_width,image_length,image_depth], name="telescope_images")

    # Load neural network model
    sys.path.append(params['modeldirectory'])
    network_module = importlib.import_module(params['networkmodule'])
    network = getattr(network_module, params['networkfunction'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=params, is_training=is_training)

    if params['pretrainedweights']:
        tf.contrib.framework.init_from_checkpoint(params['pretrainedweights'],{'Network/':'Network/'})

    output_flattened = tf.layers.flatten(output)

    logits = tf.layers.dense(output_flattened,units=num_gamma_hadron_classes)

    return logits
