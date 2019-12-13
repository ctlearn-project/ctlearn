import importlib
import sys

import tensorflow as tf
from ctlearn.ct_heads import *

def single_tel_model(features, model_params, example_description, training):
    
    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])
    num_classes = len(model_params['label_names']['particletype'])
    
    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_tel']['network']['module'])
    network = getattr(network_module,
                      model_params['single_tel']['network']['function'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=model_params, training=training)

    if model_params['single_tel']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['single_tel']['pretrained_weights'],{'Network/':'Network/'})
        
    output_flattened = tf.layers.flatten(output)
    
    logits = {}
    multihead_array = []
    for task in model_params['label_names']:
        if num_classes != 2 and task == 'particletype':
            multihead_array.append(model_params['multitask_heads'][task](output_flattened, logits, num_classes))
        else:
            multihead_array.append(model_params['multitask_heads'][task](output_flattened, logits))

    return multihead_array, logits
