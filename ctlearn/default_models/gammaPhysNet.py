import importlib
import sys

import tensorflow as tf
from ctlearn.ct_heads import *

def gammaPhysNet_model(features, model_params, example_description, training):
    
    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])
    num_classes = len(model_params['label_names']['particletype'])
    
    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['gammaPhysNet']['network']['module'])
    network = getattr(network_module,
                      model_params['gammaPhysNet']['network']['function'])

    with tf.variable_scope("Network"):
        output = network(telescope_data, params=model_params, training=training)

    if model_params['gammaPhysNet']['pretrained_weights']:    tf.contrib.framework.init_from_checkpoint(model_params['gammaPhysNet']['pretrained_weights'],{'Network/':'Network/'})
            
    output_gobalpooled = tf.reduce_mean(output, axis=[1,2])
    output_flattened = tf.layers.flatten(output)
    
    logits = {}
    multihead_array = []
    
    # Particle type classififcation
    multihead_array.append(particletype_head(output_flattened, logits, num_classes))
    
    # Arrival direction and impact parameter estimation
    logit_units = 256
    direction_impact_output = tf.layers.dense(output_flattened, units=logit_units, activation=tf.nn.relu)
    multihead_array.append(direction_head(direction_impact_output, logits))
    multihead_array.append(impact_head(direction_impact_output, logits))

    # Energy estimation
    energy_output = tf.layers.dense(output_gobalpooled, units=logit_units, activation=tf.nn.relu)
    multihead_array.append(energy_head(energy_output, logits))

    return multihead_array, logits
