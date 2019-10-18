import importlib
import sys

import tensorflow as tf
from single_tel import single_tel_model

def gammaPhysNet2_model(features, labels, mode, params):
    
    training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    training_params = params['training']
    learning_tasks = params['model']['learning_tasks']

    output = single_tel_model(features, params['model'],
                   params['example_description'], training)
    output_gobalpooled = tf.reduce_mean(output, axis=[1,2])
    output_flattened = tf.layers.flatten(output)

    labels_dict = {}
    logits_dict = {}

    # Particle type classififcation
    num_classes = len(params['model']['label_names']['class_label'])
    particletype_logit_units = 1 if num_classes == 2 else num_classes
    prediction_gammahadron_classification = tf.layers.dense(output_flattened, units=particletype_logit_units)
        
    logits_dict.update({'particle_type': prediction_gammahadron_classification})
    labels_dict.update({'particle_type': labels['class_label']})
        
    if num_classes == 2:
        gammahadron_classification_head = tf.contrib.estimator.binary_classification_head(name='particle_type')
    else:
        gammahadron_classification_head = tf.contrib.estimator.multi_class_head(name='particle_type', n_classes=num_classes)
         
    # Arrival direction, impact parameter and x max estimation
    direction_impact_xmax_logit_units = 60
    direction_impact_xmax_output = tf.layers.dense(output_flattened, units=direction_impact_xmax_logit_units, activation=tf.nn.relu)

    prediction_direction_regression = tf.layers.dense(direction_impact_xmax_output, units=2)
    prediction_impact_regression = tf.layers.dense(direction_impact_xmax_output, units=2)
    prediction_xmax_regression = tf.layers.dense(direction_impact_xmax_output, units=1)

    logits_dict.update({'direction': prediction_direction_regression})
    logits_dict.update({'impact': prediction_impact_regression})
    logits_dict.update({'x_max': prediction_xmax_regression})

    labels_dict.update({'direction': tf.reshape([labels['alt'],labels['az']],[-1,2])})
    labels_dict.update({'impact': tf.reshape([labels['core_x'],labels['core_y']],[-1,2])})
    labels_dict.update({'x_max': labels['x_max']})

    direction_regression_head = tf.contrib.estimator.regression_head(name='direction',label_dimension=2)
    impact_regression_head = tf.contrib.estimator.regression_head(name='impact',label_dimension=2)
    xmax_regression_head = tf.contrib.estimator.regression_head(name='x_max',label_dimension=1)

    # Energy estimation
    energy_logit_units = 60
    energy_output = tf.layers.dense(output_gobalpooled, units=energy_logit_units, activation=tf.nn.relu)
    energy_regression = tf.layers.dense(energy_output, units=1)
    
    concat_tensor = tf.concat([energy_regression, prediction_impact_regression, prediction_direction_regression, prediction_xmax_regression, prediction_gammahadron_classification], 1)
    
    prediction_energy_regression = tf.layers.dense(concat_tensor, units=1)
        
    logits_dict.update({'energy': prediction_energy_regression})
    labels_dict.update({'energy': labels['mc_energy']})
        
    energy_regression_head = tf.contrib.estimator.regression_head(name='energy',label_dimension=1)
        
    # Scale the learning rate so batches with fewer triggered
    # telescopes don't have smaller gradients
    # Only apply learning rate scaling for array-level models
    if (training_params['scale_learning_rate'] and model_params['model']['function'] in ['cnn_rnn_model', 'variable_input_model']):
        trigger_rate = tf.reduce_mean(tf.cast(
            features['telescope_triggers'], tf.float32),
                                      name="trigger_rate")
        trigger_rate = tf.maximum(trigger_rate, 0.1) # Avoid division by 0
        scaling_factor = tf.reciprocal(trigger_rate, name="scaling_factor")
        learning_rate = tf.multiply(scaling_factor,
                                    training_params['base_learning_rate'],
                                    name="learning_rate")
    else:
        learning_rate = training_params['base_learning_rate']
        
    # Select optimizer with appropriate arguments
    
    # Dict of optimizer_name: (optimizer_fn, optimizer_args)
    optimizers = {
        'Adadelta': (tf.train.AdadeltaOptimizer,
                     dict(learning_rate=learning_rate)),
        'Adam': (tf.train.AdamOptimizer,
                 dict(learning_rate=learning_rate,
                      epsilon=training_params['adam_epsilon'])),
        'RMSProp': (tf.train.RMSPropOptimizer,
                    dict(learning_rate=learning_rate)),
        'SGD': (tf.train.GradientDescentOptimizer,
                dict(learning_rate=learning_rate))
        }

    optimizer_fn, optimizer_args = optimizers[training_params['optimizer']]
    optimizer = optimizer_fn(**optimizer_args)

    # Combine the several heads in the multi_head class
    multi_head = tf.contrib.estimator.multi_head([gammahadron_classification_head, direction_regression_head, impact_regression_head, xmax_regression_head, energy_regression_head])

    return multi_head.create_estimator_spec(features, mode, logits_dict, labels_dict, optimizer)
