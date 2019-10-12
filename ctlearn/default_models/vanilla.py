import importlib
import sys

import tensorflow as tf
from single_tel import single_tel_model

def vanilla_model(features, labels, mode, params):
    
    training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    training_params = params['training']
    learning_tasks = params['model']['learning_tasks']

    output_flattened = single_tel_model(features, params['model'],
                   params['example_description'], training)

    labels_dict = {}
    logits_dict = {}
    multihead_array = []
    if 'gammahadron_classification' in learning_tasks:
        num_classes = len(params['model']['label_names']['class_label'])
           
        logit_units = 1 if num_classes == 2 else num_classes
        prediction_gammahadron_classification = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'class_label': prediction_gammahadron_classification})
        # Compute class-weighted softmax-cross-entropy
        true_classes = tf.cast(labels['class_label'], tf.int32,
                                  name="true_classes")
           
        labels_dict.update({'class_label': tf.equal(true_classes,1)})
        
        if num_classes == 2:
            gammahadron_classification_head = tf.contrib.estimator.binary_classification_head(name='class_label')
        else:
            gammahadron_classification_head = tf.contrib.estimator.multi_class_head(name='class_label', n_classes=num_classes)
            
        multihead_array.append(gammahadron_classification_head)
        
    if 'energy_regression' in learning_tasks:
        logit_units = 1
        prediction_energy_regression = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'mc_energy': prediction_energy_regression})
        labels_dict.update({'mc_energy': labels['mc_energy']})
        
        energy_regression_head = tf.contrib.estimator.regression_head(name='mc_energy',label_dimension=logit_units)
        
        multihead_array.append(energy_regression_head)
        
    if 'direction_regression' in learning_tasks:
        logit_units = 2
        prediction_direction_regression = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'arrival_direction': prediction_direction_regression})
        labels_dict.update({'arrival_direction': [labels['alt'],labels['az']]})
        
        direction_regression_head = tf.contrib.estimator.regression_head(name='arrival_direction',label_dimension=logit_units)
        
        multihead_array.append(direction_regression_head)
    
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
    multi_head = tf.contrib.estimator.multi_head(multihead_array)
        
    return multi_head.create_estimator_spec(features, mode, logits_dict, labels_dict, optimizer)
