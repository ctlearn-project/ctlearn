import importlib
import sys

import tensorflow as tf
from single_tel import single_tel_model

def vanilla_model(features, labels, mode, params):
    
    training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    evaluation = True if mode == tf.estimator.ModeKeys.EVAL else False

    training_params = params['training']
    learning_tasks = params['model']['learning_tasks']

    output = single_tel_model(features, params['model'],
                   params['example_description'], training)
    output_flattened = tf.layers.flatten(output)

    labels_dict = {}
    logits_dict = {}
    multihead_array = []
    if 'gammahadron_classification' in learning_tasks:
        num_classes = len(params['model']['label_names']['class_label'])
           
        logit_units = 1 if num_classes == 2 else num_classes
        prediction_gammahadron_classification = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'particle_type': prediction_gammahadron_classification})
        if training or evaluation:
            labels_dict.update({'particle_type': labels['class_label']})
        
        if num_classes == 2:
            gammahadron_classification_head = tf.contrib.estimator.binary_classification_head(name='particle_type')
        else:
            gammahadron_classification_head = tf.contrib.estimator.multi_class_head(name='particle_type', n_classes=num_classes)
            
        multihead_array.append(gammahadron_classification_head)
        
    if 'energy_regression' in learning_tasks:
        logit_units = 1
        prediction_energy_regression = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'energy': prediction_energy_regression})
        if training or evaluation:
            labels_dict.update({'energy': tf.math.log(labels['mc_energy'])})
        
        energy_regression_head = tf.contrib.estimator.regression_head(name='energy',label_dimension=logit_units)
        
        multihead_array.append(energy_regression_head)
        
    if 'direction_regression' in learning_tasks:
        logit_units = 2
        prediction_direction_regression = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'direction': prediction_direction_regression})
        if training or evaluation:
            labels_dict.update({'direction': tf.reshape([labels['alt'],labels['az']],[-1,2])})
        
        direction_regression_head = tf.contrib.estimator.regression_head(name='direction',label_dimension=logit_units)
        
        multihead_array.append(direction_regression_head)
    
    if 'impact_regression' in learning_tasks:
        logit_units = 2
        prediction_impact_regression = tf.layers.dense(output_flattened, units=logit_units)
        
        logits_dict.update({'impact': prediction_impact_regression})
        if training or evaluation:
            labels_dict.update({'impact': tf.reshape([labels['core_x']*0.001,labels['core_y']*0.001],[-1,2])})
        
        impact_regression_head = tf.contrib.estimator.regression_head(name='impact',label_dimension=logit_units)
        
        multihead_array.append(impact_regression_head)
        
    if 'showermaximum_regression' in learning_tasks:
        logit_units = 1
        prediction_xmax_regression = tf.layers.dense(output_flattened, units=logit_units)
    
        logits_dict.update({'x_max': prediction_xmax_regression})
        if training or evaluation:
            labels_dict.update({'x_max': labels['x_max']*0.001})
    
        xmax_regression_head = tf.contrib.estimator.regression_head(name='x_max',label_dimension=logit_units)
        
        multihead_array.append(xmax_regression_head)
    
    # Combine the several heads in the multi_head class
    multi_head = tf.contrib.estimator.multi_head(multihead_array)
    
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
        
    return multi_head.create_estimator_spec(features=features, mode=mode, logits=logits_dict, labels=labels_dict, optimizer=optimizer)
