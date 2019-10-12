import importlib
import sys

import tensorflow as tf
from single_tel import single_tel_model

def vanilla_classification_model(features, labels, mode, params):
    
    training = True if mode == tf.estimator.ModeKeys.TRAIN else False
    training_params = params['training']

    output_flattened = single_tel_model(features, params['model'],
                   params['example_description'], training)

    num_classes = len(params['model']['label_names']['class_label'])
    
    logit_units = 1 if num_classes == 2 else num_classes
    logits = tf.layers.dense(output_flattened, units=logit_units)
    
    # Compute class-weighted softmax-cross-entropy
    true_classes = tf.cast(labels['class_label'], tf.int32,
                           name="true_classes")
    
    labels_dict = {'class_label': tf.equal(true_classes,1)}
    
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

    if num_classes == 2:
        classification_head = tf.contrib.estimator.binary_classification_head(name='class_label')
    else:
        classification_head = tf.contrib.estimator.multi_class_head(name='class_label', n_classes=num_classes)
        
    return classification_head.create_estimator_spec(features, mode, logits, labels_dict['class_label'], optimizer)
