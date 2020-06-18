import tensorflow as tf
from tensorflow.python.ops import math_ops

import numpy as np

def classification_loss(labels, logits, params):

    # Compute class-weighted softmax-cross-entropy
    true_classes = tf.cast(labels, tf.int32,
                           name="true_classes")

    # Get class weights
    if params['apply_class_weights']:
        class_weights = tf.constant(params['class_weights'],
                                    dtype=tf.float32, name="class_weights")
        weights = tf.gather(class_weights, true_classes, name="weights")
    else:
        weights = 1.0
    
    num_classes = logits.get_shape().as_list()[1]

    onehot_labels = tf.one_hot(indices=true_classes, depth=num_classes)

    # compute cross-entropy loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits, weights=weights)
                                           
    # add regularization loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss] + regularization_losses)
        
    return loss
    
def regression_loss(loss_type, labels, logits, params, class_labels=None):
    
    logits = tf.convert_to_tensor(logits)
    labels = tf.cast(labels, tf.float32)
    loss = tf.math.abs(logits - labels)

    if class_labels is not None:
        mask = tf.equal(class_labels, tf.constant(1, dtype=tf.int8))
        loss = tf.boolean_mask(loss, mask)

    loss = tf.math.reduce_mean(loss)

    # add regularization loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss] + regularization_losses)
   
    return loss
