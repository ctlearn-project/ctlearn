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
    
    if 'MeanSquaredError' == loss_type or 'mse' == loss_type:
        loss = tf.math.square(loss)
    
    # TODO: multiclass handling! Set all non gammas to 0 and gammas to 1 in class_labels
    num_gammas_in_batch = tf.cast(params['batch_size'], tf.float32)
    if class_labels is not None:
        num_gammas_in_batch = tf.cast(tf.math.count_nonzero(class_labels), tf.float32)
        mask = tf.expand_dims(tf.cast(class_labels, tf.float32),1)
        loss = tf.math.multiply(loss, mask)

    loss = tf.math.reduce_sum(loss)
    if 'MeanAbsoluteError' == loss_type or 'mae' == loss_type:
        loss = tf.math.divide(loss, num_gammas_in_batch)

    # add regularization loss
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([loss] + regularization_losses)
   
    return loss
