import tensorflow as tf
import numpy as np


def single_tel_model(features, labels, params, is_training):
    
    # Reshape and cast inputs into proper dimensions and types
    image_width, image_length, image_depth = params['image_shapes']['MSTS']
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, image_width, image_length, image_depth])
    telescope_data = tf.cast(telescope_data, tf.float32)

    # Reshape labels to vector as expected by tf.one_hot
    gamma_hadron_labels = labels['gamma_hadron_labels']
    gamma_hadron_labels = tf.reshape(gamma_hadron_labels, [-1])
    gamma_hadron_labels = tf.cast(gamma_hadron_labels, tf.int32)

    # Choose the CNN base
    if params['cnn_block'] == 'alexnet':
        from ctalearn.models.alexnet import alexnet_block as cnn_block
    elif params['cnn_block'] == 'mobilenet':
        from ctalearn.models.mobilenet import mobilenet_block as cnn_block
    elif params['cnn_block'] == 'resnet':
        from ctalearn.models.resnet import resnet_block as cnn_block
    elif params['cnn_block'] == 'densenet':
        from ctalearn.models.densenet import densenet_block as cnn_block
    else:
        sys.exit("Error: No valid CNN block specified.")

    output = cnn_block(telescope_data,triggers=None,
                params=params,
                is_training=is_training)

    output_flattened = tf.layers.flatten(output)

    logits = tf.layers.dense(output_flattened,units=num_gamma_hadron_classes)

    onehot_labels = tf.one_hot(
            indices=gamma_hadron_labels,
            depth=num_gamma_hadron_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits)

    return loss, logits
