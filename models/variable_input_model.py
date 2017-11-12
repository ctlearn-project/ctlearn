import tensorflow as tf
import numpy as np

from models.alexnet import alexnet_block, alexnet_head
from models.mobilenet import mobilenet_block, mobilenet_head
from models.resnet import resnet_block, resnet_head

NUM_CLASSES = 2

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form 
# [NUM_BATCHES, NUM_ARRAY_FEATURES]
def combine_telescopes_as_vectors(telescope_outputs, telescope_positions, 
        telescope_triggers):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Flatten output features to get feature vectors
        telescope_features = tf.contrib.layers.flatten(telescope_features)
        # Get the telescope x and y position and if it triggered
        telescope_position = telescope_positions[i, :]
        telescope_position = tf.tile(tf.expand_dims(telescope_position, 0),
                [tf.shape(telescope_features)[0], 1])
        telescope_trigger = tf.expand_dims(telescope_triggers[:, i], 1)
        # Insert auxiliary input into each feature vector
        telescope_features = tf.concat([telescope_features, 
            telescope_position, telescope_trigger], 1)
        array_inputs.append(telescope_features)
    array_features = tf.concat(array_inputs, axis=1)
    return array_features

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form
# [NUM_BATCHES, TEL_OUTPUT_WIDTH, TEL_OUTPUT_HEIGHT, TEL_OUTPUT_CHANNELS + 
#       AUXILIARY_INPUTS_PER_TELESCOPE]
def combine_telescopes_as_feature_maps(telescope_outputs, telescope_positions, 
        telescope_triggers):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Get the telescope x and y position and if it triggered
        telescope_position = telescope_positions[i, :] # [2]
        telescope_trigger = telescope_triggers[:, i] # [NUM_BATCH]
        # Tile the position along the batch, width, and height dimensions
        telescope_position = tf.reshape(telescope_position, [1, 1, 1, -1])
        telescope_position = tf.tile(telescope_position,
                tf.concat([tf.shape(telescope_features)[:-1], [1]], 0))
        # Tile the trigger along the width, height, and channel dimensions
        telescope_trigger = tf.reshape(telescope_trigger, [-1, 1, 1, 1])
        telescope_trigger = tf.tile(telescope_trigger,
                tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0))
        # Insert auxiliary input as additional channels in feature maps
        telescope_features = tf.concat([telescope_features, 
            telescope_position, telescope_trigger], 3)
        array_inputs.append(telescope_features)
    array_features = tf.concat(array_inputs, axis=3)
    return array_features

#for use with train_datasets
def variable_input_model(tel_data, trig_list, tel_pos_tensor, labels, num_tel,
        image_shape, is_training):
 
    # Reshape and cast inputs into proper dimensions and types
    image_width, image_length, image_depth = image_shape
    tel_data = tf.reshape(tel_data, [-1, num_tel, image_width, image_length, 
        image_depth])
    # Reshape labels to vector as expected by tf.one_hot
    labels = tf.reshape(labels, [-1])
    trig_list = tf.reshape(trig_list, [-1, num_tel])
    trig_list = tf.cast(trig_list, tf.float32)
    # TODO: move number of aux inputs (2) to be defined as a constant
    tel_pos_tensor = tf.reshape(tel_pos_tensor, [num_tel, 2])
    
    # Split data by telescope by switching the batch and telescope dimensions
    # leaving width, length, and channel depth unchanged
    tel_data_by_telescope = tf.transpose(tel_data, perm=[1, 0, 2, 3, 4])

    # Define the network being used. Each CNN block analyzes a single
    # telescope. The outputs for non-triggering telescopes are zeroed out 
    # (effectively, those channels are dropped out).
    # Unlike standard dropout, this zeroing-out procedure is performed both at
    # training and test time since it encodes meaningful aspects of the data.
    # The telescope outputs are then stacked into input for the array-level
    # network, either into 1D feature vectors or into 3D convolutional 
    # feature maps, depending on the requirements of the network head.
    # The array-level processing is then performed by the network head. The
    # logits are returned and fed into a classifier.
    cnn_block = mobilenet_block
    combine_telescopes = combine_telescopes_as_feature_maps
    network_head = mobilenet_head

    # Process the input for each telescope
    telescope_outputs = []
    for i in range(num_tel):
        telescope_features = cnn_block(tf.gather(tel_data_by_telescope, i), i,
                tf.gather(trig_list, i, axis=1), is_training=is_training)
        telescope_outputs.append(telescope_features)

    with tf.variable_scope("NetworkHead"):
        # Process the single telescope data into array-level input
        array_features = combine_telescopes(telescope_outputs, tel_pos_tensor, 
                trig_list)
        # Process the combined array features
        logits = network_head(array_features, num_classes=NUM_CLASSES, 
                is_training=is_training)

    with tf.variable_scope("Outputs"):

        # Calculate loss (for both TRAIN and EVAL modes) 
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), 
                depth=NUM_CLASSES)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
                logits=logits)

        # Calculate outputs
        predictions = {
                "classes": tf.argmax(logits, axis=1),
                "probabilities": tf.nn.softmax(logits)
                }

        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.cast(predictions['classes'], tf.int8), labels), tf.float32))

    return loss, accuracy, logits, predictions
