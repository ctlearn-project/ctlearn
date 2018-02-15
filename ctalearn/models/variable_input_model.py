import tensorflow as tf

from ctalearn.models.alexnet import (alexnet_block,
        alexnet_head_feature_vector, alexnet_head_feature_map)
from ctalearn.models.mobilenet import mobilenet_block, mobilenet_head
from ctalearn.models.resnet import (resnet_block, resnet_head)
from ctalearn.models.densenet import densenet_block

# Drop out all outputs if the telescope was not triggered
def apply_trigger_dropout(inputs,triggers):
    # Reshape triggers from [BATCH_SIZE] to [BATCH_SIZE, WIDTH, HEIGHT, 
    # NUM_CHANNELS]
    triggers = tf.reshape(triggers, [-1, 1, 1, 1])
    triggers = tf.tile(triggers, tf.concat([[1], tf.shape(inputs)[1:]], 0))
    
    return tf.multiply(inputs, triggers)

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form 
# [NUM_BATCHES, NUM_ARRAY_FEATURES]
def combine_telescopes_as_vectors(telescope_outputs, telescope_positions, 
        telescope_triggers):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Flatten output features to get feature vectors
        telescope_features = tf.layers.flatten(telescope_features)
        telescope_position = telescope_positions[:, i, :]
        telescope_trigger = tf.expand_dims(telescope_triggers[:, i], 1)
        # Insert auxiliary input into each feature vector
        telescope_features = tf.concat([telescope_features, 
            telescope_position, telescope_trigger], 1)
        array_inputs.append(telescope_features)
    array_features = tf.concat(array_inputs, axis=1)
    return array_features

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form
# [NUM_BATCHES, TEL_OUTPUT_WIDTH, TEL_OUTPUT_HEIGHT, (TEL_OUTPUT_CHANNELS + 
#       NUM_AUXILIARY_INPUTS_PER_TELESCOPE) * NUM_TELESCOPES]
def combine_telescopes_as_feature_maps(telescope_outputs, telescope_positions, 
        telescope_triggers):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Get the telescope position and if it triggered
        # [NUM_BATCH, NUM_AUX_INFO]
        telescope_position = telescope_positions[:, i, :]
        telescope_trigger = telescope_triggers[:, i] # [NUM_BATCH]
        # Tile the position along the width and height dimensions
        telescope_position = tf.expand_dims(telescope_position, 1)
        telescope_position = tf.expand_dims(telescope_position, 1)
        telescope_position = tf.tile(telescope_position,
                tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0))
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

def variable_input_model(features, labels, params, is_training):
   
    # Reshape and cast inputs into proper dimensions and types
    num_telescope_types = len(params['processed_telescope_types']) 
    if not num_telescope_types == 1:
        raise ValueError('Must use a single telescope type for Variable Input Model. Number used: {}'.format(num_telescope_types))
    telescope_type = params['processed_telescope_types'][0]
    image_width, image_length, image_depth = params['processed_image_shapes'][telescope_type]
    num_telescopes = params['processed_num_telescopes'][telescope_type]
    num_position_coordinates = params['num_position_coordinates']
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, num_telescopes, 
        image_width, image_length, image_depth])
    telescope_data = tf.cast(telescope_data, tf.float32)

    telescope_triggers = features['telescope_triggers']
    telescope_triggers = tf.reshape(telescope_triggers, [-1, num_telescopes])
    telescope_triggers = tf.cast(telescope_triggers, tf.float32)

    telescope_positions = features['telescope_positions']
    telescope_positions = tf.reshape(telescope_positions, [-1, num_telescopes,
        num_position_coordinates])
    telescope_positions = tf.cast(telescope_positions, tf.float32)
    
    # Reshape labels to vector as expected by tf.one_hot
    gamma_hadron_labels = labels['gamma_hadron_label']
    gamma_hadron_labels = tf.reshape(gamma_hadron_labels, [-1])
    gamma_hadron_labels = tf.cast(gamma_hadron_labels, tf.int32)

    # Split data by telescope by switching the batch and telescope dimensions
    # leaving width, length, and channel depth unchanged
    telescope_data = tf.transpose(telescope_data, perm=[1, 0, 2, 3, 4])

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

    # Choose the CNN block
    if params['cnn_block'] == 'alexnet':
        cnn_block = alexnet_block
    elif params['cnn_block'] == 'mobilenet':
        cnn_block = mobilenet_block
    elif params['cnn_block'] == 'resnet':
        cnn_block = resnet_block
    elif params['cnn_block'] == 'densenet':
        cnn_block = densenet_block
    else:
        raise ValueError("Invalid CNN block specified: {}.".format(params['cnn_block']))

    # Choose the network head and telescope combination method
    if params['network_head'] == 'alexnet_fc':
        network_head = alexnet_head_feature_vector
        combine_telescopes = combine_telescopes_as_vectors
    elif params['network_head'] == 'alexnet_conv':
        network_head = alexnet_head_feature_map
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'mobilenet':
        network_head = mobilenet_head
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'resnet':
        network_head = resnet_head
        combine_telescopes = combine_telescopes_as_feature_maps
    else:
        raise ValueError("Invalid network head specified: {}.".format(params['network_head']))
    
    # Process the input for each telescope
    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        if telescope_index == 0:
            reuse = None
        else:
            reuse = True
        with tf.variable_scope("CNN_block", reuse=reuse):
            telescope_features = cnn_block(
                tf.gather(telescope_data, telescope_index), 
                params=params,
                is_training=is_training,
                reuse=reuse)

        if params['pretrained_weights']:
            tf.contrib.framework.init_from_checkpoint(params['pretrained_weights'],{'CNN_block/':'CNN_block/'})

        telescope_features = apply_trigger_dropout(telescope_features,
                tf.gather(telescope_triggers, telescope_index, axis=1))
        telescope_outputs.append(telescope_features)

    with tf.variable_scope("NetworkHead"):
        # Process the single telescope data into array-level input
        array_features = combine_telescopes(
                telescope_outputs, 
                telescope_positions, 
                telescope_triggers)
        # Process the combined array features
        logits = network_head(array_features, params=params,
                is_training=is_training)
        
    # Calculate loss
    onehot_labels = tf.one_hot(
            indices=gamma_hadron_labels,
            depth=num_gamma_hadron_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits)

    return loss, logits
