import importlib
import sys

import tensorflow as tf

# Drop out all outputs if the telescope was not triggered
def apply_trigger_dropout(inputs, triggers):
    # Reshape triggers from [BATCH_SIZE] to [BATCH_SIZE, WIDTH, HEIGHT,
    # NUM_CHANNELS]
    triggers = tf.reshape(triggers, [-1, 1, 1, 1])
    triggers = tf.tile(triggers, tf.concat([[1], tf.shape(inputs)[1:]], 0))

    return tf.multiply(inputs, triggers)


# Given a list of telescope output features and tensors storing the telescope
# auxiliary parameters (e.g. positions) and trigger list, return a tensor of
# array features of the form [NUM_BATCHES, NUM_TEL, NUM_ARRAY_FEATURES]
def combine_telescopes_as_vectors(
    telescope_outputs, telescope_aux_inputs, telescope_triggers, is_training
):
    array_inputs = []
    combined_telescope_features = []
    combined_telescope_aux_inputs = []
    combined_telescope_triggers = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Flatten output features to get feature vectors
        combined_telescope_features.append(tf.layers.flatten(telescope_features))
        combined_telescope_aux_inputs.append(telescope_aux_inputs[:, i, :])
        combined_telescope_triggers.append(tf.expand_dims(telescope_triggers[:, i], 1))

    # combine telescope features
    combined_telescope_features = tf.stack(
        combined_telescope_features, axis=1, name="combined_telescope_features"
    )

    # aux inputs and telescope triggers are already normalized when loaded
    combined_telescope_aux_inputs = tf.stack(
        combined_telescope_aux_inputs, axis=1, name="combined_telescope_aux_inputs"
    )
    combined_telescope_triggers = tf.stack(
        combined_telescope_triggers, axis=1, name="combined_telescope_triggers"
    )

    # Insert auxiliary input into each feature vector
    array_features = tf.concat(
        [
            combined_telescope_features,
            combined_telescope_aux_inputs,
            combined_telescope_triggers,
        ],
        axis=2,
        name="combined_array_features",
    )

    return array_features


# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form
# [NUM_BATCHES, TEL_OUTPUT_WIDTH, TEL_OUTPUT_HEIGHT, (TEL_OUTPUT_CHANNELS +
#       NUM_AUXILIARY_INPUTS_PER_TELESCOPE) * NUM_TELESCOPES]
def combine_telescopes_as_feature_maps(
    telescope_outputs, telescope_aux_inputs, telescope_triggers, is_training
):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Get the telescope auxiliary parameters (e.g. position)
        # [NUM_BATCH, NUM_AUX_PARAMS]
        telescope_aux_input = telescope_aux_inputs[:, i, :]
        # Get whether the telescope triggered [NUM_BATCH]
        telescope_trigger = telescope_triggers[:, i]
        # Tile the aux params along the width and height dimensions
        telescope_aux_input = tf.expand_dims(telescope_aux_input, 1)
        telescope_aux_input = tf.expand_dims(telescope_aux_input, 1)
        telescope_aux_input = tf.tile(
            telescope_aux_input,
            tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0),
        )
        # Tile the trigger along the width, height, and channel dimensions
        telescope_trigger = tf.reshape(telescope_trigger, [-1, 1, 1, 1])
        telescope_trigger = tf.tile(
            telescope_trigger,
            tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0),
        )
        # Insert auxiliary input as additional channels in feature maps
        telescope_features = tf.concat(
            [telescope_features, telescope_aux_input, telescope_trigger], 3
        )
        array_inputs.append(telescope_features)
    array_features = tf.concat(array_inputs, axis=3)

    return array_features


def variable_input_model(features, model_params, example_description, training):

    # Reshape inputs into proper dimensions
    telescope_aux_inputs = []
    for (name, f), d in zip(features.items(), example_description):
        if name == "image":
            telescope_data = tf.reshape(f, [-1, *d["shape"]])
            num_telescopes = d["shape"][0]
        if name == "trigger":
            telescope_triggers = tf.cast(f, tf.float32)
        if name in ["x", "y", "z"]:
            telescope_aux_inputs.append(f)
    telescope_aux_inputs = tf.reshape(
        telescope_aux_inputs,
        [-1, num_telescopes, len(telescope_aux_inputs)],
        name="telescope_aux_inputs",
    )

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

    # Load CNN block and network head models
    sys.path.append(model_params["model_directory"])
    cnn_block_module = importlib.import_module(
        model_params["variable_input_model"]["cnn_block"]["module"]
    )
    cnn_block = getattr(
        cnn_block_module, model_params["variable_input_model"]["cnn_block"]["function"]
    )
    network_head_module = importlib.import_module(
        model_params["variable_input_model"]["network_head"]["module"]
    )
    network_head = getattr(
        network_head_module,
        model_params["variable_input_model"]["network_head"]["function"],
    )
    if model_params["variable_input_model"]["telescope_combination"] == "vector":
        combine_telescopes = combine_telescopes_as_vectors
    elif (
        model_params["variable_input_model"]["telescope_combination"] == "feature_maps"
    ):
        combine_telescopes = combine_telescopes_as_feature_maps
    else:
        raise ValueError(
            "Invalid telescope combination: {}.".format(
                model_params["telescope_combination"]
            )
        )

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
                params=model_params,
                training=training,
                reuse=reuse,
            )

        if model_params["variable_input_model"]["pretrained_weights"]:
            tf.contrib.framework.init_from_checkpoint(
                model_params["variable_input_model"]["pretrained_weights"],
                {"CNN_block/": "CNN_block/"},
            )

        telescope_features = apply_trigger_dropout(
            telescope_features, tf.gather(telescope_triggers, telescope_index, axis=1)
        )
        telescope_outputs.append(telescope_features)

    # Process the single telescope data into array-level input
    array_features = combine_telescopes(
        telescope_outputs, telescope_aux_inputs, telescope_triggers, training
    )

    with tf.variable_scope("NetworkHead"):
        # Process the combined array features
        output = network_head(array_features, params=model_params, training=training)

    return output
