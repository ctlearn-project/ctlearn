import tensorflow as tf
import numpy as np

def cnn_rnn_model(features, labels, params, is_training):
    
    # Reshape and cast inputs into proper dimensions and types
    image_width, image_length, image_depth = params['image_shape']
    num_telescopes = params['num_telescopes']
    num_auxiliary_inputs = params['num_auxiliary_inputs']
    num_gamma_hadron_classes = params['num_gamma_hadron_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, num_telescopes, 
        image_width, image_length, image_depth])
    telescope_data = tf.cast(telescope_data, tf.float32)

    telescope_triggers = features['telescope_triggers']
    telescope_triggers = tf.reshape(telescope_triggers, [-1, num_telescopes])
    telescope_triggers = tf.cast(telescope_triggers, tf.float32)

    telescope_positions = features['telescope_positions']
    telescope_positions = tf.reshape(telescope_positions, 
            [num_telescopes, num_auxiliary_inputs])
    telescope_positions = tf.cast(telescope_positions, tf.float32)
 
    # Reshape labels to vector as expected by tf.one_hot
    gamma_hadron_labels = labels['gamma_hadron_labels']
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
        from ctalearn.models.alexnet import alexnet_block as cnn_block
    elif params['cnn_block'] == 'mobilenet':
        from ctalearn.models.mobilenet import mobilenet_block as cnn_block
    elif params['cnn_block'] == 'resnet':
        from ctalearn.models.resnet import resnet_block as cnn_block
    else:
        sys.exit("Error: No valid CNN block specified.")
      
    used_tel_data = tf.sign(tf.reduce_max(telescope_data,[2,3,4]))
    num_tels_triggered = tf.reduce_sum(used_tel_data, 1).to_int32()
  
    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        if telescope_index == 0:
            reuse = None
        else:
            reuse = True
 
        output = cnn_block(
                    tf.gather(telescope_data, telescope_index), 
                    tf.gather(telescope_triggers, telescope_index, axis=1),
                    params=params,
                    is_training=is_training,
                    reuse=reuse)
       
        output_flattened = tf.contrib.layers.flatten(output)
        image_embedding = tf.layers.dense(inputs=output, units=1024, activation=tf.nn.relu,reuse=reuse)
        telescope_position = tf.gather(telescope_positions,telescope_index,axis=1)
        image_embedding_with_features = tf.concat([image_embeding,telescope_position], 1)
        telescope_outputs.append(image_embedding_with_features)

    embeddings = tf.stack(telescope_outputs,axis=1)
    
    attention_cell = tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.LSTMCell(15360),num_telescopes)

    output, state = tf.nn.dynamic_rnn(
                        attention_cell,
                        embeddings,
                        dtype=tf.float32,
                        sequence_length=num_tels_triggered)

    logits = tf.layers.dense(inputs=state,units=num_gamma_hadron_classes)

    onehot_labels = tf.one_hot(
            indices=gamma_hadron_labels,
            depth=num_gamma_hadron_classes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits)

    return loss, logits
