import tensorflow as tf

from ctalearn.models.alexnet import alexnet_block
from ctalearn.models.mobilenet import mobilenet_block
from ctalearn.models.resnet import resnet_block
from ctalearn.models.densenet import densenet_block

LSTM_SIZE = 2048

def cnn_rnn_model(features, labels, params, is_training):
    
    # Reshape and cast inputs into proper dimensions and types
    image_width, image_length, image_depth = params['image_shape']
    if params['segment_images']:
        image_width = image_length = params['bounding_box_size']
    num_telescopes = params['num_telescopes']['MSTS']
    num_position_coordinates = params['num_position_coordinates']
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, num_telescopes, 
        image_width, image_length, image_depth])

    telescope_triggers = features['telescope_triggers']
    telescope_triggers = tf.reshape(telescope_triggers, [-1, num_telescopes])

    telescope_positions = features['telescope_positions']
    telescope_positions = tf.reshape(telescope_positions, [-1, num_telescopes,
        num_position_coordinates])
 
    # Reshape labels to vector as expected by tf.one_hot
    gamma_hadron_labels = labels['gamma_hadron_label']
    gamma_hadron_labels = tf.reshape(gamma_hadron_labels, [-1])

    # Transpose telescope_data from [batch_size,num_tel,length,width,channels]
    # to [num_tel,batch_size,length,width,channels].
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

    #calculate number of valid images per event
    num_tels_triggered = tf.to_int32(tf.reduce_sum(telescope_triggers,1))

    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        reuse = None if telescope_index == 0 else True
       
       
        with tf.variable_scope("CNN_block"):
            output = cnn_block(tf.gather(telescope_data, telescope_index),
                params=params, reuse=reuse, is_training=is_training)

        if params['pretrained_weights']:
            tf.contrib.framework.init_from_checkpoint(params['pretrained_weights'],{'CNN_block/':'CNN_block/'})

        output = cnn_block(tf.gather(telescope_data, telescope_index),
                params=params, reuse=reuse)

        #flatten output of embedding CNN to (batch_size, _)
        output_flattened = tf.layers.flatten(output)

        #compute image embedding for each telescope (batch_size,1024)
        image_embedding = tf.layers.dense(inputs=output_flattened, units=1024, activation=tf.nn.relu,reuse=reuse)
        telescope_outputs.append(image_embedding)

    #combine image embeddings (batch_size,num_tel,num_units_embedding)
    embeddings = tf.stack(telescope_outputs,axis=1)
    #add telescope position auxiliary input to each embedding (batch_size, num_tel, num_units_embedding+3)
    embeddings = tf.concat([embeddings,telescope_positions],axis=2)

    #implement attention mechanism with range num_tel (covering all timesteps)
    #define LSTM cell size
    attention_cell = tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(LSTM_SIZE),num_telescopes)

    # outputs = shape(batch_size, num_tel, output_size)
    outputs, _  = tf.nn.dynamic_rnn(
                        attention_cell,
                        embeddings,
                        dtype=tf.float32,
                        sequence_length=num_tels_triggered)

    # (batch_size*num_tel,output_size)
    outputs_reshaped = tf.reshape(outputs, [-1, LSTM_SIZE])
    #indices (0 except at every n+(num_tel-1) where n in range(batch_size))
    indices = tf.range(0, tf.shape(outputs)[0]) * outputs.get_shape()[1] + (outputs.get_shape()[1] - 1)
    #partition outputs to select only the last LSTM output for each example in the batch
    partitions = tf.reduce_sum(tf.one_hot(indices, tf.shape(outputs_reshaped)[0],dtype='int32'),0)
    partitioned_output = tf.dynamic_partition(outputs_reshaped, partitions, 2)    
    #shape (batch_size, output_size)
    last_output = partitioned_output[1]

    logits = tf.layers.dense(inputs=last_output,units=num_gamma_hadron_classes)

    onehot_labels = tf.one_hot(
            indices=gamma_hadron_labels,
            depth=num_gamma_hadron_classes)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, 
            logits=logits)

    return loss, logits
