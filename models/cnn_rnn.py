import importlib
import sys

import tensorflow as tf

LSTM_SIZE = 2048

def cnn_rnn_model(features, params, training):

    # Get hyperparameters
    dropout_rate = float(params.get('dropoutrate', 0.5))

    # Reshape inputs into proper dimensions
    num_telescope_types = len(params['processed_telescope_types']) 
    if not num_telescope_types == 1:
        raise ValueError('Must use a single telescope type for CNN-RNN. Number used: {}'.format(num_telescope_types))
    telescope_type = params['processed_telescope_types'][0]
    image_width, image_length, image_depth = params['processed_image_shapes'][telescope_type]
    num_telescopes = params['processed_num_telescopes'][telescope_type]
    num_aux_inputs = sum(params['processed_aux_input_nums'].values())
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, num_telescopes, 
        image_width, image_length, image_depth])

    telescope_triggers = features['telescope_triggers']
    telescope_triggers = tf.reshape(telescope_triggers, [-1, num_telescopes])
    telescope_triggers = tf.cast(telescope_triggers, tf.float32)

    telescope_aux_inputs = features['telescope_aux_inputs']
    telescope_aux_inputs = tf.reshape(telescope_aux_inputs, [-1, num_telescopes,
        num_aux_inputs])

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

    # Load CNN block model
    sys.path.append(params['modeldirectory'])
    cnn_block_module = importlib.import_module(params['cnnblockmodule'])
    cnn_block = getattr(cnn_block_module, params['cnnblockfunction'])

    #calculate number of valid images per event
    num_tels_triggered = tf.to_int32(tf.reduce_sum(telescope_triggers,1))

    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        reuse = None if telescope_index == 0 else True
              
        with tf.variable_scope("CNN_block"):
            output = cnn_block(tf.gather(telescope_data, telescope_index),
                params=params, reuse=reuse, training=training)

        if params['pretrainedweights']:
            tf.contrib.framework.init_from_checkpoint(params['pretrainedweights'],{'CNN_block/':'CNN_block/'})

        #flatten output of embedding CNN to (batch_size, _)
        image_embedding = tf.layers.flatten(output, name='image_embedding')
        image_embedding_dropout = tf.layers.dropout(image_embedding, training=training)
        telescope_outputs.append(image_embedding_dropout)

    with tf.variable_scope("NetworkHead"):

        #combine image embeddings (batch_size, num_tel, num_units_embedding)
        embeddings = tf.stack(telescope_outputs,axis=1)

        #add telescope position auxiliary input to each embedding (batch_size, num_tel, num_units_embedding+3)
        #embeddings = tf.concat([embeddings,telescope_aux_inputs],axis=2)

        #implement attention mechanism with range num_tel (covering all timesteps)
        #define LSTM cell size
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE) 
        #attention_cell = tf.contrib.rnn.AttentionCellWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(LSTM_SIZE),num_telescopes)

        # outputs = shape(batch_size, num_tel, output_size)
        outputs, _  = tf.nn.dynamic_rnn(
                            rnn_cell,
                            embeddings,
                            dtype=tf.float32,
                            swap_memory=True,
                            sequence_length=num_tels_triggered)

        # (batch_size, max_num_tel * LSTM_SIZE)
        outputs = tf.layers.flatten(outputs)
        #last_output = tf.gather(outputs, num_telescopes-1, axis=1, name="rnn_output")
        output_dropout = tf.layers.dropout(outputs, rate=dropout_rate,
                training=training, name="rnn_output_dropout")
        
        """
        #indices (0 except at every n+(num_tel-1) where n in range(batch_size))
        indices = tf.range(0, tf.shape(outputs)[0]) * outputs.get_shape()[1] + (outputs.get_shape()[1] - 1)
        #partition outputs to select only the last LSTM output for each example in the batch
        partitions = tf.reduce_sum(tf.one_hot(indices, tf.shape(outputs_reshaped)[0],dtype='int32'),0)
        partitioned_output = tf.dynamic_partition(outputs_reshaped, partitions, 2)    
        #shape (batch_size, output_size)
        last_output = partitioned_output[1]
        """

        fc1 = tf.layers.dense(inputs=output_dropout, units=1024, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc1")
        dropout_1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate,
                training=training)
        
        fc2 = tf.layers.dense(inputs=dropout_1, units=512, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc2")
        dropout_2 = tf.layers.dropout(inputs=fc2, rate=dropout_rate,
                training=training)

        logits = tf.layers.dense(inputs=dropout_2, units=num_gamma_hadron_classes, name="logits")

    return logits
