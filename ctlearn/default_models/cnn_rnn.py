import importlib
import sys

import tensorflow as tf

LSTM_SIZE = 2048

def cnn_rnn_model(features, model_params, example_description, training):

    # Get hyperparameters
    dropout_rate = model_params['cnn_rnn'].get('dropout_rate', 0.5)

    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name.endswith('images'):
            telescope_data = tf.reshape(f, [-1, *d['shape']])
            num_telescopes = d['shape'][0]
        if name.endswith('triggers'):
            telescope_triggers = tf.cast(f, tf.float32)

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
    sys.path.append(model_params['model_directory'])
    cnn_block_module = importlib.import_module(model_params['cnn_rnn']['cnn_block']['module'])
    cnn_block = getattr(cnn_block_module, model_params['cnn_rnn']['cnn_block']['function'])

    #calculate number of valid images per event
    num_tels_triggered = tf.to_int32(tf.reduce_sum(telescope_triggers,1))

    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        reuse = None if telescope_index == 0 else True

        with tf.variable_scope("CNN_block"):
            output = cnn_block(tf.gather(telescope_data, telescope_index),
                params=model_params, reuse=reuse, training=training)

        if model_params['cnn_rnn']['pretrained_weights']:
            tf.contrib.framework.init_from_checkpoint(model_params['cnn_rnn']['pretrained_weights'],{'CNN_block/':'CNN_block/'})

        #flatten output of embedding CNN to (batch_size, _)
        image_embedding = tf.layers.flatten(output, name='image_embedding')
        image_embedding_dropout = tf.layers.dropout(image_embedding, training=training)
        telescope_outputs.append(image_embedding_dropout)

    with tf.variable_scope("NetworkHead"):

        #combine image embeddings (batch_size, num_tel, num_units_embedding)
        embeddings = tf.stack(telescope_outputs,axis=1)

        #implement attention mechanism with range num_tel (covering all timesteps)
        #define LSTM cell size
        rnn_cell = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE)
        outputs, _  = tf.nn.dynamic_rnn(
                            rnn_cell,
                            embeddings,
                            dtype=tf.float32,
                            swap_memory=True,
                            sequence_length=num_tels_triggered)

        # (batch_size, max_num_tel * LSTM_SIZE)
        outputs = tf.layers.flatten(outputs)
        output_dropout = tf.layers.dropout(outputs, rate=dropout_rate,
                training=training, name="rnn_output_dropout")

        fc1 = tf.layers.dense(inputs=output_dropout, units=1024, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc1")
        dropout_1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate,
                training=training)

        fc2 = tf.layers.dense(inputs=dropout_1, units=512, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc2")
        dropout_2 = tf.layers.dropout(inputs=fc2, rate=dropout_rate,
                training=training)

    return dropout_2
