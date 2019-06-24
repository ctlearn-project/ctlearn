import importlib
import sys

import tensorflow as tf

LSTM_SIZE = 2048

def attention_model(features, model_params, example_description, training):

    # Get hyperparameters
    dropout_rate = model_params['attention'].get('dropout_rate', 0.5)

    # Reshape inputs into proper dimensions
    for (name, f), d in zip(features.items(), example_description):
        if name == 'image':
            telescope_data = tf.reshape(f, [-1, *d['shape']])
            num_telescopes = d['shape'][0]
        if name == 'trigger':
            telescope_triggers = tf.cast(f, tf.float32)

    num_classes = len(model_params['label_names']['class_label'])
    
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
    cnn_block_module = importlib.import_module(model_params['attention']['cnn_block']['module'])
    cnn_block = getattr(cnn_block_module, model_params['attention']['cnn_block']['function'])

    #calculate number of valid images per event
    num_tels_triggered = tf.to_int32(tf.reduce_sum(telescope_triggers,1))

    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        reuse = None if telescope_index == 0 else True
              
        with tf.variable_scope("CNN_block"):
            output = cnn_block(tf.gather(telescope_data, telescope_index),
                params=model_params, reuse=reuse, training=training)

        if model_params['attention']['pretrained_weights']:
            tf.contrib.framework.init_from_checkpoint(model_params['attention']['pretrained_weights'],{'CNN_block/':'CNN_block/'})

        #flatten output of embedding CNN to (batch_size, _)
        image_embedding = tf.layers.flatten(output, name='image_embedding')
        image_embedding_dropout = tf.layers.dropout(image_embedding, training=training)
        telescope_outputs.append(image_embedding_dropout)

    with tf.variable_scope("NetworkHead"):
        print(f"single_embedding.shape = {telescope_outputs[0].shape}")
        # assign the attention scores
        attention_scores_list = [tf.layers.dense(inputs=embedding, units=1) for embedding in telescope_outputs]
        attention_scores_unsqueezed = tf.stack(attention_scores_list, axis=1, name="attention_scores_unsqueezed")
        print(f"attention_scores_unsqueezed.shape = {attention_scores_unsqueezed.shape}")
        attention_scores = tf.squeeze(attention_scores_unsqueezed, axis=2, name="attention_scores")
        print(f"attention_scores.shape = {attention_scores.shape}")

        # normalize the attention scores
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        print(f"attention_weights.shape = {attention_weights.shape}")

        #combine image embeddings (batch_size, num_tel, num_units_embedding)
        embeddings = tf.stack(telescope_outputs,axis=1)
        print(f"embeddings.shape = {embeddings.shape}")

        #take attention-weighted mean
        attention_output = tf.einsum('ij,ijk->ik', attention_weights, embeddings, name="attention_output")
        print(f"attention_output.shape = {attention_output.shape}")

        attention_output_dropout = tf.layers.dropout(attention_output, rate=dropout_rate,
                training=training, name="attention_output_dropout")
        
        fc1 = tf.layers.dense(inputs=attention_output_dropout, units=1024, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc1")
        dropout_1 = tf.layers.dropout(inputs=fc1, rate=dropout_rate,
                training=training)
        
        fc2 = tf.layers.dense(inputs=dropout_1, units=512, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.004), name="fc2")
        dropout_2 = tf.layers.dropout(inputs=fc2, rate=dropout_rate,
                training=training)

        logits = tf.layers.dense(inputs=dropout_2, units=num_classes,
                                 name="logits")

    return logits
