import importlib
import sys

import tensorflow as tf

LSTM_SIZE = 2048

def cnn_rnn_model(data, model_params):

    # Get hyperparameters
    dropout_rate = model_params.get('dropout_rate', 0.5)

    # Load neural network model
    network_name = model_params.get('name', 'CNNRNN')
    trainable_backbone = model_params.get('trainable_backbone', True)
    pretrained_weights = model_params.get('pretrained_weights', None)
    if pretrained_weights:
        loaded_model = tf.keras.models.load_model(pretrained_weights)
        network = loaded_model.get_layer('ThinResNet')
        network.trainable = trainable_backbone
    else:
        sys.path.append(model_params['model_directory'])
        network_module = importlib.import_module(model_params['network']['module'])
        network = getattr(network_module, model_params['network']['function'])

    telescope_triggers = tf.keras.Input(shape=data.trg_shape, name='triggers')
    model_input = [telescope_triggers]
    num_telescopes = data.num_tels

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

    #calculate number of valid images per event
    num_tels_triggered = tf.cast(tf.reduce_sum(telescope_triggers,1), tf.int32)

    model_input = [telescope_triggers]
    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        lala_data = tf.keras.Input(shape=data.img_shape, name=f'images_tel{telescope_index}')
        model_input.append(lala_data)

        if pretrained_weights:
            output = network(lala_data)
        else:
            output = network(lala_data, params=model_params, name=f'{network_name}_tel{telescope_index}')

        #flatten output of embedding CNN to (batch_size, _)
        image_embedding = tf.keras.layers.Flatten(name=f'image_embedding_tel{telescope_index}')(output)
        image_embedding_dropout = tf.keras.layers.Dropout(rate=0.2)(image_embedding)
        telescope_outputs.append(image_embedding_dropout)

    #combine image embeddings (batch_size, num_tel, num_units_embedding)
    embeddings = tf.stack(telescope_outputs,axis=1)

    #implement attention mechanism with range num_tel (covering all timesteps)
    #define LSTM cell size
    outputs = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(LSTM_SIZE))(embeddings)

    # (batch_size, max_num_tel * LSTM_SIZE)
    outputs = tf.keras.layers.Flatten()(outputs)
    output_dropout = tf.keras.layers.Dropout(rate=dropout_rate, name="rnn_output_dropout")(outputs)

    fc1 = tf.keras.layers.Dense(units=1024, kernel_regularizer=tf.keras.regularizers.L2(l2=0.004), name="fc1")(output_dropout)
    dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)(fc1)

    fc2 = tf.keras.layers.Dense(units=512, kernel_regularizer=tf.keras.regularizers.L2(l2=0.004), name="fc2")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)(fc2)

    model = tf.keras.Model(model_input, dropout_2, name=network_name)

    return model_input, model

