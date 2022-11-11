import importlib
import sys

import tensorflow as tf

LSTM_SIZE = 2048


def cnn_rnn_model(data, model_params):

    # Get hyperparameters
    dropout_rate = model_params.get("dropout_rate", 0.5)

    # Define the network being used. Each CNN block analyzes a single
    # telescope. The outputs for non-triggering telescopes are zeroed out
    # (effectively, those channels are dropped out).
    # Unlike standard dropout, this zeroing-out procedure is performed both at
    # training and test time since it encodes meaningful aspects of the data.
    # The telescope outputs are then stacked into input for the array-level
    # network, either into 1D feature vectors or into 3D convolutional
    # feature maps, depending on the requirements of the network head.
    # The array-level processing is then performed by the network head. The
    # logits are returned and fed into a classifier/regressor.
    backbone_name = model_params.get("name", "CNNRNN")
    trainable_backbone = model_params.get("trainable_backbone", True)
    pretrained_weights = model_params.get("pretrained_weights", None)
    if pretrained_weights:
        loaded_model = tf.keras.models.load_model(pretrained_weights)
        for layer in loaded_model.layers:
            if layer.name.endswith("_block"):
                model = loaded_model.get_layer(layer.name)
                model.trainable = trainable_backbone
    else:
        sys.path.append(model_params["model_directory"])
        engine_module = importlib.import_module(model_params["engine"]["module"])
        engine = getattr(engine_module, model_params["engine"]["function"])
        network_input = tf.keras.Input(shape=data.singleimg_shape, name=f"images")
        engine_output = engine(
            network_input, params=model_params, name=model_params["engine"]["function"]
        )
        output = tf.keras.layers.GlobalAveragePooling2D(
            name=backbone_name + "_global_avgpool"
        )(engine_output)
        model = tf.keras.Model(
            network_input, output, name=model_params["engine"]["function"]
        )

    telescope_data = tf.keras.Input(shape=data.img_shape, name=f"images")
    telescope_triggers = tf.keras.Input(shape=(*data.trg_shape, 1), name=f"triggers")

    output = tf.keras.layers.TimeDistributed(model)(telescope_data)
    dropout = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(rate=0.2))(output)
    mask_output = tf.keras.layers.Multiply()([dropout, telescope_triggers])
    mask_layer = tf.keras.layers.Masking(mask_value=0)(mask_output)

    outputs = tf.keras.layers.LSTM(LSTM_SIZE, name="LSTM")(mask_layer)

    output_dropout = tf.keras.layers.Dropout(
        rate=dropout_rate, name="rnn_output_dropout"
    )(outputs)

    fc1 = tf.keras.layers.Dense(
        units=1024, kernel_regularizer=tf.keras.regularizers.L2(l2=0.004), name="fc1"
    )(output_dropout)
    dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)(fc1)

    fc2 = tf.keras.layers.Dense(
        units=512, kernel_regularizer=tf.keras.regularizers.L2(l2=0.004), name="fc2"
    )(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)(fc2)

    cnnrnn_model = tf.keras.Model(
        [telescope_data, telescope_triggers], dropout_2, name=backbone_name
    )

    return cnnrnn_model, [telescope_data, telescope_triggers]
