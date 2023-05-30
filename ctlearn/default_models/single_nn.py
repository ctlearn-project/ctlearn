import importlib
import sys

import tensorflow as tf


def single_nn_model(data, model_params):

    # Load neural network model
    network_input, network_output = [], []
    if data.img_shape != None:
        network_input_img = tf.keras.Input(shape=data.img_shape, name=f"images")
        network_input.append(network_input_img)

    if data.prm_shape != None and data.mode == "train":
        network_input_prm = tf.keras.Input(shape=data.prm_shape, name=f"parameters")
        network_input.append(network_input_prm)

    backbone_name = model_params.get("name", "SingleNN") + "_backbone"
    trainable_backbone = model_params.get("trainable_backbone", True)
    pretrained_weights = model_params.get("pretrained_weights", None)
    if pretrained_weights:
        loaded_model = tf.keras.models.load_model(pretrained_weights)
        for layer in loaded_model.layers:
            if "_backbone" in layer.name:
                singlenn_model = loaded_model.get_layer(layer.name)
                singlenn_model.trainable = trainable_backbone
    else:
        sys.path.append(model_params["model_directory"])
        if data.img_shape != None:
            engine_cnn_module = importlib.import_module(
                model_params["engine_cnn"]["module"]
            )
            engine_cnn = getattr(
                engine_cnn_module, model_params["engine_cnn"]["function"]
            )
            # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
            # x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
            init_layer = model_params.get("init_layer", False)
            if init_layer:
                network_input_img = tf.keras.layers.Conv2D(
                    filters=init_layer["filters"],
                    kernel_size=init_layer["kernel_size"],
                    strides=init_layer["strides"],
                    name=backbone_name + "_conv1_conv",
                )(network_input_img)
            # x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
            init_max_pool = model_params.get("init_max_pool", False)
            if init_max_pool:
                network_input_img = tf.keras.layers.MaxPool2D(
                    pool_size=init_max_pool["size"],
                    strides=init_max_pool["strides"],
                    name=backbone_name + "_pool1_pool",
                )(network_input_img)
            engine_output_cnn = engine_cnn(
                network_input_img, params=model_params, name=backbone_name
            )
            network_output = output_cnn = tf.keras.layers.GlobalAveragePooling2D(
                name=backbone_name + "_global_avgpool"
            )(engine_output_cnn)

        if data.prm_shape != None and data.mode == "train":
            engine_mlp_module = importlib.import_module(
                model_params["engine_mlp"]["module"]
            )
            engine_mlp = getattr(
                engine_mlp_module, model_params["engine_mlp"]["function"]
            )
            engine_output_mlp = engine_mlp(
                network_input_prm, params=model_params, name=backbone_name
            )
            network_output = output_mlp = tf.keras.layers.Flatten()(engine_output_mlp)
            if data.img_shape != None:
                network_output = tf.keras.layers.Concatenate()([output_cnn, output_mlp])

        singlenn_model = tf.keras.Model(
            network_input, network_output, name=backbone_name
        )
    return singlenn_model, network_input
