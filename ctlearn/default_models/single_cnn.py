import importlib
import sys

import tensorflow as tf


def single_cnn_model(data, model_params):

    # Load neural network model
    network_input_img = tf.keras.Input(shape=data.img_shape, name=f"images")
    flag_prm = 0
    if data.prm_shape != None:
        flag_prm = 1
        network_input_param = tf.keras.Input(shape=data.prm_shape, name=f"parameters")
    backbone_name = model_params.get("name", "CNN") + "_block"
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
        engine_cnn_module = importlib.import_module(model_params["engine_cnn"]["module"])
        engine_cnn = getattr(engine_cnn_module, model_params["engine_cnn"]["function"])
        if flag_prm == 1:
            engine_mlp_module = importlib.import_module(model_params["engine_mlp"]["module"])
            engine_mlp = getattr(engine_mlp_module, model_params["engine_mlp"]["function"])
            engine_output_mlp = engine_mlp(network_input_param, params=model_params, name=backbone_name)
            output_mlp = tf.keras.layers.Flatten()(engine_output_mlp)
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

        engine_output_cnn = engine_cnn(network_input_img, params=model_params, name=backbone_name)
        output_cnn = tf.keras.layers.GlobalAveragePooling2D(
            name=backbone_name + "_global_avgpool"
        )(engine_output_cnn)
        if flag_prm == 1:
            concat = tf.keras.layers.Concatenate()([output_cnn, output_mlp])
            singlecnn_model = tf.keras.Model(inputs=[network_input_img, network_input_param], outputs = [concat], name=backbone_name)
            return singlecnn_model, [network_input_img, network_input_param]
        else:
            singlecnn_model = tf.keras.Model(network_input_img, output_cnn, name=backbone_name)
            return singlecnn_model, [network_input_img]
