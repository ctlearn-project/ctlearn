import importlib
import sys

import tensorflow as tf


def single_cnn_model(data, model_params):

    # Load neural network model
    network_input = tf.keras.Input(shape=data.img_shape, name=f"images")
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
        engine_module = importlib.import_module(model_params["engine"]["module"])
        engine = getattr(engine_module, model_params["engine"]["function"])

        # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
        # x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
        init_layer = model_params.get("init_layer", False)
        if init_layer:
            network_input = tf.keras.layers.Conv2D(
                filters=init_layer["filters"],
                kernel_size=init_layer["kernel_size"],
                strides=init_layer["strides"],
                name=backbone_name + "_conv1_conv",
            )(network_input)
        # x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
        init_max_pool = model_params.get("init_max_pool", False)
        if init_max_pool:
            network_input = tf.keras.layers.MaxPool2D(
                pool_size=init_max_pool["size"],
                strides=init_max_pool["strides"],
                name=backbone_name + "_pool1_pool",
            )(network_input)

        engine_output = engine(network_input, params=model_params, name=backbone_name)

        output = tf.keras.layers.GlobalAveragePooling2D(
            name=backbone_name + "_global_avgpool"
        )(engine_output)

        singlecnn_model = tf.keras.Model(network_input, output, name=backbone_name)

    return singlecnn_model, [network_input]
