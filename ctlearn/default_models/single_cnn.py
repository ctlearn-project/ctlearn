import importlib
import sys

import tensorflow as tf
import tensorflow.keras.layers as tf_layers


def single_cnn_model(data, model_params):
    # Load neural network model
    network_input, network_output = [], []
    if data.wvf_pos is not None:
        network_input_wvf = tf.keras.Input(shape=data.wvf_shape, name=f"waveforms")
        waveform3D = len(data.wvf_shape) == 4
        network_input.append(network_input_wvf)
    if data.img_pos is not None:
        network_input_img = tf.keras.Input(shape=data.img_shape, name=f"images")
        network_input.append(network_input_img)

    backbone_name = model_params.get("name", "CNN") + "_block"
    trainable_backbone = model_params.get("trainable_backbone", True)
    pretrained_weights = model_params.get("pretrained_weights", None)
    if pretrained_weights:
        loaded_model = tf.keras.models.load_model(f"{pretrained_weights}/ctlearn_model/")
        for layer in loaded_model.layers:
            if layer.name.endswith("_block"):
                model = loaded_model.get_layer(layer.name)
                model.trainable = trainable_backbone
    else:
        sys.path.append(model_params["model_directory"])

        # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
        # x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
        init_layer = model_params.get("init_layer", False)
        init_max_pool = model_params.get("init_max_pool", False)
        if data.wvf_pos is not None:
            backbone_name_wvf = backbone_name + "_wvf"
            engine_wvf_module = importlib.import_module(
                model_params["waveform_engine"]["module"]
            )
            engine_wvf = getattr(
                engine_wvf_module, model_params["waveform_engine"]["function"]
            )
            if init_layer:
                if waveform3D:
                    network_input_wvf = tf_layers.Conv3D(
                        filters=init_layer["filters"],
                        kernel_size=init_layer["kernel_size"],
                        strides=init_layer["strides"],
                        name=backbone_name_wvf + "_conv1_conv",
                    )(network_input_wvf)
                else:
                    network_input_wvf = tf_layers.Conv2D(
                        filters=init_layer["filters"],
                        kernel_size=init_layer["kernel_size"],
                        strides=init_layer["strides"],
                        name=backbone_name_wvf + "_conv1_conv",
                    )(network_input_wvf)
            # x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
            if init_max_pool:
                if waveform3D:
                    network_input_wvf = tf_layers.MaxPool3D(
                        pool_size=init_max_pool["size"],
                        strides=init_max_pool["strides"],
                        name=backbone_name_wvf + "_pool1_pool",
                    )(network_input_wvf)
                else:
                    network_input_wvf = tf_layers.MaxPool2D(
                        pool_size=init_max_pool["size"],
                        strides=init_max_pool["strides"],
                        name=backbone_name_wvf + "_pool1_pool",
                    )(network_input_wvf)

            engine_output_wvf = engine_wvf(
                network_input_wvf, params=model_params, name=backbone_name_wvf
            )

            if waveform3D:
                network_output = output_wvf = tf_layers.GlobalAveragePooling3D(
                    name=backbone_name_wvf + "_global_avgpool"
                )(engine_output_wvf)
            else:
                network_output = output_wvf = tf_layers.GlobalAveragePooling2D(
                    name=backbone_name_wvf + "_global_avgpool"
                )(engine_output_wvf)

        if data.img_pos is not None:
            backbone_name_img = backbone_name + "_img"
            engine_img_module = importlib.import_module(
                model_params["image_engine"]["module"]
            )
            engine_img = getattr(
                engine_img_module, model_params["image_engine"]["function"]
            )
            if init_layer:
                network_input_img = tf_layers.Conv2D(
                    filters=init_layer["filters"],
                    kernel_size=init_layer["kernel_size"],
                    strides=init_layer["strides"],
                    name=backbone_name_img + "_conv1_conv",
                )(network_input_img)
            # x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
            if init_max_pool:
                network_input_img = tf_layers.MaxPool2D(
                    pool_size=init_max_pool["size"],
                    strides=init_max_pool["strides"],
                    name=backbone_name_img + "_pool1_pool",
                )(network_input_img)

            engine_output_img = engine_img(
                network_input_img, params=model_params, name=backbone_name_img
            )

            network_output = output_img = tf_layers.GlobalAveragePooling2D(
                name=backbone_name_img + "_global_avgpool"
            )(engine_output_img)
            if data.wvf_pos is not None:
                network_output = tf.keras.layers.Concatenate()([output_wvf, output_img])

        singlecnn_model = tf.keras.Model(
            network_input, network_output, name=backbone_name
        )

    return singlecnn_model, network_input
