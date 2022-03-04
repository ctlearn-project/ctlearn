import importlib
import sys

import tensorflow as tf

def single_cnn_model(data, model_params):

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['network']['module'])
    network = getattr(network_module,
                      model_params['network']['function'])
    network_name = model_params.get('name', 'SingleCNN')

    x = tf.keras.Input(shape=data.img_shape, name='images')

    # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
    #x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
    init_layer = model_params.get('init_layer', False)
    if init_layer:
        x = tf.keras.layers.Conv2D(filters=init_layer['filters'], kernel_size=init_layer['kernel_size'],
                strides=init_layer['strides'], name=network_name+'_conv1_conv')(x)
    #x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
    init_max_pool = model_params.get('init_max_pool', False)
    if init_max_pool:
        x = tf.keras.layers.MaxPool2D(pool_size=init_max_pool['size'],
                strides=init_max_pool['strides'], name=network_name+'_pool1_pool')(x)

    output = network(x, params=model_params, name=network_name)
    output = tf.keras.layers.GlobalAveragePooling2D(name=network_name+'_global_avgpool')(output)

    model = tf.keras.Model(x, output, name=network_name)

    model.trainable = model_params.get('trainable_backbone', True)

    return x, model

