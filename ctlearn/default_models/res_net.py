import importlib
import sys

import tensorflow as tf

def res_net_model(data, model_params):

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['res_net']['network']['module'])
    network = getattr(network_module,
                      model_params['res_net']['network']['function'])
    trainable = model_params['res_net'].get('trainable_backbone', True)
    resnet_name = model_params['resnet_engine'].get('name', 'ResNet')

    x = tf.keras.Input(shape=data.img_shape, name='images')
    
    # The original ResNet implementation use this padding, but we pad the images in the ImageMapper.
    #x = tf.pad(telescope_data, tf.constant([[3, 3], [3, 3]]), name='conv1_pad')
    init_layer = model_params['res_net'].get('init_layer', False)
    if init_layer:
        x = tf.keras.layers.Conv2D(filters=init_layer['filters'], kernel_size=init_layer['kernel_size'],
                strides=init_layer['strides'], name=resnet_name+'_conv1_conv')(x)
    #x = tf.pad(x, tf.constant([[1, 1], [1, 1]]), name='pool1_pad')
    init_max_pool = model_params['res_net'].get('init_max_pool', False)
    if init_max_pool:
        x = tf.keras.layers.MaxPool2D(pool_size=init_max_pool['size'],
                strides=init_max_pool['strides'], name=resnet_name+'_pool1_pool')(x)

    output = network(x, params=model_params, trainable=trainable, name=resnet_name)
    output = tf.keras.layers.GlobalAveragePooling2D(name=resnet_name+'_global_avgpool')(output)

    return x, tf.keras.Model(x, output, name=resnet_name)

