import importlib
import sys

import tensorflow as tf

def single_cnn_model(data, model_params):

    # Load neural network model
    sys.path.append(model_params['model_directory'])
    network_module = importlib.import_module(model_params['single_cnn']['network']['module'])
    network = getattr(network_module,
                      model_params['single_cnn']['network']['function'])

    x = tf.keras.Input(shape=data.img_shape, name='images')

    output = network(x, params=model_params)
    output = tf.keras.layers.GlobalAveragePooling2D(name='global_avgpool')(output)

    return x, tf.keras.Model(x, output, name='SingleCNN')
