import importlib
import sys
import tensorflow as tf

def param_model(data, model_params):
    # Load neural network model
    network_name = "Param_block"
    network_input = tf.keras.Input(shape=data.prm_shape, name=f"parameters")
    network_module = importlib.import_module(model_params["network"]["module"])
    network = getattr(network_module, model_params["network"]["function"])
    network_output = network(network_input, params = model_params, name  = network_name)
    model = tf.keras.Model(network_input, network_output, name  = network_name)
    dot_img_file = '/home/sahil/deeplearning/model_1.png'
    tf.keras.utils.plot_model(model, to_file = dot_img_file, show_shapes=True)
    return model, [network_input]
