import tensorflow as tf

#from ctalearn.models.basic import basic_conv_block
#from ctalearn.models.alexnet import alexnet_block
#from ctalearn.models.mobilenet import mobilenet_block
#from ctalearn.models.resnet import resnet_block
#from ctalearn.models.densenet import densenet_block

def single_tel_model(features, labels, params, is_training):
    
    # Reshape inputs into proper dimensions
    num_telescope_types = len(params['processed_telescope_types']) 
    if num_telescope_types != 1:
        raise ValueError('Must use a single telescope type for single telescope model. Number used: {}'.format(num_telescope_types))
    telescope_type = params['processed_telescope_types'][0]
    image_width, image_length, image_depth = params['processed_image_shapes'][telescope_type]
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data,[-1,image_width,image_length,image_depth], name="telescope_images")

    sys.path.append('/home/shevek/brill/ctalearn/models/')

    model_module = importlib.import_module('basic')
    cnn_block = getattr(model_module, 'basic_conv_block')
    ## Choose the CNN block
    #if params['cnn_block'] == 'alexnet':
    #    cnn_block = alexnet_block
    #elif params['cnn_block'] == 'mobilenet':
    #    cnn_block = mobilenet_block
    #elif params['cnn_block'] == 'resnet':
    #    cnn_block = resnet_block
    #elif params['cnn_block'] == 'densenet':
    #    cnn_block = densenet_block
    #elif params['cnn_block'] == 'basic':
    #    cnn_block = basic_conv_block
    #else:
    #    raise ValueError("Invalid CNN block specified: {}.".format(params['cnn_block']))

    with tf.variable_scope("CNN_block"):
        output = cnn_block(telescope_data, params=params, is_training=is_training)

    if params['pretrained_weights']:
        tf.contrib.framework.init_from_checkpoint(params['pretrained_weights'],{'CNN_block/':'CNN_block/'})

    output_flattened = tf.layers.flatten(output)

    logits = tf.layers.dense(output_flattened,units=num_gamma_hadron_classes)

    return logits
