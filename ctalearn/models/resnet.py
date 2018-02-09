# Based on Deep Residual Learning for Image Recognition (He et. al.)
# and Identity Mappings in Deep Residual Networks (He et. al.)

import tensorflow as tf

# based on ResNet-50
RESNET_BLOCK_PARAMS = [
        ([[1,64],[3,64],[1,256]],3),
        ([[1,128],[3,128],[1,512]],4),
        ([[1,256],[3,256],[1,1024]],6),
        ([[1,512],[3,512],[1,2048]],3)
        ]

RESNET_HEAD_PARAMS = [
        ([[3,64],[3,64]],4)
        ]

# a composite layer applying batch normalization, followed by a relu activation, followed by a convolution
def BN_relu_conv(inputs, kernel_size, num_filters, stride=1):
    
    x = tf.layers.batch_normalization(inputs, training=training, fused=True)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, filters=num_filters, kernel_size=kernel_size, stride=stride, padding='valid')

    return x

# a residual block comprised of multiple stacked BN_relu_conv layers with a shortcut connection
# block_params is a list of form [[kernel_size_1,num_filters_1],[kernel_size_2,num_filters_2],...]
# which describes the kernel size and number of filters for each BN_relu_conv composite layer
# in a given residual block
def residual_block(inputs, block_params, downsample=True):

    x = inputs
    for i in range(len(block_params)):
        # for the first convolution of each block (except the first) downsample by
        # applying the convolution with stride 2
        stride = 2 if (i == 0 and downsample) else 1
        x = BN_relu_conv(x, block_params[i][0], block_params[i][1], stride=stride)

    # shortcut identity connection to input
    # when downsampling, do 1x1 convolution with stride 2 to convert 
    # input to same shape as the output
    if downsample:
        shortcut_connection = BN_relu_conv(inputs, 1, block_params[-1][1], stride=2)
    else:
        shortcut_connection = inputs
    x = tf.add(x,shortcut_connection)

    return x

# a residual layer comprised of multiple identical residual blocks
# the first performs downsampling, the rest are identical
def residual_layer(inputs, block_params, num_blocks, downsample=True):
    
    x = inputs
    for i in range(num_blocks):
        with tf.variable_scope("residual_block_{}".format(i+1)):
            downsample = True if (downsample and i == 0) else False
            x = residual_block(x, block_params, downsample=downsample)

    return x

# resnet cnn block (based on a resnet-type model without the final average pooling and logits
# consists of several residual layers of different types (each, besides the first, performing downsampling)
def resnet_block(inputs, params=None, is_training=True, reuse=None):

    # Get hyperparameters
    if params is None: params = {}
    dropout_keep_prob = params.get('dropout_keep_prob', 0.9)
    num_classes = params.get('num_gamma_hadron_classes', 2)

    with tf.variable_scope("ResNet_block",reuse=reuse):
        with tf.variable_scope("initial_conv"):
            x = tf.layers.conv2d(inputs, filters=64, kernel_size=7, stride=2, padding='valid')
            x = tf.layers.max_pooling2d(x, pool_size=3, strides=2, padding='same')

        for i in len(RESNET_BLOCK_PARAMS):
            with tf.variable_scope("residual_layer_{}".format(i+1)):
                downsample = False if i > 0 else True
                x = residual_layer(x, RESNET_BLOCK_PARAMS[i][0], RESNET_BLOCK_PARAMS[i][1], downsample=downsample)

    return x 

# resnet head (based on a reduced resnet-type model with added pooling and logits)
def resnet_head(inputs, params=None, is_training=True, reuse=None):

    # Get hyperparameters
    if params is None: params = {}
    dropout_keep_prob = params.get('dropout_keep_prob', 0.9)
    num_classes = params.get('num_gamma_hadron_classes', 2)

    with tf.variable_scope("ResNet_head"):
        # reduce number of filters via a bottleneck convolution
        with tf.variable_scope("bottleneck"):
            x = tf.layers.conv2d(inputs, filters=64, kernel_size=1, padding='valid')

        for i in len(RESNET_HEAD_PARAMS):
            with tf.variable_scope("residual_layer_{}".format(i+1)):
                downsample = False if i > 0 else True
                x = residual_layer(x, RESNET_HEAD_PARAMS[i][0], RESNET_HEAD_PARAMS[i][1], downsample=downsample)

        x = tf.layers.flatten(x)
        logits = tf.layers.dense(x,num_classes)

        return logits
