import tensorflow as tf

#credit to https://github.com/tensorflow/models/blob/master/official/resnet for implementation

RESNET_HEAD_SIZE = 18
RESNET_BLOCK_SIZE = 18

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

DATA_FORMAT = 'channels_last'

def batch_norm_relu(inputs, is_training, data_format):
    """Performs a batch normalization followed by a ReLU."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
            scale=True, training=is_training, fused=True)
    inputs = tf.nn.relu(inputs)
    return inputs

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=data_format)


def building_block(inputs, filters, is_training, projection_shortcut, strides,
data_format):
    """Standard building block for residual networks with BN before convolutions.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
    mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
    a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
    downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)

    return inputs + shortcut


def bottleneck_block(inputs, filters, is_training, projection_shortcut,
strides, data_format):
    """Bottleneck block variant for residual networks with BN before convolutions.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
    third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
    mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
    a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
    downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)
    inputs = conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)

    return inputs + shortcut


def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name,
data_format):
    """Creates one layer of blocks for the ResNet model.

    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
    [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
    `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
    greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
    model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

    Returns:
    The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
                inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
            data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)


def resnet_base(inputs,size,scope_name,reuse,is_training):

    #valid resnet configurations
    model_params = {
    18: {'block': building_block, 'layers': [2, 2, 2, 2]},
    34: {'block': building_block, 'layers': [3, 4, 6, 3]},
    50: {'block': bottleneck_block, 'layers': [3, 4, 6, 3]},
    101: {'block': bottleneck_block, 'layers': [3, 4, 23, 3]},
    152: {'block': bottleneck_block, 'layers': [3, 8, 36, 3]},
    200: {'block': bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if size not in model_params:
        raise ValueError('Not a valid resnet_size:', resnet_size)
    
    params = model_params[size]
    block_fn = params['block']
    layers = params['layers']

    with tf.variable_scope(scope_name, inputs, reuse=reuse):
        inputs = block_layer(inputs=inputs, filters=64, block_fn=block_fn, blocks=layers[0],
                strides=1, is_training=is_training, name='block_layer1',data_format=DATA_FORMAT)
        inputs = block_layer(
                inputs=inputs, filters=128, block_fn=block_fn, blocks=layers[1],
                strides=2, is_training=is_training, name='block_layer2',
                data_format=DATA_FORMAT)
        inputs = block_layer(
                inputs=inputs, filters=256, block_fn=block_fn, blocks=layers[2],
                strides=2, is_training=is_training, name='block_layer3',
                data_format=DATA_FORMAT)
        inputs = block_layer(
                inputs=inputs, filters=512, block_fn=block_fn, blocks=layers[3],
                strides=2, is_training=is_training, name='block_layer4',
                data_format=DATA_FORMAT)

        inputs = batch_norm_relu(inputs, is_training, DATA_FORMAT)
        
    return inputs


def resnet_block(inputs, params=None, is_training=True, reuse=None):

    #preliminary convolution and pooling on raw input
    inputs = conv2d_fixed_padding(
    inputs=inputs, filters=64, kernel_size=7, strides=2,
            data_format=DATA_FORMAT)
    inputs = tf.identity(inputs, 'initial_conv')
    inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=3, strides=2, padding='SAME',
            data_format=DATA_FORMAT)
    inputs = tf.identity(inputs, 'initial_max_pool')

    #resnet block
    inputs = resnet_base(inputs, RESNET_BLOCK_SIZE, "RESNET_BLOCK", reuse,
            is_training)

    output = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=5, strides=3, padding='SAME',
            data_format=DATA_FORMAT)
 
    return output   

def resnet_head(inputs, params=None, is_training=True): 

    # Get hyperparameters
    if not params:
        params = {}
    num_classes = params.get('num_gamma_hadron_classes', 2)
    
    #conv and pool
    inputs = block_layer(
                inputs=inputs, filters=128, block_fn=building_block, blocks=4,
                strides=2, is_training=is_training, name='block_layer2',
                data_format=DATA_FORMAT)
   
    with tf.variable_scope('Logits'):
        inputs = tf.layers.average_pooling2d(
                inputs=inputs, pool_size=2, strides=1, padding='VALID',
                data_format=DATA_FORMAT)
        inputs = tf.identity(inputs, 'final_avg_pool')
        inputs = tf.contrib.layers.flatten(inputs)
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        logits = tf.identity(inputs, 'final_dense')
    
    return logits

def resnet_head_feature_vector(inputs, params=None, is_training=True): 

    #preliminary convolution and pooling on raw input
    #inputs = conv2d_fixed_padding(
    #inputs=inputs, filters=64, kernel_size=5, strides=2,
    #        data_format=DATA_FORMAT)
    #inputs = tf.identity(inputs, 'initial_conv')
    #inputs = tf.layers.max_pooling2d(
    #        inputs=inputs, pool_size=3, strides=2, padding='SAME',
    #        data_format=DATA_FORMAT)
    #inputs = tf.identity(inputs, 'initial_max_pool')
    #
    #inputs = resnet_base(inputs,RESNET_HEAD_SIZE,"RESNET_HEAD",False,is_training)
    
    # Get hyperparameters
    if not params:
        params = {}
    num_classes = params.get('num_gamma_hadron_classes', 2)
    
    with tf.variable_scope('Logits'):
        #inputs = tf.layers.average_pooling2d(
                #inputs=inputs, pool_size=2, strides=1, padding='VALID',
                #data_format=DATA_FORMAT)
        #inputs = tf.identity(inputs, 'final_avg_pool')
        #inputs = tf.contrib.layers.flatten(inputs)
        #inputs = tf.layers.dense(inputs=inputs, units=1024)
        inputs = tf.layers.dense(inputs=inputs, units=512)
        inputs = tf.layers.dense(inputs=inputs, units=512)
        inputs = tf.layers.dense(inputs=inputs, units=num_classes)
        logits = tf.identity(inputs, 'final_dense')
    
    return logits
