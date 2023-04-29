import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow_addons.layers import SpectralNormalization


class ConvBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        use_batchnorm=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
        kernel_initializer='orthogonal',
        spectral_norm=True
    ):
        super(ConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        if spectral_norm:
            self.conv = SpectralNormalization(self.conv)
      
        self.batch_normalization = layers.BatchNormalization()
        self.activation = activation
        self.dropout = layers.Dropout(drop_value)
        

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class Discriminator(keras.Model):
    def __init__(self, d_config):
        super(Discriminator, self).__init__(name='discriminator')
        spectral_norm = d_config['spectral_normalization']
        self.zeropadding = layers.ZeroPadding2D(**d_config['layers']['zeropadding'])
        self.conv_blocks = []
        for block_config in d_config['layers']['conv_blocks']:
            self.conv_blocks.append(ConvBlock(**block_config, spectral_norm=spectral_norm))
        
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.2)        
        self.dense = layers.Dense(1, kernel_initializer='orthogonal') # default activation is linear
        if spectral_norm:
            self.dense = SpectralNormalization(self.dense)
    

    def call(self, inputs, training=False):
        x = self.zeropadding(inputs)
        for block in self.conv_blocks:
            x = block(inputs=x, training=training)

        x = self.flatten(x)
        x = self.dropout(inputs=x, training=training)
        x = self.dense(x)

        return x


def get_discriminator_loss(name='bce', label_smoothing=0.1):
    def wasserstein_loss(labels, d_outputs):
        alphas = -(labels/(1-label_smoothing)*2-1) # map labels to fake=1 and real=-1
        loss = tf.reduce_mean(alphas*d_outputs) # fake_loss - real_loss
        
        return loss

    loss_functions = {
        # from_logits=True to avoid using sigmoid activation when defining the discriminator
        'bce': losses.BinaryCrossentropy(from_logits=True),
        'least_squares': losses.MeanSquaredError(),
        'wasserstein': wasserstein_loss
    }

    loss_function = loss_functions[name]
    def discriminator_loss(labels, d_outputs):
        # Apply label smoothing
        smoothed_labels = labels*(1-label_smoothing)
        loss = loss_function(smoothed_labels, d_outputs)

        return loss

    return discriminator_loss


def get_discriminator_optimizer(parameters, name='adam'):
    if name == 'adam':
        optimizer = optimizers.Adam(**parameters)
    elif name == 'rms':
        optimizer = optimizers.RMSprop(**parameters)
    
    return optimizer
