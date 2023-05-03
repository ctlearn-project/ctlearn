import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow_addons.layers import SpectralNormalization
import numpy as np


class TransposeConvBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        use_batchnorm=True,
        use_bias=False,
        use_dropout=False,
        drop_value=0.3,
        kernel_initializer='orthogonal',
        spectral_norm=True
    ):
        super(TransposeConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.conv_transpose = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer
        )

        if spectral_norm:
            self.conv_transpose = SpectralNormalization(self.conv_transpose)

        self.batch_normalization = layers.BatchNormalization()
        self.activation = activation
        self.dropout = layers.Dropout(drop_value)
        

    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class UpsampleBlock(layers.Layer):
    def __init__(
        self,
        filters,
        activation=layers.LeakyReLU(0.2),
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_batchnorm=True,
        use_bias=False,
        use_dropout=False,
        drop_value=0.3,
        kernel_initializer='orthogonal',
        spectral_norm=True
    ):
        super(UpsampleBlock, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.upsampling = layers.UpSampling2D(up_size)
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
        x = self.upsampling(inputs)
        x = self.conv(x)
        if self.use_batchnorm:
            x = self.batch_normalization(inputs=x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout(inputs=x, training=training)
        
        return x


class Generator(keras.Model):
    def __init__(self, g_config):
        super(Generator, self).__init__(name='generator')
        spectral_norm = g_config['spectral_normalization']
        self.latent_dim = g_config['latent_dim']
        self.dense = layers.Dense(**g_config['layers']['dense'])
        if spectral_norm:
            self.dense = SpectralNormalization(self.dense)

        self.batch_normalization = layers.BatchNormalization()
        self.activation = layers.LeakyReLU(0.2)
        self.reshape = layers.Reshape(**g_config['layers']['reshape'])
        self.upsample_blocks = []
        for block_config in g_config['layers']['upsample_blocks']:
            if g_config['upsampling']:
                block = UpsampleBlock(**block_config, spectral_norm=spectral_norm)
            else:
                block = TransposeConvBlock(**block_config, spectral_norm=spectral_norm)

            self.upsample_blocks.append(block)

        #layers.Activation("tanh"))
        self.cropping = layers.Cropping2D(**g_config['layers']['cropping'])


    def _get_generator_inputs(self, labels):
        # Sample random points in the latent space and concatenate the labels.
        batch_size = tf.shape(list(labels.values())[0])[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        g_inputs = random_latent_vectors
        for task in labels.values():
            g_inputs = tf.concat([g_inputs, task], axis=1) # TODO: axis=-1?

        return g_inputs
    

    def call(self, labels, training=False):
        g_inputs = self._get_generator_inputs(labels)
        x = self.dense(g_inputs)
        x = self.batch_normalization(inputs=x, training=training)
        x = self.activation(x)
        x = self.reshape(x)
        for block in self.upsample_blocks:
            x = block(inputs=x, training=training)
        
        x = self.cropping(x)

        return x


def get_generator_loss(name='bce', weights=np.array([1, 1, 1, 1]), label_smoothing=0.1):
    # Define the Wasserstein loss function
    def wasserstein_loss(labels, d_outputs):
        alphas = -(labels/(1-label_smoothing)*2-1) # map labels to fake=1 and real=-1
        loss = tf.reduce_mean(alphas*d_outputs) # fake_loss - real_loss
        
        return loss

    # Define the main loss functions (the contribution that asses the degree of realism)
    main_loss_functions = {
        # from_logits=True to avoid using sigmoid activation when defining the discriminator
        'bce': losses.BinaryCrossentropy(from_logits=True),
        'least_squares': losses.MeanSquaredError(),
        'wasserstein': wasserstein_loss
    }

    main_loss = main_loss_functions[name] # main loss function (for d_labels)
    c_loss = losses.CategoricalCrossentropy() # classification loss function
    r_loss = losses.MeanSquaredError() # regression loss function

    # Define the generator loss function
    def generator_loss(d_outputs, p_outputs, d_labels, p_labels):
        # Apply label smoothing
        smoothed_d_labels = d_labels*(1-label_smoothing)
        # Compute main loss
        loss = weights[0]*main_loss(smoothed_d_labels, d_outputs)
        # Add the contributions from the predictor depending on which labels are used for conditioning
        if 'particletype' in p_labels.keys():
            loss += weights[1]*c_loss(p_labels['particletype'], p_outputs['particletype'])

        if 'energy' in p_labels.keys():
            loss += weights[2]*r_loss(p_labels['energy'], p_outputs['energy'])

        if 'direction' in p_labels.keys():
            loss += weights[3]*r_loss(p_labels['direction'], p_outputs['direction'])
                    
        return loss

    return generator_loss


def get_generator_optimizer(parameters, name='adam'):
    if name == 'adam':
        optimizer = optimizers.Adam(**parameters)
    elif name == 'rms':
        optimizer = optimizers.RMSprop(**parameters)
    
    return optimizer
