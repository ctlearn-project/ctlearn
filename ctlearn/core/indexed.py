import keras
import tensorflow as tf

class NeighborGatherLayer(tf.keras.layers.Layer):
    """
    Custom layer to gather neighbor features.
    
    Given an input tensor with shape (batch, pixels_per_patch, seq_length, channels)
    and neighbor_indices with shape (pixels_per_patch, K) where K includes the pixel itself 
    and its neighbors (invalid indices should be -1), this layer returns a tensor with shape 
    (batch, pixels_per_patch, K, seq_length, channels) where invalid neighbor entries are zeroed out.
    """
    
    def __init__(self, neighbor_indices, use_3d_conv, **kwargs):
        super().__init__(**kwargs)
        # Convert neighbor_indices into constants.
        self.indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32) # (L, K)
        # Create mask: 0 where indices = -1.
        self.mask = tf.cast(tf.not_equal(self.indices, -1), tf.float32)  # (L, K)
        self.use_3d_conv = use_3d_conv
        
    def call(self, inputs):
        """
        Parameters:
          inputs: Tensor of shape (batch, L, channels)
        
        Returns:
          Tensor of shape (batch, L, K, channels) with gathered neighbor features.
        """
        batch_size = tf.shape(inputs)[0]
        # Expand indices to have a batch dimension.
        tiled_indices = tf.expand_dims(self.indices, axis=0)  # (1, L, K)
        tiled_indices = tf.tile(tiled_indices, [batch_size, 1, 1])  # (batch, L, K)
        
        # Gather neighbor features along the pixel dimension (axis 1)
        neighbor_feats = tf.gather(inputs, tiled_indices, batch_dims=1, axis=1)
        # neighbor_feats shape: 2D: (batch, L, K, channels), 3D: (batch, L, K, T, channels)
        
        # Expand mask to match neighbor_feats shape:
        tiled_mask = tf.expand_dims(self.mask, axis=0)  # (1, L, K)
        tiled_mask = tf.tile(tiled_mask, [batch_size, 1, 1])  # (batch, L, K)
        tiled_mask = tf.expand_dims(tiled_mask, axis=-1)  # (batch, L, K, 1)
        if self.use_3d_conv:
          tiled_mask = tf.expand_dims(tiled_mask, axis=-1)  # (batch, L, K, 1, 1)
        
        # Apply the mask (casting is necessary to match neighbor_feats.dtype)
        neighbor_feats = neighbor_feats * tf.cast(tiled_mask, neighbor_feats.dtype) 
        return neighbor_feats
      
    def get_config(self):
      config = super().get_config()
      config.update({
          "use_3d_conv": self.use_3d_conv,
      })
      return config


class IndexedConvolutionLayer(tf.keras.layers.Layer):
    """
    Layer to perform the hexagonal convolutions using the output of the Gather Layers.
    """
    def __init__(self, use_3d_conv, temporal_kernel_size, filters, name, **kwargs):
      super().__init__(name=name, **kwargs)

      self.use_3d_conv = use_3d_conv
      self.temporal_kernel_size = temporal_kernel_size
      self.filters = filters

      if use_3d_conv:
          self.conv = tf.keras.layers.Conv3D(
              filters=filters,
              kernel_size=(1, 7, temporal_kernel_size),
              strides=(1, 1, 1),
              padding="valid",
              activation="relu",
              name=name
          )
      else:
          self.conv = tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=(1, 7),
              strides=(1, 1),
              padding="valid",
              activation="relu",
              name=name
          )

    def call(self, x):
      x = self.conv(x)
      return tf.squeeze(x, axis=2)

    def get_config(self):
      config = super().get_config()
      config.update({
          "use_3d_conv": self.use_3d_conv,
          "temporal_kernel_size": self.temporal_kernel_size,
          "filters": self.filters,
      })
      return config


class IndexedPoolingLayer(tf.keras.layers.Layer):
    """
    Layer to perform the hexagonal poolings using the output of the Gather Layers.
    """
    def __init__(self, use_3d_conv, pooling_type, temporal_pool_size, name, **kwargs):
      super().__init__(name=name, **kwargs)
      self.use_3d_conv = use_3d_conv
      self.pooling_type = pooling_type.lower()
      self.temporal_pool_size = temporal_pool_size

      if self.use_3d_conv:
        if self.pooling_type.lower() == "max":
            self.pool = tf.keras.layers.MaxPool3D(
                pool_size=(1, 7, self.temporal_pool_size),
                strides=(1, 1, 1),
                padding='valid',
                name=name
            )
        else:
            self.pool = tf.keras.layers.AveragePooling3D(
                pool_size=(1, 7, self.temporal_pool_size),
                strides=(1, 1, 1),
                padding='valid',
                name=name
            )
      else:
        if self.pooling_type.lower() == "max":
            self.pool = tf.keras.layers.MaxPool2D(
                pool_size=(1, 7),
                strides=(1, 1),
                padding='valid',
                name=name
            )
        else:
            self.pool = tf.keras.layers.AveragePooling2D(
                pool_size=(1, 7),
                strides=(1, 1),
                padding='valid',
                name=name
            )

    def call(self, x):
        x = self.pool(x)
        return tf.squeeze(x, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "use_3d_conv": self.use_3d_conv,
            "pooling_type": self.pooling_type,
            "temporal_pool_size": self.temporal_pool_size,
        })
        return config
