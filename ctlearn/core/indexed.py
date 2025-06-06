import keras
import tensorflow as tf


class NeighborGatherLayer3D(tf.keras.layers.Layer):
    """
    Custom layer to gather neighbor features.
    
    Given an input tensor with shape (batch, pixels_per_patch, seq_length, channels)
    and neighbor_indices with shape (pixels_per_patch, K) where K includes the pixel itself 
    and its neighbors (invalid indices should be -1), this layer returns a tensor with shape 
    (batch, pixels_per_patch, K, seq_length, channels) where invalid neighbor entries are zeroed out.
    """
    
    def __init__(self, neighbor_indices, **kwargs):
        super().__init__(**kwargs)
        # Convert neighbor_indices into constants.
        self.indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32) # (L, K)
        # Create mask: 0 where indices = -1.
        self.mask = tf.cast(tf.not_equal(self.indices, -1), tf.float32)  # (L, K)
        
    def call(self, inputs):
        """
        Parameters:
          inputs: Tensor of shape (batch, L, T, channels)
        
        Returns:
          Tensor of shape (batch, L, K, T, channels) with gathered neighbor features.
        """
        batch_size = tf.shape(inputs)[0]
        # Expand indices to have a batch dimension.
        tiled_indices = tf.expand_dims(self.indices, axis=0)  # (1, L, K)
        tiled_indices = tf.tile(tiled_indices, [batch_size, 1, 1])  # (batch, L, K)
        
        # Gather neighbor features along the pixel dimension (axis 1)
        neighbor_feats = tf.gather(inputs, tiled_indices, batch_dims=1, axis=1)
        # neighbor_feats shape: (batch, L, K, T, channels)
        
        # Expand mask to match neighbor_feats shape:
        tiled_mask = tf.expand_dims(self.mask, axis=0)  # (1, L, K)
        tiled_mask = tf.tile(tiled_mask, [batch_size, 1, 1])  # (batch, L, K)
        tiled_mask = tf.expand_dims(tiled_mask, axis=-1)  # (batch, L, K, 1)
        tiled_mask = tf.expand_dims(tiled_mask, axis=-1)  # (batch, L, K, 1, 1)
        
        # Apply the mask (casting is necessary to match neighbor_feats.dtype)
        neighbor_feats = neighbor_feats * tf.cast(tiled_mask, neighbor_feats.dtype)
        # tf.print("neighbor_feats shape:", tf.shape(neighbor_feats), "Example slice:", neighbor_feats[0, :, :, 2, 0])  # shape: (K,)
        return neighbor_feats


class NeighborGatherLayer2D(tf.keras.layers.Layer):
    """
    Custom layer to gather neighbor features.
    
    Given an input tensor with shape (batch, pixels_per_patch, seq_length, channels)
    and neighbor_indices with shape (pixels_per_patch, K) where K includes the pixel itself 
    and its neighbors (invalid indices should be -1), this layer returns a tensor with shape 
    (batch, pixels_per_patch, K, seq_length, channels) where invalid neighbor entries are zeroed out.
    """
    
    def __init__(self, neighbor_indices, **kwargs):
        super().__init__(**kwargs)
        # Convert neighbor_indices into constants.
        self.indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32) # (L, K)
        # Create mask: 0 where indices = -1.
        self.mask = tf.cast(tf.not_equal(self.indices, -1), tf.float32)  # (L, K)
        
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
        # neighbor_feats shape: (batch, L, K, channels)
        
        # Expand mask to match neighbor_feats shape:
        tiled_mask = tf.expand_dims(self.mask, axis=0)  # (1, L, K)
        tiled_mask = tf.tile(tiled_mask, [batch_size, 1, 1])  # (batch, L, K)
        tiled_mask = tf.expand_dims(tiled_mask, axis=-1)  # (batch, L, K, 1)
        
        # Apply the mask (casting is necessary to match neighbor_feats.dtype)
        neighbor_feats = neighbor_feats * tf.cast(tiled_mask, neighbor_feats.dtype)
        # tf.print("neighbor_feats shape:", tf.shape(neighbor_feats), "Example slice:", neighbor_feats[0, :, :, 0])  # shape: (K,)
        return neighbor_feats


