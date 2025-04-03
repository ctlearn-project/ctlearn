import tensorflow as tf
from tensorflow import keras

def prepare_mask(neighbor_indices):
    """
    Prepares the mask from the neighbor array (shape: (L, K)).

    Parameters:
      neighbor_indices: array (numpy or similar) with shape (L, K), where L is the number of pixels 
                        and K is the maximum number of neighbors. -1 is used to indicate the absence of a neighbor.

    Returns:
      indices: tensor of shape (L, K) unchanged (using -1 in the masking logic).
      mask: tensor of shape (L, K) with 1.0 where the index is valid and 0.0 where it is -1.
    """
    indices = tf.convert_to_tensor(neighbor_indices, dtype=tf.int32)
    mask = tf.cast(tf.not_equal(indices, -1), tf.float32)
    return indices, mask


class IndexedConv3D(keras.layers.Layer):
    """
    Indexed 3D convolutional layer that operates on a pixel array with a time dimension.

    The input is expected to have shape (batch, L, T, in_channels) and 
    neighbor_indices should have shape (L, K) (with -1 indicating a missing neighbor).
    
    The convolution applies a kernel across both the pixel (neighbors) and time dimensions.
    For valid convolution along time (no padding), the operation is:
    
      out(b, l, t, o) = sum_{k=0}^{K-1} sum_{\tau=0}^{temporal_kernel_size-1} sum_{i} 
                         kernel[k, \tau, i, o] * input(b, indices[l,k], t + \tau, i)
    
    The layer supports an optional bias term.
    """
    def __init__(self, out_channels, indices, temporal_kernel_size,
                 use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.use_bias = use_bias
        self.indices = indices
        self.num_neighbors = self.indices.shape[-1]
        self.neighbor_indices, self.mask = prepare_mask(self.indices)
        self.indices_for_gather = tf.where(tf.equal(self.neighbor_indices, -1), tf.zeros_like(self.neighbor_indices), self.neighbor_indices)

        
        self.indices = tf.expand_dims(self.indices_for_gather, axis=0)  # (1, L, K)
        self.mask = tf.expand_dims(self.mask, axis=0)                     # (1, L, K)
        self.mask = tf.expand_dims(mask, axis=-1)                         # (1, L, K, 1)


    def build(self, input_shape):
        # input_shape is expected to be (batch, L, T, in_channels)
        in_channels = input_shape[-1]


        # Create kernel of shape (K, temporal_kernel_size, in_channels, out_channels)
        self.kernel = self.add_weight(
            shape=(self.num_neighbors, self.temporal_kernel_size, in_channels, self.out_channels),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        super().build(input_shape)

    def get_batched_indices_and_mask(self, batch_size):

        indices = tf.tile(self.indices, [batch_size, 1, 1])               # (batch, L, K)
        mask = tf.tile(self.mask, [batch_size, 1, 1, 1])                  # (batch, L, K, 1)

        return indices, mask

    def call(self, inputs):
        """
        inputs: Tensor of shape (batch, L, T, in_channels)
        
        Steps:
          1. Expand neighbor indices and mask for batch dimension.
          2. Replace -1 indices with 0 so tf.gather works.
          3. Gather neighbor features: result shape (batch, L, K, T, in_channels)
          4. Apply mask along the neighbor dimension.
          5. Extract sliding windows over the time dimension (valid convolution):
             result shape becomes (batch, L, K, T_out, temporal_kernel_size, in_channels) where
             T_out = T - temporal_kernel_size + 1.
          6. Perform the weighted sum over neighbor (K) and time-window (temporal_kernel_size) dimensions.
        """
        batch_size = tf.shape(inputs)[0]
        indices, mask = self.get_batched_indices_and_mask(batch_size)

        # indices = tf.expand_dims(self.neighbor_indices, axis=0)  # (1, L, K)
        # mask = tf.expand_dims(self.mask, axis=0)  # (1, L, K)
        # mask = tf.expand_dims(mask, axis=-1)  # (1, L, K, 1)

        # indices_for_gather = tf.tile(self.indices_for_gather, [batch_size, 1, 1])            # (batch, L, K)
        # mask = tf.tile(mask, [batch_size, 1, 1, 1])                   # (batch, L, K, 1)

        # Gather neighbor features along pixel dimension.
        # Inputs shape: (batch, L, T, in_channels) -> gathered: (batch, L, K, T, in_channels)
        # neighbor_feats = tf.gather(inputs, indices_for_gather, batch_dims=1)
        # Apply mask to cancel contributions from missing neighbors.
        # Convert mask to the correct dtype and broadcast to (batch, L, K, sequence_length, in_channels)
        # neighbor_feats = neighbor_feats * tf.cast(tf.expand_dims(mask, axis=-1), dtype=neighbor_feats.dtype)

        neighbor_feats = tf.gather(inputs, indices, batch_dims=1)
        neighbor_feats = neighbor_feats * tf.cast(tf.expand_dims(mask, axis=-1), dtype=neighbor_feats.dtype)
        
        # Compute valid output length along time dimension.
        # T_out = T - self.temporal_kernel_size + 1

        # Extract sliding windows from the time dimension.
        # We use tf.signal.frame on axis=3 (time axis).
        # Resulting shape: (batch, L, K, T_out, temporal_kernel_size, in_channels)
        neighbor_frames = tf.signal.frame(neighbor_feats, frame_length=self.temporal_kernel_size, frame_step=1, axis=3)

        # Now perform the convolution:
        # We need to sum over the neighbor and time window dimensions.
        # neighbor_frames shape: (batch, L, K, T_out, temporal_kernel_size, in_channels)
        # kernel shape: (K, temporal_kernel_size, in_channels, out_channels)
        # We perform an Einstein summation over dimensions:
        #   'b l k t tau i, k tau i o -> b l t o'
        out = tf.einsum('b l k t d i, k d i o -> b l t o', neighbor_frames, self.kernel)
        
        if self.use_bias:
            out = tf.add(out, self.bias)
        # Output shape: (batch, L, T_out, out_channels)
        return out


class IndexedConv2D(keras.layers.Layer):
    """
    Indexed convolutional layer in TensorFlow that operates on a pixel array.

    The input is expected to have shape (batch, L, in_channels) and 
    neighbor_indices should have shape (L, K). The value -1 indicates no neighbor.

    The operation is:
      out(b, l, o) = sum_{k=0}^{K-1} sum_{i} kernel[k, i, o] * input(b, indices[l,k], i)
    To avoid issues with negative indices, -1 values are temporarily replaced with 0,
    and a mask is used to cancel the contribution of missing neighbors.
    """
    def __init__(self, out_channels, neighbor_indices, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.neighbor_indices, self.mask = prepare_mask(neighbor_indices)  # Both of shape (L, K)
        self.num_neighbors = self.neighbor_indices.shape[-1]
        self.use_bias = use_bias

    # def get_config(self):
    #     config = super().get_config()
    #     # config.update({
    #     #     "filters": self.filters,
    #     #     "kernel_size": self.kernel_size,
    #     #     "indices": self.neighbor_indices.tolist() if isinstance(self.indices, np.ndarray) else self.neighbor_indices
    #     # })
    #     # return config

    def build(self, input_shape):
        in_channels = input_shape[-1]
        # Layer weights: (K, in_channels, out_channels)
        self.kernel = self.add_weight(
            shape=(self.num_neighbors, in_channels, self.out_channels),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.out_channels,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, L, in_channels)
        batch_size = tf.shape(inputs)[0]
        L = tf.shape(inputs)[1]
        
        # To use tf.gather with batch_dims=1, we expand and tile the indices and mask for each batch.
        # Originally, self.neighbor_indices and self.mask have shape (L, K). We expand them to (1, L, K) and then tile them.
        indices = tf.expand_dims(self.neighbor_indices, axis=0)  # (1, L, K)
        indices = tf.tile(indices, [batch_size, 1, 1])  # (batch, L, K)
        mask = tf.expand_dims(self.mask, axis=0)  # (1, L, K)
        mask = tf.tile(mask, [batch_size, 1, 1])  # (batch, L, K)

        # Replace -1 with 0 in indices so tf.gather works (without affecting the mask)
        indices_for_gather = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), indices)
        
        # tf.gather with batch_dims=1: result -> (batch, L, K, in_channels)
        neighbor_feats = tf.gather(inputs, indices_for_gather, batch_dims=1)
        # Apply mask to cancel contributions from missing neighbors
        neighbor_feats = neighbor_feats * tf.cast(tf.expand_dims(mask, axis=-1), dtype=neighbor_feats.dtype)
        # neighbor_feats = neighbor_feats * tf.expand_dims(mask, axis=-1)
        # Use tf.einsum to combine neighbors with weights.
        # 'b' = batch, 'l' = pixel, 'k' = neighbors, 'i' = input channels, 'o' = output channels.
        out = tf.einsum('blki,kio->blo', neighbor_feats, self.kernel)
        if self.use_bias:
            out = out + self.bias
        # out: (batch, L, out_channels)
        return out

class IndexedMaxPool2D(keras.layers.Layer):
    """
    Indexed max pooling layer in TensorFlow.
    Operates on an input of shape (batch, L, channels) using neighbor_indices of shape (L, K).
    """
    def __init__(self, neighbor_indices, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_indices, self.mask = prepare_mask(neighbor_indices)
        self.num_neighbors = self.neighbor_indices.shape[-1]

    def call(self, inputs):
        # inputs: (batch, L, channels)
        batch_size = tf.shape(inputs)[0]
        # Expand and tile indices and mask for the batch
        indices = tf.expand_dims(self.neighbor_indices, axis=0)
        indices = tf.tile(indices, [batch_size, 1, 1])
        mask = tf.expand_dims(self.mask, axis=0)
        mask = tf.tile(mask, [batch_size, 1, 1])
        
        # Replace -1 with 0 for tf.gather
        indices_for_gather = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), indices)
        neighbor_feats = tf.gather(inputs, indices_for_gather, batch_dims=1)
        neighbor_feats = neighbor_feats * tf.expand_dims(mask, axis=-1)
        out = tf.reduce_max(neighbor_feats, axis=2)
        return out

class IndexedAveragePool2D(keras.layers.Layer):
    """
    Indexed average pooling layer in TensorFlow.
    Operates on an input of shape (batch, L, channels) using neighbor_indices of shape (L, K).
    """
    def __init__(self, neighbor_indices, **kwargs):
        super().__init__(**kwargs)
        self.neighbor_indices, self.mask = prepare_mask(neighbor_indices)
        self.num_neighbors = self.neighbor_indices.shape[-1]

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        indices = tf.expand_dims(self.neighbor_indices, axis=0)
        indices = tf.tile(indices, [batch_size, 1, 1])
        mask = tf.expand_dims(self.mask, axis=0)
        mask = tf.tile(mask, [batch_size, 1, 1])
        
        indices_for_gather = tf.where(tf.equal(indices, -1), tf.zeros_like(indices), indices)
        neighbor_feats = tf.gather(inputs, indices_for_gather, batch_dims=1)
        neighbor_feats = neighbor_feats * tf.expand_dims(mask, axis=-1)
        out = tf.reduce_mean(neighbor_feats, axis=2)
        return out