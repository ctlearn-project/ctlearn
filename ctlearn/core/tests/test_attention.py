import pytest
import torch
import keras
import tensorflow as tf

from ctlearn.core.keras.model import (
    channel_squeeze_excite_block,
    spatial_squeeze_excite_block,
    dual_squeeze_excite_block,
)
from ctlearn.core.pytorch.attention import(
    ChannelSqueezeExciteBlock,
    SpatialSqueezeExciteBlock,
    DualSqueezeExciteBlock,
)

def build_keras_se_model(block_fn, input_shape, **kwargs):
    """Wraps a Keras functional squeeze-excite block in a Keras Model."""
    inputs = keras.Input(shape=input_shape)
    outputs = block_fn(inputs, name="se_block", **kwargs)
    return keras.Model(inputs=inputs, outputs=outputs)


@pytest.mark.parametrize(
    "k_fn, p_class, kwargs",
    [
        (channel_squeeze_excite_block, ChannelSqueezeExciteBlock, {"ratio": 4}),
        (spatial_squeeze_excite_block, SpatialSqueezeExciteBlock, {}),
        (dual_squeeze_excite_block, DualSqueezeExciteBlock, {"ratio": 16}),
    ],
)
@pytest.mark.parametrize(
    "batch, height, width, channels",
    [
        (1, 8, 8, 16),
        (4, 32, 32, 64),
    ],
)
def test_output_shape_parity(k_fn, p_class, kwargs, batch, height, width, channels):
    """Verifies that output shapes match between Keras (BHWC) and PyTorch (BCHW)."""
    # Keras Input: (Batch, H, W, C)
    x_k = tf.random.normal((batch, height, width, channels))
    k_model = build_keras_se_model(k_fn, input_shape=(height, width, channels), **kwargs)
    k_out = k_model(x_k)

    # PyTorch Input: (Batch, C, H, W)
    x_p = torch.randn(batch, channels, height, width)
    p_module = p_class(in_channels=channels, **kwargs)
    p_module.eval()
    with torch.no_grad():
        p_out = p_module(x_p)

    # Check output shape correspondence
    assert k_out.shape == (batch, height, width, channels)
    assert p_out.shape == (batch, channels, height, width)

    # Verify equivalent dimensions (transpose PyTorch output to BHWC)
    p_out_bhwc = p_out.permute(0, 2, 3, 1)
    assert k_out.shape == p_out_bhwc.shape
