"""
This module defines the squeeze-excite blocks for channel-wise and/or spatial-wise attention mechanisms in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "DualSqueezeExciteBlock",
    "ChannelSqueezeExciteBlock",
    "SpatialSqueezeExciteBlock",
]

class DualSqueezeExciteBlock(nn.Module):
    """
    A channel & spatial (dual) squeeze-excite block in PyTorch.
    Concurrently applies channel and spatial scaling, then sums the results.
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.cse = ChannelSqueezeExciteBlock(in_channels=in_channels, ratio=ratio)
        self.sse = SpatialSqueezeExciteBlock(in_channels=in_channels)

    def forward(self, x):
        # Combines cse and sse by element-wise addition
        return self.cse(x) + self.sse(x)


class ChannelSqueezeExciteBlock(nn.Module):
    """
    A channel-wise squeeze-excite (cSE) block in PyTorch.
    """
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        # Keras uses Dense layers on global pooled tensors. 
        # In PyTorch, we can achieve this elegantly using 1x1 Convolutions, 
        # avoiding the need to flatten and unflatten the spatial grid.
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global Average Pooling keeping spatial dims: (B, C, H, W) -> (B, C, 1, 1)
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))
        # Compute channel scale factor
        excitation = self.gate(squeeze)
        # Multiply input tensor by the scale factor across the channel dimension
        return x * excitation


class SpatialSqueezeExciteBlock(nn.Module):
    """
    A spatial squeeze-excite (sSE) block in PyTorch.
    """
    def __init__(self, in_channels):
        super().__init__()
        # A 1x1 convolution projecting channels down to 1 spatial mask
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # Create a spatial landscape mask via sigmoid
        spatial_mask = torch.sigmoid(self.spatial_conv(x))
        # Multiply input tensor element-wise across spatial layout
        return x * spatial_mask