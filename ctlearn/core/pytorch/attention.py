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
        reduced_channels = in_channels // ratio
        # Using nn.Linear to match Keras Dense layers
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=True)

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        # Global Average Pooling keeping dimensions: (B, C, H, W) -> (B, C, 1, 1)
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))
        # Flatten for Linear layers: (B, C, 1, 1) -> (B, C)
        squeeze = squeeze.view(batch_size, channels)
        # Dense projections with ReLU and Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        # Reshape back to broadcast across spatial dimensions: (B, C) -> (B, C, 1, 1)
        excitation = excitation.view(batch_size, channels, 1, 1)
        # Scale input tensor
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