"""
Simplified Dual-Input Transformer Network Module

This module implements a simplified dual-input transformer architecture that processes
two modalities (charge and timing) by concatenating them at the input level rather than
using separate backbones. This approach is more memory-efficient and has fewer parameters
than the dual-backbone variant.

The architecture combines convolutional layers with transformer encoder blocks, providing
both local feature extraction and global context modeling through self-attention.

Classes:
    BottleneckTransformerBlock: Bottleneck block with transformer encoder layer
    TransformerDBB: Single-backbone transformer for dual-input data
    TransformerDuo: Factory function for creating TransformerDBB models

References:
    - "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)
    - "BoTNet: Bottleneck Transformers for Visual Recognition" (Srinivas et al., CVPR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckTransformerBlock(nn.Module):
    """
    Bottleneck block with integrated transformer encoder layer.
    
    This block combines the efficiency of bottleneck architectures with the
    global receptive field of transformers. It uses dimensionality reduction
    before applying self-attention, making it computationally efficient while
    still capturing long-range dependencies.
    
    Architecture:
        Input → Conv1x1(reduce) → Norm → ReLU →
        Transformer(self-attention) →
        Conv1x1(expand) → Norm → (+) Shortcut → ReLU → Output
    
    Attributes:
        expansion (int): Channel expansion factor (set to 1)
        conv1 (nn.Conv2d): 1x1 conv for channel reduction
        norm1 (nn.BatchNorm2d or nn.GroupNorm): Normalization after reduction
        transformer_block (nn.TransformerEncoderLayer): Self-attention layer
        conv2 (nn.Conv2d): 1x1 conv for channel expansion
        norm2 (nn.BatchNorm2d or nn.GroupNorm): Normalization after expansion
        shortcut (nn.Sequential): Skip connection (identity or 1x1 conv)
    """
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=4, use_gn=False):
        """
        Initialize the bottleneck transformer block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int, optional): Stride for the shortcut connection. Defaults to 1
            reduction (int, optional): Channel reduction factor. Defaults to 4
            use_gn (bool, optional): Whether to use GroupNorm instead of BatchNorm.
                Defaults to False
        """
        super(BottleneckTransformerBlock, self).__init__()
        
        # Channel reduction
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels // reduction, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        if use_gn:
            self.norm1 = nn.GroupNorm(32, out_channels // reduction)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels // reduction)

        # Transformer encoder with multi-head self-attention
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=out_channels // reduction, 
            nhead=8
        )
        
        # Channel expansion
        self.conv2 = nn.Conv2d(
            out_channels // reduction, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        if use_gn:
            self.norm2 = nn.GroupNorm(32, out_channels)
        else:
            self.norm2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the bottleneck transformer block.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, H, W)
        """
        # Reduce channels
        out = F.relu(self.norm1(self.conv1(x)))
        
        # Prepare for transformer: (B, C, H, W) → (H*W, B, C)
        b, c, h, w = out.size()
        out = out.view(b, c, -1).permute(2, 0, 1)
        
        # Apply self-attention
        out = self.transformer_block(out)
        
        # Reshape back: (H*W, B, C) → (B, C, H, W)
        out = out.permute(1, 2, 0).view(b, c, h, w)
        
        # Expand channels
        out = self.norm2(self.conv2(out))
        
        # Add residual connection
        out += self.shortcut(x)
        
        # Final activation
        out = F.relu(out)
        return out

class TransformerDBB(nn.Module):
    """
    Simplified single-backbone transformer for dual-input data.
    
    This architecture concatenates charge and timing inputs at the channel level
    before processing through a single transformer-augmented backbone. This approach
    is more parameter-efficient than dual-backbone architectures while still allowing
    the model to learn joint representations of both modalities.
    
    Architecture Overview:
        Input_1 (charge) ↘
                          → Concatenate → Single Backbone (Conv + Transformer) → Output
        Input_2 (timing) ↗
    
    Single Backbone:
        Concat(x1, x2) → Conv7x7 → Norm → ReLU →
        Layer1 (Transformer blocks) →
        Layer2 (Transformer blocks, stride=2) →
        Layer3 (Transformer blocks, stride=2) →
        Layer4 (Transformer blocks, stride=2) →
        Adaptive AvgPool → Dropout → FC → Output
    
    Advantages over Dual-Backbone:
        - ~50% fewer parameters (single backbone vs two)
        - Lower memory consumption
        - Faster training and inference
        - Implicit early fusion of modalities
    
    Attributes:
        in_channels (int): Current number of channels (updated during layer construction)
        use_gn (bool): Whether to use GroupNorm instead of BatchNorm
        conv1 (nn.Conv2d): Initial convolution
        norm1 (nn.Module): Initial normalization
        layer1-4 (nn.Sequential): Transformer block layers
        dropout (nn.Dropout): Dropout for regularization
        adaptive_pool (nn.AdaptiveAvgPool2d): Global average pooling
        fc (nn.Linear): Final classification/regression layer
    """
    
    def __init__(self, block, layers, num_inputs=2, num_classes=1, use_gn=False, dropout_rate=0.5):
        """
        Initialize the simplified transformer network.
        
        Args:
            block: Block class to use (BottleneckTransformerBlock)
            layers (list): Number of blocks in each layer [layer1, layer2, layer3, layer4]
            num_inputs (int, optional): Total number of input channels (charge + timing).
                Defaults to 2. Should be sum of channels from both modalities.
            num_classes (int, optional): Number of output classes/values. Defaults to 1
            use_gn (bool, optional): Whether to use GroupNorm. Defaults to False
            dropout_rate (float, optional): Dropout probability. Defaults to 0.5
            
        Note:
            With num_inputs=2, the network expects concatenated inputs where:
            - Channel 0: Charge information
            - Channel 1: Timing information
        """
        super(TransformerDBB, self).__init__()
        self.in_channels = 64
        self.use_gn = use_gn

        # Initial convolution: processes concatenated inputs
        # Accepts num_inputs channels (e.g., 2 for charge + timing)
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1 = nn.GroupNorm(32, 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)
        
        # Transformer block layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Global pooling and final layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Create a layer consisting of multiple transformer blocks.
        
        Args:
            block: Block class (BottleneckTransformerBlock)
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential container of transformer blocks
        """
        layers = []
        # First block may downsample
        layers.append(block(self.in_channels, out_channels, stride, use_gn=self.use_gn))
        self.in_channels = out_channels
        # Remaining blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_gn=self.use_gn))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        """
        Forward pass through the simplified transformer network.
        
        Process:
        1. Concatenate inputs along channel dimension
        2. Process through single backbone (Conv + Transformer layers)
        3. Global average pooling
        4. Dropout for regularization
        5. Final classification/regression layer
        
        Args:
            x1 (torch.Tensor): First input (charge images)
                Shape: (batch_size, 1, H, W)
            x2 (torch.Tensor): Second input (timing images)
                Shape: (batch_size, 1, H, W)
                
        Returns:
            torch.Tensor: Output predictions
                Shape: (batch_size, num_classes)
                
        Input Fusion:
            Early fusion via concatenation: x = [x1 | x2]
            This allows the network to learn joint features from both modalities
            from the very first layer, unlike dual-backbone approaches where
            fusion happens later.
            
        Example:
            >>> model = TransformerDBB(...)
            >>> x1 = torch.randn(16, 1, 120, 120)  # Charge
            >>> x2 = torch.randn(16, 1, 120, 120)  # Timing
            >>> output = model(x1, x2)
            >>> print(output.shape)  # torch.Size([16, 1])
        """
        # Concatenate inputs along channel dimension
        # x1: (B, 1, H, W), x2: (B, 1, H, W) → x: (B, 2, H, W)
        x = torch.cat((x1, x2), dim=1)

        # Process through single backbone
        out = F.relu(self.norm1(self.conv1(x)))  # Initial conv
        out = self.layer1(out)  # Transformer blocks
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global pooling and prediction
        out = self.adaptive_pool(out)  # (B, 512, 1, 1)
        out = out.view(out.size(0), -1)  # (B, 512)
        out = self.dropout(out)  # Regularization
        out = self.fc(out)  # (B, num_classes)
        
        return out

def TransformerDuo(num_blocks=[3, 4, 6, 3], num_inputs=2, num_classes=2, use_gn=True, dropout_rate=0.3):
    """
    Factory function to create a simplified dual-input transformer network.
    
    This function instantiates a TransformerDBB model with the specified configuration.
    It provides a convenient interface for creating transformer-based models that
    process two input modalities through early fusion.
    
    Args:
        num_blocks (list, optional): Number of blocks in each layer. Defaults to [3, 4, 6, 3]
            Similar to ResNet50 architecture
        num_inputs (int, optional): Total input channels. Defaults to 2
            Should equal the sum of channels from all input modalities
        num_classes (int, optional): Number of output classes/values. Defaults to 2
        use_gn (bool, optional): Whether to use GroupNorm. Defaults to True
            Recommended for stability with small batches
        dropout_rate (float, optional): Dropout probability. Defaults to 0.3
            Lower than dual-backbone default (0.5) since single backbone
            
    Returns:
        TransformerDBB: Instantiated simplified transformer model
        
    Example:
        >>> # Create model for binary classification
        >>> model = TransformerDuo(
        ...     num_blocks=[3, 4, 6, 3],
        ...     num_inputs=2,
        ...     num_classes=2,
        ...     use_gn=True,
        ...     dropout_rate=0.3
        ... )
        >>> 
        >>> # Forward pass
        >>> x1 = torch.randn(8, 1, 120, 120)  # Charge
        >>> x2 = torch.randn(8, 1, 120, 120)  # Timing
        >>> output = model(x1, x2)
        >>> print(output.shape)  # torch.Size([8, 2])
        
    Comparison with Dual-Backbone:
        Simplified (this):
            - Pros: Fewer parameters, faster, lower memory
            - Cons: Less modality-specific feature learning
            
        Dual-Backbone:
            - Pros: More modality-specific features, potentially higher accuracy
            - Cons: More parameters, slower, higher memory
    """
    return TransformerDBB(
        BottleneckTransformerBlock, 
        num_blocks, 
        num_inputs=num_inputs, 
        num_classes=num_classes, 
        use_gn=use_gn, 
        dropout_rate=dropout_rate
    )