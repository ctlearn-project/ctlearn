"""
Dual-Backbone Transformer Network Module

This module implements a dual-backbone architecture combining Convolutional Neural Networks
with Transformer encoder blocks for processing Cherenkov telescope data. The architecture
uses bottleneck transformer blocks that integrate self-attention mechanisms into a
convolutional backbone, providing both local feature extraction and global context modeling.

The dual-backbone design allows independent processing of charge and timing information
from telescope cameras before fusion for final predictions.

Classes:
    BottleneckTransformerBlock: Bottleneck block with transformer encoder layer
    TransformerDBB: Dual-backbone transformer network
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
    
    The transformer block processes spatial features as a sequence, allowing
    each position to attend to all other positions, capturing global context
    that convolutional layers might miss.
    
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
                If stride > 1, downsampling is applied in the shortcut
            reduction (int, optional): Channel reduction factor. Defaults to 4
                The intermediate dimension is out_channels // reduction
                Higher reduction = fewer parameters but may lose information
            use_gn (bool, optional): Whether to use GroupNorm instead of BatchNorm.
                Defaults to False
                GroupNorm is more stable for small batch sizes
        """
        super(BottleneckTransformerBlock, self).__init__()
        
        # Channel reduction: compress features before transformer
        # Reduces computational cost of self-attention
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

        # Transformer encoder layer for global context
        # Uses multi-head self-attention with 8 heads
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=out_channels // reduction, 
            nhead=8
        )
        
        # Channel expansion: restore original channel dimension
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

        # Skip connection: adjust channels/spatial dims if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the bottleneck transformer block.
        
        Process:
        1. Reduce channels with 1x1 convolution
        2. Normalize and activate
        3. Reshape for transformer (flatten spatial dimensions)
        4. Apply self-attention via transformer encoder
        5. Reshape back to spatial format
        6. Expand channels with 1x1 convolution
        7. Add skip connection
        8. Final ReLU activation
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, H, W)
                
        Transformer Processing:
            The spatial dimensions (H, W) are flattened into a sequence of length H*W,
            where each position can attend to all other positions. This allows the
            model to capture long-range dependencies that convolutional layers miss.
        """
        # Channel reduction
        out = F.relu(self.norm1(self.conv1(x)))  # (B, C_reduced, H, W)
        
        # Prepare for transformer: (B, C, H, W) → (H*W, B, C)
        b, c, h, w = out.size()
        out = out.view(b, c, -1).permute(2, 0, 1)  # Flatten spatial dims
        
        # Self-attention: each position attends to all positions
        out = self.transformer_block(out)  # (H*W, B, C)
        
        # Reshape back to spatial format: (H*W, B, C) → (B, C, H, W)
        out = out.permute(1, 2, 0).view(b, c, h, w)
        
        # Channel expansion
        out = self.norm2(self.conv2(out))  # (B, out_channels, H, W)
        
        # Add residual connection
        out += self.shortcut(x)
        
        # Final activation
        out = F.relu(out)
        return out

class TransformerDBB(nn.Module):
    """
    Dual-Backbone Transformer Network for multi-modal telescope data.
    
    This architecture uses two independent transformer-augmented backbones to process
    different input modalities (charge and timing images) before fusing features for
    final prediction. Each backbone combines convolutional layers with transformer
    blocks to capture both local patterns and global context.
    
    Architecture Overview:
        Input_1 (charge) → Backbone_1 (Conv + Transformer) → Features_1
                                                                ↓
                                                              Fusion
                                                                ↓
        Input_2 (timing) → Backbone_2 (Conv + Transformer) → Features_2
                                                                ↓
                                                         Global Pool → FC → Output
    
    Each Backbone:
        Conv7x7 → Norm → ReLU →
        Layer1 (Transformer blocks) →
        Layer2 (Transformer blocks, stride=2) →
        Layer3 (Transformer blocks, stride=2) →
        Layer4 (Transformer blocks, stride=2) →
        Adaptive AvgPool
    
    Attributes:
        in_channels (int): Current number of channels (updated during layer construction)
        use_gn (bool): Whether to use GroupNorm instead of BatchNorm
        use_concat (bool): Whether to concatenate or add backbone outputs
        conv1_a, conv1_b (nn.Conv2d): Initial convolutions for each backbone
        norm1_a, norm1_b (nn.Module): Initial normalization for each backbone
        layer1-4_a, layer1-4_b (nn.Sequential): Transformer block layers for each backbone
        dropout (nn.Dropout): Dropout for regularization
        adaptive_pool (nn.AdaptiveAvgPool2d): Global average pooling
        fc (nn.Linear): Final classification/regression layer
    """
    
    def __init__(self, block, layers, num_inputs=1, num_classes=1, use_gn=False, use_concat=False, dropout_rate=0.5):
        """
        Initialize the dual-backbone transformer network.
        
        Args:
            block: Block class to use (e.g., BottleneckTransformerBlock)
            layers (list): Number of blocks in each layer [layer1, layer2, layer3, layer4]
                Example: [3, 4, 6, 3] for a ResNet50-like architecture
            num_inputs (int, optional): Number of input channels per backbone. Defaults to 1
            num_classes (int, optional): Number of output classes/values. Defaults to 1
            use_gn (bool, optional): Whether to use GroupNorm. Defaults to False
            use_concat (bool, optional): Whether to concatenate backbone outputs.
                Defaults to False (uses addition)
            dropout_rate (float, optional): Dropout probability. Defaults to 0.5
        """
        super(TransformerDBB, self).__init__()
        self.in_channels = 64
        self.use_gn = use_gn
        self.use_concat = use_concat

        # Backbone 1: Process first input modality (charge images)
        self.conv1_a = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1_a = nn.GroupNorm(32, 64)
        else:
            self.norm1_a = nn.BatchNorm2d(64)
        
        # Transformer block layers for backbone 1
        self.layer1_a = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_a = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_a = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_a = self._make_layer(block, 512, layers[3], stride=2)

        # Reset in_channels for second backbone
        self.in_channels = 64
        
        # Backbone 2: Process second input modality (timing images)
        self.conv1_b = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1_b = nn.GroupNorm(32, 64)
        else:
            self.norm1_b = nn.BatchNorm2d(64)

        # Transformer block layers for backbone 2
        self.layer1_b = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_b = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_b = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_b = self._make_layer(block, 512, layers[3], stride=2)

        # Regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Global pooling and final layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final FC layer dimension depends on fusion strategy
        if self.use_concat:
            self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Create a layer consisting of multiple transformer blocks.
        
        Args:
            block: Block class (BottleneckTransformerBlock)
            out_channels (int): Number of output channels
            blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block (for downsampling)
            
        Returns:
            nn.Sequential: Sequential container of transformer blocks
        """
        layers = []
        # First block may have stride > 1 for downsampling
        layers.append(block(self.in_channels, out_channels, stride, use_gn=self.use_gn))
        self.in_channels = out_channels
        # Remaining blocks maintain spatial dimensions
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_gn=self.use_gn))
        return nn.Sequential(*layers)

    def forward(self, x1):
        if x1.shape[1] >= 2:
            x1, x2 = torch.split(x1, [1, x1.shape[1]-1], dim=1)
        else:
            x2 = x1
        """
        Forward pass through the dual-backbone transformer network.
        
        Process:
        1. Process each input through its backbone (Conv + Transformer layers)
        2. Apply global average pooling to each backbone's output
        3. Flatten spatial dimensions
        4. Apply dropout
        5. Fuse features (concatenate or add)
        6. Final classification/regression layer
        
        Args:
            x1 (torch.Tensor): First input (charge images)
                Shape: (batch_size, num_inputs, H, W)
            x2 (torch.Tensor): Second input (timing images)
                Shape: (batch_size, 1, H, W)
                
        Returns:
            torch.Tensor: Output predictions
                Shape: (batch_size, num_classes)
                
        Feature Fusion:
            Concatenation (use_concat=True):
                - Preserves all information from both backbones
                - Doubles feature dimension
                - More parameters in final layer
                
            Addition (use_concat=False):
                - Forces features into shared space
                - Fewer parameters
                - May lose complementary information
        """
        # Backbone 1: Process charge images
        out1 = F.relu(self.norm1_a(self.conv1_a(x1)))  # Initial conv
        out1 = self.layer1_a(out1)  # Transformer blocks
        out1 = self.layer2_a(out1)
        out1 = self.layer3_a(out1)
        out1 = self.layer4_a(out1)
        out1 = self.adaptive_pool(out1)  # Global pooling
        out1 = out1.view(out1.size(0), -1)  # Flatten
        out1 = self.dropout(out1)  # Regularization

        # Backbone 2: Process timing images
        out2 = F.relu(self.norm1_b(self.conv1_b(x2)))  # Initial conv
        out2 = self.layer1_b(out2)  # Transformer blocks
        out2 = self.layer2_b(out2)
        out2 = self.layer3_b(out2)
        out2 = self.layer4_b(out2)
        out2 = self.adaptive_pool(out2)  # Global pooling
        out2 = out2.view(out2.size(0), -1)  # Flatten
        out2 = self.dropout(out2)  # Regularization

        # Fusion: Combine features from both backbones
        if self.use_concat:
            out = torch.cat((out1, out2), dim=1)  # Concatenate
        else:
            out = out1 + out2  # Element-wise addition

        # Final prediction
        out = self.fc(out)
        return out

def TransformerDuo(num_blocks=[3, 4, 6, 3], num_inputs=1, num_classes=2, use_gn=True, use_concat=False, dropout_rate=0.5):
    """
    Factory function to create a Dual-Backbone Transformer network.
    
    This function instantiates a TransformerDBB model with the specified configuration.
    It provides a convenient interface for creating transformer-based models with
    different depths and configurations.
    
    Args:
        num_blocks (list, optional): Number of blocks in each layer. Defaults to [3, 4, 6, 3]
            Similar to ResNet50 architecture:
            - Layer 1: 3 blocks (64 channels)
            - Layer 2: 4 blocks (128 channels, stride=2)
            - Layer 3: 6 blocks (256 channels, stride=2)
            - Layer 4: 3 blocks (512 channels, stride=2)
        num_inputs (int, optional): Number of input channels. Defaults to 1
        num_classes (int, optional): Number of output classes/values. Defaults to 2
        use_gn (bool, optional): Whether to use GroupNorm. Defaults to True
            Recommended for small batch sizes or distributed training
        use_concat (bool, optional): Whether to concatenate backbone outputs.
            Defaults to False
        dropout_rate (float, optional): Dropout probability. Defaults to 0.5
        
    Returns:
        TransformerDBB: Instantiated dual-backbone transformer model
        
    Example:
        >>> # Create model for binary classification
        >>> model = TransformerDuo(
        ...     num_blocks=[3, 4, 6, 3],
        ...     num_inputs=1,
        ...     num_classes=2,
        ...     use_gn=True,
        ...     use_concat=False,
        ...     dropout_rate=0.5
        ... )
        >>> 
        >>> # Forward pass
        >>> x1 = torch.randn(8, 1, 120, 120)  # Charge images
        >>> x2 = torch.randn(8, 1, 120, 120)  # Timing images
        >>> output = model(x1, x2)
        >>> print(output.shape)  # torch.Size([8, 2])
        
    Notes:
        - The default configuration creates a lighter model than standard ResNet50
        - Transformer blocks add global context modeling at each layer
        - GroupNorm is more stable than BatchNorm for small batches
        - Higher dropout rates help prevent overfitting with transformers
    """
    return TransformerDBB(
        BottleneckTransformerBlock, 
        num_blocks,
        num_inputs, 
        num_classes=num_classes, 
        use_gn=use_gn, 
        use_concat=use_concat, 
        dropout_rate=dropout_rate
    )