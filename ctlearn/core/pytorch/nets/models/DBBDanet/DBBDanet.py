"""
Dual-Backbone DANet (Dual Attention Network) Module

This module implements a dual-backbone architecture with attention mechanisms for
processing Cherenkov telescope images. It combines two separate backbones to process
image and timing information independently, then fuses the features for final prediction.

The architecture incorporates both channel and spatial attention mechanisms to improve
feature representation and focus on the most relevant information in telescope images.

Classes:
    ChannelAttentionModule: Channel attention mechanism using global pooling
    SpatialAttentionModule: Spatial attention mechanism using convolutional layers
    DANet: Single backbone with dual attention mechanisms
    DBBDanet: Dual-backbone network combining two DANet architectures

References:
    - DANet: "Dual Attention Network for Scene Segmentation" (CVPR 2019)
    - CBAM: "Convolutional Block Attention Module" (ECCV 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module using global pooling.
    
    This module learns to emphasize informative channels and suppress less useful ones
    by exploiting inter-channel relationships. It uses both average and max pooling to
    capture different aspects of channel-wise statistics.
    
    Architecture:
        Input → [AvgPool, MaxPool] → Shared MLP → Element-wise Sum → Sigmoid → Channel Weights
    
    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling
        max_pool (nn.AdaptiveMaxPool2d): Global max pooling
        fc (nn.Sequential): Shared MLP for channel attention
        sigmoid (nn.Sigmoid): Activation for attention weights
    """
    
    def __init__(self, in_channels, reduction=16):
        """
        Initialize the Channel Attention Module.
        
        Args:
            in_channels (int): Number of input channels
            reduction (int, optional): Channel reduction ratio for the MLP bottleneck.
                Defaults to 16. Higher values reduce parameters but may lose information.
                
        Example:
            >>> cam = ChannelAttentionModule(in_channels=512, reduction=16)
            >>> x = torch.randn(8, 512, 7, 7)
            >>> out = cam(x)  # Same shape as input, but with channel attention applied
        """
        super(ChannelAttentionModule, self).__init__()
        
        # Global pooling operations to capture channel-wise statistics
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Output: (B, C, 1, 1)
        
        # Shared MLP: Channel reduction → ReLU → Channel restoration
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply channel attention to the input feature map.
        
        Process:
        1. Apply global average pooling and max pooling separately
        2. Pass both through shared MLP
        3. Sum the two attention maps
        4. Apply sigmoid to get attention weights in [0, 1]
        5. Multiply with original input (element-wise)
        
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W)
                B: batch size, C: channels, H: height, W: width
                
        Returns:
            torch.Tensor: Channel-attended feature map with same shape as input
                Important channels are emphasized, less important ones suppressed
        """
        # Process through average pooling path
        avg_out = self.fc(self.avg_pool(x))  # (B, C, 1, 1)
        
        # Process through max pooling path
        max_out = self.fc(self.max_pool(x))  # (B, C, 1, 1)
        
        # Combine both paths and apply sigmoid
        out = avg_out + max_out  # (B, C, 1, 1)
        
        # Apply attention weights to input
        return self.sigmoid(out) * x  # (B, C, H, W)

class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module using channel pooling.
    
    This module learns to focus on important spatial locations in the feature map
    by exploiting inter-spatial relationships. It uses both average and max pooling
    across channels to generate a spatial attention map.
    
    Architecture:
        Input → [AvgPool(channel), MaxPool(channel)] → Concat → Conv → Sigmoid → Spatial Weights
    
    Attributes:
        conv (nn.Conv2d): Convolutional layer to generate spatial attention
        sigmoid (nn.Sigmoid): Activation for attention weights
    """
    
    def __init__(self, kernel_size=7):
        """
        Initialize the Spatial Attention Module.
        
        Args:
            kernel_size (int, optional): Size of convolutional kernel. Defaults to 7.
                Larger kernels capture wider spatial context but increase computation.
                Common choices: 3, 5, 7
                
        Example:
            >>> sam = SpatialAttentionModule(kernel_size=7)
            >>> x = torch.randn(8, 512, 7, 7)
            >>> out = sam(x)  # Same shape, but with spatial attention applied
        """
        super(SpatialAttentionModule, self).__init__()
        
        # Calculate padding to maintain spatial dimensions
        padding = kernel_size // 2
        
        # Convolution to process concatenated spatial statistics
        # Input: 2 channels (avg + max), Output: 1 channel (attention map)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to the input feature map.
        
        Process:
        1. Compute average across channels
        2. Compute max across channels
        3. Concatenate the two maps
        4. Apply convolution to generate spatial attention map
        5. Apply sigmoid to get attention weights in [0, 1]
        6. Multiply with original input (element-wise)
        
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W)
                
        Returns:
            torch.Tensor: Spatially-attended feature map with same shape as input
                Important spatial locations are emphasized
        """
        # Average pooling across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Max pooling across channel dimension
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate spatial statistics
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Generate spatial attention map
        x_att = self.conv(x_cat)  # (B, 1, H, W)
        
        # Apply attention weights to input
        return self.sigmoid(x_att) * x  # (B, C, H, W)

class DANet(nn.Module):
    """
    Dual Attention Network (DANet) - Single backbone with attention mechanisms.
    
    This network implements a CNN backbone with both channel and spatial attention
    mechanisms. It progressively reduces spatial dimensions while increasing channel
    depth, then applies dual attention before classification.
    
    Architecture:
        Input → Conv1 → BN → ReLU → MaxPool →
        Layer1 (64→128) → Layer2 (128→256) → Layer3 (256→512) →
        Channel Attention → Spatial Attention →
        Global AvgPool → Dropout → FC
    
    Attributes:
        conv1 (nn.Conv2d): Initial convolution layer
        bn1 (nn.BatchNorm2d): Batch normalization
        relu (nn.ReLU): Activation function
        maxpool (nn.MaxPool2d): Max pooling layer
        layer1, layer2, layer3: Feature extraction blocks
        cam (ChannelAttentionModule): Channel attention
        sam (SpatialAttentionModule): Spatial attention
        avgpool (nn.AdaptiveAvgPool2d): Global average pooling
        dropout (nn.Dropout): Dropout for regularization
        fc (nn.Linear): Final classification layer
    """
    
    def __init__(self, num_inputs=1, num_classes=2, dropout_rate=0.3):
        """
        Initialize the DANet model.
        
        Args:
            num_inputs (int, optional): Number of input channels. Defaults to 1.
                For telescope images: 1 for charge only, 2 for charge+timing
            num_classes (int, optional): Number of output classes. Defaults to 2.
                For particle classification: 2 (gamma vs proton)
                For regression: 1
            dropout_rate (float, optional): Dropout probability. Defaults to 0.3.
                Higher values = more regularization but may underfit
        """
        super(DANet, self).__init__()
        
        # Initial convolution: large kernel for receptive field
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Progressive feature extraction layers
        self.layer1 = self._make_layer(64, 128, 2)   # 64 → 128 channels
        self.layer2 = self._make_layer(128, 256, 2)  # 128 → 256 channels
        self.layer3 = self._make_layer(256, 512, 2)  # 256 → 512 channels
        
        # Dual attention modules
        self.cam = ChannelAttentionModule(512)  # Channel attention on 512 channels
        self.sam = SpatialAttentionModule()     # Spatial attention
        
        # Final classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling to (B, 512, 1, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        """
        Create a sequence of convolutional blocks.
        
        Each block consists of: Conv → BatchNorm → ReLU
        The first block changes channel dimension, subsequent blocks maintain it.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            blocks (int): Number of convolutional blocks
            
        Returns:
            nn.Sequential: Sequential container of conv blocks
        """
        layers = []
        # First block: change channel dimension
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks: maintain channel dimension
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the DANet.
        
        Args:
            x (torch.Tensor): Input tensor with shape (B, C_in, H, W)
                B: batch size, C_in: input channels (1 or 2)
                H, W: image dimensions (typically 120x120)
                
        Returns:
            torch.Tensor: Output predictions with shape (B, num_classes)
                For classification: logits (pre-softmax)
                For regression: predicted values
        """
        # Initial convolution and downsampling
        x = self.conv1(x)      # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # (B, 64, H/4, W/4)
        
        # Feature extraction layers
        x = self.layer1(x)     # (B, 128, H/4, W/4)
        x = self.layer2(x)     # (B, 256, H/4, W/4)
        x = self.layer3(x)     # (B, 512, H/4, W/4)
        
        # Apply dual attention mechanisms with residual connections
        x = self.cam(x) + x    # Channel attention + residual
        x = self.sam(x) + x    # Spatial attention + residual
        
        # Global pooling and classification
        x = self.avgpool(x)    # (B, 512, 1, 1)
        x = torch.flatten(x, 1) # (B, 512)
        x = self.dropout(x)    # Regularization
        x = self.fc(x)         # (B, num_classes)

        return x
    
class DBBDanet(nn.Module):
    """
    Dual-Backbone DANet for multi-modal telescope data processing.
    
    This architecture uses two separate DANet backbones to process different
    input modalities (e.g., charge and timing information) independently,
    then fuses their features for final prediction. This allows each backbone
    to specialize in extracting relevant features from its input modality.
    
    Fusion Strategies:
        - Concatenation: Preserves all information but doubles feature dimension
        - Addition: Reduces dimension but may lose information
    
    Attributes:
        task (str): Task type ('type', 'energy', 'direction')
        use_concat (bool): Whether to concatenate or add backbone outputs
        backbone_1 (DANet): First backbone for primary input (charge)
        backbone_2 (DANet): Second backbone for secondary input (timing)
        fc (nn.Linear): Final classification/regression layer
        dropout (nn.Dropout): Dropout for regularization
    """
    
    def __init__(self, task, num_inputs=1, num_classes=2, use_concat=False, dropout_rate=0.3):
        """
        Initialize the Dual-Backbone DANet.
        
        Args:
            task (str): Task type to perform
                Options: 'type' (classification), 'energy' (regression), 'direction' (regression)
            num_inputs (int, optional): Number of input channels per backbone. Defaults to 1.
                Typically 1 (grayscale images)
            num_classes (int, optional): Number of output classes/values. Defaults to 2.
                For 'type': 2 (gamma, proton)
                For 'energy': 1 (energy value)
                For 'direction': 2 or 3 (angular coordinates)
            use_concat (bool, optional): Whether to concatenate backbone outputs.
                Defaults to False (use addition instead)
                True: More parameters, preserves all information
                False: Fewer parameters, may lose some information
            dropout_rate (float, optional): Dropout probability. Defaults to 0.3.
                
        Example:
            >>> # For particle classification with concatenation
            >>> model = DBBDanet(task='type', num_inputs=1, num_classes=2, use_concat=True)
            >>> 
            >>> # For energy regression with addition
            >>> model = DBBDanet(task='energy', num_inputs=1, num_classes=1, use_concat=False)
        """
        super(DBBDanet, self).__init__()    

        self.task = task
        self.use_concat = use_concat
        
        # Initialize two separate DANet backbones
        self.backbone_1 = DANet(num_inputs=num_inputs, num_classes=num_classes, dropout_rate=dropout_rate)
        self.backbone_2 = DANet(num_inputs=num_inputs, num_classes=num_classes, dropout_rate=dropout_rate)

        # Get feature dimension from backbone
        num_features = self.backbone_1.fc.in_features  # Typically 512
        
        # Adjust feature dimension based on fusion strategy
        if self.use_concat:
            num_features *= 2  # Double if concatenating

        # Create new final layers
        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)

        # Remove original final layers from backbones (use as feature extractors)
        self.backbone_1.fc = nn.Identity()
        self.backbone_1.dropout = nn.Identity()

        self.backbone_2.fc = nn.Identity()
        self.backbone_2.dropout = nn.Identity()

    def forward(self, x):
        if x.shape[1] >= 2:
            x, y = torch.split(x, [1, x.shape[1]-1], dim=1)
        else:
            y = x
        """
        Forward pass through the dual-backbone network.
        
        Process:
        1. Extract features from both inputs using separate backbones
        2. Fuse features (concatenate or add)
        3. Apply dropout and final classification/regression layer
        4. Return task-specific output
        
        Args:
            x (torch.Tensor): First input (charge image) with shape (B, C, H, W)
            y (torch.Tensor): Second input (timing image) with shape (B, C, H, W)
                Both inputs should have the same spatial dimensions
                
        Returns:
            tuple: (classification, energy, direction) where:
                - classification: Class logits if task=='type', else None
                - energy: Energy prediction if task=='energy', else None
                - direction: Direction prediction if task=='direction', else None
                Only one of the three is non-None based on self.task
                
        Note:
            The dual-backbone design allows the model to learn separate
            representations for different input modalities before fusion,
            which can be more effective than early fusion approaches.
        """
        # Initialize outputs (only one will be non-None)
        energy = None
        classification = None
        direction = None

        # Extract features from both backbones independently
        feature_1 = self.backbone_1(x)  # Features from charge image
        feature_2 = self.backbone_2(y)  # Features from timing image

        # Fuse features based on selected strategy
        if self.use_concat:
            # Concatenate: Preserves all information
            out = torch.cat((feature_1, feature_2), dim=1)  # (B, 2*num_features)
        else:
            # Addition: Element-wise fusion
            out = feature_1 + feature_2  # (B, num_features)

        # Apply regularization and final layer
        out = self.dropout(out)
        out = self.fc(out)  # (B, num_classes)

        # Assign output based on task type
        if self.task == "type":
            classification = out  # Classification logits
        elif self.task == "energy":
            energy = out  # Energy prediction
        elif self.task == "direction":
            direction = out  # Direction prediction

        return classification, energy, direction