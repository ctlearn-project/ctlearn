"""
Dual-Backbone EfficientNet Model Module

This module implements a dual-backbone architecture using EfficientNet models
as feature extractors for processing Cherenkov telescope data. EfficientNet
is a family of efficient neural networks that achieve state-of-the-art accuracy
with fewer parameters through compound scaling.

The dual-backbone approach processes charge and timing information independently
before fusing features for final predictions, with specialized heads for different tasks.

Classes:
    MemoryEfficientSwish: Memory-efficient Swish activation function
    SEBlock: Squeeze-and-Excitation attention block
    DoubleBBEfficientNet: Dual-backbone EfficientNet for multi-task learning

References:
    - EfficientNet: "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019)
    - Squeeze-and-Excitation: "Squeeze-and-Excitation Networks" (CVPR 2018)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from ctlearn.core.pytorch.nets.models.EffientNet_pytorch.model import EfficientNet

class MemoryEfficientSwish(nn.Module):
    """
    Memory-efficient implementation of Swish activation function.
    
    Swish (also known as SiLU - Sigmoid Linear Unit) is a smooth, non-monotonic
    activation function defined as: f(x) = x · σ(x) where σ is the sigmoid function.
    
    This implementation is memory-efficient because it doesn't store intermediate
    values during the forward pass, reducing memory consumption during backpropagation.
    
    Mathematical Formula:
        Swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    
    Properties:
        - Smooth and continuously differentiable
        - Non-monotonic (can decrease for negative inputs)
        - Bounded below (approaches 0 for large negative x)
        - Unbounded above (approaches x for large positive x)
        - Self-gated: combines input with its sigmoid
    
    Example:
        >>> swish = MemoryEfficientSwish()
        >>> x = torch.randn(32, 128)
        >>> output = swish(x)
    """
    
    def forward(self, x):
        """
        Apply Swish activation element-wise.
        
        Args:
            x (torch.Tensor): Input tensor of any shape
            
        Returns:
            torch.Tensor: Activated tensor with same shape as input
        """
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    
    This module implements channel-wise attention that adaptively recalibrates
    channel-wise feature responses by explicitly modeling interdependencies
    between channels. It "squeezes" global spatial information into a channel
    descriptor and "excites" channels through a gating mechanism.
    
    Architecture:
        Input → Global Avg Pool → FC (reduce) → PReLU → FC (expand) → Swish → Scale Input
    
    The SE block improves representational power by allowing the network to
    emphasize informative features and suppress less useful ones.
    
    Attributes:
        avg_pool (nn.AdaptiveAvgPool2d): Global average pooling layer
        fc (nn.Sequential): Two-layer MLP for channel attention
            - First layer: Channel reduction (compression)
            - Second layer: Channel expansion (excitation)
    """
    
    def __init__(self, channel, reduction=16):
        """
        Initialize the Squeeze-and-Excitation block.
        
        Args:
            channel (int): Number of input channels
            reduction (int, optional): Reduction ratio for the bottleneck.
                Defaults to 16. Higher values reduce parameters but may lose information.
                Common values: 4, 8, 16
                
        Example:
            >>> se_block = SEBlock(channel=512, reduction=16)
            >>> x = torch.randn(8, 512, 7, 7)
            >>> out = se_block(x)  # Same shape as input
        """
        super(SEBlock, self).__init__()
        
        # Global average pooling: (B, C, H, W) → (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Two-layer MLP for channel attention
        self.fc = nn.Sequential(
            # Squeeze: Reduce channel dimension
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),  # Parametric ReLU activation
            # Excitation: Restore channel dimension
            nn.Linear(channel // reduction, channel, bias=False),
            MemoryEfficientSwish()  # Final activation for attention weights
        )

    def forward(self, x):
        """
        Apply channel attention to input feature map.
        
        Process:
        1. Global average pool to get channel statistics (B, C, H, W) → (B, C)
        2. Pass through MLP to get channel attention weights
        3. Reshape weights to (B, C, 1, 1)
        4. Scale input features by attention weights (element-wise multiplication)
        
        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W)
                B: batch size, C: channels, H: height, W: width
                
        Returns:
            torch.Tensor: Attention-weighted feature map with same shape as input
                Important channels are emphasized, less important ones suppressed
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)  # (B, C, 1, 1) → (B, C)
        
        # Excitation: Learn channel attention weights
        y = self.fc(y).view(b, c, 1, 1)  # (B, C) → (B, C, 1, 1)
        
        # Scale input by attention weights
        return x * y.expand_as(x)

class DoubleBBEfficientNet(nn.Module):
    """
    Dual-Backbone EfficientNet for multi-task telescope data analysis.
    
    This architecture uses two pretrained EfficientNet backbones to process
    different input modalities (charge and timing images) independently,
    then fuses their features for task-specific predictions. Supports multiple
    EfficientNet variants (B0-B7) and different tasks (classification, regression).
    
    Architecture Overview:
        Input_1 (charge) → EfficientNet_1 ↘
                                            → Fusion → Task-specific Head → Output
        Input_2 (timing) → EfficientNet_2 ↗
    
    Fusion Strategy:
        - Addition fusion for computational efficiency
        - 1x1 convolution for feature refinement
        - Global average pooling for spatial reduction
    
    Task-Specific Heads:
        - Classification: Two-layer MLP with Swish activation
        - Energy: Dual-head (classification + regression)
        - Direction: Two-layer MLP with dropout
    
    Attributes:
        task (str): Task type ('type', 'energy', 'direction')
        num_outputs (int): Number of output values
        backbone1, backbone2 (EfficientNet): Feature extraction backbones
        fusion_conv (nn.Conv2d): 1x1 conv for feature fusion
        Task-specific layers (fc_*, prelu_*, dropout_*, etc.)
    """
    
    def __init__(self, model_variant: str = "efficientnet-b3", task: str = "Energy", 
                 num_outputs=2, device_str="cuda", energy_bins=None):
        """
        Initialize the Dual-Backbone EfficientNet model.
        
        Args:
            model_variant (str, optional): EfficientNet variant to use.
                Defaults to "efficientnet-b3"
                Options: 'efficientnet-b0' to 'efficientnet-b7'
                Larger models (b5, b7) have more parameters and accuracy
                
            task (str, optional): Task type. Defaults to "Energy"
                Options: 'type' (classification), 'energy' (regression), 'direction'
                
            num_outputs (int, optional): Number of output values. Defaults to 2
                For 'type': 2 (gamma vs proton)
                For 'energy': Variable (classification bins + regression)
                For 'direction': 2 or 3 (angular coordinates)
                
            device_str (str, optional): Device string. Defaults to "cuda"
                
            energy_bins (list or None, optional): Energy bin edges for classification.
                Defaults to None. Used only for energy task with dual-head approach.
                
        Raises:
            ValueError: If model_variant is not tested/supported
            
        Example:
            >>> # Classification with EfficientNet-B3
            >>> model = DoubleBBEfficientNet(
            ...     model_variant='efficientnet-b3',
            ...     task='type',
            ...     num_outputs=2
            ... )
            
            >>> # Energy regression with larger model
            >>> model = DoubleBBEfficientNet(
            ...     model_variant='efficientnet-b5',
            ...     task='energy',
            ...     num_outputs=10
            ... )
        """
        super(DoubleBBEfficientNet, self).__init__()
        
        # Store configuration
        self.task = task.lower()
        self.num_outputs = num_outputs
        self.energy_bins = energy_bins
        self.device = torch.device(device_str)
        
        # Define architecture parameters based on model variant
        hidden_size = 512
        if 'b3' in model_variant:
            feature_size = 1536  # EfficientNet-B3 feature dimension
        elif 'b5' in model_variant: 
            feature_size = 2048  # EfficientNet-B5 feature dimension
        else:
            raise ValueError(f"Model variant {model_variant} not tested. Adapt the feature_size.")
        
        # Initialize backbones based on task
        if self.task == "type":
            # Classification task: use batch normalization
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs, use_batch_norm=True
            )
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs, use_batch_norm=True
            )
            
        elif self.task == "energy":
            # Energy regression task
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs
            )
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs
            )
            
        elif self.task == "direction":
            # Direction reconstruction task
            self.backbone1 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs
            )
            self.backbone2 = EfficientNet.from_pretrained(
                model_variant, in_channels=1, num_classes=num_outputs
            )
        
        # Fusion module: 1x1 convolution to refine fused features
        self.fusion_conv = nn.Conv2d(
            in_channels=feature_size, 
            out_channels=feature_size, 
            kernel_size=1
        )
        
        # Task-specific heads
        if self.task == "type":
            # Classification head: two-layer MLP
            self.fc_classification_1 = nn.Linear(feature_size, hidden_size)
            self.fc_classification_2 = nn.Linear(hidden_size, num_outputs)

        if self.task == "energy":
            # Dual-head architecture for energy prediction
            # Head 1: Energy range classification
            self.fc_energy_1 = nn.Linear(feature_size, hidden_size)
            self.fc_energy_2 = nn.Linear(hidden_size, int(num_outputs - 1))
            
            # Head 2: Fine-grained regression
            self.fc_energy_1_reg = nn.Linear(feature_size, hidden_size)
            self.fc_energy_2_reg = nn.Linear(hidden_size, 1)
            
        if self.task == "direction":
            # Direction head with dropout
            self.fc_direction_1 = nn.Linear(feature_size, hidden_size)
            self.fc_direction_2 = nn.Linear(hidden_size, num_outputs)
        
        # Activation and regularization layers
        self.swish = MemoryEfficientSwish()
        self.batch_norm = nn.BatchNorm1d(feature_size)
        
        # Task-specific dropout
        self.dropout_energy_1 = nn.Dropout(0.1)        
        self.dropout_energy_2 = nn.Dropout(0.3)
        self.dropout_direction = nn.Dropout(0.3)
        
        # Parametric activations
        self.prelu_direction = nn.PReLU(num_parameters=hidden_size)
        self.prelu_energy_1 = nn.PReLU(num_parameters=hidden_size)
        self.prelu_energy_2 = nn.PReLU()
           
    def extract_feature_vector(self, x1, x2):
        """
        Extract and fuse features from both backbones.
        
        This method processes both inputs through separate EfficientNet backbones,
        fuses the features, and produces a compact feature vector for final prediction.
        
        Process:
        1. Extract features from both inputs using separate backbones
        2. Fuse features using element-wise addition
        3. Refine fused features with 1x1 convolution
        4. Apply global average pooling
        5. Flatten to feature vector
        
        Args:
            x1 (torch.Tensor): First input (charge image)
                Shape: (batch_size, 1, height, width)
            x2 (torch.Tensor): Second input (timing image)
                Shape: (batch_size, 1, height, width)
                
        Returns:
            torch.Tensor: Fused feature vector
                Shape: (batch_size, feature_size)
                
        Note:
            Addition fusion is used for computational efficiency.
            Alternative fusion strategies (concatenation, weighted sum) are possible.
        """
        # Extract features from both backbones
        x1 = self.backbone1.extract_features(x1)
        x2 = self.backbone2.extract_features(x2)

        # Fusion: Element-wise addition
        # Alternative: torch.cat((x1, x2), dim=1) for concatenation
        fused_features = torch.add(x1, x2)
        
        # Refine fused features with 1x1 convolution
        fused_features = self.fusion_conv(fused_features)
        
        # Global average pooling: (B, C, H, W) → (B, C, 1, 1)
        fused_features = F.adaptive_avg_pool2d(fused_features, 1)
        
        # Flatten: (B, C, 1, 1) → (B, C)
        fused_features = fused_features.view(fused_features.size(0), -1)

        return fused_features
    
    def forward(self, x1, x2):
        """
        Forward pass through the dual-backbone network.
        
        Process:
        1. Extract and fuse features from both inputs
        2. Apply normalization and dropout (for energy/direction)
        3. Pass through task-specific prediction head
        4. Return predictions in standardized format
        
        Args:
            x1 (torch.Tensor): First input (charge image)
                Shape: (batch_size, 1, height, width)
            x2 (torch.Tensor): Second input (timing image)
                Shape: (batch_size, 1, height, width)
                
        Returns:
            tuple: (classification, energy, direction) where:
                - classification: For 'type' task
                    [logits, features] where logits: (batch_size, 2)
                - energy: For 'energy' task
                    [class_logits, regression_value]
                    class_logits: (batch_size, num_bins-1)
                    regression_value: (batch_size, 1)
                - direction: For 'direction' task
                    [predictions, features] where predictions: (batch_size, num_outputs)
                    
                Only one of the three is non-None based on self.task
                
        Task-Specific Processing:
            Classification (type):
                - Two-layer MLP with Swish activation
                - Returns logits and feature vector
                
            Energy (energy):
                - Dual-head architecture
                - Classification head: Predicts energy range/bin
                - Regression head: Predicts fine-grained value
                - Combines coarse and fine predictions
                
            Direction (direction):
                - Two-layer MLP with PReLU and dropout
                - Predicts angular offsets or coordinates
        """
        # Initialize outputs
        energy = [None, None]
        classification = [None, None]
        direction = [None, None]

        # Extract fused features from both backbones
        fused_features = self.extract_feature_vector(x1, x2)

        # Task-specific prediction heads
        if self.task == "type":
            # Classification head
            classification = self.fc_classification_1(fused_features)
            classification = self.fc_classification_2(classification)
            classification = self.swish(classification)
            
        if self.task == "energy":
            # Apply normalization and dropout
            fused_features = self.batch_norm(fused_features)
            fused_features = self.dropout_energy_1(fused_features)
            
            # Classification head: Predict energy bin
            energy_class = self.fc_energy_1(fused_features)
            energy_class = self.fc_energy_2(energy_class)
            energy_class = self.swish(energy_class)
            energy_pred_class = self.swish(energy_class)
            
            # Regression head: Predict fine-grained energy
            energy_reg = self.fc_energy_1_reg(fused_features)
            energy_reg = self.dropout_energy_2(energy_reg)
            energy_reg = self.prelu_energy_1(energy_reg)
            energy_regresion = self.fc_energy_2_reg(energy_reg)

            # Combine both predictions
            energy = [energy_pred_class, energy_regresion]

        if self.task == "direction":
            # Direction head with dropout
            fused_features = self.dropout_direction(fused_features)
            direction = self.fc_direction_1(fused_features)
            direction = self.fc_direction_2(direction)

        # Return in standardized format
        return [classification, fused_features], energy, direction

    def eval(self):
        """
        Set the model to evaluation mode.
        
        Overrides the default eval() to ensure both backbones are also
        set to evaluation mode. This is important for proper handling of
        batch normalization and dropout layers.
        """
        super().eval()  
        self.backbone1.eval()
        self.backbone2.eval()

    def train(self, mode=True):
        """
        Set the model to training or evaluation mode.
        
        Overrides the default train() to ensure both backbones follow
        the same mode. This ensures consistent behavior of batch normalization
        and dropout across all components.
        
        Args:
            mode (bool, optional): Whether to set training mode (True) or
                evaluation mode (False). Defaults to True.
        """
        super().train(mode) 
        self.backbone1.train(mode)
        self.backbone2.train(mode)
