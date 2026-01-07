"""
Dual-Backbone RegNet Model Module

This module implements a dual-backbone architecture using RegNet (Regularized Networks)
as feature extractors for processing Cherenkov telescope data. RegNet is a family of
efficient networks designed with design principles derived from network design spaces.

The dual-backbone approach allows processing charge and timing information independently
before fusing features for final predictions.

Classes:
    SingleChannelRegNet: Single RegNet backbone for one input modality
    DBBRegNet: Dual-backbone architecture combining two RegNet backbones

References:
    - "Designing Network Design Spaces" (Radosavovic et al., CVPR 2020)
    - RegNet: https://arxiv.org/abs/2003.13678
"""

import torch.nn.functional as F
import torch
from torchvision import models
import torch.nn as nn


class SingleChannelRegNet(nn.Module):
    """
    Single-channel RegNet backbone for feature extraction.
    
    This class wraps a pretrained RegNet model from torchvision and adapts it
    for single-channel telescope images. The first convolutional layer is modified
    to accept custom input channels, and the final layer is adjusted for the
    desired number of outputs.
    
    RegNet Architecture:
        - Stage-based design with regularized block patterns
        - Efficient depth and width distributions
        - Better accuracy-efficiency trade-offs than EfficientNet
        
    Available RegNet Variants:
        - regnet_y_400mf: ~4M parameters, 400MFLOPs
        - regnet_y_800mf: ~6M parameters, 800MFLOPs (default)
        - regnet_y_1_6gf: ~11M parameters, 1.6GFLOPs
        - regnet_x_16gf: ~54M parameters, 16GFLOPs
        
    Attributes:
        regnet (models.RegNet): Modified RegNet model from torchvision
    """
    
    def __init__(self, num_inputs=1, num_classes=2):
        """
        Initialize the single-channel RegNet backbone.
        
        Args:
            num_inputs (int, optional): Number of input channels. Defaults to 1.
                For telescope data: 1 (grayscale charge or timing image)
            num_classes (int, optional): Number of output classes/values. Defaults to 2.
                For classification: 2 (gamma vs proton)
                For regression: 1 (energy or direction components)
                
        Modifications:
            1. First conv layer: Modified to accept num_inputs channels
               Original: Conv2d(3, 32, ...) for RGB images
               Modified: Conv2d(num_inputs, 32, ...) for grayscale
               
            2. Final layer: Modified to output num_classes values
               Original: fc layer for ImageNet (1000 classes)
               Modified: fc layer for custom task
               
        Example:
            >>> # Create backbone for grayscale images, binary classification
            >>> backbone = SingleChannelRegNet(num_inputs=1, num_classes=2)
            >>> x = torch.randn(8, 1, 120, 120)  # Batch of 8 grayscale images
            >>> out = backbone(x)  # Shape: (8, 2)
        """
        super(SingleChannelRegNet, self).__init__()
        
        # Load pretrained RegNet-Y-800MF
        # RegNet-Y variants have squeeze-and-excitation (SE) blocks
        self.regnet = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
        
        # Alternative RegNet variants (commented out):
        # regnet_y_400mf: Smaller, faster, less accurate
        # self.regnet = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
        
        # regnet_x_16gf: Much larger, slower, potentially more accurate
        # self.regnet = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT)
        
        # regnet_x_1_6gf: Medium size without SE blocks
        # self.regnet = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.DEFAULT)
        
        # regnet_y_1_6gf: Medium size with SE blocks
        # self.regnet = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
        
        # Modify first convolutional layer for custom input channels
        # Original stem: Conv2d(3, 32) for RGB images
        # Modified stem: Conv2d(num_inputs, 32) for grayscale/custom
        self.regnet.stem[0] = nn.Conv2d(
            num_inputs, 32, 
            kernel_size=(3, 3), 
            stride=(2, 2), 
            padding=(1, 1), 
            bias=False
        )
        
        # Modify final fully connected layer for custom number of outputs
        num_features = self.regnet.fc.in_features  # Get size from pretrained model
        self.regnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        """
        Forward pass through RegNet.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_inputs, H, W)
                Typically (batch_size, 1, 120, 120) for telescope images
                
        Returns:
            torch.Tensor: Output predictions with shape (batch_size, num_classes)
                For classification: logits before softmax
                For regression: predicted values
                
        Architecture Flow:
            Input → Stem (Conv+BN+ReLU) →
            Stage 1 (depth=1) → Stage 2 (depth=3) →
            Stage 3 (depth=7) → Stage 4 (depth=12) →
            AvgPool → FC → Output
        """
        return self.regnet(x)
    
class DBBRegNet(nn.Module):
    """
    Dual-Backbone RegNet for multi-modal telescope data.
    
    This architecture uses two separate RegNet backbones to process different
    input modalities (e.g., charge and timing images) independently, then fuses
    their features for final prediction. This allows each backbone to specialize
    in extracting relevant features from its input modality.
    
    Fusion Strategies:
        - Concatenation (use_concat=True): Preserves all information
          Output features: 2 × num_features
          Pros: Retains distinct information from both streams
          Cons: Doubles parameter count in final layer
          
        - Addition (use_concat=False): Reduces dimension
          Output features: num_features
          Pros: Fewer parameters, forces feature alignment
          Cons: May lose complementary information
    
    Attributes:
        use_concat (bool): Whether to concatenate or add backbone outputs
        task (str): Task type ('type', 'energy', 'direction')
        bb1 (SingleChannelRegNet): First backbone for primary input
        bb2 (SingleChannelRegNet): Second backbone for secondary input
        dropout (nn.Dropout): Dropout layer for regularization
        fc (nn.Linear): Final classification/regression layer
    """
    
    def __init__(self, task, use_concat=False, num_inputs=1, num_classes=2, dropout_rate=0.1):
        """
        Initialize the Dual-Backbone RegNet.
        
        Args:
            task (str): Task type to perform
                Options: 'type' (classification), 'energy' (regression), 'direction' (regression)
                Case-insensitive, will be converted to lowercase
            use_concat (bool, optional): Whether to concatenate backbone outputs.
                Defaults to False (uses addition)
                True: Concatenate features (more parameters, preserves information)
                False: Add features (fewer parameters, forces alignment)
            num_inputs (int, optional): Number of input channels per backbone. Defaults to 1.
            num_classes (int, optional): Number of output classes/values. Defaults to 2.
                For 'type': 2 (gamma, proton)
                For 'energy': 1 (energy value)
                For 'direction': 2 or 3 (angular coordinates)
            dropout_rate (float, optional): Dropout probability. Defaults to 0.1.
                Applied before final layer for regularization
                
        Architecture:
            Input_1 → Backbone_1 ↘
                                   → Fusion → Dropout → FC → Output
            Input_2 → Backbone_2 ↗
            
        Example:
            >>> # Particle classification with concatenation
            >>> model = DBBRegNet(task='type', use_concat=True, num_classes=2)
            >>> charge = torch.randn(16, 1, 120, 120)
            >>> timing = torch.randn(16, 1, 120, 120)
            >>> cls, energy, direction = model(charge, timing)
            >>> print(cls.shape)  # torch.Size([16, 2])
            
            >>> # Energy regression with addition
            >>> model = DBBRegNet(task='energy', use_concat=False, num_classes=1)
            >>> cls, energy, direction = model(charge, timing)
            >>> print(energy.shape)  # torch.Size([16, 1])
        """
        super(DBBRegNet, self).__init__()
        
        # Store configuration
        self.use_concat = use_concat
        self.task = task.lower()  # Normalize to lowercase for consistency
        
        # Initialize two independent RegNet backbones
        self.bb1 = SingleChannelRegNet(num_inputs=num_inputs, num_classes=num_classes)
        self.bb2 = SingleChannelRegNet(num_inputs=num_inputs, num_classes=num_classes)

        # Get feature dimension from backbone
        num_features = self.bb1.regnet.fc.in_features
        
        # Adjust feature dimension based on fusion strategy
        if self.use_concat:
            num_features *= 2  # Double for concatenation

        # Remove original classification heads from backbones
        # Use backbones as feature extractors only
        self.bb1.regnet.fc = nn.Identity()  # Replace fc with identity (no-op)
        self.bb2.regnet.fc = nn.Identity()
        
        # Regularization layer
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        
        # Final task-specific prediction layer
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x, y):
        """
        Forward pass through the dual-backbone network.
        
        Process:
        1. Extract features from both inputs using separate backbones
        2. Fuse features (concatenate or add)
        3. Apply dropout for regularization
        4. Apply final layer for task-specific prediction
        5. Route output to appropriate task variable
        
        Args:
            x (torch.Tensor): First input (charge image) with shape (B, C, H, W)
                B: batch size, C: channels (typically 1), H/W: image dimensions
            y (torch.Tensor): Second input (timing image) with same shape as x
                
        Returns:
            tuple: (classification, energy, direction) where:
                - classification: Class logits if task=='type', else None
                  Shape: (batch_size, 2) for binary classification
                - energy: Energy prediction if task=='energy', else None
                  Shape: (batch_size, 1) for energy regression
                - direction: Direction prediction if task=='direction', else None
                  Shape: (batch_size, 2) or (batch_size, 3) for angular coordinates
                  
                Only one of the three is non-None based on self.task
                
        Feature Extraction Details:
            - Both backbones process their inputs independently
            - No gradient flow between backbones during feature extraction
            - Each backbone can learn modality-specific representations
            
        Fusion Details:
            Concatenation mode (use_concat=True):
                out = [features_1 | features_2]  # Shape: (B, 2F)
                Retains all information from both modalities
                
            Addition mode (use_concat=False):
                out = features_1 + features_2  # Shape: (B, F)
                Forces features to be in same space
                Acts as implicit alignment/fusion
                
        Example:
            >>> model = DBBRegNet(task='type', use_concat=True)
            >>> x = torch.randn(32, 1, 120, 120)  # Charge images
            >>> y = torch.randn(32, 1, 120, 120)  # Timing images
            >>> cls, energy, direction = model(x, y)
            >>> 
            >>> # Only classification is non-None
            >>> assert cls is not None
            >>> assert energy is None
            >>> assert direction is None
            >>> print(cls.shape)  # torch.Size([32, 2])
        """
        # Initialize outputs (only one will be non-None)
        energy = None
        classification = None
        direction = None

        # Extract features from both backbones independently
        feature_1 = self.bb1(x)  # Features from charge image
        feature_2 = self.bb2(y)  # Features from timing image

        # Fuse features based on selected strategy
        if self.use_concat:
            # Concatenate features along feature dimension
            # Preserves all information from both modalities
            out = torch.cat((feature_1, feature_2), dim=1)  # Shape: (B, 2F)
        else:
            # Element-wise addition of features
            # Forces features into shared representation space
            out = feature_1 + feature_2  # Shape: (B, F)
        
        # Apply dropout for regularization
        out = self.dropout(out)
        
        # Apply final prediction layer
        out = self.fc(out)  # Shape: (B, num_classes)
         
        # Route output to appropriate task-specific variable
        if self.task == "type":
            classification = out  # Classification logits
        elif self.task == "energy":
            energy = out  # Energy predictions
        elif self.task == "direction":
            direction = out  # Direction predictions

        # Return tuple format (for compatibility with multi-task frameworks)
        return classification, energy, direction