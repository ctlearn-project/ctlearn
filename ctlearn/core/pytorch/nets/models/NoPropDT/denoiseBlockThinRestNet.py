"""
Denoising Block with Thin ResNet Architecture Module

This module implements a denoising block architecture combining residual networks
with denoising capabilities for diffusion-based models. It's specifically designed
for the NoProp-DT (No-Propagation Denoising Transformer) model in CTLearn.

The architecture uses a lightweight ResNet for feature extraction from images,
combined with MLP layers for processing noisy embeddings, making it suitable
for progressive denoising in diffusion models.

Classes:
    AdaptiveBatchNorm2d: Adaptive batch normalization with learnable parameters
    ResidualBlock: Residual block with batch normalization and skip connections
    DenoiseBlock: Main denoising block combining image and embedding features
    MemoryEfficientSwish: Memory-efficient Swish activation function

References:
    - "Deep Residual Learning for Image Recognition" (He et al., CVPR 2016)
    - "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
"""

import torch
from torch import nn
import torch.nn.functional as F

class MemoryEfficientSwish(nn.Module):
    """
    Memory-efficient implementation of Swish activation function.
    
    Swish (also known as SiLU - Sigmoid Linear Unit) is defined as:
    f(x) = x · sigmoid(x)
    
    This implementation avoids storing intermediate activations during
    forward pass, reducing memory consumption during backpropagation.
    
    Properties:
        - Smooth and continuously differentiable
        - Non-monotonic (can decrease for negative inputs)
        - Self-gated: combines input with its sigmoid
        - Better gradient flow than ReLU in some cases
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

class AdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive Batch Normalization with learnable interpolation parameters.
    
    This layer combines the input with batch-normalized input using learnable
    parameters 'a' and 'b'. This allows the network to learn how much to rely
    on the original input versus the normalized version.
    
    Mathematical Formula:
        output = a * x + b * BatchNorm(x)
        
        where a and b are learnable scalar parameters initialized to 1 and 0
        respectively, so initially: output = x (identity mapping).
    
    Attributes:
        bn (nn.BatchNorm2d): Standard batch normalization layer
        a (nn.Parameter): Learnable weight for original input (initialized to 1)
        b (nn.Parameter): Learnable weight for normalized input (initialized to 0)
    
    Benefits:
        - Allows network to choose between normalized and unnormalized features
        - Can help with training stability in diffusion models
        - Provides more flexibility than standard batch normalization
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):
        """
        Initialize Adaptive Batch Normalization.
        
        Args:
            num_features (int): Number of channels in the input
            eps (float, optional): Small value for numerical stability. Defaults to 1e-5
            momentum (float, optional): Momentum for running statistics. Defaults to 0.5
                Note: Higher momentum (0.5 vs typical 0.1) gives more weight to current batch
            affine (bool, optional): Whether to learn affine parameters. Defaults to True
        """
        super(AdaptiveBatchNorm2d, self).__init__()
        
        # Standard batch normalization
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        
        # Learnable interpolation parameters
        # a=1, b=0 initially makes this an identity mapping
        self.a = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        """
        Apply adaptive batch normalization.
        
        Process:
        1. Apply standard batch normalization to input
        2. Combine original input and normalized input using learnable weights
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Adaptively normalized tensor with same shape as input
                output = a * x + b * BatchNorm(x)
        """
        return self.a * x + self.b * self.bn(x)

class ResidualBlock(nn.Module):
    """
    Residual block with adaptive batch normalization and skip connections.
    
    This block implements the core building block of ResNet architectures,
    with two convolutional layers, adaptive batch normalization, and a
    residual skip connection. If input and output channels differ, a 1x1
    convolution adjusts the skip connection.
    
    Architecture:
        Input → [Conv3x3 → AdaptiveBN → ReLU → Conv3x3 → AdaptiveBN] → (+) → ReLU → Output
                                                                          ↑
                                                                          |
                                                                    Skip Connection
                                                                    (1x1 conv if needed)
    
    Attributes:
        conv_block (nn.Sequential): Main convolutional path with two conv layers
        shortcut (nn.Sequential): Skip connection (identity or 1x1 conv)
        relu (nn.ReLU): Final activation function
    """
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        
        # Main convolutional path
        self.conv_block = nn.Sequential(
            # First convolution: maintains spatial dimensions with padding
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveBatchNorm2d(out_channels),
            nn.ReLU(),
            # Second convolution: refines features
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveBatchNorm2d(out_channels)
        )

        # Skip connection: adjust channels if needed
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # Use 1x1 convolution to match channel dimensions
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        # Final activation after adding residual
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Process:
        1. Pass input through main convolutional path
        2. Add skip connection (possibly adjusted with 1x1 conv)
        3. Apply final ReLU activation
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_channels, H, W)
                
        Benefits of Residual Connection:
            - Helps gradient flow during backpropagation
            - Allows learning identity mappings when beneficial
            - Enables training of very deep networks
        """
        # Main path + skip connection + activation
        return self.relu(self.conv_block(x) + self.shortcut(x))


class DenoiseBlock(nn.Module):
    """
    Denoising block combining image features and noisy embeddings.
    
    This block is the core component of the NoProp-DT diffusion model. It processes
    telescope images through a lightweight ResNet while simultaneously processing
    noisy class embeddings through MLPs, then combines both to produce a denoised
    embedding and class logits.
    
    Architecture Overview:
        Image → ThinResNet → Image Features (256-dim)
                                                        ↘
        Noisy Embedding → MLP (with residual) → Embedding Features (256-dim) → Concat → MLP → Logits
                                                                                                  ↓
                                                                                    z_next = z_prev + logits @ W_embed
    
    The ThinResNet Path:
        Input (1, H, W) →
        ResBlock(1→32) → MaxPool → Dropout →
        ResBlock(32→64) → MaxPool → Dropout →
        ResBlock(64→128) → AdaptiveAvgPool →
        Flatten → Linear(128→256)
    
    The Embedding Path (with residual):
        z_prev (embedding_dim) →
        Linear(→256) → BN → ReLU → h1 →
        Linear(→256) → BN → ReLU → h2 →
        Linear(→256) → BN → h3 →
        z_feat = h3 + h1 (residual)
    
    The Fusion Path:
        Concat(image_feat, z_feat) →
        Linear(512→256) → BN → ReLU →
        Linear(256→128) → BN → ReLU →
        Linear(128→num_classes) → logits
    
    Attributes:
        use_softmax (bool): Whether to apply softmax to logits
        conv_path (nn.Sequential): CNN for image feature extraction
        fc_z1, fc_z2, fc_z3 (nn.Linear): MLP layers for embedding processing
        bn_z1, bn_z2, bn_z3 (nn.BatchNorm1d): Batch norms for embedding MLP
        fc_f1, fc_f2 (nn.Linear): MLP layers for fusing features
        bn_f1, bn_f2 (nn.BatchNorm1d): Batch norms for fusion MLP
        fc_out (nn.Linear): Output layer producing class logits
    """
    
    def __init__(self, embedding_dim, num_classes, use_softmax=False, num_channels=1):
        """
        Initialize the denoising block.
        
        Args:
            embedding_dim (int): Dimension of class embeddings
                Typical values: 128, 256
            num_classes (int): Number of output classes
                For telescope data: 2 (gamma vs proton)
            use_softmax (bool, optional): Whether to apply softmax to logits.
                Defaults to False. Set to True for probability outputs
            num_channels (int, optional): Number of input image channels.
                Defaults to 1 (grayscale telescope images)
        """
        super().__init__()
        self.use_softmax = use_softmax
        
        # ThinResNet convolutional path for image feature extraction
        # Progressively increases channels: 1 → 32 → 64 → 128
        # Reduces spatial dimensions: H,W → H/2,W/2 → H/4,W/4 → 1,1
        self.conv_path = nn.Sequential(
            # Stage 1: Initial feature extraction
            ResidualBlock(num_channels, 32),
            nn.MaxPool2d(2),  # Downsample by 2x
            nn.Dropout(0.2),  # Regularization
            
            # Stage 2: Mid-level features
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),  # Downsample by 2x
            nn.Dropout(0.2),
            
            # Stage 3: High-level features
            ResidualBlock(64, 128),
            
            # Global pooling and projection
            nn.AdaptiveAvgPool2d((1, 1)),  # Reduce to (batch, 128, 1, 1)
            nn.Flatten(),  # (batch, 128)
            nn.Linear(128, 256),  # Project to 256-dim
        )

        # MLP for processing noisy embedding with residual connection
        # Three-layer network with skip connection from first to last layer
        self.fc_z1 = nn.Linear(embedding_dim, 256)
        self.bn_z1 = nn.BatchNorm1d(256)

        self.fc_z2 = nn.Linear(256, 256)
        self.bn_z2 = nn.BatchNorm1d(256)

        self.fc_z3 = nn.Linear(256, 256)
        self.bn_z3 = nn.BatchNorm1d(256)

        # MLP for combining image and embedding features
        # Fuses 512-dim (256 + 256) down to num_classes
        self.fc_f1 = nn.Linear(256 + 256, 256)  # Concatenated features
        self.bn_f1 = nn.BatchNorm1d(256)
        
        self.fc_f2 = nn.Linear(256, 128)
        self.bn_f2 = nn.BatchNorm1d(128)
        
        self.fc_out = nn.Linear(128, num_classes)  # Final logits

    def forward(self, x, z_prev, W_embed):
        """
        Forward pass through the denoising block.
        
        This method combines image features with noisy embeddings to produce
        both denoised embeddings and class predictions. The denoising happens
        through the interaction between image features and the noisy embedding.
        
        Process:
        1. Extract features from image using ThinResNet
        2. Process noisy embedding through MLP with residual connection
        3. Concatenate image and embedding features
        4. Generate class logits through fusion MLP
        5. Update embedding: z_next = z_prev + logits @ W_embed
        
        Args:
            x (torch.Tensor): Input images
                Shape: (batch_size, num_channels, height, width)
                Typically: (batch_size, 1, 120, 120) for telescope images
                
            z_prev (torch.Tensor): Previous/current noisy embedding
                Shape: (batch_size, embedding_dim)
                Contains noise that will be reduced in this step
                
            W_embed (torch.Tensor): Class embedding matrix
                Shape: (num_classes, embedding_dim)
                Each row is the embedding for one class
                
        Returns:
            tuple: (z_next, logits) where:
                - z_next (torch.Tensor): Denoised embedding
                    Shape: (batch_size, embedding_dim)
                    Cleaner than z_prev, closer to true class embedding
                - logits (torch.Tensor): Class predictions
                    Shape: (batch_size, num_classes)
                    If use_softmax=True: softmax probabilities
                    If use_softmax=False: raw logits
                    
        Denoising Mechanism:
            The update rule z_next = z_prev + logits @ W_embed works by:
            1. logits indicate which class is most likely
            2. W_embed provides the ideal embedding for each class
            3. The product logits @ W_embed pulls z_prev toward the correct class embedding
            4. Over multiple diffusion steps, this progressively cleans the embedding
            
        Example:
            >>> block = DenoiseBlock(embedding_dim=128, num_classes=2)
            >>> x = torch.randn(32, 1, 120, 120)  # Batch of 32 images
            >>> z_prev = torch.randn(32, 128)  # Noisy embeddings
            >>> W_embed = torch.randn(2, 128)  # Class embeddings
            >>> z_next, logits = block(x, z_prev, W_embed)
            >>> print(z_next.shape)  # torch.Size([32, 128])
            >>> print(logits.shape)  # torch.Size([32, 2])
        """
        # Step 1: Extract features from input image
        # Shape: (batch_size, 256)
        x_feat = self.conv_path(x)

        # Step 2: Process noisy embedding through MLP with residual connection
        # First layer
        h1 = F.relu(self.bn_z1(self.fc_z1(z_prev)))  # (batch_size, 256)
        
        # Second layer
        h2 = F.relu(self.bn_z2(self.fc_z2(h1)))  # (batch_size, 256)
        
        # Third layer (no ReLU here)
        h3 = self.bn_z3(self.fc_z3(h2))  # (batch_size, 256)

        # Residual connection: add h1 to h3
        # This helps gradient flow and preserves early features
        z_feat = h3 + h1  # (batch_size, 256)

        # Step 3: Concatenate image and embedding features
        # Combines information from both modalities
        h_f = torch.cat([x_feat, z_feat], dim=1)  # (batch_size, 512)

        # Step 4: Process combined features through fusion MLP
        # First fusion layer
        h_f = F.relu(self.bn_f1(self.fc_f1(h_f)))  # (batch_size, 256)
        
        # Second fusion layer
        h_f = F.relu(self.bn_f2(self.fc_f2(h_f)))  # (batch_size, 128)

        # Step 5: Compute logits for all classes
        logits = self.fc_out(h_f)  # (batch_size, num_classes)

        # Step 6: Optionally convert logits to probabilities
        if self.use_softmax: 
            p = F.softmax(logits, dim=1)
        else:
            p = logits

        # Step 7: Compute next denoised embedding
        # Update rule: z_next = z_prev + logits @ W_embed
        # This pulls z_prev toward the embedding of the predicted class
        z_next = z_prev + logits @ W_embed  # (batch_size, embedding_dim)

        return z_next, logits
