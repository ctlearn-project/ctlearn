"""
Convolutional Neural Network Building Blocks Module

This module provides custom CNN building blocks for CTLearn neural network architectures.
It includes specialized layers for uncertainty quantification (evidential learning) and
residual blocks optimized for Cherenkov telescope image analysis.

Classes:
    Dirichlet: Evidential layer for classification with uncertainty quantification
    NormalInvGamma: Evidential layer for regression with uncertainty quantification
    ResBlock: Residual convolutional block with Gabor filter initialization
"""

import torch
import torch.nn as nn
from ctlearn.core.pytorch.net_utils import ModelHelper
import torch.nn.functional as F

class Dirichlet(nn.Module):
    """
    Dirichlet distribution layer for evidential classification.
    
    This layer outputs parameters of a Dirichlet distribution, which is used
    in evidential deep learning to quantify classification uncertainty. The
    Dirichlet parameters (alphas) represent the concentration of probability
    mass for each class, providing both predictions and uncertainty estimates.
    
    Mathematical Background:
        Given evidence e_k for each class k, the Dirichlet parameters are:
        α_k = e_k + 1
        
        Where evidence is transformed through:
        e_k = softplus(z_k) to ensure positivity
        
    Attributes:
        dense (nn.Linear): Linear layer to compute raw evidence values
        out_units (int): Number of output classes
    
    Methods:
        evidence: Apply softplus activation to ensure positive evidence
        forward: Compute Dirichlet parameters from input features
    """
    
    def __init__(self, in_features, out_units):
        """
        Initialize the Dirichlet layer.
        
        Args:
            in_features (int): Number of input features from previous layer
            out_units (int): Number of output classes (Dirichlet dimensions)
        """
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = out_units

    def evidence(self, x):
        """
        Transform raw outputs to positive evidence values.
        
        Uses softplus activation: log(1 + exp(x)) which is smooth and always positive.
        This ensures that evidence values are strictly positive as required for
        Dirichlet parameters.
        
        Args:
            x (torch.Tensor): Raw evidence values from linear layer
            
        Returns:
            torch.Tensor: Positive evidence values
        """
        return F.softplus(x)

    def forward(self, x):
        """
        Compute Dirichlet distribution parameters.
        
        This method transforms input features into Dirichlet parameters (alphas)
        which characterize the predictive distribution over classes. Higher alpha
        values indicate higher confidence in that class.
        
        Args:
            x (torch.Tensor): Input features with shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Dirichlet parameters (alphas) with shape (batch_size, out_units)
                Each alpha_k > 1, where higher values indicate more evidence for class k
                
        Note:
            The sum of alphas S = Σα_k represents total evidence.
            Class probabilities can be computed as: p_k = α_k / S
            Uncertainty can be quantified using various metrics on the Dirichlet distribution
        """
        # Compute raw evidence
        out = self.dense(x)
        # Transform to Dirichlet parameters (add 1 to ensure α > 1)
        alpha = self.evidence(out) + 1
        return alpha
    
class NormalInvGamma(nn.Module):
    """
    Normal Inverse Gamma distribution layer for evidential regression.
    
    This layer outputs parameters of a Normal Inverse Gamma (NIG) distribution,
    used in evidential deep learning for regression with uncertainty quantification.
    The NIG distribution provides both aleatoric (data) and epistemic (model) uncertainty.
    
    Mathematical Background:
        The NIG distribution is parameterized by (μ, v, α, β):
        - μ: Predicted mean
        - v: Virtual observation count (epistemic uncertainty)
        - α: Shape parameter (controls uncertainty distribution)
        - β: Scale parameter (aleatoric uncertainty)
        
        Predictive variance: var = β / (v * (α - 1))
        
    Attributes:
        dense (nn.Linear): Linear layer to compute 4 parameters
        out_units (int): Number of regression outputs (typically 1)
    
    Methods:
        evidence: Apply softplus to ensure positive parameters
        forward: Compute NIG parameters or predictions based on mode
    """
    
    def __init__(self, in_features, out_units):
        """
        Initialize the Normal Inverse Gamma layer.
        
        Args:
            in_features (int): Number of input features from previous layer
            out_units (int): Number of regression outputs (usually 1)
        """
        super().__init__()
        # Output 4 parameters per regression target: μ, log(v), log(α), log(β)
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        """
        Transform raw outputs to positive parameter values.
        
        Uses softplus activation to ensure v, α, and β are strictly positive
        as required by the Normal Inverse Gamma distribution.
        
        Args:
            x (torch.Tensor): Raw parameter values
            
        Returns:
            torch.Tensor: Positive parameter values
        """
        return F.softplus(x)

    def forward(self, x):
        """
        Compute NIG distribution parameters or predictions.
        
        During training, returns all four NIG parameters for loss computation.
        During inference, returns mean prediction and uncertainty estimate.
        
        Args:
            x (torch.Tensor): Input features with shape (batch_size, in_features)
            
        Returns:
            Training mode:
                tuple: (mu, v, alpha, beta) - all NIG parameters
                    - mu: Mean predictions (batch_size, out_units)
                    - v: Virtual observations (batch_size, out_units)
                    - alpha: Shape parameters (batch_size, out_units)
                    - beta: Scale parameters (batch_size, out_units)
                    
            Inference mode:
                tuple: (mu, var) - predictions and uncertainties
                    - mu: Mean predictions (batch_size, out_units)
                    - var: Predictive variance (batch_size, out_units)
                    
        Note:
            The predictive variance combines both aleatoric and epistemic uncertainty:
            var = β / (v * (α - 1))
            
            Higher v means lower epistemic uncertainty (more confident)
            Higher β means higher aleatoric uncertainty (noisier data)
        """
        # Compute raw parameters
        out = self.dense(x)
        # Split into 4 components
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        
        # Transform to ensure positivity
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1  # Ensure α > 1
        beta = self.evidence(logbeta)
        
        # Return appropriate outputs based on mode
        if self.training:
            # Return all parameters for evidential loss computation
            return mu, v, alpha, beta
        else:
            # Return prediction and uncertainty for inference
            # Predictive variance from NIG distribution
            var = torch.sqrt(beta / (v * (alpha - 1)))
            return mu, var 
    
class ResBlock(nn.Module):
    """
    Residual convolutional block with Gabor filter initialization.
    
    This block implements a residual connection with convolutional layers,
    batch normalization, and pooling. It's optimized for processing Cherenkov
    telescope images by initializing filters with Gabor kernels that are
    particularly effective at detecting oriented features in shower images.
    
    Architecture:
        Input → Conv2d → BatchNorm → LeakyReLU → (+) → MaxPool → Conv1x1 → Dropout → Output
                                                    ↑
                                                    |
                                              Residual Connection
    
    Attributes:
        conv (nn.Conv2d): Main convolutional layer
        conv_dropout (nn.Dropout2d): Spatial dropout for regularization
        batch_norm (nn.BatchNorm2d): Batch normalization layer
        pool (nn.MaxPool2d): Max pooling layer (2x2)
        activation (nn.LeakyReLU): Activation function
        conv_out (nn.Conv2d): 1x1 conv for channel adjustment
    
    Methods:
        forward: Process input through residual block
    """
    
    def __init__(self, n_chans_in, n_chans_out, kernel_size=3, conv_drop_pro=0.2):
        """
        Initialize the residual block.
        
        Args:
            n_chans_in (int): Number of input channels
            n_chans_out (int): Number of output channels
            kernel_size (int, optional): Size of convolutional kernel. Defaults to 3
            conv_drop_pro (float, optional): Dropout probability. Defaults to 0.2
                
        Initialization Strategy:
            1. Kaiming normal initialization for main conv layer weights
            2. Batch norm weights initialized to 0.5
            3. Batch norm biases initialized to 0
            4. First few filters initialized with Gabor kernels for edge detection
        """
        super(ResBlock, self).__init__()

        # Main convolutional layer (maintains channel dimension)
        self.conv = nn.Conv2d(
            n_chans_in, 
            n_chans_in, 
            kernel_size=kernel_size, 
            padding=int(kernel_size/2),  # Same padding
            bias=False  # Bias not needed before batch norm
        )
        
        # Spatial dropout for regularization
        self.conv_dropout = nn.Dropout2d(p=conv_drop_pro)
        
        # Batch normalization for training stability
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans_in)
        
        # Max pooling to reduce spatial dimensions
        self.pool = nn.MaxPool2d(2)
        
        # LeakyReLU activation (allows small negative gradients)
        self.activation = nn.LeakyReLU()
        
        # 1x1 convolution to adjust channel dimension
        self.conv_out = nn.Conv2d(
            n_chans_in, 
            n_chans_out, 
            kernel_size=1, 
            padding=0, 
            bias=False
        )

        # Initialize conv layer with Kaiming normal
        # Appropriate for LeakyReLU activation
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        
        # Initialize batch norm parameters
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

        # Initialize first filters with Gabor kernels
        # Gabor filters are effective for detecting oriented features
        # particularly useful for elongated shower images
        kernels = ModelHelper.GaborKernels(size=kernel_size, showPlots=False)
        for i in range(min(self.conv.weight.shape[0], len(kernels))):
            with torch.no_grad():
                # Scale Gabor kernel by 100 for appropriate magnitude
                self.conv.weight[i, :] = torch.nn.Parameter(
                    torch.tensor(kernels[i] * 100)
                )

    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Processing Steps:
        1. Apply convolution to extract features
        2. Normalize with batch norm
        3. Apply activation function
        4. Add residual connection (skip connection)
        5. Reduce spatial dimensions with pooling
        6. Adjust channels with 1x1 convolution
        7. Apply dropout for regularization
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_chans_in, height, width)
            
        Returns:
            torch.Tensor: Output tensor with shape 
                (batch_size, n_chans_out, height/2, width/2)
                
        Note:
            The residual connection helps gradient flow during backpropagation
            and allows the network to learn identity mappings when beneficial.
            
            Spatial dimensions are halved due to max pooling with stride 2.
        """
        # Convolutional feature extraction
        out = self.conv(x)
        
        # Normalize activations
        out = self.batch_norm(out)
        
        # Non-linear activation
        out = self.activation(out)
        
        # Add residual connection (element-wise addition)
        # This helps with gradient flow and allows learning identity mappings
        out = out + x
        
        # Reduce spatial dimensions
        out = self.pool(out)
        
        # Adjust number of channels
        out = self.conv_out(out)
        
        # Apply dropout for regularization
        out = self.conv_dropout(out)
        
        return out