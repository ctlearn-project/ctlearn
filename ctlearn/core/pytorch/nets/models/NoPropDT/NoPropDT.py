"""
NoProp-DT (No-Propagation Denoising Transformer) Model Module

This module implements the NoProp-DT architecture, a denoising diffusion model
for classification tasks on Cherenkov telescope data. The model uses progressive
denoising through multiple diffusion steps to refine predictions.

The NoProp-DT approach treats classification as a denoising task where noisy
class embeddings are progressively cleaned through T diffusion steps, ultimately
producing a clean classification prediction.

Classes:
    SimplifiedDenoiseBlock: Simplified denoise block for debugging/testing
    NoPropDT: Main NoProp-DT model for classification with diffusion

References:
    - "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
    - "Classifier-Free Diffusion Guidance" (Ho & Salimans, 2022)
"""

import torch
from torch import nn
from .denoiseBlockThinRestNet import DenoiseBlock
import math 

class SimplifiedDenoiseBlock(nn.Module):
    """
    Simplified denoising block for testing and debugging.
    
    This is a lightweight version of the full DenoiseBlock, useful for rapid
    prototyping and debugging the diffusion process. It uses simple CNNs for
    feature extraction and MLPs for processing embeddings.
    
    Architecture:
        Image → CNN → Features (64-dim)
                                        ↘
        Embedding → MLP → Features (256-dim) → Concat → MLP → Logits → Updated Embedding
    
    Attributes:
        conv_path (nn.Sequential): CNN for image feature extraction
        fc_z (nn.Sequential): MLP for embedding processing
        combined (nn.Sequential): MLP for combining features and producing logits
    """
    
    def __init__(self, embedding_dim, num_classes):
        """
        Initialize the simplified denoise block.
        
        Args:
            embedding_dim (int): Dimension of class embeddings
            num_classes (int): Number of output classes
        """
        super().__init__()
        # Simplified image feature extractor using CNN
        self.conv_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial dimensions by 2x
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduce spatial dimensions by 2x again
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling to fixed size
            nn.Flatten()  # Flatten to (batch_size, 64)
        )
        
        # Simplified embedding processor using MLP
        self.fc_z = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU()
        )
        
        # Combined processor: fuses image and embedding features
        self.combined = nn.Sequential(
            nn.Linear(256 + 64, 128),  # Input: concatenated features
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output: class logits
        )

    def forward(self, x, z_prev, W_embed):
        """
        Forward pass through the simplified denoise block.
        
        Args:
            x (torch.Tensor): Input images with shape (batch_size, 1, H, W)
            z_prev (torch.Tensor): Previous embedding estimate (batch_size, embedding_dim)
            W_embed (torch.Tensor): Class embedding matrix (num_classes, embedding_dim)
            
        Returns:
            tuple: (z_next, logits)
                - z_next: Updated embedding (batch_size, embedding_dim)
                - logits: Class predictions (batch_size, num_classes)
        """
        # Extract image features
        x_feat = self.conv_path(x)  # (batch_size, 64)
        
        # Process current embedding
        z_feat = self.fc_z(z_prev)  # (batch_size, 256)
        
        # Combine image and embedding features
        combined = torch.cat([x_feat, z_feat], dim=1)  # (batch_size, 320)
        logits = self.combined(combined)  # (batch_size, num_classes)
        
        # Update embedding using logits and class embeddings
        # This pulls z_prev toward the class embedding indicated by logits
        z_next = z_prev + logits @ W_embed  # (batch_size, embedding_dim)
        
        return z_next, logits
    
class NoPropDT(nn.Module):
    """
    NoProp-DT: No-Propagation Denoising Transformer for classification.
    
    This model implements a diffusion-based approach to classification where
    predictions are refined through multiple denoising steps. Each step removes
    noise from class embeddings, progressively clarifying the prediction.
    
    Key Concepts:
    
    Diffusion Process:
        - Forward: Add noise to true class embedding
        - Reverse: Learn to denoise and recover true class
        - Training: Match denoised output to clean embedding
        - Inference: Start from noise, denoise T steps, classify
    
    Class Embeddings (W_embed):
        - Each class has a learnable embedding vector
        - These vectors represent "ideal" class representations
        - Denoising process pulls noisy vectors toward these ideals
    
    Architecture Flow:
        Noisy Embedding → [DenoiseBlock_1 → ... → DenoiseBlock_T] → Classifier → Prediction
    
    Attributes:
        num_classes (int): Number of output classes
        embedding_dim (int): Dimension of class embedding space
        T (int): Number of diffusion steps
        eta (float): Learning rate scaling factor for diffusion loss
        W_embed (nn.Parameter): Learnable class embeddings (num_classes, embedding_dim)
        blocks (nn.ModuleList): List of T denoising blocks
        classifier (nn.Linear): Final classification head
        alpha_bar (torch.Tensor): Noise schedule parameters
        snr_diff (torch.Tensor): SNR differences for loss weighting
    """
    
    def __init__(self, num_outputs, embedding_dim=128, T=3, eta=0.1):
        """
        Initialize the NoProp-DT model.
        
        Args:
            num_outputs (int): Number of output classes
                For telescope data: 2 (gamma vs proton)
            embedding_dim (int, optional): Dimension of class embeddings. Defaults to 128
                Higher values allow richer representations but increase computation
            T (int, optional): Number of diffusion steps. Defaults to 3
                More steps allow finer denoising but slower inference
                Typical values: 3-10
            eta (float, optional): Diffusion loss scaling factor. Defaults to 0.1
                Controls the weight of denoising loss vs classification loss
                
        Example:
            >>> # Create model for binary classification
            >>> model = NoPropDT(num_outputs=2, embedding_dim=128, T=5)
            >>> x = torch.randn(32, 1, 120, 120)  # Batch of images
            >>> output, _, _ = model(x)  # Classification logits
            >>> print(output.shape)  # torch.Size([32, 2])
        """
        super().__init__()

        num_classes = num_outputs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        # Initialize learnable class embeddings
        # Each class gets a random embedding vector that will be learned during training
        # Small initial values (0.02 std) for stable training
        self.W_embed = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * 0.02, 
            requires_grad=True
        )
        
        # Create T denoising blocks for progressive refinement
        # Each block learns to remove one layer of noise
        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_classes) for _ in range(T)
        ])
        
        # Final classifier: maps clean embedding to class logits
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Noise schedule: determines how much noise at each step
        # Uses cosine schedule for smooth noise progression
        self.register_buffer('alpha_bar', self._cosine_schedule(T))
        
        # SNR differences: used to weight denoising loss at each step
        # Steps with larger SNR improvements get higher weight
        self.register_buffer('snr_diff', self._calculate_snr_diff(self.alpha_bar))

    def _cosine_schedule(self, T):
        """
        Generate a cosine noise schedule for diffusion.
        
        The cosine schedule provides smooth noise progression from high to low,
        which has been shown to improve training stability compared to linear schedules.
        
        Mathematical Formula:
            α̅_t = cos²((t/T + s) / (1 + s) × π/2)
            
            where s = 0.008 is a small offset to prevent numerical issues
        
        Args:
            T (int): Total number of diffusion steps
            
        Returns:
            torch.Tensor: Alpha bar values for each step with shape (T,)
                Values decrease from ~1.0 (low noise) to ~0.0 (high noise)
                
        Properties:
            - Monotonically decreasing: α̅_1 > α̅_2 > ... > α̅_T
            - Smooth transitions between steps
            - Prevents extreme noise levels at boundaries
        """
        # Create timestep array from 1 to T
        t = torch.arange(1, T + 1, dtype=torch.float32)
        
        # Apply cosine schedule with small offset for numerical stability
        alpha_bar = torch.cos((t / T + 0.008) / 1.008 * (math.pi / 2)) ** 2
        
        return alpha_bar

    def _calculate_snr_diff(self, alpha_bar):
        """
        Calculate Signal-to-Noise Ratio differences for loss weighting.
        
        SNR differences quantify how much signal quality improves from one
        diffusion step to the next. Steps with larger improvements receive
        higher weight in the loss function.
        
        Mathematical Formula:
            SNR_t = α̅_t / (1 - α̅_t)
            SNR_diff_t = SNR_t - SNR_{t-1}
        
        Args:
            alpha_bar (torch.Tensor): Noise schedule parameters with shape (T,)
            
        Returns:
            torch.Tensor: SNR differences with shape (T,)
                All values are positive (clamped to minimum 1e-5)
                
        Usage:
            Used in training loss to weight each denoising step:
            loss_t = snr_diff_t × MSE(denoised_t, target)
        """
        # Calculate SNR at each step
        snr = alpha_bar / (1 - alpha_bar + 1e-8)  # Add epsilon for numerical stability
        
        # Prepend SNR_0 = 0 (no signal before first step)
        snr_prev = torch.cat([torch.tensor([0.]), snr[:-1]])
        
        # Compute differences and clamp to ensure positivity
        return torch.clamp(snr - snr_prev, min=1e-5)

    def forward_denoise(self, x, z_prev, t):
        """
        Perform one step of denoising at timestep t.
        
        This method applies the t-th denoising block to refine the current
        embedding estimate. The block uses both the input image and the
        current noisy embedding to produce a cleaner estimate.
        
        Args:
            x (torch.Tensor): Input images with shape (batch_size, 1, H, W)
            z_prev (torch.Tensor): Current noisy embedding (batch_size, embedding_dim)
            t (int): Current timestep index (0 to T-1)
            
        Returns:
            torch.Tensor: Refined embedding after denoising
                Shape: (batch_size, embedding_dim)
                Has less noise than z_prev
                
        Process:
            1. Denoise block extracts features from image
            2. Combines image features with current embedding
            3. Outputs refined embedding with reduced noise
        """
        # Apply t-th denoising block
        # Returns (denoised_embedding, logits), we only need the embedding
        return self.blocks[t](x, z_prev, self.W_embed)[0]

    def inference(self, x):
        """
        Perform full inference through all diffusion steps.
        
        This method executes the complete denoising process:
        1. Start from zero embedding (or random noise during training)
        2. Progressively denoise through T steps
        3. Classify the final clean embedding
        
        Args:
            x (torch.Tensor): Input images with shape (batch_size, 1, H, W)
            
        Returns:
            torch.Tensor: Classification logits with shape (batch_size, num_classes)
                
        Behavior:
            Training (self.training=True):
                - Can start from random noise for exploration
                
            Evaluation (self.training=False):
                - Starts from zeros for deterministic predictions
                
        Example:
            >>> model.eval()
            >>> x = torch.randn(16, 1, 120, 120)
            >>> logits = model.inference(x)
            >>> predictions = torch.softmax(logits, dim=1)
            >>> classes = predictions.argmax(dim=1)
        """
        B = x.size(0)  # Batch size
        
        # Initialize embedding
        # Evaluation: Start from zeros for deterministic behavior
        # Training: Could use noise for stochastic exploration (commented out)
        z = torch.zeros(B, self.embedding_dim, device=x.device)
        
        # Progressive denoising through T steps
        for t in range(self.T):
            z = self.forward_denoise(x, z, t)
        
        # Classify the final clean embedding
        return self.classifier(z)

    def forward(self, x):
        """
        Forward pass through the model.
        
        This method provides a standardized interface compatible with other
        models in CTLearn. It wraps the inference method and returns outputs
        in the expected tuple format.
        
        Args:
            x (torch.Tensor): Input images with shape (batch_size, 1, H, W)
            
        Returns:
            tuple: (classification, energy, direction) where:
                - classification: Logits for particle type (batch_size, num_classes)
                - energy: None (not used for classification)
                - direction: None (not used for classification)
                
        Note:
            The tuple format (classification, energy, direction) is maintained
            for compatibility with multi-task training frameworks, even though
            this model only performs classification.
            
        Example:
            >>> model = NoPropDT(num_outputs=2)
            >>> x = torch.randn(32, 1, 120, 120)
            >>> cls, energy, direction = model(x)
            >>> print(cls.shape)  # torch.Size([32, 2])
            >>> print(energy)  # None
            >>> print(direction)  # None
        """
        return self.inference(x), None, None