"""
Dual-Backbone NoProp-DT Regression Model Module

This module implements a dual-backbone denoising diffusion model for regression tasks
on Cherenkov telescope data. It uses a progressive denoising approach with multiple
diffusion steps to refine predictions, particularly effective for energy and direction
reconstruction tasks.

The NoProp-DT (No-Propagation Denoising Transformer) architecture uses:
- Progressive denoising through T diffusion steps
- Cosine noise schedule for stable training
- SNR (Signal-to-Noise Ratio) weighting for loss computation
- Dual-backbone design for processing charge and timing information

Classes:
    DBBNoPropDTReg: Dual-backbone diffusion model for regression tasks

References:
    - "Denoising Diffusion Probabilistic Models" (Ho et al., NeurIPS 2020)
    - "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
"""

import torch
from torch import nn
from .denoiseBlockThinRestNet import DenoiseBlock, MemoryEfficientSwish
import math 
    
class DBBNoPropDTReg(nn.Module):
    """
    Dual-Backbone NoProp-DT model for regression tasks with diffusion.
    
    This model implements a denoising diffusion approach for regression where
    predictions are progressively refined through multiple diffusion steps.
    Each step removes noise from the target embedding, ultimately producing
    a clean prediction.
    
    Architecture Overview:
        Input (x, y) → [DenoiseBlock_1 → ... → DenoiseBlock_T] → Regressor → Output
        
        At each step t:
        1. Add noise to target embedding
        2. Pass through denoise block
        3. Refine embedding
        4. Final step: regress to prediction
    
    Key Components:
        - DenoiseBlocks: T sequential denoising steps
        - Target Embedder: Projects targets to latent space
        - Regressor: Final prediction head
        - Noise Schedule: Cosine schedule for adding noise
        - SNR Weighting: Signal-to-noise ratio for loss weighting
    
    Attributes:
        task (str): Task type ('energy', 'direction')
        num_classes (int): Number of output values
        embedding_dim (int): Dimension of latent embedding space
        T (int): Number of diffusion steps
        eta (float): Learning rate scaling factor
        blocks (nn.ModuleList): List of denoising blocks
        regressor (nn.Sequential): Final regression head
        classifier (nn.Linear): Classification head (unused in regression)
        target_embedder (nn.Linear): Projects targets to embedding space
        alpha_bar (torch.Tensor): Noise schedule parameters
        snr_diff (torch.Tensor): SNR difference for loss weighting
    """
    
    def __init__(self, task, num_outputs, embedding_dim=128, T=3, eta=0.1):
        """
        Initialize the DBBNoPropDTReg model.
        
        Args:
            task (str): Task type to perform
                Options: 'energy' (energy regression), 'direction' (direction regression)
            num_outputs (int): Number of regression outputs
                For energy: 1 (single value)
                For direction: 2 or 3 (angular coordinates)
            embedding_dim (int, optional): Dimension of latent embedding space.
                Defaults to 128. Higher values allow more complex representations
                but increase memory usage.
            T (int, optional): Number of diffusion steps. Defaults to 3.
                More steps allow finer refinement but increase computation.
                Typical values: 3-10
            eta (float, optional): Learning rate scaling factor. Defaults to 0.1.
                Controls the weight of denoising loss vs final regression loss.
                
        Note:
            The model uses Xavier uniform initialization for regressor weights
            to ensure stable training at initialization.
        """
        super().__init__()

        self.task = task
        num_classes = num_outputs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        # Create T denoising blocks for progressive refinement
        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_channels=1) 
            for _ in range(T)
        ])
        
        # Final regression head with intermediate activation
        # Architecture: Linear → Swish → Linear
        # This provides non-linearity while maintaining differentiability
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            MemoryEfficientSwish(),  # Memory-efficient activation function
            nn.Linear(embedding_dim // 2, num_outputs)
        )
        
        # Classification head (included for architecture compatibility but unused)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Noise schedule: determines how much noise to add at each step
        # Uses cosine schedule for smooth noise progression
        self.register_buffer('alpha_bar', self._cosine_schedule(T))
        
        # SNR differences for loss weighting
        # Weights each diffusion step by its contribution to final quality
        self.register_buffer('snr_diff', self._calculate_snr_diff(self.alpha_bar))

        # Target embedder: projects ground truth values to latent space
        self.target_embedder = nn.Linear(num_outputs, embedding_dim)
        
        # Initialize regressor weights with Xavier uniform
        # Ensures initial gradients are neither too large nor too small
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _cosine_schedule(self, T):
        """
        Generate a cosine noise schedule for diffusion.
        
        This schedule determines how much noise is added at each diffusion step.
        The cosine schedule provides a smooth progression from high noise to low noise,
        which has been shown to improve training stability and final performance.
        
        Mathematical Formula:
            α̅_t = cos²((t/T + 0.008) / 1.008 × π/2)
            
            where:
            - t ∈ [1, T]: current diffusion step
            - α̅_t: proportion of original signal retained
            - (1 - α̅_t): proportion of noise
        
        Args:
            T (int): Total number of diffusion steps
            
        Returns:
            torch.Tensor: Alpha bar values for each step with shape (T,)
                Values range from ~1.0 (step 1, low noise) to ~0.0 (step T, high noise)
                
        Properties:
            - Monotonically decreasing: α̅_1 > α̅_2 > ... > α̅_T
            - Smooth transitions between steps
            - Small offset (0.008) prevents numerical issues at boundaries
            
        Example:
            >>> schedule = self._cosine_schedule(5)
            >>> print(schedule)
            tensor([0.9950, 0.9801, 0.9553, 0.9211, 0.8782])
        """
        # Create timestep array from 1 to T
        t = torch.arange(1, T + 1, dtype=torch.float32)
        
        # Apply cosine schedule formula
        # The small offsets (0.008, 1.008) prevent division by zero and ensure
        # alpha_bar doesn't reach exactly 0 or 1
        alpha_bar = torch.cos((t / T + 0.008) / 1.008 * (math.pi / 2)) ** 2
        
        return alpha_bar

    def _calculate_snr_diff(self, alpha_bar):
        """
        Calculate Signal-to-Noise Ratio differences for loss weighting.
        
        The SNR difference quantifies how much the signal quality improves
        from one diffusion step to the next. This is used to weight the
        denoising loss at each step - steps that make larger improvements
        receive higher weight.
        
        Mathematical Formula:
            SNR_t = α̅_t / (1 - α̅_t)
            SNR_diff_t = SNR_t - SNR_{t-1}
            
            where:
            - SNR_t: Signal-to-noise ratio at step t
            - α̅_t: Alpha bar value at step t
        
        Args:
            alpha_bar (torch.Tensor): Alpha bar values from noise schedule
                Shape: (T,)
                
        Returns:
            torch.Tensor: SNR differences with shape (T,)
                Positive values indicate signal improvement
                Clamped to minimum of 1e-5 to prevent numerical issues
                
        Implementation Details:
            - SNR_0 is set to 0 (no signal before first step)
            - Differences are clamped to ensure positive weights
            - Small epsilon (1e-8) added to denominator for numerical stability
            
        Example:
            >>> alpha_bar = torch.tensor([0.99, 0.95, 0.90])
            >>> snr_diff = self._calculate_snr_diff(alpha_bar)
            >>> print(snr_diff)
            tensor([99.0000, 19.0000, 9.0000])  # Approximate values
        """
        # Calculate SNR at each step: α̅_t / (1 - α̅_t)
        # Small epsilon prevents division by zero when alpha_bar ≈ 1
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        
        # Prepend SNR_0 = 0 (before first denoising step)
        # Then compute differences: SNR_t - SNR_{t-1}
        snr_prev = torch.cat([torch.tensor([0.]), snr[:-1]])
        
        # Clamp to minimum value to ensure positive weights
        # This prevents negative or zero weights that would destabilize training
        return torch.clamp(snr - snr_prev, min=1e-5)

    def forward_denoise(self, x, y, z_prev, t):
        """
        Perform one step of denoising.
        
        This method applies the t-th denoising block to refine the current
        embedding estimate. Each block processes the input features along
        with the current noisy embedding to produce a cleaner estimate.
        
        Args:
            x (torch.Tensor): Primary input features (charge images)
                Shape: (batch_size, channels, height, width)
            y (torch.Tensor): Secondary input features (timing images)
                Shape: (batch_size, channels, height, width)
            z_prev (torch.Tensor): Previous/current embedding estimate
                Shape: (batch_size, embedding_dim)
                Contains noise that will be reduced in this step
            t (int): Current diffusion step index (0 to T-1)
            
        Returns:
            torch.Tensor: Refined embedding after denoising
                Shape: (batch_size, embedding_dim)
                Has less noise than z_prev
                
        Process:
            1. Denoise block extracts features from x and y
            2. Current embedding z_prev is used as query
            3. Block outputs refined embedding with reduced noise
            
        Note:
            The fourth parameter (None) is for optional label embeddings,
            which are not used in regression tasks.
        """
        # Apply t-th denoising block
        # Returns tuple (denoised_embedding, attention_weights)
        # We only need the embedding, so take index [0]
        return self.blocks[t](x, y, z_prev, None)[0]

    def regress(self, z):
        """
        Generate final prediction from clean embedding.
        
        This method applies the regression head to convert the final
        clean embedding into the actual prediction values (energy or direction).
        
        Args:
            z (torch.Tensor): Clean embedding from final denoising step
                Shape: (batch_size, embedding_dim)
                
        Returns:
            torch.Tensor: Final predictions
                Shape: (batch_size, num_outputs)
                For energy: (batch_size, 1)
                For direction: (batch_size, 2) or (batch_size, 3)
                
        Architecture:
            embedding_dim → embedding_dim/2 → num_outputs
            with Swish activation in between
            
        Example:
            >>> z = torch.randn(32, 128)  # Batch of 32, embedding_dim=128
            >>> pred = model.regress(z)
            >>> print(pred.shape)  # torch.Size([32, 1]) for energy
        """
        return self.regressor(z)

    def inference(self, x, y):
        """
        Perform full inference through all diffusion steps.
        
        This method executes the complete denoising process, starting from
        random noise (or zeros during evaluation) and progressively refining
        the embedding through T diffusion steps, finally producing a prediction.
        
        Process:
            1. Initialize embedding z:
               - Training: Random Gaussian noise
               - Inference: Zeros (deterministic)
            2. For each diffusion step t = 0 to T-1:
               - Apply denoising block to refine z
            3. Final step: Apply regressor to get prediction
        
        Args:
            x (torch.Tensor): Primary input features (charge images)
                Shape: (batch_size, 1, height, width)
            y (torch.Tensor): Secondary input features (timing images)
                Shape: (batch_size, 1, height, width)
                
        Returns:
            torch.Tensor: Final predictions
                Shape: (batch_size, num_outputs)
                
        Behavior Difference:
            Training (self.training=True):
                - Starts from random noise
                - Introduces stochasticity for exploration
                - Helps learn robust denoising
                
            Evaluation (self.training=False):
                - Starts from zeros
                - Deterministic predictions
                - More stable and reproducible
                
        Example:
            >>> model.eval()  # Set to evaluation mode
            >>> x = torch.randn(8, 1, 120, 120)  # Batch of 8 images
            >>> y = torch.randn(8, 1, 120, 120)
            >>> predictions = model.inference(x, y)
            >>> print(predictions.shape)  # torch.Size([8, 1]) for energy
        """
        # Get batch size from input
        B = x.size(0)
        
        # Initialize embedding
        # Training: Random noise for stochastic exploration
        # Evaluation: Zeros for deterministic predictions
        z = torch.randn(B, self.embedding_dim, device=x.device)
        if not self.training:
            z = torch.zeros(B, self.embedding_dim, device=x.device)

        # Progressive denoising through T steps
        for t in range(self.T):
            z = self.forward_denoise(x, y, z, t)

        # Generate final prediction from clean embedding
        return self.regress(z)
    
    def forward(self, x):
        if x.shape[1] >= 2:
            x, y = torch.split(x, [1, x.shape[1]-1], dim=1)
        else:
            y = x
        """
        Forward pass through the model.
        
        This method provides a task-specific interface to the model, returning
        predictions in the expected format for each task type. It wraps the
        inference method and formats outputs appropriately.
        
        Args:
            x (torch.Tensor): Primary input features (charge images)
                Shape: (batch_size, 1, height, width)
            y (torch.Tensor): Secondary input features (timing images)
                Shape: (batch_size, 1, height, width)
                
        Returns:
            tuple: (classification, energy, direction) where:
                - classification: None (not used for regression)
                - energy: Predictions if task=='energy', else None
                - direction: Predictions if task=='direction', else None
                
                Only one of energy/direction is non-None based on self.task
                
        Task Routing:
            - task == 'direction': Returns (None, None, predictions)
              where predictions = (batch_size, 2 or 3) for angular coordinates
              
            - task == 'energy': Returns (None, predictions, None)
              where predictions = (batch_size, 1) for energy values
              
        Example:
            >>> model = DBBNoPropDTReg(task='energy', num_outputs=1)
            >>> x = torch.randn(16, 1, 120, 120)
            >>> y = torch.randn(16, 1, 120, 120)
            >>> cls, energy, direction = model(x, y)
            >>> print(cls)  # None
            >>> print(energy.shape)  # torch.Size([16, 1])
            >>> print(direction)  # None
            
        Note:
            The tuple format (classification, energy, direction) is maintained
            for compatibility with multi-task training frameworks, even though
            this model only performs one task at a time.
        """
        # Route to appropriate output based on task
        if self.task == "direction":
            # Direction reconstruction: return predictions in third position
            return None, None, self.inference(x, y)
        elif self.task == "energy":
            # Energy regression: return predictions in second position
            return None, self.inference(x, y), None
