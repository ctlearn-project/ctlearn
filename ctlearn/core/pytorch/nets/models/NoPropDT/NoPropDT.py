# NoProp-DT model

import torch
from torch import nn

# from .denoiseBlock import DenoiseBlock
from .denoiseBlockThinRestNet import DenoiseBlock

import math 

class SimplifiedDenoiseBlock(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        # Simplified image feature extractor
        self.conv_path = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Simplified embedding processor
        self.fc_z = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU()
        )
        
        # Combined processor
        self.combined = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, z_prev, W_embed):
        # Image features
        x_feat = self.conv_path(x)
        
        # Process embedding
        z_feat = self.fc_z(z_prev)
        
        # Combine features
        combined = torch.cat([x_feat, z_feat], dim=1)
        logits = self.combined(combined)
        
        # Update embedding
        z_next = z_prev + logits @ W_embed
        
        return z_next, logits
    
class NoPropDT(nn.Module):
    def __init__(self, num_outputs, embedding_dim=128, T=3, eta=0.1):
        super().__init__()

        num_classes = num_outputs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        # Initialize learnable class embeddings
        self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.02, requires_grad=True)
        
        # Create denoising blocks
        self.blocks = nn.ModuleList([
            DenoiseBlock(embedding_dim, num_classes) for _ in range(T)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Improved noise schedule
        self.register_buffer('alpha_bar', self._cosine_schedule(T))
        self.register_buffer('snr_diff', self._calculate_snr_diff(self.alpha_bar))

    def _cosine_schedule(self, T):
        t = torch.arange(1, T+1, dtype=torch.float32)
        alpha_bar = torch.cos((t / T + 0.008) / 1.008 * (math.pi/2))**2
        return alpha_bar

    def _calculate_snr_diff(self, alpha_bar):
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        snr_prev = torch.cat([torch.tensor([0.]), snr[:-1]])
        return torch.clamp(snr - snr_prev, min=1e-5)

    def forward_denoise(self, x, z_prev, t):
        return self.blocks[t](x, z_prev, self.W_embed)[0]

    def inference(self, x):
        B = x.size(0)
        z = torch.zeros(B, self.embedding_dim, device=x.device)
        
        for t in range(self.T):
            z = self.forward_denoise(x, z, t)
        
        return self.classifier(z)

    def forward(self, x):
        return self.inference(x), None, None
    
# class NoPropDT(nn.Module):
#     def __init__(self, num_outputs, embedding_dim, T, eta, use_softmax=False, num_channels=1):
#         super().__init__()
#         num_classes = num_outputs
#         self.num_classes = num_classes  # Total number of classes (e.g., 10 for MNIST)
#         self.embedding_dim = embedding_dim  # Size of the vector that represents each class
#         self.T = T  # Number of denoising steps (number of DenoiseBlocks)
#         self.eta = eta  # A hyperparameter used in the loss function

#         # Create a list of T denoising blocks. Each block learns to reduce noise.
#         self.blocks = nn.ModuleList([
#             DenoiseBlock(embedding_dim, num_classes,use_softmax,num_channels) for _ in range(T)
#         ])

#         # Learnable matrix that holds one vector (embedding) per class (e.g., 10 rows for 10 digits)
#         self.W_embed = nn.Parameter(torch.randn(num_classes, embedding_dim) * 1.1, requires_grad=True)

#         # Final classifier layer to predict class label from embedding
#         self.classifier = nn.Linear(embedding_dim, num_classes)

#         # --- Prepare cosine noise schedule for diffusion process ---

#         # t = [1, 2, ..., T]
#         t = torch.arange(1, T+1, dtype=torch.float32)

#         # Calculate alpha_t using cosine schedule
#         alpha_t = torch.cos(t / T * (math.pi/2))**2

#         # alpha_bar is cumulative product of alpha_t, used to scale noise
#         alpha_bar = torch.cumprod(alpha_t, dim=0)

#         # Calculate signal-to-noise ratio (SNR)
#         snr = alpha_bar / (1 - alpha_bar + 1e-8)

#         # Previous SNR (shifted by one timestep)
#         snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)

#         # Difference in SNR between steps, used to weight denoising loss
#         snr_diff = snr - snr_prev
#         snr_diff = torch.clamp(snr_diff, min=1e-5)


#         #----------------------------
#         # t = torch.arange(1, T + 1, dtype=torch.float32)
#         # alpha_t = torch.cos(t / T * (math.pi / 2)) ** 2
#         # alpha_bar = torch.cumprod(alpha_t, dim=0)
#         # # snr = alpha_bar / (1 - alpha_bar)
#         # snr = alpha_bar / (1 - alpha_bar + 1e-8)
#         # snr_prev = torch.cat([torch.tensor([0.], dtype=snr.dtype), snr[:-1]], dim=0)
#         # snr_diff = snr - snr_prev
#         # snr_diff = torch.clamp(snr_diff, min=1e-5)

#         #----------------------------
#         # Save alpha_bar and snr_diff inside the model so they move to GPU automatically
#         self.register_buffer('alpha_bar', alpha_bar)
#         self.register_buffer('snr_diff', snr_diff)

#     # Perform denoising at step t using DenoiseBlock[t]
#     def forward_denoise(self, x, z_prev, t):
#         return self.blocks[t](x, z_prev, self.W_embed)[0]

#     # Use final denoised vector to predict the class
#     def classify(self, z):
#         return self.classifier(z)

#     # Run all denoising steps in order to produce final prediction
#     def inference(self, x):
#         B = x.size(0)  # Batch size
#         # Start with random noise as initial z
#         z = torch.randn(B, self.embedding_dim, device=x.device)
        
#         if not self.training:
#             z = torch.zeros(B, self.embedding_dim, device=x.device)
         
#         # Pass through all denoising blocks one by one
#         for t in range(self.T):
#             z = self.forward_denoise(x, z, t)
        
#         # Use final denoised result to classify
#         return self.classify(z)

#     def forward(self, x):
#         return self.inference(x), None, None