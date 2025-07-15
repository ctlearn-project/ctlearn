# NoProp-DT model

import torch
from torch import nn
import torch.nn.functional as F
# from .denoiseBlock import DenoiseBlock
from .denoiseBlockThinRestNet import DenoiseBlock, MemoryEfficientSwish

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
    
class NoPropDTReg(nn.Module):
    def __init__(self, task, num_outputs, embedding_dim=128, T=3, eta=0.1,num_blocks=[2, 3, 3, 3]):
        super().__init__()

        self.task = task
        num_classes = num_outputs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        self.blocks = nn.ModuleList([DenoiseBlock(embedding_dim,num_channels=1,num_blocks=num_blocks) for _ in range(T)])

        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            # MemoryEfficientSwish(),
            nn.Linear(embedding_dim//2, num_outputs)
)
        
        # Final classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Improved noise schedule
        self.register_buffer('alpha_bar', self._cosine_schedule(T))
        self.register_buffer('snr_diff', self._calculate_snr_diff(self.alpha_bar))


        self.target_embedder = nn.Linear(num_outputs, embedding_dim)
        
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _cosine_schedule(self, T):
        t = torch.arange(1, T+1, dtype=torch.float32)
        alpha_bar = torch.cos((t / T + 0.008) / 1.008 * (math.pi/2))**2
        return alpha_bar

    def _calculate_snr_diff(self, alpha_bar):
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        snr_prev = torch.cat([torch.tensor([0.]), snr[:-1]])
        return torch.clamp(snr - snr_prev, min=1e-5)

    def forward_denoise(self, x, z_prev, t):
        return self.blocks[t](x, z_prev, self.target_embedder)[0]

    def regress(self, z):
        return self.regressor(z)

    def inference(self, x):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        if not self.training:
            z = torch.zeros(B, self.embedding_dim, device=x.device)

        for t in range(self.T):
            z = self.forward_denoise(x, z, t)

        return self.regress(z)
    
    def forward(self, x):

        if self.task=="direction":
            return None, None, self.inference(x)
        elif self.task=="energy":
            return None, F.tanh(self.inference(x))*4, None
