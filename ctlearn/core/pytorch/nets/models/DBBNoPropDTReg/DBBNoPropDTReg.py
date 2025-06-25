# NoProp-DT model

import torch
from torch import nn

# from .denoiseBlock import DenoiseBlock
from .denoiseBlockThinRestNet import DenoiseBlock, MemoryEfficientSwish

import math 
    
class DBBNoPropDTReg(nn.Module):
    def __init__(self, task, num_outputs, embedding_dim=128, T=3, eta=0.1):
        super().__init__()

        self.task = task
        num_classes = num_outputs
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.T = T
        self.eta = eta

        self.blocks = nn.ModuleList([DenoiseBlock(embedding_dim,num_channels=1) for _ in range(T)])
        # self.regressor = nn.Linear(embedding_dim, num_outputs)
        self.regressor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            MemoryEfficientSwish(),
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

    def forward_denoise(self, x, y, z_prev, t):
        return self.blocks[t](x, y, z_prev, None)[0]

    def regress(self, z):
        return self.regressor(z)

    def inference(self, x, y ):
        B = x.size(0)
        z = torch.randn(B, self.embedding_dim, device=x.device)
        if not self.training:
            z = torch.zeros(B, self.embedding_dim, device=x.device)

        for t in range(self.T):
            z = self.forward_denoise(x, y, z, t)

        return self.regress(z)
    
    def forward(self, x, y):

        if self.task=="direction":
            return None, None, self.inference(x,y)
        elif self.task=="energy":
            return None, self.inference(x,y), None
