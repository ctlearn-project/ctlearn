# Denoising block
import torch
from torch import nn
import torch.nn.functional as F

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.5, affine=True):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        # self.a = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        # self.b = nn.Parameter(torch.FloatTensor(1, 1, 1, 1))
        self.a = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        return self.a * x + self.b * self.bn(x)

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             # AdaptiveBatchNorm2d(out_channels),
#             nn.PReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             # AdaptiveBatchNorm2d(out_channels)
#         )

#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1),
#                 nn.BatchNorm2d(out_channels)
#             )

#         self.relu = nn.PReLU()

#     def forward(self, x):
#         return self.relu(self.conv_block(x) + self.shortcut(x))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, drop_path_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.GELU()

        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(8, out_channels)
            )

        try:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path_rate)
        except ImportError:
            self.drop_path = nn.Identity()

        # self.final_act = nn.GELU()
        self.final_act = MemoryEfficientSwish()
    def forward(self, x):
        residual = self.act1(self.norm1(self.conv1(x)))
        residual = self.act2(self.norm2(self.conv2(residual)))
        # residual = self.act1((self.conv1(x)))
        # residual = self.act2((self.conv2(residual)))

        residual = self.se(residual)

        out = self.drop_path(residual) + self.shortcut(x)
        return self.final_act(out)
    
class DenoiseBlock(nn.Module):
    def __init__(self, embedding_dim, num_channels=1):
        super().__init__()
        
        self.conv_path = nn.Sequential(
            ResidualBlock(num_channels, 32),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            ResidualBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
        )

        self.fc_z1 = nn.Linear(embedding_dim, 256)
        self.bn_z1 = nn.BatchNorm1d(256)
        self.fc_z2 = nn.Linear(256, 256)
        self.bn_z2 = nn.BatchNorm1d(256)
        self.fc_z3 = nn.Linear(256, 256)
        self.bn_z3 = nn.BatchNorm1d(256)

        self.fc_f1 = nn.Linear(512, 256)
        self.bn_f1 = nn.BatchNorm1d(256)
        self.fc_f2 = nn.Linear(256, 128)
        self.bn_f2 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, embedding_dim)

        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.act3 = nn.PReLU()
        self.act_f1 = nn.PReLU()
        self.act_f2 = nn.PReLU()

    def forward(self, x, z_prev, _):
        x_feat = self.conv_path(x)

        h1 = self.act1(self.bn_z1(self.fc_z1(z_prev)))
        h2 = self.act2(self.bn_z2(self.fc_z2(h1)))
        h3 = self.bn_z3(self.fc_z3(h2))

        z_feat = h3 + h1

        h_f = torch.cat([x_feat, z_feat], dim=1)

        h_f = self.act_f1(self.bn_f1(self.fc_f1(h_f)))
        h_f = self.act_f2(self.bn_f2(self.fc_f2(h_f)))
        z_next = self.fc_out(h_f)

        return z_next, None
