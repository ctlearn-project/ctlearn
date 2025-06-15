# Denoising block
import torch
from torch import nn
import torch.nn.functional as F

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveBatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            AdaptiveBatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv_block(x) + self.shortcut(x))


class DenoiseBlock(nn.Module):
    def __init__(self, embedding_dim, num_classes, use_softmax=False, num_channels=1):
        super().__init__()
        self.use_softmax = use_softmax
        # ThinResNet convolutional path
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
            # nn.BatchNorm1d(256)
        )

        # Fully connected layers for processing noisy embedding vector z_prev
        self.fc_z1 = nn.Linear(embedding_dim, 256)
        self.bn_z1 = nn.BatchNorm1d(256)

        self.fc_z2 = nn.Linear(256, 256)
        self.bn_z2 = nn.BatchNorm1d(256)

        self.fc_z3 = nn.Linear(256, 256)
        self.bn_z3 = nn.BatchNorm1d(256)

        # Layers to combine image and embedding features
        self.fc_f1 = nn.Linear(256 + 256, 256)
        self.bn_f1 = nn.BatchNorm1d(256)
        self.fc_f2 = nn.Linear(256, 128)
        self.bn_f2 = nn.BatchNorm1d(128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, z_prev, W_embed):
        # Extract features from the input image x
        x_feat = self.conv_path(x)

        # Process the noisy class embedding z_prev
        h1 = F.relu(self.bn_z1(self.fc_z1(z_prev)))
        h2 = F.relu(self.bn_z2(self.fc_z2(h1)))
        h3 = self.bn_z3(self.fc_z3(h2))

        z_feat = h3 + h1  # Residual connection

        # Concatenate image and embedding features
        h_f = torch.cat([x_feat, z_feat], dim=1)

        # Process combined features through fully connected layers
        h_f = F.relu(self.bn_f1(self.fc_f1(h_f)))
        h_f = F.relu(self.bn_f2(self.fc_f2(h_f)))
        # h_f =self.bn_f2(self.fc_f2(h_f))
        # h_f = F.relu(self.fc_f1(h_f))
        # h_f = F.relu(self.fc_f2(h_f))

        # Compute logits for all classes
        logits = self.fc_out(h_f)

        # Convert logits to probability distribution over classes
        if self.use_softmax: 
            p = F.softmax(logits, dim=1)
        else:
            p = logits

        # Compute the next denoised embedding
        # z_next = p @ W_embed
        z_next = z_prev + logits @ W_embed

        return z_next, logits
