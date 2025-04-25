import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckTransformerBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=4, use_gn=False):
        super(BottleneckTransformerBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, stride=1, bias=False)
        if use_gn:
            self.norm1 = nn.GroupNorm(32, out_channels // reduction)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels // reduction)

        self.transformer_block = nn.TransformerEncoderLayer(d_model=out_channels // reduction, nhead=8)
        
        self.conv2 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, stride=1, bias=False)
        if use_gn:
            self.norm2 = nn.GroupNorm(32, out_channels)
        else:
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        b, c, h, w = out.size()
        out = out.view(b, c, -1).permute(2, 0, 1)  # Prepare for transformer
        out = self.transformer_block(out)
        out = out.permute(1, 2, 0).view(b, c, h, w)
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TransformerDBB(nn.Module):
    def __init__(self, block, layers, num_inputs=2, num_classes=1, use_gn=False, dropout_rate=0.5):
        super(TransformerDBB, self).__init__()
        self.in_channels = 64
        self.use_gn = use_gn

        # Backbone único
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1 = nn.GroupNorm(32, 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, use_gn=self.use_gn))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_gn=self.use_gn))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # Concatenar las entradas a lo largo del canal
        x = torch.cat((x1, x2), dim=1)

        # Backbone único
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def TransformerDuo(num_blocks=[3, 4, 6, 3], num_inputs=2, num_classes=2, use_gn=True, dropout_rate=0.3):
    # Configurar un modelo más ligero
    return TransformerDBB(BottleneckTransformerBlock, num_blocks, num_inputs=num_inputs, num_classes=num_classes, use_gn=use_gn, dropout_rate=dropout_rate)

# Instancia del modelo
# bottleneck_transformer_duo = TransformerDuo(num_blocks=[3, 4, 6, 3], num_classes=1, use_gn=True, dropout_rate=0.3)