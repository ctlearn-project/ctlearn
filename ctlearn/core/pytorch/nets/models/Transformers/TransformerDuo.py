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
    def __init__(self, block, layers, num_inputs=1, num_classes=1, use_gn=False, use_concat=False, dropout_rate=0.5):
        super(TransformerDBB, self).__init__()
        self.in_channels = 64
        self.use_gn = use_gn
        self.use_concat = use_concat

        # Backbone 1
        self.conv1_a = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1_a = nn.GroupNorm(32, 64)
        else:
            self.norm1_a = nn.BatchNorm2d(64)
        
        self.layer1_a = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_a = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_a = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_a = self._make_layer(block, 512, layers[3], stride=2)

        # Backbone 2
        self.conv1_b = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if use_gn:
            self.norm1_b = nn.GroupNorm(32, 64)
        else:
            self.norm1_b = nn.BatchNorm2d(64)

        self.layer1_b = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_b = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_b = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_b = self._make_layer(block, 512, layers[3], stride=2)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_concat:
            self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, use_gn=self.use_gn))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, use_gn=self.use_gn))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # Backbone 1
        out1 = F.relu(self.norm1_a(self.conv1_a(x1)))
        out1 = self.layer1_a(out1)
        out1 = self.layer2_a(out1)
        out1 = self.layer3_a(out1)
        out1 = self.layer4_a(out1)
        out1 = self.adaptive_pool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.dropout(out1)

        # Backbone 2
        out2 = F.relu(self.norm1_b(self.conv1_b(x2)))
        out2 = self.layer1_b(out2)
        out2 = self.layer2_b(out2)
        out2 = self.layer3_b(out2)
        out2 = self.layer4_b(out2)
        out2 = self.adaptive_pool(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.dropout(out2)

        # Combine outputs
        if self.use_concat:
            out = torch.cat((out1, out2), dim=1)
        else:
            out = out1 + out2

        out = self.fc(out)
        return out

def TransformerDuo(num_blocks=[3, 4, 6, 3], num_inputs=1, num_classes=2,use_gn=True, use_concat=False, dropout_rate=0.5):
    # Here we configure fewer blocks for a lighter model
    return TransformerDBB(BottleneckTransformerBlock, num_blocks,num_inputs, num_classes=num_classes, use_gn=use_gn, use_concat=use_concat, dropout_rate=dropout_rate)

# Instancia del modelo
# bottleneck_transformer_duo = TransformerDuo(BottleneckTransformerBlock, [3, 4, 6, 3], num_classes=1, use_gn=True, use_concat=False, dropout_rate=0.5)