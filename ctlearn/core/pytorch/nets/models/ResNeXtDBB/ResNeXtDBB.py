import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.se_channels = max(in_channels // reduction_ratio, 1)  # Evitar que los canales sean menos de 1
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, self.se_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.se_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(batch_size, channels)
        # Excitation: Dos capas densas
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        # Recalibrar los canales
        return x * y.expand_as(x)
    
class ResNeXtBlock(nn.Module):
    expansion = 4  # Correcto ajuste del factor de expansión

    def __init__(self, in_channels, out_channels, stride=1, groups=32, use_gn=False, reduction_ratio=16):
        super(ResNeXtBlock, self).__init__()
        
        # Primera capa de convolución
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
        
        # Segunda capa de convolución con agrupación
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.norm2 = nn.GroupNorm(32, out_channels) if use_gn else nn.BatchNorm2d(out_channels)
        
        # Tercera capa de convolución
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(32, out_channels * self.expansion) if use_gn else nn.BatchNorm2d(out_channels * self.expansion)

        # Bloque Squeeze-and-Excitation
        self.se_block = SqueezeExcitation(out_channels * self.expansion, reduction_ratio)

        # Atajo (shortcut) para ajustar el número de canales si es necesario
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, out_channels * self.expansion) if use_gn else nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.prelu_1 = nn.PReLU() 
        self.prelu_2 = nn.PReLU() 
        self.prelu_3 = nn.PReLU() 

    def forward(self, x):
        # Aplicar las capas de convolución y las normalizaciones con ReLU
        out = self.prelu_1(self.norm1(self.conv1(x)))
        out = self.prelu_2(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        
        # Apply Squeeze-and-Excitation
        out = self.se_block(out)
        
        out += self.shortcut(x)
        out = self.prelu_3(out)
        return out

class ResNeXtDBB(nn.Module):
    def __init__(self,task, block, layers, num_inputs=1, num_classes=1, use_gn=False, use_concat=False, dropout_rate=0.5):
        super(ResNeXtDBB, self).__init__()
        self.in_channels = 64
        self.use_gn = use_gn
        self.use_concat = use_concat
        self.task = task
        # Backbone 1
        self.conv1_a = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_a = nn.GroupNorm(32, 64) if use_gn else nn.BatchNorm2d(64)
        
        self.layer1_a = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_a = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_a = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_a = self._make_layer(block, 512, layers[3], stride=2)

        # Backbone 2
        self.in_channels = 64  # Reset in_channels for the second backbone
        self.conv1_b = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1_b = nn.GroupNorm(32, 64) if use_gn else nn.BatchNorm2d(64)

        self.layer1_b = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2_b = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_b = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_b = self._make_layer(block, 512, layers[3], stride=2)

        self.prelu_1 = nn.PReLU() 
        self.prelu_2 = nn.PReLU() 
 
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Ajuste de los canales para la capa completamente conectada
        if self.use_concat:
            self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, use_gn=self.use_gn))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, use_gn=self.use_gn))
        return nn.Sequential(*layers)

    def forward(self, x1, x2):

        classification=None
        energy=None
        direction=None

        # Backbone 1
        # out1 = F.relu(self.norm1_a(self.conv1_a(x1)))
        out1 = self.prelu_1(self.conv1_a(x1))

        out1 = self.layer1_a(out1)
        out1 = self.layer2_a(out1)
        out1 = self.layer3_a(out1)
        out1 = self.layer4_a(out1)
        out1 = self.adaptive_pool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.dropout(out1)

        # Backbone 2
        # out2 = F.relu(self.norm1_b(self.conv1_b(x2)))
        out2 = self.prelu_2(self.conv1_b(x2))
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

        if self.task == "type":
            classification = out
        elif self.task == "energy":
            energy = out
        elif self.task == "direction":
            direction = out

        return classification, energy, direction

def ResNeXtDuo(task,num_blocks=[2, 2, 2, 2], num_inputs=1, num_classes=2, use_gn=True, use_concat=False, dropout_rate=0.5):
    # Here we configure fewer blocks for a lighter model
    return ResNeXtDBB(task,ResNeXtBlock, num_blocks, num_inputs, num_classes=num_classes, use_gn=use_gn, use_concat=use_concat, dropout_rate=dropout_rate)

# Instancia del modelo
# resnext_duo = ResNeXtDuo(ResNeXtBlock, [3, 4, 6, 3], num_classes=1, use_gn=True, use_concat=True, dropout_rate=0.5)