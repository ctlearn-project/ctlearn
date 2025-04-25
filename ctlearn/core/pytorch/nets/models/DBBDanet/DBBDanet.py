import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x

class DANet(nn.Module):
    def __init__(self, num_inputs=1, num_classes=2, dropout_rate=0.3):
        super(DANet, self).__init__()
        
        # Basic CNN backbone
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2)
        self.layer3 = self._make_layer(256, 512, 2)
        
        # DANet attention modules
        self.cam = ChannelAttentionModule(512)
        self.sam = SpatialAttentionModule()

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Apply attention modules
        x = self.cam(x) + x  # Channel Attention
        x = self.sam(x) + x  # Spatial Attention
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Dropout before the final fully connected layer
        x = self.fc(x)


        return x
    
class DBBDanet(nn.Module):
    def __init__(self, task , num_inputs=1, num_classes=2, use_concat=False, dropout_rate=0.3):
        super(DBBDanet, self).__init__()    

        self.task = task
        self.use_concat = use_concat
        self.backbone_1 = DANet(num_inputs=num_inputs, num_classes=num_classes, dropout_rate=dropout_rate)
        self.backbone_2 = DANet(num_inputs=num_inputs, num_classes=num_classes, dropout_rate=dropout_rate)

        # num_features = 512
        num_features = self.backbone_1.fc.in_features
        if self.use_concat:
            num_features*=2

        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate,inplace=True)

        self.backbone_1.fc = nn.Identity()
        self.backbone_1.dropout = nn.Identity()

        self.backbone_2.fc = nn.Identity()
        self.backbone_2.dropout = nn.Identity()

    def forward(self, x, y):  
        energy = None
        classification = None
        direction = None

        feature_1 = self.backbone_1(x)        
        feature_2 = self.backbone_2(y)    

        # Combine outputs
        if self.use_concat:
            out = torch.cat((feature_1, feature_2), dim=1)
        else:
            out = feature_1 + feature_2

        out = self.dropout(out)  # Dropout before the final fully connected layer
        out = self.fc(out)

        if self.task == "type":
            classification = out
        elif self.task == "energy":
            energy = out
        elif self.task == "direction":
            direction = out

        return classification, energy, direction    