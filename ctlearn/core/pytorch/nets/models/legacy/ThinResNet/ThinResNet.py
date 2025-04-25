import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ThinResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,use_bn=False):
        super(ThinResNet, self).__init__()
        self.in_channels = 64
        self.use_bn=use_bn
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.bn1 = nn.Sequential()
            
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # Reducing the number of layers and filters to make it "thin"
        self.linear = nn.Linear(256 * block.expansion, num_classes)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def thin_resnet34():
    # Here we configure fewer blocks for a lighter model
    return ThinResNet(BasicBlock, [2, 2, 2])

# model = thin_resnet34()
# print(model)

# # Set the model to evaluation mode (as we are just testing with a forward pass)
# model.eval()

# # Create dummy input tensors
# # Assuming the input images are 224x224 pixels with 1 input channel (grayscale)
# x = torch.randn(1, 1, 224, 224)  # Batch size of 1

# # Forward pass through the model
# with torch.no_grad():  # We don't need to calculate gradients here
#     output = model(x)

# # Print the output tensor
# print("Output:", output)
