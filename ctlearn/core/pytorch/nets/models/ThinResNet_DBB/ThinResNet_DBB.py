import torch
import torch.nn as nn
import torch.nn.functional as F
from ctlearn.core.pytorch.nets.block.cnn_blocks import NormalInvGamma

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
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # Ensuring dimension match
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, reduction=16,use_bn=True):


        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else: 
            self.bn1 = nn.Identity()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.Identity()

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
        out = self.se(out)
        return out
    
class ThinResNet_DBB(nn.Module):
    def __init__(self,task, block=BasicBlock, num_blocks=[2, 3, 3, 3], num_inputs=1, num_outputs=2,use_bn=False,dropout=0.0):
        super(ThinResNet_DBB, self).__init__()

        # block = BasicBlock
        self.in_channels = 64
        self.use_bn=use_bn
        self.task = task
        self.conv1 = nn.Conv2d(num_inputs, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.bn1 = nn.Identity()

        self.layer1_1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_1 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_1 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Reducing the number of layers and filters to make it "thin"
        self.fc_1 = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.fc_2 = nn.Linear(512 * block.expansion, num_outputs)
        self.bn_final = nn.BatchNorm1d(512 * block.expansion)  # BatchNorm layer
        self.prelu = nn.PReLU(num_parameters=512 * block.expansion)  # Define Leaky ReLU
        if self.task == "direction":
            self.normal_inv = NormalInvGamma(512 * block.expansion,num_outputs)

        # self.fc_1_separation = nn.Linear(512 * block.expansion, 512 * block.expansion)
        # self.fc_2_separation = nn.Linear(512 * block.expansion, 4)

        self.in_channels = 64
        self.conv2 = nn.Conv2d(num_inputs, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_2 = self._make_layer(block, 256, num_blocks[2], stride=2)
        
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(64)
        else:
            self.bn2 = nn.Identity()        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.dropout = nn.Dropout(dropout)
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_bn=self.use_bn))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y):

        energy = None
        classification = None
        direction = None

        # out_1 = F.relu(self.bn1(self.conv1(x)))    
        out_1 = F.relu(self.conv1(x)) 
        out_1 = self.layer1_1(out_1)
        out_1 = self.layer2_1(out_1)
        out_1 = self.layer3_1(out_1)

        # out_2 = F.relu(self.bn2(self.conv2(y)))
        out_2 = F.relu((self.conv2(y)))        
        out_2 = self.layer1_2(out_2)
        out_2 = self.layer2_2(out_2)
        out_2 = self.layer3_2(out_2)
        out = out_1 + out_2 

        # out = self.layer3(out)
        out = self.layer4(out)
        out = self.adaptive_pool(out)
        out_feature = out.view(out.size(0), -1)
        out = self.dropout(out_feature) 

        # if self.training:
        #     out_sep = self.fc_1_separation(out)
        #     out_sep = self.fc_2_separation(out_sep)
        
        out = self.fc_1(out)
        # out = self.bn_final(out)
        # out = self.prelu(out)
        # Original
        # out = self.fc_2(out)

       
        if self.task == "type":
            out = self.fc_2(out)
            classification = out            
        if self.task == "energy":
            out = self.fc_2(out)
            # energy = [out, out_feature]
            energy = out

        if self.task == "direction":
            direction = self.normal_inv(out)

            # direction = [direction, out_feature]
            # if self.training:
            #     out = self.normal_inv(out)
            #     direction = out
            # else: 
            #     direction = self.normal_inv(out)
                

        # if self.training:
        #     out = torch.cat((out, out_sep), dim=1)
            
        return classification, energy, direction

# def thin_resnet34(num_blocks=[2, 3, 3, 3], num_inputs=1, num_classes=2):
#     # Here we configure fewer blocks for a lighter model
#     return ThinResNet_DBB(BasicBlock, num_blocks,num_inputs,num_classes)

# def create_model(num_blocks=[2, 3, 3, 3], num_inputs=1, num_classes=2, use_bn=False, dropout=0.0):
#     return ThinResNet_DBB(
#         block=BasicBlock,
#         num_blocks=num_blocks,
#         num_inputs=num_inputs,
#         num_classes=num_classes,
#         use_bn=use_bn,
#         dropout=dropout
#     )

# model = thin_resnet34()
# print(model)

# # Set the model to evaluation mode (as we are just testing with a forward pass)
# model.eval()

# # Create dummy input tensors
# # Assuming the input images are 224x224 pixels with 1 input channel (grayscale)
# x = torch.randn(1, 1, 224, 224)  # Batch size of 1
# y = torch.randn(1, 1, 224, 224)  # Batch size of 1

# # Forward pass through the model
# with torch.no_grad():  # We don't need to calculate gradients here
#     output = model(x, y)

# # Print the output tensor
# print("Output:", output)