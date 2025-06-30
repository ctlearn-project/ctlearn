import torch
import torch.nn as nn
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

    def __init__(self, in_channels, out_channels, stride=1, reduction=16,use_bn=False):


        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.use_bn = use_bn

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


        self.se = SEBlock(out_channels, reduction)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = (self.conv2(out))    
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.se(out)
        return out
    
class DenoiseBlock(nn.Module):
    def __init__(self, embedding_dim, num_channels=1, block=BasicBlock, num_blocks=[2, 3, 3, 3], num_inputs=1, num_outputs=2,use_bn=False,dropout=0.0):
        super().__init__()

        # block = BasicBlock
        self.in_channels = 64
        self.use_bn=use_bn
 
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1_1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2_1 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3_1 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        # Reducing the number of layers and filters to make it "thin"
        # self.fc_1 = nn.Linear(embedding_dim , embedding_dim)
        # self.fc_2 = nn.Linear(embedding_dim , 256)
        self.bn_final = nn.BatchNorm1d(512 * block.expansion)  # BatchNorm layer
        self.prelu = nn.PReLU(num_parameters=512 * block.expansion)  # Define Leaky ReLU

            
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.dropout = nn.Dropout(dropout)        
   
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
 
        out_1 = F.relu(self.conv1(x)) 
        out_1 = self.layer1_1(out_1)
        out_1 = self.layer2_1(out_1)
        out_1 = self.layer3_1(out_1)

        x_feat = self.layer4(out_1)
        x_feat= self.adaptive_pool(x_feat)
        x_feat = x_feat.view(x_feat.size(0), -1)


        # h1 = self.act1(self.bn_z1(self.fc_z1(z_prev)))
        # h2 = self.act2(self.bn_z2(self.fc_z2(h1)))
        h1 = self.act1((self.fc_z1(z_prev)))
        h2 = self.act2((self.fc_z2(h1)))

        h3 = self.bn_z3(self.fc_z3(h2))
        z_feat = h3 + h1
        h_f = torch.cat([x_feat, z_feat], dim=1)
        # h_f = self.act_f1(self.bn_f1(self.fc_f1(h_f)))
        # h_f = self.act_f2(self.bn_f2(self.fc_f2(h_f)))
        h_f = self.act_f1((self.fc_f1(h_f)))
        h_f = self.act_f2((self.fc_f2(h_f)))        
        z_next = self.fc_out(h_f)
        return z_next, None

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_bn=self.use_bn))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
