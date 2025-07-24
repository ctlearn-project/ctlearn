import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.coord_conv import CoordConvTh
# from lib.dataset import get_decoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=not bn)
        self.bn = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim=None):
        super().__init__()
        mid_dim = mid_dim or out_dim // 2
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, bn=True, relu=True)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, bn=True, relu=True)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, bn=True, relu=False)
        self.skip = ConvBlock(inp_dim, out_dim, 1, bn=True, relu=False) if inp_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return self.relu(out + self.skip(x))

class Hourglass(nn.Module):
    def __init__(self, n, f):
        super().__init__()
        self.up1 = ResBlock(f, f)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResBlock(f, f)
        if n > 1:
            self.low2 = Hourglass(n - 1, f)
        else:
            self.low2 = ResBlock(f, f)
        self.low3 = ResBlock(f, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        up1 = self.up1(x)
        low1 = self.low1(self.pool1(x))
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        if up2.shape[-2:] != up1.shape[-2:]:
            up2 = F.interpolate(up2, size=up1.shape[-2:], mode='nearest') 
        return up1 + up2

class StackedHGNet(nn.Module):
    def __init__(self,task, input_channels=3, nstack=2, nlevels=4, in_channel=256, output_dim=3, use_bn=True, use_stn=False):
        super().__init__()
        self.task = task
        self.nstack = nstack
        self.pre = nn.Sequential(
            ConvBlock(input_channels, 64, 7, 2, bn=use_bn, relu=True),
            ResBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResBlock(128, 128),
            ResBlock(128, in_channel)
        )
        self.hgs = nn.ModuleList([
            Hourglass(nlevels, in_channel) for _ in range(nstack)
        ])
        self.features = nn.ModuleList([
            nn.Sequential(
                ResBlock(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=use_bn, relu=True)
            ) for _ in range(nstack)
        ])
        # self.out_regression = nn.ModuleList([
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Flatten(),
        #         nn.Linear(in_channel, output_dim)
        #     ) for _ in range(nstack)
        # ])

        self.out_regression = nn.ModuleList([
        nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channel, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, output_dim)
        ) for _ in range(nstack)
        ])
        
        # self.out_regression = nn.ModuleList([
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Flatten(),
        #         nn.Linear(in_channel, 128),
        #         nn.ReLU(),
        #         nn.BatchNorm1d(128),
        #         nn.Linear(128, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, output_dim)
        #     ) for _ in range(nstack)
        # ])
        # Head auxiliar de heatmap (un canal)
        # self.out_heatmap = nn.ModuleList([
        #     ConvBlock(in_channel, 1, 1, relu=False, bn=False) for _ in range(nstack)
        # ])
        self.merge_features = nn.ModuleList([
            ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
            for _ in range(nstack - 1)
        ])

    def forward(self, x):
        x = self.pre(x)
        regression_outputs = []
        # heatmap_outputs = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            regression_output = self.out_regression[i](feature)
            # heatmap_output = self.out_heatmap[i](feature)
            regression_outputs.append(regression_output)
            # heatmap_outputs.append(heatmap_output)
            if i < self.nstack - 1:
                x = x + self.merge_features[i](feature)

        if self.task=="direction":
            return None, None, regression_outputs[-1]
        elif self.task=="energy":
            return None, regression_outputs[-1], None
        else:
            raise ValueError(f"No implemented")
