import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re

nfnet_params = {
    'F0': {'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3], 'drop_rate': 0.2},
    'F1': {'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6], 'drop_rate': 0.3},
    'F2': {'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9], 'drop_rate': 0.4},
    'F3': {'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12], 'drop_rate': 0.4},
    'F4': {'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15], 'drop_rate': 0.5},
    'F5': {'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18], 'drop_rate': 0.5},
    'F6': {'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21], 'drop_rate': 0.5},
    'F7': {'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24], 'drop_rate': 0.5},
}

class VPGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input) * 1.7015043497085571

class VPReLU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=True) * 1.7139588594436646

activations_dict = {'gelu': VPGELU(), 'relu': VPReLU()}

class NFNet(nn.Module):
    def __init__(self, num_classes: int, variant: str = 'F0', stochdepth_rate: float = None, alpha: float = 0.2, se_ratio: float = 0.5, activation: str = 'gelu'):
        super(NFNet, self).__init__()
        if variant not in nfnet_params:
            raise RuntimeError(f"Variant {variant} does not exist and could not be loaded.")
        block_params = nfnet_params[variant]
        self.activation = activations_dict[activation]
        self.drop_rate = block_params['drop_rate']
        self.num_classes = num_classes
        self.stem = Stem(activation=activation)
        num_blocks, index = sum(block_params['depth']), 0
        blocks = []
        expected_std = 1.0
        in_channels = block_params['width'][0] // 2
        block_args = zip(block_params['width'], block_params['depth'], [0.5] * 4, [128] * 4, [1, 2, 2, 2])
        for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1. / expected_std
                block_sd_rate = stochdepth_rate * index / num_blocks if stochdepth_rate is not None else 0
                out_channels = block_width
                blocks.append(NFBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride if block_index == 0 else 1,
                    alpha=alpha,
                    beta=beta,
                    se_ratio=se_ratio,
                    group_size=group_size,
                    stochdepth_rate=block_sd_rate,
                    activation=activation
                ))
                in_channels = out_channels
                index += 1
                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5
        self.body = nn.Sequential(*blocks)
        final_conv_channels = 2 * in_channels
        self.final_conv = WSConv2D(in_channels=out_channels, out_channels=final_conv_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(1)
        if self.drop_rate > 0.:
            self.dropout = nn.Dropout(self.drop_rate)
        self.linear = nn.Linear(final_conv_channels, self.num_classes)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        out = self.stem(x)
        out = self.body(out)
        out = self.activation(self.final_conv(out))
        pool = torch.mean(out, dim=(2, 3))
        if self.training and self.drop_rate > 0.:
            pool = self.dropout(pool)
        return self.linear(pool)

class Stem(nn.Module):
    def __init__(self, activation: str = 'gelu'):
        super(Stem, self).__init__()
        self.activation = activations_dict[activation]
        self.conv0 = WSConv2D(in_channels=1, out_channels=16, kernel_size=3, stride=2)  # For grayscale images
        self.conv1 = WSConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = WSConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = WSConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv4 = WSConv2D(in_channels=128, out_channels=128, kernel_size=3, stride=2)  # Additional layer

    def forward(self, x):
        out = self.activation(self.conv0(x))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = self.conv4(out)
        return out
    
class NFBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5, se_ratio: float = 0.5, stride: int = 1, beta: float = 1.0, alpha: float = 0.2, group_size: int = 1, stochdepth_rate: float = None, activation: str = 'gelu'):
        super(NFBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.activation = activations_dict[activation]
        self.beta, self.alpha = beta, alpha
        self.group_size = group_size
        width = int(self.out_channels * expansion)
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride
        self.conv0 = WSConv2D(in_channels=self.in_channels, out_channels=self.width, kernel_size=1)
        self.conv1 = WSConv2D(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=stride, padding=1, groups=self.groups)
        self.conv1b = WSConv2D(in_channels=self.width, out_channels=self.width, kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.conv2 = WSConv2D(in_channels=self.width, out_channels=self.out_channels, kernel_size=1)
        self.use_projection = self.stride > 1 or self.in_channels != self.out_channels
        if self.use_projection:
            self.shortcut = nn.Sequential()
            if self.stride > 1:
                self.shortcut.add_module('avg_pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0 if self.in_channels == 1536 else 1))
            self.shortcut.add_module('conv', WSConv2D(self.in_channels, self.out_channels, kernel_size=1))
        self.squeeze_excite = SqueezeExcite(self.out_channels, self.out_channels, se_ratio=self.se_ratio, activation=activation)
        self.skip_gain = nn.Parameter(torch.zeros(()))
        self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0. and stochdepth_rate < 1.
        if self.use_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)

    def forward(self, x):
        out = self.activation(x) * self.beta
        if self.use_projection:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out = self.activation(self.conv0(out))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv1b(out))
        out = self.conv2(out)
        out = (self.squeeze_excite(out) * 2) * out
        if self.use_stochdepth:
            out = self.stoch_depth(out)
        return out * self.alpha * self.skip_gain + shortcut
    
class WSConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(WSConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer('eps', torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer('fan_in', torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight), persistent=False)

    def standardized_weights(self):
        mean = torch.mean(self.weight, axis=[1,2,3], keepdims=True)
        var = torch.var(self.weight, axis=[1,2,3], keepdims=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain
        
    def forward(self, x):
        return F.conv2d(x, self.standardized_weights(), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, out_channels, se_ratio, activation='relu'):
        super(SqueezeExcite, self).__init__()
        self.se_reduce = nn.Conv2d(in_channels, int(in_channels * se_ratio), 1)
        self.se_expand = nn.Conv2d(int(in_channels * se_ratio), out_channels, 1)
        self.activation = activations_dict[activation]

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.se_reduce(scale)
        scale = self.activation(scale)
        scale = self.se_expand(scale)
        scale = torch.sigmoid(scale)
        return x * scale

class StochDepth(nn.Module):
    def __init__(self, p: float):
        super(StochDepth, self).__init__()
        self.prob = p

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.prob:
            return x * 0
        return x