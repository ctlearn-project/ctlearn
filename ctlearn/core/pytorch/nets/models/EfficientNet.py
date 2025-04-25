# import pytorch
from torch import nn
import math
basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, groups=groups),
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)
    

# class MBBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size,
#         stride, padding, expand_ratio, reduction=2,
#     ):
#         super(MBBlock, self).__init__()
#         # self.use_residual = in_channels == out_channels and stride == 1
#         hidden_dim = in_channels * expand_ratio
#         self.expand = in_channels != hidden_dim

#         # This is for squeeze and excitation block
#         reduced_dim = int(in_channels / reduction)

#         if self.expand:
#             self.expand_conv = ConvBlock(in_channels, hidden_dim,
#                 kernel_size=3,stride=1,padding=1)

#         self.conv = nn.Sequential(
#                 ConvBlock(hidden_dim,hidden_dim,kernel_size,
#                   stride,padding,groups=hidden_dim),
#                 SqueezeExcitation(hidden_dim, reduced_dim),
#                 nn.Conv2d(hidden_dim, out_channels, 1),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self, inputs):
#         if self.expand:
#           x = self.expand_conv(inputs)
#         else:
#           x = inputs
#         return self.conv(x)
    
class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, expand_ratio, reduction=4):  # Changed 'ratio' to 'expand_ratio' and adjusted 'reduction'
        super(MBBlock, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim

        reduced_dim = int(hidden_dim / reduction)  # Use 'hidden_dim' not 'in_channels'

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                                         kernel_size=1, stride=1, padding=0)  # Typically a 1x1 conv

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size,
                      stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0),  # Kernel size 1, stride 1, padding 0
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        return self.conv(x)
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)
    

class EfficientNet(nn.Module):
    def __init__(self, model_name, num_channels, output):
        super(EfficientNet, self).__init__()
        self.num_channels = num_channels
        phi, resolution, dropout = scale_values[model_name]
        self.depth_factor, self.width_factor = alpha**phi, beta**phi
        self.last_channels = math.ceil(1280 * self.width_factor)
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channels, output),
        )

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features = [ConvBlock(self.num_channels, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for k, c_o, repeat, s, n in basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * math.ceil(int(c_o * self.width_factor) / 4)
            num_layers = math.ceil(repeat * self.depth_factor)

            for layer in range(num_layers):
                if layer == 0:
                  stride = s
                else:
                  stride = 1
                features.append(
                        MBBlock(in_channels,out_channels,expand_ratio=k,
                        stride=stride,kernel_size=n,padding=n// 2)
                    )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, self.last_channels, 
            kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

    def forward(self, x):
        x = self.avgpool(self.extractor(x))
        return self.classifier(self.flatten(x))    