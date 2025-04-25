import torch
from torch import nn

Eff_V2_SETTINGS = {
    # expansion factor, k, stride, n_in, n_out, num_layers, use_fusedMBCONV
    's' : [
        [1, 3, 1, 24, 24, 2, True],
        [4, 3, 2, 24, 48, 4, True],
        [4, 3, 2, 48, 64, 4, True],
        [4, 3, 2, 64, 128, 6, False],
        [6, 3, 1, 128, 160, 9, False],
        [6, 3, 2, 160, 256, 15, False]
    ],
    
    'm' : [
        [1, 3, 1, 24, 24, 3, True],
        [4, 3, 2, 24, 48, 5, True],
        [4, 3, 2, 48, 80, 5, True],
        [4, 3, 2, 80, 160, 7, False],
        [6, 3, 1, 160, 176, 14, False],
        [6, 3, 2, 176, 304, 18, False],
        [6, 3, 1, 304, 512, 5, False]
    ],
    
    'l' : [
        [1, 3, 1, 32, 32, 4, True],
        [4, 3, 2, 32, 64, 7, True],
        [4, 3, 2, 64, 96, 7, True],
        [4, 3, 2, 96, 192, 10, False],
        [6, 3, 1, 192, 224, 19, False],
        [6, 3, 2, 224, 384, 25, False],
        [6, 3, 1, 384, 640, 7, False]
    ]
}

class ConvBnAct(nn.Module):
    
    def __init__(
        self,
        n_in, # in_channels
        n_out, # out_channels
        k_size = 3, # Kernel Size
        stride = 1, 
        padding = 0,
        groups = 1, 
        act = True, 
        bn = True, 
        bias = False
    ):
        super(ConvBnAct, self).__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = k_size, stride = stride,
                              padding = padding, groups = groups,bias = bias
                             )
        self.batch_norm = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.activation = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        
        return x
    
#--------------------------------------------------------------------------------------------

'''Squeeze and Excitation Class'''

class SqueezeExcitation(nn.Module):
    
    def __init__(
        self,
        n_in, # In_channels
        reduced_dim
    ):
        super(SqueezeExcitation, self).__init__()
      
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(nn.Conv2d(n_in, reduced_dim, kernel_size=1),
                                   nn.SiLU(),
                                   nn.Conv2d(reduced_dim, n_in, kernel_size=1),
                                   nn.Sigmoid()
                                   )
        
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excite(y)
            
        return x * y
        
#--------------------------------------------------------------------------------------

''' Stochastic Depth Class'''

class StochasticDepth(nn.Module):
    
    def __init__(
        self,
        survival_prob = 0.8
    ):
        super(StochasticDepth, self).__init__()
        
        self.p =  survival_prob
        
    def forward(self, x):
        
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
        
        return torch.div(x, self.p) * binary_tensor
        
#-------------------------------------------------------------------------------

'''MBCONV Class'''

class MBConvN(nn.Module):
    
    def __init__(
        self,
        n_in, # In_channels
        n_out, # out_channels
        k_size = 3, # kernel_size
        stride = 1,
        expansion_factor = 4,
        reduction_factor = 4, # SqueezeExcitation Block
        survival_prob = 0.8 # StochasticDepth Block
    ):
        super(MBConvN, self).__init__()
        reduced_dim = int(n_in//4)
        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1)//2
        
        self.use_residual = (n_in == n_out) and (stride == 1)
        self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded_dim, k_size = 1)
        self.depthwise_conv = ConvBnAct(expanded_dim, expanded_dim,
                                        k_size, stride = stride,
                                        padding = padding, groups = expanded_dim
                                       )
        self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = ConvBnAct(expanded_dim, n_out, k_size = 1, act = False)
        
    def forward(self, x):
        
        residual = x.clone()
        x = self.expand(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            x += residual
            
        return x
    
#--------------------------------------------------------------------------------------

'''Fused-MBCONV Class'''

class FusedMBConvN(nn.Module):
    
    def __init__(
        self,
        n_in, # In_channels
        n_out, # out_channels
        k_size = 3, # kernel_size
        stride = 1,
        expansion_factor = 4,
        reduction_factor = 4, # SqueezeExcitation Block
        survival_prob = 0.8 # StochasticDepth Block
    ):
        super(FusedMBConvN, self).__init__()
        
        reduced_dim = int(n_in//4)
        expanded_dim = int(expansion_factor * n_in)
        padding = (k_size - 1)//2
        
        self.use_residual = (n_in == n_out) and (stride == 1)
        #self.expand = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded_dim, k_size = 1)
        self.conv = ConvBnAct(n_in, expanded_dim,
                              k_size, stride = stride,
                              padding = padding, groups = 1
                             )
        #self.se = SqueezeExcitation(expanded_dim, reduced_dim)
        self.drop_layers = StochasticDepth(survival_prob)
        self.pointwise_conv = nn.Identity() if (expansion_factor == 1) else ConvBnAct(expanded_dim, n_out, k_size = 1, act = False)
        
    def forward(self, x):
        
        residual = x.clone()
        #x = self.conv(x)
        x = self.conv(x)
        #x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_residual:
            x = self.drop_layers(x)
            x += residual
            
        return x
    
#-----------------------------------------------------------------------------------------------

class EfficientNetV2(nn.Module):
    
    def __init__(
    self,
    version = 's',
    dropout_rate = 0.2,
    in_channels=1,
    num_classes = 1000
    ):
        super(EfficientNetV2, self).__init__()
        last_channel = 1280
        self.features = self._feature_extractor(version,in_channels, last_channel)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace = True),
            nn.Linear(last_channel, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        
        return x
        
    def _feature_extractor(self, version,in_channels, last_channel):
        
        # Extract the Config
        config = Eff_V2_SETTINGS[version]
        
        layers = []
        layers.append(ConvBnAct(in_channels, config[0][3], k_size = 3, stride = 2, padding = 1))
        #in_channel = config[0][3]
        
        for (expansion_factor, k, stride, n_in, n_out, num_layers, use_fused) in config:
            
            if use_fused:
                layers += [FusedMBConvN(n_in if repeat==0 else n_out, 
                                        n_out,
                                        k_size=k,
                                        stride = stride if repeat==0 else 1,
                                        expansion_factor=expansion_factor
                                       ) for repeat in range(num_layers)
                          ]
            else:
                
                layers += [MBConvN(n_in if repeat==0 else n_out, 
                                   n_out,
                                   k_size=k,
                                   stride = stride if repeat==0 else 1,
                                   expansion_factor=expansion_factor
                                   ) for repeat in range(num_layers)
                      ]
                
        layers.append(ConvBnAct(config[-1][4], last_channel, k_size = 1))   
            
        return nn.Sequential(*layers)