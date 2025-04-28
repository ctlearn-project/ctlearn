import torch
import torch.nn as nn
from ctlearn.core.pytorch.net_utils import ModelHelper
import torch.nn.functional as F

class Dirichlet(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        alpha = self.evidence(out) + 1
        return alpha
    
class NormalInvGamma(nn.Module):
    """Defines the Normal Inverse Gamma distribution layer."""
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        # return mu, v, alpha, beta
    
        if self.training:
            return mu, v, alpha, beta
        else:
            var = torch.sqrt(beta / (v * (alpha - 1)))

            return mu, var 
    
class ResBlock(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size=3, conv_drop_pro=0.2):

        super(ResBlock, self).__init__()

        self.conv = nn.Conv2d(n_chans_in, n_chans_in, kernel_size=kernel_size, padding=int(kernel_size/2), bias=False)
        self.conv_dropout = nn.Dropout2d(p=conv_drop_pro)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans_in)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.LeakyReLU()
        self.conv_out = nn.Conv2d(n_chans_in, n_chans_out, kernel_size=1, padding=0, bias=False)

        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)

        #  Init Filters
        kernels = ModelHelper.GaborKernels(size=kernel_size, showPlots=False)
        for i in range(min(self.conv.weight.shape[0], len(kernels))):
            with torch.no_grad():
                self.conv.weight[i, :] = torch.nn.Parameter(torch.tensor(kernels[i]*100))

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = out + x
        out = self.pool(out)
        out = self.conv_out(out)
        out = self.conv_dropout(out)
        return out