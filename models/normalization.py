import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

def instance_std(x, eps=1e-5):
    N,C,H,W = x.size()
    x1 = x.reshape(N*C,-1)
    var = x1.var(dim=-1, keepdim=True)+eps
    return var.sqrt().reshape(N,C,1,1)

def group_std(x, groups, eps = 1e-5):
    N, C, H, W = x.size()
    x1 = x.reshape(N,groups,-1)
    var = (x1.var(dim=-1, keepdim = True)+eps).reshape(N,groups,-1)
    return (x1 / var.sqrt()).reshape(N,C,H,W)


class BatchNorm2dRelu(nn.Module):
    def __init__(self,in_channels):
        super(BatchNorm2dRelu,self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        output = self.layer(x)
        return output


class EvoNorm2dB0(nn.Module):
    def __init__(self,in_channels,nonlinear=True,momentum=0.9,eps = 1e-5):
        super(EvoNorm2dB0, self).__init__()
        self.nonlinear = nonlinear
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter(torch.Tensor(1,in_channels,1,1))
        self.beta = Parameter(torch.Tensor(1,in_channels,1,1))
        if nonlinear:
            self.v = Parameter(torch.Tensor(1,in_channels,1,1))
        self.register_buffer('running_var', torch.ones(1, in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        if self.nonlinear:
            init.ones_(self.v)

    def forward(self, x):
        N, C, H, W = x.size()
        if self.training:
            x1 = x.permute(1, 0, 2, 3).reshape(C, -1)
            var = x1.var(dim=1).reshape(1, C, 1, 1)
            self.running_var.copy_(self.momentum * self.running_var + (1 - self.momentum) * var)
        else:
            var = self.running_var
        if self.nonlinear:
            den = torch.max((var+self.eps).sqrt(), self.v * x + instance_std(x))
            return x / den * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


class EvoNorm2dS0(nn.Module):
    def __init__(self,in_channels,groups=8,nonlinear=True):
        super(EvoNorm2dS0, self).__init__()
        self.nonlinear = nonlinear
        self.groups = groups
        self.gamma = Parameter(torch.Tensor(1,in_channels,1,1))
        self.beta = Parameter(torch.Tensor(1,in_channels,1,1))
        if nonlinear:
            self.v = Parameter(torch.Tensor(1,in_channels,1,1))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        if self.nonlinear:
            init.ones_(self.v)
    def forward(self, x):
        if self.nonlinear:
            num = torch.sigmoid(self.v * x)
            std = group_std(x,self.groups)
            return num * std * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta