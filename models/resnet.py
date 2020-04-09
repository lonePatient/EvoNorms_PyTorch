import torch.nn as nn
import torch.nn.functional as F
from .normalization import BatchNorm2dRelu,EvoNorm2dB0,EvoNorm2dS0

def get_norm(channels,norm_type):
    if norm_type == 'bn':
        norm_layer = BatchNorm2dRelu(channels)
    elif norm_type == 'enb0':
        norm_layer = EvoNorm2dB0(channels)
    else:
        norm_layer = EvoNorm2dS0(channels)
    return norm_layer


class PreActBasic(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride,norm_type):
        super().__init__()
        self.residual = nn.Sequential(
            get_norm(in_channels, norm_type),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            get_norm(out_channels, norm_type),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBasic.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut

class PreActBottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride,norm_type):
        super().__init__()
        self.residual = nn.Sequential(
            get_norm(in_channels,norm_type),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),
            get_norm(out_channels,norm_type),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            get_norm(out_channels, norm_type),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        shortcut = self.shortcut(x)
        return res + shortcut

class PreActResNet(nn.Module):
    def __init__(self, block, num_block,norm_type='bn',class_num=10):
        super().__init__()
        self.input_channels = 64
        self.norm_type=norm_type
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            get_norm(64,norm_type)
        )
        self.stage1 = self._make_layers(block, num_block[0], 64, 1)
        self.stage2 = self._make_layers(block,num_block[1], 128, 2)
        self.stage3 = self._make_layers(block,num_block[2], 256, 2)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2)
        self.linear = nn.Linear(self.input_channels, class_num)

    def _make_layers(self, block, block_num, out_channels, stride):
        layers = []
        layers.append(block(self.input_channels, out_channels, stride,self.norm_type))
        self.input_channels = out_channels * block.expansion
        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1,self.norm_type))
            self.input_channels = out_channels * block.expansion
            block_num -= 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet18(norm_type):
    return PreActResNet(PreActBasic, [2, 2, 2, 2],norm_type)


def resnet34(norm_type):
    return PreActResNet(PreActBasic, [3, 4, 6, 3],norm_type)


def resnet50(norm_type):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3],norm_type)


def resnet101(norm_type):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3],norm_type)


def resnet152(norm_type):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3],norm_type)