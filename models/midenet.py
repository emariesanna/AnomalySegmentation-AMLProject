import torch
import torch.nn as nn
import torch.nn.functional as F

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_out = self.conv(x)
        
        pool_out = self.pool(x)
        
        out = torch.cat([conv_out, pool_out], dim=1)
        return self.relu(self.bn(out))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dilation=1):
        super().__init__()
        stride = 2 if downsample else 1
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.match_channels = None
        if in_channels != out_channels or downsample:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.match_channels:
            residual = self.match_channels(residual)
        return self.relu(out + residual)

class MidENet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.initial = DownsampleBlock(3, 16)
        self.bottleneck1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True),
            Bottleneck(64, 64),
            Bottleneck(64, 64)
        )
        self.bottleneck2 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128, dilation=2),
            Bottleneck(128, 128, dilation=4)
        )
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)  # restore input resolution
        return self.classifier(x)