import torch
import torch.nn as nn
from .mobilevitv2 import AttnWrapper

class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        return x * self.sig(avg + mx)

class SpatialGate(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 8)
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size, padding=pad, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return x * self.sig(self.conv(x))

class BAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.cg = ChannelGate(channels, reduction)
        self.sg = SpatialGate(channels, reduction, kernel_size)
    def forward(self, x):
        return self.sg(self.cg(x))

class MobileViTv2_BAM(AttnWrapper):
    def __init__(self, num_classes: int, variant: str = "mobilevitv2_050"):
        tmp = nn.Identity()
        super().__init__(num_classes, attention_module=tmp, variant=variant)
        channels = self.head.in_features
        self.attn = BAM(channels)
