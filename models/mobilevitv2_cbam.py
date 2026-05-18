import torch
import torch.nn as nn
from .mobilevitv2 import AttnWrapper

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=(2,3))
        mx  = x.amax(dim=(2,3))
        w = self.mlp(avg) + self.mlp(mx)
        w = self.sig(w).unsqueeze(-1).unsqueeze(-1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        a = torch.cat([avg, mx], dim=1)
        w = self.sig(self.conv(a))
        return x * w

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))

class MobileViTv2_CBAM(AttnWrapper):
    def __init__(self, num_classes: int, variant: str = "mobilevitv2_050"):
        tmp = nn.Identity()
        super().__init__(num_classes, attention_module=tmp, variant=variant)
        channels = self.head.in_features
        self.attn = CBAM(channels)
