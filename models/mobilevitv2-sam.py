# models/mobilevitv2-sam.py
import torch
import torch.nn as nn
from .mobilevitv2 import AttnWrapper

class SAM(nn.Module):
    """
    Spatial Attention Module 
    """
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

class MobileViTv2_SAM(AttnWrapper):
    def __init__(self, num_classes: int, variant: str = "mobilevitv2_050"):
        super().__init__(num_classes, attention_module=SAM(), variant=variant)
