# models/mobilevitv2-c2psa.py
import torch
import torch.nn as nn
from .mobilevitv2 import AttnWrapper

class C2PSA(nn.Module):
    """
    Custom Compact Parallel Spatial Attention:

    """
    def __init__(self, kernel_large=7):
        super().__init__()
        padL = kernel_large // 2
        self.branch3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.branchL = nn.Conv2d(2, 1, kernel_large, padding=padL, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        a = torch.cat([avg, mx], dim=1)  # (B,2,H,W)
        w = self.branch3(a) + self.branchL(a)
        w = self.sig(w)
        return x * w

class MobileViTv2_C2PSA(AttnWrapper):
    def __init__(self, num_classes: int, variant: str = "mobilevitv2_050"):
        super().__init__(num_classes, attention_module=C2PSA(), variant=variant)
