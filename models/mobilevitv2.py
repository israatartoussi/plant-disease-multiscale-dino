import torch
import torch.nn as nn
import timm

class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))  # (B,C,H,W)->(B,C)

class MobileViTv2Classifier(nn.Module):
    def __init__(self, num_classes: int, variant: str = "mobilevitv2_050"):
        super().__init__()
        self.backbone = timm.create_model(variant, pretrained=True, num_classes=0, features_only=False)
        self.pool = GlobalAvgPool()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone.forward_features(dummy)  # (1,C,H,W)
            c = feats.shape[1]
        self.head = nn.Linear(c, num_classes)

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        f = self.forward_features(x)
        z = self.pool(f)
        return self.head(z)

class AttnWrapper(nn.Module):

    def __init__(self, num_classes: int, attention_module: nn.Module, variant: str = "mobilevitv2_050"):
        super().__init__()
        self.backbone = timm.create_model(variant, pretrained=True, num_classes=0, features_only=False)
        self.attn = attention_module
        self.pool = GlobalAvgPool()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            c = self.backbone.forward_features(dummy).shape[1]
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        f = self.backbone.forward_features(x)  # (B,C,H,W)
        f = self.attn(f)
        z = self.pool(f)
        return self.head(z)
