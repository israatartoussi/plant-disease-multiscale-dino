import torch.nn as nn
from dinounet.dinov3.vision_transformer import vit_base

class DinoV3ContextEncoder(nn.Module):
    def __init__(self, pretrained_weights=None):
        super().__init__()
        self.backbone = vit_base()
        if pretrained_weights is not None:
            self.backbone.load_state_dict(
                torch.load(pretrained_weights, map_location="cpu"),
                strict=False
            )

    def forward(self, x):
        # returns dict, we take patch tokens
        out = self.backbone.forward_features(x)
        patch_tokens = out["x_norm_patchtokens"]  # [B, N, D]
        return patch_tokens
