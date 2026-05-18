import torch
import torch.nn as nn
import timm

from dinounet.dinov3.models.vision_transformer import vit_small
from models.cross_attention import CrossAttentionBlock


class FusionVariant3(nn.Module):
    """
    Variant 3 Fusion:
      - ViT-tiny (timm) as Query encoder -> tokens [B,197,192]
      - DINOv3 ViT-small as Context encoder -> patch tokens [B,196,384]
      - Cross-attention: Q from ViT tokens, K/V from DINO tokens
      - Head uses (CLS + GAP over patches)
    """

    def __init__(self, num_classes: int, freeze_dinov3: bool = True):
        super().__init__()

        # 1) Query encoder
        self.vit_q = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.vit_q.reset_classifier(0)
        self.dim_q = 192

        # 2) Context encoder
        self.dino_ctx = vit_small()
        self.dim_kv = 384

        if freeze_dinov3:
            for p in self.dino_ctx.parameters():
                p.requires_grad = False

        # 3) Cross-attention
        self.fusion = CrossAttentionBlock(dim_q=self.dim_q, dim_kv=self.dim_kv, num_heads=6)

        # 4) Head
        self.cls_norm = nn.LayerNorm(self.dim_q)
        self.gap_norm = nn.LayerNorm(self.dim_q)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim_q * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

        # for Grad-CAM: keep last query tokens feature map
        self._last_query_tokens = None  # [B,197,192] (or refined tokens)

    @torch.no_grad()
    def _dino_tokens(self, x):
        out = self.dino_ctx.forward_features(x)
        return out["x_norm_patchtokens"]  # [B,196,384]

    def _vit_tokens(self, x):
        return self.vit_q.forward_features(x)  # [B,197,192]

    def forward(self, x, return_features: bool = False, return_tokens: bool = False):
        q_tokens = self._vit_tokens(x)               # [B,197,192]
        kv_tokens = self._dino_tokens(x)             # [B,196,384]

        refined = self.fusion(q_tokens, kv_tokens)   # [B,197,192]
        self._last_query_tokens = refined            # save for Grad-CAM

        cls = self.cls_norm(refined[:, 0])           # [B,192]
        gap = self.gap_norm(refined[:, 1:].mean(1))  # [B,192]

        feat = torch.cat([cls, gap], dim=-1)         # [B,384]
        logits = self.classifier(feat)               # [B,num_classes]

        if return_tokens:
            # useful for debugging / visualizations
            return logits, feat, refined, kv_tokens

        if return_features:
            return logits, feat

        return logits
