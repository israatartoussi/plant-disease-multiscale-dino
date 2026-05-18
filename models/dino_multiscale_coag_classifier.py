import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DINO_REPO = ROOT / "third_party" / "dinov3_repo"
if str(DINO_REPO) not in sys.path:
    sys.path.insert(0, str(DINO_REPO))

from dinounet.dinov3.models.vision_transformer import vit_small


# ─── Attention Gate ───────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    AG(x, g): gates x using g as the gating signal.
    x, g: [B, C, H, W]
    """
    def __init__(self, in_channels: int, inter_channels: int = None):
        super().__init__()
        inter_channels = inter_channels or max(in_channels // 2, 1)

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.conv_psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        alpha = self.conv_psi(self.relu(self.conv_x(x) + self.conv_g(g)))
        return x * alpha


# ─── Channel Attention ────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(avg + mx)


# ─── Spatial Attention ────────────────────────────────────────────────────────
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


# ─── CBAM ─────────────────────────────────────────────────────────────────────
class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_att(self.channel_att(x))


# ─── CoAG block ───────────────────────────────────────────────────────────────
class CoAG(nn.Module):
    """
    Co-Attention Gate between two feature maps Si and Sj.

    Steps (following Figure 2.a of MambaCAFU):
        1. AG(Si, Sj)  — gate Si using Sj
        2. AG(Sj, Si)  — gate Sj using Si
        3. Concat( AG(Si,Sj), AG(Sj,Si) )   → [B, 2C, H, W]
        4. Channel Attention                  → [B, 2C, H, W]
        5. 1×1 Conv to project back to C      → [B, C,  H, W]

    Input:  Si, Sj  [B, C, H, W]
    Output: fused   [B, C, H, W]
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.ag_ij = AttentionGate(in_channels)   # gates Si using Sj
        self.ag_ji = AttentionGate(in_channels)   # gates Sj using Si

        self.ca    = ChannelAttention(in_channels * 2, reduction=reduction)
        self.proj  = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, si: torch.Tensor, sj: torch.Tensor) -> torch.Tensor:
        gated_ij = self.ag_ij(si, sj)           # Si gated by Sj
        gated_ji = self.ag_ji(sj, si)           # Sj gated by Si
        fused    = torch.cat([gated_ij, gated_ji], dim=1)   # [B, 2C, H, W]
        fused    = self.ca(fused)
        fused    = self.proj(fused)             # [B, C, H, W]
        return fused


# ─── DINOv3 Encoder ───────────────────────────────────────────────────────────
class DinoV3ContextEncoder(nn.Module):
    """
    Returns intermediate patch-token features from selected transformer blocks.
    Output: [S1, S2, S3, S4], each [B, N, D]
    """
    def __init__(self, pretrained_weights=None,
                 out_indices=(2, 5, 8, 11),
                 freeze_backbone=False):
        super().__init__()
        self.backbone   = vit_small()
        self.out_indices = list(out_indices)
        self.embed_dim   = getattr(self.backbone, "embed_dim", 384)

        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print(f"[DINOv3] Missing: {len(missing)}  Unexpected: {len(unexpected)}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        if x.dim() == 4:
            if x.shape[1] == self.embed_dim:
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == self.embed_dim:
                x = x.flatten(1, 2)
        # else already [B, N, C]

        if hasattr(self.backbone, "cls_token") and self.backbone.cls_token is not None:
            x = torch.cat((self.backbone.cls_token.expand(B, -1, -1), x), dim=1)

        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            if self.backbone.pos_embed.shape[1] == x.shape[1]:
                x = x + self.backbone.pos_embed

        if hasattr(self.backbone, "pos_drop") and self.backbone.pos_drop is not None:
            x = self.backbone.pos_drop(x)

        feats = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.out_indices:
                feats.append(x[:, 1:, :] if x.shape[1] > 1 else x)

        return feats


# ─── Main classifier ──────────────────────────────────────────────────────────
class DinoMultiScaleClassifier(nn.Module):
    """
    5th solution — CoAG Consecutive Fusion

    Pipeline:
        Input
          -> DINOv3 backbone  (frozen)
          -> S1, S2, S3, S4   [B, N, D]
          -> Reshape to        [B, D, H, W]
          -> CoAG(S1,S2), CoAG(S2,S3), CoAG(S3,S4)
          -> Concat[ S1, CoAG12, CoAG23, CoAG34 ]   [B, 4D, H, W]
          -> CBAM
          -> 1x1 Conv + GAP + FC
          -> logits
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_weights: str = None,
        freeze_backbone: bool = False,
        out_indices=(2, 5, 8, 11),
        cbam_reduction: int = 16,
        coag_reduction: int = 16,
        classifier_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = DinoV3ContextEncoder(
            pretrained_weights=pretrained_weights,
            out_indices=out_indices,
            freeze_backbone=freeze_backbone,
        )

        D = self.encoder.embed_dim       # 384 for vit_small

        # Three CoAG blocks for consecutive scale pairs
        self.coag_12 = CoAG(D, reduction=coag_reduction)   # S1 ↔ S2
        self.coag_23 = CoAG(D, reduction=coag_reduction)   # S2 ↔ S3
        self.coag_34 = CoAG(D, reduction=coag_reduction)   # S3 ↔ S4

        # After concat: [S1, CoAG12, CoAG23, CoAG34] → 4D channels
        fused_dim = D * 4

        self.cbam = CBAM(fused_dim, reduction=cbam_reduction)

        self.classifier = nn.Sequential(
            nn.Conv2d(fused_dim, classifier_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(classifier_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_map(f: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, D, H, W]"""
        B, N, D = f.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"N={N} is not a perfect square"
        return f.transpose(1, 2).reshape(B, D, H, W)

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)                      # [S1,S2,S3,S4] [B,N,D]
        assert len(feats) == 4

        s1, s2, s3, s4 = [self._to_map(f) for f in feats]   # [B,D,16,16]

        coag12 = self.coag_12(s1, s2)   # [B, D, 16, 16]
        coag23 = self.coag_23(s2, s3)
        coag34 = self.coag_34(s3, s4)

        # S1 kept as anchor (low-level reference)
        fused = torch.cat([s1, coag12, coag23, coag34], dim=1)  # [B, 4D, 16, 16]

        fused  = self.cbam(fused)
        logits = self.classifier(fused)
        return logits