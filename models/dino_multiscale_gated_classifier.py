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
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "SpatialAttention kernel_size should be 3 or 7"
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class GatedCNN(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        g = self.gate(x)
        return x + f * g


class DinoV3ContextEncoder(nn.Module):
    """
    Returns intermediate DINOv3 features from selected transformer blocks.

    Output:
        feats = [S1, S2, S3, S4]
        each tensor shape: [B, N, D]
    """

    def __init__(
        self,
        pretrained_weights: str = None,
        out_indices=(2, 5, 8, 11),   # 0-based -> layers 3,6,9,12
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = vit_small()
        self.out_indices = list(out_indices)
        self.embed_dim = getattr(self.backbone, "embed_dim", 384)

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print("[DINOv3] Missing keys:", len(missing))
            print("[DINOv3] Unexpected keys:", len(unexpected))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        B = x.shape[0]

        x = self.backbone.patch_embed(x)

        # Convert patch embedding output to tokens [B, N, C]
        if x.dim() == 4:
            if x.shape[1] == self.embed_dim:
                # [B, C, H, W] -> [B, N, C]
                x = x.flatten(2).transpose(1, 2)
            elif x.shape[-1] == self.embed_dim:
                # [B, H, W, C] -> [B, N, C]
                x = x.flatten(1, 2)
            else:
                raise ValueError(f"Unexpected 4D patch_embed output shape: {x.shape}")
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unexpected patch_embed output shape: {x.shape}")

        if hasattr(self.backbone, "cls_token") and self.backbone.cls_token is not None:
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if hasattr(self.backbone, "pos_embed") and self.backbone.pos_embed is not None:
            pos_embed = self.backbone.pos_embed
            if pos_embed.shape[1] == x.shape[1]:
                x = x + pos_embed
            else:
                raise ValueError(
                    f"pos_embed mismatch: pos_embed={pos_embed.shape}, tokens={x.shape}"
                )

        if hasattr(self.backbone, "pos_drop") and self.backbone.pos_drop is not None:
            x = self.backbone.pos_drop(x)

        feats = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.out_indices:
                feats.append(x[:, 1:, :])  # remove CLS

        return feats


class DinoMultiScaleClassifier(nn.Module):
    """
    Gated CNN + CBAM version

    Pipeline:
        Input
          -> DINOv3 backbone
          -> Extract S1,S2,S3,S4
          -> Tokens -> feature maps
          -> Gated CNN on each scale
          -> Resize to same resolution
          -> Concat along channels
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
        classifier_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = DinoV3ContextEncoder(
            pretrained_weights=pretrained_weights,
            out_indices=out_indices,
            freeze_backbone=freeze_backbone,
        )

        self.embed_dim = self.encoder.embed_dim
        self.num_scales = len(out_indices)
        self.fused_dim = self.embed_dim * self.num_scales

        self.gated_blocks = nn.ModuleList([
            GatedCNN(self.embed_dim) for _ in range(self.num_scales)
        ])

        self.cbam = CBAM(
            in_channels=self.fused_dim,
            reduction=cbam_reduction,
            spatial_kernel=7,
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(self.fused_dim, classifier_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(classifier_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def _tokens_to_feature_maps(self, feats):
        maps = []

        for f in feats:
            B, N, D = f.shape
            H = W = int(math.sqrt(N))
            assert H * W == N, f"Patch tokens N={N} is not a square"

            fm = f.transpose(1, 2).reshape(B, D, H, W)
            maps.append(fm)

        return maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)

        assert len(feats) > 0, "No intermediate features were extracted from DINOv3"
        assert len(feats) == self.num_scales, (
            f"Expected {self.num_scales} features, got {len(feats)}"
        )

        maps = self._tokens_to_feature_maps(feats)

        # Gated CNN on each scale
        maps = [
            self.gated_blocks[i](m) for i, m in enumerate(maps)
        ]

        target_hw = maps[0].shape[-2:]
        maps = [
            F.interpolate(m, size=target_hw, mode="bilinear", align_corners=False)
            if m.shape[-2:] != target_hw else m
            for m in maps
        ]

        fused = torch.cat(maps, dim=1)   # [B, 4D, H, W]
        fused = self.cbam(fused)
        logits = self.classifier(fused)

        return logits