import timm
import torch.nn as nn

class ViTQueryEncoder(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # no classifier
        )

    def forward(self, x):
        # output: [B, 197, D]
        tokens = self.vit.forward_features(x)
        return tokens
