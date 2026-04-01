"""
EfficientNet-B0 fine-tuned for emotion classification.

Uses `timm` library for pretrained weights.
Handles grayscale → 3-channel conversion for pretrained models.
"""

import torch
import torch.nn as nn
import timm


class EmotionEfficientNet(nn.Module):
    """
    EfficientNet-B0 fine-tuned for FER-2013.

    Key decisions:
    - Uses pretrained ImageNet weights (transfer learning)
    - Converts 1-channel grayscale to 3-channel by repeating
    - Replaces classifier head with custom head
    - Supports backbone freezing for initial epochs
    """

    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # remove classifier, we add our own
        )

        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 1280 for B0

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def freeze_backbone(self):
        """Freeze backbone weights — only train classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🧊 Backbone frozen — only training classifier head")

    def unfreeze_backbone(self):
        """Unfreeze all weights for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen — full fine-tuning active")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle grayscale: repeat 1 channel → 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)  # (B, 1280)
        logits = self.classifier(features)  # (B, num_classes)
        return logits


if __name__ == "__main__":
    model = EmotionEfficientNet(num_classes=7, pretrained=False)

    # Test with grayscale input (what FER-2013 gives us)
    dummy_gray = torch.randn(4, 1, 224, 224)
    out = model(dummy_gray)
    print(f"Grayscale Input:  {dummy_gray.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test freeze/unfreeze
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (frozen): {trainable:,}")

    model.unfreeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params (unfrozen): {trainable:,}")
