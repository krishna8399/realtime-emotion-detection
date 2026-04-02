"""
EfficientNet-B0 fine-tuned for emotion classification.

Uses `timm` library for pretrained weights.
Handles grayscale -> 3-channel conversion for pretrained models.
"""

import torch
import torch.nn as nn
import timm  # PyTorch Image Models library for pretrained architectures


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

        # Load EfficientNet-B0 with ImageNet pretrained weights
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,  # download weights trained on ImageNet-1k
            num_classes=0,          # remove the default 1000-class head; we add our own
        )

        # Get the feature vector size output by the backbone (1280 for EfficientNet-B0)
        self.feature_dim = self.backbone.num_features  # 1280 for B0

        # Custom classifier head: 1280 → 512 → 128 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),  # compress backbone features
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),               # regularize to prevent overfitting on small dataset
            nn.Linear(512, 128),               # further compress
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),           # lighter dropout closer to output
            nn.Linear(128, num_classes),       # final layer outputs one score per emotion class
        )

    def freeze_backbone(self):
        """Freeze backbone weights — only train classifier head."""
        for param in self.backbone.parameters():
            param.requires_grad = False  # prevent gradients from flowing into backbone
        print("Backbone frozen — only training classifier head")

    def unfreeze_backbone(self):
        """Unfreeze all weights for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True  # allow gradients to update backbone weights too
        print("Backbone unfrozen — full fine-tuning active")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EfficientNet expects 3-channel input; repeat grayscale channel 3 times
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)

        features = self.backbone(x)       # extract feature vector: (B, 1280)
        logits = self.classifier(features)  # classify: (B, 1280) → (B, num_classes)
        return logits


if __name__ == "__main__":
    model = EmotionEfficientNet(num_classes=7, pretrained=False)

    # Test with grayscale input (what FER-2013 gives us)
    dummy_gray = torch.randn(4, 1, 224, 224)  # batch of 4 grayscale images at 224×224
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
