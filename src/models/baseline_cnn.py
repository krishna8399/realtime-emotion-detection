"""
Baseline CNN for emotion classification.

A simple 4-block CNN to establish baseline performance.
This is intentionally simple — we want to beat this with EfficientNet.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """
    Simple CNN: 4 conv blocks → global average pooling → classifier.
    Input: (B, 1, 48, 48) grayscale images
    Output: (B, 7) emotion logits
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 → 32 channels, spatial: 48×48 → 24×24 after pooling
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # extract low-level features (edges, textures)
            nn.BatchNorm2d(32),   # normalize activations to stabilize training
            nn.ReLU(inplace=True),  # non-linearity; inplace saves memory
            nn.Conv2d(32, 32, kernel_size=3, padding=1),  # second conv to learn more complex patterns
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # halve spatial dimensions (48→24) by keeping max value in 2×2 window
            nn.Dropout2d(0.25),   # randomly zero entire feature maps to prevent co-adaptation

            # Block 2: 32 → 64 channels, spatial: 24×24 → 12×12 after pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # double channels to learn richer features
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # halve spatial dimensions (24→12)
            nn.Dropout2d(0.25),

            # Block 3: 64 → 128 channels, spatial: 12×12 → 6×6 after pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),   # halve spatial dimensions (12→6)
            nn.Dropout2d(0.25),

            # Block 4: 128 → 256 channels, spatial stays 6×6 (no pooling here)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # deepen representation before pooling
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # collapse any spatial size → (B, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),              # (B, 256, 1, 1) → (B, 256)
            nn.Linear(256, 128),       # first FC layer: reduce 256 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),       # dropout before final layer to reduce overfitting
            nn.Linear(128, num_classes),  # output layer: 128 → 7 class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # pass through conv blocks to extract features
        x = self.global_pool(x)    # pool to fixed size regardless of input resolution
        x = self.classifier(x)     # classify into emotion classes
        return x


if __name__ == "__main__":
    # Quick test
    model = BaselineCNN(num_classes=7)
    dummy = torch.randn(4, 1, 48, 48)  # batch of 4 grayscale 48×48 images
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
