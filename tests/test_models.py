"""Tests for BaselineCNN and EmotionEfficientNet model architectures."""

import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet
from src.data.dataset import NUM_CLASSES


# ── BaselineCNN ───────────────────────────────────────────────────────────────

class TestBaselineCNN:
    def setup_method(self):
        self.model = BaselineCNN(num_classes=NUM_CLASSES)
        self.model.eval()

    def test_output_shape_48(self):
        """Standard 48×48 FER-2013 input produces correct output shape."""
        x = torch.randn(4, 1, 48, 48)
        out = self.model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_output_shape_224(self):
        """Global avg pool means any resolution works without error."""
        x = torch.randn(2, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_batch_size_1(self):
        """Single-sample batch works (no batch norm issues)."""
        x = torch.randn(1, 1, 48, 48)
        out = self.model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_param_count(self):
        """Model should have ~1.2M parameters."""
        params = sum(p.numel() for p in self.model.parameters())
        assert 1_000_000 < params < 2_000_000, f"Unexpected param count: {params:,}"

    def test_output_is_logits_not_probs(self):
        """Output should be raw logits — not softmax probabilities."""
        x = torch.randn(4, 1, 48, 48)
        out = self.model(x)
        # Logits are not constrained to [0,1] or sum to 1
        assert not (out >= 0).all() or out.sum(dim=1).abs().max() > 0.1

    def test_no_nan_in_output(self):
        """Forward pass should not produce NaN values."""
        x = torch.randn(4, 1, 48, 48)
        out = self.model(x)
        assert not torch.isnan(out).any()

    def test_freeze_unfreeze_not_applicable(self):
        """BaselineCNN has no backbone — freeze/unfreeze methods should not exist."""
        assert not hasattr(self.model, "freeze_backbone")
        assert not hasattr(self.model, "unfreeze_backbone")


# ── EmotionEfficientNet ───────────────────────────────────────────────────────

class TestEmotionEfficientNet:
    def setup_method(self):
        # pretrained=False to avoid downloading weights in tests
        self.model = EmotionEfficientNet(num_classes=NUM_CLASSES, pretrained=False)
        self.model.eval()

    def test_output_shape_grayscale(self):
        """Grayscale (1-channel) input is internally expanded to 3 channels."""
        x = torch.randn(4, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_output_shape_rgb(self):
        """3-channel input also works directly."""
        x = torch.randn(4, 3, 224, 224)
        out = self.model(x)
        assert out.shape == (4, NUM_CLASSES)

    def test_batch_size_1(self):
        x = torch.randn(1, 1, 224, 224)
        out = self.model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_param_count(self):
        """Model should have ~4.7M parameters."""
        params = sum(p.numel() for p in self.model.parameters())
        assert 4_000_000 < params < 6_000_000, f"Unexpected param count: {params:,}"

    def test_no_nan_in_output(self):
        x = torch.randn(4, 1, 224, 224)
        out = self.model(x)
        assert not torch.isnan(out).any()

    def test_freeze_backbone(self):
        """Freezing backbone makes only classifier trainable."""
        self.model.freeze_backbone()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        frozen = [p for p in self.model.backbone.parameters() if p.requires_grad]
        assert len(frozen) == 0, "Backbone params should all be frozen"
        assert len(trainable) > 0, "Classifier params should still be trainable"

    def test_unfreeze_backbone(self):
        """After unfreezing, all params are trainable."""
        self.model.freeze_backbone()
        self.model.unfreeze_backbone()
        frozen = [p for p in self.model.parameters() if not p.requires_grad]
        assert len(frozen) == 0, "All params should be trainable after unfreeze"

    def test_freeze_reduces_trainable_params(self):
        """Freezing backbone significantly reduces trainable parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        self.model.freeze_backbone()
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # Classifier is much smaller than backbone
        assert trainable < total * 0.2, "Frozen model should have <20% trainable params"
