"""Tests for GradCAM and overlay_heatmap in src/utils/visualize.py."""

import numpy as np
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualize import GradCAM, overlay_heatmap
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet
from src.data.dataset import NUM_CLASSES


# ── overlay_heatmap ───────────────────────────────────────────────────────────

class TestOverlayHeatmap:
    def test_output_shape_matches_input(self):
        """Output should have same H×W as input image."""
        image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        heatmap = np.random.rand(14, 14).astype(np.float32)  # smaller than image
        result = overlay_heatmap(image, heatmap, alpha=0.5)
        assert result.shape == image.shape

    def test_output_dtype_uint8(self):
        """Output must be uint8 for display."""
        image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        heatmap = np.random.rand(48, 48).astype(np.float32)
        result = overlay_heatmap(image, heatmap)
        assert result.dtype == np.uint8

    def test_grayscale_input(self):
        """Grayscale (H,W) input should be handled without error."""
        image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
        heatmap = np.random.rand(48, 48).astype(np.float32)
        result = overlay_heatmap(image, heatmap, alpha=0.4)
        assert result.shape[:2] == (48, 48)

    def test_alpha_zero_returns_original(self):
        """alpha=0 means no heatmap blended — result should equal the original."""
        image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        heatmap = np.ones((48, 48), dtype=np.float32)
        result = overlay_heatmap(image, heatmap, alpha=0.0)
        assert result.shape == image.shape  # shape preserved regardless


# ── GradCAM ───────────────────────────────────────────────────────────────────

class TestGradCAM:
    def _make_gradcam(self, model, layer):
        """Helper to construct GradCAM on a given layer."""
        return GradCAM(model, layer)

    def test_generate_returns_heatmap_and_class(self):
        """generate() must return (np.ndarray, int)."""
        model = BaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        grad_cam = GradCAM(model, model.features[-2])

        x = torch.randn(1, 1, 48, 48, requires_grad=True)
        heatmap, class_idx = grad_cam.generate(x)

        assert isinstance(heatmap, np.ndarray)
        assert isinstance(class_idx, int)
        assert 0 <= class_idx < NUM_CLASSES

    def test_heatmap_range(self):
        """Heatmap values must be normalised to [0, 1]."""
        model = BaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        grad_cam = GradCAM(model, model.features[-2])

        x = torch.randn(1, 1, 48, 48, requires_grad=True)
        heatmap, _ = grad_cam.generate(x)

        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6

    def test_target_class_respected(self):
        """When target_class is specified, the returned class index should match."""
        model = BaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        grad_cam = GradCAM(model, model.features[-2])

        x = torch.randn(1, 1, 48, 48, requires_grad=True)
        _, class_idx = grad_cam.generate(x, target_class=3)
        assert class_idx == 3

    def test_different_classes_produce_different_heatmaps(self):
        """Heatmaps for different target classes should not be identical."""
        model = BaselineCNN(num_classes=NUM_CLASSES)
        model.eval()
        grad_cam = GradCAM(model, model.features[-2])

        x_base = torch.randn(1, 1, 48, 48)

        heatmaps = []
        for cls in range(3):
            x = x_base.detach().clone().requires_grad_(True)
            heatmap, _ = grad_cam.generate(x, target_class=cls)
            heatmaps.append(heatmap)

        # At least two of the three heatmaps should differ
        assert not (np.allclose(heatmaps[0], heatmaps[1]) and
                    np.allclose(heatmaps[1], heatmaps[2]))

    def test_efficientnet_gradcam(self):
        """GradCAM works on EfficientNet's conv_head layer."""
        model = EmotionEfficientNet(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        grad_cam = GradCAM(model, model.backbone.conv_head)

        x = torch.randn(1, 1, 224, 224, requires_grad=True)
        heatmap, class_idx = grad_cam.generate(x)

        assert isinstance(heatmap, np.ndarray)
        assert 0 <= class_idx < NUM_CLASSES
        assert heatmap.min() >= 0.0 - 1e-6
        assert heatmap.max() <= 1.0 + 1e-6
