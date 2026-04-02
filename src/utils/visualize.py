"""
Grad-CAM visualization for emotion classification.

Shows what regions of the face the model focuses on
when making predictions — great for explainability.

Usage:
    python src/utils/visualize.py --checkpoint models/checkpoints/best_efficientnet_b0.pt --image path/to/face.jpg
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import EMOTION_LABELS
from src.models.efficientnet import EmotionEfficientNet
from src.models.baseline_cnn import BaselineCNN


class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping.

    Highlights which regions of the input image are most important
    for the model's prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None   # will be filled by backward hook
        self.activations = None  # will be filled by forward hook

        # Hooks intercept the forward and backward passes at the target layer
        target_layer.register_forward_hook(self._save_activation)        # captures feature maps
        target_layer.register_full_backward_hook(self._save_gradient)    # captures gradients

    def _save_activation(self, module, input, output):
        self.activations = output.detach()  # save feature maps from the target layer

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # save gradients flowing back through target layer

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input image
            target_class: class index to visualize (None = predicted class)

        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        self.model.eval()

        # Forward pass — triggers _save_activation hook
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()  # use predicted class if not specified

        # Backward pass on the target class score — triggers _save_gradient hook
        self.model.zero_grad()
        output[0, target_class].backward()  # compute gradients for target class only

        # Global average pool the gradients over spatial dimensions → importance weight per channel
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)

        # Weighted sum of activation maps: channels with larger gradients contribute more
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # only keep positive contributions (negative = suppressing class)

        # Normalize heatmap to [0, 1] for visualization
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # +eps to avoid division by zero

        return cam, target_class


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image."""
    # Scale heatmap back up to original image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply JET colormap: low values = blue, high values = red
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )

    # Handle grayscale images by converting to BGR for blending
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blend original image and heatmap: result = image*(1-alpha) + heatmap*alpha
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def visualize_gradcam(
    checkpoint_path: str,
    image_path: str,
    save_path: str = "assets/gradcam.png",
):
    """Generate and save Grad-CAM visualization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and reconstruct model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_name = config["model"]["name"]
    image_size = config["data"]["image_size"]

    if model_name == "efficientnet_b0":
        model = EmotionEfficientNet(
            num_classes=config["model"]["num_classes"],
            pretrained=False,  # don't re-download ImageNet weights; we have trained weights
        )
        # Target the last conv layer before global pooling — captures highest-level features
        target_layer = model.backbone.conv_head
    elif model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=config["model"]["num_classes"])
        # Target the second-to-last layer in features (last conv before final ReLU)
        target_layer = model.features[-2]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])  # restore trained weights
    model.to(device)

    # Load and preprocess the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original = image.copy()  # keep unmodified copy for visualization
    resized = cv2.resize(image, (image_size, image_size))
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5  # normalize to [-1, 1]
    # Add batch and channel dimensions: (H, W) → (1, 1, H, W)
    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)

    # Generate heatmap using Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap, pred_class = grad_cam.generate(input_tensor)

    # Get class probabilities for annotation (inference only, no grad needed)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze()  # convert logits to probabilities

    # Create 3-panel visualization figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: original grayscale face
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original Face")
    axes[0].axis("off")

    # Panel 2: raw Grad-CAM heatmap (no image overlay)
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    axes[1].imshow(heatmap_resized, cmap="jet")  # jet = blue→green→red heat scale
    axes[1].set_title(f"Grad-CAM: {EMOTION_LABELS[pred_class]}")
    axes[1].axis("off")

    # Panel 3: heatmap blended onto face — shows which facial regions drove the prediction
    overlay = overlay_heatmap(original, heatmap, alpha=0.4)
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Overlay ({probs[pred_class]:.1%} confidence)")
    axes[2].axis("off")

    # Title shows predicted emotion + all class probabilities
    plt.suptitle(
        f"Predicted: {EMOTION_LABELS[pred_class].upper()} | "
        + " | ".join(f"{EMOTION_LABELS[i]}: {probs[i]:.1%}" for i in range(len(EMOTION_LABELS))),
        fontsize=9,
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # create output dir if needed
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f" Grad-CAM saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--save", type=str, default="assets/gradcam.png")
    args = parser.parse_args()
    visualize_gradcam(args.checkpoint, args.image, args.save)
