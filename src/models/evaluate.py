"""
Model evaluation with detailed metrics and visualizations.

Usage:
    python src/models/evaluate.py --checkpoint models/checkpoints/best_efficientnet_b0.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # statistical plotting library (used for heatmaps)
import torch
from sklearn.metrics import (
    classification_report,  # per-class precision, recall, F1
    confusion_matrix,        # matrix of true vs predicted labels
    f1_score,                # weighted F1 across all classes
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import (
    EMOTION_LABELS,
    FER2013Dataset,
    get_val_transforms,
)
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet
from typing import Tuple


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Load a checkpoint and reconstruct the model architecture. Returns (model, config)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)  # load saved state dict + config
    config = checkpoint["config"]  # retrieve training config embedded in the checkpoint

    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    # Reconstruct the same model architecture used during training
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        model = EmotionEfficientNet(num_classes=num_classes, pretrained=False)  # no ImageNet weights needed
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])  # restore learned weights
    model.to(device)
    model.eval()  # set to eval mode: disable dropout, use frozen batch norm stats

    return model, config


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "assets/confusion_matrix.png",
) -> None:
    """Plot and save a side-by-side raw + normalized confusion matrix."""
    labels = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]  # ordered label names
    cm = confusion_matrix(y_true, y_pred)  # raw count matrix: cm[i,j] = predicted j when true is i
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # normalize each row to sum to 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: raw counts — shows absolute number of predictions per cell
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Right plot: normalized — shows per-class recall (diagonal = accuracy per class)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # create assets/ dir if needed
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f" Confusion matrix saved to {save_path}")
    plt.close()


def evaluate(checkpoint_path: str) -> None:
    """Run evaluation on the FER-2013 test split and print per-class metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)

    image_size = config["data"]["image_size"]
    data_dir = config["data"]["data_dir"]

    # Load test split with validation transforms (no augmentation)
    dataset = FER2013Dataset(
        root_dir=f"{data_dir}/test",
        transform=get_val_transforms(image_size),
        grayscale=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():  # disable gradient tracking — not needed for inference
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)          # get raw logits
            _, predicted = outputs.max(1)    # take argmax to get predicted class
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print per-class precision, recall, F1, and support
    label_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Overall summary metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")  # weighted by class frequency
    acc = (all_preds == all_labels).mean() * 100  # overall accuracy
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Weighted F1-Score: {f1:.4f}")

    # Save confusion matrix plot to disk
    plot_confusion_matrix(all_labels, all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
