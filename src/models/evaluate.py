"""
Model evaluation with detailed metrics and visualizations.

Usage:
    python src/models/evaluate.py --checkpoint models/checkpoints/best_efficientnet_b0.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
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


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        model = EmotionEfficientNet(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def plot_confusion_matrix(y_true, y_pred, save_path: str = "assets/confusion_matrix.png"):
    labels = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f" Confusion matrix saved to {save_path}")
    plt.close()


def evaluate(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(checkpoint_path, device)

    image_size = config["data"]["image_size"]
    data_dir = config["data"]["data_dir"]

    dataset = FER2013Dataset(
        root_dir=f"{data_dir}/test",
        transform=get_val_transforms(image_size),
        grayscale=True,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    label_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # Overall metrics
    f1 = f1_score(all_labels, all_preds, average="weighted")
    acc = (all_preds == all_labels).mean() * 100
    print(f"Overall Accuracy: {acc:.2f}%")
    print(f"Weighted F1-Score: {f1:.4f}")

    # Confusion matrix
    plot_confusion_matrix(all_labels, all_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint)
