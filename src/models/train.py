"""
Training loop with Weights & Biases experiment tracking.

Usage:
    python src/models/train.py --config configs/baseline_cnn.yaml
    python src/models/train.py --config configs/efficientnet.yaml
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml  # for reading YAML config files
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # learning rate scheduler
from tqdm import tqdm  # progress bars for training loops

import wandb  # Weights & Biases for experiment tracking

# Add project root to path so src.* imports work regardless of where script is run from
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders, NUM_CLASSES, EMOTION_LABELS
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)  # parse YAML into a Python dict


def create_model(config: dict) -> nn.Module:
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", NUM_CLASSES)

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return EmotionEfficientNet(
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),  # use ImageNet weights by default
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> dict:
    model.train()  # set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)  # progress bar over batches
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)  # move data to GPU/CPU

        optimizer.zero_grad()          # clear gradients from previous step
        outputs = model(images)        # forward pass: compute predictions
        loss = criterion(outputs, labels)  # compute cross-entropy loss
        loss.backward()                # backprop: compute gradients w.r.t. all parameters
        optimizer.step()               # update weights using computed gradients

        running_loss += loss.item() * images.size(0)  # accumulate weighted loss (scaled by batch size)
        _, predicted = outputs.max(1)  # get index of highest logit (predicted class)
        total += labels.size(0)        # count total samples processed
        correct += predicted.eq(labels).sum().item()  # count correct predictions

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%",
        })

    return {
        "train_loss": running_loss / total,       # average loss per sample
        "train_acc": 100.0 * correct / total,     # accuracy percentage
    }


@torch.no_grad()  # disable gradient computation for validation (saves memory and speed)
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()  # set model to eval mode (disables dropout, uses running batch norm stats)
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []   # collect all predictions for metrics
    all_labels = []  # collect all ground truth labels for metrics

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)            # forward pass only (no backward)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)      # predicted class index
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())   # move to CPU for sklearn metrics
        all_labels.extend(labels.cpu().numpy())

    return {
        "val_loss": running_loss / total,
        "val_acc": 100.0 * correct / total,
        "predictions": all_preds,   # full list for confusion matrix etc.
        "labels": all_labels,
    }


def train(config_path: str):
    config = load_config(config_path)

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize W&B run — logs config and tracks metrics online
    wandb.init(
        project=config["wandb"]["project"],  # W&B project name
        name=config["wandb"]["name"],        # run display name
        tags=config["wandb"].get("tags", []),
        config=config,  # log full config so we can reproduce the run
    )

    # Data
    image_size = config["data"]["image_size"]
    grayscale = True  # FER-2013 is grayscale

    train_loader, val_loader = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        image_size=image_size,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        grayscale=grayscale,
    )

    # Build model and move to target device
    model = create_model(config).to(device)
    print(f"Model: {config['model']['name']}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally freeze backbone for first N epochs (EfficientNet only)
    freeze_epochs = config["model"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()  # train only classifier head initially

    # Label smoothing softens targets: instead of [0,1,0,...] it uses [0.01,0.93,0.01,...]
    # Helps prevent overconfident predictions
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer setup
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]   # L2 regularization strength
    opt_name = config["training"]["optimizer"]

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)  # AdamW decouples weight decay
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Cosine annealing: smoothly decays LR from `lr` down to `lr * 0.01` over all epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],  # full cycle length = total epochs
        eta_min=lr * 0.01,                   # minimum LR at end of schedule
    )

    # Early stopping state
    es_config = config["training"]["early_stopping"]
    best_val_acc = 0.0
    patience_counter = 0  # counts epochs since last improvement

    # Create checkpoint directory if it doesn't exist
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Main training loop
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # Unfreeze backbone after freeze_epochs and reset optimizer with lower LR for backbone
        if epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
            # Use separate param groups: backbone gets 10x lower LR than classifier
            # (backbone weights are already good from ImageNet; we don't want to distort them)
            optimizer = AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr * 0.1},  # fine-tune gently
                    {"params": model.classifier.parameters(), "lr": lr},       # train head normally
                ],
                weight_decay=wd,
            )

        # Run one full pass over training data
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_metrics = validate(model, val_loader, criterion, device)

        # Advance LR scheduler after each epoch
        scheduler.step()

        # Log all metrics to W&B for this epoch
        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc": train_metrics["train_acc"],
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "lr": optimizer.param_groups[0]["lr"],  # current learning rate
        })

        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.2f}%")
        print(f"  Val   Loss: {val_metrics['val_loss']:.4f} | Val   Acc: {val_metrics['val_acc']:.2f}%")

        # Save checkpoint if this is the best model so far
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            patience_counter = 0  # reset patience since we improved
            save_path = ckpt_dir / f"best_{config['model']['name']}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),       # learnable weights
                "optimizer_state_dict": optimizer.state_dict(),  # optimizer state (for resuming)
                "val_acc": best_val_acc,
                "config": config,  # save config so we can reconstruct the model later
            }, save_path)
            print(f"Saved best model (val_acc: {best_val_acc:.2f}%)")
            wandb.run.summary["best_val_acc"] = best_val_acc  # update W&B run summary
        else:
            patience_counter += 1  # no improvement this epoch
            print(f"  No improvement ({patience_counter}/{es_config['patience']})")

        # Stop training if val accuracy hasn't improved for `patience` consecutive epochs
        if patience_counter >= es_config["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()  # mark W&B run as complete and upload final data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
