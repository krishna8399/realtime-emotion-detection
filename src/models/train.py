"""
Training loop with Weights & Biases experiment tracking.

Usage:
    python src/models/train.py --config configs/baseline_cnn.yaml
    python src/models/train.py --config configs/efficientnet.yaml
    python src/models/train.py --config configs/efficientnet_v2.yaml
"""

import argparse
import copy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts with no display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import create_dataloaders, NUM_CLASSES, EMOTION_LABELS
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet


def lr_finder(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_iter: int = 100,
    smoothing: float = 0.05,
    save_path: str = "assets/lr_finder.png",
) -> float:
    """
    Fast.ai-style learning rate finder.

    How it works:
      1. Save the model's current weights (we restore them after)
      2. Start with a very small LR and run one mini-batch
      3. Multiply LR by a constant factor each step (exponential ramp)
      4. Record the smoothed loss at each LR
      5. Stop when loss explodes (5x the minimum seen)
      6. Plot loss vs LR on a log scale
      7. Return the LR just before the loss starts rising steeply
         (heuristic: 1/10th of the LR with the steepest negative gradient)

    The optimal LR to use for training is roughly one order of magnitude
    below the minimum-loss LR — this is the "valley" region where the
    loss descends fastest without diverging.
    """
    print("\nRunning LR Finder...")

    # Save original weights so the finder doesn't pollute the model
    original_state = copy.deepcopy(model.state_dict())

    # Use SGD with momentum — same as fast.ai's finder; works well across architectures
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=0.9)

    # Multiplicative factor to increase LR each step
    # After num_iter steps: start_lr * factor^num_iter = end_lr
    factor = (end_lr / start_lr) ** (1.0 / num_iter)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=factor)

    lrs, losses = [], []
    avg_loss = 0.0
    best_loss = float("inf")

    model.train()
    data_iter = iter(dataloader)

    for i in tqdm(range(num_iter), desc="LR Finder"):
        # Wrap around the dataloader if we exhaust it before num_iter steps
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Exponential moving average smoothing to reduce noise in the loss curve
        avg_loss = smoothing * loss.item() + (1 - smoothing) * avg_loss
        # Bias correction for early steps (same trick as Adam)
        smoothed_loss = avg_loss / (1 - (1 - smoothing) ** (i + 1))

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(smoothed_loss)

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        # Stop early if loss has exploded (5x the best seen) — no point continuing
        if smoothed_loss > 5 * best_loss and i > 10:
            print(f"  Loss diverged at LR={current_lr:.2e} — stopping early")
            break

        scheduler.step()

    # Restore original weights — the finder is purely diagnostic
    model.load_state_dict(original_state)

    # --- Find suggested LR ---
    # Compute gradient of loss w.r.t. LR index; steepest descent = best region
    losses_arr = np.array(losses)
    lrs_arr = np.array(lrs)
    # Gradient of loss: negative = still improving, positive = diverging
    gradients = np.gradient(losses_arr)
    # Suggested LR: point of steepest negative gradient, divided by 10 for safety margin
    steepest_idx = np.argmin(gradients)
    suggested_lr = lrs_arr[steepest_idx] / 10.0

    # --- Plot ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lrs_arr, losses_arr, color="steelblue", linewidth=1.5, label="Smoothed loss")
    ax.axvline(
        suggested_lr, color="red", linestyle="--", linewidth=1.5,
        label=f"Suggested LR: {suggested_lr:.2e}",
    )
    ax.axvline(
        lrs_arr[steepest_idx], color="orange", linestyle=":", linewidth=1.2,
        label=f"Min gradient LR: {lrs_arr[steepest_idx]:.2e}",
    )
    ax.set_xscale("log")   # log scale makes the exponential ramp linear and easy to read
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Loss (smoothed)")
    ax.set_title("LR Finder — Loss vs Learning Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  LR finder plot saved to {save_path}")
    print(f"  Suggested LR: {suggested_lr:.2e}  (min-gradient LR / 10)")
    return suggested_lr


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> nn.Module:
    model_name = config["model"]["name"]
    num_classes = config["model"].get("num_classes", NUM_CLASSES)

    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    elif model_name == "efficientnet_b0":
        return EmotionEfficientNet(
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Improvement 1: Class-weighted loss
# ---------------------------------------------------------------------------

def compute_class_weights(data_dir: str, device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to penalise the loss more
    for rare classes (e.g. disgust: 436 samples) and less for common
    ones (e.g. happy: 7215 samples).

    Formula: weight_c = total / (num_classes * count_c)
    This is the sklearn 'balanced' strategy applied to PyTorch.
    """
    train_dir = Path(data_dir) / "train"
    counts = []
    for emotion in sorted(EMOTION_LABELS.values()):  # iterate in index order
        d = train_dir / emotion
        n = len(list(d.glob("*.jpg")) + list(d.glob("*.png"))) if d.exists() else 1
        counts.append(n)

    counts = np.array(counts, dtype=np.float32)
    total = counts.sum()
    # Higher weight for rarer classes; normalized so mean weight = 1
    weights = total / (len(counts) * counts)
    weights = weights / weights.mean()  # keep weights around 1.0 on average

    print("Class weights:")
    for i, (emotion, w) in enumerate(zip(sorted(EMOTION_LABELS.values()), weights)):
        print(f"  {emotion}: {w:.3f}  (n={int(counts[i])})")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Improvement 2: Mixup augmentation
# ---------------------------------------------------------------------------

def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    """
    Mixup: blend two random samples from the same batch.

    Creates virtual training examples:
        mixed_x = λ * x_i + (1-λ) * x_j
        loss    = λ * CE(pred, y_i) + (1-λ) * CE(pred, y_j)

    This forces the model to behave linearly between classes, improving
    generalisation on ambiguous boundary cases (e.g. fear vs sad).

    alpha controls Beta distribution shape:
        alpha → 0  : λ ≈ 1 (almost no mixing, safer)
        alpha = 0.2: light mixing (recommended starting point)
        alpha = 1.0: uniform mixing (aggressive)
    """
    lam = np.random.beta(alpha, alpha)  # mixing coefficient drawn from Beta distribution
    batch_size = images.size(0)
    # Random permutation to pair each sample with a different sample in the batch
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels            # original labels
    labels_b = labels[index]     # labels of the randomly paired samples
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Compute mixed loss: weighted sum of loss for both sets of labels."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


# ---------------------------------------------------------------------------
# Improvement 3: Test Time Augmentation (TTA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def tta_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test Time Augmentation: run inference on multiple augmented versions
    of each image and average the softmax probabilities.

    Augmentations used:
        1. Original image
        2. Horizontal flip  (emotions are roughly mirror-symmetric)
        3. Slight brightness increase (+10%)
        4. Slight brightness decrease (-10%)

    Averaging predictions reduces variance from random augmentation
    effects, consistently gaining ~0.5-1% accuracy at no training cost.
    """
    # 1. Original
    probs = F.softmax(model(images), dim=1)

    # 2. Horizontal flip — face emotions are roughly symmetric left/right
    flipped = torch.flip(images, dims=[3])  # flip along width dimension
    probs = probs + F.softmax(model(flipped), dim=1)

    # 3. Slightly brighter (simulate different lighting conditions)
    brighter = torch.clamp(images + 0.1, -1.0, 1.0)
    probs = probs + F.softmax(model(brighter), dim=1)

    # 4. Slightly darker
    darker = torch.clamp(images - 0.1, -1.0, 1.0)
    probs = probs + F.softmax(model(darker), dim=1)

    # Average over 4 augmentations → more stable prediction
    return probs / 4.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    mixup_alpha: float = 0.0,  # 0.0 = disabled; > 0 enables mixup
) -> dict:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0:
            # Apply mixup: blend pairs of images and compute blended loss
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            # Accuracy: count as correct if the dominant label matches (lam > 0.5 → use labels_a)
            _, predicted = outputs.max(1)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%",
        })

    return {
        "train_loss": running_loss / total,
        "train_acc": 100.0 * correct / total,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_tta: bool = False,  # toggle TTA on/off
) -> dict:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        if use_tta:
            # TTA: average predictions over multiple augmented views
            avg_probs = tta_predict(model, images)
            predicted = avg_probs.argmax(dim=1)
            # Compute loss on original images (TTA is inference-only)
            outputs = model(images)
            loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "val_loss": running_loss / total,
        "val_acc": 100.0 * correct / total,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(config_path: str):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"]["name"],
        tags=config["wandb"].get("tags", []),
        config=config,
    )

    image_size = config["data"]["image_size"]
    grayscale = True

    train_loader, val_loader = create_dataloaders(
        data_dir=config["data"]["data_dir"],
        image_size=image_size,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        grayscale=grayscale,
    )

    model = create_model(config).to(device)
    print(f"Model: {config['model']['name']}")
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")

    freeze_epochs = config["model"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    # ---- Loss function (Improvement 1: class weights) ----------------------
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    if config["training"].get("class_weighted_loss", False):
        # Compute per-class weights inversely proportional to class frequency
        class_weights = compute_class_weights(config["data"]["data_dir"], device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        print("Using class-weighted loss")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- LR Finder (optional) ----------------------------------------------
    if config["training"].get("run_lr_finder", False):
        suggested_lr = lr_finder(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            save_path="assets/lr_finder.png",
        )
        # Override config LR with the suggested value and log it
        print(f"  Overriding config LR {lr:.2e} → suggested {suggested_lr:.2e}")
        lr = suggested_lr
        wandb.log({"lr_finder/suggested_lr": suggested_lr})

    # ---- Read augmentation/TTA flags from config ---------------------------
    mixup_alpha = config["training"].get("mixup_alpha", 0.0)   # 0 = disabled
    use_tta = config["training"].get("use_tta", False)
    if mixup_alpha > 0:
        print(f"Mixup enabled (alpha={mixup_alpha})")
    if use_tta:
        print("TTA enabled for validation")

    # ---- Optimizer ---------------------------------------------------------
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    opt_name = config["training"]["optimizer"]

    if opt_name == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=lr * 0.01,
    )

    es_config = config["training"]["early_stopping"]
    best_val_acc = 0.0
    patience_counter = 0

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"{'='*60}")

        if epoch == freeze_epochs + 1 and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
            optimizer = AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": lr * 0.1},
                    {"params": model.classifier.parameters(), "lr": lr},
                ],
                weight_decay=wd,
            )

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            mixup_alpha=mixup_alpha,  # pass mixup setting
        )

        val_metrics = validate(
            model, val_loader, criterion, device,
            use_tta=use_tta,  # pass TTA setting
        )

        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train_loss": train_metrics["train_loss"],
            "train_acc": train_metrics["train_acc"],
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_metrics["val_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.2f}%")
        print(f"  Val   Loss: {val_metrics['val_loss']:.4f} | Val   Acc: {val_metrics['val_acc']:.2f}%")

        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            patience_counter = 0
            save_path = ckpt_dir / f"best_{config['model']['name']}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": config,
            }, save_path)
            print(f"Saved best model (val_acc: {best_val_acc:.2f}%)")
            wandb.run.summary["best_val_acc"] = best_val_acc
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{es_config['patience']})")

        if patience_counter >= es_config["patience"]:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    train(args.config)
