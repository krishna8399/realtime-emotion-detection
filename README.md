# Real-Time Face & Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Demo](assets/demo.gif)

A production-ready emotion detection pipeline that detects faces using MediaPipe and classifies emotions (angry, disgust, fear, happy, neutral, sad, surprise) using a fine-tuned EfficientNet-B0. Deployed as an interactive Streamlit web app with per-face confidence bars, Grad-CAM explainability, and emotion timeline charts for video.

## Architecture

```
Image / Video Input
       │
       ▼
┌──────────────┐
│  MediaPipe    │  Face detection + bounding box
│  Face Detect  │
└──────┬───────┘
       │ Cropped face (224×224, grayscale)
       ▼
┌──────────────┐
│ EfficientNet │  Fine-tuned on FER-2013
│    B0        │  7-class softmax classifier
└──────┬───────┘
       │ Emotion label + confidence scores
       ▼
┌──────────────┐
│  Streamlit   │  Annotated output + Grad-CAM heatmaps
│  Frontend    │  + Emotion timeline (video mode)
└──────────────┘
```

## Results

| Model | Test Accuracy | Weighted F1 | Params | GPU Inference |
|-------|--------------|-------------|--------|---------------|
| Baseline CNN | 65.00% | 0.6414 | 1.2M | — |
| EfficientNet-B0 v1 | 67.64%* | 0.6753 | 4.73M | — |
| EfficientNet-B0 v2 | **68.00%** | **0.6791** | 4.73M | **17 ms / 59 FPS** |

*v1 figure is best validation accuracy (no separate test run). v2 figures are on the held-out FER-2013 test set (7,178 images).

> v2 improvements over v1: class-weighted loss + Mixup augmentation (α=0.2) + Test Time Augmentation (4 views)

**CPU inference (GTX 1650 host, model on CPU):** ~37 ms/frame &nbsp;|&nbsp; **GPU inference:** ~17 ms/frame (59 FPS)

### Per-Class Performance — EfficientNet-B0 v2

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|----|---------|
| angry   | 0.63 | 0.57 | 0.60 | 958 |
| disgust | 0.57 | **0.74** | 0.65 | 111 |
| fear    | 0.53 | 0.54 | 0.53 | 1,024 |
| happy   | **0.88** | 0.86 | **0.87** | 1,774 |
| neutral | 0.62 | 0.67 | 0.64 | 1,233 |
| sad     | 0.56 | 0.52 | 0.54 | 1,247 |
| surprise| 0.77 | **0.84** | 0.80 | 831 |
| **weighted avg** | **0.68** | **0.68** | **0.68** | 7,178 |

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The matrix confirms two dominant confusion patterns: **fear ↔ sad** and **angry ↔ neutral**. Both are discussed in the "What I Learned" section below.

### Grad-CAM Explainability

The app includes live Grad-CAM heatmaps (accessible via the **Explainability** tab after uploading an image). Warm regions show what the model focused on for each of the top-3 predicted emotions. Example: for "happy", activation concentrates on the mouth corners; for "fear", it shifts to the eyes and forehead.

### LR Finder

![LR Finder](assets/lr_finder.png)

The fast.ai-style LR finder identified **lr ≈ 3×10⁻⁴** as the optimal learning rate (steepest loss descent before divergence). This was used directly in the final EfficientNet-B0 v2 training run.

---

## Quick Start

### Option 1: Docker (Recommended — no environment setup needed)

```bash
docker build -t emotion-detection .
docker run -p 8501:8501 emotion-detection
```

Open **http://localhost:8501** in your browser.

> **Note:** The Docker image (~3.4 GB) includes PyTorch with CUDA but runs on CPU inside the container by default. Pass `--gpus all` if your host has the NVIDIA Container Toolkit installed.

### Option 2: Local Setup

**Requirements:** Python 3.10, a Kaggle account (for the dataset), and optionally a CUDA-capable GPU.

```bash
# 1. Clone
git clone https://github.com/krishna8399/realtime-emotion-detection.git
cd realtime-emotion-detection

# 2. Create environment
conda create -n emotion-det python=3.10 -y
conda activate emotion-det

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU only) Reinstall PyTorch with CUDA support
#    The default requirements.txt installs CPU-only torch.
#    For CUDA 12.4 (adjust the cu124 tag to match your driver):
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 \
    --force-reinstall

# 5. Download FER-2013 from Kaggle
#    First, place your kaggle.json at ~/.kaggle/kaggle.json (chmod 600 on Linux/Mac)
python src/data/download_data.py

# 6. Train baseline CNN (~30 min on GPU)
python src/models/train.py --config configs/baseline_cnn.yaml

# 7. Train EfficientNet-B0 with all improvements (~20 epochs, ~45 min on GPU)
python src/models/train.py --config configs/efficientnet_v2.yaml

# 8. Evaluate and generate confusion matrix
python src/models/evaluate.py --checkpoint models/checkpoints/best_efficientnet_b0.pt

# 9. Run the Streamlit app
streamlit run src/app/app.py
```

**Windows note:** `num_workers` is set to `0` in all configs to avoid multiprocessing issues. This is already the default.

---

## Project Structure

```
realtime-emotion-detection/
├── README.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── configs/
│   ├── baseline_cnn.yaml       # Baseline CNN hyperparameters
│   ├── efficientnet.yaml       # EfficientNet v1 config
│   └── efficientnet_v2.yaml    # EfficientNet v2 (class weights + Mixup + TTA)
├── src/
│   ├── data/
│   │   ├── download_data.py    # Download FER-2013 from Kaggle
│   │   └── dataset.py          # PyTorch Dataset + augmentations
│   ├── models/
│   │   ├── baseline_cnn.py     # 4-block CNN baseline (1.2M params)
│   │   ├── efficientnet.py     # Fine-tuned EfficientNet-B0 (4.73M params)
│   │   ├── train.py            # Training loop with W&B logging
│   │   └── evaluate.py         # Evaluation + confusion matrix
│   ├── utils/
│   │   └── visualize.py        # Grad-CAM implementation
│   └── app/
│       ├── app.py              # Streamlit application
│       ├── detector.py         # MediaPipe face detection wrapper
│       └── predictor.py        # Inference pipeline
├── assets/
│   ├── confusion_matrix.png    # Generated by evaluate.py
│   └── lr_finder.png           # Generated by train.py (run_lr_finder: true)
└── models/
    └── checkpoints/            # Saved model weights (gitignored)
```

## Tech Stack

- **Deep Learning**: PyTorch 2.x, timm (EfficientNet-B0), torchvision
- **Computer Vision**: OpenCV, MediaPipe, albumentations
- **Experiment Tracking**: Weights & Biases
- **Deployment**: Streamlit, Docker
- **Dataset**: FER-2013 — 35,887 grayscale 48×48 images, 7 emotion classes

## Training Details

| Setting | Value |
|---------|-------|
| Dataset split | 28,709 train / 3,589 val / 7,178 test |
| Input resolution | 48×48 → resized to 224×224 |
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Backbone freeze | First 3 epochs frozen, then full fine-tune |
| Optimizer | AdamW (lr=3×10⁻⁴, weight_decay=0.01) |
| Scheduler | Cosine annealing |
| Label smoothing | 0.1 |
| Class-weighted loss | Yes (inverse frequency; disgust ×4.4, happy ×0.27) |
| Mixup | α=0.2 |
| TTA at inference | 4 views: original, horizontal flip, +brightness, −brightness |
| Early stopping | Patience=5, min_delta=0.001 |
| Epochs trained | 20 |

---

## What I Learned

**1. Class imbalance is the first thing to fix, not the last.**
FER-2013 has a 16:1 ratio between its largest class (happy, 7,215 samples) and smallest (disgust, 436). Before adding class-weighted loss, the model's disgust recall was essentially zero — it learned to predict happy/neutral and still achieved ~66% accuracy because those dominate the dataset. Inverse-frequency weighting boosted disgust recall from near-zero to 74% at the cost of just ~0.5% overall accuracy. The lesson: accuracy on imbalanced data is a misleading metric until you look at per-class recall.

**2. Fear and sad are genuinely ambiguous — not just a model problem.**
The confusion matrix shows the model's two biggest mistakes are fear→sad and sad→fear. This isn't a training failure; FER-2013's own inter-annotator agreement on fear is the lowest of all seven emotions. Fear and sadness share the same facial Action Units (AU1+AU4: inner brow raise + brow lowerer), and the difference lies mostly in subtle mouth tension that 48×48 grayscale crops cannot reliably capture. Any model trained on this data will hit this ceiling.

**3. Transfer learning wins even across domain mismatches.**
EfficientNet-B0 was pretrained on ImageNet RGB images of objects, but FER-2013 is grayscale face crops — about as different a domain as you can get while still being images. Despite this, loading ImageNet weights and fine-tuning for 20 epochs beat a purpose-built 4-block CNN trained for 30 epochs by 3 percentage points. The early convolutional layers learn general edge and texture detectors that are useful for faces regardless of pretraining domain.

**4. Mixup closes the generalization gap more than it moves the accuracy number.**
Before Mixup the model had a ~27% train/val accuracy gap, a clear sign of overfitting. After adding Mixup (α=0.2), the gap shrank to ~12% while overall test accuracy only improved by ~1%. Mixup's value isn't primarily accuracy — it's that it forces the model to interpolate between samples rather than memorize them, producing better-calibrated confidence scores and a model that degrades more gracefully on out-of-distribution faces.

**5. The LR finder saves more time than it costs.**
Running the fast.ai-style learning rate finder (exponential ramp + EMA smoothing, ~1 epoch of compute) identified lr ≈ 3×10⁻⁴ as the region of steepest loss descent. Using this directly with AdamW + cosine annealing converged cleanly without any manual grid search. The finder also revealed that lr > 1×10⁻³ causes immediate loss divergence for this fine-tuning task — useful to know before committing to a 45-minute training run.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Krishna Singh** — MSc Artificial Intelligence @ IU Berlin
- GitHub: [@krishna8399](https://github.com/krishna8399)
- LinkedIn: [krishna839](https://linkedin.com/in/krishna839)
