# 😊 Real-Time Face & Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready emotion detection pipeline that captures video input, detects faces using MediaPipe, and classifies emotions (happy, sad, angry, surprise, neutral, fear, disgust) using a fine-tuned EfficientNet-B0. Deployed as an interactive Streamlit web app with real-time inference.

<!-- TODO: Add demo GIF here after building -->
<!-- ![Demo](assets/demo.gif) -->

## 🏗️ Architecture

```
Video/Webcam Input
       │
       ▼
┌──────────────┐
│  MediaPipe    │  Face Detection
│  Face Detect  │  + Bounding Box
└──────┬───────┘
       │ Cropped Face (48x48, grayscale)
       ▼
┌──────────────┐
│ EfficientNet │  Fine-tuned on FER-2013
│    B0        │  7-class classification
└──────┬───────┘
       │ Emotion + Confidence
       ▼
┌──────────────┐
│  Streamlit   │  Annotated video feed
│  Frontend    │  + Analytics dashboard
└──────────────┘
```

## 📊 Results

| Model | Accuracy | F1-Score | Inference (CPU) | Inference (GPU) |
|-------|----------|----------|-----------------|-----------------|
| Baseline CNN | - | - | - | - |
| EfficientNet-B0 | - | - | - | - |

<!-- TODO: Fill in after training -->

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker build -t emotion-detection .
docker run -p 8501:8501 emotion-detection
```

### Option 2: Local Setup
```bash
# Clone
git clone https://github.com/krishna8399/realtime-emotion-detection.git
cd realtime-emotion-detection

# Create environment
conda create -n emotion-det python=3.10 -y
conda activate emotion-det

# Install dependencies
pip install -r requirements.txt

# Download & prepare data
python src/data/download_data.py
python src/data/prepare_data.py

# Train baseline
python src/models/train.py --config configs/baseline_cnn.yaml

# Train EfficientNet
python src/models/train.py --config configs/efficientnet.yaml

# Run app
streamlit run src/app/app.py
```

## 📁 Project Structure

```
realtime-emotion-detection/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── configs/
│   ├── baseline_cnn.yaml       # Baseline CNN hyperparameters
│   └── efficientnet.yaml       # EfficientNet fine-tuning config
├── src/
│   ├── data/
│   │   ├── download_data.py    # Download FER-2013 from Kaggle
│   │   ├── prepare_data.py     # Preprocessing & splits
│   │   └── dataset.py          # PyTorch Dataset + augmentations
│   ├── models/
│   │   ├── baseline_cnn.py     # Simple CNN (baseline)
│   │   ├── efficientnet.py     # Fine-tuned EfficientNet-B0
│   │   ├── train.py            # Training loop with W&B logging
│   │   └── evaluate.py         # Evaluation + confusion matrix
│   ├── utils/
│   │   ├── visualize.py        # Grad-CAM + attention maps
│   │   └── metrics.py          # Custom metrics
│   └── app/
│       ├── app.py              # Streamlit application
│       ├── detector.py         # MediaPipe face detection
│       └── predictor.py        # Inference pipeline
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── tests/
│   └── test_dataset.py         # Unit tests
├── docker/
│   └── docker-compose.yml
└── assets/
    └── demo.gif                # Demo recording
```

## 🔧 Tech Stack

- **Deep Learning**: PyTorch, torchvision, timm (EfficientNet)
- **Computer Vision**: OpenCV, MediaPipe, albumentations
- **Experiment Tracking**: Weights & Biases
- **Deployment**: Streamlit, Docker
- **Data**: FER-2013 (35,887 images, 7 emotion classes)

## 📈 Training Details

- **Dataset**: FER-2013 — 28,709 training / 3,589 validation / 3,589 test images
- **Image Size**: 48×48 grayscale → resized to 224×224 for EfficientNet
- **Augmentations**: Horizontal flip, rotation (±15°), brightness/contrast, random crop
- **Optimizer**: AdamW with cosine annealing LR schedule
- **Training**: ~15 epochs with early stopping (patience=5)

## 🧠 What I Learned

<!-- TODO: Fill this in as you build — recruiters love seeing your thought process -->
- 
- 
- 

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Krishna Singh** — MSc Artificial Intelligence @ IU Berlin
- GitHub: [@krishna8399](https://github.com/krishna8399)
- LinkedIn: [krishna839](https://linkedin.com/in/krishna839)
