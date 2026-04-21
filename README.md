---
title: Real-Time Emotion Detection
emoji: 😊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# 😊 Real-Time Emotion Detection

Detects faces and classifies emotions using a fine-tuned **EfficientNet-B0** trained on FER-2013.

**68% test accuracy** across 7 emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## Features
- Upload an image → get per-face emotion labels with confidence bars
- Upload a video → emotion timeline chart over time
- Grad-CAM explainability tab showing what the model focuses on
- JSON download of all predictions

## Model
EfficientNet-B0 fine-tuned with class-weighted loss, Mixup augmentation, and Test Time Augmentation.

Built by [Krishna Singh](https://github.com/krishna8399) — MSc AI @ IU Berlin
