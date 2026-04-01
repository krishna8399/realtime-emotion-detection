"""Tests for the FER2013 dataset loader."""

import numpy as np
import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import (
    EMOTION_LABELS,
    LABEL_TO_IDX,
    NUM_CLASSES,
    get_train_transforms,
    get_val_transforms,
)


def test_emotion_labels():
    assert NUM_CLASSES == 7
    assert len(EMOTION_LABELS) == 7
    assert EMOTION_LABELS[0] == "angry"
    assert EMOTION_LABELS[3] == "happy"
    assert EMOTION_LABELS[6] == "surprise"


def test_label_to_idx():
    assert LABEL_TO_IDX["happy"] == 3
    assert LABEL_TO_IDX["angry"] == 0
    assert LABEL_TO_IDX["neutral"] == 4


def test_train_transforms():
    transform = get_train_transforms(image_size=48)
    dummy_image = np.random.randint(0, 255, (48, 48, 1), dtype=np.uint8)
    result = transform(image=dummy_image)
    tensor = result["image"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 48, 48)


def test_val_transforms():
    transform = get_val_transforms(image_size=48)
    dummy_image = np.random.randint(0, 255, (48, 48, 1), dtype=np.uint8)
    result = transform(image=dummy_image)
    tensor = result["image"]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 48, 48)


def test_transforms_different_sizes():
    for size in [48, 96, 224]:
        transform = get_val_transforms(image_size=size)
        dummy = np.random.randint(0, 255, (100, 80, 1), dtype=np.uint8)
        result = transform(image=dummy)
        assert result["image"].shape == (1, size, size)
