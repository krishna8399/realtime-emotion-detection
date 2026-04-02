"""
PyTorch Dataset for FER-2013 with albumentations augmentations.

FER-2013 has images organized in folders:
    data/fer2013/train/{emotion_name}/image_xxxx.jpg
    data/fer2013/test/{emotion_name}/image_xxxx.jpg

Emotion labels: angry, disgust, fear, happy, neutral, sad, surprise
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import albumentations as A  # image augmentation library
import cv2  # OpenCV for reading images
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2  # converts numpy array to PyTorch tensor
from torch.utils.data import DataLoader, Dataset

# FER-2013 emotion mapping: index → label name
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

LABEL_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}  # reverse map: label name → index
NUM_CLASSES = len(EMOTION_LABELS)  # total number of emotion classes (7)


def get_train_transforms(image_size: int = 48) -> A.Compose:
    """Training augmentations — makes model robust to real-world variations."""
    return A.Compose([
        A.Resize(image_size, image_size),  # resize to model input size
        A.HorizontalFlip(p=0.5),  # randomly flip face horizontally (emotions are symmetric)
        A.Rotate(limit=15, p=0.3),  # small random rotation up to ±15 degrees
        A.RandomBrightnessContrast(
            brightness_limit=0.2,  # vary brightness by ±20%
            contrast_limit=0.2,    # vary contrast by ±20%
            p=0.3,
        ),
        A.GaussNoise(noise_scale_factor=0.1, p=0.2),  # add random Gaussian noise to simulate sensor noise
        A.CoarseDropout(
            num_holes_range=(1, 1), hole_height_range=(4, 8), hole_width_range=(4, 8),
            p=0.2,  # randomly erase small patches to simulate occlusion
        ),
        A.Normalize(mean=[0.5], std=[0.5]),  # scale pixel values from [0,255] to [-1,1]
        ToTensorV2(),  # convert numpy (H, W, C) to torch tensor (C, H, W)
    ])


def get_val_transforms(image_size: int = 48) -> A.Compose:
    """Validation/test transforms — just resize and normalize."""
    return A.Compose([
        A.Resize(image_size, image_size),  # resize to match training input size
        A.Normalize(mean=[0.5], std=[0.5]),  # same normalization as training
        ToTensorV2(),  # convert to tensor
    ])


class FER2013Dataset(Dataset):
    """
    FER-2013 Dataset loader.

    Expects folder structure:
        root_dir/
            angry/
                image_0001.jpg
                ...
            happy/
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[A.Compose] = None,
        grayscale: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale

        # Collect all image paths and their corresponding labels
        self.samples = []
        for emotion_name in sorted(os.listdir(self.root_dir)):  # iterate each emotion folder
            emotion_dir = self.root_dir / emotion_name
            if not emotion_dir.is_dir():  # skip non-directory entries (e.g. files)
                continue

            label = LABEL_TO_IDX.get(emotion_name.lower())  # map folder name to class index
            if label is None:  # skip folders that don't match known emotions
                continue

            for img_name in os.listdir(emotion_dir):  # iterate images in this emotion folder
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # only image files
                    self.samples.append((emotion_dir / img_name, label))  # store (path, label) pair

        print(f"Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)  # total number of images in the dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]  # get image path and label for this index

        # Read image from disk — check for None immediately (corrupt/missing file)
        if self.grayscale:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)  # load as single-channel
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = np.expand_dims(image, axis=-1)  # add channel dim: (H, W) → (H, W, 1)
        else:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # load as BGR 3-channel
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert BGR to RGB for display

        # Apply augmentations/transforms
        if self.transform:
            augmented = self.transform(image=image)  # albumentations expects keyword arg
            image = augmented["image"]  # extract transformed image tensor from result dict

        return image, label  # return (tensor, class_index) pair


def create_dataloaders(
    data_dir: str,
    image_size: int = 48,
    batch_size: int = 64,
    num_workers: int = 4,
    grayscale: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    train_dir = os.path.join(data_dir, "train")  # path to training images
    test_dir = os.path.join(data_dir, "test")    # path to validation/test images

    train_dataset = FER2013Dataset(
        root_dir=train_dir,
        transform=get_train_transforms(image_size),  # use augmented transforms for training
        grayscale=grayscale,
    )

    val_dataset = FER2013Dataset(
        root_dir=test_dir,
        transform=get_val_transforms(image_size),  # use plain transforms for validation
        grayscale=grayscale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # shuffle training data each epoch to prevent ordering bias
        num_workers=num_workers,
        pin_memory=True,    # speeds up CPU→GPU transfer by using pinned (page-locked) memory
        drop_last=True,     # drop the last incomplete batch to keep batch sizes consistent
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # no shuffling for validation — order doesn't matter
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
