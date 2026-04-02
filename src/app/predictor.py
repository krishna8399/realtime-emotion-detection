"""
Emotion prediction pipeline.

Combines face detection + emotion classification into a single
easy-to-use interface for the Streamlit app.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2  # OpenCV for image preprocessing
import numpy as np
import torch
import torch.nn.functional as F  # for softmax

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import EMOTION_LABELS
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet
from src.app.detector import FaceDetector


@dataclass
class EmotionPrediction:
    """Prediction result for a single face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) face bounding box in the original frame
    emotion: str                     # predicted emotion label (e.g. "happy")
    confidence: float                # probability of the predicted emotion (0.0–1.0)
    all_probs: dict                  # {emotion_name: probability} for all 7 classes
    face_crop_bgr: np.ndarray = None  # raw BGR face crop — used by Grad-CAM in the app


class EmotionPredictor:
    """
    Full pipeline: frame → face detection → emotion classification.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        face_confidence: float = 0.5,
    ):
        # Automatically pick GPU if available, otherwise use CPU
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load saved checkpoint (weights + config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        self.image_size = config["data"]["image_size"]  # input resolution the model was trained on

        model_name = config["model"]["name"]
        num_classes = config["model"]["num_classes"]

        # Reconstruct the exact model architecture from the saved config
        if model_name == "baseline_cnn":
            self.model = BaselineCNN(num_classes=num_classes)
        elif model_name == "efficientnet_b0":
            self.model = EmotionEfficientNet(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.load_state_dict(checkpoint["model_state_dict"])  # restore trained weights
        self.model.to(self.device)
        self.model.eval()  # disable dropout for inference

        # Initialize face detector with given confidence threshold
        self.face_detector = FaceDetector(min_confidence=face_confidence)

        print(f" Loaded {model_name} (val_acc: {checkpoint.get('val_acc', 'N/A')})")
        print(f"   Device: {self.device}")

    def preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """Preprocess a face crop for the model."""
        # Convert BGR to grayscale if needed
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop  # already single-channel

        # Resize to the model's expected input resolution
        resized = cv2.resize(gray, (self.image_size, self.image_size))

        # Normalize to [-1, 1] range — same as training transforms
        normalized = resized.astype(np.float32) / 255.0  # [0, 255] → [0.0, 1.0]
        normalized = (normalized - 0.5) / 0.5            # [0.0, 1.0] → [-1.0, 1.0]

        # Add batch and channel dimensions: (H, W) → (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()  # no gradients needed during inference
    def predict_frame(self, frame: np.ndarray) -> List[EmotionPrediction]:
        """
        Run full pipeline on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of EmotionPrediction for each detected face
        """
        # Step 1: detect all faces in the frame
        faces = self.face_detector.detect(frame)

        predictions = []
        for face in faces:
            # Step 2: preprocess each face crop into a model-ready tensor
            input_tensor = self.preprocess_face(face.face_crop)

            # Step 3: run forward pass to get raw logits
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze()  # convert logits to probabilities summing to 1

            # Step 4: get the top prediction
            top_idx = probs.argmax().item()    # index of highest probability class
            top_prob = probs[top_idx].item()   # confidence score for that class

            # Build a dict of all class probabilities for display in the UI
            all_probs = {
                EMOTION_LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(EMOTION_LABELS))
            }

            predictions.append(EmotionPrediction(
                bbox=face.bbox,
                emotion=EMOTION_LABELS[top_idx],  # human-readable label
                confidence=top_prob,
                all_probs=all_probs,
                face_crop_bgr=face.face_crop,  # store crop for Grad-CAM explainability
            ))

        return predictions

    def annotate_frame(
        self,
        frame: np.ndarray,
        predictions: List[EmotionPrediction],
    ) -> np.ndarray:
        """
        Draw emotion bounding boxes and labels onto a copy of the frame.

        Each face gets a colored rectangle (color varies by emotion) and a filled
        label background showing the emotion name and confidence percentage.
        The original frame is not modified.
        """
        annotated = frame.copy()  # don't modify the original frame

        # Distinct BGR color for each emotion to make annotations easy to distinguish
        colors = {
            "angry": (0, 0, 255),       # red
            "disgust": (0, 128, 0),     # dark green
            "fear": (128, 0, 128),      # purple
            "happy": (0, 255, 255),     # yellow
            "neutral": (200, 200, 200),  # light grey
            "sad": (255, 0, 0),         # blue
            "surprise": (0, 255, 0),    # bright green
        }

        for pred in predictions:
            x, y, w, h = pred.bbox
            color = colors.get(pred.emotion, (255, 255, 255))  # white fallback

            # Draw face bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Measure text size so we can draw a solid background behind the label
            label = f"{pred.emotion} ({pred.confidence:.0%})"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
            )
            # Draw filled rectangle as label background
            cv2.rectangle(
                annotated,
                (x, y - label_h - 10),
                (x + label_w, y),
                color, -1,  # -1 = filled
            )

            # Draw label text in black on top of the colored background
            cv2.putText(
                annotated, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2,  # black text, thickness 2
            )

        return annotated
