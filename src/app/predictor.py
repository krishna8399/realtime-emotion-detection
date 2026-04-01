"""
Emotion prediction pipeline.

Combines face detection + emotion classification into a single
easy-to-use interface for the Streamlit app.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import EMOTION_LABELS
from src.models.baseline_cnn import BaselineCNN
from src.models.efficientnet import EmotionEfficientNet
from src.app.detector import FaceDetector


@dataclass
class EmotionPrediction:
    """Prediction result for a single face."""
    bbox: Tuple[int, int, int, int]
    emotion: str
    confidence: float
    all_probs: dict  # {emotion_name: probability}


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
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint["config"]
        self.image_size = config["data"]["image_size"]

        model_name = config["model"]["name"]
        num_classes = config["model"]["num_classes"]

        if model_name == "baseline_cnn":
            self.model = BaselineCNN(num_classes=num_classes)
        elif model_name == "efficientnet_b0":
            self.model = EmotionEfficientNet(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Face detector
        self.face_detector = FaceDetector(min_confidence=face_confidence)

        print(f"✅ Loaded {model_name} (val_acc: {checkpoint.get('val_acc', 'N/A')})")
        print(f"   Device: {self.device}")

    def preprocess_face(self, face_crop: np.ndarray) -> torch.Tensor:
        """Preprocess a face crop for the model."""
        # Convert to grayscale
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        # Resize
        resized = cv2.resize(gray, (self.image_size, self.image_size))

        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5

        # To tensor: (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_frame(self, frame: np.ndarray) -> List[EmotionPrediction]:
        """
        Run full pipeline on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of EmotionPrediction for each detected face
        """
        # Detect faces
        faces = self.face_detector.detect(frame)

        predictions = []
        for face in faces:
            # Preprocess
            input_tensor = self.preprocess_face(face.face_crop)

            # Predict
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze()

            # Get top prediction
            top_idx = probs.argmax().item()
            top_prob = probs[top_idx].item()

            # All probabilities
            all_probs = {
                EMOTION_LABELS[i]: round(probs[i].item(), 4)
                for i in range(len(EMOTION_LABELS))
            }

            predictions.append(EmotionPrediction(
                bbox=face.bbox,
                emotion=EMOTION_LABELS[top_idx],
                confidence=top_prob,
                all_probs=all_probs,
            ))

        return predictions

    def annotate_frame(
        self,
        frame: np.ndarray,
        predictions: List[EmotionPrediction],
    ) -> np.ndarray:
        """Draw bounding boxes and emotion labels on frame."""
        annotated = frame.copy()

        # Color map for emotions
        colors = {
            "angry": (0, 0, 255),
            "disgust": (0, 128, 0),
            "fear": (128, 0, 128),
            "happy": (0, 255, 255),
            "neutral": (200, 200, 200),
            "sad": (255, 0, 0),
            "surprise": (0, 255, 0),
        }

        for pred in predictions:
            x, y, w, h = pred.bbox
            color = colors.get(pred.emotion, (255, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label background
            label = f"{pred.emotion} ({pred.confidence:.0%})"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
            )
            cv2.rectangle(
                annotated,
                (x, y - label_h - 10),
                (x + label_w, y),
                color, -1,
            )

            # Label text
            cv2.putText(
                annotated, label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 2,
            )

        return annotated
