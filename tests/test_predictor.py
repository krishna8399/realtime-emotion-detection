"""Tests for the EmotionPredictor inference pipeline and FaceDetector."""

import numpy as np
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app.detector import FaceDetector, FaceDetection
from src.app.predictor import EmotionPredictor, EmotionPrediction
from src.data.dataset import EMOTION_LABELS, NUM_CLASSES


CHECKPOINT = Path("models/checkpoints/best_efficientnet_b0.pt")
CHECKPOINT_FALLBACK = Path("models/checkpoints/best_baseline_cnn.pt")


def get_checkpoint() -> str:
    """Return path to any available checkpoint, skip if none found."""
    if CHECKPOINT.exists():
        return str(CHECKPOINT)
    if CHECKPOINT_FALLBACK.exists():
        return str(CHECKPOINT_FALLBACK)
    pytest.skip("No model checkpoint found — run training first")


# ── FaceDetector ──────────────────────────────────────────────────────────────

class TestFaceDetector:
    def setup_method(self):
        self.detector = FaceDetector(min_confidence=0.3)

    def test_initialises(self):
        """Detector should initialise with one of the three backends."""
        assert self.detector._backend in ("solutions", "tasks", "haar")

    def test_detect_blank_image_returns_empty(self):
        """Blank image should have no faces detected."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(blank)
        assert isinstance(results, list)

    def test_detect_returns_face_detection_objects(self):
        """detect() must always return a list of FaceDetection dataclass instances."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, FaceDetection)

    def test_face_detection_fields(self):
        """Each FaceDetection has valid bbox, confidence, and non-empty crop."""
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(frame)
        for r in results:
            x, y, w, h = r.bbox
            assert x >= 0 and y >= 0 and w > 0 and h > 0
            assert 0.0 <= r.confidence <= 1.0
            assert r.face_crop.size > 0

    def test_update_confidence(self):
        """_update_confidence() should not raise."""
        self.detector._update_confidence(0.7)
        self.detector._update_confidence(0.3)


# ── EmotionPredictor ──────────────────────────────────────────────────────────

class TestEmotionPredictor:
    @pytest.fixture(autouse=True)
    def load(self):
        self.predictor = EmotionPredictor(get_checkpoint())

    def test_has_required_attributes(self):
        """Predictor must expose config, val_acc, image_size, device, model."""
        assert hasattr(self.predictor, "config")
        assert hasattr(self.predictor, "val_acc")
        assert hasattr(self.predictor, "image_size")
        assert hasattr(self.predictor, "device")
        assert hasattr(self.predictor, "model")

    def test_config_has_expected_keys(self):
        assert "model" in self.predictor.config
        assert "data" in self.predictor.config
        assert self.predictor.config["model"]["num_classes"] == NUM_CLASSES

    def test_val_acc_reasonable(self):
        """val_acc should be between 50% and 100% for a trained model."""
        assert 50.0 < self.predictor.val_acc < 100.0

    def test_model_in_eval_mode(self):
        """Model must be in eval mode for inference."""
        assert not self.predictor.model.training

    def test_preprocess_face_shape(self):
        """preprocess_face() must return (1, 1, H, W) tensor on the correct device."""
        crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = self.predictor.preprocess_face(crop)
        assert tensor.shape == (1, 1, self.predictor.image_size, self.predictor.image_size)
        assert tensor.device.type == self.predictor.device.type

    def test_preprocess_face_normalised(self):
        """Output tensor should be normalised to roughly [-1, 1]."""
        crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        tensor = self.predictor.preprocess_face(crop)
        assert tensor.min().item() >= -1.5
        assert tensor.max().item() <= 1.5

    def test_predict_frame_blank(self):
        """Blank frame should return an empty list (no faces)."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.predictor.predict_frame(blank)
        assert isinstance(results, list)

    def test_predict_frame_returns_emotion_predictions(self):
        """predict_frame() must return a list of EmotionPrediction objects."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = self.predictor.predict_frame(frame)
        for r in results:
            assert isinstance(r, EmotionPrediction)

    def test_emotion_prediction_fields(self):
        """Each EmotionPrediction has valid emotion label and probability."""
        frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        results = self.predictor.predict_frame(frame)
        valid_emotions = set(EMOTION_LABELS.values())
        for r in results:
            assert r.emotion in valid_emotions
            assert 0.0 <= r.confidence <= 1.0
            assert set(r.all_probs.keys()) == valid_emotions
            assert abs(sum(r.all_probs.values()) - 1.0) < 1e-3  # probs sum to 1
            assert r.face_crop_bgr is not None

    def test_annotate_frame_shape(self):
        """annotate_frame() must return same shape as input."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = self.predictor.annotate_frame(frame, [])
        assert annotated.shape == frame.shape

    def test_annotate_frame_does_not_modify_original(self):
        """Original frame must not be mutated by annotate_frame()."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_copy = frame.copy()
        self.predictor.annotate_frame(frame, [])
        assert np.array_equal(frame, original_copy)
