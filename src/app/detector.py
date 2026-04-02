"""
Face detection using MediaPipe.

Supports both the legacy mp.solutions API (mediapipe < 0.10.14) and
the new Tasks API (mediapipe >= 0.10.14, which removed mp.solutions).
Falls back to OpenCV Haar cascades if neither MediaPipe API is available.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetection:
    """A single detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box in pixel coordinates
    confidence: float                # detection confidence score
    face_crop: np.ndarray            # cropped face region as numpy array (BGR)


def _has_solutions_api() -> bool:
    """Return True if this mediapipe version has the legacy mp.solutions API."""
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection")


def _has_tasks_api() -> bool:
    """Return True if this mediapipe version has the new Tasks API."""
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        _ = vision.FaceDetector  # noqa: F841
        return True
    except (ImportError, AttributeError):
        return False


class FaceDetector:
    """
    MediaPipe-based face detector with automatic API version detection.

    - mediapipe < 0.10.14 : uses mp.solutions.face_detection (legacy API)
    - mediapipe >= 0.10.14: uses mediapipe.tasks (new Tasks API)
    - fallback             : OpenCV Haar cascade if neither is available
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._mp_face_detection = None  # legacy solutions handle
        self._tasks_detector = None     # new tasks handle
        self._haar_cascade = None       # fallback

        if _has_solutions_api():
            self._init_solutions(min_confidence)
        elif _has_tasks_api():
            self._init_tasks(min_confidence)
        else:
            self._init_haar()

    def _init_solutions(self, min_confidence: float) -> None:
        """Initialize using legacy mp.solutions API (mediapipe < 0.10.14)."""
        self.mp_face_detection = mp.solutions.face_detection
        self._mp_face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence,
        )
        self._backend = "solutions"
        print("FaceDetector: using MediaPipe solutions API")

    def _init_tasks(self, min_confidence: float) -> None:
        """Initialize using new MediaPipe Tasks API (mediapipe >= 0.10.14)."""
        from mediapipe.tasks.python import vision, BaseOptions

        # The Tasks API requires a model file — download if not present
        model_path = self._ensure_model_file()

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_confidence,
        )
        self._tasks_detector = vision.FaceDetector.create_from_options(options)
        self._backend = "tasks"
        print("FaceDetector: using MediaPipe Tasks API")

    def _init_haar(self) -> None:
        """Fallback: OpenCV Haar cascade face detector."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        self._backend = "haar"
        print("FaceDetector: using OpenCV Haar cascade (fallback)")

    @staticmethod
    def _ensure_model_file():
        """Download the BlazeFace model for the Tasks API if not already present."""
        from pathlib import Path
        import urllib.request

        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "blaze_face_short_range.tflite"

        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/1/"
                "blaze_face_short_range.tflite"
            )
            print(f"Downloading face detector model to {model_path}…")
            urllib.request.urlretrieve(url, model_path)

        return model_path

    def _update_confidence(self, min_confidence: float) -> None:
        """Recreate the detector with a new confidence threshold (called by the app slider)."""
        if self._backend == "solutions":
            if self._mp_face_detection:
                self._mp_face_detection.close()
            self._mp_face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        elif self._backend == "tasks":
            self._init_tasks(min_confidence)
        # Haar cascade doesn't support confidence filtering — threshold applied post-detection

    def detect(self, frame: np.ndarray, padding: float = 0.2) -> List[FaceDetection]:
        """
        Detect faces in a BGR frame.

        Args:
            frame:   BGR image (from OpenCV)
            padding: extra padding around each bbox as a fraction of its size

        Returns:
            List of FaceDetection objects
        """
        if self._backend == "solutions":
            return self._detect_solutions(frame, padding)
        elif self._backend == "tasks":
            return self._detect_tasks(frame, padding)
        else:
            return self._detect_haar(frame, padding)

    def _detect_solutions(self, frame: np.ndarray, padding: float) -> List[FaceDetection]:
        """Detect using legacy mp.solutions API."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face_detection.process(rgb)

        detections = []
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                bw, bh = int(bbox.width * w), int(bbox.height * h)

                x, y, bw, bh = self._apply_padding(x, y, bw, bh, w, h, padding)
                crop = frame[y:y + bh, x:x + bw]
                if crop.size > 0:
                    detections.append(FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=det.score[0],
                        face_crop=crop,
                    ))
        return detections

    def _detect_tasks(self, frame: np.ndarray, padding: float) -> List[FaceDetection]:
        """Detect using new MediaPipe Tasks API."""
        import mediapipe as mp_tasks  # noqa: F811

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_tasks.Image(image_format=mp_tasks.ImageFormat.SRGB, data=rgb)
        result = self._tasks_detector.detect(mp_image)

        detections = []
        for det in result.detections:
            bb = det.bounding_box
            x, y = bb.origin_x, bb.origin_y
            bw, bh = bb.width, bb.height
            score = det.categories[0].score if det.categories else 0.5

            x, y, bw, bh = self._apply_padding(x, y, bw, bh, w, h, padding)
            crop = frame[y:y + bh, x:x + bw]
            if crop.size > 0:
                detections.append(FaceDetection(
                    bbox=(x, y, bw, bh),
                    confidence=score,
                    face_crop=crop,
                ))
        return detections

    def _detect_haar(self, frame: np.ndarray, padding: float) -> List[FaceDetection]:
        """Detect using OpenCV Haar cascade fallback."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detections = []
        if len(faces) > 0:
            for (x, y, bw, bh) in faces:
                x, y, bw, bh = self._apply_padding(x, y, bw, bh, w, h, padding)
                crop = frame[y:y + bh, x:x + bw]
                if crop.size > 0:
                    detections.append(FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=0.9,  # Haar doesn't give confidence scores
                        face_crop=crop,
                    ))
        return detections

    @staticmethod
    def _apply_padding(
        x: int, y: int, bw: int, bh: int,
        img_w: int, img_h: int,
        padding: float,
    ) -> Tuple[int, int, int, int]:
        """Expand a bounding box by padding fraction, clamped to image bounds."""
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        bw = min(img_w - x, bw + 2 * pad_x)
        bh = min(img_h - y, bh + 2 * pad_y)
        return x, y, bw, bh

    def __del__(self):
        if self._mp_face_detection and self._backend == "solutions":
            try:
                self._mp_face_detection.close()
            except Exception:
                pass
