"""
Face detection using MediaPipe.

Provides real-time face detection with bounding boxes
for the Streamlit app pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceDetection:
    """A single detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    face_crop: np.ndarray  # cropped face region


class FaceDetector:
    """
    MediaPipe-based face detector.

    Advantages over Haar cascades:
    - More accurate, handles angles/occlusion better
    - GPU-accelerated when available
    - Returns confidence scores
    """

    def __init__(self, min_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range (< 2m), 1 = long-range
            min_detection_confidence=min_confidence,
        )

    def detect(self, frame: np.ndarray, padding: float = 0.2) -> List[FaceDetection]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image (from OpenCV)
            padding: Extra padding around face bbox (fraction)

        Returns:
            List of FaceDetection objects
        """
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        detections = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Add padding
                pad_x = int(bw * padding)
                pad_y = int(bh * padding)
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                bw = min(w - x, bw + 2 * pad_x)
                bh = min(h - y, bh + 2 * pad_y)

                # Crop face
                face_crop = frame[y:y + bh, x:x + bw]

                if face_crop.size > 0:
                    detections.append(FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=detection.score[0],
                        face_crop=face_crop,
                    ))

        return detections

    def __del__(self):
        self.detector.close()
