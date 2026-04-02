"""
Face detection using MediaPipe.

Provides real-time face detection with bounding boxes
for the Streamlit app pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2  # OpenCV for image processing
import mediapipe as mp  # Google's MediaPipe for face detection
import numpy as np


@dataclass
class FaceDetection:
    """A single detected face."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box in pixel coordinates
    confidence: float                # detection confidence score from MediaPipe
    face_crop: np.ndarray            # cropped face region as numpy array (BGR)


class FaceDetector:
    """
    MediaPipe-based face detector.

    Advantages over Haar cascades:
    - More accurate, handles angles/occlusion better
    - GPU-accelerated when available
    - Returns confidence scores
    """

    def __init__(self, min_confidence: float = 0.5):
        self.mp_face_detection = mp.solutions.face_detection  # MediaPipe face detection module
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 = short-range model (best for < 2m, i.e. webcam), 1 = long-range
            min_detection_confidence=min_confidence,  # discard detections below this threshold
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
        h, w = frame.shape[:2]  # get frame height and width for coordinate conversion

        # MediaPipe requires RGB input, but OpenCV reads as BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)  # run detection

        detections = []
        if results.detections:  # None if no faces found
            for detection in results.detections:
                # MediaPipe returns normalized coords (0.0–1.0); convert to pixel coords
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Expand bounding box by `padding` fraction to include forehead/chin
                pad_x = int(bw * padding)
                pad_y = int(bh * padding)
                x = max(0, x - pad_x)           # clamp to image boundaries
                y = max(0, y - pad_y)
                bw = min(w - x, bw + 2 * pad_x)  # don't exceed image width
                bh = min(h - y, bh + 2 * pad_y)  # don't exceed image height

                # Crop the face region from the original frame
                face_crop = frame[y:y + bh, x:x + bw]

                if face_crop.size > 0:  # skip empty crops (edge case when bbox is at image boundary)
                    detections.append(FaceDetection(
                        bbox=(x, y, bw, bh),
                        confidence=detection.score[0],  # score is a list; take first element
                        face_crop=face_crop,
                    ))

        return detections

    def __del__(self):
        self.detector.close()  # release MediaPipe resources when object is garbage collected
