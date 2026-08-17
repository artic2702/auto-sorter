"""
Face detection engine — finds faces in images using YuNet.

=== WHAT IS THIS FILE? ===
This is the core AI component of the app. It takes a photo and returns
a list of detected faces with their locations and key points.

=== HOW YUNET FACE DETECTION WORKS ===
YuNet is a neural network trained on millions of face images. When given
a new photo, it scans the image at multiple scales looking for patterns
that match "face-like" features (eyes, nose, mouth arrangement).

For each face found, YuNet outputs:
  1. BOUNDING BOX (x, y, w, h) — rectangle around the face
  2. LANDMARKS — 5 key points:
     - Left eye center
     - Right eye center
     - Nose tip
     - Left mouth corner
     - Right mouth corner
  3. CONFIDENCE — how sure it is (0.0 to 1.0)

Visual example:
  ┌─────────────────┐
  │   •         •   │  ← left eye, right eye
  │                 │
  │        •        │  ← nose tip
  │                 │
  │     •     •     │  ← left mouth, right mouth
  └─────────────────┘
        ↑ bounding box

=== MULTI-SCALE DETECTION ===
YuNet works on a fixed input size (e.g., 320×320). But your photo might
be 4000×3000. If we just resize to 320×320, small faces get squished to
a few pixels and are missed.

Our approach: we tell YuNet the actual image size so it handles the
scaling internally. OpenCV's FaceDetectorYN does multi-scale detection
natively when given the correct input dimensions.
"""

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from autosorter.paths import YUNET_MODEL


@dataclass
class DetectedFace:
    """
    One face detected in an image.

    A dataclass is like a simple container class that Python auto-generates
    __init__, __repr__, and __eq__ for. It's cleaner than a regular class
    when you just need to hold data.

    Usage:
        face = DetectedFace(bbox=(100, 50, 180, 200), ...)
        print(face.bbox)        # (100, 50, 180, 200)
        print(face.confidence)  # 0.95
    """
    # Bounding box: (x, y, width, height) in original image pixels
    bbox: Tuple[int, int, int, int]

    # 5 facial landmarks as [(x1,y1), (x2,y2), ...] in original image pixels
    # Order: left_eye, right_eye, nose, left_mouth, right_mouth
    landmarks: List[Tuple[float, float]]

    # Detection confidence: 0.0 (uncertain) to 1.0 (very confident)
    confidence: float

    def landmarks_json(self) -> str:
        """Serialize landmarks to JSON for database storage."""
        return json.dumps(self.landmarks)

    def quality_score(self) -> float:
        """
        Estimate face quality for recognition.

        Higher quality = better for face recognition. Based on:
        1. Face SIZE — bigger faces have more detail (more important)
        2. Detection CONFIDENCE — high confidence usually means frontal, clear face

        A face that's 200px wide is much better than one that's 30px wide.
        We use the geometric mean of width and height as the size metric.

        Quality score ranges from 0.0 to ~1.0:
        - < 0.2: tiny face, probably unusable for recognition
        - 0.2-0.5: small but might work
        - 0.5-0.8: good
        - > 0.8: excellent (large, clear, frontal)
        """
        _, _, w, h = self.bbox
        # Geometric mean of dimensions (better than just width or just height)
        face_size = (w * h) ** 0.5

        # Normalize size to 0-1 range (256px+ = max quality)
        # min() caps at 1.0 so huge faces don't get scores > 1.0
        size_score = min(face_size / 256.0, 1.0)

        # Combine size (70% weight) and confidence (30% weight)
        # Size matters more because a blurry large face is still better
        # than a crisp tiny face for recognition.
        return 0.7 * size_score + 0.3 * self.confidence


class FaceDetector:
    """
    Detects faces in images using YuNet.

    This class wraps OpenCV's FaceDetectorYN. We create it ONCE and reuse
    it for every image (loading the model is expensive, running it is cheap).

    Usage:
        detector = FaceDetector()
        faces = detector.detect(image)
        for face in faces:
            print(f"Face at {face.bbox}, confidence={face.confidence:.2f}")
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        min_face_size: int = 60,
    ) -> None:
        """
        Args:
            confidence_threshold: Minimum confidence to keep a face.
                0.7 is conservative — filters out most false positives.
                Lower = more faces found (but more false detections).
                Higher = fewer faces found (but more reliable).

            nms_threshold: Non-Maximum Suppression threshold.
                When YuNet detects the same face multiple times (at slightly
                different positions), NMS merges them into one detection.
                0.3 means: if two detections overlap by more than 30%,
                keep only the one with higher confidence.

            min_face_size: Minimum face size in pixels (width or height).
                Any detected face smaller than this is discarded.
                Real human faces in normal photos are typically 60px+.
                This filters out tiny false positives (cup handles, knots
                in wood, patterns on clothing, etc.).
        """
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._min_face_size = min_face_size

        # Load the YuNet model.
        # We initialize with a dummy size (320, 320) — we'll update it
        # per image in the detect() method.
        self._detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",                    # config (not needed for ONNX)
            (320, 320),            # initial input size (updated per image)
            self._confidence_threshold,
            self._nms_threshold,
        )

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in an image.

        Args:
            image: A BGR numpy array of shape (height, width, 3).
                   This is the format returned by cv2.imdecode().

        Returns:
            A list of DetectedFace objects, sorted by confidence (highest first).
            Empty list if no faces found.

        HOW THIS WORKS INTERNALLY:
        1. We tell YuNet the image dimensions
        2. YuNet scans the image at multiple scales
        3. For each face-like region, it outputs 15 numbers:
           [x, y, w, h, x_le, y_le, x_re, y_re, x_n, y_n, x_lm, y_lm, x_rm, y_rm, score]
           where le=left_eye, re=right_eye, n=nose, lm=left_mouth, rm=right_mouth
        4. We parse these into DetectedFace objects
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # Tell YuNet the actual image dimensions so it can detect
        # faces of all sizes, not just 320×320-scale faces.
        self._detector.setInputSize((w, h))

        # Run detection!
        # retval: status code (we don't use it)
        # results: numpy array of shape (N, 15) where N = number of faces
        #          or None if no faces found
        _, results = self._detector.detect(image)

        if results is None:
            return []

        faces = []
        for row in results:
            # Parse the 15 numbers from YuNet's output
            # Row format: [x, y, w, h, lm1_x, lm1_y, lm2_x, lm2_y, ..., score]
            x, y, face_w, face_h = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            # Clamp bounding box to image boundaries
            # (YuNet sometimes returns coordinates slightly outside the image)
            x = max(0, x)
            y = max(0, y)
            face_w = min(face_w, w - x)
            face_h = min(face_h, h - y)

            # Skip degenerate boxes (zero or negative dimensions)
            if face_w <= 0 or face_h <= 0:
                continue

            # MINIMUM SIZE FILTER: skip tiny detections.
            # A real face in a normal photo is at least 60px wide.
            # Anything smaller is likely a false positive (cup, logo, pattern).
            if face_w < self._min_face_size or face_h < self._min_face_size:
                continue

            # Parse the 5 landmarks
            landmarks = [
                (float(row[4]),  float(row[5])),   # left eye
                (float(row[6]),  float(row[7])),   # right eye
                (float(row[8]),  float(row[9])),   # nose tip
                (float(row[10]), float(row[11])),  # left mouth corner
                (float(row[12]), float(row[13])),  # right mouth corner
            ]

            # Confidence is the last value
            confidence = float(row[14])

            faces.append(DetectedFace(
                bbox=(x, y, face_w, face_h),
                landmarks=landmarks,
                confidence=confidence,
            ))

        # Sort by confidence, highest first
        faces.sort(key=lambda f: f.confidence, reverse=True)

        return faces
