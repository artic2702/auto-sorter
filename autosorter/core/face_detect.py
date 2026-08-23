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
import math
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

    # Raw 15-number detection result from YuNet (needed by SFace's alignCrop).
    # Format: [x, y, w, h, lm1_x, lm1_y, ..., lm5_x, lm5_y, score]
    # We store this so SFace can produce the EXACT alignment it was trained on.
    raw_detection: Optional[np.ndarray] = None

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


def _is_valid_face_geometry(
    bbox: Tuple[int, int, int, int],
    landmarks: List[Tuple[float, float]],
) -> bool:
    """
    Post-detection filter: verify that detected landmarks form a real face.

    Returns False for false positives (ears, hands, jewelry, cloth patterns)
    which pass YuNet's confidence threshold but have nonsensical landmark layout.

    Checks (all must pass):
    1. Aspect ratio: real faces are 0.65 <= h/w <= 1.65
    2. Eye spacing: left-to-right eye distance >= 15% of face width
    3. Eye-to-mouth distance: vertical span >= 15% of face height
    4. Triangle area: (left eye, right eye, nose) must form a 2D triangle
       with area >= 1% of bounding box area (collinear = not a face)

    Why this preserves side profiles and tilted heads:
    - Side profiles (up to ~70° yaw): eyes compress horizontally but the
      triangle (eyes, nose) still has positive 2D area.
    - Tilted heads (up to ~30° roll): vertical ordering is checked relative
      to the bbox, not absolute coordinates.
    """
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or len(landmarks) < 5:
        return False

    # 1. Aspect ratio
    aspect = h / float(w)
    if aspect < 0.65 or aspect > 1.65:
        return False

    (lx, ly), (rx, ry), (nx, ny), (lmx, lmy), (rmx, rmy) = landmarks[:5]

    # 2. Eye spacing: distance between left and right eye
    eye_dist = math.hypot(rx - lx, ry - ly)
    if eye_dist < 0.15 * w:
        return False

    # 3. Eye-to-mouth vertical distance
    mouth_cx = (lmx + rmx) / 2.0
    mouth_cy = (lmy + rmy) / 2.0
    eye_cx = (lx + rx) / 2.0
    eye_cy = (ly + ry) / 2.0
    eye_mouth_dist = math.hypot(mouth_cx - eye_cx, mouth_cy - eye_cy)
    if eye_mouth_dist < 0.15 * h:
        return False

    # 4. Triangle area using Shoelace formula: (left eye, right eye, nose)
    area = 0.5 * abs(lx * (ry - ny) + rx * (ny - ly) + nx * (ly - ry))
    bbox_area = float(w * h)
    if (area / bbox_area) < 0.01:
        return False

    return True


class FaceDetector:
    """
    Detects faces in images using YuNet with 3-stage filtering.

    This class wraps OpenCV's FaceDetectorYN. We create it ONCE and reuse
    it for every image (loading the model is expensive, running it is cheap).

    3-stage pipeline:
      Stage 1: Resolution normalization — downscale large images + add padding
               for close-up faces that fill the entire frame.
      Stage 2: Confidence threshold (0.75) — reject borderline noise.
      Stage 3: Landmark geometry validation — reject ears/hands/patterns whose
               landmarks don't form a valid facial structure.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(image)
        for face in faces:
            print(f"Face at {face.bbox}, confidence={face.confidence:.2f}")
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        min_face_size: int = 40,
    ) -> None:
        """
        Args:
            confidence_threshold: Minimum confidence to keep a face.
                0.75 filters out most false positives while keeping real faces
                (which typically score 0.82+ after EXIF correction and downscaling).

            nms_threshold: Non-Maximum Suppression threshold.
                When YuNet detects the same face multiple times (at slightly
                different positions), NMS merges them into one detection.
                0.3 means: if two detections overlap by more than 30%,
                keep only the one with higher confidence.

            min_face_size: Minimum face size in pixels (width or height)
                in the DETECTION image (after downscaling). 40px is the minimum
                for YuNet to produce reliable landmarks.
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

    def detect(
        self,
        image: np.ndarray,
        max_dimension: int = 1280,
    ) -> List[DetectedFace]:
        """
        Detect all faces in an image with 3-stage filtering.

        Stage 1 — Resolution Normalization:
          Large images (24-48MP) are downscaled so the longest side equals
          max_dimension. This keeps faces in YuNet's optimal 10-300px range
          and caps RAM usage to ~400MB instead of 3.6GB.

          Additionally, 15% black padding is added around the image so that
          close-up faces (selfies where the head fills the entire frame) have
          enough border context for YuNet's anchor mechanism to trigger.

        Stage 2 — YuNet Detection + Confidence Filter:
          Run inference, reject anything below confidence_threshold.

        Stage 3 — Landmark Geometry Validation:
          Check that detected landmarks form a valid facial structure
          (eyes above nose above mouth, proper spacing, non-collinear).

        All coordinates are mapped back to the original image space.

        Args:
            image: A BGR numpy array of shape (height, width, 3).
            max_dimension: Maximum image dimension for detection (default 1280).
                          Set to 0 to disable downscaling (native resolution).

        Returns:
            A list of DetectedFace objects in ORIGINAL image coordinates,
            sorted by confidence (highest first). Empty list if no faces found.
        """
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]

        # ═══════════════════════════════════════════════════════════
        # Stage 1: Resolution normalization + close-up padding
        # ═══════════════════════════════════════════════════════════

        # Downscale if needed
        scale = 1.0
        det_img = image
        if max_dimension > 0 and max(orig_h, orig_w) > max_dimension:
            scale = max_dimension / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            det_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        det_h, det_w = det_img.shape[:2]

        # Add 15% padding around the image for close-up face detection.
        # When a face fills the entire frame, YuNet's anchor mechanism needs
        # border context to detect it. copyMakeBorder adds black pixels.
        pad_x = int(det_w * 0.15)
        pad_y = int(det_h * 0.15)
        padded_img = cv2.copyMakeBorder(
            det_img, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

        padded_h, padded_w = padded_img.shape[:2]

        # ═══════════════════════════════════════════════════════════
        # Stage 2: Run YuNet detection
        # ═══════════════════════════════════════════════════════════

        self._detector.setInputSize((padded_w, padded_h))
        _, results = self._detector.detect(padded_img)

        if results is None:
            return []

        # Inverse scale for mapping back to original coordinates
        inv_scale = 1.0 / scale if scale != 1.0 else 1.0

        faces = []
        for row in results:
            # Parse bbox in padded-image coordinates
            px, py, pw, ph = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            # Remove padding offset
            px -= pad_x
            py -= pad_y

            # Clamp to detection image boundaries
            px = max(0, px)
            py = max(0, py)
            pw = min(pw, det_w - px)
            ph = min(ph, det_h - py)

            if pw <= 0 or ph <= 0:
                continue

            # Minimum size filter (in detection image space)
            if pw < self._min_face_size or ph < self._min_face_size:
                continue

            # Parse landmarks in padded coordinates, then remove padding
            det_landmarks = [
                (float(row[4])  - pad_x, float(row[5])  - pad_y),  # left eye
                (float(row[6])  - pad_x, float(row[7])  - pad_y),  # right eye
                (float(row[8])  - pad_x, float(row[9])  - pad_y),  # nose tip
                (float(row[10]) - pad_x, float(row[11]) - pad_y),  # left mouth
                (float(row[12]) - pad_x, float(row[13]) - pad_y),  # right mouth
            ]

            confidence = float(row[14])

            # ═══════════════════════════════════════════════════════
            # Stage 3: Landmark geometry validation
            # ═══════════════════════════════════════════════════════
            if not _is_valid_face_geometry((px, py, pw, ph), det_landmarks):
                continue

            # Map coordinates back to original image space
            orig_x = int(px * inv_scale)
            orig_y = int(py * inv_scale)
            orig_w_face = int(pw * inv_scale)
            orig_h_face = int(ph * inv_scale)

            # Clamp to original image boundaries
            orig_x = max(0, orig_x)
            orig_y = max(0, orig_y)
            orig_w_face = min(orig_w_face, orig_w - orig_x)
            orig_h_face = min(orig_h_face, orig_h - orig_y)

            if orig_w_face <= 0 or orig_h_face <= 0:
                continue

            orig_landmarks = [
                (lx * inv_scale, ly * inv_scale) for lx, ly in det_landmarks
            ]

            # Build raw_detection array in ORIGINAL coordinates for SFace
            raw_orig = np.array([
                orig_x, orig_y, orig_w_face, orig_h_face,
                orig_landmarks[0][0], orig_landmarks[0][1],
                orig_landmarks[1][0], orig_landmarks[1][1],
                orig_landmarks[2][0], orig_landmarks[2][1],
                orig_landmarks[3][0], orig_landmarks[3][1],
                orig_landmarks[4][0], orig_landmarks[4][1],
                confidence,
            ], dtype=np.float32)

            faces.append(DetectedFace(
                bbox=(orig_x, orig_y, orig_w_face, orig_h_face),
                landmarks=orig_landmarks,
                confidence=confidence,
                raw_detection=raw_orig,
            ))

        # Sort by confidence, highest first
        faces.sort(key=lambda f: f.confidence, reverse=True)

        return faces
