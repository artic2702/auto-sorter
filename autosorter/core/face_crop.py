"""
Face cropping and alignment — prepares face images for display and recognition.

=== WHAT IS THIS FILE? ===
After YuNet detects a face, we need to:
1. CROP it out of the photo (for display as a face thumbnail)
2. ALIGN it so the eyes are level (critical for recognition accuracy)

=== WHY ALIGNMENT MATTERS ===
Imagine comparing two photos of the same person:
  - Photo A: head tilted 30° to the right
  - Photo B: head straight

Without alignment, the faceprints (128 numbers) would be quite different
because the pixels are in different positions. After alignment (rotating
so eyes are horizontal, centering the face), the faceprints become very
similar — making recognition much more accurate.

The alignment process:
  1. Find the left eye and right eye positions (from landmarks)
  2. Calculate the angle between them
  3. Compute a rotation + scale transform
  4. Apply it to produce a standardized 112×112 face crop

Before alignment:          After alignment:
  ┌──────────┐              ┌──────────┐
  │     •    │              │  •    •  │  ← eyes now level
  │  •       │              │          │
  │     •    │   ──────►    │     •    │
  │   •   •  │              │   •   •  │
  └──────────┘              └──────────┘
  (head tilted)             (standardized)

=== WHAT IS AN AFFINE TRANSFORM? ===
An affine transform is a mathematical operation that can:
  - ROTATE an image
  - SCALE it (zoom in/out)
  - TRANSLATE it (shift left/right/up/down)
All at once, in a single step. It's defined by a 2×3 matrix.

OpenCV's cv2.getAffineTransform() computes this matrix given 3 source
points and 3 destination points. We use the two eye positions + a third
derived point to compute where the face should end up in the 112×112 output.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Standard output size for aligned face crops.
# SFace (the recognition model in M3) expects exactly 112×112.
ALIGNED_FACE_SIZE = 112

# Where the eyes should be in the aligned output.
# These are the "ideal" eye positions in a 112×112 image, based on
# the standard used by SFace and most face recognition models.
# Left eye at roughly (0.35 * 112, 0.35 * 112) = (39, 39)
# Right eye at roughly (0.65 * 112, 0.35 * 112) = (73, 39)
_LEFT_EYE_TARGET = (0.35 * ALIGNED_FACE_SIZE, 0.35 * ALIGNED_FACE_SIZE)
_RIGHT_EYE_TARGET = (0.65 * ALIGNED_FACE_SIZE, 0.35 * ALIGNED_FACE_SIZE)


def align_face(
    image: np.ndarray,
    landmarks: List[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """
    Produce an aligned 112×112 face crop from the original image.

    This is the image that will be fed to SFace for recognition (M3).

    Args:
        image:     The full original image (BGR, any size).
        landmarks: 5 facial landmarks from YuNet:
                   [left_eye, right_eye, nose, left_mouth, right_mouth]

    Returns:
        A 112×112 BGR numpy array (the aligned face), or None on failure.

    HOW THE MATH WORKS:
    1. Get left eye (x1, y1) and right eye (x2, y2) from landmarks
    2. Calculate the angle: angle = atan2(y2-y1, x2-x1)
       → If eyes are level, angle = 0
       → If head tilted right, angle > 0
    3. Calculate the distance between eyes: dist = sqrt((x2-x1)² + (y2-y1)²)
    4. Calculate scale: scale = desired_eye_distance / actual_eye_distance
    5. Build a rotation matrix centered on the midpoint between the eyes
    6. Apply to the image → output: 112×112 with eyes at standard positions
    """
    if landmarks is None or len(landmarks) < 2:
        return None

    # Extract eye positions
    left_eye = np.array(landmarks[0], dtype=np.float32)    # (x, y)
    right_eye = np.array(landmarks[1], dtype=np.float32)   # (x, y)

    # Calculate angle between eyes (in degrees)
    # atan2 gives us the angle in radians; we convert to degrees for OpenCV
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    # If eyes are perfectly level: angle = 0
    # If right eye is higher: angle < 0 (need to rotate clockwise)

    # Calculate distance between eyes (in pixels)
    eye_dist = np.sqrt(dx * dx + dy * dy)
    if eye_dist < 1.0:
        return None  # Eyes are at the same point — degenerate case

    # Desired eye distance in the output image
    desired_dist = _RIGHT_EYE_TARGET[0] - _LEFT_EYE_TARGET[0]

    # Scale factor: how much to zoom in/out
    scale = desired_dist / eye_dist

    # Center of rotation = midpoint between the eyes
    center = ((left_eye[0] + right_eye[0]) / 2.0,
              (left_eye[1] + right_eye[1]) / 2.0)

    # Build the rotation matrix
    # cv2.getRotationMatrix2D(center, angle, scale) returns a 2×3 matrix
    # that rotates around 'center' by 'angle' degrees and scales by 'scale'
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Adjust the translation so the face center ends up in the right place
    # in the 112×112 output. Without this, the face would be rotated and
    # scaled but positioned incorrectly.
    desired_center_x = ALIGNED_FACE_SIZE * 0.5
    desired_center_y = ALIGNED_FACE_SIZE * 0.45  # slightly above center (more forehead)
    M[0, 2] += desired_center_x - center[0]
    M[1, 2] += desired_center_y - center[1]

    # Apply the transform!
    # cv2.warpAffine applies the 2×3 matrix M to every pixel in the image,
    # producing a new image of size (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE)
    aligned = cv2.warpAffine(
        image,
        M,
        (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE),
        flags=cv2.INTER_LINEAR,         # bilinear interpolation (smooth)
        borderMode=cv2.BORDER_CONSTANT, # fill missing areas with black
        borderValue=(0, 0, 0),
    )

    return aligned


def crop_face_thumb(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Crop a face region from the image with some padding for context.

    Unlike align_face(), this just does a simple rectangular crop with
    extra margin around the face. Used for face thumbnail display.

    Args:
        image:   The full original image.
        bbox:    (x, y, w, h) bounding box from YuNet.
        padding: How much extra space around the face (0.3 = 30%).
                 This shows some hair, neck, and background for context.

    Returns:
        The cropped face region, or None if the bbox is invalid.

    Example with padding=0.3:
        Original bbox: (100, 50, 80, 100)
        Padding: 80 * 0.3 = 24 horizontal, 100 * 0.3 = 30 vertical
        Final crop: (76, 20, 128, 160) — bigger rectangle, centered
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Calculate padding in pixels
    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    # Expand the bounding box by padding, but clamp to image boundaries
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + bw + pad_w)
    y2 = min(h, y + bh + pad_h)

    # Crop (numpy slicing: [rows, columns] = [y:y+h, x:x+w])
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return crop


def save_face_thumb(
    face_crop: np.ndarray,
    face_thumbs_dir: Path,
    face_id: int,
    size: int = ALIGNED_FACE_SIZE,
) -> str:
    """
    Resize a face crop to a standard size and save it as JPEG.

    Args:
        face_crop:      The cropped face image (any size).
        face_thumbs_dir: Directory to save face thumbnails.
        face_id:         Database face ID (used for filename).
        size:            Output size (square). Default 112×112.

    Returns:
        The path to the saved face thumbnail.
    """
    face_thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Resize to standard size
    thumb = cv2.resize(face_crop, (size, size), interpolation=cv2.INTER_AREA)

    # Save as JPEG
    filename = f"face_{face_id:06d}.jpg"
    save_path = face_thumbs_dir / filename

    success, encoded = cv2.imencode(
        ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if success:
        encoded.tofile(str(save_path))

    return str(save_path)
