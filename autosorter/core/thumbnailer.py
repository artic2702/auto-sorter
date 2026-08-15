"""
Thumbnail generator — creates small preview images for the UI grid.

=== WHAT IS THIS FILE? ===
When we display a grid of 500 photos, we can't load the full-size originals
(each could be 10-20 MB). Instead, we create tiny thumbnails (~15-30 KB each)
and display those. The original is only loaded when the user clicks to view.

=== WHY NOT cv2.imread()? ===
On Windows, cv2.imread("D:\\Photos\\अर्णव\\pic.jpg") SILENTLY FAILS if the
path contains non-ASCII characters (Hindi, Chinese, emoji, etc.).
It returns None with no error message. This is a known OpenCV bug.

THE FIX: Read the file as raw bytes using np.fromfile() (which handles Unicode
paths correctly), then decode the bytes into an image using cv2.imdecode().
This two-step approach works for ALL filenames on ALL Windows systems.

=== WHY 256×256? ===
- Small enough to be fast: ~15 KB per thumbnail, loads in <1ms
- Large enough to recognize faces and content at a glance
- Standard size used by Windows Explorer, macOS Finder, etc.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# Maximum dimension for thumbnails (pixels).
# The thumbnail will fit inside a 256×256 box while keeping aspect ratio.
THUMB_MAX = 256

# JPEG quality for thumbnails (0-100). 85 is a good balance between
# quality and file size. Below 70, you start seeing visible artifacts.
THUMB_QUALITY = 85


def _read_image_safe(file_path: str) -> Optional[np.ndarray]:
    """
    Read an image file in a way that handles Unicode Windows paths.

    This is the CORRECT way to load images on Windows:
      1. np.fromfile() reads raw bytes from disk (handles any filename)
      2. cv2.imdecode() converts those bytes into a pixel array

    Args:
        file_path: Full path to the image file.

    Returns:
        A NumPy array of shape (height, width, 3) in BGR color order,
        or None if the file couldn't be decoded (corrupt, unsupported format).

    WHAT IS BGR?
    OpenCV uses BGR (Blue-Green-Red) instead of the more common RGB.
    This is a historical accident from 1990s camera hardware. It doesn't
    matter for our purposes — we just need to be consistent.
    """
    try:
        # Step 1: Read raw bytes from disk
        # np.fromfile reads the file as a flat array of bytes (uint8 = 0-255)
        raw_bytes = np.fromfile(file_path, dtype=np.uint8)

        # Step 2: Decode bytes into an image
        # cv2.IMREAD_COLOR = load as 3-channel BGR color image
        # Even if the original is grayscale, we get 3 channels (for consistency)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        return img  # None if decoding failed (corrupt file)

    except (OSError, ValueError):
        # File doesn't exist, permission denied, or invalid data
        return None


def generate_thumbnail(
    image_path: str,
    thumbs_dir: Path,
    image_id: int,
) -> Optional[Tuple[str, int, int]]:
    """
    Create a thumbnail for a single image.

    Args:
        image_path: Full path to the original image file.
        thumbs_dir: Directory where thumbnails are saved.
        image_id:   Database row ID (used to name the thumbnail file).

    Returns:
        A tuple of (thumb_path, original_width, original_height) on success,
        or None if the image couldn't be read.

    HOW THE RESIZE WORKS:
    Original: 4000×3000 (landscape photo)
    Max: 256
    Scale = 256 / 4000 = 0.064
    New size = 4000 × 0.064 = 256, 3000 × 0.064 = 192
    Result: 256×192 thumbnail (fits in 256×256 box, aspect ratio preserved)

    Original: 1080×1920 (portrait phone photo)
    Scale = 256 / 1920 = 0.133
    New size = 1080 × 0.133 = 144, 1920 × 0.133 = 256
    Result: 144×256 thumbnail
    """
    # Read the original image
    img = _read_image_safe(image_path)
    if img is None:
        return None

    # Get original dimensions
    orig_h, orig_w = img.shape[:2]
    # shape is (height, width, channels) — note height comes FIRST!
    # This is because images are stored as 2D arrays of rows (height) × columns (width).

    if orig_h == 0 or orig_w == 0:
        return None  # Degenerate image

    # Calculate scale factor to fit within THUMB_MAX × THUMB_MAX
    scale = THUMB_MAX / max(orig_h, orig_w)

    if scale < 1.0:
        # Image is bigger than 256px → shrink it
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        # cv2.INTER_AREA is the best interpolation method for shrinking.
        # It averages pixels together (anti-aliasing), avoiding jagged edges.
        # Other methods (INTER_LINEAR, INTER_NEAREST) look worse when downscaling.
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Image is already small enough → use as-is
        thumb = img

    # Create thumbs directory if it doesn't exist
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Name the thumbnail by its database ID: "00001.jpg", "00002.jpg", etc.
    # Zero-padded to 6 digits so they sort correctly in file explorers.
    thumb_filename = f"{image_id:06d}.jpg"
    thumb_path = thumbs_dir / thumb_filename

    # Save as JPEG with our chosen quality
    # cv2.imencode returns (success_bool, encoded_bytes)
    success, encoded = cv2.imencode(
        ".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY]
    )

    if not success:
        return None

    # Write bytes to disk (handles Unicode paths, unlike cv2.imwrite)
    encoded.tofile(str(thumb_path))

    return str(thumb_path), orig_w, orig_h


def compute_sha256(file_path: str) -> Optional[str]:
    """
    Compute the SHA-256 hash of a file.

    WHY WE HASH FILES:
    If the user moves "beach.jpg" from D:\\Photos to D:\\Backup\\Photos,
    the PATH changes but the FILE CONTENT is identical. The SHA-256 hash
    of the content stays the same, so we can detect: "this is the same file,
    just in a new location" — and reattach all its face data instead of
    re-processing.

    WHAT IS SHA-256?
    It's a mathematical function that turns any amount of data into a fixed
    64-character string. Like a fingerprint for files:
    - Same content → always the same hash
    - Even 1 bit different → completely different hash
    - Practically impossible for two different files to have the same hash

    We read the file in 64 KB chunks to avoid loading huge files entirely
    into memory.
    """
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:  # "rb" = read binary
            while True:
                chunk = f.read(65536)  # 64 KB at a time
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()  # Returns a 64-character hex string
    except OSError:
        return None
