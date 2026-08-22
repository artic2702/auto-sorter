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
import io
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Register HEIF opener so PIL Image.open() automatically supports .heic / .heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

try:
    import rawpy
except ImportError:
    rawpy = None  # type: ignore[assignment]


# Maximum dimension for thumbnails (pixels).
# The thumbnail will fit inside a 256×256 box while keeping aspect ratio.
THUMB_MAX = 256

# JPEG quality for thumbnails (0-100). 85 is a good balance between
# quality and file size. Below 70, you start seeing visible artifacts.
THUMB_QUALITY = 85

# File extensions recognized as Camera RAW
_RAW_EXTENSIONS = frozenset({
    ".cr2", ".cr3",
    ".nef", ".nrw",
    ".arw", ".srf", ".sr2",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
})


def _decode_raw_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode a camera RAW image (.CR2, .NEF, .ARW, .DNG, etc.).

    LOW-END PC OPTIMIZATION:
    Instead of doing slow Bayer demosaicing on a 30-60 MB sensor dump
    (which takes 2-4 seconds and spikes RAM), we extract the embedded
    full-resolution / preview JPEG stored inside the RAW file by the camera.
    This extraction takes ~5 ms and uses virtually zero extra RAM.
    """
    if rawpy is None:
        return None

    try:
        with rawpy.imread(io.BytesIO(raw_bytes)) as raw:
            # Fast path A: Extract embedded JPEG/Bitmap preview (instant: ~5 ms)
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    thumb_bytes = np.frombuffer(thumb.data, dtype=np.uint8)
                    img = cv2.imdecode(thumb_bytes, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    return cv2.cvtColor(thumb.data, cv2.COLOR_RGB2BGR)
            except Exception:
                pass  # Fallback to postprocessing if no thumbnail exists

            # Fallback path B: Fast half-size demosaic if no embedded preview
            rgb = raw.postprocess(
                half_size=True,
                use_camera_wb=True,
                no_auto_bright=True,
                output_color=rawpy.ColorSpace.sRGB,
            )
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _read_image_safe(file_path: str) -> Optional[np.ndarray]:
    """
    Read an image file in a way that handles Unicode Windows paths,
    standard formats, HEIC/HEIF, and Camera RAW formats.

    Decoding pipeline:
      1. np.fromfile() reads raw bytes from disk (handles Unicode paths)
      2. Fast path: cv2.imdecode() for JPEG, PNG, WebP, BMP, TIFF
      3. RAW path: rawpy embedded preview extraction for .CR2, .NEF, .ARW, .DNG, etc.
      4. Fallback: PIL Image.open() (with pillow-heif) for .heic, .heif, etc.

    Args:
        file_path: Full path to the image file.

    Returns:
        A NumPy array of shape (height, width, 3) in BGR color order,
        or None if the file couldn't be decoded (corrupt, unsupported format).
    """
    try:
        # Step 1: Read raw bytes from disk (Unicode path safe)
        raw_bytes = np.fromfile(file_path, dtype=np.uint8)
        if len(raw_bytes) == 0:
            return None

        # Check extension
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()

        # Step 2: Fast path for standard formats (OpenCV)
        if ext_lower not in _RAW_EXTENSIONS and ext_lower not in {".heic", ".heif"}:
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                return img

        # Step 3: Camera RAW files (.CR2, .NEF, .ARW, .DNG, etc.)
        if ext_lower in _RAW_EXTENSIONS:
            raw_img = _decode_raw_image(raw_bytes.tobytes())
            if raw_img is not None:
                return raw_img

        # Step 4: HEIC/HEIF and other formats via PIL (handles pillow-heif)
        try:
            with Image.open(io.BytesIO(raw_bytes.tobytes())) as pil_img:
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                rgb_array = np.array(pil_img)
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        # Final attempt: try cv2.imdecode in case file had unusual extension
        return cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

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
