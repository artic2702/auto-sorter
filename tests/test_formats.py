"""
Format tests: HEIC/HEIF, Camera RAW, and extended format support.

=== WHAT ARE WE TESTING? ===
1. Scanner: recognizes all RAW extensions (.cr2, .nef, .arw, .dng, etc.) and HEIC (.heic, .heif)
2. Image loader (_read_image_safe):
   - Standard JPEG/PNG/WebP reading
   - Unicode path handling
   - Corrupted and empty files return None without crashing
   - HEIC / RAW format decode pathways
3. Thumbnail generation from loaded images
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image
import pytest

from autosorter.core.scanner import is_image_file, IMAGE_EXTENSIONS
from autosorter.core.thumbnailer import (
    _read_image_safe,
    _decode_raw_image,
    generate_thumbnail,
)


class TestFormatRecognition:
    """Tests for image format recognition in scanner."""

    @pytest.mark.parametrize("ext", [
        ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
        ".heic", ".heif",
        ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef",
    ])
    def test_recognizes_all_supported_extensions(self, ext: str):
        """All supported extensions return True for is_image_file."""
        assert is_image_file(f"photo{ext}") is True
        assert is_image_file(f"photo{ext.upper()}") is True

    def test_rejects_non_images(self):
        """Non-image extensions return False."""
        assert is_image_file("document.pdf") is False
        assert is_image_file("script.py") is False
        assert is_image_file("data.json") is False
        assert is_image_file("archive.zip") is False


class TestImageReader:
    """Tests for _read_image_safe and decoders."""

    def test_read_standard_jpeg(self, tmp_path: Path):
        """Standard JPEG loads as BGR numpy array."""
        img_path = tmp_path / "test.jpg"
        dummy = np.zeros((100, 150, 3), dtype=np.uint8)
        dummy[:, :] = (120, 180, 240)  # BGR
        cv2.imwrite(str(img_path), dummy)

        loaded = _read_image_safe(str(img_path))
        assert loaded is not None
        assert loaded.shape == (100, 150, 3)
        assert np.array_equal(loaded[0, 0], [120, 180, 240])

    def test_read_unicode_path(self, tmp_path: Path):
        """Handles non-ASCII Unicode paths cleanly on Windows."""
        img_path = tmp_path / "फोटो_अर्णव_🎉.jpg"
        dummy = np.zeros((80, 80, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", dummy)
        encoded.tofile(str(img_path))

        loaded = _read_image_safe(str(img_path))
        assert loaded is not None
        assert loaded.shape == (80, 80, 3)

    def test_read_empty_file_returns_none(self, tmp_path: Path):
        """Empty 0-byte file returns None safely."""
        empty = tmp_path / "empty.jpg"
        empty.write_bytes(b"")
        assert _read_image_safe(str(empty)) is None

    def test_read_corrupt_file_returns_none(self, tmp_path: Path):
        """Corrupted junk data returns None safely without crashing."""
        corrupt = tmp_path / "corrupt.heic"
        corrupt.write_bytes(b"NOT_A_REAL_HEIC_FILE_HEADER_123456789")
        assert _read_image_safe(str(corrupt)) is None

    def test_read_nonexistent_file_returns_none(self, tmp_path: Path):
        """Nonexistent path returns None safely."""
        missing = tmp_path / "does_not_exist.raw"
        assert _read_image_safe(str(missing)) is None

    def test_heic_pil_fallback_path(self, tmp_path: Path):
        """Simulates PIL / pillow-heif decoding path for HEIC/HEIF files."""
        # Create a valid image file named .heic (PIL can open it via fallback)
        heic_path = tmp_path / "sample.heic"
        pil_img = Image.new("RGB", (60, 40), color=(255, 0, 0))  # Red
        pil_img.save(heic_path, format="PNG")  # saved with PNG container but .heic extension

        loaded = _read_image_safe(str(heic_path))
        assert loaded is not None
        assert loaded.shape == (40, 60, 3)
        # Red in RGB is (255, 0, 0) -> in BGR is (0, 0, 255)
        assert loaded[0, 0, 2] == 255
        assert loaded[0, 0, 0] == 0

    def test_decode_raw_with_embedded_thumb_mock(self):
        """Test _decode_raw_image extracting embedded JPEG thumbnail."""
        dummy_jpeg = np.zeros((50, 50, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode(".jpg", dummy_jpeg)

        mock_raw = MagicMock()
        mock_thumb = MagicMock()
        mock_thumb.format = 1  # ThumbFormat.JPEG
        mock_thumb.data = jpeg_bytes.tobytes()
        mock_raw.extract_thumb.return_value = mock_thumb

        with patch("autosorter.core.thumbnailer.rawpy") as mock_rawpy:
            mock_rawpy.imread.return_value.__enter__.return_value = mock_raw
            mock_rawpy.ThumbFormat.JPEG = 1

            result = _decode_raw_image(b"fake_raw_bytes")
            assert result is not None
            assert result.shape == (50, 50, 3)

    def test_thumbnail_generation_for_extended_formats(self, tmp_path: Path):
        """generate_thumbnail succeeds on supported image formats."""
        img_path = tmp_path / "raw_photo.png"
        dummy = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy)

        thumbs_dir = tmp_path / "thumbs"
        res = generate_thumbnail(str(img_path), thumbs_dir, 1)

        assert res is not None
        thumb_path, orig_w, orig_h = res
        assert orig_w == 800
        assert orig_h == 600
        assert Path(thumb_path).is_file()
