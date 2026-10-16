"""
M1 tests: database, scanner, thumbnailer.

=== WHAT ARE THESE TESTS? ===
Each test function checks ONE specific behavior. If any test fails,
we know exactly what's broken. Tests use "tmp_path" — a temporary
folder that pytest creates fresh for each test and cleans up after.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.scanner import IMAGE_EXTENSIONS, is_image_file, scan_folder
from autosorter.core.thumbnailer import compute_sha256, generate_thumbnail


# ──────────────────────────────────────────────────────────
# Helper: create a small test JPEG file on disk
# ──────────────────────────────────────────────────────────

def _make_test_image(path: Path, width: int = 100, height: int = 80) -> None:
    """
    Create a small random JPEG image at the given path.

    We need real image files (not empty files) because the thumbnailer
    actually reads and decodes them. This function creates a valid JPEG
    with random colored pixels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a random image array
    rng = np.random.default_rng(seed=42)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    # Encode as JPEG bytes and write to disk
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    encoded.tofile(str(path))


# ══════════════════════════════════════════════════════════
# DATABASE TESTS
# ══════════════════════════════════════════════════════════

class TestDatabase:
    """Tests for the Database class."""

    def test_creates_tables(self, tmp_path: Path):
        """Opening a database creates the images and meta tables."""
        db = Database(tmp_path / "test.db")
        # Query SQLite's internal table list
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "images" in tables
        assert "meta" in tables
        db.close()

    def test_schema_version_stored(self, tmp_path: Path):
        """The schema version is written to the meta table."""
        db = Database(tmp_path / "test.db")
        cursor = db._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "3"
        db.close()

    def test_upsert_insert(self, tmp_path: Path):
        """upsert_image inserts a new row and returns a valid ID."""
        db = Database(tmp_path / "test.db")
        row_id = db.upsert_image("photo.jpg", 12345, 1720000000.0)
        assert row_id > 0
        assert db.count_images() == 1
        db.close()

    def test_upsert_update(self, tmp_path: Path):
        """upsert_image with the same path replaces the existing row."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("photo.jpg", 12345, 1720000000.0)
        db.upsert_image("photo.jpg", 99999, 1720000001.0)
        # Should still be 1 row, not 2
        assert db.count_images() == 1
        row = db.get_image_by_path("photo.jpg")
        assert row["size"] == 99999  # Updated value
        db.close()

    def test_is_unchanged_true(self, tmp_path: Path):
        """is_unchanged returns True when size and mtime match."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("photo.jpg", 12345, 1720000000.0)
        assert db.is_unchanged("photo.jpg", 12345, 1720000000.0) is True
        db.close()

    def test_is_unchanged_false_different_size(self, tmp_path: Path):
        """is_unchanged returns False when file size differs."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("photo.jpg", 12345, 1720000000.0)
        assert db.is_unchanged("photo.jpg", 99999, 1720000000.0) is False
        db.close()

    def test_is_unchanged_false_unknown_file(self, tmp_path: Path):
        """is_unchanged returns False for a file not in the database."""
        db = Database(tmp_path / "test.db")
        assert db.is_unchanged("unknown.jpg", 12345, 1720000000.0) is False
        db.close()

    def test_count_by_status(self, tmp_path: Path):
        """count_images can filter by status."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("a.jpg", 100, 1.0, status="pending")
        db.upsert_image("b.jpg", 200, 2.0, status="pending")
        db.upsert_image("c.jpg", 300, 3.0, status="processed")
        assert db.count_images() == 3
        assert db.count_images(status="pending") == 2
        assert db.count_images(status="processed") == 1
        db.close()

    def test_iter_pending(self, tmp_path: Path):
        """iter_pending yields only pending images in order."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("a.jpg", 100, 1.0, status="pending")
        db.upsert_image("b.jpg", 200, 2.0, status="processed")
        db.upsert_image("c.jpg", 300, 3.0, status="pending")
        pending = [row["path"] for row in db.iter_pending()]
        assert pending == ["a.jpg", "c.jpg"]
        db.close()

    def test_context_manager(self, tmp_path: Path):
        """Database works as a context manager (with statement)."""
        with Database(tmp_path / "test.db") as db:
            db.upsert_image("photo.jpg", 123, 1.0)
            assert db.count_images() == 1
        # After 'with' block, db is closed — no crash


# ══════════════════════════════════════════════════════════
# SCANNER TESTS
# ══════════════════════════════════════════════════════════

class TestScanner:
    """Tests for the file scanner."""

    def test_is_image_file(self):
        """is_image_file recognizes common image extensions."""
        assert is_image_file("photo.jpg") is True
        assert is_image_file("PHOTO.JPG") is True  # case insensitive
        assert is_image_file("image.png") is True
        assert is_image_file("notes.txt") is False
        assert is_image_file("data.csv") is False
        assert is_image_file("app.exe") is False

    def test_scan_finds_images(self, tmp_path: Path):
        """Scanner finds image files and ignores non-image files."""
        # Create test files
        _make_test_image(tmp_path / "photo1.jpg")
        _make_test_image(tmp_path / "photo2.png")
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b,c")

        db = Database(tmp_path / "data" / "test.db")
        result = scan_folder(tmp_path, db)

        assert result.found == 2   # only jpg and png
        assert result.new == 2
        assert result.skipped == 0
        db.close()

    def test_scan_recursive(self, tmp_path: Path):
        """Scanner finds images in subdirectories."""
        _make_test_image(tmp_path / "root.jpg")
        _make_test_image(tmp_path / "vacation" / "beach.jpg")
        _make_test_image(tmp_path / "vacation" / "2024" / "sunset.jpg")

        db = Database(tmp_path / "data" / "test.db")
        result = scan_folder(tmp_path, db)

        assert result.found == 3
        assert result.new == 3
        db.close()

    def test_scan_skips_unchanged(self, tmp_path: Path):
        """Second scan skips files that haven't changed."""
        _make_test_image(tmp_path / "photo.jpg")

        db = Database(tmp_path / "data" / "test.db")

        # First scan
        result1 = scan_folder(tmp_path, db)
        assert result1.new == 1

        # Second scan — same file, unchanged
        result2 = scan_folder(tmp_path, db)
        assert result2.found == 1
        assert result2.new == 0
        assert result2.skipped == 1

        db.close()

    def test_scan_cancellation(self, tmp_path: Path):
        """Scanner stops when cancel_check returns True."""
        for i in range(20):
            _make_test_image(tmp_path / f"photo_{i:02d}.jpg")

        db = Database(tmp_path / "data" / "test.db")
        # Cancel immediately
        result = scan_folder(tmp_path, db, cancel_check=lambda: True)

        # Should have found very few (or zero) files
        assert result.found < 20
        db.close()


# ══════════════════════════════════════════════════════════
# THUMBNAILER TESTS
# ══════════════════════════════════════════════════════════

class TestThumbnailer:
    """Tests for thumbnail generation."""

    def test_generate_thumbnail(self, tmp_path: Path):
        """Generating a thumbnail creates a JPEG file with correct dimensions."""
        # Create a 400x300 test image
        img_path = tmp_path / "large.jpg"
        _make_test_image(img_path, width=400, height=300)

        thumbs_dir = tmp_path / "thumbs"
        result = generate_thumbnail(str(img_path), thumbs_dir, image_id=1)

        assert result is not None
        thumb_path, orig_w, orig_h = result
        assert orig_w == 400
        assert orig_h == 300

        # Thumbnail file should exist
        assert Path(thumb_path).is_file()

        # Thumbnail should be smaller than original
        thumb_size = Path(thumb_path).stat().st_size
        orig_size = img_path.stat().st_size
        assert thumb_size <= orig_size

    def test_thumbnail_preserves_aspect_ratio(self, tmp_path: Path):
        """Thumbnail preserves the aspect ratio of the original."""
        img_path = tmp_path / "wide.jpg"
        _make_test_image(img_path, width=800, height=400)  # 2:1 aspect ratio

        thumbs_dir = tmp_path / "thumbs"
        result = generate_thumbnail(str(img_path), thumbs_dir, image_id=1)
        assert result is not None
        thumb_path = result[0]

        # Read thumbnail and check dimensions
        thumb = cv2.imread(thumb_path)
        assert thumb is not None
        th, tw = thumb.shape[:2]
        # Should be 256x128 (scaled down 2:1)
        assert tw == 256
        assert th == 128

    def test_thumbnail_nonexistent_file(self, tmp_path: Path):
        """Thumbnailing a non-existent file returns None (doesn't crash)."""
        result = generate_thumbnail(
            str(tmp_path / "nope.jpg"), tmp_path / "thumbs", image_id=1
        )
        assert result is None

    def test_sha256_consistent(self, tmp_path: Path):
        """SHA-256 hash of the same file is always the same."""
        img_path = tmp_path / "photo.jpg"
        _make_test_image(img_path)

        hash1 = compute_sha256(str(img_path))
        hash2 = compute_sha256(str(img_path))
        assert hash1 is not None
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters

    def test_sha256_nonexistent(self, tmp_path: Path):
        """SHA-256 of a non-existent file returns None."""
        result = compute_sha256(str(tmp_path / "nope.jpg"))
        assert result is None
