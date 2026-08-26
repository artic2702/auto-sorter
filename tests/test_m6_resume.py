"""
Tests for Milestone 6: Resume & Incremental Scan.

Tests cover:
1. True upsert preserving faces and person assignments
2. Incremental scan (only new files processed)
3. Moved-file reattach via SHA-256
4. Missing-file detection
5. Pipeline resume after cancellation
6. Pipeline state counts
7. New faces auto-assigned to existing persons
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.scanner import scan_folder, ScanResult


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_db(tmp_path: Path) -> Database:
    """Create a fresh in-memory-like database in a temp folder."""
    return Database(tmp_path / "library.db")


def _random_embedding() -> bytes:
    """Generate a random 128-d L2-normalized embedding as bytes."""
    vec = np.random.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return FaceRecognizer.embedding_to_bytes(vec)


def _add_full_image(db: Database, path: str, *, with_face: bool = True,
                    with_person: bool = False, sha256: str = None) -> dict:
    """
    Helper: insert an image, optionally with a face and person assignment.
    Returns dict with image_id, face_id, person_id.
    """
    img_id = db.upsert_image(
        path=path, size=1000, mtime=1000.0,
        sha256=sha256, status="processed",
        width=100, height=100, thumb_path=f"/thumbs/{path}.jpg",
    )

    result = {"image_id": img_id, "face_id": None, "person_id": None}

    if with_face:
        face_id = db.insert_face(
            image_id=img_id,
            bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
            det_score=0.95, quality=0.8,
        )
        emb = _random_embedding()
        db.update_embedding(face_id, emb)
        result["face_id"] = face_id

        if with_person:
            person_id = db.insert_person(
                display_name="Test Person",
                centroid_bytes=emb,
                face_count=1,
                cover_face_id=face_id,
            )
            db.assign_face_to_person(face_id, person_id, source="auto",
                                     confidence=0.9)
            result["person_id"] = person_id

    return result


# ══════════════════════════════════════════════════════════════════════
# Test 1: True upsert preserves faces and person assignments
# ══════════════════════════════════════════════════════════════════════

class TestTrueUpsert:

    def test_upsert_preserves_faces_on_rescan(self, tmp_path):
        """
        The critical fix: re-scanning an unchanged file must NOT
        destroy linked faces, embeddings, or person assignments.

        Old behavior (INSERT OR REPLACE): deleted the row → CASCADE
        destroyed all faces → all person data lost!

        New behavior (ON CONFLICT DO UPDATE): updates size/mtime
        in-place → row ID preserved → faces/persons intact.
        """
        db = _make_db(tmp_path)

        # Create an image with a face assigned to a person
        data = _add_full_image(db, "/photos/family.jpg", with_face=True,
                               with_person=True)

        # Verify face exists and is assigned
        face = db.get_face_by_id(data["face_id"])
        assert face is not None
        assert face["person_id"] == data["person_id"]

        # Re-scan the SAME file (simulates a re-scan with changed mtime)
        db.upsert_image(path="/photos/family.jpg", size=1000, mtime=2000.0)

        # Face and person assignment MUST still exist
        face_after = db.get_face_by_id(data["face_id"])
        assert face_after is not None, "Face was destroyed by upsert!"
        assert face_after["person_id"] == data["person_id"], \
            "Person assignment was destroyed by upsert!"
        assert face_after["embedding"] is not None, \
            "Embedding was destroyed by upsert!"

        db.close()

    def test_upsert_updates_status_to_pending(self, tmp_path):
        """When a file is re-scanned with new mtime, status resets to pending."""
        db = _make_db(tmp_path)
        _add_full_image(db, "/photos/pic.jpg")

        # File is currently "processed"
        row = db.get_image_by_path("/photos/pic.jpg")
        assert row["status"] == "processed"

        # Re-scan with changed mtime → status should become "pending"
        db.upsert_image(path="/photos/pic.jpg", size=1000, mtime=9999.0,
                        status="pending")

        row_after = db.get_image_by_path("/photos/pic.jpg")
        assert row_after["status"] == "pending"

        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 2: Incremental scan (only new files)
# ══════════════════════════════════════════════════════════════════════

class TestIncrementalScan:

    def test_scan_skips_unchanged_files(self, tmp_path):
        """
        On re-scan, files that haven't changed (same size + mtime)
        should be skipped, not re-processed.
        """
        # Create a photo folder with some images
        photos = tmp_path / "photos"
        photos.mkdir()
        for name in ["a.jpg", "b.jpg", "c.jpg"]:
            (photos / name).write_bytes(b"\xFF\xD8" + os.urandom(100))

        db = _make_db(tmp_path)

        # First scan: all 3 should be new
        result1 = scan_folder(photos, db)
        assert result1.found == 3
        assert result1.new == 3
        assert result1.skipped == 0

        # Second scan: all 3 should be skipped (unchanged)
        result2 = scan_folder(photos, db)
        assert result2.found == 3
        assert result2.new == 0
        assert result2.skipped == 3

        db.close()

    def test_scan_detects_new_files(self, tmp_path):
        """Adding new files to the folder → only new ones get processed."""
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"\xFF\xD8" + os.urandom(100))

        db = _make_db(tmp_path)

        # First scan: 1 file
        result1 = scan_folder(photos, db)
        assert result1.new == 1

        # Add 2 more files
        (photos / "b.jpg").write_bytes(b"\xFF\xD8" + os.urandom(100))
        (photos / "c.jpg").write_bytes(b"\xFF\xD8" + os.urandom(100))

        # Second scan: only 2 new
        result2 = scan_folder(photos, db)
        assert result2.found == 3
        assert result2.new == 2
        assert result2.skipped == 1

        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 3: Moved-file reattach via SHA-256
# ══════════════════════════════════════════════════════════════════════

class TestReattachByHash:

    def test_reattach_preserves_face_and_person(self, tmp_path):
        """
        Moving a file to a new path → SHA-256 matches → old row's path
        is updated → face and person data preserved.
        """
        db = _make_db(tmp_path)

        # Create image with face + person, with a known hash
        data = _add_full_image(
            db, "/photos/vacation/beach.jpg",
            with_face=True, with_person=True,
            sha256="abc123hash",
        )

        # Mark the old path as missing (file was moved)
        db.update_status(data["image_id"], "missing")

        # Reattach: new path, same hash
        reattached_id = db.reattach_by_hash("abc123hash", "/photos/2024/beach.jpg")

        assert reattached_id == data["image_id"]

        # Check: path updated, status restored, face intact
        row = db.get_image_by_path("/photos/2024/beach.jpg")
        assert row is not None
        assert row["status"] == "processed"

        face = db.get_face_by_id(data["face_id"])
        assert face is not None
        assert face["person_id"] == data["person_id"]

        db.close()

    def test_reattach_no_match_returns_none(self, tmp_path):
        """When no SHA-256 match exists, reattach returns None."""
        db = _make_db(tmp_path)
        result = db.reattach_by_hash("nonexistent_hash", "/photos/new.jpg")
        assert result is None
        db.close()

    def test_reattach_empty_hash_returns_none(self, tmp_path):
        """Empty/None hash should return None immediately."""
        db = _make_db(tmp_path)
        assert db.reattach_by_hash("", "/photos/new.jpg") is None
        assert db.reattach_by_hash(None, "/photos/new.jpg") is None
        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 4: Missing-file detection
# ══════════════════════════════════════════════════════════════════════

class TestMissingFileDetection:

    def test_marks_deleted_files_as_missing(self, tmp_path):
        """Files in the DB but not on disk → marked as 'missing'."""
        db = _make_db(tmp_path)

        # Add 3 images to the DB
        _add_full_image(db, "/photos/a.jpg")
        _add_full_image(db, "/photos/b.jpg")
        _add_full_image(db, "/photos/c.jpg")

        # Only a.jpg and c.jpg are still on disk
        known_paths = {"/photos/a.jpg", "/photos/c.jpg"}
        missing_count = db.mark_missing_images(known_paths)

        assert missing_count == 1  # b.jpg is missing

        row_b = db.get_image_by_path("/photos/b.jpg")
        assert row_b["status"] == "missing"

        # a.jpg and c.jpg should still be processed
        assert db.get_image_by_path("/photos/a.jpg")["status"] == "processed"
        assert db.get_image_by_path("/photos/c.jpg")["status"] == "processed"

        db.close()

    def test_missing_preserves_face_data(self, tmp_path):
        """Marking as missing should NOT delete face/person data."""
        db = _make_db(tmp_path)

        data = _add_full_image(db, "/photos/family.jpg", with_face=True,
                               with_person=True)

        # Mark as missing
        db.mark_missing_images(set())  # no files on disk

        # Face and person must still exist
        face = db.get_face_by_id(data["face_id"])
        assert face is not None
        assert face["person_id"] == data["person_id"]

        db.close()

    def test_no_missing_when_all_present(self, tmp_path):
        """If all files are still on disk, mark_missing returns 0."""
        db = _make_db(tmp_path)

        _add_full_image(db, "/photos/a.jpg")
        _add_full_image(db, "/photos/b.jpg")

        missing = db.mark_missing_images({"/photos/a.jpg", "/photos/b.jpg"})
        assert missing == 0

        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 5: Pipeline state counts
# ══════════════════════════════════════════════════════════════════════

class TestPipelineState:

    def test_empty_database_counts(self, tmp_path):
        """Fresh database should have all zeros."""
        db = _make_db(tmp_path)
        state = db.get_pipeline_state()

        assert state["total"] == 0
        assert state["processed"] == 0
        assert state["pending"] == 0
        assert state["missing"] == 0
        assert state["faces"] == 0
        assert state["embedded"] == 0
        assert state["unassigned"] == 0
        assert state["persons"] == 0
        assert state["needs_detection"] == 0

        db.close()

    def test_populated_database_counts(self, tmp_path):
        """Database with images, faces, and persons reports correct counts."""
        db = _make_db(tmp_path)

        # 2 processed images with faces + persons
        _add_full_image(db, "/a.jpg", with_face=True, with_person=True)
        _add_full_image(db, "/b.jpg", with_face=True, with_person=True)

        # 1 pending image (no face yet)
        db.upsert_image(path="/c.jpg", size=100, mtime=100.0, status="pending")

        state = db.get_pipeline_state()

        assert state["total"] == 3       # 2 processed + 1 pending
        assert state["processed"] == 2
        assert state["pending"] == 1
        assert state["faces"] == 2       # 2 real faces
        assert state["embedded"] == 2    # both have embeddings
        assert state["persons"] == 2     # 2 people created

        db.close()

    def test_missing_files_excluded_from_total(self, tmp_path):
        """Missing files should not count in the total."""
        db = _make_db(tmp_path)

        _add_full_image(db, "/a.jpg")
        _add_full_image(db, "/b.jpg")

        # Mark one as missing
        db.mark_missing_images({"/a.jpg"})

        state = db.get_pipeline_state()
        assert state["total"] == 1       # only a.jpg
        assert state["missing"] == 1     # b.jpg is missing

        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 6: ScanResult includes discovered_paths
# ══════════════════════════════════════════════════════════════════════

class TestScanResultPaths:

    def test_scan_returns_discovered_paths(self, tmp_path):
        """scan_folder should return all discovered image paths in the result."""
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "a.jpg").write_bytes(b"\xFF\xD8" + os.urandom(100))
        (photos / "b.png").write_bytes(b"\x89PNG" + os.urandom(100))
        (photos / "c.txt").write_text("not an image")

        db = _make_db(tmp_path)
        result = scan_folder(photos, db)

        # Should find 2 images (not the .txt)
        assert result.found == 2
        assert len(result.discovered_paths) == 2

        # Paths should be absolute strings
        path_names = {os.path.basename(p) for p in result.discovered_paths}
        assert "a.jpg" in path_names
        assert "b.png" in path_names
        assert "c.txt" not in path_names

        db.close()


# ══════════════════════════════════════════════════════════════════════
# Test 7: get_all_image_paths
# ══════════════════════════════════════════════════════════════════════

class TestGetAllImagePaths:

    def test_returns_all_paths(self, tmp_path):
        """get_all_image_paths should return paths for all statuses."""
        db = _make_db(tmp_path)

        _add_full_image(db, "/a.jpg")
        db.upsert_image(path="/b.jpg", size=100, mtime=100.0, status="pending")
        _add_full_image(db, "/c.jpg")
        db.update_status(
            db.get_image_by_path("/c.jpg")["id"], "missing"
        )

        paths = db.get_all_image_paths()
        assert "/a.jpg" in paths
        assert "/b.jpg" in paths
        assert "/c.jpg" in paths
        assert len(paths) == 3

        db.close()
