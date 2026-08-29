"""
Tests for Milestone 7 — Smart Albums + Safe Export.

Tests cover:
  1. Database album query methods (group photos, unknown faces, text & docs)
  2. Document classifier heuristics
  3. Export engine (folder structure, group duplication, source safety, collisions)
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.doc_classifier import (
    _score_filename,
    _score_resolution,
    _score_aspect_ratio,
    classify_image,
)
from autosorter.core.export import (
    _safe_folder_name,
    _unique_dest_path,
    export_library,
)
from autosorter.core.face_embed import FaceRecognizer


# ══════════════════════════════════════════════════════════════════════
# FIXTURES — shared setup for test classes
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database with some test data."""
    db = Database(tmp_path / "library.db")

    # --- Insert 4 test images ---
    # Image 1: group photo (will have 2 person faces)
    db.upsert_image(path=str(tmp_path / "group_photo.jpg"), size=1000, mtime=1.0)
    img1 = db.get_image_by_path(str(tmp_path / "group_photo.jpg"))
    db.update_thumb(img1["id"], str(tmp_path / "thumb1.jpg"), 4032, 3024)
    db.update_status(img1["id"], "processed")

    # Image 2: solo photo (will have 1 face)
    db.upsert_image(path=str(tmp_path / "solo_photo.jpg"), size=2000, mtime=2.0)
    img2 = db.get_image_by_path(str(tmp_path / "solo_photo.jpg"))
    db.update_thumb(img2["id"], str(tmp_path / "thumb2.jpg"), 4032, 3024)
    db.update_status(img2["id"], "processed")

    # Image 3: screenshot (high doc_score)
    db.upsert_image(path=str(tmp_path / "Screenshot_2024.png"), size=500, mtime=3.0)
    img3 = db.get_image_by_path(str(tmp_path / "Screenshot_2024.png"))
    db.update_thumb(img3["id"], str(tmp_path / "thumb3.png"), 1920, 1080)
    db.update_status(img3["id"], "processed")
    db.update_doc_score(img3["id"], 0.85)

    # Image 4: another photo (no faces, no doc_score)
    db.upsert_image(path=str(tmp_path / "landscape.jpg"), size=3000, mtime=4.0)
    img4 = db.get_image_by_path(str(tmp_path / "landscape.jpg"))
    db.update_thumb(img4["id"], str(tmp_path / "thumb4.jpg"), 6000, 4000)
    db.update_status(img4["id"], "processed")

    # --- Create 2 persons ---
    emb1 = np.random.randn(128).astype(np.float32)
    emb1 /= np.linalg.norm(emb1)
    emb2 = np.random.randn(128).astype(np.float32)
    emb2 /= np.linalg.norm(emb2)
    centroid1 = FaceRecognizer.embedding_to_bytes(emb1)
    centroid2 = FaceRecognizer.embedding_to_bytes(emb2)

    p1_id = db.insert_person("Arnav", centroid1)
    p2_id = db.insert_person("Mom", centroid2)

    # --- Insert faces ---
    emb1_bytes = FaceRecognizer.embedding_to_bytes(emb1)
    emb2_bytes = FaceRecognizer.embedding_to_bytes(emb2)

    # Image 1 (group): 2 faces from different persons
    f1 = db.insert_face(img1["id"], 10, 10, 50, 50, 0.95, landmarks="[]", quality=0.8)
    db.update_embedding(f1, emb1_bytes)
    db.assign_face_to_person(f1, p1_id, source="auto", confidence=0.9)

    f2 = db.insert_face(img1["id"], 100, 10, 50, 50, 0.92, landmarks="[]", quality=0.75)
    db.update_embedding(f2, emb2_bytes)
    db.assign_face_to_person(f2, p2_id, source="auto", confidence=0.85)

    # Image 2 (solo): 1 face
    f3 = db.insert_face(img2["id"], 10, 10, 50, 50, 0.90, landmarks="[]", quality=0.7)
    db.update_embedding(f3, emb1_bytes)
    db.assign_face_to_person(f3, p1_id, source="auto", confidence=0.88)

    # Image 2 also has an unknown face (no person_id)
    f4 = db.insert_face(img2["id"], 200, 10, 50, 50, 0.80, landmarks="[]", quality=0.5)
    db.update_embedding(f4, emb2_bytes)
    # f4 is NOT assigned to any person → unknown

    yield db, tmp_path, p1_id, p2_id
    db.close()


# ══════════════════════════════════════════════════════════════════════
# TEST: Database Album Queries
# ══════════════════════════════════════════════════════════════════════

class TestGroupPhotos:
    """Test the get_group_photos() query."""

    def test_returns_multi_face_images(self, tmp_db):
        """Images with ≥2 distinct person faces should be returned."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_group_photos()

        # Only image 1 has 2 different persons
        assert len(results) == 1
        assert results[0]["path"] == str(tmp_path / "group_photo.jpg")

    def test_excludes_single_face_images(self, tmp_db):
        """Images with only 1 person should NOT appear in group photos."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_group_photos()
        paths = [r["path"] for r in results]

        # solo_photo.jpg has 1 assigned person + 1 unknown → only 1 distinct person_id
        assert str(tmp_path / "solo_photo.jpg") not in paths


class TestUnknownFaces:
    """Test the get_unknown_faces() query."""

    def test_returns_unassigned_faces(self, tmp_db):
        """Faces with no person_id should appear in the unknown faces album."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_unknown_faces()

        # f4 in image 2 is unassigned
        assert len(results) == 1
        assert results[0]["person_id"] is None

    def test_excludes_assigned_faces(self, tmp_db):
        """Faces assigned to a person should NOT appear in unknown faces."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_unknown_faces()

        # All results should have person_id = None
        for r in results:
            assert r["person_id"] is None


class TestTextAndDocs:
    """Test the get_text_and_docs() query."""

    def test_returns_high_doc_score_images(self, tmp_db):
        """Images with doc_score >= 0.5 should appear in text & docs album."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_text_and_docs()

        assert len(results) == 1
        assert results[0]["path"] == str(tmp_path / "Screenshot_2024.png")

    def test_excludes_low_doc_score_images(self, tmp_db):
        """Images with doc_score < 0.5 should NOT appear."""
        db, tmp_path, p1_id, p2_id = tmp_db
        results = db.get_text_and_docs()
        paths = [r["path"] for r in results]

        assert str(tmp_path / "group_photo.jpg") not in paths
        assert str(tmp_path / "solo_photo.jpg") not in paths


class TestImagesForPerson:
    """Test the get_images_for_person() query."""

    def test_returns_person_images(self, tmp_db):
        """Should return all images containing a specific person's faces."""
        db, tmp_path, p1_id, p2_id = tmp_db

        # Arnav (p1) appears in image 1 (group) and image 2 (solo)
        results = db.get_images_for_person(p1_id)
        assert len(results) == 2

        # Mom (p2) appears only in image 1 (group)
        results = db.get_images_for_person(p2_id)
        assert len(results) == 1


# ══════════════════════════════════════════════════════════════════════
# TEST: Document Classifier Heuristics
# ══════════════════════════════════════════════════════════════════════

class TestDocClassifierFilename:
    """Test filename pattern detection."""

    def test_screenshot_filename_scores_high(self):
        assert _score_filename("Screenshot_20240615_143000.png") == 1.0

    def test_screen_shot_filename_scores_high(self):
        assert _score_filename("Screen Shot 2024-06-15 at 2.30.00 PM.png") == 1.0

    def test_normal_photo_filename_scores_zero(self):
        assert _score_filename("IMG_2611.JPG") == 0.0

    def test_dsc_photo_filename_scores_zero(self):
        assert _score_filename("DSC_0001.NEF") == 0.0


class TestDocClassifierResolution:
    """Test screen resolution detection."""

    def test_hd_resolution_scores_high(self):
        assert _score_resolution(1920, 1080) == 1.0

    def test_iphone_resolution_scores_high(self):
        assert _score_resolution(1170, 2532) == 1.0

    def test_camera_resolution_scores_zero(self):
        assert _score_resolution(4032, 3024) == 0.0

    def test_unusual_resolution_scores_zero(self):
        assert _score_resolution(3456, 2304) == 0.0


class TestDocClassifierAspectRatio:
    """Test aspect ratio scoring."""

    def test_16_9_ratio_scores_positive(self):
        score = _score_aspect_ratio(1920, 1080)
        assert score > 0.0

    def test_3_2_camera_ratio_scores_zero(self):
        # 3:2 = 1.5 ratio — common camera ratio, not a screen ratio
        score = _score_aspect_ratio(6000, 4000)
        assert score == 0.0


class TestDocClassifierIntegration:
    """Test the combined classify_image function."""

    def test_screenshot_scores_high(self):
        """A screenshot with typical name + resolution should score high."""
        score = classify_image(
            path="C:/Users/test/Screenshot_2024.png",
            thumbnail_path=None,  # no pixel analysis
            width=1920,
            height=1080,
            face_count=0,
        )
        # Filename (0.35) + resolution (0.20) + face penalty (0.10) = 0.65 minimum
        assert score >= 0.5

    def test_camera_photo_scores_low(self):
        """A camera photo with faces should score very low."""
        score = classify_image(
            path="D:/Photos/IMG_2611.JPG",
            thumbnail_path=None,
            width=4032,
            height=3024,
            face_count=3,
        )
        # No filename match, no resolution match, has faces → very low
        assert score < 0.3


# ══════════════════════════════════════════════════════════════════════
# TEST: Export Engine
# ══════════════════════════════════════════════════════════════════════

class TestSafeFolderName:
    """Test filename sanitization for export folder names."""

    def test_normal_name(self):
        assert _safe_folder_name("Arnav") == "Arnav"

    def test_name_with_spaces(self):
        assert _safe_folder_name("Person 1") == "Person 1"

    def test_name_with_forbidden_chars(self):
        result = _safe_folder_name('Test: "quotes" <brackets>')
        assert ":" not in result
        assert '"' not in result
        assert "<" not in result
        assert ">" not in result

    def test_empty_name(self):
        assert _safe_folder_name("") == "Unknown"


class TestUniqueDestPath:
    """Test filename collision resolution."""

    def test_no_collision(self, tmp_path):
        result = _unique_dest_path(tmp_path, "photo.jpg")
        assert result == tmp_path / "photo.jpg"

    def test_collision_appends_suffix(self, tmp_path):
        # Create an existing file
        (tmp_path / "photo.jpg").write_text("existing")

        result = _unique_dest_path(tmp_path, "photo.jpg")
        assert result == tmp_path / "photo_2.jpg"

    def test_multiple_collisions(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("existing")
        (tmp_path / "photo_2.jpg").write_text("existing")

        result = _unique_dest_path(tmp_path, "photo.jpg")
        assert result == tmp_path / "photo_3.jpg"


class TestExportEngine:
    """Test the export_library function."""

    def test_creates_person_folders(self, tmp_db, tmp_path):
        """Export should create a folder for each person."""
        db, db_path, p1_id, p2_id = tmp_db
        export_dir = tmp_path / "export_test"

        # Create fake source files so copy works
        (db_path / "group_photo.jpg").write_bytes(b"fake_jpg_1")
        (db_path / "solo_photo.jpg").write_bytes(b"fake_jpg_2")
        (db_path / "Screenshot_2024.png").write_bytes(b"fake_png")
        (db_path / "landscape.jpg").write_bytes(b"fake_jpg_3")

        result = export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )

        assert result["persons_exported"] == 2
        assert (export_dir / "Arnav").is_dir()
        assert (export_dir / "Mom").is_dir()

    def test_copies_group_photo_to_each_person(self, tmp_db, tmp_path):
        """Group photos should appear in each participant's folder."""
        db, db_path, p1_id, p2_id = tmp_db
        export_dir = tmp_path / "export_test2"

        (db_path / "group_photo.jpg").write_bytes(b"fake_group_jpg")
        (db_path / "solo_photo.jpg").write_bytes(b"fake_solo_jpg")

        export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )

        # group_photo.jpg has both Arnav and Mom
        assert (export_dir / "Arnav" / "group_photo.jpg").is_file()
        assert (export_dir / "Mom" / "group_photo.jpg").is_file()

    def test_never_modifies_source(self, tmp_db, tmp_path):
        """Source files should remain untouched after export."""
        db, db_path, p1_id, p2_id = tmp_db
        export_dir = tmp_path / "export_test3"

        src_content = b"original_content_12345"
        src_file = db_path / "group_photo.jpg"
        src_file.write_bytes(src_content)
        (db_path / "solo_photo.jpg").write_bytes(b"solo_content")

        export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )

        # Source file should be completely unchanged
        assert src_file.read_bytes() == src_content

    def test_export_resume_skips_existing(self, tmp_db, tmp_path):
        """Re-running export should skip files that already exist."""
        db, db_path, p1_id, p2_id = tmp_db
        export_dir = tmp_path / "export_test4"

        (db_path / "group_photo.jpg").write_bytes(b"fake_jpg")
        (db_path / "solo_photo.jpg").write_bytes(b"fake_jpg_2")

        # First export
        result1 = export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )
        copied1 = result1["files_copied"]

        # Second export — should skip everything
        result2 = export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )

        assert result2["files_skipped"] > 0
        assert result2["files_copied"] == 0

    def test_export_handles_filename_collision(self, tmp_db, tmp_path):
        """Two different source files with the same name should both be exported."""
        db, db_path, p1_id, p2_id = tmp_db
        export_dir = tmp_path / "export_test5"

        (db_path / "group_photo.jpg").write_bytes(b"fake_group")
        (db_path / "solo_photo.jpg").write_bytes(b"fake_solo")

        # Pre-create a file in Arnav's folder to cause collision
        arnav_folder = export_dir / "Arnav"
        arnav_folder.mkdir(parents=True)
        (arnav_folder / "group_photo.jpg").write_bytes(b"pre_existing_different_size")

        export_library(
            db=db,
            export_root=export_dir,
            include_group=False,
            include_unknown=False,
            include_docs=False,
        )

        # Both the original and the _2 version should exist
        assert (arnav_folder / "group_photo.jpg").is_file()
        assert (arnav_folder / "group_photo_2.jpg").is_file()
