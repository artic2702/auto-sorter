"""
M2 tests: face detection, face cropping, database face operations.

=== WHAT ARE WE TESTING? ===
1. Database: faces table exists, insert/query works, skip-already-detected
2. FaceDetector: finds faces in test images, returns empty for blank images
3. Face cropping: alignment produces 112×112 crops
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_detect import DetectedFace, FaceDetector
from autosorter.core.face_crop import align_face, crop_face_thumb, ALIGNED_FACE_SIZE


# ──────────────────────────────────────────────────────────
# Helper: create test images
# ──────────────────────────────────────────────────────────

def _make_test_image(path: Path, width: int = 400, height: int = 300) -> None:
    """Create a random JPEG test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    img = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    encoded.tofile(str(path))


def _create_face_like_image() -> np.ndarray:
    """
    Create a synthetic image with a drawn face for testing.

    We draw circles for eyes and an arc for a mouth on a skin-toned
    background. YuNet is trained on real faces, so this may or may
    not be detected — we use it primarily for non-crash testing.
    """
    img = np.full((400, 400, 3), (180, 200, 220), dtype=np.uint8)  # skin-ish bg

    # Draw a face-like oval
    cv2.ellipse(img, (200, 200), (120, 160), 0, 0, 360, (140, 170, 200), -1)

    # Eyes
    cv2.circle(img, (160, 170), 15, (50, 50, 50), -1)  # left eye
    cv2.circle(img, (240, 170), 15, (50, 50, 50), -1)  # right eye

    # Pupils
    cv2.circle(img, (160, 170), 6, (0, 0, 0), -1)
    cv2.circle(img, (240, 170), 6, (0, 0, 0), -1)

    # Nose
    cv2.circle(img, (200, 210), 8, (120, 150, 180), -1)

    # Mouth
    cv2.ellipse(img, (200, 260), (40, 15), 0, 0, 180, (80, 80, 120), 2)

    return img


# ══════════════════════════════════════════════════════════
# DATABASE FACE TESTS
# ══════════════════════════════════════════════════════════

class TestDatabaseFaces:
    """Tests for the faces table and face-related DB methods."""

    def test_faces_table_exists(self, tmp_path: Path):
        """Opening a database creates the faces table."""
        db = Database(tmp_path / "test.db")
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "faces" in tables
        db.close()

    def test_faces_index_exists(self, tmp_path: Path):
        """The faces(image_id) index is created."""
        db = Database(tmp_path / "test.db")
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indices = {row[0] for row in cursor.fetchall()}
        assert "idx_faces_image_id" in indices
        db.close()

    def test_insert_and_get_face(self, tmp_path: Path):
        """insert_face + get_faces_for_image round trip."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        face_id = db.insert_face(
            image_id=img_id,
            bbox_x=100, bbox_y=50, bbox_w=80, bbox_h=100,
            det_score=0.95,
            landmarks='[[160,70],[240,70],[200,100],[170,130],[230,130]]',
            quality=0.8,
        )

        assert face_id > 0
        faces = db.get_faces_for_image(img_id)
        assert len(faces) == 1
        assert faces[0]["bbox_x"] == 100
        assert faces[0]["det_score"] == 0.95
        db.close()

    def test_multiple_faces_per_image(self, tmp_path: Path):
        """An image can have multiple faces (group photo)."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("group.jpg", 200, 2.0, status="processed")

        db.insert_face(img_id, 50, 50, 80, 100, 0.9)
        db.insert_face(img_id, 200, 50, 80, 100, 0.85)
        db.insert_face(img_id, 350, 50, 80, 100, 0.92)

        faces = db.get_faces_for_image(img_id)
        assert len(faces) == 3
        assert db.count_faces() == 3
        db.close()

    def test_has_faces_detected(self, tmp_path: Path):
        """has_faces_detected returns True after faces are inserted."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        assert db.has_faces_detected(img_id) is False
        db.insert_face(img_id, 100, 50, 80, 100, 0.95)
        assert db.has_faces_detected(img_id) is True
        db.close()

    def test_mark_no_faces(self, tmp_path: Path):
        """mark_no_faces inserts a sentinel so the image is skipped on re-runs."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("empty.jpg", 100, 1.0, status="processed")

        db.mark_no_faces(img_id)
        assert db.has_faces_detected(img_id) is True

        # The sentinel should have det_score=-1
        faces = db.get_faces_for_image(img_id)
        assert len(faces) == 1
        assert faces[0]["det_score"] == -1.0
        db.close()

    def test_iter_images_needing_detection(self, tmp_path: Path):
        """Only processed images without face data need detection."""
        db = Database(tmp_path / "test.db")

        # Image 1: processed, no faces → needs detection
        id1 = db.upsert_image("a.jpg", 100, 1.0, status="processed")

        # Image 2: processed, has faces → already done
        id2 = db.upsert_image("b.jpg", 200, 2.0, status="processed")
        db.insert_face(id2, 10, 10, 50, 50, 0.9)

        # Image 3: pending (not yet thumbnailed) → not ready
        id3 = db.upsert_image("c.jpg", 300, 3.0, status="pending")

        needing = list(db.iter_images_needing_detection())
        assert len(needing) == 1
        assert needing[0]["path"] == "a.jpg"
        db.close()

    def test_count_images_needing_detection(self, tmp_path: Path):
        """count_images_needing_detection matches iter results."""
        db = Database(tmp_path / "test.db")
        db.upsert_image("a.jpg", 100, 1.0, status="processed")
        db.upsert_image("b.jpg", 200, 2.0, status="processed")
        id3 = db.upsert_image("c.jpg", 300, 3.0, status="processed")
        db.mark_no_faces(id3)

        assert db.count_images_needing_detection() == 2
        db.close()

    def test_foreign_key_cascade(self, tmp_path: Path):
        """Deleting an image cascades to delete its faces."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        db.insert_face(img_id, 10, 10, 50, 50, 0.9)
        assert db.count_faces() == 1

        # Delete the image
        db._conn.execute("DELETE FROM images WHERE id = ?", (img_id,))
        db._conn.commit()

        # Face should be gone too (cascade delete)
        assert db.count_faces() == 0
        db.close()


# ══════════════════════════════════════════════════════════
# FACE DETECTOR TESTS
# ══════════════════════════════════════════════════════════

class TestFaceDetector:
    """Tests for the YuNet face detector."""

    def test_detector_loads(self):
        """FaceDetector initializes without errors."""
        detector = FaceDetector()
        assert detector is not None

    def test_no_faces_in_blank_image(self):
        """A blank image should produce zero detections."""
        detector = FaceDetector()
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        faces = detector.detect(blank)
        assert len(faces) == 0

    def test_no_faces_in_none(self):
        """Passing None should return empty list (not crash)."""
        detector = FaceDetector()
        assert detector.detect(None) == []

    def test_no_faces_in_empty(self):
        """Empty array should return empty list."""
        detector = FaceDetector()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector.detect(empty) == []

    def test_detected_face_dataclass(self):
        """DetectedFace dataclass stores and serializes correctly."""
        face = DetectedFace(
            bbox=(100, 50, 80, 100),
            landmarks=[(120.0, 70.0), (180.0, 70.0), (150.0, 95.0),
                       (130.0, 120.0), (170.0, 120.0)],
            confidence=0.95,
        )
        assert face.bbox == (100, 50, 80, 100)
        assert face.confidence == 0.95
        assert "120.0" in face.landmarks_json()

    def test_quality_score_range(self):
        """Quality score should be between 0 and 1."""
        # Small face → low quality
        small = DetectedFace(bbox=(0, 0, 20, 20), landmarks=[], confidence=0.9)
        assert 0.0 <= small.quality_score() <= 1.0
        assert small.quality_score() < 0.5  # small face = low quality

        # Large face → high quality
        large = DetectedFace(bbox=(0, 0, 300, 300), landmarks=[], confidence=0.95)
        assert large.quality_score() > 0.8  # large face = high quality

    def test_detect_returns_list_of_detected_faces(self):
        """Detector always returns a list (even if empty)."""
        detector = FaceDetector()
        img = _create_face_like_image()
        result = detector.detect(img)
        assert isinstance(result, list)
        # Each item should be a DetectedFace
        for face in result:
            assert isinstance(face, DetectedFace)


# ══════════════════════════════════════════════════════════
# FACE CROP TESTS
# ══════════════════════════════════════════════════════════

class TestFaceCrop:
    """Tests for face alignment and cropping."""

    def test_align_face_output_shape(self):
        """align_face produces a 112×112 image."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        # Simulated landmarks: eyes at (150,180) and (250,180)
        landmarks = [
            (150.0, 180.0),  # left eye
            (250.0, 180.0),  # right eye
            (200.0, 220.0),  # nose
            (170.0, 260.0),  # left mouth
            (230.0, 260.0),  # right mouth
        ]
        aligned = align_face(img, landmarks)
        assert aligned is not None
        assert aligned.shape == (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE, 3)

    def test_align_face_with_tilted_head(self):
        """align_face handles a tilted face (eyes at different heights)."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        # Right eye is higher than left (head tilted)
        landmarks = [
            (150.0, 200.0),  # left eye
            (250.0, 170.0),  # right eye (higher)
            (200.0, 220.0),
            (170.0, 260.0),
            (230.0, 260.0),
        ]
        aligned = align_face(img, landmarks)
        assert aligned is not None
        assert aligned.shape == (ALIGNED_FACE_SIZE, ALIGNED_FACE_SIZE, 3)

    def test_align_face_none_landmarks(self):
        """align_face returns None for missing landmarks."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        assert align_face(img, None) is None
        assert align_face(img, []) is None

    def test_crop_face_thumb(self):
        """crop_face_thumb extracts a region from the image."""
        img = np.ones((400, 400, 3), dtype=np.uint8) * 128
        crop = crop_face_thumb(img, (100, 50, 80, 100), padding=0.3)
        assert crop is not None
        assert crop.shape[0] > 0  # has height
        assert crop.shape[1] > 0  # has width

    def test_crop_face_thumb_clamped(self):
        """crop_face_thumb clamps to image boundaries."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Face near the edge — padding would go out of bounds
        crop = crop_face_thumb(img, (80, 80, 30, 30), padding=0.5)
        assert crop is not None
        # Result should be within image bounds (no crash)

    def test_crop_face_thumb_none_image(self):
        """crop_face_thumb returns None for None input."""
        assert crop_face_thumb(None, (0, 0, 50, 50)) is None
