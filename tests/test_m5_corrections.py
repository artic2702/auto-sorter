"""
M5 tests: manual corrections, merge persons, not-this-person, reassign.

=== WHAT ARE WE TESTING? ===
1. Merge persons: combines faces, deletes source person, updates target centroid & count.
2. Unassign face ("Not this person"): removes face, recalculates centroid, updates cover.
3. Reassign face ("Move to..."): moves face to existing person, updates both centroids.
4. Create person for face: creates new person seeded by face.
5. Cover face preservation: highest quality face remains cover face after merge/unassign.
6. Persistence: all corrections survive database close and reopen.
"""

from pathlib import Path

import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


def _make_embedding(seed: int = 42) -> np.ndarray:
    """Create a deterministic L2-normalized 128-dim embedding."""
    rng = np.random.default_rng(seed=seed)
    emb = rng.standard_normal(128).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _insert_face(
    db: Database,
    image_id: int,
    quality: float = 0.8,
    seed: int = 42,
    person_id: int = None,
) -> int:
    """Insert a test face with embedding."""
    emb = _make_embedding(seed)
    emb_bytes = FaceRecognizer.embedding_to_bytes(emb)
    face_id = db.insert_face(
        image_id=image_id,
        bbox_x=10,
        bbox_y=10,
        bbox_w=100,
        bbox_h=100,
        det_score=0.95,
        landmarks="[]",
        quality=quality,
    )
    db.update_embedding(face_id, emb_bytes)
    if person_id is not None:
        db.assign_face_to_person(face_id, person_id, source="auto", confidence=1.0)
    return face_id


class TestMergePersons:
    """Tests for db.merge_persons()."""

    def test_merge_two_persons(self, tmp_path: Path):
        """Merging Person A into Person B moves faces, deletes A, updates B."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Person 1 with 2 faces
        p1 = db.insert_person("Arnav 1", None, 0)
        f1 = _insert_face(db, img_id, quality=0.8, seed=1, person_id=p1)
        f2 = _insert_face(db, img_id, quality=0.9, seed=2, person_id=p1)
        db.recompute_person_centroid(p1)

        # Person 2 with 1 face
        p2 = db.insert_person("Arnav 2", None, 0)
        f3 = _insert_face(db, img_id, quality=0.7, seed=3, person_id=p2)
        db.recompute_person_centroid(p2)

        assert db.count_persons() == 2

        # Merge Person 2 into Person 1
        db.merge_persons(source_person_id=p2, target_person_id=p1)

        # Person 2 should be deleted
        assert db.count_persons() == 1
        assert db.get_person_by_id(p2) is None

        # Person 1 should have all 3 faces
        p1_faces = list(db.get_faces_for_person(p1))
        assert len(p1_faces) == 3
        p1_fids = {f["id"] for f in p1_faces}
        assert p1_fids == {f1, f2, f3}

        # Person 1 face count and cover face updated
        p1_row = db.get_person_by_id(p1)
        assert p1_row["face_count"] == 3
        assert p1_row["cover_face_id"] == f2  # quality 0.9 is highest
        assert p1_row["centroid"] is not None

        db.close()

    def test_merge_same_person_noop(self, tmp_path: Path):
        """Merging a person into itself is a safe no-op."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Arnav", None, 0)
        _insert_face(db, img_id, quality=0.8, seed=1, person_id=p1)
        db.recompute_person_centroid(p1)

        db.merge_persons(source_person_id=p1, target_person_id=p1)
        assert db.count_persons() == 1
        db.close()


class TestUnassignFace:
    """Tests for db.unassign_face() ("Not this person")."""

    def test_unassign_removes_face_and_updates_centroid(self, tmp_path: Path):
        """Unassigning a face removes it from the person and updates centroid."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Arnav", None, 0)
        f1 = _insert_face(db, img_id, quality=0.9, seed=1, person_id=p1)
        f2 = _insert_face(db, img_id, quality=0.7, seed=2, person_id=p1)
        db.recompute_person_centroid(p1)

        # Unassign f1 (was cover face)
        old_pid = db.unassign_face(f1)
        assert old_pid == p1

        # f1 is now unassigned
        f1_row = db.get_face_by_id(f1)
        assert f1_row["person_id"] is None
        assert f1_row["assign_source"] == "user"

        # Person 1 now has 1 face, and f2 is the new cover
        p1_row = db.get_person_by_id(p1)
        assert p1_row["face_count"] == 1
        assert p1_row["cover_face_id"] == f2

        db.close()

    def test_unassign_last_face_deletes_empty_person(self, tmp_path: Path):
        """Unassigning the only face of a person deletes the empty person."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Arnav", None, 0)
        f1 = _insert_face(db, img_id, quality=0.9, seed=1, person_id=p1)
        db.recompute_person_centroid(p1)

        db.unassign_face(f1)
        assert db.count_persons() == 0
        assert db.get_person_by_id(p1) is None
        db.close()


class TestReassignFace:
    """Tests for db.reassign_face() and db.create_person_for_face()."""

    def test_reassign_between_persons(self, tmp_path: Path):
        """Moving a face from Person 1 to Person 2 updates both."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Arnav", None, 0)
        f1 = _insert_face(db, img_id, quality=0.9, seed=1, person_id=p1)
        f2 = _insert_face(db, img_id, quality=0.8, seed=2, person_id=p1)
        db.recompute_person_centroid(p1)

        p2 = db.insert_person("Dad", None, 0)
        f3 = _insert_face(db, img_id, quality=0.7, seed=3, person_id=p2)
        db.recompute_person_centroid(p2)

        # Move f2 to Person 2
        db.reassign_face(f2, p2)

        # Check Person 1 (now 1 face)
        p1_row = db.get_person_by_id(p1)
        assert p1_row["face_count"] == 1

        # Check Person 2 (now 2 faces, f2 is cover since 0.8 > 0.7)
        p2_row = db.get_person_by_id(p2)
        assert p2_row["face_count"] == 2
        assert p2_row["cover_face_id"] == f2

        db.close()

    def test_create_person_for_face(self, tmp_path: Path):
        """Creating a new person for an unassigned face."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        f1 = _insert_face(db, img_id, quality=0.9, seed=1, person_id=None)
        new_pid = db.create_person_for_face(f1, "Rohan")

        assert db.count_persons() == 1
        p_row = db.get_person_by_id(new_pid)
        assert p_row["display_name"] == "Rohan"
        assert p_row["cover_face_id"] == f1
        assert p_row["face_count"] == 1

        f1_row = db.get_face_by_id(f1)
        assert f1_row["person_id"] == new_pid
        assert f1_row["assign_source"] == "user"

        db.close()


class TestCorrectionsPersistence:
    """Corrections persist after closing and reopening the database."""

    def test_corrections_survive_reopen(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        db = Database(db_path)
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Person A", None, 0)
        p2 = db.insert_person("Person B", None, 0)
        f1 = _insert_face(db, img_id, quality=0.9, seed=1, person_id=p1)
        f2 = _insert_face(db, img_id, quality=0.8, seed=2, person_id=p2)
        db.recompute_person_centroid(p1)
        db.recompute_person_centroid(p2)

        # Merge p2 into p1
        db.merge_persons(p2, p1)
        db.close()

        # Reopen DB
        db2 = Database(db_path)
        assert db2.count_persons() == 1
        p1_row = db2.get_person_by_id(p1)
        assert p1_row["face_count"] == 2
        assert db2.get_person_by_id(p2) is None
        db2.close()


class TestCleanUpPersonAlbum:
    """Tests for clean_up_person_album()."""

    def test_cleanup_shifts_misplaced_face(self, tmp_path: Path):
        """A face placed in Person 1 that is much closer to Person 2 is shifted."""
        from autosorter.core.cluster import clean_up_person_album
        db = Database(tmp_path / "test.db")
        img1 = db.upsert_image("photo1.jpg", 100, 1.0, status="processed")
        img2 = db.upsert_image("photo2.jpg", 100, 1.0, status="processed")
        img3 = db.upsert_image("photo3.jpg", 100, 1.0, status="processed")

        # Person 1 anchor (seed 1)
        p1 = db.insert_person("Person 1", None, 0)
        f1 = _insert_face(db, img1, quality=0.9, seed=1, person_id=p1)
        db.recompute_person_centroid(p1)

        # Person 2 anchor (seed 2)
        p2 = db.insert_person("Person 2", None, 0)
        f2 = _insert_face(db, img2, quality=0.9, seed=2, person_id=p2)
        db.recompute_person_centroid(p2)

        # Misplaced face: has seed 2 (matches Person 2), but was auto-assigned to Person 1 in img3
        f_misplaced = _insert_face(db, img3, quality=0.85, seed=2, person_id=p1)
        db.recompute_person_centroid(p1)

        assert len(db.get_faces_for_person(p1)) == 2
        assert len(db.get_faces_for_person(p2)) == 1

        # Run cleanup on Person 1
        summary = clean_up_person_album(db, p1)

        assert "Person 2" in summary
        assert summary["Person 2"] == 1

        # Misplaced face should now be in Person 2
        f_check = db.get_face_by_id(f_misplaced)
        assert f_check["person_id"] == p2
        assert len(db.get_faces_for_person(p1)) == 1
        assert len(db.get_faces_for_person(p2)) == 2
        db.close()

    def test_cleanup_preserves_user_locked_face(self, tmp_path: Path):
        """Faces with assign_source = 'user' are never moved by cleanup."""
        from autosorter.core.cluster import clean_up_person_album
        db = Database(tmp_path / "test.db")
        img1 = db.upsert_image("photo1.jpg", 100, 1.0, status="processed")
        img2 = db.upsert_image("photo2.jpg", 100, 1.0, status="processed")
        img3 = db.upsert_image("photo3.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Person 1", None, 0)
        f1 = _insert_face(db, img1, quality=0.9, seed=1, person_id=p1)
        db.recompute_person_centroid(p1)

        p2 = db.insert_person("Person 2", None, 0)
        f2 = _insert_face(db, img2, quality=0.9, seed=2, person_id=p2)
        db.recompute_person_centroid(p2)

        # User locked face into Person 1
        f_locked = _insert_face(db, img3, quality=0.85, seed=2, person_id=p1)
        db.assign_face_to_person(f_locked, p1, source="user")

        summary = clean_up_person_album(db, p1)
        assert summary == {}
        assert db.get_face_by_id(f_locked)["person_id"] == p1
        db.close()


class TestDeletePerson:
    """Tests for db.delete_person()."""

    def test_delete_person_removes_person_and_unassigns_faces(self, tmp_path: Path):
        """Deleting a person removes the person record and unassigns its faces."""
        db = Database(tmp_path / "test.db")
        img1 = db.upsert_image("photo1.jpg", 100, 1.0, status="processed")
        img2 = db.upsert_image("photo2.jpg", 100, 1.0, status="processed")

        p1 = db.insert_person("Person 1", None, 0)
        f1 = _insert_face(db, img1, quality=0.9, seed=1, person_id=p1)
        f2 = _insert_face(db, img2, quality=0.8, seed=1, person_id=p1)

        p2 = db.insert_person("Person 2", None, 0)
        f3 = _insert_face(db, img1, quality=0.7, seed=2, person_id=p2)

        assert db.count_persons() == 2
        assert len(db.get_faces_for_person(p1)) == 2

        # Delete Person 1
        db.delete_person(p1)

        # Person 1 is deleted
        assert db.count_persons() == 1
        assert db.get_person_by_id(p1) is None
        assert db.get_person_by_id(p2) is not None

        # Faces f1 and f2 are now unassigned
        f1_row = db.get_face_by_id(f1)
        f2_row = db.get_face_by_id(f2)
        assert f1_row["person_id"] is None
        assert f2_row["person_id"] is None

        # Person 2's face is untouched
        f3_row = db.get_face_by_id(f3)
        assert f3_row["person_id"] == p2
        db.close()


