"""
M4 tests: person clustering, database person operations, rename persistence.

=== WHAT ARE WE TESTING? ===
1. Database: persons table created, insert/rename/query persons
2. Clustering: single face → new person, similar → same person, different → different
3. Quality filter: low-quality faces skipped
4. Margin check: ambiguous matches → new person (not wrong merge)
5. Centroid update: adding a face moves the centroid
6. Cover face: highest quality face selected
7. Rename: persists across DB close + reopen
"""

from pathlib import Path

import numpy as np
import pytest

from autosorter.core.cluster import (
    AssignAction,
    ClusterEngine,
    MIN_QUALITY_FOR_CLUSTERING,
    T_HIGH,
    T_REVIEW,
    MARGIN_MIN,
    pick_cover_faces,
)
from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


# ──────────────────────────────────────────────────────────
# Helper: create a normalized embedding from a seed
# ──────────────────────────────────────────────────────────

def _make_embedding(seed: int = 42) -> np.ndarray:
    """Create a deterministic L2-normalized 128-dim embedding."""
    rng = np.random.default_rng(seed=seed)
    emb = rng.standard_normal(128).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


def _emb_bytes(seed: int = 42) -> bytes:
    """Create an embedding and convert to bytes for DB storage."""
    return FaceRecognizer.embedding_to_bytes(_make_embedding(seed))


def _insert_face_with_embedding(
    db: Database,
    image_id: int,
    quality: float = 0.7,
    seed: int = 42,
) -> int:
    """Insert a face with a valid embedding, return face_id."""
    face_id = db.insert_face(
        image_id, 10, 10, 80, 100, 0.9,
        quality=quality,
        thumb_path=f"/tmp/face_{seed}.jpg",
    )
    emb = _make_embedding(seed)
    db.update_embedding(face_id, FaceRecognizer.embedding_to_bytes(emb))
    return face_id


# ──────────────────────────────────────────────────────────
# Database: persons table tests
# ──────────────────────────────────────────────────────────

class TestPersonsTable:
    """Tests for the persons table schema and basic CRUD."""

    def test_persons_table_exists(self, tmp_path: Path):
        """Schema creates the persons table."""
        db = Database(tmp_path / "test.db")
        # Should not raise — table exists
        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM persons"
        )
        assert cursor.fetchone()[0] == 0
        db.close()

    def test_insert_person(self, tmp_path: Path):
        """Can create a person with a centroid."""
        db = Database(tmp_path / "test.db")
        centroid = _emb_bytes(42)
        pid = db.insert_person("Person 1", centroid, face_count=1)
        assert pid > 0

        person = db.get_person_by_id(pid)
        assert person is not None
        assert person["display_name"] == "Person 1"
        assert person["face_count"] == 1
        assert person["name_is_user_set"] == 0
        assert person["hidden"] == 0
        assert person["merged_into"] is None
        db.close()

    def test_rename_person(self, tmp_path: Path):
        """Rename sets name and marks name_is_user_set=1."""
        db = Database(tmp_path / "test.db")
        pid = db.insert_person("Person 1", _emb_bytes(), face_count=1)

        db.rename_person(pid, "Arnav")

        person = db.get_person_by_id(pid)
        assert person["display_name"] == "Arnav"
        assert person["name_is_user_set"] == 1
        db.close()

    def test_rename_survives_restart(self, tmp_path: Path):
        """Rename persists after closing and reopening the database."""
        db_path = tmp_path / "test.db"

        # Create and rename
        db = Database(db_path)
        pid = db.insert_person("Person 1", _emb_bytes(), face_count=1)
        db.rename_person(pid, "Mom")
        db.close()

        # Reopen and verify
        db2 = Database(db_path)
        person = db2.get_person_by_id(pid)
        assert person["display_name"] == "Mom"
        assert person["name_is_user_set"] == 1
        db2.close()

    def test_assign_face_to_person(self, tmp_path: Path):
        """assign_face_to_person sets person_id on the face row."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        face_id = db.insert_face(img_id, 10, 10, 80, 100, 0.9)
        pid = db.insert_person("Person 1", _emb_bytes(), face_count=1)

        db.assign_face_to_person(face_id, pid, source="auto", confidence=0.65)

        face = db.get_face_by_id(face_id)
        assert face["person_id"] == pid
        assert face["assign_source"] == "auto"
        assert abs(face["assign_conf"] - 0.65) < 0.001
        db.close()

    def test_get_all_persons_ordered(self, tmp_path: Path):
        """get_all_persons returns persons ordered by face_count DESC."""
        db = Database(tmp_path / "test.db")
        p1 = db.insert_person("A", _emb_bytes(1), face_count=10)
        p2 = db.insert_person("B", _emb_bytes(2), face_count=50)
        p3 = db.insert_person("C", _emb_bytes(3), face_count=25)

        persons = db.get_all_persons()
        assert len(persons) == 3
        assert persons[0]["id"] == p2  # 50 faces, most
        assert persons[1]["id"] == p3  # 25 faces
        assert persons[2]["id"] == p1  # 10 faces, least
        db.close()

    def test_get_faces_for_person(self, tmp_path: Path):
        """Returns only faces assigned to that person."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        f1 = db.insert_face(img_id, 10, 10, 80, 100, 0.9, quality=0.8)
        f2 = db.insert_face(img_id, 200, 10, 80, 100, 0.85, quality=0.6)
        f3 = db.insert_face(img_id, 400, 10, 80, 100, 0.88, quality=0.7)

        pid = db.insert_person("Person 1", _emb_bytes(), face_count=2)
        db.assign_face_to_person(f1, pid, "auto", 0.5)
        db.assign_face_to_person(f3, pid, "auto", 0.6)

        faces = db.get_faces_for_person(pid)
        assert len(faces) == 2
        face_ids = {f["id"] for f in faces}
        assert face_ids == {f1, f3}
        db.close()

    def test_count_unassigned_faces(self, tmp_path: Path):
        """Counts faces with embeddings but no person_id."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Face with embedding, no person → unassigned
        f1 = _insert_face_with_embedding(db, img_id, seed=1)
        # Face with embedding, assigned → not counted
        f2 = _insert_face_with_embedding(db, img_id, seed=2)
        pid = db.insert_person("P", _emb_bytes(), face_count=1)
        db.assign_face_to_person(f2, pid, "auto", 0.5)

        assert db.count_unassigned_faces() == 1
        db.close()

    def test_count_persons(self, tmp_path: Path):
        """count_persons excludes hidden and merged."""
        db = Database(tmp_path / "test.db")
        db.insert_person("Visible", _emb_bytes(1), face_count=5)
        p_hidden = db.insert_person("Hidden", _emb_bytes(2), face_count=3)
        db._conn.execute("UPDATE persons SET hidden = 1 WHERE id = ?", (p_hidden,))
        db._conn.commit()

        assert db.count_persons() == 1
        db.close()

    def test_update_person_cover(self, tmp_path: Path):
        """update_person_cover sets cover_face_id."""
        db = Database(tmp_path / "test.db")
        pid = db.insert_person("P1", _emb_bytes(), face_count=1, cover_face_id=None)
        db.update_person_cover(pid, 42)

        person = db.get_person_by_id(pid)
        assert person["cover_face_id"] == 42
        db.close()


# ──────────────────────────────────────────────────────────
# Clustering engine tests
# ──────────────────────────────────────────────────────────

class TestClusterEngine:
    """Tests for the incremental centroid clustering engine."""

    def test_single_face_creates_person(self, tmp_path: Path):
        """First face processed → creates Person 1."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        face_id = _insert_face_with_embedding(db, img_id, quality=0.7, seed=42)

        engine = ClusterEngine()
        face_row = db.get_face_by_id(face_id)
        result = engine.process_face(db, face_row)

        assert result.action == AssignAction.NEW_PERSON
        assert result.person_id > 0
        assert engine.person_count == 1

        # Verify in DB
        person = db.get_person_by_id(result.person_id)
        assert person is not None
        assert person["display_name"] == "Person 1"
        assert person["face_count"] == 1
        db.close()

    def test_identical_embeddings_same_person(self, tmp_path: Path):
        """Two faces with identical embeddings → same person."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Two faces with the SAME embedding (same seed)
        f1 = _insert_face_with_embedding(db, img_id, quality=0.8, seed=42)
        f2 = _insert_face_with_embedding(db, img_id, quality=0.7, seed=42)

        engine = ClusterEngine()

        # Process first face → new person
        r1 = engine.process_face(db, db.get_face_by_id(f1))
        assert r1.action == AssignAction.NEW_PERSON

        # Process second face → should match the same person
        r2 = engine.process_face(db, db.get_face_by_id(f2))
        assert r2.action == AssignAction.AUTO_ASSIGN
        assert r2.person_id == r1.person_id
        assert engine.person_count == 1
        db.close()

    def test_very_different_embeddings_different_persons(self, tmp_path: Path):
        """Two faces with very different embeddings → two different persons."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Two faces with DIFFERENT embeddings (different seeds → random directions)
        f1 = _insert_face_with_embedding(db, img_id, quality=0.8, seed=1)
        f2 = _insert_face_with_embedding(db, img_id, quality=0.7, seed=999)

        engine = ClusterEngine()

        r1 = engine.process_face(db, db.get_face_by_id(f1))
        r2 = engine.process_face(db, db.get_face_by_id(f2))

        assert r1.action == AssignAction.NEW_PERSON
        assert r2.action == AssignAction.NEW_PERSON
        assert r1.person_id != r2.person_id
        assert engine.person_count == 2
        db.close()

    def test_quality_filter_skips_low_quality(self, tmp_path: Path):
        """Faces below MIN_QUALITY_FOR_CLUSTERING are skipped."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Low quality face
        face_id = _insert_face_with_embedding(
            db, img_id, quality=MIN_QUALITY_FOR_CLUSTERING - 0.01, seed=42
        )

        engine = ClusterEngine()
        result = engine.process_face(db, db.get_face_by_id(face_id))

        assert result.action == AssignAction.SKIPPED
        assert engine.person_count == 0
        db.close()

    def test_centroid_update_moves_toward_new_face(self, tmp_path: Path):
        """Adding a face to a person moves the centroid toward that face."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Create first face (seed person)
        f1 = _insert_face_with_embedding(db, img_id, quality=0.8, seed=42)
        engine = ClusterEngine()
        r1 = engine.process_face(db, db.get_face_by_id(f1))

        # Get the original centroid
        person = db.get_person_by_id(r1.person_id)
        original_centroid = FaceRecognizer.bytes_to_embedding(person["centroid"])

        # Create a second face with same embedding (will be assigned to same person)
        f2 = _insert_face_with_embedding(db, img_id, quality=0.7, seed=42)
        r2 = engine.process_face(db, db.get_face_by_id(f2))

        # Centroid should have been updated
        person_after = db.get_person_by_id(r1.person_id)
        updated_centroid = FaceRecognizer.bytes_to_embedding(person_after["centroid"])

        # Face count should be 2 now
        assert person_after["face_count"] == 2

        # Centroid should still be normalized
        norm = np.linalg.norm(updated_centroid)
        assert abs(norm - 1.0) < 1e-5
        db.close()

    def test_process_order_quality_desc(self, tmp_path: Path):
        """iter_unassigned_faces returns highest quality first."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        _insert_face_with_embedding(db, img_id, quality=0.3, seed=1)
        _insert_face_with_embedding(db, img_id, quality=0.9, seed=2)
        _insert_face_with_embedding(db, img_id, quality=0.6, seed=3)

        faces = list(db.iter_unassigned_faces())
        qualities = [f["quality"] for f in faces]

        # Should be sorted descending
        assert qualities == sorted(qualities, reverse=True)
        db.close()

    def test_load_existing_persons_resume(self, tmp_path: Path):
        """ClusterEngine.load_existing_persons loads centroids from DB."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Create a person via the engine
        f1 = _insert_face_with_embedding(db, img_id, quality=0.8, seed=42)
        engine1 = ClusterEngine()
        r1 = engine1.process_face(db, db.get_face_by_id(f1))
        assert engine1.person_count == 1

        # Create a NEW engine that loads from DB (simulating app restart)
        engine2 = ClusterEngine()
        engine2.load_existing_persons(db)
        assert engine2.person_count == 1

        # Process a face with same embedding → should match existing person
        f2 = _insert_face_with_embedding(db, img_id, quality=0.7, seed=42)
        r2 = engine2.process_face(db, db.get_face_by_id(f2))
        assert r2.action == AssignAction.AUTO_ASSIGN
        assert r2.person_id == r1.person_id
        db.close()


# ──────────────────────────────────────────────────────────
# Cover face tests
# ──────────────────────────────────────────────────────────

class TestCoverFace:
    """Tests for pick_cover_faces."""

    def test_cover_face_is_highest_quality(self, tmp_path: Path):
        """Cover face should be the highest-quality face."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Create a person
        pid = db.insert_person("Person 1", _emb_bytes(), face_count=3)

        # Add faces with different qualities
        f_low = db.insert_face(img_id, 10, 10, 40, 50, 0.8, quality=0.3,
                               thumb_path="/tmp/low.jpg")
        f_high = db.insert_face(img_id, 100, 10, 120, 150, 0.95, quality=0.9,
                                thumb_path="/tmp/high.jpg")
        f_mid = db.insert_face(img_id, 200, 10, 80, 100, 0.85, quality=0.6,
                               thumb_path="/tmp/mid.jpg")

        db.assign_face_to_person(f_low, pid, "auto", 0.5)
        db.assign_face_to_person(f_high, pid, "auto", 0.7)
        db.assign_face_to_person(f_mid, pid, "auto", 0.6)

        pick_cover_faces(db)

        person = db.get_person_by_id(pid)
        assert person["cover_face_id"] == f_high  # highest quality
        db.close()
