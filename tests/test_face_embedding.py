"""
M3 tests: face embeddings, cosine similarity, database BLOB storage.

=== WHAT ARE WE TESTING? ===
1. SFace model: loads, produces 128-dim embeddings
2. Embeddings: deterministic, L2-normalized (unit length)
3. Cosine similarity: same face = 1.0, different = < 1.0
4. BLOB round-trip: numpy → bytes → SQLite → bytes → numpy
5. Database: embedding storage, queries for faces needing embedding
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


# ──────────────────────────────────────────────────────────
# SFace recognizer tests
# ──────────────────────────────────────────────────────────

class TestFaceRecognizer:
    """Tests for the SFace face recognition model."""

    def test_model_loads(self):
        """FaceRecognizer initializes without errors."""
        recognizer = FaceRecognizer()
        assert recognizer is not None

    def test_compute_returns_128_dim(self):
        """SFace outputs exactly 128 floats."""
        recognizer = FaceRecognizer()
        # Create a dummy 112×112 face image (random pixels)
        rng = np.random.default_rng(seed=42)
        face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = recognizer.compute(face)
        assert embedding is not None
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32

    def test_embedding_is_l2_normalized(self):
        """Embeddings should have unit length (L2 norm = 1.0)."""
        recognizer = FaceRecognizer()
        rng = np.random.default_rng(seed=42)
        face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        embedding = recognizer.compute(face)
        assert embedding is not None
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5, f"Expected norm ~1.0, got {norm}"

    def test_embedding_is_deterministic(self):
        """Same input → same embedding (no randomness in inference)."""
        recognizer = FaceRecognizer()
        rng = np.random.default_rng(seed=42)
        face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        emb1 = recognizer.compute(face)
        emb2 = recognizer.compute(face)
        assert emb1 is not None and emb2 is not None
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_self_similarity_is_one(self):
        """Cosine similarity of an embedding with itself = 1.0."""
        recognizer = FaceRecognizer()
        rng = np.random.default_rng(seed=42)
        face = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        emb = recognizer.compute(face)
        assert emb is not None
        sim = FaceRecognizer.compare(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_different_inputs_different_embeddings(self):
        """Different face images → different embeddings → similarity < 1.0."""
        recognizer = FaceRecognizer()
        rng = np.random.default_rng(seed=42)
        face_a = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        rng2 = np.random.default_rng(seed=999)
        face_b = rng2.integers(0, 255, (112, 112, 3), dtype=np.uint8)
        emb_a = recognizer.compute(face_a)
        emb_b = recognizer.compute(face_b)
        assert emb_a is not None and emb_b is not None
        sim = FaceRecognizer.compare(emb_a, emb_b)
        assert sim < 1.0, f"Different faces should have sim < 1.0, got {sim}"

    def test_compute_rejects_wrong_size(self):
        """SFace only accepts 112×112 — other sizes return None."""
        recognizer = FaceRecognizer()
        wrong = np.zeros((64, 64, 3), dtype=np.uint8)
        assert recognizer.compute(wrong) is None

    def test_compute_rejects_none(self):
        """Passing None returns None (no crash)."""
        recognizer = FaceRecognizer()
        assert recognizer.compute(None) is None


# ──────────────────────────────────────────────────────────
# BLOB serialization tests
# ──────────────────────────────────────────────────────────

class TestEmbeddingSerialization:
    """Tests for numpy ↔ bytes conversion."""

    def test_roundtrip(self):
        """embedding → bytes → embedding preserves values exactly."""
        rng = np.random.default_rng(seed=42)
        original = rng.standard_normal(128).astype(np.float32)
        blob = FaceRecognizer.embedding_to_bytes(original)
        recovered = FaceRecognizer.bytes_to_embedding(blob)
        np.testing.assert_array_equal(original, recovered)

    def test_blob_size_is_512(self):
        """128 × float32 = 128 × 4 bytes = 512 bytes exactly."""
        emb = np.zeros(128, dtype=np.float32)
        blob = FaceRecognizer.embedding_to_bytes(emb)
        assert len(blob) == 512


# ──────────────────────────────────────────────────────────
# Database embedding storage tests
# ──────────────────────────────────────────────────────────

class TestDatabaseEmbeddings:
    """Tests for embedding CRUD in the database."""

    def test_update_and_retrieve_embedding(self, tmp_path: Path):
        """update_embedding stores a BLOB that we can read back."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        face_id = db.insert_face(img_id, 10, 10, 80, 100, 0.9,
                                 thumb_path="/tmp/face_1.jpg")

        # Create a fake embedding
        emb = np.ones(128, dtype=np.float32) * 0.5
        emb_bytes = FaceRecognizer.embedding_to_bytes(emb)

        db.update_embedding(face_id, emb_bytes)

        # Read it back
        face = db.get_face_by_id(face_id)
        assert face is not None
        assert face["embedding"] is not None
        recovered = FaceRecognizer.bytes_to_embedding(face["embedding"])
        np.testing.assert_array_almost_equal(emb, recovered)
        db.close()

    def test_iter_faces_needing_embedding(self, tmp_path: Path):
        """Only faces with thumb but no embedding are yielded."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Face 1: has thumb, no embedding → needs embedding
        f1 = db.insert_face(img_id, 10, 10, 80, 100, 0.9,
                            thumb_path="/tmp/face_1.jpg")

        # Face 2: has thumb AND embedding → already done
        f2 = db.insert_face(img_id, 200, 10, 80, 100, 0.85,
                            thumb_path="/tmp/face_2.jpg")
        emb = np.zeros(128, dtype=np.float32)
        db.update_embedding(f2, FaceRecognizer.embedding_to_bytes(emb))

        # Face 3: sentinel (det_score=-1) → should be skipped
        db.mark_no_faces(img_id)

        needing = list(db.iter_faces_needing_embedding())
        assert len(needing) == 1
        assert needing[0]["id"] == f1
        db.close()

    def test_count_faces_needing_embedding(self, tmp_path: Path):
        """Count matches the iterator results."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        db.insert_face(img_id, 10, 10, 80, 100, 0.9,
                       thumb_path="/tmp/face_1.jpg")
        db.insert_face(img_id, 200, 10, 80, 100, 0.85,
                       thumb_path="/tmp/face_2.jpg")

        assert db.count_faces_needing_embedding() == 2
        db.close()

    def test_get_all_embeddings(self, tmp_path: Path):
        """get_all_embeddings returns only faces with embeddings."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")

        # Face with embedding
        f1 = db.insert_face(img_id, 10, 10, 80, 100, 0.9,
                            thumb_path="/tmp/face_1.jpg")
        emb = np.ones(128, dtype=np.float32)
        db.update_embedding(f1, FaceRecognizer.embedding_to_bytes(emb))

        # Face without embedding
        f2 = db.insert_face(img_id, 200, 10, 80, 100, 0.85,
                            thumb_path="/tmp/face_2.jpg")

        all_embs = db.get_all_embeddings()
        assert len(all_embs) == 1
        assert all_embs[0]["id"] == f1
        db.close()

    def test_get_face_by_id(self, tmp_path: Path):
        """get_face_by_id returns the correct face row."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        face_id = db.insert_face(img_id, 42, 84, 80, 100, 0.95)

        face = db.get_face_by_id(face_id)
        assert face is not None
        assert face["bbox_x"] == 42
        assert face["bbox_y"] == 84

        # Non-existent ID
        assert db.get_face_by_id(9999) is None
        db.close()

    def test_embedding_blob_survives_db_roundtrip(self, tmp_path: Path):
        """Full round trip: numpy → bytes → SQLite → bytes → numpy."""
        db = Database(tmp_path / "test.db")
        img_id = db.upsert_image("photo.jpg", 100, 1.0, status="processed")
        face_id = db.insert_face(img_id, 10, 10, 80, 100, 0.9,
                                 thumb_path="/tmp/face.jpg")

        # Create a meaningful embedding (L2-normalized)
        rng = np.random.default_rng(seed=42)
        emb = rng.standard_normal(128).astype(np.float32)
        emb = emb / np.linalg.norm(emb)  # L2 normalize

        # Store → retrieve → verify
        db.update_embedding(face_id, FaceRecognizer.embedding_to_bytes(emb))

        face = db.get_face_by_id(face_id)
        recovered = FaceRecognizer.bytes_to_embedding(face["embedding"])

        # Check values are identical (not just close — binary should be exact)
        np.testing.assert_array_equal(emb, recovered)
        db.close()
