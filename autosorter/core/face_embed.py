"""
Face embedding engine — converts face images into 128-number faceprints using SFace.

=== WHAT IS THIS FILE? ===
This is the RECOGNITION component. While face_detect.py (YuNet) finds WHERE
faces are, this module figures out WHO each face is — by computing a unique
"faceprint" (128 numbers) for each face.

=== WHAT IS A FACE EMBEDDING? ===
An embedding is a vector of 128 floating-point numbers that DESCRIBES a face.
Think of it like a fingerprint, but for faces:

  Face photo → SFace model → [0.023, -0.145, 0.089, ..., 0.034]  (128 numbers)

Key property: SAME person → SIMILAR numbers, DIFFERENT person → DIFFERENT numbers.

Example (simplified to 3 numbers for illustration):
  Your face, photo 1:   [0.82, -0.15, 0.43]
  Your face, photo 2:   [0.80, -0.17, 0.45]  ← very similar!
  Someone else's face:  [-0.31, 0.67, -0.22]  ← completely different!

=== HOW SFACE WORKS (simplified) ===
SFace is a deep neural network trained on millions of labeled face images.
During training, it learned to:
1. IGNORE irrelevant details (lighting, expression, glasses, minor aging)
2. PRESERVE identity-specific features (face shape, eye spacing, nose structure)

The 128 numbers encode these identity features in a compressed form.
We NEVER retrain the model — it's "frozen" and just used as a measurement tool.

=== WHAT IS COSINE SIMILARITY? ===
To compare two faceprints, we use "cosine similarity" — a measure of how
similar two vectors are in direction (ignoring magnitude):

  similarity = dot_product(A, B) / (|A| × |B|)

  Range: -1.0 (opposite) to 1.0 (identical direction)
  In practice for faces: 0.0 (completely different) to 1.0 (same person)

WHY COSINE AND NOT EUCLIDEAN DISTANCE?
Both work, but cosine similarity:
- Is scale-invariant (doesn't matter if one vector is "louder" than another)
- After L2 normalization, cosine similarity = simple dot product (very fast)
- Is what SFace was trained with, so it gives the best accuracy

=== WHAT IS L2 NORMALIZATION? ===
L2 normalization scales a vector so its length (magnitude) = 1.0:

  normalized = vector / sqrt(sum(vector²))

After normalization, cosine similarity simplifies to:
  similarity = dot_product(A, B)   ← that's it! One multiplication + sum

This is why we normalize embeddings before storing them — it makes
similarity computation a single numpy operation on the entire database.
"""

from typing import Optional

import cv2
import numpy as np

from autosorter.paths import SFACE_MODEL


class FaceRecognizer:
    """
    Computes face embeddings using the SFace model.

    This class wraps OpenCV's FaceRecognizerSF. Like FaceDetector, we create
    it ONCE and reuse it for every face (loading the model is expensive).

    Usage:
        recognizer = FaceRecognizer()
        embedding = recognizer.compute(aligned_face_112x112)
        similarity = recognizer.compare(embedding1, embedding2)
    """

    def __init__(self) -> None:
        """Load the SFace ONNX model."""
        self._recognizer = cv2.FaceRecognizerSF.create(
            str(SFACE_MODEL),
            "",  # config (not needed for ONNX)
        )

    def compute(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a 128-dimensional embedding for an aligned face image.

        Args:
            aligned_face: A 112×112 BGR numpy array (from align_face()).
                          MUST be exactly 112×112 — SFace was trained on this size.

        Returns:
            A 1D numpy array of 128 float32 values, L2-normalized.
            Returns None if the input is invalid.

        WHAT HAPPENS INSIDE:
        1. SFace preprocesses the image (normalize pixel values to 0-1)
        2. Forward pass through the neural network (many matrix multiplications)
        3. Output layer produces 128 numbers
        4. We L2-normalize the result

        Performance: ~10ms per face on a modern CPU (very fast!)
        """
        if aligned_face is None:
            return None

        # Validate input shape
        if aligned_face.shape[:2] != (112, 112):
            return None

        # SFace's feature() method does the forward pass
        # Returns shape (1, 128) — a batch of 1 embedding
        raw_embedding = self._recognizer.feature(aligned_face)

        # Flatten from (1, 128) to (128,)
        embedding = raw_embedding.flatten().astype(np.float32)

        # L2 normalize: scale so the vector has length 1.0
        # This makes cosine_similarity(A, B) = dot(A, B)
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return None  # Degenerate embedding (shouldn't happen)
        embedding = embedding / norm

        return embedding

    @staticmethod
    def compare(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Since both embeddings are L2-normalized, this is just a dot product!

        Args:
            embedding1: First face embedding (128,)
            embedding2: Second face embedding (128,)

        Returns:
            Similarity score from -1.0 to 1.0.
            In practice for faces:
                > 0.5:  very likely the same person
                0.3-0.5: possibly the same person (needs review)
                < 0.3:  different people

        Note: OpenCV's published SFace threshold is 0.363 (cosine).
        We use slightly different thresholds calibrated for our use case.
        """
        # Since both are L2-normalized: cosine = dot product
        return float(np.dot(embedding1, embedding2))

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """
        Convert a numpy embedding to bytes for database storage.

        A 128×float32 embedding becomes exactly 512 bytes.

        WHY NOT pickle?
        pickle is Python-specific and includes metadata overhead.
        .tobytes() is raw, fast, and language-agnostic. We know the
        format (128×float32) so we can reconstruct it trivially.
        """
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(blob: bytes) -> np.ndarray:
        """
        Convert database BLOB bytes back to a numpy embedding.

        Inverse of embedding_to_bytes().
        """
        return np.frombuffer(blob, dtype=np.float32).copy()
