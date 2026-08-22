"""
Incremental centroid clustering — groups faces into people.

=== WHAT IS THIS FILE? ===
This is the BRAIN of people discovery. Given a set of face embeddings,
it figures out which faces belong to the same person by grouping similar
embeddings into clusters.

=== HOW INCREMENTAL CENTROID CLUSTERING WORKS ===
Unlike batch clustering (K-means, DBSCAN), we process faces ONE AT A TIME:

  1. Take the first (highest quality) face → create "Person 1"
  2. Take the next face → compare to all existing persons' centroids
     - Similar enough? → assign to that person, update centroid
     - Not similar? → create a new person
  3. Repeat for all faces

WHY INCREMENTAL (not batch)?
- Works naturally with streaming data (scan in progress)
- Resumable: can stop and continue later (M6)
- Each face is assigned exactly once → deterministic results
- No need to know the number of clusters in advance

=== WHAT IS A CENTROID? ===
The centroid is the AVERAGE embedding of all faces in a cluster.
Think of it as the "center" of the cluster in 128-dimensional space.

Person "Arnav" has 50 photos:
  centroid = average(embedding_1, embedding_2, ..., embedding_50)

To check if a new face is Arnav, we compare it to Arnav's centroid:
  similarity(new_face, arnav_centroid) → 0.62  (high → probably Arnav!)

=== CONSERVATIVE APPROACH ===
From the plan: "the app would rather show the same person as two cards
(1-click Merge fixes it) than mix two different people together."

This means our thresholds are STRICT. We'd rather create "Person 5"
and "Person 8" that are both Arnav (user merges later) than accidentally
put Arnav's faces into Dad's cluster (unfixable in M4).

=== MARGIN CHECK ===
Even if the best score is above threshold, we check the MARGIN:
  best_score - second_best_score >= MARGIN_MIN

If the margin is too small, the face is ambiguous (could be either person)
→ we create a new person instead of guessing wrong.

Example:
  Face vs Arnav: 0.48
  Face vs Dad:   0.45
  Margin: 0.03 < 0.08 → TOO CLOSE! Create new person.
  (Later, user can merge if it's actually Arnav.)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


# ══════════════════════════════════════════════════════════════════
# Thresholds — the key knobs that control clustering behavior
# ══════════════════════════════════════════════════════════════════

# Above this score: AUTO-ASSIGN to the best-matching person.
# SFace's published threshold is 0.363 cosine. We use 0.38 — slightly
# above the published value, but much lower than the initial 0.45 which
# caused massive over-splitting on real photos with varied lighting/angles.
T_HIGH = 0.38

# Between T_REVIEW and T_HIGH: uncertain match.
# In M4, we still auto-assign these (no review queue yet).
# In M5, these will go to the "Needs Review" queue.
T_REVIEW = 0.28

# Best match must beat second-best by at least this much.
# Prevents assigning a face when two people look equally similar.
# Lowered from 0.08 to 0.04 — the original was too strict and caused
# legitimate matches to be rejected when multiple similar-looking people exist.
MARGIN_MIN = 0.04

# Faces below this quality threshold are skipped for clustering.
# They'd poison centroids with noise. They're still stored in the DB
# and shown in photos, but not assigned to any person.
MIN_QUALITY_FOR_CLUSTERING = 0.25


class AssignAction(Enum):
    """What happened when we processed a face."""
    AUTO_ASSIGN = "auto"       # High confidence → assigned automatically
    REVIEW_ASSIGN = "review"   # Medium confidence → assigned (review queue in M5)
    NEW_PERSON = "new"         # No match → created a new person
    SKIPPED = "skipped"        # Quality too low → not clustered


@dataclass
class AssignResult:
    """Result of processing one face through the clustering engine."""
    face_id: int
    person_id: int           # The person this face was assigned to (or new person)
    action: AssignAction     # What happened
    confidence: float        # Similarity score (0.0 for new persons)


class ClusterEngine:
    """
    Incremental centroid-based face clustering engine.

    Maintains a matrix of all person centroids for fast similarity
    computation via matrix multiplication.

    Usage:
        engine = ClusterEngine()
        engine.load_existing_persons(db)

        for face in db.iter_unassigned_faces():
            result = engine.process_face(db, face)
            # result tells you: assigned to person X, action = auto/new/etc.

    HOW THE FAST SIMILARITY WORKS:
    Instead of a Python loop comparing each face to each person:
        for person in persons:
            sim = dot(face_emb, person_centroid)  ← slow!

    We stack ALL centroids into a matrix and do ONE operation:
        similarities = centroid_matrix @ face_emb  ← fast!

    For 100 persons, this computes all 100 similarities at once
    using optimized numpy/BLAS code. Even 10,000 persons would be fast.
    """

    def __init__(self) -> None:
        # Map of person_id → index in the centroid matrix
        self._person_ids: List[int] = []

        # Map of person_id → current face count (for centroid updates)
        self._face_counts: Dict[int, int] = {}

        # (N, 128) matrix of L2-normalized centroids, one row per person
        # None until at least one person exists
        self._centroids: Optional[np.ndarray] = None

    @property
    def person_count(self) -> int:
        """Number of persons currently tracked."""
        return len(self._person_ids)

    def load_existing_persons(self, db: Database) -> None:
        """
        Load all existing persons' centroids from the database.

        Called at startup before processing new faces. This way,
        if the user restarts the app, we continue from where we left off.
        """
        persons = db.get_all_persons()

        if not persons:
            return

        centroids = []
        for person in persons:
            if person["centroid"] is None:
                continue

            centroid = FaceRecognizer.bytes_to_embedding(person["centroid"])
            self._person_ids.append(person["id"])
            self._face_counts[person["id"]] = person["face_count"]
            centroids.append(centroid)

        if centroids:
            self._centroids = np.stack(centroids)

    def process_face(
        self,
        db: Database,
        face_row,
    ) -> AssignResult:
        """
        Process a single unassigned face: assign to existing person or create new.

        This is the core method called for every face during clustering.

        Args:
            db:       Database instance (for creating persons/assignments).
            face_row: A sqlite3.Row from iter_unassigned_faces().

        Returns:
            AssignResult describing what happened.
        """
        face_id = face_row["id"]
        quality = face_row["quality"] or 0.0

        # Quality gate: skip low-quality faces
        if quality < MIN_QUALITY_FOR_CLUSTERING:
            return AssignResult(
                face_id=face_id,
                person_id=0,
                action=AssignAction.SKIPPED,
                confidence=0.0,
            )

        # Get the face's embedding
        embedding = FaceRecognizer.bytes_to_embedding(face_row["embedding"])

        # If no persons exist yet, create the first one
        if self._centroids is None or len(self._person_ids) == 0:
            person_id = self._create_new_person(db, face_id, embedding)
            return AssignResult(
                face_id=face_id,
                person_id=person_id,
                action=AssignAction.NEW_PERSON,
                confidence=0.0,
            )

        # Compute similarity to ALL persons at once (matrix multiplication)
        # (N, 128) @ (128,) → (N,) — one similarity per person
        similarities = self._centroids @ embedding

        # Find the best and second-best matches
        sorted_indices = np.argsort(similarities)[::-1]
        best_idx = sorted_indices[0]
        best_score = float(similarities[best_idx])
        best_person_id = self._person_ids[best_idx]

        # Second-best score (for margin check)
        second_score = 0.0
        if len(sorted_indices) > 1:
            second_score = float(similarities[sorted_indices[1]])

        margin = best_score - second_score

        # Decision logic
        if best_score >= T_HIGH and margin >= MARGIN_MIN:
            # HIGH CONFIDENCE: auto-assign to best person
            self._assign_to_person(
                db, face_id, best_person_id, embedding, quality, best_score,
                source="auto",
            )
            return AssignResult(
                face_id=face_id,
                person_id=best_person_id,
                action=AssignAction.AUTO_ASSIGN,
                confidence=best_score,
            )

        elif best_score >= T_REVIEW and margin >= MARGIN_MIN:
            # MEDIUM CONFIDENCE: assign but mark for review (M5).
            # In M4 we treat this as auto-assign since there's no review UI yet.
            self._assign_to_person(
                db, face_id, best_person_id, embedding, quality, best_score,
                source="review",
            )
            return AssignResult(
                face_id=face_id,
                person_id=best_person_id,
                action=AssignAction.REVIEW_ASSIGN,
                confidence=best_score,
            )

        else:
            # LOW CONFIDENCE or AMBIGUOUS: create a new person
            person_id = self._create_new_person(db, face_id, embedding)
            return AssignResult(
                face_id=face_id,
                person_id=person_id,
                action=AssignAction.NEW_PERSON,
                confidence=best_score,
            )

    def _create_new_person(
        self,
        db: Database,
        face_id: int,
        embedding: np.ndarray,
    ) -> int:
        """
        Create a new person seeded with this face.

        The face's embedding becomes the person's initial centroid.
        """
        # Auto-name: "Person 1", "Person 2", ...
        person_number = db.get_next_person_number()
        display_name = f"Person {person_number}"

        # This face's embedding IS the centroid (only member of the cluster)
        centroid_bytes = FaceRecognizer.embedding_to_bytes(embedding)

        person_id = db.insert_person(
            display_name=display_name,
            centroid_bytes=centroid_bytes,
            face_count=1,
            cover_face_id=face_id,
        )

        # Assign the face to this new person
        db.assign_face_to_person(face_id, person_id, source="auto", confidence=1.0)

        # Update our in-memory state
        self._person_ids.append(person_id)
        self._face_counts[person_id] = 1

        if self._centroids is None:
            self._centroids = embedding.reshape(1, -1)
        else:
            self._centroids = np.vstack([self._centroids, embedding.reshape(1, -1)])

        return person_id

    def _assign_to_person(
        self,
        db: Database,
        face_id: int,
        person_id: int,
        embedding: np.ndarray,
        quality: float,
        score: float,
        source: str,
    ) -> None:
        """
        Assign a face to an existing person and update the centroid.

        CENTROID UPDATE (quality-weighted incremental):
        Instead of recomputing the average from scratch every time
        (which would require loading all embeddings), we update
        incrementally:

          new_centroid = (old_centroid * old_count + new_embedding * quality)
                         / (old_count + quality)

        This is a weighted running average. High-quality faces (clear,
        frontal, large) contribute more to the centroid than low-quality
        ones (blurry, small, profile).

        The result is re-normalized to unit length for cosine similarity.
        """
        # Assign face in database
        db.assign_face_to_person(face_id, person_id, source=source, confidence=score)

        # Find this person's index in our arrays
        idx = self._person_ids.index(person_id)
        old_centroid = self._centroids[idx]
        old_count = self._face_counts[person_id]

        # Quality-weighted incremental centroid update
        # Quality acts as the weight for this face (high quality = more influence)
        weight = max(quality, 0.1)  # floor at 0.1 to avoid zero-weight
        total_weight = old_count + weight
        new_centroid = (old_centroid * old_count + embedding * weight) / total_weight

        # Re-normalize to unit length (critical for cosine similarity)
        norm = np.linalg.norm(new_centroid)
        if norm > 1e-10:
            new_centroid = new_centroid / norm

        # Update in-memory state
        new_count = old_count + 1
        self._centroids[idx] = new_centroid
        self._face_counts[person_id] = new_count

        # Persist to database
        centroid_bytes = FaceRecognizer.embedding_to_bytes(new_centroid)
        db.update_person_centroid(person_id, centroid_bytes, new_count)


def pick_cover_faces(db: Database) -> None:
    """
    For each person, set the cover face to the highest-quality face.

    Called after clustering is complete. The cover face is the thumbnail
    displayed on the person card in the People panel.
    """
    for person in db.get_all_persons():
        faces = db.get_faces_for_person(person["id"])
        if faces:
            # faces are ordered by quality DESC, so first = best
            best_face = faces[0]
            db.update_person_cover(person["id"], best_face["id"])

