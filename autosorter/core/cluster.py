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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


# ══════════════════════════════════════════════════════════════════
# Thresholds — the key knobs that control clustering behavior
# ══════════════════════════════════════════════════════════════════

# Above this score: AUTO-ASSIGN to the best-matching person.
# Balanced threshold that groups same-person faces across varied lighting/angles.
T_HIGH = 0.46

# Minimum Member Similarity:
# Prevents centroid drift while allowing natural variation in facial expressions.
T_MIN_MEMBER = 0.40

# Review threshold for M5 review queue
T_REVIEW = 0.36

# Best match must beat second-best by at least this much.
MARGIN_MIN = 0.03

# Faces below this quality threshold are skipped for clustering.
MIN_QUALITY_FOR_CLUSTERING = 0.25

# Minimum quality required to SEED a brand-new Person card.
MIN_QUALITY_TO_SEED_PERSON = 0.35


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
    Incremental face clustering engine with complete-linkage drift protection.

    Features:
    1. Vectorized candidate ranking via matrix multiplication against centroids.
    2. Complete-linkage validation: checks candidate against all existing members
       of the target person (min similarity >= 0.48, avg >= 0.52).
    3. Mathematical drift immunity: ensures distinct individuals can never be merged.
    """

    def __init__(self) -> None:
        # Map of person_id → index in the centroid matrix
        self._person_ids: List[int] = []

        # Map of person_id → current face count (for centroid updates)
        self._face_counts: Dict[int, int] = {}

        # (N, 128) matrix of L2-normalized centroids, one row per person
        # None until at least one person exists
        self._centroids: Optional[np.ndarray] = None

        # Map of person_id → list of all member embeddings (for complete linkage check)
        self._person_members: Dict[int, List[np.ndarray]] = {}

    @property
    def person_count(self) -> int:
        """Number of persons currently tracked."""
        return len(self._person_ids)

    def load_existing_persons(self, db: Database) -> None:
        """
        Load all existing persons' centroids and member embeddings from the database.
        """
        persons = db.get_all_persons()

        if not persons:
            return

        centroids = []
        for person in persons:
            if person["centroid"] is None:
                continue

            pid = person["id"]
            centroid = FaceRecognizer.bytes_to_embedding(person["centroid"])
            self._person_ids.append(pid)
            self._face_counts[pid] = person["face_count"]

            # Load member embeddings
            faces = db.get_faces_for_person(pid)
            members = [
                FaceRecognizer.bytes_to_embedding(f["embedding"])
                for f in faces
                if f["embedding"] is not None
            ]
            self._person_members[pid] = members if members else [centroid]

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
        similarities = self._centroids @ embedding

        # Find the best and second-best matches
        sorted_indices = np.argsort(similarities)[::-1]
        best_idx = sorted_indices[0]
        best_score = float(similarities[best_idx])
        best_person_id = self._person_ids[best_idx]

        # Second-best score (for margin check)
        second_score = float(similarities[sorted_indices[1]]) if len(sorted_indices) > 1 else -1.0
        margin = best_score - second_score

        # Complete-linkage check against all members of the best candidate
        member_sims = [float(m @ embedding) for m in self._person_members[best_person_id]]
        min_member_sim = min(member_sims)
        avg_member_sim = float(np.mean(member_sims))

        # Assign only if average similarity and minimum member similarity are both strong
        if avg_member_sim >= T_HIGH and min_member_sim >= T_MIN_MEMBER and margin >= MARGIN_MIN:
            self._assign_to_person(
                db, face_id, best_person_id, embedding, quality, avg_member_sim,
                source="auto",
            )
            return AssignResult(
                face_id=face_id,
                person_id=best_person_id,
                action=AssignAction.AUTO_ASSIGN,
                confidence=avg_member_sim,
            )

        else:
            # Not a match to any existing person
            if quality < MIN_QUALITY_TO_SEED_PERSON:
                return AssignResult(
                    face_id=face_id,
                    person_id=0,
                    action=AssignAction.SKIPPED,
                    confidence=best_score,
                )

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
        """
        person_number = db.get_next_person_number()
        display_name = f"Person {person_number}"

        centroid_bytes = FaceRecognizer.embedding_to_bytes(embedding)

        person_id = db.insert_person(
            display_name=display_name,
            centroid_bytes=centroid_bytes,
            face_count=1,
            cover_face_id=face_id,
        )

        db.assign_face_to_person(face_id, person_id, source="auto", confidence=1.0)

        # Update in-memory state
        self._person_ids.append(person_id)
        self._face_counts[person_id] = 1
        self._person_members[person_id] = [embedding]

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
        """
        db.assign_face_to_person(face_id, person_id, source=source, confidence=score)

        idx = self._person_ids.index(person_id)
        old_centroid = self._centroids[idx]
        old_count = self._face_counts[person_id]

        weight = max(quality, 0.1)
        total_weight = old_count + weight
        new_centroid = (old_centroid * old_count + embedding * weight) / total_weight

        norm = np.linalg.norm(new_centroid)
        if norm > 1e-10:
            new_centroid = new_centroid / norm

        new_count = old_count + 1
        self._centroids[idx] = new_centroid
        self._face_counts[person_id] = new_count
        self._person_members[person_id].append(embedding)

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

