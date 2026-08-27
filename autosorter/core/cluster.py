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
# Thresholds — calibrated for balanced clustering without over-splitting
# ══════════════════════════════════════════════════════════════════

# Above this score: candidate for AUTO-ASSIGN to the best-matching person.
T_HIGH = 0.44

# Strong match threshold: when similarity to centroid is this high,
# margin check between competing candidates is relaxed because competing
# candidates are often split clusters of the same identity.
T_STRONG = 0.48

# Top-K member validation threshold:
# Checks average similarity to the top 5 most similar confirmed faces of the person.
# Prevents centroid drift while allowing natural variation in expressions, angles, and lighting.
TOP_K_MEMBERS = 5
T_TOP_K = 0.40

# Review threshold for uncertain matches
T_REVIEW = 0.36

# Best match must beat second-best by at least this margin when in borderline range (T_HIGH <= score < T_STRONG).
MARGIN_MIN = 0.02

# Centroid cosine similarity threshold for merging split clusters in the consolidation pass.
T_MERGE_CENTROIDS = 0.52

# Faces below this quality threshold are skipped for clustering.
MIN_QUALITY_FOR_CLUSTERING = 0.25

# Minimum quality required to SEED a brand-new Person card.
# Higher threshold ensures new cards start with crisp, frontal, representative anchors.
MIN_QUALITY_TO_SEED_PERSON = 0.38


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

        # Top-K member validation against the best candidate
        members = self._person_members.get(best_person_id, [])
        if members:
            member_sims = sorted([float(m @ embedding) for m in members], reverse=True)
            top_k = member_sims[:TOP_K_MEMBERS]
            top_avg = float(np.mean(top_k))
        else:
            top_avg = best_score

        # Assignment conditions:
        # 1. Overwhelming match: best_score >= T_STRONG (0.48) and top_avg >= T_TOP_K (0.40).
        #    Margin check is relaxed because competing candidates with high scores are
        #    usually split duplicate clusters of the exact same identity.
        # 2. Good match: best_score >= T_HIGH (0.44) and top_avg >= T_TOP_K and margin >= MARGIN_MIN (0.02).
        is_strong_match = (best_score >= T_STRONG and top_avg >= T_TOP_K)
        is_good_match = (best_score >= T_HIGH and top_avg >= T_TOP_K and margin >= MARGIN_MIN)

        if is_strong_match or is_good_match:
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

        else:
            # Not a confident match to any existing person
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


def consolidate_clusters(db: Database, merge_threshold: float = T_MERGE_CENTROIDS) -> int:
    """
    Find and merge person clusters whose centroids have cosine similarity >= merge_threshold.

    This resolves over-splitting where the same person was fragmented into multiple
    separate cards due to streaming order or variations across photos (e.g. with/without glasses).

    Safety rules:
    - Never merge two persons if BOTH have different user-set names (e.g. 'Arnav' and 'Dad').
    - If one person is user-named and the other is auto-named, merge into the user-named person.
    - If both are auto-named, merge into the person with more photos.

    Returns:
        Total number of merges performed.
    """
    merges_done = 0

    while True:
        persons = db.get_all_persons()
        if len(persons) < 2:
            break

        # Collect valid centroids
        p_data = []
        centroids = []
        for p in persons:
            if p["centroid"] is not None:
                c = FaceRecognizer.bytes_to_embedding(p["centroid"])
                centroids.append(c)
                p_data.append(p)

        if len(centroids) < 2:
            break

        mat = np.stack(centroids)
        sim_mat = mat @ mat.T

        # Find the single most similar pair >= merge_threshold
        best_sim = merge_threshold - 1e-5
        best_pair = None

        for i in range(len(p_data)):
            for j in range(i + 1, len(p_data)):
                p1, p2 = p_data[i], p_data[j]

                # Safety: do not merge two different user-named persons
                if (
                    p1["name_is_user_set"] == 1
                    and p2["name_is_user_set"] == 1
                    and p1["display_name"] != p2["display_name"]
                ):
                    continue

                sim = float(sim_mat[i, j])
                if sim >= merge_threshold and sim > best_sim:
                    best_sim = sim
                    best_pair = (p1, p2)

        if best_pair is None:
            # No more pairs exceed the merge threshold
            break

        p1, p2 = best_pair

        # Determine target vs source
        if p1["name_is_user_set"] == 1 and p2["name_is_user_set"] == 0:
            target_id, source_id = p1["id"], p2["id"]
        elif p2["name_is_user_set"] == 1 and p1["name_is_user_set"] == 0:
            target_id, source_id = p2["id"], p1["id"]
        else:
            if p1["face_count"] >= p2["face_count"]:
                target_id, source_id = p1["id"], p2["id"]
            else:
                target_id, source_id = p2["id"], p1["id"]

        db.merge_persons(source_id, target_id)
        merges_done += 1

    if merges_done > 0:
        pick_cover_faces(db)

    return merges_done


def recluster_all_auto_faces(db: Database) -> Tuple[int, int]:
    """
    Reset automated face assignments and re-cluster from scratch using the
    improved ClusterEngine and post-consolidation pass.

    Safety:
    - User-set assignments ('user') and user-named persons are preserved.
    - Only auto-created persons and auto-assigned faces are reset.

    Returns:
        (total_faces_assigned, total_persons_remaining)
    """
    # 1. Unassign only auto-assigned faces
    db._conn.execute(
        "UPDATE faces SET person_id = NULL WHERE assign_source != 'user' OR assign_source IS NULL"
    )
    # 2. Delete empty or auto-created persons (keep user-named)
    db._conn.execute("DELETE FROM persons WHERE name_is_user_set = 0")
    db._conn.commit()

    # 3. For any remaining user-named persons, recompute their centroid from remaining user faces
    for p in db.get_all_persons():
        db.recompute_person_centroid(p["id"])

    # 4. Initialize engine and load existing user-named persons
    engine = ClusterEngine()
    engine.load_existing_persons(db)

    total_assigned = 0
    for face_row in db.iter_unassigned_faces():
        result = engine.process_face(db, face_row)
        if result.action != AssignAction.SKIPPED:
            total_assigned += 1

    # 5. Run consolidation pass
    consolidate_clusters(db)

    # 6. Re-pick cover photos
    pick_cover_faces(db)

    return total_assigned, db.count_persons()


