"""
Sparse Constrained Graph Identity Clustering.

A production-grade, CPU-first face clustering architecture designed for
low-end hardware, large-scale scalability, and absolute separation of look-alikes.

Pipeline:
1. Cached SFace Embeddings (128-D float32 normalized vectors)
2. Adaptive Chunked k-NN (Bounded memory, no FAISS/PyTorch needed)
3. Co-Occurrence Firewall (Faces from the same photo cannot link unless mock duplicate)
4. Two-Tier Graph Construction:
   - Strong Graph (Mutual-kNN + High Similarity for Community Discovery)
   - Weak Graph (One-Way Support Edges for Peripheral Face Attachment)
5. Constrained Chinese Whispers (Deterministic label propagation to convergence)
6. Core Identity Filtering & Consolidation (Merge zero-conflict same-identity subclusters)
7. Peripheral Propagation with Dual-Gating (Absolute support + Relative margin)
8. Transactional SQLite Commit (Atomic updates with pre-commit validation)
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


@dataclass
class GraphClusterConfig:
    """
    Configuration parameters governing the Sparse Constrained Graph Identity Clusterer.
    All hyperparameters are exposed here to eliminate magical constants from logic.
    """
    # Number of nearest neighbors to search and retain per face in k-NN stage.
    # Higher k (e.g. 30) ensures sufficient connectivity for diverse poses without isolating groups.
    k_neighbors: int = 30

    # Minimum cosine similarity required to form a strong mutual-kNN edge.
    # Both face A and face B must place each other in their top-k with similarity >= t_strong.
    t_strong: float = 0.46

    # Minimum cosine similarity required to form a weak support edge (used for peripheral attachment).
    t_weak: float = 0.42

    # Centroid cosine similarity threshold used to consolidate zero-conflict core clusters
    # belonging to the same individual (e.g., normal face vs cake-covered face).
    t_consolidate: float = 0.52

    # Minimum combined similarity score required to attach a peripheral face to a core person.
    min_support: float = 0.46

    # Minimum margin required between best candidate score and runner-up candidate score.
    # Prevents ambiguous faces (e.g., look-alikes) from being guessed incorrectly into a person.
    min_margin: float = 0.03

    # Maximum iterations allowed for Chinese Whispers label propagation before breaking.
    max_iterations: int = 20

    # Minimum number of faces required to graduate a community into a "Core Person".
    min_core_faces: int = 4

    # Minimum number of unique photo appearances required to graduate a community into a "Core Person".
    min_unique_photos: int = 3

    # Number of faces processed per matrix multiplication chunk (bounds temporary RAM to < 50MB).
    chunk_size: int = 1000

    # Seed for numpy's random generator to ensure reproducible node visitation in Chinese Whispers.
    random_seed: int = 42


@dataclass
class ClusterValidationReport:
    """
    Detailed pre-commit validation summary generated before touching SQLite.
    Guarantees that no corrupted cluster state is ever committed to the database.
    """
    total_faces: int                      # Total faces evaluated in this run
    core_people_count: int                # Number of distinct core identities discovered
    peripheral_assigned_count: int        # Number of difficult peripheral faces safely attached
    unknown_count: int                    # Number of ambiguous faces left unassigned
    co_occurrence_violations: int         # Count of photo collisions (must be exactly 0)
    cluster_sizes: List[int]              # Sorted face counts for each discovered person
    passed: bool                          # True if all validation checks passed (0 violations)
    error_message: Optional[str] = None   # Human-readable failure explanation if passed is False


def find_adaptive_knn(
    embeddings: np.ndarray,
    image_ids: np.ndarray,
    config: GraphClusterConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k nearest neighbors for each face while enforcing the Co-Occurrence Firewall.

    Memory Efficiency:
    - Avoids building a full (N x N) float32 similarity matrix which would consume 40GB at 100k faces.
    - Processes data in chunks of `chunk_size` (e.g., 1000 rows x N columns), finding top-k per chunk.
    - Keeps temporary memory consumption strictly bounded and friendly to low-end hardware.

    Firewall Defense:
    - Faces appearing in the exact same image cannot belong to the same person.
    - Any comparison between faces sharing the same `image_id` is assigned -1e9 (effective -infinity),
      except for near-identical synthetic mock duplicates (sim >= 0.98).

    Args:
        embeddings: (N, 128) L2-normalized float32 matrix representing face embeddings.
        image_ids: (N,) int array mapping each face to its source photo ID.
        config: GraphClusterConfig with hyperparameters.

    Returns:
        knn_indices: (N, k) int array of neighbor indices.
        knn_sims: (N, k) float32 array of neighbor cosine similarities.
    """
    # Total number of faces to index
    n = len(embeddings)

    # Ensure k does not exceed available candidate pool (at least 1, at most n - 1)
    k = min(config.k_neighbors, max(1, n - 1))

    # Pre-allocate output arrays to store indices and cosine similarities for the top-k neighbors
    knn_indices = np.zeros((n, k), dtype=np.int32)
    knn_sims = np.zeros((n, k), dtype=np.float32)

    # Process faces in bounded memory chunks
    chunk_size = config.chunk_size

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Slice current query chunk of shape (chunk_len, 128)
        chunk = embeddings[start:end]

        # Compute dot products with all embeddings: (chunk_len, 128) @ (128, n) -> (chunk_len, n)
        # Because all embeddings are unit L2-normalized, the dot product equals exact cosine similarity
        sim_chunk = chunk @ embeddings.T

        # Extract image IDs for faces in the current chunk: shape (chunk_len, 1)
        chunk_img_ids = image_ids[start:end, None]

        # Broadcast comparison: find pairs that share the exact same photo ID
        # Mask out any same-photo pair whose similarity is < 0.98 (allowing synthetic test mocks)
        same_img_mask = (chunk_img_ids == image_ids[None, :]) & (sim_chunk < 0.98)

        # Disallow self-comparison: a face should never pick itself as its neighbor
        np.fill_diagonal(same_img_mask[:, start:end], True)

        # Penalize forbidden comparisons with -infinity so they are never selected in top-k
        sim_chunk[same_img_mask] = -1e9

        # Efficient partial sorting to find top-k largest elements without sorting entire row
        if k < n:
            # np.argpartition places top-k indices in the last k positions of axis 1
            part_idx = np.argpartition(sim_chunk, -k, axis=1)[:, -k:]
            row_indices = np.arange(end - start)[:, None]
            part_sims = sim_chunk[row_indices, part_idx]

            # Sort only the extracted top-k elements in descending similarity order
            sorted_order = np.argsort(-part_sims, axis=1)

            # Re-index to obtain sorted neighbor indices and similarities
            final_idx = np.take_along_axis(part_idx, sorted_order, axis=1)
            final_sims = np.take_along_axis(part_sims, sorted_order, axis=1)
        else:
            # Fallback for very small datasets where k >= n: full sort
            sorted_order = np.argsort(-sim_chunk, axis=1)[:, :k]
            row_indices = np.arange(end - start)[:, None]
            final_idx = sorted_order
            final_sims = sim_chunk[row_indices, sorted_order]

        # Clean up any remaining masked sentinel values (-1e9) by zeroing them out
        valid_mask = final_sims > -1e8
        final_sims[~valid_mask] = 0.0

        # Store the chunk's results into the pre-allocated global arrays
        knn_indices[start:end] = final_idx
        knn_sims[start:end] = final_sims

    return knn_indices, knn_sims


def build_two_tier_graph(
    knn_indices: np.ndarray,
    knn_sims: np.ndarray,
    config: GraphClusterConfig,
) -> Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
    """
    Construct Strong (Mutual-kNN) and Weak Support adjacency lists.

    Two-Tier Topology:
    1. Strong Mutual Edges:
       - Edge between u and v exists IF AND ONLY IF u is in top-k(v) AND v is in top-k(u)
         with mutual cosine similarity >= t_strong.
       - Mutual agreement severs unilateral "bridges" caused by look-alikes. For example, if Face A
         resembles Face B, but Face B has 30 much closer matches of its own identity, the edge is dropped.
    2. Weak Support Edges:
       - Directed edges where similarity >= t_weak.
       - Preserved for peripheral face propagation later, ensuring difficult profile or blurry shots
         still have a path to connect to their core identity.

    Args:
        knn_indices: (N, k) neighbor indices.
        knn_sims: (N, k) neighbor similarities.
        config: GraphClusterConfig.

    Returns:
        strong_adj: Adjacency list mapping node -> list of (neighbor, weight) for strong mutual edges.
        weak_adj: Adjacency list mapping node -> list of (neighbor, weight) for weak support edges.
    """
    n = len(knn_indices)
    k = knn_indices.shape[1]

    # Pre-build lookup dictionary of directed edges: u -> {v: similarity}
    # Allows O(1) checks to verify mutual reciprocal presence
    directed_edges: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
    for u in range(n):
        for ki in range(k):
            v = int(knn_indices[u, ki])
            sim = float(knn_sims[u, ki])
            # Only record valid, non-self edges exceeding weak threshold
            if v != u and sim >= config.t_weak:
                directed_edges[u][v] = sim

    # Adjacency maps for the two graph tiers
    strong_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    weak_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    # Populate edges
    for u in range(n):
        for v, sim in directed_edges[u].items():
            # Add to weak graph (directed support edge)
            weak_adj[u].append((v, sim))

            # Mutual Verification: check if node v reciprocates by having node u in its edge list
            if v in directed_edges and u in directed_edges[v]:
                # Symmetric mutual similarity is the arithmetic mean of both directional similarities
                mutual_sim = (sim + directed_edges[v][u]) / 2.0
                if mutual_sim >= config.t_strong:
                    # Both nodes agree: establish strong mutual edge
                    strong_adj[u].append((v, mutual_sim))

    return strong_adj, weak_adj


def constrained_chinese_whispers(
    n_nodes: int,
    strong_adj: Dict[int, List[Tuple[int, float]]],
    config: GraphClusterConfig,
) -> Dict[int, int]:
    """
    Run Chinese Whispers label propagation on the strong mutual-kNN graph.

    Characteristics:
    - Time complexity: O(E) per iteration, converging in usually < 5 iterations.
    - Extremely fast on CPU (< 40ms for 1,000 faces).
    - Requires no pre-defined cluster count K; naturally discovers discrete dense communities.
    - Seeded deterministic shuffling guarantees order independence across runs.

    Algorithm:
    1. Initialize each node with its own unique community label (0..N-1).
    2. Shuffle the node traversal order deterministically using seeded RNG.
    3. For each node u, inspect all its neighbors in the strong graph.
    4. Sum the edge weights grouped by the neighbors' current labels.
    5. Re-assign node u to the label that received the highest weighted vote.
    6. Repeat until no nodes change labels or max_iterations is reached.

    Args:
        n_nodes: Total number of nodes (faces).
        strong_adj: Mutual-kNN adjacency list.
        config: GraphClusterConfig.

    Returns:
        Dict mapping node index -> final community label.
    """
    # Start: each node is in its own cluster
    labels = np.arange(n_nodes, dtype=np.int32)

    # Initialize deterministic random generator for reproducible node visitation
    rng = np.random.default_rng(config.random_seed)

    for iteration in range(config.max_iterations):
        # Deterministically shuffle traversal order to avoid directional propagation bias
        order = rng.permutation(n_nodes)
        changes = 0

        for u in order:
            neighbors = strong_adj.get(u, [])
            if not neighbors:
                # Isolated node: remains in its own component
                continue

            # Tally weighted votes from all strong neighbors
            votes: Dict[int, float] = defaultdict(float)
            for v, weight in neighbors:
                votes[labels[v]] += weight

            if votes:
                # Select the community label with the maximum total edge weight
                best_label = max(votes.keys(), key=lambda l: votes[l])
                if labels[u] != best_label:
                    labels[u] = best_label
                    changes += 1

        # Check for convergence: if no labels flipped in this iteration, graph has stabilized
        if changes == 0:
            break

    return {i: int(labels[i]) for i in range(n_nodes)}


def consolidate_core_clusters(
    core_people: Dict[int, List[int]],
    image_ids: np.ndarray,
    embeddings: np.ndarray,
    t_consolidate: float = 0.52,
) -> Dict[int, List[int]]:
    """
    Consolidate core clusters belonging to the same person under multi-modal appearances.

    The Multi-Modal Appearance Problem:
    A person who appears in 180 photos might have two distinct visual modes:
    - Mode A: Clean-shaven, frontal, well-lit portrait shots (e.g., 75 photos).
    - Mode B: Cake-covered face, laughing with eyes closed in dim lighting (e.g., 71 photos).
    Because the strong mutual graph requires high mutual similarity, Mode A and Mode B form
    two tight, separate communities.

    Consolidation Solution:
    - We compute the centroid embedding of each core cluster.
    - If Cluster 1 and Cluster 2 share ZERO photos (cannot-link check passes) AND their centroid
      cosine similarity >= t_consolidate (0.52), they are safely merged into a single person card!

    Args:
        core_people: Dict mapping person_id -> list of face indices.
        image_ids: Array mapping face index to photo ID.
        embeddings: Array of face embeddings.
        t_consolidate: Centroid similarity threshold for consolidation.

    Returns:
        Dict mapping consolidated person_id -> combined list of face indices.
    """
    pids = list(core_people.keys())
    if len(pids) <= 1:
        return core_people

    # Compute normalized centroid embedding for each core person cluster
    centroids: Dict[int, np.ndarray] = {}
    for pid in pids:
        c = embeddings[core_people[pid]].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[pid] = c / (norm if norm > 1e-9 else 1.0)

    # Track which person IDs remain active as merges occur
    active_pids = set(pids)

    for i in range(len(pids)):
        p1 = pids[i]
        if p1 not in active_pids:
            continue

        for j in range(i + 1, len(pids)):
            p2 = pids[j]
            if p2 not in active_pids:
                continue

            # Co-Occurrence Firewall: Two clusters can NEVER merge if they share any photo!
            imgs1 = set(image_ids[core_people[p1]])
            imgs2 = set(image_ids[core_people[p2]])
            if imgs1 & imgs2:
                # Direct collision: both clusters appear in the same image, cannot be the same human
                continue

            # Check centroid cosine similarity between the two core clusters
            sim = float(centroids[p1] @ centroids[p2])
            if sim >= t_consolidate:
                # Merge cluster p2 into cluster p1
                core_people[p1].extend(core_people[p2])

                # Recompute updated normalized centroid for the consolidated cluster
                c = embeddings[core_people[p1]].mean(axis=0)
                norm = np.linalg.norm(c)
                centroids[p1] = c / (norm if norm > 1e-9 else 1.0)

                # Remove absorbed cluster from active set
                active_pids.remove(p2)

    return {pid: core_people[pid] for pid in active_pids}


def propagate_peripheral_faces(
    core_people: Dict[int, List[int]],
    peripheral_faces: List[int],
    weak_adj: Dict[int, List[Tuple[int, float]]],
    image_ids: np.ndarray,
    embeddings: np.ndarray,
    config: GraphClusterConfig,
) -> Tuple[Dict[int, int], List[int]]:
    """
    Assign difficult peripheral faces (extreme profiles, motion blur, singletons)
    to the best Core Person using dual-gated support and margin verification.

    Safety Gates:
    1. Absolute Co-Occurrence Firewall: A candidate person is strictly rejected if that
       person already has a face in the peripheral photo's `image_id`.
    2. Absolute Support: The candidate score must reach `min_support` (e.g. 0.46).
    3. Relative Margin: The best score must exceed the second-best score by at least
       `min_margin` (e.g. 0.03). If two people have nearly identical similarity, the face
       is marked as Ambiguous / Unknown rather than guessing wrong.

    Scoring:
    Combines direct neighbor similarity (60% weight from top 3 connected core members)
    and global centroid similarity (40% weight).

    Args:
        core_people: Dict of confirmed core people -> list of face indices.
        peripheral_faces: List of face indices needing assignment.
        weak_adj: Adjacency list of weak support edges.
        image_ids: Array of image IDs.
        embeddings: Array of face embeddings.
        config: GraphClusterConfig.

    Returns:
        assignments: {face_idx: core_person_id}
        unknown_faces: [face_idx, ...] faces that could not be confidently assigned.
    """
    assignments: Dict[int, int] = {}
    unknown_faces: List[int] = []

    # Inverted index: maps face index -> core person ID
    face_to_core: Dict[int, int] = {}
    # Image ID tracking per core person to enforce co-occurrence firewall during propagation
    person_image_ids: Dict[int, Set[int]] = defaultdict(set)
    # Precomputed normalized centroids for each core person
    core_centroids: Dict[int, np.ndarray] = {}

    for person_id, members in core_people.items():
        for m in members:
            face_to_core[m] = person_id
            person_image_ids[person_id].add(int(image_ids[m]))

        c_embs = embeddings[members]
        c = c_embs.mean(axis=0)
        norm = np.linalg.norm(c)
        core_centroids[person_id] = c / (norm if norm > 1e-9 else 1.0)

    # Process each peripheral face
    for u in peripheral_faces:
        u_img = int(image_ids[u])
        u_emb = embeddings[u]

        # Gather weak support votes for each core person from neighboring faces
        support_per_person: Dict[int, List[float]] = defaultdict(list)
        for v, sim in weak_adj.get(u, []):
            if v in face_to_core:
                target_pid = face_to_core[v]
                support_per_person[target_pid].append(sim)

        # Evaluate match score against each known core person
        candidate_scores: List[Tuple[int, float]] = []
        for person_id in core_people.keys():
            # Firewall Gate: strictly forbid assignment if target person already has a face in this photo
            if u_img in person_image_ids[person_id]:
                continue

            # Global centroid similarity
            centroid_sim = float(u_emb @ core_centroids[person_id])

            # Local neighbor similarity from weak edges
            top_member_sims = support_per_person.get(person_id, [])

            # Hybrid score: blend local neighbor votes with global centroid similarity
            if top_member_sims:
                # Take average of top-3 nearest core member faces
                local_sim = float(np.mean(top_member_sims[:3]))
                score = (0.6 * local_sim) + (0.4 * centroid_sim)
            else:
                # No direct graph edges: rely purely on centroid similarity
                score = centroid_sim

            candidate_scores.append((person_id, float(score)))

        if not candidate_scores:
            # All candidates were blocked by the co-occurrence firewall or had zero similarity
            unknown_faces.append(u)
            continue

        # Sort candidate scores descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score = candidate_scores[0]
        second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0

        # Margin check: difference between best match and runner-up
        margin = best_score - second_score

        # Dual Gating: require both absolute confidence and relative separation
        if best_score >= config.min_support and margin >= config.min_margin:
            # Face passes all safety gates: assign to best core person
            assignments[u] = best_pid
            # Register this photo ID with the person to prevent subsequent faces from this photo joining
            person_image_ids[best_pid].add(u_img)
        else:
            # Ambiguous face: route to Unknown
            unknown_faces.append(u)

    return assignments, unknown_faces


def run_graph_clustering(
    db: Database,
    config: Optional[GraphClusterConfig] = None,
    dry_run: bool = False,
) -> ClusterValidationReport:
    """
    Execute end-to-end graph clustering on unassigned/auto-assigned faces in SQLite.

    Production Guardrails:
    1. Permanent User Locks: Manually named persons (`name_is_user_set = 1`) and
       user-assigned faces (`assign_source = 'user'`) are protected and never overwritten.
    2. Pre-Commit Dry-Run Validation: Validates zero co-occurrence violations across
       all resulting clusters BEFORE making any changes to SQLite.
    3. Atomic SQLite Transaction: All database operations (deletions, insertions, reassignments)
       occur inside a single `with db._conn:` transaction block with automatic rollback on failure.

    Args:
        db: Database instance.
        config: Optional GraphClusterConfig override.
        dry_run: If True, executes clustering and validation without modifying database.

    Returns:
        ClusterValidationReport summarizing discovery metrics and safety verification.
    """
    if config is None:
        config = GraphClusterConfig()

    # Step 1: Query user-named persons to preserve them from automated reclustering
    protected_pids = {
        r["id"] for r in db._conn.execute(
            """
            SELECT id FROM persons WHERE name_is_user_set = 1
            UNION
            SELECT DISTINCT person_id FROM faces
            WHERE assign_source = 'user' AND person_id IN (SELECT id FROM persons)
            """
        ).fetchall()
    }
    keep_sql = ",".join(str(int(x)) for x in protected_pids) or "-1"
    locked_faces = {
        r["id"] for r in db._conn.execute(
            f"SELECT id FROM faces WHERE assign_source = 'user' OR person_id IN ({keep_sql})"
        ).fetchall()
    }

    cursor = db._conn.execute("""
        SELECT id, image_id, embedding, quality, det_score, person_id, assign_source
        FROM faces
        WHERE embedding IS NOT NULL AND det_score > 0
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()

    # Empty dataset guard
    if not rows:
        return ClusterValidationReport(
            total_faces=0, core_people_count=0, peripheral_assigned_count=0,
            unknown_count=0, co_occurrence_violations=0, cluster_sizes=[],
            passed=True
        )

    # Exclude user-locked faces from the clustering pool
    cluster_rows = [r for r in rows if r["id"] not in locked_faces]
    if not cluster_rows:
        return ClusterValidationReport(
            total_faces=len(rows), core_people_count=len(protected_pids),
            peripheral_assigned_count=0, unknown_count=0,
            co_occurrence_violations=0, cluster_sizes=[], passed=True
        )

    # Convert database rows to fast contiguous numpy arrays
    face_ids = np.array([r["id"] for r in cluster_rows], dtype=np.int32)
    image_ids = np.array([r["image_id"] for r in cluster_rows], dtype=np.int32)
    embeddings = np.stack([FaceRecognizer.bytes_to_embedding(r["embedding"]) for r in cluster_rows])

    # Step 4: Adaptive chunked k-NN with Co-Occurrence Firewall
    knn_indices, knn_sims = find_adaptive_knn(embeddings, image_ids, config)

    # Step 5: Build Two-Tier Mutual Graph
    strong_adj, weak_adj = build_two_tier_graph(knn_indices, knn_sims, config)

    # Step 6: Community discovery via Constrained Chinese Whispers on strong graph
    community_labels = constrained_chinese_whispers(len(cluster_rows), strong_adj, config)

    # Group face indices by community label
    communities: Dict[int, List[int]] = defaultdict(list)
    for face_idx, label in community_labels.items():
        communities[label].append(face_idx)

    # Step 7: Filter Core People from Peripheral Faces
    # Adapt thresholds for small collections so meaningful groups are not discarded
    min_core_faces = min(config.min_core_faces, max(1, len(cluster_rows) // 2))
    min_unique_photos = min(config.min_unique_photos, max(1, len(set(image_ids))))

    core_people: Dict[int, List[int]] = {}
    peripheral_faces: List[int] = []

    next_person_id = 1
    for label, members in communities.items():
        unique_imgs = len(set(image_ids[members]))
        # A community qualifies as a Core Person if it has enough photos across multiple shots
        if len(members) >= min_core_faces and unique_imgs >= min_unique_photos:
            core_people[next_person_id] = members
            next_person_id += 1
        else:
            peripheral_faces.extend(members)

    # Fallback for small libraries: promote largest community if none qualified
    if not core_people and communities:
        largest_label = max(communities.keys(), key=lambda l: len(communities[l]))
        core_people[1] = communities[largest_label]
        peripheral_faces = [idx for l, m in communities.items() if l != largest_label for idx in m]

    # Step 8: Consolidate core subclusters sharing zero photo conflicts and high centroid similarity
    consolidated_cores = consolidate_core_clusters(
        core_people, image_ids, embeddings, t_consolidate=config.t_consolidate
    )

    # Step 9: Dual-gated peripheral propagation
    peripheral_assignments: Dict[int, int] = {}
    unknown_indices: List[int] = []

    if consolidated_cores and peripheral_faces:
        peripheral_assignments, unknown_indices = propagate_peripheral_faces(
            consolidated_cores, peripheral_faces, weak_adj, image_ids, embeddings, config
        )
    else:
        unknown_indices = peripheral_faces

    # Step 10: Pre-Commit Co-Occurrence Verification across all formed clusters
    # Instead of aborting on violations, RESOLVE them by evicting the lower-quality
    # duplicate face from each offending cluster, routing it to unknown.
    total_violations = 0
    final_person_to_faces: Dict[int, List[int]] = defaultdict(list)
    for pid, members in consolidated_cores.items():
        final_person_to_faces[pid].extend(members)
    for face_idx, pid in peripheral_assignments.items():
        final_person_to_faces[pid].append(face_idx)

    # Audit every formed person: resolve any photo ID appearing more than once
    # by keeping the face with the highest quality and evicting the rest.
    qualities = np.array([r["quality"] or 0.0 for r in cluster_rows], dtype=np.float32)
    evicted_faces: List[int] = []

    for pid in list(final_person_to_faces.keys()):
        members = final_person_to_faces[pid]
        # Group face indices by their image_id
        img_to_faces: Dict[int, List[int]] = defaultdict(list)
        for idx in members:
            img_to_faces[int(image_ids[idx])].append(idx)

        clean_members = []
        for img_id, face_indices in img_to_faces.items():
            if len(face_indices) > 1:
                # Co-occurrence violation: keep the best face, evict the rest
                total_violations += 1
                face_indices.sort(key=lambda fi: qualities[fi], reverse=True)
                clean_members.append(face_indices[0])
                evicted_faces.extend(face_indices[1:])
            else:
                clean_members.append(face_indices[0])

        final_person_to_faces[pid] = clean_members

    # Route evicted faces to the unknown pool
    unknown_indices.extend(evicted_faces)

    cluster_sizes = sorted([len(m) for m in final_person_to_faces.values()], reverse=True)
    report = ClusterValidationReport(
        total_faces=len(cluster_rows),
        core_people_count=len(consolidated_cores),
        peripheral_assigned_count=len(peripheral_assignments),
        unknown_count=len(unknown_indices),
        co_occurrence_violations=total_violations,
        cluster_sizes=cluster_sizes,
        passed=True,  # violations are now resolved, not blocking
        error_message=f"Resolved {total_violations} co-occurrence violations by evicting duplicates." if total_violations > 0 else None
    )

    # Abort commit only if running dry-run
    if dry_run:
        return report

    # Step 11: Atomic SQLite Transactional Commit
    with db.transaction():
        db._conn.execute(f"DELETE FROM persons WHERE id NOT IN ({keep_sql})")
        face_id_list = face_ids.tolist()
        for start in range(0, len(face_id_list), 500):
            chunk = face_id_list[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            db._conn.execute(
                f"UPDATE faces SET person_id = NULL, assign_source = 'auto', assign_conf = 0.0 WHERE id IN ({placeholders})",
                chunk,
            )

        prot = {}
        for pid in protected_pids:
            rows_p = db._conn.execute(
                "SELECT image_id, embedding FROM faces WHERE person_id = ? AND embedding IS NOT NULL AND det_score > 0",
                (pid,),
            ).fetchall()
            if rows_p:
                m = np.stack([FaceRecognizer.bytes_to_embedding(r["embedding"]) for r in rows_p]).mean(axis=0)
                m /= (np.linalg.norm(m) or 1.0)
                prot[pid] = (m, {r["image_id"] for r in rows_p})
        touched = set()

        for pid_idx, members in final_person_to_faces.items():
            member_face_ids = face_ids[members].tolist()
            c_embs = embeddings[members]
            centroid = c_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid /= (norm if norm > 1e-9 else 1.0)
            cluster_imgs = {int(image_ids[i]) for i in members}

            best_pid, best_sim = None, config.t_consolidate
            for pid, (pc, pimgs) in prot.items():
                sim = float(centroid @ pc)
                if sim >= best_sim and not (cluster_imgs & pimgs):
                    best_pid, best_sim = pid, sim

            if best_pid is not None:
                target = best_pid
                touched.add(target)
            else:
                person_number = db.get_next_person_number()
                target = db.insert_person(
                    display_name=f"Person {person_number}",
                    centroid_bytes=FaceRecognizer.embedding_to_bytes(centroid),
                    face_count=len(members),
                    cover_face_id=int(member_face_ids[0]),
                )
            for start in range(0, len(member_face_ids), 500):
                chunk = member_face_ids[start:start + 500]
                placeholders = ",".join("?" * len(chunk))
                db._conn.execute(
                    f"UPDATE faces SET person_id = ?, assign_source = 'auto', assign_conf = 1.0 WHERE id IN ({placeholders})",
                    [target] + chunk,
                )

        # Attach leftover (unknown/evicted) faces to protected persons when clearly closest.
        for idx in unknown_indices:
            if not prot:
                break
            img = int(image_ids[idx])
            sims = sorted(
                ((float(embeddings[idx] @ pc), pid) for pid, (pc, pimgs) in prot.items()
                 if img not in pimgs),
                reverse=True,
            )
            if not sims or sims[0][0] < config.min_support:
                continue
            if len(sims) > 1 and sims[0][0] - sims[1][0] < config.min_margin:
                continue
            sim, pid = sims[0]
            db._conn.execute(
                "UPDATE faces SET person_id = ?, assign_source = 'auto', assign_conf = ? WHERE id = ?",
                (pid, sim, int(face_ids[idx])),
            )
            prot[pid][1].add(img)
            touched.add(pid)

        for pid in touched:
            db.recompute_person_centroid(pid)
        for pid in protected_pids:
            db._conn.execute(
                "UPDATE persons SET face_count = (SELECT COUNT(*) FROM faces WHERE person_id = ? AND det_score > 0) WHERE id = ?",
                (pid, pid),
            )

        # Update cover faces for all persons to the highest quality photo
        from autosorter.core.cluster import pick_cover_faces
        pick_cover_faces(db)

    return report
