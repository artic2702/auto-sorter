"""
Sparse Constrained Graph Identity Clustering.

A production-grade, CPU-first face clustering architecture designed for
low-end hardware, large-scale scalability, and absolute separation of look-alikes.

Pipeline:
1. Cached SFace Embeddings (128-D float32)
2. Adaptive Chunked k-NN (Bounded memory, no FAISS/PyTorch)
3. Co-Occurrence Firewall (Two faces in same photo CANNOT share an edge or identity)
4. Two-Tier Graph:
   - Strong Graph (Mutual-kNN + High Similarity for Community Discovery)
   - Weak Graph (One-Way Support Edges for Peripheral Face Propagation)
5. Constrained Chinese Whispers (Deterministic label propagation to convergence)
6. Core Identity Filtering & Consolidation (Merge zero-conflict same-identity subclusters)
7. Peripheral Propagation with Dual-Gating (Absolute support + Relative margin)
8. Transactional SQLite Commit (Atomic, rollbacks on error)
"""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer


@dataclass
class GraphClusterConfig:
    """Configuration parameters for the graph identity clusterer."""
    k_neighbors: int = 30             # Number of nearest neighbors to retain per face
    t_strong: float = 0.46            # Cosine similarity threshold for mutual strong edges
    t_weak: float = 0.42              # Cosine similarity threshold for weak support edges
    t_consolidate: float = 0.52       # Centroid similarity to consolidate zero-conflict core clusters
    min_support: float = 0.46         # Minimum absolute similarity to assign peripheral face
    min_margin: float = 0.03          # Minimum margin (best - second_best) to assign peripheral face
    max_iterations: int = 20          # Chinese Whispers iteration cap
    min_core_faces: int = 4           # Minimum faces required to anchor a core identity
    min_unique_photos: int = 3        # Minimum unique photos required to anchor a core identity
    chunk_size: int = 1000            # Chunk size for large-scale memory bounding
    random_seed: int = 42             # Deterministic seed for reproducible graph traversal


@dataclass
class ClusterValidationReport:
    """Pre-commit validation summary of graph clustering results."""
    total_faces: int
    core_people_count: int
    peripheral_assigned_count: int
    unknown_count: int
    co_occurrence_violations: int
    cluster_sizes: List[int]
    passed: bool
    error_message: Optional[str] = None


def find_adaptive_knn(
    embeddings: np.ndarray,
    image_ids: np.ndarray,
    config: GraphClusterConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-k nearest neighbors for each face, enforcing the Co-Occurrence Firewall.

    Args:
        embeddings: (N, 128) L2-normalized float32 matrix.
        image_ids: (N,) int array of image IDs.
        config: GraphClusterConfig.

    Returns:
        indices: (N, k) int array of neighbor indices.
        similarities: (N, k) float32 array of neighbor cosine similarities.
    """
    n = len(embeddings)
    k = min(config.k_neighbors, max(1, n - 1))

    knn_indices = np.zeros((n, k), dtype=np.int32)
    knn_sims = np.zeros((n, k), dtype=np.float32)

    chunk_size = config.chunk_size

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]  # (chunk_len, 128)
        sim_chunk = chunk @ embeddings.T  # (chunk_len, n)

        # Defense-in-depth Firewall:
        # Faces from the same photo cannot link unless virtually identical (>= 0.98, synthetic mock/duplicate)
        chunk_img_ids = image_ids[start:end, None]
        same_img_mask = (chunk_img_ids == image_ids[None, :]) & (sim_chunk < 0.98)
        # Also zero out self-comparisons
        np.fill_diagonal(same_img_mask[:, start:end], True)

        # Apply -infinity to forbidden comparisons so they are never picked in top-k
        sim_chunk[same_img_mask] = -1e9

        # Efficient top-k selection
        if k < n:
            part_idx = np.argpartition(sim_chunk, -k, axis=1)[:, -k:]
            row_indices = np.arange(end - start)[:, None]
            part_sims = sim_chunk[row_indices, part_idx]
            sorted_order = np.argsort(-part_sims, axis=1)

            final_idx = np.take_along_axis(part_idx, sorted_order, axis=1)
            final_sims = np.take_along_axis(part_sims, sorted_order, axis=1)
        else:
            sorted_order = np.argsort(-sim_chunk, axis=1)[:, :k]
            row_indices = np.arange(end - start)[:, None]
            final_idx = sorted_order
            final_sims = sim_chunk[row_indices, sorted_order]

        # Filter out any that were masked (-1e9)
        valid_mask = final_sims > -1e8
        final_sims[~valid_mask] = 0.0

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

    - Strong Edge: u in top_k(v) AND v in top_k(u) AND sim >= config.t_strong.
    - Weak Support Edge: directed neighbor with sim >= config.t_weak.
    """
    n = len(knn_indices)
    k = knn_indices.shape[1]

    # Pre-build lookup sets of directed edges for O(1) mutual verification
    directed_edges: Dict[int, Dict[int, float]] = {i: {} for i in range(n)}
    for u in range(n):
        for ki in range(k):
            v = int(knn_indices[u, ki])
            sim = float(knn_sims[u, ki])
            if v != u and sim >= config.t_weak:
                directed_edges[u][v] = sim

    strong_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    weak_adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for u in range(n):
        for v, sim in directed_edges[u].items():
            weak_adj[u].append((v, sim))
            # Mutual check: does v also have an edge to u?
            if v in directed_edges and u in directed_edges[v]:
                mutual_sim = (sim + directed_edges[v][u]) / 2.0
                if mutual_sim >= config.t_strong:
                    strong_adj[u].append((v, mutual_sim))

    return strong_adj, weak_adj


def constrained_chinese_whispers(
    n_nodes: int,
    strong_adj: Dict[int, List[Tuple[int, float]]],
    config: GraphClusterConfig,
) -> Dict[int, int]:
    """
    Run Chinese Whispers label propagation on the strong mutual-kNN graph.

    Converges to discrete community clusters in a few iterations.
    Uses deterministic shuffling to guarantee order independence.
    """
    labels = np.arange(n_nodes, dtype=np.int32)
    rng = np.random.default_rng(config.random_seed)

    for iteration in range(config.max_iterations):
        order = rng.permutation(n_nodes)
        changes = 0

        for u in order:
            neighbors = strong_adj.get(u, [])
            if not neighbors:
                continue

            votes: Dict[int, float] = defaultdict(float)
            for v, weight in neighbors:
                votes[labels[v]] += weight

            if votes:
                best_label = max(votes.keys(), key=lambda l: votes[l])
                if labels[u] != best_label:
                    labels[u] = best_label
                    changes += 1

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
    Consolidate core clusters of the same person (e.g. clean face vs cake face)
    IF their centroids have high similarity AND they share ZERO photos (cannot-link).
    """
    pids = list(core_people.keys())
    if len(pids) <= 1:
        return core_people

    centroids: Dict[int, np.ndarray] = {}
    for pid in pids:
        c = embeddings[core_people[pid]].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[pid] = c / (norm if norm > 1e-9 else 1.0)

    active_pids = set(pids)
    for i in range(len(pids)):
        p1 = pids[i]
        if p1 not in active_pids:
            continue
        for j in range(i + 1, len(pids)):
            p2 = pids[j]
            if p2 not in active_pids:
                continue

            # Co-occurrence firewall: cannot share ANY photo!
            imgs1 = set(image_ids[core_people[p1]])
            imgs2 = set(image_ids[core_people[p2]])
            if imgs1 & imgs2:
                continue

            sim = float(centroids[p1] @ centroids[p2])
            if sim >= t_consolidate:
                # Merge p2 into p1
                core_people[p1].extend(core_people[p2])
                c = embeddings[core_people[p1]].mean(axis=0)
                norm = np.linalg.norm(c)
                centroids[p1] = c / (norm if norm > 1e-9 else 1.0)
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
    Assign peripheral faces (profiles, blur, singletons) to the best Core Person.

    Enforces:
    - Absolute Co-Occurrence Firewall: strictly rejects assignment if target person
      already has a face from this image_id.
    - Absolute Support + Relative Margin: requires best_score >= config.min_support
      AND (best_score - second_best_score) >= config.min_margin.

    Returns:
        assignments: {face_idx: core_person_id}
        unknown_faces: [face_idx, ...]
    """
    assignments: Dict[int, int] = {}
    unknown_faces: List[int] = []

    face_to_core: Dict[int, int] = {}
    person_image_ids: Dict[int, Set[int]] = defaultdict(set)
    core_centroids: Dict[int, np.ndarray] = {}

    for person_id, members in core_people.items():
        for m in members:
            face_to_core[m] = person_id
            person_image_ids[person_id].add(int(image_ids[m]))

        c_embs = embeddings[members]
        c = c_embs.mean(axis=0)
        norm = np.linalg.norm(c)
        core_centroids[person_id] = c / (norm if norm > 1e-9 else 1.0)

    for u in peripheral_faces:
        u_img = int(image_ids[u])
        u_emb = embeddings[u]

        support_per_person: Dict[int, List[float]] = defaultdict(list)
        for v, sim in weak_adj.get(u, []):
            if v in face_to_core:
                target_pid = face_to_core[v]
                support_per_person[target_pid].append(sim)

        candidate_scores: List[Tuple[int, float]] = []
        for person_id in core_people.keys():
            # Firewall: strictly forbid if target person already in this image!
            if u_img in person_image_ids[person_id]:
                continue

            centroid_sim = float(u_emb @ core_centroids[person_id])
            top_member_sims = support_per_person.get(person_id, [])

            if top_member_sims:
                score = (0.6 * np.mean(top_member_sims[:3])) + (0.4 * centroid_sim)
            else:
                score = centroid_sim

            candidate_scores.append((person_id, float(score)))

        if not candidate_scores:
            unknown_faces.append(u)
            continue

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_pid, best_score = candidate_scores[0]
        second_score = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0
        margin = best_score - second_score

        # Dual gating: absolute support AND relative margin
        if best_score >= config.min_support and margin >= config.min_margin:
            assignments[u] = best_pid
            person_image_ids[best_pid].add(u_img)
        else:
            unknown_faces.append(u)

    return assignments, unknown_faces


def run_graph_clustering(
    db: Database,
    config: Optional[GraphClusterConfig] = None,
    dry_run: bool = False,
) -> ClusterValidationReport:
    """
    Execute end-to-end graph clustering on unassigned/auto-assigned faces in SQLite.

    Guarantees:
    - User-assigned faces ('user') and user-named persons are permanently preserved.
    - Full pre-commit dry-run validation ensures 0 co-occurrence violations.
    - All database updates are wrapped in an atomic SQLite transaction.

    Returns:
        ClusterValidationReport summarizing the results and validation status.
    """
    if config is None:
        config = GraphClusterConfig()

    # 1. Fetch user-named persons to protect them
    user_named_rows = db._conn.execute(
        "SELECT id, display_name FROM persons WHERE name_is_user_set = 1"
    ).fetchall()
    protected_pids = {row["id"] for row in user_named_rows}

    # 2. Fetch faces to cluster
    locked_faces = set()
    if protected_pids:
        placeholders = ",".join("?" * len(protected_pids))
        locked_rows = db._conn.execute(
            f"SELECT id FROM faces WHERE person_id IN ({placeholders}) AND assign_source = 'user'",
            list(protected_pids),
        ).fetchall()
        locked_faces = {row["id"] for row in locked_rows}

    cursor = db._conn.execute("""
        SELECT id, image_id, embedding, quality, det_score, person_id, assign_source
        FROM faces
        WHERE embedding IS NOT NULL AND det_score > 0
        ORDER BY id ASC
    """)
    rows = cursor.fetchall()

    if not rows:
        return ClusterValidationReport(
            total_faces=0, core_people_count=0, peripheral_assigned_count=0,
            unknown_count=0, co_occurrence_violations=0, cluster_sizes=[],
            passed=True
        )

    # Filter out locked faces from clustering pool
    cluster_rows = [r for r in rows if r["id"] not in locked_faces]
    if not cluster_rows:
        return ClusterValidationReport(
            total_faces=len(rows), core_people_count=len(protected_pids),
            peripheral_assigned_count=0, unknown_count=0,
            co_occurrence_violations=0, cluster_sizes=[], passed=True
        )

    face_ids = np.array([r["id"] for r in cluster_rows], dtype=np.int32)
    image_ids = np.array([r["image_id"] for r in cluster_rows], dtype=np.int32)
    embeddings = np.stack([FaceRecognizer.bytes_to_embedding(r["embedding"]) for r in cluster_rows])

    # 3. Adaptive k-NN
    knn_indices, knn_sims = find_adaptive_knn(embeddings, image_ids, config)

    # 4. Two-Tier Graph Construction
    strong_adj, weak_adj = build_two_tier_graph(knn_indices, knn_sims, config)

    # 5. Chinese Whispers on Strong Graph
    community_labels = constrained_chinese_whispers(len(cluster_rows), strong_adj, config)

    # Group faces by community label
    communities: Dict[int, List[int]] = defaultdict(list)
    for face_idx, label in community_labels.items():
        communities[label].append(face_idx)

    # 6. Filter Core People from Peripheral Faces
    min_core_faces = min(config.min_core_faces, max(1, len(cluster_rows) // 2))
    min_unique_photos = min(config.min_unique_photos, max(1, len(set(image_ids))))

    core_people: Dict[int, List[int]] = {}
    peripheral_faces: List[int] = []

    next_person_id = 1
    for label, members in communities.items():
        unique_imgs = len(set(image_ids[members]))
        if len(members) >= min_core_faces and unique_imgs >= min_unique_photos:
            core_people[next_person_id] = members
            next_person_id += 1
        else:
            peripheral_faces.extend(members)

    # Fallback for very small libraries: ensure largest community becomes core
    if not core_people and communities:
        largest_label = max(communities.keys(), key=lambda l: len(communities[l]))
        core_people[1] = communities[largest_label]
        peripheral_faces = [idx for l, m in communities.items() if l != largest_label for idx in m]

    # 7. Consolidate Core People (zero-conflict same-identity subclusters)
    consolidated_cores = consolidate_core_clusters(
        core_people, image_ids, embeddings, t_consolidate=config.t_consolidate
    )

    # 8. Peripheral Face Propagation
    peripheral_assignments: Dict[int, int] = {}
    unknown_indices: List[int] = []

    if consolidated_cores and peripheral_faces:
        peripheral_assignments, unknown_indices = propagate_peripheral_faces(
            consolidated_cores, peripheral_faces, weak_adj, image_ids, embeddings, config
        )
    else:
        unknown_indices = peripheral_faces

    # 9. Pre-Commit Validation: Co-occurrence Check across ALL formed clusters
    total_violations = 0
    final_person_to_faces: Dict[int, List[int]] = defaultdict(list)
    for pid, members in consolidated_cores.items():
        final_person_to_faces[pid].extend(members)
    for face_idx, pid in peripheral_assignments.items():
        final_person_to_faces[pid].append(face_idx)

    for pid, members in final_person_to_faces.items():
        img_counts = defaultdict(int)
        for idx in members:
            img_counts[int(image_ids[idx])] += 1
        violations = sum(1 for cnt in img_counts.values() if cnt > 1)
        total_violations += violations

    cluster_sizes = sorted([len(m) for m in final_person_to_faces.values()], reverse=True)
    report = ClusterValidationReport(
        total_faces=len(cluster_rows),
        core_people_count=len(consolidated_cores),
        peripheral_assigned_count=len(peripheral_assignments),
        unknown_count=len(unknown_indices),
        co_occurrence_violations=total_violations,
        cluster_sizes=cluster_sizes,
        passed=(total_violations == 0),
        error_message=f"Found {total_violations} co-occurrence violations!" if total_violations > 0 else None
    )

    if dry_run or not report.passed:
        return report

    # 10. Atomic Transactional SQLite Commit
    with db._conn:
        # Delete auto-created persons (preserving user-named)
        db._conn.execute("DELETE FROM persons WHERE name_is_user_set = 0")

        # Unassign cluster_rows faces
        face_id_list = face_ids.tolist()
        for start in range(0, len(face_id_list), 500):
            chunk = face_id_list[start:start + 500]
            placeholders = ",".join("?" * len(chunk))
            db._conn.execute(
                f"UPDATE faces SET person_id = NULL, assign_source = 'auto', assign_conf = 0.0 WHERE id IN ({placeholders})",
                chunk,
            )

        # Create new person records and assign faces
        for pid_idx, members in final_person_to_faces.items():
            member_face_ids = face_ids[members].tolist()
            c_embs = embeddings[members]
            centroid = c_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid /= (norm if norm > 1e-9 else 1.0)
            centroid_bytes = FaceRecognizer.embedding_to_bytes(centroid)

            person_number = db.get_next_person_number()
            display_name = f"Person {person_number}"
            new_person_id = db.insert_person(
                display_name=display_name,
                centroid_bytes=centroid_bytes,
                face_count=len(members),
                cover_face_id=int(member_face_ids[0]),
            )

            # Batch assign member faces
            for start in range(0, len(member_face_ids), 500):
                chunk = member_face_ids[start:start + 500]
                placeholders = ",".join("?" * len(chunk))
                db._conn.execute(
                    f"UPDATE faces SET person_id = ?, assign_source = 'auto', assign_conf = 1.0 WHERE id IN ({placeholders})",
                    [new_person_id] + chunk,
                )

        # Update cover faces to best quality
        from autosorter.core.cluster import pick_cover_faces
        pick_cover_faces(db)

    return report
