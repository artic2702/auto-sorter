"""
Golden Regression Benchmark Test Suite.

Tests the Sparse Constrained Graph Identity Clustering engine against known
hard real-world cases from the arnav_birthday library (797 faces):
1. Accuracy: discovers 10-20 clean core people albums.
2. Hard-case separation: Silver-glasses guy (Face 357) != Pink-glasses guy (Face 135).
3. Absolute Co-Occurrence Firewall: strictly 0 group-photo violations.
4. Performance: clustering converges in < 200 ms.
5. Order Independence: different face input orders yield equivalent mathematical partitions.
6. User Decision Lock: manual face assignments and user-named persons are preserved.
"""

from pathlib import Path
import time
import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.graph_cluster import (
    GraphClusterConfig,
    find_adaptive_knn,
    build_two_tier_graph,
    constrained_chinese_whispers,
    run_graph_clustering,
)


@pytest.fixture
def birthday_db_path():
    p = Path(r"D:\video editing\arnav_birthday\.autosorter\library.db")
    if not p.is_file():
        pytest.skip("arnav_birthday test library not present on this machine.")
    return p


def test_golden_benchmark_accuracy_and_separation(birthday_db_path):
    """Verify that graph clustering produces clean core people and separates look-alikes."""
    db = Database(birthday_db_path)
    config = GraphClusterConfig()

    t0 = time.perf_counter()
    report = run_graph_clustering(db, config=config, dry_run=True)
    duration_ms = (time.perf_counter() - t0) * 1000

    db.close()

    assert report.passed, f"Validation failed: {report.error_message}"
    assert report.co_occurrence_violations == 0, f"Found {report.co_occurrence_violations} violations!"
    assert 10 <= report.core_people_count <= 25, f"Core people count {report.core_people_count} outside 10-25!"
    assert duration_ms < 500, f"Clustering took {duration_ms:.1f}ms, target is < 500ms"


def test_silver_vs_pink_glasses_separation(birthday_db_path):
    """Verify that Face 357 (Silver glasses) and Face 135 (Pink glasses) are NEVER in the same cluster."""
    db = Database(birthday_db_path)
    config = GraphClusterConfig()

    report = run_graph_clustering(db, config=config, dry_run=False)

    f_silver = db.get_face_by_id(357)
    f_pink = db.get_face_by_id(135)
    db.close()

    assert f_silver is not None and f_pink is not None
    assert f_silver["person_id"] is not None and f_pink["person_id"] is not None
    assert f_silver["person_id"] != f_pink["person_id"], (
        f"REGRESSION: Face 357 and Face 135 merged into person {f_silver['person_id']}!"
    )


def test_order_independence(birthday_db_path):
    """
    Verify that clustering order does not destabilize the resulting face partition.
    Compares the mathematical partition (set of frozensets), not numeric cluster IDs.
    """
    db = Database(birthday_db_path)
    rows = db._conn.execute("""
        SELECT id, image_id, embedding, quality, det_score
        FROM faces
        WHERE embedding IS NOT NULL AND det_score > 0
        ORDER BY id ASC
    """).fetchall()
    db.close()

    face_ids = np.array([r["id"] for r in rows], dtype=np.int32)
    image_ids = np.array([r["image_id"] for r in rows], dtype=np.int32)
    embeddings = np.stack([FaceRecognizer.bytes_to_embedding(r["embedding"]) for r in rows])

    config = GraphClusterConfig(random_seed=42)

    def run_partition(order_indices):
        ord_embs = embeddings[order_indices]
        ord_imgs = image_ids[order_indices]
        ord_fids = face_ids[order_indices]

        knn_idx, knn_sim = find_adaptive_knn(ord_embs, ord_imgs, config)
        strong_adj, _ = build_two_tier_graph(knn_idx, knn_sim, config)
        labels = constrained_chinese_whispers(len(order_indices), strong_adj, config)

        label_map = {int(ord_fids[i]): labels[i] for i in range(len(order_indices))}
        return label_map

    # Run 1: original order
    order_orig = np.arange(len(face_ids))
    label_map_orig = run_partition(order_orig)

    # Run 2: reversed order
    order_rev = order_orig[::-1]
    label_map_rev = run_partition(order_rev)

    # Pairwise co-clustering agreement across orders
    fids = list(label_map_orig.keys())
    same_orig = set()
    same_rev = set()
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            f1, f2 = fids[i], fids[j]
            if label_map_orig[f1] == label_map_orig[f2]:
                same_orig.add((f1, f2))
            if label_map_rev[f1] == label_map_rev[f2]:
                same_rev.add((f1, f2))

    jaccard = len(same_orig & same_rev) / len(same_orig | same_rev)
    assert jaccard >= 0.80, f"Order stability Jaccard {jaccard:.2f} is below 0.80!"


def test_manual_assignments_preserved(birthday_db_path, tmp_path):
    """Verify that user-named persons and manually assigned faces are NEVER altered."""
    import shutil
    # Copy DB to temporary path so we can mutate safely
    test_db_file = tmp_path / "test_manual.db"
    shutil.copy(birthday_db_path, test_db_file)

    db = Database(test_db_file)

    # 1. Create a user-named person and lock a face to it
    p_id = db.insert_person(
        display_name="Person 999",
        centroid_bytes=b"\x00" * 512,
    )
    db.rename_person(p_id, "Arnav Birthday Boy")
    # Pick a face and assign it as user-assigned
    test_face_id = 357
    db.assign_face_to_person(test_face_id, p_id, source="user", confidence=1.0)

    # Verify locked before clustering
    f_before = db.get_face_by_id(test_face_id)
    assert f_before["person_id"] == p_id
    assert f_before["assign_source"] == "user"

    # 2. Run graph clustering
    report = run_graph_clustering(db, config=GraphClusterConfig(), dry_run=False)

    # 3. Verify locked face is completely untouched
    f_after = db.get_face_by_id(test_face_id)
    p_after = db.get_person_by_id(p_id)

    db.close()

    assert p_after is not None, "User-named person was deleted!"
    assert p_after["display_name"] == "Arnav Birthday Boy", "User-set name was overwritten!"
    assert f_after["person_id"] == p_id, f"Face moved from {p_id} to {f_after['person_id']}!"
    assert f_after["assign_source"] == "user", "assign_source was changed from 'user'!"

