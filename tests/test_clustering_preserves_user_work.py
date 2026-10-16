"""Regression tests: re-running graph clustering must never undo user work."""
from pathlib import Path

import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.graph_cluster import run_graph_clustering


class Lib:
    """Tiny synthetic library: tight identities in 128-d, one image per face."""

    def __init__(self, tmp_path: Path):
        self.db = Database(tmp_path / "t.db")
        self.rng = np.random.default_rng(1)
        self.base = {k: self.rng.normal(size=128) for k in "ABC"}
        self.n = 0

    def add(self, who: str, image_id: int | None = None) -> int:
        v = self.base[who] + self.rng.normal(0, 0.02, 128)
        v = (v / np.linalg.norm(v)).astype(np.float32)
        if image_id is None:
            self.n += 1
            image_id = self.db.upsert_image(f"/img{self.n}.jpg", 1, 1.0, status="processed")
        fid = self.db.insert_face(image_id, 0, 0, 50, 50, 0.99, quality=0.9)
        self.db.update_embedding(fid, v.tobytes())
        return fid

    def person_of(self, face_id: int):
        return self.db.get_face_by_id(face_id)["person_id"]


@pytest.fixture
def lib(tmp_path: Path):
    lb = Lib(tmp_path)
    lb.a = [lb.add("A") for _ in range(8)]
    lb.b = [lb.add("B") for _ in range(8)]
    run_graph_clustering(lb.db)
    yield lb
    lb.db.close()


def test_rename_survives_rerun_and_new_photo_is_filed(lib):
    mom = lib.person_of(lib.a[0])
    lib.db.rename_person(mom, "Mom")
    new_face = lib.add("A")
    run_graph_clustering(lib.db)
    assert lib.db.get_person_by_id(mom)["display_name"] == "Mom"
    assert len(lib.db.get_faces_for_person(mom)) == 9
    assert lib.person_of(new_face) == mom
    assert lib.db.get_person_by_id(mom)["face_count"] == 9
    assert lib.db.count_persons() == 2  # no duplicate of Mom


def test_user_moved_face_stays_put(lib):
    target = lib.person_of(lib.b[0])
    lib.db.reassign_face(lib.a[1], target)
    run_graph_clustering(lib.db)
    row = lib.db.get_face_by_id(lib.a[1])
    assert row["person_id"] == target and row["assign_source"] == "user"


def test_not_this_person_stays_unassigned(lib):
    lib.db.unassign_face(lib.a[2])
    run_graph_clustering(lib.db)
    assert lib.person_of(lib.a[2]) is None


def test_stranger_is_not_attached_to_named_person(lib):
    mom = lib.person_of(lib.a[0])
    lib.db.rename_person(mom, "Mom")
    stranger = lib.add("C")
    run_graph_clustering(lib.db)
    assert lib.person_of(stranger) != mom


def test_same_person_never_twice_in_one_photo(lib):
    mom = lib.person_of(lib.a[0])
    lib.db.rename_person(mom, "Mom")
    img = lib.db.upsert_image("/pair.jpg", 1, 1.0, status="processed")
    lib.add("A", img)
    lib.add("A", img)
    run_graph_clustering(lib.db)
    dup = lib.db._conn.execute(
        "SELECT 1 FROM faces WHERE person_id IS NOT NULL "
        "GROUP BY person_id, image_id HAVING COUNT(*) > 1"
    ).fetchall()
    assert dup == []


def test_failure_during_write_rolls_back(lib, monkeypatch):
    for _ in range(6):
        lib.add("C")  # a new unprotected cluster forces insert_person
    before = (
        lib.db._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
        lib.db._conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL").fetchone()[0],
    )

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(lib.db, "insert_person", boom)
    with pytest.raises(RuntimeError):
        run_graph_clustering(lib.db)
    after = (
        lib.db._conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0],
        lib.db._conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL").fetchone()[0],
    )
    assert before == after


def test_person_numbers_never_collide(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    ids = [db.insert_person(f"Person {db.get_next_person_number()}", b"x") for _ in range(3)]
    db.delete_person(ids[0])
    new = db.insert_person(f"Person {db.get_next_person_number()}", b"x")
    names = [r["display_name"] for r in db.get_all_persons()]
    assert len(names) == len(set(names))
    assert db.get_person_by_id(new) is not None
    db.close()


def test_rerun_without_any_user_work_does_not_duplicate_people(lib):
    """Regression: with no protected persons, old auto persons must be replaced, not kept."""
    assert lib.db.count_persons() == 2
    for _ in range(3):
        run_graph_clustering(lib.db)
    assert lib.db.count_persons() == 2
