"""Regression tests: schema migrations, upsert contract, stale-face invalidation."""
import sqlite3
from pathlib import Path

from autosorter.core.db import Database


def _uv(db):
    return db._conn.execute("PRAGMA user_version").fetchone()[0]


def _person_with_face(db, image_id, name=None):
    p = db.insert_person("Person X", b"\x00" * 512)
    if name:
        db.rename_person(p, name)
    f = db.insert_face(image_id, 1, 1, 1, 1, 0.9, quality=0.5)
    db.update_embedding(f, b"\x00" * 512)
    db.assign_face_to_person(f, p, "user" if name else "auto", 1.0)
    return p, f


def test_fresh_db_is_latest_version(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    assert _uv(db) >= 2
    assert "error_count" in {r[1] for r in db._conn.execute("PRAGMA table_info(images)")}
    db.close()


def test_legacy_db_is_upgraded_and_reopen_is_idempotent(tmp_path: Path):
    path = tmp_path / "t.db"
    Database(path).close()
    raw = sqlite3.connect(path)
    raw.execute("ALTER TABLE images DROP COLUMN error_count")
    raw.execute("PRAGMA user_version = 0")
    raw.commit()
    raw.close()
    db = Database(path)
    assert "error_count" in {r[1] for r in db._conn.execute("PRAGMA table_info(images)")}
    v = _uv(db)
    db.close()
    db = Database(path)
    assert _uv(db) == v
    db.close()


def test_rescan_with_changed_mtime_keeps_faces_and_assignments(tmp_path: Path):
    """The Milestone-6 contract: re-scan must never destroy faces or person data."""
    db = Database(tmp_path / "t.db")
    i = db.upsert_image("/a.jpg", 1000, 1000.0, status="processed")
    p, f = _person_with_face(db, i, "Mom")
    i2 = db.upsert_image("/a.jpg", 1000, 2000.0)
    assert i2 == i
    assert db.get_face_by_id(f)["person_id"] == p
    assert db.get_image_by_path("/a.jpg")["status"] == "pending"
    db.close()


def test_upsert_returns_the_existing_row_id(tmp_path: Path):
    """lastrowid is unreliable after ON CONFLICT DO UPDATE; the id must be looked up."""
    db = Database(tmp_path / "t.db")
    a = db.upsert_image("/a.jpg", 1, 1.0)
    db.upsert_image("/b.jpg", 1, 1.0)
    assert db.upsert_image("/a.jpg", 2, 2.0) == a
    db.close()


def test_upsert_resets_error_count(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    i = db.upsert_image("/a.jpg", 1, 1.0)
    db._conn.execute("UPDATE images SET error_count = 2 WHERE id = ?", (i,))
    db.upsert_image("/a.jpg", 2, 2.0)
    assert db.get_image_by_path("/a.jpg")["error_count"] == 0
    db.close()


def test_same_dimensions_keep_faces_when_thumbnail_regenerated(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    i = db.upsert_image("/a.jpg", 1, 1.0, status="processed")
    db.update_thumb(i, "t.jpg", 100, 80)
    _, f = _person_with_face(db, i, "Mom")
    db.update_thumb(i, "t2.jpg", 100, 80)
    assert db.get_face_by_id(f) is not None
    db.close()


def test_changed_dimensions_drop_stale_faces_and_fix_persons(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    i1 = db.upsert_image("/a.jpg", 1, 1.0, status="processed")
    i2 = db.upsert_image("/b.jpg", 1, 1.0, status="processed")
    db.update_thumb(i1, "t.jpg", 100, 80)
    p, f1 = _person_with_face(db, i1)
    f2 = db.insert_face(i2, 1, 1, 1, 1, 0.9, quality=0.5)
    db.update_embedding(f2, b"\x00" * 512)
    db.assign_face_to_person(f2, p, "auto", 1.0)
    db.update_thumb(i1, "t.jpg", 50, 40)  # file was cropped
    assert db.get_faces_for_image(i1) == []
    assert db.get_person_by_id(p)["face_count"] == 1
    db.close()


def test_user_named_person_survives_losing_all_faces(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    i = db.upsert_image("/a.jpg", 1, 1.0, status="processed")
    db.update_thumb(i, "t.jpg", 100, 80)
    p, _ = _person_with_face(db, i, "Mom")
    db.update_thumb(i, "t.jpg", 10, 10)
    person = db.get_person_by_id(p)
    assert person is not None and person["face_count"] == 0
    db.close()


def test_unnamed_empty_person_is_deleted(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    p = db.insert_person("Person 1", b"\x00" * 512)
    db.recompute_person_centroid(p)
    assert db.get_person_by_id(p) is None
    db.close()
