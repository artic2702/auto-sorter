"""Regression tests: Database.transaction() is atomic and nestable."""
from pathlib import Path

import pytest

from autosorter.core.db import Database


def _count(db: Database, table: str) -> int:
    return db._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_transaction_rolls_back_on_error(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.insert_person("A", b"x" * 8)
            db.insert_person("B", b"x" * 8)
            raise RuntimeError("boom")
    assert _count(db, "persons") == 0
    db.close()


def test_nested_transaction_rolls_back_everything(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.insert_person("A", b"x" * 8)
            with db.transaction():
                db.insert_person("B", b"x" * 8)
            raise RuntimeError("boom")
    assert _count(db, "persons") == 0
    db.close()


def test_transaction_commits_on_success(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    with db.transaction():
        db.insert_person("A", b"x" * 8)
    assert _count(db, "persons") == 1
    assert not db._conn.in_transaction
    db.close()


def test_calls_outside_transaction_still_autocommit(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    db.insert_person("A", b"x" * 8)
    assert not db._conn.in_transaction
    db.close()
