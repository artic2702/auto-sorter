"""Regression tests: failed images are retried a bounded number of times."""
from pathlib import Path

from autosorter.core.db import Database


def test_error_is_retried_until_max_attempts(tmp_path: Path):
    db = Database(tmp_path / "t.db")
    i = db.upsert_image("/a.jpg", 1, 1.0, status="processed")
    seen = []
    for _ in range(4):
        db.mark_image_error(i, "detect failed")
        seen.append(db.requeue_failed_images(max_attempts=3))
    row = db.get_image_by_path("/a.jpg")
    assert seen == [1, 1, 0, 0]
    assert row["status"] == "error" and row["error_count"] == 4
    db.close()
