"""Scan-stage failures are recorded as retryable errors and re-queued on the next run."""
from pathlib import Path

from PIL import Image

from autosorter.core.db import Database
from autosorter.core.pipeline_worker import PipelineWorker


def _summary() -> dict:
    return {"files_found": 0, "files_new": 0, "files_skipped": 0,
            "files_reattached": 0, "files_missing": 0, "files_requeued": 0,
            "errors": 0}


def test_corrupt_image_is_retried_then_given_up(tmp_path: Path):
    photos, data = tmp_path / "photos", tmp_path / "data"
    photos.mkdir()
    Image.new("RGB", (64, 64), "white").save(photos / "ok.png")
    (photos / "bad.jpg").write_bytes(b"not a real jpeg")
    db = Database(data / "library.db")
    requeued = []
    for _ in range(4):
        summary = _summary()
        PipelineWorker(photos, data)._run_scan_stage(db, summary)
        requeued.append(summary["files_requeued"])
    bad = db.get_image_by_path(str(photos / "bad.jpg"))
    good = db.get_image_by_path(str(photos / "ok.png"))
    assert good["status"] == "processed"
    assert bad["status"] == "error" and bad["error_count"] == 3
    assert requeued == [0, 1, 1, 0]
    db.close()
