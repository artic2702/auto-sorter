"""Detection stage: unified thread pool, bounded submission, retryable errors, doc scoring."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autosorter.core import pipeline_worker as pw
from autosorter.core.db import Database


def _summary() -> dict:
    return {"faces_detected": 0, "faces_embedded": 0, "errors": 0}


def _fake_face(i: int) -> dict:
    emb = np.zeros(128, dtype=np.float32)
    emb[i % 128] = 1.0
    return {
        "bbox": (1, 2, 30, 40), "confidence": 0.99, "landmarks_json": "[]",
        "quality": 0.8, "aligned_face": np.zeros((112, 112, 3), np.uint8), "embedding": emb,
    }


@pytest.fixture
def env(tmp_path: Path):
    thumb = tmp_path / "t.png"
    Image.new("RGB", (32, 32), "white").save(thumb)
    db = Database(tmp_path / "data" / "library.db")
    ids = {}
    for name in ["good1", "good2", "good3", "empty", "none", "boom"]:
        i = db.upsert_image(f"/{name}.jpg", 1, 1.0, status="pending")
        db.update_thumb(i, str(thumb), 32, 32)
        db.update_status(i, "processed")
        ids[name] = i
    worker = pw.PipelineWorker(tmp_path / "photos", tmp_path / "data")
    worker._worker_count = 3
    yield db, worker, ids
    db.close()


def _install_fake(monkeypatch):
    def fake(image_id, path, max_dim):
        name = Path(path).stem
        if name == "boom":
            raise RuntimeError("worker crashed")
        if name == "none":
            return None
        if name == "empty":
            return []
        return [_fake_face(image_id), _fake_face(image_id + 1)]
    monkeypatch.setattr(pw, "_detect_one_image", fake)
    monkeypatch.setattr(pw, "_preflight_models", lambda: None)


def test_detect_stage_outcomes(env, monkeypatch):
    db, worker, ids = env
    _install_fake(monkeypatch)
    summary = _summary()
    worker._run_detect_stage(db, summary)

    for name in ("good1", "good2", "good3"):
        faces = [f for f in db.get_faces_for_image(ids[name]) if f["det_score"] > 0]
        assert len(faces) == 2
        assert all(f["embedding"] is not None and f["thumb_path"] for f in faces)
        assert all(Path(f["thumb_path"]).is_file() for f in faces)
    assert summary["faces_detected"] == 6 and summary["faces_embedded"] == 6
    assert summary["errors"] == 2

    # no faces -> sentinel, never re-queued
    assert db.get_image_by_path("/empty.jpg")["status"] == "processed"
    # failures -> retryable errors, NOT a permanent "no faces" sentinel
    for name in ("none", "boom"):
        row = db.get_image_by_path(f"/{name}.jpg")
        assert row["status"] == "error" and row["error_count"] == 1
        assert db.get_faces_for_image(ids[name]) == []
    assert db.count_images_needing_detection() == 0


def test_cancel_stops_new_work_and_leaves_consistent_state(env, monkeypatch):
    db, worker, ids = env

    def fake(image_id, path, max_dim):
        worker._cancel = True  # user presses Cancel while the first image is running
        return [_fake_face(image_id)]
    monkeypatch.setattr(pw, "_detect_one_image", fake)
    monkeypatch.setattr(pw, "_preflight_models", lambda: None)
    worker._worker_count = 1
    worker._run_detect_stage(db, _summary())
    assert 1 <= db.count_faces() < 6
    for img in db.iter_images_needing_detection():  # still-queued images have no partial faces
        assert db.get_faces_for_image(img["id"]) == []


def test_legacy_doc_scores_are_rescored_once(env, monkeypatch):
    db, worker, ids = env
    _install_fake(monkeypatch)
    worker._run_detect_stage(db, _summary())
    calls = []
    monkeypatch.setattr(
        pw, "classify_image",
        lambda **kw: calls.append(kw["face_count"]) or 0.25,
    )
    db.set_meta("doc_score_version", "1")  # simulate a DB scored by the old code
    assert worker._rescore_legacy_doc_scores(db) is True
    assert sorted(calls) == [0, 2, 2, 2]  # 3 images with faces + the no-face image
    n = len(calls)
    assert worker._rescore_legacy_doc_scores(db) is True
    assert len(calls) == n  # flag set -> not repeated
    assert db.get_meta("doc_score_version") == "2"


def test_missing_models_abort_instead_of_marking_every_image_failed(env, monkeypatch):
    db, worker, ids = env

    def no_models():
        raise FileNotFoundError("model missing")
    monkeypatch.setattr(pw, "_preflight_models", no_models)
    with pytest.raises(FileNotFoundError):
        worker._run_detect_stage(db, _summary())
    for name in ids:
        assert db.get_image_by_path(f"/{name}.jpg")["error_count"] == 0
