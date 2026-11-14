"""
Framework-agnostic background runners for Auto-Sorter's pipeline and export engine.

Provides pure-Python threading runners (zero Qt/PySide6 dependency) with
callback hooks for stage transitions, throttled progress reporting, completion,
and error handling. Used by the Electron JSON-RPC bridge (`autosorter.bridge`).
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

from autosorter.core.bounded import run_bounded
from autosorter.core.db import Database
from autosorter.core.doc_classifier import classify_image
from autosorter.core.export import export_library
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.graph_cluster import GraphClusterConfig, run_graph_clustering
from autosorter.core.perf import (
    SpeedMode,
    get_rss_bytes,
    is_memory_pressure,
    set_current_thread_below_normal,
    worker_count_for_mode,
)
from autosorter.core.scanner import scan_folder
from logging.handlers import RotatingFileHandler
from autosorter.core.thumbnailer import (
    _read_image_safe,
    compute_sha256,
    generate_thumbnail,
)

log = logging.getLogger(__name__)

_MEMORY_CHECK_INTERVAL = 10
_MEMORY_BACKPRESSURE_SLEEP = 0.5
_PROGRESS_MIN_INTERVAL = 0.06  # ~16 updates/sec max so IPC stays silky smooth

_thread_local = threading.local()


def _preflight_models() -> None:
    """Verify YuNet and SFace ONNX models load cleanly before spawning workers."""
    FaceDetector()
    FaceRecognizer()


def _get_thread_models():
    """Return thread-local (FaceDetector, FaceRecognizer) instances."""
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = FaceDetector()
        _thread_local.recognizer = FaceRecognizer()
    return _thread_local.detector, _thread_local.recognizer


def detect_one_image(
    image_id: int,
    image_path: str,
    max_dimension: int,
) -> Optional[List[dict]]:
    """
    Detect faces and compute SFace 128-D embeddings for a single image on a worker thread.
    """
    image = _read_image_safe(image_path)
    if image is None:
        return None

    detector, recognizer = _get_thread_models()
    faces = detector.detect(image, max_dimension=max_dimension)
    if len(faces) == 0:
        return []

    results = []
    for face in faces:
        aligned_sface = recognizer.align_crop(image, face.raw_detection)
        if aligned_sface is None:
            aligned_sface = align_face(image, face.landmarks)

        embedding = None
        if aligned_sface is not None:
            embedding = recognizer.compute(aligned_sface)

        results.append(
            {
                "bbox": face.bbox,
                "confidence": face.confidence,
                "landmarks_json": face.landmarks_json(),
                "quality": face.quality_score(),
                "aligned_face": aligned_sface,
                "embedding": embedding,
            }
        )
    return results


class PipelineRunner:
    """
    Runs the 4-stage Auto-Sorter pipeline (Scan -> Detect -> Embed -> Identify)
    on a background daemon thread with graceful cancellation and throttled events.
    """

    TOTAL_STAGES = 4

    def __init__(
        self,
        photo_folder: Path,
        data_folder: Path,
        *,
        speed_mode: SpeedMode = SpeedMode.NORMAL,
        max_dimension: int = 1280,
        on_stage: Optional[Callable[[str, int, int], None]] = None,
        on_progress: Optional[Callable[[int, int, str, int, str], None]] = None,
        on_finished: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._photo_folder = photo_folder
        self._data_folder = data_folder
        self._speed_mode = speed_mode
        self._max_dimension = max_dimension
        self._worker_count = worker_count_for_mode(speed_mode)

        self._on_stage = on_stage
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_error = on_error

        self._cancel = False
        self._running = False
        self._current_stage_num = 0
        self._current_stage_name = ""
        self._last_progress: dict = {
            "done": 0,
            "total": 0,
            "detail": "",
            "stage_num": 0,
            "stage_name": "",
        }
        self._last_emit_ts = 0.0
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def snapshot(self) -> dict:
        rss = get_rss_bytes()
        return {
            "running": self._running,
            "cancelling": self._cancel and self._running,
            "photo_folder": str(self._photo_folder),
            "data_folder": str(self._data_folder),
            "speed_mode": self._speed_mode.value,
            "worker_count": self._worker_count,
            "rss_mb": round(rss / (1024 * 1024), 1) if rss > 0 else 0,
            **self._last_progress,
        }

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Pipeline is already running")
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="autosorter-pipeline", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel = True

    def _emit_stage(self, stage_name: str, stage_num: int) -> None:
        self._current_stage_num = stage_num
        self._current_stage_name = stage_name
        self._last_progress = {
            "done": 0,
            "total": 0,
            "detail": f"{stage_name}…",
            "stage_num": stage_num,
            "stage_name": stage_name,
        }
        self._last_emit_ts = 0.0
        if self._on_stage:
            self._on_stage(stage_name, stage_num, self.TOTAL_STAGES)

    def _emit_progress(
        self, done: int, total: int, detail: str, *, force: bool = False
    ) -> None:
        now = time.monotonic()
        is_boundary = (done == 0) or (total > 0 and done >= total)
        if not force and not is_boundary and (now - self._last_emit_ts) < _PROGRESS_MIN_INTERVAL:
            self._last_progress = {
                "done": done,
                "total": total,
                "detail": detail,
                "stage_num": self._current_stage_num,
                "stage_name": self._current_stage_name,
            }
            return

        self._last_emit_ts = now
        self._last_progress = {
            "done": done,
            "total": total,
            "detail": detail,
            "stage_num": self._current_stage_num,
            "stage_name": self._current_stage_name,
        }
        if self._on_progress:
            self._on_progress(
                done,
                total,
                detail,
                self._current_stage_num,
                self._current_stage_name,
            )

    def _run(self) -> None:
        set_current_thread_below_normal()
        db: Optional[Database] = None
        file_handler: Optional[RotatingFileHandler] = None
        root_logger = logging.getLogger()
        try:
            self._data_folder.mkdir(parents=True, exist_ok=True)
            log_path = self._data_folder / "autosorter.log"
            file_handler = RotatingFileHandler(
                log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root_logger.addHandler(file_handler)
            db = Database(self._data_folder / "library.db")

            summary = {
                "files_found": 0,
                "files_new": 0,
                "files_skipped": 0,
                "files_reattached": 0,
                "files_missing": 0,
                "files_requeued": 0,
                "faces_detected": 0,
                "faces_embedded": 0,
                "faces_assigned": 0,
                "persons_found": 0,
                "persons_created": 0,
                "errors": 0,
                "cancelled": False,
                "worker_count": self._worker_count,
                "peak_rss_mb": 0,
            }

            if not self._cancel:
                self._run_scan_stage(db, summary)
            if not self._cancel:
                self._run_detect_stage(db, summary)
            if not self._cancel:
                self._run_embed_stage(db, summary)
            if not self._cancel:
                self._run_identity_stage(db, summary)

            summary["persons_created"] = summary["persons_found"]
            rss = get_rss_bytes()
            if rss > 0:
                summary["peak_rss_mb"] = round(rss / (1024 * 1024), 1)
            summary["cancelled"] = self._cancel

            self._running = False
            if self._on_finished:
                self._on_finished(summary)
        except Exception as exc:
            log.exception("Pipeline execution failed")
            self._running = False
            if self._on_error:
                self._on_error(str(exc))
        finally:
            self._running = False
            if db is not None:
                db.close()
            if file_handler is not None:
                root_logger.removeHandler(file_handler)
                file_handler.close()

    def _run_scan_stage(self, db: Database, summary: dict) -> None:
        self._emit_stage("Scanning files", 1)
        summary["files_requeued"] = db.requeue_failed_images(max_attempts=3)

        scan_result = scan_folder(
            root=self._photo_folder,
            db=db,
            progress_callback=lambda f, n, s: self._emit_progress(
                f, f, f"Discovering files: {f:,} found ({n:,} new, {s:,} unchanged)"
            ),
            cancel_check=lambda: self._cancel,
        )

        summary["files_found"] = scan_result.found
        summary["files_new"] = scan_result.new
        summary["files_skipped"] = scan_result.skipped
        summary["errors"] += scan_result.errors

        if self._cancel:
            return

        missing_count = db.mark_missing_images(scan_result.discovered_paths)
        summary["files_missing"] = missing_count

        if self._cancel:
            return

        thumbs_dir = self._data_folder / "thumbs"
        pending_count = db.count_images(status="pending")
        if pending_count == 0:
            self._emit_progress(
                scan_result.found,
                max(1, scan_result.found),
                f"Scan complete: {scan_result.found:,} files verified",
                force=True,
            )
            return

        done = 0
        self._emit_progress(0, pending_count, f"Generating thumbnails: 0 / {pending_count:,}", force=True)

        for row in db.iter_pending():
            if self._cancel:
                break

            image_id = row["id"]
            image_path = row["path"]

            sha256 = compute_sha256(image_path)
            if sha256:
                reattached_id = db.reattach_by_hash(sha256, image_path)
                if reattached_id is not None:
                    summary["files_reattached"] += 1
                    done += 1
                    self._emit_progress(
                        done,
                        pending_count,
                        f"Thumbnails: {done:,} / {pending_count:,} (reattached moved file)",
                    )
                    continue

            thumb_result = generate_thumbnail(image_path, thumbs_dir, image_id)
            if thumb_result is not None:
                thumb_path, width, height = thumb_result
                db.update_thumb(image_id, thumb_path, width, height)
                if sha256:
                    db.set_image_sha256(image_id, sha256)
                db.update_status(image_id, "processed")
            else:
                db.mark_image_error(image_id, "decode failed")
                summary["errors"] += 1

            done += 1
            self._emit_progress(
                done,
                pending_count,
                f"Thumbnails: {done:,} / {pending_count:,}",
            )

    def _run_detect_stage(self, db: Database, summary: dict) -> None:
        self._emit_stage("Detecting faces", 2)
        if not self._rescore_legacy_doc_scores(db):
            return

        rows = list(db.iter_images_needing_detection())
        total = len(rows)
        if total == 0:
            return

        _preflight_models()
        face_thumbs_dir = self._data_folder / "face_thumbs"
        n_workers = max(1, min(self._worker_count, total))
        done = total_faces = total_embedded = 0
        self._emit_progress(
            0, total, f"Analyzing photos: 0 / {total:,} · {n_workers} worker{'s' if n_workers != 1 else ''}", force=True
        )

        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="detect",
            initializer=set_current_thread_below_normal,
        ) as executor:
            for row, result, err in run_bounded(
                executor,
                rows,
                lambda r: detect_one_image(r["id"], r["path"], self._max_dimension),
                max_inflight=n_workers * 2,
                should_cancel=lambda: self._cancel,
            ):
                image_id = row["id"]
                if err is not None:
                    db.mark_image_error(image_id, f"detect failed: {err!r}")
                    summary["errors"] += 1
                elif result is None:
                    db.mark_image_error(image_id, "decode failed during detection")
                    summary["errors"] += 1
                else:
                    with db.transaction():
                        if not result:
                            db.mark_no_faces(image_id)
                        for face_data in result:
                            face_id = db.insert_face(
                                image_id=image_id,
                                bbox_x=face_data["bbox"][0],
                                bbox_y=face_data["bbox"][1],
                                bbox_w=face_data["bbox"][2],
                                bbox_h=face_data["bbox"][3],
                                det_score=face_data["confidence"],
                                landmarks=face_data["landmarks_json"],
                                quality=face_data["quality"],
                            )
                            total_faces += 1
                            if face_data["aligned_face"] is None:
                                continue
                            db.set_face_thumb(
                                face_id,
                                save_face_thumb(
                                    face_data["aligned_face"], face_thumbs_dir, face_id
                                ),
                            )
                            if face_data["embedding"] is not None:
                                db.update_embedding(
                                    face_id,
                                    FaceRecognizer.embedding_to_bytes(face_data["embedding"]),
                                )
                                total_embedded += 1
                        score = classify_image(
                            path=row["path"],
                            thumbnail_path=row["thumb_path"],
                            width=row["width"],
                            height=row["height"],
                            face_count=len(result),
                        )
                        if score > 0.0:
                            db.update_doc_score(image_id, score)

                done += 1
                self._emit_progress(
                    done,
                    total,
                    f"Photos: {done:,} / {total:,} · {total_faces:,} faces found ({n_workers} worker{'s' if n_workers != 1 else ''})",
                )
                if done % _MEMORY_CHECK_INTERVAL == 0 and is_memory_pressure():
                    time.sleep(_MEMORY_BACKPRESSURE_SLEEP)

        summary["faces_detected"] += total_faces
        summary["faces_embedded"] += total_embedded

    def _rescore_legacy_doc_scores(self, db: Database) -> bool:
        if db.get_meta("doc_score_version") == "2":
            return True
        images = db.get_images_for_doc_rescoring()
        for n, img in enumerate(images, 1):
            if self._cancel:
                return False
            score = classify_image(
                path=img["path"],
                thumbnail_path=img["thumb_path"],
                width=img["width"],
                height=img["height"],
                face_count=img["face_count"],
            )
            db.update_doc_score(img["id"], score)
            if n % 200 == 0:
                self._emit_progress(
                    n, len(images), f"Updating document scores: {n:,} / {len(images):,}"
                )
        db.set_meta("doc_score_version", "2")
        return True

    def _run_embed_stage(self, db: Database, summary: dict) -> None:
        remaining = db.count_faces_needing_embedding()
        if remaining == 0:
            return

        self._emit_stage("Computing faceprints", 3)
        recognizer = FaceRecognizer()
        done = 0

        for face_row in db.iter_faces_needing_embedding():
            if self._cancel:
                break
            thumb_path = face_row["thumb_path"]
            if thumb_path is None:
                done += 1
                continue
            aligned = _read_image_safe(thumb_path)
            if aligned is None:
                done += 1
                continue
            embedding = recognizer.compute(aligned)
            if embedding is not None:
                db.update_embedding(
                    face_row["id"], FaceRecognizer.embedding_to_bytes(embedding)
                )
                summary["faces_embedded"] += 1
            done += 1
            self._emit_progress(done, remaining, f"Faceprints: {done:,} / {remaining:,}")

    def _run_identity_stage(self, db: Database, summary: dict) -> None:
        db.clear_orphaned_person_ids()
        total = db.count_unassigned_faces()
        if total == 0:
            summary["persons_found"] = db.count_persons()
            return

        self._emit_stage("Identifying people", 4)
        self._emit_progress(0, total, f"Clustering {total:,} faces…", force=True)

        report = run_graph_clustering(db, config=GraphClusterConfig(), dry_run=False)
        summary["faces_assigned"] = report.total_faces - report.unknown_count
        summary["persons_found"] = db.count_persons()
        self._emit_progress(
            total,
            total,
            f"Discovered {summary['persons_found']:,} people",
            force=True,
        )


class ExportRunner:
    """
    Runs safe copy-only library export on a background daemon thread.
    """

    def __init__(
        self,
        data_folder: Path,
        export_root: Path,
        *,
        person_ids: Optional[List[int]] = None,
        include_group: bool = True,
        include_unknown: bool = True,
        include_docs: bool = True,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_finished: Optional[Callable[[Dict[str, object]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._data_folder = data_folder
        self._export_root = export_root
        self._person_ids = person_ids
        self._include_group = include_group
        self._include_unknown = include_unknown
        self._include_docs = include_docs

        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_error = on_error

        self._cancelled = False
        self._running = False
        self._last_emit_ts = 0.0
        self._last_progress = {"done": 0, "total": 0, "detail": ""}
        self._thread: Optional[threading.Thread] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def snapshot(self) -> dict:
        return {
            "running": self._running,
            "cancelling": self._cancelled and self._running,
            "export_root": str(self._export_root),
            **self._last_progress,
        }

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Export is already running")
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="autosorter-export", daemon=True
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled = True

    def _emit_progress(self, done: int, total: int, detail: str) -> None:
        now = time.monotonic()
        is_boundary = (done == 0) or (total > 0 and done >= total)
        self._last_progress = {"done": done, "total": total, "detail": detail}
        if not is_boundary and (now - self._last_emit_ts) < _PROGRESS_MIN_INTERVAL:
            return
        self._last_emit_ts = now
        if self._on_progress:
            self._on_progress(done, total, detail)

    def _run(self) -> None:
        set_current_thread_below_normal()
        db: Optional[Database] = None
        try:
            db = Database(self._data_folder / "library.db")
            result: Dict[str, object] = dict(
                export_library(
                    db=db,
                    export_root=self._export_root,
                    person_ids=self._person_ids,
                    include_group=self._include_group,
                    include_unknown=self._include_unknown,
                    include_docs=self._include_docs,
                    progress_callback=self._emit_progress,
                    cancel_check=lambda: self._cancelled,
                )
            )
            result["cancelled"] = self._cancelled
            result["export_root"] = str(self._export_root)
            self._running = False
            if self._on_finished:
                self._on_finished(result)
        except Exception as exc:
            log.exception("Export execution failed")
            self._running = False
            if self._on_error:
                self._on_error(str(exc))
        finally:
            self._running = False
            if db is not None:
                db.close()
