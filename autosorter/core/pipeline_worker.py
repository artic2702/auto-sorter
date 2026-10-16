"""
Unified pipeline worker — runs all 4 stages in a single background thread.

=== WHAT IS THIS FILE? (M6) ===
The pipeline executes all 4 stages automatically in sequence as methods
of PipelineWorker:

  Stage 1: SCAN      — _run_scan_stage: find new files, hash them, generate thumbnails,
                        detect moved files (SHA-256 reattach), mark missing
  Stage 2: DETECT    — _run_detect_stage: run YuNet face detection on all new images
  Stage 3: EMBED     — _run_embed_stage: run SFace to compute 128-d embeddings for new faces
  Stage 4: IDENTIFY  — _run_cluster_stage: run incremental centroid clustering on unassigned faces

=== WHY ONE WORKER? ===
1. One button, one progress display — much simpler UX
2. Each stage commits per-item to SQLite → crash-safe resume
3. On relaunch, each stage simply skips already-processed items
4. Cancel is checked between every single image/face

=== RESUME MAGIC ===
SQLite IS the checkpoint. When the pipeline is interrupted:
- Stage 1: images already in DB with status='processed' are skipped
- Stage 2: images with face rows are skipped (iter_images_needing_detection)
- Stage 3: faces with embeddings are skipped (iter_faces_needing_embedding)
- Stage 4: faces with person_id are skipped (iter_unassigned_faces)
On restart, each stage picks up exactly where it left off.

=== M8 PERFORMANCE ENHANCEMENTS ===
- Multi-worker thread pool for Stage 2 (face detection is the bottleneck)
- Speed dial support: Quiet (1) / Normal (half cores) / Fast (all cores)
- Process priority set to below-normal so the PC stays usable
- Bounded queues cap decoded-image RAM to ~500 MB even at max workers
- Memory monitoring with backpressure when RSS approaches 1.2 GB
- Batch DB commits reduce SQLite I/O by ~10x
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)

from autosorter.core.doc_classifier import classify_image
from autosorter.core.bounded import run_bounded
from autosorter.core.db import Database
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.perf import (
    SpeedMode,
    worker_count_for_mode,
    set_current_thread_below_normal,
    is_memory_pressure,
    get_rss_bytes,
)
from autosorter.core.scanner import scan_folder
from autosorter.core.thumbnailer import compute_sha256, generate_thumbnail, _read_image_safe


# Number of faces to accumulate before committing to SQLite.
# Reduces I/O: 1 commit per 20 faces instead of 1 commit per face.
# If the app crashes, at most BATCH_SIZE faces need re-processing (< 1 second).
_BATCH_COMMIT_SIZE = 20

# How often to check memory pressure (every N images in Stage 2)
_MEMORY_CHECK_INTERVAL = 10

# Sleep time when memory pressure is detected (seconds)
_MEMORY_BACKPRESSURE_SLEEP = 0.5


class PipelineWorker(QThread):
    """
    Background thread that runs the full scan→detect→embed→cluster pipeline.

    Signals:
        stage_changed(stage_name, stage_num, total_stages)
            Emitted when moving to a new stage.
        progress(done, total, detail_text)
            Emitted after each processed item within a stage.
        finished(summary_dict)
            Emitted when all stages complete. The dict contains counts
            for everything that happened: files scanned, faces found,
            people discovered, files reattached, files missing, etc.
        error(message)
            Emitted on serious failure.
    """

    stage_changed = Signal(str, int, int)       # (stage_name, stage_num, total_stages)
    progress = Signal(int, int, str)             # (done, total, detail_text)
    pipeline_finished = Signal(dict)                      # (summary_dict)
    error = Signal(str)                          # (error_message)

    # Total number of stages in the pipeline
    TOTAL_STAGES = 4

    def __init__(
        self,
        photo_folder: Path,
        data_folder: Path,
        *,
        max_dimension: int = 1280,
        speed_mode: SpeedMode = SpeedMode.NORMAL,
    ) -> None:
        """
        Args:
            photo_folder: The user's photo folder to scan.
            data_folder:  The .autosorter data directory.
            max_dimension: Max image dimension for YuNet (controls RAM usage).
            speed_mode:   Speed dial setting (Quiet/Normal/Fast).
        """
        super().__init__()
        self._photo_folder = photo_folder
        self._data_folder = data_folder
        self._max_dimension = max_dimension
        self._speed_mode = speed_mode
        self._worker_count = worker_count_for_mode(speed_mode)
        self._cancel = False

    @property
    def worker_count(self) -> int:
        """Number of detection workers that will be used."""
        return self._worker_count

    def cancel(self) -> None:
        """Request graceful cancellation (checked between every item)."""
        self._cancel = True

    def run(self) -> None:
        """
        Main pipeline — runs all 4 stages in sequence.

        Each stage is a separate method for clarity. If cancelled,
        we skip remaining stages and emit finished with partial results.
        """
        db: Optional[Database] = None
        try:
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
                "errors": 0,
                "cancelled": False,
                "worker_count": self._worker_count,
                "peak_rss_mb": 0,
            }

            # ═══════════════════════════════════════════════════════════
            # STAGE 1: SCAN — discover files, thumbnails, reattach, missing
            # ═══════════════════════════════════════════════════════════
            if not self._cancel:
                self._run_scan_stage(db, summary)

            # ═══════════════════════════════════════════════════════════
            # STAGE 2: DETECT — YuNet face detection on new images
            # (M8: multi-worker thread pool for parallel detection)
            # ═══════════════════════════════════════════════════════════
            if not self._cancel:
                self._run_detect_stage(db, summary)

            # ═══════════════════════════════════════════════════════════
            # STAGE 3: EMBED — SFace embeddings for new faces
            # (Note: Stage 2 already computes embeddings inline for new
            #  detections. Stage 3 catches any faces from interrupted
            #  previous runs that have thumbnails but no embeddings.)
            # ═══════════════════════════════════════════════════════════
            if not self._cancel:
                self._run_embed_stage(db, summary)

            # ═══════════════════════════════════════════════════════════
            # STAGE 4: IDENTIFY — cluster unassigned faces into persons
            # ═══════════════════════════════════════════════════════════
            if not self._cancel:
                self._run_identity_stage(db, summary)

            # Record peak memory usage
            rss = get_rss_bytes()
            if rss > 0:
                summary["peak_rss_mb"] = round(rss / (1024 * 1024), 1)

            summary["cancelled"] = self._cancel
            self.pipeline_finished.emit(summary)

        except Exception as e:
            log.exception("Pipeline execution failed")
            self.error.emit(f"Pipeline failed: {e}")
        finally:
            if db is not None:
                db.close()

    # ──────────────────────────────────────────────────────────────────
    # Stage 1: Scan
    # ──────────────────────────────────────────────────────────────────

    def _run_scan_stage(self, db: Database, summary: dict) -> None:
        """
        Stage 1: Walk the folder tree, register new files, generate thumbnails,
        detect moved files via SHA-256, and mark missing files.
        """
        self.stage_changed.emit("Scanning files", 1, self.TOTAL_STAGES)
        summary["files_requeued"] = db.requeue_failed_images(max_attempts=3)

        # Phase 1a: File discovery
        scan_result = scan_folder(
            root=self._photo_folder,
            db=db,
            progress_callback=lambda f, n, s: self.progress.emit(
                f, f, f"Discovering: {f} files ({n} new, {s} unchanged)"
            ),
            cancel_check=lambda: self._cancel,
        )

        summary["files_found"] = scan_result.found
        summary["files_new"] = scan_result.new
        summary["files_skipped"] = scan_result.skipped
        summary["errors"] += scan_result.errors

        if self._cancel:
            return

        # Phase 1b: Mark missing files (deleted from disk since last scan)
        missing_count = db.mark_missing_images(scan_result.discovered_paths)
        summary["files_missing"] = missing_count

        if self._cancel:
            return

        # Phase 1c: Generate thumbnails for new/pending images
        thumbs_dir = self._data_folder / "thumbs"
        pending_count = db.count_images(status="pending")
        done = 0

        for row in db.iter_pending():
            if self._cancel:
                break

            image_id = row["id"]
            image_path = row["path"]

            # Compute SHA-256 first (for moved-file detection)
            sha256 = compute_sha256(image_path)

            # Try to reattach by hash (file was moved/renamed)
            if sha256:
                reattached_id = db.reattach_by_hash(sha256, image_path)
                if reattached_id is not None:
                    # The file was already processed under another path.
                    # reattach_by_hash() moved that row to the new path and
                    # already removed the duplicate pending row.
                    summary["files_reattached"] += 1
                    done += 1
                    self.progress.emit(
                        done, pending_count,
                        f"Thumbnails: {done}/{pending_count} (reattached)"
                    )
                    continue

            # Generate thumbnail
            thumb_result = generate_thumbnail(image_path, thumbs_dir, image_id)

            if thumb_result is not None:
                thumb_path, width, height = thumb_result
                db.update_thumb(image_id, thumb_path, width, height)

                # Store the SHA-256 hash for future moved-file detection
                if sha256:
                    db.set_image_sha256(image_id, sha256)

                db.update_status(image_id, "processed")


            else:
                db.mark_image_error(image_id, "decode failed")
                summary["errors"] += 1

            done += 1
            self.progress.emit(
                done, pending_count,
                f"Thumbnails: {done}/{pending_count}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Face Detection + Inline Embedding (M8: Multi-Worker)
    # ──────────────────────────────────────────────────────────────────

    def _run_detect_stage(self, db: Database, summary: dict) -> None:
        """
        Stage 2: detect faces, compute embeddings and score documents.

        Workers (a thread pool, size = speed dial; 1 in Quiet mode) do all the
        CPU-heavy work. This thread is the ONLY writer to SQLite, and writes one
        image per transaction, so a crash can never leave a half-saved image.
        Work is submitted through a sliding window (run_bounded), so a slow
        image never stalls the other workers and RAM stays bounded.
        """
        self.stage_changed.emit("Detecting faces", 2, self.TOTAL_STAGES)
        if not self._rescore_legacy_doc_scores(db):
            return  # cancelled while re-scoring

        rows = list(db.iter_images_needing_detection())
        total = len(rows)
        if total == 0:
            return

        _preflight_models()  # systemic problem -> abort the run, don't mark every photo as failed
        face_thumbs_dir = self._data_folder / "face_thumbs"
        n_workers = max(1, min(self._worker_count, total))
        done = total_faces = total_embedded = 0
        self.progress.emit(0, total, f"Faces: 0/{total} images - {n_workers} worker(s)")

        with ThreadPoolExecutor(
            max_workers=n_workers,
            thread_name_prefix="detect",
            initializer=set_current_thread_below_normal,
        ) as executor:
            for row, result, err in run_bounded(
                executor,
                rows,
                lambda r: _detect_one_image(r["id"], r["path"], self._max_dimension),
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
                                save_face_thumb(face_data["aligned_face"], face_thumbs_dir, face_id),
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
                self.progress.emit(
                    done, total,
                    f"Faces: {done}/{total} images ({total_faces} faces, "
                    f"{total_embedded} embeddings) - {n_workers} worker(s)",
                )
                if done % _MEMORY_CHECK_INTERVAL == 0 and is_memory_pressure():
                    time.sleep(_MEMORY_BACKPRESSURE_SLEEP)

        summary["faces_detected"] += total_faces
        summary["faces_embedded"] += total_embedded

    def _rescore_legacy_doc_scores(self, db: Database) -> bool:
        """One-time fix: doc scores computed before v2 used face_count=0 for every image.

        Returns False only if cancelled (the flag is then NOT set, so it re-runs next time).
        """
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
                self.progress.emit(n, len(images), f"Updating document scores: {n}/{len(images)}")
        db.set_meta("doc_score_version", "2")
        return True

    def _run_embed_stage(self, db: Database, summary: dict) -> None:
        """
        Stage 3: Compute embeddings for any faces that have thumbnails
        but no embeddings yet. This catches faces from interrupted
        previous runs where detection completed but embedding didn't.
        """
        remaining = db.count_faces_needing_embedding()
        if remaining == 0:
            return  # Nothing to do — skip this stage entirely

        self.stage_changed.emit("Computing embeddings", 3, self.TOTAL_STAGES)

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
                emb_bytes = FaceRecognizer.embedding_to_bytes(embedding)
                db.update_embedding(face_row["id"], emb_bytes)
                summary["faces_embedded"] += 1

            done += 1
            self.progress.emit(
                done, remaining,
                f"Embeddings: {done}/{remaining}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Stage 4: Identity Clustering
    # ──────────────────────────────────────────────────────────────────

    def _run_identity_stage(self, db: Database, summary: dict) -> None:
        """
        Stage 4: Run incremental centroid clustering on all faces that
        have embeddings but no person assignment yet.
        """
        # Clean up any orphaned face references (pointing to non-existent persons)
        db.clear_orphaned_person_ids()

        total = db.count_unassigned_faces()
        if total == 0:
            # Nothing to cluster — just report existing state
            summary["persons_found"] = db.count_persons()
            return

        self.stage_changed.emit("Identifying people", 4, self.TOTAL_STAGES)
        self.progress.emit(0, total, f"Clustering {total} faces...")

        from autosorter.core.graph_cluster import run_graph_clustering, GraphClusterConfig
        report = run_graph_clustering(db, config=GraphClusterConfig(), dry_run=False)

        summary["faces_assigned"] = report.total_faces - report.unknown_count
        summary["persons_found"] = db.count_persons()
        self.progress.emit(total, total, f"People: {summary['persons_found']} people found")


# ══════════════════════════════════════════════════════════════════════
# Worker function for multi-threaded detection (M8)
# ══════════════════════════════════════════════════════════════════════

# Thread-local storage for model instances.
# Each thread gets its own FaceDetector and FaceRecognizer to avoid
# any shared state. The models are ~25 MB each, so 4 threads = ~200 MB.
def _preflight_models() -> None:
    """Load both models once on the calling thread; raises if files are missing or corrupt."""
    FaceDetector()
    FaceRecognizer()


_thread_local = threading.local()


def _get_thread_models():
    """
    Get or create per-thread FaceDetector and FaceRecognizer instances.

    Using thread-local storage ensures each thread has its own model,
    avoiding any locking or GIL contention on shared model state.
    The models are loaded lazily on first use per thread.
    """
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = FaceDetector()
        _thread_local.recognizer = FaceRecognizer()
    return _thread_local.detector, _thread_local.recognizer


def _detect_one_image(
    image_id: int,
    image_path: str,
    max_dimension: int,
) -> Optional[List[dict]]:
    """
    Detect faces and compute embeddings for a single image.

    This function runs in a worker thread. It does ALL CPU-intensive work:
    - Load image from disk (Unicode-safe)
    - Run YuNet face detection
    - Align each face using SFace's alignment
    - Compute SFace embedding

    Returns a list of face data dicts (or None if the image can't be decoded).
    The main thread handles all SQLite writes.

    Args:
        image_id:      Database image ID.
        image_path:    Full path to the image file.
        max_dimension: Maximum dimension for YuNet detection.

    Returns:
        None if the image can't be decoded.
        Empty list if no faces found.
        List of dicts with keys: bbox, confidence, landmarks_json, quality,
        aligned_face (numpy array or None), embedding (numpy array or None).
    """
    # Load image (Unicode path safe)
    image = _read_image_safe(image_path)
    if image is None:
        return None

    # Get thread-local model instances
    detector, recognizer = _get_thread_models()

    # Run YuNet face detection
    faces = detector.detect(image, max_dimension=max_dimension)

    if len(faces) == 0:
        return []

    results = []
    for face in faces:
        # Align face using SFace's built-in alignment (best accuracy)
        aligned_sface = recognizer.align_crop(image, face.raw_detection)

        # Fallback to our custom alignment if SFace alignment fails
        if aligned_sface is None:
            aligned_sface = align_face(image, face.landmarks)

        # Compute embedding on the aligned face
        embedding = None
        if aligned_sface is not None:
            embedding = recognizer.compute(aligned_sface)

        results.append({
            "bbox": face.bbox,
            "confidence": face.confidence,
            "landmarks_json": face.landmarks_json(),
            "quality": face.quality_score(),
            "aligned_face": aligned_sface,
            "embedding": embedding,
        })

    return results
"""
Pipeline Worker module.
Unified 4-stage background worker for scan→detect→embed→cluster.
"""
