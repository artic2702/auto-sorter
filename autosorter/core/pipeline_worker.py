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

import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QThread, Signal

from autosorter.core.doc_classifier import classify_image
from autosorter.core.db import Database
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.perf import (
    SpeedMode,
    worker_count_for_mode,
    set_below_normal_priority,
    restore_normal_priority,
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
    finished = Signal(dict)                      # (summary_dict)
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
        try:
            # Set below-normal process priority so the PC stays usable
            set_below_normal_priority()

            db_path = self._data_folder / "library.db"
            db = Database(db_path)

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
            db.close()

            # Restore normal process priority
            restore_normal_priority()

            self.finished.emit(summary)

        except Exception as e:
            restore_normal_priority()
            self.error.emit(f"Pipeline failed: {e}")

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
                    # This file was already processed under a different path!
                    # Remove the duplicate pending row (the new path)
                    # The old row has been updated to point to the new path.
                    # Delete the pending row that was just created.
                    existing = db.get_image_by_path(image_path)
                    if existing and existing["id"] != reattached_id:
                        db._conn.execute(
                            "DELETE FROM images WHERE id = ?",
                            (existing["id"],),
                        )
                        db._conn.commit()
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
                    db._conn.execute(
                        "UPDATE images SET sha256 = ? WHERE id = ?",
                        (sha256, image_id),
                    )

                db.update_status(image_id, "processed")

                # ── M7: Classify image as photo vs screenshot/document ──
                # Run the lightweight heuristic classifier on the thumbnail
                # we just generated. This adds < 5ms per image and avoids
                # a separate pass over the database later.
                face_count = len(db.get_faces_for_image(image_id))
                doc_score = classify_image(
                    path=image_path,
                    thumbnail_path=thumb_path,
                    width=width,
                    height=height,
                    face_count=face_count,
                )
                if doc_score > 0.0:
                    db.update_doc_score(image_id, doc_score)

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
        Stage 2: Run YuNet face detection AND SFace embedding on all
        processed images that don't have face rows yet.

        M8 ENHANCEMENT: Uses a thread pool for parallel detection.
        - Each worker has its own FaceDetector + FaceRecognizer instances
          (models are ~25 MB each, thread-safe when each thread has its own).
        - Workers load + decode + detect + embed in parallel.
        - Only the main thread writes to SQLite (single-writer safety).
        - Bounded task submission prevents memory blowup.

        WHY THREADS (not processes)?
        OpenCV releases the GIL during C++ inference. YuNet detection and
        SFace embedding are 95%+ C++ code, so threads scale almost linearly.
        Threads also share the same address space, avoiding the overhead of
        pickling large numpy arrays across process boundaries.
        """
        self.stage_changed.emit("Detecting faces", 2, self.TOTAL_STAGES)

        total = db.count_images_needing_detection()
        if total == 0:
            return

        face_thumbs_dir = self._data_folder / "face_thumbs"
        n_workers = self._worker_count
        done = 0
        total_faces = 0
        total_embedded = 0
        uncommitted = 0

        # ── Single-worker fast path (Quiet mode or very small batches) ──
        if n_workers <= 1 or total <= 2:
            self._run_detect_stage_single(db, summary, total, face_thumbs_dir)
            return

        # ── Multi-worker path ──
        # Collect all image rows that need detection upfront.
        # This avoids cursor conflicts when workers are running.
        image_rows = list(db.iter_images_needing_detection())
        total = len(image_rows)

        if total == 0:
            return

        self.progress.emit(
            0, total,
            f"Faces: 0/{total} images (0 faces) — {n_workers} workers"
        )

        # Use ThreadPoolExecutor for parallel detection.
        # max_workers is capped at the number of images to avoid idle threads.
        actual_workers = min(n_workers, total)

        with ThreadPoolExecutor(
            max_workers=actual_workers,
            thread_name_prefix="detect",
        ) as executor:
            # Submit all tasks. Each future holds the detection results.
            # We submit in batches to limit in-flight decoded images.
            futures: List[Tuple[Future, dict]] = []
            batch_size = actual_workers * 2  # bounded inflight tasks

            row_idx = 0
            while row_idx < total and not self._cancel:
                # Submit a batch of tasks
                batch_end = min(row_idx + batch_size, total)
                batch_futures = []

                for i in range(row_idx, batch_end):
                    if self._cancel:
                        break
                    row = image_rows[i]
                    future = executor.submit(
                        _detect_one_image,
                        row["id"],
                        row["path"],
                        self._max_dimension,
                    )
                    batch_futures.append((future, row))

                # Collect results from this batch (maintains order, limits memory)
                for future, row in batch_futures:
                    if self._cancel:
                        future.cancel()
                        continue

                    try:
                        result = future.result(timeout=120)  # 2 min timeout per image
                    except Exception as exc:
                        # Worker crashed — record a retryable error and continue
                        db.mark_image_error(row["id"], f"detect failed: {exc!r}")
                        summary["errors"] += 1
                        done += 1
                        self.progress.emit(
                            done, total,
                            f"Faces: {done}/{total} images ({total_faces} faces) — {n_workers}w"
                        )
                        continue

                    image_id = row["id"]

                    if result is None:
                        # Image couldn't be decoded
                        db.mark_image_error(image_id, "decode failed during detection")
                        summary["errors"] += 1
                    elif len(result) == 0:
                        # No faces found in this image
                        db.mark_no_faces(image_id)
                    else:
                        # Write all detected faces to DB (single-writer thread)
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

                            if face_data["aligned_face"] is not None:
                                thumb_path = save_face_thumb(
                                    face_data["aligned_face"],
                                    face_thumbs_dir,
                                    face_id,
                                )
                                db._conn.execute(
                                    "UPDATE faces SET thumb_path = ? WHERE id = ?",
                                    (thumb_path, face_id),
                                )

                                if face_data["embedding"] is not None:
                                    emb_bytes = FaceRecognizer.embedding_to_bytes(
                                        face_data["embedding"]
                                    )
                                    db.update_embedding(face_id, emb_bytes)
                                    total_embedded += 1

                            uncommitted += 1
                            total_faces += 1

                        # Batch commit: only write to disk every N faces
                        if uncommitted >= _BATCH_COMMIT_SIZE:
                            db._conn.commit()
                            uncommitted = 0

                    done += 1
                    self.progress.emit(
                        done, total,
                        f"Faces: {done}/{total} images ({total_faces} faces, "
                        f"{total_embedded} emb) — {n_workers}w"
                    )

                    # Memory pressure check
                    if done % _MEMORY_CHECK_INTERVAL == 0 and is_memory_pressure():
                        time.sleep(_MEMORY_BACKPRESSURE_SLEEP)

                row_idx = batch_end

        # Final commit for any remaining uncommitted faces
        if uncommitted > 0:
            db._conn.commit()

        summary["faces_detected"] += total_faces
        summary["faces_embedded"] += total_embedded

    def _run_detect_stage_single(
        self, db: Database, summary: dict, total: int, face_thumbs_dir: Path
    ) -> None:
        """
        Single-threaded detection fallback (for Quiet mode or small batches).
        Identical to the original M6 implementation.
        """
        detector = FaceDetector()
        recognizer = FaceRecognizer()

        done = 0
        total_faces = 0
        total_embedded = 0
        uncommitted = 0

        for row in db.iter_images_needing_detection():
            if self._cancel:
                break

            image_id = row["id"]
            image_path = row["path"]

            image = _read_image_safe(image_path)

            if image is None:
                db.mark_image_error(image_id, "decode failed during detection")
                summary["errors"] += 1
                done += 1
                self.progress.emit(
                    done, total,
                    f"Faces: {done}/{total} images ({total_faces} faces)"
                )
                continue

            faces = detector.detect(image, max_dimension=self._max_dimension)

            if len(faces) == 0:
                db.mark_no_faces(image_id)
            else:
                for face in faces:
                    aligned_display = align_face(image, face.landmarks)
                    aligned_sface = recognizer.align_crop(
                        image, face.raw_detection
                    )

                    face_id = db.insert_face(
                        image_id=image_id,
                        bbox_x=face.bbox[0],
                        bbox_y=face.bbox[1],
                        bbox_w=face.bbox[2],
                        bbox_h=face.bbox[3],
                        det_score=face.confidence,
                        landmarks=face.landmarks_json(),
                        quality=face.quality_score(),
                    )

                    thumb_to_save = (
                        aligned_sface if aligned_sface is not None
                        else aligned_display
                    )
                    if thumb_to_save is not None:
                        thumb_path = save_face_thumb(
                            thumb_to_save, face_thumbs_dir, face_id
                        )
                        db._conn.execute(
                            "UPDATE faces SET thumb_path = ? WHERE id = ?",
                            (thumb_path, face_id),
                        )

                        embed_input = (
                            aligned_sface if aligned_sface is not None
                            else aligned_display
                        )
                        embedding = recognizer.compute(embed_input)
                        if embedding is not None:
                            emb_bytes = FaceRecognizer.embedding_to_bytes(
                                embedding
                            )
                            db.update_embedding(face_id, emb_bytes)
                            total_embedded += 1

                        uncommitted += 1

                    total_faces += 1

                # Batch commit
                if uncommitted >= _BATCH_COMMIT_SIZE:
                    db._conn.commit()
                    uncommitted = 0

            done += 1
            self.progress.emit(
                done, total,
                f"Faces: {done}/{total} images ({total_faces} faces, "
                f"{total_embedded} embeddings)"
            )

        # Final commit
        if uncommitted > 0:
            db._conn.commit()

        summary["faces_detected"] += total_faces
        summary["faces_embedded"] += total_embedded

    # ──────────────────────────────────────────────────────────────────
    # Stage 3: Embed remaining faces (catch-up for interrupted runs)
    # ──────────────────────────────────────────────────────────────────

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
        db._conn.execute("""
            UPDATE faces
            SET person_id = NULL
            WHERE person_id IS NOT NULL
              AND person_id NOT IN (SELECT id FROM persons)
        """)
        db._conn.commit()

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
