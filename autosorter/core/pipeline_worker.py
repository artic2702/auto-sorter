"""
Unified pipeline worker — runs all 4 stages in a single background thread.

=== WHAT IS THIS FILE? (M6) ===
Before M6, the user had to click 3 separate buttons:
  1. "Start Scan"      → ScanWorker
  2. "Detect Faces"    → DetectWorker
  3. "Identify People" → IdentityWorker

This was tedious and error-prone. M6 replaces all three with a single
"▶ Start" button that runs everything automatically in sequence:

  Stage 1: SCAN      — find new files, hash them, generate thumbnails,
                        detect moved files (SHA-256 reattach), mark missing
  Stage 2: DETECT    — run YuNet face detection on all new images
  Stage 3: EMBED     — run SFace to compute 128-d embeddings for new faces
  Stage 4: IDENTIFY  — run incremental centroid clustering on unassigned faces

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
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from autosorter.core.cluster import (
    ClusterEngine,
    AssignAction,
    pick_cover_faces,
    consolidate_clusters,
)
from autosorter.core.doc_classifier import classify_image
from autosorter.core.db import Database
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.scanner import scan_folder
from autosorter.core.thumbnailer import compute_sha256, generate_thumbnail, _read_image_safe


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
    ) -> None:
        """
        Args:
            photo_folder: The user's photo folder to scan.
            data_folder:  The .autosorter data directory.
            max_dimension: Max image dimension for YuNet (controls RAM usage).
        """
        super().__init__()
        self._photo_folder = photo_folder
        self._data_folder = data_folder
        self._max_dimension = max_dimension
        self._cancel = False

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
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            summary = {
                "files_found": 0,
                "files_new": 0,
                "files_skipped": 0,
                "files_reattached": 0,
                "files_missing": 0,
                "faces_detected": 0,
                "faces_embedded": 0,
                "faces_assigned": 0,
                "persons_found": 0,
                "errors": 0,
                "cancelled": False,
            }

            # ═══════════════════════════════════════════════════════════
            # STAGE 1: SCAN — discover files, thumbnails, reattach, missing
            # ═══════════════════════════════════════════════════════════
            if not self._cancel:
                self._run_scan_stage(db, summary)

            # ═══════════════════════════════════════════════════════════
            # STAGE 2: DETECT — YuNet face detection on new images
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

            summary["cancelled"] = self._cancel
            db.close()
            self.finished.emit(summary)

        except Exception as e:
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
                db.update_status(image_id, "error", error="decode failed")
                summary["errors"] += 1

            done += 1
            self.progress.emit(
                done, pending_count,
                f"Thumbnails: {done}/{pending_count}"
            )

    # ──────────────────────────────────────────────────────────────────
    # Stage 2: Face Detection + Inline Embedding
    # ──────────────────────────────────────────────────────────────────

    def _run_detect_stage(self, db: Database, summary: dict) -> None:
        """
        Stage 2: Run YuNet face detection AND SFace embedding on all
        processed images that don't have face rows yet.

        Detect + embed is done in one pass (same as old DetectWorker)
        because the image is already loaded in memory.
        """
        self.stage_changed.emit("Detecting faces", 2, self.TOTAL_STAGES)

        detector = FaceDetector()
        recognizer = FaceRecognizer()
        face_thumbs_dir = self._data_folder / "face_thumbs"

        total = db.count_images_needing_detection()
        done = 0
        total_faces = 0
        total_embedded = 0

        for row in db.iter_images_needing_detection():
            if self._cancel:
                break

            image_id = row["id"]
            image_path = row["path"]

            image = _read_image_safe(image_path)

            if image is None:
                db.mark_no_faces(image_id)
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

                        db._conn.commit()

                    total_faces += 1

            done += 1
            self.progress.emit(
                done, total,
                f"Faces: {done}/{total} images ({total_faces} faces, "
                f"{total_embedded} embeddings)"
            )

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
"""
Pipeline Worker module.
Unified 4-stage background worker for scan→detect→embed→cluster.
"""
