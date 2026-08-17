"""
Face detection + embedding worker — runs YuNet AND SFace on all scanned images.

=== WHAT CHANGED FROM M2? ===
M2: detect faces → save face crop → done
M3: detect faces → save face crop → compute faceprint → store in DB

=== THE FULL PIPELINE (M1 → M2 → M3, all in one pass) ===
For each image:
  1. Load image from disk (Unicode-safe)
  2. Run YuNet face detection → bounding boxes + landmarks
  3. For each face:
     a. Align face using landmarks → 112×112 crop
     b. Save face crop to disk
     c. Run SFace on the crop → 128-number embedding
     d. L2-normalize the embedding
     e. Store everything in the faces table

=== WHY ONE PASS INSTEAD OF TWO? ===
We COULD run detection first, then a separate embedding pass.
But doing it in one pass is:
- Faster: the image is already loaded in memory
- Simpler: one button, one progress bar
- Atomic: each face is fully processed before moving to the next

The only downside: if we add a better recognition model later,
we need to re-embed all faces. But that's a future problem (M8).
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from autosorter.core.db import Database
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.thumbnailer import _read_image_safe


class DetectWorker(QThread):
    """
    Background thread that runs face detection AND embedding on all images.

    Signals:
        progress(done, total, faces_found, embedded)  — after each image
        finished(total_images, total_faces, total_embedded, errors)  — done
        error(message)  — if something goes seriously wrong
    """

    # Updated signals to include embedding counts
    progress = Signal(int, int, int, int)   # (done, total, faces, embedded)
    finished = Signal(int, int, int, int)   # (images, faces, embedded, errors)
    error = Signal(str)                     # (error_message)

    def __init__(self, data_folder: Path) -> None:
        """
        Args:
            data_folder: The .autosorter directory containing library.db
                         and where face_thumbs/ will be created.
        """
        super().__init__()
        self._data_folder = data_folder
        self._cancel = False

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._cancel = True

    def run(self) -> None:
        """
        Main detection + embedding loop (runs in background thread).

        For each unprocessed image:
        1. Load image → detect faces (YuNet)
        2. For each face: align → save thumb → compute embedding (SFace) → store
        """
        try:
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            # Load BOTH models once (expensive to load, cheap to run)
            detector = FaceDetector()
            recognizer = FaceRecognizer()

            face_thumbs_dir = self._data_folder / "face_thumbs"

            # Count images needing detection
            total = db.count_images_needing_detection()
            done = 0
            total_faces = 0
            total_embedded = 0
            errors = 0

            for row in db.iter_images_needing_detection():
                if self._cancel:
                    break

                image_id = row["id"]
                image_path = row["path"]

                # Load the full image (Unicode-safe)
                image = _read_image_safe(image_path)

                if image is None:
                    db.mark_no_faces(image_id)
                    errors += 1
                    done += 1
                    self.progress.emit(done, total, total_faces, total_embedded)
                    continue

                # === STEP 1: Detect faces (YuNet) ===
                faces = detector.detect(image)

                if len(faces) == 0:
                    db.mark_no_faces(image_id)
                else:
                    # === STEP 2: Process each face ===
                    for face in faces:
                        # Align face with OUR method (for display thumbnail)
                        aligned_display = align_face(image, face.landmarks)

                        # Align face with SFACE's method (for embedding)
                        # SFace's alignCrop uses the exact alignment the model
                        # was trained on → much more accurate embeddings
                        aligned_sface = recognizer.align_crop(
                            image, face.raw_detection
                        )

                        # Insert face into database FIRST to get the face_id
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

                        # Save display thumbnail
                        thumb_to_save = aligned_sface if aligned_sface is not None else aligned_display
                        if thumb_to_save is not None:
                            thumb_path = save_face_thumb(
                                thumb_to_save, face_thumbs_dir, face_id
                            )
                            db._conn.execute(
                                "UPDATE faces SET thumb_path = ? WHERE id = ?",
                                (thumb_path, face_id),
                            )

                            # === STEP 3: Compute embedding (SFace) ===
                            # Use SFace-aligned crop for best accuracy
                            embed_input = aligned_sface if aligned_sface is not None else aligned_display
                            embedding = recognizer.compute(embed_input)
                            if embedding is not None:
                                # Convert numpy array to raw bytes for storage
                                emb_bytes = FaceRecognizer.embedding_to_bytes(
                                    embedding
                                )
                                db.update_embedding(face_id, emb_bytes)
                                total_embedded += 1

                            db._conn.commit()

                        total_faces += 1

                done += 1
                self.progress.emit(done, total, total_faces, total_embedded)

            # === Also embed any faces from previous runs that lack embeddings ===
            # (handles upgrade from M2 → M3: faces detected before embedding existed)
            remaining = db.count_faces_needing_embedding()
            if remaining > 0 and not self._cancel:
                embed_done = 0
                for face_row in db.iter_faces_needing_embedding():
                    if self._cancel:
                        break

                    thumb_path = face_row["thumb_path"]
                    if thumb_path is None:
                        continue

                    # Load the aligned face crop from disk
                    aligned = _read_image_safe(thumb_path)
                    if aligned is None:
                        continue

                    # Compute embedding
                    embedding = recognizer.compute(aligned)
                    if embedding is not None:
                        emb_bytes = FaceRecognizer.embedding_to_bytes(embedding)
                        db.update_embedding(face_row["id"], emb_bytes)
                        total_embedded += 1

                    embed_done += 1
                    self.progress.emit(
                        done, total, total_faces, total_embedded
                    )

            db.close()
            self.finished.emit(done, total_faces, total_embedded, errors)

        except Exception as e:
            self.error.emit(f"Face detection failed: {e}")
