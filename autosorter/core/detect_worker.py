"""
Face detection worker — runs YuNet on all scanned images in a background thread.

=== WHAT IS THIS FILE? ===
After M1's scanning + thumbnailing is done, images have status='processed'.
This worker picks up those images, runs face detection on each one, saves
face crops, and writes the results to the faces table.

=== THE PIPELINE (M1 → M2) ===
M1 did:  Photo folder → scan files → thumbnails → status='processed'
M2 adds: status='processed' → detect faces → save face crops → faces table

=== HOW IT AVOIDS RE-WORK ===
On first run: all processed images get face detection.
On re-run: only images WITHOUT any face rows are processed.
This is handled by db.iter_images_needing_detection() which uses a
LEFT JOIN to find images with no entries in the faces table.
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from autosorter.core.db import Database
from autosorter.core.face_crop import align_face, save_face_thumb
from autosorter.core.face_detect import FaceDetector
from autosorter.core.thumbnailer import _read_image_safe


class DetectWorker(QThread):
    """
    Background thread that runs face detection on all scanned images.

    Signals:
        progress(done, total, faces_found)  — emitted after each image
        finished(total_images, total_faces, errors)  — when done
        error(message)  — if something goes seriously wrong
    """

    progress = Signal(int, int, int)     # (done, total, total_faces_so_far)
    finished = Signal(int, int, int)     # (total_images, total_faces, errors)
    error = Signal(str)                  # (error_message)

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
        Main detection loop (runs in background thread).

        For each image:
        1. Load the full image from disk
        2. Run YuNet face detection
        3. For each face found:
           a. Align and crop the face
           b. Save the face crop as a JPEG
           c. Insert a row into the faces table
        4. If no faces found, insert a sentinel row so we skip it next time
        """
        try:
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            # Create the FaceDetector ONCE (loading the model is expensive)
            detector = FaceDetector()

            face_thumbs_dir = self._data_folder / "face_thumbs"

            # Count how many images need detection
            total = db.count_images_needing_detection()
            done = 0
            total_faces = 0
            errors = 0

            for row in db.iter_images_needing_detection():
                if self._cancel:
                    break

                image_id = row["id"]
                image_path = row["path"]

                # Load the full image (Unicode-safe)
                image = _read_image_safe(image_path)

                if image is None:
                    # Image can't be read (deleted, corrupted, etc.)
                    db.mark_no_faces(image_id)
                    errors += 1
                    done += 1
                    self.progress.emit(done, total, total_faces)
                    continue

                # Run face detection!
                faces = detector.detect(image)

                if len(faces) == 0:
                    # No faces found — mark so we skip this image next time
                    db.mark_no_faces(image_id)
                else:
                    # Process each detected face
                    for face in faces:
                        # Align the face (rotate so eyes are level)
                        aligned = align_face(image, face.landmarks)

                        # Insert into database FIRST to get the face_id
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

                        # Save aligned face thumbnail
                        if aligned is not None:
                            thumb_path = save_face_thumb(
                                aligned, face_thumbs_dir, face_id
                            )
                            # Update the face row with the thumb path
                            db._conn.execute(
                                "UPDATE faces SET thumb_path = ? WHERE id = ?",
                                (thumb_path, face_id),
                            )
                            db._conn.commit()

                        total_faces += 1

                done += 1
                self.progress.emit(done, total, total_faces)

            db.close()
            self.finished.emit(done, total_faces, errors)

        except Exception as e:
            self.error.emit(f"Face detection failed: {e}")
