"""
Identity worker — runs face clustering in a background thread.

=== WHAT IS THIS FILE? ===
Like ScanWorker (M1) and DetectWorker (M2-M3), this runs a slow operation
in a background thread so the UI stays responsive.

The identity pipeline:
  1. Load existing persons' centroids from DB
  2. For each unassigned face (sorted by quality, best first):
     - Compare to all known persons (one matrix multiply)
     - Assign to best person, or create a new one
  3. Pick cover faces for all persons
  4. Emit finished signal

=== WHEN IS THIS RUN? ===
After face detection + embedding (M3) completes, the user clicks
"Identify People" and this worker runs. It processes all faces that
have embeddings but no person_id assignment.

On subsequent runs (e.g., after scanning new photos), it only processes
the NEW unassigned faces — existing assignments are preserved.
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from autosorter.core.cluster import AssignAction, ClusterEngine, pick_cover_faces
from autosorter.core.db import Database


class IdentityWorker(QThread):
    """
    Background thread that clusters face embeddings into persons.

    Signals:
        progress(done, total, persons_found)  — after each face
        finished(total_assigned, total_persons) — when done
        error(message) — on serious failure
    """

    progress = Signal(int, int, int)   # (done, total, persons_found)
    finished = Signal(int, int)        # (total_assigned, total_persons)
    error = Signal(str)                # (error_message)

    def __init__(self, data_folder: Path) -> None:
        """
        Args:
            data_folder: The .autosorter directory containing library.db.
        """
        super().__init__()
        self._data_folder = data_folder
        self._cancel = False

    def cancel(self) -> None:
        """Request graceful cancellation."""
        self._cancel = True

    def run(self) -> None:
        """
        Main clustering loop (runs in background thread).

        Steps:
        1. Open DB, count unassigned faces
        2. Load existing persons into ClusterEngine
        3. Process each unassigned face
        4. Pick cover faces for all persons
        5. Emit finished
        """
        try:
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            # Initialize the clustering engine with existing persons
            engine = ClusterEngine()
            engine.load_existing_persons(db)

            # Count how many faces need processing
            total = db.count_unassigned_faces()
            done = 0
            assigned = 0

            # Process each unassigned face
            for face_row in db.iter_unassigned_faces():
                if self._cancel:
                    break

                result = engine.process_face(db, face_row)

                if result.action != AssignAction.SKIPPED:
                    assigned += 1

                done += 1
                self.progress.emit(done, total, engine.person_count)

            # Set cover faces for all persons
            if not self._cancel:
                pick_cover_faces(db)

            total_persons = engine.person_count
            db.close()
            self.finished.emit(assigned, total_persons)

        except Exception as e:
            self.error.emit(f"Identity clustering failed: {e}")
