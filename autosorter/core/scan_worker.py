"""
Scan worker — runs the scanning and thumbnailing pipeline in a background thread.

=== WHAT IS THIS FILE? ===
The scanner and thumbnailer are slow operations (scanning 10,000 photos can take
minutes). If we run them on the main UI thread, the entire window freezes —
you can't click buttons, the progress bar won't move, and Windows shows
"Not Responding".

The solution: run them in a BACKGROUND THREAD using Qt's QThread.

=== HOW THREADS WORK (simple analogy) ===
Think of your app as a restaurant:
  - Main thread = the WAITER (takes orders, serves food, talks to customers)
  - Background thread = the CHEF (cooks food in the kitchen)

The waiter can't cook AND serve at the same time. So:
  - Waiter (UI thread) handles clicks, displays progress, stays responsive
  - Chef (ScanWorker thread) scans files, makes thumbnails, writes to DB

They communicate via SIGNALS (like a kitchen bell):
  - Chef rings bell → "Order ready!" → Waiter picks it up and serves it
  - In Qt: worker.progress.emit(342, 1200) → UI updates "342 / 1200"

=== WHAT ARE Qt SIGNALS AND SLOTS? ===
A Signal is a notification that something happened:
  - progress = Signal(int, int)  ← "I scanned file #X out of Y"
  - finished = Signal(int, int)  ← "All done! Total X, errors Y"

A Slot is a function that handles that notification:
  - def on_progress(scanned, total):
  -     label.setText(f"Scanning: {scanned} / {total}")

You CONNECT them: worker.progress.connect(on_progress)
Now every time the worker emits progress, on_progress runs automatically on the UI thread.
"""

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from autosorter.core.db import Database
from autosorter.core.scanner import scan_folder
from autosorter.core.thumbnailer import compute_sha256, generate_thumbnail


class ScanWorker(QThread):
    """
    Background thread that scans a folder, generates thumbnails, and updates the DB.

    Signals (notifications sent to the UI):
        scan_progress(found, new, skipped)  — emitted as files are discovered
        thumb_progress(done, total)         — emitted as thumbnails are generated
        finished(total_images, errors)      — emitted when everything is done
        error(message)                      — emitted if something goes seriously wrong

    Usage from the UI:
        worker = ScanWorker(photo_folder, data_folder)
        worker.scan_progress.connect(self.update_scan_label)
        worker.thumb_progress.connect(self.update_thumb_progress)
        worker.finished.connect(self.on_scan_done)
        worker.start()  # ← this calls run() in a new thread
    """

    # --- Define the signals ---
    # Signal(int, int, int) means: this signal carries 3 integers.
    # The UI connects a function to each signal to handle the data.
    scan_progress = Signal(int, int, int)   # (found, new, skipped)
    thumb_progress = Signal(int, int)       # (done, total)
    finished = Signal(int, int)             # (total_images, total_errors)
    error = Signal(str)                     # (error_message)

    def __init__(self, photo_folder: Path, data_folder: Path) -> None:
        """
        Args:
            photo_folder: The folder containing the user's photos to scan.
            data_folder:  Where to store the database and thumbnails.
                          Typically photo_folder / ".autosorter"
        """
        super().__init__()

        self._photo_folder = photo_folder
        self._data_folder = data_folder
        self._cancel = False  # Set to True when user clicks Cancel

    def cancel(self) -> None:
        """
        Request cancellation. The worker checks this flag regularly and stops.

        Why a flag instead of killing the thread?
        Forcefully killing a thread can leave the database in a bad state
        (half-written row, open transaction). By using a flag, the worker
        finishes its current file cleanly, then stops. The database stays healthy.
        """
        self._cancel = True

    def _is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel

    def run(self) -> None:
        """
        The main method that runs in the background thread.

        QThread calls this automatically when you call worker.start().
        NEVER call run() directly — always use start().

        The pipeline:
        Phase 1: SCAN — find all image files and add them to the database
        Phase 2: THUMBNAIL — for each pending image, generate a thumbnail
        """
        try:
            # --- Phase 1: Scan for files ---
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            # scan_folder walks the directory and adds files to the DB.
            # We pass our progress callback and cancel checker.
            result = scan_folder(
                root=self._photo_folder,
                db=db,
                progress_callback=lambda f, n, s: self.scan_progress.emit(f, n, s),
                cancel_check=self._is_cancelled,
            )

            if self._cancel:
                db.close()
                self.finished.emit(result.found, result.errors)
                return

            # --- Phase 2: Generate thumbnails for new/pending images ---
            thumbs_dir = self._data_folder / "thumbs"
            pending_count = db.count_images(status="pending")
            done = 0
            total_errors = result.errors

            for row in db.iter_pending():
                if self._cancel:
                    break

                image_id = row["id"]
                image_path = row["path"]

                # Generate thumbnail
                thumb_result = generate_thumbnail(image_path, thumbs_dir, image_id)

                if thumb_result is not None:
                    thumb_path, width, height = thumb_result

                    # Compute file hash (for moved-file detection in M6)
                    sha256 = compute_sha256(image_path)

                    # Update the database with thumbnail info
                    db.update_thumb(image_id, thumb_path, width, height)

                    # Update the upsert with sha256
                    if sha256:
                        db._conn.execute(
                            "UPDATE images SET sha256 = ? WHERE id = ?",
                            (sha256, image_id),
                        )

                    # Mark as processed
                    db.update_status(image_id, "processed")
                else:
                    # Image couldn't be decoded (corrupt, unsupported format)
                    db.update_status(image_id, "error", error="decode failed")
                    total_errors += 1

                done += 1
                self.thumb_progress.emit(done, pending_count)

            db.close()
            total = db_path  # just for the count
            self.finished.emit(result.found, total_errors)

        except Exception as e:
            # Catch any unexpected errors so the thread doesn't crash silently.
            # The UI will show this error message to the user.
            self.error.emit(f"Scan failed: {e}")
