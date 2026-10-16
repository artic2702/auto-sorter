"""
Export worker — runs the export engine in a background QThread.

=== WHAT IS THIS FILE? (M7) ===
The export operation can take a long time for large libraries (copying
thousands of files). This worker runs it in a background thread so the
UI stays responsive, with progress signals and cancel support.

=== SIGNAL FLOW ===
    ExportWorker.start()
         │
         ├── progress(done, total, detail)  → updates progress bar
         ├── finished(result_dict)          → shows completion summary
         └── error(message)                → shows error dialog
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from autosorter.core.db import Database
from autosorter.core.export import export_library

log = logging.getLogger(__name__)


class ExportWorker(QThread):
    """
    Background thread that runs the photo export engine.

    Signals:
        progress(done, total, detail_text):
            Emitted after each file is processed.
        finished(result_dict):
            Emitted when export completes. Dict contains counts.
        error(message):
            Emitted on serious failure.
    """

    # Qt signals for communicating with the main UI thread
    progress = Signal(int, int, str)      # (done, total, detail_text)
    finished = Signal(dict)               # (result_dict)
    error = Signal(str)                   # (error_message)

    def __init__(
        self,
        data_folder: Path,
        export_root: Path,
        person_ids: Optional[list] = None,
        include_group: bool = True,
        include_unknown: bool = True,
        include_docs: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        # Store export configuration — these are read by the run() method
        self._data_folder = data_folder
        self._export_root = export_root
        # Selective person export filter: None = all people, list = specific person IDs
        self._person_ids = person_ids
        self._include_group = include_group
        self._include_unknown = include_unknown
        self._include_docs = include_docs
        # Cancel flag — set to True by the main thread to stop export
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation. The export will stop after the current file."""
        self._cancelled = True

    def run(self) -> None:
        """
        Execute the export in the background thread.

        Opens its own database connection (SQLite connections cannot be
        shared across threads) and runs the export engine with progress
        reporting and cancel checking.
        """
        try:
            # Open a dedicated database connection for this thread
            db_path = self._data_folder / "library.db"
            db = Database(db_path)

            # Run the export engine with our progress and cancel callbacks
            result = export_library(
                db=db,
                export_root=self._export_root,
                person_ids=self._person_ids,
                include_group=self._include_group,
                include_unknown=self._include_unknown,
                include_docs=self._include_docs,
                progress_callback=self._on_progress,
                cancel_check=self._is_cancelled,
            )

            db.close()

            # Add cancellation flag to the result
            result["cancelled"] = self._cancelled

            # Emit the finished signal with the result summary
            self.finished.emit(result)

        except Exception as e:
            log.exception("Export worker execution failed")
            self.error.emit(str(e))

    def _on_progress(self, done: int, total: int, detail: str) -> None:
        """Forward progress from the export engine to the Qt signal."""
        self.progress.emit(done, total, detail)

    def _is_cancelled(self) -> bool:
        """Check if the user has requested cancellation."""
        return self._cancelled
