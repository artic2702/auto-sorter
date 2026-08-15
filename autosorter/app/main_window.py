"""
Main application window — M1: Folder selection, scanning, and live progress.

=== WHAT CHANGED FROM M0? ===
M0 had just a placeholder label saying "under construction".
Now we have a REAL UI with:
  1. Two buttons to pick the photo folder and data folder
  2. A "Start Scan" button that launches the background scanner
  3. A progress bar and live counter that update in real-time
  4. A "Cancel" button to stop scanning

=== UI LAYOUT (what the user sees) ===
┌─────────────────────────────────────────────────────┐
│  Auto-Sorter 0.1.0                           [─][□][×]│
├─────────────────────────────────────────────────────┤
│                                                     │
│   📁 Photo folder:  D:\\Users\\You\\Photos    [Browse] │
│   📁 Data folder:   (auto)                  [Browse] │
│                                                     │
│   [▶ Start Scan]                                    │
│                                                     │
│   ████████████████████░░░░░░░░  75%                 │
│   Phase 1: Found 900 images (850 new, 50 unchanged) │
│   Phase 2: Thumbnails 637 / 850                     │
│                                                     │
│   [Cancel]                                          │
│                                                     │
│ ─────────────────────────────────────────────────── │
│ Status: Scanning...                                 │
└─────────────────────────────────────────────────────┘

=== QT LAYOUT SYSTEM (how widgets are arranged) ===
Qt uses "layouts" to arrange widgets (buttons, labels, etc.) automatically.
- QVBoxLayout: stacks widgets VERTICALLY (top to bottom)
- QHBoxLayout: stacks widgets HORIZONTALLY (left to right)
We nest them: a VBox contains several HBoxes, each HBox is one "row" of widgets.
"""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from autosorter import __version__
from autosorter.core.scan_worker import ScanWorker


class MainWindow(QMainWindow):
    """The main application window with scan controls and progress display."""

    def __init__(self) -> None:
        super().__init__()

        # --- Window properties ---
        self.setWindowTitle(f"Auto-Sorter {__version__}")
        self.resize(720, 420)
        # Minimum size so the window can't be shrunk too small to see
        self.setMinimumSize(500, 300)

        # --- State ---
        self._photo_folder: Path | None = None  # User's selected photo folder
        self._data_folder: Path | None = None    # Where DB + thumbnails go
        self._worker: ScanWorker | None = None   # Background scanner thread

        # --- Build the UI ---
        self._build_ui()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        """
        Create all the widgets and arrange them in layouts.

        WIDGET = any UI element (button, label, text box, progress bar, etc.)
        LAYOUT = invisible container that positions widgets automatically

        We build from the inside out:
        1. Create individual widgets (buttons, labels)
        2. Add them to layouts (HBox for rows, VBox for the page)
        3. Set the VBox as the window's content
        """
        # The "central widget" is Qt's name for the main content area of a window.
        # QMainWindow requires you to set one. We create a plain QWidget and
        # put our layout inside it.
        central = QWidget()
        self.setCentralWidget(central)

        # Main vertical layout — everything stacks top to bottom
        layout = QVBoxLayout(central)
        layout.setSpacing(12)           # 12px gap between each row
        layout.setContentsMargins(24, 24, 24, 24)  # 24px padding around edges

        # --- Title ---
        title = QLabel("Auto-Sorter")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Privacy-first offline photo organizer")
        subtitle.setStyleSheet("font-size: 13px; color: gray;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)  # Extra gap before the folder pickers

        # --- Photo folder row ---
        # QHBoxLayout arranges widgets left-to-right in one row:
        #   [ Label ] [ Path display ] [ Browse button ]
        photo_row = QHBoxLayout()
        photo_row.addWidget(QLabel("📁 Photo folder:"))
        self._photo_path_label = QLabel("(not selected)")
        self._photo_path_label.setStyleSheet("color: gray;")
        photo_row.addWidget(self._photo_path_label, stretch=1)
        # stretch=1 means this label takes up all remaining horizontal space

        photo_browse = QPushButton("Browse…")
        photo_browse.setFixedWidth(90)
        photo_browse.clicked.connect(self._pick_photo_folder)
        # clicked.connect() links the button click SIGNAL to our handler SLOT
        photo_row.addWidget(photo_browse)
        layout.addLayout(photo_row)

        # --- Data folder row ---
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("📁 Data folder:"))
        self._data_path_label = QLabel("(auto — inside photo folder)")
        self._data_path_label.setStyleSheet("color: gray;")
        data_row.addWidget(self._data_path_label, stretch=1)

        data_browse = QPushButton("Browse…")
        data_browse.setFixedWidth(90)
        data_browse.clicked.connect(self._pick_data_folder)
        data_row.addWidget(data_browse)
        layout.addLayout(data_row)

        layout.addSpacing(8)

        # --- Start / Cancel buttons ---
        btn_row = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start Scan")
        self._start_btn.setFixedHeight(36)
        self._start_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 20px;"
        )
        self._start_btn.clicked.connect(self._start_scan)
        self._start_btn.setEnabled(False)  # Disabled until a folder is selected
        btn_row.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.clicked.connect(self._cancel_scan)
        self._cancel_btn.setVisible(False)  # Hidden until a scan is running
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch()  # Push buttons to the left
        layout.addLayout(btn_row)

        # --- Progress area ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._scan_label = QLabel("")
        self._scan_label.setVisible(False)
        layout.addWidget(self._scan_label)

        self._thumb_label = QLabel("")
        self._thumb_label.setVisible(False)
        layout.addWidget(self._thumb_label)

        # Push everything to the top (spacer at the bottom eats remaining space)
        layout.addStretch()

        # --- Status bar at the bottom ---
        self._status_label = QLabel("Ready — select a photo folder to begin.")
        self._status_label.setStyleSheet(
            "color: gray; border-top: 1px solid #ddd; padding-top: 8px;"
        )
        layout.addWidget(self._status_label)

    # ==================================================================
    # Folder selection handlers
    # ==================================================================

    def _pick_photo_folder(self) -> None:
        """
        Open a native Windows folder picker dialog.

        QFileDialog.getExistingDirectory() opens the standard Windows folder
        browser. It returns the selected path as a string, or an empty string
        if the user clicked Cancel.
        """
        folder = QFileDialog.getExistingDirectory(
            self,                           # parent window
            "Select your photo folder",     # dialog title
        )
        if folder:
            self._photo_folder = Path(folder)
            self._photo_path_label.setText(folder)
            self._photo_path_label.setStyleSheet("")  # Remove gray color

            # Auto-set data folder inside the photo folder
            if self._data_folder is None:
                self._data_folder = self._photo_folder / ".autosorter"
                self._data_path_label.setText(str(self._data_folder))

            self._start_btn.setEnabled(True)
            self._status_label.setText("Ready — click Start Scan.")

    def _pick_data_folder(self) -> None:
        """Let the user choose a custom location for the database + thumbnails."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select data storage folder"
        )
        if folder:
            self._data_folder = Path(folder)
            self._data_path_label.setText(folder)
            self._data_path_label.setStyleSheet("")

    # ==================================================================
    # Scan control
    # ==================================================================

    def _start_scan(self) -> None:
        """
        Launch the background scan worker.

        This method:
        1. Creates a ScanWorker (background thread)
        2. Connects its signals to our UI update methods
        3. Starts the thread
        4. Updates button visibility (hide Start, show Cancel)
        """
        if self._photo_folder is None or self._data_folder is None:
            return

        # Create the background worker
        self._worker = ScanWorker(self._photo_folder, self._data_folder)

        # CONNECT signals to slots:
        # When the worker emits scan_progress → call _on_scan_progress
        # When the worker emits thumb_progress → call _on_thumb_progress
        # When the worker emits finished → call _on_finished
        # When the worker emits error → call _on_error
        self._worker.scan_progress.connect(self._on_scan_progress)
        self._worker.thumb_progress.connect(self._on_thumb_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # Show progress UI, hide start button
        self._start_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # 0,0 = indeterminate (spinning)
        self._scan_label.setVisible(True)
        self._scan_label.setText("Phase 1: Discovering images…")
        self._thumb_label.setVisible(False)
        self._status_label.setText("Scanning…")

        # START the background thread
        # This calls worker.run() in a new thread — the UI stays responsive!
        self._worker.start()

    def _cancel_scan(self) -> None:
        """Request the worker to stop gracefully."""
        if self._worker:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling…")
            self._status_label.setText("Cancelling… (finishing current file)")

    # ==================================================================
    # Signal handlers (called automatically when the worker emits signals)
    # ==================================================================

    def _on_scan_progress(self, found: int, new: int, skipped: int) -> None:
        """
        Called every time the scanner finds another file.

        This runs on the UI THREAD (Qt automatically marshals signals across
        threads), so it's safe to update widgets here.
        """
        self._scan_label.setText(
            f"Phase 1: Found {found} images ({new} new, {skipped} unchanged)"
        )

    def _on_thumb_progress(self, done: int, total: int) -> None:
        """Called as each thumbnail is generated."""
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._thumb_label.setVisible(True)
        self._thumb_label.setText(f"Phase 2: Thumbnails {done} / {total}")

    def _on_finished(self, total: int, errors: int) -> None:
        """Called when the entire scan + thumbnail pipeline is complete."""
        # Reset UI
        self._start_btn.setEnabled(True)
        self._start_btn.setText("▶  Re-Scan")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setVisible(False)

        # Show summary
        error_text = f", {errors} errors" if errors else ""
        self._status_label.setText(
            f"✅ Scan complete: {total} images found{error_text}."
        )
        self._worker = None

    def _on_error(self, message: str) -> None:
        """Called if the worker hits an unexpected error."""
        self._start_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"❌ Error: {message}")
        self._status_label.setStyleSheet(
            "color: red; border-top: 1px solid #ddd; padding-top: 8px;"
        )
        self._worker = None
