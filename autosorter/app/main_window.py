"""
Main application window — M6: Unified Pipeline + Resume.

=== WHAT CHANGED FROM M5? ===
M5 had: 3 separate buttons (Start Scan, Detect Faces, Identify People)
M6 replaces them with:
  1. Single "▶ Start" button → runs all 4 stages automatically
  2. "▶ Resume" button text when previous run was interrupted
  3. Library state summary on folder selection
  4. Stage-aware progress display: "Stage 2/4: Detecting faces… 12/340"

=== UI LAYOUT (M6) ===
┌─────────────────────────────────────────────────────┐
│  Auto-Sorter 0.1.0                                  │
├─────────────────────────────────────────────────────┤
│ 📁 Photo folder: D:\\Photos               [Browse]  │
│ 📁 Data folder:  (auto)                   [Browse]  │
│                                                     │
│ Library: 1,234 photos · 456 faces · 12 people       │
│ 50 new photos found                                 │
│                                                     │
│ [▶ Start] [Cancel]                                  │
│                                                     │
│ Stage 2/4: Detecting faces…  12/340 images          │
│ ████████████████████░░░░░░░░  33%                   │
│                                                     │
│ ┌──── People (12 found) ───────────────────────┐    │
│ │ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐        │    │
│ │ │(face)│ │(face)│ │(face)│ │(face)│        │    │
│ │ │Arnav │ │ Mom  │ │Per.3 │ │Per.4 │        │    │
│ │ │623 ph│ │412 ph│ │89 ph │ │45 ph │        │    │
│ │ └──────┘ └──────┘ └──────┘ └──────┘        │    │
│ └──────────────────────────────────────────────┘    │
│                                                     │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│ │thumb│ │thumb│ │thumb│ │thumb│ │thumb│ ...        │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
│                                                     │
│ Status: ✅ Done. 1,234 photos · 12 people.         │
└─────────────────────────────────────────────────────┘
"""

from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter import __version__
from autosorter.core.db import Database
from autosorter.core.pipeline_worker import PipelineWorker
from autosorter.app.photo_viewer import PhotoViewer
from autosorter.app.people_panel import PeoplePanel


# Thumbnail display size in the grid (pixels)
GRID_THUMB_SIZE = 120


class ClickableThumb(QLabel):
    """
    A clickable thumbnail label.

    QLabel doesn't have a clicked signal by default, so we override
    mousePressEvent to detect clicks. When clicked, it opens the
    PhotoViewer for this image.

    WHY NOT USE QPushButton?
    QPushButton can display an icon, but it adds button styling (borders,
    hover effects) that makes the grid look ugly. A plain QLabel with a
    pixmap looks cleaner — like a real photo grid.
    """

    def __init__(self, image_row, parent_window, parent=None):
        super().__init__(parent)
        self._image_row = image_row
        self._parent_window = parent_window
        self.setCursor(Qt.CursorShape.PointingHandCursor)  # hand cursor on hover
        self.setFixedSize(GRID_THUMB_SIZE, GRID_THUMB_SIZE)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "border: 2px solid transparent; border-radius: 4px;"
        )

        # Load and display the thumbnail
        if image_row["thumb_path"]:
            pixmap = QPixmap(image_row["thumb_path"])
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    GRID_THUMB_SIZE - 4, GRID_THUMB_SIZE - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.setPixmap(scaled)
            else:
                self.setText("❌")
        else:
            self.setText("❌")

    def enterEvent(self, event):
        """Mouse enters the thumbnail — show a highlight border."""
        self.setStyleSheet(
            "border: 2px solid #4CAF50; border-radius: 4px;"
        )

    def leaveEvent(self, event):
        """Mouse leaves — remove highlight."""
        self.setStyleSheet(
            "border: 2px solid transparent; border-radius: 4px;"
        )

    def mousePressEvent(self, event):
        """Click → open the PhotoViewer for this image."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._parent_window._open_photo_viewer(self._image_row)


class MainWindow(QMainWindow):
    """The main application window with unified pipeline and photo grid."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"Auto-Sorter {__version__}")
        self.resize(900, 650)
        self.setMinimumSize(600, 400)

        # --- State ---
        self._photo_folder: Path | None = None
        self._data_folder: Path | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._viewers: list = []  # Keep references to open PhotoViewer windows

        self._build_ui()

    # ==================================================================
    # UI Construction
    # ==================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 16, 20, 16)

        # --- Title ---
        title = QLabel("Auto-Sorter")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel("Privacy-first offline photo organizer")
        subtitle.setStyleSheet("font-size: 13px; color: gray;")
        layout.addWidget(subtitle)

        layout.addSpacing(4)

        # --- Folder rows ---
        photo_row = QHBoxLayout()
        photo_row.addWidget(QLabel("📁 Photo folder:"))
        self._photo_path_label = QLabel("(not selected)")
        self._photo_path_label.setStyleSheet("color: gray;")
        photo_row.addWidget(self._photo_path_label, stretch=1)
        photo_browse = QPushButton("Browse…")
        photo_browse.setFixedWidth(90)
        photo_browse.clicked.connect(self._pick_photo_folder)
        photo_row.addWidget(photo_browse)
        layout.addLayout(photo_row)

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

        layout.addSpacing(4)

        # --- Library state summary (M6) ---
        self._library_label = QLabel("")
        self._library_label.setStyleSheet("font-size: 13px; color: #555;")
        self._library_label.setVisible(False)
        layout.addWidget(self._library_label)

        # --- Action buttons row (M6: single Start button) ---
        btn_row = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedHeight(36)
        self._start_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 20px;"
        )
        self._start_btn.clicked.connect(self._start_pipeline)
        self._start_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.clicked.connect(self._cancel_pipeline)
        self._cancel_btn.setVisible(False)
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # --- Stage label (M6: shows current stage) ---
        self._stage_label = QLabel("")
        self._stage_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        self._stage_label.setVisible(False)
        layout.addWidget(self._stage_label)

        # --- Progress area ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.setVisible(False)
        layout.addWidget(self._progress_label)

        # --- People panel (M4) ---
        self._people_panel: PeoplePanel | None = None
        self._people_container = QWidget()
        self._people_container.setVisible(False)
        people_layout = QVBoxLayout(self._people_container)
        people_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._people_container)

        # --- Thumbnail grid (inside a scroll area) ---
        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setVisible(False)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(6)
        self._grid_scroll.setWidget(self._grid_container)
        layout.addWidget(self._grid_scroll, stretch=1)

        # Spacer (only used when grid is hidden)
        self._spacer_label = QLabel("")
        layout.addWidget(self._spacer_label, stretch=1)

        # --- Status bar ---
        self._status_label = QLabel("Ready — select a photo folder to begin.")
        self._status_label.setStyleSheet(
            "color: gray; border-top: 1px solid #ddd; padding-top: 8px;"
        )
        layout.addWidget(self._status_label)

    # ==================================================================
    # Folder selection
    # ==================================================================

    def _pick_photo_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select your photo folder"
        )
        if folder:
            self._photo_folder = Path(folder)
            self._photo_path_label.setText(folder)
            self._photo_path_label.setStyleSheet("")

            if self._data_folder is None:
                self._data_folder = self._photo_folder / ".autosorter"
                self._data_path_label.setText(str(self._data_folder))

            self._start_btn.setEnabled(True)

            # Show library state if DB already exists (M6: resume support)
            self._update_library_summary()

    def _pick_data_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select data storage folder"
        )
        if folder:
            self._data_folder = Path(folder)
            self._data_path_label.setText(folder)
            self._data_path_label.setStyleSheet("")

            # Update library summary with new data folder
            self._update_library_summary()

    def _update_library_summary(self) -> None:
        """
        Show a summary of the existing library state (M6).

        If the database already exists from a previous run, show counts
        and change the button text to "Resume" instead of "Start".
        """
        if self._data_folder is None:
            return

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            self._library_label.setVisible(False)
            self._start_btn.setText("▶  Start")
            self._status_label.setText("Ready — click Start.")
            return

        try:
            db = Database(db_path)
            state = db.get_pipeline_state()
            db.close()

            parts = []
            if state["total"] > 0:
                parts.append(f"{state['total']} photos")
            if state["faces"] > 0:
                parts.append(f"{state['faces']} faces")
            if state["persons"] > 0:
                parts.append(f"{state['persons']} people")

            if parts:
                self._library_label.setText(
                    f"📚 Library: {' · '.join(parts)}"
                )
                self._library_label.setVisible(True)

            # Decide button text: "Start" vs "Resume" vs "Re-Scan"
            has_work = (
                state["pending"] > 0
                or state["needs_detection"] > 0
                or state["unassigned"] > 0
            )
            if state["total"] == 0:
                self._start_btn.setText("▶  Start")
                self._status_label.setText("Ready — click Start to begin scanning.")
            elif has_work:
                self._start_btn.setText("▶  Resume")
                self._status_label.setText(
                    f"Previous run interrupted. "
                    f"{state['pending']} pending, "
                    f"{state['unassigned']} unassigned faces."
                )
            else:
                self._start_btn.setText("▶  Re-Scan")
                self._status_label.setText(
                    f"✅ Library up to date. Click Re-Scan to check for new photos."
                )

            # Auto-load existing data
            if state["total"] > 0:
                self._load_thumbnail_grid()
            if state["persons"] > 0:
                self._show_people_panel()

        except Exception:
            self._library_label.setVisible(False)
            self._start_btn.setText("▶  Start")

    # ==================================================================
    # Pipeline control (M6)
    # ==================================================================

    def _start_pipeline(self) -> None:
        """Launch the unified pipeline (scan → detect → embed → cluster)."""
        if self._photo_folder is None or self._data_folder is None:
            return

        self._pipeline_worker = PipelineWorker(
            self._photo_folder, self._data_folder
        )
        self._pipeline_worker.stage_changed.connect(self._on_stage_changed)
        self._pipeline_worker.progress.connect(self._on_pipeline_progress)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_error)

        self._start_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate initially
        self._stage_label.setVisible(True)
        self._stage_label.setText("Starting pipeline…")
        self._progress_label.setVisible(True)
        self._status_label.setText("Pipeline running…")
        self._status_label.setStyleSheet(
            "color: gray; border-top: 1px solid #ddd; padding-top: 8px;"
        )

        self._pipeline_worker.start()

    def _cancel_pipeline(self) -> None:
        """Cancel the running pipeline."""
        if self._pipeline_worker:
            self._pipeline_worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")
        self._status_label.setText("Cancelling… (finishing current item)")

    # ==================================================================
    # Signal handlers — Pipeline (M6)
    # ==================================================================

    def _on_stage_changed(self, stage_name: str, stage_num: int,
                          total_stages: int) -> None:
        """Update UI when pipeline moves to a new stage."""
        self._stage_label.setText(
            f"Stage {stage_num}/{total_stages}: {stage_name}"
        )
        # Reset progress bar to indeterminate for the new stage
        self._progress_bar.setRange(0, 0)

    def _on_pipeline_progress(self, done: int, total: int,
                              detail: str) -> None:
        """Update progress bar and label during pipeline execution."""
        if total > 0:
            self._progress_bar.setRange(0, total)
            self._progress_bar.setValue(done)
        self._progress_label.setText(detail)

    def _on_pipeline_finished(self, summary: dict) -> None:
        """Handle pipeline completion — show results and enable re-scan."""
        self._start_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setVisible(False)
        self._stage_label.setVisible(False)
        self._progress_label.setVisible(False)

        # Build summary message
        if summary.get("cancelled"):
            self._start_btn.setText("▶  Resume")
            self._status_label.setText(
                f"⏸️ Paused. "
                f"{summary['files_found']} files found, "
                f"{summary['faces_detected']} faces. "
                f"Click Resume to continue."
            )
        else:
            self._start_btn.setText("▶  Re-Scan")

            parts = []
            if summary["files_new"] > 0:
                parts.append(f"{summary['files_new']} new photos processed")
            if summary["files_reattached"] > 0:
                parts.append(f"{summary['files_reattached']} moved files reattached")
            if summary["files_missing"] > 0:
                parts.append(f"{summary['files_missing']} files missing")
            if summary["faces_detected"] > 0:
                parts.append(f"{summary['faces_detected']} faces found")
            if summary["persons_found"] > 0:
                parts.append(f"{summary['persons_found']} people")
            if summary["errors"] > 0:
                parts.append(f"{summary['errors']} errors")

            if parts:
                self._status_label.setText(f"✅ Done. {'. '.join(parts)}.")
            else:
                self._status_label.setText("✅ Library up to date. No new photos found.")

        self._pipeline_worker = None

        # Refresh thumbnail grid and people panel
        self._load_thumbnail_grid()
        self._update_library_summary_after_pipeline()

    def _update_library_summary_after_pipeline(self) -> None:
        """Update library label and auto-show people panel after pipeline."""
        if self._data_folder is None:
            return

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        try:
            db = Database(db_path)
            state = db.get_pipeline_state()
            db.close()

            parts = []
            if state["total"] > 0:
                parts.append(f"{state['total']} photos")
            if state["faces"] > 0:
                parts.append(f"{state['faces']} faces")
            if state["persons"] > 0:
                parts.append(f"{state['persons']} people")

            if parts:
                self._library_label.setText(
                    f"📚 Library: {' · '.join(parts)}"
                )
                self._library_label.setVisible(True)

            if state["persons"] > 0:
                self._show_people_panel()

        except Exception:
            pass

    def _on_error(self, message: str) -> None:
        """Handle pipeline errors."""
        self._start_btn.setEnabled(True)
        self._start_btn.setText("▶  Resume")
        self._cancel_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        self._stage_label.setVisible(False)
        self._progress_label.setVisible(False)
        self._status_label.setText(f"❌ Error: {message}")
        self._status_label.setStyleSheet(
            "color: red; border-top: 1px solid #ddd; padding-top: 8px;"
        )
        self._pipeline_worker = None

    # ==================================================================
    # Thumbnail Grid (M2)
    # ==================================================================

    def _load_thumbnail_grid(self) -> None:
        """
        Load all scanned image thumbnails into a clickable grid.

        We read the database for all processed images, then create
        ClickableThumb widgets for each one, arranged in a grid layout.
        """
        if self._data_folder is None:
            return

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        db = Database(db_path)
        images = db.get_all_images_with_thumbs()
        db.close()

        if not images:
            return

        # Clear existing grid
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Calculate how many columns fit in the window
        cols = max(1, (self.width() - 60) // (GRID_THUMB_SIZE + 6))

        # Add thumbnails to the grid
        for i, img_row in enumerate(images):
            thumb = ClickableThumb(img_row, self)
            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(thumb, row, col)

        # Show the grid, hide the spacer
        self._grid_scroll.setVisible(True)
        self._spacer_label.setVisible(False)

    # ==================================================================
    # Photo Viewer (M2)
    # ==================================================================

    def _open_photo_viewer(self, image_row) -> None:
        """
        Open a PhotoViewer window for the clicked image.

        Loads face data from the database and passes it to the viewer.
        """
        if self._data_folder is None:
            return

        db_path = self._data_folder / "library.db"
        db = Database(db_path)
        faces = db.get_faces_for_image(image_row["id"])
        db.close()

        viewer = PhotoViewer(
            image_row["path"], faces, data_folder=self._data_folder
        )
        viewer.changed.connect(self._on_viewer_changed)
        viewer.show()
        # Keep a reference so the window doesn't get garbage-collected
        self._viewers.append(viewer)

    def _on_viewer_changed(self) -> None:
        """When face assignments change in photo viewer, reload people panel."""
        if self._people_panel is not None:
            self._people_panel.load_persons()

    # ==================================================================
    # People Panel (M4)
    # ==================================================================

    def _show_people_panel(self) -> None:
        """Create or refresh the people panel."""
        if self._data_folder is None:
            return

        if self._people_panel is None:
            self._people_panel = PeoplePanel(self._data_folder)
            self._people_container.layout().addWidget(self._people_panel)

        self._people_panel.load_persons()
        self._people_container.setVisible(True)
