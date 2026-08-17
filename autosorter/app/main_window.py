"""
Main application window — M2: Scan + Face Detection + Thumbnail Grid + Photo Viewer.

=== WHAT CHANGED FROM M1? ===
M1 had: folder picker, scan button, progress labels.
M2 adds:
  1. "Detect Faces" button — appears after scan, runs YuNet on all images
  2. Thumbnail grid — shows all scanned photos as clickable thumbnails
  3. Click a thumbnail → opens PhotoViewer with face boxes drawn on the image
  4. Detection progress — "Detecting faces: 42 / 58 images (127 faces found)"

=== UI LAYOUT (M2) ===
┌─────────────────────────────────────────────────────┐
│  Auto-Sorter 0.1.0                                  │
├─────────────────────────────────────────────────────┤
│ 📁 Photo folder: D:\\Photos               [Browse]  │
│ 📁 Data folder:  (auto)                   [Browse]  │
│                                                     │
│ [▶ Start Scan] [🔍 Detect Faces]                    │
│                                                     │
│ Progress: Detecting faces 42/58 (127 faces)         │
│ ████████████████████░░░░░░░░  72%                   │
│                                                     │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│ │thumb│ │thumb│ │thumb│ │thumb│ │thumb│ │thumb│  │
│ │  1  │ │  2  │ │  3  │ │  4  │ │  5  │ │  6  │  │
│ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  │
│ ┌─────┐ ┌─────┐ ┌─────┐ ...                        │
│                                                     │
│ Status: ✅ Detection complete: 127 faces in 58 imgs │
└─────────────────────────────────────────────────────┘

=== NEW CONCEPT: QFlowLayout (custom layout) ===
Qt doesn't have a built-in "flow layout" (like CSS flexbox wrap).
We simulate one by placing thumbnails in a grid inside a scroll area.
Each thumbnail is a clickable QLabel with a pixmap.
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
from autosorter.core.detect_worker import DetectWorker
from autosorter.core.scan_worker import ScanWorker
from autosorter.app.photo_viewer import PhotoViewer


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
    """The main application window with scan, detection, and photo grid."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(f"Auto-Sorter {__version__}")
        self.resize(900, 650)
        self.setMinimumSize(600, 400)

        # --- State ---
        self._photo_folder: Path | None = None
        self._data_folder: Path | None = None
        self._worker: ScanWorker | None = None
        self._detect_worker: DetectWorker | None = None
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

        # --- Action buttons row ---
        btn_row = QHBoxLayout()

        self._start_btn = QPushButton("▶  Start Scan")
        self._start_btn.setFixedHeight(36)
        self._start_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 20px;"
        )
        self._start_btn.clicked.connect(self._start_scan)
        self._start_btn.setEnabled(False)
        btn_row.addWidget(self._start_btn)

        # NEW in M2: Detect Faces button
        self._detect_btn = QPushButton("🔍  Detect Faces")
        self._detect_btn.setFixedHeight(36)
        self._detect_btn.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 4px 20px;"
        )
        self._detect_btn.clicked.connect(self._start_detection)
        self._detect_btn.setVisible(False)  # Hidden until scan completes
        btn_row.addWidget(self._detect_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.clicked.connect(self._cancel_action)
        self._cancel_btn.setVisible(False)
        btn_row.addWidget(self._cancel_btn)

        btn_row.addStretch()
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
            self._status_label.setText("Ready — click Start Scan.")

    def _pick_data_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Select data storage folder"
        )
        if folder:
            self._data_folder = Path(folder)
            self._data_path_label.setText(folder)
            self._data_path_label.setStyleSheet("")

    # ==================================================================
    # Scan control (M1)
    # ==================================================================

    def _start_scan(self) -> None:
        if self._photo_folder is None or self._data_folder is None:
            return

        self._worker = ScanWorker(self._photo_folder, self._data_folder)
        self._worker.scan_progress.connect(self._on_scan_progress)
        self._worker.thumb_progress.connect(self._on_thumb_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._worker.error.connect(self._on_error)

        self._start_btn.setEnabled(False)
        self._detect_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._scan_label.setVisible(True)
        self._scan_label.setText("Phase 1: Discovering images…")
        self._thumb_label.setVisible(False)
        self._status_label.setText("Scanning…")

        self._worker.start()

    # ==================================================================
    # Detection control (M2)
    # ==================================================================

    def _start_detection(self) -> None:
        """Launch face detection on all scanned images."""
        if self._data_folder is None:
            return

        self._detect_worker = DetectWorker(self._data_folder)
        self._detect_worker.progress.connect(self._on_detect_progress)
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.error.connect(self._on_error)

        self._start_btn.setEnabled(False)
        self._detect_btn.setEnabled(False)
        self._cancel_btn.setVisible(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._scan_label.setVisible(True)
        self._scan_label.setText("Detecting faces…")
        self._thumb_label.setVisible(False)
        self._status_label.setText("Running face detection…")

        self._detect_worker.start()

    def _cancel_action(self) -> None:
        """Cancel whichever worker is currently running."""
        if self._worker:
            self._worker.cancel()
        if self._detect_worker:
            self._detect_worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")
        self._status_label.setText("Cancelling… (finishing current file)")

    # ==================================================================
    # Signal handlers — Scan (M1)
    # ==================================================================

    def _on_scan_progress(self, found: int, new: int, skipped: int) -> None:
        self._scan_label.setText(
            f"Phase 1: Found {found} images ({new} new, {skipped} unchanged)"
        )

    def _on_thumb_progress(self, done: int, total: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._thumb_label.setVisible(True)
        self._thumb_label.setText(f"Phase 2: Thumbnails {done} / {total}")

    def _on_scan_finished(self, total: int, errors: int) -> None:
        self._start_btn.setEnabled(True)
        self._start_btn.setText("▶  Re-Scan")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setVisible(False)

        # Show the Detect Faces button
        self._detect_btn.setVisible(True)
        self._detect_btn.setEnabled(True)

        error_text = f", {errors} errors" if errors else ""
        self._status_label.setText(
            f"✅ Scan complete: {total} images found{error_text}. "
            f"Click 'Detect Faces' to find faces."
        )
        self._worker = None

        # Load the thumbnail grid
        self._load_thumbnail_grid()

    # ==================================================================
    # Signal handlers — Detection (M2)
    # ==================================================================

    def _on_detect_progress(self, done: int, total: int, faces: int) -> None:
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)
        self._scan_label.setText(
            f"Detecting faces: {done} / {total} images  ({faces} faces found)"
        )

    def _on_detect_finished(self, total_images: int, total_faces: int,
                            errors: int) -> None:
        self._start_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._detect_btn.setText("🔍  Re-Detect")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._progress_bar.setVisible(False)
        self._scan_label.setVisible(False)

        error_text = f", {errors} errors" if errors else ""
        self._status_label.setText(
            f"✅ Detection complete: {total_faces} faces found "
            f"in {total_images} images{error_text}. "
            f"Click any photo to see face boxes."
        )
        self._detect_worker = None

    # ==================================================================
    # Common signal handlers
    # ==================================================================

    def _on_error(self, message: str) -> None:
        self._start_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        self._cancel_btn.setVisible(False)
        self._progress_bar.setVisible(False)
        self._status_label.setText(f"❌ Error: {message}")
        self._status_label.setStyleSheet(
            "color: red; border-top: 1px solid #ddd; padding-top: 8px;"
        )
        self._worker = None
        self._detect_worker = None

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

        viewer = PhotoViewer(image_row["path"], faces)
        viewer.show()
        # Keep a reference so the window doesn't get garbage-collected
        self._viewers.append(viewer)
