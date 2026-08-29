"""
Albums panel — virtual smart albums + export UI.

=== WHAT IS THIS FILE? (M7) ===
Displays virtual smart albums as clickable cards:
  👥 Group Photos   — images with 2+ different people
  ❓ Unknown Faces  — faces not assigned to any person
  📄 Text & Docs    — screenshots, receipts, text images

Each album card shows an icon, title, and item count. Clicking opens
an AlbumDetailView with a scrollable grid of photos/faces.

Also provides the "📤 Export Library" button that launches the export
engine to copy photos into organized person folders.

=== UI LAYOUT ===
┌─────────────────────────────────────────────────────┐
│  📁 Albums                           [📤 Export]    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ 👥       │  │ ❓       │  │ 📄       │          │
│  │ Group    │  │ Unknown  │  │ Text &   │          │
│  │ Photos   │  │ Faces    │  │ Documents│          │
│  │  42 imgs │  │  15 faces│  │  8 imgs  │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter.core.db import Database
from autosorter.core.export_worker import ExportWorker


# ══════════════════════════════════════════════════════════════════════
# ALBUM CARD WIDGET
# ══════════════════════════════════════════════════════════════════════
# Each virtual album is displayed as a clickable card with an icon,
# title, and item count. Clicking opens the album detail view.

# Size of album card widgets (pixels)
ALBUM_CARD_WIDTH = 140
ALBUM_CARD_HEIGHT = 100


class AlbumCard(QWidget):
    """
    A clickable card representing a virtual smart album.

    Displays an emoji icon, album title, and item count. On click,
    emits the 'clicked' signal so the parent panel can open the
    detail view for this album type.
    """

    # Signal emitted when the card is clicked, carrying the album type string
    clicked = Signal(str)

    def __init__(
        self,
        album_type: str,
        icon: str,
        title: str,
        count: int,
        parent=None,
    ):
        super().__init__(parent)
        # Store the album type identifier (e.g., "group", "unknown", "docs")
        self._album_type = album_type

        # Fixed size for consistent grid layout
        self.setFixedSize(ALBUM_CARD_WIDTH, ALBUM_CARD_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Apply card styling — rounded corners, subtle border, hover effect
        self.setStyleSheet("""
            QWidget {
                background: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 8px;
            }
            QWidget:hover {
                background: #e8f0fe;
                border-color: #4285f4;
            }
        """)

        # Build the card layout: icon on top, title below, count at bottom
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Large emoji icon
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 28px; border: none; background: transparent;")
        layout.addWidget(icon_label)

        # Album title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #333; "
            "border: none; background: transparent;"
        )
        layout.addWidget(title_label)

        # Item count
        count_text = f"{count} {'items' if count != 1 else 'item'}"
        self._count_label = QLabel(count_text)
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_label.setStyleSheet(
            "font-size: 11px; color: #888; border: none; background: transparent;"
        )
        layout.addWidget(self._count_label)

    def mousePressEvent(self, event):
        """Handle mouse click — emit the clicked signal with album type."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._album_type)


# ══════════════════════════════════════════════════════════════════════
# ALBUM DETAIL VIEW
# ══════════════════════════════════════════════════════════════════════
# Opens as a dialog showing a scrollable grid of thumbnails for the
# selected album. Supports click-to-open in PhotoViewer.

# Size of thumbnails in the album detail grid
DETAIL_THUMB_SIZE = 120


class AlbumDetailView(QDialog):
    """
    A scrollable grid view showing all photos/faces in a virtual album.

    Displays thumbnails in a responsive grid layout. Clicking a thumbnail
    opens the full photo in the Photo Viewer (via the parent main window).
    """

    def __init__(
        self,
        title: str,
        items: list,
        data_folder: Path,
        is_face_album: bool = False,
        parent=None,
    ):
        """
        Args:
            title:          Window title (e.g., "👥 Group Photos").
            items:          List of image rows (or face rows for unknown album).
            data_folder:    Path to the .autosorter data folder.
            is_face_album:  If True, items are face rows; show face thumbs.
                            If False, items are image rows; show image thumbs.
            parent:         Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(700, 500)
        self.setMinimumSize(400, 300)

        self._data_folder = data_folder
        self._items = items
        self._is_face_album = is_face_album

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the album detail view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header with album title and item count
        header = QLabel(f"{self.windowTitle()}  —  {len(self._items)} items")
        header.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(header)

        # Scrollable grid of thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        grid_container = QWidget()
        self._grid_layout = QGridLayout(grid_container)
        self._grid_layout.setSpacing(6)
        scroll.setWidget(grid_container)
        layout.addWidget(scroll, stretch=1)

        # Populate the grid
        self._populate_grid()

    def _populate_grid(self) -> None:
        """Fill the grid with clickable thumbnail widgets."""
        # Calculate columns based on dialog width
        cols = max(1, (self.width() - 60) // (DETAIL_THUMB_SIZE + 6))

        for i, item in enumerate(self._items):
            row = i // cols
            col = i % cols

            # Create a thumbnail label
            thumb_label = QLabel()
            thumb_label.setFixedSize(DETAIL_THUMB_SIZE, DETAIL_THUMB_SIZE)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet(
                "border: 2px solid transparent; border-radius: 4px;"
            )

            # Load the appropriate thumbnail
            if self._is_face_album:
                # For unknown faces, show the face thumbnail
                thumb_path = item["thumb_path"] if item["thumb_path"] else None
            else:
                # For image albums, show the image thumbnail
                thumb_path = item["thumb_path"] if item["thumb_path"] else None

            if thumb_path:
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    scaled = pixmap.scaled(
                        DETAIL_THUMB_SIZE - 4, DETAIL_THUMB_SIZE - 4,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    thumb_label.setPixmap(scaled)
                else:
                    thumb_label.setText("❌")
            else:
                thumb_label.setText("❌")

            self._grid_layout.addWidget(thumb_label, row, col)

    def resizeEvent(self, event):
        """Reflow the grid when the dialog is resized."""
        super().resizeEvent(event)
        # Clear and repopulate with new column count
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._populate_grid()


# ══════════════════════════════════════════════════════════════════════
# EXPORT OPTIONS DIALOG
# ══════════════════════════════════════════════════════════════════════

class ExportOptionsDialog(QDialog):
    """
    Dialog for configuring export options before starting.

    Lets the user choose which categories to include in the export
    and select the destination folder.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📤 Export Library")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Description
        desc = QLabel(
            "Export your photo library into organized person folders.\n"
            "Source photos are NEVER modified — only copies are created."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; color: #555;")
        layout.addWidget(desc)

        # Checkboxes for export categories
        self.cb_group = QCheckBox("Include Group Photos folder")
        self.cb_group.setChecked(True)
        layout.addWidget(self.cb_group)

        self.cb_unknown = QCheckBox("Include Unknown faces folder")
        self.cb_unknown.setChecked(False)
        layout.addWidget(self.cb_unknown)

        self.cb_docs = QCheckBox("Include Text & Documents folder")
        self.cb_docs.setChecked(True)
        layout.addWidget(self.cb_docs)

        # Destination folder selection
        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Export to:"))
        self._dest_label = QLabel("(not selected)")
        self._dest_label.setStyleSheet("color: gray;")
        dest_row.addWidget(self._dest_label, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._pick_dest)
        dest_row.addWidget(browse_btn)
        layout.addLayout(dest_row)

        # OK / Cancel buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._ok_btn = QPushButton("📤 Export")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Store the selected destination path
        self.export_path: Optional[Path] = None

    def _pick_dest(self) -> None:
        """Open a folder picker for the export destination."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select export destination"
        )
        if folder:
            self.export_path = Path(folder)
            self._dest_label.setText(folder)
            self._dest_label.setStyleSheet("")
            self._ok_btn.setEnabled(True)


# ══════════════════════════════════════════════════════════════════════
# ALBUMS PANEL (main widget embedded in MainWindow)
# ══════════════════════════════════════════════════════════════════════

class AlbumsPanel(QWidget):
    """
    The albums panel shown in the main window — displays album cards
    and the export button.

    This panel is a compact horizontal row of album cards. Clicking a
    card opens a detail dialog. The export button launches the export
    engine with a progress bar.
    """

    def __init__(self, data_folder: Path, parent=None):
        super().__init__(parent)
        self._data_folder = data_folder
        self._export_worker: Optional[ExportWorker] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the albums panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(8)

        # Header row: "Albums" title + Export button
        header_row = QHBoxLayout()

        header_label = QLabel("📁 Albums")
        header_label.setStyleSheet("font-size: 15px; font-weight: bold;")
        header_row.addWidget(header_label)
        header_row.addStretch()

        self._export_btn = QPushButton("📤 Export Library")
        self._export_btn.setFixedHeight(30)
        self._export_btn.setStyleSheet(
            "font-size: 12px; padding: 4px 12px;"
        )
        self._export_btn.clicked.connect(self._on_export_clicked)
        header_row.addWidget(self._export_btn)

        layout.addLayout(header_row)

        # Album cards row — horizontal layout with the 3 album cards
        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(10)
        self._cards_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(self._cards_row)

        # Export progress bar (hidden by default)
        self._export_progress = QProgressBar()
        self._export_progress.setVisible(False)
        layout.addWidget(self._export_progress)

        self._export_label = QLabel("")
        self._export_label.setStyleSheet("font-size: 12px; color: #555;")
        self._export_label.setVisible(False)
        layout.addWidget(self._export_label)

    def load_albums(self) -> None:
        """
        Refresh the album cards with current counts from the database.

        Called after pipeline completion and whenever the library changes.
        """
        # Clear existing cards
        while self._cards_row.count():
            item = self._cards_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        try:
            db = Database(db_path)

            # Count items for each album
            group_count = len(db.get_group_photos())
            unknown_count = len(db.get_unknown_faces())
            docs_count = len(db.get_text_and_docs())

            db.close()
        except Exception:
            return

        # Create and add album cards
        if group_count > 0:
            card = AlbumCard("group", "👥", "Group\nPhotos", group_count)
            card.clicked.connect(self._on_album_clicked)
            self._cards_row.addWidget(card)

        if unknown_count > 0:
            card = AlbumCard("unknown", "❓", "Unknown\nFaces", unknown_count)
            card.clicked.connect(self._on_album_clicked)
            self._cards_row.addWidget(card)

        if docs_count > 0:
            card = AlbumCard("docs", "📄", "Text &\nDocs", docs_count)
            card.clicked.connect(self._on_album_clicked)
            self._cards_row.addWidget(card)

        # Add a stretch at the end so cards stay left-aligned
        self._cards_row.addStretch()

    def _on_album_clicked(self, album_type: str) -> None:
        """
        Handle album card click — open the detail view dialog.

        Fetches the appropriate items from the database and opens
        an AlbumDetailView dialog showing all photos/faces.
        """
        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        db = Database(db_path)

        if album_type == "group":
            items = db.get_group_photos()
            title = "👥 Group Photos"
            is_face = False
        elif album_type == "unknown":
            items = db.get_unknown_faces()
            title = "❓ Unknown Faces"
            is_face = True
        elif album_type == "docs":
            items = db.get_text_and_docs()
            title = "📄 Text & Documents"
            is_face = False
        else:
            db.close()
            return

        db.close()

        if not items:
            QMessageBox.information(self, title, "No items in this album.")
            return

        # Open the album detail view as a modal dialog
        dialog = AlbumDetailView(
            title=title,
            items=items,
            data_folder=self._data_folder,
            is_face_album=is_face,
            parent=self,
        )
        dialog.exec()

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    def _on_export_clicked(self) -> None:
        """
        Handle Export button click — show options dialog and start export.
        """
        # Show export options dialog
        dialog = ExportOptionsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        if dialog.export_path is None:
            return

        # Disable the export button while running
        self._export_btn.setEnabled(False)
        self._export_btn.setText("Exporting…")
        self._export_progress.setVisible(True)
        self._export_progress.setRange(0, 0)  # indeterminate initially
        self._export_label.setVisible(True)
        self._export_label.setText("Starting export…")

        # Create and start the export worker thread
        self._export_worker = ExportWorker(
            data_folder=self._data_folder,
            export_root=dialog.export_path,
            include_group=dialog.cb_group.isChecked(),
            include_unknown=dialog.cb_unknown.isChecked(),
            include_docs=dialog.cb_docs.isChecked(),
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.error.connect(self._on_export_error)
        self._export_worker.start()

    def _on_export_progress(self, done: int, total: int, detail: str) -> None:
        """Update progress bar during export."""
        if total > 0:
            self._export_progress.setRange(0, total)
            self._export_progress.setValue(done)
        self._export_label.setText(detail)

    def _on_export_finished(self, result: dict) -> None:
        """Handle export completion — show summary and re-enable button."""
        self._export_btn.setEnabled(True)
        self._export_btn.setText("📤 Export Library")
        self._export_progress.setVisible(False)
        self._export_label.setVisible(False)
        self._export_worker = None

        # Build a readable summary message
        parts = []
        if result["persons_exported"] > 0:
            parts.append(f"{result['persons_exported']} person folders")
        if result["files_copied"] > 0:
            parts.append(f"{result['files_copied']} files copied")
        if result["files_skipped"] > 0:
            parts.append(f"{result['files_skipped']} files skipped (already exported)")
        if result.get("cancelled"):
            parts.append("⚠️ Export was cancelled")

        summary = ". ".join(parts) if parts else "No files to export."

        QMessageBox.information(
            self,
            "📤 Export Complete",
            f"Export finished!\n\n{summary}",
        )

    def _on_export_error(self, message: str) -> None:
        """Handle export error."""
        self._export_btn.setEnabled(True)
        self._export_btn.setText("📤 Export Library")
        self._export_progress.setVisible(False)
        self._export_label.setVisible(False)
        self._export_worker = None

        QMessageBox.critical(
            self,
            "Export Error",
            f"Export failed:\n\n{message}",
        )
