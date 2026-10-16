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
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
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
        self.resize(720, 520)
        self.setMinimumSize(400, 300)

        self._data_folder = data_folder
        self._items = items
        self._is_face_album = is_face_album
        self._current_cols = -1
        self._viewers: list = []

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the album detail view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Header with album title and item count
        header = QLabel(f"{self.windowTitle()}  —  {len(self._items)} items")
        header.setStyleSheet("font-size: 15px; font-weight: bold;")
        layout.addWidget(header)

        # Helper hint
        hint = QLabel("<span style='color: #888; font-size: 11px;'>💡 Click any photo to view in full resolution</span>")
        layout.addWidget(hint)

        # Scrollable grid of thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(8)
        self._grid_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._grid_container)
        layout.addWidget(scroll, stretch=1)

        # Populate initial grid
        self._populate_grid()

    def _populate_grid(self) -> None:
        """
        Fill the grid with clickable thumbnail widgets.

        Avoids ghost/overlapping widget artifacts by:
        1. Checking if column count actually changed before rebuilding.
        2. Hiding and unparenting previous widgets immediately before deleteLater.
        """
        # Calculate columns based on dialog width
        cols = max(1, (self.width() - 50) // (DETAIL_THUMB_SIZE + 8))

        # Only rebuild if column count changed or grid is empty
        if cols == self._current_cols and self._grid_layout.count() > 0:
            return
        self._current_cols = cols

        # Immediately hide and unparent previous widgets to prevent visual ghosts
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        for i, item in enumerate(self._items):
            row = i // cols
            col = i % cols

            # Create a clickable thumbnail label
            thumb_label = QLabel()
            thumb_label.setFixedSize(DETAIL_THUMB_SIZE, DETAIL_THUMB_SIZE)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setCursor(Qt.CursorShape.PointingHandCursor)
            thumb_label.setStyleSheet(
                "border: 1px solid #ccc; border-radius: 6px; background: #111;"
            )

            # Load the appropriate thumbnail
            if self._is_face_album:
                thumb_path = item["thumb_path"] if item["thumb_path"] else None
            else:
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

            # Click thumbnail to open full photo in PhotoViewer
            # sqlite3.Row doesn't have .get() — convert to dict first
            item_dict = dict(item)
            image_path = item_dict.get("path") if not self._is_face_album else item_dict.get("image_path")
            image_id = item_dict.get("id") if not self._is_face_album else item_dict.get("image_id")
            if image_path and image_id:
                thumb_label.mousePressEvent = lambda ev, ip=image_path, iid=image_id: (
                    self._open_photo_viewer(ip, iid) if ev.button() == Qt.MouseButton.LeftButton else None
                )

            self._grid_layout.addWidget(thumb_label, row, col)

    def _open_photo_viewer(self, image_path: str, image_id: int) -> None:
        """Open the full photo in PhotoViewer when clicked."""
        try:
            db_path = self._data_folder / "library.db"
            with Database(db_path) as db:
                faces = db.get_faces_for_image(image_id)

            from autosorter.app.photo_viewer import PhotoViewer
            viewer = PhotoViewer(image_path, faces, data_folder=self._data_folder)
            viewer.show()
            self._viewers.append(viewer)
        except Exception:
            pass

    def resizeEvent(self, event):
        """Reflow the grid when the dialog is resized."""
        super().resizeEvent(event)
        self._populate_grid()


# ══════════════════════════════════════════════════════════════════════
# EXPORT OPTIONS DIALOG
# ══════════════════════════════════════════════════════════════════════

class ExportOptionsDialog(QDialog):
    """
    Dialog for configuring export options before starting.

    Lets the user choose:
    1. Which people to export:
       - 'All people' (default): exports everyone found in the library
       - 'Selected people only': checkable list of specific people to export
    2. Which smart albums to include (Group Photos, Unknown, Docs)
    3. Destination folder on disk
    """

    def __init__(self, data_folder: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📤 Export Library")
        self.setMinimumWidth(460)

        self._data_folder = data_folder
        self._persons: list = []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # User-facing explanation
        desc = QLabel(
            "Export your photo library into organized person folders.\n"
            "Source photos are NEVER modified — only copies are created."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 13px; color: #555;")
        layout.addWidget(desc)

        # ── Section 1: People selection (All vs Specific People) ──
        people_header = QLabel("<b>People to Export:</b>")
        people_header.setStyleSheet("font-size: 13px; color: #222; margin-top: 4px;")
        layout.addWidget(people_header)

        # Radio button choices
        self._rb_all_people = QRadioButton("Export all people")
        self._rb_all_people.setChecked(True)
        self._rb_selected_people = QRadioButton("Export selected people only")

        self._people_btn_group = QButtonGroup(self)
        self._people_btn_group.addButton(self._rb_all_people)
        self._people_btn_group.addButton(self._rb_selected_people)

        layout.addWidget(self._rb_all_people)
        layout.addWidget(self._rb_selected_people)

        # Container for the checkable person list (shown only when "Selected" radio is active)
        self._people_list_container = QWidget()
        list_layout = QVBoxLayout(self._people_list_container)
        list_layout.setContentsMargins(18, 2, 0, 4)
        list_layout.setSpacing(4)

        # Quick action buttons: Select All / Deselect All
        sel_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.setFixedHeight(24)
        select_all_btn.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        select_all_btn.clicked.connect(self._select_all_people)
        sel_row.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.setFixedHeight(24)
        deselect_all_btn.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        deselect_all_btn.clicked.connect(self._deselect_all_people)
        sel_row.addWidget(deselect_all_btn)
        sel_row.addStretch()
        list_layout.addLayout(sel_row)

        # Scrollable list with checkboxes for each person
        self._people_list = QListWidget()
        self._people_list.setMaximumHeight(140)
        self._people_list.itemChanged.connect(lambda _: self._update_ok_button())
        list_layout.addWidget(self._people_list)
        layout.addWidget(self._people_list_container)

        # Hidden by default because "All people" is pre-selected
        self._people_list_container.setVisible(False)

        # Connect radio toggles to show/hide the person checklist
        self._rb_all_people.toggled.connect(self._on_people_toggle)
        self._rb_selected_people.toggled.connect(self._on_people_toggle)

        # Load persons from database into the checklist
        self._load_persons()

        # ── Section 2: Smart Albums Checkboxes ──
        albums_header = QLabel("<b>Smart Albums to Include:</b>")
        albums_header.setStyleSheet("font-size: 13px; color: #222; margin-top: 4px;")
        layout.addWidget(albums_header)

        self.cb_group = QCheckBox("Include Group Photos folder")
        self.cb_group.setChecked(True)
        layout.addWidget(self.cb_group)

        self.cb_unknown = QCheckBox("Include Unknown faces folder")
        self.cb_unknown.setChecked(False)
        layout.addWidget(self.cb_unknown)

        self.cb_docs = QCheckBox("Include Text & Documents folder")
        self.cb_docs.setChecked(True)
        layout.addWidget(self.cb_docs)

        # ── Section 3: Destination Folder Selection ──
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

        # ── Action Buttons: OK (Export) / Cancel ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._ok_btn = QPushButton("📤 Export")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self._on_ok_clicked)
        btn_row.addWidget(self._ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Selected destination folder
        self.export_path: Optional[Path] = None

    def _load_persons(self) -> None:
        """Load all visible persons from SQLite and populate the checklist."""
        if self._data_folder is None:
            return

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        try:
            with Database(db_path) as db:
                self._persons = db.get_all_persons()
        except Exception:
            return

        # Update radio label with the total count of persons discovered
        self._rb_all_people.setText(f"Export all people ({len(self._persons)} people)")

        # Fill the checklist with person display names and photo counts
        self._people_list.clear()
        for p in self._persons:
            count = p["face_count"]
            item = QListWidgetItem(f"{p['display_name']} ({count} {'photos' if count != 1 else 'photo'})")
            # Store the person database ID in custom user role
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            item.setCheckState(Qt.CheckState.Checked)
            self._people_list.addItem(item)

    def _on_people_toggle(self) -> None:
        """Show checklist when 'Selected only' is active; hide when 'All' is active."""
        is_selected = self._rb_selected_people.isChecked()
        self._people_list_container.setVisible(is_selected)
        self._update_ok_button()

    def _select_all_people(self) -> None:
        """Check all people in the list."""
        for i in range(self._people_list.count()):
            self._people_list.item(i).setCheckState(Qt.CheckState.Checked)
        self._update_ok_button()

    def _deselect_all_people(self) -> None:
        """Uncheck all people in the list."""
        for i in range(self._people_list.count()):
            self._people_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._update_ok_button()

    def selected_person_ids(self) -> Optional[list]:
        """
        Return the list of selected person IDs.

        Returns:
            None: if 'Export all people' is selected (meaning export everybody).
            list[int]: if 'Export selected only' is selected (only the checked IDs).
        """
        if self._rb_all_people.isChecked():
            return None

        # Gather IDs of all checked items
        selected_ids = []
        for i in range(self._people_list.count()):
            item = self._people_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_ids.append(item.data(Qt.ItemDataRole.UserRole))
        return selected_ids

    def _update_ok_button(self) -> None:
        """Enable Export button only if a destination is picked and people are selected."""
        has_dest = self.export_path is not None
        if self._rb_selected_people.isChecked():
            # In selective mode, make sure at least one person is checked
            has_people = any(
                self._people_list.item(i).checkState() == Qt.CheckState.Checked
                for i in range(self._people_list.count())
            )
            self._ok_btn.setEnabled(has_dest and has_people)
        else:
            self._ok_btn.setEnabled(has_dest)

    def _pick_dest(self) -> None:
        """Open a folder picker for the export destination."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select export destination"
        )
        if folder:
            self.export_path = Path(folder)
            self._dest_label.setText(folder)
            self._dest_label.setStyleSheet("")
            self._update_ok_button()

    def _on_ok_clicked(self) -> None:
        """Validate selections before closing dialog."""
        if self._rb_selected_people.isChecked():
            chosen = self.selected_person_ids()
            if not chosen:
                QMessageBox.warning(self, "No People Selected", "Please check at least one person to export.")
                return
        self.accept()


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
            with Database(db_path) as db:
                # Count items for each album
                group_count = len(db.get_group_photos())
                unknown_count = len(db.get_unknown_faces())
                docs_count = len(db.get_text_and_docs())
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

        with Database(db_path) as db:
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
                return

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
        # Show export options dialog configured with current data folder
        dialog = ExportOptionsDialog(data_folder=self._data_folder, parent=self)
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

        # Create and start the export worker thread with selective person filter
        self._export_worker = ExportWorker(
            data_folder=self._data_folder,
            export_root=dialog.export_path,
            person_ids=dialog.selected_person_ids(),
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
