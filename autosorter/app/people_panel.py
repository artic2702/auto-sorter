"""
People panel — displays person cards with face thumbnails, rename, merge, and corrections.

=== FEATURES IN M5 ===
1. Grid of Person Cards with large avatar thumbnails (110x110), photo count, rename (✏️), and merge (🔗).
2. Person Detail View:
   - Large face thumbnails (120x120) showing all photos without truncation.
   - Double-click any face to open the FULL original photo in the Photo Viewer!
   - Dedicated "🔍 Full Photo" button on every face cell.
   - Right-click any face:
       • 🔍 View Full Photo
       • ➕ Move to NEW Person Card... (creates a new section and shifts face)
       • ➡️ Move to Existing Person...
       • ❌ Not this person (Remove/Unassign)
       • ⭐ Set as Cover Face
   - "🔗 Merge with..." dialog with search filter and double-click to merge.
"""

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QIcon, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter.core.db import Database


# Size of face thumbnails on person cards
CARD_FACE_SIZE = 110

# Size of face thumbnails in the Person Detail panel
DETAIL_FACE_SIZE = 120


class MergePersonDialog(QDialog):
    """
    Dialog to select another person to merge into with live search filter and 1-click select.
    """

    def __init__(
        self,
        source_person_id: int,
        source_name: str,
        data_folder: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._source_id = source_person_id
        self._source_name = source_name
        self._data_folder = data_folder
        self._selected_target_id: Optional[int] = None

        self.setWindowTitle(f"Merge '{source_name}' into another person")
        self.resize(400, 480)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header_label = QLabel(
            f"Merge all photos of <b>{self._source_name}</b> into:"
        )
        header_label.setStyleSheet("font-size: 14px;")
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Search filter box
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("🔍 Type name to filter...")
        self._search_edit.setStyleSheet(
            "padding: 6px; font-size: 13px; background-color: #222; border: 1px solid #555; border-radius: 4px; color: #fff;"
        )
        self._search_edit.textChanged.connect(self._filter_list)
        layout.addWidget(self._search_edit)

        # List of persons
        self._list_widget = QListWidget()
        self._list_widget.setIconSize(QSize(44, 44))
        self._list_widget.setStyleSheet(
            "QListWidget { background-color: #1e1e1e; border: 1px solid #444; border-radius: 6px; padding: 4px; }"
            "QListWidget::item { padding: 8px; border-radius: 4px; color: #eee; font-size: 13px; margin-bottom: 2px; }"
            "QListWidget::item:selected { background-color: #2E7D32; color: white; font-weight: bold; }"
            "QListWidget::item:hover { background-color: #333; }"
        )
        self._list_widget.itemDoubleClicked.connect(self._on_merge_clicked)

        self._populate_list()
        layout.addWidget(self._list_widget, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("padding: 6px 16px; font-size: 13px;")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        merge_btn = QPushButton("🔗 Merge into Selected")
        merge_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; font-weight: bold; padding: 7px 20px; border-radius: 4px; font-size: 13px;"
        )
        merge_btn.clicked.connect(self._on_merge_clicked)
        btn_row.addWidget(merge_btn)

        layout.addLayout(btn_row)

    def _populate_list(self) -> None:
        self._list_widget.clear()
        db = Database(self._data_folder / "library.db")
        persons = db.get_all_persons()

        filter_text = self._search_edit.text().strip().lower()

        for p in persons:
            if p["id"] == self._source_id:
                continue

            name = p["display_name"]
            if filter_text and filter_text not in name.lower():
                continue

            count = p["face_count"]
            item_text = f"  {name}  ({count} photo{'s' if count != 1 else ''})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            item.setData(Qt.ItemDataRole.UserRole + 1, name)

            if p["cover_face_id"]:
                c_face = db.get_face_by_id(p["cover_face_id"])
                if c_face and c_face["thumb_path"] and Path(c_face["thumb_path"]).exists():
                    pix = QPixmap(c_face["thumb_path"])
                    if not pix.isNull():
                        item.setIcon(QIcon(pix))

            self._list_widget.addItem(item)

        db.close()

    def _filter_list(self) -> None:
        self._populate_list()

    def _on_merge_clicked(self) -> None:
        current_item = self._list_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Select Person", "Please select a person to merge into.")
            return

        target_id = current_item.data(Qt.ItemDataRole.UserRole)
        target_name = current_item.data(Qt.ItemDataRole.UserRole + 1)

        confirm = QMessageBox.question(
            self,
            "Confirm Merge",
            f"Are you sure you want to merge '{self._source_name}' into '{target_name}'?\n\n"
            f"All photos of '{self._source_name}' will be moved under '{target_name}'.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if confirm == QMessageBox.StandardButton.Yes:
            self._selected_target_id = target_id
            self.accept()

    def get_selected_target_id(self) -> Optional[int]:
        return self._selected_target_id


class MoveFaceDialog(QDialog):
    """
    Dialog to move a single face to an existing person or create a new person.
    """

    def __init__(
        self,
        current_person_id: Optional[int],
        data_folder: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._current_person_id = current_person_id
        self._data_folder = data_folder
        self._target_person_id: Optional[int] = None
        self._create_new_name: Optional[str] = None

        self.setWindowTitle("Move Face to Person")
        self.resize(380, 440)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Select destination person:"))

        self._list_widget = QListWidget()
        self._list_widget.setIconSize(QSize(40, 40))
        self._list_widget.setStyleSheet(
            "QListWidget { background-color: #222; border: 1px solid #444; border-radius: 6px; padding: 4px; }"
            "QListWidget::item { padding: 8px; border-radius: 4px; color: #eee; font-size: 13px; }"
            "QListWidget::item:selected { background-color: #1976D2; color: white; font-weight: bold; }"
        )
        self._list_widget.itemDoubleClicked.connect(self._on_move_clicked)

        # First option: Create New Person
        new_item = QListWidgetItem("➕  Create New Person...")
        new_item.setData(Qt.ItemDataRole.UserRole, -1)
        self._list_widget.addItem(new_item)

        db = Database(self._data_folder / "library.db")
        persons = db.get_all_persons()

        for p in persons:
            if p["id"] == self._current_person_id:
                continue

            count = p["face_count"]
            item_text = f"👤  {p['display_name']} ({count} photos)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, p["id"])

            if p["cover_face_id"]:
                c_face = db.get_face_by_id(p["cover_face_id"])
                if c_face and c_face["thumb_path"] and Path(c_face["thumb_path"]).exists():
                    pix = QPixmap(c_face["thumb_path"])
                    if not pix.isNull():
                        item.setIcon(QIcon(pix))

            self._list_widget.addItem(item)

        db.close()
        layout.addWidget(self._list_widget, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        move_btn = QPushButton("Move Face")
        move_btn.setStyleSheet(
            "background-color: #1976D2; color: white; font-weight: bold; padding: 6px 18px; border-radius: 4px;"
        )
        move_btn.clicked.connect(self._on_move_clicked)
        btn_row.addWidget(move_btn)

        layout.addLayout(btn_row)

    def _on_move_clicked(self) -> None:
        item = self._list_widget.currentItem()
        if not item:
            return

        val = item.data(Qt.ItemDataRole.UserRole)
        if val == -1:
            name, ok = QInputDialog.getText(self, "New Person", "Enter name for new person:")
            if ok and name.strip():
                self._create_new_name = name.strip()
                self._target_person_id = -1
                self.accept()
        else:
            self._target_person_id = val
            self.accept()

    def get_result(self) -> tuple[Optional[int], Optional[str]]:
        return self._target_person_id, self._create_new_name


class FaceCellWidget(QWidget):
    """
    Individual large face thumbnail cell with double-click to view full photo,
    dedicated "Full Photo" button, and right-click menu.
    """

    open_photo_requested = Signal(int)    # (face_id)
    unassign_requested = Signal(int)      # (face_id)
    reassign_requested = Signal(int)      # (face_id)
    create_new_person_requested = Signal(int)  # (face_id)
    set_cover_requested = Signal(int)     # (face_id)

    def __init__(self, face_row, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._face = face_row
        self._face_id = face_row["id"]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Image label - transparent to mouse events so parent cell gets double-click
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(DETAIL_FACE_SIZE, DETAIL_FACE_SIZE)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._thumb_label.setStyleSheet(
            "border: 2px solid #444; border-radius: 8px; background-color: #1e1e1e;"
        )

        if self._face["thumb_path"] and Path(self._face["thumb_path"]).exists():
            pix = QPixmap(self._face["thumb_path"])
            if not pix.isNull():
                self._thumb_label.setPixmap(
                    pix.scaled(
                        DETAIL_FACE_SIZE - 4, DETAIL_FACE_SIZE - 4,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                self._thumb_label.setText("👤")
        else:
            self._thumb_label.setText("👤")

        layout.addWidget(self._thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Dedicated "🔍 Full Photo" button
        open_btn = QPushButton("🔍 Full Photo")
        open_btn.setStyleSheet(
            "QPushButton { font-size: 11px; padding: 2px 6px; background-color: #2b2b2b; color: #ccc; border: 1px solid #444; border-radius: 3px; }"
            "QPushButton:hover { background-color: #388E3C; color: white; border-color: #4CAF50; }"
        )
        open_btn.clicked.connect(lambda: self.open_photo_requested.emit(self._face_id))
        layout.addWidget(open_btn)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click opens the full original photo!"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_photo_requested.emit(self._face_id)
            event.accept()

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #444; padding: 4px; }"
            "QMenu::item { padding: 7px 24px; font-size: 13px; border-radius: 3px; }"
            "QMenu::item:selected { background-color: #2E7D32; color: white; }"
            "QMenu::separator { height: 1px; background-color: #444; margin: 4px 0; }"
        )

        view_act = menu.addAction("🔍 View Full Original Photo")
        menu.addSeparator()
        new_person_act = menu.addAction("➕ Move to NEW Person Card...")
        move_act = menu.addAction("➡️ Move to Existing Person...")
        menu.addSeparator()
        unassign_act = menu.addAction("❌ Not this person (Remove)")
        cover_act = menu.addAction("⭐ Set as Cover Face")

        action = menu.exec(self.mapToGlobal(pos))
        if action == view_act:
            self.open_photo_requested.emit(self._face_id)
        elif action == new_person_act:
            self.create_new_person_requested.emit(self._face_id)
        elif action == move_act:
            self.reassign_requested.emit(self._face_id)
        elif action == unassign_act:
            self.unassign_requested.emit(self._face_id)
        elif action == cover_act:
            self.set_cover_requested.emit(self._face_id)


class PersonDetailPanel(QWidget):
    """
    Shows all face thumbnails for a selected person with correction controls and full-photo view.
    """

    changed = Signal()  # Emitted when faces are unassigned, moved, or person merged

    def __init__(
        self,
        person_row,
        faces: list,
        data_folder: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person = person_row
        self._faces = faces
        self._data_folder = data_folder
        self._viewers: list = []

        self.setWindowTitle(f"{person_row['display_name']} — Photos & Corrections")
        self.resize(900, 680)
        self.setMinimumSize(500, 400)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Header toolbar
        header = QHBoxLayout()
        self._title_label = QLabel(
            f"<b style='font-size: 18px;'>👤 {self._person['display_name']}</b>"
            f"  <span style='color: #aaa; font-size: 14px;'>— {len(self._faces)} photos</span>"
        )
        header.addWidget(self._title_label)
        header.addStretch()

        rename_btn = QPushButton("✏️ Rename")
        rename_btn.setStyleSheet(
            "background-color: #333; color: white; padding: 6px 14px; border-radius: 4px; font-size: 13px;"
        )
        rename_btn.clicked.connect(self._on_rename_clicked)
        header.addWidget(rename_btn)

        merge_btn = QPushButton("🔗 Merge with...")
        merge_btn.setToolTip("Merge this person card into another person")
        merge_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; padding: 6px 16px; border-radius: 4px; font-weight: bold; font-size: 13px;"
        )
        merge_btn.clicked.connect(self._on_merge_clicked)
        header.addWidget(merge_btn)

        recheck_btn = QPushButton("🔍 Re-check Album")
        recheck_btn.setToolTip("Scan this album for misplaced photos and automatically shift them to their rightful owners")
        recheck_btn.setStyleSheet(
            "background-color: #1565C0; color: white; padding: 6px 14px; border-radius: 4px; font-weight: bold; font-size: 13px;"
        )
        recheck_btn.clicked.connect(self._on_recheck_clicked)
        header.addWidget(recheck_btn)

        layout.addLayout(header)

        # Subtitle hint
        hint = QLabel(
            "<span style='color: #888; font-size: 12px;'>"
            "💡 <b>Double-click</b> any face (or click <b>🔍 Full Photo</b>) to open the original photo. "
            "<b>Right-click</b> to remove or shift to a new person card."
            "</span>"
        )
        layout.addWidget(hint)

        # Scrollable face grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #333; border-radius: 8px; background-color: #161616; }"
        )

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background-color: #161616;")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._populate_grid()

        self._scroll.setWidget(self._grid_container)
        layout.addWidget(self._scroll, stretch=1)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding: 6px 22px; font-size: 13px;")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _populate_grid(self) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = max(1, (self.width() - 60) // (DETAIL_FACE_SIZE + 24))
        for i, face in enumerate(self._faces):
            cell = FaceCellWidget(face)
            cell.open_photo_requested.connect(self._on_open_full_photo)
            cell.unassign_requested.connect(self._on_unassign_face)
            cell.reassign_requested.connect(self._on_reassign_face)
            cell.create_new_person_requested.connect(self._on_create_new_person_for_face)
            cell.set_cover_requested.connect(self._on_set_cover_face)
            self._grid_layout.addWidget(cell, i // cols, i % cols)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_grid_layout") and self._faces:
            self._populate_grid()

    def _reload_data(self) -> None:
        db = Database(self._data_folder / "library.db")
        person = db.get_person_by_id(self._person["id"])
        if not person:
            db.close()
            self.changed.emit()
            self.close()
            return

        self._person = person
        self._faces = list(db.get_faces_for_person(self._person["id"]))
        db.close()

        self._title_label.setText(
            f"<b style='font-size: 18px;'>👤 {self._person['display_name']}</b>"
            f"  <span style='color: #aaa; font-size: 14px;'>— {len(self._faces)} photos</span>"
        )
        self._populate_grid()
        self.changed.emit()

    def _on_open_full_photo(self, face_id: int) -> None:
        """Open the full original photo in PhotoViewer."""
        db = Database(self._data_folder / "library.db")
        face = db.get_face_by_id(face_id)
        if not face:
            db.close()
            return

        img_row = db._conn.execute(
            "SELECT * FROM images WHERE id = ?", (face["image_id"],)
        ).fetchone()
        if not img_row:
            db.close()
            return

        img_path = img_row["path"]
        faces = db.get_faces_for_image(face["image_id"])
        db.close()

        from autosorter.app.photo_viewer import PhotoViewer
        viewer = PhotoViewer(img_path, faces, data_folder=self._data_folder)
        viewer.changed.connect(self._reload_data)
        viewer.show()
        self._viewers.append(viewer)

    def _on_rename_clicked(self) -> None:
        new_name, ok = QInputDialog.getText(
            self, "Rename Person", "Enter new name for this person:", text=self._person["display_name"]
        )
        if ok and new_name.strip():
            db = Database(self._data_folder / "library.db")
            db.rename_person(self._person["id"], new_name.strip())
            db.close()
            self._reload_data()

    def _on_create_new_person_for_face(self, face_id: int) -> None:
        name, ok = QInputDialog.getText(
            self, "New Person Card", "Enter name for new person card:"
        )
        if ok and name.strip():
            db = Database(self._data_folder / "library.db")
            db.create_person_for_face(face_id, name.strip())
            db.close()
            self._reload_data()

    def _on_unassign_face(self, face_id: int) -> None:
        db = Database(self._data_folder / "library.db")
        db.unassign_face(face_id)
        db.close()
        self._reload_data()

    def _on_reassign_face(self, face_id: int) -> None:
        dialog = MoveFaceDialog(self._person["id"], self._data_folder, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_id, new_name = dialog.get_result()
            db = Database(self._data_folder / "library.db")
            if target_id == -1 and new_name:
                db.create_person_for_face(face_id, new_name)
            elif target_id and target_id > 0:
                db.reassign_face(face_id, target_id)
            db.close()
            self._reload_data()

    def _on_set_cover_face(self, face_id: int) -> None:
        db = Database(self._data_folder / "library.db")
        db.update_person_cover(self._person["id"], face_id)
        db.close()
        self._reload_data()

    def _on_merge_clicked(self) -> None:
        dialog = MergePersonDialog(
            self._person["id"], self._person["display_name"], self._data_folder, self
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_id = dialog.get_selected_target_id()
            if target_id:
                db = Database(self._data_folder / "library.db")
                db.merge_persons(source_person_id=self._person["id"], target_person_id=target_id)
                db.close()
                self.changed.emit()
                self.close()

    def _on_recheck_clicked(self) -> None:
        """Scan this album for misplaced faces and shift them to better matching persons."""
        from autosorter.core.cluster import clean_up_person_album
        db = Database(self._data_folder / "library.db")
        try:
            summary = clean_up_person_album(db, self._person["id"])
        finally:
            db.close()

        if not summary:
            QMessageBox.information(
                self,
                "Album Verified",
                f"All photos in {self._person['display_name']} are correctly placed!\nNo misfiled photos were found.",
            )
            return

        lines = [f"• Shifted {count} photo(s) to <b>{name}</b>" for name, count in summary.items()]
        msg = f"<b>Re-check complete!</b><br><br>" + "<br>".join(lines)
        QMessageBox.information(self, "Re-Check Complete", msg)
        self._reload_data()



class PersonCard(QWidget):
    """
    A single person card showing large cover face (110x110), name, count, and actions.
    """

    renamed = Signal(int, str)           # (person_id, new_name)
    clicked = Signal(int)                # (person_id)
    merge_requested = Signal(int, str)   # (person_id, display_name)

    def __init__(
        self,
        person_row,
        cover_thumb_path: Optional[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person_id = person_row["id"]
        self._display_name = person_row["display_name"]
        self._face_count = person_row["face_count"]
        self._cover_thumb_path = cover_thumb_path

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(175, 225)

        self._build_ui()
        self._apply_style(selected=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # --- Cover face thumbnail ---
        self._face_label = QLabel()
        self._face_label.setFixedSize(CARD_FACE_SIZE, CARD_FACE_SIZE)
        self._face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._face_label.setStyleSheet(
            "border: 2px solid #555; border-radius: 55px; background-color: #2a2a2a;"
        )

        if self._cover_thumb_path and Path(self._cover_thumb_path).exists():
            pixmap = QPixmap(self._cover_thumb_path)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    CARD_FACE_SIZE - 4, CARD_FACE_SIZE - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._face_label.setPixmap(scaled)
            else:
                self._face_label.setText("👤")
                self._face_label.setStyleSheet(self._face_label.styleSheet() + "font-size: 42px;")
        else:
            self._face_label.setText("👤")
            self._face_label.setStyleSheet(self._face_label.styleSheet() + "font-size: 42px;")

        layout.addWidget(self._face_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Name label ---
        self._name_label = QLabel(self._display_name)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e0e0e0;")
        self._name_label.setWordWrap(True)
        layout.addWidget(self._name_label)

        # --- Name editor ---
        self._name_edit = QLineEdit(self._display_name)
        self._name_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_edit.setStyleSheet(
            "font-size: 14px; font-weight: bold; padding: 3px; "
            "border: 1px solid #4CAF50; border-radius: 4px; "
            "background-color: #1a1a1a; color: #e0e0e0;"
        )
        self._name_edit.setVisible(False)
        self._name_edit.returnPressed.connect(self._finish_rename)
        layout.addWidget(self._name_edit)

        # --- Bottom toolbar: Count + Action buttons ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(4)

        count_text = f"{self._face_count} photo{'s' if self._face_count != 1 else ''}"
        self._count_label = QLabel(count_text)
        self._count_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._count_label.setStyleSheet("font-size: 12px; color: #888;")
        bottom_row.addWidget(self._count_label)

        bottom_row.addStretch()

        self._merge_btn = QPushButton("🔗")
        self._merge_btn.setFixedSize(24, 24)
        self._merge_btn.setToolTip("Merge this person into another")
        self._merge_btn.setStyleSheet("border: none; font-size: 14px; padding: 0; color: #aaa;")
        self._merge_btn.clicked.connect(self._on_merge_clicked)
        bottom_row.addWidget(self._merge_btn)

        self._rename_btn = QPushButton("✏️")
        self._rename_btn.setFixedSize(24, 24)
        self._rename_btn.setToolTip("Rename this person")
        self._rename_btn.setStyleSheet("border: none; font-size: 14px; padding: 0; color: #aaa;")
        self._rename_btn.clicked.connect(self._start_rename)
        bottom_row.addWidget(self._rename_btn)

        layout.addLayout(bottom_row)

    def _on_merge_clicked(self) -> None:
        self.merge_requested.emit(self._person_id, self._display_name)

    def _apply_style(self, selected: bool = False) -> None:
        border = "#4CAF50" if selected else "#3a3a3a"
        self.setStyleSheet(
            f"PersonCard {{ background-color: #2d2d2d; border: 2px solid {border}; border-radius: 12px; }}"
        )

    def _start_rename(self) -> None:
        self._name_label.setVisible(False)
        self._name_edit.setText(self._display_name)
        self._name_edit.setVisible(True)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_rename(self) -> None:
        new_name = self._name_edit.text().strip()
        if not new_name:
            new_name = self._display_name

        self._display_name = new_name
        self._name_label.setText(new_name)
        self._name_edit.setVisible(False)
        self._name_label.setVisible(True)

        self.renamed.emit(self._person_id, new_name)

    def focusOutEvent(self, event) -> None:
        if self._name_edit.isVisible():
            self._finish_rename()
        super().focusOutEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if click was on buttons
            pos = event.pos()
            if not self._merge_btn.geometry().contains(pos) and not self._rename_btn.geometry().contains(pos):
                self.clicked.emit(self._person_id)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_rename()


class PeoplePanel(QWidget):
    """
    Main people panel — shows person cards in a scrollable grid with merge & corrections.
    """

    def __init__(
        self,
        data_folder: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._data_folder = data_folder
        self._detail_windows: list = []

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        self._title_label = QLabel("<b style='font-size: 16px;'>👤 People</b>")
        header_row.addWidget(self._title_label)
        header_row.addStretch()

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.clicked.connect(self.load_persons)
        header_row.addWidget(refresh_btn)

        layout.addLayout(header_row)

        # Scrollable card grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")

        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setSpacing(12)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._scroll.setWidget(self._grid_container)
        layout.addWidget(self._scroll, stretch=1)

    def load_persons(self) -> None:
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        db_path = self._data_folder / "library.db"
        if not db_path.is_file():
            return

        db = Database(db_path)
        persons = db.get_all_persons()

        if not persons:
            placeholder = QLabel(
                "<span style='color: gray; font-size: 13px;'>"
                "No people discovered yet. Run 'Identify People' first."
                "</span>"
            )
            self._grid_layout.addWidget(placeholder, 0, 0)
            db.close()
            self._title_label.setText("<b style='font-size: 16px;'>👤 People</b>")
            return

        self._title_label.setText(
            f"<b style='font-size: 16px;'>👤 People</b>"
            f"  <span style='color: gray;'>({len(persons)} people found)</span>"
        )

        cols = max(1, (self.width() - 40) // 185)
        for i, person in enumerate(persons):
            cover_thumb = None
            if person["cover_face_id"]:
                cover_face = db.get_face_by_id(person["cover_face_id"])
                if cover_face:
                    cover_thumb = cover_face["thumb_path"]

            card = PersonCard(person, cover_thumb)
            card.renamed.connect(self._on_person_renamed)
            card.clicked.connect(self._on_person_clicked)
            card.merge_requested.connect(self._on_card_merge_requested)

            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(card, row, col)

        db.close()

    def _on_person_renamed(self, person_id: int, new_name: str) -> None:
        db = Database(self._data_folder / "library.db")
        db.rename_person(person_id, new_name)
        db.close()

    def _on_person_clicked(self, person_id: int) -> None:
        db = Database(self._data_folder / "library.db")
        person = db.get_person_by_id(person_id)
        faces = db.get_faces_for_person(person_id)
        db.close()

        if person is None:
            return

        detail = PersonDetailPanel(person, faces, self._data_folder)
        detail.changed.connect(self.load_persons)
        detail.show()
        self._detail_windows.append(detail)

    def _on_card_merge_requested(self, person_id: int, display_name: str) -> None:
        dialog = MergePersonDialog(person_id, display_name, self._data_folder, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_id = dialog.get_selected_target_id()
            if target_id:
                db = Database(self._data_folder / "library.db")
                db.merge_persons(source_person_id=person_id, target_person_id=target_id)
                db.close()
                self.load_persons()
