"""
People panel — displays person cards with face thumbnails and rename.

=== WHAT IS THIS FILE? ===
The People panel shows all discovered persons as cards in a grid.
Each card has:
  - Cover face thumbnail (the best face crop)
  - Display name ("Person 1" or user-set name like "Arnav")
  - Photo count ("623 photos")
  - Rename button (or double-click the name)

Clicking a card expands it to show all face thumbnails for that person.

=== RENAME FLOW ===
The user can rename a person two ways:
  1. Click the "✏️" rename button
  2. Double-click the person's name

Both open an inline QLineEdit. On Enter or focus-lost:
  - db.rename_person(person_id, new_name) is called
  - The name is persisted immediately (survives restart)
  - The card updates to show the new name
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter.core.db import Database


# Size of face thumbnails on person cards
CARD_FACE_SIZE = 96

# Max face thumbs to show in the expanded person detail
MAX_DETAIL_FACES = 30


class PersonCard(QWidget):
    """
    A single person card showing cover face, name, and count.

    Signals:
        renamed(person_id, new_name): emitted when user renames a person
        clicked(person_id): emitted when the card is clicked
    """

    renamed = Signal(int, str)   # (person_id, new_name)
    clicked = Signal(int)        # (person_id)

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
        self.setFixedSize(160, 200)

        self._build_ui()
        self._apply_style(selected=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # --- Cover face thumbnail ---
        self._face_label = QLabel()
        self._face_label.setFixedSize(CARD_FACE_SIZE, CARD_FACE_SIZE)
        self._face_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._face_label.setStyleSheet(
            "border: 2px solid #555; border-radius: 48px; "
            "background-color: #2a2a2a;"
        )

        if self._cover_thumb_path:
            pixmap = QPixmap(self._cover_thumb_path)
            if not pixmap.isNull():
                # Scale and make circular-looking crop
                scaled = pixmap.scaled(
                    CARD_FACE_SIZE - 4, CARD_FACE_SIZE - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._face_label.setPixmap(scaled)
            else:
                self._face_label.setText("👤")
                self._face_label.setStyleSheet(
                    self._face_label.styleSheet() + "font-size: 36px;"
                )
        else:
            self._face_label.setText("👤")
            self._face_label.setStyleSheet(
                self._face_label.styleSheet() + "font-size: 36px;"
            )

        layout.addWidget(self._face_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Name label (double-click to edit) ---
        self._name_label = QLabel(self._display_name)
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #e0e0e0;"
        )
        self._name_label.setWordWrap(True)
        layout.addWidget(self._name_label)

        # --- Name editor (hidden by default) ---
        self._name_edit = QLineEdit(self._display_name)
        self._name_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_edit.setStyleSheet(
            "font-size: 13px; font-weight: bold; padding: 2px; "
            "border: 1px solid #4CAF50; border-radius: 3px; "
            "background-color: #1a1a1a; color: #e0e0e0;"
        )
        self._name_edit.setVisible(False)
        self._name_edit.returnPressed.connect(self._finish_rename)
        layout.addWidget(self._name_edit)

        # --- Count + rename row ---
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(4)

        count_text = f"{self._face_count} photo{'s' if self._face_count != 1 else ''}"
        self._count_label = QLabel(count_text)
        self._count_label.setStyleSheet("font-size: 11px; color: #888;")
        bottom_row.addWidget(self._count_label)

        bottom_row.addStretch()

        rename_btn = QPushButton("✏️")
        rename_btn.setFixedSize(24, 24)
        rename_btn.setToolTip("Rename this person")
        rename_btn.setStyleSheet(
            "border: none; font-size: 14px; padding: 0;"
        )
        rename_btn.clicked.connect(self._start_rename)
        bottom_row.addWidget(rename_btn)

        layout.addLayout(bottom_row)

    def _apply_style(self, selected: bool = False) -> None:
        """Apply card styling."""
        border = "#4CAF50" if selected else "#3a3a3a"
        self.setStyleSheet(
            f"PersonCard {{"
            f"  background-color: #2d2d2d;"
            f"  border: 2px solid {border};"
            f"  border-radius: 10px;"
            f"}}"
        )

    def _start_rename(self) -> None:
        """Show the name editor inline."""
        self._name_label.setVisible(False)
        self._name_edit.setText(self._display_name)
        self._name_edit.setVisible(True)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _finish_rename(self) -> None:
        """Commit the rename."""
        new_name = self._name_edit.text().strip()
        if not new_name:
            new_name = self._display_name  # revert to old name if empty

        self._display_name = new_name
        self._name_label.setText(new_name)
        self._name_edit.setVisible(False)
        self._name_label.setVisible(True)

        self.renamed.emit(self._person_id, new_name)

    def focusOutEvent(self, event) -> None:
        """Finish rename when focus leaves the card."""
        if self._name_edit.isVisible():
            self._finish_rename()
        super().focusOutEvent(event)

    def mousePressEvent(self, event) -> None:
        """Click → emit clicked signal."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._person_id)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click → start rename."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._start_rename()


class PersonDetailPanel(QWidget):
    """
    Shows all face thumbnails for a selected person.

    Opens as a popup/section below the person cards, showing a grid
    of face crops belonging to that person.
    """

    def __init__(
        self,
        person_row,
        faces: list,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._person = person_row
        self._faces = faces

        self.setWindowTitle(f"{person_row['display_name']} — Faces")
        self.resize(600, 400)
        self.setMinimumSize(300, 200)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Header
        header = QLabel(
            f"<b style='font-size: 16px;'>{self._person['display_name']}</b>"
            f"  <span style='color: gray;'>— {len(self._faces)} faces</span>"
        )
        layout.addWidget(header)

        # Face grid in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        grid.setSpacing(6)

        cols = 5
        shown = min(len(self._faces), MAX_DETAIL_FACES)
        for i in range(shown):
            face = self._faces[i]
            cell = self._make_face_cell(face)
            grid.addWidget(cell, i // cols, i % cols)

        if len(self._faces) > MAX_DETAIL_FACES:
            more = QLabel(
                f"<i style='color: gray;'>"
                f"… and {len(self._faces) - MAX_DETAIL_FACES} more faces</i>"
            )
            grid.addWidget(
                more, shown // cols + 1, 0, 1, cols,
                Qt.AlignmentFlag.AlignCenter,
            )

        scroll.setWidget(grid_container)
        layout.addWidget(scroll, stretch=1)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def _make_face_cell(self, face_row) -> QWidget:
        """Create a single face thumbnail cell."""
        cell = QWidget()
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(2, 2, 2, 2)
        cell_layout.setSpacing(2)

        thumb = QLabel()
        thumb.setFixedSize(80, 80)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet("border: 1px solid #444; border-radius: 3px;")

        if face_row["thumb_path"]:
            pixmap = QPixmap(face_row["thumb_path"])
            if not pixmap.isNull():
                thumb.setPixmap(pixmap.scaled(
                    76, 76,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            else:
                thumb.setText("❌")
        else:
            thumb.setText("❌")

        cell_layout.addWidget(thumb, alignment=Qt.AlignmentFlag.AlignCenter)

        # Quality score
        quality = face_row["quality"]
        q_text = f"Q: {quality:.0%}" if quality else "Q: N/A"
        q_label = QLabel(q_text)
        q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        q_label.setStyleSheet("font-size: 10px; color: #888;")
        cell_layout.addWidget(q_label)

        return cell


class PeoplePanel(QWidget):
    """
    Main people panel — shows person cards in a scrollable grid.

    Usage:
        panel = PeoplePanel(data_folder)
        panel.load_persons()  # reads from DB and populates the grid
    """

    def __init__(
        self,
        data_folder: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._data_folder = data_folder
        self._detail_windows: list = []  # Keep refs to open detail windows

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        self._title_label = QLabel(
            "<b style='font-size: 15px;'>👤 People</b>"
        )
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
        self._grid_layout.setSpacing(10)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._scroll.setWidget(self._grid_container)
        layout.addWidget(self._scroll, stretch=1)

    def load_persons(self) -> None:
        """Load all persons from the database and populate the card grid."""
        # Clear existing cards
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
            self._title_label.setText(
                "<b style='font-size: 15px;'>👤 People</b>"
            )
            return

        self._title_label.setText(
            f"<b style='font-size: 15px;'>👤 People</b>"
            f"  <span style='color: gray;'>({len(persons)} people found)</span>"
        )

        # Build person cards
        cols = max(1, (self.width() - 40) // 170)
        for i, person in enumerate(persons):
            # Get cover face thumb path
            cover_thumb = None
            if person["cover_face_id"]:
                cover_face = db.get_face_by_id(person["cover_face_id"])
                if cover_face:
                    cover_thumb = cover_face["thumb_path"]

            card = PersonCard(person, cover_thumb)
            card.renamed.connect(self._on_person_renamed)
            card.clicked.connect(self._on_person_clicked)

            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(card, row, col)

        db.close()

    def _on_person_renamed(self, person_id: int, new_name: str) -> None:
        """Handle person rename — persist to database."""
        db_path = self._data_folder / "library.db"
        db = Database(db_path)
        db.rename_person(person_id, new_name)
        db.close()

    def _on_person_clicked(self, person_id: int) -> None:
        """Handle person card click — show face detail panel."""
        db_path = self._data_folder / "library.db"
        db = Database(db_path)
        person = db.get_person_by_id(person_id)
        faces = db.get_faces_for_person(person_id)
        db.close()

        if person is None:
            return

        detail = PersonDetailPanel(person, faces)
        detail.show()
        self._detail_windows.append(detail)
