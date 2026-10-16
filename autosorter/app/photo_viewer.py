"""
Photo viewer — displays a full-size image with interactive face bounding boxes.

=== WHAT CHANGED IN M5? ===
M5: Clicking a face bounding box opens an action menu:
  - View assigned person name
  - "Assign / Move to Person..."
  - "Not this person" (unassign)
  - "Set as Cover Face"
  - "Find Similar Faces"
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from autosorter.core.db import Database
from autosorter.core.thumbnailer import _read_image_safe


# Colors for face bounding boxes (cycle through these for multiple faces)
_FACE_COLORS = [
    QColor(0, 255, 100),    # green
    QColor(255, 100, 0),    # orange
    QColor(0, 150, 255),    # blue
    QColor(255, 50, 200),   # pink
    QColor(255, 255, 0),    # yellow
    QColor(150, 0, 255),    # purple
    QColor(0, 255, 255),    # cyan
    QColor(255, 0, 0),      # red
]


class ClickableImageLabel(QLabel):
    """
    A QLabel that detects clicks on face bounding boxes.
    """

    def __init__(self, parent_viewer: "PhotoViewer", parent=None):
        super().__init__(parent)
        self._parent_viewer = parent_viewer
        self._face_rects: List[Tuple[QRect, dict]] = []
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_face_rects(self, rects: List[Tuple[QRect, dict]]) -> None:
        """Store the face rectangles for hit testing."""
        self._face_rects = rects

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Check if the click landed on a face box."""
        click_pos = event.pos()

        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        offset_x = (self.width() - pixmap.width()) // 2
        offset_y = (self.height() - pixmap.height()) // 2

        adjusted_x = click_pos.x() - offset_x
        adjusted_y = click_pos.y() - offset_y

        for rect, face_row in self._face_rects:
            if rect.contains(adjusted_x, adjusted_y):
                self._parent_viewer._on_face_clicked(face_row, event.globalPosition().toPoint())
                return


class PhotoViewer(QWidget):
    """
    A window that displays a photo with interactive face detection overlays.
    """

    changed = Signal()  # Emitted when face assignments are changed

    def __init__(
        self,
        image_path: str,
        faces: list,
        data_folder: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self._image_path = image_path
        self._faces = [f for f in faces if f["det_score"] >= 0]
        self._data_folder = data_folder
        self._similar_panels: list = []

        self.setWindowTitle(f"Photo Viewer — {Path(image_path).name}")
        self.resize(920, 720)
        self.setMinimumSize(400, 300)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info bar
        info_bar = QHBoxLayout()
        info_bar.setContentsMargins(12, 8, 12, 8)

        face_count = len(self._faces)
        face_text = f"{face_count} face{'s' if face_count != 1 else ''} detected"
        click_hint = " — click any face box to view or reassign person" if face_count > 0 else ""
        self._info_label = QLabel(
            f"📷 <b>{Path(self._image_path).name}</b>  —  {face_text}{click_hint}"
        )
        self._info_label.setStyleSheet("font-size: 13px;")
        info_bar.addWidget(self._info_label)

        info_bar.addStretch()

        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.close)
        info_bar.addWidget(close_btn)

        layout.addLayout(info_bar)

        # Scrollable image area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = ClickableImageLabel(self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        scroll.setWidget(self._image_label)
        layout.addWidget(scroll)

        self._load_and_draw()

    def _load_and_draw(self) -> None:
        img = _read_image_safe(self._image_path)
        if img is None:
            self._image_label.setText("❌ Could not load image")
            return

        h, w = img.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg)

        if pixmap.isNull():
            self._image_label.setText("❌ Could not load image")
            return

        max_w = max(800, self.width() - 40)
        max_h = max(600, self.height() - 80)
        display_pixmap = pixmap.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        scale_x = display_pixmap.width() / pixmap.width()
        scale_y = display_pixmap.height() / pixmap.height()

        face_rects: List[Tuple[QRect, dict]] = []

        # Refresh faces from DB to get latest person assignments
        if self._data_folder:
            db_path = self._data_folder / "library.db"
            if db_path.exists():
                db = Database(db_path)
                fresh_faces = []
                for f in self._faces:
                    refreshed = db.get_face_by_id(f["id"])
                    fresh_faces.append(refreshed if refreshed else f)
                self._faces = fresh_faces
                db.close()

        if self._faces:
            painter = QPainter(display_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            db = Database(self._data_folder / "library.db") if self._data_folder else None

            for i, face in enumerate(self._faces):
                color = _FACE_COLORS[i % len(_FACE_COLORS)]
                pen = QPen(color)
                pen.setWidth(3)
                painter.setPen(pen)

                x = int(face["bbox_x"] * scale_x)
                y = int(face["bbox_y"] * scale_y)
                w = int(face["bbox_w"] * scale_x)
                h = int(face["bbox_h"] * scale_y)

                painter.drawRect(x, y, w, h)
                face_rects.append((QRect(x, y, w, h), face))

                # Person name or confidence tag
                person_name = None
                if db and face["person_id"]:
                    p = db.get_person_by_id(face["person_id"])
                    if p:
                        person_name = p["display_name"]

                tag_text = person_name if person_name else f"{face['det_score']:.0%}"
                painter.setPen(QPen(color))
                text_y = y - 6 if y > 22 else y + 16
                painter.drawText(x + 2, text_y, tag_text)

                # Landmarks
                if face["landmarks"]:
                    try:
                        landmarks = json.loads(face["landmarks"])
                        painter.setPen(QPen(QColor(0, 255, 0), 2))
                        for lm_x, lm_y in landmarks:
                            px = int(lm_x * scale_x)
                            py = int(lm_y * scale_y)
                            painter.drawEllipse(px - 3, py - 3, 6, 6)
                    except (json.JSONDecodeError, TypeError):
                        pass

            if db:
                db.close()

            painter.end()

        self._image_label.set_face_rects(face_rects)
        self._image_label.setPixmap(display_pixmap)

    def _on_face_clicked(self, face_row: dict, global_pos: QPoint) -> None:
        """Called when a face bounding box is clicked."""
        if not self._data_folder:
            return

        face_id = face_row["id"]
        db = Database(self._data_folder / "library.db")
        fresh_face = db.get_face_by_id(face_id)
        if not fresh_face:
            db.close()
            return

        person_name = "Unassigned"
        if fresh_face["person_id"]:
            person = db.get_person_by_id(fresh_face["person_id"])
            if person:
                person_name = person["display_name"]
        db.close()

        # Build context popup menu
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #444; }"
            "QMenu::item { padding: 6px 20px; }"
            "QMenu::item:selected { background-color: #4CAF50; color: white; }"
        )

        title_act = menu.addAction(f"👤 Person: {person_name}")
        title_act.setEnabled(False)
        menu.addSeparator()

        assign_act = menu.addAction("➡️ Assign / Move to Person...")
        unassign_act = menu.addAction("❌ Remove from Person (Unassign)") if fresh_face["person_id"] else None
        cover_act = menu.addAction("⭐ Set as Cover Face") if fresh_face["person_id"] else None
        menu.addSeparator()
        similar_act = menu.addAction("🔍 Find Similar Faces")

        action = menu.exec(global_pos)

        if action == assign_act:
            from autosorter.app.people_panel import MoveFaceDialog
            dialog = MoveFaceDialog(fresh_face["person_id"], self._data_folder, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                target_id, new_name = dialog.get_result()
                db = Database(self._data_folder / "library.db")
                if target_id == -1 and new_name:
                    db.create_person_for_face(face_id, new_name)
                elif target_id and target_id > 0:
                    db.reassign_face(face_id, target_id)
                db.close()
                self._load_and_draw()
                self.changed.emit()

        elif action == unassign_act and unassign_act is not None:
            db = Database(self._data_folder / "library.db")
            db.unassign_face(face_id)
            db.close()
            self._load_and_draw()
            self.changed.emit()

        elif action == cover_act and cover_act is not None:
            db = Database(self._data_folder / "library.db")
            db.update_person_cover(fresh_face["person_id"], face_id)
            db.close()
            self._load_and_draw()
            self.changed.emit()

        elif action == similar_act:
            from autosorter.app.similar_panel import SimilarPanel
            panel = SimilarPanel(self._data_folder, face_id)
            panel.show()
            self._similar_panels.append(panel)
