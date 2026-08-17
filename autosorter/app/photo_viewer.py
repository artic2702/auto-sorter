"""
Photo viewer — displays a full-size image with CLICKABLE face bounding boxes.

=== WHAT CHANGED FROM M2? ===
M2: Face boxes were just visual overlays (painted onto the image).
M3: Face boxes are now CLICKABLE. Clicking a face opens the "Find Similar"
    panel showing the most similar faces ranked by cosine similarity.

=== HOW CLICKABLE REGIONS WORK ===
QPainter draws on a pixmap (flat image) — there's no concept of "clicking
a drawn rectangle." So we need a different approach:

APPROACH: Track face rectangles in display coordinates.
When the user clicks the image, we check if the click point falls inside
any of the face rectangles. If yes, we open the SimilarPanel for that face.

This is called HIT TESTING:
  click at (245, 180)
  face box 1: (100, 50, 200, 250) → 100 ≤ 245 ≤ 300, 50 ≤ 180 ≤ 300 → HIT!
  face box 2: (400, 100, 150, 200) → 400 ≤ 245? NO → miss
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


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

    WHY A SUBCLASS?
    QLabel doesn't know about face boxes — it just displays a pixmap.
    We subclass it to add:
    1. A list of face rectangles (in display coordinates)
    2. A mousePressEvent that checks if the click hit any face box
    3. A callback to open the SimilarPanel for the clicked face
    """

    def __init__(self, parent_viewer: "PhotoViewer", parent=None):
        super().__init__(parent)
        self._parent_viewer = parent_viewer
        # List of (QRect, face_row) pairs — populated when we draw boxes
        self._face_rects: List[Tuple[QRect, dict]] = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_face_rects(self, rects: List[Tuple[QRect, dict]]) -> None:
        """Store the face rectangles for hit testing."""
        self._face_rects = rects

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Check if the click landed on a face box.

        HIT TESTING: For each face rectangle, check if the click point
        (event.pos()) is inside the rectangle. The FIRST hit wins
        (faces are drawn in order, so the topmost visually wins).
        """
        if event.button() != Qt.MouseButton.LeftButton:
            return

        click_pos = event.pos()

        # Account for label alignment offset
        # The pixmap might be centered in the label if the label is larger
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return

        # Calculate offset (pixmap is centered in label)
        offset_x = (self.width() - pixmap.width()) // 2
        offset_y = (self.height() - pixmap.height()) // 2

        # Adjust click position relative to pixmap
        adjusted_x = click_pos.x() - offset_x
        adjusted_y = click_pos.y() - offset_y

        for rect, face_row in self._face_rects:
            if rect.contains(adjusted_x, adjusted_y):
                self._parent_viewer._on_face_clicked(face_row)
                return


class PhotoViewer(QWidget):
    """
    A window that displays a photo with clickable face detection overlays.

    M3 addition: clicking a face box opens the "Find Similar" panel.
    """

    def __init__(
        self,
        image_path: str,
        faces: list,
        data_folder: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            image_path:  Full path to the original image file.
            faces:       List of face rows from the database.
            data_folder: Path to .autosorter directory (for SimilarPanel).
            parent:      Parent widget (optional).
        """
        super().__init__(parent)

        self._image_path = image_path
        self._faces = [f for f in faces if f["det_score"] >= 0]
        self._data_folder = data_folder
        self._similar_panels: list = []  # Keep refs to open panels

        self.setWindowTitle(f"Photo Viewer — {Path(image_path).name}")
        self.resize(900, 700)
        self.setMinimumSize(400, 300)

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the viewer layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Info bar at the top ---
        info_bar = QHBoxLayout()
        info_bar.setContentsMargins(12, 8, 12, 8)

        face_count = len(self._faces)
        face_text = f"{face_count} face{'s' if face_count != 1 else ''} detected"
        click_hint = " — click a face box to find similar" if face_count > 0 else ""
        self._info_label = QLabel(
            f"📷 {Path(self._image_path).name}  —  {face_text}{click_hint}"
        )
        self._info_label.setStyleSheet("font-size: 13px;")
        info_bar.addWidget(self._info_label)

        info_bar.addStretch()

        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.close)
        info_bar.addWidget(close_btn)

        layout.addLayout(info_bar)

        # --- Scrollable image area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Use our clickable label instead of plain QLabel
        self._image_label = ClickableImageLabel(self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        scroll.setWidget(self._image_label)
        layout.addWidget(scroll)

        self._load_and_draw()

    def _load_and_draw(self) -> None:
        """Load the image, draw face overlays, and display it."""
        pixmap = QPixmap(self._image_path)

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

        # Track face rectangles for click detection
        face_rects: List[Tuple[QRect, dict]] = []

        if self._faces:
            painter = QPainter(display_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

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

                # Store the rectangle for hit testing
                face_rects.append((QRect(x, y, w, h), face))

                # Confidence label
                conf = face["det_score"]
                label_text = f"{conf:.0%}"
                painter.setPen(QPen(color))
                text_y = y - 5 if y > 20 else y + 18
                painter.drawText(x + 2, text_y, label_text)

                # "Click me" hint
                painter.drawText(x + 2, y + h + 15, "🔍 click")

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

            painter.end()

        # Pass the rectangles to the clickable label for hit testing
        self._image_label.set_face_rects(face_rects)
        self._image_label.setPixmap(display_pixmap)

    def _on_face_clicked(self, face_row) -> None:
        """Called when a face bounding box is clicked."""
        if self._data_folder is None:
            return

        face_id = face_row["id"]

        # Check if this face has an embedding
        if face_row["embedding"] is None:
            self._info_label.setText(
                "⚠️ This face has no embedding yet. Run 'Detect Faces' first."
            )
            return

        # Import here to avoid circular imports
        from autosorter.app.similar_panel import SimilarPanel

        panel = SimilarPanel(self._data_folder, face_id)
        panel.show()
        self._similar_panels.append(panel)

        self._info_label.setText(
            f"🔍 Showing similar faces for face #{face_id}"
        )
