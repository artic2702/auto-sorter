"""
Photo viewer — displays a full-size image with face bounding boxes overlaid.

=== WHAT IS THIS FILE? ===
When the user clicks a photo thumbnail in the grid, this viewer opens and
shows the full image with colored rectangles drawn around each detected face
and green dots on the facial landmarks (eyes, nose, mouth corners).

=== HOW QT DRAWS IMAGES ===
Qt uses a "pixmap" (QPixmap) to display images. A pixmap is the image data
in a format optimized for screen display. We:
  1. Load the image file into a QPixmap
  2. Scale it to fit the window while keeping aspect ratio
  3. Draw face boxes and landmarks on top using QPainter
  4. Display the result in a QLabel

=== WHAT IS QPainter? ===
QPainter is Qt's drawing tool. Think of it as a virtual pen:
  - painter.drawRect(x, y, w, h) draws a rectangle
  - painter.drawEllipse(x, y, w, h) draws a circle/ellipse
  - painter.setPen(color) changes the pen color/thickness
You "paint" on top of the pixmap, then display the combined result.
"""

import json
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# Colors for face bounding boxes (cycle through these for multiple faces)
# Using bright, distinct colors so each face is clearly visible
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


class PhotoViewer(QWidget):
    """
    A window that displays a photo with face detection overlays.

    Usage:
        viewer = PhotoViewer(image_path, faces_data)
        viewer.show()
    """

    def __init__(
        self,
        image_path: str,
        faces: list,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            image_path: Full path to the original image file.
            faces:      List of face rows from the database (sqlite3.Row objects).
                        Each has: bbox_x, bbox_y, bbox_w, bbox_h, landmarks, det_score
            parent:     Parent widget (optional).
        """
        super().__init__(parent)

        self._image_path = image_path
        self._faces = [f for f in faces if f["det_score"] >= 0]  # exclude sentinels

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
        self._info_label = QLabel(f"📷 {Path(self._image_path).name}  —  {face_text}")
        self._info_label.setStyleSheet("font-size: 13px;")
        info_bar.addWidget(self._info_label)

        info_bar.addStretch()

        close_btn = QPushButton("✕ Close")
        close_btn.clicked.connect(self.close)
        info_bar.addWidget(close_btn)

        layout.addLayout(info_bar)

        # --- Scrollable image area ---
        # QScrollArea provides scroll bars if the image is larger than the window
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # The image label displays the pixmap
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: #1a1a1a;")

        scroll.setWidget(self._image_label)
        layout.addWidget(scroll)

        # Load and display the image with face boxes
        self._load_and_draw()

    def _load_and_draw(self) -> None:
        """Load the image, draw face overlays, and display it."""
        # Load image into a QPixmap
        pixmap = QPixmap(self._image_path)

        if pixmap.isNull():
            self._image_label.setText("❌ Could not load image")
            return

        # Scale to fit the window while keeping aspect ratio
        # We limit the display size so huge images don't eat all RAM
        max_w = max(800, self.width() - 40)
        max_h = max(600, self.height() - 80)
        display_pixmap = pixmap.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Calculate the scale factor (original → display size)
        # We need this to map face coordinates from original image space
        # to display image space
        scale_x = display_pixmap.width() / pixmap.width()
        scale_y = display_pixmap.height() / pixmap.height()

        # Draw face boxes and landmarks on the display pixmap
        if self._faces:
            painter = QPainter(display_pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            for i, face in enumerate(self._faces):
                # Pick a color (cycle through the list)
                color = _FACE_COLORS[i % len(_FACE_COLORS)]

                # --- Draw bounding box ---
                pen = QPen(color)
                pen.setWidth(3)
                painter.setPen(pen)

                # Scale coordinates from original image to display size
                x = int(face["bbox_x"] * scale_x)
                y = int(face["bbox_y"] * scale_y)
                w = int(face["bbox_w"] * scale_x)
                h = int(face["bbox_h"] * scale_y)

                painter.drawRect(x, y, w, h)

                # --- Draw confidence label ---
                conf = face["det_score"]
                label_text = f"{conf:.0%}"
                painter.setPen(QPen(color))
                # Draw text above the box (or inside if at top edge)
                text_y = y - 5 if y > 20 else y + 18
                painter.drawText(x + 2, text_y, label_text)

                # --- Draw landmarks (small dots) ---
                if face["landmarks"]:
                    try:
                        landmarks = json.loads(face["landmarks"])
                        painter.setPen(QPen(QColor(0, 255, 0), 2))  # green dots
                        for lm_x, lm_y in landmarks:
                            # Scale landmark coordinates
                            px = int(lm_x * scale_x)
                            py = int(lm_y * scale_y)
                            painter.drawEllipse(px - 3, py - 3, 6, 6)
                    except (json.JSONDecodeError, TypeError):
                        pass

            painter.end()

        self._image_label.setPixmap(display_pixmap)
