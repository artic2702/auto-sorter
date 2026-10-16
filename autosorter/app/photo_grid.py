"""
Virtualized photo grid using Qt Model/View architecture.

Replaces the one-widget-per-photo (QLabel) approach with a memory-efficient
QListView in IconMode, supporting 10,000-100,000+ photos with instant loading.
"""

from typing import Any, Optional, Sequence

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QPixmapCache
from PySide6.QtWidgets import QListView, QWidget

# Set 64 MB cache for thumbnails once at module import (64 * 1024 KB)
QPixmapCache.setCacheLimit(64 * 1024)

GRID_THUMB_SIZE = 120


class PhotoGridModel(QAbstractListModel):
    """Model backing the photo grid. Loads thumbnails lazily via QPixmapCache."""

    def __init__(self, rows: Sequence[Any], thumb_size: int = GRID_THUMB_SIZE) -> None:
        super().__init__()
        self._rows = rows
        self._ts = thumb_size
        self.data_call_count = 0

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        self.data_call_count += 1
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.DecorationRole:
            key = row["thumb_path"]
            if not key:
                return None
            key_str = str(key)
            pm = QPixmapCache.find(key_str)
            if pm is None or pm.isNull():
                pm = QPixmap(key_str)
                if pm.isNull():
                    return None
                pm = pm.scaled(
                    self._ts,
                    self._ts,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                QPixmapCache.insert(key_str, pm)
            return QIcon(pm)
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None


class PhotoGridView(QListView):
    """Virtualized photo grid view displaying thumbnails with uniform sizes."""

    photo_activated = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None, thumb_size: int = GRID_THUMB_SIZE) -> None:
        super().__init__(parent)
        self._ts = thumb_size
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setUniformItemSizes(True)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(QSize(self._ts, self._ts))
        self.setGridSize(QSize(self._ts + 12, self._ts + 12))
        self.setSpacing(6)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QListView {"
            "  background-color: #1e1e1e;"
            "  border: none;"
            "  outline: none;"
            "}"
            "QListView::item {"
            "  border: 2px solid transparent;"
            "  border-radius: 4px;"
            "}"
            "QListView::item:hover {"
            "  border: 2px solid #4CAF50;"
            "  border-radius: 4px;"
            "}"
            "QListView::item:selected {"
            "  border: 2px solid #2196F3;"
            "  border-radius: 4px;"
            "  background-color: transparent;"
            "}"
        )
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, index: QModelIndex) -> None:
        if index.isValid():
            row = index.data(Qt.ItemDataRole.UserRole)
            if row is not None:
                self.photo_activated.emit(row)
