import gc
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from autosorter.app.main_window import MainWindow
from autosorter.core.db import Database


@pytest.fixture
def main_window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    yield window
    window.deleteLater()
    app.sendPostedEvents(None, 0)
    app.processEvents()
    gc.collect()


def _setup_mock_library(folder: Path, num_photos: int = 5, num_persons: int = 2):
    data_dir = folder / ".autosorter"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "library.db"
    with Database(db_path) as db:
        for i in range(num_photos):
            img_id = db.upsert_image(
                str(folder / f"photo_{i}.jpg"),
                100,
                1.0,
                sha256=f"hash_{i}",
                thumb_path=str(data_dir / f"thumb_{i}.jpg"),
                status="processed",
            )
            # Add a face for the image
            face_id = db.insert_face(
                img_id,
                10, 10, 50, 50,
                0.95,
                landmarks="[]",
                quality=0.9,
                thumb_path=str(data_dir / f"face_{i}.jpg"),
            )
            if i < num_persons:
                centroid = b"\x00" * 512
                pid = db.insert_person(f"Person_{i}", centroid)
                db.assign_face_to_person(face_id, pid)
    return folder, data_dir


def test_photo_folder_switch_updates_data_folder_and_labels(tmp_path):
    app = QApplication.instance() or QApplication([])

    folder_a = tmp_path / "photos_a"
    folder_a.mkdir()
    folder_b = tmp_path / "photos_b"
    folder_b.mkdir()

    window = MainWindow()

    # Select Folder A
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_a)):
        window._pick_photo_folder()

    assert window._photo_folder == folder_a
    assert window._data_folder == folder_a / ".autosorter"
    assert window._photo_path_label.text() == str(folder_a)
    assert window._data_path_label.text() == str(folder_a / ".autosorter")

    # Select Folder B (simulating user picking a second folder)
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_b)):
        window._pick_photo_folder()

    # Crucial assertion: _data_folder must follow folder B!
    assert window._photo_folder == folder_b
    assert window._data_folder == folder_b / ".autosorter"
    assert window._photo_path_label.text() == str(folder_b)
    assert window._data_path_label.text() == str(folder_b / ".autosorter")
    window.close()
    app.processEvents()


def test_custom_data_folder_preserved_when_switching_photos(tmp_path):
    app = QApplication.instance() or QApplication([])

    folder_a = tmp_path / "photos_a"
    folder_a.mkdir()
    folder_b = tmp_path / "photos_b"
    folder_b.mkdir()
    custom_data = tmp_path / "custom_data"
    custom_data.mkdir()

    window = MainWindow()

    # Pick custom data folder
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(custom_data)):
        window._pick_data_folder()

    assert window._custom_data_folder is True
    assert window._data_folder == custom_data

    # Pick photo folder A
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_a)):
        window._pick_photo_folder()

    assert window._photo_folder == folder_a
    assert window._data_folder == custom_data  # Should stay custom

    # Pick photo folder B
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_b)):
        window._pick_photo_folder()

    assert window._photo_folder == folder_b
    assert window._data_folder == custom_data  # Still stays custom
    window.close()
    app.processEvents()


def test_switching_from_populated_to_empty_folder_clears_views(tmp_path):
    app = QApplication.instance() or QApplication([])

    folder_a = tmp_path / "photos_a"
    _setup_mock_library(folder_a, num_photos=4, num_persons=2)
    folder_b = tmp_path / "photos_b"
    folder_b.mkdir()  # empty, no database

    window = MainWindow()
    window.show()

    # Select populated Folder A
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_a)):
        window._pick_photo_folder()

    assert window._photo_grid.isVisible() is True
    assert window._people_container.isVisible() is True
    assert window._albums_container.isVisible() is True
    assert "photos" in window._library_label.text()

    # Switch to empty Folder B
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_b)):
        window._pick_photo_folder()

    # Views must be hidden / cleared
    assert window._photo_grid.isVisible() is False
    assert window._people_container.isVisible() is False
    assert window._albums_container.isVisible() is False
    assert window._library_label.isVisible() is False
    assert window._start_btn.text() == "▶  Start"
    window.close()
    app.processEvents()


def test_switching_between_two_populated_folders_updates_panels(tmp_path):
    app = QApplication.instance() or QApplication([])

    folder_a = tmp_path / "photos_a"
    _setup_mock_library(folder_a, num_photos=10, num_persons=3)

    folder_b = tmp_path / "photos_b"
    _setup_mock_library(folder_b, num_photos=4, num_persons=1)

    window = MainWindow()

    # Select Folder A
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_a)):
        window._pick_photo_folder()

    assert "3 people" in window._library_label.text()
    assert window._people_panel._data_folder == folder_a / ".autosorter"
    assert window._albums_panel._data_folder == folder_a / ".autosorter"

    # Select Folder B
    with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value=str(folder_b)):
        window._pick_photo_folder()

    # Summary and panels must reflect Folder B!
    assert "1 people" in window._library_label.text()
    assert window._people_panel._data_folder == folder_b / ".autosorter"
    assert window._albums_panel._data_folder == folder_b / ".autosorter"
    window.close()
    app.processEvents()
