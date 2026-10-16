"""Tests for virtualized photo grid model and view."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel

from autosorter.app.photo_grid import PhotoGridModel, PhotoGridView


def test_photo_grid_virtualization():
    app = QApplication.instance() or QApplication([])

    total_rows = 50000
    fake_rows = [
        {"id": i, "thumb_path": f"/fake/thumb_{i}.jpg", "path": f"/fake/img_{i}.jpg"}
        for i in range(total_rows)
    ]

    model = PhotoGridModel(fake_rows, thumb_size=120)
    assert model.rowCount() == total_rows

    view = PhotoGridView()
    view.setModel(model)
    view.resize(1000, 800)
    view.show()
    app.processEvents()

    # Zero QLabel child widgets created
    labels = view.findChildren(QLabel)
    assert len(labels) == 0

    # data() called far fewer than 50,000 times (< 2,000)
    assert model.data_call_count < 2000

    # Signal test: photo_activated fires with the right row on click
    activated_rows = []
    view.photo_activated.connect(lambda row: activated_rows.append(row))

    test_idx = model.index(5)
    view.clicked.emit(test_idx)

    assert len(activated_rows) == 1
    assert activated_rows[0]["id"] == 5
    assert activated_rows[0]["path"] == "/fake/img_5.jpg"
