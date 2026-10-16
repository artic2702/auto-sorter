"""Entry point: python -m autosorter"""

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from autosorter.app.main_window import MainWindow
from autosorter.logging_setup import setup_logging
from autosorter.paths import APP_ICON


def main() -> int:
    setup_logging(None)
    app = QApplication(sys.argv)
    app.setApplicationName("Auto-Sorter")
    app.setOrganizationName("AutoSorter")
    if APP_ICON.exists():
        app.setWindowIcon(QIcon(str(APP_ICON)))

    window = MainWindow()
    if APP_ICON.exists():
        window.setWindowIcon(QIcon(str(APP_ICON)))
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
