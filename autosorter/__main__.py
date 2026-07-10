"""Entry point: python -m autosorter"""

import sys

from PySide6.QtWidgets import QApplication

from autosorter.app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Auto-Sorter")
    app.setOrganizationName("AutoSorter")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
