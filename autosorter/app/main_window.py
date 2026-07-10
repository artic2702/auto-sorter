"""Main application window (M0: empty shell, screens arrive in later milestones)."""

from PySide6.QtWidgets import QLabel, QMainWindow

from autosorter import __version__


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Auto-Sorter {__version__}")
        self.resize(1000, 700)

        placeholder = QLabel(
            "Auto-Sorter\n\nOffline photo organizer — under construction (M0 scaffold)."
        )
        placeholder.setStyleSheet("font-size: 16px; color: gray;")
        placeholder.setContentsMargins(24, 24, 24, 24)
        self.setCentralWidget(placeholder)
