"""Application logging configuration."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(data_folder: Path | None = None) -> None:
    """Configure root logger with a console handler and an optional rotating file handler."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT)

    # Ensure single console handler
    console_handler = None
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            console_handler = h
            break

    if console_handler is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
    else:
        console_handler.setFormatter(formatter)

    # Attach file handler if data_folder is provided
    if data_folder is not None:
        folder = Path(data_folder)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        log_path = folder / "autosorter.log"

        has_file_handler = False
        for h in list(root.handlers):
            if isinstance(h, RotatingFileHandler):
                if getattr(h, "baseFilename", None) == str(log_path.resolve()):
                    has_file_handler = True
                else:
                    root.removeHandler(h)
                    h.close()

        if not has_file_handler:
            file_handler = RotatingFileHandler(
                log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
