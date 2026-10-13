"""Central place for locating bundled resources (works in dev and in PyInstaller builds)."""

import sys
from pathlib import Path


def package_root() -> Path:
    """Directory containing the autosorter package resources."""
    if getattr(sys, "frozen", False):  # PyInstaller build
        candidates = []
        if hasattr(sys, "_MEIPASS"):
            meipass = Path(sys._MEIPASS)
            candidates.extend([
                meipass / "autosorter",
                meipass,
            ])
        exe_dir = Path(sys.executable).parent
        candidates.extend([
            exe_dir / "_internal" / "autosorter",
            exe_dir / "_internal",
            exe_dir / "autosorter",
            exe_dir,
        ])
        for c in candidates:
            if (c / "models").exists():
                return c
        return Path(getattr(sys, "_MEIPASS", exe_dir)) / "autosorter"
    return Path(__file__).parent


MODELS_DIR = package_root() / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
APP_ICON = package_root() / "app.png"
