"""Central place for locating bundled resources (works in dev and in PyInstaller builds)."""

import sys
from pathlib import Path


def package_root() -> Path:
    """Directory containing the autosorter package resources."""
    if getattr(sys, "frozen", False):  # PyInstaller onedir build
        return Path(sys._MEIPASS) / "autosorter"  # type: ignore[attr-defined]
    return Path(__file__).parent


MODELS_DIR = package_root() / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
