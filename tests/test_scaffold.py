"""M0 smoke tests: models are present, load into OpenCV, and produce sane output shapes."""

import cv2
import numpy as np

from autosorter.paths import SFACE_MODEL, YUNET_MODEL


def test_model_files_exist():
    assert YUNET_MODEL.is_file(), f"missing {YUNET_MODEL}"
    assert SFACE_MODEL.is_file(), f"missing {SFACE_MODEL}"
    # Guard against accidentally committing Git-LFS pointer files instead of real models.
    assert YUNET_MODEL.stat().st_size > 100_000
    assert SFACE_MODEL.stat().st_size > 30_000_000


def test_yunet_detector_loads_and_runs():
    detector = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320))
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    _, faces = detector.detect(blank)
    assert faces is None or len(faces) == 0  # no faces in a blank image


def test_sface_recognizer_loads_and_embeds():
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
    # SFace expects an aligned 112x112 BGR crop; a random one is fine for a shape check.
    rng = np.random.default_rng(seed=42)
    crop = rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8)
    embedding = recognizer.feature(crop)
    assert embedding.shape == (1, 128)  # the 128-number "faceprint"
    assert np.isfinite(embedding).all()
