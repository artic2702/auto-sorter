"""
Tests for Milestone 8: Performance, Speed Dial, Low-Spec Optimization, and Memory Cap.

Tests cover:
1. Speed dial worker count calculations for Quiet, Normal, and Fast modes
2. Process priority controls (below-normal for responsiveness)
3. Memory monitoring (RSS tracking, soft/hard limits, backpressure)
4. Reduced-size image decode (max_decode_dim via PIL / OpenCV)
5. Multi-worker detection pipeline execution & single-worker fallback
6. Stage 4 graph clustering performance at scale
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from autosorter.core.db import Database
from autosorter.core.face_embed import FaceRecognizer
from autosorter.core.graph_cluster import run_graph_clustering, GraphClusterConfig
from autosorter.core.perf import (
    SpeedMode,
    get_cpu_count,
    worker_count_for_mode,
    set_below_normal_priority,
    restore_normal_priority,
    get_rss_bytes,
    is_memory_pressure,
    MEMORY_SOFT_LIMIT,
    MEMORY_HARD_LIMIT,
)
from autosorter.core.pipeline_worker import PipelineWorker, _detect_one_image
from autosorter.core.thumbnailer import _read_image_safe


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _create_test_image(path: Path, width: int = 640, height: int = 480, color=(128, 128, 128)) -> Path:
    """Create a synthetic solid color image."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Draw simple shapes
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (200, 200, 200), -1)
    cv2.imwrite(str(path), img)
    return path


def _random_embedding() -> bytes:
    """Generate a random 128-d L2-normalized embedding as bytes."""
    vec = np.random.randn(128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return FaceRecognizer.embedding_to_bytes(vec)


# ══════════════════════════════════════════════════════════════════════
# 1. Speed Dial Tests
# ══════════════════════════════════════════════════════════════════════

class TestSpeedDial:
    """Tests for SpeedDial mode calculations and worker allocations."""

    def test_speed_mode_values(self):
        assert SpeedMode.QUIET.value == "quiet"
        assert SpeedMode.NORMAL.value == "normal"
        assert SpeedMode.FAST.value == "fast"

    def test_quiet_mode_always_one_worker(self):
        with patch("autosorter.core.perf.get_cpu_count", return_value=16):
            assert worker_count_for_mode(SpeedMode.QUIET) == 1
        with patch("autosorter.core.perf.get_cpu_count", return_value=1):
            assert worker_count_for_mode(SpeedMode.QUIET) == 1

    def test_normal_mode_half_cores(self):
        with patch("autosorter.core.perf.get_cpu_count", return_value=8):
            assert worker_count_for_mode(SpeedMode.NORMAL) == 4
        with patch("autosorter.core.perf.get_cpu_count", return_value=4):
            assert worker_count_for_mode(SpeedMode.NORMAL) == 2
        with patch("autosorter.core.perf.get_cpu_count", return_value=1):
            assert worker_count_for_mode(SpeedMode.NORMAL) == 1

    def test_fast_mode_all_cores(self):
        with patch("autosorter.core.perf.get_cpu_count", return_value=8):
            assert worker_count_for_mode(SpeedMode.FAST) == 8
        with patch("autosorter.core.perf.get_cpu_count", return_value=16):
            assert worker_count_for_mode(SpeedMode.FAST) == 16
        with patch("autosorter.core.perf.get_cpu_count", return_value=1):
            assert worker_count_for_mode(SpeedMode.FAST) == 1

    def test_pipeline_worker_speed_mode_wiring(self, tmp_path):
        worker_quiet = PipelineWorker(
            photo_folder=tmp_path,
            data_folder=tmp_path,
            speed_mode=SpeedMode.QUIET,
        )
        assert worker_quiet.worker_count == 1

        with patch("autosorter.core.perf.get_cpu_count", return_value=8):
            worker_normal = PipelineWorker(
                photo_folder=tmp_path,
                data_folder=tmp_path,
                speed_mode=SpeedMode.NORMAL,
            )
            assert worker_normal.worker_count == 4

            worker_fast = PipelineWorker(
                photo_folder=tmp_path,
                data_folder=tmp_path,
                speed_mode=SpeedMode.FAST,
            )
            assert worker_fast.worker_count == 8


# ══════════════════════════════════════════════════════════════════════
# 2. Process Priority Tests
# ══════════════════════════════════════════════════════════════════════

class TestProcessPriority:
    """Tests for process priority management."""

    def test_set_and_restore_priority(self):
        # On Windows, this should execute kernel32 SetPriorityClass
        result_lower = set_below_normal_priority()
        result_restore = restore_normal_priority()

        if sys.platform == "win32":
            assert result_lower is True
            assert result_restore is True
        else:
            # On non-Windows platforms, returns bool without crashing
            assert isinstance(result_lower, bool)
            assert isinstance(result_restore, bool)


# ══════════════════════════════════════════════════════════════════════
# 3. Memory Monitoring Tests
# ══════════════════════════════════════════════════════════════════════

class TestMemoryMonitoring:
    """Tests for memory limits and RSS monitoring."""

    def test_rss_bytes_returns_positive_number(self):
        rss = get_rss_bytes()
        assert isinstance(rss, int)
        # On Windows/Linux/macOS, current process RSS should be > 10 MB
        assert rss > 10 * 1024 * 1024

    def test_memory_limits_configured_correctly(self):
        assert MEMORY_SOFT_LIMIT == int(1.2 * 1024 * 1024 * 1024)
        assert MEMORY_HARD_LIMIT == int(1.5 * 1024 * 1024 * 1024)

    def test_is_memory_pressure_logic(self):
        # Under 1.2 GB -> False
        with patch("autosorter.core.perf.get_rss_bytes", return_value=500 * 1024 * 1024):
            assert is_memory_pressure() is False

        # Over 1.2 GB -> True
        with patch("autosorter.core.perf.get_rss_bytes", return_value=int(1.3 * 1024 * 1024 * 1024)):
            assert is_memory_pressure() is True

        # Failed measurement (0) -> False (no false alarm)
        with patch("autosorter.core.perf.get_rss_bytes", return_value=0):
            assert is_memory_pressure() is False


# ══════════════════════════════════════════════════════════════════════
# 4. Reduced-Size Image Decode Tests
# ══════════════════════════════════════════════════════════════════════

class TestReducedSizeDecode:
    """Tests for max_decode_dim downscaling during image decode."""

    def test_read_image_safe_respects_max_decode_dim(self, tmp_path):
        img_path = tmp_path / "large_photo.jpg"
        # Create an 800x600 image
        _create_test_image(img_path, width=800, height=600)

        # Decode without restriction -> original dimensions
        full_img = _read_image_safe(str(img_path))
        assert full_img is not None
        assert full_img.shape == (600, 800, 3)

        # Decode with max_decode_dim=400 -> shrunk
        shrunk_img = _read_image_safe(str(img_path), max_decode_dim=400)
        assert shrunk_img is not None
        h, w = shrunk_img.shape[:2]
        assert max(h, w) <= 400
        # Aspect ratio preserved (800x600 -> 400x300)
        assert w == 400
        assert h == 300

    def test_read_image_safe_smaller_image_not_upscaled(self, tmp_path):
        img_path = tmp_path / "small_photo.jpg"
        _create_test_image(img_path, width=200, height=150)

        img = _read_image_safe(str(img_path), max_decode_dim=500)
        assert img is not None
        assert img.shape == (150, 200, 3)


# ══════════════════════════════════════════════════════════════════════
# 5. Multi-Worker & Single-Worker Detection Pipeline Tests
# ══════════════════════════════════════════════════════════════════════

class TestDetectionPipelineWorkers:
    """Tests for multi-worker detection execution in PipelineWorker."""

    def test_detect_one_image_invalid_path(self):
        res = _detect_one_image(999, "non_existent_file.jpg", 1280)
        assert res is None

    def test_detect_one_image_blank_image(self, tmp_path):
        img_path = tmp_path / "blank.jpg"
        _create_test_image(img_path, width=400, height=300)
        res = _detect_one_image(1, str(img_path), 1280)
        # Should decode successfully but find no faces
        assert res == []

    def test_pipeline_runs_multi_worker_to_completion(self, tmp_path):
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create 4 test images with distinct colors (unique hashes)
        for i in range(4):
            _create_test_image(
                photos_dir / f"test_{i}.jpg",
                width=320,
                height=240,
                color=(50 + i * 40, 100, 150),
            )

        # Run pipeline with 2 workers (explicit multi-worker)
        with patch("autosorter.core.perf.get_cpu_count", return_value=4):
            worker = PipelineWorker(
                photos_dir,
                data_dir,
                speed_mode=SpeedMode.NORMAL,
            )
            assert worker.worker_count == 2

            results = []
            errors = []
            worker.finished.connect(results.append)
            worker.error.connect(errors.append)
            worker.run()  # run synchronously for test

            assert errors == []
            assert len(results) == 1
            summary = results[0]
            assert summary["files_found"] == 4
            assert summary["files_new"] == 4
            assert summary["worker_count"] == 2
            assert summary["cancelled"] is False
            assert summary["errors"] == 0

            # Verify database records
            db = Database(data_dir / "library.db")
            assert db.count_images() == 4
            assert db.count_images(status="processed") == 4
            db.close()

    def test_pipeline_runs_quiet_single_worker(self, tmp_path):
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for i in range(2):
            _create_test_image(
                photos_dir / f"quiet_{i}.jpg",
                width=320,
                height=240,
                color=(60 + i * 50, 120, 180),
            )

        worker = PipelineWorker(
            photos_dir,
            data_dir,
            speed_mode=SpeedMode.QUIET,
        )
        assert worker.worker_count == 1

        results = []
        errors = []
        worker.finished.connect(results.append)
        worker.error.connect(errors.append)
        worker.run()

        assert errors == []
        assert len(results) == 1
        summary = results[0]
        assert summary["files_found"] == 2
        assert summary["worker_count"] == 1


# ══════════════════════════════════════════════════════════════════════
# 6. Stage 4 Clustering Performance at Scale
# ══════════════════════════════════════════════════════════════════════

class TestClusteringPerformance:
    """Verify that Stage 4 graph clustering completes rapidly on large datasets."""

    def test_graph_clustering_500_faces_completes_under_one_second(self, tmp_path):
        db = Database(tmp_path / "perf.db")

        # Insert 50 images with 10 faces each = 500 faces
        # Simulate 5 distinct persons (100 faces per cluster)
        person_bases = [
            np.random.randn(128).astype(np.float32) for _ in range(5)
        ]
        for base in person_bases:
            base /= np.linalg.norm(base)

        face_count = 500
        for i in range(face_count):
            img_id = db.upsert_image(
                path=f"img_{i}.jpg",
                size=1000,
                mtime=1000.0,
                status="processed",
            )
            # Pick one of the 5 persons and add small noise (similarity > 0.7)
            base = person_bases[i % 5]
            noisy = base + np.random.randn(128).astype(np.float32) * 0.05
            noisy /= np.linalg.norm(noisy)
            emb_bytes = FaceRecognizer.embedding_to_bytes(noisy)

            face_id = db.insert_face(
                image_id=img_id,
                bbox_x=10, bbox_y=10, bbox_w=50, bbox_h=50,
                det_score=0.95,
                landmarks="[]",
                quality=0.8,
            )
            db.update_embedding(face_id, emb_bytes)

        db._conn.commit()

        # Measure clustering time
        start_time = time.perf_counter()
        report = run_graph_clustering(db, config=GraphClusterConfig(), dry_run=False)
        elapsed = time.perf_counter() - start_time

        # Must be well under 1 second for 500 faces
        assert elapsed < 1.0
        assert report.total_faces == 500
        assert db.count_persons() >= 5
        db.close()
