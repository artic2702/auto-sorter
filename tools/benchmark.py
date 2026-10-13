"""
Benchmark tool for Auto-Sorter (Milestone 8).

Measures pipeline throughput (img/s), wall clock time, and peak memory (RSS)
across different worker configurations (Quiet, Normal, Fast).

Usage:
    python tools/benchmark.py --images 20
    python tools/benchmark.py --images 50 --output benchmark_results.md
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from autosorter.core.perf import (
    SpeedMode,
    worker_count_for_mode,
    get_cpu_count,
    get_rss_bytes,
)
from autosorter.core.pipeline_worker import PipelineWorker


def generate_benchmark_images(folder: Path, count: int, width: int = 1280, height: int = 960) -> None:
    """Generate synthetic test images with gradient backgrounds and circular shapes."""
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Generating {count} synthetic test images ({width}x{height}) in {folder}...")

    for i in range(count):
        # Create gradient image with varying colors
        img = np.zeros((height, width, 3), dtype=np.uint8)
        base_color = ((i * 37) % 255, (i * 73) % 255, (i * 109) % 255)
        img[:] = base_color

        # Draw contrast circles
        center_x = width // 2 + int(np.sin(i) * (width // 4))
        center_y = height // 2 + int(np.cos(i) * (height // 4))
        radius = min(width, height) // 6
        cv2.circle(img, (center_x, center_y), radius, (240, 240, 240), -1)
        cv2.circle(img, (center_x - radius // 3, center_y - radius // 4), radius // 6, (40, 40, 40), -1)
        cv2.circle(img, (center_x + radius // 3, center_y - radius // 4), radius // 6, (40, 40, 40), -1)

        file_path = folder / f"bench_{i:04d}.jpg"
        cv2.imwrite(str(file_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print("Image generation complete.")


def run_single_benchmark(
    photo_folder: Path,
    speed_mode: SpeedMode,
    image_count: int,
) -> Dict[str, Any]:
    """Run the pipeline for a given speed mode and measure metrics."""
    data_dir = Path(tempfile.mkdtemp(prefix="autosorter_bench_data_"))
    try:
        workers = worker_count_for_mode(speed_mode)
        mode_name = speed_mode.value.capitalize()
        print(f"\n--- Testing {mode_name} mode ({workers} workers) on {image_count} images ---")

        worker = PipelineWorker(
            photo_folder=photo_folder,
            data_folder=data_dir,
            speed_mode=speed_mode,
        )

        results: List[dict] = []
        errors: List[str] = []
        worker.finished.connect(results.append)
        worker.error.connect(errors.append)

        start_time = time.perf_counter()
        worker.run()
        elapsed = time.perf_counter() - start_time

        if errors:
            raise RuntimeError(f"Pipeline error: {errors[0]}")

        summary = results[0] if results else {}
        peak_rss_mb = summary.get("peak_rss_mb", 0)
        if peak_rss_mb == 0:
            peak_rss_mb = round(get_rss_bytes() / (1024 * 1024), 1)

        img_per_sec = image_count / elapsed if elapsed > 0 else 0

        return {
            "mode": mode_name,
            "workers": workers,
            "images": image_count,
            "time_sec": round(elapsed, 2),
            "throughput_img_s": round(img_per_sec, 2),
            "peak_rss_mb": peak_rss_mb,
            "faces_found": summary.get("faces_detected", 0),
        }
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Format benchmark rows as a markdown table."""
    headers = [
        "Mode",
        "Workers",
        "Images",
        "Total Time (s)",
        "Throughput (img/s)",
        "Peak RSS (MB)",
        "RAM Cap (< 1.5 GB)",
    ]

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    lines = [header_line, separator_line]
    for r in rows:
        ram_ok = "PASS (<1.5GB)" if r["peak_rss_mb"] < 1536 else "FAIL (>1.5GB)"
        row_str = (
            f"| {r['mode']} "
            f"| {r['workers']} "
            f"| {r['images']} "
            f"| {r['time_sec']}s "
            f"| {r['throughput_img_s']} img/s "
            f"| {r['peak_rss_mb']} MB "
            f"| {ram_ok} |"
        )
        lines.append(row_str)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-Sorter Performance Benchmark")
    parser.add_argument(
        "--images",
        type=int,
        default=20,
        help="Number of synthetic test images to generate (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output markdown report file",
    )
    args = parser.parse_args()

    cpu_count = get_cpu_count()
    print("=" * 60)
    print(" Auto-Sorter M8 Performance Benchmark")
    print(f" System CPU cores (logical): {cpu_count}")
    print(f" Test images: {args.images}")
    print("=" * 60)

    bench_dir = Path(tempfile.mkdtemp(prefix="autosorter_bench_photos_"))
    try:
        generate_benchmark_images(bench_dir, count=args.images)

        modes = [SpeedMode.QUIET, SpeedMode.NORMAL, SpeedMode.FAST]
        results = []

        for mode in modes:
            res = run_single_benchmark(bench_dir, mode, args.images)
            results.append(res)
            print(
                f"  -> {res['mode']}: {res['time_sec']}s "
                f"({res['throughput_img_s']} img/s), Peak RSS: {res['peak_rss_mb']} MB"
            )

        print("\n" + "=" * 60)
        print(" Benchmark Results Summary")
        print("=" * 60 + "\n")

        table_md = format_markdown_table(results)
        print(table_md)

        if args.output:
            out_path = Path(args.output)
            report = (
                f"# Auto-Sorter Benchmark Report\n\n"
                f"- **CPU Logical Cores:** {cpu_count}\n"
                f"- **Images Tested:** {args.images}\n"
                f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"### Performance by Speed Dial Mode\n\n"
                f"{table_md}\n"
            )
            out_path.write_text(report, encoding="utf-8")
            print(f"\nWrote benchmark report to: {out_path.resolve()}")

    finally:
        shutil.rmtree(bench_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
