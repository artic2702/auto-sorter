"""
Performance utilities for Auto-Sorter.

=== WHAT IS THIS FILE? (M8) ===
This module provides low-level performance controls:
1. Thread priority — set below-normal so worker threads yield to the UI thread
2. RSS monitoring — track memory usage to enforce the 1.5 GB cap
3. Worker count — calculate thread counts for the speed dial
4. Batch commit — reduce SQLite I/O by batching writes

=== WHY BELOW-NORMAL THREAD PRIORITY? ===
When running face detection on 20,000+ photos, worker threads can max out CPU
cores. Instead of lowering process-wide priority (which would slow down our own
UI thread), we lower the priority of the background worker threads.
Worker threads run at full speed when idle, but yield immediately to the UI
thread and other interactive applications.

=== WHY RSS MONITORING? ===
The plan guarantees RAM < 1.5 GB. Without monitoring, a 4-worker thread
pool processing 8000×6000 images could spike to 3+ GB. We sample RSS
periodically and apply backpressure (pause the producer queue) if it
exceeds our soft limit.
"""

import os
import sys
from enum import Enum


# ══════════════════════════════════════════════════════════════════════
# Speed Dial — worker count modes
# ══════════════════════════════════════════════════════════════════════

class SpeedMode(Enum):
    """Speed dial modes for the pipeline."""
    QUIET = "quiet"       # 1 worker — minimal CPU
    NORMAL = "normal"     # half the cores
    FAST = "fast"         # all cores


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    os.cpu_count() returns the number of LOGICAL cores (including
    hyperthreading). On a 4-core/8-thread CPU, this returns 8.
    We use logical cores because OpenCV releases the GIL and can
    benefit from hyperthreading for I/O-bound decode operations.

    Returns at least 1 even if os.cpu_count() returns None.
    """
    return os.cpu_count() or 1


def worker_count_for_mode(mode: SpeedMode) -> int:
    """
    Calculate the number of detection workers for a given speed mode.

    Args:
        mode: The user's selected speed dial setting.

    Returns:
        Number of worker threads to use.

    Examples (on an 8-core machine):
        QUIET  → 1 worker
        NORMAL → 4 workers (8 // 2)
        FAST   → 8 workers
    """
    cpus = get_cpu_count()

    if mode == SpeedMode.QUIET:
        return 1
    elif mode == SpeedMode.NORMAL:
        return max(1, cpus // 2)
    elif mode == SpeedMode.FAST:
        return max(1, cpus)
    else:
        return max(1, cpus // 2)


# ══════════════════════════════════════════════════════════════════════
# Thread Priority (per-thread)
# ══════════════════════════════════════════════════════════════════════

def set_current_thread_below_normal() -> bool:
    """Lower the priority of the CALLING thread only (Windows). No-op elsewhere.

    Used as the ThreadPoolExecutor initializer so decoding/detection threads yield
    to the UI thread, which keeps normal priority.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        THREAD_PRIORITY_BELOW_NORMAL = -1
        return bool(kernel32.SetThreadPriority(kernel32.GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL))
    except (OSError, AttributeError):
        return False


MEMORY_SOFT_LIMIT = int(1.2 * 1024 * 1024 * 1024)

# Hard memory limit (1.5 GB — absolute cap)
MEMORY_HARD_LIMIT = int(1.5 * 1024 * 1024 * 1024)


def get_rss_bytes() -> int:
    """
    Get the current Resident Set Size (physical memory used) in bytes.

    On Windows, uses GetProcessMemoryInfo via ctypes.
    On other platforms, tries /proc/self/status (Linux) or resource module.

    Returns 0 if memory info is unavailable (never blocks or crashes).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
            handle = kernel32.GetCurrentProcess()

            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
            return 0

        else:
            # Linux: read from /proc/self/status
            try:
                with open("/proc/self/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            # Format: "VmRSS:    12345 kB"
                            return int(line.split()[1]) * 1024
            except FileNotFoundError:
                pass

            # Fallback: resource module (macOS, BSD)
            import resource
            # ru_maxrss is in bytes on macOS, KB on Linux
            usage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                return usage.ru_maxrss  # bytes on macOS
            return usage.ru_maxrss * 1024  # KB on Linux

    except Exception:
        return 0


def is_memory_pressure() -> bool:
    """
    Check if we're approaching the memory limit.

    Returns True if RSS exceeds the soft limit (1.2 GB), signaling
    that the producer should pause and let workers drain their queues.
    """
    rss = get_rss_bytes()
    if rss == 0:
        return False  # Can't measure → assume OK
    return rss > MEMORY_SOFT_LIMIT
