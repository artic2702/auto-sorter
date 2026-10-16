"""
File scanner — walks a directory tree and registers images in the database.

=== WHAT IS THIS FILE? ===
When the user selects a photo folder, this module walks through every file
in that folder (and all sub-folders), checks if it's a photo, and adds it
to the database. On re-scans, it skips files that haven't changed.

=== HOW THE SCANNER WORKS (step by step) ===
1. User picks a folder like "D:\\Photos"
2. Scanner walks every file in D:\\Photos (and subdirectories)
3. For each file:
   a. Is it a photo? (check extension: .jpg, .png, etc.) → if no, skip
   b. Is it already in the database with the same size + date? → if yes, skip
   c. Otherwise → add to database as status='pending'
4. Return a summary: "Found 1200 files, 50 new, 1150 skipped"

=== WHY os.scandir() INSTEAD OF os.walk()? ===
os.scandir() is faster because it gets file metadata (size, type) in the
same system call that reads the directory. os.walk() needs extra calls.
For folders with 50,000+ files, this makes a noticeable difference.
"""

import os
from pathlib import Path
from typing import NamedTuple

from autosorter.core.db import Database


# NamedTuple is like a simple class with named fields.
# It's immutable (can't be changed after creation) and lightweight.
# We use it to return scan results in a clean, readable way:
#   result = ScanResult(found=100, new=20, skipped=80, errors=0)
#   log.info(result.new)  # 20
class ScanResult(NamedTuple):
    found: int    # total image files found on disk
    new: int      # files that were added to the database
    skipped: int  # files already in DB and unchanged
    errors: int   # files that caused problems (permissions, etc.)
    discovered_paths: set  # all image paths found on disk (for missing detection)


# The set of file extensions we consider "photos".
# We use a frozenset (an immutable set) because:
#   1. Set lookup is O(1) — checking "is .jpg in this set?" is instant
#   2. Frozen = can't accidentally add/remove items later
#
# WHY THESE FORMATS?
# - .jpg/.jpeg: ~95% of all photos from cameras and phones
# - .png: screenshots, edited photos, graphics
# - .bmp: old Windows format, some scanners still use it
# - .tiff/.tif: professional photography, scanners
# - .webp: modern Google format, increasingly common
# - .heic/.heif: iPhone & modern mobile photos
# - Camera RAW formats:
#     - .cr2, .cr3 (Canon)
#     - .nef, .nrw (Nikon)
#     - .arw, .srf, .sr2 (Sony)
#     - .dng (Adobe / Leica / Google Pixel RAW)
#     - .raf (Fujifilm)
#     - .orf (Olympus)
#     - .rw2 (Panasonic)
#     - .pef (Pentax)
IMAGE_EXTENSIONS = frozenset({
    # Standard Web / Mobile formats
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tiff", ".tif", ".webp",
    ".heic", ".heif",
    # Camera RAW formats
    ".cr2", ".cr3",
    ".nef", ".nrw",
    ".arw", ".srf", ".sr2",
    ".dng",
    ".raf",
    ".orf",
    ".rw2",
    ".pef",
})


def is_image_file(path: str) -> bool:
    """
    Check if a file path has a recognized image extension.

    Uses case-insensitive comparison because Windows doesn't care about case:
    "photo.JPG" and "photo.jpg" are the same file.

    Args:
        path: Full path or just filename.

    Returns:
        True if the file extension is in our IMAGE_EXTENSIONS set.
    """
    # os.path.splitext splits "photo.jpg" into ("photo", ".jpg")
    # [1] gets the extension, .lower() makes it case-insensitive
    _, ext = os.path.splitext(path)
    return ext.lower() in IMAGE_EXTENSIONS


# Directories to always skip during scanning.
# These are either our own data folders or system folders that never contain user photos.
SKIP_DIR_NAMES = frozenset({
    ".autosorter",  # our own data folder (DB + thumbnails)
    ".git",         # version control
    "$RECYCLE.BIN", # Windows recycle bin
    "System Volume Information",  # Windows system folder
})


def scan_folder(
    root: Path,
    db: Database,
    *,
    progress_callback=None,
    cancel_check=None,
) -> ScanResult:
    """
    Walk a folder tree, find all images, and register them in the database.

    Args:
        root:              The top-level folder to scan (e.g., Path("D:/Photos"))
        db:                Database instance to store results in
        progress_callback: Optional function called with (found, new, skipped)
                          after each file. The UI uses this to update the counter.
        cancel_check:      Optional function that returns True if the user clicked
                          "Cancel". We check this regularly to stop scanning.

    Returns:
        A ScanResult with counts of what happened.

    HOW THE WALK WORKS:
    Imagine this folder structure:
        D:\\Photos\\
          ├── vacation/
          │   ├── beach.jpg      ← found!
          │   └── notes.txt      ← ignored (not an image)
          ├── family/
          │   └── birthday.png   ← found!
          └── selfie.jpg         ← found!

    os.scandir() visits each item in a directory. We use a stack (list) to
    process directories depth-first without recursion (which could crash on
    very deeply nested folders).
    """
    found = 0
    new = 0
    skipped = 0
    errors = 0
    discovered_paths: set = set()  # Track all image paths found on disk (M6)

    # We use a manual stack instead of os.walk() for two reasons:
    # 1. We can check cancel_check between directories
    # 2. os.scandir() gives us DirEntry objects with cached stat info (faster)
    #
    # "Stack" = a list where we add to the end and remove from the end (LIFO).
    # Start with the root folder.
    dirs_to_visit = [root]

    while dirs_to_visit:
        # Check if the user clicked "Cancel"
        if cancel_check and cancel_check():
            break

        current_dir = dirs_to_visit.pop()

        try:
            # os.scandir() returns DirEntry objects for each item in the directory.
            # DirEntry has .name, .path, .is_file(), .is_dir(), .stat() — all cached.
            entries = list(os.scandir(current_dir))
        except PermissionError:
            # Some folders (like System Volume Information) are locked.
            # We skip them silently instead of crashing.
            errors += 1
            continue
        except OSError:
            # Other OS errors (network drives disconnected, etc.)
            errors += 1
            continue

        for entry in entries:
            # Check cancellation between files too (responsive to Cancel button)
            if cancel_check and cancel_check():
                break

            if entry.is_dir(follow_symlinks=False):
                # It's a subdirectory → add to our stack to visit later.
                # follow_symlinks=False prevents infinite loops from symbolic links.
                #
                # BUT skip directories that are our own data or system folders.
                # Also skip any directory starting with '.' (hidden folders on
                # both Windows and Linux, e.g., .autosorter, .git, .thumbnails)
                if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                    continue  # Don't descend into this folder
                dirs_to_visit.append(entry.path)

            elif entry.is_file(follow_symlinks=False):
                # It's a file → check if it's an image
                if not is_image_file(entry.name):
                    continue  # Not an image, skip silently

                found += 1
                discovered_paths.add(entry.path)  # M6: track for missing detection

                try:
                    # Get file metadata from the operating system
                    stat = entry.stat()
                    file_size = stat.st_size      # size in bytes
                    file_mtime = stat.st_mtime    # last modified time

                    # Check: is this file already in the DB and unchanged?
                    if db.is_unchanged(entry.path, file_size, file_mtime):
                        skipped += 1
                    else:
                        # New or modified file → add to database
                        db.upsert_image(
                            path=entry.path,
                            size=file_size,
                            mtime=file_mtime,
                        )
                        new += 1

                except OSError:
                    # File disappeared between scandir and stat, or is locked
                    errors += 1

                # Notify the UI about progress
                if progress_callback:
                    progress_callback(found, new, skipped)

    return ScanResult(
        found=found, new=new, skipped=skipped, errors=errors,
        discovered_paths=discovered_paths,
    )
