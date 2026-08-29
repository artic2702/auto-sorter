"""
Safe file export engine — copies photos into organized person folders.

=== WHAT IS THIS FILE? (M7) ===
The export engine reads the Auto-Sorter database and copies photos into
a clean folder structure organized by person. Group photos are copied
into each participant's folder. Source files are NEVER modified or deleted.

=== EXPORT FOLDER STRUCTURE ===
    Export Target/
    ├── Arnav/
    │   ├── IMG_2611.jpg
    │   ├── IMG_2612.jpg        ← group photo (also appears in Mom/)
    │   └── ...
    ├── Mom/
    │   ├── IMG_2612.jpg        ← same group photo
    │   └── ...
    ├── Group Photos/
    │   ├── IMG_2620.jpg
    │   └── ...
    ├── Unknown/
    │   └── ...
    └── Text & Documents/
        ├── Screenshot_2024.png
        └── ...

=== SAFETY GUARANTEES ===
1. Source files are opened read-only (shutil.copy2 reads + writes to dest).
2. No source path is ever passed to os.remove, os.rename, or shutil.move.
3. Export can be cancelled between any two files without corruption.
4. Re-running export skips files that already exist at the destination
   with matching size (resume support after interruption).
"""

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from autosorter.core.db import Database


def _safe_folder_name(name: str) -> str:
    """
    Sanitize a person display name for use as a folder name on Windows.

    Windows forbids these characters in folder names:
        \\ / : * ? \" < > |
    We also strip leading/trailing whitespace and dots (Windows quirk:
    folder names cannot end with a dot).

    Args:
        name: The raw display name (e.g., "Person 1", "Mom ❤️").

    Returns:
        A filesystem-safe folder name.
    """
    # Replace forbidden characters with underscores
    forbidden = r'\/:*?"<>|'
    safe = name
    for char in forbidden:
        safe = safe.replace(char, "_")

    # Strip leading/trailing whitespace and dots
    safe = safe.strip().strip(".")

    # Fall back to "Unknown" if the name is empty after sanitization
    return safe if safe else "Unknown"


def _unique_dest_path(dest_folder: Path, filename: str) -> Path:
    """
    Generate a unique destination file path, handling filename collisions.

    If 'IMG_2611.jpg' already exists in dest_folder, try 'IMG_2611_2.jpg',
    then 'IMG_2611_3.jpg', etc.

    Args:
        dest_folder: The target folder to place the file in.
        filename:    The original filename.

    Returns:
        A Path that does not yet exist in dest_folder.
    """
    dest = dest_folder / filename
    if not dest.exists():
        return dest

    # Split filename into stem and suffix: "IMG_2611.jpg" → ("IMG_2611", ".jpg")
    stem = Path(filename).stem
    suffix = Path(filename).suffix

    # Try incrementing suffixes: _2, _3, _4, ...
    counter = 2
    while True:
        candidate = dest_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def export_library(
    db: Database,
    export_root: Path,
    person_ids: Optional[List[int]] = None,
    include_group: bool = True,
    include_unknown: bool = True,
    include_docs: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, int]:
    """
    Export the photo library into organized person folders.

    This is the core export function. It reads persons from the database,
    creates a folder for each person, and copies their photos into the
    corresponding folders. Group photos are copied into each participant's
    folder so every person's folder is complete.

    Args:
        db:                The database connection.
        export_root:       The root folder to export into (e.g., D:\\Export).
        person_ids:        Optional list of specific person database IDs to export.
                           If None, ALL visible persons in the library are exported.
                           If specified (e.g. [5] or [1, 7]), ONLY those chosen
                           persons' folders will be created and exported.
        include_group:     If True, also create a "Group Photos" folder.
        include_unknown:   If True, export photos with unknown faces.
        include_docs:      If True, export screenshots/documents.
        progress_callback: Optional callback(done, total, detail) for progress.
        cancel_check:      Optional callback() -> bool. Return True to cancel.

    Returns:
        Dict with counts: {"persons_exported", "files_copied", "files_skipped",
                           "group_photos", "unknown_photos", "doc_photos"}

    Safety:
        Source files are NEVER modified, moved, or deleted. Only copy operations.
    """
    # Initialize result counters
    result = {
        "persons_exported": 0,
        "files_copied": 0,
        "files_skipped": 0,
        "group_photos": 0,
        "unknown_photos": 0,
        "doc_photos": 0,
    }

    # Ensure the export root directory exists on disk
    export_root.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Filter target persons and count total work for progress reporting ──
    all_persons = db.get_all_persons()

    # If the user specified specific person IDs, filter the list down to only those people
    if person_ids is not None:
        target_ids = set(person_ids)
        persons = [p for p in all_persons if p["id"] in target_ids]
    else:
        # Otherwise export all visible, non-merged persons
        persons = all_persons

    total_work = 0

    # Pre-fetch image lists per person to count total files for the progress bar
    person_images: Dict[int, list] = {}
    for p in persons:
        # Query all images that contain at least one face of this person
        imgs = db.get_images_for_person(p["id"])
        person_images[p["id"]] = imgs
        total_work += len(imgs)

    # Add counts for optional smart albums
    group_images = db.get_group_photos() if include_group else []
    unknown_faces = db.get_unknown_faces() if include_unknown else []
    doc_images = db.get_text_and_docs() if include_docs else []
    total_work += len(group_images) + len(unknown_faces) + len(doc_images)

    done = 0

    # ── Phase 2: Export each target person's photos ──
    for p in persons:
        # Check for user cancellation between persons
        if cancel_check and cancel_check():
            break

        # Create a safe, sanitized folder name on Windows (no forbidden chars like \ / : * ? " < > |)
        folder_name = _safe_folder_name(p["display_name"])
        person_folder = export_root / folder_name
        person_folder.mkdir(parents=True, exist_ok=True)

        imgs = person_images[p["id"]]
        for img in imgs:
            if cancel_check and cancel_check():
                break

            src = Path(img["path"])
            if not src.is_file():
                done += 1
                continue

            # Resume support: check if the file was already exported
            # (same filename and same size in the destination folder)
            candidate = person_folder / src.name
            if candidate.exists() and candidate.stat().st_size == src.stat().st_size:
                result["files_skipped"] += 1
            else:
                dest = _unique_dest_path(person_folder, src.name)
                shutil.copy2(str(src), str(dest))
                result["files_copied"] += 1

            done += 1
            if progress_callback:
                progress_callback(done, total_work, f"Exporting {folder_name}…")

        result["persons_exported"] += 1

    # ── Phase 3: Export Group Photos album ──
    if include_group and group_images:
        group_folder = export_root / "Group Photos"
        group_folder.mkdir(parents=True, exist_ok=True)

        for img in group_images:
            if cancel_check and cancel_check():
                break

            src = Path(img["path"])
            if not src.is_file():
                done += 1
                continue

            dest = group_folder / src.name
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                result["files_skipped"] += 1
            else:
                dest = _unique_dest_path(group_folder, src.name)
                shutil.copy2(str(src), str(dest))
                result["group_photos"] += 1
                result["files_copied"] += 1

            done += 1
            if progress_callback:
                progress_callback(done, total_work, "Exporting Group Photos…")

    # ── Phase 4: Export Unknown faces (by their source images) ──
    if include_unknown and unknown_faces:
        unknown_folder = export_root / "Unknown"
        unknown_folder.mkdir(parents=True, exist_ok=True)

        # Deduplicate by image_id — multiple unknown faces in the same image
        # should only copy the image once
        seen_image_ids: Set[int] = set()

        for face in unknown_faces:
            if cancel_check and cancel_check():
                break

            img_id = face["image_id"]
            if img_id in seen_image_ids:
                done += 1
                continue
            seen_image_ids.add(img_id)

            src_path = face["image_path"]
            if not src_path or not Path(src_path).is_file():
                done += 1
                continue

            src = Path(src_path)
            dest = unknown_folder / src.name
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                result["files_skipped"] += 1
            else:
                dest = _unique_dest_path(unknown_folder, src.name)
                shutil.copy2(str(src), str(dest))
                result["unknown_photos"] += 1
                result["files_copied"] += 1

            done += 1
            if progress_callback:
                progress_callback(done, total_work, "Exporting Unknown…")

    # ── Phase 5: Export Text & Documents ──
    if include_docs and doc_images:
        docs_folder = export_root / "Text & Documents"
        docs_folder.mkdir(parents=True, exist_ok=True)

        for img in doc_images:
            if cancel_check and cancel_check():
                break

            src = Path(img["path"])
            if not src.is_file():
                done += 1
                continue

            dest = docs_folder / src.name
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                result["files_skipped"] += 1
            else:
                dest = _unique_dest_path(docs_folder, src.name)
                shutil.copy2(str(src), str(dest))
                result["doc_photos"] += 1
                result["files_copied"] += 1

            done += 1
            if progress_callback:
                progress_callback(done, total_work, "Exporting Documents…")

    return result
