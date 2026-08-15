"""
Database layer for Auto-Sorter.

=== WHAT IS THIS FILE? ===
This file handles ALL communication with our SQLite database.
Think of the database as a giant spreadsheet saved in a single file (library.db).
This file provides a clean Python class (Database) so the rest of our code
never has to write raw SQL — it just calls methods like db.upsert_image(...).

=== WHY SQLITE? ===
- It's a SINGLE FILE on disk. No server to install. No passwords. No setup.
- It's built into Python (import sqlite3 — already there, zero install).
- It's crash-safe: if the app crashes mid-write, the database is never corrupted.
- It's FAST: can handle millions of rows easily on any laptop.

=== WHAT IS WAL MODE? ===
WAL = Write-Ahead Logging. By default, SQLite locks the entire database when writing.
WAL mode changes this so:
  - Multiple threads can READ at the same time
  - One thread can WRITE while others READ
  - If the app crashes, the database automatically recovers
This is critical because our UI thread reads from the DB while the scanner writes to it.
"""

import sqlite3
from pathlib import Path
from typing import Optional


# The SQL statements that create our tables.
# "IF NOT EXISTS" means: create it only the first time; don't crash if it already exists.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    -- INTEGER PRIMARY KEY AUTOINCREMENT: SQLite auto-generates a unique ID for each row
    -- (1, 2, 3, 4...). We never have to set this manually.
    id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- TEXT UNIQUE NOT NULL: the file path is a string, must be unique (no duplicates),
    -- and cannot be empty (NOT NULL). This is how we find "did we already scan this file?"
    path        TEXT    UNIQUE NOT NULL,

    -- File metadata used to detect changes:
    -- If size or mtime changed since last scan → the file was modified → re-process it.
    size        INTEGER NOT NULL,          -- file size in bytes
    mtime       REAL    NOT NULL,          -- last modified time (seconds since 1970)

    -- SHA-256 hash of the file contents. Computed during thumbnail generation.
    -- Used to detect MOVED files: different path but same hash = same file.
    sha256      TEXT,

    -- Image dimensions (filled in when we decode the image for thumbnailing)
    width       INTEGER,
    height      INTEGER,

    -- EXIF "Date Taken" if the camera recorded it (e.g., "2024-06-15 14:30:00")
    taken_at    TEXT,

    -- Path to the small thumbnail we generate (e.g., "thumbs/00001.jpg")
    thumb_path  TEXT,

    -- How likely this image is a screenshot or document (0.0 = photo, 1.0 = document)
    -- We'll use this in Milestone 7 for the "Text & Documents" album.
    doc_score   REAL    DEFAULT 0.0,

    -- Processing status:
    --   'pending'   = found on disk, not yet processed (face detection hasn't run)
    --   'processed' = fully processed (faces extracted, thumbnail made)
    --   'error'     = something went wrong (see 'error' column)
    --   'missing'   = file was in DB but no longer exists on disk
    status      TEXT    DEFAULT 'pending',

    -- If status='error', this stores what went wrong (e.g., "corrupt JPEG")
    error       TEXT
);

-- Meta table: stores key-value settings like the schema version.
-- This lets us upgrade the database format in future versions.
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Current schema version. If we ever change the table structure,
# we increment this and write migration code.
_SCHEMA_VERSION = "1"


class Database:
    """
    Manages the SQLite database for Auto-Sorter.

    Usage:
        db = Database(Path("C:/Users/You/Photos/.autosorter/library.db"))
        db.upsert_image(path="/photos/pic.jpg", size=123456, mtime=1720000000.0)
        row = db.get_image_by_path("/photos/pic.jpg")
        db.close()

    You can also use it as a context manager:
        with Database(path) as db:
            db.upsert_image(...)
        # automatically closed when the 'with' block ends
    """

    def __init__(self, db_path: Path) -> None:
        """
        Open (or create) the database at the given path.

        Parameters:
            db_path: Where to store the .db file. Parent directories are created
                     automatically if they don't exist.
        """
        # Create parent directories if needed (e.g., .autosorter/ folder)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to SQLite. check_same_thread=False allows multiple threads
        # to use this connection — safe because we'll only WRITE from one thread.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)

        # Return rows as sqlite3.Row objects instead of plain tuples.
        # This lets us access columns by name: row["path"] instead of row[0]
        self._conn.row_factory = sqlite3.Row

        self._setup()

    def _setup(self) -> None:
        """Create tables and configure the database on first open."""
        # --- Enable WAL mode ---
        # PRAGMA is SQLite's way of setting internal options.
        # journal_mode=WAL enables Write-Ahead Logging (explained at top of file).
        self._conn.execute("PRAGMA journal_mode=WAL")

        # --- Foreign keys ---
        # SQLite has foreign keys (linking tables together) disabled by default.
        # We enable them so that if we delete an image, its faces are also deleted.
        self._conn.execute("PRAGMA foreign_keys=ON")

        # --- Create tables ---
        # executescript() runs multiple SQL statements separated by semicolons.
        self._conn.executescript(_SCHEMA)

        # --- Store schema version ---
        # "INSERT OR IGNORE" means: insert only if the key doesn't exist yet.
        # On a fresh DB this sets version="1". On an existing DB it does nothing.
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", _SCHEMA_VERSION),
        )
        self._conn.commit()  # Save everything to disk right now.

    # ------------------------------------------------------------------
    # Image operations
    # ------------------------------------------------------------------

    def upsert_image(
        self,
        path: str,
        size: int,
        mtime: float,
        *,  # everything after * must be passed as keyword arguments
        sha256: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        taken_at: Optional[str] = None,
        thumb_path: Optional[str] = None,
        status: str = "pending",
        error: Optional[str] = None,
    ) -> int:
        """
        Insert a new image row, or update it if the path already exists.

        "Upsert" = "Update or Insert". This is important because:
        - First scan: file is new → INSERT a row with status='pending'
        - Re-scan: file changed → UPDATE the existing row with new size/mtime
        - This single method handles both cases cleanly.

        Returns:
            The row ID of the inserted/updated image.
        """
        # "INSERT OR REPLACE" is SQLite's upsert. Because 'path' has a UNIQUE
        # constraint, if a row with this path already exists, it gets replaced.
        cursor = self._conn.execute(
            """
            INSERT OR REPLACE INTO images
                (path, size, mtime, sha256, width, height, taken_at,
                 thumb_path, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, size, mtime, sha256, width, height, taken_at,
             thumb_path, status, error),
        )
        self._conn.commit()
        # lastrowid gives us the auto-generated ID of the row we just wrote.
        return cursor.lastrowid  # type: ignore[return-value]

    def get_image_by_path(self, path: str) -> Optional[sqlite3.Row]:
        """
        Look up an image by its file path.

        Returns:
            A Row object (dict-like) if found, or None if not in the database.

        Example:
            row = db.get_image_by_path("/photos/sunset.jpg")
            if row:
                print(row["status"])  # "pending" or "processed"
        """
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE path = ?", (path,)
        )
        return cursor.fetchone()  # Returns one row or None

    def is_unchanged(self, path: str, size: int, mtime: float) -> bool:
        """
        Check if a file is already in the DB with the same size and mtime.

        This is the "skip unchanged files" logic:
        - Same path + same size + same modification time = file hasn't changed
        - No need to re-process it, saving lots of time on re-scans.

        WHY size + mtime (not just path)?
        - If you edit a photo (crop, filter), its size and/or mtime change.
        - We detect that and re-process it to find faces in the new version.
        """
        row = self.get_image_by_path(path)
        if row is None:
            return False  # Never seen this file before
        return row["size"] == size and abs(row["mtime"] - mtime) < 0.001

    def count_images(self, status: Optional[str] = None) -> int:
        """
        Count images in the database.

        Args:
            status: If given, count only images with this status.
                    If None, count all images.

        Examples:
            db.count_images()                  # total images
            db.count_images(status="pending")   # how many still need processing
            db.count_images(status="processed") # how many are done
        """
        if status is None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM images")
        else:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM images WHERE status = ?", (status,)
            )
        return cursor.fetchone()[0]

    def iter_pending(self):
        """
        Yield all images that still need processing (status='pending').

        This is a Python GENERATOR (uses 'yield' instead of 'return').
        Instead of loading ALL pending images into memory at once (could be
        100,000+ rows), it gives them to you one at a time — memory efficient.

        Usage:
            for row in db.iter_pending():
                process(row["path"])
        """
        cursor = self._conn.execute(
            "SELECT * FROM images WHERE status = 'pending' ORDER BY id"
        )
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row

    def update_status(self, image_id: int, status: str,
                      error: Optional[str] = None) -> None:
        """
        Update the processing status of an image.

        Called after thumbnailing/face detection:
          - Success: db.update_status(img_id, "processed")
          - Failure: db.update_status(img_id, "error", error="corrupt JPEG")
        """
        self._conn.execute(
            "UPDATE images SET status = ?, error = ? WHERE id = ?",
            (status, error, image_id),
        )
        self._conn.commit()

    def update_thumb(self, image_id: int, thumb_path: str,
                     width: int, height: int) -> None:
        """
        Store the thumbnail path and image dimensions after generating a thumbnail.
        """
        self._conn.execute(
            "UPDATE images SET thumb_path = ?, width = ?, height = ? WHERE id = ?",
            (thumb_path, width, height, image_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection. Always call this when done."""
        self._conn.close()

    def __enter__(self):
        """Support 'with Database(path) as db:' syntax."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically close when exiting a 'with' block."""
        self.close()
        return False  # Don't suppress exceptions
