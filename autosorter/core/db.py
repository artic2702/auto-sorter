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

-- ═══════════════════════════════════════════════════════════════════
-- FACES TABLE (added in M2)
-- ═══════════════════════════════════════════════════════════════════
-- Each row = one face detected in one image.
-- An image can have 0, 1, or many faces (group photos).
-- The relationship is: images (1) ──▶ (many) faces
--
-- REFERENCES images(id): this is a FOREIGN KEY — it means face.image_id
-- MUST point to a valid row in the images table. If we delete an image,
-- SQLite can automatically delete its faces too (ON DELETE CASCADE).
CREATE TABLE IF NOT EXISTS faces (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Which image this face belongs to
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,

    -- BOUNDING BOX: the rectangle around the face in the original image.
    -- (x, y) = top-left corner; (w, h) = width and height in pixels.
    -- Example: bbox_x=340, bbox_y=120, bbox_w=180, bbox_h=200
    -- means "a 180×200 rectangle starting at pixel (340, 120)"
    bbox_x      INTEGER NOT NULL,
    bbox_y      INTEGER NOT NULL,
    bbox_w      INTEGER NOT NULL,
    bbox_h      INTEGER NOT NULL,

    -- LANDMARKS: 5 facial key points stored as a JSON string.
    -- Format: [[left_eye_x, left_eye_y], [right_eye_x, right_eye_y],
    --          [nose_x, nose_y], [left_mouth_x, left_mouth_y],
    --          [right_mouth_x, right_mouth_y]]
    -- These are used to ALIGN the face (rotate so eyes are level)
    -- before feeding it to the recognition model (M3).
    landmarks   TEXT,

    -- DETECTION CONFIDENCE: how sure YuNet is that this is a real face.
    -- Range: 0.0 (no confidence) to 1.0 (very confident).
    -- We filter out anything below 0.7 to avoid false positives
    -- (e.g., a knot on a tree trunk that vaguely looks like a face).
    det_score   REAL NOT NULL,

    -- QUALITY SCORE: how good this face crop is for recognition.
    -- Based on face size, sharpness, and how frontal the pose is.
    -- Low-quality faces (tiny, blurry, extreme angle) are stored but
    -- excluded from matching — they would poison the person's average.
    quality     REAL,

    -- Path to the 112×112 aligned face crop thumbnail.
    -- This is the image fed to SFace for recognition in M3.
    thumb_path  TEXT,

    -- ── These columns stay NULL until later milestones ──
    -- embedding: 128×float32 faceprint vector (filled in M3)
    embedding   BLOB,
    -- person_id: which person this face belongs to (filled in M4)
    person_id   INTEGER,
    -- How the assignment was made: 'auto' (AI decided), 'user' (human corrected),
    -- or 'review' (confirmed via the review queue)
    assign_source TEXT,
    -- Confidence of the person assignment (0.0-1.0)
    assign_conf REAL
);

-- INDEX: speeds up the most common query "get all faces for image X".
-- Without this index, SQLite would scan EVERY face row. With it, instant lookup.
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);

-- Index for looking up faces by person (used in People panel and clustering)
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);

-- ═══════════════════════════════════════════════════════════════════
-- PERSONS TABLE (added in M4)
-- ═══════════════════════════════════════════════════════════════════
-- Each row = one discovered person (a cluster of similar faces).
-- Initially named "Person 1", "Person 2", etc. The user renames them.
--
-- A person IS a cluster. Renaming just sets display_name.
-- The centroid (average embedding) is the mathematical center of the
-- cluster — used to compare new faces against known people.
CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Display name shown in the UI. Starts as "Person 1", user renames.
    display_name    TEXT NOT NULL,

    -- 0 = auto-generated name, 1 = user has renamed this person.
    -- User-named persons are treated with more care (never auto-merged).
    name_is_user_set INTEGER DEFAULT 0,

    -- The face ID used as the thumbnail on the person card.
    -- Automatically set to the highest-quality face.
    cover_face_id   INTEGER,

    -- Average embedding of all faces in this cluster (128×f32 = 512 bytes).
    -- Updated incrementally as new faces are assigned.
    centroid        BLOB,

    -- Cached count of faces assigned to this person.
    -- Denormalized for performance (avoids COUNT(*) on every card render).
    face_count      INTEGER DEFAULT 0,

    -- If this person was merged into another, stores the target person ID.
    -- NULL = not merged. Merged persons are hidden from the UI. (M5)
    merged_into     INTEGER,

    -- 1 = user has hidden this person from the People panel.
    hidden          INTEGER DEFAULT 0
);
"""

# Current schema version. Bumped to "3" because we added the persons table (M4).
# On existing databases, the CREATE TABLE IF NOT EXISTS handles the upgrade
# gracefully — it only creates the table if it doesn't exist yet.
_SCHEMA_VERSION = "3"


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
    # Face operations (M2)
    # ------------------------------------------------------------------

    def insert_face(
        self,
        image_id: int,
        bbox_x: int,
        bbox_y: int,
        bbox_w: int,
        bbox_h: int,
        det_score: float,
        *,
        landmarks: Optional[str] = None,
        quality: Optional[float] = None,
        thumb_path: Optional[str] = None,
    ) -> int:
        """
        Insert a detected face into the faces table.

        Called by the face detection worker after YuNet finds a face.
        Each face is linked to its parent image via image_id.

        Returns:
            The auto-generated face ID.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO faces
                (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
                 det_score, landmarks, quality, thumb_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
             det_score, landmarks, quality, thumb_path),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_faces_for_image(self, image_id: int) -> list:
        """
        Get all detected faces for a given image.

        Returns:
            A list of Row objects (dict-like), one per face.
            Empty list if no faces were detected in this image.

        Example:
            faces = db.get_faces_for_image(42)
            for face in faces:
                print(f"Face at ({face['bbox_x']}, {face['bbox_y']}), "
                      f"confidence={face['det_score']:.2f}")
        """
        cursor = self._conn.execute(
            "SELECT * FROM faces WHERE image_id = ? ORDER BY id",
            (image_id,),
        )
        return cursor.fetchall()

    def count_faces(self) -> int:
        """Count total faces detected across all images."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]

    def has_faces_detected(self, image_id: int) -> bool:
        """
        Check if face detection has already run on this image.

        We check for ANY face rows (including zero-face results tracked
        via a special marker). This prevents re-running detection on
        images we've already processed.

        For images with no faces, we store a marker in the images table
        (face_count column) so we know detection was run but found nothing.
        """
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM faces WHERE image_id = ?",
            (image_id,),
        )
        return cursor.fetchone()[0] > 0

    def iter_images_needing_detection(self):
        """
        Yield all processed images that haven't had face detection run yet.

        An image needs detection if:
        1. Its status is 'processed' (thumbnailing is done)
        2. It has no rows in the faces table yet

        Uses a LEFT JOIN to efficiently find images without face entries.

        WHAT IS A LEFT JOIN?
        Imagine two spreadsheets side by side:
          - Left: images table
          - Right: faces table
        LEFT JOIN means: show ALL rows from the left (images), and for
        each one, attach matching rows from the right (faces). If there's
        no match, the right side is NULL.

        So "WHERE f.id IS NULL" means "images that have NO faces at all".
        """
        cursor = self._conn.execute(
            """
            SELECT i.*
            FROM images i
            LEFT JOIN faces f ON f.image_id = i.id
            WHERE i.status = 'processed' AND f.id IS NULL
            ORDER BY i.id
            """
        )
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row

    def count_images_needing_detection(self) -> int:
        """Count images that still need face detection."""
        cursor = self._conn.execute(
            """
            SELECT COUNT(*)
            FROM images i
            LEFT JOIN faces f ON f.image_id = i.id
            WHERE i.status = 'processed' AND f.id IS NULL
            """
        )
        return cursor.fetchone()[0]

    def mark_no_faces(self, image_id: int) -> None:
        """
        Record that detection was run on this image but found zero faces.

        We insert a sentinel row with det_score=-1 so that
        iter_images_needing_detection() skips this image on re-runs.
        Without this, we'd re-detect no-face images every time.
        """
        self._conn.execute(
            """
            INSERT INTO faces (image_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score)
            VALUES (?, 0, 0, 0, 0, -1.0)
            """,
            (image_id,),
        )
        self._conn.commit()

    def get_all_images_with_thumbs(self) -> list:
        """
        Get all images that have thumbnails (for the photo grid).

        Returns images ordered by ID, only those with status='processed'
        and a valid thumb_path.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM images
            WHERE status = 'processed' AND thumb_path IS NOT NULL
            ORDER BY id
            """
        )
        return cursor.fetchall()

    # ------------------------------------------------------------------
    # Embedding operations (M3)
    # ------------------------------------------------------------------

    def update_embedding(self, face_id: int, embedding_bytes: bytes) -> None:
        """
        Store a faceprint (embedding) for a detected face.

        The embedding is a 128×float32 vector (512 bytes) computed by SFace.
        We store it as a BLOB (Binary Large Object) — raw bytes in SQLite.

        WHY BLOB AND NOT JSON?
        A 128-float vector as JSON: "[0.0234, -0.1456, ...]" = ~1500 bytes
        As a binary BLOB: 128 × 4 bytes = 512 bytes (3× smaller, instant to load)
        For 100,000 faces, this saves ~100 MB of database size.

        Args:
            face_id:         The face's database ID.
            embedding_bytes: The embedding as raw bytes (numpy .tobytes()).
        """
        self._conn.execute(
            "UPDATE faces SET embedding = ? WHERE id = ?",
            (embedding_bytes, face_id),
        )
        self._conn.commit()

    def iter_faces_needing_embedding(self):
        """
        Yield all faces that have a thumbnail but no embedding yet.

        A face needs an embedding if:
        1. It has a thumb_path (aligned face crop exists on disk)
        2. Its embedding column is NULL
        3. Its det_score > 0 (not a sentinel/no-face marker)
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM faces
            WHERE thumb_path IS NOT NULL
              AND embedding IS NULL
              AND det_score > 0
            ORDER BY id
            """
        )
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row

    def count_faces_needing_embedding(self) -> int:
        """Count faces that still need embedding computation."""
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM faces
            WHERE thumb_path IS NOT NULL
              AND embedding IS NULL
              AND det_score > 0
            """
        )
        return cursor.fetchone()[0]

    def get_all_embeddings(self) -> list:
        """
        Get all face IDs and their embeddings for similarity search.

        Returns a list of rows with: id, image_id, embedding, thumb_path, quality.
        Only includes real faces (det_score > 0) that have embeddings.

        WHAT IS THIS FOR?
        When the user clicks a face and asks "Find Similar", we need to
        compare that face's embedding against ALL other embeddings.
        This loads them all into memory for fast numpy computation.
        For 100k faces, this is ~50 MB of data — fits easily in RAM.
        """
        cursor = self._conn.execute(
            """
            SELECT id, image_id, embedding, thumb_path, quality, det_score
            FROM faces
            WHERE embedding IS NOT NULL AND det_score > 0
            ORDER BY id
            """
        )
        return cursor.fetchall()

    def get_face_by_id(self, face_id: int):
        """
        Get a single face row by its ID.

        Returns:
            A sqlite3.Row object, or None if not found.
        """
        cursor = self._conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        )
        return cursor.fetchone()

    # ------------------------------------------------------------------
    # Person operations (M4)
    # ------------------------------------------------------------------

    def insert_person(
        self,
        display_name: str,
        centroid_bytes: bytes,
        face_count: int = 0,
        cover_face_id: Optional[int] = None,
    ) -> int:
        """
        Create a new person (cluster) in the database.

        A person starts as "Person 1", "Person 2", etc. The centroid is
        the average faceprint of all faces in the cluster — initially
        just the seed face's embedding.

        Args:
            display_name:   The name shown in the UI (e.g., "Person 1").
            centroid_bytes: The 128×f32 centroid embedding as raw bytes.
            face_count:     Initial face count (usually 1 for a new person).
            cover_face_id:  The face to use as the card thumbnail.

        Returns:
            The auto-generated person ID.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO persons
                (display_name, centroid, face_count, cover_face_id)
            VALUES (?, ?, ?, ?)
            """,
            (display_name, centroid_bytes, face_count, cover_face_id),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_person_centroid(
        self,
        person_id: int,
        centroid_bytes: bytes,
        face_count: int,
    ) -> None:
        """
        Update a person's centroid and face count after a new face is assigned.

        The centroid is recomputed incrementally by the clustering engine
        and stored here. face_count is the new total (not a delta).
        """
        self._conn.execute(
            """
            UPDATE persons SET centroid = ?, face_count = ?
            WHERE id = ?
            """,
            (centroid_bytes, face_count, person_id),
        )
        self._conn.commit()

    def rename_person(self, person_id: int, new_name: str) -> None:
        """
        Rename a person and mark the name as user-set.

        User-set names are treated with more care:
        - They are never overwritten by auto-naming
        - In M5, user-named persons are never auto-merged

        Args:
            person_id: The person to rename.
            new_name:  The new display name.
        """
        self._conn.execute(
            """
            UPDATE persons SET display_name = ?, name_is_user_set = 1
            WHERE id = ?
            """,
            (new_name, person_id),
        )
        self._conn.commit()

    def assign_face_to_person(
        self,
        face_id: int,
        person_id: int,
        source: str = "auto",
        confidence: float = 0.0,
    ) -> None:
        """
        Assign a face to a person.

        Args:
            face_id:    The face to assign.
            person_id:  The person to assign it to.
            source:     How the assignment was made:
                        'auto' = clustering algorithm decided
                        'user' = human explicitly assigned (M5)
                        'review' = confirmed via review queue (M5)
            confidence: The similarity score that led to this assignment.
        """
        self._conn.execute(
            """
            UPDATE faces SET person_id = ?, assign_source = ?, assign_conf = ?
            WHERE id = ?
            """,
            (person_id, source, confidence, face_id),
        )
        self._conn.commit()

    def get_all_persons(self) -> list:
        """
        Get all visible, non-merged persons, ordered by face count (most first).

        This is the main query for the People panel — shows person cards
        sorted by how many photos they appear in.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM persons
            WHERE hidden = 0 AND merged_into IS NULL
            ORDER BY face_count DESC
            """
        )
        return cursor.fetchall()

    def get_person_by_id(self, person_id: int):
        """
        Get a single person row by ID.

        Returns:
            A sqlite3.Row object, or None if not found.
        """
        cursor = self._conn.execute(
            "SELECT * FROM persons WHERE id = ?", (person_id,)
        )
        return cursor.fetchone()

    def get_faces_for_person(self, person_id: int) -> list:
        """
        Get all faces assigned to a person, ordered by quality (best first).

        Used by the People panel to show face thumbnails for a person,
        and by the clustering engine to pick the cover face.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM faces
            WHERE person_id = ? AND det_score > 0
            ORDER BY quality DESC
            """,
            (person_id,),
        )
        return cursor.fetchall()

    def count_unassigned_faces(self) -> int:
        """
        Count faces that have embeddings but haven't been assigned to a person.

        These are the faces that need clustering in M4.
        Only counts real faces (det_score > 0) with valid embeddings.
        """
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM faces
            WHERE embedding IS NOT NULL
              AND person_id IS NULL
              AND det_score > 0
            """
        )
        return cursor.fetchone()[0]

    def iter_unassigned_faces(self):
        """
        Yield faces that need clustering: have embeddings but no person.

        Ordered by QUALITY DESCENDING — best faces first. This is critical
        because high-quality faces make better cluster seeds (more reliable
        centroids). Processing the best faces first means clusters start
        with strong anchors.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM faces
            WHERE embedding IS NOT NULL
              AND person_id IS NULL
              AND det_score > 0
            ORDER BY quality DESC
            """
        )
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row

    def get_next_person_number(self) -> int:
        """
        Get the next person number for auto-naming ("Person N").

        Returns the count of existing persons + 1, so names are sequential.
        """
        cursor = self._conn.execute("SELECT COUNT(*) FROM persons")
        return cursor.fetchone()[0] + 1

    def update_person_cover(self, person_id: int, face_id: int) -> None:
        """
        Set the cover face for a person card.

        The cover face is the thumbnail shown on the person card in the
        People panel. We pick the highest-quality face automatically.
        """
        self._conn.execute(
            "UPDATE persons SET cover_face_id = ? WHERE id = ?",
            (face_id, person_id),
        )
        self._conn.commit()

    def count_persons(self) -> int:
        """Count all visible, non-merged persons."""
        cursor = self._conn.execute(
            """
            SELECT COUNT(*) FROM persons
            WHERE hidden = 0 AND merged_into IS NULL
            """
        )
        return cursor.fetchone()[0]

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
