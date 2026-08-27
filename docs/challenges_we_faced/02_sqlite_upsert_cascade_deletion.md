# Challenge 02 — The `INSERT OR REPLACE` Foreign Key Cascade Disaster

## 🚨 The Symptom
During incremental re-scans (Milestone 6), re-scanning a folder that had already been scanned, detected, and organized would sometimes wipe out all face detections, embeddings, and person cards, forcing the app to re-detect all faces from scratch.

---

## 🔍 Root Cause Analysis

In SQLite, our database schema defines a foreign key constraint with cascade deletion:
```sql
CREATE TABLE faces (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id  INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    ...
);
```

In `autosorter/core/db.py`, the image registration function was written as:
```python
# OLD CODE (M1 - M5):
cursor = self._conn.execute(
    """
    INSERT OR REPLACE INTO images
        (path, size, mtime, sha256, width, height, taken_at,
         thumb_path, status, error)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (...)
)
```

### The SQLite Trap: What `INSERT OR REPLACE` Actually Does
Many developers assume `INSERT OR REPLACE` is synonymous with an SQL `UPSERT` (update existing columns). **It is not.**

In SQLite standard semantics, when `INSERT OR REPLACE` encounters a row that violates a `UNIQUE` constraint (here, `path UNIQUE`):
1. SQLite **DELETES the old row entirely**.
2. Deleting that row fires all foreign key `ON DELETE CASCADE` triggers!
3. All child rows in the `faces` table pointing to that `image_id` are **instantly and permanently deleted**!
4. SQLite then inserts a **brand-new row** with a **new auto-increment ID**.
5. All thumbnail paths, 128-d embeddings, person associations, and manual user corrections linked to the previous ID were lost.

---

## 🛠️ The Solution

In SQLite 3.24+, true upsert syntax (`INSERT ... ON CONFLICT DO UPDATE`) was introduced.

We refactored `db.upsert_image`:
```python
# NEW CODE (M6+):
cursor = self._conn.execute(
    """
    INSERT INTO images
        (path, size, mtime, sha256, width, height, taken_at,
         thumb_path, status, error)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(path) DO UPDATE SET
        size   = excluded.size,
        mtime  = excluded.mtime,
        status = excluded.status,
        error  = excluded.error
    """,
    (...)
)
```

### Why `ON CONFLICT DO UPDATE` Fixes the Issue:
- The existing row is updated **in place**.
- The row's primary key `id` is **preserved**.
- No `DELETE` operation occurs, meaning `ON DELETE CASCADE` is never triggered.
- All child rows in `faces`, their embeddings, person assignments, and manual corrections stay 100% intact across re-scans.

---

## 📊 Verification & Tests

We added unit tests in `tests/test_m6_resume.py`:
```python
def test_upsert_preserves_faces_on_rescan(self, tmp_path):
    # Inserts image + face + person
    # Re-scans image with updated mtime
    # Asserts face, embedding, and person assignment are still intact
```
The test passes and confirms that re-scanning unchanged or modified files never destroys face records.

---

## 🧠 Key Takeaway
**Never use `INSERT OR REPLACE` in SQLite when child tables reference the parent table with `ON DELETE CASCADE`.** Always use `INSERT ... ON CONFLICT(column) DO UPDATE` to perform in-place column updates without triggering deletion triggers.
