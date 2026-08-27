# Challenge 05 — Moved & Renamed Photo Reattachment via SHA-256

## 🚨 The Symptom
Users frequently organize their photo libraries outside the application:
- Moving photos into year or event subfolders (e.g., `D:\Photos\vacation\beach.jpg` $\rightarrow$ `D:\Photos\2024\beach.jpg`).
- Renaming camera files from `IMG_0042.JPG` to `Arnav_Birthday_42.JPG`.

In early versions of the app:
- The scanner relied solely on the file path string (`path TEXT UNIQUE`) as the primary identifier.
- Moving or renaming a file caused the old path to be discarded and the new path to be treated as an unknown, brand-new photo.
- All previously detected faces, thumbnails, embeddings, and human-corrected person assignments were lost for that photo, forcing full re-detection and re-clustering.

---

## 🔍 Root Cause Analysis

A file's location on disk (`path`) is a mutable pointer, not an intrinsic identity:
1. When a photo is moved:
   - `os.scandir` discovers `2024/beach.jpg` as a new path not present in `images`.
   - The original path `vacation/beach.jpg` no longer exists on disk.
2. If the database deletes the old row and inserts the new row, the foreign keys (`faces.image_id`) point to dead records.
3. If the database simply marks the old path missing and creates a new row, the library ends up with duplicate records and lost historical assignments.

---

## 🛠️ The Solution

We implemented a two-part identification strategy combining **fast metadata checks** with **SHA-256 cryptographic content hashing**:

### 1. SHA-256 Hashing on Thumbnail Generation
During thumbnail generation, we compute the SHA-256 hash of the photo's raw contents and store it in `images.sha256`:
```python
def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
```

### 2. In-Place Reattachment (`db.reattach_by_hash`)
When a new file path is discovered during a scan:
1. We compute its SHA-256 hash.
2. We query SQLite for any existing row matching that hash:
   ```python
   def reattach_by_hash(self, sha256: str, new_path: str) -> Optional[int]:
       cursor = self._conn.execute(
           """
           SELECT id, path FROM images
           WHERE sha256 = ? AND path != ?
           ORDER BY
               CASE status WHEN 'missing' THEN 0 ELSE 1 END,
               id
           LIMIT 1
           """,
           (sha256, new_path),
       )
       row = cursor.fetchone()
       if row is None:
           return None

       # Update path in-place — preserves row ID and all linked faces!
       self._conn.execute(
           "UPDATE images SET path = ?, status = 'processed' WHERE id = ?",
           (new_path, row["id"]),
       )
       self._conn.commit()
       return row["id"]
   ```
3. Prioritizing `status = 'missing'` ensures that when `vacation/beach.jpg` goes missing and `2024/beach.jpg` appears, the existing record is updated in-place.
4. All child rows in `faces` retain their foreign key link to `images.id`. The face bounding boxes, embeddings, and person assignments are preserved with zero re-processing!

---

## 📊 Verification & Tests

Added unit tests in `tests/test_m6_resume.py`:
- `test_reattach_preserves_face_and_person`: Moves an image to a new path, marks old path missing, triggers reattach, and asserts that face and person data survive.
- `test_marks_deleted_files_as_missing`: Verifies deleted files are marked `missing` without wiping records.

---

## 🧠 Key Takeaway
**Never rely on file paths as primary identity keys in local file management software.** Track files by content hashes (SHA-256) so moves and renames are seamlessly reconciled in-place.
