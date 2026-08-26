# Milestone 6 — Resume & Incremental Scan

## 🎯 Goal
Make Auto-Sorter **feel instant on every launch after the first scan**:
1. **One button** — "▶ Start" runs the entire pipeline (scan → detect → embed → cluster) automatically.
2. **Resume** — kill the app mid-pipeline, relaunch → it picks up exactly where it stopped.
3. **New photos only** — add 50 photos to the folder → re-scan processes only those 50.
4. **Moved/renamed file reattach** — move a photo to another subfolder → SHA-256 hash links it to the existing faces and person assignments.
5. **Deleted file cleanup** — files removed from disk are marked `missing` in the DB (not deleted — they might come back).

---

## 📦 Concepts Learned

### 1. SQLite IS the Checkpoint

The key insight of M6: **SQLite is the crash-recovery mechanism**. We don't need a separate checkpoint file, progress log, or resume state. Every item is committed to the database individually, and each pipeline stage simply queries for "what hasn't been done yet":

```
Stage 1 (Scan):     SELECT * FROM images WHERE status = 'pending'
Stage 2 (Detect):   SELECT * FROM images WHERE no face rows exist
Stage 3 (Embed):    SELECT * FROM faces WHERE embedding IS NULL
Stage 4 (Cluster):  SELECT * FROM faces WHERE person_id IS NULL
```

If the app crashes between item 500 and 501:
- Items 1–500 are already committed to SQLite
- On restart, each query skips them automatically
- Work resumes from item 501 with zero wasted effort

```
First run (killed at 60%):
  Images:    [■■■■■■░░░░]  600/1000 processed
  Faces:     [■■■■░░░░░░]  400/1000 detected
  Embeddings [■■■░░░░░░░]  300/1000 computed
  People:    [■■░░░░░░░░]  200/1000 clustered

Restart (picks up exactly):
  Images:    [■■■■■■■■■■]  1000/1000 (400 new, 600 skipped)
  Faces:     [■■■■■■■■■■]  1000/1000 (600 new, 400 skipped)
  Embeddings [■■■■■■■■■■]  1000/1000 (700 new, 300 skipped)
  People:    [■■■■■■■■■■]  1000/1000 (800 new, 200 skipped)
```

---

### 2. The Upsert Bug — INSERT OR REPLACE vs ON CONFLICT

**The most critical bug fix in this milestone.** The original `upsert_image` used `INSERT OR REPLACE`:

```sql
-- OLD (DANGEROUS):
INSERT OR REPLACE INTO images (path, size, mtime, ...) VALUES (?, ?, ?, ...)
```

**What `INSERT OR REPLACE` actually does:**
1. Checks if a row with the same `path` UNIQUE key exists
2. If yes: **DELETES the entire old row** (fires ON DELETE CASCADE!)
3. Inserts a new row with a **new auto-increment ID**

**The cascade catastrophe:**
```sql
faces(image_id) REFERENCES images(id) ON DELETE CASCADE
```
When the old image row is deleted, ALL linked faces are automatically deleted too. This means:
- Re-scanning an unchanged photo folder **silently destroys** all face detection work
- All embeddings: gone
- All person assignments: gone
- Hours of clustering and manual corrections: gone

**The fix — true upsert with ON CONFLICT:**
```sql
-- NEW (SAFE):
INSERT INTO images (path, size, mtime, ...)
VALUES (?, ?, ?, ...)
ON CONFLICT(path) DO UPDATE SET
    size   = excluded.size,
    mtime  = excluded.mtime,
    status = excluded.status
```

This updates the row **in place** — preserving the original row ID and all foreign key relationships. Faces, embeddings, and person assignments survive re-scans perfectly.

---

### 3. Moved-File Reattach via SHA-256

When a user reorganizes their photo folder (moves files to subfolders, renames them), the file path changes but the content is identical. Auto-Sorter detects this using SHA-256 content hashing:

```
Before:  /Photos/vacation/beach.jpg   SHA256=abc123
                                       ↳ 3 faces detected, assigned to Arnav, Mom, Dad

User moves file:  /Photos/2024/summer/beach.jpg

After scan:
  1. Old path /Photos/vacation/beach.jpg → NOT found on disk → marked 'missing'
  2. New path /Photos/2024/summer/beach.jpg → NEW file found
  3. Compute SHA256 of new file → abc123 (matches!)
  4. Call db.reattach_by_hash("abc123", new_path)
  5. UPDATE images SET path = new_path, status = 'processed' WHERE id = old_id
  6. All 3 faces and person assignments are preserved!
```

The implementation prioritizes `missing` rows over `processed` rows in the hash match, ensuring moved files connect to the right record:

```python
def reattach_by_hash(self, sha256, new_path):
    cursor = self._conn.execute("""
        SELECT id, path FROM images
        WHERE sha256 = ? AND path != ?
        ORDER BY
            CASE status WHEN 'missing' THEN 0 ELSE 1 END,
            id
        LIMIT 1
    """, (sha256, new_path))
```

---

### 4. The Unified Pipeline Architecture

Before M6: 3 separate QThread workers, 3 buttons, manual sequencing.
After M6: 1 PipelineWorker, 1 button, automatic 4-stage execution.

```
┌──────────────────────────────────────────────────────────────────┐
│  PipelineWorker(QThread)                                         │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Stage 1  │→ │ Stage 2  │→ │ Stage 3  │→ │   Stage 4    │    │
│  │  SCAN    │  │  DETECT  │  │  EMBED   │  │  IDENTIFY    │    │
│  │          │  │          │  │          │  │              │    │
│  │ Walk dirs│  │ YuNet on │  │ SFace on │  │ Cluster into │    │
│  │ Thumbnail│  │ new imgs │  │ new faces│  │ persons      │    │
│  │ SHA-256  │  │          │  │          │  │              │    │
│  │ Reattach │  │          │  │          │  │ pick_cover   │    │
│  │ Missing  │  │          │  │          │  │              │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│       ↕              ↕             ↕              ↕              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   SQLite (WAL mode)                       │   │
│  │  Per-item commits → crash-safe → skip-already-done       │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
         │
    Qt Signals
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  MainWindow (UI thread)                                          │
│                                                                  │
│  stage_changed("Detecting faces", 2, 4)                          │
│    → "Stage 2/4: Detecting faces"                                │
│                                                                  │
│  progress(42, 340, "Faces: 42/340 images (15 faces)")           │
│    → progress bar + detail label                                 │
│                                                                  │
│  finished({files_new: 50, faces: 15, persons: 3, ...})          │
│    → "✅ Done. 50 new photos processed. 3 people."              │
└──────────────────────────────────────────────────────────────────┘
```

**Signals:**
- `stage_changed(name, num, total)` — updates the stage indicator
- `progress(done, total, detail_text)` — drives the progress bar and label
- `finished(summary_dict)` — populates the completion message
- `error(message)` — shows error state with "Resume" button

---

### 5. Missing-File Detection Strategy

When a file is deleted from disk, Auto-Sorter marks it as `status='missing'` but does **not** delete the database row or its face/person data. Why?

1. **The file might come back** — user could undo the delete, restore from backup, or it's on an external drive that's temporarily disconnected.
2. **The file might have been moved** — `reattach_by_hash` will connect it to the new path.
3. **Person data is precious** — hours of manual corrections and name assignments shouldn't be lost because a USB drive was unplugged.

```python
def mark_missing_images(self, known_paths: set) -> int:
    # Compare DB paths against paths found on disk
    rows = self._conn.execute(
        "SELECT id, path FROM images WHERE status != 'missing'"
    ).fetchall()
    
    missing_count = 0
    for row in rows:
        if row["path"] not in known_paths:
            self._conn.execute(
                "UPDATE images SET status = 'missing' WHERE id = ?",
                (row["id"],)
            )
            missing_count += 1
    return missing_count
```

---

### 6. Smart Button State Machine

The Start button adapts its text and behavior based on the library state:

```
No database exists          → "▶ Start"
Database exists, work left  → "▶ Resume"  (pending/unassigned items)
Database exists, all done   → "▶ Re-Scan" (checks for new photos)
```

On folder selection, the UI immediately shows a library summary:
```
📚 Library: 1,234 photos · 456 faces · 12 people
Previous run interrupted. 50 pending, 23 unassigned faces.
```

---

## 📋 Files Created / Modified in M6

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/db.py` | MODIFIED | True upsert fix, `mark_missing_images`, `reattach_by_hash`, `get_pipeline_state`, `get_all_image_paths` |
| `autosorter/core/pipeline_worker.py` | NEW | Unified 4-stage background worker |
| `autosorter/core/scanner.py` | MODIFIED | `discovered_paths` set in `ScanResult` for missing-file detection |
| `autosorter/app/main_window.py` | MODIFIED | Single Start/Resume button, library summary, stage progress |
| `tests/test_m6_resume.py` | NEW | 15 tests covering upsert, reattach, missing, pipeline state |

---

## ✅ How to Verify M6 Works

```powershell
# Run the full test suite (129 tests)
.\.venv\Scripts\python -m pytest tests/ -v

# Launch the app
.\.venv\Scripts\python -m autosorter
```

### Manual Verification Flow:
1. **First Run:** Browse → Select photo folder → Click "▶ Start" → All 4 stages run automatically → People panel appears.
2. **Resume Test:** Kill the app mid-pipeline → Relaunch → Select same folder → Button says "▶ Resume" → Click it → Picks up exactly where stopped.
3. **Incremental Test:** Add new photos to the folder → Click "▶ Re-Scan" → Only new photos are processed.
4. **Moved File Test:** Move a photo to another subfolder → Re-Scan → Photo is reattached via SHA-256, faces and person assignments preserved.
5. **Deleted File Test:** Delete a photo → Re-Scan → Photo marked as missing in the DB (face data preserved).
