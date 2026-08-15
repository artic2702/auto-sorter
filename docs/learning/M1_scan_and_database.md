# Milestone 1 — Scan Folder → SQLite + Thumbnails + Live Progress

## 🎯 Goal
User picks a photo folder → app finds every image, saves metadata to a database, generates small thumbnails, and shows live progress — all without the UI freezing.

---

## 📦 Concepts Learned

### 1. SQLite — Your App's Memory

**What is a database?**
Imagine a spreadsheet saved as a single file. Instead of opening Excel to read it, your code reads/writes it directly. That's SQLite.

```
library.db (one file on disk)
├── Table: images
│   ├── Row 1: id=1, path="vacation/beach.jpg", size=4521899, status="processed"
│   ├── Row 2: id=2, path="family/birthday.png", size=1283744, status="pending"
│   └── Row 3: id=3, path="selfie.jpg", size=892341, status="processed"
└── Table: meta
    └── Row 1: key="schema_version", value="1"
```

**Why SQLite (not a regular file)?**

| Feature | Text/JSON file | SQLite |
|---|---|---|
| Find one image by path | Read ENTIRE file, search | Instant (indexed lookup) |
| Count pending images | Read + parse + count | `SELECT COUNT(*) WHERE status='pending'` |
| Crash mid-write | File corrupted 💀 | Database safe ✅ (atomic writes) |
| 100,000 rows | Slow, huge memory | Fast, low memory |

**Key SQL Commands We Use**:

```sql
-- Create a table (like creating a spreadsheet with column headers)
CREATE TABLE IF NOT EXISTS images (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,  -- auto-numbering
    path    TEXT UNIQUE NOT NULL,                -- no duplicates allowed
    size    INTEGER NOT NULL,
    status  TEXT DEFAULT 'pending'
);

-- Insert a row
INSERT INTO images (path, size) VALUES ('beach.jpg', 4521899);

-- Find a specific image
SELECT * FROM images WHERE path = 'beach.jpg';

-- Count images
SELECT COUNT(*) FROM images;
SELECT COUNT(*) FROM images WHERE status = 'pending';

-- Update a row
UPDATE images SET status = 'processed' WHERE id = 1;

-- Upsert (insert or replace if path already exists)
INSERT OR REPLACE INTO images (path, size, status) VALUES ('beach.jpg', 4521899, 'pending');
```

**WAL Mode (Write-Ahead Logging)**:
```python
conn.execute("PRAGMA journal_mode=WAL")
```
Normal SQLite: when writing, the database is LOCKED — no one can read.
WAL mode: readers and writers work simultaneously. The UI thread can read (to display photos) while the scanner thread writes (to add new images). Also crash-safe.

---

### 2. File Walking — How the Scanner Finds Photos

**The problem**: Given a folder like `D:\Photos`, find every `.jpg`, `.png`, etc. inside it and all subfolders, recursively.

**Our solution** uses `os.scandir()` with a manual stack:

```python
dirs_to_visit = [root]        # Start with the root folder

while dirs_to_visit:           # Keep going until no more folders
    current = dirs_to_visit.pop()   # Take the next folder

    for entry in os.scandir(current):
        if entry.is_dir():
            dirs_to_visit.append(entry.path)   # Visit this subfolder later
        elif entry.is_file():
            if is_image_file(entry.name):      # Check extension
                # Process this image...
```

**Visual example**:
```
Step 1: dirs_to_visit = [D:\Photos]
  → Pop D:\Photos, scan it
  → Found: vacation/ (dir), family/ (dir), selfie.jpg (file ✅)
  → dirs_to_visit = [D:\Photos\vacation, D:\Photos\family]

Step 2: Pop D:\Photos\family
  → Found: birthday.png (file ✅)
  → dirs_to_visit = [D:\Photos\vacation]

Step 3: Pop D:\Photos\vacation
  → Found: beach.jpg (file ✅), notes.txt (file ❌ not an image)
  → dirs_to_visit = [] ← EMPTY, we're done!

Result: found 3 images
```

**Why os.scandir() instead of os.walk()?**
`os.scandir()` returns `DirEntry` objects that cache file metadata (size, type). One system call gets everything. `os.walk()` needs extra `stat()` calls for each file → slower on large folders.

**Skip unchanged files (the resume trick)**:
```python
if db.is_unchanged(entry.path, file_size, file_mtime):
    skipped += 1    # Already processed, skip!
else:
    db.upsert_image(path=entry.path, size=file_size, mtime=file_mtime)
    new += 1        # New file, add to DB
```
This is why re-scans are instant — we compare `size + mtime` (modification time) against what's stored in the DB. If both match, the file hasn't changed.

---

### 3. Thumbnail Generation — Why and How

**The problem**: A single phone photo is ~5-15 MB. Loading 500 photos into memory = 5 GB. Your PC would freeze or crash.

**The solution**: Create tiny 256×256 thumbnails (~15-30 KB each). The UI shows thumbnails; originals are loaded only when clicked.

```python
# Step 1: Read image (Unicode-safe for Windows)
raw_bytes = np.fromfile(file_path, dtype=np.uint8)
img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

# Step 2: Calculate new size (fit in 256×256 box, keep aspect ratio)
scale = 256 / max(height, width)
new_w = int(width * scale)
new_h = int(height * scale)

# Step 3: Resize (INTER_AREA = best quality for shrinking)
thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Step 4: Save as compressed JPEG
success, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
encoded.tofile(str(thumb_path))
```

**Why `np.fromfile()` + `cv2.imdecode()` instead of `cv2.imread()`?**

This is a critical Windows-specific issue:
```python
# ❌ BROKEN on Windows with Unicode paths:
img = cv2.imread("D:\\Photos\\अर्णव\\pic.jpg")   # Returns None silently!

# ✅ WORKS with any filename:
raw = np.fromfile("D:\\Photos\\अर्णव\\pic.jpg", dtype=np.uint8)
img = cv2.imdecode(raw, cv2.IMREAD_COLOR)         # Works perfectly!
```
`cv2.imread()` uses C's `fopen()` internally, which can't handle Unicode paths on Windows. `np.fromfile()` uses Python's file I/O, which handles them correctly.

**Aspect ratio math example**:
```
Original: 4000×3000 (landscape)
max(4000, 3000) = 4000
scale = 256 / 4000 = 0.064
new_w = 4000 × 0.064 = 256
new_h = 3000 × 0.064 = 192
Result: 256×192 ✅ (fits in 256×256 box, aspect ratio preserved)
```

---

### 4. SHA-256 Hashing — Detecting Moved Files

**The problem**: User moves `beach.jpg` from `D:\Photos` to `D:\Backup`. The PATH changed, but it's the SAME file. Without hashing, we'd re-process it from scratch and lose all face data.

**The solution**: Compute a unique "fingerprint" (hash) of the file's contents.

```python
import hashlib

h = hashlib.sha256()
with open(file_path, "rb") as f:
    while True:
        chunk = f.read(65536)    # Read 64 KB at a time
        if not chunk:
            break
        h.update(chunk)
print(h.hexdigest())
# → "a1b2c3d4e5f6...7890" (64-character string, always the same for the same file)
```

**Properties of SHA-256**:
- Same content → always the same hash (deterministic)
- Change 1 byte → completely different hash (avalanche effect)
- Different files → (virtually) impossible to get the same hash
- Fast: ~500 MB/s on modern CPUs

---

### 5. Threading — Why the UI Freezes (and How to Fix It)

**The problem**:
```python
# If scanning runs on the main thread:
def _start_scan(self):
    scan_folder(root, db)           # Takes 30 seconds...
    # ⚠️ During those 30 seconds:
    # - UI is FROZEN (can't click anything)
    # - Progress bar doesn't move
    # - Windows shows "Not Responding"
```

**Why it freezes**: Qt's main thread runs a single event loop:
```
Wait for event → Handle it → Wait for event → Handle it → ...
```
If "Handle it" takes 30 seconds (scanning), no other events are processed. Clicks, redraws, and progress updates all queue up and nothing happens.

**The fix — QThread (background thread)**:

```
Main Thread (UI):  [event loop running, responsive]
                     ↑ signals (progress updates)
                     |
Worker Thread:     [scanning files, making thumbnails]
```

```python
class ScanWorker(QThread):
    progress = Signal(int, int)      # Define a signal
    finished = Signal(int, int)

    def run(self):                   # Runs in background thread
        for file in scan_all_files():
            process(file)
            self.progress.emit(done, total)   # Send update to UI
        self.finished.emit(total, errors)
```

In the UI:
```python
worker = ScanWorker(folder, data_dir)
worker.progress.connect(self.update_progress_bar)   # Connect signal → slot
worker.finished.connect(self.on_done)
worker.start()       # ← Starts run() in a NEW thread
# Main thread continues immediately → UI stays responsive!
```

---

### 6. Qt Signals and Slots — Thread-Safe Communication

**The restaurant analogy**:
- **Main thread** = Waiter (serves customers, takes orders)
- **Worker thread** = Chef (cooks food in the kitchen)
- **Signal** = Kitchen bell 🔔 ("order ready!")
- **Slot** = Waiter's response ("pick up food and serve it")

```python
# DEFINING a signal (in the worker):
class ScanWorker(QThread):
    scan_progress = Signal(int, int, int)   # carries 3 integers

# EMITTING a signal (from the worker thread):
    self.scan_progress.emit(found, new, skipped)   # "Ring the bell!"

# CONNECTING signal to slot (in the UI):
    worker.scan_progress.connect(self._on_scan_progress)   # "When bell rings, do this"

# THE SLOT (runs on the UI thread automatically):
    def _on_scan_progress(self, found, new, skipped):
        self._scan_label.setText(f"Found {found} images ({new} new)")
```

**Why not just call the UI function directly from the worker?**
Qt widgets are NOT thread-safe. Updating a label from a background thread can crash the app or corrupt the display. Signals automatically "marshal" the call to the main thread, making it safe.

---

### 7. Qt Layout System — Arranging Widgets

```python
# Vertical layout (top to bottom)
layout = QVBoxLayout()
layout.addWidget(title_label)        # Row 1
layout.addWidget(subtitle_label)     # Row 2
layout.addLayout(photo_row)          # Row 3 (a horizontal sub-layout)
layout.addWidget(start_button)       # Row 4

# Horizontal layout (left to right) — used for one row
photo_row = QHBoxLayout()
photo_row.addWidget(QLabel("📁 Photo folder:"))
photo_row.addWidget(path_label, stretch=1)   # stretch=1: take remaining space
photo_row.addWidget(browse_button)
```

**Visual result**:
```
┌─────────────────────────────────────┐
│ Auto-Sorter              ← VBox row 1
│ Privacy-first...         ← VBox row 2
│ 📁 Photo: /path  [Browse] ← HBox inside VBox row 3
│ [▶ Start Scan]           ← VBox row 4
└─────────────────────────────────────┘
```

`addStretch()` adds invisible elastic space — pushes widgets to one side:
```python
btn_row.addWidget(start_button)
btn_row.addStretch()    # ← pushes everything LEFT, empty space on RIGHT
```

---

## 📋 Files Created / Modified in M1

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/__init__.py` | NEW | Makes core/ a Python package |
| `autosorter/core/db.py` | NEW | SQLite database layer (create, read, write) |
| `autosorter/core/scanner.py` | NEW | File walker (find images, skip unchanged) |
| `autosorter/core/thumbnailer.py` | NEW | Thumbnail generator + SHA-256 hashing |
| `autosorter/core/scan_worker.py` | NEW | Background QThread (scanner + thumbnailer) |
| `autosorter/app/main_window.py` | MODIFIED | Real UI: folder pickers, progress bar, buttons |
| `tests/test_m1_scan.py` | NEW | 20 automated tests for DB, scanner, thumbnailer |

---

## 🐛 Bug Found and Fixed

**Bug**: Re-scanning doubled the image count (58 → 116 → 174...).

**Cause**: Thumbnails were saved in `.autosorter/thumbs/` INSIDE the photo folder. The scanner found those `.jpg` thumbnails and treated them as new photos!

**Fix**: Added a `SKIP_DIR_NAMES` set to the scanner:
```python
SKIP_DIR_NAMES = frozenset({
    ".autosorter",  # our own data folder
    ".git",         # version control
    "$RECYCLE.BIN", # Windows recycle bin
})

# In the scanning loop:
if entry.is_dir():
    if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
        continue   # Don't scan this folder
```

**Lesson**: Always think about what your app creates and whether it could interfere with its own logic.

---

## ✅ How to Verify M1 Works

```powershell
# All 23 tests pass
.\.venv\Scripts\python -m pytest tests/ -v

# App opens, scan a folder, see progress
.\.venv\Scripts\python -m autosorter

# Expected behavior:
# 1. Select a folder with photos → Start Scan
# 2. Phase 1: finds images (live counter)
# 3. Phase 2: generates thumbnails (progress bar)
# 4. Status: "✅ Scan complete: X images found"
# 5. Re-Scan → instant! "0 new, X unchanged"
```

---

## 🧪 How Data Flows Through M1

```
User clicks "Browse" → picks D:\Photos
User clicks "Start Scan"
        │
        ▼
    ScanWorker.start()                    [launches background thread]
        │
        ▼
    scan_folder(D:\Photos, db)            [Phase 1: File Discovery]
        │
        ├── os.scandir(D:\Photos)
        │   ├── vacation/beach.jpg  → is_image? ✅ → in DB? ❌ → INSERT (pending)
        │   ├── notes.txt           → is_image? ❌ → skip
        │   └── vacation/           → is_dir? ✅  → add to stack
        │
        │   emit scan_progress(found=1, new=1, skipped=0) ──→ UI updates label
        │
        ▼
    for row in db.iter_pending():          [Phase 2: Thumbnailing]
        │
        ├── beach.jpg → read → resize → save thumb → compute hash
        │   → db.update_thumb(id, thumb_path, width, height)
        │   → db.update_status(id, "processed")
        │
        │   emit thumb_progress(done=1, total=1) ──→ UI updates progress bar
        │
        ▼
    emit finished(total=1, errors=0) ──→ UI shows "✅ Scan complete: 1 image"
```
