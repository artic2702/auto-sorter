# Milestone 7 — Smart Albums & Safe Export

## 🎯 Goal

Provide intelligent organization and safe export tools without touching or risking user originals:
1. **Virtual Smart Albums** — Instant browsing for **Group Photos**, **Unknown Faces**, and **Text & Documents** (powered by zero-copy SQLite queries).
2. **Lightweight Document/Screenshot Classifier** — High-speed heuristic scoring (< 5ms per image) that separates screenshots, receipts, and text graphics from camera photos without heavy ML dependencies.
3. **Non-Destructive Safe Export Engine** — An export pipeline that copies photos into clean, organized person folders (`Export/Arnav/IMG_001.jpg`), duplicates group photos into each participant's folder, resolves filename collisions, and supports resumable copying.

---

## 📦 Concepts Learned

### 1. Virtual Albums vs. Physical Folder Duplication

In naive photo managers, organizing photos into "Group Photos" or "People" albums creates copies on disk or moves files around, breaking directory structures and consuming double storage.

Auto-Sorter uses **Virtual Smart Albums**:
- The files on disk remain in their original directories, completely untouched.
- Albums are **live views** generated dynamically by SQLite queries on metadata and face relations.
- An album with 5,000 photos costs **0 bytes of extra disk space**.

```
┌────────────────────────────────────────────────────────┐
│                   Original Photo Library               │
│                  (D:\Photos\2024\...)                  │
└───────────────────────────┬────────────────────────────┘
                            │ Read-Only Queries
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       👥 Group Photos  ❓ Unknown   📄 Text & Docs
       (≥2 people)      (Unassigned)  (doc_score ≥ 0.5)
```

---

### 2. The Group Photo SQL Relationship

A "group photo" is defined as any photograph containing **two or more distinct identified individuals**. 

```sql
SELECT i.*
FROM images i
WHERE i.status = 'processed'
  AND i.id IN (
      SELECT f.image_id
      FROM faces f
      WHERE f.person_id IS NOT NULL
      GROUP BY f.image_id
      HAVING COUNT(DISTINCT f.person_id) >= 2
  )
ORDER BY COALESCE(i.taken_at, '') DESC, i.id DESC;
```

#### Why `COUNT(DISTINCT f.person_id)` instead of `COUNT(f.id)`?
1. An image could contain 2 detections of the **same** person (e.g. reflection in a mirror, or multi-detector artifact). That is not a group photo.
2. Unassigned/unknown faces (`person_id IS NULL`) are filtered out by `WHERE f.person_id IS NOT NULL` so an ambiguous background passerby does not misclassify a portrait as a group shot.

---

### 3. Lightweight Heuristic Document Classification

Rather than shipping a heavy ResNet or MobileNet document classifier (which would bloat the binary by 50–100MB and consume CPU cycles), `doc_classifier.py` uses **fast statistical heuristics** on the 200px thumbnail generated during scanning:

$$\text{doc\_score} = 0.35 s_{\text{name}} + 0.20 s_{\text{res}} + 0.05 s_{\text{ratio}} + 0.20 s_{\text{entropy}} + 0.10 s_{\text{edges}} + 0.10 s_{\text{face\_penalty}}$$

```
Signal                  Weight   Description
─────────────────────────────────────────────────────────────────────────────
Filename Pattern        0.35     Matches regex: "Screenshot", "Screen Shot", "Capture", "Snip"
Screen Resolution       0.20     Exact match with monitor/phone resolutions (1920×1080, 1170×2532)
Aspect Ratio            0.05     Screen aspect ratios (16:9, 19.5:9, 20:9)
Color Entropy           0.20     Shannon entropy on Saturation histogram (flat UI = low entropy)
Edge Density            0.10     Canny edge fraction (text characters generate high edge counts)
Face Penalty            0.10     Presence of real detected human faces reduces doc likelihood
```

#### Why Color Entropy Works:
Camera photos have continuous gradients (sky, skin, fabric, nature). Screenshots and documents consist of large blocks of solid white, dark mode gray, and discrete UI buttons. Computing Shannon entropy on the HSV Saturation histogram cleanly separates them:
$$H = -\sum_{i} p_i \log_2(p_i)$$
- Low entropy ($H < 1.5$) $\rightarrow$ flat monochrome colors $\rightarrow$ document/screenshot.
- High entropy ($H > 3.0$) $\rightarrow$ rich natural color spectrum $\rightarrow$ photograph.

---

### 4. Non-Destructive Safe Export Engine

The Export feature copies files to a user-selected external directory. It follows strict production safety invariants:

1. **Read-Only Source Invariant**: Sources are opened via `shutil.copy2` (preserving file timestamps and metadata). Never `os.remove`, `shutil.move`, or write mode on source paths.
2. **Windows Path Sanitization**: Person names like `Mom ❤️` or `John: Work` cannot be valid directory names on Windows. `_safe_folder_name` strips forbidden characters (`\ / : * ? " < > |`) and trailing dots/whitespace.
3. **Deterministic Collision Resolution**: If two different source files share the same filename (e.g. `IMG_0001.JPG` from different cameras/dates), `_unique_dest_path` appends `_2`, `_3` before copying.
4. **Resumable Copying**: Before copying, the destination is checked. If `dest.exists()` and `dest.stat().st_size == src.stat().st_size`, the copy is skipped. An interrupted 20GB export resumes instantly without re-copying existing files.
5. **Group Photo Duplication**: If Photo A contains both Alice and Bob, it is copied into `Export/Alice/PhotoA.jpg` and `Export/Bob/PhotoA.jpg`. Why not symlinks? On Windows, creating symlinks requires Administrator privileges or Developer Mode enabled in Windows Settings. Physical copying ensures zero friction on consumer PCs.

---

## 📋 Files Created / Modified in M7

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/doc_classifier.py` | NEW | Fast heuristic document & screenshot classifier |
| `autosorter/core/export.py` | NEW | Safe folder export engine with resume, collision handling & group duplication |
| `autosorter/core/export_worker.py` | NEW | `QThread` background runner for export with Qt progress signals |
| `autosorter/app/albums_panel.py` | NEW | Smart album cards UI, detail grid views, and export options dialog |
| `autosorter/core/db.py` | MODIFIED | Added `get_group_photos`, `get_unknown_faces`, `get_images_for_person`, `get_text_and_docs`, and `update_doc_score` |
| `autosorter/core/pipeline_worker.py` | MODIFIED | Integrated inline document classification during Stage 1 thumbnail generation |
| `autosorter/app/main_window.py` | MODIFIED | Integrated Albums panel into the main layout between People and the photo grid |
| `tests/test_m7_albums.py` | NEW | 31 comprehensive unit tests covering queries, heuristics, and export engine |
| `PLAN.md` | MODIFIED | Marked M7 as complete in progress list and decisions log |

---

## ✅ How to Verify M7 Works

```powershell
# Run the M7 test suite (31 tests)
.\.venv\Scripts\python -m pytest tests/test_m7_albums.py -v

# Run the complete project test suite (172 tests)
.\.venv\Scripts\python -m pytest tests/ -v

# Launch the app
.\.venv\Scripts\python -m autosorter
```

### Manual Verification Steps:
1. **Virtual Albums View**:
   - Open a folder with scanned photos.
   - Notice the **📁 Albums** section displaying:
     - **👥 Group Photos** card with photo count.
     - **❓ Unknown Faces** card with unassigned face count.
     - **📄 Text & Docs** card with detected document count.
   - Click any card $\rightarrow$ an **Album Detail View** modal pops up with a clean scrollable grid of thumbnails.
2. **Library Export (All or Selected People)**:
   - Click **`📤 Export Library`** in the Albums panel.
   - Select either **"Export all people"** or **"Export selected people only"** with the interactive checklist.
   - Select optional smart albums (Group Photos, Unknown, Text & Docs) and choose an export target folder.
   - Watch the real-time progress bar.
   - Verify the destination folder contains organized subfolders per person, group photos copied to each participant, and all source photos are 100% untouched!
3. **Single-Person Export (1-Click)**:
   - Open any person card by clicking on it.
   - Click the orange **`📤 Export Person`** button in the header toolbar.
   - Pick an export folder $\rightarrow$ exports only that person's photos directly into a named folder!

