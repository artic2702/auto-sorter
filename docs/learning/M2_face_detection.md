# Milestone 2 — Face Detection + Face Thumbs + Boxes in Viewer

## 🎯 Goal
After scanning photos, the app runs **YuNet face detection** on every image, finds all faces, saves aligned face crops, and lets you click any photo to see **green rectangles** drawn around each detected face.

---

## 📦 Concepts Learned

### 1. Face Detection — How AI "Sees" Faces

**The problem**: Given a photo of 4000×3000 pixels (12 million pixels), find every face.

**How YuNet works** (simplified):
1. Scan the image at multiple scales (zoom levels)
2. At each scale, slide a small window across the image
3. For each window position, ask: "Does this look like a face?"
4. If yes → output the location + confidence score

```
Original photo (4000×3000):
┌──────────────────────────────────────┐
│                                      │
│    ┌──┐         ┌──┐                 │
│    │😊│         │😊│                 │  ← Two faces detected!
│    └──┘         └──┘                 │
│                                      │
│              ┌──────┐                │
│              │ 🏔️   │                │  ← Not a face (ignored)
│              └──────┘                │
└──────────────────────────────────────┘
```

**For each face, YuNet outputs 15 numbers**:
```
[x, y, w, h,                          ← bounding box (rectangle)
 left_eye_x, left_eye_y,              ← 5 landmarks
 right_eye_x, right_eye_y,
 nose_x, nose_y,
 left_mouth_x, left_mouth_y,
 right_mouth_x, right_mouth_y,
 confidence]                           ← how sure (0.0 to 1.0)
```

Visual of a detected face:
```
  ┌─────────────────┐  ← bounding box (x, y, w, h)
  │   •         •   │  ← left eye, right eye (landmarks)
  │                 │
  │        •        │  ← nose tip
  │                 │
  │     •     •     │  ← left mouth, right mouth
  └─────────────────┘
```

---

### 2. Confidence Threshold — Filtering False Positives

YuNet gives every detection a **confidence score** (0.0 to 1.0):
- **0.95+**: "I'm very sure this is a face" (frontal, well-lit)
- **0.7-0.95**: "Probably a face" (angled, partially occluded)
- **0.3-0.7**: "Maybe a face?" (blurry, tiny, or not a face at all)
- **Below 0.3**: "Random noise"

We set a threshold of **0.7** — anything below is discarded.

```python
# In FaceDetector.__init__:
confidence_threshold: float = 0.7   # Minimum to keep
```

**The tradeoff**:
```
Higher threshold (0.9) → fewer detections, almost zero false positives
                        but misses angled/blurry faces

Lower threshold (0.3)  → catches more faces, but also detects cups,
                        logos, tree knots as "faces"

Our choice (0.7)       → good balance. Review queue (M5) handles edge cases.
```

---

### 3. Minimum Face Size — Another Filter

Even with 0.7 confidence, the model sometimes detects tiny objects (cup handles, decorative patterns) as "faces" — especially in game screenshots.

**Fix**: Any detection smaller than 60×60 pixels is discarded.

```python
# Real faces in photos are typically 60px+ wide
min_face_size: int = 60

# In the detect loop:
if face_w < self._min_face_size or face_h < self._min_face_size:
    continue  # Too small to be a real face
```

**Why 60px?** A face needs at least ~60 pixels across to have enough detail for two eyes, a nose, and a mouth to be distinguishable. Below that, even humans struggle to tell if something is a face.

---

### 4. Face Alignment — Why Rotation Matters

**The problem**: SFace (recognition model, M3) expects faces to be standardized — eyes level, face centered. But people tilt their heads!

**The solution**: Use the eye positions from landmarks to compute a rotation.

```
Before alignment:          After alignment:
  ┌──────────┐              ┌──────────┐
  │     •    │              │  •    •  │  ← eyes now level
  │  •       │              │          │
  │     •    │   ──────►    │     •    │
  │   •   •  │              │   •   •  │
  └──────────┘              └──────────┘
  (head tilted 20°)         (standardized 112×112)
```

**The math** (simplified):
```python
# 1. Get eye positions
left_eye  = landmarks[0]   # (x1, y1)
right_eye = landmarks[1]   # (x2, y2)

# 2. Calculate the tilt angle
angle = atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
# If eyes are level: angle = 0°
# If head tilted right: angle > 0°

# 3. Calculate scale (zoom to fit 112×112)
eye_distance = sqrt((x2-x1)² + (y2-y1)²)
scale = desired_distance / eye_distance

# 4. Build rotation matrix and apply
M = cv2.getRotationMatrix2D(center, angle, scale)
aligned = cv2.warpAffine(image, M, (112, 112))
```

---

### 5. Affine Transforms — The Math Behind Alignment

An **affine transform** is a 2×3 matrix that combines rotation, scaling, and translation into one operation:

```
┌         ┐   ┌               ┐   ┌     ┐
│ new_x   │ = │ a  b  tx      │ × │ x   │
│ new_y   │   │ c  d  ty      │   │ y   │
└         ┘   └               ┘   │ 1   │
                                   └     ┘

Where:
  a, d = scale (zoom)
  b, c = rotation
  tx, ty = translation (shift)
```

`cv2.warpAffine(image, M, output_size)` applies this matrix to every pixel in the image, producing a new image where:
- The face is centered
- The eyes are horizontal
- The size is exactly 112×112

**Why 112×112?** That's the input size SFace was trained on. Using a different size would reduce recognition accuracy.

---

### 6. Dataclasses — Clean Data Containers

```python
from dataclasses import dataclass

@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    landmarks: List[Tuple[float, float]]
    confidence: float
```

**What `@dataclass` does**: Python auto-generates `__init__`, `__repr__`, and `__eq__` for you.

Without dataclass (boilerplate):
```python
class DetectedFace:
    def __init__(self, bbox, landmarks, confidence):
        self.bbox = bbox
        self.landmarks = landmarks
        self.confidence = confidence
    
    def __repr__(self):
        return f"DetectedFace(bbox={self.bbox}, ...)"
    
    def __eq__(self, other):
        return (self.bbox == other.bbox and ...)
```

With dataclass (clean):
```python
@dataclass
class DetectedFace:
    bbox: Tuple[int, int, int, int]
    landmarks: List[Tuple[float, float]]
    confidence: float
```

Same result, 1/3 the code. Use `@dataclass` whenever you need a simple container class.

---

### 7. Database Foreign Keys — Linking Tables

The `faces` table links to the `images` table:

```sql
CREATE TABLE faces (
    id       INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    ...
);
```

**`REFERENCES images(id)`**: This is a **foreign key** — it means `face.image_id` MUST point to a valid `images.id`. You can't insert a face with `image_id=999` if there's no image with `id=999`.

**`ON DELETE CASCADE`**: If you delete an image, ALL its faces are automatically deleted too. This prevents "orphan" faces pointing to non-existent images.

```
images table:               faces table:
┌────┬────────────┐         ┌────┬──────────┬────────┐
│ id │ path       │         │ id │ image_id │ bbox   │
├────┼────────────┤    ┌──► ├────┼──────────┼────────┤
│ 1  │ beach.jpg  │◄───┤    │ 1  │ 1        │ (x,y)  │
│    │            │◄───┘    │ 2  │ 1        │ (x2,y2)│
│ 2  │ selfie.jpg │◄──────►│ 3  │ 2        │ (x,y)  │
└────┴────────────┘         └────┴──────────┴────────┘

Delete image 1 → faces 1 and 2 are auto-deleted (CASCADE)
```

---

### 8. LEFT JOIN — Finding Images Without Faces

```sql
SELECT i.*
FROM images i
LEFT JOIN faces f ON f.image_id = i.id
WHERE i.status = 'processed' AND f.id IS NULL
```

**LEFT JOIN** explained with a restaurant analogy:
- **images** table = list of customers
- **faces** table = list of orders
- **LEFT JOIN** = "show me ALL customers, and attach their orders if they have any"
- **WHERE f.id IS NULL** = "only show customers with NO orders"

Result: all processed images that have zero faces detected → need detection.

---

### 9. Database INDEX — Making Queries Fast

```sql
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
```

**Without index**: To find faces for image #42, SQLite scans EVERY row in the faces table.
**With index**: SQLite jumps directly to the right rows (like a book's index).

```
Without index (10,000 faces):
  Scan row 1... no. Row 2... no. Row 3... no. ... Row 9,999... no. Row 10,000... yes!
  → O(N) = slow

With index:
  Look up "image_id=42" in index → jump to row 7,843 → done!
  → O(log N) = fast
```

---

### 10. QPainter — Drawing on Images in Qt

```python
# Start painting on the pixmap
painter = QPainter(pixmap)

# Set pen color and width
pen = QPen(QColor(0, 255, 100))  # green
pen.setWidth(3)                   # 3px thick lines
painter.setPen(pen)

# Draw a rectangle (face bounding box)
painter.drawRect(x, y, w, h)

# Draw a small circle (landmark dot)
painter.drawEllipse(px - 3, py - 3, 6, 6)

# Done painting
painter.end()
```

**Important**: You must call `painter.end()` before displaying the pixmap. Think of it like closing a paint can — you can't show the painting while you're still painting on it.

---

### 11. Sentinel Pattern — Marking "No Faces" Images

**The problem**: If an image has no faces, how do we avoid re-detecting it every time?

**Solution**: Insert a **sentinel row** — a fake face entry with `det_score=-1`:

```python
def mark_no_faces(self, image_id: int) -> None:
    self._conn.execute(
        "INSERT INTO faces (..., det_score) VALUES (..., -1.0)",
        (image_id,),
    )
```

On re-run, `iter_images_needing_detection()` uses `LEFT JOIN ... WHERE f.id IS NULL`. Since the sentinel row exists, the image has `f.id IS NOT NULL` → skipped!

Real faces have `det_score > 0`, so the sentinel (-1) is easily filtered out in display code.

---

## 🐛 Bugs Found and Fixed

### Bug 1: False Positives on Game Screenshots
**Problem**: YuNet detected cups and patterns as "faces" in Roblox screenshots.
**Fix**: Added `min_face_size=60` filter — real faces are 60px+, cups are smaller.

### Bug 2: Too Many False Positives with Low Threshold
**Problem**: Lowering confidence threshold to 0.5 created more false detections.
**Fix**: Raised back to 0.7. Better to miss edge cases than pollute data with false faces. The review system (M5) handles missed faces.

---

## 📋 Files Created / Modified in M2

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/db.py` | MODIFIED | Added `faces` table, face CRUD methods, LEFT JOIN queries |
| `autosorter/core/face_detect.py` | NEW | YuNet face detection engine + min size filter |
| `autosorter/core/face_crop.py` | NEW | Face alignment (affine transform) + cropping |
| `autosorter/core/detect_worker.py` | NEW | Background QThread for batch face detection |
| `autosorter/app/main_window.py` | MODIFIED | Detect button, thumbnail grid, click-to-view |
| `autosorter/app/photo_viewer.py` | NEW | Image viewer with face boxes + landmarks |
| `tests/test_m2_faces.py` | NEW | 22 tests for DB faces, detector, cropping |
| `tests/test_m1_scan.py` | MODIFIED | Updated schema version assertion (1→2) |

---

## ✅ How to Verify M2 Works

```powershell
# All 45 tests pass
.\.venv\Scripts\python -m pytest tests/ -v

# Run the app
.\.venv\Scripts\python -m autosorter

# Expected flow:
# 1. Browse → select photo folder → Start Scan
# 2. After scan: thumbnail grid appears + "Detect Faces" button
# 3. Click "Detect Faces" → progress shows faces being detected
# 4. Click any thumbnail → PhotoViewer opens with green face boxes
# 5. Re-detect → already-detected images are skipped (instant)
```

---

## 🧪 How Data Flows Through M2

```
User clicks "Detect Faces"
        │
        ▼
    DetectWorker.start()                    [background thread]
        │
        ▼
    db.iter_images_needing_detection()      [find unprocessed images]
        │
        ├── image "beach.jpg" → no face rows → needs detection
        │
        ▼
    Load image (np.fromfile + cv2.imdecode)
        │
        ▼
    detector.detect(image)                  [YuNet inference]
        │
        ├── Face found! bbox=(340,120,180,200), confidence=0.92
        │   │
        │   ├── align_face(image, landmarks) → 112×112 crop
        │   ├── db.insert_face(image_id, bbox, det_score, ...)
        │   └── save_face_thumb(aligned, face_thumbs_dir, face_id)
        │
        ├── Face found! bbox=(50,80,120,150), confidence=0.85
        │   └── (same process)
        │
        └── emit progress(done=1, total=58, faces=2)
                │
                ▼
            UI updates: "Detecting faces: 1/58 (2 faces found)"

User clicks a thumbnail
        │
        ▼
    db.get_faces_for_image(image_id)        [fetch face data]
        │
        ▼
    PhotoViewer(image_path, faces)          [open viewer]
        │
        ├── Load image as QPixmap
        ├── Scale to fit window
        ├── QPainter: draw green rectangles at face bboxes
        ├── QPainter: draw green dots at landmarks
        └── Display in QLabel
```
