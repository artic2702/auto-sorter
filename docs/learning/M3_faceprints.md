# Milestone 3 — Faceprints (Embeddings) + "Find Similar" Panel

## 🎯 Goal
After detecting faces (M2), compute a **128-number faceprint** for each face using SFace, store it in the database as a BLOB, and let users click a face to see the most similar faces ranked by cosine similarity.

---

## 📦 Concepts Learned

### 1. Face Embeddings — The Core Idea

An **embedding** (also called a "faceprint") is a vector of 128 floating-point numbers that describes a face's identity:

```
Face photo → SFace neural network → [0.023, -0.145, 0.089, ..., 0.034]
                                      └─── 128 numbers ───┘
```

**The key property:**
- SAME person, different photos → similar numbers
- DIFFERENT people → different numbers

```
Your face, photo 1:    [0.82, -0.15, 0.43, ...]
Your face, photo 2:    [0.80, -0.17, 0.45, ...]  ← very close!
Someone else's face:   [-0.31, 0.67, -0.22, ...]  ← completely different!
```

SFace was trained on millions of labeled face images. It learned to:
- **IGNORE**: lighting, expression, glasses, mild aging
- **PRESERVE**: face shape, eye spacing, nose structure

We NEVER retrain SFace. It's a frozen measurement tool — like a ruler.

---

### 2. L2 Normalization — Making Math Simple

**L2 normalization** scales a vector so its length (magnitude) = 1.0:

```python
normalized = vector / sqrt(sum(vector²))
```

**Why?** After normalization, cosine similarity becomes a simple dot product:

```
Before normalization:
  similarity = dot(A, B) / (|A| × |B|)    ← 3 operations

After normalization (|A| = |B| = 1.0):
  similarity = dot(A, B)                   ← 1 operation!
```

For 100,000 faces, this saves millions of operations per search.

```python
# In our code:
embedding = model.feature(face_image).flatten()
norm = np.linalg.norm(embedding)  # sqrt(sum(x²))
embedding = embedding / norm       # now length = 1.0
```

---

### 3. Cosine Similarity — Comparing Faceprints

**Cosine similarity** measures how similar two vectors are in *direction* (not magnitude):

```
    A •
     /|
    / | similarity = cos(θ)
   /  |
  /θ__|________
  O         B •

θ = 0°  → cos(0°) = 1.0  → identical direction (same person)
θ = 90° → cos(90°) = 0.0 → perpendicular (different people)
```

**For our app:**
```
> 0.50: very likely same person (auto-assign in M4)
0.30-0.50: possibly same person (needs review in M5)
< 0.30: different people
```

The math:
```python
# Since both embeddings are L2-normalized:
similarity = np.dot(embedding_A, embedding_B)
# That's it! One line. Result between -1.0 and 1.0.
```

---

### 4. Matrix Multiplication — Searching All Faces at Once

**The problem:** Compare one face against 100,000 others.

**Naive approach (Python loop):**
```python
for face in all_faces:
    sim = np.dot(query, face)  # 100,000 iterations → slow
```

**Matrix approach (one operation):**
```python
# Stack all embeddings into a matrix: shape (100000, 128)
all_embeddings = np.stack(embeddings_list)

# Compute ALL similarities at once:
similarities = all_embeddings @ query_embedding  # shape (100000,)

# Sort and take top 10
top_indices = np.argsort(similarities)[::-1][:10]
```

The `@` operator is matrix multiplication. NumPy runs it in optimized C code using BLAS libraries — it's **hundreds of times faster** than a Python loop.

```
Matrix multiplication visualization:

[face_1: 0.2, -0.1, ...]         [query:]     [sim_1: 0.82]
[face_2: -0.3, 0.5, ...]    @    [ 0.15 ]  =  [sim_2: 0.15]
[face_3: 0.1, 0.4, ...]          [-0.22 ]     [sim_3: 0.91]
[...100k rows...]                 [ ...  ]     [...100k results]

One operation, all similarities computed!
```

---

### 5. BLOB Storage — Binary Data in SQLite

A **BLOB** (Binary Large Object) is raw bytes stored in a database column.

```
JSON storage:  "[0.023, -0.145, 0.089, ...]"  = ~1500 bytes (text)
BLOB storage:  raw 128×float32 bytes            = 512 bytes (binary)
```

**3× smaller**, and instant to load (no parsing needed).

```python
# Storing:
embedding = np.array([0.023, -0.145, ...], dtype=np.float32)
blob = embedding.tobytes()  # 128 × 4 bytes = 512 bytes
db.execute("UPDATE faces SET embedding = ? WHERE id = ?", (blob, face_id))

# Loading:
row = db.execute("SELECT embedding FROM faces WHERE id = ?", (face_id,))
embedding = np.frombuffer(row["embedding"], dtype=np.float32)
```

**For 100,000 faces:**
- JSON: 150 MB in the database
- BLOB: 50 MB in the database ← our choice

---

### 6. The Detection + Embedding Pipeline

M3 merges detection and embedding into a single pass:

```
For each image:
  ┌──────────────────────────────────────────────────────┐
  │ 1. Load image from disk (np.fromfile + cv2.imdecode) │
  │                                                      │
  │ 2. YuNet: detect faces → bounding boxes + landmarks  │
  │                                                      │
  │ 3. For each face:                                    │
  │    a. align_face() → 112×112 crop                    │
  │    b. save_face_thumb() → JPEG on disk               │
  │    c. SFace: compute embedding → 128 floats          │
  │    d. L2 normalize → unit vector                     │
  │    e. Store in DB (bbox + landmarks + embedding BLOB) │
  └──────────────────────────────────────────────────────┘
```

**Why one pass?**
- Image is already in memory — no need to reload for embedding
- One button click, one progress bar — simpler UX
- Each face is fully processed atomically

---

### 7. Hit Testing — Making Drawn Rectangles Clickable

QPainter draws on a flat pixmap — there's no concept of "clicking a drawn shape." We need **hit testing**:

```python
# Store face rectangles when drawing them:
face_rects = [(QRect(x, y, w, h), face_data), ...]

# When user clicks, check each rectangle:
def mousePressEvent(self, event):
    click = event.pos()
    for rect, face in face_rects:
        if rect.contains(click.x(), click.y()):
            open_similar_panel(face)  # HIT!
            return
```

This is exactly how game engines handle click detection on sprites — store bounding boxes, loop through them on click.

---

### 8. Upgrade Path — Handling M2 → M3 Databases

Users who ran M2 have faces in the DB without embeddings. The detect worker handles this:

```python
# After new detections, also embed any old faces:
remaining = db.count_faces_needing_embedding()
if remaining > 0:
    for face_row in db.iter_faces_needing_embedding():
        aligned = load_image(face_row["thumb_path"])
        embedding = recognizer.compute(aligned)
        db.update_embedding(face_row["id"], embedding.tobytes())
```

This means: run "Detect Faces" once on an M2 database → all existing faces get embeddings automatically.

---

## 📋 Files Created / Modified in M3

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/face_embed.py` | NEW | SFace embedding engine (compute + compare + BLOB serialization) |
| `autosorter/core/detect_worker.py` | MODIFIED | Inline embedding during detection + M2→M3 upgrade path |
| `autosorter/core/db.py` | MODIFIED | Embedding CRUD methods (update, iter_needing, get_all, get_by_id) |
| `autosorter/app/similar_panel.py` | NEW | "Find Similar" panel with ranked face grid |
| `autosorter/app/photo_viewer.py` | MODIFIED | Clickable face boxes with hit testing |
| `autosorter/app/main_window.py` | MODIFIED | Updated signals for embedding counts |
| `tests/test_m3_embed.py` | NEW | 16 tests for SFace, serialization, and DB embeddings |

---

## ✅ How to Verify M3 Works

```powershell
# All 61 tests pass
.\.venv\Scripts\python -m pytest tests/ -v

# Run the app
.\.venv\Scripts\python -m autosorter

# Test flow:
# 1. Browse → pick photo folder → Start Scan
# 2. Click "Detect Faces" → progress shows "X faces, Y embeddings"
# 3. Click any photo thumbnail → see face boxes with "🔍 click" hints
# 4. Click a face box → "Find Similar" panel opens!
# 5. Same person's faces should have GREEN borders (score > 0.5)
# 6. Different people should have RED borders (score < 0.3)
```
