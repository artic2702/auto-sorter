# Milestone 4 — People Discovery + Rename

## 🎯 Goal
After face detection + embedding (M3), automatically **cluster faces into person groups**, display them as person cards, and let the user rename each person. This is the **first full vertical slice** — the app goes from "a bag of faces" to "here are your people."

---

## 📦 Concepts Learned

### 1. Clustering — Turning Faces into People

**The problem:** We have 200 face embeddings (128 numbers each). We know faces of the same person have similar embeddings. But we don't know *how many* people there are, or *which* faces belong together.

**Clustering** = automatically grouping similar items. There are many algorithms:

```
K-Means:   "Split into K groups"     → Problem: we don't know K
DBSCAN:    "Find dense neighborhoods" → Problem: batch-only, hard to resume
Agglom.:   "Merge closest pairs"      → Problem: O(n²) memory

Incremental Centroid:  "One at a time, compare to known people"
                       → ✅ Our choice!
```

---

### 2. Incremental Centroid Clustering & Drift Protection

Process one face at a time, in order of quality (best faces first):

```
Face 1 (best quality): No persons exist → Create "Person 1"
                        Person 1's centroid = this face's embedding

Face 2: Compare to Person 1
         Similarity = 0.72 → HIGH (>= 0.46) and min member check (>= 0.40)
         → Assign to Person 1
         Update Person 1's centroid (now average of 2 faces)

Face 3: Compare to Person 1
         Similarity = 0.05 → LOW → Create "Person 2"
         Person 2's centroid = this face's embedding

Face 4: Compare to Person 1: 0.68
         Compare to Person 2: 0.71
         Best = Person 2, margin = 0.71 - 0.68 = 0.03 <= 0.03 → AMBIGUOUS
         → Create "Person 3" (don't guess wrong!)

...repeat for all faces
```

#### 🛡️ The Centroid Drift ("Chaining") Problem & Solution
* **The Problem**: In simple running-average clustering, adding a slightly ambiguous face pulls the centroid towards it. Over 40 photos, the centroid gradually migrates across embedding space and swallows different people who look somewhat alike (e.g. same age/hair/glasses).
* **The Solution**: We enforce **Member Validation**:
  - `avg_similarity >= T_HIGH (0.46)`
  - `min_member_similarity >= T_MIN_MEMBER (0.40)`: A new face cannot contradict ANY existing face already in that person's card.
  - This mathematically prevents cluster drift across distinct people.

---

### 3. SFace Similarity Dynamics & Calibrated Thresholds

In SFace 128-dimensional embedding space:
- **Same person** across varied lighting, angles, expressions: **`0.50 – 0.75`**
- **Different people** who look similar (same age, hair, skin tone, glasses): **`0.48 – 0.54`**
- **Completely different people**: **`< 0.35`**

```python
T_HIGH = 0.46        # High confidence match threshold
T_MIN_MEMBER = 0.40  # Minimum similarity across all cluster members
MARGIN_MIN = 0.03    # Separation over second-best match
```

A **centroid** is the average embedding of all faces belonging to a person:

```
Person "Arnav" has 3 faces:
  Face 1: [0.82, -0.15, 0.43, ...]
  Face 2: [0.80, -0.17, 0.45, ...]
  Face 3: [0.79, -0.14, 0.44, ...]

Centroid = average = [0.803, -0.153, 0.440, ...]
```

**Why use the centroid instead of comparing to every face?**
- 50 faces per person × 100 people = 5,000 comparisons per new face
- 100 centroids = 100 comparisons per new face ← **50× faster**

The centroid is a better "representative" anyway — it averages out noise from individual photos (bad lighting, weird angles).

---

### 4. Quality-Weighted Centroid Updates

Not all faces are equal. A clear, frontal, well-lit face should influence the centroid more than a blurry profile shot:

```python
# Simple average (BAD — blurry face has equal influence):
new_centroid = (old_centroid * n + new_embedding) / (n + 1)

# Quality-weighted average (GOOD — clear faces dominate):
weight = face_quality  # 0.0 to 1.0
new_centroid = (old_centroid * n + new_embedding * weight) / (n + weight)
```

**Example:**
```
Person has centroid C from 10 good faces (weight = 10)
New blurry face arrives (quality = 0.3)

weighted = (C * 10 + blurry * 0.3) / (10.3)
                                      ↑
                     Blurry face barely moves the centroid!
```

After updating, we **re-normalize** to unit length (critical for cosine similarity).

---

### 5. Processing Order Matters — Best Faces First

We sort unassigned faces by quality DESCENDING before clustering:

```
Quality 0.95 → Processed 1st → Seeds a new cluster with a STRONG centroid
Quality 0.85 → Processed 2nd → Matches reliably against strong centroids
Quality 0.40 → Processed last → By now, centroids are well-established
```

**Why this matters:**
```
BAD order (worst first):
  Blurry face → Seeds Person 1 with weak centroid
  Same person, clear face → Doesn't match weak centroid → Creates Person 2!
  → Same person split into two clusters 💥

GOOD order (best first):
  Clear face → Seeds Person 1 with strong centroid
  Same person, blurry face → Matches strong centroid → Correctly assigned ✅
```

```sql
-- In db.py:
SELECT * FROM faces
WHERE embedding IS NOT NULL AND person_id IS NULL AND det_score > 0
ORDER BY quality DESC   ← this line is critical!
```

---

### 6. Thresholds and the Conservative Philosophy

Three thresholds control clustering behavior:

```
T_HIGH = 0.38     →  "I'm confident this is the same person"
T_REVIEW = 0.28   →  "Maybe the same person, but not sure"
MARGIN_MIN = 0.04  →  "Best match must beat second-best by this much"
```

**The margin check prevents ambiguous assignments:**
```
Face vs Arnav: 0.42
Face vs Dad:   0.40
Margin: 0.02 < 0.04 → TOO CLOSE! Create new person.

Face vs Arnav: 0.42
Face vs Mom:   0.15
Margin: 0.27 > 0.04 → CLEAR WINNER! Assign to Arnav.
```

**Conservative philosophy: prefer splitting over merging.**
- Splitting (same person = 2 cards) → easy fix: 1-click Merge (M5)
- Wrong merge (2 different people in 1 card) → hard to fix, confusing

**Real-world tuning:** We started at T_HIGH=0.45, which caused massive over-splitting (20 persons when there were 3 real people). Lowered to 0.38 after testing on real photos — now produces the right clusters while keeping zero wrong merges.

---

### 7. Fast Similarity — Matrix Multiply Against All Persons

Instead of a Python loop, we compute similarity to ALL persons at once:

```python
# Slow: Python loop
for person in persons:
    sim = np.dot(face_embedding, person.centroid)

# Fast: One matrix multiply
# centroids shape: (N_persons, 128)
# face_embedding shape: (128,)
similarities = centroids @ face_embedding  # shape: (N_persons,)
# All N similarities computed in one optimized BLAS call!
```

For 100 persons, this computes all 100 similarities in **one CPU instruction batch**. Even 10,000 persons would be near-instant.

---

### 8. The Rename Pattern — Persistent User Labels

Rename uses two database columns:

```sql
display_name     TEXT     -- "Person 1" → user changes to "Arnav"
name_is_user_set INTEGER  -- 0 = auto-generated, 1 = user named
```

**Why track `name_is_user_set`?**
- User-named persons are never auto-merged (M5 safety)
- Auto-named persons can be auto-renamed if we add heuristics later
- When showing "Rename" hints, skip already-named persons

```python
# In db.py:
def rename_person(self, person_id, new_name):
    self._conn.execute(
        "UPDATE persons SET display_name = ?, name_is_user_set = 1 WHERE id = ?",
        (new_name, person_id)
    )
    self._conn.commit()  # Persisted immediately — survives crash/restart
```

---

### 9. Inline Editing in Qt — QLabel ↔ QLineEdit Swap

To make names editable without a dialog popup:

```python
# Two widgets stacked in the same spot:
self._name_label = QLabel("Person 1")      # Visible by default
self._name_edit = QLineEdit("Person 1")    # Hidden by default

def _start_rename(self):
    self._name_label.setVisible(False)      # Hide the label
    self._name_edit.setVisible(True)        # Show the editor
    self._name_edit.setFocus()              # Cursor goes there
    self._name_edit.selectAll()             # Text is pre-selected

def _finish_rename(self):
    new_name = self._name_edit.text().strip()
    self._name_label.setText(new_name)
    self._name_edit.setVisible(False)       # Hide editor
    self._name_label.setVisible(True)       # Show label again
    self.renamed.emit(self._person_id, new_name)  # Tell parent to save
```

**Triggers:** Double-click name OR click ✏️ button → `_start_rename()`
**Commit:** Press Enter OR click away → `_finish_rename()`

---

### 10. Denormalization — Caching Counts in the Database

```sql
-- NORMALIZED approach (always correct, but slow):
SELECT person_id, COUNT(*) FROM faces GROUP BY person_id
-- Scans the entire faces table every time the UI renders!

-- DENORMALIZED approach (cached, fast):
-- persons table has:  face_count INTEGER DEFAULT 0
-- Updated whenever a face is assigned/removed
SELECT face_count FROM persons WHERE id = ?
-- Instant lookup, no scan needed
```

**The tradeoff:**
- Normalized = always correct, but slow (recompute every render)
- Denormalized = fast, but could get out of sync

We update `face_count` atomically when assigning faces, so it stays in sync. If it ever drifts, a simple `UPDATE persons SET face_count = (SELECT COUNT(*) FROM faces WHERE person_id = persons.id)` fixes it.

---

## 📋 Files Created / Modified in M4

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/cluster.py` | NEW | Incremental centroid clustering engine |
| `autosorter/core/identity_worker.py` | NEW | Background QThread for clustering |
| `autosorter/app/people_panel.py` | NEW | Person cards UI + inline rename + detail view |
| `autosorter/core/db.py` | MODIFIED | `persons` table + 12 new person CRUD methods, schema v3 |
| `autosorter/app/main_window.py` | MODIFIED | "Identify People" button + People panel + identity signals |
| `tests/test_m4_cluster.py` | NEW | 18 tests for DB persons, clustering, cover faces |
| `tests/test_m1_scan.py` | MODIFIED | Schema version assertion updated (2 → 3) |

---

## ✅ How to Verify M4 Works

```powershell
# All 79 tests pass
.\.venv\Scripts\python -m pytest tests/ -v

# Run the app
.\.venv\Scripts\python -m autosorter

# Test flow:
# 1. Browse → pick photo folder → Start Scan
# 2. Click "Detect Faces" → wait for completion
# 3. Click "Identify People" → progress shows "X/Y faces (N people found)"
# 4. People panel appears with person cards!
# 5. Each card shows: cover face thumbnail + name + photo count
# 6. Double-click a name → type new name → press Enter → renamed!
# 7. Close app → reopen → rename is still there (survives restart)
# 8. Click a person card → see all their face thumbnails
```
