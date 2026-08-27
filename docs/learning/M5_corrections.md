# Milestone 5 — Corrections & Manual Library Organization

## 🎯 Goal
Provide fast, intuitive, and robust manual organization tools to correct any AI misidentifications, split cards, or false matches:
1. **Merge Person Cards** — Combine 2 split cards of the same person into one with 1 click.
2. **"Not this person"** — Remove an incorrectly assigned face from a person card.
3. **"Move to..."** — Reassign a face to an existing person card.
4. **"Move to NEW Person Card"** — Split a face out into a brand new person section in 2 clicks.
5. **"Set as Cover"** — Pick which face photo represents the person card avatar.
6. **Interactive Box-Click** — Click any face bounding box in the full photo viewer to view or reassign its identity.
7. **Full Photo Inspection** — Double-click any face or click `🔍 Full Photo` to inspect the high-resolution original image.

---

## 📦 Concepts Learned

### 1. The Human-in-the-Loop Philosophy

No computer vision face clustering model is 100% accurate across all real-world conditions (harsh shadows, profiles, heavy motion blur, aging, glasses). 

Instead of pretending the AI never makes mistakes, Auto-Sorter treats the AI as a **first-pass drafter** and gives the user **1-click correction superpowers**:

```
AI First-Pass (M4):           User Correction (M5):
┌─────────────────────────┐   ┌───────────────────────────────────────────┐
│ Discovers 12 people     │ → │ • 2 cards are the same person → [🔗 Merge] │
│ Auto-clusters 95%+ faces│   │ • 1 wrong face in card → [❌ Not This]    │
│                         │   │ • Face belongs to new person → [➕ Move New]│
└─────────────────────────┘   └───────────────────────────────────────────┘
```

**Safety Principle:** All manual corrections set `assign_source = 'user'` and `assign_conf = 1.0`. Automated re-scans will **never overwrite or undo user decisions**.

---

### 2. Centroid Cleansing & Recomputation Math

When a user merges two people, removes a face ("Not this person"), or shifts a face to another card, the person's identity vector (centroid) must be updated immediately so that future incremental scans and searches remain accurate.

#### The Quality-Weighted Average Formula:
$$\vec{C}_{\text{raw}} = \sum_{i=1}^{N} q_i \cdot \vec{e}_i$$

Where:
- $\vec{e}_i$ is the 128-dimensional embedding vector of face $i$.
- $q_i = \max(\text{quality}_i, 0.1)$ is the YuNet face quality score (confidence + sharpness + frontal angle).
- $N$ is the total number of remaining faces for this person.

#### Normalization (L2-Norm):
$$\vec{C} = \frac{\vec{C}_{\text{raw}}}{\|\vec{C}_{\text{raw}}\|_2}$$

```python
# Implementation in db.py:
def recompute_person_centroid(self, person_id: int) -> None:
    faces = self.get_faces_for_person(person_id)
    if not faces:
        # If no faces remain, cleanly delete the empty person card
        self._conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self._conn.commit()
        return

    embs = []
    weights = []
    for f in faces:
        if f["embedding"] is not None:
            emb = FaceRecognizer.bytes_to_embedding(f["embedding"])
            weight = max(f["quality"] or 0.5, 0.1)
            embs.append(emb * weight)
            weights.append(weight)

    if embs:
        sum_emb = np.sum(embs, axis=0)
        total_weight = sum(weights)
        centroid = sum_emb / total_weight
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
        centroid_bytes = FaceRecognizer.embedding_to_bytes(centroid)
    else:
        centroid_bytes = None

    # Cover face is always the highest-quality face remaining
    best_face_id = faces[0]["id"]
    face_count = len(faces)

    self._conn.execute(
        "UPDATE persons SET centroid = ?, face_count = ?, cover_face_id = ? WHERE id = ?",
        (centroid_bytes, face_count, best_face_id, person_id),
    )
    self._conn.commit()
```

---

### 3. 1-Click Merge Architecture

When Person A ("Arnav with glasses") and Person B ("Arnav without glasses") are merged into Person A:

```
Before Merge:
Person 1 (Arnav):   [Face 10, Face 12]       Centroid = C_1
Person 2 (Arnav 2): [Face 45, Face 46, Face 50] Centroid = C_2

Execute db.merge_persons(source_id=2, target_id=1):
1. UPDATE faces SET person_id = 1, assign_source = 'user' WHERE person_id = 2
2. DELETE FROM persons WHERE id = 2
3. Recompute Person 1 centroid across all 5 faces: [10, 12, 45, 46, 50]
4. Pick highest-quality face among all 5 as cover avatar.

After Merge:
Person 1 (Arnav):   [Face 10, Face 12, Face 45, Face 46, Face 50] (5 photos)
Person 2:           DELETED
```

#### Merge Dialog Features:
- **Live Search Filter:** Type in `🔍 Type name to filter...` to search through hundreds of people in milliseconds.
- **Visual Cover Thumbnails:** Displays each candidate's circular face avatar and photo count.
- **Double-Click Confirmation:** Double-click any person in the list to select and confirm the merge in one smooth motion.

---

### 4. "Move to NEW Person Card" — Instant Section Splitting

When an album has 1 wrong face mixed in with 40 correct faces:
1. User right-clicks the wrong face $\rightarrow$ **`➕ Move to NEW Person Card...`**
2. Prompts for a name (e.g. *"Rohan"*).
3. Auto-Sorter:
   - Creates `Person N` named *"Rohan"*.
   - Seeds the new card's centroid with that face's embedding.
   - Sets that face as Rohan's cover photo.
   - Moves the face out of the original person's album.
   - Cleanses the original person's centroid so it never drifts toward Rohan again!

---

### 5. Interactive Hit-Testing in Full Photo Viewer

When viewing a full photograph with multiple people:
1. Face bounding boxes are stored in original image coordinates: `(bbox_x, bbox_y, bbox_w, bbox_h)`.
2. When rendered on screen, the image is scaled to fit the viewer window: `scale_x = display_w / original_w`.
3. Display rectangles are computed: `rect = QRect(x * scale_x, y * scale_y, w * scale_x, h * scale_y)`.
4. **Hit-Testing on Mouse Click:**
   ```python
   # In ClickableImageLabel.mousePressEvent:
   click_pos = event.pos() - offset  # centered image offset
   for rect, face_row in self._face_rects:
       if rect.contains(click_pos):
           self._parent_viewer._on_face_clicked(face_row, event.globalPosition().toPoint())
           return
   ```
5. A popup menu appears directly at the clicked face showing:
   - Current assignment (`👤 Person: Arnav` or `Unassigned`)
   - `➡️ Assign / Move to Person...`
   - `❌ Remove from Person (Unassign)`
   - `⭐ Set as Cover Face`
   - `🔍 Find Similar Faces`

---

### 6. Qt & PySide6 Technical Lessons

#### 🛡️ Issue 1: Event Consumption by Child QLabels
* **Problem:** In `FaceCellWidget`, clicking or double-clicking the face photo failed because the child `QLabel` displaying the image consumed all mouse events, preventing the parent widget's `mouseDoubleClickEvent` from firing.
* **Solution:** Set `setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)` on all decorative child labels (`_thumb_label`, `_name_label`, `_count_label`). Now mouse clicks pass straight to the parent cell container!

#### 🛡️ Issue 2: Shiboken C++ Type Conversion on Custom Python Types
* **Problem:** Defining a PySide6 signal as `open_photo_requested = Signal(dict)` and emitting an internal `sqlite3.Row` object triggered a fatal Shiboken C++ conversion exception:
  `Shiboken::Conversions::_pythonToCppCopy: Cannot copy-convert sqlite3.Row to C++`
* **Solution:** Signals should always pass primitive types (`Signal(int)` passing `face_id` or `person_id`). The receiver queries the database cleanly by ID.

#### 🛡️ Issue 3: Responsive Dynamic Layout Reflow
* **Problem:** Fixed grid columns clipped face cards on smaller screens and left vast empty margins on 4K monitors.
* **Solution:** Implement `resizeEvent(self, event)` to dynamically recalculate column counts on the fly:
  `cols = max(1, (self.width() - 60) // (DETAIL_FACE_SIZE + 24))`
  The grid automatically reflows smoothly as the window is resized or maximized.

---

### 8. Semi-Supervised Active Learning — "🔍 Re-check Album"

Moving 10 misfiled photos one-by-one is tedious. Auto-Sorter leverages **Semi-Supervised Active Learning** through the **"🔍 Re-check Album"** engine (`clean_up_person_album` in `cluster.py`):

1. When a user manually moves **one photo** of a person to their correct card (e.g. Person 7), that photo becomes a confirmed anchor.
2. Person 7's centroid shifts to include that new angle/pose.
3. Clicking **`🔍 Re-check Album`** sweeps all remaining auto-assigned photos in the current album.
4. For every face:
   - Evaluates similarity against all other person centroids.
   - Enforces the **Co-Occurrence Firewall** (cannot move if the target person already appears in that image).
   - If a face is significantly closer to another person ($\text{sim} \ge 0.46$ and margin $\ge 0.03$), it is **automatically shifted** to that person's card!
5. All affected centroids and counts are updated atomically in SQLite.

---

### 9. Complete Person Deletion

When an unwanted person card is discovered (e.g. background strangers or erroneous clusters), the user can delete it in 1 click:
- Available via the red **`🗑️ Delete Person`** button in the album header and the **`🗑️`** icon on each person card.
- Calls `db.delete_person(person_id)`:
  ```sql
  UPDATE faces SET person_id = NULL, assign_source = 'auto', assign_conf = 0.0 WHERE person_id = ?;
  DELETE FROM persons WHERE id = ?;
  ```
- Unassigns all member faces cleanly so they return to the unassigned pool, deletes the card, and reloads the People grid.

---

## 📋 Files Created / Modified in M5

| File | Status | Purpose |
|---|---|---|
| `autosorter/core/db.py` | MODIFIED | Added `merge_persons`, `delete_person`, `unassign_face`, `reassign_face`, `create_person_for_face`, and `recompute_person_centroid` |
| `autosorter/core/cluster.py` | MODIFIED | Added `clean_up_person_album` for active semi-supervised album re-checking |
| `autosorter/app/people_panel.py` | MODIFIED | Added `MergePersonDialog`, `MoveFaceDialog`, `FaceCellWidget`, `Re-check Album` button, and card/header `Delete Person` actions |
| `autosorter/app/photo_viewer.py` | MODIFIED | Added interactive face box click popover with assignment actions + reload signals |
| `autosorter/app/main_window.py` | MODIFIED | Connected `viewer.changed` signals to automatically refresh People Panel on any correction |
| `tests/test_m5_corrections.py` | MODIFIED | 10 comprehensive unit tests including album cleanup and person deletion |
| `docs/learning/M5_corrections.md` | MODIFIED | In-depth learning documentation for active learning and card deletion |


---

## ✅ How to Verify M5 Works

```powershell
# Run the full test suite (114 passing tests)
.\.venv\Scripts\python -m pytest tests/ -v

# Launch the app
.\.venv\Scripts\python -m autosorter
```

### Verification Flow:
1. **Merge Test:**
   - Click the **🔗** icon on any person card (or click **"🔗 Merge with..."** inside their detail view).
   - Type in the filter bar or double-click the target person $\rightarrow$ cards combine instantly, count updates, and duplicate card disappears.
2. **"Not This Person" Test:**
   - Open a person album $\rightarrow$ right-click a face $\rightarrow$ **"❌ Not this person"** $\rightarrow$ face is removed and centroid is cleansed.
3. **"Move to NEW Person Card" Test:**
   - Right-click a face $\rightarrow$ **"➕ Move to NEW Person Card..."** $\rightarrow$ type a name $\rightarrow$ a brand new person section is created for that face.
4. **Full Photo Inspection:**
   - Double-click any face (or click the **`🔍 Full Photo`** button) $\rightarrow$ opens the full-resolution photo in the Photo Viewer.
5. **Interactive Box-Click:**
   - In the Photo Viewer, click any colored face bounding box $\rightarrow$ shows assigned person name and lets you reassign or unassign on the spot!
6. **Persistence Test:**
   - Close the app and relaunch $\rightarrow$ all merges, new person sections, and reassignments are preserved exactly.
