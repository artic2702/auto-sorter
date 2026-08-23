# Milestone 5 — Corrections & Manual Library Organization

## 🎯 Goal
Provide fast, intuitive manual tools to fix any AI misidentifications or split person cards:
1. **Merge Person Cards** — Combine 2 split cards of the same person into one with 1 click.
2. **"Not this person"** — Remove an incorrectly assigned face from a person card.
3. **"Move to..."** — Reassign a face to another person or create a new person card.
4. **"Set as Cover"** — Pick which face photo represents the person card.
5. **Interactive Box-Click** — Click any face bounding box in the full photo viewer to view or change its person assignment.
6. **Full Photo Inspection** — Double-click any face in an album or click `🔍 Full Photo` to inspect the high-resolution original image.

---

## 📦 Concepts Learned

### 1. The Human-in-the-Loop Philosophy
No face clustering model is 100% perfect under real-world conditions (drastic lighting changes, angle/profile views, sunglasses, aging).
Rather than pretending the AI is infallible, Auto-Sorter gives the user **1-click correction superpowers**:
- If the AI splits a person into two cards (e.g. child vs adult, or with glasses vs without): **Merge them**.
- If a photo contains a wrong person: **Remove or shift it to a new card in 2 clicks**.
- All manual corrections are tagged with `assign_source = 'user'` and `assign_conf = 1.0`, meaning re-scans and automated updates will **never overwrite user decisions**.

---

### 2. Mathematical Centroid Recomputation
When faces are added, removed, or merged, the person's identity vector (centroid) must be updated immediately so future comparisons are accurate:

$$\vec{C} = \frac{\sum_{i=1}^N q_i \cdot \vec{e}_i}{\sum_{i=1}^N q_i}$$

Where:
- $\vec{e}_i$ is the 128-dimensional embedding of face $i$.
- $q_i = \max(\text{quality}_i, 0.1)$ is the face quality weight.
- $\vec{C}$ is L2-normalized: $\vec{C} \leftarrow \frac{\vec{C}}{\|\vec{C}\|}$.

**Why quality weighting matters:**
High-resolution, front-facing, sharp photos dominate the average vector, while blurry or extreme-angle photos contribute less, keeping the centroid crisp and representative.

---

### 3. Database Operations for Corrections

#### 🔗 1-Click Merge (`db.merge_persons(source_id, target_id)`)
```python
def merge_persons(self, source_person_id: int, target_person_id: int) -> None:
    # 1. Move all faces from source to target
    self._conn.execute(
        "UPDATE faces SET person_id = ?, assign_source = 'user', assign_conf = 1.0 WHERE person_id = ?",
        (target_person_id, source_person_id),
    )
    # 2. Delete the source person record
    self._conn.execute("DELETE FROM persons WHERE id = ?", (source_person_id,))
    self._conn.commit()

    # 3. Recompute centroid, face count, and cover photo for target person
    self.recompute_person_centroid(target_person_id)
```

#### ❌ "Not This Person" (`db.unassign_face(face_id)`)
```python
def unassign_face(self, face_id: int) -> Optional[int]:
    face = self.get_face_by_id(face_id)
    old_person_id = face["person_id"]
    
    self._conn.execute(
        "UPDATE faces SET person_id = NULL, assign_source = 'user', assign_conf = NULL WHERE id = ?",
        (face_id,),
    )
    self._conn.commit()

    if old_person_id is not None:
        self.recompute_person_centroid(old_person_id)
    return old_person_id
```

#### ➕ Create New Person from Face (`db.create_person_for_face(face_id, name)`)
```python
def create_person_for_face(self, face_id: int, display_name: Optional[str] = None) -> int:
    face = self.get_face_by_id(face_id)
    person_id = self.insert_person(
        display_name=display_name or f"Person {self.get_next_person_number()}",
        centroid_bytes=face["embedding"],
        face_count=1,
        cover_face_id=face_id,
    )
    self._conn.execute(
        "UPDATE faces SET person_id = ?, assign_source = 'user', assign_conf = 1.0 WHERE id = ?",
        (person_id, face_id),
    )
    self._conn.commit()
    return person_id
```

---

### 4. UI Architecture & PySide6 Lessons

```
┌─────────────────────────────────────────────────────────────┐
│ PeoplePanel                                                 │
│ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
│ │ PersonCard 1  │ │ PersonCard 2  │ │ PersonCard 3  │ ...  │
│ │ [Avatar 110px]│ │ [Avatar 110px]│ │ [Avatar 110px]│      │
│ │  Name ✏️   🔗 │ │  Name ✏️   🔗 │ │  Name ✏️   🔗 │      │
│ └───────┬───────┘ └───────────────┘ └───────────────┘      │
└─────────┼───────────────────────────────────────────────────┘
          │ Click Card
          ▼
┌─────────────────────────────────────────────────────────────┐
│ PersonDetailPanel (900x680)                                 │
│ Header: [Name] — [Count] photos | [✏️ Rename] [🔗 Merge with]│
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Scrollable Face Grid (120x120 thumbnails)               │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │ │
│ │ │ Face Thumb  │ │ Face Thumb  │ │ Face Thumb  │         │ │
│ │ │ 🔍 Full Pic │ │ 🔍 Full Pic │ │ 🔍 Full Pic │         │ │
│ │ └──────┬──────┘ └─────────────┘ └─────────────┘         │ │
│ └────────┼────────────────────────────────────────────────┘ │
└──────────┼──────────────────────────────────────────────────┘
           │ Double-click or click 'Full Pic'
           ▼
┌─────────────────────────────────────────────────────────────┐
│ PhotoViewer (Full Resolution Photo)                         │
│ [Draws colored bounding boxes on faces]                    │
│ Click face box → Context Menu:                             │
│   • Assigned to: [Name]                                    │
│   • ➡️ Move / Assign to Person...                          │
│   • ❌ Remove from Person                                  │
│   • ⭐ Set as Cover Face                                   │
│   • 🔍 Find Similar Faces                                  │
└─────────────────────────────────────────────────────────────┘
```

#### 💡 Key Qt Gotchas Resolved:
1. **Mouse Event Interception by Child QLabels**:
   - `QLabel` widgets consume mouse events by default. When an image label is placed inside a clickable cell, double-clicks do not bubble up to the cell.
   - **Fix**: Set `label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)`.
2. **Shiboken C++ Signal Type Safety**:
   - Passing `sqlite3.Row` dictionaries across `Signal(dict)` triggers Shiboken conversion exceptions in C++.
   - **Fix**: Use primitive integer types (`Signal(int)`) passing the `face_id` or `person_id`.
3. **Responsive Grid Wrapping**:
   - Instead of hard-coded row/column counts, dynamic column calculation in `resizeEvent` recalculates `cols = max(1, (width - 60) // (FACE_SIZE + 24))` so grids re-flow smoothly on any display scale.

---

## 🧪 Test Coverage

| Test Case | What It Validates |
|---|---|
| `test_merge_two_persons` | Reassigns all faces, updates target centroid/count/cover, deletes source card |
| `test_merge_same_person_noop` | Merging person into self is safe no-op |
| `test_unassign_removes_face_and_updates_centroid` | Unassigns face, cleanses centroid, picks new cover |
| `test_unassign_last_face_deletes_empty_person` | Removing only face cleanly deletes empty card |
| `test_reassign_between_persons` | Moves face between cards, updates both centroids |
| `test_create_person_for_face` | Seeds new person card from single face |
| `test_corrections_survive_reopen` | All corrections persist across database restart |
