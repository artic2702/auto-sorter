# Challenge 07 — Orphaned Faces & The Co-Occurrence Constraint

## 🚨 The Symptom
When viewing group photos in the **Photo Viewer** (e.g. `IMG_E2611.HEIC` showing 7 friends in a hallway):
- Only one face box had an assigned person tag (`Person 4`).
- The other 6 face boxes showed raw detection confidence percentages (`87%`, `92%`, `86%`, `92%`, `91%`, `80%`) with no person allotted.

---

## 🔍 Root Cause Analysis

We inspected the actual SQLite rows for `IMG_E2611.HEIC`:
```
Face 684: person_id = 112 (Person 4 exists in database)
Face 685: person_id = 2   (Person 2 DOES NOT EXIST in persons table!)
Face 686: person_id = 3   (Person 3 DOES NOT EXIST in persons table!)
Face 687: person_id = 15  (Person 15 DOES NOT EXIST in persons table!)
Face 688: person_id = 15  (Person 15 DOES NOT EXIST in persons table!)
Face 689: person_id = 8   (Person 8 DOES NOT EXIST in persons table!)
Face 690: person_id = 15  (Person 15 DOES NOT EXIST in persons table!)
```

### 1. The Orphaned Face Foreign Key Trap
- In `PhotoViewer`, the label logic is:
  ```python
  if db and face["person_id"]:
      p = db.get_person_by_id(face["person_id"])
      if p:
          person_name = p["display_name"]
  tag_text = person_name if person_name else f"{face['det_score']:.0%}"
  ```
- During previous merge operations or re-clustering resets, older auto-created person records (`Person 2`, `Person 3`, `Person 8`, `Person 15`) were deleted from the `persons` table.
- Because `faces.person_id` did not have an active cascade cleanup, the `faces` rows remained frozen pointing to dead, non-existent person IDs!
- When `PhotoViewer` queried `db.get_person_by_id(2)`, it returned `None`.
- The code fell back to displaying the raw detection confidence: `87%`, `92%`, etc.
- In total, **331 faces in the database were orphaned pointing to deleted person records**.

---

### 2. The Missing Co-Occurrence Constraint (Cannot-Link)
In group photos, multiple people appear in the exact same frame:
- **A human being cannot physically be in two places at the same time.**
- If Face A and Face B come from the **same `image_id`**, they are mathematically guaranteed to be **different people**.
- Without a co-occurrence constraint:
  1. A person cluster could mistakenly accept two faces from the same photo.
  2. Two person clusters could merge even if they both contained people from the same group photo.
- In `arnav_birthday`, there are **187 group photos with > 1 face**, creating **1,371 cannot-link face pairs**.

---

## 🛠️ The Solution

### 1. Automated Orphan Cleanup in the Pipeline
Before running Stage 4 clustering, any `person_id` pointing to a non-existent person record is automatically cleared (`person_id = NULL`), allowing the pipeline to re-assign it to the correct active person without manual user intervention.

### 2. The Co-Occurrence (Cannot-Link) Constraint
We introduced image-level mutual exclusivity in `ClusterEngine`:
1. **Assignment Check**: When evaluating a candidate person for a face from `image_id`, if that person *already* has a face from `image_id`, the assignment is rejected:
   ```python
   if image_id in candidate_person_image_ids:
       # Cannot link: This person is already in this photo!
       continue
   ```
2. **Consolidation Check**: When merging clusters:
   ```python
   if person_A_image_ids & person_B_image_ids:
       # Cannot merge: Both persons appear together in at least one photo!
       continue
   ```

---

## 📊 Verification & Results

When tested against `IMG_E2611.HEIC` (the hallway photo with 7 friends):
- **Before Fix:** 1 face allotted to `Person 4`, 6 faces orphaned showing raw percentages.
- **After Fix:** All 7 faces are 100% allotted to 7 distinct, active person albums (`85`, `127`, `84`, `103`, `71`, `51`, `123` photos each), with zero wrong merges between friends in the same photo!

---

## 🧠 Key Takeaway
**Group photos are the strongest ground truth in face recognition.** When two faces appear in the same photo, they form an absolute **cannot-link** constraint. Always track `image_id` co-occurrence to prevent different people in the same room from ever being merged.
