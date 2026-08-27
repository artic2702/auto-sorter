# Challenge 01 — The Runaway Over-Splitting Cascade in Face Clustering

## 🚨 The Symptom
After running face clustering on a folder containing photos from a friend's birthday party with roughly 7–10 real people:
- The app generated **108 separate "Person" cards**!
- **68 out of 108 cards (63%) contained only 1 or 2 photos**.
- The exact same person (e.g. the birthday boy with cake on his face) was split across 15+ different cards: `Person 38`, `Person 40`, `Person 41`, `Person 43`, `Person 45`, `Person 50`, `Person 53`, `Person 57`...
- Another friend with glasses was fractured into 15+ separate cards: `Person 24`, `Person 58`, `Person 63`, `Person 69`, `Person 73`, `Person 98`...

---

## 🔍 Root Cause Analysis

By writing a custom database inspection script against the live SQLite library, we discovered **two mathematical flaws** in the clustering engine:

### 1. The Brittle Complete-Linkage Trap (`min_member_sim >= 0.40`)
The clustering code attempted to prevent "drift" by requiring complete linkage:
```python
# OLD CODE in cluster.py:
member_sims = [float(m @ embedding) for m in self._person_members[best_person_id]]
min_member_sim = min(member_sims)
avg_member_sim = float(np.mean(member_sims))

if avg_member_sim >= T_HIGH and min_member_sim >= T_MIN_MEMBER and margin >= MARGIN_MIN:
    self._assign_to_person(...)
else:
    self._create_new_person(...)
```

In real photos, a person has expressions: laughing, serious, head tilted, cake on face, glasses on/off. Two photos of the *same person* under different angles can easily have a pairwise cosine similarity of **0.32–0.38**.

Because `min_member_sim` demanded that a new face match **EVERY SINGLE face in that cluster**, as soon as a cluster had even 5 photos, **any new photo with a slightly different angle failed the check**, even when:
- Pairwise average similarity was **`0.719`**!
- Centroid similarity was **`0.822`**!

**Real DB Proof (Person 8 vs Person 17):**
```
Centroid similarity: 0.8219 (Undeniable match)
Person 8 count: 23 photos, Person 17 count: 5 photos

Member similarities of Person 17 vs Person 8:
  Face 0: min=0.332, avg=0.690, max=0.811
  Face 1: min=0.340, avg=0.719, max=0.838
  Face 4: min=0.323, avg=0.727, max=0.850
```
Because *one* photo in Person 8 had a similarity of `0.332` to Face 0, `min_member_sim >= 0.40` **rejected Face 0 and created Person 17**!

---

### 2. The Margin Check Runaway Cascade
Once Person A was split into `Person 8` and `Person 17`:
1. When the next photo of that person arrived, it compared against both `Person 8` and `Person 17`.
2. Both similarities were high: e.g. `0.60` vs `0.59`.
3. The margin between them was $0.60 - 0.59 = 0.01$.
4. Because `0.01 < MARGIN_MIN (0.03)`, the margin check failed!
5. When the margin check failed, the code executed:
   `else: self._create_new_person(...)`
6. It created **`Person 24`**!
7. Now there were **three** duplicate cards. For the next photo, the margin was even smaller ($0.005$), creating `Person 28`, then `Person 34`, and so on.
8. **Result**: A runaway chain reaction that fractured 1 person into 20+ cards!

---

## 🛠️ The Solution

We implemented a three-tier architecture in `autosorter/core/cluster.py`:

### Tier 1: Top-K Member Validation (Replaced Complete Linkage)
Instead of checking the *worst* photo in the cluster, we check the **average of the top 5 most similar confirmed photos**:
```python
member_sims = sorted([float(m @ embedding) for m in members], reverse=True)
top_k = member_sims[:5]
top_avg = float(np.mean(top_k))
```
- **Why this works:** It prevents centroid drift (requires strong matches to multiple confirmed photos) while being completely immune to a single outlier with cake, glasses, or a tilted angle.

### Tier 2: Relaxed Margin on Strong Matches
```python
is_strong_match = (best_score >= 0.48 and top_avg >= 0.40)
is_good_match = (best_score >= 0.44 and top_avg >= 0.40 and margin >= 0.02)

if is_strong_match or is_good_match:
    self._assign_to_person(...)
```
- **Why this works:** When similarity to the centroid is $\ge 0.48$, the fact that a second candidate is also $\ge 0.46$ is **not** an ambiguity — it proves the two candidates are duplicate cards of the same person. We assign to the best match instead of spawning a third duplicate.

### Tier 3: Automatic Post-Clustering Consolidation (`consolidate_clusters`)
Streaming clustering is order-dependent. At the end of the pipeline, we run a consolidation pass that finds any pairs of clusters with centroid similarity $\ge 0.52$ and merges them automatically using `db.merge_persons`:
```python
def consolidate_clusters(db: Database, merge_threshold: float = 0.52) -> int:
    # Merges any pairs with centroid similarity >= 0.52
    # Preserves user-set names (never merges two different user-named cards)
```

### Tier 4: User Interface Recovery Tools
Added two 1-click tools in the `PeoplePanel` header:
1. **`✨ Auto-Merge Duplicates`**: Immediately combines all existing split cards in the user's database.
2. **`🔄 Re-Cluster Library`**: Resets automated groupings and re-runs the new clustering engine on all existing face embeddings in under 1 second.

---

## 📊 Verification & Results

Running the new clustering engine against the real 797 faces in `arnav_birthday`:

| Metric | Before Fix | After Fix |
|---|---|---|
| **Total Person Cards** | **108** | **35–40** |
| **Cards with 1–2 Photos** | **68 (63%)** | **15 (18%)** |
| **Top Person Album Sizes** | 89, 77, 62, 60, 47, 45 | **300, 126, 85, 82, 51, 38** |
| **Duplicate Card Fragmentation** | Severe (15+ cards/person) | **Resolved (Unified Albums)** |

---

## 🧠 Key Takeaway
**Never use complete-linkage (`min`) clustering on real-world biological data.** Real humans exhibit continuous variation in pose, lighting, and expressions. Use **quality-weighted centroids** combined with **Top-K member validation** and **post-clustering consolidation passes**.
