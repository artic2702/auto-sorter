# 🛠️ Challenges We Faced & How We Solved Them

This directory is a living engineering journal documenting the real-world bugs, mathematical traps, operating system quirks, and architectural challenges encountered during the development of **Auto-Sorter** — and the precise solutions implemented to fix them.

---

## 📑 Table of Contents

| # | Challenge | Subsystem | Severity | Summary |
|---|---|---|---|---|
| **01** | [The Runaway Over-Splitting Cascade in Face Clustering](01_face_clustering_oversplitting.md) | AI & Math | 🔴 Critical | Complete-linkage (`min_member_sim >= 0.40`) caused 108 split cards for a 7-person party. Replaced with Top-K validation, relaxed strong margins, and automatic centroid consolidation. |
| **02** | [The `INSERT OR REPLACE` Foreign Key Cascade Disaster](02_sqlite_upsert_cascade_deletion.md) | Database | 🔴 Critical | `INSERT OR REPLACE` deleted existing image rows on re-scans, triggering `ON DELETE CASCADE` and wiping out all detected faces. Fixed with `ON CONFLICT DO UPDATE`. |
| **03** | [Qt Event Bubbling & Shiboken C++ Conversion Crashes](03_qt_event_bubbling_and_shiboken_crashes.md) | UI (PySide6) | 🟠 High | Inner `QLabel` consumed all mouse clicks in person cards, and emitting `sqlite3.Row` across Qt signals crashed C++. Solved with `WA_TransparentForMouseEvents` and primitive signal types. |
| **04** | [Windows Unicode Path Incompatibility in OpenCV](04_windows_unicode_paths_opencv.md) | OS & I/O | 🟠 High | `cv2.imread` fails silently on Windows when file paths contain non-ASCII characters, Hindi, spaces, or emojis. Fixed using `np.fromfile` and `imdecode`. |
| **05** | [Moved & Renamed Photo Reattachment via SHA-256](05_moved_file_reattach_sha256.md) | Storage | 🟡 Medium | Moving photos to new folders caused duplicate scanning and lost identity history. Solved using SHA-256 content hashing to reattach paths in-place. |
| **06** | [False Positive Face Filtering on Text & Posters](06_false_positive_face_filtering.md) | Computer Vision | 🟡 Medium | YuNet occasionally detected text posters as faces. Solved with geometric landmark validation (aspect ratio, eye spacing, triangle area) and stricter person seed thresholds. |
| **07** | [Orphaned Faces & The Co-Occurrence Constraint](07_orphaned_faces_and_cooccurrence.md) | AI & Database | 🔴 Critical | Faces in group photos showed raw percentages instead of names because their person IDs pointed to deleted records. Solved with orphan cleanup and co-occurrence cannot-link constraints. |
| **08** | [The Greedy Centroid Trap: Scaling Face Identity with Sparse Constrained Graphs & Active Learning](08_sparse_constrained_graph_clustering.md) | AI & Graph Theory | 🔴 Critical | Greedy single-pass clustering drifted on look-alikes and fractured into 108 cards. Solved with chunked k-NN, two-tier mutual graph, Chinese Whispers, zero-conflict consolidation, and 1-click active learning album re-check. |

---

## 💡 How to Use This Knowledge Base
Each document follows a strict engineering breakdown:
1. **The Symptom** — What user experience or crash occurred.
2. **Root Cause Analysis** — Mathematical or code-level deep dive into why it happened.
3. **The Solution** — The exact code changes, algorithms, and architectural adjustments.
4. **Verification & Proof** — Real data, before/after metrics, and tests verifying the fix.
5. **Key Takeaway** — Reusable engineering lesson for future projects.
