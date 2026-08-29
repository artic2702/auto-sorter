# Auto-Sorter — Offline Photo Organizer for Windows

> Living plan document. Progress checkboxes updated after every milestone; any change gets a line in the Decisions log at the bottom.

## Progress

- [x] **M0** — Scaffold: git, Python 3.12 venv, deps, models downloaded, empty window, first test
- [x] **M1** — Scan folder → SQLite + thumbnails + live-count UI
- [x] **M2** — Face detection (worker pool) + face thumbs + boxes in viewer
- [x] **M3** — Faceprints (align + SFace) stored; dev "find similar" panel
- [x] **M4** — People discovery + rename (incremental centroid clustering, person cards, inline rename)
- [x] **M5** — Corrections: merge, not-this-person, reassign, box-click assign
- [x] **M6** — Resume/incremental: kill-and-relaunch resume, new-photos-only, moved-file reattach
- [x] **M7** — Albums (Group/Unknown/Review/Text&Docs) + optional Export
- [ ] **M8** — Performance: 20k+ test, speed dial tuning, low-spec benchmark, memory cap
- [ ] **M9** — PyInstaller + Inno Setup installer, verified offline on clean Windows

---

## What we're building (one paragraph)

A Windows app: pick a folder full of photos → the app finds every face, automatically groups photos by person ("Person 1, Person 2…"), you name each person **once**, and from then on new photos are filed under the right person automatically. Also auto-detects screenshots/documents into a "Text & Documents" album, and group photos into a "Group Photos" album. 100% offline, original photos never touched, ends as a double-click installer (.exe).

## The core idea (how it remembers people)

- The AI model (SFace) is a frozen **face-measuring machine**: give it a face, it outputs 128 numbers — a **faceprint**. Same person → nearly identical numbers. It is never trained/retrained on user photos.
- **First scan:** every face → faceprint → similar faceprints auto-grouped → "Person 1 (623 photos)". User names the *group* once → all 623 photos labeled instantly.
- **New photos later:** faceprint compared to each known person's stored average.
  - Very similar → auto-filed (no question).
  - Somewhat similar → "Is this Arnav? ✓/✗" in a Needs Review list.
  - Not similar → new "Person N" or Unknown.
- Each confirmed face improves the stored average. Everything lives in one local SQLite database file.
- **Careful by default:** the app would rather show the same person as two cards (1-click Merge fixes it) than mix two different people together.

## The first-scan problem — and the solution

Research: even Apple Photos, with a dedicated AI chip, needs **several nights** to analyze a few thousand photos, and only runs while charging + locked ([Apple ML research](https://machinelearning.apple.com/research/recognizing-people-photos)). A slow first scan is universal; the fix is making it **gentle and interruptible**, not skipping it:

1. **Low Windows process priority** — scan only uses CPU that's otherwise idle; PC stays fully usable.
2. **Speed dial:** Quiet (1 worker) / Normal (half the cores) / Fast (all cores, for overnight).
3. **Pause / shutdown anytime** — SQLite is the checkpoint; every finished image is saved; relaunch offers "Resume (12,340 / 30,000 done)". Nothing reprocessed, nothing lost.
4. **Live results** — person cards appear *while* scanning; newest photos processed first so value shows in minutes.
5. **One-time cost** — afterwards only new/changed files are processed (detected via file size+date, hash for moved files).
6. **Reduced-size decode** — faces are detected on a downscaled image (`cv2.IMREAD_REDUCED_COLOR_2/4`); decoding is the real bottleneck and this cuts it ~4×.

**Realistic first-scan times** (verified in milestone M8 on real low-end hardware):

| PC | Speed | 10,000 photos |
|---|---|---|
| Old dual-core, 4 GB RAM (minimum spec) | ~0.5–1.5 img/s | one overnight run |
| i5/Ryzen 5, 8 GB (recommended) | ~3–6 img/s | ~1 hour |

RAM stays capped (~1–1.5 GB) no matter how big the library is, via bounded queues. **No GPU ever used.** Minimum spec: Windows 10 64-bit, any dual-core x64 from ~2015, 4 GB RAM.

## Why these exact models (researched, not assumed)

**digiKam** — a popular free photo manager — ships offline face recognition using **exactly YuNet (detection) + SFace (recognition) on CPU** ([digiKam 8.6](https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/)). Proven in production. Measured speeds: YuNet ~1.6 ms/image on desktop CPU ([paper](https://link.springer.com/article/10.1007/s11633-023-1423-y)); SFace ~10 ms/face. Both models are **Apache-2.0 licensed including the weights** (OpenCV Zoo) — safe to publish on GitHub, share, even sell later. (Rejected: InsightFace ArcFace/SCRFD — better accuracy but weights are non-commercial + trained on a retracted dataset.)

## Tech stack (small on purpose)

| Piece | Choice | Why |
|---|---|---|
| Language | Python 3.12 (venv; system 3.14 lacks reliable wheels) | |
| UI | PySide6 (Qt) | Native Windows desktop, handles 100k-item grids via virtualization |
| Face AI | OpenCV built-ins: `cv2.FaceDetectorYN` (YuNet) + `cv2.FaceRecognizerSF` (SFace) | No ONNX Runtime needed → smaller install, one dependency does decode+detect+recognize |
| Math/grouping | NumPy, own incremental centroid clusterer | Exact cosine search is milliseconds even at 100k faces; streaming & resumable |
| Storage | SQLite (WAL mode) | The database IS the crash-recovery checkpoint |
| Packaging | PyInstaller (onedir) + Inno Setup | Standard path; Nuitka fallback if antivirus complains |

Not in MVP: onnxruntime, FAISS, scikit-learn, OCR, HEIC (all have clean extension points later).

## Architecture (one process, modular monolith)

```
UI (Qt main thread) ── signals ──▶ progress, live results
        │ facade calls
        ▼
JobController:
  Scanner ─▶ [queue] ─▶ FaceWorker ×N ─▶ [queue] ─▶ IdentityWorker (1)
  walk files,          decode small, detect,        compare faceprint to people,
  hash, EXIF,          align, embed, quality,       auto-assign / review / new person
  doc-heuristic        thumbnails                   update centroids
        └──────────────────┴─────────────┬──────────────┘
                                   DBWriter thread (single writer, batched commits)
```

- Workers run at below-normal OS priority; count set by the speed dial.
- Rules: sources opened **read-only** everywhere except the explicit Export feature. Never `cv2.imread(path)` on Windows (Unicode paths break) — always `np.fromfile` + `imdecode`.

## Database (SQLite, v1)

```
images(id, path UNIQUE, size, mtime, sha256, dhash, w, h, taken_at, doc_score,
       status: pending|processed|error|missing, error)
faces(id, image_id, bbox, landmarks, det_score, quality, embedding BLOB(128×f32),
      thumb_path, person_id NULL, assign_source: auto|user|review, assign_conf)
persons(id, display_name, name_is_user_set, cover_face_id, centroid BLOB,
        face_count, merged_into NULL, hidden)
negatives(face_id, person_id)          -- "not this person", permanent veto
review_items(id, face_id, suggested_person_id, confidence, status)
jobs(id, root_path, state, counters_json, timestamps)
meta(key, value)                       -- schema version, calibrated thresholds
```

- A "cluster" IS a `persons` row; renaming just sets the name. Embeddings L2-normalized (cosine = dot product).
- **User corrections beat automation, permanently:** `assign_source='user'` faces are never auto-moved; `negatives` checked before every auto-assign; both survive re-scans.

## Matching & confidence (careful mode)

```
new face → similarity to every person's centroid (one matvec)
         → refine vs best person's top-K confirmed faces → score s, margin = best − second
s ≥ T_HIGH and margin ok  → auto-assign, update centroid
T_REVIEW ≤ s < T_HIGH     → Needs Review ("Is this Arnav?")
s < T_REVIEW              → new Person N / Unknown
```

- Low-quality faces (tiny/blurry) stored but excluded from matching — they poison averages.
- Age-gap handling: child-Arnav and adult-Arnav become two cards + an automatic **merge suggestion** in Review; never auto-merged once a person is user-named.
- Thresholds are **not hard-coded**: `tools/calibrate.py` runs the real pipeline on labeled pairs (LFW + personal set), picks T_HIGH at ≤0.1% false-accept, stores in `meta`. OpenCV's published SFace threshold (0.363 cosine) is the sanity anchor.

## UI screens

1. **Home/Library** — pick photo folder + data folder, Start; live counters, progress, Pause/Resume/Cancel, speed dial; photo grid fills in live.
2. **People** — person cards (face, name, count); Rename, Merge, Hide.
3. **Person detail** — their photos; per-face: Not this person / Move to… / Set cover.
4. **Photo viewer** — face boxes drawn; click a box → assign to person or create new.
5. **Needs Review** — ✓/✗ queue for uncertain matches + merge suggestions.
6. **Albums** — Group Photos, Unknown People, Text & Documents (all virtual = DB queries; optional Export copies files only after explicit confirm).

## Project layout

```
d:\Auto_sorter\
  autosorter/
    __main__.py
    app/        screens/ widgets/ models/ facade.py
    core/       db/ scan/ faceengine/ identity/ classify/ pipeline/ export/
    models/     yunet.onnx + sface.onnx + LICENSE files
  tools/        calibrate.py
  tests/
  packaging/    pyinstaller spec + installer.iss
```

## Build order (app runs after every step)

| # | Milestone | Done when |
|---|---|---|
| M0 | Scaffold: git, Py3.12 venv, deps, models downloaded, empty window | `python -m autosorter` opens; pytest green |
| M1 | Scan folder → SQLite + thumbnails + live-count UI | 5k photos scanned, UI never freezes, re-run skips unchanged |
| M2 | Face detection (worker pool) + face thumbs + boxes in viewer | faces visible & aligned on personal test set |
| M3 | Faceprints (align + SFace) stored; dev "find similar" panel | same-person similarity clearly > different-person |
| M4 | **People discovery + rename** ← first full vertical slice | on 1–3k personal photos: top people found, zero wrong merges, rename survives restart |
| M5 | **Corrections: merge, not-this-person, reassign, box-click assign** | every correction survives restart + full re-scan |
| M6 | **Resume/incremental: kill-and-relaunch resume, new-photos-only, moved-file reattach** | kill mid-scan → resumes exactly; new photos of named person auto-file |
| M7 | Albums (Group/Unknown/Review/Text&Docs) + optional Export | group photo under each person with ONE file on disk; export never touches sources |
| M8 | Performance: 20k+ test, speed dial tuning, low-spec benchmark, memory cap verified | documented img/s on min-spec machine; RAM < 1.5 GB; multiprocessing fallback if threads underperform |
| M9 | PyInstaller + Inno Setup installer | clean Windows VM, **network disabled**: install → scan → name → restart → resume all work; zero network calls observed |

## Main risks

- **SFace weaker than big models on profiles/age gaps** → conservative thresholds + Review queue turn errors into questions, not mistakes; model layer is swappable.
- **PyInstaller antivirus false positives** → onedir, no UPX; Nuitka fallback; expect SmartScreen "unsigned" warning.
- **Python threads limited by GIL** → OpenCV releases the GIL so threads should scale; M8 has an explicit go/no-go to switch workers to multiprocessing.
- **Windows paths** (Unicode, >260 chars) → imdecode rule + long-path handling + tests with Hindi/emoji filenames.

## Out of scope for now (extension points exist)

OCR/search, semantic search, pets/objects, video, blur/best-shot detection, duplicate-cleanup UI (dHash already stored), HEIC, OpenVINO, Fast/Accurate model modes.

---

## Decisions log

- 2026-07-10 — Plan approved. Stack: Python 3.12 + PySide6 + OpenCV (YuNet + SFace, Apache-2.0) + SQLite. Conservative clustering. Finish line = installer (M9).
- 2026-07-10 — Repo pushed to https://github.com/artic2702/auto-sorter (user's request).
- 2026-07-10 — M0 done. License check on downloaded weights: YuNet is actually **MIT** (not Apache-2.0), SFace is Apache-2.0 — both permissive, license files committed next to the models. Models verified loading + running via pytest.
- 2026-08-20 — M1 done. High-performance scanner, SQLite WAL DB, thumbnails with EXIF auto-rotation, background thread worker with Qt live signals.
- 2026-08-21 — M2 done. Face detection pipeline using YuNet, landmark extraction, affine alignment, quality scoring, and live photo viewer overlay.
- 2026-08-22 — M3 done. Face embeddings using SFace, 128-d L2 normalized faceprints in BLOB storage, Dev Find Similar matrix search panel.
- 2026-08-23 — M4 done. Incremental centroid clustering with drift protection and member validation, person cards, inline renaming.
- 2026-08-24 — M5 done. Manual corrections: 1-click merge cards with live search filter, "Not this person", "Move to...", "Create new person card", cover photo selection, and interactive box-click assignment in full photo viewer. All 114 tests passing.
- 2026-08-27 — M6 done. Unified single-button pipeline (scan→detect→embed→cluster), crash-safe SQLite resume, SHA-256 moved-file reattach, missing-file detection, true upsert fix (INSERT OR REPLACE → ON CONFLICT preserving faces). All 129 tests passing.
- 2026-08-29 — M7 done. Virtual smart albums (Group Photos, Unknown Faces, Text & Documents) powered by SQL queries. Lightweight heuristic document/screenshot classifier (filename, resolution, color entropy, edge density). Safe export engine copying photos into organized person folders with group photo duplication, collision resolution, and resume support. All 170 tests passing.
