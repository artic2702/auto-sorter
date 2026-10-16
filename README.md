# Auto-Sorter

**Privacy-first, 100% offline photo organizer for Windows.**

Point it at a folder of photos → it finds every face, automatically groups photos by person, lets you name each person once, and files new photos under the right person automatically. Screenshots and documents are sorted into their own album. Your photos never leave your PC and the originals are never modified.

## Status

Milestones M0–M9 complete: full offline scanning, face detection, clustering, manual corrections, virtual albums, performance speed dial, and standalone packaging — see [PLAN.md](PLAN.md) for roadmap and progress.

## Highlights

- **Fully offline** — no cloud, no accounts, no uploads. Face analysis runs locally on CPU.
- **People discovery** — faces are grouped into "Person 1, Person 2…"; name a group once and every photo (past and future) is labeled.
- **Gentle first scan** — runs at low priority in the background, pause/shutdown anytime, resumes exactly where it stopped.
- **Originals untouched** — organization is virtual (database albums); files are only copied if you explicitly export.
- **Low-end friendly** — targets Windows 10/11, dual-core CPU, 4 GB RAM, no GPU.

## Privacy Note

The `.autosorter/` data folder inside your photo folder stores face thumbnails, database records, and biometric faceprint embeddings.
Do not share or publish that folder if privacy matters.
Your original photos are never modified or uploaded anywhere.

## Tech

Python 3.12 · PySide6 (Qt) · OpenCV YuNet (face detection) + SFace (face recognition, both Apache-2.0 incl. weights) · SQLite

## Development setup

```powershell
py install 3.12            # if not present (Python Install Manager)
py -V:3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m autosorter    # run the app
```

## Run the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

## Documentation

- [Project Roadmap & Plan](PLAN.md)
- [Learning Guides Overview](docs/learning/README.md)
  - [M0 — Scaffold](docs/learning/M0_scaffold.md)
  - [M1 — Scan & Database](docs/learning/M1_scan_and_database.md)
  - [M2 — Face Detection](docs/learning/M2_face_detection.md)
  - [M3 — Faceprints](docs/learning/M3_faceprints.md)
  - [M4 — People Discovery](docs/learning/M4_people_discovery.md)
  - [M5 — Corrections](docs/learning/M5_corrections.md)
  - [M6 — Resume & Incremental](docs/learning/M6_resume_incremental.md)
  - [M7 — Albums & Safe Export](docs/learning/M7_albums_and_export.md)
- [Challenges We Faced & Solutions](docs/challenges_we_faced/README.md)

## License

All rights reserved until a license is chosen. Bundled model weights have their own permissive licenses in `autosorter/models/` (YuNet: MIT, SFace: Apache-2.0).
