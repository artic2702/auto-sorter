# Auto-Sorter

**Privacy-first, 100% offline photo organizer for Windows.**

Point it at a folder of photos → it finds every face, automatically groups photos by person, lets you name each person once, and files new photos under the right person automatically. Screenshots and documents are sorted into their own album. Your photos never leave your PC and the originals are never modified.

## Status

🚧 In development — see [PLAN.md](PLAN.md) for the roadmap and current progress.

## Highlights

- **Fully offline** — no cloud, no accounts, no uploads. Face analysis runs locally on CPU.
- **People discovery** — faces are grouped into "Person 1, Person 2…"; name a group once and every photo (past and future) is labeled.
- **Careful by default** — uncertain matches go to a Needs Review queue instead of being guessed.
- **Gentle first scan** — runs at low priority in the background, pause/shutdown anytime, resumes exactly where it stopped.
- **Originals untouched** — organization is virtual (database albums); files are only copied if you explicitly export.
- **Low-end friendly** — targets Windows 10/11, dual-core CPU, 4 GB RAM, no GPU.

## Tech

Python 3.12 · PySide6 (Qt) · OpenCV YuNet (face detection) + SFace (face recognition, both Apache-2.0 incl. weights) · SQLite

## Development setup

```powershell
py install 3.12            # if not present (Python Install Manager)
py -V:3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m autosorter    # run the app
.\.venv\Scripts\python -m pytest        # run tests
```

## License

TBD.
