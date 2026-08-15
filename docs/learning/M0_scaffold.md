# Milestone 0 — Scaffold

## 🎯 Goal
Set up the project foundation so that one command opens an empty window, and the AI models are verified working.

---

## 📦 Concepts Learned

### 1. Virtual Environments (`.venv`)

**Problem**: If you install Python packages globally (for all projects), they can conflict. Project A needs `numpy 1.26`, but Project B needs `numpy 2.0` — installing one breaks the other.

**Solution**: A **virtual environment** is an isolated Python installation just for THIS project. It lives inside the `.venv/` folder.

```
D:\Auto_sorter\
  .venv/                    ← isolated Python + packages
    Scripts/
      python.exe            ← THIS Python, with OUR packages
      pip.exe
    Lib/
      site-packages/
        PySide6/            ← installed only for this project
        cv2/
        numpy/
```

**Commands**:
```powershell
# Create a virtual environment
py -V:3.12 -m venv .venv

# Activate it (tells your terminal to use .venv's Python)
.\.venv\Scripts\Activate

# Install packages INTO this environment
pip install -r requirements.txt

# Run code using this environment's Python
.\.venv\Scripts\python -m autosorter
```

**Why Python 3.12 (not 3.14)?**
Some libraries (like PySide6, OpenCV) ship pre-built binary "wheels" for specific Python versions. Python 3.14 is too new — the wheels don't exist yet. Using 3.12 guarantees everything installs smoothly.

---

### 2. Project Structure

```
D:\Auto_sorter\
  autosorter/               ← The main Python PACKAGE (our code)
    __init__.py             ← Makes this folder a package; sets __version__
    __main__.py             ← Entry point: runs when you do "python -m autosorter"
    paths.py                ← Centralized file paths (model locations)
    app/                    ← UI-related code
      __init__.py
      main_window.py        ← The Qt window
    models/                 ← AI model files (.onnx)
      face_detection_yunet_2023mar.onnx
      face_recognition_sface_2021dec.onnx
      LICENSE-yunet.txt
      LICENSE-sface.txt
  tests/                    ← Automated tests
    test_m0_scaffold.py
  .venv/                    ← Virtual environment (not committed to git)
  .gitignore                ← Tells git to ignore .venv, __pycache__, etc.
  requirements.txt          ← List of packages to install
  pyproject.toml            ← Project metadata + pytest config
  README.md                 ← Project description
  PLAN.md                   ← Detailed roadmap
```

**Key rule**: `__init__.py` makes a folder into a Python package. Without it, `from autosorter.app import main_window` would fail with `ModuleNotFoundError`.

---

### 3. `__main__.py` — The Entry Point

When you run `python -m autosorter`, Python looks for `autosorter/__main__.py` and executes it. Here's what it does:

```python
import sys
from PySide6.QtWidgets import QApplication
from autosorter.app.main_window import MainWindow

def main() -> int:
    # QApplication: THE core Qt object. Every Qt app needs exactly ONE.
    # It manages the event loop (listening for clicks, key presses, etc.)
    app = QApplication(sys.argv)
    app.setApplicationName("Auto-Sorter")

    # Create and show our window
    window = MainWindow()
    window.show()

    # app.exec() starts the EVENT LOOP:
    # → Wait for user action (click, type, resize)
    # → Handle it (update UI, run code)
    # → Repeat forever until the window is closed
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
```

**What is an Event Loop?**
Think of it like a receptionist at a desk:
1. Sit and wait for someone to come in
2. Handle their request (button click → run a function)
3. Go back to waiting
4. Repeat until the office closes (window closed)

Without `app.exec()`, the window would flash open and immediately close.

---

### 4. AI Models — What Are ONNX Files?

**ONNX** (Open Neural Network Exchange) is a universal format for AI models. It's like PDF for documents — any tool can read it, regardless of what created it.

Our two models:

#### YuNet (`face_detection_yunet_2023mar.onnx` — 227 KB)
- **Input**: An image (any size, resized internally to 320×320)
- **Output**: List of face locations, each with:
  - Bounding box: `[x, y, width, height]`
  - 5 landmarks: left eye, right eye, nose, left mouth, right mouth
  - Confidence score: 0.0 to 1.0 (how sure it is that this is a face)
- **Speed**: ~1.6ms per image on desktop CPU
- **License**: MIT (do anything you want with it)

#### SFace (`face_recognition_sface_2021dec.onnx` — 37 MB)
- **Input**: An aligned face crop (112×112 pixels)
- **Output**: A 128-number vector (the "faceprint")
- **Purpose**: Comparing faces. Same person → similar numbers.
- **Speed**: ~10ms per face on desktop CPU
- **License**: Apache-2.0 (do anything, just include the license)

**Why these and not bigger/better models?**
| Model | Accuracy | Size | Speed | License | Our choice? |
|---|---|---|---|---|---|
| YuNet + SFace | Good | 37 MB | Fast | ✅ Free | ✅ YES |
| ArcFace (InsightFace) | Better | 250 MB | Slower | ❌ Non-commercial | No |
| FaceNet (Google) | Great | 90 MB | Medium | ⚠️ Complex | No |
| Apple Vision | Best | Huge | Needs GPU | ❌ Apple only | No |

We chose the smallest, fastest, most permissive option. The accuracy is "good enough" — and our **conservative thresholds** + **Review queue** catch the rare mistakes.

---

### 5. `paths.py` — Finding Files in Dev AND in Packaged App

```python
import sys
from pathlib import Path

def package_root() -> Path:
    if getattr(sys, "frozen", False):        # Are we inside a PyInstaller .exe?
        return Path(sys._MEIPASS) / "autosorter"  # Yes → use the temp extract dir
    return Path(__file__).parent              # No → use the source code directory

MODELS_DIR = package_root() / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
```

**Why this is smart**: When we eventually package the app as a `.exe` (M9), PyInstaller bundles all files into a temp directory. `sys._MEIPASS` points to that directory. This code handles both cases with zero changes.

---

### 6. Pytest — Automated Testing

**What is a test?** A function that checks: "does this code do what I expect?"

```python
def test_model_files_exist():
    assert YUNET_MODEL.is_file(), f"missing {YUNET_MODEL}"
```

- `assert` = "I declare this MUST be true"
- If true → test **passes** ✅
- If false → test **fails** ❌ with the message after the comma

**Why test?**
Every time you change code, you can run `pytest` and instantly know if you broke something. Without tests, you'd have to manually open the app and click through everything — slow and error-prone.

**Running tests**:
```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```
The `-v` flag means "verbose" — shows each test name and result instead of just dots.

---

## 📋 Files Created in M0

| File | Purpose |
|---|---|
| `autosorter/__init__.py` | Package marker + version string |
| `autosorter/__main__.py` | Entry point (creates Qt app + window) |
| `autosorter/paths.py` | Locates model files (works in dev and .exe) |
| `autosorter/app/__init__.py` | Package marker for app/ |
| `autosorter/app/main_window.py` | Empty placeholder window |
| `autosorter/models/*.onnx` | AI model files |
| `autosorter/models/LICENSE-*.txt` | Model licenses |
| `tests/test_m0_scaffold.py` | Smoke tests: models exist, load, run |
| `requirements.txt` | Package dependencies |
| `pyproject.toml` | Project config + pytest settings |
| `.gitignore` | Files git should ignore |

---

## ✅ How to Verify M0 Works

```powershell
# Tests pass
.\.venv\Scripts\python -m pytest tests/test_m0_scaffold.py -v

# App opens (empty window)
.\.venv\Scripts\python -m autosorter
```
