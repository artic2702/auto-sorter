# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for Auto-Sorter's standalone Python RPC sidecar (`autosorter-engine.exe`).

Bundled inside the Electron installer's `resources/engine/` directory so end users
do NOT need Python, OpenCV, NumPy, Pillow, or ONNX models installed on their PC.
"""

from pathlib import Path

block_cipher = None

REPO_ROOT = Path(SPECPATH).parent.parent.resolve()

datas = [
    (str(REPO_ROOT / "autosorter" / "models" / "*"), "autosorter/models"),
    (str(REPO_ROOT / "packaging" / "app.png"), "autosorter"),
]

hiddenimports = [
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ExifTags",
    "sqlite3",
    "ctypes",
    "ctypes.wintypes",
    "hashlib",
    "json",
    "math",
    "shutil",
    "concurrent.futures",
    "logging.handlers",
    "autosorter",
    "autosorter.paths",
    "autosorter.bridge",
    "autosorter.core.engine",
    "autosorter.core.db",
    "autosorter.core.scanner",
    "autosorter.core.thumbnailer",
    "autosorter.core.face_detect",
    "autosorter.core.face_embed",
    "autosorter.core.face_crop",
    "autosorter.core.graph_cluster",
    "autosorter.core.cluster",
    "autosorter.core.doc_classifier",
    "autosorter.core.export",
    "autosorter.core.bounded",
    "autosorter.core.perf",
]

excludes = [
    "PySide6",
    "shiboken6",
    "PyQt5",
    "PyQt6",
    "pytest",
    "_pytest",
    "unittest",
    "tkinter",
    "matplotlib",
    "scipy",
    "IPython",
]

a = Analysis(
    [str(REPO_ROOT / "autosorter" / "bridge.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Strip unused OpenCV FFmpeg video codec DLL (~28 MB; Auto-Sorter processes still images only)
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0].lower()]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Note: `console=True` ensures Win32 stdin/stdout pipes are initialized cleanly;
# Electron spawns `autosorter-engine.exe` with `windowsHide: true` (CREATE_NO_WINDOW)
# so no console window ever appears on the user's screen.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="autosorter-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(REPO_ROOT / "electron" / "packaging" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="autosorter-engine",
)
