# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification file for Auto-Sorter.
Builds a standalone onedir distribution for Windows with zero external runtime dependencies.
"""

import sys
from pathlib import Path

block_cipher = None

# Base repository root
REPO_ROOT = Path(SPECPATH).parent.resolve()

datas = [
    (str(REPO_ROOT / "autosorter" / "models" / "*"), "autosorter/models"),
    (str(REPO_ROOT / "autosorter" / "app.png"), "autosorter"),
    (str(REPO_ROOT / "packaging" / "app.ico"), "packaging"),
]

hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "rawpy",
    "pillow_heif",
    "sqlite3",
    "hashlib",
    "json",
    "math",
    "shutil",
    "concurrent.futures",
    "autosorter",
    "autosorter.paths",
    "autosorter.core.db",
    "autosorter.core.scanner",
    "autosorter.core.thumbnailer",
    "autosorter.core.face_detect",
    "autosorter.core.face_embed",
    "autosorter.core.graph_cluster",
    "autosorter.core.pipeline_worker",
    "autosorter.core.export_worker",
    "autosorter.core.perf",
    "autosorter.app.main_window",
    "autosorter.app.people_panel",
    "autosorter.app.photo_viewer",
    "autosorter.app.similar_panel",
    "autosorter.app.albums_panel",
    "autosorter.app.photo_grid",
]

excludes = [
    "pytest",
    "_pytest",
    "unittest",
    "tkinter",
    "matplotlib",
    "scipy",
    "IPython",
]

a = Analysis(
    [str(REPO_ROOT / "autosorter" / "__main__.py")],
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

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoSorter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Avoid UPX to prevent antivirus false positives
    console=False,  # Windows GUI application (no command prompt window)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(REPO_ROOT / "packaging" / "app.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoSorter",
)
