"""
Automated packaging and installer build script for Auto-Sorter.

Workflow:
1. Ensure application icon is generated in packaging/
2. Build standalone onedir bundle using PyInstaller
3. Verify bundled executable and model files
4. Compile Inno Setup installer using ISCC
5. Print build artifacts summary
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_iscc() -> Path | None:
    """Find Inno Setup Compiler executable."""
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
    ]
    # Check PATH
    which_iscc = shutil.which("iscc")
    if which_iscc:
        candidates.insert(0, Path(which_iscc))

    for c in candidates:
        if c.is_file():
            return c
    return None


def run_build(skip_installer: bool = False) -> int:
    repo_root = Path(__file__).parent.parent.resolve()
    packaging_dir = repo_root / "packaging"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"

    print("=" * 60)
    print("Auto-Sorter M9 Build Pipeline")
    print("=" * 60)

    # Step 1: Ensure icons exist
    ico_path = packaging_dir / "app.ico"
    if not ico_path.exists():
        print("\n[1/4] Generating application icons...")
        from packaging.generate_icon import generate_app_icon
        generate_app_icon(packaging_dir)
    else:
        print("\n[1/4] Application icons already exist.")

    # Ensure app.png in autosorter package
    app_png = repo_root / "autosorter" / "app.png"
    if not app_png.exists():
        shutil.copy2(packaging_dir / "app.png", app_png)

    # Step 2: Run PyInstaller
    print("\n[2/4] Running PyInstaller build...")
    spec_file = packaging_dir / "autosorter.spec"
    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec_file),
    ]

    ret = subprocess.run(pyinstaller_cmd, cwd=str(repo_root))
    if ret.returncode != 0:
        print("ERROR: PyInstaller build failed!")
        return ret.returncode

    # Step 3: Verify Bundle Integrity
    print("\n[3/4] Verifying PyInstaller bundle integrity...")
    bundle_dir = dist_dir / "AutoSorter"
    exe_path = bundle_dir / "AutoSorter.exe"
    if not exe_path.is_file():
        print(f"ERROR: Executable not found at {exe_path}!")
        return 1

    # Search for model files in bundle
    found_yunet = list(bundle_dir.glob("**/face_detection_yunet_2023mar.onnx"))
    found_sface = list(bundle_dir.glob("**/face_recognition_sface_2021dec.onnx"))

    if not found_yunet:
        print("ERROR: YuNet model not found in bundle!")
        return 1
    if not found_sface:
        print("ERROR: SFace model not found in bundle!")
        return 1

    print(f"  [OK] Executable found: {exe_path.name} ({exe_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  [OK] YuNet model verified at: {found_yunet[0].relative_to(bundle_dir)}")
    print(f"  [OK] SFace model verified at: {found_sface[0].relative_to(bundle_dir)}")

    # Step 4: Build Inno Setup Installer
    if skip_installer:
        print("\n[4/4] Skipping Inno Setup installer (--skip-installer specified).")
        return 0

    print("\n[4/4] Compiling Inno Setup installer...")
    iscc_path = find_iscc()
    if not iscc_path:
        print("WARNING: ISCC.exe (Inno Setup Compiler) not found in standard paths.")
        print("Please install Inno Setup 6 to compile the installer.")
        return 0

    print(f"  Found ISCC at: {iscc_path}")
    iss_file = packaging_dir / "installer.iss"
    iscc_cmd = [str(iscc_path), str(iss_file)]

    ret = subprocess.run(iscc_cmd, cwd=str(packaging_dir))
    if ret.returncode != 0:
        print("ERROR: Inno Setup compilation failed!")
        return ret.returncode

    installer_exe = dist_dir / "installer" / "AutoSorter-Setup-0.1.0.exe"
    if installer_exe.is_file():
        size_mb = installer_exe.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"SUCCESS: Installer created successfully!")
        print(f"Output: {installer_exe}")
        print(f"Size:   {size_mb:.2f} MB")
        print("=" * 60)
    else:
        print("WARNING: Installer file not found at expected location.")

    return 0


if __name__ == "__main__":
    skip = "--skip-installer" in sys.argv
    sys.exit(run_build(skip_installer=skip))
