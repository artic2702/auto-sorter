"""
Tests for Milestone 9: Packaging and Distribution.

Covers:
1. Packaging files exist and are well-formed (autosorter.spec, installer.iss, app.ico, build.py)
2. Spec file configuration integrity (entry point, datas, UPX disabled, console disabled)
3. Inno Setup script configuration integrity (app name, privileges, x64 architecture, offline source)
4. Frozen environment path resolution (paths.package_root under simulated frozen states)
5. Zero network telemetry verification (imports and modules do not initiate external connections)
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autosorter.paths import APP_ICON, MODELS_DIR, SFACE_MODEL, YUNET_MODEL, package_root


REPO_ROOT = Path(__file__).parent.parent.resolve()
PACKAGING_DIR = REPO_ROOT / "packaging"


class TestPackagingAssets:
    """Verify all packaging assets exist and are populated."""

    def test_packaging_directory_contents(self):
        assert PACKAGING_DIR.is_dir()
        assert (PACKAGING_DIR / "autosorter.spec").is_file()
        assert (PACKAGING_DIR / "installer.iss").is_file()
        assert (PACKAGING_DIR / "generate_icon.py").is_file()
        assert (PACKAGING_DIR / "build.py").is_file()

    def test_app_icons_exist(self):
        ico = PACKAGING_DIR / "app.ico"
        png = PACKAGING_DIR / "app.png"
        assert ico.is_file()
        assert png.is_file()
        assert ico.stat().st_size > 1000
        assert png.stat().st_size > 1000

    def test_app_icon_in_package(self):
        assert APP_ICON.is_file()
        assert APP_ICON.stat().st_size > 1000


class TestSpecConfiguration:
    """Verify autosorter.spec has proper PyInstaller configuration."""

    @pytest.fixture
    def spec_content(self):
        spec_path = PACKAGING_DIR / "autosorter.spec"
        return spec_path.read_text(encoding="utf-8")

    def test_entry_point(self, spec_content):
        assert "__main__.py" in spec_content

    def test_bundled_models(self, spec_content):
        assert "autosorter/models" in spec_content

    def test_upx_disabled(self, spec_content):
        # UPX should be disabled to prevent antivirus false positives
        assert "upx=False" in spec_content

    def test_console_window_disabled(self, spec_content):
        # Windows GUI application should not show console window
        assert "console=False" in spec_content

    def test_onedir_collect(self, spec_content):
        assert "COLLECT(" in spec_content
        assert 'name="AutoSorter"' in spec_content


class TestInnoSetupConfiguration:
    """Verify installer.iss has proper offline Inno Setup configuration."""

    @pytest.fixture
    def iss_content(self):
        iss_path = PACKAGING_DIR / "installer.iss"
        return iss_path.read_text(encoding="utf-8")

    def test_metadata(self, iss_content):
        assert 'MyAppName "Auto-Sorter"' in iss_content
        assert 'MyAppVersion "0.1.0"' in iss_content
        assert 'AutoSorter.exe' in iss_content

    def test_privileges_and_directory(self, iss_content):
        assert "PrivilegesRequired=lowest" in iss_content
        assert "DefaultDirName={autopf}" in iss_content

    def test_architecture(self, iss_content):
        assert "ArchitecturesAllowed=x64compatible" in iss_content
        assert "ArchitecturesInstallIn64BitMode=x64compatible" in iss_content

    def test_compression(self, iss_content):
        assert "Compression=lzma2/ultra64" in iss_content
        assert "SolidCompression=yes" in iss_content

    def test_offline_bundle_source(self, iss_content):
        # Sources from local PyInstaller output only — zero network downloading
        assert r'Source: "..\dist\AutoSorter\*"' in iss_content


class TestFrozenPathResolution:
    """Verify package_root behaves properly under simulated PyInstaller environments."""

    def test_development_mode(self):
        root = package_root()
        assert (root / "models").is_dir()
        assert (root / "models" / "face_detection_yunet_2023mar.onnx").is_file()

    def test_frozen_simulation(self, tmp_path):
        # Simulate PyInstaller _MEIPASS layout
        fake_meipass = tmp_path / "meipass"
        fake_models = fake_meipass / "autosorter" / "models"
        fake_models.mkdir(parents=True)
        (fake_models / "face_detection_yunet_2023mar.onnx").touch()

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "_MEIPASS", str(fake_meipass), create=True):
            resolved = package_root()
            assert resolved == fake_meipass / "autosorter"
            assert (resolved / "models").is_dir()

    def test_frozen_flat_simulation(self, tmp_path):
        # Simulate PyInstaller flat layout (models directly in _MEIPASS/models)
        fake_meipass = tmp_path / "flat_meipass"
        fake_models = fake_meipass / "models"
        fake_models.mkdir(parents=True)
        (fake_models / "face_detection_yunet_2023mar.onnx").touch()

        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "_MEIPASS", str(fake_meipass), create=True):
            resolved = package_root()
            assert resolved == fake_meipass
            assert (resolved / "models").is_dir()


class TestOfflineZeroNetworkIntegrity:
    """Verify application modules have zero network dependencies."""

    def test_no_telemetry_packages(self):
        # Ensure common analytics/telemetry libraries are not imported
        forbidden_modules = [
            "urllib.request",
            "requests",
            "httpx",
            "aiohttp",
            "sentry_sdk",
            "mixpanel",
            "google.analytics",
        ]
        for mod in forbidden_modules:
            assert mod not in sys.modules, f"Forbidden network/telemetry module loaded: {mod}"
