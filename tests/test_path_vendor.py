"""Tests for bundled PATH vendor paths."""

from __future__ import annotations

from aggressor_wrappers.paths import DEFAULT_PATH_SCRIPT, PATH_VENDOR_ROOT


def test_bundled_path_script_exists() -> None:
    assert PATH_VENDOR_ROOT.is_dir()
    assert DEFAULT_PATH_SCRIPT.is_file()
    assert DEFAULT_PATH_SCRIPT.name == "path1.1.py"
