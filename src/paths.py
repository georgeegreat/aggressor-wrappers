"""Package root paths (config, legacy scripts)."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PACKAGE_ROOT / "legacy"
PATH_VENDOR_ROOT = PACKAGE_ROOT / "vendor" / "PATH"
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.cfg"
DEFAULT_APPNN_SCRIPT = LEGACY_ROOT / "appnn_converter.R"
DEFAULT_PATH_SCRIPT = PATH_VENDOR_ROOT / "path1.1.py"


def bht_reference_root() -> Path:
    """
    Locate the BHT_amyloid repository root for reference CSVs.

    Works when ``aggressor_wrappers`` lives inside ``BHT_amyloid/`` or as a
    sibling directory under ``Amyloids_data/``.
    """
    parent = PACKAGE_ROOT.parent
    if (parent / "all").is_dir():
        return parent
    sibling = parent / "BHT_amyloid"
    if (sibling / "all").is_dir():
        return sibling
    return sibling
