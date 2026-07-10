"""Cache raw predictor outputs for reproducibility."""

from __future__ import annotations

import shutil
from pathlib import Path

from aggressor_wrappers.core.config import AppConfig, load_config


def cache_dir(config: AppConfig | None = None, override: str | Path | None = None) -> Path:
    cfg = config or load_config()
    root = Path(override) if override is not None else Path(cfg.cache.root)
    return root.expanduser()


def raw_cache_path(
    protein_id: str,
    predictor: str,
    source: str | Path,
    *,
    config: AppConfig | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """Return ``{root}/{protein_id}/{predictor}/raw.{ext}``."""
    root = cache_dir(config, cache_root)
    ext = Path(source).suffix or ".bin"
    safe_id = _safe_segment(protein_id)
    safe_predictor = _safe_segment(predictor)
    return root / safe_id / safe_predictor / f"raw{ext}"


def store_raw_cache(
    protein_id: str,
    predictor: str,
    source: str | Path,
    *,
    config: AppConfig | None = None,
    cache_root: str | Path | None = None,
    force: bool = False,
) -> Path | None:
    """
    Copy ``source`` into the cache tree.

    Returns destination path, or ``None`` if caching is disabled.
    """
    cfg = config or load_config()
    if not force and not cfg.cache.enabled:
        return None

    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot cache missing file: {source}")

    dest = raw_cache_path(
        protein_id,
        predictor,
        source,
        config=cfg,
        cache_root=cache_root,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def clear_cache_dir(
    config: AppConfig | None = None,
    cache_root: str | Path | None = None,
) -> Path | None:
    """
    Remove the cache root directory if it looks like an aggressor cache tree.

    Refuses to delete a non-empty directory that does not contain
    ``{protein_id}/{predictor}/raw*`` entries, to avoid wiping an unrelated
    ``cache/`` folder in the working directory.
    """
    root = cache_dir(config, cache_root)
    if not root.is_dir():
        return None
    if any(root.iterdir()) and not _looks_like_aggressor_cache(root):
        return None
    shutil.rmtree(root)
    return root


def _looks_like_aggressor_cache(root: Path) -> bool:
    """True when ``root`` contains at least one ``*/*/raw*`` cache entry."""
    try:
        for protein_dir in root.iterdir():
            if not protein_dir.is_dir():
                continue
            for pred_dir in protein_dir.iterdir():
                if not pred_dir.is_dir():
                    continue
                if any(pred_dir.glob("raw*")):
                    return True
    except OSError:
        return False
    return False


def _safe_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned or "protein"
