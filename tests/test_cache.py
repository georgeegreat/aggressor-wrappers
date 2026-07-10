"""Tests for raw-output cache."""

from __future__ import annotations

from pathlib import Path

from aggressor_wrappers.core.cache import clear_cache_dir, raw_cache_path, store_raw_cache
from aggressor_wrappers.core.config import AppConfig, CacheConfig, MetascoreConfig


def test_clear_cache_dir(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "protein" / "waltz").mkdir(parents=True)
    cfg = AppConfig(cache=CacheConfig(root=str(root), enabled=True))

    removed = clear_cache_dir(cfg)
    assert removed == root
    assert not root.exists()


def test_store_raw_cache(tmp_path: Path) -> None:
    source = tmp_path / "input.dat"
    source.write_text("1\t0.5\n")
    cfg = AppConfig(cache=CacheConfig(root=str(tmp_path / "cache"), enabled=True))

    dest = store_raw_cache("RPS2_human", "waltz", source, config=cfg)
    assert dest is not None
    assert dest.is_file()
    assert dest.read_text() == source.read_text()
    assert dest == raw_cache_path("RPS2_human", "waltz", source, config=cfg, cache_root=cfg.cache.root)
