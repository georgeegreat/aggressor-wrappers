"""Discover and classify standard CSV inputs for merge."""

from __future__ import annotations

import re
from pathlib import Path

from aggressor_wrappers.core.schema import resolve_predictor_key


def expand_csv_inputs(items: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("No CSV inputs found")
    return paths


def guess_predictor_from_filename(path: Path) -> str:
    """Infer predictor registry key from filename tokens."""
    stem = path.stem.lower()
    tokens = set(re.split(r"[_\-.]+", stem))

    token_map = {
        "crossbeta": "crossbeta",
        "archcandy": "archcandy",
        "aggreprot": "aggreprot",
        "aggrescan": "aggrescan",
        "appnn": "appnn",
        "pasta": "pasta",
        "path": "path",
        "waltz": "waltz",
    }
    for token, key in token_map.items():
        if token in tokens:
            return resolve_predictor_key(key)

    if "cross-beta-predictor" in stem or "cross-beta" in stem:
        return resolve_predictor_key("crossbeta")

    raise ValueError(f"Cannot infer predictor from filename: {path.name}")
