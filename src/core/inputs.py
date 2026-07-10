"""Discover and classify standard CSV inputs for merge."""

from __future__ import annotations

import re
from pathlib import Path

from aggressor_wrappers.core.schema import resolve_predictor_key

# Registry keys used for filename inference (longest first for suffix matching).
_FILENAME_PREDICTORS = (
    "crossbeta",
    "archcandy",
    "aggreprot",
    "aggrescan",
    "appnn",
    "pasta",
    "waltz",
    "path",
)

# Short/generic tokens that must appear as a filename suffix (e.g. ``*_PATH.csv``).
_SUFFIX_ONLY_TOKENS = frozenset({"path"})

# Multi-token aliases checked as substrings before token splitting.
_MULTI_TOKEN_ALIASES = (
    ("cross-beta-predictor", "crossbeta"),
    ("cross-beta", "crossbeta"),
    ("cross_beta", "crossbeta"),
    ("arch-candy", "archcandy"),
)


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
    """
    Infer predictor registry key from filename.

    Prefers suffix matches (``RPS2_PATH.csv``, ``foo_waltz.csv``) and multi-token
    aliases (``cross-beta``). Ambiguous multi-token names raise ``ValueError``.
    """
    stem = path.stem.lower()

    for needle, key in _MULTI_TOKEN_ALIASES:
        if needle in stem:
            return resolve_predictor_key(key)

    # Prefer explicit suffix: ``*_path``, ``*-waltz``, etc. (longest keys first).
    for key in sorted(_FILENAME_PREDICTORS, key=len, reverse=True):
        if stem.endswith(f"_{key}") or stem.endswith(f"-{key}"):
            return resolve_predictor_key(key)

    tokens = set(re.split(r"[_\-.]+", stem))
    matches = [
        key
        for key in _FILENAME_PREDICTORS
        if key in tokens and key not in _SUFFIX_ONLY_TOKENS
    ]
    if len(matches) == 1:
        return resolve_predictor_key(matches[0])
    if len(matches) > 1:
        matches.sort(key=len, reverse=True)
        if len(matches[0]) > len(matches[1]):
            return resolve_predictor_key(matches[0])
        raise ValueError(
            f"Ambiguous predictor in filename {path.name!r}: {matches}. "
            "Pass --predictor explicitly."
        )

    raise ValueError(f"Cannot infer predictor from filename: {path.name}")
