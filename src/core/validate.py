"""Validation helpers for merged predictor tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_reference_table(path: str | Path) -> pd.DataFrame:
    """Load wide or historical BHT ``*_all.csv`` reference."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    use_index = "position" not in header
    return pd.read_csv(path, index_col=0 if use_index else None)


def compare_score_columns(
    merged: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> dict[str, float]:
    """
    Compare ``*_score`` columns present in both frames.

    Returns per-column max absolute difference. Empty dict means all match.
    """
    if len(merged) != len(reference):
        raise ValueError(
            f"Row count mismatch: merged={len(merged)} reference={len(reference)}"
        )

    mismatches: dict[str, float] = {}
    ref_score_cols = [c for c in reference.columns if c.endswith("_score")]
    for col in ref_score_cols:
        if col not in merged.columns:
            mismatches[col] = float("nan")
            continue
        merged_vals = merged[col].values
        ref_vals = reference[col].values
        if not np.allclose(merged_vals, ref_vals, rtol=rtol, atol=atol):
            mismatches[col] = float(np.max(np.abs(merged_vals - ref_vals)))
    return mismatches
