#!/usr/bin/env python3
"""
Estimate metascore weights via least squares.

Usage:
  python scripts/calibrate_weights.py \\
      --merged-csv merged/RPS2_merged.csv \\
      --metascore-csv ../BHT_amyloid/metascores/RPS2_metascore_table.csv

Prints a config snippet for config.cfg to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY
from aggressor_wrappers.core.validate import load_reference_table


def main() -> int:
    args = parse_args()
    merged_df = load_reference_table(args.merged_csv)
    meta = pd.read_csv(args.metascore_csv)

    wt_col = next(c for c in meta.columns if "metascore" in c.lower() and "wt" in c.lower())
    y = meta[wt_col].values[: len(merged_df)]

    keys: list[str] = []
    cols: list[str] = []
    for key, spec in PREDICTOR_REGISTRY.items():
        if spec.score_column in merged_df.columns:
            keys.append(key)
            cols.append(spec.score_column)

    x = merged_df[cols].values
    weights, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    weights = np.maximum(weights, 0)
    if weights.sum() == 0:
        raise SystemExit("Regression produced zero weights")
    weights = weights / weights.sum()

    print("# Paste into config.cfg under [metascore.presets.<name>]")
    for key, weight in zip(keys, weights, strict=True):
        print(f"{key} = {weight:.6f}")
    pred = x @ weights
    rms = float(np.sqrt(np.mean((pred - y) ** 2)))
    print(f"# RMS vs {wt_col}: {rms:.6f}", file=__import__("sys").stderr)
    return 0


def load_merged_scores(path: Path) -> pd.DataFrame:
    """Load wide merged CSV or historical BHT reference table."""
    return load_reference_table(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-csv",
        type=Path,
        required=True,
        help="Wide merged CSV (position, aa_name, *_score columns)",
    )
    parser.add_argument("--metascore-csv", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
