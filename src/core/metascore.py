"""Metascore computation."""

from __future__ import annotations

import warnings

import pandas as pd

from aggressor_wrappers.core.config import AppConfig, load_config
from aggressor_wrappers.core.schema import get_predictor_spec


def compute_weighted_metascore(
    wide_df: pd.DataFrame,
    *,
    config: AppConfig | None = None,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Linear weighted sum of predictor score columns.

    ``wide_df`` must contain ``{Predictor}_score`` columns listed in weights.
    Missing predictors are skipped with a warning; remaining weights are
    renormalised so the sum stays on a comparable scale.
    """
    cfg = config or load_config()
    active_weights = weights or cfg.metascore.weights
    if not active_weights:
        raise ValueError("No metascore weights configured")

    if cfg.metascore.method != "weighted_sum":
        raise NotImplementedError(f"Metascore method not implemented: {cfg.metascore.method}")

    total = pd.Series(0.0, index=wide_df.index)
    used = 0.0
    missing: list[str] = []
    for key, weight in active_weights.items():
        spec = get_predictor_spec(key)
        col = spec.score_column
        if col not in wide_df.columns:
            missing.append(col)
            continue
        total = total + wide_df[col].astype(float) * weight
        used += weight

    if used == 0:
        raise ValueError("None of the configured weight columns are present in wide_df")

    if missing:
        warnings.warn(
            f"metascore: missing score column(s) {missing}; "
            f"renormalising over remaining weight sum {used:.4f}",
            UserWarning,
            stacklevel=2,
        )

    if used < 0.999:
        total = total / used

    return total


def metascore_table(
    wide_df: pd.DataFrame,
    *,
    column_name: str = "metascore",
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Attach a metascore column to a wide merge table."""
    out = wide_df.copy()
    out[column_name] = compute_weighted_metascore(out, config=config)
    return out
