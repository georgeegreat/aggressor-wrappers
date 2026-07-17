"""Pluggable metascore methods.

``[metascore] method`` in ``config.cfg`` selects a registered combiner from
``METASCORE_REGISTRY``. Two built-ins ship with the package:

* ``zscore_consensus`` — standardise score columns, apply polarity, combine with
  preset weights (after standardisation the weights express model confidence).
* ``fractional_consensus`` — weighted mean of ``{predictor}_bin`` columns using
  the same preset weights (renormalised over predictors present).

External consensus functions (e.g. amyloscope-backed tiering) can register via
``@register_metascore`` without editing pipeline code.

Why not a raw weighted sum of score columns? The panel's scales are
incommensurable (WALTZ/PATH 0–100 vs APPNN 0–1 vs PASTA free energy), and PASTA
is polarity-inverted at binarisation time but has no direction field on
``PredictorSpec``. ``zscore_consensus`` fixes both issues before applying weights.
"""

from __future__ import annotations

from typing import Callable, Protocol

import pandas as pd

from aggressor_wrappers.core.config import AppConfig, load_config
from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY, get_predictor_spec

# --------------------------------------------------------------------------- #
# Predictor polarity.
#
# This belongs in ``PredictorSpec`` (which has ``default_threshold`` but no
# direction). It is kept here so this module stays a purely additive overlay;
# fold it into schema.py when convenient.
#
# True  -> higher score = more amyloidogenic
# False -> lower  score = more amyloidogenic (PASTA pairing free energy)
# --------------------------------------------------------------------------- #
HIGHER_IS_AMYLOIDOGENIC: dict[str, bool] = {
    "aggreprot": True,
    "aggrescan": True,
    "appnn": True,
    "archcandy": True,
    "crossbeta": True,
    "pasta": False,
    "path": True,
    "waltz": True,
}


def polarity(key: str) -> int:
    """+1 if higher score means more amyloidogenic, -1 if inverted."""
    return 1 if HIGHER_IS_AMYLOIDOGENIC.get(key, True) else -1


class MetascoreFn(Protocol):
    def __call__(
        self,
        wide_df: pd.DataFrame,
        *,
        config: AppConfig,
        weights: dict[str, float] | None = None,
    ) -> pd.Series: ...


METASCORE_REGISTRY: dict[str, MetascoreFn] = {}


def register_metascore(name: str) -> Callable[[MetascoreFn], MetascoreFn]:
    """Register a metascore method under ``[metascore] method = <name>``.

    External code (e.g. an amyloscope-backed consensus) can register its own
    combiner without modifying this package::

        from aggressor_wrappers.core.metascore_plugins import register_metascore

        @register_metascore("amyloscope_consensus")
        def my_consensus(wide_df, *, config, weights=None):
            ...
            return pd.Series(...)
    """

    def decorator(fn: MetascoreFn) -> MetascoreFn:
        METASCORE_REGISTRY[name] = fn
        return fn

    return decorator


def available_methods() -> list[str]:
    return sorted(METASCORE_REGISTRY)


def compute_metascore(
    wide_df: pd.DataFrame,
    *,
    config: AppConfig | None = None,
    weights: dict[str, float] | None = None,
    method: str | None = None,
) -> pd.Series:
    """Dispatch to the configured metascore method."""
    cfg = config or load_config()
    name = method or cfg.metascore.method
    if name not in METASCORE_REGISTRY:
        raise NotImplementedError(
            f"Metascore method {name!r} is not registered. "
            f"Available: {available_methods()}"
        )
    return METASCORE_REGISTRY[name](wide_df, config=cfg, weights=weights)


def _active_weights(cfg: AppConfig, weights: dict[str, float] | None) -> dict[str, float]:
    active = weights or cfg.metascore.weights
    if not active:
        raise ValueError("No metascore weights configured")
    return active


def _score_columns(wide_df: pd.DataFrame, keys) -> dict[str, str]:
    """Map predictor key -> score column, keeping only columns present."""
    out = {}
    for key in keys:
        col = get_predictor_spec(key).score_column
        if col in wide_df.columns:
            out[key] = col
    return out


# --------------------------------------------------------------------------- #
# Built-in methods
# --------------------------------------------------------------------------- #


@register_metascore("zscore_consensus")
def zscore_consensus(
    wide_df: pd.DataFrame,
    *,
    config: AppConfig,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Polarity-corrected, standardised weighted consensus.

    Each predictor's score column is standardised to zero mean and unit variance
    over the table, multiplied by its polarity (+1, or −1 for PASTA), then
    combined with weights from the active metascore preset (renormalised over
    predictors present in ``wide_df``).

    Degenerate (zero-variance) columns contribute nothing rather than producing
    NaNs — a tool that fired nowhere on this protein carries no information.
    """
    active = _active_weights(config, weights)
    cols = _score_columns(wide_df, active)
    if not cols:
        raise ValueError("None of the configured weight columns are present in wide_df")

    total = pd.Series(0.0, index=wide_df.index)
    used = 0.0
    for key, col in cols.items():
        series = wide_df[col].astype(float)
        sd = series.std()
        if not sd or pd.isna(sd):
            continue  # zero variance carries no signal
        z = (series - series.mean()) / sd
        total = total + z * polarity(key) * active[key]
        used += active[key]

    if used == 0:
        raise ValueError("All configured predictors had zero variance")
    return total / used


@register_metascore("fractional_consensus")
def fractional_consensus(
    wide_df: pd.DataFrame,
    *,
    config: AppConfig,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Fraction of the panel calling each residue amyloidogenic.

    Combines the per-tool ``{predictor}_bin`` columns rather than the raw
    scores. Those binary calls were produced by each tool's own parser using its
    own threshold *and* its own direction (``binary_from_scores(...,
    greater_or_equal=...)``), so this combiner is scale-free and
    polarity-correct by construction.

    With preset weights this is a confidence-weighted fraction on the 0–1 scale;
    weights are renormalised over ``{predictor}_bin`` columns present.
    """
    active = weights if weights is not None else config.metascore.weights
    keys = list(active) if active else list(PREDICTOR_REGISTRY)

    total = pd.Series(0.0, index=wide_df.index)
    used = 0.0
    for key in keys:
        col = get_predictor_spec(key).bin_column
        if col not in wide_df.columns:
            continue
        weight = float(active[key]) if active else 1.0
        total = total + wide_df[col].astype(float).clip(0, 1) * weight
        used += weight

    if used == 0:
        raise ValueError(
            "No {predictor}_bin columns present; fractional_consensus needs the "
            "binarised calls (run merge, or use zscore_consensus on raw scores)"
        )
    return total / used
