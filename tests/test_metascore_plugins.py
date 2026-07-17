"""Pluggable metascore methods: registry, polarity, and correctness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.core.metascore_plugins import (
    available_methods,
    compute_metascore,
    polarity,
    register_metascore,
)


def _wide() -> pd.DataFrame:
    hot = np.array([False] * 10 + [True] * 10)
    df = pd.DataFrame(
        {
            "waltz_score": np.where(hot, 90.0, 20.0),
            "APPNN_score": np.where(hot, 0.9, 0.1),
            # PASTA: inverted — lower (more negative) is more amyloidogenic
            "pasta_score": np.where(hot, -7.0, -1.0),
        }
    )
    df["waltz_bin"] = (df.waltz_score >= 73).astype(int)
    df["APPNN_bin"] = (df.APPNN_score >= 0.5).astype(int)
    df["pasta_bin"] = (df.pasta_score < -2.8).astype(int)
    return df


def test_pasta_polarity_is_declared_inverted():
    assert polarity("pasta") == -1
    assert polarity("waltz") == 1


def test_builtin_methods_are_registered():
    assert available_methods() == ["fractional_consensus", "zscore_consensus"]


def test_default_config_uses_zscore_consensus():
    assert load_config().metascore.method == "zscore_consensus"


def test_corrected_methods_rank_amyloidogenic_residues_higher():
    df, cfg = _wide(), load_config()
    for method in ("zscore_consensus", "fractional_consensus"):
        meta = compute_metascore(df, config=cfg, method=method)
        assert meta[10:].mean() > meta[:10].mean(), method


def test_fractional_consensus_is_a_fraction():
    df, cfg = _wide(), load_config()
    meta = compute_metascore(df, config=cfg, method="fractional_consensus")
    assert meta.min() >= 0.0 and meta.max() <= 1.0


def test_external_method_can_be_registered():
    """An external consensus plugs in without editing this package."""

    @register_metascore("_test_external")
    def _external(wide_df, *, config, weights=None):  # noqa: ARG001
        bins = [c for c in wide_df.columns if c.endswith("_bin")]
        return wide_df[bins].sum(axis=1) / len(bins)

    df, cfg = _wide(), load_config()
    meta = compute_metascore(df, config=cfg, method="_test_external")
    assert "_test_external" in available_methods()
    assert meta[10:].mean() > meta[:10].mean()


def test_unknown_method_raises_with_available_list():
    with pytest.raises(NotImplementedError, match="not registered"):
        compute_metascore(_wide(), config=load_config(), method="nope")
