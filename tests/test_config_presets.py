"""Tests for metascore weight presets and widemerge reference validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.core.validate import compare_score_columns, load_reference_table
from aggressor_wrappers.paths import bht_reference_root

FIXTURES = Path(__file__).parent / "fixtures"


def test_functional_amyloid_preset_weights() -> None:
    cfg = load_config()
    weights = cfg.metascore.presets["functional_amyloids"]
    assert weights["waltz"] == pytest.approx(0.24)
    assert weights["path"] == pytest.approx(0.22)
    assert weights["appnn"] == pytest.approx(0.20)
    assert weights["crossbeta"] == pytest.approx(0.10)
    assert weights["pasta"] == pytest.approx(0.16)
    assert weights["archcandy"] == pytest.approx(0.05)
    assert weights["aggrescan"] == pytest.approx(0.03)
    assert "aggreprot" not in weights


def test_pathogenic_amyloid_preset_weights() -> None:
    weights = load_config().metascore.presets["pathogenic_amyloids"]
    assert weights["crossbeta"] == pytest.approx(0.20)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_widemerge_reference_self_compare(tmp_path: Path) -> None:
    ref_path = bht_reference_root() / "all" / "RPS2_human_all.csv"
    if not ref_path.is_file():
        pytest.skip("BHT reference CSV not available")

    ref = load_reference_table(ref_path)
    out = tmp_path / "merged.csv"
    ref.to_csv(out, index=False)

    mismatches = compare_score_columns(ref, ref)
    assert mismatches == {}


def test_runner_timeout_zero_from_config() -> None:
    cfg = load_config()
    assert cfg.runners["path"]["timeout_seconds"] == 0
    assert cfg.runners["appnn"]["timeout_seconds"] == 0
    assert cfg.cache.enabled is False


def test_runner_batch_config_defaults() -> None:
    from aggressor_wrappers.core.config import runner_batch_config

    path_cfg = runner_batch_config("path")
    assert path_cfg.parallel_jobs == 2
    assert path_cfg.sequences_per_run == 1

    appnn_cfg = runner_batch_config("appnn")
    assert appnn_cfg.parallel_jobs == 1
    assert appnn_cfg.sequences_per_run == 0

    waltz_cfg = runner_batch_config("waltz")
    assert waltz_cfg.sequences_per_run == 10

    pasta_cfg = runner_batch_config("pasta")
    assert pasta_cfg.sequences_per_run == 10

    archcandy_cfg = runner_batch_config("archcandy")
    assert archcandy_cfg.sequences_per_run == 1
    assert archcandy_cfg.parallel_jobs == 2

    crossbeta_cfg = runner_batch_config("crossbeta")
    assert crossbeta_cfg.sequences_per_run == 1
    assert crossbeta_cfg.parallel_jobs == 2

    aggreprot_cfg = runner_batch_config("aggreprot")
    assert aggreprot_cfg.sequences_per_run == 3


def test_pipeline_predictors_from_config() -> None:
    from aggressor_wrappers.core.config import default_pipeline_predictors

    assert default_pipeline_predictors() == [
        "path",
        "appnn",
        "waltz",
        "pasta",
        "archcandy",
        "crossbeta",
        "aggreprot",
    ]


def test_guess_predictor_import() -> None:
    from aggressor_wrappers.core.inputs import guess_predictor_from_filename

    assert guess_predictor_from_filename(Path("RPS2_PATH.csv")) == "path"
    assert guess_predictor_from_filename(Path("RPL27_crossbeta.csv")) == "crossbeta"
    assert guess_predictor_from_filename(Path("foo_cross-beta-predictor.csv")) == "crossbeta"
    assert guess_predictor_from_filename(Path("sample_waltz.csv")) == "waltz"

    with pytest.raises(ValueError, match="Cannot infer"):
        guess_predictor_from_filename(Path("hot_path_tracker.csv"))

    # Suffix wins when one predictor is the filename ending.
    assert guess_predictor_from_filename(Path("foo_path_waltz.csv")) == "waltz"

    with pytest.raises(ValueError, match="Ambiguous"):
        guess_predictor_from_filename(Path("pasta_waltz_sample.csv"))


def test_get_runner_wires_path_and_appnn_thresholds() -> None:
    from aggressor_wrappers.runners.registry import get_runner

    path_runner = get_runner("path")
    assert path_runner.threshold_percentile == pytest.approx(75.0)

    appnn_runner = get_runner("appnn")
    assert appnn_runner.score_threshold == pytest.approx(0.5)


def test_normalise_predictors_accepts_aliases() -> None:
    from aggressor_wrappers.batch.pipeline import _normalise_predictors

    assert _normalise_predictors(["cross-beta", "ArchCandy", "PATH"]) == [
        "crossbeta",
        "archcandy",
        "path",
    ]
