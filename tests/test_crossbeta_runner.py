"""Cross-Beta runner and batch integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.predictors.crossbeta import CrossBetaParser
from aggressor_wrappers.runners.crossbeta import CrossBetaRunner
from aggressor_wrappers.runners.registry import get_runner, list_runners

FIXTURES = Path(__file__).parent / "fixtures"
API_JSON = FIXTURES / "crossbeta_api.json"
LEGACY_JSON = Path(__file__).resolve().parents[2].parent / (
    "RPL27 and RPL36/Cross-beta predictor/RPL27.json"
)
BHT_RPL27 = Path(__file__).resolve().parents[2].parent / "BHT_amyloid/all/RPL27_human_all.csv"


def test_crossbeta_runner_registered() -> None:
    assert "crossbeta" in list_runners()


def test_get_runner_crossbeta_from_config() -> None:
    runner = get_runner("crossbeta")
    assert isinstance(runner, CrossBetaRunner)
    assert runner.threshold == pytest.approx(0.54)
    assert runner.window_size == "auto"
    assert runner.confidence_threshold == pytest.approx(0.54)


def test_crossbeta_parser_accepts_api_list_format() -> None:
    if not API_JSON.is_file():
        pytest.skip("crossbeta API fixture missing")
    result = CrossBetaParser().parse(API_JSON, protein_id="t", sequence="MADKG")
    assert result.scores[1] == pytest.approx(0.6)
    assert result.binary[3] == 1


@pytest.mark.skipif(not LEGACY_JSON.is_file(), reason="RPL27 legacy JSON not available")
def test_legacy_rpl27_bins_match_bht_reference_at_054() -> None:
    import pandas as pd

    legacy = json.loads(LEGACY_JSON.read_text())
    key = next(iter(legacy))
    sequence = "".join(item["amino_acid"] for item in legacy[key][0]["AA_list"])
    ref = pd.read_csv(BHT_RPL27)
    result = CrossBetaParser(confidence_threshold=0.54).parse(
        LEGACY_JSON, protein_id="RPL27", sequence=sequence
    )
    ref_bins = ref["cross-beta-predictor_bin"].astype(int).tolist()
    assert result.binary == ref_bins


def test_batch_pipeline_skip_run_crossbeta(tmp_path: Path) -> None:
    if not API_JSON.is_file():
        pytest.skip("crossbeta API fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(">APP\nMADKG\n")

    out = tmp_path / "results"
    work = out / "cross-beta-predictor" / "work" / "batch_1"
    work.mkdir(parents=True)
    (work / "APP_crossbeta.json").write_text(API_JSON.read_text())

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["crossbeta"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"APP"}
    assert (out / "cross-beta-predictor" / "parsed" / "APP_cross-beta-predictor.csv").is_file()
    assert any("[Cross-Beta]" in line for line in logs)
