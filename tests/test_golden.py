"""Golden-file tests using reference tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggressor_wrappers.core.config import load_config, list_metascore_presets
from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.core.merge import merge_predictor_tables
from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY, PredictorResult
from aggressor_wrappers.core.validate import load_reference_table
from aggressor_wrappers.paths import bht_reference_root
from aggressor_wrappers.predictors.crossbeta import CrossBetaParser

DATA = Path(__file__).resolve().parents[2]
BHT = bht_reference_root()


@pytest.fixture
def rps2_sequence() -> tuple[str, str]:
    fasta = DATA / "RPS2.fasta"
    if not fasta.is_file():
        fasta = DATA.parent / "RPS2.fasta"
    if not fasta.is_file():
        pytest.skip("RPS2.fasta not available")
    return read_first_sequence(fasta)


def _reference_table_to_results(
    table: pd.DataFrame,
    *,
    protein_id: str,
    sequence: str,
) -> list[PredictorResult]:
    results: list[PredictorResult] = []
    for spec in PREDICTOR_REGISTRY.values():
        score_col, bin_col = spec.score_column, spec.bin_column
        if score_col not in table.columns or bin_col not in table.columns:
            continue
        results.append(
            PredictorResult(
                protein_id=protein_id,
                sequence=sequence,
                spec=spec,
                scores=[float(x) for x in table[score_col]],
                binary=[int(x) for x in table[bin_col]],
            )
        )
    return results


def test_golden_merge_roundtrip_rps2(rps2_sequence: tuple[str, str]) -> None:
    """Re-merge predictor columns from BHT reference → identical wide scores."""
    ref_path = BHT / "all" / "RPS2_human_all.csv"
    if not ref_path.is_file():
        pytest.skip("Reference all CSV not available")

    protein_id, sequence = rps2_sequence
    ref = load_reference_table(ref_path)
    assert len(ref) == len(sequence)

    results = _reference_table_to_results(ref, protein_id=protein_id, sequence=sequence)
    assert len(results) >= 7

    wide = merge_predictor_tables(results)
    score_cols = [c for c in ref.columns if c.endswith("_score")]
    for col in score_cols:
        np.testing.assert_allclose(
            wide[col].values,
            ref[col].values,
            rtol=1e-9,
            atol=1e-9,
            err_msg=col,
        )


def test_golden_crossbeta_rpl27_json() -> None:
    json_path = DATA / "RPL27 and RPL36" / "Cross-beta predictor" / "RPL27.json"
    if not json_path.is_file():
        json_path = DATA.parent / "RPL27 and RPL36" / "Cross-beta predictor" / "RPL27.json"
    if not json_path.is_file():
        pytest.skip("RPL27 cross-beta JSON not available")

    import json

    payload = json.loads(json_path.read_text())
    first_key = next(iter(payload))
    sequence = "".join(item["amino_acid"] for item in payload[first_key][0]["AA_list"])

    result = CrossBetaParser().parse(json_path, protein_id="RPL27", sequence=sequence)
    assert result.length == len(sequence)
    assert result.scores[0] == pytest.approx(0.7143500706559989)
    assert result.binary[0] == 1


def test_config_active_weights_sum_to_one() -> None:
    cfg = load_config()
    total = sum(cfg.metascore.weights.values())
    assert 0.999 <= total <= 1.001


def test_metascore_presets_configured() -> None:
    cfg = load_config()
    names = list_metascore_presets(cfg)
    assert "functional_amyloids" in names
    assert "pathogenic_amyloids" in names
    assert "predictor_specificity" in names
    assert cfg.metascore.preset == "predictor_specificity"
    for name in names:
        total = sum(cfg.metascore.presets[name].values())
        assert 0.999 <= total <= 1.001
