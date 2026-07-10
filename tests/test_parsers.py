"""Tests for aggressor_wrappers parsers and merge."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.core.merge import merge_predictor_tables
from aggressor_wrappers.core.schema import get_predictor_spec
from aggressor_wrappers.predictors.aggreprot import AggreProtParser
from aggressor_wrappers.predictors.archcandy import ArchCandyParser
from aggressor_wrappers.predictors.crossbeta import CrossBetaParser
from aggressor_wrappers.predictors.pasta import PASTAParser
from aggressor_wrappers.predictors.waltz import WALTZParser

FIXTURES = Path(__file__).parent / "fixtures"
SEQUENCE = "MADKG"  # 5 residues
WALTZ_DETAILED = FIXTURES / "waltz_detailed.txt"


@pytest.fixture
def waltz_detailed_mini(tmp_path: Path) -> Path:
    path = tmp_path / "waltz.txt"
    path.write_text(
        ">t\n"
        "Positions\tSequence\tAverage score per residue\n"
        "2-3\tAD\t95.0\n"
        "5-5\tG\t80.0\n"
    )
    return path


@pytest.fixture
def pasta_energy(tmp_path: Path) -> Path:
    path = tmp_path / "pasta.txt"
    path.write_text("-3\n-6\n-4\n-2\n-7\n")
    return path


@pytest.fixture
def aggreprot_csv(tmp_path: Path) -> Path:
    path = tmp_path / "aggreprot.csv"
    path.write_text(
        "header\n"
        "position,amino_acid,aggregation,sasa,transmembrane,struct_position\n"
        "1,M,0.1,1,0,1\n"
        "2,A,0.3,1,0,2\n"
        "3,D,0.1,1,0,3\n"
        "4,K,0.5,1,0,4\n"
        "5,G,0.1,1,0,5\n"
    )
    return path


@pytest.fixture
def archcandy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "archcandy.csv"
    pd.DataFrame(
        {"Start": [2], "Stop": [3], "Score": [0.85]},
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def crossbeta_json(tmp_path: Path) -> Path:
    path = tmp_path / "crossbeta.json"
    payload = {
        "query": [
            {
                "AA_list": [
                    {"index": i, "amino_acid": aa, "mean_confidence": score}
                    for i, (aa, score) in enumerate(zip(SEQUENCE, [0.1, 0.6, 0.2, 0.8, 0.3]))
                ]
            }
        ]
    }
    path.write_text(json.dumps(payload))
    return path


def test_waltz_parser(waltz_detailed_mini: Path) -> None:
    result = WALTZParser().parse(waltz_detailed_mini, protein_id="t", sequence=SEQUENCE)
    assert result.scores[1] == 95.0
    assert result.scores[2] == 95.0
    assert result.scores[4] == 80.0
    assert result.binary[1] == 1
    assert result.binary[0] == 0
    df = result.to_dataframe()
    assert list(df.columns) == ["position", "aa_name", "waltz_score", "waltz_bin"]


def test_waltz_parser_detailed_text() -> None:
    if not WALTZ_DETAILED.is_file():
        pytest.skip("waltz detailed fixture missing")
    sequence = "M" * 211 + "SDVWWG" + "A" * 88  # APP_human length 305
    result = WALTZParser().parse(WALTZ_DETAILED, protein_id="APP_human", sequence=sequence)
    assert result.binary[205:211] == [1, 1, 1, 1, 1, 1]
    assert result.scores[205] == pytest.approx(94.314381)
    assert result.binary.count(1) == 6


def test_waltz_parser_detailed_text_no_regions() -> None:
    if not WALTZ_DETAILED.is_file():
        pytest.skip("waltz detailed fixture missing")
    sequence = "M" * 249
    result = WALTZParser().parse(WALTZ_DETAILED, protein_id="RPS6_human", sequence=sequence)
    assert result.binary == [0] * len(sequence)
    assert result.scores == [0.0] * len(sequence)


def test_pasta_parser(pasta_energy: Path) -> None:
    result = PASTAParser().parse(pasta_energy, protein_id="t", sequence=SEQUENCE)
    assert result.binary[1] == 1  # -6 < -2.8
    assert result.binary[4] == 1  # -7 < -2.8


def test_aggreprot_parser(aggreprot_csv: Path) -> None:
    result = AggreProtParser().parse(aggreprot_csv, protein_id="t", sequence=SEQUENCE)
    assert result.binary[1] == 1  # 0.3 >= 0.25
    assert result.binary[3] == 1  # 0.5 >= 0.25


def test_archcandy_parser(archcandy_csv: Path) -> None:
    result = ArchCandyParser().parse(archcandy_csv, protein_id="t", sequence=SEQUENCE)
    assert result.scores[1] == 0.85
    assert result.binary[1] == 1
    assert result.binary[0] == 0


def test_archcandy_parser_cumulative_overlapping_regions(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "archcandy_app.csv"
    if not fixture.is_file():
        pytest.skip("archcandy fixture missing")
    sequence = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    cumulative = ArchCandyParser(score_mode="cumulative").parse(
        fixture, protein_id="APP", sequence=sequence
    )
    highest = ArchCandyParser(score_mode="highest").parse(
        fixture, protein_id="APP", sequence=sequence
    )
    # Position 15 is covered by multiple regions in the fixture CSV.
    assert cumulative.scores[14] > highest.scores[14]
    assert cumulative.binary[14] == 1


def test_crossbeta_parser(crossbeta_json: Path) -> None:
    result = CrossBetaParser(confidence_threshold=0.5).parse(
        crossbeta_json, protein_id="t", sequence=SEQUENCE
    )
    assert result.binary[1] == 1
    assert result.binary[3] == 1


def test_merge_wide_format(waltz_detailed_mini: Path, pasta_energy: Path) -> None:
    waltz = WALTZParser().parse(waltz_detailed_mini, protein_id="t", sequence=SEQUENCE)
    pasta = PASTAParser().parse(pasta_energy, protein_id="t", sequence=SEQUENCE)
    wide = merge_predictor_tables([waltz, pasta])
    assert list(wide.columns[:2]) == ["position", "aa_name"]
    assert wide.iloc[0]["position"] == 1
    assert wide.iloc[0]["aa_name"] == "M"
    assert "waltz_score" in wide.columns
    assert "pasta_score" in wide.columns


def test_merge_rejects_duplicate_predictor(waltz_detailed_mini: Path, pasta_energy: Path) -> None:
    waltz = WALTZParser().parse(waltz_detailed_mini, protein_id="t", sequence=SEQUENCE)
    with pytest.raises(ValueError, match="Duplicate predictor"):
        merge_predictor_tables([waltz, waltz])


def test_crossbeta_length_mismatch(crossbeta_json: Path) -> None:
    with pytest.raises(ValueError, match="does not match sequence length"):
        CrossBetaParser().parse(crossbeta_json, protein_id="t", sequence="MA")


def test_reference_bht_columns_match_registry() -> None:
    """Historical BHT all/RPS2_human_all.csv score columns match registry."""
    from aggressor_wrappers.paths import bht_reference_root

    ref_path = bht_reference_root() / "all" / "RPS2_human_all.csv"
    if not ref_path.exists():
        pytest.skip("Reference CSV not available")
    ref = pd.read_csv(ref_path, index_col=0)
    expected_score_cols = {spec.score_column for spec in map(get_predictor_spec, (
        "aggreprot", "aggrescan", "appnn", "archcandy", "crossbeta", "pasta", "path", "waltz"
    ))}
    missing = expected_score_cols - set(ref.columns)
    assert not missing, f"Reference table missing columns: {missing}"
