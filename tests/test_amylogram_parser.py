"""AmyloGramParser: registration, both input shapes, and the regression it closes.

The bug this file exists for is not a wrong number but a missing one. AmyloGram
had a runner and a projection module and no ``BasePredictorParser`` subclass, so
it was absent from ``PARSER_REGISTRY``; ``batch.pipeline._merge_protein`` gates
on ``list_parsers()`` membership and skipped it with a bare ``continue``. The
predictor ran on every protein of a 372-sequence sweep, wrote correct parsed
tables, and never reached a merged table -- and the consensus denominator was 9
where the config declared 10, with nothing in the log to say so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.predictors.amylogram import (
    AmyloGramParser,
    detect_output_shape,
)
from aggressor_wrappers.predictors.registry import PARSER_REGISTRY, get_parser, list_parsers

SEQUENCE = "MKLVFFAEDVGSNKGAIIGLMVGGVV"


def _per_residue_csv(path: Path, sequence: str, scores: list[float]) -> Path:
    pd.DataFrame(
        {
            "Number": range(1, len(sequence) + 1),
            "Residue": list(sequence),
            "Score": scores,
        }
    ).to_csv(path, index=False)
    return path


def _merged_style_csv(path: Path, sequence: str, scores: list[float]) -> Path:
    """The shape `batch.pipeline` writes into <predictor>/parsed/."""
    pd.DataFrame(
        {
            "position": range(1, len(sequence) + 1),
            "aa_name": list(sequence),
            "amylogram_score": scores,
            "amylogram_bin": [int(s >= 0.5) for s in scores],
        }
    ).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Registration -- the actual regression
# --------------------------------------------------------------------------- #
def test_amylogram_is_registered() -> None:
    assert "amylogram" in list_parsers()
    assert "amylogram" in PARSER_REGISTRY


def test_amylogram_reaches_the_merge_gate() -> None:
    """`_merge_protein` gates on exactly this membership test."""
    requested = ["appnn", "pasta", "amylogram"]
    assert [k for k in requested if k not in list_parsers()] == []


def test_get_parser_amylogram_constructs() -> None:
    parser = get_parser("amylogram")
    assert isinstance(parser, AmyloGramParser)
    assert parser.threshold == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Shape detection
# --------------------------------------------------------------------------- #
def test_detects_per_residue(tmp_path: Path) -> None:
    path = _per_residue_csv(tmp_path / "r.csv", SEQUENCE, [0.1] * len(SEQUENCE))
    assert detect_output_shape(pd.read_csv(path)) == "per_residue"


def test_detects_merged_style_per_residue(tmp_path: Path) -> None:
    path = _merged_style_csv(tmp_path / "m.csv", SEQUENCE, [0.1] * len(SEQUENCE))
    assert detect_output_shape(pd.read_csv(path)) == "per_residue"


def test_detects_per_peptide(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    pd.DataFrame({"name": ["w1", "w2"], "probability": [0.9, 0.1]}).to_csv(
        path, index=False
    )
    assert detect_output_shape(pd.read_csv(path)) == "per_peptide"


def test_unrecognised_shape_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="neither a per-residue table"):
        detect_output_shape(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# Per-residue path
# --------------------------------------------------------------------------- #
def test_per_residue_is_read_through_unchanged(tmp_path: Path) -> None:
    scores = [i / len(SEQUENCE) for i in range(len(SEQUENCE))]
    path = _per_residue_csv(tmp_path / "r.csv", SEQUENCE, scores)
    result = AmyloGramParser().parse(path, protein_id="t", sequence=SEQUENCE)
    assert result.scores == pytest.approx(scores)
    assert result.binary == [int(s >= 0.5) for s in scores]
    assert "already projected" in result.metadata["parsed_from"]


def test_per_residue_length_mismatch_raises(tmp_path: Path) -> None:
    path = _per_residue_csv(tmp_path / "r.csv", SEQUENCE[:10], [0.1] * 10)
    with pytest.raises(ValueError, match="10 rows"):
        AmyloGramParser().parse(path, protein_id="t", sequence=SEQUENCE)


def test_mis_filed_output_is_caught(tmp_path: Path) -> None:
    """A swapped export parses perfectly and attributes one chain's APRs to another."""
    other = "A" * len(SEQUENCE)
    path = _per_residue_csv(tmp_path / "r.csv", other, [0.1] * len(SEQUENCE))
    with pytest.raises(ValueError, match="does not match the sequence"):
        AmyloGramParser().parse(path, protein_id="t", sequence=SEQUENCE)


def test_threshold_is_applied(tmp_path: Path) -> None:
    scores = [0.3, 0.7] * (len(SEQUENCE) // 2)
    path = _per_residue_csv(tmp_path / "r.csv", SEQUENCE, scores)
    strict = AmyloGramParser(threshold=0.8).parse(
        path, protein_id="t", sequence=SEQUENCE
    )
    assert sum(strict.binary) == 0
    loose = AmyloGramParser(threshold=0.2).parse(
        path, protein_id="t", sequence=SEQUENCE
    )
    assert sum(loose.binary) == len(SEQUENCE)


# --------------------------------------------------------------------------- #
# Per-peptide path -- the `--skip-run` reparse
# --------------------------------------------------------------------------- #
def test_per_peptide_is_projected(tmp_path: Path) -> None:
    """Reparsing raw output reproduces the runner's projection."""
    n_windows = len(SEQUENCE) - 6 + 1
    probs = [0.0] * n_windows
    probs[0] = 1.0  # one amyloidogenic hexapeptide at the N-terminus
    path = tmp_path / "p.csv"
    pd.DataFrame(
        {"name": [f"w{i + 1}" for i in range(n_windows)], "probability": probs}
    ).to_csv(path, index=False)

    result = AmyloGramParser(aggregation="max").parse(
        path, protein_id="t", sequence=SEQUENCE
    )
    assert len(result.scores) == len(SEQUENCE)
    # `max` raises every residue of the positive window and nothing else.
    assert result.binary[:6] == [1] * 6
    assert sum(result.binary) == 6
    assert "projected" in result.metadata["parsed_from"]
    assert "coverage_depth" in result.aux


def test_support_aggregation_is_stricter_than_max(tmp_path: Path) -> None:
    """The aggregation rule changes APR extent, which is the documented point."""
    n_windows = len(SEQUENCE) - 6 + 1
    probs = [0.0] * n_windows
    probs[0] = 1.0
    path = tmp_path / "p.csv"
    pd.DataFrame(
        {"name": [f"w{i + 1}" for i in range(n_windows)], "probability": probs}
    ).to_csv(path, index=False)

    widest = AmyloGramParser(aggregation="max").parse(
        path, protein_id="t", sequence=SEQUENCE
    )
    strictest = AmyloGramParser(aggregation="support", support_fraction=1.0).parse(
        path, protein_id="t", sequence=SEQUENCE
    )
    assert sum(strictest.binary) < sum(widest.binary)
