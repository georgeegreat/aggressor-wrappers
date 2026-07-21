"""Local-tool parsers: TANGO, AmyloGram, AmyloDeep, ArchCandy.

Binary-dependent tests skip when the licensed tool is absent, so CI stays green
on a machine that has no TANGO binary or ArchCandy JAR.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from aggressor_wrappers.predictors.amylodeep import parse_amylodeep
from aggressor_wrappers.predictors.amylogram import (
    project_windows,
    sliding_windows,
)
from aggressor_wrappers.predictors.tango import tango_profile

AB42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

TANGO_TABLE = "res\taa\tBeta\tTurn\tHelix\tAggregation\tConc-Stab_Aggregation\n"


def _tango_file(tmp_path: Path, sequence: str, agg: list[float]) -> Path:
    path = tmp_path / "t.txt"
    with path.open("w") as fh:
        fh.write(TANGO_TABLE)
        for i, (aa, a) in enumerate(zip(sequence, agg, strict=True), start=1):
            fh.write(f"{i:02d}\t  {aa}\t  0.0\t  0.1\t0.000\t{a:.3f}\t{a:.3f}\n")
    return path


# --------------------------------------------------------------------------- #
# TANGO
# --------------------------------------------------------------------------- #


def test_tango_profile_is_one_based_and_matches_sequence(tmp_path):
    seq = "MVGGVV"
    path = _tango_file(tmp_path, seq, [0.0, 10.0, 90.0, 90.0, 10.0, 0.0])
    scores, aux, meta = tango_profile(path, seq)
    assert len(scores) == len(seq)
    assert scores[2] == pytest.approx(90.0)
    assert meta["score_column"] == "Aggregation"


def test_tango_retains_secondary_structure_channels(tmp_path):
    """Beta/Turn/Helix are what make TANGO more than an aggregation scale."""
    seq = "MVGGVV"
    path = _tango_file(tmp_path, seq, [0.0] * 6)
    _, aux, _ = tango_profile(path, seq)
    for channel in ("beta", "turn", "helix", "conc_stab_aggregation"):
        assert channel in aux
        assert len(aux[channel]) == len(seq)


def test_tango_rejects_mismatched_sequence(tmp_path):
    """A stale output file must fail loudly, not silently misalign positions."""
    path = _tango_file(tmp_path, "MVGGVV", [0.0] * 6)
    with pytest.raises(ValueError, match="does not match"):
        tango_profile(path, "MVGGVVIATV")


@pytest.mark.skipif(shutil.which("tango") is None, reason="TANGO binary not installed")
def test_tango_runner_against_real_binary(tmp_path):
    from aggressor_wrappers.runners.tango import TANGORunner

    fasta = tmp_path / "ab.fasta"
    fasta.write_text(f">AB42\n{AB42}\n")
    res = TANGORunner(binary_path="tango").run(
        fasta=fasta, protein_id="AB42", work_dir=tmp_path
    )
    assert res.length == len(AB42)
    assert max(res.scores) > 50  # Abeta has strong APRs


# --------------------------------------------------------------------------- #
# AmyloGram
# --------------------------------------------------------------------------- #


def test_sliding_windows_count():
    w = sliding_windows(AB42, 6)
    assert len(w) == len(AB42) - 6 + 1
    assert w[0] == (1, AB42[:6])


def test_short_sequence_is_scored_not_dropped():
    """A peptide shorter than the window is still a legitimate query."""
    assert sliding_windows("MVGG", 6) == [(1, "MVGG")]


def test_projection_max_covers_full_window():
    w = sliding_windows(AB42, 6)
    probs = {f"w{s}": (0.95 if "LVFF" in p else 0.05) for s, p in w}
    scores = project_windows(w, probs, len(AB42), aggregation="max")
    flagged = [i + 1 for i, s in enumerate(scores) if s > 0.5]
    assert AB42[flagged[0] - 1 : flagged[-1]] == "QKLVFFAE"


def test_mean_aggregation_dilutes_isolated_signal():
    """Documents why 'max' is the default rather than 'mean'."""
    w = sliding_windows(AB42, 6)
    probs = {f"w{s}": (0.95 if "LVFF" in p else 0.05) for s, p in w}
    max_hits = sum(s > 0.5 for s in project_windows(w, probs, len(AB42), aggregation="max"))
    mean_hits = sum(s > 0.5 for s in project_windows(w, probs, len(AB42), aggregation="mean"))
    assert mean_hits < max_hits


def test_missing_window_probability_raises():
    w = sliding_windows("MVGGVVIATV", 6)
    with pytest.raises(KeyError):
        project_windows(w, {"w1": 0.5}, 10)


# --------------------------------------------------------------------------- #
# AmyloDeep
# --------------------------------------------------------------------------- #


def _amylodeep_csv(tmp_path: Path, n_rows: int, length: int) -> Path:
    path = tmp_path / "ad.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence_id", "position", "probability", "sequence_length"])
        for i in range(n_rows):
            w.writerow(["s", i, 0.9, length])
    return path


def test_amylodeep_detects_window_output(tmp_path):
    """22 rows for a 31-residue sequence implies a window of 10."""
    scores, meta = parse_amylodeep(_amylodeep_csv(tmp_path, 22, 31))
    assert meta["granularity"] == "window"
    assert meta["window_size"] == 10
    assert len(scores) == 31
    assert all(s > 0 for s in scores)  # no residue left unscored


def test_amylodeep_detects_per_residue_output(tmp_path):
    scores, meta = parse_amylodeep(_amylodeep_csv(tmp_path, 31, 31))
    assert meta["granularity"] == "per_residue"
    assert meta["window_size"] == 1
    assert len(scores) == 31


def test_amylodeep_positions_are_zero_based(tmp_path):
    _, meta = parse_amylodeep(_amylodeep_csv(tmp_path, 31, 31))
    assert meta["position_base"] == 0
