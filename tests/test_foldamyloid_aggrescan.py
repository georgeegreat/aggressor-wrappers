"""FoldAmyloid and AGGRESCAN: contract, parsing, and the mis-filed-input guard.

The FoldAmyloid fixture is a VERBATIM capture of the service's reply, not a
hand-written approximation, so a change in its wire format fails here rather
than in a run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggressor_wrappers.predictors.aggrescan import AggrescanParser, hotspot_mask
from aggressor_wrappers.predictors.foldamyloid import FoldAmyloidParser
from aggressor_wrappers.runners.foldamyloid import (
    CONTACTS_8A,
    FoldAmyloidRunner,
    parse_long_table,
    scale_string,
)
from aggressor_wrappers.runners.registry import get_runner

FIXTURES = Path(__file__).parent / "fixtures"
WORKER_JSON = FIXTURES / "foldamyloid_worker_abeta42.json"
FA_CSV = FIXTURES / "foldamyloid_abeta42.csv"

# Abeta(1-42). KLVFFA (16-21) is the segment shown to be necessary and
# sufficient for fibril formation (Tjernberg et al., 1996, JBC 271:8545).
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _spans(positions: list[int]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for i in positions:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def test_scale_is_the_model_not_a_formatting_detail() -> None:
    """p2 carries the contact scale; 21 entries incl. X, in the wire format."""
    assert len(CONTACTS_8A) == 21
    assert CONTACTS_8A["W"] == pytest.approx(28.48)  # most contacts
    assert CONTACTS_8A["G"] == pytest.approx(17.11)  # fewest
    rendered = scale_string()
    assert rendered.startswith("A 19.89|")
    assert rendered.endswith("X 20.73|")


@pytest.mark.skipif(not WORKER_JSON.is_file(), reason="worker capture missing")
def test_long_table_lives_in_tbl_and_recovers_the_abeta_core() -> None:
    reply = json.loads(WORKER_JSON.read_text())
    assert "tbl" in reply, "per-residue data is in 'tbl', not 'seq'"
    rows = parse_long_table(reply["tbl"])
    assert "".join(r[1] for r in rows) == ABETA42
    flagged = _spans([n for n, _, flag, _ in rows if flag])
    assert (16, 21) in flagged, "KLVFFA must be called"
    assert (32, 36) in flagged, "IGLMV must be called"


@pytest.mark.skipif(not FA_CSV.is_file(), reason="foldamyloid fixture missing")
def test_parser_uses_the_tool_flag_and_keeps_the_profile() -> None:
    result = FoldAmyloidParser().parse(FA_CSV, protein_id="ABETA42", sequence=ABETA42)
    assert result.metadata["binarised_from"] == "tool_flag"
    assert sum(result.binary) == 11
    # F19 sits at the profile maximum: an aromatic in the middle of the core.
    assert result.scores.index(max(result.scores)) + 1 == 19
    assert result.aux["tool_flag"] == result.binary


@pytest.mark.skipif(not FA_CSV.is_file(), reason="foldamyloid fixture missing")
def test_mis_filed_output_is_rejected_rather_than_silently_attributed() -> None:
    """A predictor file for the wrong protein must not parse successfully.

    Two FoldAmyloid exports in this project carried each other's protein. A
    swapped file parses perfectly and attributes one chain's APRs to another,
    which no downstream check can detect -- so the residue identities the tool
    itself reported are compared against the query sequence.
    """
    with pytest.raises(ValueError, match="does not match the sequence"):
        FoldAmyloidParser().parse(FA_CSV, protein_id="OTHER", sequence="M" * 42)


def test_aggrescan_marker_ignores_whitespace_artifacts() -> None:
    """Stray non-breaking spaces in Prediction are not Hot Spot calls.

    AGGRESCAN leaves the marker blank outside Hot Spots but its Mac Roman
    exports carry 0xCA there; under a "not empty" rule that becomes a phantom
    single-residue APR at the N-terminus.
    """
    import pandas as pd

    raw = pd.Series(["\xca\xca", "", "1", "1", None, "0", "true"])
    assert hotspot_mask(raw) == [0, 0, 1, 1, 0, 0, 1]


def test_aggrescan_parser_round_trip(tmp_path: Path) -> None:
    seq = "MKVLAAG"
    csv = tmp_path / "x.csv"
    csv.write_text(
        "Number,AA,a4v,HSA,NHSA,a4vAHS,Prediction\r\n"
        "1,M,-0.064,0,0,0,\xca\xca\r\n"
        "2,K,-0.064,0,0,0,\r\n"
        "3,V,0.348,2.111,0.302,0.282,1\r\n"
        "4,L,0.391,2.111,0.302,0.282,1\r\n"
        "5,A,0.336,2.111,0.302,0.282,1\r\n"
        "6,A,0.164,0,0,0,\r\n"
        "7,G,-0.143,0,0,0,\r\n",
        encoding="latin-1",
    )
    result = AggrescanParser().parse(csv, protein_id="X", sequence=seq)
    assert result.binary == [0, 0, 1, 1, 1, 0, 0]
    assert result.scores[3] == pytest.approx(0.391)


def test_aggrescan_accepts_semicolon_exports(tmp_path: Path) -> None:
    """Semicolon-delimited exports must load; three real files were this shape."""
    csv = tmp_path / "semi.csv"
    csv.write_text(
        "Number;AA;Score;HSA;NHSA;a4vAHS;Disorder\n"
        "1;M;0.1;0;0;0;\n2;K;0.5;1;1;1;1\n3;V;0.5;1;1;1;1\n",
        encoding="utf-8",
    )
    result = AggrescanParser().parse(csv, protein_id="X", sequence="MKV")
    assert result.binary == [0, 1, 1]


def test_foldamyloid_runner_registered_and_web_only() -> None:
    runner = get_runner("foldamyloid")
    assert isinstance(runner, FoldAmyloidRunner)
    assert runner.averaging_frame == 5
    assert runner.threshold == pytest.approx(21.4)
