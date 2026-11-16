"""TANGO output obtained elsewhere must be reusable without the binary.

The profile logic lived in predictors/tango.py all along, used by the runner after
it invoked TANGO. What was missing was a parser CLASS and its registry entry -- and
the scheduler's parse-only path and --skip-run both go through get_parser(), so a
directory of perfectly good TANGO output could not be read without the binary that
made it. On a machine where that binary cannot execute at all (the distributed
Linux ELF build, on macOS) output obtained elsewhere was unusable.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.predictors.registry import get_parser, list_parsers

#: A real TANGO table: tab-separated, zero-padded 1-based res, leading spaces
#: inside cells, and the 5 % convention on the Aggregation column.
TABLE = """\
res\taa\tBeta\tTurn\tHelix\tAggregation\tConc-Stab_Aggregation
01\t  M\t  0.0\t  0.4\t0.000\t0.000\t0.000
02\t  A\t  0.0\t  0.6\t0.000\t0.000\t0.000
03\t  V\t 10.3\t  0.0\t0.000\t97.233\t97.233
04\t  L\t 11.1\t  0.0\t0.000\t96.100\t96.100
05\t  K\t  0.0\t  0.8\t0.183\t0.000\t0.000
"""
SEQUENCE = "MAVLK"


def _write(tmp_path, table=TABLE):
    path = tmp_path / "RPS2.txt"
    path.write_text(table, encoding="utf-8")
    return path


def test_tango_is_registered_so_parse_only_can_reach_it():
    assert "tango" in list_parsers()
    assert type(get_parser("tango")).__name__ == "TangoParser"


def test_an_existing_table_is_read_without_running_anything(tmp_path):
    result = get_parser("tango").parse(
        _write(tmp_path), protein_id="RPS2", sequence=SEQUENCE
    )
    assert len(result.scores) == len(SEQUENCE)
    assert result.metadata["reused_existing_output"] is True
    called = [i + 1 for i, flag in enumerate(result.binary) if flag]
    assert called == [3, 4], "5 % is TANGO's cut on Aggregation"


def test_the_protein_identity_is_verified_from_the_files_own_residues(tmp_path):
    """This is what makes reuse safe. The table carries its own `aa` column, so a
    file from another protein is caught rather than aligned onto this one -- which
    would corrupt every position downstream while looking fine."""
    with pytest.raises(Exception) as excinfo:
        get_parser("tango").parse(
            _write(tmp_path), protein_id="RPS2", sequence="MAQQQ"
        )
    assert "aa" in str(excinfo.value).lower() or "sequence" in str(excinfo.value).lower()


def test_a_length_disagreement_is_refused(tmp_path):
    """tango_profile's own residue check fires first and is stricter than a length
    comparison -- it reads the file's `aa` column -- so the length guard in the
    parser is a backstop rather than the primary defence. Either way the file is
    refused; the test asserts that, not which message."""
    with pytest.raises(Exception):
        get_parser("tango").parse(
            _write(tmp_path), protein_id="RPS2", sequence="MAVLKMAVLK"
        )


def test_the_threshold_is_configurable_and_recorded(tmp_path):
    lenient = get_parser("tango", threshold=0.0).parse(
        _write(tmp_path), protein_id="RPS2", sequence=SEQUENCE
    )
    assert lenient.metadata["threshold"] == 0.0
    assert sum(lenient.binary) > 2


def test_beta_is_not_offered_as_a_score_column():
    """Beta-propensity is a different quantity from aggregation, on its own scale.
    SCORE_COLUMNS excludes it deliberately, so a panel cannot end up with one tool
    reporting something other than what its threshold assumes."""
    with pytest.raises(ValueError, match="score_column must be one of"):
        get_parser("tango", score_column="Beta")


def test_the_alternative_aggregation_column_is_recorded_distinctly(tmp_path):
    """Conc-Stab_Aggregation is the concentration-stabilised variant. It is not
    interchangeable with Aggregation, so which one produced the numbers has to be
    readable off the result."""
    result = get_parser("tango", score_column="Conc-Stab_Aggregation").parse(
        _write(tmp_path), protein_id="RPS2", sequence=SEQUENCE
    )
    assert result.metadata["score_column"] == "Conc-Stab_Aggregation"


def test_an_unknown_column_is_rejected_at_construction():
    with pytest.raises(ValueError, match="score_column must be one of"):
        get_parser("tango", score_column="Aggregaton")
