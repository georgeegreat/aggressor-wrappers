"""A prediction over part of a sequence is a prediction about that part.

Both of this project's RPS12 files cover residues 10-129 of a 132-residue
chain. Refused, the protein lost two predictors. Accepted at face value, each
file aligns at residue 1 and every call lands nine residues early -- a complete,
plausible, wrong answer that nothing downstream could detect.

So a short file is accepted when its placement can be ESTABLISHED, and the
residues outside it are marked not-predicted rather than predicted-negative.
Those are different claims: a zero says the tool looked and found nothing.
"""

from __future__ import annotations

import math

import pytest

from aggressor_wrappers.core.coverage import (
    Coverage,
    PartialCoverageError,
    align_partial,
    parse_offsets,
    place,
)
from aggressor_wrappers.predictors.foldamyloid import FoldAmyloidParser
from aggressor_wrappers.predictors.waltz import WALTZParser, parse_per_residue

#: RPS12's real case, in miniature: a 20-residue chain whose file covers 10-19.
SEQUENCE = "MAEEGIAAGGVMDVNTALQE"
BLOCK = SEQUENCE[9:19]          # residues 10-19


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


def test_a_unique_match_places_the_block():
    assert align_partial(BLOCK, SEQUENCE) == 9


def test_no_match_is_a_different_protein():
    with pytest.raises(PartialCoverageError) as excinfo:
        align_partial("WWWWWW", SEQUENCE)

    assert "not a truncation of this one" in str(excinfo.value)


def test_a_repeated_block_cannot_be_placed():
    """Proof means ONE offset. Two is a coin toss with a protein on it."""
    with pytest.raises(PartialCoverageError) as excinfo:
        align_partial("AA", "MAAKAAK")

    assert "cannot be placed" in str(excinfo.value)


def test_a_longer_block_is_not_a_sub_range():
    with pytest.raises(PartialCoverageError) as excinfo:
        align_partial(SEQUENCE + "KK", SEQUENCE)

    assert "not a sub-range" in str(excinfo.value)


def test_uncovered_positions_are_nan_not_zero():
    placed, coverage = place([1.0, 2.0], offset=2, length=5)

    assert math.isnan(placed[0]) and math.isnan(placed[1])
    assert placed[2:4] == [1.0, 2.0]
    assert coverage.start == 3 and coverage.stop == 4 and coverage.missing == 3


def test_coverage_describes_itself_for_a_report():
    assert "complete" in Coverage(1, 20, 20).describe()
    text = Coverage(10, 129, 132).describe()
    assert "residues 10-129 of 132" in text and "12 not predicted" in text


def test_offsets_are_declared_per_protein():
    assert parse_offsets("RPS12:9, RPL22:0") == {"RPS12": 9, "RPL22": 0}


def test_a_bare_offset_is_refused():
    """One offset for a whole panel would shift every complete file with it."""
    with pytest.raises(ValueError) as excinfo:
        parse_offsets("9")

    assert "belongs to one protein" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# FoldAmyloid places itself, because it reports residues
# --------------------------------------------------------------------------- #


def _foldamyloid_csv(tmp_path, residues, *, start=1, flagged=()):
    rows = ["Num,Res,Fold,Value"]
    for i, residue in enumerate(residues, start=start):
        mark = "f" if i in flagged else ""
        rows.append(f"{i},{residue},{mark},22.5")
    path = tmp_path / "x.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_a_partial_export_is_placed_by_its_residues(tmp_path):
    """Numbered 1..10 but holding residues 10-19: the residues win."""
    path = _foldamyloid_csv(tmp_path, BLOCK, start=1, flagged={3, 4, 5, 6, 7})

    result = FoldAmyloidParser().parse(path, protein_id="P", sequence=SEQUENCE)

    assert result.metadata["coverage_start"] == 10
    assert result.metadata["coverage_stop"] == 19
    # The flags land where the residues put them, not where the file numbers do.
    assert result.binary[11] == 1 and result.binary[2] == 0
    assert result.aux["covered"][0] == 0 and result.aux["covered"][9] == 1
    assert math.isnan(result.scores[0])


def test_a_complete_export_is_unchanged(tmp_path):
    path = _foldamyloid_csv(tmp_path, SEQUENCE, flagged={1, 2})

    result = FoldAmyloidParser().parse(path, protein_id="P", sequence=SEQUENCE)

    assert result.metadata["coverage_start"] == 1
    assert result.metadata["coverage_stop"] == len(SEQUENCE)
    assert all(result.aux["covered"])
    assert not any(math.isnan(value) for value in result.scores)


def test_a_mis_filed_export_is_still_refused(tmp_path):
    """Partial coverage must not become a way for the wrong file to parse."""
    path = _foldamyloid_csv(tmp_path, "WWWWWWWWWW")

    with pytest.raises(ValueError) as excinfo:
        FoldAmyloidParser().parse(path, protein_id="P", sequence=SEQUENCE)

    assert "does not match the sequence" in str(excinfo.value)


def test_partial_can_be_switched_off(tmp_path):
    path = _foldamyloid_csv(tmp_path, BLOCK, start=1)

    with pytest.raises(ValueError) as excinfo:
        FoldAmyloidParser(accept_partial=False).parse(
            path, protein_id="P", sequence=SEQUENCE
        )

    assert "accept_partial is off" in str(excinfo.value)


def test_a_profile_with_holes_is_refused(tmp_path):
    rows = ["Num,Res,Fold,Value", "1,M,,22.5", "5,I,,22.5"]
    path = tmp_path / "x.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        FoldAmyloidParser().parse(path, protein_id="P", sequence=SEQUENCE)

    assert "not contiguous" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# WALTZ cannot place itself, so the offset is declared
# --------------------------------------------------------------------------- #


def _dat(values, *, start=1):
    return "".join(
        f"{i}\t{v:.6f}\n" for i, v in enumerate(values, start=start)
    )


def test_a_partial_dat_without_a_declaration_is_refused():
    text = _dat([0.0] * 10)

    with pytest.raises(ValueError) as excinfo:
        parse_per_residue(text, sequence=SEQUENCE, source="p.dat")

    message = str(excinfo.value)
    assert "covers 10 of 20" in message
    assert "offsets" in message


def test_a_declared_offset_places_the_dat():
    text = _dat([0.0, 0.0, 96.3, 96.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    scores, binary, coverage = parse_per_residue(
        text, sequence=SEQUENCE, source="p.dat", offset=9
    )

    assert coverage.start == 10 and coverage.stop == 19
    assert binary[11] == 1 and binary[2] == 0
    assert math.isnan(scores[0])


def test_the_parser_reads_the_declaration_for_its_protein(tmp_path):
    path = tmp_path / "WaltzJob_1_RPS12.dat"
    path.write_text(_dat([0.0] * 10), encoding="utf-8")

    parser = WALTZParser(offsets="RPS12:9")
    result = parser.parse(path, protein_id="RPS12", sequence=SEQUENCE)

    assert result.metadata["coverage_start"] == 10


def test_a_declaration_for_another_protein_does_not_leak(tmp_path):
    """The whole point of keying offsets by protein."""
    path = tmp_path / "p.dat"
    path.write_text(_dat([0.0] * 10), encoding="utf-8")

    parser = WALTZParser(offsets="RPL22:9")
    with pytest.raises(ValueError):
        parser.parse(path, protein_id="RPS12", sequence=SEQUENCE)


def test_a_complete_dat_needs_no_declaration():
    text = _dat([0.0] * len(SEQUENCE))

    scores, binary, coverage = parse_per_residue(text, sequence=SEQUENCE)

    assert coverage.is_complete
    assert not any(math.isnan(value) for value in scores)
