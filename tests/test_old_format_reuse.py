"""Predictions saved from a web server years ago must still be usable.

WALTZ and FoldAmyloid are both web services, and both have been down for long
stretches. What a person has in that situation is a file saved from the server
when it worked — under the server's own filename, in the server's own download
format — and neither was recognised.

WALTZ's parser read only the detailed text the HTML result page yields; the
``.dat`` the server packages in its ZIP (``position<TAB>score``, one line per
residue) went unread. Both runners' ``discover_outputs`` looked for exactly the
filename that runner writes itself, so a directory of genuine predictions named
``WaltzJob_1761904229_RPL22.dat`` or ``RPL22.csv`` discovered nothing, and the
predictor dropped out of the panel while still counting in its denominator.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.core.raw_discovery import (
    FOLDAMYLOID_RULES,
    FOLDAMYLOID_SUFFIXES,
    WALTZ_RULES,
    WALTZ_SUFFIXES,
    AmbiguousRawOutput,
    find_protein_file,
    find_protein_files,
)
from aggressor_wrappers.predictors.waltz import (
    WALTZParser,
    looks_per_residue,
    parse_per_residue,
)

#: A real WALTZ .dat: RPL22's, trimmed. One segment at 7-10, zero elsewhere.
DAT = "".join(
    f"{i}\t{96.321070 if 7 <= i <= 10 else 0.0:.6f}\n" for i in range(1, 13)
)
SEQUENCE = "MAPVKKLVVKGG"

#: The detailed format, for the discrimination tests.
DETAILED = ">RPL22\nPositions\tx\tScore\n7-10\t.\t96.32\n"


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #


def test_the_per_residue_download_is_recognised():
    assert looks_per_residue(DAT) is True


def test_the_detailed_download_is_not_mistaken_for_it():
    """A '>' header is the thing the per-residue format never has."""
    assert looks_per_residue(DETAILED) is False


def test_a_detailed_file_whose_first_region_starts_at_one_is_not_mistaken():
    """'1-9' reads as a position only if the '-' is ignored."""
    text = ">P\nPositions\tx\tScore\n1-9\t.\t96.32\n"
    assert looks_per_residue(text) is False


def test_detection_ignores_the_extension(tmp_path):
    """A .txt saved from the ZIP holds per-residue data; a .dat may hold either."""
    as_txt = tmp_path / "RPL22_waltz.txt"
    as_txt.write_text(DAT, encoding="utf-8")

    result = WALTZParser().parse(as_txt, protein_id="RPL22", sequence=SEQUENCE)

    assert result.metadata["layout"] == "per_residue"


# --------------------------------------------------------------------------- #
# Per-residue parsing
# --------------------------------------------------------------------------- #


def test_a_segment_becomes_a_call(tmp_path):
    path = tmp_path / "WaltzJob_1761904229_RPL22.dat"
    path.write_text(DAT, encoding="utf-8")

    result = WALTZParser().parse(path, protein_id="RPL22", sequence=SEQUENCE)

    assert result.binary == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert result.scores[6] == pytest.approx(96.32107)
    assert result.scores[0] == 0.0


def test_an_all_zero_file_is_a_scored_negative_not_an_error(tmp_path):
    """WALTZ finding nothing is a result; two of four real files look like this."""
    path = tmp_path / "p.dat"
    path.write_text("".join(f"{i}\t0.000000\n" for i in range(1, 13)), encoding="utf-8")

    result = WALTZParser().parse(path, protein_id="P", sequence=SEQUENCE)

    assert sum(result.binary) == 0
    assert len(result.scores) == len(SEQUENCE)


def test_a_short_file_is_refused_unless_its_offset_is_declared():
    """The .dat has no residue identities, so nothing in it can place it.

    A 10-line file against a 12-residue sequence is equally consistent with
    residues 1-10 and with 3-12, and the two readings put every segment two
    residues apart. Only the person who knows what was submitted can say.
    """
    short = "".join(f"{i}\t0.000000\n" for i in range(1, 11))

    with pytest.raises(ValueError) as excinfo:
        parse_per_residue(short, sequence=SEQUENCE, source="p.dat")

    message = str(excinfo.value)
    assert "covers 10 of 12 positions" in message
    assert "offsets" in message        # names the way to resolve it


def test_a_declared_offset_places_a_short_file():
    """RPS12's real case: 120 residues of 132, beginning at residue 10."""
    short = "".join(
        f"{i}\t{96.0 if i <= 2 else 0.0:.6f}\n" for i in range(1, 11)
    )

    scores, binary, coverage = parse_per_residue(
        short, sequence=SEQUENCE, source="p.dat", offset=2
    )

    assert coverage.start == 3 and coverage.stop == 12
    assert coverage.missing == 2
    # The segment lands where the declaration puts it, not at residue 1.
    assert binary == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # Residues the tool never saw are NaN, not 0: those are different claims.
    assert scores[0] != scores[0] and scores[1] != scores[1]


def test_a_longer_file_is_refused_as_a_different_protein():
    """A profile LONGER than the sequence is not a truncation of it."""
    longer = "".join(f"{i}\t0.000000\n" for i in range(1, 15))

    with pytest.raises(ValueError) as excinfo:
        parse_per_residue(longer, sequence=SEQUENCE, source="p.dat")

    assert "not a sub-range" in str(excinfo.value)


def test_a_repeated_position_is_refused():
    text = "".join(f"{i}\t0.0\n" for i in range(1, 13)) + "5\t1.0\n"

    with pytest.raises(ValueError) as excinfo:
        parse_per_residue(text, sequence=SEQUENCE, source="p.dat")

    assert "appears twice" in str(excinfo.value)


def test_blank_lines_are_tolerated():
    text = "\n".join(f"{i}\t0.0" for i in range(1, 13)) + "\n\n"

    scores, binary, coverage = parse_per_residue(text, sequence=SEQUENCE)

    assert len(scores) == 12 and sum(binary) == 0
    assert coverage.is_complete


# --------------------------------------------------------------------------- #
# Discovery under the names files actually have
# --------------------------------------------------------------------------- #


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("1\t0.0\n", encoding="utf-8")


def test_the_waltz_server_filename_is_found(tmp_path):
    _touch(tmp_path, "WaltzJob_1761904229_RPL22.dat")

    found = find_protein_files(
        tmp_path, ["RPL22"], WALTZ_RULES, suffixes=WALTZ_SUFFIXES
    )

    assert found["RPL22"].name == "WaltzJob_1761904229_RPL22.dat"


def test_a_bare_rename_is_found(tmp_path):
    _touch(tmp_path, "RPL22.dat")

    found = find_protein_files(
        tmp_path, ["RPL22"], WALTZ_RULES, suffixes=WALTZ_SUFFIXES
    )

    assert found["RPL22"].name == "RPL22.dat"


def test_the_canonical_name_wins_over_a_looser_match(tmp_path):
    """A resumed run must pick up what the runner itself wrote."""
    _touch(tmp_path, "RPL22_waltz.txt", "RPL22.dat")

    path = find_protein_file(tmp_path, "RPL22", WALTZ_RULES, suffixes=WALTZ_SUFFIXES)

    assert path.name == "RPL22_waltz.txt"


def test_a_prefix_cannot_claim_a_longer_name(tmp_path):
    """RPL36 and RPL36a are both in this panel.

    A 'RPL36*' glob hands one protein the other's track, which is the worst
    failure available here: it produces a complete, plausible, wrong answer.
    """
    _touch(
        tmp_path,
        "WaltzJob_1761904229_RPL36.dat",
        "WaltzJob_1761904229_RPL36a.dat",
    )

    found = find_protein_files(
        tmp_path, ["RPL36", "RPL36a"], WALTZ_RULES, suffixes=WALTZ_SUFFIXES
    )

    assert found["RPL36"].name == "WaltzJob_1761904229_RPL36.dat"
    assert found["RPL36a"].name == "WaltzJob_1761904229_RPL36a.dat"


def test_two_files_claiming_one_protein_are_refused(tmp_path):
    _touch(tmp_path, "RPL22.dat", "RPL22.txt")

    with pytest.raises(AmbiguousRawOutput) as excinfo:
        find_protein_file(tmp_path, "RPL22", WALTZ_RULES, suffixes=WALTZ_SUFFIXES)

    message = str(excinfo.value)
    assert "RPL22.dat" in message and "RPL22.txt" in message


def test_files_of_other_kinds_are_not_candidates(tmp_path):
    """The directory also holds logs, sidecars and parsed CSVs."""
    _touch(tmp_path, "RPL22.log", "RPL22.job.json")

    assert find_protein_file(
        tmp_path, "RPL22", WALTZ_RULES, suffixes=WALTZ_SUFFIXES
    ) is None


def test_a_missing_protein_is_simply_absent(tmp_path):
    _touch(tmp_path, "RPL22.dat")

    found = find_protein_files(
        tmp_path, ["RPL22", "RPL29"], WALTZ_RULES, suffixes=WALTZ_SUFFIXES
    )

    assert set(found) == {"RPL22"}


def test_foldamyloid_finds_a_bare_rename(tmp_path):
    """FoldAmyloid's server offers no filename, so this is the common case."""
    directory = tmp_path / "foldamyloid"
    directory.mkdir()
    (directory / "RPL22.csv").write_text("Num,Res,Fold,Value\n", encoding="utf-8")

    found = find_protein_files(
        directory, ["RPL22"], FOLDAMYLOID_RULES, suffixes=FOLDAMYLOID_SUFFIXES
    )

    assert found["RPL22"].name == "RPL22.csv"


def test_foldamyloid_prefers_its_own_output(tmp_path):
    directory = tmp_path / "foldamyloid"
    directory.mkdir()
    for name in ("RPL22.csv", "RPL22_foldamyloid.csv"):
        (directory / name).write_text("Num,Res,Fold,Value\n", encoding="utf-8")

    path = find_protein_file(
        directory, "RPL22", FOLDAMYLOID_RULES, suffixes=FOLDAMYLOID_SUFFIXES
    )

    assert path.name == "RPL22_foldamyloid.csv"


def test_the_runners_expose_the_widened_discovery(tmp_path):
    """The rules have to be reachable through the protocol the scheduler calls."""
    from aggressor_wrappers.runners.foldamyloid import FoldAmyloidRunner
    from aggressor_wrappers.runners.waltz import WALTZRunner

    waltz_dir = tmp_path / "waltz"
    _touch(waltz_dir, "WaltzJob_1_RPL22.dat")
    fold_dir = tmp_path / "foldamyloid"
    fold_dir.mkdir()
    (fold_dir / "RPL22.csv").write_text("Num,Res,Fold,Value\n", encoding="utf-8")

    assert "RPL22" in WALTZRunner().discover_outputs(waltz_dir, ["RPL22"])
    assert "RPL22" in FoldAmyloidRunner().discover_outputs(fold_dir, ["RPL22"])


# --------------------------------------------------------------------------- #
# A protein no predictor could produce output for is a gap, not a stop.
# --------------------------------------------------------------------------- #


def test_the_merge_survives_a_protein_with_no_parsed_output(tmp_path):
    """_merge_protein raised, and took the whole merge step with it.

    On a seven-protein panel one protein (RPS12, whose saved files turned out
    to be a 120-residue fragment of a 132-residue chain) had no output from
    either tool. The merge of the other six -- already parsed by every
    predictor in the panel -- died with it.
    """
    from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline

    fasta = tmp_path / "panel.faa"
    fasta.write_text(">GOOD\nMAVLKMAVLK\n>GAP\nMAVLKMAVLKMA\n", encoding="utf-8")

    out = tmp_path / "out"
    drop = out / "waltz"
    drop.mkdir(parents=True)
    # A file for GOOD only; GAP has nothing anywhere.
    (drop / "GOOD.dat").write_text(
        "".join(f"{i}\t0.000000\n" for i in range(1, 11)), encoding="utf-8"
    )

    log: list[str] = []
    merged = run_multifasta_pipeline(
        fasta,
        out,
        predictors=["waltz"],
        skip_run=True,
        log=log.append,
    )

    assert set(merged) == {"GOOD"}
    joined = "\n".join(log)
    assert "GAP: NO merged table" in joined
    assert "1 of 2 protein(s) merged" in joined


def test_a_run_where_nothing_merges_still_fails(tmp_path):
    """Reporting a gap is not the same as tolerating an empty run."""
    from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline

    fasta = tmp_path / "panel.faa"
    fasta.write_text(">GAP\nMAVLKMAVLK\n", encoding="utf-8")
    out = tmp_path / "out"
    (out / "waltz").mkdir(parents=True)

    with pytest.raises((RuntimeError, FileNotFoundError)):
        run_multifasta_pipeline(
            fasta, out, predictors=["waltz"], skip_run=True, log=lambda _: None
        )
