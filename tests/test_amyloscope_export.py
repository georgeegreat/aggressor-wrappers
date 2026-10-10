"""Bridge from merged predictor tables to amyloscope inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggressor_wrappers.core.amyloscope_export import export_run


def _wide() -> pd.DataFrame:
    hot = np.array([False] * 10 + [True] * 10)
    df = pd.DataFrame(
        {
            "waltz_score": np.where(hot, 90.0, 20.0),
            "APPNN_score": np.where(hot, 0.9, 0.1),
            "pasta_score": np.where(hot, -7.0, -1.0),
        }
    )
    df["waltz_bin"] = (df.waltz_score >= 73).astype(int)
    df["APPNN_bin"] = (df.APPNN_score >= 0.5).astype(int)
    df["pasta_bin"] = (df.pasta_score < -2.8).astype(int)
    return df

# --------------------------------------------------------------------------- #
# amyloscope bridge
# --------------------------------------------------------------------------- #


def test_export_preserves_pasta_polarity(tmp_path: Path):
    """The exported APR flag must be true where PASTA is most negative."""
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST.csv", index=False)

    out = tmp_path / "amylo"
    config = export_run(merged, out)
    assert config.exists()

    track = pd.read_csv(out / "tracks" / "pasta" / "TEST.csv")
    assert list(track.columns) == ["Number", "Residue", "Score", "APR"]
    apr = track["APR"].astype(bool).to_numpy()
    # residues 11-20 carry the strong (negative) PASTA signal
    assert apr[10:].all()
    assert not apr[:10].any()


def test_generated_config_uses_flag_detection_and_available_denominator(tmp_path: Path):
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST.csv", index=False)

    text = export_run(merged, tmp_path / "amylo").read_text()
    # thresholds/polarity are NOT restated downstream: one source of truth
    assert "method: flag, column: APR" in text
    # a dead predictor must not make a tier unreachable
    assert "denominator: available" in text




# --- The CLI handoff --------------------------------------------------------
#
# export_run existed and passed its tests while being reachable only from
# Python, and its tests used <protein_id>.csv fixtures while the pipeline writes
# <protein_id>_merged.csv -- so the naming mismatch below was invisible.

import pandas as pd
import pytest

from aggressor_wrappers.core.amyloscope_export import protein_id_from_merged

SEQ = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _pipeline_run(root, protein_ids=("RPS2", "FXR1")):
    """A batch run's merged directory, named the way the pipeline names it."""
    merged = root / "output" / "merged"
    merged.mkdir(parents=True)
    n = len(SEQ)
    for pid in protein_ids:
        pd.DataFrame(
            {
                "position": range(1, n + 1),
                "aa_name": list(SEQ),
                "aggrescan_score": [0.1] * n,
                "aggrescan_bin": [1 if 15 <= i < 21 else 0 for i in range(n)],
                "pasta_score": [-2.0] * n,
                "pasta_bin": [1 if 15 <= i < 21 else 0 for i in range(n)],
            }
        ).to_csv(merged / f"{pid}_merged.csv", index=False)
    return root / "output"


@pytest.mark.parametrize(
    "stem,expected",
    [("RPS2_merged", "RPS2"), ("RPS2", "RPS2"), ("FXR1_merged_merged", "FXR1_merged")],
)
def test_protein_id_strips_only_the_pipeline_suffix(tmp_path, stem, expected):
    assert protein_id_from_merged(tmp_path / f"{stem}.csv") == expected


def test_exported_ids_match_the_proteins_not_the_filenames(tmp_path):
    """amyloscope matches tracks to proteins by id, so 'RPS2_merged' would drop
    the protein from the analysis rather than fail."""
    from aggressor_wrappers.cli.export_amyloscope import main

    run = _pipeline_run(tmp_path)
    assert main([str(run / "merged"), "-o", str(tmp_path / "amylo")]) == 0

    tracks = tmp_path / "amylo" / "tracks" / "aggrescan"
    assert sorted(p.name for p in tracks.iterdir()) == ["FXR1.csv", "RPS2.csv"]
    config = (tmp_path / "amylo" / "config.yaml").read_text()
    assert "id: RPS2\n" in config
    assert "_merged" not in config


def test_the_run_directory_is_accepted_as_well_as_merged(tmp_path):
    """Pointing at <output>/ is the obvious mistake; the CSVs are one level down."""
    from aggressor_wrappers.cli.export_amyloscope import main

    run = _pipeline_run(tmp_path)
    assert main([str(run), "-o", str(tmp_path / "amylo")]) == 0
    assert (tmp_path / "amylo" / "config.yaml").is_file()


def test_an_inverted_polarity_tool_crosses_as_a_flag_not_a_threshold(tmp_path):
    """PASTA's energy is amyloidogenic when LOW. The parser already applied that
    direction, so what crosses is the call, and amyloscope needs no per-tool cut."""
    from aggressor_wrappers.cli.export_amyloscope import main

    run = _pipeline_run(tmp_path)
    main([str(run), "-o", str(tmp_path / "amylo")])

    track = pd.read_csv(tmp_path / "amylo" / "tracks" / "pasta" / "RPS2.csv")
    assert track.loc[track["APR"], "Number"].tolist() == list(range(16, 22))
    assert (track["Score"] < 0).all(), "native scale is carried through unchanged"
    assert "detection: {method: flag, column: APR" in (
        tmp_path / "amylo" / "config.yaml"
    ).read_text()


def test_an_empty_merged_directory_fails_with_the_reason(tmp_path):
    from aggressor_wrappers.cli.export_amyloscope import main

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty), "-o", str(tmp_path / "amylo")]) == 2


def test_the_subcommand_is_reachable_from_the_root_cli(tmp_path):
    """A handoff nobody can invoke is an integration that works only in pytest."""
    from aggressor_wrappers.cli.app import main as root_main
    from aggressor_wrappers.cli.batch import is_subcommand

    assert is_subcommand(["export-amyloscope"])
    run = _pipeline_run(tmp_path)
    assert root_main(["export-amyloscope", str(run), "-o", str(tmp_path / "a")]) == 0


# --- The config has to name things amyloscope recognises --------------------
#
# A 372-protein export reported success and then failed inside amyloscope. Four
# separate things were wrong, and each one alone is enough to break a run:
# the adapter named nothing, the tool names were display names where amyloscope
# keys on canonical ones, the printed command used -c for a positional argument,
# and tracks from an earlier export were still in the directory.

from aggressor_wrappers.cli.export_amyloscope import main
from aggressor_wrappers.core.amyloscope_export import (
    AMYLOSCOPE_TOOL_NAMES,
    NORMALISED_ADAPTER,
    REQUIRED_TRACK_COLUMNS,
    verify_export,
)


def test_tool_names_are_amyloscopes_canonical_keys_not_display_names(tmp_path):
    """amyloscope keys MECHANISM_FAMILIES and its adapters on lowercase canonical
    names. A tool declared as "Aggrescan" loads but cannot be placed in a family,
    so it silently drops out of the composition that decides whether a tier moves.
    """
    run = _pipeline_run(tmp_path)
    main([str(run), "-o", str(tmp_path / "amylo")])
    config = (tmp_path / "amylo" / "config.yaml").read_text()

    assert "name: aggrescan" in config
    assert "name: Aggrescan" not in config
    assert "name: PASTA" not in config


def test_pasta_is_renamed_because_the_two_packages_spell_it_differently():
    assert AMYLOSCOPE_TOOL_NAMES["pasta"] == "pasta2"


def test_a_predictor_with_no_counterpart_is_skipped_loudly():
    """PATH has no amyloscope adapter and no mechanism family. Writing it into the
    config would name a tool the analysis layer cannot place."""
    assert AMYLOSCOPE_TOOL_NAMES["path"] is None


def test_the_adapter_is_the_one_that_reads_a_normalised_track(tmp_path):
    run = _pipeline_run(tmp_path)
    main([str(run), "-o", str(tmp_path / "amylo")])
    config = (tmp_path / "amylo" / "config.yaml").read_text()
    assert f"adapter: {NORMALISED_ADAPTER}" in config


def test_every_track_carries_the_columns_the_config_reads(tmp_path):
    run = _pipeline_run(tmp_path)
    main([str(run), "-o", str(tmp_path / "amylo")])
    for track in (tmp_path / "amylo" / "tracks").rglob("*.csv"):
        header = pd.read_csv(track, nrows=0).columns.tolist()
        for column in REQUIRED_TRACK_COLUMNS:
            assert column in header, f"{track} lacks {column}"


def test_verification_catches_a_track_from_an_older_export(tmp_path):
    """The actual failure: a track with Number, Residue, Score and no APR, left in
    the directory by a previous version. The export must refuse rather than report
    success and let amyloscope hit a KeyError 372 proteins later."""
    run = _pipeline_run(tmp_path)
    out = tmp_path / "amylo"
    main([str(run), "-o", str(out)])

    stale = out / "tracks" / "aggrescan" / "RPS2.csv"
    pd.DataFrame(
        {"Number": range(1, len(SEQ) + 1), "Residue": list(SEQ), "Score": [0.1] * len(SEQ)}
    ).to_csv(stale, index=False)

    proteins = {"RPS2": {"sequence": SEQ}}
    with pytest.raises(ValueError) as excinfo:
        verify_export(out, ["aggrescan"], proteins)
    message = str(excinfo.value)
    assert "APR" in message
    assert "older export" in message
    assert "--clean" in message


def test_verification_catches_a_length_disagreement(tmp_path):
    run = _pipeline_run(tmp_path)
    out = tmp_path / "amylo"
    main([str(run), "-o", str(out)])

    short = out / "tracks" / "aggrescan" / "RPS2.csv"
    pd.DataFrame(
        {
            "Number": range(1, 11),
            "Residue": list(SEQ)[:10],
            "Score": [0.1] * 10,
            "APR": [0] * 10,
        }
    ).to_csv(short, index=False)

    with pytest.raises(ValueError, match="10 rows against a 42-residue"):
        verify_export(out, ["aggrescan"], {"RPS2": {"sequence": SEQ}})


def test_a_missing_track_is_reported_not_skipped(tmp_path):
    with pytest.raises(ValueError, match="declared in the config but absent"):
        verify_export(tmp_path, ["aggrescan"], {"RPS2": {"sequence": SEQ}})


def test_the_cli_refuses_a_directory_that_already_has_tracks(tmp_path):
    run = _pipeline_run(tmp_path)
    out = tmp_path / "amylo"
    assert main([str(run), "-o", str(out)]) == 0
    assert main([str(run), "-o", str(out)]) == 2, "a second export must not mix"
    assert main([str(run), "-o", str(out), "--clean"]) == 0


def test_the_printed_command_matches_amyloscopes_cli(tmp_path, capsys):
    """`config` is positional in amyloscope's run subcommand, so -c is not a
    command that runs. This was printed to the user as the thing to do next."""
    run = _pipeline_run(tmp_path)
    main([str(run), "-o", str(tmp_path / "amylo")])
    printed = capsys.readouterr().out
    assert "amyloscope run " in printed
    assert "amyloscope run -c" not in printed
    assert "amyloscope validate " in printed


def test_a_merged_table_longer_than_its_sequence_is_refused(tmp_path):
    """The WALTZ symptom: 793 positions against a 792-residue sequence."""
    from aggressor_wrappers.core.amyloscope_export import export_protein

    merged = tmp_path / "RPS2_merged.csv"
    n = len(SEQ) + 1
    pd.DataFrame(
        {
            "position": range(1, n + 1),
            "aa_name": (list(SEQ) * 2)[:n],
            "aggrescan_score": [0.1] * n,
            "aggrescan_bin": [0] * n,
        }
    ).to_csv(merged, index=False)

    with pytest.raises(ValueError) as excinfo:
        export_protein(merged, "RPS2", tmp_path / "out", sequence=SEQ)
    message = str(excinfo.value)
    assert f"{n} rows for a {len(SEQ)}-residue" in message
    assert "duplicate header" in message
    assert "uniq -d" in message, "the error should carry the command that finds it"
