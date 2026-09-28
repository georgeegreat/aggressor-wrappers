"""AmyloDeep: a native crash must say so.

The runner inspected only whether an output file appeared, so a child that died
on SIGSEGV produced "AmyloDeep produced no output" with empty stdout and stderr
-- a crash inside a compiled extension never reaches Python's excepthook, so
there was nothing else to quote. The signal was available in
``proc.returncode`` the whole time and was being discarded.
"""

from __future__ import annotations

import signal
import subprocess
from pathlib import Path

import pytest

from aggressor_wrappers.runners import amylodeep as mod
from aggressor_wrappers.runners.amylodeep import AmyloDeepRunner, _signal_report

SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _died(signal_number: int, *, stdout: str = "", stderr: str = ""):
    """A CompletedProcess as subprocess.run reports a signal death."""
    return subprocess.CompletedProcess(
        args=["amylodeep"], returncode=-signal_number, stdout=stdout, stderr=stderr
    )


def test_segfault_is_named_and_not_blamed_on_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _died(signal.SIGSEGV))
    runner = AmyloDeepRunner(use_compat_shim=False)
    with pytest.raises(RuntimeError) as excinfo:
        runner._run_one(SEQUENCE, tmp_path / "out.csv")

    message = str(excinfo.value)
    assert "SIGSEGV" in message
    assert "produced no output" not in message
    # The wrong conclusion this exists to prevent.
    assert "NOT an out-of-memory" in message
    assert "Adding RAM will not change this" in message


def test_sigkill_reads_as_memory_and_sigsegv_does_not():
    """The two must not give the same advice: only one of them is about RAM."""
    segv = _signal_report(signal.SIGSEGV, SEQUENCE, "")
    kill = _signal_report(signal.SIGKILL, SEQUENCE, "")
    assert "out-of-memory kill" in kill
    assert "NOT an out-of-memory" in segv
    assert "OpenMP" in segv and "OpenMP" not in kill


def test_report_names_the_two_stacks_that_collide():
    """The leading hypothesis is specific to this tool, so it is stated.

    AmyloDeep is the only predictor in the panel that loads a PyTorch model
    (ESM2) and a JAX model (UniRep, via jax_unirep) into one process.
    """
    message = _signal_report(signal.SIGSEGV, SEQUENCE, "")
    assert "ESM2" in message and "jax_unirep" in message
    assert "KMP_DUPLICATE_LIB_OK" in message


def test_report_does_not_blame_sequence_length():
    """Per the preprint, scoring is a running window (default 10), no stated max.

    So a long protein is many small inferences, and length is the wrong first
    suspect. The report says so and names the experiment that settles it.
    """
    message = _signal_report(signal.SIGSEGV, "A" * 334, "")
    assert "RUNNING WINDOW" in message
    assert "10-mer" in message
    assert "334 residues" in message  # still reported, as a fact rather than a cause


def test_captured_output_is_quoted_when_there_is_any(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: _died(signal.SIGSEGV, stderr="libomp.dylib already initialized"),
    )
    with pytest.raises(RuntimeError, match="libomp.dylib already initialized"):
        AmyloDeepRunner(use_compat_shim=False)._run_one(SEQUENCE, tmp_path / "out.csv")


def test_a_clean_nonzero_exit_still_uses_the_existing_diagnostics(monkeypatch, tmp_path):
    """Signal handling must not shadow the Hub and pkg_resources cases."""
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            args=["amylodeep"], returncode=1,
            stdout="", stderr="could not reach huggingface.co",
        ),
    )
    with pytest.raises(RuntimeError, match="model weights"):
        AmyloDeepRunner(use_compat_shim=False)._run_one(SEQUENCE, tmp_path / "out.csv")


def test_amylodeep_is_available_to_the_panel_but_does_not_vote_by_default():
    """Registered as a parser, absent from the panel -- and the two are separate.

    Registering a parser makes a tool *scoreable*, so its agreement with the
    biophysical panel can be measured and attributed. The config decides whether
    it votes. Keeping those apart is what lets a pLM tool be studied without
    silently moving every tier of a fractional consensus, and it is the pair of
    facts most at risk of drifting into each other.
    """
    from aggressor_wrappers.core.config import default_pipeline_predictors
    from aggressor_wrappers.predictors.registry import get_parser, list_parsers

    assert "amylodeep" in list_parsers()
    assert type(get_parser("amylodeep")).__name__ == "AmyloDeepParser"
    assert "amylodeep" not in default_pipeline_predictors()


def test_parser_projects_and_binarises_a_window_table(tmp_path):
    """The parser is the place the 0-based window table becomes a residue track."""
    from aggressor_wrappers.predictors.registry import get_parser

    window = 6
    n_windows = len(SEQUENCE) - window + 1
    rows = ["sequence_id,position,probability,sequence_length"]
    rows += [
        # 0-based window start 15 covers 0-based 15..20, i.e. 1-based 16..21 --
        # KLVFFA, the segment shown to be necessary and sufficient for fibril
        # formation (Tjernberg et al., 1996, JBC 271:8545). Chosen so the
        # off-by-one this test exists to catch is visible as a wrong PEPTIDE, not
        # just a wrong index.
        f"P,{i},{0.9 if i == 15 else 0.01},{len(SEQUENCE)}" for i in range(n_windows)
    ]
    raw = tmp_path / "amylodeep.csv"
    raw.write_text("\n".join(rows) + "\n")

    result = get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)
    assert len(result.scores) == len(SEQUENCE)
    assert result.metadata["granularity"] == "window"
    assert result.metadata["window_size"] == window
    called = [i + 1 for i, flag in enumerate(result.binary) if flag]
    assert called == list(range(16, 22))
    assert "".join(SEQUENCE[i - 1] for i in called) == "KLVFFA"


def test_parser_refuses_a_file_for_a_different_protein(tmp_path):
    from aggressor_wrappers.predictors.registry import get_parser

    raw = tmp_path / "amylodeep.csv"
    raw.write_text("sequence_id,position,probability,sequence_length\nP,0,0.9,12\n")
    with pytest.raises(ValueError, match="probably for a different protein"):
        get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)


# --- AmyloDeep 0.4 changed both the column names and the grain ---------------
#
# 0.3 wrote one row per window with a 0-based `position`; 0.4 writes one row per
# residue with a 1-based `residue_number` by default, and its window table names
# the start `window_start_0based`. A parser that only knew `position` read
# neither, so the tool became unreadable the moment it was upgraded -- and the
# runner, passing no --resolution, inherited whichever default was installed.


def _residue_table_04(path, scores, *, window=6, aggregate="mean", heads=5):
    """A residue table as amylodeep 0.4 writes it by default."""
    header = (
        "sequence_id,sequence_length,window_size,aggregate,heads_used,"
        "avg_probability,max_probability,residue_number,residue,probability,"
        "coverage_depth,window_min,window_mean,window_max,window_support,"
        "window_std,window_at_start"
    )
    avg = sum(scores) / len(scores)
    rows = [header]
    for index, value in enumerate(scores, start=1):
        depth = min(window, index, len(scores) - index + 1)
        rows.append(
            f"P,{len(scores)},{window},{aggregate},{heads},{avg:.4f},"
            f"{max(scores)},{index},{SEQUENCE[index - 1]},{value},{depth},"
            f"{min(scores)},{value},{max(scores)},0.5,0.1,{value}"
        )
    path.write_text("\n".join(rows) + "\n")
    return path


def test_parser_reads_the_04_residue_table(tmp_path):
    """KLVFFA again, this time already projected by the tool."""
    from aggressor_wrappers.predictors.registry import get_parser

    scores = [0.01] * len(SEQUENCE)
    for index in range(15, 21):          # 0-based 15..20 == 1-based 16..21
        scores[index] = 0.9
    raw = _residue_table_04(tmp_path / "amylodeep.csv", scores)

    result = get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)
    assert len(result.scores) == len(SEQUENCE)
    assert result.metadata["granularity"] == "per_residue"
    assert result.metadata["index_column"] == "residue_number"
    assert result.metadata["position_base"] == 1
    called = [i + 1 for i, flag in enumerate(result.binary) if flag]
    assert "".join(SEQUENCE[i - 1] for i in called) == "KLVFFA"


def test_an_already_projected_table_is_not_projected_again(tmp_path):
    """Projecting twice would widen every boundary by window_size - 1 residues.

    The 0.4 table carries window_size 6 because the TOOL aggregated over six
    windows. Reporting that as this parser's own window size would be wrong in the
    other direction, so the metadata records both the size and who applied it.
    """
    from aggressor_wrappers.predictors.registry import get_parser

    scores = [0.01] * len(SEQUENCE)
    scores[20] = 0.9                     # one residue only
    raw = _residue_table_04(tmp_path / "amylodeep.csv", scores)

    result = get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)
    called = [i + 1 for i, flag in enumerate(result.binary) if flag]
    assert called == [21], "a projected table must pass through unchanged"
    assert result.metadata["window_size"] == 6
    assert result.metadata["projected_by"] == "amylodeep"
    assert result.metadata["aggregation"] == "mean"


def test_the_two_projections_are_distinguishable_in_the_metadata(tmp_path):
    """Which rule produced the numbers decides how far a shoulder can be trusted,
    so it must be readable off the result rather than inferred from the version."""
    from aggressor_wrappers.predictors.registry import get_parser

    window = 6
    n_windows = len(SEQUENCE) - window + 1
    rows = ["sequence_id,position,probability,sequence_length"]
    rows += [
        f"P,{i},{0.9 if i == 15 else 0.01},{len(SEQUENCE)}" for i in range(n_windows)
    ]
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("\n".join(rows) + "\n")

    projected = _residue_table_04(tmp_path / "new.csv", [0.01] * len(SEQUENCE))

    old = get_parser("amylodeep").parse(legacy, protein_id="P", sequence=SEQUENCE)
    new = get_parser("amylodeep").parse(projected, protein_id="P", sequence=SEQUENCE)
    assert old.metadata["projected_by"] == "aggressor-wrappers"
    assert old.metadata["aggregation"] == "max_over_covering_windows"
    assert new.metadata["projected_by"] == "amylodeep"
    assert new.metadata["aggregation"] == "mean"


def test_a_four_head_run_is_recorded(tmp_path):
    from aggressor_wrappers.predictors.registry import get_parser

    raw = _residue_table_04(tmp_path / "a.csv", [0.01] * len(SEQUENCE), heads=4)
    result = get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)
    assert result.metadata["heads_used"] == 4


def test_coverage_depth_range_is_carried(tmp_path):
    """A terminal residue rests on fewer windows than a mid-chain one."""
    from aggressor_wrappers.predictors.registry import get_parser

    raw = _residue_table_04(tmp_path / "a.csv", [0.01] * len(SEQUENCE), window=6)
    result = get_parser("amylodeep").parse(raw, protein_id="P", sequence=SEQUENCE)
    assert result.metadata["coverage_depth_min"] == 1
    assert result.metadata["coverage_depth_max"] == 6


def test_an_unknown_column_layout_is_refused_by_name(tmp_path):
    from aggressor_wrappers.predictors.amylodeep import parse_amylodeep

    raw = tmp_path / "a.csv"
    raw.write_text("sequence_id,residue_index,probability\nP,1,0.9\n")
    with pytest.raises(ValueError, match="no position column"):
        parse_amylodeep(raw, sequence=SEQUENCE)


# --- The runner asks for the grain instead of inheriting it ------------------


def test_runner_requests_the_residue_grain_by_default(tmp_path):
    argv = AmyloDeepRunner(use_compat_shim=False)._argv(SEQUENCE, tmp_path / "o.csv")
    assert "--resolution" in argv
    assert argv[argv.index("--resolution") + 1] == "residue"
    assert argv[-1] == SEQUENCE, "the sequence stays the last positional argument"


def test_window_and_aggregate_reach_the_command_line(tmp_path):
    runner = AmyloDeepRunner(use_compat_shim=False, window_size=6, aggregate="mean")
    argv = runner._argv(SEQUENCE, tmp_path / "o.csv")
    assert argv[argv.index("--window-size") + 1] == "6"
    assert argv[argv.index("--aggregate") + 1] == "mean"


def test_flags_can_be_omitted_for_an_older_installation(tmp_path):
    runner = AmyloDeepRunner(use_compat_shim=False, resolution=None)
    argv = runner._argv(SEQUENCE, tmp_path / "o.csv")
    assert "--resolution" not in argv


def test_a_bad_option_value_is_rejected_at_construction():
    with pytest.raises(ValueError, match="resolution must be"):
        AmyloDeepRunner(resolution="per-residue")
    with pytest.raises(ValueError, match="aggregate must be"):
        AmyloDeepRunner(aggregate="average")


def test_an_argparse_rejection_reads_as_a_version_mismatch(monkeypatch, tmp_path):
    """The failure mode this replaces: a flag the installed version does not have
    produced no output file and therefore "AmyloDeep produced no output", which
    points at the model rather than at the argument list."""
    rejected = subprocess.CompletedProcess(
        args=["amylodeep"],
        returncode=2,
        stdout="",
        stderr="amylodeep: error: unrecognized arguments: --resolution residue",
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: rejected)
    runner = AmyloDeepRunner(use_compat_shim=False)
    with pytest.raises(RuntimeError) as excinfo:
        runner._run_one(SEQUENCE, tmp_path / "out.csv")
    message = str(excinfo.value)
    assert "predates" in message
    assert "0.4" in message
    assert "resolution" in message


def test_the_shipped_config_sets_the_grain_explicitly():
    """The registry only WARNS about a key no runner accepts, so a misspelled
    option here would be a silent revert to whatever the installed CLI defaults
    to. Pin the three keys against the constructor that consumes them."""
    from aggressor_wrappers.core.config import load_config
    from aggressor_wrappers.runners.registry import _construct

    options = dict(load_config("config.cfg").runners.get("amylodeep", {}))
    assert options["resolution"] == "residue"
    assert options["window_size"] == 6
    assert options["aggregate"] == "mean"

    runner = _construct(AmyloDeepRunner, options, key="amylodeep")
    assert runner.resolution == "residue"
    assert runner.window_size == 6
    assert runner.aggregate == "mean"


def test_blank_config_values_omit_the_flags_rather_than_failing():
    """config.cfg tells the reader to blank the three keys for an older CLI, and a
    blanked ini key arrives as an empty string, not as None."""
    runner = AmyloDeepRunner(
        use_compat_shim=False, resolution="", aggregate="", window_size=""
    )
    argv = runner._argv(SEQUENCE, Path("/tmp/o.csv"))
    assert "--resolution" not in argv
    assert "--aggregate" not in argv
    assert "--window-size" not in argv
