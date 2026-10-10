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
    from aggressor_wrappers.predictors.registry import get_parser, list_parsers

    assert "amylodeep" in list_parsers()
    assert type(get_parser("amylodeep")).__name__ == "AmyloDeepParser"

    # The separation is that REGISTRATION does not confer a vote -- a property of
    # the code, which a config cannot change. Asserting amylodeep's absence from
    # the shipped roster tested the opposite: it made a legitimate decision to
    # include it (an orthologue sweep needs a local panel) look like a regression.
    from aggressor_wrappers.core.config import default_pipeline_predictors
    from aggressor_wrappers.runners.registry import list_runners

    # Every voter must be REACHABLE, by one of two routes: a registered parser, or
    # a runner that writes the standard per-residue CSV itself and so never needs
    # one (AmyloGram does exactly that, which is why "every voter has a parser" is
    # the wrong invariant and failed on a correct configuration).
    panel = default_pipeline_predictors()
    reachable = set(list_parsers()) | set(list_runners())
    unreachable = [key for key in panel if key not in reachable]
    assert not unreachable, (
        f"panel member(s) {unreachable} have neither a parser nor a runner, so the "
        f"run fails after the predictors have already executed"
    )
    # And the separation itself: more tools are registered than vote.
    assert len(reachable) > len(panel), (
        "if every registered tool voted, the opt-in distinction would not exist"
    )


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


def test_model_dir_and_offline_flag_reach_the_command_line(tmp_path):
    """Which copy of the 3 GB of weights a run used is part of its provenance."""
    runner = AmyloDeepRunner(
        use_compat_shim=False, model_dir=tmp_path / "models", allow_download=False
    )
    argv = runner._argv(SEQUENCE, tmp_path / "o.csv")
    assert argv[argv.index("--model-dir") + 1] == str(tmp_path / "models")
    assert "--no-download" in argv


def test_a_model_dir_that_does_not_exist_yet_is_accepted():
    """The tool downloads INTO that directory, so refusing an absent path would
    break first use -- the failure belongs to the tool, which names what is
    missing, not to the runner's constructor."""
    runner = AmyloDeepRunner(use_compat_shim=False, model_dir="/nonexistent/models")
    assert runner.model_dir == Path("/nonexistent/models")


def test_a_tilde_in_the_model_dir_is_expanded(tmp_path):
    """A config file is text: ~ arrives literally and would create a directory
    called '~' beside the working directory rather than resolving to $HOME."""
    runner = AmyloDeepRunner(use_compat_shim=False, model_dir="~/amylodeep_models")
    assert "~" not in str(runner.model_dir)
    assert runner.model_dir.is_absolute()


def test_download_is_allowed_by_default():
    assert AmyloDeepRunner(use_compat_shim=False).allow_download is True
    assert "--no-download" not in AmyloDeepRunner(use_compat_shim=False)._argv(
        SEQUENCE, Path("/tmp/o.csv")
    )


def test_an_absolute_executable_path_satisfies_the_availability_check(tmp_path):
    """A conda env is reached by pointing `executable` into its bin/ without
    activating it, which only works if availability accepts an absolute path."""
    fake = tmp_path / "bin" / "amylodeep"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    assert AmyloDeepRunner(executable=str(fake), use_compat_shim=False).is_available()
    assert not AmyloDeepRunner(
        executable=str(tmp_path / "bin" / "absent"), use_compat_shim=False
    ).is_available()


def test_the_shipped_config_pins_the_model_directory():
    from aggressor_wrappers.core.config import load_config
    from aggressor_wrappers.runners.registry import _construct

    options = dict(load_config("config.cfg").runners.get("amylodeep", {}))
    runner = _construct(AmyloDeepRunner, options, key="amylodeep")
    assert runner.model_dir is not None, "model_dir must be pinned, not inherited"
    assert runner.model_dir.is_absolute()


# --- The compat shim's availability check asked the wrong question -----------
#
# With use_compat_shim on (the default), _argv builds [python, bootstrap, ...] and
# the console script is never called. is_available() checked the console script,
# so an amylodeep installed in its own conda env satisfied the gate while the shim
# ran under whichever python3 came first -- typically the system one, which cannot
# import the package. The run then died with a bare ModuleNotFoundError from inside
# a generated file, after the panel was already running.


def _interpreter(tmp_path, *, importable: bool):
    """A fake python whose `-c "import amylodeep"` succeeds or fails."""
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit %d\n" % (0 if importable else 1))
    fake.chmod(0o755)
    return fake


def test_availability_checks_the_interpreter_when_the_shim_is_used(tmp_path):
    good = _interpreter(tmp_path / "good", importable=True)
    bad = _interpreter(tmp_path / "bad", importable=False)
    assert AmyloDeepRunner(python=str(good), use_compat_shim=True).is_available()
    assert not AmyloDeepRunner(python=str(bad), use_compat_shim=True).is_available()


def test_availability_checks_the_executable_when_the_shim_is_off(tmp_path):
    """Without the shim the console script IS what runs, so check that instead."""
    script = tmp_path / "bin" / "amylodeep"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    runner = AmyloDeepRunner(executable=str(script), use_compat_shim=False)
    assert runner.is_available()


def test_the_refusal_names_the_interpreter_and_the_config_key(tmp_path):
    """An interpreter that cannot import it must point at `python`, not
    `executable` -- setting the latter has no effect while the shim is on."""
    bad = _interpreter(tmp_path, importable=False)
    runner = AmyloDeepRunner(python=str(bad), use_compat_shim=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        runner.require_available()
    message = str(excinfo.value)
    assert str(bad) in message
    assert "python =" in message
    assert "has no effect" in message


def test_a_module_not_found_at_runtime_is_explained(monkeypatch, tmp_path):
    """Belt and braces: if the interpreter changes between the check and the run,
    the symptom still has to name the cause rather than quote a traceback."""
    failed = subprocess.CompletedProcess(
        args=["python", "_run_amylodeep.py"],
        returncode=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "_run_amylodeep.py", line 39, in <module>\n'
            "    from amylodeep.cli import main\n"
            "ModuleNotFoundError: No module named 'amylodeep'"
        ),
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: failed)
    runner = AmyloDeepRunner(use_compat_shim=True)
    with pytest.raises(RuntimeError) as excinfo:
        runner._run_one(SEQUENCE, tmp_path / "out.csv")
    message = str(excinfo.value)
    assert "cannot import amylodeep" in message
    assert "[runners.amylodeep]" in message


def test_the_backend_locator_agrees_with_the_runner(tmp_path):
    """The backend policy decided availability from `executable` too, which is how
    a run got past the gate in the first place. Both must ask one question."""
    from aggressor_wrappers.runners.registry import _amylodeep_local_tool

    with_shim = _amylodeep_local_tool({"python": "/envs/amylodeep/bin/python"})
    assert with_shim.executable == "/envs/amylodeep/bin/python"

    without = _amylodeep_local_tool(
        {"use_compat_shim": False, "executable": "/envs/amylodeep/bin/amylodeep"}
    )
    assert without.executable == "/envs/amylodeep/bin/amylodeep"


def test_a_blank_use_compat_shim_string_from_ini_is_read_as_false():
    """configparser hands through strings; 'false' must not be truthy."""
    from aggressor_wrappers.runners.registry import _amylodeep_local_tool

    tool = _amylodeep_local_tool({"use_compat_shim": "false", "executable": "ad"})
    assert tool.executable == "ad"


# --- A broken jax must not stop the tool before it starts -------------------
#
# The compat shim applies a WORKAROUND for a jax_unirep bug, so failing to apply it
# cannot be fatal. The first version caught only ImportError, and jax's own jaxlib
# version check raises RuntimeError -- so a mismatched jax/jaxlib pair killed the
# bootstrap before AmyloDeep ran, and a problem confined to ONE of five heads
# presented as a total failure.

JAXLIB_MISMATCH = (
    "RuntimeError: jaxlib is version 0.10.2, but this version of jax requires "
    "version >= 0.11.2."
)


def test_a_jax_jaxlib_mismatch_is_named_with_the_paired_install(monkeypatch, tmp_path):
    failed = subprocess.CompletedProcess(
        args=["python", "_run_amylodeep.py"],
        returncode=1,
        stdout="",
        stderr=JAXLIB_MISMATCH,
    )
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: failed)
    runner = AmyloDeepRunner(use_compat_shim=True)
    with pytest.raises(RuntimeError) as excinfo:
        runner._run_one(SEQUENCE, tmp_path / "out.csv")
    message = str(excinfo.value)
    assert "mismatched" in message
    assert "jax[cpu]" in message, "install them as a PAIR, not separately"
    assert "jax_unirep" in message, "the consumer may break on a newer pair"
    assert "four heads" in message, "only one head needs jax"


def test_the_bootstrap_survives_a_jax_that_raises_more_than_importerror():
    """Run the generated bootstrap's _patch() with a jax that raises RuntimeError
    on import, exactly as a mismatched jaxlib does."""
    import re
    import sys
    import types

    from aggressor_wrappers.core.compat import BOOTSTRAP_SOURCE

    namespace: dict = {"__name__": "not_main", "sys": sys}
    exec(compile(BOOTSTRAP_SOURCE, "_run_amylodeep.py", "exec"), namespace)

    class Exploding(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError(
                "jaxlib is version 0.10.2, but this version of jax requires "
                "version >= 0.11.2."
            )

    saved = {k: v for k, v in sys.modules.items() if k.startswith("jax")}
    for key in list(sys.modules):
        if key.startswith("jax"):
            del sys.modules[key]
    sys.modules["jax"] = Exploding("jax")
    sys.modules["jax.numpy"] = Exploding("jax.numpy")
    try:
        namespace["_patch"]()          # must return, not raise
    finally:
        for key in list(sys.modules):
            if key.startswith("jax"):
                del sys.modules[key]
        sys.modules.update(saved)

    assert "_patch" in namespace


def test_the_bootstrap_still_returns_when_jax_is_simply_absent():
    import sys

    from aggressor_wrappers.core.compat import BOOTSTRAP_SOURCE

    namespace: dict = {"__name__": "not_main", "sys": sys}
    exec(compile(BOOTSTRAP_SOURCE, "_run_amylodeep.py", "exec"), namespace)

    saved = {k: v for k, v in sys.modules.items() if k.startswith("jax")}
    for key in list(sys.modules):
        if key.startswith("jax"):
            del sys.modules[key]
    sys.modules["jax"] = None          # import raises ImportError
    try:
        namespace["_patch"]()
    finally:
        for key in list(sys.modules):
            if key.startswith("jax"):
                del sys.modules[key]
        sys.modules.update(saved)
