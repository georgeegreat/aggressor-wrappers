"""One model load per batch, not one per protein.

AmyloDeep loads ~3.2 GB of artefacts at startup -- the 2.6 GB frozen encoder,
ESM2-150M, three heads, three calibrators. execute_batch looped over the records
and launched a subprocess each, so a 372-protein sweep paid that 372 times and read
roughly 1.2 TB from disk to do work that needs one load. The CLI already accepts a
multifasta and loops internally with the models resident.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.runners import amylodeep as mod
from aggressor_wrappers.runners.amylodeep import AmyloDeepRunner

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _fasta(tmp_path, ids):
    path = tmp_path / "proteins.faa"
    path.write_text("".join(f">{i}\n{ABETA42}\n" for i in ids), encoding="utf-8")
    return path


def _fake_cli(monkeypatch, *, calls: list, fail: bool = False):
    """Stand in for the tool: write one combined CSV for every -i record."""

    def run(argv, **kwargs):
        calls.append(argv)
        if fail:
            return subprocess.CompletedProcess(argv, 1, "", "boom")
        fasta = Path(argv[argv.index("--input") + 1])
        dest = Path(argv[argv.index("--output") + 1])
        ids = [
            line[1:].strip()
            for line in fasta.read_text(encoding="utf-8").splitlines()
            if line.startswith(">")
        ]
        rows = []
        for pid in ids:
            for i, residue in enumerate(ABETA42, start=1):
                rows.append(
                    {
                        "sequence_id": pid,
                        "sequence_length": len(ABETA42),
                        "window_size": 6,
                        "aggregate": "mean",
                        "heads_used": 5,
                        "residue_number": i,
                        "residue": residue,
                        "probability": 0.1,
                    }
                )
        pd.DataFrame(rows).to_csv(dest, index=False)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(AmyloDeepRunner, "require_available", lambda self: None)


def test_a_whole_batch_is_one_invocation(monkeypatch, tmp_path):
    calls: list = []
    _fake_cli(monkeypatch, calls=calls)
    runner = AmyloDeepRunner(use_compat_shim=False)

    runner.execute_batch(_fasta(tmp_path, [f"P{i}" for i in range(20)]), tmp_path / "w")
    assert len(calls) == 1, (
        f"{len(calls)} invocations for 20 proteins: each one reloads ~3.2 GB of "
        f"artefacts"
    )
    assert "--input" in calls[0], "the multifasta goes in, not one sequence"


def test_every_protein_still_gets_its_own_file(monkeypatch, tmp_path):
    """discover_outputs, the parser and resume all expect <protein_id>.csv."""
    _fake_cli(monkeypatch, calls=[])
    runner = AmyloDeepRunner(use_compat_shim=False)
    ids = ["RPS2", "FXR1", "APP"]

    out_dir = runner.execute_batch(_fasta(tmp_path, ids), tmp_path / "w")
    found = runner.discover_outputs(out_dir, ids)
    assert set(found) == set(ids)
    for pid, path in found.items():
        frame = pd.read_csv(path)
        assert set(frame["sequence_id"]) == {pid}, "rows must not leak between files"
        assert len(frame) == len(ABETA42)


def test_finished_proteins_are_not_recomputed(monkeypatch, tmp_path):
    """A sweep interrupted after three days keeps what it finished."""
    calls: list = []
    _fake_cli(monkeypatch, calls=calls)
    runner = AmyloDeepRunner(use_compat_shim=False)
    ids = ["A", "B", "C", "D"]
    fasta = _fasta(tmp_path, ids)
    work = tmp_path / "w"

    runner.execute_batch(fasta, work)          # first pass: all four
    first_argv = calls[0]
    assert len([x for x in first_argv if x == "--input"]) == 1

    # Simulate a restart with two already done.
    (work / "amylodeep_out" / "C.csv").unlink()
    (work / "amylodeep_out" / "D.csv").unlink()
    calls.clear()
    runner.execute_batch(fasta, work)

    assert len(calls) == 1
    pending = Path(calls[0][calls[0].index("--input") + 1]).read_text()
    assert ">C" in pending and ">D" in pending
    assert ">A" not in pending and ">B" not in pending


def test_a_fully_finished_batch_launches_nothing(monkeypatch, tmp_path):
    calls: list = []
    _fake_cli(monkeypatch, calls=calls)
    runner = AmyloDeepRunner(use_compat_shim=False)
    fasta = _fasta(tmp_path, ["A", "B"])
    work = tmp_path / "w"

    runner.execute_batch(fasta, work)
    calls.clear()
    runner.execute_batch(fasta, work)
    assert calls == [], "nothing left to do must cost nothing"


def test_the_batch_argv_carries_every_configured_flag(monkeypatch, tmp_path):
    calls: list = []
    _fake_cli(monkeypatch, calls=calls)
    runner = AmyloDeepRunner(
        use_compat_shim=False,
        model_dir=tmp_path / "models",
        allow_download=False,
        resolution="residue",
        window_size=6,
        aggregate="mean",
    )
    runner.execute_batch(_fasta(tmp_path, ["A"]), tmp_path / "w")
    argv = calls[0]
    assert argv[argv.index("--model-dir") + 1] == str(tmp_path / "models")
    assert "--no-download" in argv
    assert argv[argv.index("--resolution") + 1] == "residue"
    assert argv[argv.index("--window-size") + 1] == "6"
    assert argv[argv.index("--aggregate") + 1] == "mean"


def test_a_failed_batch_names_the_log(monkeypatch, tmp_path):
    _fake_cli(monkeypatch, calls=[], fail=True)
    runner = AmyloDeepRunner(use_compat_shim=False)
    with pytest.raises(RuntimeError, match="Full captured output"):
        runner.execute_batch(_fasta(tmp_path, ["A"]), tmp_path / "w")


def test_output_without_sequence_id_is_refused_rather_than_mis_split(
    monkeypatch, tmp_path
):
    """Splitting on a missing column would silently assign every row to one
    protein, which is worse than failing."""

    def run(argv, **kwargs):
        dest = Path(argv[argv.index("--output") + 1])
        pd.DataFrame({"residue_number": [1, 2], "probability": [0.1, 0.2]}).to_csv(
            dest, index=False
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mod.subprocess, "run", run)
    monkeypatch.setattr(AmyloDeepRunner, "require_available", lambda self: None)
    runner = AmyloDeepRunner(use_compat_shim=False)
    with pytest.raises(ValueError, match="no sequence_id column"):
        runner.execute_batch(_fasta(tmp_path, ["A"]), tmp_path / "w")


# --- device pass-through ----------------------------------------------------


def test_the_device_reaches_both_argv_builders(monkeypatch, tmp_path):
    calls: list = []
    _fake_cli(monkeypatch, calls=calls)
    runner = AmyloDeepRunner(use_compat_shim=False, device="mps")

    single = runner._argv("MKTFF", tmp_path / "o.csv")
    assert single[single.index("--device") + 1] == "mps"

    runner.execute_batch(_fasta(tmp_path, ["A"]), tmp_path / "w")
    batch = calls[0]
    assert batch[batch.index("--device") + 1] == "mps"


def test_no_device_flag_by_default_so_an_older_amylodeep_still_runs(tmp_path):
    argv = AmyloDeepRunner(use_compat_shim=False)._argv("MKTFF", tmp_path / "o.csv")
    assert "--device" not in argv


def test_a_blank_config_value_omits_the_flag(tmp_path):
    runner = AmyloDeepRunner(use_compat_shim=False, device="")
    assert "--device" not in runner._argv("MKTFF", tmp_path / "o.csv")


def test_an_unknown_device_is_rejected_at_construction():
    with pytest.raises(ValueError, match="device must be cpu"):
        AmyloDeepRunner(device="gpu")


def test_the_shipped_config_pins_the_device_explicitly():
    from aggressor_wrappers.core.config import load_config
    from aggressor_wrappers.runners.registry import _construct

    options = dict(load_config("config.cfg").runners.get("amylodeep", {}))
    runner = _construct(AmyloDeepRunner, options, key="amylodeep")
    assert runner.device == "cpu", "the default must be the published configuration"
