"""Scratch files are removed; the deliverable is not.

AmyloGram writes a peptide FASTA per query width and a generated R script, then
scores them. None of it was ever removed: run() built a mkdtemp with no owner, and
the batch path created one directory per protein and left it. On a 372-protein
sweep that is 372 directories of intermediates, and the disk grows with the number
of proteins already finished rather than staying flat.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.runners.amylogram import AmyloGramRunner

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _fasta(tmp_path, ids=("P1", "P2")):
    path = tmp_path / "proteins.fasta"
    path.write_text("".join(f">{i}\n{ABETA42}\n" for i in ids), encoding="utf-8")
    return path


def _fake_scoring(monkeypatch, runner):
    """Score without R: write the CSV the parser expects and nothing else."""
    import aggressor_wrappers.runners.amylogram as mod

    def _score(self, peptides_fasta: Path, work: Path) -> Path:
        # Keyed on the peptide FASTA's own headers and written with the column
        # names the parser accepts (name, probability). A first version used
        # "seq,prob", which the parser rejects -- a fixture bug that failed five
        # tests and looked like a cleanup bug.
        names = [
            line[1:].strip()
            for line in peptides_fasta.read_text(encoding="utf-8").splitlines()
            if line.startswith(">")
        ]
        out_csv = work / f"{peptides_fasta.stem}_amylogram.csv"
        rows = ["name,probability"] + [f"{name},0.1" for name in names]
        out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return out_csv

    monkeypatch.setattr(mod.AmyloGramRunner, "_score_peptides", _score)
    monkeypatch.setattr(mod.AmyloGramRunner, "require_available", lambda self: None)


def test_execute_batch_leaves_only_the_published_outputs(monkeypatch, tmp_path):
    runner = AmyloGramRunner(write_per_residue=False)
    _fake_scoring(monkeypatch, runner)

    out_dir = runner.execute_batch(_fasta(tmp_path), tmp_path / "work")
    entries = sorted(p.name for p in out_dir.iterdir())
    assert entries == ["P1.csv", "P2.csv"], (
        f"scratch left behind: {entries}. Only the published CSVs are results."
    )
    assert not [p for p in out_dir.iterdir() if p.is_dir()]


def test_discover_outputs_finds_what_execute_batch_published(monkeypatch, tmp_path):
    """Cleanup must not remove the thing the scheduler then looks for."""
    runner = AmyloGramRunner(write_per_residue=False)
    _fake_scoring(monkeypatch, runner)

    out_dir = runner.execute_batch(_fasta(tmp_path), tmp_path / "work")
    found = runner.discover_outputs(out_dir, ["P1", "P2"])
    assert set(found) == {"P1", "P2"}
    assert all(path.is_file() for path in found.values())


def test_keep_intermediates_preserves_them_for_one_run(monkeypatch, tmp_path):
    runner = AmyloGramRunner(write_per_residue=False, keep_intermediates=True)
    _fake_scoring(monkeypatch, runner)

    out_dir = runner.execute_batch(_fasta(tmp_path, ids=("P1",)), tmp_path / "work")
    assert (out_dir / "P1").is_dir(), "opted in, so the scratch must survive"
    assert any((out_dir / "P1").glob("*windows.fasta"))


def test_a_caller_supplied_work_dir_is_never_removed(monkeypatch, tmp_path):
    """It belongs to the caller. Only a directory this call created is this call's
    to delete -- the distinction the leaking mkdtemp never made."""
    runner = AmyloGramRunner(write_per_residue=False)
    _fake_scoring(monkeypatch, runner)

    work = tmp_path / "mine"
    work.mkdir()
    runner.run(fasta=_fasta(tmp_path, ids=("P1",)), protein_id="P1", work_dir=work)
    assert work.is_dir()
    assert any(work.iterdir()), "the caller's directory keeps its contents"


def test_a_temporary_work_dir_is_removed(monkeypatch, tmp_path):
    runner = AmyloGramRunner(write_per_residue=False)
    _fake_scoring(monkeypatch, runner)

    result = runner.run(fasta=_fasta(tmp_path, ids=("P1",)), protein_id="P1")
    assert result.protein_id == "P1"
    source = Path(result.metadata["source"])
    assert not source.parent.exists(), (
        f"{source.parent} survived the call that created it"
    )


def test_scratch_is_removed_even_when_scoring_fails(monkeypatch, tmp_path):
    """A failing protein must not leave its directory behind either, or a sweep
    with a few bad sequences accumulates exactly the directories worth cleaning."""
    import aggressor_wrappers.runners.amylogram as mod

    monkeypatch.setattr(mod.AmyloGramRunner, "require_available", lambda self: None)
    monkeypatch.setattr(
        mod.AmyloGramRunner,
        "_score_peptides",
        lambda self, peptides, work: (_ for _ in ()).throw(RuntimeError("R died")),
    )
    runner = AmyloGramRunner(write_per_residue=False)
    with pytest.raises(RuntimeError, match="R died"):
        runner.run(fasta=_fasta(tmp_path, ids=("P1",)), protein_id="P1")

    leftovers = [p for p in Path(tmp_path).iterdir() if p.name.startswith("amylogram_")]
    assert not leftovers
