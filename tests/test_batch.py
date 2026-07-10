"""Batch multifasta pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import chunk_items, run_multifasta_pipeline

FIXTURES = Path(__file__).parent / "fixtures"
SEQUENCE = "MADKG"


def _write_fasta(path: Path, records: dict[str, str]) -> None:
    lines = [f">{pid}\n{seq}" for pid, seq in records.items()]
    path.write_text("\n".join(lines) + "\n")


def test_chunk_items() -> None:
    items = [("a", "M"), ("b", "A"), ("c", "D")]
    assert chunk_items(items, 2) == [[("a", "M"), ("b", "A")], [("c", "D")]]
    assert chunk_items(items, 10) == [items]


def test_chunk_items_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_items([("a", "M")], 0)


def test_batch_pipeline_skip_run_merge(tmp_path: Path) -> None:
    multifasta = tmp_path / "proteins.fasta"
    _write_fasta(multifasta, {"protA": SEQUENCE, "protB": SEQUENCE})

    path_results = FIXTURES / "path_results.csv"
    appnn_a = FIXTURES / "appnn_sample.csv"
    appnn_b = FIXTURES / "appnn_sample.csv"
    if not path_results.is_file() or not appnn_a.is_file():
        pytest.skip("runner fixtures missing")

    out = tmp_path / "results"

    path_batch = out / "PATH" / "work" / "batch_1"
    path_batch.mkdir(parents=True)
    (path_batch / "results.csv").write_text(path_results.read_text())

    path_batch_2 = out / "PATH" / "work" / "batch_2"
    path_batch_2.mkdir(parents=True)
    (path_batch_2 / "results.csv").write_text(path_results.read_text())

    appnn_batch = out / "APPNN" / "work" / "batch_1" / "APPNN_parsed"
    appnn_batch.mkdir(parents=True)
    (appnn_batch / "protA_APPNN.csv").write_text(appnn_a.read_text())
    (appnn_batch / "protB_APPNN.csv").write_text(appnn_b.read_text())

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["path", "appnn"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"protA", "protB"}
    for protein_id in ("protA", "protB"):
        assert merged[protein_id].is_file()
        assert (out / "PATH" / "parsed" / f"{protein_id}_PATH.csv").is_file()
        assert (out / "APPNN" / "parsed" / f"{protein_id}_APPNN.csv").is_file()

    assert any("[PATH]" in line for line in logs)
    assert any("[APPNN]" in line for line in logs)
    assert any("[merge]" in line for line in logs)


def test_batch_unknown_predictor_raises(tmp_path: Path) -> None:
    multifasta = tmp_path / "proteins.fasta"
    _write_fasta(multifasta, {"protA": SEQUENCE})

    with pytest.raises(ValueError, match="Unknown predictor"):
        run_multifasta_pipeline(
            multifasta,
            tmp_path / "results",
            predictors=["path", "not_a_tool"],
            skip_run=True,
        )


def test_batch_help() -> None:
    from aggressor_wrappers.cli.batch import main

    assert main(["--help"]) == 0


def test_aggressor_wrappers_script_help() -> None:
    from aggressor_wrappers.cli.batch import run_pipeline

    assert run_pipeline(["--help"], prog="aggressor-wrappers.py") == 0


def test_app_dispatch_batch_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["batch", "--help"]) == 0
