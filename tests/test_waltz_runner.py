"""WALTZ runner and batch integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.predictors.waltz import split_detailed_sections
from aggressor_wrappers.runners.registry import get_runner, list_runners
from aggressor_wrappers.runners.waltz import WALTZRunner

FIXTURES = Path(__file__).parent / "fixtures"
WALTZ_DETAILED = FIXTURES / "waltz_detailed.txt"


def test_waltz_runner_registered() -> None:
    assert "waltz" in list_runners()


def test_split_detailed_sections() -> None:
    if not WALTZ_DETAILED.is_file():
        pytest.skip("waltz detailed fixture missing")
    sections = split_detailed_sections(WALTZ_DETAILED.read_text())
    assert "APP_human" in sections
    assert "RPL27_human" in sections
    assert "Positions\tSequence\tAverage score per residue" in sections["APP_human"]


def test_waltz_runner_split_outputs(tmp_path: Path) -> None:
    if not WALTZ_DETAILED.is_file():
        pytest.skip("waltz detailed fixture missing")

    combined = tmp_path / "waltz_combined.txt"
    combined.write_text(WALTZ_DETAILED.read_text())
    runner = WALTZRunner()
    mapping = runner._split_combined_output(
        combined,
        ["APP_human", "RPL27_human", "RPS6_human"],
        tmp_path,
    )
    assert set(mapping) == {"APP_human", "RPL27_human", "RPS6_human"}
    assert (tmp_path / "APP_human_waltz.txt").is_file()


def test_batch_pipeline_skip_run_waltz(tmp_path: Path) -> None:
    if not WALTZ_DETAILED.is_file():
        pytest.skip("waltz detailed fixture missing")

    proteins_fasta = Path(__file__).resolve().parents[2] / "proteins.fasta"
    if not proteins_fasta.is_file():
        pytest.skip("proteins.fasta missing")

    from aggressor_wrappers.core.fasta import read_fasta

    sequences = read_fasta(proteins_fasta)
    subset = {pid: sequences[pid] for pid in ("APP_human", "RPS6_human")}

    multifasta = tmp_path / "proteins.fasta"
    lines = [f">{pid}\n{seq}" for pid, seq in subset.items()]
    multifasta.write_text("\n".join(lines) + "\n")

    out = tmp_path / "results"
    waltz_work = out / "waltz" / "work" / "batch_1"
    waltz_work.mkdir(parents=True)
    runner = WALTZRunner()
    runner._split_combined_output(
        WALTZ_DETAILED,
        ["APP_human", "RPS6_human"],
        waltz_work,
    )

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["waltz"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"APP_human", "RPS6_human"}
    app_csv = out / "waltz" / "parsed" / "APP_human_waltz.csv"
    assert app_csv.is_file()
    assert any("[WALTZ]" in line for line in logs)


def test_get_runner_waltz_from_config() -> None:
    runner = get_runner("waltz")
    assert isinstance(runner, WALTZRunner)
    assert runner.output_format == "text_long"
