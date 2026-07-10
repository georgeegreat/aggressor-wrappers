"""ArchCandy runner and batch integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.runners.archcandy import ArchCandyRunner
from aggressor_wrappers.runners.registry import get_runner, list_runners

FIXTURES = Path(__file__).parent / "fixtures"
ARCHCANDY_CSV = FIXTURES / "archcandy_app.csv"
APP_SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def test_archcandy_runner_registered() -> None:
    assert "archcandy" in list_runners()


def test_get_runner_archcandy_from_config() -> None:
    runner = get_runner("archcandy")
    assert isinstance(runner, ArchCandyRunner)
    assert runner.threshold == pytest.approx(0.4)
    assert runner.transmembrane is False
    assert runner.score_mode == "cumulative"


def test_batch_pipeline_skip_run_archcandy(tmp_path: Path) -> None:
    if not ARCHCANDY_CSV.is_file():
        pytest.skip("archcandy fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(f">APP\n{APP_SEQUENCE}\n")

    out = tmp_path / "results"
    work = out / "ArchCandy" / "work" / "batch_1"
    work.mkdir(parents=True)
    (work / "APP_archcandy.csv").write_text(ARCHCANDY_CSV.read_text())

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["archcandy"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"APP"}
    assert (out / "ArchCandy" / "parsed" / "APP_ArchCandy.csv").is_file()
    assert any("[ArchCandy]" in line for line in logs)


def test_batch_pipeline_skip_run_archcandy_parallel_work_layout(tmp_path: Path) -> None:
    """--skip-run finds raw CSV under per-protein work dirs (parallel_jobs layout)."""
    if not ARCHCANDY_CSV.is_file():
        pytest.skip("archcandy fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(f">APP\n{APP_SEQUENCE}\n")

    out = tmp_path / "results"
    work = out / "ArchCandy" / "work" / "APP"
    work.mkdir(parents=True)
    (work / "APP_archcandy.csv").write_text(ARCHCANDY_CSV.read_text())

    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["archcandy"],
        skip_run=True,
    )

    assert set(merged) == {"APP"}
    assert (out / "ArchCandy" / "parsed" / "APP_ArchCandy.csv").is_file()
