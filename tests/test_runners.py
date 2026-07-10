"""Runner tests (mocked subprocess — no live PATH/APPNN execution)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aggressor_wrappers.runners.appnn import APPNNRunner
from aggressor_wrappers.runners.path import PATHRunner

FIXTURES = Path(__file__).parent / "fixtures"
SEQUENCE = "MADKG"


@pytest.fixture
def tiny_fasta(tmp_path: Path) -> Path:
    path = tmp_path / "test.fasta"
    path.write_text(f">test_protein\n{SEQUENCE}\n")
    return path


def test_path_runner_skip_run(tiny_fasta: Path) -> None:
    results = FIXTURES / "path_results.csv"
    runner = PATHRunner(threshold_percentile=75.0)
    result = runner.run(
        fasta=tiny_fasta,
        results_csv=results,
        skip_run=True,
    )
    assert result.length == len(SEQUENCE)
    assert result.spec.score_column == "PATH_score"
    df = result.to_dataframe()
    assert list(df.columns) == ["position", "aa_name", "PATH_score", "PATH_bin"]


def test_path_runner_requires_script_when_not_skip_run(tiny_fasta: Path, tmp_path: Path) -> None:
    missing = tmp_path / "missing_path1.1.py"
    runner = PATHRunner(script=str(missing))
    with pytest.raises(FileNotFoundError, match="PATH script not found"):
        runner.run(fasta=tiny_fasta)


@patch("aggressor_wrappers.runners.path.subprocess.run")
def test_path_runner_invokes_subprocess(mock_run: MagicMock, tiny_fasta: Path, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    work = tmp_path / "path_out"
    work.mkdir()
    results = work / "results.csv"
    results.write_text((FIXTURES / "path_results.csv").read_text())

    fake_script = tmp_path / "path1.1py"
    fake_script.write_text("# stub")

    runner = PATHRunner(script=str(fake_script), python="python3")
    result = runner.run(fasta=tiny_fasta, work_dir=work)

    mock_run.assert_called_once()
    assert result.length == len(SEQUENCE)
    assert mock_run.call_args[0][0][:3] == ["python3", str(fake_script), "-f"]


def test_appnn_runner_skip_run(tiny_fasta: Path) -> None:
    appnn_csv = FIXTURES / "appnn_sample.csv"
    runner = APPNNRunner(score_threshold=0.5)
    result = runner.run(
        fasta=tiny_fasta,
        raw_csv=appnn_csv,
        skip_run=True,
    )
    assert result.binary[1] == 1
    assert result.binary[3] == 1
    df = result.to_dataframe()
    assert list(df.columns) == ["position", "aa_name", "APPNN_score", "APPNN_bin"]


@patch("aggressor_wrappers.runners.appnn.subprocess.run")
def test_appnn_runner_invokes_rscript(mock_run: MagicMock, tiny_fasta: Path, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    out_dir = tmp_path / "APPNN_parsed"
    out_dir.mkdir()
    (out_dir / "test_protein_APPNN.csv").write_text(
        (FIXTURES / "appnn_sample.csv").read_text()
    )

    script = tmp_path / "appnn_converter.R"
    script.write_text("# stub")

    runner = APPNNRunner(rscript="Rscript", converter_script=script)
    result = runner.run(fasta=tiny_fasta, work_dir=tmp_path)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "Rscript"
    assert result.length == len(SEQUENCE)


def test_run_cli_help() -> None:
    from aggressor_wrappers.cli.run import main

    assert main(["--help"]) == 0
