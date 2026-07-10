"""PASTA runner and batch integration tests."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.runners.pasta import PASTARunner, _encode_multipart
from aggressor_wrappers.runners.registry import get_runner, list_runners

FIXTURES = Path(__file__).parent / "fixtures"
PASTA_PROFILE = FIXTURES / "pasta_test1.dat"
PASTA_ARCHIVE = Path("/tmp/pasta_test.tar.gz")


def test_pasta_runner_registered() -> None:
    assert "pasta" in list_runners()


def test_encode_multipart() -> None:
    body, content_type = _encode_multipart(
        fields={"npair": "22", "amount": "-2.8"},
        files={"mutantfastafile": ("test.fasta", b">a\nAC\n", "text/plain")},
    )
    assert b"npair" in body
    assert b"mutantfastafile" in body
    assert "multipart/form-data" in content_type


def test_pasta_runner_extract_profiles(tmp_path: Path) -> None:
    if not PASTA_ARCHIVE.is_file():
        pytest.skip("pasta test archive missing")

    archive_copy = tmp_path / "batch.tar.gz"
    archive_copy.write_bytes(PASTA_ARCHIVE.read_bytes())
    runner = PASTARunner()
    mapping = runner._extract_profiles(
        archive_copy,
        ["test1", "test2", "test3"],
        tmp_path,
    )
    assert set(mapping) == {"test1", "test2", "test3"}
    assert (tmp_path / "test1_pasta.dat").is_file()


def test_get_runner_pasta_from_config() -> None:
    runner = get_runner("pasta")
    assert isinstance(runner, PASTARunner)
    assert runner.npair == 22
    assert runner.amount == pytest.approx(-2.8)
    assert runner.energy_threshold == pytest.approx(-2.8)


def test_batch_pipeline_skip_run_pasta(tmp_path: Path) -> None:
    if not PASTA_PROFILE.is_file() or not PASTA_ARCHIVE.is_file():
        pytest.skip("pasta fixtures missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(">test1\nGGGGGG\n>test2\nISFLIF\n")

    out = tmp_path / "results"
    pasta_work = out / "pasta" / "work" / "batch_1"
    pasta_work.mkdir(parents=True)
    runner = PASTARunner()
    runner._extract_profiles(PASTA_ARCHIVE, ["test1", "test2"], pasta_work)

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["pasta"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"test1", "test2"}
    assert (out / "pasta" / "parsed" / "test1_pasta.csv").is_file()
    assert any("[PASTA]" in line for line in logs)


def test_pasta_profile_member_name() -> None:
    members = {
        "predictions/test1.fasta.seq.aggr_profile.dat": object(),
        "predictions/test2.fasta.seq.aggr_profile.dat": object(),
    }
    assert PASTARunner._profile_member_name("test1", members) is not None
