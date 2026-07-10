"""Tests for pipeline resume / checkpoint logic."""

from __future__ import annotations

from pathlib import Path

from aggressor_wrappers.batch.resume import (
    partition_items_for_resume,
    predictor_parsed_path,
    resume_status,
)
from tests.helpers import write_standard_parsed_csv

SEQUENCE = "MADKG"


def test_resume_status_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    assert resume_status(path, runner_key="path", protein_id="protA", sequence=SEQUENCE) == "missing"


def test_resume_status_valid(tmp_path: Path) -> None:
    path = tmp_path / "protA_PATH.csv"
    write_standard_parsed_csv(path, runner_key="path", protein_id="protA", sequence=SEQUENCE)
    assert resume_status(path, runner_key="path", protein_id="protA", sequence=SEQUENCE) == "valid"


def test_resume_status_invalid_sequence(tmp_path: Path) -> None:
    path = tmp_path / "protA_PATH.csv"
    write_standard_parsed_csv(path, runner_key="path", protein_id="protA", sequence="MADK")
    assert resume_status(path, runner_key="path", protein_id="protA", sequence=SEQUENCE) == "invalid"


def test_partition_items_for_resume_disabled() -> None:
    items = [("protA", SEQUENCE), ("protB", SEQUENCE)]
    partition = partition_items_for_resume(
        items,
        parsed_dir=Path("/unused"),
        runner_key="path",
        tag="PATH",
        resume=False,
    )
    assert partition.pending == items
    assert partition.skipped_ids == []
    assert partition.invalidated_ids == []


def test_partition_items_for_resume_mixed(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "PATH" / "parsed"
    parsed_dir.mkdir(parents=True)
    valid_path = predictor_parsed_path(parsed_dir, protein_id="protA", tag="PATH")
    stale_path = predictor_parsed_path(parsed_dir, protein_id="protB", tag="PATH")
    write_standard_parsed_csv(valid_path, runner_key="path", protein_id="protA", sequence=SEQUENCE)
    write_standard_parsed_csv(stale_path, runner_key="path", protein_id="protB", sequence="MADK")

    items = [("protA", SEQUENCE), ("protB", SEQUENCE), ("protC", SEQUENCE)]
    logs: list[str] = []
    partition = partition_items_for_resume(
        items,
        parsed_dir=parsed_dir,
        runner_key="path",
        tag="PATH",
        resume=True,
        emit=logs.append,
    )

    assert partition.skipped_ids == ["protA"]
    assert partition.invalidated_ids == ["protB"]
    assert partition.pending == [("protB", SEQUENCE), ("protC", SEQUENCE)]
    assert any("reusing parsed output" in line and "protA" in line for line in logs)
    assert any("stale parsed output" in line and "protB" in line for line in logs)
