"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec


def write_standard_parsed_csv(
    path: Path,
    *,
    runner_key: str,
    protein_id: str,
    sequence: str,
) -> None:
    """Write a minimal valid standard-format parsed table for resume/batch tests."""
    spec = get_predictor_spec(runner_key)
    length = len(sequence)
    PredictorResult(
        protein_id,
        sequence,
        spec,
        scores=[0.1] * length,
        binary=[0] * length,
    ).to_csv(path)
