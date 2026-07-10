"""Cross-Beta JSON (CRBM datastore / REST API) → standard table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class CrossBetaParser(BasePredictorParser):
    spec = get_predictor_spec("crossbeta")

    def __init__(self, confidence_threshold: float = 0.54) -> None:
        self.confidence_threshold = confidence_threshold

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        with Path(source).open() as handle:
            data = json.load(handle)

        sequence_data = _extract_aa_list(data)
        if not sequence_data:
            return PredictorResult(
                protein_id=protein_id,
                sequence=sequence,
                spec=self.spec,
                scores=[0.0] * len(sequence),
                binary=[0] * len(sequence),
                metadata={"confidence_threshold": self.confidence_threshold},
            )

        if len(sequence_data) != len(sequence):
            raise ValueError(
                f"Cross-Beta residue count ({len(sequence_data)}) "
                f"does not match sequence length ({len(sequence)})"
            )

        scores = [0.0] * len(sequence)
        for item in sequence_data:
            pos = int(item["index"]) + 1
            idx = pos - 1
            if not 0 <= idx < len(sequence):
                raise ValueError(f"Cross-Beta position out of range: {pos}")
            scores[idx] = float(item["mean_confidence"])

        binary = binary_from_scores(scores, threshold=self.confidence_threshold)

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"confidence_threshold": self.confidence_threshold},
        )


def _extract_aa_list(data: Any) -> list[dict[str, Any]]:
    """Accept legacy dict payloads and modern list payloads from the CRBM API."""
    if isinstance(data, list):
        if not data or "AA_list" not in data[0]:
            raise ValueError("Cross-Beta JSON list missing AA_list")
        return data[0]["AA_list"]

    if isinstance(data, dict):
        if not data:
            raise ValueError("Cross-Beta JSON is empty")
        first_key = next(iter(data))
        entries = data[first_key]
        if not entries or "AA_list" not in entries[0]:
            raise ValueError("Cross-Beta JSON missing AA_list")
        return entries[0]["AA_list"]

    raise ValueError("Cross-Beta JSON must be an object or list")
