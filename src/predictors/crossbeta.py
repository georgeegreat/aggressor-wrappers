"""Cross-Beta JSON (CRBM datastore) → standard table."""

from __future__ import annotations

import json
from pathlib import Path

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class CrossBetaParser(BasePredictorParser):
    spec = get_predictor_spec("crossbeta")

    def __init__(self, confidence_threshold: float = 0.5) -> None:
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

        if not data:
            raise ValueError("Cross-Beta JSON is empty")

        first_key = next(iter(data))
        entries = data[first_key]
        if not entries or "AA_list" not in entries[0]:
            raise ValueError("Cross-Beta JSON missing AA_list")

        sequence_data = entries[0]["AA_list"]
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
