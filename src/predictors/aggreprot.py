"""AggreProt CSV export → standard table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class AggreProtParser(BasePredictorParser):
    spec = get_predictor_spec("aggreprot")

    def __init__(self, aggregation_threshold: float = 0.25) -> None:
        self.aggregation_threshold = aggregation_threshold

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        df = pd.read_csv(source, header=1)
        drop_cols = [c for c in ("struct_position", "amino_acid", "sasa", "transmembrane") if c in df.columns]
        df = df.drop(columns=drop_cols)

        position_col = "position" if "position" in df.columns else "aa"
        score_col = "aggregation" if "aggregation" in df.columns else "score"

        scores = [0.0] * len(sequence)
        for _, row in df.iterrows():
            pos = int(row[position_col])
            idx = pos - 1
            if 0 <= idx < len(sequence):
                scores[idx] = float(row[score_col])

        binary = binary_from_scores(scores, threshold=self.aggregation_threshold)

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"aggregation_threshold": self.aggregation_threshold},
        )
