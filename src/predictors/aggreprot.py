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

        position_col = "position" if "position" in df.columns else "aa"
        score_col = "aggregation" if "aggregation" in df.columns else "score"

        scores = [0.0] * len(sequence)
        # AggreProt also reports solvent accessibility and a transmembrane flag
        # per residue. These were previously dropped; they are retained as aux
        # channels because they are not redundant with the aggregation score:
        # burial (SASA) is precisely what determines whether an aggregation-prone
        # segment is sterically available to pair, and a TM segment's
        # hydrophobicity is a well-known false-positive source for
        # hydrophobicity-driven predictors.
        aux_cols = [c for c in ("sasa", "transmembrane") if c in df.columns]
        aux: dict[str, list] = {name: [float("nan")] * len(sequence) for name in aux_cols}

        for _, row in df.iterrows():
            pos = int(row[position_col])
            idx = pos - 1
            if 0 <= idx < len(sequence):
                scores[idx] = float(row[score_col])
                for name in aux_cols:
                    try:
                        aux[name][idx] = float(row[name])
                    except (TypeError, ValueError):
                        aux[name][idx] = float("nan")

        binary = binary_from_scores(scores, threshold=self.aggregation_threshold)

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"aggregation_threshold": self.aggregation_threshold},
            aux=aux,
        )
