"""ArchCandy region CSV → standard per-residue table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class ArchCandyParser(BasePredictorParser):
    spec = get_predictor_spec("archcandy")

    def __init__(self, *, score_mode: str = "cumulative") -> None:
        if score_mode not in {"cumulative", "highest"}:
            raise ValueError(f"Unsupported ArchCandy score_mode: {score_mode!r}")
        self.score_mode = score_mode

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        regions = pd.read_csv(source)
        regions = _normalise_region_columns(regions)

        scores = [0.0] * len(sequence)
        binary = [0] * len(sequence)

        for _, row in regions.iterrows():
            start = int(row["start"]) - 1
            stop = int(row["stop"]) - 1
            score = float(row["score"])
            if start > stop:
                raise ValueError(f"ArchCandy region has Start > Stop: {start + 1}-{stop + 1}")
            for pos in range(start, stop + 1):
                if 0 <= pos < len(sequence):
                    if self.score_mode == "cumulative":
                        scores[pos] += score
                    else:
                        scores[pos] = max(scores[pos], score)
                    binary[pos] = 1

        if self.score_mode == "cumulative":
            scores = [round(value, 3) for value in scores]

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"score_mode": self.score_mode},
        )


def _normalise_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=rename)
    required = {"start", "stop", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ArchCandy CSV missing columns: {sorted(missing)}")
    return df
