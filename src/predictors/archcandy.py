"""ArchCandy region CSV → standard per-residue table."""

from __future__ import annotations

import warnings

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
        # ArchCandy predicts discrete beta-arches, each with a topology string
        # ('GBPL', 'BLLPBL', ...) describing the predicted structural motif. That
        # topology is the tool's distinctive output — it is what separates
        # ArchCandy from a generic propensity scale — and has no per-residue
        # representation, so it is retained at region level rather than lost.
        native: list[dict] = []
        best_arch: list[str] = [""] * len(sequence)
        best_score: list[float] = [float("-inf")] * len(sequence)

        for _, row in regions.iterrows():
            start = int(row["start"]) - 1
            stop = int(row["stop"]) - 1
            score = float(row["score"])
            if start > stop:
                raise ValueError(f"ArchCandy region has Start > Stop: {start + 1}-{stop + 1}")
            arch = str(row["arch"]) if "arch" in regions.columns else ""
            native.append(
                {
                    "start": start + 1,
                    "stop": stop + 1,
                    "score": score,
                    "arch": arch,
                }
            )
            for pos in range(start, stop + 1):
                if 0 <= pos < len(sequence):
                    if self.score_mode == "cumulative":
                        scores[pos] += score
                    else:
                        scores[pos] = max(scores[pos], score)
                    binary[pos] = 1
                    if score > best_score[pos]:
                        best_score[pos] = score
                        best_arch[pos] = arch

        if self.score_mode == "cumulative":
            scores = [round(value, 3) for value in scores]
            peak = max(scores, default=0.0)
            max_single = max((r["score"] for r in native), default=0.0)
            if peak > max_single + 1e-9:
                # ArchCandy's score is a confidence in [0, 1] (Ahmed et al. 2015).
                # Summing overlapping arches makes the per-residue value
                # confidence x multiplicity rather than confidence: it is unbounded
                # above, and a residue lying under several *weak* arches can clear a
                # threshold that no single arch clears. 'highest' keeps the value on
                # ArchCandy's own scale.
                warnings.warn(
                    f"ArchCandy score_mode='cumulative' inflated a per-residue score to "
                    f"{peak:.3f} for {protein_id}, above the best single arch "
                    f"({max_single:.3f}): overlapping arches are summed, so the score "
                    f"conflates confidence with the number of overlapping predictions "
                    f"and is no longer comparable to ArchCandy's own threshold. "
                    f"Use score_mode='highest' to keep scores on ArchCandy's scale.",
                    UserWarning,
                    stacklevel=2,
                )

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={"score_mode": self.score_mode, "n_arches": len(native)},
            aux={"arch": best_arch},
            regions=native,
        )


def _normalise_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=rename)
    required = {"start", "stop", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ArchCandy CSV missing columns: {sorted(missing)}")
    return df
