"""ArchCandy region CSV → standard per-residue table.

This is the ArchCandy **1.0** parser (the downloadable JAR's output, and any
historical 1.0 file); ArchCandy 2.0's REST output is ``archcandy_v2.py``.

On the threshold. ArchCandy emits beta-arch candidates each carrying a
confidence in [0, 1], and its own calibration treats low-scoring arches as
non-significant -- 0.56 is the 1.0 published boundary (Ahmed et al., 2015),
0.57 is 2.0's. This parser previously called EVERY residue lying under ANY
arch, whatever its score, so an arch the tool itself classes as insignificant
still produced a positive per-residue call. ``ArchCandyLocalRunner`` goes to
real trouble to carry ``local_threshold`` through, and the per-residue call then
ignored it.

The threshold is applied to the BEST SINGLE arch covering a residue, never to
the cumulative score. Under ``score_mode='cumulative'`` the per-residue value is
a sum over overlapping arches and can exceed 1.0, so comparing it with a [0, 1]
confidence boundary would call residues that lie under several weak arches and
no strong one -- which is the same conflation of confidence with multiplicity
the ``cumulative`` warning below is about. Binarising on the best arch keeps the
call on ArchCandy's scale regardless of how the score column is reported.
"""

from __future__ import annotations

import warnings

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

#: ArchCandy 1.0's published significance boundary (Ahmed et al., 2015).
#: Version-specific: 2.0 uses 0.57 and the two must not be mixed.
PUBLISHED_THRESHOLD_1X = 0.56


class ArchCandyParser(BasePredictorParser):
    spec = get_predictor_spec("archcandy")

    def __init__(
        self,
        threshold: float | None = PUBLISHED_THRESHOLD_1X,
        *,
        score_mode: str = "cumulative",
    ) -> None:
        """``threshold`` binarises on the best single arch covering each residue.

        Pass ``threshold=None`` for the previous behaviour -- every residue under
        any arch is called, whatever its confidence. That is available because a
        caller may genuinely want arch COVERAGE rather than significant arch
        coverage, but it is no longer the default, because a predictor that
        cannot say no is not contributing evidence to a consensus.
        """
        if score_mode not in {"cumulative", "highest"}:
            raise ValueError(f"Unsupported ArchCandy score_mode: {score_mode!r}")
        self.score_mode = score_mode
        self.threshold = None if threshold is None else float(threshold)

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
                    if score > best_score[pos]:
                        best_score[pos] = score
                        best_arch[pos] = arch

        # Binarise on the best single arch, not on `scores`, which under
        # score_mode='cumulative' is a sum and no longer a confidence.
        if self.threshold is None:
            binary = [1 if s > float("-inf") else 0 for s in best_score]
        else:
            binary = [1 if s >= self.threshold else 0 for s in best_score]

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
            metadata={
                "score_mode": self.score_mode,
                "n_arches": len(native),
                "threshold": self.threshold,
                "binarised_from": (
                    "best arch >= threshold" if self.threshold is not None
                    else "any arch coverage (threshold disabled)"
                ),
            },
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
