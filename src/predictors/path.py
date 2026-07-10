"""PATH results.csv → standard per-residue scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

from aggressor_wrappers.core.fasta import STANDARD_AA


class PATHParser(BasePredictorParser):
    spec = get_predictor_spec("path")

    def __init__(
        self,
        threshold_percentile: float = 75.0,
    ) -> None:
        self.threshold_percentile = threshold_percentile

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        results = pd.read_csv(source)
        hexapeptide_scores = self._load_hexapeptide_scores(results)
        per_residue, global_threshold = self._per_residue_scores(sequence, hexapeptide_scores)
        binary = binary_from_scores(per_residue, threshold=global_threshold)
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=[float(x) for x in per_residue],
            binary=binary,
            metadata={"threshold": global_threshold, "threshold_percentile": self.threshold_percentile},
        )

    def _load_hexapeptide_scores(self, results: pd.DataFrame) -> dict[str, float]:
        dope_max = results["dope"].max()
        span = float(dope_max - results["dope"].min())
        hexapeptide_dope = results.groupby("seq", sort=False)["dope"].min()
        if span == 0.0:
            return {seq: 0.0 for seq in hexapeptide_dope.index}
        return {
            seq: (dope_max - dope) / span
            for seq, dope in hexapeptide_dope.items()
        }

    def _per_residue_scores(
        self,
        sequence: str,
        hexapeptide_scores: dict[str, float],
    ) -> tuple[list[float], float]:
        seq_len = len(sequence)
        score_sum = np.zeros(seq_len, dtype=float)
        count = np.zeros(seq_len, dtype=int)
        window = 6

        for i in range(seq_len - window + 1):
            hexapeptide = sequence[i : i + window]
            if not all(aa in STANDARD_AA for aa in hexapeptide):
                continue
            if hexapeptide in hexapeptide_scores:
                score = hexapeptide_scores[hexapeptide]
                score_sum[i : i + window] += score
                count[i : i + window] += 1

        per_residue = np.zeros(seq_len)
        mask = count > 0
        per_residue[mask] = score_sum[mask] / count[mask]

        global_threshold = float(
            np.percentile(np.fromiter(hexapeptide_scores.values(), dtype=float), self.threshold_percentile)
        )
        return [float(x) for x in per_residue], global_threshold
