"""PASTA per-residue energy file → standard table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, binary_from_scores, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class PASTAParser(BasePredictorParser):
    spec = get_predictor_spec("pasta")

    def __init__(self, energy_threshold: float = -2.8) -> None:
        self.energy_threshold = energy_threshold

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        values = pd.read_csv(source, header=None, names=["energy"])
        if len(values) != len(sequence):
            raise ValueError(
                f"PASTA length mismatch: {len(values)} energies vs {len(sequence)} residues"
            )

        energies = [float(x) for x in values["energy"]]
        binary = binary_from_scores(
            energies,
            threshold=self.energy_threshold,
            greater_or_equal=False,
        )

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=energies,
            binary=binary,
            metadata={"energy_threshold": self.energy_threshold},
        )
