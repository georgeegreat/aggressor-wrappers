"""Base class for predictor runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aggressor_wrappers.core.schema import PredictorResult


class BasePredictorRunner(ABC):
    """Run an external predictor and return a standard :class:`PredictorResult`."""

    @abstractmethod
    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        **kwargs,
    ) -> PredictorResult:
        raise NotImplementedError
