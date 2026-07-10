"""Base class for predictor output parsers (phase 0: parse-only)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from aggressor_wrappers.core.schema import PredictorResult, PredictorSpec


class BasePredictorParser(ABC):
    """Parse raw tool output into a :class:`PredictorResult`."""

    spec: PredictorSpec

    @abstractmethod
    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        raise NotImplementedError
