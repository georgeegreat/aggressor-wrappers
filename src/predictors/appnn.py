"""APPNN CSV (from appnn_converter.R) → standard table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser


class APPNNParser(BasePredictorParser):
    spec = get_predictor_spec("appnn")

    def __init__(self, score_threshold: float = 0.5) -> None:
        """``score_threshold`` is accepted here as well as on :meth:`parse`.

        It has to be. ``score_threshold`` is a ``[predictors.appnn]`` key, and
        ``registry.get_parser`` constructs every parser as
        ``PARSER_REGISTRY[key](**options)`` with the options read from that
        section -- so a class declaring no ``__init__`` raised
        ``TypeError: APPNNParser() takes no arguments`` for anyone who went
        through ``get_parser``. The run path was unaffected (``runners/appnn.py``
        builds ``APPNNParser()`` and passes the threshold to ``parse``), but
        ``pipeline``'s reparse / ``--skip-run`` branch goes through
        ``get_parser``, so reparsing cached APPNN output could not work at all.

        A value passed to :meth:`parse` still wins, so the existing call site
        keeps its behaviour; the constructor supplies the default.
        """
        self.score_threshold = float(score_threshold)

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        score_threshold: float | None = None,
        **kwargs,
    ) -> PredictorResult:
        score_threshold = (
            self.score_threshold if score_threshold is None else float(score_threshold)
        )
        df = pd.read_csv(source)

        position_col = _pick_column(df, ("aminoacid_position", "position"))
        score_col = _pick_column(df, ("aminoacid_score", "score", "APPNN_score"))
        aa_col = _pick_column(df, ("aminoacid", "aa_name"), required=False)

        ordered = df.sort_values(position_col)

        if aa_col is not None:
            seq_from_file = "".join(str(a) for a in ordered[aa_col])
            if len(seq_from_file) != len(sequence):
                raise ValueError(
                    f"APPNN row count ({len(seq_from_file)}) "
                    f"does not match sequence length ({len(sequence)})"
                )
            if seq_from_file != sequence:
                raise ValueError("APPNN amino acids disagree with supplied sequence")
            sequence = seq_from_file

        positions = ordered[position_col].astype(int)
        if (positions < 1).any() or (positions > len(sequence)).any():
            bad = int(positions[(positions < 1) | (positions > len(sequence))].iloc[0])
            raise ValueError(f"APPNN position out of range: {bad}")

        idx = positions.to_numpy(dtype=int) - 1
        score_values = ordered[score_col].astype(float).to_numpy()

        scores = np.zeros(len(sequence), dtype=float)
        binary = np.zeros(len(sequence), dtype=int)
        scores[idx] = score_values

        if "hotspot_region" in ordered.columns:
            hotspot = ordered["hotspot_region"].fillna(0).astype(int).to_numpy()
            binary[idx] = np.where(
                hotspot == 1,
                1,
                (score_values >= score_threshold).astype(int),
            )
        else:
            binary[idx] = (score_values >= score_threshold).astype(int)

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores.tolist(),
            binary=binary.tolist(),
            metadata={"score_threshold": score_threshold},
        )


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], *, required: bool = True) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"Expected one of {candidates}, got {list(df.columns)}")
    return None
