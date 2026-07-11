"""Shared data models and column naming conventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class PredictorSpec:
    """Metadata for one amyloidogenicity predictor."""

    key: str
    """Short registry key, e.g. ``path``."""

    display_name: str
    """Human-readable name shown in logs."""

    score_column: str
    """Column name in merged wide tables, e.g. ``PATH_score``."""

    bin_column: str
    """Binary column name, e.g. ``PATH_bin``."""

    default_threshold: float | None = None
    """Default cutoff when the tool exposes a tunable threshold."""


PREDICTOR_REGISTRY: dict[str, PredictorSpec] = {
    "aggreprot": PredictorSpec(
        key="aggreprot",
        display_name="AggreProt",
        score_column="aggreprot_score",
        bin_column="aggreprot_bin",
        default_threshold=0.25,
    ),
    "aggrescan": PredictorSpec(
        key="aggrescan",
        display_name="Aggrescan",
        score_column="aggrescan_score",
        bin_column="aggrescan_bin",
    ),
    "appnn": PredictorSpec(
        key="appnn",
        display_name="APPNN",
        score_column="APPNN_score",
        bin_column="APPNN_bin",
    ),
    "archcandy": PredictorSpec(
        key="archcandy",
        display_name="ArchCandy",
        score_column="ArchCandy_score",
        bin_column="ArchCandy_bin",
    ),
    "crossbeta": PredictorSpec(
        key="crossbeta",
        display_name="Cross-Beta",
        score_column="cross-beta-predictor_score",
        bin_column="cross-beta-predictor_bin",
        default_threshold=0.54,
    ),
    "pasta": PredictorSpec(
        key="pasta",
        display_name="PASTA",
        score_column="pasta_score",
        bin_column="pasta_bin",
        default_threshold=-5.0,
    ),
    "path": PredictorSpec(
        key="path",
        display_name="PATH",
        score_column="PATH_score",
        bin_column="PATH_bin",
        default_threshold=75.0,
    ),
    "waltz": PredictorSpec(
        key="waltz",
        display_name="WALTZ",
        score_column="waltz_score",
        bin_column="waltz_bin",
    ),
}

# Aliases accepted by CLI and filename inference.
PREDICTOR_ALIASES: dict[str, str] = {
    "cross-beta": "crossbeta",
    "cross-beta-predictor": "crossbeta",
    "cross_beta": "crossbeta",
    "arch-candy": "archcandy",
    "ArchCandy": "archcandy",
    "APPNN": "appnn",
    "PATH": "path",
}


def resolve_predictor_key(name: str) -> str:
    """Normalise user-facing predictor name to registry key."""
    lowered = name.strip().lower().replace(" ", "-")
    if lowered in PREDICTOR_REGISTRY:
        return lowered
    if lowered in PREDICTOR_ALIASES:
        return PREDICTOR_ALIASES[lowered]
    if name in PREDICTOR_ALIASES:
        return PREDICTOR_ALIASES[name]
    raise KeyError(f"Unknown predictor: {name!r}. Known: {sorted(PREDICTOR_REGISTRY)}")


def get_predictor_spec(name: str) -> PredictorSpec:
    return PREDICTOR_REGISTRY[resolve_predictor_key(name)]


@dataclass
class PredictorResult:
    """Per-residue output from a single predictor for one protein."""

    protein_id: str
    sequence: str
    spec: PredictorSpec
    scores: list[float]
    binary: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.sequence)
        if len(self.scores) != n:
            raise ValueError(
                f"{self.spec.key}: expected {n} scores, got {len(self.scores)}"
            )
        if len(self.binary) != n:
            raise ValueError(
                f"{self.spec.key}: expected {n} binary flags, got {len(self.binary)}"
            )

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_dataframe(self) -> pd.DataFrame:
        """Standard per-predictor table (1-based ``position``)."""
        return pd.DataFrame(
            {
                "position": range(1, self.length + 1),
                "aa_name": list(self.sequence),
                self.spec.score_column: self.scores,
                self.spec.bin_column: self.binary,
            }
        )

    def to_csv(self, path: str | pd.PathLike[str]) -> None:
        self.to_dataframe().to_csv(path, index=False)


def standard_columns(spec: PredictorSpec) -> tuple[str, str]:
    return spec.score_column, spec.bin_column


def read_standard_csv(
    path: str | pd.PathLike[str],
    spec: PredictorSpec,
    *,
    protein_id: str = "protein",
    sequence: str | None = None,
) -> PredictorResult:
    """Load a previously exported standard CSV back into ``PredictorResult``."""
    df = pd.read_csv(path)
    score_col, bin_col = standard_columns(spec)

    if score_col not in df.columns or bin_col not in df.columns:
        raise ValueError(f"{path}: expected columns {score_col!r} and {bin_col!r}")

    if "position" in df.columns:
        df = df.sort_values("position")

    if "aa_name" in df.columns:
        seq = "".join(str(a) for a in df["aa_name"])
    elif sequence is not None:
        seq = sequence
    else:
        raise ValueError(f"{path}: need aa_name column or explicit sequence")

    if sequence is not None and seq != sequence:
        raise ValueError(
            f"{path}: sequence from aa_name ({len(seq)} aa) "
            f"does not match supplied FASTA ({len(sequence)} aa)"
        )

    if len(df) != len(seq):
        raise ValueError(
            f"{path}: row count ({len(df)}) does not match sequence length ({len(seq)})"
        )

    return PredictorResult(
        protein_id=protein_id,
        sequence=seq,
        spec=spec,
        scores=[float(x) for x in df[score_col]],
        binary=[int(x) for x in df[bin_col]],
    )


def scores_from_mapping(
    sequence: str,
    position_to_score: Mapping[int, float],
    *,
    default: float = 0.0,
) -> list[float]:
    """Build a per-residue score list from sparse 1-based positions."""
    scores = [default] * len(sequence)
    for pos, value in position_to_score.items():
        idx = pos - 1
        if 0 <= idx < len(sequence):
            scores[idx] = float(value)
    return scores


def binary_from_scores(
    scores: Sequence[float],
    *,
    threshold: float,
    greater_or_equal: bool = True,
) -> list[int]:
    if greater_or_equal:
        return [1 if s >= threshold else 0 for s in scores]
    return [1 if s < threshold else 0 for s in scores]
