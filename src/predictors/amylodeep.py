"""Parse AmyloDeep output into a per-residue profile.

Two output generations are read, and the difference matters because only one of
them has already placed its scores on a residue axis.

**0.4 and later, per residue** (the default, and what this package asks for)::

    sequence_id,...,window_size,aggregate,heads_used,...,residue_number,residue,
    probability,coverage_depth,window_min,...,window_at_start
    input_sequence,...,6,mean,5,...,1,M,0.3121,1,...

``residue_number`` is 1-based and the projection over covering windows was done
by the tool, which records the ``window_size`` and ``aggregate`` it used. Those
values are carried into the metadata as reported rather than re-derived, and no
second projection is applied here: projecting twice would widen every boundary.

**0.3 and earlier, per window**::

    sequence_id,position,probability,sequence_length,avg_probability,max_probability
    input_sequence,0,0.793,31,0.7744,0.945

Here two properties need care, and both are silent corruptions if missed.

**Positions are 0-based.** Every other predictor in this package, and
``PredictorResult`` itself, is 1-based (``position``/``Number`` start at 1). Read
verbatim, an AmyloDeep profile is shifted one residue toward the N-terminus
relative to every other tool — a displacement small enough to survive inspection
and large enough to break a consensus that requires 80 % window overlap.

**The rows may be windows, not residues.** If the file contains
``sequence_length`` rows the output is per-residue. If it contains fewer, the
model scored overlapping windows and ``position`` is a window *start*, so the
implied window is ``sequence_length - n_rows + 1``. Assigning a window's
probability to its start residue alone would mislocate the signal by roughly half
a window; the score is therefore spread across the residues the window covers,
and each residue takes the maximum probability of the windows containing it
(``max``, not ``mean``: a residue inside one strongly amyloidogenic window should
not be diluted by weak neighbours, which is the same convention ArchCandy's
``highest`` mode uses).

That fallback is deliberately the wider rule, and it is a fallback: ``max`` can
exceed the evidence by up to ``window_size - 1`` residues, which is why 0.4's own
``mean`` projection is preferred and why the metadata always says which of the two
produced the numbers.

Which case applies is *detected and recorded* in the result metadata rather than
assumed, so the choice is visible in the output instead of buried here.

``avg_probability`` and ``max_probability`` are constant per sequence — they are
protein-level summaries, not per-residue signal, so they are carried as metadata
rather than repeated down a column.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

#: The index column, by generation. 0.4 writes ``residue_number`` (1-based, one
#: row per residue) or ``window_start_0based`` under ``--resolution window``; 0.3
#: wrote ``position``. Checked in this order so a 0.4 residue table is recognised
#: as such even though it also carries window columns.
INDEX_COLUMNS = ("residue_number", "position", "window_start_0based")
REQUIRED = {"probability"}


def _index_column(df, path) -> str:
    for name in INDEX_COLUMNS:
        if name in df.columns:
            return name
    raise ValueError(
        f"{path}: AmyloDeep output has no position column (looked for "
        f"{list(INDEX_COLUMNS)}); file has {list(df.columns)}"
    )


def _load(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("results", "predictions", "data"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        return pd.DataFrame(data)
    return pd.read_csv(p)


def parse_amylodeep(
    path: str | Path,
    *,
    sequence: str | None = None,
    sequence_id: str | None = None,
) -> tuple[list[float], dict]:
    """Return ``(per_residue_scores, metadata)`` from an AmyloDeep output file.

    ``sequence`` is used only to determine the expected length when the file's
    own ``sequence_length`` column is absent; it is never used to invent values.
    """
    df = _load(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: AmyloDeep output lacks column(s) {sorted(missing)}")

    if sequence_id is not None and "sequence_id" in df.columns:
        subset = df[df["sequence_id"].astype(str) == str(sequence_id)]
        if not subset.empty:
            df = subset

    index_column = _index_column(df, path)

    if "sequence_length" in df.columns and df["sequence_length"].notna().any():
        length = int(df["sequence_length"].dropna().iloc[0])
    elif sequence is not None:
        length = len(sequence)
    else:
        length = int(df[index_column].max()) + 1

    positions = df[index_column].astype(int).to_numpy()
    probs = df["probability"].astype(float).to_numpy()

    # AmyloDeep 0.3 was 0-based; 0.4's residue_number is 1-based. This package is
    # 1-based throughout, so the base is read from the data rather than assumed
    # from the column name -- a file written by a version this parser has not seen
    # should be wrong loudly, not quietly.
    base = int(positions.min())
    if base not in (0, 1):
        raise ValueError(f"{path}: unexpected minimum position {base}; expected 0 or 1")
    zero_based = base == 0

    n_rows = len(positions)
    per_residue = n_rows >= length
    window = 1 if per_residue else (length - n_rows + 1)

    scores = [0.0] * length
    for pos, prob in zip(positions, probs, strict=True):
        start = pos if zero_based else pos - 1      # -> 0-based index
        for offset in range(window):
            idx = start + offset
            if 0 <= idx < length:
                scores[idx] = max(scores[idx], float(prob))

    meta = {
        "source": str(path),
        "rows": n_rows,
        "sequence_length": length,
        "index_column": index_column,
        "position_base": 0 if zero_based else 1,
        "granularity": "per_residue" if per_residue else "window",
        "window_size": window,
        "aggregation": "max_over_covering_windows" if window > 1 else "direct",
    }

    # A 0.4 residue table has already been projected, and it records how. Report
    # the tool's own window size and rule instead of this parser's, which would
    # otherwise claim window_size 1 / "direct" for numbers aggregated over six
    # windows -- the projection would then be invisible downstream, and the
    # shoulder width it implies unaccounted for. projected_by names who did it.
    if per_residue and "window_size" in df.columns and df["window_size"].notna().any():
        meta["window_size"] = int(df["window_size"].dropna().iloc[0])
        meta["projected_by"] = "amylodeep"
        if "aggregate" in df.columns and df["aggregate"].notna().any():
            meta["aggregation"] = str(df["aggregate"].dropna().iloc[0])
    elif window > 1:
        meta["projected_by"] = "aggressor-wrappers"

    # Carried through where 0.4 supplies them: coverage_depth marks the termini,
    # where a value rests on fewer windows, and heads_used flags a run that
    # dropped the XGBoost head and averaged four rather than five.
    if "heads_used" in df.columns and df["heads_used"].notna().any():
        meta["heads_used"] = int(df["heads_used"].dropna().iloc[0])
    if per_residue and "coverage_depth" in df.columns and df["coverage_depth"].notna().any():
        depth = df["coverage_depth"].dropna().astype(int)
        meta["coverage_depth_min"] = int(depth.min())
        meta["coverage_depth_max"] = int(depth.max())
    for col in ("avg_probability", "max_probability"):
        if col in df.columns and df[col].notna().any():
            meta[col] = float(df[col].dropna().iloc[0])
    return scores, meta


class AmyloDeepParser(BasePredictorParser):
    """Register :func:`parse_amylodeep` on the standard parser interface.

    Kept as a thin adapter over the function, which predates it and is used
    directly by the runner. The class exists so AmyloDeep can reach a consensus
    table through the same path as every other predictor -- and so that path is
    an explicit, reviewable choice rather than a side effect of a runner existing.

    AmyloDeep is OPT-IN. It is absent from `predictors` in config.cfg, and this
    registration does not change that: registering a parser makes a tool
    *available* to the panel, while the config decides whether it votes. That
    separation matters for a fractional consensus, where adding one voter
    changes every tier.
    """

    spec = get_predictor_spec("amylodeep")

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = 0.5 if threshold is None else float(threshold)

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        scores, meta = parse_amylodeep(
            source, sequence=sequence, sequence_id=kwargs.get("sequence_id", protein_id)
        )
        if len(scores) != len(sequence):
            raise ValueError(
                f"{source}: AmyloDeep reports {len(scores)} positions for a "
                f"{len(sequence)}-residue sequence ({protein_id}). The file is "
                f"probably for a different protein."
            )
        meta = {**meta, "threshold": self.threshold, "binarised_from": "probability_threshold"}
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=[1 if value > self.threshold else 0 for value in scores],
            metadata=meta,
        )
