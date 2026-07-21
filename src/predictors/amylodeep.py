"""Parse AmyloDeep output into a per-residue profile.

AmyloDeep emits CSV/JSON of the form::

    sequence_id,position,probability,sequence_length,avg_probability,max_probability
    input_sequence,0,0.793,31,0.7744,0.945
    input_sequence,1,0.8404,31,0.7744,0.945

Two properties need care, and both are silent corruptions if missed.

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

REQUIRED = {"position", "probability"}


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

    if "sequence_length" in df.columns and df["sequence_length"].notna().any():
        length = int(df["sequence_length"].dropna().iloc[0])
    elif sequence is not None:
        length = len(sequence)
    else:
        length = int(df["position"].max()) + 1

    positions = df["position"].astype(int).to_numpy()
    probs = df["probability"].astype(float).to_numpy()

    # AmyloDeep is 0-based; this package is 1-based.
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
        "position_base": 0 if zero_based else 1,
        "granularity": "per_residue" if per_residue else "window",
        "window_size": window,
        "aggregation": "max_over_covering_windows" if window > 1 else "direct",
    }
    for col in ("avg_probability", "max_probability"):
        if col in df.columns and df[col].notna().any():
            meta[col] = float(df[col].dropna().iloc[0])
    return scores, meta
