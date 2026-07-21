"""Parse the local Cross-Beta predictor (``CB_RF_pred.py``) output.

Written against the source at
https://github.com/Valentin-Gonay/cross-beta-predictor.

The per-residue file is **semicolon**-delimited (not comma), with columns::

    Query_name;Sequence_length;Average_protein_prediction;AR_position;Amino_acids_score

Two fields are stringified Python literals rather than scalars:

``Amino_acids_score``
    A list of *single-key dictionaries*, one per residue, keyed by the amino
    acid: ``[{'M': 0.42}, {'K': 0.51}, ...]``. Unusual, but useful — it carries
    the residue identity alongside the score, which gives a free integrity check
    against the input FASTA. A mismatch means the file describes a different
    protein, and silently accepting it would misalign every position.
``AR_position``
    The amyloidogenic regions, or the literal ``None`` when the tool found none.
    "No regions" is a legitimate biological result, so it yields an empty region
    list rather than an error.

Note the tool writes into a ``results/`` directory relative to its own working
directory regardless of what ``-o`` says; the runner accounts for that.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

COLUMNS = (
    "Query_name",
    "Sequence_length",
    "Average_protein_prediction",
    "AR_position",
    "Amino_acids_score",
)


def _literal(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in ("None", "nan", "NaN"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"could not parse Cross-Beta field: {text[:80]!r}") from exc


def read_crossbeta_csv(path: str | Path) -> pd.DataFrame:
    """Read the semicolon-delimited Cross-Beta per-residue result file."""
    df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: not a Cross-Beta per-residue result (missing {missing}). "
            f"Got columns: {list(df.columns)}"
        )
    return df


def crossbeta_profile(
    path: str | Path,
    sequence: str,
    *,
    query_name: str | None = None,
) -> tuple[list[float], list[tuple[int, int]], dict]:
    """Return ``(scores, regions, metadata)`` for one protein.

    ``regions`` are 1-based inclusive ``(start, stop)`` pairs where the tool
    reported an amyloidogenic region.
    """
    df = read_crossbeta_csv(path)
    if query_name is not None:
        subset = df[df["Query_name"].astype(str) == str(query_name)]
        if not subset.empty:
            df = subset
    row = df.iloc[0]

    residues = _literal(row["Amino_acids_score"]) or []
    scores: list[float] = []
    observed: list[str] = []
    for item in residues:
        if isinstance(item, dict):
            (aa, value), = item.items()
            observed.append(str(aa))
            scores.append(float(value))
        else:  # tolerate a plain list of scores
            scores.append(float(item))

    if observed:
        seen = "".join(observed)
        if seen != sequence:
            raise ValueError(
                f"{path}: Cross-Beta output describes a {len(seen)}-residue "
                f"sequence that does not match the {len(sequence)}-residue input"
            )
    elif len(scores) != len(sequence):
        raise ValueError(
            f"{path}: Cross-Beta returned {len(scores)} scores for a "
            f"{len(sequence)}-residue sequence"
        )

    regions: list[tuple[int, int]] = []
    for entry in _literal(row["AR_position"]) or []:
        if isinstance(entry, dict):
            start = entry.get("start") or entry.get("Start")
            stop = entry.get("stop") or entry.get("end") or entry.get("Stop")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            start, stop = entry[0], entry[1]
        else:
            continue
        if start is not None and stop is not None:
            regions.append((int(start), int(stop)))

    meta = {
        "source": str(path),
        "query_name": str(row["Query_name"]),
        "average_protein_prediction": float(row["Average_protein_prediction"]),
        "n_regions": len(regions),
    }
    return scores, regions, meta
