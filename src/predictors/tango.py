"""Parse TANGO per-residue output.

TANGO writes a tab-separated table, one row per residue::

    res	aa	Beta	Turn	Helix	Aggregation	Conc-Stab_Aggregation
    01	  D	  0.0	  0.1	0.000	0.000	0.000
    40	  V	 10.3	  0.0	0.000	97.233	97.233

Properties confirmed by running the binary:

* ``res`` is **1-based** and zero-padded, matching this package's convention.
* Fields carry leading whitespace inside the tab-delimited cells, so values must
  be stripped rather than cast directly.
* The ``aa`` column reproduces the input sequence, which gives a free integrity
  check: if it disagrees with the FASTA, the output belongs to a different
  sequence and should not be trusted. That check is cheap and catches the class
  of error where a stale output file is reused after the input changed.

**Which column is the score.** ``Aggregation`` is TANGO's β-aggregation
propensity as a percentage of the population (0-100), and is the quantity the
literature thresholds — conventionally at 5 % over a run of ≥5 residues.
``Conc-Stab_Aggregation`` is the concentration- and stability-corrected variant;
it is *not* redundant (it differs whenever the concentration or stability terms
bite), so it is retained rather than discarded.

``Beta``, ``Turn`` and ``Helix`` are TANGO's secondary-structure propensities.
They are what make TANGO more than an aggregation scale — a segment predicted
β-aggregating *and* β-structured is a different claim from one predicted
aggregating in a helical context — so they are kept as auxiliary channels
instead of being dropped at parse time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCORE_COLUMNS = ("Aggregation", "Conc-Stab_Aggregation")
AUX_COLUMNS = ("Beta", "Turn", "Helix", "Conc-Stab_Aggregation")
_EXPECTED = ("res", "aa", "Beta", "Turn", "Helix", "Aggregation")


def parse_tango_table(path: str | Path) -> pd.DataFrame:
    """Read a TANGO output file into a tidy frame (1-based ``res``)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in _EXPECTED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: TANGO output lacks column(s) {missing}")

    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df["res"] = df["res"].astype(int)
    for col in df.columns:
        if col not in ("res", "aa"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def tango_profile(
    path: str | Path,
    sequence: str,
    *,
    score_column: str = "Aggregation",
) -> tuple[list[float], dict[str, list[float]], dict]:
    """Return ``(scores, aux, metadata)`` for one sequence.

    Raises if the ``aa`` column contradicts ``sequence``: a mismatch means the
    file describes a different protein, and silently aligning it would corrupt
    every downstream position.
    """
    if score_column not in SCORE_COLUMNS:
        raise ValueError(f"score_column must be one of {SCORE_COLUMNS}")

    df = parse_tango_table(path)
    observed = "".join(df["aa"].tolist())
    if observed != sequence:
        raise ValueError(
            f"{path}: TANGO output describes a sequence of length {len(observed)} "
            f"that does not match the {len(sequence)}-residue input "
            f"(first mismatch at position "
            f"{next((i + 1 for i, (a, b) in enumerate(zip(observed, sequence, strict=False)) if a != b), min(len(observed), len(sequence)) + 1)})"
        )

    scores = [float(v) for v in df[score_column].fillna(0.0)]
    aux = {
        col.lower().replace("-", "_"): [float(v) for v in df[col].fillna(0.0)]
        for col in AUX_COLUMNS
        if col in df.columns and col != score_column
    }
    meta = {
        "score_column": score_column,
        "source": str(path),
        "max_aggregation": float(df["Aggregation"].max()),
    }
    return scores, aux, meta
