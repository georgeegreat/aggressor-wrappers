"""Parse the ArchCandy command-line ``Summary_*.txt`` output.

The local ArchCandy JAR writes, per sequence,
``<outdir>/<header>/Original_Sequence/Summary_<header>.txt``, whose
``CANDIDATES DETAILS`` section is a **fixed-width** table::

    Number     Sequence                    Score       Arc Type      Start Position  Stop Position ...
    1          HHQKLVFFAEDVGSNKGAIIGL      0.643       GBPL          13              34            ...
    12         AIIGLMVGGVVIATVIVITL        0.629       6 Res Arc 1   30              49            ...

Two properties of that table drive the implementation:

* **``Arc Type`` contains spaces** (``6 Res Arc 1``, ``5 Res Arc``), so the table
  cannot be split on whitespace — a naive ``line.split()`` silently shifts every
  column after the arc type. Column boundaries are therefore taken from the
  header line itself, which also means a change in the JAR's padding does not
  break the parser.
* **Start/Stop are 1-based and inclusive** (verified against the sequence:
  ``13..34`` is exactly ``HHQKLVFFAEDVGSNKGAIIGL``), matching the convention
  used throughout this package, so no index conversion is applied.

The result is the same region-level shape the web ArchCandy parser produces, so
either backend feeds the identical downstream consensus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SECTION = "CANDIDATES DETAILS"
_COLUMNS = [
    "Number",
    "Sequence",
    "Score",
    "Arc Type",
    "Start Position",
    "Stop Position",
    "Length",
    "Candidate / query %",
    "Location",
]

# Canonical output names, matching the web ArchCandy CSV the existing parser reads.
_RENAME = {
    "Number": "id",
    "Sequence": "sequence",
    "Score": "score",
    "Arc Type": "arch",
    "Start Position": "start",
    "Stop Position": "stop",
    "Length": "length",
    "Location": "location",
}


def _column_spans(header: str) -> list[tuple[str, int, int]]:
    """Derive (name, start, end) spans from the header line."""
    starts = []
    for name in _COLUMNS:
        idx = header.find(name)
        if idx < 0:
            raise ValueError(f"ArchCandy summary header lacks column {name!r}")
        starts.append((name, idx))
    starts.sort(key=lambda t: t[1])
    spans = []
    for i, (name, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(header) + 4096
        spans.append((name, start, end))
    return spans


def parse_summary(path: str | Path) -> pd.DataFrame:
    """Return the candidate table from an ArchCandy ``Summary_*.txt``.

    An empty frame (with the right columns) is returned when ArchCandy reports
    no candidates — that is a legitimate biological result, not an error, and it
    must not raise or the panel would lose the whole predictor for that protein.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Number") and "Arc Type" in line:
            header_idx = i
            break

    empty = pd.DataFrame(columns=list(_RENAME.values()))
    if header_idx is None:
        # "This sequence does not contain candidates for amyloid fibrils."
        if "does not contain candidates" in text:
            return empty
        raise ValueError(f"{path}: no CANDIDATES DETAILS table and no 'no candidates' note")

    spans = _column_spans(lines[header_idx])
    rows: list[dict] = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            if rows:
                break  # blank line ends the table
            continue
        if not re.match(r"\s*\d+\s", line):
            break  # a following section heading
        row = {name: line[start:end].strip() for name, start, end in spans}
        rows.append(row)

    if not rows:
        return empty

    df = pd.DataFrame(rows)
    df = df.rename(columns=_RENAME)[list(_RENAME.values())]
    for col, cast in (("id", int), ("start", int), ("stop", int), ("length", int)):
        df[col] = df[col].astype(cast)
    df["score"] = df["score"].astype(float)
    return df


def summary_path_for(output_dir: str | Path, header: str) -> Path:
    """Locate the Summary file ArchCandy writes for a given FASTA header."""
    return (
        Path(output_dir) / header / "Original_Sequence" / f"Summary_{header}.txt"
    )


def find_summaries(output_dir: str | Path) -> dict[str, Path]:
    """Map FASTA header -> Summary file for every sequence in an output folder."""
    root = Path(output_dir)
    found: dict[str, Path] = {}
    for summary in root.glob("*/Original_Sequence/Summary_*.txt"):
        found[summary.parent.parent.name] = summary
    return found
