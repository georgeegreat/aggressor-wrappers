"""FoldAmyloid web runner (bioinfo.protres.ru) + parse pipeline.

FoldAmyloid publishes no API. The contract below was recovered by observing the
page's own traffic, not inferred from its HTML, and every field is reproduced as
the browser sends it::

    POST http://bioinfo.protres.ru/fold-amyloid/worker.php
    Content-Type: application/x-www-form-urlencoded

    cmd = do
    p1  = FASTA text (header + sequence)
    p2  = the scale table, "A 19.89|C 23.52|...|X 20.73|"
    p3  = averaging frame           (default 5)
    p4  = 3                         (sent verbatim by the page; see MODE below)
    p5  = threshold                 (default 21.4)
    p6  = threshold, repeated

The reply is JSON, and the per-residue data is **base64-encoded HTML** in the
``tbl`` field — not in ``seq``, which carries only the coloured sequence view::

    {"res": true, "msg": "",
     "seq": "<b64: sequence with <span class='F'> on flagged residues>",
     "tbl": "<b64: '  Num  Res Fold  Value' table, one row per residue>",
     "nam": "<query name>", "pro": "<scale name>", "mrg": "<threshold>",
     "tmp": "<job token; plots at tmp/<token>.png and tmp/<token>L.png>"}

Mechanistically, FoldAmyloid scores each residue by its **expected number of
contacts within 8 A**, averaged over a 5-residue frame, and flags residues whose
averaged profile exceeds 21.4 (Garbuzynskiy, Lobanov & Galzitskaya, 2010,
*Bioinformatics* 26:326). The premise is packing density rather than
hydrophobicity: a cross-beta spine demands tightly interdigitated side chains, so
the segments capable of forming one are those predicted to be densely
contacting. Sending the scale table as a parameter is therefore not incidental —
it *is* the model, and changing ``p2`` changes what is being predicted.

Sanity check used during development: on Abeta(1-42) the service flags residues
16-21, ``KLVFFA`` — the segment shown to be necessary and sufficient for fibril
formation (Tjernberg et al., 1996, *J. Biol. Chem.* 271:8545) — plus 31-35
(``IGLMV``) in the C-terminal hydrophobic stretch.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_fasta, read_first_sequence
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.foldamyloid import FoldAmyloidParser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "http://bioinfo.protres.ru/fold-amyloid/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"

#: The default scale: expected number of contacts within 8 A, per residue type.
#: Sent verbatim in ``p2``. These twenty-one values ARE the predictor; they are
#: kept here rather than being fetched from the page so that a run is
#: reproducible if the server's defaults ever change, and so that the numbers a
#: result depends on are visible in the repository.
CONTACTS_8A = {
    "A": 19.89, "C": 23.52, "D": 17.41, "E": 17.46, "F": 27.18, "G": 17.11,
    "H": 21.72, "I": 25.71, "K": 17.67, "L": 25.36, "M": 24.82, "N": 18.49,
    "P": 17.43, "Q": 19.23, "R": 21.03, "S": 18.19, "T": 19.81, "V": 23.93,
    "W": 28.48, "Y": 25.93, "X": 20.73,
}

#: Sent as ``p4``. The page emits 3 unconditionally and no control on the form
#: changes it; it is reproduced rather than reinterpreted, because guessing at
#: an undocumented parameter is how a wrapper silently stops matching the web
#: tool it claims to reproduce.
MODE = "3"

_TAG_RE = re.compile(r"<[^>]+>")
_ROW_RE = re.compile(r"^\s*(\d+)\s+([A-Za-z])\s+(f?)\s*([-\d.]+)\s*$")


def scale_string(scale: dict[str, float] | None = None) -> str:
    """Render a residue scale into the ``p2`` wire format."""
    table = scale or CONTACTS_8A
    return "".join(f"{aa} {value}|" for aa, value in table.items())


def parse_long_table(tbl_b64: str) -> list[tuple[int, str, bool, float]]:
    """Decode the base64 ``tbl`` field into ``(number, residue, flag, value)``.

    Why a line regex rather than ``csv.DictReader`` over the de-tagged text:
    **the Fold column is blank for unflagged residues.** A whitespace-delimited
    reader collapses ``    1   D       18.253`` to three fields and
    ``   16   K   f   21.582`` to four, so the flag is silently dropped from
    every negative row and mis-assigned on positives — the one column the
    predictor exists to produce. ``pandas.read_fwf`` with explicit column spans
    does handle it, and is the right choice if a DataFrame is wanted.

    Cost is not the deciding factor, measured on this capture replicated to
    length (mean of 5 runs, per call):

        rows      regex     DictReader   read_fwf
          42    0.03 ms      0.08 ms     0.56 ms
         300    0.20 ms      0.51 ms     0.92 ms
        3000    2.03 ms      4.91 ms     6.19 ms
       30000   21.77 ms     52.70 ms    63.77 ms

    At protein scale the whole parse is a fifth of a millisecond and is dwarfed
    by the HTTP round trip; the regex is nonetheless both the fastest and the
    only one of the three that is correct out of the box.

    The ``-----`` rule row under the header is skipped by the row pattern rather
    than by a comment character. amyloscope's original adapter passed
    ``comment="-"`` to drop it, which also truncates any data line at its first
    hyphen — harmless while the contact profile stays positive, fatal for a
    build that writes ``-`` as its negative Fold marker.
    """
    text = base64.b64decode(tbl_b64).decode("utf-8", errors="replace")
    rows: list[tuple[int, str, bool, float]] = []
    for line in text.splitlines():
        plain = _TAG_RE.sub("", line)
        match = _ROW_RE.match(plain)
        if not match:
            continue
        number, residue, flag, value = match.groups()
        rows.append((int(number), residue.upper(), flag.lower() == "f", float(value)))
    if not rows:
        raise RuntimeError(
            "FoldAmyloid returned no parseable rows in its 'tbl' field. The "
            "service changed its reply shape, or the submission was rejected."
        )
    return rows


class FoldAmyloidRunner(BasePredictorRunner):
    """Submit one sequence per job to FoldAmyloid's worker endpoint."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        averaging_frame: int = 5,
        threshold: float = 21.4,
        scale: dict[str, float] | None = None,
        timeout_seconds: int = 600,
        **_ignored,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_FOLDAMYLOID_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        self.averaging_frame = int(averaging_frame)
        self.threshold = float(threshold)
        self.scale = dict(scale) if scale else dict(CONTACTS_8A)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def _post(self, fasta_text: str) -> dict:
        payload = urllib.parse.urlencode(
            {
                "cmd": "do",
                "p1": fasta_text,
                "p2": scale_string(self.scale),
                "p3": str(self.averaging_frame),
                "p4": MODE,
                "p5": f"{self.threshold}",
                "p6": f"{self.threshold}",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            urllib.parse.urljoin(self.base_url, "worker.php"),
            data=payload,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json, text/plain, */*",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"FoldAmyloid returned non-JSON ({len(body)} bytes): {body[:200]!r}"
            ) from exc
        if not data.get("res"):
            raise RuntimeError(f"FoldAmyloid rejected the job: {data.get('msg') or data}")
        if "tbl" not in data:
            raise RuntimeError(
                f"FoldAmyloid reply has no 'tbl' field (keys: {sorted(data)}). "
                f"The per-residue table lives there, not in 'seq'."
            )
        return data

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run FoldAmyloid on a single-record FASTA; return a Num,Res,Fold,Value CSV.

        The CSV is written in the service's own export layout, BOM and all, so
        that a file produced here and a file downloaded by hand from the web UI
        are interchangeable inputs to the parser and to amyloscope.
        """
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        protein_id, sequence = read_first_sequence(fasta_path)
        data = self._post(f">{protein_id}\n{sequence}\n")
        rows = parse_long_table(data["tbl"])

        dest = cwd / f"{protein_id}_foldamyloid.csv"
        lines = ["Num,Res,Fold,Value"]
        lines += [f"{n},{r},{'f' if flag else ''},{v}" for n, r, flag, v in rows]
        dest.write_text("﻿" + "\r\n".join(lines) + "\r\n", encoding="utf-8")
        self.last_raw_path = dest
        return dest

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"FoldAmyloid batch expects exactly one sequence per job, got {len(records)}"
            )
        return self.execute(fasta_path, work_dir)

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        out_root = Path(output_dir)
        return {
            pid: out_root / f"{pid}_foldamyloid.csv"
            for pid in protein_ids
            if (out_root / f"{pid}_foldamyloid.csv").is_file()
        }

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_csv: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_csv is not None:
            raw_path = Path(raw_csv)
        elif skip_run:
            raise ValueError("Provide raw_csv when --skip-run is set")
        else:
            cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
            raw_path = self.execute(fasta_path, cwd)

        self.last_raw_path = raw_path
        return FoldAmyloidParser(threshold=self.threshold).parse(
            raw_path, protein_id=protein_id, sequence=sequence
        )
