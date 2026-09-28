"""AGGRESCAN web runner — andromeda.uab.cat.

AGGRESCAN publishes no API and offers no per-residue download: the server's only
file is ``aap_<job>.txt.xls``, which holds the ten global descriptors and nothing
positional. The per-residue profile exists **only in the result HTML**, so it is
scraped — which is where the stray non-breaking spaces in the historical CSVs
come from, 0xCA being a Mac Roman nbsp carried out of the page by a copy-paste.

Contract, captured from the site's own traffic::

    POST http://andromeda.uab.cat/bioinf/cgi-bin/aap/aap_ov.pl
    Content-Type: application/x-www-form-urlencoded
        sequence = <FASTA text>

    -> 200, the result page itself (synchronous; no job polling)
       plus /bioinf/aap/<job>/aap_<job>.txt.xls  (global descriptors only)

**The table is anchored on its header row, not on markup.** The result page
carries the profile in one ``<tr>`` of six ``<td>`` cells — one cell per column,
values separated by ``<br>`` — immediately after the header row whose cells read
``#  AA  a4v  HSA  NHSA  a4vAHS``. Anchoring there is not a stylistic choice: a
real page (334-residue CTSV) opens 42 ``<small>`` tags and closes 29, and
contains a literal ``<smal>`` typo, so any ``<small>``-delimited scan reads
across cell boundaries in a way that depends on where the imbalance happens to
fall. The header row is the one landmark the page states explicitly.

**Hot Spots come from NHSA, not from a re-derived threshold rule.** AGGRESCAN
(Conchillo-Sole et al., 2007, *BMC Bioinformatics* 8:65) describes a Hot Spot as
a region held above the Hot Spot Threshold for a minimum run, but reimplementing
that description does not reproduce the server's own calls. ``NHSA > 0`` does,
exactly, on both captures held here:

===============  ============  ==================  ====================
capture          reported nHS  ``NHSA > 0``        a4v > -0.02, run >= 5
===============  ============  ==================  ====================
Abeta(1-42)      2             2  (17-22, 30-42)   2  (identical)
CTSV (334 aa)    11            11 (identical)      12
===============  ============  ==================  ====================

On CTSV the threshold rule invents a hot spot at 105-111 and extends three more
by 3-4 residues each (230-232, 245-248, 332-334): 101 hot residues against the
server's 84, i.e. 30.2 % breadth reported where the tool itself calls 25.1 %.
The error is one-directional — it never under-calls — so it inflates AGGRESCAN's
apparent breadth and manufactures exactly the kind of shoulder that downstream
cluster analysis is meant to adjudicate. ``NHSA > 0`` also matches the page's
own red highlighting residue for residue (84/84), which is the server saying
which residues it considers hot.

``HSA`` is not the mask either: on Abeta42 the isolated Y10 carries
``HSA = 0.122`` and ``a4vAHS = 0.102`` while ``NHSA = 0`` — the shared area is
assigned before the run-length requirement is applied, so only NHSA reflects the
final call. ``hot_spot_runs()`` keeps the threshold rule for pages that predate
the NHSA column, and the runner still checks whichever mask it used against the
``nHS`` the server reported.

Validation cases: Abeta(1-42) nHS = 2 — 17-22 ``LVFFAE``, 30-42
``AIIGLMVGGVVIA``; a3vSA 0.064, Na4vSS 6.4. CTSV_Homo_sapiens nHS = 11 —
1-18, 75-79, 138-144, 182-187, 205-210, 223-229, 238-244, 259-265, 277-286,
298-302, 326-331; a3vSA -0.080, Na4vSS -8.2.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_fasta, read_first_sequence
from aggressor_wrappers.core.net import PermanentToolError
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.aggrescan import AggrescanParser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "http://andromeda.uab.cat/bioinf/aggrescan/"
#: Absolute path, deliberately. The CGI lives one level ABOVE the tool page, so
#: joining it relatively against base_url works only when base_url happens to be
#: the /bioinf/ root and silently 404s when base_url is the tool's own page --
#: which is the URL anyone would naturally put in a config key called base_url.
#: Anchoring to the host removes the trap: any base_url on the right host works.
_CGI_PATH = "/bioinf/cgi-bin/aap/aap_ov.pl"
_USER_AGENT = f"aggressor-wrappers/{__version__}"

#: AGGRESCAN's published Hot Spot criteria.
HOT_SPOT_THRESHOLD = -0.02   # a4v above which a residue is "aggregation prone"
MIN_HOT_SPOT_RUN = 5         # consecutive residues required to call a Hot Spot

_TAG_RE = re.compile(r"<[^>]+>")
_TR_RE = re.compile(r"<tr\b[^>]*>", re.IGNORECASE)
_TR_END_RE = re.compile(r"</tr\s*>", re.IGNORECASE)
_TD_RE = re.compile(r"<td\b[^>]*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_RED_RE = re.compile(r'COLOR\s*=\s*"?red"?', re.IGNORECASE)

_COLUMNS = ("Number", "AA", "a4v", "HSA", "NHSA", "a4vAHS")
#: The header labels as the page prints them, in order.
_HEADER_LABELS = ("#", "AA", "a4v", "HSA", "NHSA", "a4vAHS")

#: Export link on the result page. Global descriptors only -- no positional data
#: -- so it is recorded for provenance rather than parsed.
_EXPORT_RE = re.compile(r'href\s*=\s*"(/bioinf/aap/(\d+)/aap_\2\.txt\.xls)"', re.IGNORECASE)


def _cell_text(cell: str) -> str:
    return _TAG_RE.sub("", cell).replace("&nbsp;", " ").replace("\xa0", " ").strip()


def _cell_values(cell: str) -> list[str]:
    """One column's values, in residue order, from a ``<br>``-separated cell."""
    return [text for text in (_cell_text(part) for part in _BR_RE.split(cell)) if text]


def _rows(html: str) -> list[str]:
    """Table rows, each truncated at its own ``</tr>``.

    Truncating matters: the profile row is followed by further cells that belong
    to the next row, and a split on the opening tag alone would carry them in.
    """
    return [_TR_END_RE.split(chunk)[0] for chunk in _TR_RE.split(html)[1:]]


def _result_table(html: str) -> tuple[list[list[str]], list[bool]]:
    """Six columns plus the page's own red-highlight mask, or ``([], [])``.

    The anchor is the header row that names all six columns; the profile is the
    row after it. Both are located by content, so nothing depends on how many
    tables the page happens to wrap around them.
    """
    rows = _rows(html)
    for index, row in enumerate(rows[:-1]):
        flat = _TAG_RE.sub(" ", row)
        if not all(label in flat for label in _HEADER_LABELS):
            continue
        cells = _TD_RE.split(rows[index + 1])[1:]
        columns = [_cell_values(cell) for cell in cells]
        columns = [column for column in columns if column][:6]
        if len(columns) != 6:
            continue
        marked = [
            bool(_RED_RE.search(part))
            for part in _BR_RE.split(cells[0])
            if _cell_text(part)
        ]
        return columns, marked
    return [], []


def hot_spot_runs(
    a4v: list[float],
    *,
    threshold: float = HOT_SPOT_THRESHOLD,
    min_run: int = MIN_HOT_SPOT_RUN,
) -> list[tuple[int, int]]:
    """1-based ``(start, stop)`` runs of at least ``min_run`` residues above ``threshold``."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(a4v, start=1):
        if value > threshold:
            start = index if start is None else start
        elif start is not None:
            if index - start >= min_run:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(a4v) - start + 1 >= min_run:
        runs.append((start, len(a4v)))
    return runs


def hot_spot_mask(nhsa: list[float]) -> list[bool]:
    """AGGRESCAN's own Hot Spot membership, read rather than re-derived.

    ``NHSA`` (Normalized Hot Spot Area) is zero outside a Hot Spot and positive
    inside one, so the column *is* the call. See the module docstring for the
    two captures this was checked against, and for why the threshold rule in
    ``hot_spot_runs()`` is not equivalent.
    """
    return [value > 0 for value in nhsa]


def mask_runs(mask: list[bool]) -> list[tuple[int, int]]:
    """1-based ``(start, stop)`` runs of consecutive True values."""
    runs: list[list[int]] = []
    for position, flag in enumerate(mask, start=1):
        if not flag:
            continue
        if runs and runs[-1][1] == position - 1:
            runs[-1][1] = position
        else:
            runs.append([position, position])
    return [(start, stop) for start, stop in runs]


#: Descriptor cells of the summary table, e.g. ``(nHS)`` in "Number of Hot
#: Spots (nHS):". The page states each descriptor's short name in parentheses,
#: which is the only stable handle on a two-column layout.
_DESCRIPTOR_RE = re.compile(r"\(([A-Za-z0-9]+)\)\s*:\s*$")


#: The 20 standard amino acids. Verified against the live CGI: these, in either
#: case, are accepted; B, J, O, U, X, Z, '*' and '-' are not.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

_RED_CHAR_RE = re.compile(r'<FONT\s+COLOR="red">(.*?)</FONT>', re.IGNORECASE | re.DOTALL)


def _crlf(text: str) -> str:
    """Normalise to CRLF, which is the difference between a result and an error.

    The CGI splits FASTA records on CRLF because an HTML ``<textarea>`` always
    submits CRLF -- the HTML specification requires the normalisation, so every
    submission the authors ever saw had it. A client that sends bare LF hands the
    script one unsplittable line, and it answers ``UNIDENTIFIED ERROR``, whose
    text blames a copy-paste from MS Word. Verified on the live service: the same
    sequence, byte for byte, returns a 15 643-byte result table with CRLF and an
    860-byte error page with LF.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")


def validate_sequence(sequence: str) -> list[tuple[int, str]]:
    """1-based ``(position, character)`` for residues AGGRESCAN will reject.

    Pre-flight rather than post-mortem: a 372-protein sweep should not spend a
    request to learn that a sequence has an ``X`` in it, and the caller gets the
    positions instead of "some characters".
    """
    return [
        (index, character)
        for index, character in enumerate(sequence.upper(), start=1)
        if character not in STANDARD_RESIDUES
    ]


def _raise_for_service_error(html: str) -> None:
    """Report AGGRESCAN's own refusal as a refusal, not as a layout change.

    The CGI answers 200 with its usual page -- title, stylesheet and all -- and
    puts an error where the table would be, so "is this a result page?" cannot
    be answered from the shell. An unqualified "the layout changed" is then
    wrong twice over: the layout is fine, and it sends the operator into the
    parser instead of to the request that was sent. The three refusals below are
    everything the live service produced across a probe of line endings,
    alphabets, lengths from 3 to 2000 residues, and header spellings.
    """
    flat = re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()
    upper = flat.upper()

    if "CHARACTERS OF YOUR SEQUENCE ARE NOT ALLOWED" in upper:
        # This page echoes the sequence with the offending characters wrapped in
        # <FONT COLOR="red">, so the answer is on the page rather than inferred.
        bad = sorted({c for run in _RED_CHAR_RE.findall(html) for c in _TAG_RE.sub("", run).strip()})
        raise PermanentToolError(
            f"AGGRESCAN rejected the sequence: it contains characters outside the "
            f"20 standard amino acids{' -- ' + ', '.join(bad) if bad else ''}. "
            f"B, J, O, U, X, Z, '*' and '-' are all refused; lower case is NOT a "
            f"problem (verified). Screen with validate_sequence() before "
            f"submitting to avoid spending a request on this."
        )

    if "PLEASE USE FASTA FORMAT" in upper:
        raise PermanentToolError(
            "AGGRESCAN rejected the submission: no '>' header. It is a "
            "multisequence analyser and uses '>' to find where each record "
            "starts and ends, so a bare sequence is refused."
        )

    if "UNIDENTIFIED ERROR" in upper:
        raise PermanentToolError(
            "AGGRESCAN returned UNIDENTIFIED ERROR. The CGI splits FASTA records "
            "on CRLF -- an HTML <textarea> always submits CRLF, so that is the "
            "only line ending the service has ever been given. A body with bare "
            "LF arrives as one unsplittable line and is refused this way, and so "
            "is a header with no sequence under it. The page blames a copy-paste "
            "from MS Word, which is a red herring for a programmatic client: "
            "check the line endings and that the record is non-empty."
        )


def parse_summary(html: str) -> dict[str, float]:
    """The global descriptors, keyed by the short names the page itself prints.

    The summary is TWO columns -- every label in one, every value in the other --
    so a label and its value are nowhere near each other in the source. Reading
    "the first number after the words Number of Hot Spots" therefore returns 100,
    off the *next* label ("Normalized nHS for 100 residues"), and a cross-check
    against that silently becomes a check against a constant. Labels and values
    are collected separately and paired by order, and if the two runs are not the
    same length nothing is returned -- an unavailable check beats a wrong one.
    """
    labels: list[str] = []
    values: list[float] = []
    for attributes, body in re.findall(
        r"<td\b([^>]*)>(.*?)</td>", html, re.IGNORECASE | re.DOTALL
    ):
        text = _cell_text(_BR_RE.sub(" ", body))
        if not text:
            continue
        descriptor = _DESCRIPTOR_RE.search(text)
        if descriptor:
            labels.append(descriptor.group(1))
        elif 'colspan="5"' in attributes.lower():
            try:
                values.append(float(text))
            except ValueError:
                continue
    if not labels or len(values) < len(labels):
        return {}
    # A second summary table repeats the same descriptors; the first run wins.
    return dict(zip(labels, values[: len(labels)]))


def parse_result_page(html: str) -> tuple[list[dict], int | None]:
    """Return per-residue records and the server's reported ``nHS``."""
    columns, marked = _result_table(html)
    if len(columns) < 6:
        _raise_for_service_error(html)
        # Distinguish "the layout changed" from "this is not the result page",
        # which is what a mis-resolved CGI URL produces. The first line of the
        # body is usually enough to tell them apart, so it is quoted.
        snippet = _TAG_RE.sub(" ", html)[:200].strip().replace("\n", " ")
        looks_like_error = any(
            marker in html.lower() for marker in ("404", "not found", "<title>error")
        )
        hint = (
            " The response looks like an error page, not a result page -- check "
            "that base_url points at the AGGRESCAN host (the CGI is resolved "
            "against scheme+host, not against base_url's path)."
            if looks_like_error
            else " The page layout changed."
        )
        raise RuntimeError(
            f"AGGRESCAN result page yielded {len(columns)} data columns, expected 6 "
            f"({', '.join(_COLUMNS)}) in the row after the "
            f"'{' '.join(_HEADER_LABELS)}' header.{hint} First bytes: {snippet!r}"
        )
    number, aa, a4v, hsa, nhsa, ahs = columns[:6]
    lengths = {len(b) for b in (number, aa, a4v, hsa, nhsa, ahs)}
    if len(lengths) != 1:
        raise RuntimeError(
            f"AGGRESCAN columns have unequal lengths {sorted(lengths)}; the "
            f"column-major cells cannot be zipped safely."
        )
    records = [
        {
            "Number": int(number[i]),
            "AA": aa[i],
            "a4v": float(a4v[i]),
            "HSA": float(hsa[i]),
            "NHSA": float(nhsa[i]),
            "a4vAHS": float(ahs[i]),
        }
        for i in range(len(number))
    ]
    # The page prints Hot Spot residues in red. It is redundant with NHSA by
    # construction, which is exactly what makes it worth checking: if the two
    # ever disagree, the column's meaning has moved and the mask is guesswork
    # again.
    if marked and len(marked) == len(records) and any(marked):
        derived = hot_spot_mask([r["NHSA"] for r in records])
        if marked != derived:
            disagreements = [
                i + 1 for i, (m, d) in enumerate(zip(marked, derived)) if m != d
            ]
            raise RuntimeError(
                f"AGGRESCAN's red Hot Spot highlighting and NHSA > 0 disagree at "
                f"residue(s) {disagreements[:10]}"
                f"{' ...' if len(disagreements) > 10 else ''}. NHSA has been the "
                f"Hot Spot mask on every capture; a disagreement means the page "
                f"changed what one of them means."
            )
    summary = parse_summary(html)
    reported = int(summary["nHS"]) if "nHS" in summary else None
    return records, reported


class AggrescanRunner(BasePredictorRunner):
    """Submit a FASTA to AGGRESCAN and write the per-residue table it never exports."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        hot_spot_threshold: float = HOT_SPOT_THRESHOLD,
        min_hot_spot_run: int = MIN_HOT_SPOT_RUN,
        verify_reported_nhs: bool = True,
        cgi_path: str | None = None,
        timeout_seconds: int = 600,
        **_ignored,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_AGGRESCAN_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        self.hot_spot_threshold = float(hot_spot_threshold)
        self.min_hot_spot_run = int(min_hot_spot_run)
        self.verify_reported_nhs = bool(verify_reported_nhs)
        self.timeout_seconds = int(timeout_seconds)
        # Resolved against the SCHEME+HOST of base_url, not against its path.
        parts = urllib.parse.urlsplit(self.base_url)
        self.cgi_url = urllib.parse.urlunsplit(
            (parts.scheme, parts.netloc, cgi_path or _CGI_PATH, "", "")
        )
        self.last_raw_path: Path | None = None

    def _post(self, fasta_text: str) -> str:
        payload = urllib.parse.urlencode({"sequence": _crlf(fasta_text)}).encode("utf-8")
        request = urllib.request.Request(
            self.cgi_url,
            data=payload,
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        protein_id, sequence = read_first_sequence(fasta_path)
        offending = validate_sequence(sequence)
        if offending:
            # Refuse locally rather than spend a request to be told the same
            # thing less precisely: the server's reply says "som characters",
            # this says which ones and where.
            shown = ", ".join(f"{char}@{position}" for position, char in offending[:12])
            raise PermanentToolError(
                f"AGGRESCAN accepts only the 20 standard amino acids; "
                f"{protein_id} has {len(offending)} other character(s): {shown}"
                f"{' ...' if len(offending) > 12 else ''}"
            )
        html = self._post(f">{protein_id}\n{sequence}\n")
        records, reported = parse_result_page(html)

        if len(records) != len(sequence):
            raise RuntimeError(
                f"AGGRESCAN returned {len(records)} residues for a "
                f"{len(sequence)}-residue query ({protein_id})"
            )
        observed = "".join(r["AA"] for r in records).upper()
        if observed != sequence.upper():
            raise RuntimeError(
                f"AGGRESCAN echoed a different sequence than was submitted for "
                f"{protein_id}; refusing to attribute the result."
            )

        nhsa = [r["NHSA"] for r in records]
        if any(value > 0 for value in nhsa):
            runs = mask_runs(hot_spot_mask(nhsa))
            source = "NHSA > 0"
        else:
            # No Hot Spot anywhere is a legitimate result, and indistinguishable
            # from an all-zero NHSA column, so fall back and let the nHS check
            # below decide which it was.
            runs = hot_spot_runs(
                [r["a4v"] for r in records],
                threshold=self.hot_spot_threshold,
                min_run=self.min_hot_spot_run,
            )
            source = (
                f"a4v > {self.hot_spot_threshold}, run >= {self.min_hot_spot_run} "
                f"(NHSA column carried no Hot Spot)"
            )
        if self.verify_reported_nhs and reported is not None and len(runs) != reported:
            raise RuntimeError(
                f"AGGRESCAN reported nHS={reported} but the mask from {source} "
                f"found {len(runs)}: {runs}. Do not ship a mask that disagrees "
                f"with the server's own count."
            )
        hot = {p for start, stop in runs for p in range(start, stop + 1)}

        dest = cwd / f"{protein_id}_aggrescan.csv"
        lines = ["Number,AA,a4v,HSA,NHSA,a4vAHS,Prediction"]
        for r in records:
            flag = "1" if r["Number"] in hot else ""
            lines.append(
                f"{r['Number']},{r['AA']},{r['a4v']},{r['HSA']},"
                f"{r['NHSA']},{r['a4vAHS']},{flag}"
            )
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.last_raw_path = dest
        return dest

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"AGGRESCAN runner submits one sequence per job, got {len(records)}"
            )
        return self.execute(fasta_path, work_dir)

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        root = Path(output_dir)
        return {
            pid: root / f"{pid}_aggrescan.csv"
            for pid in protein_ids
            if (root / f"{pid}_aggrescan.csv").is_file()
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
        return AggrescanParser().parse(raw_path, protein_id=protein_id, sequence=sequence)
