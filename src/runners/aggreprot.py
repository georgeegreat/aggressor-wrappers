r"""AggreProt web runner (loschmidt.chemi.muni.cz) + parse pipeline.

API design (reverse-engineered from the public React frontend, not documented OpenAPI):

1. ``POST /api/jobs`` — allocate a short job id (empty JSON body is accepted).
2. ``PUT /api/jobs/{id}`` — submit a JobRequest; this starts execution.
3. Poll ``GET /api/jobs/{id}`` until ``status`` is ``DONE`` or ``FAILED``.
4. Download ``GET /api/jobs/{id}.csv``.

JobRequest constraints mirrored from the web form:

- ``proteins[]``: per-record fields only (sequence, accession, header, sequenceLines,
  sequenceOneLine). Do **not** send ``structSource`` or null structure placeholders —
  the server rejects those; "no structure" means omitting structure fields entirely.
- Accessions must be unique and match ``^[\w.]+$`` (first whitespace-delimited FASTA token).
- At most three sequences per job (site limit).

The web UI threshold slider (default 0.25) affects chart display only; it is **not**
part of the submission payload. Binarisation for our pipeline uses
``[predictors.aggreprot] aggregation_threshold`` in config.cfg.

Multifasta CSV exports use six side-by-side column groups per protein; see
``split_aggreprot_csv`` below. We split combined exports so the existing
``AggreProtParser`` (single-protein, ``header=1``) stays unchanged (Open/Closed).
"""

from __future__ import annotations

import json
import os
import re
import ssl
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_first_sequence, read_fasta
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.aggreprot import AggreProtParser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://loschmidt.chemi.muni.cz/aggreprot"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_TERMINAL_STATUSES = frozenset({"DONE", "FAILED"})
# Site-enforced maximum; keep in sync with the AggreProt input form validation.
_MAX_SEQUENCES_PER_JOB = 3
# Fixed width of each protein block in combined CSV exports (position … transmembrane).
_COLUMNS_PER_PROTEIN = 6
_ACCESSION_RE = re.compile(r"^[\w.]+$")
# Brief 404s right after PUT have been observed while the job record propagates.
_POLL_404_GRACE_SECONDS = 30
_COMBINED_CSV_NAME = "aggreprot_combined.csv"


class AggreProtRunner(BasePredictorRunner):
    """
    Submit (multi-)FASTA to the AggreProt REST API, poll until completion,
    download the CSV export, and parse into standard columns.

    Configure via ``[runners.aggreprot]`` and ``[predictors.aggreprot]`` in config.cfg.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        poll_interval_seconds: int = 3,
        timeout_seconds: int | None = 1800,
        verify_ssl: bool = False,
        aggregation_threshold: float = 0.25,
        job_title: str = "",
        email: str = "",
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_AGGREPROT_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.poll_interval_seconds = int(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None
        self.verify_ssl = bool(verify_ssl)
        self.aggregation_threshold = float(aggregation_threshold)
        self.job_title = str(job_title)
        self.email = str(email)
        self.last_raw_path: Path | None = None

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run AggreProt on a single-record FASTA and return the per-protein CSV path."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"AggreProt execute expects exactly one sequence, got {len(records)}"
            )
        protein_id = next(iter(records))
        mapping = self._materialise_per_protein_csvs(records, cwd)
        raw_path = mapping[protein_id]
        self.last_raw_path = raw_path
        return raw_path

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run AggreProt on up to three FASTA records in one API job."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        records = read_fasta(fasta_path)
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}")
        if len(records) > _MAX_SEQUENCES_PER_JOB:
            raise ValueError(
                f"AggreProt accepts at most {_MAX_SEQUENCES_PER_JOB} sequences per job, "
                f"got {len(records)}"
            )
        self._materialise_per_protein_csvs(records, cwd)
        return cwd

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map FASTA protein IDs to ``{id}_aggreprot.csv`` under ``output_dir``."""
        out_root = Path(output_dir)
        mapping: dict[str, Path] = {}
        for protein_id in protein_ids:
            path = out_root / f"{protein_id}_aggreprot.csv"
            if path.is_file():
                mapping[protein_id] = path
        return mapping

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
        parser = AggreProtParser(aggregation_threshold=self.aggregation_threshold)
        return parser.parse(raw_path, protein_id=protein_id, sequence=sequence)

    def _materialise_per_protein_csvs(
        self,
        records: dict[str, str],
        work_dir: Path,
    ) -> dict[str, Path]:
        """Submit one API job and write ``{protein_id}_aggreprot.csv`` files."""
        combined_csv = work_dir / _COMBINED_CSV_NAME
        self._submit_records(records, combined_csv)
        # Preserve FASTA dict order so CSV column groups align with batch protein_ids.
        return split_aggreprot_csv(
            combined_csv.read_text(),
            list(records.keys()),
            work_dir,
        )

    def _submit_records(self, records: dict[str, str], dest_csv: Path) -> Path:
        job_id = self._init_job()
        self._submit_job(job_id, self._build_job_request(records))
        self._wait_for_completion(job_id)
        dest_csv.parent.mkdir(parents=True, exist_ok=True)
        dest_csv.write_bytes(self._download_csv(job_id))
        self.last_raw_path = dest_csv
        return dest_csv

    def _build_job_request(self, records: dict[str, str]) -> dict:
        return {
            "title": self.job_title,
            "email": self.email,
            "proteins": [
                self._build_protein_record(protein_id, sequence)
                for protein_id, sequence in records.items()
            ],
            "example": False,
        }

    def _build_protein_record(self, protein_id: str, sequence: str) -> dict:
        accession = normalise_aggreprot_accession(protein_id)
        if not _ACCESSION_RE.match(accession):
            raise ValueError(
                f"AggreProt accession must match {_ACCESSION_RE.pattern!r}, got {accession!r}"
            )
        seq = sequence.upper().replace(" ", "")
        return {
            "sequence": f">{accession}\n{seq}",
            "accession": accession,
            "header": accession,
            "sequenceLines": "\n".join(textwrap.wrap(seq, 60)),
            "sequenceOneLine": seq,
        }

    def _init_job(self) -> str:
        payload = self._read_json(self._request("POST", "/api/jobs", data={}))
        if isinstance(payload, str):
            return payload
        job_id = str(payload.get("id") or "")
        if not job_id:
            raise RuntimeError(f"AggreProt job init returned unexpected payload: {payload}")
        return job_id

    def _submit_job(self, job_id: str, request: dict) -> None:
        # PUT must be executed; building the Request alone does not send it.
        self._read_bytes(
            self._request("PUT", f"/api/jobs/{urllib.parse.quote(job_id)}", data=request)
        )

    def _wait_for_completion(self, job_id: str) -> None:
        start = time.time()
        status_url = f"/api/jobs/{urllib.parse.quote(job_id)}"
        while True:
            elapsed = time.time() - start
            if self.timeout_seconds and elapsed > self.timeout_seconds:
                raise RuntimeError(
                    f"AggreProt timed out after {self.timeout_seconds}s waiting for job {job_id}"
                )

            try:
                job = self._read_json(self._request("GET", status_url))
            except RuntimeError as exc:
                # Job record may not be readable immediately after PUT (transient 404).
                if "404" in str(exc) and elapsed < _POLL_404_GRACE_SECONDS:
                    time.sleep(self.poll_interval_seconds)
                    continue
                raise

            status = str(job.get("status") or "")
            if status in _TERMINAL_STATUSES:
                if status != "DONE":
                    raise RuntimeError(f"AggreProt job {job_id} failed with status {status}")
                return
            time.sleep(self.poll_interval_seconds)

    def _download_csv(self, job_id: str) -> bytes:
        return self._read_bytes(
            self._request(
                "GET",
                f"/api/jobs/{urllib.parse.quote(job_id)}.csv",
                accept="text/csv,*/*",
            )
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        data: dict | None = None,
        accept: str = "application/json",
    ) -> urllib.request.Request:
        url = urllib.parse.urljoin(self.base_url + "/", path.lstrip("/"))
        headers = {"User-Agent": _USER_AGENT, "Accept": accept}
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(url, data=body, headers=headers, method=method)

    def _read_json(self, request: urllib.request.Request) -> dict | str:
        return json.loads(self._read_bytes(request).decode("utf-8"))

    def _read_bytes(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(
                request,
                context=self._ssl_context(),
                timeout=self.timeout_seconds,
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"AggreProt request failed ({exc.code}) for {request.full_url}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"AggreProt request failed for {request.full_url}: {exc}") from exc

    def _ssl_context(self) -> ssl.SSLContext:
        if self.verify_ssl:
            return ssl.create_default_context()
        # Match other web runners: loschmidt uses a valid cert but we keep opt-out
        # for restricted/lab environments (same pattern as ArchCandy/Cross-Beta).
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context


def normalise_aggreprot_accession(header: str) -> str:
    """Match AggreProt FASTA accession parsing (first header token)."""
    token = header.strip().split()[0]
    if token.startswith(("sp|", "tr|")):
        token = token[3:]
    if "|" in token:
        token = token.split("|", 1)[0]
    return token


def split_aggreprot_csv(
    csv_text: str,
    protein_ids: list[str],
    dest_dir: Path,
) -> dict[str, Path]:
    """
    Split a combined AggreProt CSV into per-protein exports.

    The API uses one of two layouts:

    - **Single protein:** standard six-column table (unchanged on disk).
    - **Multifasta:** row 0 lists accessions every six columns; data columns are
      concatenated horizontally. We slice fixed-width groups so ``AggreProtParser``
      can stay single-protein.

    Parsing uses simple comma splits (no RFC 4180 quoting). This matches current
    AggreProt exports where numeric fields use plain decimals/scientific notation.
    """
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("AggreProt CSV export is empty or malformed")

    names = _protein_names_from_header(lines[0])
    _validate_csv_protein_alignment(names, protein_ids)

    if len(names) == 1:
        protein_id = protein_ids[0]
        dest = dest_dir / f"{protein_id}_aggreprot.csv"
        dest.write_text(csv_text if csv_text.endswith("\n") else csv_text + "\n")
        return {protein_id: dest}

    header_parts = lines[1].split(",")
    mapping: dict[str, Path] = {}
    for index, (name, protein_id) in enumerate(zip(names, protein_ids)):
        start = index * _COLUMNS_PER_PROTEIN
        end = start + _COLUMNS_PER_PROTEIN
        chunk_header = ",".join(header_parts[start:end])
        out_lines = [
            f"Protein {index + 1},{name},,,,",
            chunk_header,
        ]
        for row in lines[2:]:
            parts = row.split(",")
            out_lines.append(",".join(parts[start:end]))
        dest = dest_dir / f"{protein_id}_aggreprot.csv"
        dest.write_text("\n".join(out_lines) + "\n")
        mapping[protein_id] = dest
    return mapping


def _validate_csv_protein_alignment(names: list[str], protein_ids: list[str]) -> None:
    if len(names) != len(protein_ids):
        raise ValueError(
            f"AggreProt CSV lists {len(names)} protein(s) "
            f"but batch expected {len(protein_ids)}: {protein_ids}"
        )
    for name, protein_id in zip(names, protein_ids):
        if normalise_aggreprot_accession(protein_id) != name:
            raise ValueError(
                f"AggreProt CSV accession {name!r} does not match FASTA id {protein_id!r}"
            )


def _protein_names_from_header(line: str) -> list[str]:
    parts = line.split(",")
    names: list[str] = []
    index = 1
    while index < len(parts):
        name = parts[index].strip()
        if name:
            names.append(name)
        index += _COLUMNS_PER_PROTEIN
    if not names:
        raise ValueError(f"Could not parse protein names from AggreProt CSV header: {line!r}")
    return names
