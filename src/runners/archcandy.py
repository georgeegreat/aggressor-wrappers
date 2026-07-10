"""ArchCandy web runner (bioinfo.crbm.cnrs.fr) + parse pipeline."""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_first_sequence, read_fasta
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.archcandy import ArchCandyParser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://bioinfo.crbm.cnrs.fr/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_TERMINAL_STATUSES = frozenset({"DONE", "ERROR", "FAILED"})


class ArchCandyRunner(BasePredictorRunner):
    """
    Submit one FASTA record per job to the ArchCandy REST API, poll until
    completion, download the region CSV, and parse into standard columns.

    Configure via ``[runners.archcandy]`` in config.cfg.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        threshold: float = 0.4,
        transmembrane: bool = False,
        poll_interval_seconds: int = 2,
        timeout_seconds: int | None = 600,
        verify_ssl: bool = False,
        score_mode: str = "cumulative",
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_ARCHCANDY_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        self.threshold = float(threshold)
        self.transmembrane = bool(transmembrane)
        self.poll_interval_seconds = int(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None
        self.verify_ssl = bool(verify_ssl)
        self.score_mode = str(score_mode)
        self.last_raw_path: Path | None = None

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run ArchCandy on a single-record FASTA and return the region CSV path."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        protein_id, _sequence = read_first_sequence(fasta_path)
        fasta_text = self._format_submission_fasta(fasta_path, protein_id=protein_id)
        csv_path = cwd / f"{protein_id}_archcandy.csv"
        self._submit_and_download(fasta_text, csv_path)
        return csv_path

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run ArchCandy on the single FASTA record in ``fasta_path`` (one API job)."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"ArchCandy batch expects exactly one sequence per job, got {len(records)}"
            )
        protein_id = next(iter(records))
        single_fasta = cwd / f"{protein_id}.fasta"
        single_fasta.write_text(self._format_submission_fasta(fasta_path, protein_id=protein_id))
        return self.execute(single_fasta, cwd)

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map FASTA protein IDs to ``{id}_archcandy.csv`` under ``output_dir``."""
        out_root = Path(output_dir)
        mapping: dict[str, Path] = {}
        for protein_id in protein_ids:
            path = out_root / f"{protein_id}_archcandy.csv"
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
        parser = ArchCandyParser(score_mode=self.score_mode)
        return parser.parse(raw_path, protein_id=protein_id, sequence=sequence)

    def _format_submission_fasta(self, fasta_path: Path, *, protein_id: str) -> str:
        _, sequence = read_first_sequence(fasta_path)
        return f">{protein_id}\n{sequence}\n"

    def _submit_and_download(self, fasta_text: str, dest_csv: Path) -> Path:
        job_id, token = self._create_job(fasta_text)
        self._wait_for_completion(job_id, token)
        csv_bytes = self._download_file(job_id, token, file_type="csv")
        dest_csv.parent.mkdir(parents=True, exist_ok=True)
        dest_csv.write_bytes(csv_bytes)
        self.last_raw_path = dest_csv
        return dest_csv

    def _create_job(self, fasta_text: str) -> tuple[str, str]:
        payload = json.dumps(
            {
                "sequence": fasta_text,
                "threshold": self.threshold,
                "transmembrane": self.transmembrane,
            }
        ).encode()
        url = urllib.parse.urljoin(self.base_url, "api/tools/archCandy/jobs")
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "User-Agent": _USER_AGENT,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        response = self._read_json(request)
        job_id = str(response.get("jobId") or "")
        token = str(response.get("publicToken") or "")
        if not job_id or not token:
            raise RuntimeError(f"ArchCandy job creation returned incomplete payload: {response}")
        return job_id, token

    def _wait_for_completion(self, job_id: str, token: str) -> None:
        start = time.time()
        status_url = self._job_url(job_id, token)
        while True:
            elapsed = time.time() - start
            if self.timeout_seconds and elapsed > self.timeout_seconds:
                raise RuntimeError(
                    f"ArchCandy timed out after {self.timeout_seconds}s waiting for job {job_id}"
                )

            job = self._get_json(status_url)
            status = str(job.get("status") or "")
            if status in _TERMINAL_STATUSES:
                if status != "DONE":
                    message = job.get("errorMessage") or status
                    raise RuntimeError(f"ArchCandy job {job_id} failed: {message}")
                return
            time.sleep(self.poll_interval_seconds)

    def _download_file(self, job_id: str, token: str, *, file_type: str) -> bytes:
        url = self._job_file_url(job_id, token, file_type)
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
        )
        return self._read_bytes(request)

    def _job_url(self, job_id: str, token: str) -> str:
        query = urllib.parse.urlencode({"token": token})
        return urllib.parse.urljoin(
            self.base_url,
            f"api/tools/archCandy/jobs/{urllib.parse.quote(job_id)}?{query}",
        )

    def _job_file_url(self, job_id: str, token: str, file_type: str) -> str:
        query = urllib.parse.urlencode({"token": token})
        return urllib.parse.urljoin(
            self.base_url,
            f"api/tools/archCandy/jobs/{urllib.parse.quote(job_id)}/files/{file_type}?{query}",
        )

    def _get_json(self, url: str) -> dict:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
        return self._read_json(request)

    def _read_json(self, request: urllib.request.Request) -> dict:
        data = self._read_bytes(request)
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("ArchCandy API returned non-object JSON")
        return payload

    def _read_bytes(self, request: urllib.request.Request) -> bytes:
        try:
            with urllib.request.urlopen(request, context=self._ssl_context(), timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ArchCandy request failed for {request.full_url}: {exc}") from exc

    def _ssl_context(self) -> ssl.SSLContext:
        if self.verify_ssl:
            return ssl.create_default_context()
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
