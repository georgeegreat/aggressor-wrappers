"""WALTZ web runner (waltz.switchlab.org) + parse pipeline."""

from __future__ import annotations

import io
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_first_sequence, read_fasta
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.waltz import WALTZParser, split_detailed_sections
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://waltz.switchlab.org/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_RESULTS_LINK_RE = re.compile(
    r'href=["\'](OUTPUT/WaltzJob_[^/]+/WaltzJob_[^"\']+\.html)["\']',
    re.IGNORECASE,
)
_ZIP_LINK_RE = re.compile(r'href=["\'](WaltzJob_\d+\.zip)["\']', re.IGNORECASE)


class WALTZRunner(BasePredictorRunner):
    """
    Submit (multi-)FASTA to the WALTZ web form, download the job ZIP, and parse
    detailed text output into standard columns.

    Configure via ``[runners.waltz]`` in config.cfg.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        threshold: int | float = 92,
        ph: float = 7.0,
        output_format: str = "text_long",
        timeout_seconds: int | None = 180,
        combined_filename: str = "waltz_combined.txt",
    ) -> None:
        self.base_url = (base_url or os.environ.get("AGGRESSOR_WALTZ_BASE_URL") or _DEFAULT_BASE_URL).rstrip(
            "/"
        ) + "/"
        self.threshold = threshold
        self.ph = ph
        self.output_format = output_format
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None
        self.combined_filename = combined_filename
        self.last_raw_path: Path | None = None

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run WALTZ on a (multi-)FASTA file; return directory with per-protein raw text."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        combined_path = self._submit_and_download(fasta_path, cwd / self.combined_filename)
        protein_ids = list(read_fasta(fasta_path).keys())
        self._split_combined_output(combined_path, protein_ids, cwd)
        return cwd

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map FASTA protein IDs to ``{id}_waltz.txt`` files under ``output_dir``."""
        out_root = Path(output_dir)
        mapping: dict[str, Path] = {}
        for protein_id in protein_ids:
            path = out_root / f"{protein_id}_waltz.txt"
            if path.is_file():
                mapping[protein_id] = path
        return mapping

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_txt: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_txt is not None:
            raw_path = Path(raw_txt)
        elif skip_run:
            raise ValueError("Provide raw_txt when --skip-run is set")
        else:
            cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
            raw_path = self._execute_single(fasta_path, cwd, protein_id)

        self.last_raw_path = raw_path
        parser = WALTZParser()
        return parser.parse(raw_path, protein_id=protein_id, sequence=sequence)

    def _execute_single(self, fasta_path: Path, work_dir: Path, protein_id: str) -> Path:
        out_root = self.execute_batch(fasta_path, work_dir)
        raw_path = out_root / f"{protein_id}_waltz.txt"
        if not raw_path.is_file():
            raise FileNotFoundError(f"WALTZ per-protein output missing: {raw_path}")
        return raw_path

    def _submit_and_download(self, fasta_path: Path, dest_txt: Path) -> Path:
        fasta_text = fasta_path.read_text().strip() + "\n"
        payload = urllib.parse.urlencode(
            {
                "sequence": fasta_text,
                "threshold": str(self.threshold),
                "ph": str(self.ph),
                "output": self.output_format,
                "Submit": "Submit sequences",
            }
        ).encode()

        results_html = self._post(f"{self.base_url}results.cgi", payload)
        if "job ran succes" not in results_html.lower():
            raise RuntimeError("WALTZ submission failed: success message not found on results page")

        match = _RESULTS_LINK_RE.search(results_html)
        if not match:
            raise RuntimeError("WALTZ results page missing job link")

        results_url = urllib.parse.urljoin(self.base_url, match.group(1))
        detail_html = self._get(results_url)

        zip_match = _ZIP_LINK_RE.search(detail_html)
        if not zip_match:
            raise RuntimeError(f"WALTZ results page missing ZIP link: {results_url}")

        zip_url = urllib.parse.urljoin(results_url, zip_match.group(1))
        zip_bytes = self._get_bytes(zip_url)
        txt_name, txt_bytes = self._extract_txt_from_zip(zip_bytes)
        dest_txt.parent.mkdir(parents=True, exist_ok=True)
        dest_txt.write_bytes(txt_bytes)
        self.last_raw_path = dest_txt
        return dest_txt

    def _split_combined_output(
        self,
        combined_path: Path,
        protein_ids: list[str],
        out_dir: Path,
    ) -> dict[str, Path]:
        sections = split_detailed_sections(combined_path.read_text())
        mapping: dict[str, Path] = {}
        for protein_id in protein_ids:
            section = sections.get(protein_id)
            if section is None:
                for key, body in sections.items():
                    if key.strip() == protein_id.strip():
                        section = body
                        break
            if section is None:
                continue
            dest = out_dir / f"{protein_id}_waltz.txt"
            dest.write_text(section)
            mapping[protein_id] = dest
        return mapping

    def _post(self, url: str, data: bytes) -> str:
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("User-Agent", _USER_AGENT)
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        return self._read_text(request)

    def _get(self, url: str) -> str:
        request = urllib.request.Request(url, method="GET")
        request.add_header("User-Agent", _USER_AGENT)
        return self._read_text(request)

    def _get_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(url, method="GET")
        request.add_header("User-Agent", _USER_AGENT)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"WALTZ download failed for {url}: {exc}") from exc

    def _read_text(self, request: urllib.request.Request) -> str:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"WALTZ request failed for {request.full_url}: {exc}") from exc

    @staticmethod
    def _extract_txt_from_zip(zip_bytes: bytes) -> tuple[str, bytes]:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            txt_names = [name for name in archive.namelist() if name.lower().endswith(".txt")]
            if not txt_names:
                raise RuntimeError("WALTZ ZIP archive contains no .txt file")
            txt_name = txt_names[0]
            return txt_name, archive.read(txt_name)
