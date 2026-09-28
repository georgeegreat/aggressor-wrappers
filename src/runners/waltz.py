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
from aggressor_wrappers.core.net import PermanentToolError, retry_call
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

    def fetch_per_residue(self, fasta_path: Path, work_dir: str | Path) -> dict[str, Path]:
        """Harvest WALTZ's per-residue ``.dat`` files.

        **Only ``output=text_long_graph`` produces them.** Under ``text_long``
        the archive contains a single region table and nothing positional, so a
        pipeline configured for ``text_long`` can never generate the
        ``WaltzJob_<id>_<n>.dat`` files that amyloscope's ``waltz`` adapter
        reads — which is why that adapter was being pointed at files nobody
        could regenerate.

        The ``.dat`` layout is ``position<TAB>score``, zero outside
        position-specific-matrix hits, e.g. for Abeta42 every residue is 0.0
        except 16-21 at 97.993311 (``KLVFFA``). It is strictly richer than the
        region table, which is derivable from it as the non-zero runs.

        Files inside the archive are indexed by SUBMISSION ORDER
        (``..._1.dat``), not by accession, so they are mapped back through the
        order of the FASTA that was sent. Any reordering between submission and
        extraction would silently mis-assign one protein's track to another, so
        the mapping is done here, once, next to the request that fixed it.

        This is a second request when ``output_format`` is not already
        ``text_long_graph``: WALTZ decides the archive's contents at submission
        time, so the two outputs cannot be obtained from one job.
        """
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        order = list(read_fasta(fasta_path))
        zip_bytes = self._submit_zip(fasta_path, output_format="text_long_graph")
        written: dict[str, Path] = {}
        for name, data in self._extract_members(zip_bytes, ".dat"):
            index = self._dat_index(name)
            if index is None or index > len(order):
                continue
            accession = order[index - 1]
            dest = cwd / f"{accession}_waltz.dat"
            dest.write_bytes(data)
            written[accession] = dest
        if not written:
            raise RuntimeError(
                "WALTZ returned no .dat members even under text_long_graph; "
                "the archive layout changed."
            )
        return written

    @staticmethod
    def _dat_index(name: str) -> int | None:
        match = re.search(r"_(\d+)\.dat$", name)
        return int(match.group(1)) if match else None

    def _submit_zip(self, fasta_path: Path, *, output_format: str | None = None) -> bytes:
        """Submit and return the result archive's bytes."""
        fasta_text = fasta_path.read_text().strip() + "\n"
        payload = urllib.parse.urlencode(
            {
                "sequence": fasta_text,
                "threshold": str(self.threshold),
                "ph": str(self.ph),
                "output": output_format or self.output_format,
                "Submit": "Submit sequences",
            }
        ).encode()
        results_html = self._post(f"{self.base_url}results.cgi", payload)
        if "job ran succes" not in results_html.lower():
            raise RuntimeError(
                "WALTZ submission failed: success message not found on results page"
            )
        match = _RESULTS_LINK_RE.search(results_html)
        if not match:
            raise RuntimeError("WALTZ results page missing job link")
        results_url = urllib.parse.urljoin(self.base_url, match.group(1))
        detail_html = self._get(results_url)
        zip_match = _ZIP_LINK_RE.search(detail_html)
        if not zip_match:
            raise RuntimeError(f"WALTZ results page missing ZIP link: {results_url}")
        return self._get_bytes(urllib.parse.urljoin(results_url, zip_match.group(1)))

    @staticmethod
    def _extract_members(zip_bytes: bytes, suffix: str) -> list[tuple[str, bytes]]:
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            return [
                (name, archive.read(name))
                for name in sorted(archive.namelist())
                if name.endswith(suffix)
            ]

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
        def _call() -> bytes:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()

        try:
            return retry_call(_call)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"WALTZ download failed for {url}: {exc}") from exc

    def _read_text(self, request: urllib.request.Request) -> str:
        """Read a WALTZ page, retrying transient TLS/connection drops.

        waltz.switchlab.org closes connections mid-handshake often enough that
        an unretried run of a few hundred sequences is unlikely to finish: a
        372-protein sweep died after ~120 with UNEXPECTED_EOF_WHILE_READING and
        discarded every batch already parsed.
        """
        def _call() -> str:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")

        try:
            return retry_call(_call)
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
