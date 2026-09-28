"""PASTA 2.0 web runner (old.protein.bio.unipd.it) + parse pipeline."""

from __future__ import annotations

import os
import re
import tarfile
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_first_sequence, read_fasta
from aggressor_wrappers.core.net import PermanentToolError, retry_call
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.pasta import PASTAParser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "http://old.protein.bio.unipd.it/pasta2/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_BATCH_TAR_RE = re.compile(r'href=["\']([^"\']*batch\.tar(?:\.gz)?)["\']', re.IGNORECASE)
_PROFILE_SUFFIX = ".fasta.seq.aggr_profile.dat"
_FREE_ENERGY_SUFFIX = ".fasta.seq.aggr_profile.dat.free_energy"

#: PASTA rewrites each FASTA header before naming its output files, keeping
#: only word characters: dots and hyphens are DELETED, underscores survive.
#:   6087.XP_002162002.2  -> 6087XP_0021620022
#:   7029.ACYPI000203-PA  -> 7029ACYPI000203PA
#: Matching archive members against the original accession therefore fails for
#: every identifier containing a dot -- which is every STRING-style accession,
#: and why a 372-protein panel reported "missing profile" for all ten sequences
#: of its first batch while the service had in fact analysed them correctly.
_HEADER_STRIP_RE = re.compile(r"[^A-Za-z0-9_]")


def pasta_member_key(protein_id: str) -> str:
    """The name PASTA will give this accession inside its archive."""
    return _HEADER_STRIP_RE.sub("", protein_id)


def check_member_key_collisions(protein_ids: list[str]) -> None:
    """Refuse a batch whose accessions collapse onto one PASTA name.

    Because the rewrite is lossy, ``FXR1.A`` and ``FXR1A`` become the same
    member. Silently taking either would attribute one protein's energy profile
    to another, so the batch is rejected with both names instead.
    """
    seen: dict[str, str] = {}
    for protein_id in protein_ids:
        key = pasta_member_key(protein_id)
        if key in seen and seen[key] != protein_id:
            raise ValueError(
                f"PASTA strips non-word characters from FASTA headers, and "
                f"{seen[key]!r} and {protein_id!r} both become {key!r}. Rename "
                f"one, or submit them in different batches."
            )
        seen[key] = protein_id


class PASTARunner(BasePredictorRunner):
    """
    Submit (multi-)FASTA to the PASTA 2.0 web form (Regular mode), poll until
    ``batch.tar.gz`` is ready, extract per-protein aggregation profiles, and parse
    into standard columns.

    Configure via ``[runners.pasta]`` and ``[predictors.pasta]`` in config.cfg.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        npair: int = 22,
        amount: float = -2.8,
        thresdrop: str = "val1",
        poll_interval_seconds: int = 10,
        timeout_seconds: int | None = 3600,
        energy_threshold: float | None = None,
        archive_filename: str = "batch.tar.gz",
        profile_kind: str = "free_energy",
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_PASTA_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        if profile_kind not in ("free_energy", "aggregation"):
            raise ValueError("profile_kind must be 'free_energy' or 'aggregation'")
        # The archive carries BOTH .aggr_profile.dat and
        # .aggr_profile.dat.free_energy. The free-energy profile is the one to
        # threshold: PASTA scores a pairing FREE ENERGY, so more negative means a
        # more stable cross-beta pairing and the detection rule is `below`. The
        # amyloscope ribosomal config already reads .free_energy, while this
        # runner used to extract the other file and exclude that one -- the two
        # halves of the pipeline disagreed about which number PASTA's datum is.
        self.profile_kind = profile_kind
        self.npair = int(npair)
        self.amount = float(amount)
        self.thresdrop = thresdrop
        self.poll_interval_seconds = int(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None
        self.energy_threshold = float(energy_threshold if energy_threshold is not None else amount)
        self.archive_filename = archive_filename
        self.last_raw_path: Path | None = None

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run PASTA on a (multi-)FASTA file; return directory with per-protein profiles."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        archive_path = cwd / self.archive_filename
        self._submit_and_download(fasta_path, archive_path)
        protein_ids = list(read_fasta(fasta_path).keys())
        self._extract_profiles(archive_path, protein_ids, cwd)
        return cwd

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map FASTA protein IDs to ``{id}_pasta.dat`` files under ``output_dir``."""
        out_root = Path(output_dir)
        mapping: dict[str, Path] = {}
        for protein_id in protein_ids:
            for suffix in (".dat", ".txt"):
                path = out_root / f"{protein_id}_pasta{suffix}"
                if path.is_file():
                    mapping[protein_id] = path
                    break
        return mapping

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_profile: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_profile is not None:
            raw_path = Path(raw_profile)
        elif skip_run:
            raise ValueError("Provide raw_profile when --skip-run is set")
        else:
            cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
            raw_path = self._execute_single(fasta_path, cwd, protein_id)

        self.last_raw_path = raw_path
        parser = PASTAParser(energy_threshold=self.energy_threshold)
        return parser.parse(raw_path, protein_id=protein_id, sequence=sequence)

    def _execute_single(self, fasta_path: Path, work_dir: Path, protein_id: str) -> Path:
        out_root = self.execute_batch(fasta_path, work_dir)
        raw_path = out_root / f"{protein_id}_pasta.dat"
        if not raw_path.is_file():
            raise FileNotFoundError(f"PASTA per-protein output missing: {raw_path}")
        return raw_path

    def _submit_and_download(self, fasta_path: Path, dest_archive: Path) -> Path:
        job_url = self._submit_job(fasta_path)
        tar_url = self._wait_for_archive(job_url)
        tar_bytes = self._get_bytes(tar_url)
        dest_archive.parent.mkdir(parents=True, exist_ok=True)
        dest_archive.write_bytes(tar_bytes)
        self.last_raw_path = dest_archive
        return dest_archive

    def _submit_job(self, fasta_path: Path) -> str:
        """Submit the query through the field PASTA actually reads.

        The FASTA goes in ``sequence``, the main textarea. It was previously
        uploaded as ``mutantfastafile``, which on Pasta2.jsp is the file input
        of the *mutant* panel — the one guarded by the ``mutantcheck`` checkbox
        for comparing a variant against a wild type. Submitting there with no
        ``sequence`` sends an empty query, which is why this runner produced no
        usable output while the service itself was working normally.

        ``emailaddress`` is deliberately left absent and ``emailnotice`` unset:
        the service will happily email a result, and a wrapper should not enrol
        anyone in that without being asked.
        """
        fasta_text = fasta_path.read_text(encoding="utf-8")
        body, content_type = _encode_multipart(
            fields={
                "sequence": fasta_text,
                "npair": str(self.npair),
                "amount": str(self.amount),
                "thresdrop": self.thresdrop,
                "Submit Query": "",
            },
            files={},
        )
        request = urllib.request.Request(
            urllib.parse.urljoin(self.base_url, "Pasta2.jsp"),
            data=body,
            method="POST",
        )
        request.add_header("User-Agent", _USER_AGENT)
        request.add_header("Content-Type", content_type)

        def _submit() -> str:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status not in {200, 302}:
                    raise RuntimeError(f"PASTA submission returned HTTP {response.status}")
                return response.geturl()

        try:
            job_url = retry_call(_submit)
        except urllib.error.HTTPError as exc:
            if exc.code not in {302, 303, 307} or not exc.headers.get("Location"):
                raise RuntimeError(f"PASTA submission failed: HTTP {exc.code}") from exc
            job_url = urllib.parse.urljoin(self.base_url, exc.headers["Location"])
        except urllib.error.URLError as exc:
            raise RuntimeError(f"PASTA submission failed: {exc}") from exc

        if "/work/pid_" not in job_url:
            raise RuntimeError(f"PASTA submission did not return a job URL: {job_url}")
        return job_url

    def _wait_for_archive(self, job_url: str) -> str:
        start = time.time()
        while True:
            elapsed = time.time() - start
            if self.timeout_seconds and elapsed > self.timeout_seconds:
                raise RuntimeError(
                    f"PASTA timed out after {self.timeout_seconds}s waiting for batch.tar.gz"
                )

            html = self._get(job_url)
            match = _BATCH_TAR_RE.search(html)
            if match:
                return urllib.parse.urljoin(job_url, match.group(1))

            lowered = html.lower()
            if any(err in lowered for err in ("fatal error", "query failed", "too many sequences")):
                raise RuntimeError("PASTA job failed according to results page")

            time.sleep(self.poll_interval_seconds)

    def _extract_profiles(
        self,
        archive_path: Path,
        protein_ids: list[str],
        out_dir: Path,
    ) -> dict[str, Path]:
        mode = "r:gz" if archive_path.name.endswith((".tar.gz", ".tgz")) else "r"
        suffix = (
            _FREE_ENERGY_SUFFIX if self.profile_kind == "free_energy" else _PROFILE_SUFFIX
        )
        check_member_key_collisions(list(protein_ids))
        mapping: dict[str, Path] = {}
        with tarfile.open(archive_path, mode) as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(suffix)
            }
            for protein_id in protein_ids:
                source_name = self._profile_member_name(protein_id, members, suffix)
                if source_name is None:
                    continue
                dest = out_dir / f"{protein_id}_pasta.dat"
                extracted = archive.extractfile(members[source_name])
                if extracted is None:
                    raise RuntimeError(f"PASTA archive member missing: {source_name}")
                dest.write_bytes(extracted.read())
                mapping[protein_id] = dest
        return mapping

    @staticmethod
    def _profile_member_name(
        protein_id: str, members: dict[str, object], suffix: str = _PROFILE_SUFFIX
    ) -> str | None:
        """Locate one protein's profile, allowing for PASTA's header rewrite.

        The exact accession is tried first so an archive that preserved it still
        matches; the stripped form is the one that actually occurs. Results also
        come back in a different order than submitted, so position is never used.
        """
        key = pasta_member_key(protein_id)
        for candidate in (
            f"predictions/{protein_id}{suffix}",
            f"predictions/{key}{suffix}",
        ):
            if candidate in members:
                return candidate
        for name in members:
            stem = Path(name).name.removesuffix(suffix)
            if stem in (protein_id, key, f"{protein_id}.fasta", f"{key}.fasta"):
                return name
        return None

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
            raise RuntimeError(f"PASTA download failed for {url}: {exc}") from exc

    def _read_text(self, request: urllib.request.Request) -> str:
        def _call() -> str:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")

        try:
            return retry_call(_call)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"PASTA request failed for {request.full_url}: {exc}") from exc


def _encode_multipart(
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----aggressor-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )

    for name, (filename, content, content_type) in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{filename}"\r\n'
                ).encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode())
    content_type = f"multipart/form-data; boundary={boundary}"
    return b"".join(chunks), content_type
