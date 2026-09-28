"""ArchCandy 2.0 web runner — bioinfo.crbm.cnrs.fr REST API.

The BiSMM site was rebuilt: the old ``index.php?route=tools&tool=7`` entry point
is gone and the service now exposes a genuine JSON API. This is why
:mod:`aggressor_wrappers.runners.archcandy` stopped working — not a transient
outage but a moved endpoint.

Contract, captured from the site's own traffic::

    POST /api/tools/archCandy/jobs            (application/json)
        {"sequence": "<FASTA text>", "threshold": 0.4, "transmembrane": false}
     -> 201 {"jobId": "<uuid>", "status": "QUEUED", "publicToken": "<hex>"}

    GET  /api/tools/archCandy/jobs/{jobId}?token={publicToken}
     -> {"status": "QUEUED|RUNNING|DONE", "expiresAt": ..., "exitCode": 0,
         "errorMessage": null}

    GET  /api/tools/archCandy/jobs/{jobId}/files/csv?token={publicToken}
     -> ID,Sequence,Arch,Start,Stop,Score       <- the region table
    GET  /api/tools/archCandy/jobs/{jobId}/files/input?token=...
    GET  /api/tools/archCandy/jobs/{jobId}/files/selectseq?token=...

``publicToken`` is required on every follow-up call, and results **expire** (the
job reports ``expiresAt``, a few hours out), so a run that intends to keep its
raw output must download it, not bookmark the job.

**Score interpretation — read this before setting a threshold.** ArchCandy 2.0's
own documentation states the calibration explicitly: a prediction is
*non-significant* below 0.40, *ambiguous* between 0.40 and 0.57, and
*significant* above 0.57. The web default is 0.40, which is a recall-oriented
submission filter, not a significance claim. A panel that lowers the cutoff to
force calls out of an otherwise silent protein is therefore not making weaker
predictions — it is collecting arches the authors' own benchmark classifies as
non-significant, and any consensus vote built on them inherits that status.

Output columns differ from the standalone ArchCandy 1.0 build, which writes
``Number, Digram, Score, Arc_type, Position``. They are not interchangeable and
are given separate parsers.

Validation case: Abeta(1-42) at threshold 0.40 returns 14 beta-arch candidates,
the best being ``QKLVFFAEDVGSNKGAIIGLMV`` at 15-36 with score 0.723 — a single
arch spanning both hydrophobic segments (KLVFFA and IGLMV) with the bend between
them, which is the beta-arch architecture reported for Abeta42 fibrils.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_fasta, read_first_sequence
from aggressor_wrappers.core.net import PermanentToolError, retry_call
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.archcandy_v2 import ArchCandy2Parser
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://bioinfo.crbm.cnrs.fr/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_TERMINAL = frozenset({"DONE", "ERROR", "FAILED", "CANCELLED"})

#: The authors' own benchmark bands (ArchCandy-2.0 documentation).
SIGNIFICANCE_BANDS = {
    "non_significant": (0.00, 0.40),
    "ambiguous": (0.40, 0.57),
    "significant": (0.57, 1.00),
}
WEB_DEFAULT_THRESHOLD = 0.40        # ArchCandy 2.0 web form default
SIGNIFICANCE_THRESHOLD = 0.57       # ArchCandy 2.0 significance boundary
#: ArchCandy 1.0's published cutoff (Ahmed et al., 2015, Alzheimers Dement.
#: 11:681). 1.0 is the only downloadable build, so this is the number that
#: applies to a local run -- it is close to 2.0's 0.57 but is a different
#: calibration on a different model, and the two should not be swapped.
LOCAL_1_0_THRESHOLD = 0.56


def significance_band(score: float) -> str:
    """Label a score by ArchCandy 2.0's documented calibration."""
    if score < 0.40:
        return "non_significant"
    if score < 0.57:
        return "ambiguous"
    return "significant"


class ArchCandy2Runner(BasePredictorRunner):
    """Submit one sequence per job to the ArchCandy 2.0 REST API."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        threshold: float = WEB_DEFAULT_THRESHOLD,
        transmembrane: bool = False,
        score_mode: str = "highest",
        poll_interval_seconds: float = 2.0,
        timeout_seconds: int = 1800,
        **_ignored,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_ARCHCANDY2_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        self.threshold = float(threshold)
        self.transmembrane = bool(transmembrane)
        self.score_mode = score_mode
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None
        self.last_job: dict | None = None

    # ------------------------------------------------------------------ #
    def _request(self, path: str, *, payload: dict | None = None) -> tuple[int, str]:
        url = urllib.parse.urljoin(self.base_url, path.lstrip("/"))
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json, text/csv, */*"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        def _call() -> tuple[int, str]:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return response.status, response.read().decode("utf-8", errors="replace")

        try:
            return retry_call(_call)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"ArchCandy 2.0 {path} -> HTTP {exc.code}: {body}") from exc

    def submit(self, fasta_text: str) -> dict:
        status, body = self._request(
            "api/tools/archCandy/jobs",
            payload={
                "sequence": fasta_text,
                "threshold": self.threshold,
                "transmembrane": self.transmembrane,
            },
        )
        job = json.loads(body)
        for key in ("jobId", "publicToken"):
            if key not in job:
                raise RuntimeError(f"ArchCandy 2.0 submission lacked {key!r}: {job}")
        self.last_job = job
        return job

    def wait(self, job: dict) -> dict:
        deadline = time.monotonic() + self.timeout_seconds
        query = urllib.parse.urlencode({"token": job["publicToken"]})
        while True:
            _, body = self._request(f"api/tools/archCandy/jobs/{job['jobId']}?{query}")
            state = json.loads(body)
            status = str(state.get("status", "")).upper()
            if status in _TERMINAL:
                if status != "DONE" or state.get("exitCode") not in (0, None):
                    raise RuntimeError(
                        f"ArchCandy 2.0 job {job['jobId']} ended {status}: "
                        f"{state.get('errorMessage')}"
                    )
                return state
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"ArchCandy 2.0 job {job['jobId']} still {status} after "
                    f"{self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def fetch_csv(self, job: dict) -> str:
        query = urllib.parse.urlencode({"token": job["publicToken"]})
        _, body = self._request(
            f"api/tools/archCandy/jobs/{job['jobId']}/files/csv?{query}"
        )
        return body

    # ------------------------------------------------------------------ #
    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        protein_id, sequence = read_first_sequence(fasta_path)
        job = self.submit(f">{protein_id}\n{sequence}\n")
        state = self.wait(job)
        csv_text = self.fetch_csv(job)

        dest = cwd / f"{protein_id}_archcandy2.csv"
        dest.write_text(csv_text, encoding="utf-8")
        # Results expire server-side; record when, beside the data, so a stale
        # job id in a log is recognisable as expired rather than broken.
        (cwd / f"{protein_id}_archcandy2.job.json").write_text(
            json.dumps({**job, **state, "threshold": self.threshold}, indent=1)
        )
        self.last_raw_path = dest
        return dest

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"ArchCandy 2.0 accepts one sequence per job, got {len(records)}"
            )
        return self.execute(fasta_path, work_dir)

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        root = Path(output_dir)
        return {
            pid: root / f"{pid}_archcandy2.csv"
            for pid in protein_ids
            if (root / f"{pid}_archcandy2.csv").is_file()
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
        return ArchCandy2Parser(score_mode=self.score_mode, threshold=self.threshold).parse(
            raw_path, protein_id=protein_id, sequence=sequence
        )
