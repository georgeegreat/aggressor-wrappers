"""Cross-Beta-Pred web runner — bioinfo.crbm.cnrs.fr REST API.

This replaces an earlier refusal that was **wrong**. The page form at
``/crossBetaPred`` carries a ``g-recaptcha-response`` field, and on that basis
the runner declined to submit. But the CAPTCHA belongs to the page, not to the
service: the job API accepts an anonymous ``POST`` with ``credentials: omit``
and no token at all, exactly as ArchCandy 2.0 does. Nothing here works around a
CAPTCHA — the endpoint simply does not ask for one.

The earlier probe that "proved" no API existed used ``crossBetaPred`` in the
path. The route is spelled ``crossbetaPred``, with a lower-case b, so every
attempt 404'd and the absence looked real. The bundle's own tool→path map is
the authority::

    archCandy -> /tools/archCandy/...      crossBetaPred -> /tools/crossbetaPred/...

Contract, captured live::

    POST /api/tools/crossbetaPred/jobs                  (application/json)
         {"sequence": "<FASTA>", "threshold": 0.5, "windowSize": "auto"}
      -> 201 {"jobId": "<uuid>", "status": "QUEUED", "publicToken": "<hex>"}

    GET  /api/tools/crossbetaPred/jobs/{jobId}?token={publicToken}
      -> {"status": "QUEUED|RUNNING|DONE", "expiresAt", "exitCode", "errorMessage"}

    GET  /api/tools/crossbetaPred/jobs/{jobId}/files/result?token={publicToken}
    GET  /api/tools/crossbetaPred/jobs/{jobId}/files/input?token={publicToken}

``files/result`` — note ``result``, not ``csv``; there is no CSV member — returns::

    [{"prot_name": "sequence_query",
      "All_sequence_pred": 0.7548,
      "AA_list": [{"index": 0, "amino_acid": "D",
                   "score_list": [...15 values...],
                   "mean_confidence": 0.5763}, ...],
      "mean_list": [{"D": 0.5763}, ...],
      "AR_list": [[1, 42]]}]

Three details that will silently corrupt a track if missed:

* ``index`` is **0-based**; every other predictor in this panel is 1-based.
* the per-residue score is ``mean_confidence``, not ``score`` — ``score_list``
  holds the 15 window values it averages.
* ``prot_name`` is the literal ``"sequence_query"``, never the submitted
  accession, so the protein id must come from the caller and one sequence is
  submitted per job.

``score_list`` having 15 entries is the model's window size made visible:
Cross-Beta is trained on 15-residue windows, so it detects extended
aggregation-prone stretches and cannot resolve a nucleating hexapeptide. On
Abeta(1-42) its ``AR_list`` is ``[[1, 42]]`` — the entire peptide as one region,
with a profile that ramps monotonically to 0.854 at residue 38 rather than
peaking on KLVFFA. That is why it behaves as the panel's broadest caller and why
its support for a region boundary should be weighted accordingly.
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
from aggressor_wrappers.core.net import retry_call
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://bioinfo.crbm.cnrs.fr/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_TERMINAL = frozenset({"DONE", "ERROR", "FAILED", "CANCELLED"})
#: Lower-case b. The capitalised spelling 404s.
_TOOL = "crossbetaPred"


def parse_result(document, *, sequence: str) -> dict:
    """Normalise one Cross-Beta record into per-residue arrays."""
    records = document[0] if isinstance(document, list) else document
    aa_list = records.get("AA_list") or []
    if not aa_list:
        raise ValueError("Cross-Beta result has an empty AA_list")
    if len(aa_list) != len(sequence):
        raise ValueError(
            f"Cross-Beta returned {len(aa_list)} residues for a "
            f"{len(sequence)}-residue query"
        )
    observed = "".join(str(entry["amino_acid"]) for entry in aa_list)
    if observed.upper() != sequence.upper():
        raise ValueError(
            "Cross-Beta echoed a different sequence than was queried; refusing "
            "to attribute the result."
        )
    scores = [float(entry["mean_confidence"]) for entry in aa_list]
    # index is 0-based here and 1-based everywhere else in this panel.
    numbers = [int(entry["index"]) + 1 for entry in aa_list]
    in_ar = [0] * len(sequence)
    for interval in records.get("AR_list") or []:
        start, stop = int(interval[0]), int(interval[1])
        for position in range(max(1, start), min(len(sequence), stop) + 1):
            in_ar[position - 1] = 1
    return {
        "numbers": numbers,
        "scores": scores,
        "in_AR": in_ar,
        "whole_protein": records.get("All_sequence_pred"),
        "window": len(aa_list[0].get("score_list") or []),
    }


class CrossBeta2Runner(BasePredictorRunner):
    """Submit one sequence per job to the Cross-Beta-Pred REST API."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        confidence_threshold: float = 0.54,
        window_size: str | int = "auto",
        use_tool_regions: bool = True,
        poll_interval_seconds: float = 2.0,
        timeout_seconds: int = 1800,
        **_ignored,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("AGGRESSOR_CROSSBETA_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/") + "/"
        self.confidence_threshold = float(confidence_threshold)
        self.window_size = window_size
        # Cross-Beta's own AR_list embeds its windowing; a fixed cut on the
        # smoothed confidence is not equivalent and is opt-in.
        self.use_tool_regions = bool(use_tool_regions)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    def _request(self, path: str, *, payload: dict | None = None) -> str:
        url = urllib.parse.urljoin(self.base_url, path.lstrip("/"))
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json, */*"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)

        def _call() -> str:
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                return response.read().decode("utf-8", errors="replace")

        try:
            return retry_call(_call)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Cross-Beta {path} -> HTTP {exc.code}: {body}") from exc

    def submit(self, fasta_text: str) -> dict:
        job = json.loads(
            self._request(
                f"api/tools/{_TOOL}/jobs",
                payload={
                    "sequence": fasta_text,
                    "threshold": self.confidence_threshold,
                    "windowSize": self.window_size,
                },
            )
        )
        for key in ("jobId", "publicToken"):
            if key not in job:
                raise RuntimeError(f"Cross-Beta submission lacked {key!r}: {job}")
        return job

    def wait(self, job: dict) -> dict:
        deadline = time.monotonic() + self.timeout_seconds
        query = urllib.parse.urlencode({"token": job["publicToken"]})
        while True:
            state = json.loads(
                self._request(f"api/tools/{_TOOL}/jobs/{job['jobId']}?{query}")
            )
            status = str(state.get("status", "")).upper()
            if status in _TERMINAL:
                if status != "DONE" or state.get("exitCode") not in (0, None):
                    raise RuntimeError(
                        f"Cross-Beta job {job['jobId']} ended {status}: "
                        f"{state.get('errorMessage')}"
                    )
                return state
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Cross-Beta job {job['jobId']} still {status} after "
                    f"{self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def fetch_result(self, job: dict) -> str:
        query = urllib.parse.urlencode({"token": job["publicToken"]})
        return self._request(f"api/tools/{_TOOL}/jobs/{job['jobId']}/files/result?{query}")

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        protein_id, sequence = read_first_sequence(fasta_path)
        job = self.submit(f">{protein_id}\n{sequence}\n")
        self.wait(job)
        dest = cwd / f"{protein_id}_crossbeta.json"
        dest.write_text(self.fetch_result(job), encoding="utf-8")
        self.last_raw_path = dest
        return dest

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        records = read_fasta(fasta_path)
        if len(records) != 1:
            raise ValueError(
                f"Cross-Beta accepts one sequence per job (prot_name is always "
                f"'sequence_query', so a multi-sequence job cannot be demultiplexed); "
                f"got {len(records)}"
            )
        self.execute(fasta_path, work_dir)
        return Path(work_dir)

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        root = Path(output_dir)
        return {
            pid: root / f"{pid}_crossbeta.json"
            for pid in protein_ids
            if (root / f"{pid}_crossbeta.json").is_file()
        }

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_json: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_json is not None:
            raw_path = Path(raw_json)
        elif skip_run:
            raise ValueError("Provide raw_json when --skip-run is set")
        else:
            cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
            raw_path = self.execute(fasta_path, cwd)

        self.last_raw_path = raw_path
        parsed = parse_result(json.loads(Path(raw_path).read_text()), sequence=sequence)
        scores = parsed["scores"]
        binary = (
            list(parsed["in_AR"])
            if self.use_tool_regions
            else binary_from_scores(scores, threshold=self.confidence_threshold)
        )
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=get_predictor_spec("crossbeta"),
            scores=scores,
            binary=binary,
            metadata={
                "confidence_threshold": self.confidence_threshold,
                "binarised_from": "AR_list" if self.use_tool_regions else "threshold",
                "whole_protein_prediction": parsed["whole_protein"],
                "window": parsed["window"],
                "source": str(raw_path),
            },
            aux={"in_AR": [float(v) for v in parsed["in_AR"]]},
        )
