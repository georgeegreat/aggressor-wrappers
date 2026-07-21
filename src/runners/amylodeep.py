"""Run AmyloDeep locally via its ``amylodeep`` console script.

Invocation, matching the documented CLI::

    amylodeep --output <file.csv> --format csv "<SEQUENCE>"

Three operational notes, each found by installing and running the package:

* **It is sequence-at-a-time, not FASTA-at-a-time.** The CLI takes a bare
  sequence string as a positional argument, so a multifasta is handled by looping
  and writing one output file per record. That loop is cheap relative to model
  load, which is why the runner keeps a single working directory per batch.
* **First run downloads model weights from ``huggingface.co``.** In an air-gapped
  or allowlisted environment the run fails with a Hub lookup error rather than a
  clean message, so the runner surfaces that case explicitly. Pre-populating the
  HF cache is the fix.
* **It needs ``pkg_resources``** (via ``jax_unirep``), which is absent from
  setuptools >= 81; the failure is an opaque ``ModuleNotFoundError`` at import.

Output positions are 0-based and may be window starts rather than residues; that
is handled in :mod:`aggressor_wrappers.predictors.amylodeep`, not here.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.predictors.amylodeep import parse_amylodeep
from aggressor_wrappers.runners.base import BasePredictorRunner


class AmyloDeepRunner(BasePredictorRunner):
    """Execute the local ``amylodeep`` CLI, one invocation per sequence."""

    def __init__(
        self,
        *,
        executable: str = "amylodeep",
        threshold: float = 0.5,
        output_format: str = "csv",
        timeout_seconds: int = 1800,
        **_ignored,
    ) -> None:
        self.executable = executable
        self.threshold = float(threshold)
        if output_format not in ("csv", "json"):
            raise ValueError("output_format must be 'csv' or 'json'")
        self.output_format = output_format
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        return shutil.which(self.executable) is not None

    def require_available(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                f"{self.executable!r} not on PATH. Install with "
                f"`pip install amylodeep`, or use backend=web."
            )

    # ------------------------------------------------------------------ #
    def _run_one(self, sequence: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                self.executable,
                "--output",
                str(dest),
                "--format",
                self.output_format,
                sequence,
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if not dest.exists():
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if "huggingface.co" in combined or "Hub" in combined:
                raise RuntimeError(
                    "AmyloDeep could not fetch its model weights from huggingface.co. "
                    "The first run needs network access; in a restricted environment, "
                    "pre-populate the HF cache (HF_HOME) on a connected machine.\n"
                    f"{combined[:400]}"
                )
            if "pkg_resources" in combined:
                raise RuntimeError(
                    "AmyloDeep's dependency jax_unirep imports pkg_resources, which "
                    "setuptools >= 81 removed. Install `setuptools<81` in the same "
                    f"environment.\n{combined[:300]}"
                )
            raise RuntimeError(f"AmyloDeep produced no output.\n{combined[:400]}")
        return dest

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run AmyloDeep for every record; return the directory of raw outputs."""
        self.require_available()
        work = Path(work_dir)
        out_dir = work / "amylodeep_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        for protein_id, sequence in read_fasta(fasta_path).items():
            self._run_one(sequence, out_dir / f"{protein_id}.{self.output_format}")
        self.last_raw_path = out_dir
        return out_dir

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for pid in protein_ids:
            candidate = Path(output_dir) / f"{pid}.{self.output_format}"
            if candidate.exists():
                found[pid] = candidate
        return found

    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        raw_csv: str | Path | None = None,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        records = read_fasta(fasta_path)
        pid = protein_id or next(iter(records))
        sequence = records[pid]

        if raw_csv is not None:
            raw = Path(raw_csv)
        else:
            work = Path(work_dir) if work_dir else fasta_path.parent
            self.require_available()
            raw = self._run_one(
                sequence,
                Path(work) / "amylodeep_out" / f"{pid}.{self.output_format}",
            )

        self.last_raw_path = raw
        scores, meta = parse_amylodeep(raw, sequence=sequence, sequence_id=pid)
        if len(scores) != len(sequence):
            raise ValueError(
                f"AmyloDeep profile length {len(scores)} != sequence length "
                f"{len(sequence)} for {pid!r}"
            )
        binary = binary_from_scores(scores, threshold=self.threshold)
        meta["threshold"] = self.threshold
        return PredictorResult(
            protein_id=pid,
            sequence=sequence,
            spec=get_predictor_spec("amylodeep"),
            scores=scores,
            binary=binary,
            metadata=meta,
        )
