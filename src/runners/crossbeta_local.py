"""Run the local Cross-Beta predictor (``CB_RF_pred.py``).

Source: https://github.com/Valentin-Gonay/cross-beta-predictor

Invocation::

    python3 CB_RF_pred.py -i <input.fasta> -it fasta -t <threshold> -o <name>

Three properties of the script drive this runner, all read from its source:

**Output goes to the tool's own directory, not yours.** Both the model path and
the results path are resolved against ``path.dirname(__file__)``::

    result_path = path.join(current_directory, "results/")

so ``-o`` names a *file*, not a location: results always land in
``<repo>/results/<name>.csv`` regardless of the working directory. The runner
therefore collects the file from there and moves it into the pipeline's work
directory.

**That fixed location is a concurrency hazard.** Since predictors now run
concurrently (see ``batch/scheduler.py``), two invocations writing
``results/prediction_result.csv`` would race and one would silently read the
other's output. Each invocation is given a unique output name.

**IUPred3 is a separate, licensed dependency.** Cross-Beta calls IUPred3 for
disorder prediction; it is under academic licence and must be requested from
https://iupred3.elte.hu/download_new and unpacked into ``utils/iupred3/`` inside
the Cross-Beta repository. Without it the script fails at import with
``ModuleNotFoundError: No module named 'utils.iupred3'``, so the runner checks
for it up front and says so, rather than surfacing an import traceback.

Cross-Beta also has a web backend, so ``backend = auto`` can fall back when the
local checkout is absent.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.predictors.crossbeta_local import crossbeta_profile
from aggressor_wrappers.runners.base import BasePredictorRunner

SCRIPT_NAME = "CB_RF_pred.py"


class CrossBetaLocalRunner(BasePredictorRunner):
    """Execute a local checkout of cross-beta-predictor over a FASTA file."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        python: str | None = None,
        threshold: float = 0.54,
        window_size: int = 0,
        draw_graph: bool = False,
        timeout_seconds: int = 3600,
        **_ignored,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser()
        self.python = python or sys.executable
        # 0.54 is the script's own default classification threshold.
        self.threshold = float(threshold)
        # 0 lets the tool choose the prediction window.
        self.window_size = int(window_size)
        self.draw_graph = bool(draw_graph)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    @property
    def script(self) -> Path:
        return self.repo_path / SCRIPT_NAME

    @property
    def results_dir(self) -> Path:
        """Where the tool writes, regardless of cwd or -o."""
        return self.repo_path / "results"

    def is_available(self) -> bool:
        return self.script.exists() and (self.repo_path / "utils" / "iupred3").exists()

    def require_available(self) -> None:
        if not self.script.exists():
            raise FileNotFoundError(
                f"{SCRIPT_NAME} not found under {self.repo_path}. Clone "
                f"https://github.com/Valentin-Gonay/cross-beta-predictor and set "
                f"[runners.crossbeta] repo_path, or use backend=web."
            )
        if not (self.repo_path / "utils" / "iupred3").exists():
            raise FileNotFoundError(
                "Cross-Beta needs IUPred3, which is under academic licence and is "
                "not bundled. Request it from https://iupred3.elte.hu/download_new "
                f"and unpack it as {self.repo_path / 'utils' / 'iupred3'}."
            )

    # ------------------------------------------------------------------ #
    def _command(self, fasta: Path, out_name: str) -> list[str]:
        cmd = [
            self.python,
            str(self.script),
            "-i",
            str(fasta),
            "-it",
            "fasta",
            "-t",
            f"{self.threshold:g}",
            "-o",
            out_name,
        ]
        if self.window_size:
            cmd += ["-ws", str(self.window_size)]
        if self.draw_graph:
            cmd.append("-g")
        return cmd

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run Cross-Beta over a (multi-)FASTA; return the collected result file."""
        self.require_available()
        work = Path(work_dir)
        out_dir = work / "crossbeta_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Unique name: the tool writes into its own results/ directory, which is
        # shared across concurrent predictor invocations.
        out_name = f"cb_{uuid.uuid4().hex[:12]}.csv"
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._command(fasta_path, out_name),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        produced = self.results_dir / out_name
        if not produced.exists():
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if "iupred3" in combined.lower():
                raise RuntimeError(
                    "Cross-Beta failed to import IUPred3. It is academically "
                    "licensed and must be unpacked into "
                    f"{self.repo_path / 'utils' / 'iupred3'}.\n{combined[:300]}"
                )
            raise RuntimeError(f"Cross-Beta produced no result file.\n{combined[:400]}")

        collected = out_dir / produced.name
        shutil.move(str(produced), str(collected))
        self.last_raw_path = collected
        return collected

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """One result file holds every protein, so all ids map to it."""
        files = sorted(Path(output_dir).glob("cb_*.csv"))
        if not files:
            return {}
        return {pid: files[-1] for pid in protein_ids}

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
            raw = self.execute_batch(fasta_path, work)

        self.last_raw_path = raw
        scores, regions, meta = crossbeta_profile(raw, sequence, query_name=pid)
        binary = binary_from_scores(scores, threshold=self.threshold)
        meta["threshold"] = self.threshold
        return PredictorResult(
            protein_id=pid,
            sequence=sequence,
            spec=get_predictor_spec("crossbeta"),
            scores=scores,
            binary=binary,
            metadata=meta,
            regions=[{"start": s, "stop": e} for s, e in regions],
        )
