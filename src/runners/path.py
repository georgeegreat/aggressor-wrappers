"""PATH threading runner + parse pipeline."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.paths import DEFAULT_PATH_SCRIPT
from aggressor_wrappers.predictors.path import PATHParser
from aggressor_wrappers.runners.base import BasePredictorRunner


class PATHRunner(BasePredictorRunner):
    """
    Run the PATH tool (``python path1.1.py -f FASTA -o work_dir``) then parse
    ``results.csv`` into standard columns.

    Configure via ``[runners.path]`` in config.cfg or env
    ``AGGRESSOR_PATH_SCRIPT``.
    """

    def __init__(
        self,
        *,
        script: str | None = None,
        python: str = "python3",
        results_filename: str = "results.csv",
        threshold_percentile: float = 75.0,
        timeout_seconds: int | None = None,
    ) -> None:
        self.script = script or os.environ.get("AGGRESSOR_PATH_SCRIPT") or str(DEFAULT_PATH_SCRIPT)
        self.python = os.environ.get("AGGRESSOR_PATH_PYTHON") or python
        self.results_filename = results_filename
        self.last_raw_path: Path | None = None
        self.threshold_percentile = threshold_percentile
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None

    def execute(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run PATH on ``fasta_path`` and return ``results.csv`` path."""
        return self._execute_path(fasta_path, work_dir)

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        results_csv: str | Path | None = None,
        skip_run: bool = False,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if results_csv is not None:
            raw_path = Path(results_csv)
        elif skip_run:
            raise ValueError("Provide --results when --skip-run is set")
        else:
            raw_path = self._execute_path(fasta_path, work_dir)

        self.last_raw_path = raw_path
        parser = PATHParser(threshold_percentile=self.threshold_percentile)
        return parser.parse(
            raw_path,
            protein_id=protein_id,
            sequence=sequence,
        )

    def _execute_path(
        self,
        fasta_path: Path,
        work_dir: str | Path | None,
    ) -> Path:
        if not self.script:
            raise RuntimeError(
                "PATH script not configured and bundled vendor/PATH/path1.1.py is missing. "
                "Set [runners.path].script in config.cfg or AGGRESSOR_PATH_SCRIPT."
            )

        script_path = Path(self.script)
        if not script_path.is_file():
            raise FileNotFoundError(f"PATH script not found: {script_path}")

        cleanup = False
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="amyloid_path_"))
            cleanup = True
        else:
            work_dir = Path(work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

        cmd = [self.python, str(script_path), "-f", str(fasta_path), "-o", str(work_dir)]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or "").strip() or str(exc)
            raise RuntimeError(f"PATH failed: {msg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"PATH timed out after {self.timeout_seconds}s "
                "(increase [runners.path].timeout_seconds or use --skip-run)"
            ) from exc

        results_path = self._find_results_csv(work_dir)
        if cleanup:
            # Keep results in work_dir when caller provided it; for temp dirs copy out
            dest = fasta_path.parent / f"{fasta_path.stem}_{self.results_filename}"
            shutil.copy2(results_path, dest)
            shutil.rmtree(work_dir, ignore_errors=True)
            return dest
        return results_path

    def _find_results_csv(self, work_dir: Path) -> Path:
        direct = work_dir / self.results_filename
        if direct.is_file():
            return direct

        matches = sorted(work_dir.rglob(self.results_filename))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"PATH output missing {self.results_filename!r} under {work_dir}. "
            "Check PATH installation or pass --results manually."
        )
