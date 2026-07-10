"""APPNN R runner + parse pipeline."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.paths import DEFAULT_APPNN_SCRIPT
from aggressor_wrappers.predictors.appnn import APPNNParser
from aggressor_wrappers.runners.base import BasePredictorRunner


class APPNNRunner(BasePredictorRunner):
    """
    Run ``legacy/appnn_converter.R`` via Rscript, locate ``APPNN_parsed/*.csv``,
    then parse into standard columns.
    """

    def __init__(
        self,
        *,
        rscript: str = "Rscript",
        converter_script: str | Path | None = None,
        output_dir: str = "APPNN_parsed",
        score_threshold: float = 0.5,
        timeout_seconds: int | None = None,
    ) -> None:
        self.rscript = rscript
        script_path = converter_script or os.environ.get("AGGRESSOR_APPNN_SCRIPT") or DEFAULT_APPNN_SCRIPT
        self.converter_script = Path(script_path)
        self.output_dir = output_dir
        self.score_threshold = score_threshold
        self.timeout_seconds = int(timeout_seconds) if timeout_seconds else None
        self.last_raw_path: Path | None = None

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run APPNN R script on a (multi-)FASTA file; return ``APPNN_parsed`` directory."""
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        self._run_appnn_script(fasta_path, cwd)
        out_root = cwd / self.output_dir
        if not out_root.is_dir():
            raise FileNotFoundError(f"APPNN output directory missing: {out_root}")
        return out_root

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map FASTA protein IDs to ``{id}_APPNN.csv`` files under ``output_dir``."""
        out_root = Path(output_dir)
        mapping: dict[str, Path] = {}
        available = {path.stem.removesuffix("_APPNN"): path for path in out_root.glob("*_APPNN.csv")}

        for protein_id in protein_ids:
            clean_id = re.sub(r"[^A-Za-z0-9_]", "_", protein_id)
            for key in (protein_id, clean_id):
                path = out_root / f"{key}_APPNN.csv"
                if path.is_file():
                    mapping[protein_id] = path
                    break
                if key in available:
                    mapping[protein_id] = available[key]
                    break
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
            appnn_csv = Path(raw_csv)
        elif skip_run:
            raise ValueError("Provide --input when --skip-run is set")
        else:
            appnn_csv = self._execute_appnn(fasta_path, work_dir or fasta_path.parent, protein_id)

        self.last_raw_path = appnn_csv
        parser = APPNNParser()
        return parser.parse(
            appnn_csv,
            protein_id=protein_id,
            sequence=sequence,
            score_threshold=self.score_threshold,
        )

    def _execute_appnn(
        self,
        fasta_path: Path,
        work_dir: str | Path | None,
        protein_id: str,
    ) -> Path:
        cwd = Path(work_dir) if work_dir is not None else fasta_path.parent
        cwd.mkdir(parents=True, exist_ok=True)
        self._run_appnn_script(fasta_path, cwd)
        return self._find_appnn_csv(cwd, protein_id)

    def _run_appnn_script(self, fasta_path: Path, cwd: Path) -> None:
        if not self.converter_script.is_file():
            raise FileNotFoundError(f"APPNN converter not found: {self.converter_script}")

        cmd = [self.rscript, str(self.converter_script.resolve()), str(fasta_path.resolve())]
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            msg = (exc.stderr or "").strip() or str(exc)
            raise RuntimeError(f"APPNN R script failed: {msg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"APPNN timed out after {self.timeout_seconds}s") from exc

    def _find_appnn_csv(self, cwd: Path, protein_id: str) -> Path:
        out_root = cwd / self.output_dir
        if not out_root.is_dir():
            raise FileNotFoundError(
                f"APPNN output directory missing: {out_root}. "
                "Ensure R package 'appnn' is installed."
            )

        clean_id = re.sub(r"[^A-Za-z0-9_]", "_", protein_id)
        candidates = [
            out_root / f"{clean_id}_APPNN.csv",
            out_root / f"{protein_id}_APPNN.csv",
        ]
        for path in candidates:
            if path.is_file():
                return path

        matches = sorted(out_root.glob("*_APPNN.csv"))
        if len(matches) == 1:
            return matches[0]
        if matches:
            for path in matches:
                if clean_id in path.stem or protein_id in path.stem:
                    return path
            return matches[0]

        raise FileNotFoundError(f"No APPNN CSV found under {out_root}")
