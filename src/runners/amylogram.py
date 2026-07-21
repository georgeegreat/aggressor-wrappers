"""Run AmyloGram locally through Rscript.

AmyloGram is an R package, so the runner shells out the same way the APPNN runner
does. The R side is kept deliberately thin — it loads the model, scores a FASTA of
peptides, and writes ``name,probability``. The windowing and the projection back
onto residues stay in Python (:mod:`aggressor_wrappers.predictors.amylogram`),
where they are testable and where the modelling choice is visible rather than
buried in a helper script.

Requires ``Rscript`` on PATH and the ``AmyloGram`` and ``seqinr`` R packages::

    install.packages(c("AmyloGram", "seqinr"))

There is also a web server, so this predictor can fall back
(``backend = auto``) if R is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.predictors.amylogram import (
    DEFAULT_AGGREGATION,
    DEFAULT_WINDOW,
    R_SCRIPT,
    parse_amylogram_output,
    project_windows,
    write_peptide_fasta,
)
from aggressor_wrappers.runners.base import BasePredictorRunner


class AmyloGramRunner(BasePredictorRunner):
    """Score sliding hexapeptides with AmyloGram and project onto residues."""

    def __init__(
        self,
        *,
        rscript: str = "Rscript",
        script_path: str | Path | None = None,
        window: int = DEFAULT_WINDOW,
        aggregation: str = DEFAULT_AGGREGATION,
        threshold: float = 0.5,
        timeout_seconds: int = 1800,
        **_ignored,
    ) -> None:
        self.rscript = rscript
        self.script_path = Path(script_path) if script_path else None
        self.window = int(window)
        if aggregation not in ("max", "mean"):
            raise ValueError("aggregation must be 'max' or 'mean'")
        self.aggregation = aggregation
        self.threshold = float(threshold)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        return shutil.which(self.rscript) is not None

    def require_available(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                f"{self.rscript!r} not on PATH. AmyloGram is an R package: install R "
                f"plus `install.packages(c('AmyloGram','seqinr'))`, or use backend=web."
            )

    def _ensure_script(self, work: Path) -> Path:
        if self.script_path and self.script_path.exists():
            return self.script_path
        dest = work / "amylogram_predict.R"
        return dest

    # ------------------------------------------------------------------ #
    def _score_peptides(self, peptides_fasta: Path, work: Path) -> Path:
        script = self._ensure_script(work)
        out_csv = work / f"{peptides_fasta.stem}_amylogram.csv"
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [self.rscript, str(script), str(peptides_fasta), str(out_csv)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if not out_csv.exists():
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if "there is no package" in combined or "AmyloGram" in combined:
                raise RuntimeError(
                    "AmyloGram's R packages are missing. In R: "
                    f"install.packages(c('AmyloGram','seqinr')).\n{combined[:300]}"
                )
            raise RuntimeError(f"AmyloGram produced no output.\n{combined[:400]}")
        return out_csv

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

        work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        work.mkdir(parents=True, exist_ok=True)
        peptides = work / f"{pid}_windows.fasta"
        windows = write_peptide_fasta(sequence, peptides, window=self.window)

        if raw_csv is not None:
            out_csv = Path(raw_csv)
        else:
            self.require_available()
            out_csv = self._score_peptides(peptides, work)

        self.last_raw_path = out_csv
        probabilities = parse_amylogram_output(out_csv)
        scores = project_windows(
            windows, probabilities, len(sequence), aggregation=self.aggregation
        )
        binary = binary_from_scores(scores, threshold=self.threshold)
        return PredictorResult(
            protein_id=pid,
            sequence=sequence,
            spec=get_predictor_spec("amylogram"),
            scores=scores,
            binary=binary,
            metadata={
                "window": self.window,
                "aggregation": self.aggregation,
                "threshold": self.threshold,
                "n_windows": len(windows),
                "source": str(out_csv),
            },
        )
