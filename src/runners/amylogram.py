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
import warnings
import tempfile
from collections.abc import Sequence
from pathlib import Path

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.predictors.amylogram import (
    DEFAULT_AGGREGATION,
    DEFAULT_SUPPORT_FRACTION,
    DEFAULT_WINDOWS,
    coverage_depth,
    parse_amylogram_output,
    project_windows,
    to_per_residue_frame,
    write_peptide_fasta,
)
from aggressor_wrappers.runners.base import BasePredictorRunner
from aggressor_wrappers.paths import DEFAULT_AMYLOGRAM_SCRIPT


def _coerce_windows(windows, window) -> tuple[int, ...]:
    """Accept a list, a comma string (config.cfg gives strings), or a legacy int."""
    if windows is None and window is None:
        return tuple(DEFAULT_WINDOWS)
    if windows is None:
        return (int(window),)
    if isinstance(windows, str):
        parts = [p.strip() for p in windows.replace(";", ",").split(",") if p.strip()]
        values = [int(p) for p in parts]
    elif isinstance(windows, int):
        values = [windows]
    else:
        values = [int(w) for w in windows]
    if not values:
        raise ValueError("windows must name at least one width")
    if any(w < 1 for w in values):
        raise ValueError(f"window widths must be >= 1; got {values}")
    return tuple(sorted(dict.fromkeys(values)))


class AmyloGramRunner(BasePredictorRunner):
    """Score sliding hexapeptides with AmyloGram and project onto residues."""

    def __init__(
        self,
        *,
        rscript: str = "Rscript",
        script_path: str | Path | None = None,
        window: int | None = None,
        windows: Sequence[int] | str | None = None,
        aggregation: str = DEFAULT_AGGREGATION,
        support_fraction: float = DEFAULT_SUPPORT_FRACTION,
        quantile: float = 0.5,
        combine: str = "max",
        threshold: float = 0.5,
        timeout_seconds: int = 1800,
        write_per_residue: bool = True,
        **_ignored,
    ) -> None:
        self.rscript = rscript
        self.script_path = Path(script_path).expanduser() if script_path else None
        self.write_per_residue = bool(write_per_residue)
        self.last_per_residue_path: Path | None = None

        # `window` (singular) is kept so existing configs and callers keep
        # working; `windows` supersedes it.
        self.windows = _coerce_windows(windows, window)
        off_regime = [w for w in self.windows if w != 6]
        if off_regime:
            warnings.warn(
                f"AmyloGram was fitted on hexapeptides; querying it with width(s) "
                f"{off_regime} is outside that training distribution and the "
                f"returned probability is not calibrated there. Treat those "
                f"columns as a boundary-sensitivity device, not as predictions. "
                f"To widen APRs in-regime, lower support_fraction instead.",
                stacklevel=2,
            )
        self.aggregation = aggregation
        self.support_fraction = float(support_fraction)
        self.quantile = float(quantile)
        if combine not in ("max", "mean", "min"):
            raise ValueError("combine must be 'max', 'mean' or 'min'")
        self.combine = combine
        self.threshold = float(threshold)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    @property
    def window(self) -> int:
        """The primary (narrowest) query width — AmyloGram's training unit."""
        return min(self.windows)

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
        """Locate the R helper. Never invent a path that does not exist.

        The previous version fell back to ``work / "amylogram_predict.R"`` — a
        file nothing ever wrote — so with no explicit ``script_path`` the runner
        handed Rscript a non-existent script and surfaced the failure as the
        generic "AmyloGram produced no output". The packaged helper at
        ``legacy/amylogram_predict.R`` is the real default; a missing script is
        now named as such at the point of failure.
        """
        for candidate in (self.script_path, DEFAULT_AMYLOGRAM_SCRIPT):
            if candidate and Path(candidate).exists():
                return Path(candidate)
        raise FileNotFoundError(
            "AmyloGram helper script not found. Expected the packaged "
            f"{DEFAULT_AMYLOGRAM_SCRIPT}"
            + (f" or the configured {self.script_path}" if self.script_path else "")
            + ". Set [runners.amylogram] script_path in config.cfg."
        )

    def r_packages_present(self) -> bool:
        """Check that the R side can actually load AmyloGram and seqinr.

        ``shutil.which('Rscript')`` says only that R is installed. AmyloGram and
        seqinr are separate CRAN packages, and their absence is the common
        failure on a fresh machine; catching it here turns a mid-run crash into
        a pre-flight answer.
        """
        if not self.is_available():
            return False
        probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                self.rscript,
                "-e",
                'q(status = if (all(c("AmyloGram","seqinr") %in% '
                "rownames(installed.packages()))) 0 else 1)",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return probe.returncode == 0

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

        per_width: dict[int, list[float]] = {}
        sources: dict[int, str] = {}
        n_windows: dict[int, int] = {}

        for width in self.windows:
            peptides = work / f"{pid}_w{width}_windows.fasta"
            windows = write_peptide_fasta(sequence, peptides, window=width)
            n_windows[width] = len(windows)
            if raw_csv is not None and len(self.windows) == 1:
                out_csv = Path(raw_csv)
            elif raw_csv is not None:
                out_csv = Path(raw_csv).with_name(
                    f"{Path(raw_csv).stem}_w{width}{Path(raw_csv).suffix}"
                )
            else:
                self.require_available()
                out_csv = self._score_peptides(peptides, work)
            sources[width] = str(out_csv)
            probabilities = parse_amylogram_output(out_csv)
            per_width[width] = project_windows(
                windows,
                probabilities,
                len(sequence),
                aggregation=self.aggregation,
                support_fraction=self.support_fraction,
                quantile=self.quantile,
            )

        self.last_raw_path = Path(sources[self.window])

        if len(per_width) == 1:
            scores = per_width[self.window]
        else:
            stacked = list(zip(*(per_width[w] for w in self.windows), strict=True))
            reducer = {"max": max, "min": min,
                       "mean": lambda v: sum(v) / len(v)}[self.combine]
            scores = [float(reducer(column)) for column in stacked]

        binary = binary_from_scores(scores, threshold=self.threshold)

        # Coverage depth for the primary width. Terminal residues are supported
        # by fewer windows than interior ones, so every rule except `max` is
        # evaluated on a smaller sample there; carrying the depth makes that
        # visible downstream instead of leaving it as an unexplained edge.
        depth = coverage_depth(len(sequence), self.window)

        per_residue_path: Path | None = None
        if self.write_per_residue:
            frame = to_per_residue_frame(sequence, scores, protein_id=pid)
            for width in self.windows:
                frame[f"w{width}_Score"] = per_width[width]
            frame["coverage_depth"] = depth
            per_residue_path = work / f"{pid}_amylogram_per_residue.csv"
            frame.to_csv(per_residue_path, index=False)
            self.last_per_residue_path = per_residue_path

        return PredictorResult(
            protein_id=pid,
            sequence=sequence,
            spec=get_predictor_spec("amylogram"),
            scores=scores,
            binary=binary,
            metadata={
                "windows": list(self.windows),
                "window": self.window,
                "aggregation": self.aggregation,
                "support_fraction": self.support_fraction,
                "combine": self.combine if len(self.windows) > 1 else None,
                "threshold": self.threshold,
                "n_windows": n_windows,
                "source": sources[self.window],
                "sources": sources,
                "per_residue_csv": str(per_residue_path) if per_residue_path else None,
            },
            aux={"coverage_depth": depth,
                 **{f"w{w}_score": per_width[w] for w in self.windows}},
        )
