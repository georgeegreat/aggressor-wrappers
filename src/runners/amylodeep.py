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


def _signal_report(signal_number: int, sequence: str, output: str) -> str:
    """Name the signal, because a native crash leaves nothing else behind.

    ``subprocess.run`` returns a NEGATIVE returncode when the child died on a
    signal, and that number was being discarded: the caller got "AmyloDeep
    produced no output" with an empty stdout/stderr, because a crash inside a
    compiled extension never reaches Python's excepthook. The signal is the only
    evidence the run leaves in-process, so it is reported first.

    The distinction that matters is SIGSEGV vs SIGKILL. Out of memory on macOS
    and Linux kills the process (SIGKILL, or the OOM killer), and Python raises
    MemoryError when the allocation is its own; a SIGSEGV is a bad memory access
    inside native code and is not a sign of insufficient RAM. Saying so stops the
    obvious-but-wrong conclusion that a bigger machine would help.
    """
    import signal as _signal

    try:
        name = _signal.Signals(signal_number).name
    except ValueError:  # pragma: no cover - unknown platform signal
        name = f"signal {signal_number}"

    lines = [
        f"AmyloDeep died on {name} ({signal_number}) after "
        f"{len(sequence)} residues of input, without writing output."
    ]
    if signal_number in (_signal.SIGSEGV, getattr(_signal, "SIGBUS", None), _signal.SIGILL):
        lines += [
            "",
            "This is a crash inside a compiled extension, NOT an out-of-memory "
            "condition: an OOM shows up as SIGKILL (or the OOM killer), and an "
            "allocation Python itself made raises MemoryError. Adding RAM will "
            "not change this.",
            "",
            "AmyloDeep loads two independent numerical stacks in ONE process -- "
            "ESM2 through PyTorch and UniRep through jax_unirep/JAX -- which is "
            "the usual source of this failure: each ships its own OpenMP runtime, "
            "and loading both crashes on macOS. In that case:",
            "  * try KMP_DUPLICATE_LIB_OK=TRUE as a diagnostic (it confirms the "
            "cause; it is not a fix),",
            "  * or install torch and jaxlib from ONE channel into a clean env.",
            "A wheel built for a different CPU (an x86_64 build under Rosetta on "
            "Apple Silicon, or AVX-512 on a machine without it) produces the same "
            "signal; `python -c \"import platform; print(platform.machine())\"` "
            "inside the AmyloDeep environment settles that.",
            "",
            "Sequence length is unlikely to be the cause: AmyloDeep scores a "
            "RUNNING WINDOW (default 10 residues), so a long protein is many "
            "small inferences rather than one large one, and its own preprint "
            "reports no maximum length. Confirm with a short peptide -- if a "
            "10-mer crashes too, the input is not the variable.",
        ]
    elif signal_number == _signal.SIGKILL:
        lines += [
            "",
            "SIGKILL with no output is what an out-of-memory kill looks like. "
            "Check the per-window batch: AmyloDeep embeds every window with ESM2, "
            "so peak memory scales with window count, not with sequence length "
            "alone.",
        ]
    if output.strip():
        lines += ["", f"Captured output: {output.strip()[:400]}"]
    else:
        lines += [
            "",
            "The child wrote nothing to stdout or stderr, which is expected for a "
            "native crash and is why this runner reports the signal rather than "
            "quoting the output.",
        ]
    return "\n".join(lines)

class AmyloDeepRunner(BasePredictorRunner):
    """Execute the local ``amylodeep`` CLI, one invocation per sequence."""

    def __init__(
        self,
        *,
        executable: str = "amylodeep",
        python: str | None = None,
        use_compat_shim: bool = True,
        threshold: float = 0.5,
        output_format: str = "csv",
        timeout_seconds: int = 1800,
        window_size: int | None = None,
        aggregate: str | None = None,
        resolution: str | None = "residue",
        **_ignored,
    ) -> None:
        self.executable = executable
        # AmyloDeep runs as a subprocess (often in its own conda env), so the
        # jax_unirep clip fix must be applied inside THAT interpreter. When
        # enabled, the tool is launched through a generated self-contained
        # bootstrap instead of the console script. Set false to call `amylodeep`
        # directly (e.g. once the fix lands upstream).
        self.python = python
        self.use_compat_shim = bool(use_compat_shim)
        self.threshold = float(threshold)
        if output_format not in ("csv", "json"):
            raise ValueError("output_format must be 'csv' or 'json'")
        self.output_format = output_format
        self.timeout_seconds = int(timeout_seconds)
        # Ask for the grain rather than inherit it. AmyloDeep 0.3's CLI wrote one
        # row per window at window 10; 0.4 writes one row per residue at window 6.
        # Left implicit, the same config would produce two different profiles
        # depending only on which version happens to be installed, and the shift
        # is the kind that survives inspection. resolution=None omits the flag,
        # for an installation that predates it.
        # A config file expresses "omit this flag" as an empty value, so an empty
        # string means the same as None here rather than failing validation.
        resolution = resolution or None
        aggregate = aggregate or None
        window_size = window_size if window_size not in (None, "") else None
        if resolution not in (None, "residue", "window", "both"):
            raise ValueError("resolution must be 'residue', 'window', 'both' or None")
        if aggregate not in (None, "mean", "max", "support"):
            raise ValueError("aggregate must be 'mean', 'max', 'support' or None")
        if window_size is not None and int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        self.resolution = resolution
        self.aggregate = aggregate
        self.window_size = None if window_size is None else int(window_size)
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
    def _argv(self, sequence: str, dest: Path) -> list[str]:
        args = ["--output", str(dest), "--format", self.output_format]
        if self.resolution is not None:
            args += ["--resolution", self.resolution]
        if self.window_size is not None:
            args += ["--window-size", str(self.window_size)]
        if self.aggregate is not None:
            args += ["--aggregate", self.aggregate]
        args.append(sequence)
        if self.use_compat_shim:
            from aggressor_wrappers.core.compat import write_amylodeep_bootstrap

            boot = write_amylodeep_bootstrap(dest.parent / "_run_amylodeep.py")
            python = self.python or shutil.which("python3") or "python3"
            return [python, str(boot), *args]
        return [self.executable, *args]

    def _run_one(self, sequence: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._argv(sequence, dest),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if not dest.exists():
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            if proc.returncode is not None and proc.returncode < 0:
                raise RuntimeError(_signal_report(-proc.returncode, sequence, combined))
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
            if "unrecognized arguments" in combined or "invalid choice" in combined:
                raise RuntimeError(
                    "The installed amylodeep rejected a flag this runner passed, so "
                    "it predates the per-residue output (0.4). Upgrade it, or set "
                    "resolution/aggregate/window_size to None in the amylodeep "
                    "section of config.cfg to fall back to that version's defaults "
                    "-- the parser still reads a 0.3 window table, projecting it "
                    f"with `max`.\n{combined[:400]}"
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
