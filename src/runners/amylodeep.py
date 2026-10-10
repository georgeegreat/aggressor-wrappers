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

import os
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

#: Variables that make ANOTHER environment's native libraries visible to this
#: interpreter. Inherited into a cross-environment subprocess they are a direct
#: route to SIGSEGV: the child's Python is amylodeep's, while PYTHONPATH still
#: points at the caller's site-packages and DYLD_LIBRARY_PATH at the caller's
#: dylibs, so a compiled extension from one build lands in a process built
#: against another. Two numerical stacks in one process is already the fragile
#: case here; two COPIES of them from different environments is worse.
_LEAKING_VARIABLES = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONEXECUTABLE",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "VIRTUAL_ENV",
)


def child_environment(
    interpreter: str,
    *,
    base: dict[str, str] | None = None,
    kmp_duplicate_lib_ok: bool = False,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """The environment a cross-environment subprocess should run under.

    Pure, so the sanitising is testable without launching anything.

    Three things happen. The leaking variables above are removed. CONDA_PREFIX and
    the leading PATH entry are rewritten to the interpreter's own environment, so
    the child resolves its siblings (and any console scripts it shells out to)
    inside its own env rather than the caller's. Everything else is passed through,
    because HF_HOME, AMYLODEEP_MODEL_DIR, proxy settings and locale all matter and
    none of them cross ABI boundaries.
    """
    environment = dict(os.environ if base is None else base)
    for name in _LEAKING_VARIABLES:
        environment.pop(name, None)

    interpreter_path = Path(interpreter)
    bin_dir = interpreter_path.parent
    if bin_dir.name in ("bin", "Scripts"):
        prefix = bin_dir.parent
        environment["CONDA_PREFIX"] = str(prefix)
        # Prepend rather than replace: the child may need system tools too.
        path_entries = [str(bin_dir)] + [
            entry
            for entry in environment.get("PATH", "").split(os.pathsep)
            if entry and entry != str(bin_dir)
        ]
        environment["PATH"] = os.pathsep.join(path_entries)

    if kmp_duplicate_lib_ok:
        # A DIAGNOSTIC, not a fix. It lets a second OpenMP runtime load instead of
        # aborting, which tells you the duplicate was the cause -- and leaves two
        # runtimes in one process, whose behaviour is undefined.
        environment["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if extra:
        environment.update({str(k): str(v) for k, v in extra.items()})
    return environment


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
        model_dir: str | Path | None = None,
        allow_download: bool = True,
        sanitise_environment: bool = True,
        kmp_duplicate_lib_ok: bool = False,
        device: str | None = None,
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
        # Where AmyloDeep reads its weights. Worth pinning from here rather than
        # leaving to the tool: unset, it falls back to $AMYLODEEP_MODEL_DIR and
        # then ~/.amylodeep_models, so the 3 GB of artefacts a run uses would
        # depend on the environment the subprocess inherited -- and in a conda
        # env launched by a pipeline, that is not the shell the user configured.
        # An absent directory is NOT an error here: the tool downloads into it,
        # and refusing a path that does not exist yet would break first use.
        self.model_dir = (
            None if model_dir in (None, "") else Path(str(model_dir)).expanduser()
        )
        # False turns a missing artefact into a failure that names it, instead of
        # a silent multi-gigabyte download in the middle of a batch.
        self.allow_download = bool(allow_download)
        # When aggressor-wrappers and amylodeep live in different conda
        # environments, the child inherits the caller's PYTHONPATH and DYLD_* by
        # default -- which is how one env's compiled extensions end up inside the
        # other env's interpreter. False restores the old inherit-everything
        # behaviour, for comparing the two.
        self.sanitise_environment = bool(sanitise_environment)
        self.kmp_duplicate_lib_ok = bool(kmp_duplicate_lib_ok)
        # Where AmyloDeep's torch heads run. None omits the flag, so an amylodeep
        # that predates --device still works. "cpu" is the tool's own default and
        # the configuration its published numbers came from; an accelerator is
        # opted into, and the tool RAISES rather than downgrading if the backend is
        # unavailable -- which is what makes the choice recoverable from a log
        # instead of a guess. The device also lands in every output row, so a
        # merged table says which backend produced it.
        device = device or None
        if device not in (None, "cpu", "auto", "mps", "cuda"):
            raise ValueError(
                "device must be cpu, auto, mps, cuda or blank "
                f"(got {device!r})"
            )
        self.device = device
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def _interpreter(self) -> str:
        """The interpreter the compat shim will actually run under."""
        return self.python or shutil.which("python3") or "python3"

    def is_available(self) -> bool:
        """Check the thing that will actually be executed.

        With use_compat_shim (the default) the console script is never invoked:
        _argv builds [python, bootstrap, ...]. Checking `executable` therefore
        answered a question nobody asked -- an amylodeep installed in its own
        conda env put the console script somewhere PATH may or may not see, while
        the shim ran under whichever python3 came first, typically the system one,
        which cannot import amylodeep. The run then died with a bare
        `ModuleNotFoundError: No module named 'amylodeep'` from inside a generated
        file, after the panel was already running.
        """
        if self.use_compat_shim:
            return self._interpreter_can_import()
        return shutil.which(self.executable) is not None

    def _interpreter_can_import(self) -> bool:
        interpreter = self._interpreter()
        if shutil.which(interpreter) is None and not Path(interpreter).is_file():
            return False
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [interpreter, "-c", "import amylodeep"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def require_available(self) -> None:
        if self.is_available():
            return
        if self.use_compat_shim:
            interpreter = self._interpreter()
            raise FileNotFoundError(
                f"The interpreter the compat shim would use, {interpreter!r}, "
                f"cannot import amylodeep.\n"
                f"With use_compat_shim = true the console script is never called, "
                f"so setting `executable` alone has no effect -- it is `python` in "
                f"[runners.amylodeep] that must point into the environment holding "
                f"amylodeep, e.g.\n"
                f"    python = /path/to/miniconda3/envs/amylodeep/bin/python\n"
                f"Find it with: conda run -n amylodeep which python\n"
                f"Alternatively set use_compat_shim = false and point `executable` "
                f"at that env's bin/amylodeep -- but the shim exists to apply the "
                f"jax_unirep clip fix inside that interpreter, so prefer `python`."
            )
        raise FileNotFoundError(
            f"{self.executable!r} not on PATH. Install with "
            f"`pip install amylodeep`, or use backend=web."
        )

    # ------------------------------------------------------------------ #
    def _argv(self, sequence: str, dest: Path) -> list[str]:
        args = ["--output", str(dest), "--format", self.output_format]
        if self.device is not None:
            args += ["--device", self.device]
        if self.model_dir is not None:
            args += ["--model-dir", str(self.model_dir)]
        if not self.allow_download:
            args.append("--no-download")
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

    def _write_run_log(self, dest: Path, proc, combined: str) -> Path:
        """Persist the child's whole output beside where its result should have been.

        A crash inside a compiled extension prints its useful line and then dies,
        so that line is at the END of the stream -- which is exactly what a leading
        truncation discards. The excerpts in the messages below are therefore
        tails, and the whole thing is on disk regardless.
        """
        log_path = dest.with_suffix(dest.suffix + ".log")
        header = [
            f"argv: {' '.join(self._argv('<sequence>', dest))}",
            f"interpreter: {self._interpreter()}",
            f"returncode: {proc.returncode}",
            f"environment sanitised: {self.sanitise_environment}",
            f"KMP_DUPLICATE_LIB_OK: {self.kmp_duplicate_lib_ok}",
            "",
        ]
        try:
            log_path.write_text("\n".join(header) + combined, encoding="utf-8")
        except OSError:  # pragma: no cover - never mask the original failure
            pass
        return log_path

    def _run_one(self, sequence: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        environment = None
        if self.sanitise_environment:
            environment = child_environment(
                self._interpreter(),
                kmp_duplicate_lib_ok=self.kmp_duplicate_lib_ok,
            )
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._argv(sequence, dest),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env=environment,
        )
        if not dest.exists():
            combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            # The full output goes to a file BEFORE any message is built. Every
            # branch below used to inline combined[:400], which on a real crash cut
            # the text off mid-path ("...\n/Use") and threw away the line that
            # named the fault. A diagnostic that truncates where it gets
            # interesting is worse than none, because it looks complete.
            log_path = self._write_run_log(dest, proc, combined)
            log_note = f"\nFull captured output: {log_path}"
            if proc.returncode is not None and proc.returncode < 0:
                raise RuntimeError(
                    _signal_report(-proc.returncode, sequence, combined) + log_note
                )
            if "huggingface.co" in combined or "Hub" in combined:
                raise RuntimeError(
                    "AmyloDeep could not fetch its model weights from huggingface.co. "
                    "The first run needs network access; in a restricted environment, "
                    "pre-populate the HF cache (HF_HOME) on a connected machine.\n"
                    f"{combined[-1500:]}{log_note}"
                )
            if "pkg_resources" in combined:
                raise RuntimeError(
                    "AmyloDeep's dependency jax_unirep imports pkg_resources, which "
                    "setuptools >= 81 removed. Install `setuptools<81` in the same "
                    f"environment.\n{combined[-1500:]}{log_note}"
                )
            if "jaxlib is version" in combined or (
                "jax" in combined and "requires version" in combined
            ):
                raise RuntimeError(
                    "jax and jaxlib are mismatched in the AmyloDeep environment, "
                    "so the run stopped before any sequence was scored.\n"
                    "They are one library in two packages -- jax is pure Python, "
                    "jaxlib is the compiled half -- and jax checks the pair at "
                    "import. Install them together rather than separately, in the "
                    "environment `python` points at:\n"
                    f"    {self._interpreter()} -m pip install --upgrade 'jax[cpu]'\n"
                    "Then confirm the consumer still imports, since jax_unirep "
                    "predates recent jax releases and a newer pair can break it:\n"
                    f"    {self._interpreter()} -c 'import jax_unirep; print(\"ok\")'\n"
                    "If that fails, pin BOTH to a matched older pair instead of "
                    "upgrading -- only the UniRep head needs jax, and AmyloDeep can "
                    "average four heads without it (heads_used records that)."
                    f"\n{combined[-1500:]}{log_note}"
                )
            if "No module named 'amylodeep'" in combined:
                raise RuntimeError(
                    f"The compat shim ran under {self._interpreter()!r}, which "
                    f"cannot import amylodeep. Set `python` in "
                    f"[runners.amylodeep] to the interpreter of the environment "
                    f"holding it (conda run -n amylodeep which python); setting "
                    f"`executable` alone has no effect while use_compat_shim is "
                    f"true.\n{combined[-1500:]}{log_note}"
                )
            if "unrecognized arguments" in combined or "invalid choice" in combined:
                raise RuntimeError(
                    "The installed amylodeep rejected a flag this runner passed, so "
                    "it predates the per-residue output (0.4). Upgrade it, or set "
                    "resolution/aggregate/window_size to None in the amylodeep "
                    "section of config.cfg to fall back to that version's defaults "
                    "-- the parser still reads a 0.3 window table, projecting it "
                    f"with `max`.\n{combined[-1500:]}{log_note}"
                )
            raise RuntimeError(
                f"AmyloDeep produced no output.\n{combined[-1500:]}{log_note}"
            )
        return dest

    def _argv_batch(self, fasta: Path, dest: Path) -> list[str]:
        """argv for ONE invocation covering a whole multifasta."""
        args = ["--input", str(fasta), "--output", str(dest),
                "--format", self.output_format]
        if self.device is not None:
            args += ["--device", self.device]
        if self.model_dir is not None:
            args += ["--model-dir", str(self.model_dir)]
        if not self.allow_download:
            args.append("--no-download")
        if self.resolution is not None:
            args += ["--resolution", self.resolution]
        if self.window_size is not None:
            args += ["--window-size", str(self.window_size)]
        if self.aggregate is not None:
            args += ["--aggregate", self.aggregate]
        if self.use_compat_shim:
            from aggressor_wrappers.core.compat import write_amylodeep_bootstrap

            boot = write_amylodeep_bootstrap(dest.parent / "_run_amylodeep.py")
            python = self.python or shutil.which("python3") or "python3"
            return [python, str(boot), *args]
        return [self.executable, *args]

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Score every record in ONE invocation, then split the result per protein.

        The previous version looped over the records and launched one subprocess
        per protein. AmyloDeep loads about 3.2 GB of artefacts at startup -- the
        2.6 GB frozen encoder, ESM2-150M, three heads and three calibrators -- so a
        372-protein sweep paid that 372 times and read ~1.2 TB from disk to do work
        that needs one load. The CLI already accepts a multifasta and loops
        internally with the models resident, which is what this now uses.

        Granularity stays the caller's: the pipeline's [runners.amylodeep]
        sequences_per_run decides how many records reach one execute_batch call, so
        a crash costs one chunk rather than the whole sweep.

        Outputs are split back into <protein_id>.csv so discover_outputs, the
        parser and the resume logic all keep the contract they had.
        """
        self.require_available()
        work = Path(work_dir)
        out_dir = work / "amylodeep_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        records = read_fasta(Path(fasta_path))
        pending = {
            pid: seq
            for pid, seq in records.items()
            if not (out_dir / f"{pid}.{self.output_format}").is_file()
        }
        if not pending:
            self.last_raw_path = out_dir
            return out_dir

        # Only the records still missing go to the tool. A sweep interrupted after
        # three days keeps every protein it finished.
        subset = out_dir / "_pending.faa"
        subset.write_text(
            "".join(f">{pid}\n{seq}\n" for pid, seq in pending.items()),
            encoding="utf-8",
        )
        combined = out_dir / f"_combined.{self.output_format}"
        self._run_batch(subset, combined)
        self._split_combined(combined, out_dir, list(pending))

        self.last_raw_path = out_dir
        return out_dir

    def _run_batch(self, fasta: Path, dest: Path) -> Path:
        environment = None
        if self.sanitise_environment:
            environment = child_environment(
                self._interpreter(),
                kmp_duplicate_lib_ok=self.kmp_duplicate_lib_ok,
            )
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._argv_batch(fasta, dest),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
            env=environment,
        )
        if not dest.exists():
            combined_output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
            log_path = self._write_run_log(dest, proc, combined_output)
            if proc.returncode is not None and proc.returncode < 0:
                raise RuntimeError(
                    _signal_report(-proc.returncode, "<batch>", combined_output)
                    + f"\nFull captured output: {log_path}"
                )
            raise RuntimeError(
                f"AmyloDeep produced no output for the batch.\n"
                f"{combined_output[-1500:]}\nFull captured output: {log_path}"
            )
        return dest

    def _split_combined(
        self, combined: Path, out_dir: Path, protein_ids: list[str]
    ) -> dict[str, Path]:
        """One file per protein, so nothing downstream has to know about batching."""
        import pandas as pd

        if self.output_format == "json":
            import json

            rows = json.loads(combined.read_text(encoding="utf-8"))
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_csv(combined)

        if "sequence_id" not in frame.columns:
            raise ValueError(
                f"{combined}: no sequence_id column, so the batch output cannot be "
                f"split per protein. A single-sequence invocation would be needed."
            )

        written: dict[str, Path] = {}
        grouped = {str(k): g for k, g in frame.groupby("sequence_id")}
        for pid in protein_ids:
            group = grouped.get(str(pid))
            if group is None or group.empty:
                continue
            dest = out_dir / f"{pid}.{self.output_format}"
            if self.output_format == "json":
                dest.write_text(group.to_json(orient="records", indent=2), "utf-8")
            else:
                group.to_csv(dest, index=False)
            written[pid] = dest
        return written

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
