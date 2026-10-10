"""Run TANGO locally.

TANGO is a licensed academic tool distributed as a platform-specific binary; it
has no web service. It is therefore **local-only**, and any environment without
the binary (notably CI, and any platform other than the one the binary was built
for) must skip it rather than fail.

Invocation, established by running the Linux build::

    tango <output_basename> nt=N ct=N ph=7.0 te=298.15 io=0.02 seq=<SEQUENCE>

Behaviours the runner depends on, all confirmed empirically:

* ``<output_basename>`` may be a path; TANGO appends ``.txt``.
* A one-line summary goes to stdout
  (``AGG … AMYLO … TURN … HELIX … HELAGG … BETA …``); the per-residue table goes
  to the file.
* One sequence per invocation, so a multifasta is handled by looping.

The condition parameters are exposed rather than hardcoded because they change
the prediction: ``ph`` and ``te`` (temperature, K) shift the aggregation
propensity, and ``io`` (ionic strength) modulates the charge term. ``nt``/``ct``
declare whether the termini are free (charged) — for an internal fragment of a
larger protein they should be ``N``, which is why that is the default here rather
than the capped form.
"""

from __future__ import annotations

import errno
import platform

import shutil
import subprocess
from pathlib import Path

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.predictors.tango import tango_profile
from aggressor_wrappers.runners.base import BasePredictorRunner


#: Leading bytes that identify an executable's container format, and the machine
#: field that follows. Read rather than guessed: "Exec format error" (Errno 8) is
#: raised for a Linux ELF on macOS AND for a Mach-O x86-64 with no Rosetta, and
#: the two have completely different remedies -- one needs a different build, the
#: other needs one `softwareupdate` command. The distributed TANGO filenames
#: (tango_x86_64_release) name the ARCHITECTURE and not the operating system, so
#: the name cannot settle it either.
_MACHO_CPU_TYPES = {
    0x01000007: "x86_64",
    0x0100000C: "arm64",
    0x00000007: "i386",
    0x0000000C: "arm",
}
_ELF_MACHINES = {0x03: "i386", 0x3E: "x86-64", 0xB7: "aarch64", 0x28: "arm"}

#: Container formats each OS cannot execute at all, whatever else is installed.
#: Rosetta and similar translate an ARCHITECTURE, never a different OS's format.
_FOREIGN_FORMATS = {
    "Darwin": ("Linux/ELF", "Windows PE"),
    "Linux": ("macOS/Mach-O", "macOS universal", "Windows PE"),
    "Windows": ("Linux/ELF", "macOS/Mach-O", "macOS universal"),
}


def runnable_here(kind: str) -> bool:
    """Whether a format named by describe_binary() can execute on this OS."""
    foreign = _FOREIGN_FORMATS.get(platform.system(), ())
    return not any(marker in kind for marker in foreign)


def describe_binary(path: Path) -> str:
    """Name a binary's format and architecture from its magic bytes."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(20)
    except OSError as exc:  # pragma: no cover - reported, never raised onward
        return f"unreadable ({exc.strerror})"
    if len(head) < 4:
        return "too short to identify"
    if head[:2] == b"#!":
        return "a script (#! ...)"
    if head[:4] == b"\x7fELF":
        bits = {1: "32-bit", 2: "64-bit"}.get(head[4], "unknown width")
        machine = _ELF_MACHINES.get(
            int.from_bytes(head[18:20], "little"), "unknown architecture"
        )
        return f"a Linux/ELF {bits} {machine} binary"
    if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):
        cpu = _MACHO_CPU_TYPES.get(
            int.from_bytes(head[4:8], "little"), "unknown architecture"
        )
        return f"a macOS/Mach-O {cpu} binary"
    if head[:4] in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "a macOS universal (fat) binary"
    if head[:2] == b"MZ":
        return "a Windows PE executable"
    return f"an unrecognised format (magic {head[:4].hex()})"


def exec_format_report(path: Path) -> str:
    """Explain an Errno 8 for this file, with the remedy that matches BOTH the
    file's format and the host running it.

    Keying the advice on the file alone was the first version's mistake: it told a
    Linux host that macOS cannot run ELF binaries, which is true and useless. The
    remedy depends on the pair.
    """
    kind = describe_binary(path)
    host = platform.system()
    machine = platform.machine()
    lines = [
        f"{path} could not be executed: the file is {kind}, "
        f"on a {host} {machine} host."
    ]

    if not runnable_here(kind):
        lines += [
            "",
            f"{host} cannot execute that format at all. Architecture translators "
            f"(Rosetta 2 on macOS, qemu-user on Linux) translate an instruction "
            f"set, never another operating system's executable format, so "
            f"installing one will not help.",
            "A distributed filename such as tango_x86_64_release names the "
            "ARCHITECTURE and not the OS, so it cannot tell you which build you "
            "have -- the magic bytes above can.",
            f"Options: obtain the {host} build, run this one inside a container or "
            f"VM for its own OS, or drop tango from the predictor list for this run.",
        ]
    elif host == "Darwin" and "Mach-O x86_64" in kind and machine == "arm64":
        lines += [
            "",
            "A Mac binary for Intel on Apple Silicon. Rosetta 2 translates it, so "
            "this usually means Rosetta is absent:",
            "    softwareupdate --install-rosetta --agree-to-license",
        ]
    elif "script" in kind:
        lines += [
            "",
            "Its interpreter line may name a path that does not exist here, or the "
            "file may lack the executable bit.",
        ]
    else:
        lines += [
            "",
            "The format looks runnable on this host, so check the file is the "
            "executable itself and not an archive, a Git LFS pointer or a partial "
            "download, and that it carries the executable bit.",
        ]
    return "\n".join(lines)


class TANGORunner(BasePredictorRunner):
    """Execute the local TANGO binary, one invocation per sequence."""

    def __init__(
        self,
        *,
        binary_path: str | Path = "tango",
        nt: str = "N",
        ct: str = "N",
        ph: float = 7.0,
        te: float = 298.15,
        io: float = 0.02,
        threshold: float = 5.0,
        score_column: str = "Aggregation",
        timeout_seconds: int = 600,
        **_ignored,
    ) -> None:
        self.binary_path = str(binary_path)
        for name, value in (("nt", nt), ("ct", ct)):
            if str(value).upper() not in ("N", "Y"):
                raise ValueError(f"{name} must be 'N' or 'Y'")
        self.nt = str(nt).upper()
        self.ct = str(ct).upper()
        self.ph = float(ph)
        self.te = float(te)
        self.io = float(io)
        # 5 % is the conventional TANGO aggregation cutoff.
        self.threshold = float(threshold)
        self.score_column = score_column
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def _resolve(self) -> str | None:
        p = Path(self.binary_path).expanduser()
        if p.exists():
            return str(p)
        return shutil.which(self.binary_path)

    def is_available(self) -> bool:
        return self._resolve() is not None

    def require_available(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(
                f"TANGO binary not found ({self.binary_path!r}). TANGO is licensed "
                f"and platform-specific with no web fallback: set "
                f"[runners.tango] binary_path, or leave it out of the predictor list."
            )
        # Present on disk is not the same as runnable here. Checking the format up
        # front costs one file read and converts a mid-sweep Errno 8 into a
        # refusal before the panel starts.
        resolved = Path(self._resolve())
        kind = describe_binary(resolved)
        if not runnable_here(kind) or "unrecognised" in kind:
            raise FileNotFoundError(exec_format_report(resolved))

    # ------------------------------------------------------------------ #
    def _run_one(self, sequence: str, out_base: Path) -> Path:
        binary = self._resolve()
        assert binary is not None  # require_available() called by callers
        out_base.parent.mkdir(parents=True, exist_ok=True)
        # TANGO silently writes nothing when the output-name argument is long
        # (it fails at ~40 characters), so the process is run *inside* the output
        # directory with a bare basename. Path length then cannot matter, which
        # also keeps it working under pytest's deep tmp_path directories.
        argv = [
                str(Path(binary).resolve()),
                out_base.name,
                f"nt={self.nt}",
                f"ct={self.ct}",
                f"ph={self.ph:g}",
                f"te={self.te:g}",
                f"io={self.io:g}",
                f"seq={sequence}",
        ]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                cwd=out_base.parent,
            )
        except OSError as exc:
            # Errno 8 (ENOEXEC) propagated as "[Errno 8] Exec format error:
            # '<path>'", which names the file and not the reason. The reason is in
            # the file's first bytes.
            if exc.errno == errno.ENOEXEC:
                raise RuntimeError(exec_format_report(Path(binary))) from exc
            raise
        produced = out_base.with_suffix(".txt")
        if not produced.exists():
            raise RuntimeError(
                "TANGO produced no output file.\n"
                f"stdout: {(proc.stdout or '')[:300]}\n"
                f"stderr: {(proc.stderr or '')[:300]}"
            )
        return produced

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        self.require_available()
        out_dir = Path(work_dir) / "tango_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        for protein_id, sequence in read_fasta(fasta_path).items():
            self._run_one(sequence, out_dir / protein_id)
        self.last_raw_path = out_dir
        return out_dir

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        found = {}
        for pid in protein_ids:
            candidate = Path(output_dir) / f"{pid}.txt"
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
            self.require_available()
            work = Path(work_dir) if work_dir else fasta_path.parent
            raw = self._run_one(sequence, Path(work) / "tango_out" / pid)

        self.last_raw_path = raw
        scores, aux, meta = tango_profile(
            raw, sequence, score_column=self.score_column
        )
        binary = binary_from_scores(scores, threshold=self.threshold)
        meta.update(
            {
                "threshold": self.threshold,
                "nt": self.nt,
                "ct": self.ct,
                "ph": self.ph,
                "te": self.te,
                "io": self.io,
            }
        )
        return PredictorResult(
            protein_id=pid,
            sequence=sequence,
            spec=get_predictor_spec("tango"),
            scores=scores,
            binary=binary,
            metadata=meta,
            aux=aux,
        )
