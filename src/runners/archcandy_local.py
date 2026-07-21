"""Run ArchCandy locally via its headless command-line class.

Discovered by inspecting the distributed JAR (see ``LOCAL_BACKENDS.md``):

* ``java -jar ArchCandy.jar`` reaches ``fr.cnrs.crbm.ArchCandyFactory.Final``,
  which constructs a ``JFrame`` regardless of arguments — it opens the GUI, and
  dies with ``HeadlessException`` on a machine without a display. It is not a
  usable entry point for automation.
* ``fr.cnrs.crbm.ArchCandyFactory.AnalysisFactoryCommandLine`` sounds right but is
  a 481-byte stub whose ``main`` is empty; it exits 0 and does nothing.
* ``fr.cnrs.crbm.ArchCandyFactory.MakeAnalysisWithoutApplet`` is the real headless
  entry point, taking twelve positional arguments. It prints its own usage when
  the count is wrong.

Two behaviours of the JAR shape this runner:

**UniProt-style headers are mandatory.** ``FastaParse.getEntryName`` splits the
header on ``|`` and reads index 2, so an ordinary ``>RPL27`` header aborts the run
with ``ArrayIndexOutOfBoundsException``. Input headers are therefore rewritten to
``sp|<id>|<id>_ACWRAP`` before submission and mapped back afterwards.

**Output directories are named after the header**, ``|`` characters included:
``<out>/sp|RPL27|RPL27_ACWRAP/Original_Sequence/Summary_….txt``. That is legal on
macOS and Linux, hostile on Windows, and must be quoted everywhere.

The Summary table is converted to the same region CSV shape the web ArchCandy
output produces, so it is handed to the existing :class:`ArchCandyParser` and the
downstream scoring (``score_mode``, arch topology retention, per-residue
aggregation) is identical for both backends.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.schema import PredictorResult
from aggressor_wrappers.predictors.archcandy import ArchCandyParser
from aggressor_wrappers.predictors.archcandy_local import find_summaries, parse_summary
from aggressor_wrappers.runners.base import BasePredictorRunner

_CLI_CLASS = "fr.cnrs.crbm.ArchCandyFactory.MakeAnalysisWithoutApplet"
_HEADER_SUFFIX = "ACWRAP"
# FastaParse needs a description field after the entry name; its content is
# irrelevant but its presence is not (see uniprot_header).
_DESCRIPTION = "aggressor-wrappers local run"


def uniprot_header(protein_id: str) -> str:
    """Rewrite an arbitrary id into the header form ArchCandy's parser needs.

    ``FastaParse`` requires *both* the ``sp|accession|entry`` triple **and** a
    trailing description: ``>sp|X|Y`` alone raises
    ``StringIndexOutOfBoundsException`` (the class computes a substring between
    the entry-name end and a description offset that does not exist), while
    ``>sp|X|Y d`` parses. The description is therefore always emitted.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", protein_id)
    return f"sp|{safe}|{safe}_{_HEADER_SUFFIX}"


def original_id(header: str, known_ids: list[str]) -> str | None:
    """Map an ArchCandy output folder name back to the original protein id."""
    for pid in known_ids:
        if header == uniprot_header(pid):
            return pid
    return None


class ArchCandyLocalRunner(BasePredictorRunner):
    """Execute the local ArchCandy JAR headlessly over a (multi-)FASTA file."""

    def __init__(
        self,
        *,
        jar_path: str | Path,
        java: str = "java",
        threshold: float = 0.57,
        tm_filter: bool = False,
        arches: str = "arches",
        cavity: bool = False,
        cysteine: bool = False,
        score_mode: str = "highest",
        timeout_seconds: int = 1800,
        **_ignored,
    ) -> None:
        self.jar_path = Path(jar_path).expanduser()
        self.java = java
        self.threshold = float(threshold)
        self.tm_filter = bool(tm_filter)
        if arches not in ("arches", "arches-serpentines"):
            raise ValueError("arches must be 'arches' or 'arches-serpentines'")
        self.arches = arches
        self.cavity = bool(cavity)
        self.cysteine = bool(cysteine)
        # 'highest' rather than the web default 'cumulative': ArchCandy emits many
        # overlapping candidates (16 on a single 99-residue sequence in testing),
        # and summing them leaves its native [0, 1] confidence scale.
        self.score_mode = score_mode
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    # availability
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        return self.jar_path.exists() and shutil.which(self.java) is not None

    def require_available(self) -> None:
        if not self.jar_path.exists():
            raise FileNotFoundError(
                f"ArchCandy JAR not found at {self.jar_path}. Set "
                f"[runners.archcandy] jar_path, or use backend=web."
            )
        if shutil.which(self.java) is None:
            raise FileNotFoundError(
                f"{self.java!r} not on PATH; ArchCandy's local backend needs a JRE."
            )

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def _command(self, fasta: Path, out_dir: Path) -> list[str]:
        return [
            self.java,
            "-Djava.awt.headless=true",
            "-cp",
            str(self.jar_path),
            _CLI_CLASS,
            str(fasta),
            str(out_dir),
            f"{self.threshold:g}",
            "TM" if self.tm_filter else "noTM",
            self.arches,
            "True" if self.cavity else "False",
            "True" if self.cysteine else "False",
            "False",  # histoOutput   - PNG chart, not needed for parsing
            "False",  # scoreOutput   - ASCII arch diagram, not machine-readable
            "False",  # seqViewOutput
            "False",  # serpentinesOutput
            "False",  # serpentinesCardOutput
        ]

    def execute_batch(self, fasta_path: Path, work_dir: str | Path) -> Path:
        """Run ArchCandy over every record in ``fasta_path``; return the output dir."""
        self.require_available()
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_dir = work / "archcandy_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Rewrite headers: ArchCandy's FastaParse requires the sp|x|y form.
        records = read_fasta(fasta_path)
        prepared = work / "archcandy_input.fasta"
        with prepared.open("w", encoding="utf-8") as handle:
            for protein_id, sequence in records.items():
                handle.write(f">{uniprot_header(protein_id)} {_DESCRIPTION}\n{sequence}\n")

        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            self._command(prepared, out_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if "The number of arguments is incorrect" in (proc.stdout or ""):
            raise RuntimeError(
                "ArchCandy rejected the argument list; the JAR's CLI signature may "
                f"differ from the expected twelve positional options.\n{proc.stdout[:400]}"
            )
        if not find_summaries(out_dir):
            raise RuntimeError(
                "ArchCandy produced no Summary files.\n"
                f"stdout: {(proc.stdout or '')[:400]}\n"
                f"stderr: {(proc.stderr or '')[:400]}"
            )
        self.last_raw_path = out_dir
        return out_dir

    def discover_outputs(self, output_dir: Path, protein_ids: list[str]) -> dict[str, Path]:
        """Map original protein id -> its ArchCandy Summary file."""
        mapping: dict[str, Path] = {}
        for header, summary in find_summaries(output_dir).items():
            pid = original_id(header, protein_ids)
            if pid is not None:
                mapping[pid] = summary
        return mapping

    # ------------------------------------------------------------------ #
    # parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def summary_to_region_csv(summary: Path, dest: Path) -> Path:
        """Convert a Summary table into the region CSV the web parser consumes."""
        df = parse_summary(summary)
        out = pd.DataFrame(
            {
                "ID": df["id"] if not df.empty else [],
                "Sequence": df["sequence"] if not df.empty else [],
                "Arch": df["arch"] if not df.empty else [],
                "Start": df["start"] if not df.empty else [],
                "Stop": df["stop"] if not df.empty else [],
                "Score": df["score"] if not df.empty else [],
            }
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        return dest

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
            summary = Path(raw_csv)
        else:
            work = Path(work_dir) if work_dir else fasta_path.parent
            out_dir = self.execute_batch(fasta_path, work)
            found = self.discover_outputs(out_dir, [pid])
            if pid not in found:
                raise FileNotFoundError(f"No ArchCandy Summary produced for {pid!r}")
            summary = found[pid]

        self.last_raw_path = summary
        region_csv = summary.parent / f"regions_{pid}.csv"
        self.summary_to_region_csv(summary, region_csv)
        return ArchCandyParser(score_mode=self.score_mode).parse(
            region_csv, protein_id=pid, sequence=sequence
        )
