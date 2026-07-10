"""
Resume interrupted multifasta pipeline runs from existing parsed outputs.

Design notes
------------
* Validation reuses ``read_standard_csv`` rather than ad-hoc checks so resume
  criteria stay identical to merge/parse invariants (columns, row count,
  ``aa_name`` vs FASTA).
* Invalid/stale parsed files are re-run, not deleted here — the runner/parser
  overwrites them on success, which keeps resume logic side-effect free.
* Resume is disabled together with ``--skip-run`` (see ``pipeline._resume_enabled``):
  ``--skip-run`` means "re-parse from raw work files", which would be skipped
  if we reused existing ``parsed/`` tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from aggressor_wrappers.batch.logging import LogFn
from aggressor_wrappers.core.schema import get_predictor_spec, read_standard_csv

ResumeStatus = Literal["missing", "valid", "invalid"]


@dataclass(frozen=True)
class ResumePartition:
    """Proteins still to run vs already validated on disk."""

    pending: list[tuple[str, str]]
    skipped_ids: list[str]
    invalidated_ids: list[str]


def predictor_parsed_path(
    parsed_dir: Path,
    *,
    protein_id: str,
    tag: str,
) -> Path:
    """Standard path for a per-protein parsed table under ``{PREDICTOR}/parsed/``."""
    return parsed_dir / f"{protein_id}_{tag}.csv"


def resume_status(
    parsed_path: Path,
    *,
    runner_key: str,
    protein_id: str,
    sequence: str,
) -> ResumeStatus:
    """
    Check whether an existing parsed CSV can be reused for this FASTA record.

    Returns ``valid`` when the file loads and ``aa_name`` matches ``sequence``,
    ``missing`` when absent, ``invalid`` when present but inconsistent (stale run).
    """
    if not parsed_path.is_file():
        return "missing"
    spec = get_predictor_spec(runner_key)
    try:
        read_standard_csv(
            parsed_path,
            spec,
            protein_id=protein_id,
            sequence=sequence,
        )
    except Exception:
        # ValueError from schema checks, pandas parse errors, OSError, etc.
        # Treat as stale so the pipeline re-runs instead of aborting.
        return "invalid"
    return "valid"


def partition_items_for_resume(
    items: list[tuple[str, str]],
    *,
    parsed_dir: Path,
    runner_key: str,
    tag: str,
    resume: bool,
    emit: LogFn | None = None,
) -> ResumePartition:
    """
    Split batch items into pending work and already-valid outputs.

    When ``resume`` is false every item is pending (full re-run). Invalid parsed
    files are treated as pending so a changed FASTA or corrupted CSV is replaced.
    """
    if not resume:
        return ResumePartition(pending=list(items), skipped_ids=[], invalidated_ids=[])

    pending: list[tuple[str, str]] = []
    skipped_ids: list[str] = []
    invalidated_ids: list[str] = []

    for protein_id, sequence in items:
        path = predictor_parsed_path(parsed_dir, protein_id=protein_id, tag=tag)
        status = resume_status(
            path,
            runner_key=runner_key,
            protein_id=protein_id,
            sequence=sequence,
        )
        if status == "valid":
            skipped_ids.append(protein_id)
        elif status == "invalid":
            invalidated_ids.append(protein_id)
            pending.append((protein_id, sequence))
        else:
            pending.append((protein_id, sequence))

    if emit is not None:
        if skipped_ids:
            emit(
                f"[{tag}] resume: reusing parsed output for "
                f"{len(skipped_ids)} protein(s): {', '.join(skipped_ids)}"
            )
        for protein_id in invalidated_ids:
            emit(
                f"[{tag}] resume: ignoring stale parsed output for {protein_id} "
                f"(sequence mismatch or malformed CSV); will re-run"
            )

    return ResumePartition(
        pending=pending,
        skipped_ids=skipped_ids,
        invalidated_ids=invalidated_ids,
    )
