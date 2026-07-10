"""FASTA reading utilities."""

from __future__ import annotations

import warnings
from pathlib import Path

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> dict[str, str]:
    """Parse a FASTA file into ``{header_id: sequence}``."""
    path = Path(path)
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    _store_record(sequences, current_id, "".join(chunks), path=path)
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        _store_record(sequences, current_id, "".join(chunks), path=path)

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    return sequences


def read_first_sequence(path: str | Path) -> tuple[str, str]:
    """Return ``(protein_id, sequence)`` for the first record in a FASTA file."""
    items = read_fasta(path)
    protein_id, sequence = next(iter(items.items()))
    return protein_id, sequence


def normalise_sequence(sequence: str) -> str:
    """Uppercase, strip spaces, and reject non-standard amino acids."""
    return _normalise_sequence(sequence)


def _store_record(
    sequences: dict[str, str],
    protein_id: str,
    raw_sequence: str,
    *,
    path: Path,
) -> None:
    if protein_id in sequences:
        warnings.warn(
            f"Duplicate FASTA header {protein_id!r} in {path}; "
            "later record overwrites the earlier one",
            UserWarning,
            stacklevel=2,
        )
    sequences[protein_id] = _normalise_sequence(raw_sequence)


def _normalise_sequence(sequence: str) -> str:
    seq = sequence.upper().replace(" ", "")
    if not seq:
        raise ValueError("Empty protein sequence")
    invalid = sorted({aa for aa in seq if aa not in STANDARD_AA})
    if invalid:
        raise ValueError(f"Non-standard amino acids in sequence: {invalid}")
    return seq
