"""FASTA reading utilities."""

from __future__ import annotations

from pathlib import Path

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> dict[str, str]:
    """Parse a FASTA file into ``{header_id: sequence}``."""
    path = Path(path)
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = _normalise_sequence("".join(chunks))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)

    if current_id is not None:
        sequences[current_id] = _normalise_sequence("".join(chunks))

    if not sequences:
        raise ValueError(f"No sequences found in {path}")

    return sequences


def read_first_sequence(path: str | Path) -> tuple[str, str]:
    """Return ``(protein_id, sequence)`` for the first record in a FASTA file."""
    items = read_fasta(path)
    protein_id, sequence = next(iter(items.items()))
    return protein_id, sequence


def _normalise_sequence(sequence: str) -> str:
    seq = sequence.upper().replace(" ", "")
    if not seq:
        raise ValueError("Empty protein sequence")
    invalid = sorted({aa for aa in seq if aa not in STANDARD_AA})
    if invalid:
        raise ValueError(f"Non-standard amino acids in sequence: {invalid}")
    return seq
