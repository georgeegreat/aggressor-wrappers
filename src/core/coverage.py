"""Predictions that cover only part of a sequence.

A file shorter than the protein it is filed under used to be refused outright,
and for good reason: accepted at face value it aligns at residue 1 and every
call it carries is shifted by however much was trimmed. Both of this project's
RPS12 files are 120 residues of a 132-residue chain, and taken at face value
each APR would have landed nine residues early.

Refusing is not the only honest answer though. A prediction over residues
10-129 is a real prediction about residues 10-129; what it is not is a
statement about 1-9 or 130-132. So a partial file is accepted when its
placement can be **established**, and the residues outside it are marked *not
predicted* rather than predicted-negative — those are different claims, and
collapsing them inflates every denominator computed downstream.

Placement is established two ways, never guessed:

``align_partial``
    The file carries residue identities (FoldAmyloid's ``Res`` column). The
    observed string is matched against the sequence and accepted only if it
    occurs at exactly one offset. One match is proof; several means the
    sequence repeats and the file cannot be placed; none means it is not this
    protein's.

a declared offset
    The file carries no identities at all (WALTZ's ``.dat`` is positions and
    scores). Nothing inside it can place it, so the offset has to be stated by
    the person who knows what was submitted, per protein, and is recorded in
    the output beside the numbers it shifts.
"""

from __future__ import annotations

from dataclasses import dataclass


class PartialCoverageError(ValueError):
    """A short file whose placement could not be established."""


@dataclass(frozen=True)
class Coverage:
    """Which residues of a sequence a prediction actually covers, 1-based."""

    start: int
    stop: int
    length: int

    @property
    def is_complete(self) -> bool:
        return self.start == 1 and self.stop == self.length

    @property
    def missing(self) -> int:
        return self.length - (self.stop - self.start + 1)

    def flags(self) -> list[int]:
        """1 where a residue was predicted, 0 where it was not."""
        return [1 if self.start <= pos <= self.stop else 0
                for pos in range(1, self.length + 1)]

    def describe(self) -> str:
        if self.is_complete:
            return f"complete (1-{self.length})"
        return (
            f"partial: residues {self.start}-{self.stop} of {self.length} "
            f"({self.missing} not predicted)"
        )


def align_partial(observed: str, sequence: str, *, source: object = "<file>") -> int:
    """The 0-based offset at which ``observed`` sits inside ``sequence``.

    Raises :class:`PartialCoverageError` unless there is exactly one.
    """
    observed = observed.strip().upper()
    sequence = sequence.upper()
    if not observed:
        raise PartialCoverageError(f"{source}: no residue identities to align with")
    if len(observed) > len(sequence):
        raise PartialCoverageError(
            f"{source}: {len(observed)} residues against a "
            f"{len(sequence)}-residue sequence -- longer, so not a sub-range of it"
        )

    offsets = []
    start = sequence.find(observed)
    while start != -1:
        offsets.append(start)
        start = sequence.find(observed, start + 1)

    if not offsets:
        raise PartialCoverageError(
            f"{source}: the residues this file reports do not occur anywhere in "
            f"the sequence it is filed under. It is a different protein, not a "
            f"truncation of this one."
        )
    if len(offsets) > 1:
        raise PartialCoverageError(
            f"{source}: these {len(observed)} residues occur at "
            f"{len(offsets)} places in the sequence "
            f"({', '.join(str(o + 1) for o in offsets[:4])}"
            f"{', …' if len(offsets) > 4 else ''}), so the file cannot be "
            f"placed. Declare the offset, or re-run on the full sequence."
        )
    return offsets[0]


def place(
    values: list[float],
    *,
    offset: int,
    length: int,
    fill: float = float("nan"),
) -> tuple[list[float], Coverage]:
    """Put ``values`` at ``offset`` inside a ``length``-residue track.

    Positions outside the covered window take ``fill`` — NaN by default, which
    is what marks them as not predicted. Zero would be a prediction.
    """
    if offset < 0 or offset + len(values) > length:
        raise PartialCoverageError(
            f"a {len(values)}-residue block at offset {offset} does not fit a "
            f"{length}-residue sequence"
        )
    placed = [fill] * length
    placed[offset : offset + len(values)] = values
    return placed, Coverage(offset + 1, offset + len(values), length)


def parse_offsets(raw: str | dict | None) -> dict[str, int]:
    """Read a per-protein offset declaration.

    ``"RPS12:9, RPL22:0"`` -> ``{"RPS12": 9, "RPL22": 0}``. Per protein by
    design: a single offset applied to a whole panel would silently shift every
    protein whose file was in fact complete, which is the failure this module
    exists to prevent, reintroduced one level up.
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k).strip(): int(v) for k, v in raw.items()}

    offsets: dict[str, int] = {}
    for item in str(raw).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"offset declaration {item!r} is not 'PROTEIN:OFFSET'. An "
                f"offset belongs to one protein; a bare number would be applied "
                f"to all of them."
            )
        protein_id, _, value = item.partition(":")
        try:
            offsets[protein_id.strip()] = int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"offset for {protein_id.strip()!r} is not an integer: {value!r}"
            ) from exc
    return offsets
