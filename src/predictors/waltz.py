"""WALTZ output → standard per-residue table, in either format the server gives.

WALTZ offers two downloads and this parser reads both, because which one a
person has depends on when they fetched it rather than on what they wanted.

**Detailed text** (``>header`` sections, then ``start-end<TAB>?<TAB>score``)
is what the HTML result page yields and what this package's runner saves. It
carries several proteins in one file.

**Per-residue** (``position<TAB>score``, one line per residue, zero outside a
predicted segment) is the ``.dat`` the server packages in its ZIP under
``text_long_graph``, named ``WaltzJob_<job>_<protein>.dat``. It carries one
protein and no header, so the protein it belongs to is known only from the
filename -- which is why the length check below is not optional: it is the only
evidence inside the file that it is the right one.

The format is detected from the content, never from the extension. A ``.txt``
saved from the ZIP holds per-residue data and a ``.dat`` renamed by hand may
hold either, so keying on the suffix would be a guess dressed up as a rule.
"""

from __future__ import annotations

import re
from pathlib import Path

from aggressor_wrappers.core.coverage import Coverage, parse_offsets, place
from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

_SECTION_HEADER_RE = re.compile(r"^>(.+)$")


def split_detailed_sections(text: str) -> dict[str, str]:
    """Split combined WALTZ detailed output into ``{protein_id: section_text}``."""
    sections: dict[str, str] = {}
    current_id: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = _SECTION_HEADER_RE.match(line)
        if match:
            if current_id is not None:
                sections[current_id] = "\n".join(current_lines) + "\n"
            current_id = match.group(1).strip()
            current_lines = [line]
        elif current_id is not None:
            current_lines.append(line)

    if current_id is not None:
        sections[current_id] = "\n".join(current_lines) + "\n"
    return sections


def _parse_position_range(pos_range: str) -> tuple[int, int]:
    if "-" not in pos_range:
        raise ValueError(f"Invalid WALTZ position range: {pos_range!r}")
    start_s, end_s = pos_range.split("-", 1)
    return int(start_s), int(end_s)


def _apply_region_scores(
    scores: list[float],
    binary: list[int],
    *,
    sequence: str,
    start: int,
    end: int,
    score: float,
) -> None:
    for pos in range(start, end + 1):
        idx = pos - 1
        if not 0 <= idx < len(sequence):
            raise ValueError(f"WALTZ position out of range: {pos}")
        scores[idx] = score
        binary[idx] = 1 if score != 0 else 0


#: A per-residue line: an integer position, whitespace, a number. The position
#: must not contain '-', which is what separates it from the detailed format's
#: ``start-end`` first field.
_PER_RESIDUE_LINE = re.compile(r"^\s*(\d+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")


def looks_per_residue(text: str) -> bool:
    """Whether ``text`` is the two-column per-residue download.

    Decided on the first few data lines rather than on one, because a detailed
    file whose first region happens to start at position 1 would otherwise be
    misread, and on the ABSENCE of a '>' header, which the detailed format
    always has and the per-residue format never does.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    if any(line.lstrip().startswith(">") for line in lines[:5]):
        return False
    sample = lines[:5]
    return all(_PER_RESIDUE_LINE.match(line) for line in sample)


def parse_per_residue(
    text: str,
    *,
    sequence: str,
    source: object = "<text>",
    offset: int | None = None,
) -> tuple[list[float], list[int], Coverage]:
    """Read ``position<TAB>score`` into aligned score and call lists.

    Positions must cover 1..len(sequence) exactly once. A short file is the
    failure mode that matters: the ``.dat`` has no header, so the only thing
    tying it to a protein is its filename, and a file one residue short -- or
    one belonging to a different chain of similar length -- parses perfectly
    and attributes someone else's segments. Refusing on length is the only
    check the format affords.
    """
    n = len(sequence)
    values: dict[int, float] = {}
    seen: set[int] = set()

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _PER_RESIDUE_LINE.match(line)
        if match is None:
            raise ValueError(
                f"{source}: line {lineno} is not 'position<TAB>score': {line.strip()[:80]!r}"
            )
        position = int(match.group(1))
        if position in seen:
            raise ValueError(f"{source}: position {position} appears twice")
        seen.add(position)
        values[position] = float(match.group(2))

    first, last = min(values), max(values)
    if last - first + 1 != len(values):
        raise ValueError(
            f"{source}: positions {first}-{last} are not contiguous "
            f"({len(values)} lines). A WALTZ profile with holes in it cannot "
            f"be placed against a sequence."
        )
    block = [values[pos] for pos in range(first, last + 1)]

    if len(block) > n:
        raise ValueError(
            f"{source}: {len(block)} positions against a {n}-residue sequence. "
            f"A longer profile is not a sub-range of this protein -- it is a "
            f"different protein, or a different isoform of this one."
        )
    if last > n and offset is None:
        raise ValueError(
            f"{source}: position {last} outside the {n}-residue sequence. The "
            f"file is for a different protein, or a different isoform of this "
            f"one."
        )

    if offset is None:
        if len(block) != n or first != 1:
            # No residue identities anywhere in this format, so nothing INSIDE
            # the file can place it: a 120-line file against a 132-residue
            # chain is equally consistent with residues 1-120 and with 13-132,
            # and the two readings put every segment twelve residues apart.
            # Only the person who knows what was submitted can say, so the
            # offset is declared rather than inferred.
            raise ValueError(
                f"{source}: covers {len(block)} of {n} positions "
                f"({first}-{last}); {n - len(block)} absent. A per-residue "
                f"WALTZ file carries no residue "
                f"identities, so a partial one cannot be placed from its "
                f"contents. Re-run WALTZ on the full sequence, or declare where "
                f"this block starts: [predictors.waltz] offsets = "
                f"<PROTEIN>:<0-based offset>."
            )
        offset = 0

    scores, coverage = place(block, offset=offset, length=n)

    # Nonzero means "inside a predicted segment": the server writes the
    # segment's score across its residues and 0.0 everywhere else, which is the
    # same convention the detailed parser applies region by region. NaN marks a
    # residue the tool never saw, and is not a call.
    binary = [1 if (value == value and value != 0) else 0 for value in scores]
    return scores, binary, coverage


class WALTZParser(BasePredictorParser):
    spec = get_predictor_spec("waltz")

    def __init__(self, offsets: str | dict | None = None) -> None:
        """``offsets`` places a partial per-residue file, per protein.

        ``"RPS12:9"`` says RPS12's .dat begins at residue 10. Per protein by
        design: one offset applied to a whole panel would silently shift every
        protein whose file was in fact complete.
        """
        self.offsets = parse_offsets(offsets)

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        text = Path(source).read_text()
        if looks_per_residue(text):
            scores, binary, coverage = parse_per_residue(
                text,
                sequence=sequence,
                source=source,
                offset=self.offsets.get(protein_id),
            )
            layout = "per_residue"
        else:
            section = self._extract_section(text, protein_id=protein_id)
            scores, binary = self._parse_detailed_section(section, sequence=sequence)
            coverage = Coverage(1, len(sequence), len(sequence))
            layout = "detailed"

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            aux={"covered": coverage.flags()},
            metadata={
                "layout": layout,
                "source": str(source),
                "coverage_start": coverage.start,
                "coverage_stop": coverage.stop,
                "coverage": coverage.describe(),
            },
        )

    def _extract_section(self, text: str, *, protein_id: str) -> str:
        sections = split_detailed_sections(text)
        target = protein_id.strip()
        if target in sections:
            return sections[target]
        for key, body in sections.items():
            if key.strip() == target:
                return body
        known = ", ".join(sorted(sections))
        raise ValueError(f"WALTZ output missing section for {protein_id!r}. Found: {known}")

    def _parse_detailed_section(self, section: str, *, sequence: str) -> tuple[list[float], list[int]]:
        scores = [0.0] * len(sequence)
        binary = [0] * len(sequence)

        for line in section.splitlines():
            line = line.strip()
            if not line or line.startswith(">") or line.startswith("Positions"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            start, end = _parse_position_range(parts[0])
            score = float(parts[2])
            _apply_region_scores(
                scores,
                binary,
                sequence=sequence,
                start=start,
                end=end,
                score=score,
            )
        return scores, binary
