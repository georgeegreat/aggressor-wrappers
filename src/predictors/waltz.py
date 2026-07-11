"""WALTZ web detailed text output → standard per-residue table."""

from __future__ import annotations

import re
from pathlib import Path

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


class WALTZParser(BasePredictorParser):
    spec = get_predictor_spec("waltz")

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        text = Path(source).read_text()
        section = self._extract_section(text, protein_id=protein_id)
        scores, binary = self._parse_detailed_section(section, sequence=sequence)

        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
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
