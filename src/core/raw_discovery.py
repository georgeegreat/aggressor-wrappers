"""Find a protein's raw predictor output under the names it actually arrives with.

Every runner's ``discover_outputs`` looked for exactly one filename: the one
that runner writes itself. That is correct for output the runner produced and
useless for output it did not, which is the case that matters when a web
service is down and the only copy of a prediction is a file saved from the
server months ago. WALTZ's own download is ``WaltzJob_<job>_<protein>.dat``;
FoldAmyloid's is whatever the browser was told to call it. Neither is
``{protein}_waltz.txt`` or ``{protein}_foldamyloid.csv``, so a drop zone full
of perfectly good predictions discovered nothing.

Matching is on the whole stem, never a prefix. ``RPL36`` and ``RPL36a`` are
both in this project's panel, and a ``RPL36*`` glob would quietly hand one
protein the other's track -- the single worst failure available here, because
it produces a complete, plausible, wrong answer. Rules are tried in order and
the first that matches anything wins; two files matching the SAME rule is an
ambiguity and is refused by name rather than resolved by sort order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NameRule:
    """One way a protein's raw file may be named.

    ``stems`` are exact stem templates (``"{pid}_waltz"``); ``pattern`` is a
    regex matched against the whole stem, with ``{pid}`` substituted as a
    quoted literal so a protein id containing regex metacharacters cannot
    widen it.
    """

    description: str
    stems: tuple[str, ...] = ()
    pattern: str | None = None

    def matches(self, stem: str, protein_id: str) -> bool:
        if any(stem == template.format(pid=protein_id) for template in self.stems):
            return True
        if self.pattern is None:
            return False
        expanded = self.pattern.format(pid=re.escape(protein_id))
        return re.fullmatch(expanded, stem) is not None


class AmbiguousRawOutput(ValueError):
    """More than one file in the same directory claims the same protein."""


def find_protein_file(
    directory: Path,
    protein_id: str,
    rules: list[NameRule],
    *,
    suffixes: tuple[str, ...],
) -> Path | None:
    """The one raw file in ``directory`` belonging to ``protein_id``, or None.

    ``suffixes`` is checked case-insensitively and is a whitelist: a directory
    also holds parsed CSVs, logs and job sidecars, and a rule that matched one
    of those would feed the wrong file to a parser that might well accept it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None

    candidates = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    for rule in rules:
        hits = [path for path in candidates if rule.matches(path.stem, protein_id)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AmbiguousRawOutput(
                f"{directory}: {len(hits)} files claim to be {protein_id}'s "
                f"{rule.description} ({', '.join(p.name for p in hits)}). "
                f"Leave one and move the others out: choosing between them by "
                f"sort order would attribute one protein's prediction to "
                f"another without saying so."
            )
    return None


def find_protein_files(
    directory: Path,
    protein_ids: list[str],
    rules: list[NameRule],
    *,
    suffixes: tuple[str, ...],
) -> dict[str, Path]:
    """``find_protein_file`` over a batch; proteins with no file are absent."""
    found: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = find_protein_file(directory, protein_id, rules, suffixes=suffixes)
        if path is not None:
            found[protein_id] = path
    return found


#: WALTZ. The canonical names first -- those are what this package writes and
#: what a resumed run should pick up -- then the server's own download name,
#: then a bare rename.
WALTZ_RULES = [
    NameRule("canonical output", stems=("{pid}_waltz",)),
    NameRule(
        "WALTZ server download",
        pattern=r"WaltzJob[_-].*[_-]{pid}",
    ),
    NameRule("bare protein name", stems=("{pid}",)),
]
WALTZ_SUFFIXES = (".txt", ".dat")

#: FoldAmyloid. The server offers no fixed filename at all, so a bare rename is
#: the common case rather than the fallback.
FOLDAMYLOID_RULES = [
    NameRule("canonical output", stems=("{pid}_foldamyloid", "{pid}_foldamyloid_long")),
    NameRule("bare protein name", stems=("{pid}",)),
    NameRule("FoldAmyloid server download", pattern=r"(?i)foldamyloid[_-]{pid}"),
]
FOLDAMYLOID_SUFFIXES = (".csv", ".txt", ".tsv")
