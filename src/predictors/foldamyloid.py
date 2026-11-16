"""FoldAmyloid per-residue export -> standard per-residue table.

FoldAmyloid (Garbuzynskiy, Lobanov & Galzitskaya, 2010, *Bioinformatics* 26:326)
predicts amyloidogenic segments from *packing density* rather than
hydrophobicity: each residue is scored by its expected number of contacts within
8 A, averaged over a 5-residue frame, and residues whose averaged profile
exceeds 21.4 are flagged. The mechanistic claim is that a cross-beta spine
requires tightly interdigitated side chains -- a steric zipper -- so the
segments able to form one are those predicted to be densely contacting. This
makes FoldAmyloid genuinely non-redundant with the hydrophobic-cluster models
(AGGRESCAN) and the beta-pairing-energy models (PASTA 2.0) in the same panel.

Native export: ``Num,Res,Fold,Value`` with a UTF-8 BOM and CRLF line endings.
``Fold`` is ``f`` for a flagged residue and empty (or ``-``, depending on build)
otherwise; ``Value`` is the averaged contact count.

Two channels are preserved beyond the score: the tool's own ``f`` flag, which is
the authoritative call because it embeds the calibrated threshold *and* the
frame averaging, and the raw profile, so a different operating point can be
applied without re-running the server.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.coverage import (
    Coverage,
    PartialCoverageError,
    align_partial,
    place,
)
from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

_DELIMITERS = (",", ";", "\t")
_TRUE_MARKERS = frozenset({"f", "1", "true", "yes"})


def _read(source: str | Path) -> pd.DataFrame:
    """Read the export, sniffing delimiter and encoding.

    Not ``pd.read_csv(source)``: these files reach the pipeline as comma or
    semicolon separated depending on locale, and a BOM renames the first column
    if it is not stripped. Resolving columns by NAME afterwards is what stops a
    build that reorders them from producing a silently wrong track.
    """
    raw = Path(source).read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    header = next((ln for ln in text.splitlines() if ln.strip()), "")
    delimiter = max(_DELIMITERS, key=header.count)
    if header.count(delimiter) == 0:
        raise ValueError(
            f"{source}: FoldAmyloid header has no recognised delimiter: {header[:120]!r}"
        )
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _column(df: pd.DataFrame, aliases: tuple[str, ...], *, source, what: str) -> str:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    raise ValueError(
        f"{source}: no {what} column (looked for {list(aliases)}); "
        f"file has {list(df.columns)}"
    )


class FoldAmyloidParser(BasePredictorParser):
    spec = get_predictor_spec("foldamyloid")

    def __init__(
        self,
        threshold: float | None = None,
        use_tool_flag: bool = True,
        accept_partial: bool = True,
    ) -> None:
        """
        ``use_tool_flag`` (default) binarises on FoldAmyloid's own ``f`` marker.
        Setting it False re-derives the call by thresholding the profile, which
        is only equivalent if ``threshold`` matches the frame size and cutoff the
        server actually ran -- the export does not record either, so the flag is
        the safer default.
        """
        self.threshold = (
            float(threshold) if threshold is not None else (self.spec.default_threshold or 21.4)
        )
        self.use_tool_flag = bool(use_tool_flag)
        #: Whether a file covering a contiguous SUB-RANGE of the sequence is
        #: accepted. It is, when the ``Res`` column places it uniquely: a
        #: prediction over residues 10-129 is a real prediction about residues
        #: 10-129. The residues outside it are marked not-predicted, never
        #: predicted-negative. Set False to require full coverage.
        self.accept_partial = bool(accept_partial)

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        df = _read(source)
        number = _column(df, ("Num", "Number", "Position", "N"), source=source, what="residue index")
        value = _column(df, ("Value", "Score", "Contacts"), source=source, what="profile value")
        residue = _column(df, ("Res", "Residue", "AA"), source=source, what="residue identity")
        fold = next(
            (df.columns[i] for i, c in enumerate(df.columns) if str(c).strip().lower() in ("fold", "flag")),
            None,
        )

        n = len(sequence)
        # Read the file on its OWN axis first, positions 1..m as it numbers
        # them, and only then decide where that block sits in the sequence. The
        # previous version wrote straight into a sequence-length array, so a
        # file covering residues 10-129 of 132 landed at 1-120 before anything
        # could notice -- every call nine residues early, and the mismatch
        # reported as "a different protein".
        rows = []
        for _, row in df.iterrows():
            try:
                pos = int(row[number])
            except (TypeError, ValueError):
                continue
            rows.append(
                (
                    pos,
                    float(pd.to_numeric(row[value], errors="coerce") or 0.0),
                    str(row[residue]).strip().upper()[:1],
                    1 if (fold is not None
                          and str(row[fold]).strip().lower() in _TRUE_MARKERS)
                    else 0,
                )
            )
        if not rows:
            raise ValueError(f"{source}: FoldAmyloid export has no usable rows")

        rows.sort(key=lambda item: item[0])
        first, last = rows[0][0], rows[-1][0]
        if last - first + 1 != len(rows):
            raise ValueError(
                f"{source}: FoldAmyloid positions {first}-{last} are not "
                f"contiguous ({len(rows)} rows). A profile with holes in it "
                f"cannot be placed against a sequence."
            )

        block_scores = [score for _, score, _, _ in rows]
        block_observed = "".join(res for _, _, res, _ in rows)
        block_flags = [flag for _, _, _, flag in rows]

        offset = first - 1
        coverage = Coverage(1, n, n)
        if len(rows) == n and block_observed == sequence.upper():
            offset = 0
        elif block_observed and block_observed == sequence[offset : offset + len(rows)].upper():
            # The file's own numbering already places it correctly.
            pass
        elif block_observed:
            if not self.accept_partial and len(rows) != n:
                raise ValueError(
                    f"{source}: FoldAmyloid covers {len(rows)} of {n} residues "
                    f"and accept_partial is off"
                )
            # Its numbering disagrees with the sequence, or it is short. Place
            # it by its residues, which is proof rather than assumption.
            try:
                offset = align_partial(block_observed, sequence, source=source)
            except PartialCoverageError as exc:
                # The identity check predates partial coverage and its wording
                # is what every caller and test recognises a mis-filed export
                # by. Placing failed, so this IS that case; the mismatch detail
                # is what makes it actionable.
                against = sequence[: len(block_observed)].upper()
                mismatches = [
                    i + 1
                    for i, (a, b) in enumerate(zip(block_observed, against))
                    if a != b
                ]
                detail = ""
                if mismatches:
                    first = mismatches[0]
                    detail = (
                        f": {len(mismatches)} residue mismatch(es), first at "
                        f"position {first} (file says "
                        f"{block_observed[first - 1]!r}, sequence has "
                        f"{against[first - 1]!r})"
                    )
                raise ValueError(
                    f"{source}: FoldAmyloid output does not match the sequence "
                    f"given for {protein_id}{detail}. {exc}"
                ) from exc
        elif len(rows) != n:
            raise ValueError(
                f"{source}: FoldAmyloid covers {len(rows)} of {n} residues and "
                f"reports no residue identities, so it cannot be placed"
            )

        scores, coverage = place(block_scores, offset=offset, length=n)
        flagged_full, _ = place(
            [float(f) for f in block_flags], offset=offset, length=n, fill=0.0
        )
        flagged = [int(f) for f in flagged_full]
        observed = [""] * n
        for i, residue_letter in enumerate(block_observed):
            observed[offset + i] = residue_letter
        binary = [0] * n

        # Identity check. Predictor outputs have been mis-filed before -- two
        # FoldAmyloid exports in this project carry each other's protein -- and a
        # swapped file parses perfectly while attributing one chain's APRs to
        # another. Comparing the residue letters the tool itself reported against
        # the query sequence is the only cheap way to catch it.
        seen = "".join(observed)
        if seen.strip("\0 ") and any(observed):
            mismatches = [
                i + 1
                for i, (a, b) in enumerate(zip(observed, sequence))
                if a and a != b.upper()
            ]
            if mismatches:
                raise ValueError(
                    f"{source}: FoldAmyloid output does not match the sequence "
                    f"given for {protein_id}: {len(mismatches)} residue "
                    f"mismatch(es), first at position {mismatches[0]} "
                    f"(file says {observed[mismatches[0]-1]!r}, sequence has "
                    f"{sequence[mismatches[0]-1]!r}). The file is probably for a "
                    f"different protein."
                )

        binary = flagged if (self.use_tool_flag and fold is not None) else [
            # NaN marks a residue the tool never saw; it is not a call.
            1 if (s == s and s >= self.threshold) else 0 for s in scores
        ]
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            aux={"tool_flag": flagged, "covered": coverage.flags()},
            metadata={
                "threshold": self.threshold,
                "binarised_from": "tool_flag" if (self.use_tool_flag and fold is not None) else "profile",
                "source": str(source),
                "coverage_start": coverage.start,
                "coverage_stop": coverage.stop,
                "coverage": coverage.describe(),
            },
        )
