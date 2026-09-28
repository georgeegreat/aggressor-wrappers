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

    def __init__(self, threshold: float | None = None, use_tool_flag: bool = True) -> None:
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
        scores = [0.0] * n
        binary = [0] * n
        flagged = [0] * n
        observed: list[str] = [""] * n

        for _, row in df.iterrows():
            try:
                pos = int(row[number])
            except (TypeError, ValueError):
                continue
            idx = pos - 1
            if not 0 <= idx < n:
                raise ValueError(
                    f"{source}: FoldAmyloid position {pos} outside the "
                    f"{n}-residue sequence for {protein_id}"
                )
            scores[idx] = float(pd.to_numeric(row[value], errors="coerce") or 0.0)
            observed[idx] = str(row[residue]).strip().upper()[:1]
            if fold is not None:
                marker = str(row[fold]).strip().lower()
                flagged[idx] = 1 if marker in _TRUE_MARKERS else 0

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
            1 if s >= self.threshold else 0 for s in scores
        ]
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={
                "threshold": self.threshold,
                "binarised_from": "tool_flag" if (self.use_tool_flag and fold is not None) else "profile",
                "source": str(source),
            },
            aux={"tool_flag": flagged},
        )
