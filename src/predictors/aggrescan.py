"""AGGRESCAN per-residue export -> standard per-residue table.

AGGRESCAN (Conchillo-Sole et al., 2007, *BMC Bioinformatics* 8:65) derives an
intrinsic aggregation-propensity value ``a3v`` per amino acid from an in vivo
aggregation assay, then reports ``a4v``: that value averaged over a
sequence-length-dependent window. Hot Spots are the contiguous stretches where
the a4v profile rises above the calibrated threshold, and the tool flags their
residues explicitly.

Because the score is experimentally calibrated on cellular aggregation rather
than fitted to fibril structures, AGGRESCAN contributes a different kind of
evidence from the structure-based members of a panel (ArchCandy's beta-arch
compatibility, PASTA's pairing energy) -- which is the point of running them
together.

Two parsing hazards this handles explicitly:

**Delimiter and header drift.** Exports appear comma- and semicolon-separated,
and the score column is written ``a4v`` in some builds and ``Score`` in others;
the Hot Spot marker is ``Prediction`` in some and ``Disorder`` in others.
Positional column assignment turns any of these into either a crash or, worse,
a silently transposed track.

**Whitespace in the marker column.** AGGRESCAN leaves the marker blank outside
Hot Spots, but its files also carry stray non-breaking spaces there (0xCA in the
Mac Roman exports). Treating "not empty" as "predicted" converts that artifact
into a phantom single-residue APR -- observed at position 1 of the RPL27 export.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

_DELIMITERS = (",", ";", "\t")
_TRUE_MARKERS = frozenset({"true", "yes", "y", "hs", "hotspot"})


def _read(source: str | Path) -> pd.DataFrame:
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
            f"{source}: AGGRESCAN header has no recognised delimiter: {header[:120]!r}"
        )
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _column(df, aliases, *, source, what, required=True):
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    if not required:
        return None
    raise ValueError(
        f"{source}: no {what} column (looked for {list(aliases)}); "
        f"file has {list(df.columns)}"
    )


def hotspot_mask(series: pd.Series) -> list[int]:
    """Strict boolean from AGGRESCAN's Hot Spot marker column."""
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce").fillna(0)
    truthy = text.str.lower().isin(_TRUE_MARKERS)
    return [int(bool(v)) for v in ((numeric > 0) | truthy)]


class AggrescanParser(BasePredictorParser):
    spec = get_predictor_spec("aggrescan")

    def __init__(self, threshold: float | None = None, use_tool_flag: bool = True) -> None:
        """
        ``use_tool_flag`` (default) honours AGGRESCAN's own Hot Spot calls, which
        embed both its calibrated threshold and its length-dependent averaging
        window. Re-deriving them with a fixed ``threshold`` on a4v ignores the
        window, so it is opt-in.
        """
        self.threshold = float(threshold) if threshold is not None else 0.0
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
        number = _column(df, ("Number", "Position", "Num"), source=source, what="residue index")
        residue = _column(df, ("AA", "Residue", "Res", "amino_acid"), source=source, what="residue identity")
        score = _column(df, ("a4v", "Score", "a3v"), source=source, what="a4v score")
        marker = _column(
            df, ("Prediction", "Disorder", "HotSpot", "Hot_Spot", "HS"),
            source=source, what="hot-spot marker", required=False,
        )

        n = len(sequence)
        scores = [0.0] * n
        flagged = [0] * n
        observed = [""] * n
        mask = hotspot_mask(df[marker]) if marker is not None else [0] * len(df)

        for row_i, (_, row) in enumerate(df.iterrows()):
            try:
                pos = int(row[number])
            except (TypeError, ValueError):
                continue
            idx = pos - 1
            if not 0 <= idx < n:
                raise ValueError(
                    f"{source}: AGGRESCAN position {pos} outside the "
                    f"{n}-residue sequence for {protein_id}"
                )
            scores[idx] = float(pd.to_numeric(row[score], errors="coerce") or 0.0)
            observed[idx] = str(row[residue]).strip().upper()[:1]
            flagged[idx] = mask[row_i]

        mismatches = [
            i + 1 for i, (a, b) in enumerate(zip(observed, sequence)) if a and a != b.upper()
        ]
        if mismatches:
            raise ValueError(
                f"{source}: AGGRESCAN output does not match the sequence given "
                f"for {protein_id}: {len(mismatches)} residue mismatch(es), first "
                f"at position {mismatches[0]} (file says "
                f"{observed[mismatches[0]-1]!r}, sequence has "
                f"{sequence[mismatches[0]-1]!r}). The file is probably for a "
                f"different protein."
            )

        binary = flagged if (self.use_tool_flag and marker is not None) else [
            1 if s > self.threshold else 0 for s in scores
        ]
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={
                "binarised_from": "tool_flag" if (self.use_tool_flag and marker is not None) else "a4v_threshold",
                "threshold": self.threshold,
                "source": str(source),
            },
            aux={"hot_spot": flagged},
        )
