"""ArchCandy 2.0 region CSV -> per-residue table.

ArchCandy predicts **beta-arches** — beta-strand/loop/beta-strand motifs that
stack in parallel and in register to form a beta-arcade, the structural core of
most naturally occurring and disease-related amyloid fibrils (Kajava et al.,
2010, *FASEB J.* 24:1311; Ahmed et al., 2015, *Alzheimers Dement.* 11:681). It
therefore reports *segments with a topology string*, never a per-residue score,
and the projection to residues is a modelling step — exactly as for AmyloGram.

Version 2.0's CSV is ``ID,Sequence,Arch,Start,Stop,Score``. This is **not** the
standalone 1.0 layout (``Number,Digram,Score,Arc_type,Position``), so the two
get separate parsers rather than a column-position guess.

``score_mode``
    ``highest`` (default) — a residue takes the best single arch covering it,
    keeping scores on ArchCandy's own [0, 1] confidence scale and comparable to
    its published bands.
    ``cumulative`` — overlapping arches are summed, reproducing the web UI's
    cumulative tab. On Abeta42 the 14 candidates overlap heavily, so cumulative
    pushes residues far above 1.0; such a number is no longer an ArchCandy score
    and cannot be compared with any threshold the authors calibrated.

``threshold``
    Applied to the per-residue projection. ArchCandy 2.0's documentation gives
    the calibration directly: below 0.40 a prediction is non-significant,
    0.40-0.57 ambiguous, above 0.57 significant. The web default is 0.40.
    The topology of each retained arch is preserved in ``regions`` — the arch
    string (``GBPL``, ``BLLPBL``, ``BEPL``…) is the structural hypothesis and is
    lost entirely by any per-residue summary.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec
from aggressor_wrappers.predictors.base import BasePredictorParser

REQUIRED = ("Start", "Stop", "Score")


def _read(source: str | Path) -> pd.DataFrame:
    text = Path(source).read_bytes().decode("utf-8-sig", errors="replace")
    delimiter = max((",", ";", "\t"), key=text.splitlines()[0].count)
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df


class ArchCandy2Parser(BasePredictorParser):
    spec = get_predictor_spec("archcandy")

    def __init__(self, score_mode: str = "highest", threshold: float = 0.40) -> None:
        if score_mode not in ("highest", "cumulative"):
            raise ValueError("score_mode must be 'highest' or 'cumulative'")
        self.score_mode = score_mode
        self.threshold = float(threshold)

    def parse(
        self,
        source: str | Path,
        *,
        protein_id: str,
        sequence: str,
        **kwargs,
    ) -> PredictorResult:
        df = _read(source)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"{source}: ArchCandy 2.0 CSV missing {missing}; has {list(df.columns)}. "
                f"A file with 'Number,Digram,Score,Arc_type,Position' is the "
                f"standalone 1.0 layout -- use ArchCandyParser for that."
            )

        n = len(sequence)
        scores = [0.0] * n
        arch_count = [0] * n
        regions: list[dict] = []

        for _, row in df.iterrows():
            try:
                start, stop = int(row["Start"]), int(row["Stop"])
                score = float(row["Score"])
            except (TypeError, ValueError):
                continue
            if start < 1 or stop > n or stop < start:
                raise ValueError(
                    f"{source}: arch {start}-{stop} outside the {n}-residue "
                    f"sequence for {protein_id}"
                )
            regions.append(
                {
                    "start": start,
                    "stop": stop,
                    "score": score,
                    # The topology string is the structural hypothesis: which
                    # positions are strand (B), loop/bend (L, P), etc. No
                    # per-residue projection can carry it, so it is kept whole.
                    "arch": str(row.get("Arch", "")).strip(),
                    "segment": str(row.get("Sequence", "")).strip(),
                }
            )
            for pos in range(start, stop + 1):
                idx = pos - 1
                if self.score_mode == "highest":
                    scores[idx] = max(scores[idx], score)
                else:
                    scores[idx] += score
                arch_count[idx] += 1

        binary = [1 if s >= self.threshold else 0 for s in scores]
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=self.spec,
            scores=scores,
            binary=binary,
            metadata={
                "score_mode": self.score_mode,
                "threshold": self.threshold,
                "n_arches": len(regions),
                "version": "2.0",
                "source": str(source),
            },
            aux={"arch_count": arch_count},
            regions=regions,
        )
