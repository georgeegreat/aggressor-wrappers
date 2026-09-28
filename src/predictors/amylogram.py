"""AmyloGram: per-peptide probabilities mapped onto a per-residue profile.

AmyloGram returns one probability per *query sequence*, not per residue, so it
cannot join a positional consensus without an explicit modelling step.

The step is not arbitrary. AmyloGram is an n-gram model trained on
**hexapeptides** (the AmyLoad set); handing it a 300-residue protein asks it for
a judgement outside the regime it was fitted for, and the single number it
returns is not a calibrated protein-level amyloidogenicity. Scoring overlapping
hexapeptides and projecting those onto residues is therefore *closer* to the
model's training distribution than whole-protein input, not merely more
convenient.

Defaults, all overridable:

``windows = (6,)``
    AmyloGram's training unit. Windows are taken with step 1, so an interior
    residue is covered by exactly *w* of them.

    Several widths may be requested, and each is projected separately and kept
    as a ``w{n}_Score`` column so boundary sensitivity can be examined without
    re-querying. Be explicit about what that means, though: AmyloGram was fitted
    on hexapeptides, so a query longer than 6 is outside its training
    distribution and its probability is not calibrated there. Widths above 6 are
    a sensitivity device, not a better prediction, and the runner warns when one
    is used. To broaden APR extent while staying in-regime, lower
    ``support_fraction`` instead — that changes how 6-mer evidence is pooled, not
    what the model was asked.
``aggregation = "support"``
    How the probabilities of the windows covering a residue collapse into one
    score. This is the consequential choice, and neither extreme is right.

    ``max`` asks only that a residue lie in *at least one* amyloidogenic
    hexapeptide. One positive 6-mer therefore raises all six of its residues, so
    every called region is inflated by up to *w*-1 = 5 residues at each end. On
    Abeta42 ``max`` returns 14-25 where the nucleating segment is KLVFFA
    (16-21).

    ``mean`` asks for the average, which dilutes a single strongly amyloidogenic
    hexapeptide against its weaker neighbours and contracts regions onto their
    peaks. (The ArchCandy ``highest``-over-``cumulative`` argument is NOT the
    same one: there the defect is that summing leaves the tool's [0, 1] scale
    entirely. Here both rules stay on scale; what changes is APR extent.)

    ``support`` (default) makes the strictness explicit rather than implicit. A
    residue's score is the quantile of its covering-window probabilities at
    level ``1 - support_fraction``, so ``support_fraction`` reads directly as
    *what proportion of the hexapeptide evidence covering this residue must be
    positive*. ``1/window`` reproduces ``max``, ``0.5`` gives the median, ``1.0``
    gives the minimum. The default 0.5 asks for majority support.

    Why this matters biologically: the hexapeptide is the unit of the steric
    zipper, but it is not the unit of an APR. Many experimentally confirmed
    amyloid cores are far broader than six residues — the alpha-synuclein NAC
    region spans ~35, prion-forming domains span tens — and a rule that calls a
    residue on single-window evidence and a rule that demands unanimous support
    give materially different extents for exactly those cases. Fixing the rule
    silently at either extreme hides that decision inside a projection step.
``threshold = 0.5``
    The model returns a probability, so 0.5 is the natural cut; it is exposed
    because the operating point should be chosen against whatever validation set
    the panel is tuned on, not assumed.

Sequences shorter than the window are scored as a single peptide covering the
whole sequence, rather than dropped: a 5-residue peptide is still a legitimate
query for a hexapeptide model, and silently returning an empty profile would
remove the tool from the consensus without saying so.
"""

import math
from pathlib import Path

import pandas as pd

DEFAULT_WINDOW = 6
DEFAULT_WINDOWS = (6,)
DEFAULT_AGGREGATION = "support"
DEFAULT_SUPPORT_FRACTION = 0.5
VALID_AGGREGATIONS = frozenset({"max", "min", "mean", "median", "quantile", "support"})


def sliding_windows(sequence: str, window: int = DEFAULT_WINDOW) -> list[tuple[int, str]]:
    """Return ``(start_1based, peptide)`` for each window of ``sequence``."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if len(sequence) <= window:
        return [(1, sequence)]
    return [
        (i + 1, sequence[i : i + window])
        for i in range(len(sequence) - window + 1)
    ]


def write_peptide_fasta(
    sequence: str,
    dest: Path,
    *,
    window: int = DEFAULT_WINDOW,
    prefix: str = "w",
) -> list[tuple[int, str]]:
    """Write one FASTA record per window; return the window list."""
    windows = sliding_windows(sequence, window)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for start, peptide in windows:
            handle.write(f">{prefix}{start}\n{peptide}\n")
    return windows


def _aggregate(
    values: list[float],
    *,
    aggregation: str,
    support_fraction: float,
    quantile: float,
) -> float:
    """Collapse the probabilities of the windows covering one residue."""
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    n = len(ordered)
    if aggregation == "max":
        return ordered[0]
    if aggregation == "min":
        return ordered[-1]
    if aggregation == "mean":
        return sum(ordered) / n
    if aggregation == "median":
        mid = n // 2
        return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    if aggregation == "quantile":
        # quantile of the DESCENDING order: q=0 -> max, q=1 -> min.
        idx = min(n - 1, max(0, int(round(quantile * (n - 1)))))
        return ordered[idx]
    if aggregation == "support":
        # The score a residue attains if `support_fraction` of its covering
        # windows must be at least that high. Index into the descending order at
        # ceil(f * n) - 1: f = 1/n -> ordered[0] (max), f = 1 -> ordered[-1].
        k = max(1, min(n, math.ceil(support_fraction * n)))
        return ordered[k - 1]
    raise ValueError(f"unknown aggregation {aggregation!r}")


def project_windows(
    windows: list[tuple[int, str]],
    probabilities: dict[str, float] | list[float],
    sequence_length: int,
    *,
    aggregation: str = DEFAULT_AGGREGATION,
    support_fraction: float = DEFAULT_SUPPORT_FRACTION,
    quantile: float = 0.5,
    prefix: str = "w",
) -> list[float]:
    """Map per-window probabilities onto per-residue scores.

    The covering windows of each residue are collected first and reduced
    afterwards, rather than folded in as they arrive. That costs one list per
    residue and buys every order-dependent rule -- median, quantile, support --
    which a running max or running sum cannot express.
    """
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {sorted(VALID_AGGREGATIONS)}; got {aggregation!r}"
        )
    if not 0.0 < support_fraction <= 1.0:
        raise ValueError(f"support_fraction must be in (0, 1]; got {support_fraction}")

    if isinstance(probabilities, dict):
        probs = []
        for start, _peptide in windows:
            key = f"{prefix}{start}"
            if key not in probabilities:
                raise KeyError(f"AmyloGram output has no probability for window {key!r}")
            probs.append(float(probabilities[key]))
    else:
        if len(probabilities) != len(windows):
            raise ValueError(
                f"AmyloGram returned {len(probabilities)} probabilities for "
                f"{len(windows)} windows"
            )
        probs = [float(p) for p in probabilities]

    covering: list[list[float]] = [[] for _ in range(sequence_length)]
    for (start, peptide), prob in zip(windows, probs, strict=True):
        for offset in range(len(peptide)):
            idx = start - 1 + offset
            if 0 <= idx < sequence_length:
                covering[idx].append(prob)

    return [
        _aggregate(
            values,
            aggregation=aggregation,
            support_fraction=support_fraction,
            quantile=quantile,
        )
        for values in covering
    ]


def coverage_depth(sequence_length: int, window: int) -> list[int]:
    """Number of windows covering each residue.

    Terminal residues are covered by fewer windows than interior ones -- residue
    1 by a single window, residue *w* by *w*. Any rule other than ``max`` is
    therefore evaluated over a smaller sample at the termini, which is the same
    edge effect that inflates wide-probe scores in amyloid_predict. Exposed so a
    caller can down-weight or mask the first and last *w*-1 positions rather
    than discovering the asymmetry in a figure.
    """
    depth = [0] * sequence_length
    for start, _ in sliding_windows("X" * sequence_length, window):
        for offset in range(min(window, sequence_length)):
            idx = start - 1 + offset
            if 0 <= idx < sequence_length:
                depth[idx] += 1
    return depth


def parse_amylogram_output(path: str | Path) -> dict[str, float]:
    """Read the CSV written by the AmyloGram helper script.

    Expected columns: an identifier column (``name``/``id``/``seq_name``) and a
    probability column (``probability``/``prob``/``AmyloGram_probability``).
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Matching is case-insensitive on purpose. The bundled helper script writes
    # 'name,probability', but AmyloGram's own predict.ag_model returns a data
    # frame with 'Name' and 'Probability', and a user who exports that directly
    # (or opens the CSV in a spreadsheet first) would otherwise hit a confusing
    # "expected an id column" error on a file that plainly has one.
    lowered = {str(c).lower(): c for c in df.columns}
    id_col = next(
        (lowered[c] for c in ("name", "id", "seq_name", "sequence_id") if c in lowered),
        None,
    )
    prob_col = next(
        (
            lowered[c]
            for c in ("probability", "prob", "amylogram_probability", "score")
            if c in lowered
        ),
        None,
    )
    if id_col is None or prob_col is None:
        raise ValueError(
            f"{path}: expected an id column and a probability column; got {list(df.columns)}"
        )
    return {
        str(name): float(prob)
        for name, prob in zip(df[id_col], df[prob_col], strict=True)
    }



def to_per_residue_frame(
    sequence: str,
    scores: list[float],
    *,
    protein_id: str = "",
) -> "pd.DataFrame":
    """Return the ``Number, Residue, Score`` table amyloscope adapters expect.

    AmyloGram is the only tool in the panel whose native output is per *peptide*
    rather than per residue, so the projection performed here is the point at
    which its evidence becomes commensurable with WALTZ, FoldAmyloid and the
    rest. Writing that table out explicitly — rather than leaving the projection
    implicit inside a consensus call — keeps the modelling step auditable: the
    file on disk is exactly what the consensus counted.
    """
    if len(scores) != len(sequence):
        raise ValueError(
            f"{protein_id or 'sequence'}: {len(scores)} projected scores for "
            f"{len(sequence)} residues"
        )
    return pd.DataFrame(
        {
            "Number": range(1, len(sequence) + 1),
            "Residue": list(sequence),
            "Score": [float(s) for s in scores],
        }
    )
