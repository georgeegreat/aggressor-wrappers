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

``window = 6``
    AmyloGram's training unit. Windows are taken with step 1, so a residue is
    typically covered by up to six of them.
``aggregation = "max"``
    A residue takes the highest probability among the windows covering it. The
    alternative, ``mean``, smooths the profile but dilutes a single strongly
    amyloidogenic hexapeptide against its weaker neighbours — the same argument
    that makes ``highest`` preferable to ``cumulative`` for ArchCandy.
``threshold = 0.5``
    The model returns a probability, so 0.5 is the natural cut; it is exposed
    because the operating point should be chosen against whatever validation set
    the panel is tuned on, not assumed.

Sequences shorter than the window are scored as a single peptide covering the
whole sequence, rather than dropped: a 5-residue peptide is still a legitimate
query for a hexapeptide model, and silently returning an empty profile would
remove the tool from the consensus without saying so.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_WINDOW = 6
DEFAULT_AGGREGATION = "max"


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


def project_windows(
    windows: list[tuple[int, str]],
    probabilities: dict[str, float] | list[float],
    sequence_length: int,
    *,
    aggregation: str = DEFAULT_AGGREGATION,
    prefix: str = "w",
) -> list[float]:
    """Map per-window probabilities onto per-residue scores."""
    if aggregation not in ("max", "mean"):
        raise ValueError("aggregation must be 'max' or 'mean'")

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

    totals = [0.0] * sequence_length
    counts = [0] * sequence_length
    for (start, peptide), prob in zip(windows, probs, strict=True):
        for offset in range(len(peptide)):
            idx = start - 1 + offset
            if 0 <= idx < sequence_length:
                if aggregation == "max":
                    totals[idx] = max(totals[idx], prob)
                else:
                    totals[idx] += prob
                counts[idx] += 1
    if aggregation == "mean":
        return [t / c if c else 0.0 for t, c in zip(totals, counts, strict=True)]
    return totals


def parse_amylogram_output(path: str | Path) -> dict[str, float]:
    """Read the CSV written by the AmyloGram helper script.

    Expected columns: an identifier column (``name``/``id``/``seq_name``) and a
    probability column (``probability``/``prob``/``AmyloGram_probability``).
    """
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = next(
        (c for c in ("name", "id", "seq_name", "sequence_id") if c in df.columns), None
    )
    prob_col = next(
        (
            c
            for c in ("probability", "prob", "AmyloGram_probability", "Probability")
            if c in df.columns
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


R_SCRIPT = r"""#!/usr/bin/env Rscript
# Score peptides with AmyloGram and write id,probability as CSV.
# Deliberately minimal: the windowing and the projection back onto residues are
# done in Python, where they are testable and where the modelling choice is
# visible, rather than being buried in an R helper.
suppressMessages({
  library(AmyloGram)
  library(seqinr)
})
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: amylogram_predict.R <peptides.fasta> <out.csv>")
fasta_path <- args[[1]]
out_path   <- args[[2]]

seqs <- seqinr::read.fasta(fasta_path, seqtype = "AA", as.string = FALSE)
data(AmyloGram_model, package = "AmyloGram", envir = environment())
pred <- predict(AmyloGram_model, seqs)

prob <- if (is.data.frame(pred)) {
  col <- intersect(c("Probability", "probability", "prob"), colnames(pred))
  if (length(col) == 0) pred[[ncol(pred)]] else pred[[col[[1]]]]
} else as.numeric(pred)

write.csv(
  data.frame(name = names(seqs), probability = prob, stringsAsFactors = FALSE),
  out_path, row.names = FALSE, quote = FALSE
)
"""
