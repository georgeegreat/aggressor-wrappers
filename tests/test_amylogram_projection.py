"""How the hexapeptide -> residue projection distorts APR boundaries.

AmyloGram scores peptides, not residues, so the projection is a modelling step
with a measurable bias. These tests pin that bias with a generator that states
the ground truth: a window scores high inside the true APR, low outside, and in
proportion to overlap when it straddles the edge. Any error in the recovered
extent is therefore attributable to the aggregation rule alone.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.predictors.amylogram import (
    DEFAULT_AGGREGATION,
    DEFAULT_SUPPORT_FRACTION,
    coverage_depth,
    project_windows,
    sliding_windows,
)

WINDOW, HIGH, LOW, THRESHOLD = 6, 0.90, 0.10, 0.5


def recovered_extent(length, apr_start, apr_stop, rule, fraction=0.5):
    windows = sliding_windows("A" * length, WINDOW)
    probs = []
    for start, peptide in windows:
        lo, hi = start, start + len(peptide) - 1
        overlap = max(0, min(hi, apr_stop) - max(lo, apr_start) + 1)
        probs.append(LOW + (HIGH - LOW) * (overlap / len(peptide)))
    scores = project_windows(
        windows, probs, length, aggregation=rule, support_fraction=fraction
    )
    called = [i + 1 for i, s in enumerate(scores) if s >= THRESHOLD]
    return (called[0], called[-1]) if called else None


# (label, rule, fraction, expected N-terminal error, expected C-terminal error)
@pytest.mark.parametrize(
    "rule,fraction,err_n,err_c",
    [
        ("max", 1.0, -3, +3),
        ("median", 1.0, 0, 0),
        ("mean", 1.0, 0, 0),
        ("min", 1.0, +2, -2),
        ("support", 1 / WINDOW, -3, +3),   # f = 1/w reproduces max
        ("support", 0.5, -1, +1),
        ("support", 1.0, +2, -2),          # f = 1 reproduces min
    ],
)
@pytest.mark.parametrize(
    "length,start,stop",
    [
        (60, 28, 33),    # hexapeptide-scale core, as in Abeta KLVFFA
        (140, 61, 95),   # 35-residue core, as in the alpha-synuclein NAC region
    ],
)
def test_boundary_error_is_a_property_of_the_rule_not_the_apr_length(
    rule, fraction, err_n, err_c, length, start, stop
):
    """The distortion is additive in residues, so its RELATIVE cost is worst
    for short APRs: `max` doubles a 6-residue core but inflates a 35-residue
    core by only about a sixth."""
    got = recovered_extent(length, start, stop, rule, fraction)
    assert got is not None
    assert (got[0] - start, got[1] - stop) == (err_n, err_c)


def test_support_fraction_spans_max_to_min_monotonically():
    """Raising support_fraction can only shrink or preserve a called region."""
    extents = []
    for fraction in (1 / WINDOW, 0.34, 0.5, 0.67, 0.84, 1.0):
        got = recovered_extent(140, 61, 95, "support", fraction)
        extents.append(got[1] - got[0] + 1)
    assert extents == sorted(extents, reverse=True)


def test_default_is_distribution_aware_not_an_extreme():
    assert DEFAULT_AGGREGATION == "support"
    assert DEFAULT_SUPPORT_FRACTION == 0.5
    wide = recovered_extent(140, 61, 95, "max")
    default = recovered_extent(140, 61, 95, DEFAULT_AGGREGATION, DEFAULT_SUPPORT_FRACTION)
    narrow = recovered_extent(140, 61, 95, "min")
    assert (wide[1] - wide[0]) > (default[1] - default[0]) > (narrow[1] - narrow[0])


def test_coverage_depth_is_reduced_at_the_termini():
    """Every rule but `max` is evaluated on a smaller sample near the ends."""
    depth = coverage_depth(10, 6)
    assert depth == [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    assert max(depth) == 5  # a 10-residue chain never reaches full 6-fold cover
    assert coverage_depth(20, 6)[9] == 6


def test_unknown_rule_and_bad_fraction_are_rejected():
    windows = sliding_windows("A" * 20, 6)
    probs = [0.5] * len(windows)
    with pytest.raises(ValueError, match="aggregation must be one of"):
        project_windows(windows, probs, 20, aggregation="cumulative")
    with pytest.raises(ValueError, match="support_fraction"):
        project_windows(windows, probs, 20, aggregation="support", support_fraction=0.0)
