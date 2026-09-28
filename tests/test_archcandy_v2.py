"""ArchCandy 2.0: REST contract, projection modes, and score calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.predictors.archcandy_v2 import ArchCandy2Parser
from aggressor_wrappers.runners.archcandy_v2 import (
    SIGNIFICANCE_THRESHOLD,
    WEB_DEFAULT_THRESHOLD,
    ArchCandy2Runner,
    significance_band,
)
from aggressor_wrappers.runners.registry import get_runner

FIXTURE = Path(__file__).parent / "fixtures" / "archcandy_v2_abeta42.csv"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def test_calibration_bands_match_the_published_documentation() -> None:
    """ArchCandy 2.0 documents a three-band scale, not one cutoff."""
    assert WEB_DEFAULT_THRESHOLD == pytest.approx(0.40)
    assert SIGNIFICANCE_THRESHOLD == pytest.approx(0.57)
    assert significance_band(0.24) == "non_significant"
    assert significance_band(0.40) == "ambiguous"
    assert significance_band(0.56) == "ambiguous"
    assert significance_band(0.57) == "significant"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="archcandy2 fixture missing")
def test_best_arch_spans_both_abeta_hydrophobic_segments() -> None:
    """The top-scoring arch is the beta-arch reported for Abeta42 fibrils.

    15-36 QKLVFFAEDVGSNKGAIIGLMV carries KLVFFA and IGLMV as the two strands
    with the bend between them -- the arrangement a beta-arcade predictor
    should find, and the one that makes this a usable regression case.
    """
    result = ArchCandy2Parser().parse(FIXTURE, protein_id="ABETA42", sequence=ABETA42)
    best = max(result.regions, key=lambda r: r["score"])
    assert (best["start"], best["stop"]) == (15, 36)
    assert best["score"] == pytest.approx(0.723)
    assert best["segment"].startswith("QKLVFFA")
    assert significance_band(best["score"]) == "significant"
    # Topology survives the projection: it has no per-residue representation.
    assert {r["arch"] for r in result.regions} >= {"GBPL", "BLLPBL", "BEPL"}


@pytest.mark.skipif(not FIXTURE.is_file(), reason="archcandy2 fixture missing")
def test_cumulative_leaves_archcandys_own_scale() -> None:
    """Summing overlapping arches produces a number no threshold applies to."""
    highest = ArchCandy2Parser(score_mode="highest").parse(
        FIXTURE, protein_id="ABETA42", sequence=ABETA42
    )
    cumulative = ArchCandy2Parser(score_mode="cumulative").parse(
        FIXTURE, protein_id="ABETA42", sequence=ABETA42
    )
    assert max(highest.scores) == pytest.approx(0.723)
    assert max(highest.scores) <= 1.0
    assert max(cumulative.scores) > 6.0  # 6.681 on this capture
    # arch_count is what cumulative is actually measuring: overlap depth.
    assert max(cumulative.aux["arch_count"]) == 12


@pytest.mark.skipif(not FIXTURE.is_file(), reason="archcandy2 fixture missing")
def test_raising_the_threshold_from_web_default_to_significant_shrinks_calls() -> None:
    at_web = ArchCandy2Parser(threshold=WEB_DEFAULT_THRESHOLD).parse(
        FIXTURE, protein_id="ABETA42", sequence=ABETA42
    )
    at_sig = ArchCandy2Parser(threshold=SIGNIFICANCE_THRESHOLD).parse(
        FIXTURE, protein_id="ABETA42", sequence=ABETA42
    )
    assert sum(at_web.binary) == 41
    assert sum(at_sig.binary) == 30
    assert sum(at_sig.binary) < sum(at_web.binary)


def test_one_point_zero_layout_is_rejected_with_a_pointer(tmp_path: Path) -> None:
    """A standalone-1.0 CSV must not be silently misread as 2.0 output."""
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("Number,Digram,Score,Arc_type,Position\n1,XX,0.31,a,02-23\n")
    with pytest.raises(ValueError, match="standalone 1.0 layout"):
        ArchCandy2Parser().parse(legacy, protein_id="X", sequence="A" * 30)


def test_registry_web_arm_is_version_two() -> None:
    """'archcandy' must resolve to the 2.0 API, not the retired 1.x route."""
    runner = get_runner("archcandy")
    assert isinstance(runner, ArchCandy2Runner)
    assert runner.threshold == pytest.approx(0.57)
    assert not hasattr(runner, "verify_ssl")


# --------------------------------------------------------------------------- #
# ArchCandy is two programs behind one config section: the web service is 2.0,
# the downloadable JAR is 1.0 with a published 0.56 cutoff. `local_threshold`
# held the second value and nothing read it, so the local backend would have
# run at the web threshold. `backend = auto` finds no JAR on the analysis
# machine and resolves to web, which is why this never surfaced in a run.
# --------------------------------------------------------------------------- #


def test_local_threshold_reaches_the_local_backend(tmp_path):
    import warnings

    from aggressor_wrappers.runners.archcandy_local import ArchCandyLocalRunner
    from aggressor_wrappers.runners.registry import get_runner

    jar = tmp_path / "ArchCandy.jar"
    jar.write_bytes(b"")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner = get_runner(
            "archcandy", backend="local", jar_path=jar, threshold=0.57, local_threshold=0.56
        )
    assert isinstance(runner, ArchCandyLocalRunner)
    assert runner.threshold == 0.56
    assert not [w for w in caught if "local_threshold" in str(w.message)]


def test_local_threshold_does_not_leak_into_the_web_backend():
    import warnings

    from aggressor_wrappers.runners.archcandy_v2 import ArchCandy2Runner
    from aggressor_wrappers.runners.registry import get_runner

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner = get_runner(
            "archcandy", backend="web", threshold=0.57, local_threshold=0.56
        )
    assert isinstance(runner, ArchCandy2Runner)
    assert runner.threshold == 0.57
    assert not [w for w in caught if "local_threshold" in str(w.message)]
