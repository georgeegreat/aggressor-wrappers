"""Thresholds and registry resolution: the four fixes that change what a tool calls.

Each test here pins a value that was wrong in a way no exception reported.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.core.schema import get_predictor_spec
from aggressor_wrappers.predictors.aggrescan import (
    PUBLISHED_A4V_THRESHOLD,
    AggrescanParser,
)
from aggressor_wrappers.predictors.appnn import APPNNParser
from aggressor_wrappers.predictors.archcandy import (
    PUBLISHED_THRESHOLD_1X,
    ArchCandyParser,
)
from aggressor_wrappers.predictors.archcandy_v2 import ArchCandy2Parser
from aggressor_wrappers.predictors.pasta import PASTAParser
from aggressor_wrappers.predictors.registry import get_parser


# --------------------------------------------------------------------------- #
# PASTA
# --------------------------------------------------------------------------- #
def test_pasta_default_comes_from_the_registry() -> None:
    """One published value, written down once."""
    assert PASTAParser().energy_threshold == pytest.approx(
        get_predictor_spec("pasta").default_threshold
    )
    assert PASTAParser().energy_threshold == pytest.approx(-5.0)


def test_pasta_explicit_threshold_still_wins() -> None:
    assert PASTAParser(energy_threshold=-3.0).energy_threshold == pytest.approx(-3.0)


def test_pasta_call_direction_is_inverted(tmp_path: Path) -> None:
    """More negative energy = more aggregation-prone. The only inverted tool."""
    path = tmp_path / "e.dat"
    path.write_text("\n".join(["-7.0", "-1.0", "-6.0", "-0.5"]))
    result = PASTAParser().parse(path, protein_id="t", sequence="MKLV")
    assert result.binary == [1, 0, 1, 0]


# --------------------------------------------------------------------------- #
# ArchCandy -- a predictor that could not say no
# --------------------------------------------------------------------------- #
def _arch_csv(path: Path, rows: list[tuple[int, int, float]]) -> Path:
    pd.DataFrame(rows, columns=["Start", "Stop", "Score"]).to_csv(path, index=False)
    return path


def test_archcandy_rejects_insignificant_arches(tmp_path: Path) -> None:
    """An arch below ArchCandy's own calibration no longer produces a call."""
    path = _arch_csv(tmp_path / "a.csv", [(1, 4, 0.20)])
    result = ArchCandyParser().parse(path, protein_id="t", sequence="MKLVFFAE")
    assert sum(result.binary) == 0


def test_archcandy_accepts_significant_arches(tmp_path: Path) -> None:
    path = _arch_csv(tmp_path / "a.csv", [(1, 4, 0.80)])
    result = ArchCandyParser().parse(path, protein_id="t", sequence="MKLVFFAE")
    assert result.binary[:4] == [1, 1, 1, 1]
    assert sum(result.binary) == 4


def test_archcandy_threshold_uses_best_arch_not_the_sum(tmp_path: Path) -> None:
    """Three weak arches must not add up to a call.

    Under score_mode='cumulative' the per-residue score is a sum and can exceed
    1.0, so thresholding it would call residues lying under several weak arches
    and no strong one -- confidence conflated with multiplicity.
    """
    path = _arch_csv(tmp_path / "a.csv", [(1, 4, 0.25), (1, 4, 0.25), (1, 4, 0.25)])
    result = ArchCandyParser(score_mode="cumulative").parse(
        path, protein_id="t", sequence="MKLVFFAE"
    )
    assert result.scores[0] == pytest.approx(0.75)  # the sum is still reported
    assert sum(result.binary) == 0                  # but nothing is called


def test_archcandy_threshold_can_be_disabled(tmp_path: Path) -> None:
    path = _arch_csv(tmp_path / "a.csv", [(1, 4, 0.20)])
    result = ArchCandyParser(threshold=None).parse(
        path, protein_id="t", sequence="MKLVFFAE"
    )
    assert sum(result.binary) == 4


def test_archcandy_default_is_the_1x_published_value() -> None:
    assert ArchCandyParser().threshold == pytest.approx(PUBLISHED_THRESHOLD_1X)
    assert PUBLISHED_THRESHOLD_1X == pytest.approx(0.56)


# --------------------------------------------------------------------------- #
# Registry resolution
# --------------------------------------------------------------------------- #
def test_archcandy2_is_reachable_through_get_parser() -> None:
    """The alias used to shadow the versioned parser entirely.

    `resolve_predictor_key('archcandy2')` returns 'archcandy', so the exact
    registry key was never consulted and asking for the 2.0 parser returned the
    1.x one -- which at the time applied no threshold at all.
    """
    assert isinstance(get_parser("archcandy2"), ArchCandy2Parser)
    assert isinstance(get_parser("archcandy"), ArchCandyParser)


def test_unknown_parser_names_the_alternatives() -> None:
    """An unknown name is refused by `resolve_predictor_key`, which lists the
    known predictors. Note its list is PREDICTOR_REGISTRY (specs), which is a
    superset of PARSER_REGISTRY: `tango` has a spec and no parser.
    """
    with pytest.raises(KeyError, match="Known:"):
        get_parser("not_a_predictor")


# --------------------------------------------------------------------------- #
# APPNN -- a constructor that could not be constructed
# --------------------------------------------------------------------------- #
def test_appnn_accepts_its_config_key() -> None:
    """`score_threshold` is a [predictors.appnn] key and get_parser passes it.

    Without an __init__ accepting it, `get_parser('appnn')` raised
    `TypeError: APPNNParser() takes no arguments`, so the reparse / --skip-run
    branch of the pipeline could not reparse cached APPNN output at all.
    """
    parser = APPNNParser(score_threshold=0.7)
    assert parser.score_threshold == pytest.approx(0.7)
    assert isinstance(get_parser("appnn"), APPNNParser)


def test_appnn_parse_argument_overrides_the_constructor(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    pd.DataFrame(
        {
            "aminoacid_position": [1, 2, 3, 4],
            "aminoacid": list("MKLV"),
            "aminoacid_score": [0.6, 0.6, 0.1, 0.1],
        }
    ).to_csv(path, index=False)

    default = APPNNParser(score_threshold=0.5).parse(
        path, protein_id="t", sequence="MKLV"
    )
    assert sum(default.binary) == 2

    overridden = APPNNParser(score_threshold=0.5).parse(
        path, protein_id="t", sequence="MKLV", score_threshold=0.9
    )
    assert sum(overridden.binary) == 0


# --------------------------------------------------------------------------- #
# AGGRESCAN
# --------------------------------------------------------------------------- #
def test_aggrescan_fallback_is_the_calibrated_value() -> None:
    """-0.02 (Conchillo-Sole et al. 2007), not a round 0.0."""
    assert AggrescanParser().threshold == pytest.approx(PUBLISHED_A4V_THRESHOLD)
    assert PUBLISHED_A4V_THRESHOLD == pytest.approx(-0.02)
