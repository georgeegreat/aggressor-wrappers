"""AGGRESCAN: page scraping, and the Hot Spot rule the paper actually defines."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.runners.aggrescan import (
    HOT_SPOT_THRESHOLD,
    MIN_HOT_SPOT_RUN,
    AggrescanRunner,
    hot_spot_runs,
    parse_result_page,
)
from aggressor_wrappers.runners.registry import get_runner

FIXTURES = Path(__file__).parent / "fixtures"
PAGE = FIXTURES / "aggrescan_result_abeta42.html"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


@pytest.mark.skipif(not PAGE.is_file(), reason="aggrescan page fixture missing")
def test_column_major_blocks_are_recovered_in_order() -> None:
    records, reported = parse_result_page(PAGE.read_text())
    assert len(records) == len(ABETA42)
    assert "".join(r["AA"] for r in records) == ABETA42
    assert reported == 2
    assert records[18]["a4v"] == pytest.approx(1.289)  # F19, profile maximum


@pytest.mark.skipif(not PAGE.is_file(), reason="aggrescan page fixture missing")
def test_hsa_over_zero_overcalls_relative_to_the_servers_own_count() -> None:
    """The decisive test: HSA > 0 and AGGRESCAN's nHS disagree.

    An isolated Y10 clears the threshold but not the 5-residue run requirement.
    AGGRESCAN still assigns it a shared area, so a mask built from HSA alone
    reports three Hot Spots where the server reports two.
    """
    records, reported = parse_result_page(PAGE.read_text())
    by_hsa = [r["Number"] for r in records if r["HSA"] > 0]
    runs_from_hsa = []
    for position in by_hsa:
        if runs_from_hsa and position == runs_from_hsa[-1][1] + 1:
            runs_from_hsa[-1][1] = position
        else:
            runs_from_hsa.append([position, position])
    assert len(runs_from_hsa) == 3            # 10 | 17-22 | 30-42
    assert runs_from_hsa[0] == [10, 10]

    runs = hot_spot_runs([r["a4v"] for r in records])
    assert len(runs) == reported == 2
    assert runs == [(17, 22), (30, 42)]
    assert "".join(ABETA42[a - 1 : b] for a, b in runs) == "LVFFAE" + "AIIGLMVGGVVIA"


def test_published_criteria_are_the_defaults() -> None:
    assert HOT_SPOT_THRESHOLD == pytest.approx(-0.02)
    assert MIN_HOT_SPOT_RUN == 5


def test_run_shorter_than_the_minimum_is_not_a_hot_spot() -> None:
    profile = [-1.0] * 3 + [0.5] * 4 + [-1.0] * 3      # a 4-residue run
    assert hot_spot_runs(profile) == []
    profile = [-1.0] * 3 + [0.5] * 5 + [-1.0] * 3      # a 5-residue run
    assert hot_spot_runs(profile) == [(4, 8)]


def test_runner_registered_and_web_only() -> None:
    runner = get_runner("aggrescan")
    assert isinstance(runner, AggrescanRunner)
    assert runner.verify_reported_nhs is True
    assert "andromeda.uab.cat" in runner.base_url
