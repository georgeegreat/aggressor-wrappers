"""AGGRESCAN result-page parsing, against a verbatim 334-residue capture.

The Abeta42 fixture alone could not have caught either bug here: it is short
enough that the threshold rule and NHSA agree, and small enough that its markup
is well-formed. CTSV_Homo_sapiens is a real page as the server sent it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aggressor_wrappers.runners.aggrescan import (
    hot_spot_mask,
    hot_spot_runs,
    mask_runs,
    parse_result_page,
    parse_summary,
)

FIXTURES = Path(__file__).parent / "fixtures"
CTSV = FIXTURES / "aggrescan_result_ctsv.html"
ABETA = FIXTURES / "aggrescan_result_abeta42.html"

# AGGRESCAN's own Hot Spot calls for CTSV, read off the page's red highlighting
# and identical to NHSA > 0. nHS = 11, as the summary reports.
CTSV_HOT_SPOTS = [
    (1, 18), (75, 79), (138, 144), (182, 187), (205, 210),
    (223, 229), (238, 244), (259, 265), (277, 286), (298, 302), (326, 331),
]
CTSV_SUMMARY = {
    "a3vSA": -0.080, "nHS": 11.0, "NnHS": 3.293, "AAT": 29.748, "THSA": 25.432,
    "TA": -20.989, "AATr": 0.089, "THSAr": 0.076, "Na4vSS": -8.2,
}


@pytest.fixture(scope="module")
def ctsv_html() -> str:
    if not CTSV.is_file():
        pytest.skip("CTSV capture missing")
    return CTSV.read_text(encoding="utf-8", errors="replace")


def test_the_capture_really_is_malformed(ctsv_html):
    """Guards the premise: if this page were well-formed the anchor is moot.

    42 <small> opened, 29 closed, plus a literal <smal> typo. A parser that
    delimits on <small> reads across cell boundaries here.
    """
    assert len(re.findall(r"<small\b", ctsv_html, re.I)) > len(
        re.findall(r"</small>", ctsv_html, re.I)
    )
    assert "<smal>" in ctsv_html


def test_header_row_anchor_survives_the_malformed_markup(ctsv_html):
    records, reported = parse_result_page(ctsv_html)
    assert len(records) == 334
    assert reported == 11
    assert records[0] == {
        "Number": 1, "AA": "M", "a4v": 0.464,
        "HSA": 10.203, "NHSA": 0.567, "a4vAHS": 0.533,
    }
    assert "".join(r["AA"] for r in records).startswith("MNLSLVLAAFCLGIASAVPK")


def test_nhsa_reproduces_the_servers_own_hot_spot_calls(ctsv_html):
    records, reported = parse_result_page(ctsv_html)
    runs = mask_runs(hot_spot_mask([r["NHSA"] for r in records]))
    assert runs == CTSV_HOT_SPOTS
    assert len(runs) == reported


def test_threshold_rule_overcalls_and_is_why_nhsa_is_used(ctsv_html):
    """Pin the disagreement, so 'NHSA is redundant' is falsifiable, not asserted.

    The threshold rule invents 105-111 and pushes three more regions 3-4
    residues past where AGGRESCAN ends them. It never under-calls, so the error
    is one-directional: it inflates AGGRESCAN's breadth and manufactures the
    shoulders that consensus clustering then has to adjudicate.
    """
    records, _ = parse_result_page(ctsv_html)
    rule = hot_spot_runs([r["a4v"] for r in records])
    assert len(rule) == 12 and len(CTSV_HOT_SPOTS) == 11
    assert (105, 111) in rule

    def residues(runs):
        return {p for start, stop in runs for p in range(start, stop + 1)}

    server, derived = residues(CTSV_HOT_SPOTS), residues(rule)
    assert server < derived, "the rule should only ever add residues"
    assert len(derived - server) == 17
    assert len(server) / len(records) == pytest.approx(0.251, abs=0.001)
    assert len(derived) / len(records) == pytest.approx(0.302, abs=0.001)


def test_summary_pairs_labels_with_the_other_column(ctsv_html):
    """The two-column layout is why naive adjacency read nHS as 100.

    "Number of Hot Spots (nHS):" is followed in the SOURCE by the next label,
    "Normalized nHS for 100 residues", not by its own value.
    """
    assert parse_summary(ctsv_html) == pytest.approx(CTSV_SUMMARY)
    flat = re.sub(r"<[^>]+>", " ", ctsv_html)
    naive = re.search(r"Number of Hot Spots[^<]*\(nHS\)\s*:?.*?([\d.]+)", flat, re.S)
    assert naive is not None and float(naive.group(1)) == 100.0


def test_abeta42_is_unchanged_by_the_rewrite():
    """Same numbers, same calls -- the known validation case must not move."""
    records, reported = parse_result_page(
        ABETA.read_text(encoding="utf-8", errors="replace")
    )
    assert len(records) == 42
    assert "".join(r["AA"] for r in records) == (
        "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
    )
    runs = mask_runs(hot_spot_mask([r["NHSA"] for r in records]))
    assert runs == [(17, 22), (30, 42)]
    assert reported == 2


def test_y10_shows_why_hsa_is_not_the_mask():
    """Abeta42's Y10 has area assigned but is not a Hot Spot.

    HSA and a4vAHS are both non-zero at Y10 because the shared area is computed
    before the run-length requirement is applied. Only NHSA reflects the call.
    """
    records, _ = parse_result_page(ABETA.read_text(encoding="utf-8", errors="replace"))
    y10 = records[9]
    assert y10["AA"] == "Y"
    assert y10["HSA"] > 0 and y10["a4vAHS"] > 0
    assert y10["NHSA"] == 0


def test_an_error_page_is_reported_as_such_not_as_a_layout_change():
    with pytest.raises(RuntimeError, match="error page"):
        parse_result_page("<html><head><title>404 Not Found</title></head><body/></html>")
