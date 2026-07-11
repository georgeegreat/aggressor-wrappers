"""Wrapper fidelity: ArchCandy's beta-arch topology and score scale."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# --------------------------------------------------------------------------- #
# Wrapper fidelity: does each instrument's distinctive output survive?
# --------------------------------------------------------------------------- #


def _archcandy_csv(tmp_path: Path, scores=(0.15, 0.15, 0.15)) -> Path:
    df = pd.DataFrame(
        {
            "ID": ["P"] * 3,
            "Sequence": ["AAAA"] * 3,
            "Arch": ["GBPL", "BLLPBL", "GBPL"],
            "Start": [5, 8, 11],
            "Stop": [14, 17, 20],
            "Score": list(scores),
        }
    )
    path = tmp_path / "arch.csv"
    df.to_csv(path, index=False)
    return path


def test_archcandy_retains_beta_arch_topology(tmp_path: Path):
    """The arch topology string is ArchCandy's distinctive output; keep it."""
    from aggressor_wrappers.predictors.archcandy import ArchCandyParser

    res = ArchCandyParser(score_mode="highest").parse(
        _archcandy_csv(tmp_path), protein_id="P", sequence="A" * 30
    )
    regions = res.regions_dataframe()
    assert list(regions["arch"]) == ["GBPL", "BLLPBL", "GBPL"]
    assert list(regions["start"]) == [5, 8, 11]
    # per-residue best-arch channel
    assert res.aux["arch"][11] == "GBPL"


def test_archcandy_cumulative_mode_warns_when_it_inflates_the_score(tmp_path: Path):
    """Summing overlapping arches leaves ArchCandy's [0,1] confidence scale."""
    from aggressor_wrappers.predictors.archcandy import ArchCandyParser

    with pytest.warns(UserWarning, match="inflated a per-residue score"):
        res = ArchCandyParser(score_mode="cumulative").parse(
            _archcandy_csv(tmp_path), protein_id="P", sequence="A" * 30
        )
    # three arches of 0.15 sum to 0.45 and clear ArchCandy's own 0.4 threshold,
    # which no single arch clears
    assert max(res.scores) == pytest.approx(0.45)


def test_archcandy_highest_mode_stays_on_native_scale(tmp_path: Path):
    from aggressor_wrappers.predictors.archcandy import ArchCandyParser

    res = ArchCandyParser(score_mode="highest").parse(
        _archcandy_csv(tmp_path), protein_id="P", sequence="A" * 30
    )
    assert max(res.scores) == pytest.approx(0.15)


