"""Bridge from merged predictor tables to amyloscope inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aggressor_wrappers.core.amyloscope_export import (
    export_protein,
    export_run,
    parse_aggressor_track,
    register_amyloscope_adapter,
)


def _wide() -> pd.DataFrame:
    hot = np.array([False] * 10 + [True] * 10)
    df = pd.DataFrame(
        {
            "waltz_score": np.where(hot, 90.0, 20.0),
            "APPNN_score": np.where(hot, 0.9, 0.1),
            # PASTA: inverted — lower (more negative) is more amyloidogenic
            "pasta_score": np.where(hot, -7.0, -1.0),
        }
    )
    df["waltz_bin"] = (df.waltz_score >= 73).astype(int)
    df["APPNN_bin"] = (df.APPNN_score >= 0.5).astype(int)
    df["pasta_bin"] = (df.pasta_score < -2.8).astype(int)
    return df


def test_export_preserves_pasta_polarity(tmp_path: Path) -> None:
    """The exported APR flag must be true where PASTA is most negative."""
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST_merged.csv", index=False)

    out = tmp_path / "amylo"
    config = export_run(merged, out)
    assert config.exists()

    track = pd.read_csv(out / "tracks" / "pasta" / "TEST.csv")
    assert list(track.columns) == ["Number", "Residue", "Score", "APR"]
    apr = track["APR"].astype(bool).to_numpy()
    assert apr[10:].all()
    assert not apr[:10].any()


def test_generated_config_uses_flag_detection_and_available_denominator(tmp_path: Path) -> None:
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST_merged.csv", index=False)

    text = export_run(merged, tmp_path / "amylo").read_text()
    assert "method: flag, column: APR" in text
    assert "denominator: available" in text


def test_parse_aggressor_track_roundtrip(tmp_path: Path) -> None:
    track_path = tmp_path / "waltz.csv"
    pd.DataFrame(
        {
            "Number": [1, 2],
            "Residue": ["M", "A"],
            "Score": [0.5, 0.9],
            "APR": [False, True],
        }
    ).to_csv(track_path, index=False)

    parsed = parse_aggressor_track(track_path)
    assert parsed["APR"].tolist() == [False, True]


def test_register_amyloscope_adapter_when_installed() -> None:
    amyloscope = pytest.importorskip("amyloscope")
    del amyloscope  # registration is the assertion target
    assert register_amyloscope_adapter() is True
    from amyloscope.io.adapters import available_adapters

    assert "aggressor" in available_adapters()


def test_amyloscope_run_on_exported_panel(tmp_path: Path) -> None:
    pytest.importorskip("amyloscope")
    from amyloscope import run_from_file

    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "RPL27_merged.csv", index=False)

    out = tmp_path / "amylo"
    config_path = export_run(merged, out, name="test panel")
    assert register_amyloscope_adapter() is True

    cfg = yaml.safe_load(config_path.read_text())
    assert cfg["proteins"][0]["id"] == "RPL27"
    assert (out / "tracks" / "waltz" / "RPL27.csv").is_file()

    artifacts = run_from_file(config_path)
    regions = artifacts.consensus.all_regions()
    assert regions, "expected at least one consensus region from synthetic hot stretch"

    export_protein(merged / "RPL27_merged.csv", "RPL27", out)
    assert parse_aggressor_track(out / "tracks" / "pasta" / "RPL27.csv")["APR"].astype(bool).any()
