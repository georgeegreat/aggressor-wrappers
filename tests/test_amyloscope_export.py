"""Bridge from merged predictor tables to amyloscope inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggressor_wrappers.core.amyloscope_export import export_run


def _wide() -> pd.DataFrame:
    hot = np.array([False] * 10 + [True] * 10)
    df = pd.DataFrame(
        {
            "waltz_score": np.where(hot, 90.0, 20.0),
            "APPNN_score": np.where(hot, 0.9, 0.1),
            "pasta_score": np.where(hot, -7.0, -1.0),
        }
    )
    df["waltz_bin"] = (df.waltz_score >= 73).astype(int)
    df["APPNN_bin"] = (df.APPNN_score >= 0.5).astype(int)
    df["pasta_bin"] = (df.pasta_score < -2.8).astype(int)
    return df

# --------------------------------------------------------------------------- #
# amyloscope bridge
# --------------------------------------------------------------------------- #


def test_export_preserves_pasta_polarity(tmp_path: Path):
    """The exported APR flag must be true where PASTA is most negative."""
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST.csv", index=False)

    out = tmp_path / "amylo"
    config = export_run(merged, out)
    assert config.exists()

    track = pd.read_csv(out / "tracks" / "pasta" / "TEST.csv")
    assert list(track.columns) == ["Number", "Residue", "Score", "APR"]
    apr = track["APR"].astype(bool).to_numpy()
    # residues 11-20 carry the strong (negative) PASTA signal
    assert apr[10:].all()
    assert not apr[:10].any()


def test_generated_config_uses_flag_detection_and_available_denominator(tmp_path: Path):
    merged = tmp_path / "merged"
    merged.mkdir()
    df = _wide()
    df.insert(0, "aa_name", list("ACDEFGHIKLMNPQRSTVWY"))
    df.insert(0, "position", range(1, 21))
    df.to_csv(merged / "TEST.csv", index=False)

    text = export_run(merged, tmp_path / "amylo").read_text()
    # thresholds/polarity are NOT restated downstream: one source of truth
    assert "method: flag, column: APR" in text
    # a dead predictor must not make a tier unreachable
    assert "denominator: available" in text


