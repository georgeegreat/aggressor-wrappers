"""Dependency compatibility shim, and the local Cross-Beta parser."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pytest

from aggressor_wrappers.core.compat import (
    BOOTSTRAP_SOURCE,
    patch_jax_unirep_clip,
    write_amylodeep_bootstrap,
)
from aggressor_wrappers.predictors.crossbeta_local import crossbeta_profile

SEQ = "MKTFFFLLLLFTIGFCYVQF"


def _crossbeta_csv(tmp_path: Path, *, regions="[[9, 14]]", name="RPL27") -> Path:
    aa = [{a: (0.8 if 8 <= i < 14 else 0.2)} for i, a in enumerate(SEQ)]
    path = tmp_path / "cb.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(
            [
                "Query_name",
                "Sequence_length",
                "Average_protein_prediction",
                "AR_position",
                "Amino_acids_score",
            ]
        )
        w.writerow([name, len(SEQ), 0.42, regions, str(aa)])
    return path


# --------------------------------------------------------------------------- #
# compat shim
# --------------------------------------------------------------------------- #


def test_shim_is_idempotent():
    """Repeated calls must not re-patch or re-warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch_jax_unirep_clip()
        assert patch_jax_unirep_clip() is False


def test_bootstrap_is_self_contained(tmp_path):
    """It runs in AmyloDeep's interpreter, which cannot import this package."""
    path = write_amylodeep_bootstrap(tmp_path / "run.py")
    source = path.read_text()
    assert "import aggressor_wrappers" not in source
    assert "from aggressor_wrappers" not in source
    assert "jax_unirep" in BOOTSTRAP_SOURCE


# --------------------------------------------------------------------------- #
# Cross-Beta local
# --------------------------------------------------------------------------- #


def test_crossbeta_parses_semicolon_and_dict_scores(tmp_path):
    scores, regions, meta = crossbeta_profile(_crossbeta_csv(tmp_path), SEQ)
    assert len(scores) == len(SEQ)
    assert regions == [(9, 14)]
    assert meta["average_protein_prediction"] == pytest.approx(0.42)


def test_crossbeta_rejects_mismatched_sequence(tmp_path):
    """The per-residue dicts carry residue identity; use it as an integrity check."""
    with pytest.raises(ValueError, match="does not match"):
        crossbeta_profile(_crossbeta_csv(tmp_path), "MKTFFF")


def test_crossbeta_no_regions_is_not_an_error(tmp_path):
    """'No amyloidogenic region' is a result, not a failure."""
    _, regions, _ = crossbeta_profile(_crossbeta_csv(tmp_path, regions="None"), SEQ)
    assert regions == []


def test_crossbeta_rejects_wrong_file(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a;b\n1;2\n")
    with pytest.raises(ValueError, match="not a Cross-Beta"):
        crossbeta_profile(path, SEQ)
