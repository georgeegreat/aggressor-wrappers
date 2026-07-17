"""Wrapper fidelity: AggreProt's auxiliary per-residue channels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# --------------------------------------------------------------------------- #
# Wrapper fidelity: does each instrument's distinctive output survive?
# --------------------------------------------------------------------------- #


def _aggreprot_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "position": range(1, 31),
            "struct_position": range(1, 31),
            "amino_acid": list("A" * 30),
            "aggregation": [0.3] * 30,
            "sasa": [12.5] * 30,
            "transmembrane": [0.0] * 30,
        }
    )
    path = tmp_path / "aggreprot.csv"
    with path.open("w") as fh:
        fh.write("Protein 1,P,,,,\n")
        df.to_csv(fh, index=False)
    return path


def test_aggreprot_retains_sasa_and_transmembrane(tmp_path: Path):
    """SASA (burial) is not redundant with the aggregation score — keep it."""
    from aggressor_wrappers.predictors.aggreprot import AggreProtParser

    res = AggreProtParser().parse(
        _aggreprot_csv(tmp_path), protein_id="P", sequence="A" * 30
    )
    assert "sasa" in res.aux
    assert "transmembrane" in res.aux
    assert res.aux["sasa"][0] == pytest.approx(12.5)


def test_default_table_is_byte_compatible_but_aux_is_opt_in(tmp_path: Path):
    """Existing outputs must not change; aux columns appear only on request."""
    from aggressor_wrappers.predictors.aggreprot import AggreProtParser

    res = AggreProtParser().parse(
        _aggreprot_csv(tmp_path), protein_id="P", sequence="A" * 30
    )
    assert list(res.to_dataframe().columns) == [
        "position",
        "aa_name",
        "aggreprot_score",
        "aggreprot_bin",
    ]
    assert "aggreprot_sasa" in res.to_dataframe(include_aux=True).columns


def test_aux_length_is_validated():
    """A mis-sized aux column is a bug and must fail loudly, not silently align."""
    from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec

    with pytest.raises(ValueError, match="aux column"):
        PredictorResult(
            protein_id="P",
            sequence="AAA",
            spec=get_predictor_spec("aggreprot"),
            scores=[0.0, 0.0, 0.0],
            binary=[0, 0, 0],
            aux={"sasa": [1.0, 2.0]},  # wrong length
        )
