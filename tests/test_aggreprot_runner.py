"""AggreProt runner and batch integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.predictors.aggreprot import AggreProtParser
from aggressor_wrappers.runners.aggreprot import (
    AggreProtRunner,
    normalise_aggreprot_accession,
    split_aggreprot_csv,
)
from aggressor_wrappers.runners.registry import get_runner, list_runners

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_CSV = FIXTURES / "aggreprot_single.csv"
MULTI_CSV = FIXTURES / "aggreprot_multi.csv"
BHT_RPL27 = Path(__file__).resolve().parents[2] / "BHT_amyloid/all/RPL27_human_all.csv"

RPL27_SEQ = (
    "MGKFMKPGKVVLVLAGRYSGRKAVIVKNIDDGTSDRPYSHALVAGIDRYPRKVTAAMGKK"
    "KIAKRSKIKSFVKVYNYNHLMPTRYSVDIPLDKTVVNKDVFRDPALKRKARREAKVKFEE"
    "RYKTGKNKWFFQKLRF"
)


def test_aggreprot_runner_registered() -> None:
    assert "aggreprot" in list_runners()


def test_get_runner_aggreprot_from_config() -> None:
    runner = get_runner("aggreprot")
    assert isinstance(runner, AggreProtRunner)
    assert runner.aggregation_threshold == pytest.approx(0.25)
    assert runner.poll_interval_seconds == 3


def test_normalise_aggreprot_accession() -> None:
    assert normalise_aggreprot_accession("RPL27_human") == "RPL27_human"
    assert normalise_aggreprot_accession("sp|P12345|NAME extra") == "P12345"


def test_split_single_protein_csv(tmp_path: Path) -> None:
    if not SINGLE_CSV.is_file():
        pytest.skip("aggreprot single CSV fixture missing")
    text = SINGLE_CSV.read_text()
    mapping = split_aggreprot_csv(text, ["RPL27_human"], tmp_path)
    assert set(mapping) == {"RPL27_human"}
    assert mapping["RPL27_human"].read_text() == text if text.endswith("\n") else text + "\n"


def test_split_multiprotein_csv(tmp_path: Path) -> None:
    if not MULTI_CSV.is_file():
        pytest.skip("aggreprot multi CSV fixture missing")
    mapping = split_aggreprot_csv(
        MULTI_CSV.read_text(),
        ["LINB", "LINB_1"],
        tmp_path,
    )
    assert set(mapping) == {"LINB", "LINB_1"}
    single = mapping["LINB"].read_text()
    assert single.startswith("Protein 1,LINB,,,,\n")
    assert "position,struct_position,amino_acid,aggregation,sasa,transmembrane\n" in single


def test_split_csv_rejects_accession_mismatch(tmp_path: Path) -> None:
    if not SINGLE_CSV.is_file():
        pytest.skip("aggreprot single CSV fixture missing")
    with pytest.raises(ValueError, match="does not match FASTA"):
        split_aggreprot_csv(SINGLE_CSV.read_text(), ["WRONG_ID"], tmp_path)


def test_aggreprot_parser_reads_export_fixture() -> None:
    if not SINGLE_CSV.is_file():
        pytest.skip("aggreprot single CSV fixture missing")
    result = AggreProtParser(aggregation_threshold=0.24).parse(
        SINGLE_CSV,
        protein_id="RPL27_human",
        sequence=RPL27_SEQ,
    )
    assert len(result.scores) == len(RPL27_SEQ)
    assert result.scores[0] == pytest.approx(0.08343360424041743, rel=1e-6)
    assert result.binary[8] == 1  # 0.0642 >= 0.24


@pytest.mark.skipif(not BHT_RPL27.is_file(), reason="BHT RPL27 reference table missing")
def test_rpl27_bins_match_bht_reference_at_024() -> None:
    if not SINGLE_CSV.is_file():
        pytest.skip("aggreprot single CSV fixture missing")
    ref = pd.read_csv(BHT_RPL27)
    result = AggreProtParser(aggregation_threshold=0.24).parse(
        SINGLE_CSV,
        protein_id="RPL27_human",
        sequence=RPL27_SEQ,
    )
    assert result.binary == ref["aggreprot_bin"].astype(int).tolist()


def test_batch_pipeline_skip_run_aggreprot(tmp_path: Path) -> None:
    if not SINGLE_CSV.is_file():
        pytest.skip("aggreprot single CSV fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(f">RPL27_human\n{RPL27_SEQ}\n")

    out = tmp_path / "results"
    work = out / "aggreprot" / "work" / "batch_1"
    work.mkdir(parents=True)
    (work / "RPL27_human_aggreprot.csv").write_text(SINGLE_CSV.read_text())

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["aggreprot"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"RPL27_human"}
    assert (out / "aggreprot" / "parsed" / "RPL27_human_aggreprot.csv").is_file()
    assert any("[AggreProt]" in line or "[aggreprot]" in line.lower() for line in logs)


# --------------------------------------------------------------------------- #
# pdb_id must not be silently dropped.
#
# [runners.aggreprot] pdb_id was accepted by the config and by
# AggreProtWebRunner, but the runner a panel run actually constructs
# (AggreProtRunner) did not declare it, so it was filtered out with a warning
# and the job went through with no structure -- returning sasa = null for every
# residue, which is precisely the value pdb_id exists to obtain.
# --------------------------------------------------------------------------- #


def test_api_runner_declares_pdb_id_so_the_option_is_not_filtered_away():
    import inspect

    from aggressor_wrappers.runners.aggreprot import AggreProtRunner

    assert "pdb_id" in inspect.signature(AggreProtRunner.__init__).parameters


def test_api_submission_refuses_a_pdb_id_it_cannot_carry():
    """Loud failure beats a structure-free run that looks successful.

    The PUT payload has no structure channel -- the server rejects structSource
    and null structure placeholders -- so a pdb_id set here can only be ignored.
    """
    import pytest

    from aggressor_wrappers.runners.aggreprot import AggreProtRunner

    runner = AggreProtRunner(pdb_id="1IYT")
    with pytest.raises(ValueError, match="aggreprot_web"):
        runner._build_job_request({"ABETA42": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"})


def test_api_submission_is_unchanged_when_pdb_id_is_blank():
    """The configured default (pdb_id =) must stay a working unattended run."""
    from aggressor_wrappers.runners.aggreprot import AggreProtRunner

    request = AggreProtRunner(pdb_id="  ")._build_job_request(
        {"ABETA42": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"}
    )
    assert [p["accession"] for p in request["proteins"]] == ["ABETA42"]
    assert not any("struct" in key.lower() for key in request["proteins"][0])


def test_config_pdb_id_reaches_the_constructed_runner():
    from aggressor_wrappers.runners.registry import get_runner

    runner = get_runner("aggreprot", pdb_id="1IYT")
    assert runner.pdb_id == "1IYT"
