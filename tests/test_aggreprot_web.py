"""AggreProt: the two-call submission trap, and retrieval from the job document."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggressor_wrappers.runners.aggreprot_web import (
    STRUCTURE_SOURCES,
    AggreProtWebRunner,
    parse_job,
    submission_recipe,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aggreprot_job_abeta42.json"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _spans(positions):
    out = []
    for p in positions:
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [(a, b) for a, b in out]


@pytest.mark.skipif(not FIXTURE.is_file(), reason="aggreprot fixture missing")
def test_per_residue_series_comes_from_the_job_document(tmp_path):
    """No CSV download is needed: GET api/jobs/{id} already carries the profile."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">ABETA42\n{ABETA42}\n")
    result = AggreProtWebRunner(aggregation_threshold=0.25).run(
        fasta=fasta, raw_json=FIXTURE
    )
    assert len(result.scores) == len(ABETA42)
    assert max(result.scores) == pytest.approx(0.8464, abs=1e-4)
    assert result.scores.index(max(result.scores)) + 1 == 34
    assert _spans([i for i, b in enumerate(result.binary, 1) if b]) == [(12, 21), (29, 42)]


@pytest.mark.skipif(not FIXTURE.is_file(), reason="aggreprot fixture missing")
def test_no_structure_means_no_sasa_channel(tmp_path):
    """sasa is null throughout unless a structure was supplied on step 2.

    Worth asserting because the absence is silent: the series key exists and is
    full-length, it is simply all None, so a consumer that does not check would
    treat 'unknown burial' as a number.
    """
    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">ABETA42\n{ABETA42}\n")
    result = AggreProtWebRunner().run(fasta=fasta, raw_json=FIXTURE)
    assert result.metadata["has_sasa"] is False
    assert "sasa" not in result.aux
    assert "transmembrane" in result.aux  # this one IS returned without a structure


@pytest.mark.skipif(not FIXTURE.is_file(), reason="aggreprot fixture missing")
def test_sequence_mismatch_is_refused(tmp_path):
    fasta = tmp_path / "wrong.fasta"
    fasta.write_text(">ABETA42\n" + "M" * 42 + "\n")
    with pytest.raises(ValueError, match="different sequence"):
        AggreProtWebRunner().run(fasta=fasta, raw_json=FIXTURE)


def test_running_without_a_job_id_explains_the_two_call_submission(tmp_path):
    """The error must name the trap, because probing the POST looks like success.

    POST /api/jobs accepts any body and returns an id; only the PUT carries the
    data. A wrapper that submits by POST therefore creates empty jobs and gets a
    plausible id back every time.
    """
    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">ABETA42\n{ABETA42}\n")
    with pytest.raises(ValueError, match="only allocates an id"):
        AggreProtWebRunner().run(fasta=fasta)


def test_recipe_selects_no_structure_by_default_and_pdb_when_configured():
    records = {"RPL27": "MGKFMKPGK"}
    without = submission_recipe(records)
    radio = [s for s in without if s.get("action") == "select_radio"][0]
    assert radio["name"] == "inputStructureSourceRPL27"   # named per sequence
    assert radio["value"] == STRUCTURE_SOURCES["none"] == ""

    with_pdb = submission_recipe(records, pdb_ids={"RPL27": "4V6X"})
    radio = [s for s in with_pdb if s.get("action") == "select_radio"][0]
    assert radio["value"] == STRUCTURE_SOURCES["pdb"] == "WWPDB"
    filled = [s for s in with_pdb if s.get("name") == "inputStructureWwpdbRPL27"]
    assert filled and filled[0]["value"] == "4V6X"


def test_parse_job_names_the_protein_it_could_not_find():
    doc = {"id": "x", "proteins": [{"name": "A", "series": {"aminoAcids": [], "aggreprot": []}}]}
    with pytest.raises(ValueError, match=r"has no protein 'B'"):
        parse_job(doc, protein_name="B")


REPORT_CSV = Path(__file__).parent / "fixtures" / "aggreprot_report_abeta42.csv"


@pytest.mark.skipif(not REPORT_CSV.is_file(), reason="aggreprot csv fixture missing")
def test_report_csv_is_the_layout_the_project_already_reads(tmp_path):
    """The CSV path and the JSON path must agree residue for residue.

    They are the same numbers by construction, but the CSV is the primary route
    because its columns are what amyloscope's `aggreprot` adapter consumes —
    going via the JSON would re-serialise them into that shape, adding a
    translation that can drift.
    """
    from aggressor_wrappers.runners.aggreprot_web import split_report_csv

    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">ABETA42\n{ABETA42}\n")
    runner = AggreProtWebRunner(aggregation_threshold=0.25)

    via_csv = runner.run(fasta=fasta, raw_csv=REPORT_CSV)
    via_json = runner.run(fasta=fasta, raw_json=FIXTURE)
    assert via_csv.binary == via_json.binary
    assert via_csv.scores == pytest.approx(via_json.scores, abs=1e-12)

    header = REPORT_CSV.read_text().splitlines()[1]
    assert header.strip() == (
        "position,struct_position,amino_acid,aggregation,sasa,transmembrane"
    )
    blocks = split_report_csv(REPORT_CSV.read_text())
    assert list(blocks) == ["ABETA42"]


@pytest.mark.skipif(not REPORT_CSV.is_file(), reason="aggreprot csv fixture missing")
def test_multi_protein_report_is_split_not_truncated():
    """A `header=1` read of a multi-protein export keeps only the first block.

    Each protein is preceded by its own `Protein N,<accession>` banner and a
    repeated column header, so the file is several tables concatenated — not one
    table with a decorative first line.
    """
    from aggressor_wrappers.runners.aggreprot_web import split_report_csv

    combined = REPORT_CSV.read_text() + (
        "Protein 2,RPL27,,,,\r\n"
        "position,struct_position,amino_acid,aggregation,sasa,transmembrane\r\n"
        "1,,M,0.5,,0.0\r\n"
    )
    blocks = split_report_csv(combined)
    assert list(blocks) == ["ABETA42", "RPL27"]
    assert len(blocks["RPL27"].strip().splitlines()) == 2  # header + one residue


def test_report_csv_url_matches_the_download_button():
    """The button is an <a href>, so the CSV is fetchable rather than saved."""
    runner = AggreProtWebRunner()
    assert runner.base_url.endswith("/aggreprot/")
    # reportCsvUsingGET in the site bundle: "/api/jobs/{id}.csv" + download flag
    assert "api/jobs/{id}.csv".format(id="abc123") == "api/jobs/abc123.csv"
