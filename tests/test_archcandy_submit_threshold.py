"""The calling cut must not double as the service's filter.

ArchCandy 2.0 takes a threshold in the submission payload and returns only
arches at or above it. That threshold was the same number the parser binarises
with, so at the 0.57 significance boundary the raw CSV could not show anything
below 0.57 -- and on the seven-protein ribosomal panel six proteins came back
with a header and no rows. The parser wrote those as a profile of zeros, which
in the merged table, in amyloscope's tracks and in every figure is
indistinguishable from a tool that scored the protein and found nothing.

The consequence was not cosmetic: ArchCandy still counted in the consensus
denominator for all six, and on RPL36a the panel fell to four calling tools
against a lowest tier needing five, so no consensus region was attainable for
that protein at any tier.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.runners.archcandy_v2 import (
    SIGNIFICANCE_THRESHOLD,
    WEB_DEFAULT_THRESHOLD,
    ArchCandy2Runner,
    _region_rows,
)

HEADER = "ID,Sequence,Arch,Start,Stop,Score"


def test_submission_uses_the_recall_floor_not_the_calling_cut():
    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD)

    assert runner.threshold == SIGNIFICANCE_THRESHOLD
    assert runner.submit_threshold == WEB_DEFAULT_THRESHOLD
    assert runner.submit_threshold < runner.threshold


def test_the_payload_carries_the_submit_threshold(monkeypatch):
    """The number that reaches the service is the floor, not the call."""
    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD)
    seen = {}

    def _fake_request(path, *, payload=None):
        seen["payload"] = payload
        return 201, '{"jobId": "j", "publicToken": "t"}'

    monkeypatch.setattr(runner, "_request", _fake_request)
    runner.submit(">p\nMAVLK\n")

    assert seen["payload"]["threshold"] == WEB_DEFAULT_THRESHOLD


def test_a_cut_below_the_floor_submits_at_the_cut():
    """Submitting at 0.40 would discard arches a 0.30 call would accept."""
    runner = ArchCandy2Runner(threshold=0.30)

    assert runner.submit_threshold == 0.30


def test_submitting_above_the_calling_cut_is_refused():
    with pytest.raises(ValueError) as excinfo:
        ArchCandy2Runner(threshold=0.40, submit_threshold=SIGNIFICANCE_THRESHOLD)

    assert "filter out arches the call would accept" in str(excinfo.value)


def test_an_explicit_submit_threshold_is_honoured():
    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD, submit_threshold=0.0)

    assert runner.submit_threshold == 0.0


# --------------------------------------------------------------------------- #
# An empty region table is a result, not an absence.
# --------------------------------------------------------------------------- #


def test_a_header_only_table_counts_zero_regions():
    """The job reports DONE with exitCode 0 whether or not it found anything."""
    assert _region_rows(HEADER + "\n") == 0
    assert _region_rows(HEADER) == 0
    assert _region_rows("") == 0


def test_rows_are_counted_ignoring_blank_lines():
    text = HEADER + "\nRPS2,QKLVFFA,GBPL,125,160,0.675\n\n"

    assert _region_rows(text) == 1


def test_an_empty_result_is_announced(tmp_path, monkeypatch, capsys):
    """Silence from the service has to reach the log, not just the file."""
    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD)
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">RPL36\nMALRYPMAVG\n", encoding="utf-8")

    monkeypatch.setattr(runner, "submit", lambda text: {"jobId": "j", "publicToken": "t"})
    monkeypatch.setattr(runner, "wait", lambda job: {"status": "DONE", "exitCode": 0})
    monkeypatch.setattr(runner, "fetch_csv", lambda job: HEADER + "\n")

    runner.execute(fasta, tmp_path / "work")

    message = capsys.readouterr().err
    assert "RPL36" in message
    assert "NO beta-arch candidates" in message
    # Both numbers, because "no arches" at 0.57 is a different claim from
    # "no arches" at 0.40.
    assert "0.4" in message and "0.57" in message


def test_the_sidecar_records_both_thresholds_and_the_row_count(tmp_path, monkeypatch):
    """work/ is deleted at the end of a run; the counts must outlive it."""
    import json

    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD)
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">RPL36\nMALRYPMAVG\n", encoding="utf-8")

    monkeypatch.setattr(runner, "submit", lambda text: {"jobId": "j", "publicToken": "t"})
    monkeypatch.setattr(runner, "wait", lambda job: {"status": "DONE", "exitCode": 0})
    monkeypatch.setattr(runner, "fetch_csv", lambda job: HEADER + "\n")

    runner.execute(fasta, tmp_path / "work")

    sidecar = json.loads(
        (tmp_path / "work" / "RPL36_archcandy2.job.json").read_text(encoding="utf-8")
    )
    assert sidecar["threshold"] == SIGNIFICANCE_THRESHOLD
    assert sidecar["submit_threshold"] == WEB_DEFAULT_THRESHOLD
    assert sidecar["region_rows"] == 0


def test_a_populated_table_is_not_announced(tmp_path, monkeypatch, capsys):
    runner = ArchCandy2Runner(threshold=SIGNIFICANCE_THRESHOLD)
    fasta = tmp_path / "p.fasta"
    fasta.write_text(">RPS2\nMAVLKMAVLK\n", encoding="utf-8")

    monkeypatch.setattr(runner, "submit", lambda text: {"jobId": "j", "publicToken": "t"})
    monkeypatch.setattr(runner, "wait", lambda job: {"status": "DONE", "exitCode": 0})
    monkeypatch.setattr(
        runner, "fetch_csv", lambda job: HEADER + "\nRPS2,MAVLK,GBPL,1,5,0.675\n"
    )

    runner.execute(fasta, tmp_path / "work")

    assert "NO beta-arch candidates" not in capsys.readouterr().err


def test_calls_are_unchanged_by_the_lower_submission(tmp_path):
    """The point of the split: more evidence, identical calls.

    Under score_mode='highest' a residue takes the best single arch covering
    it, so an arch below the calling cut cannot flip a residue on. The score
    column gains the ambiguous band; the APR column does not move.
    """
    from aggressor_wrappers.predictors.archcandy_v2 import ArchCandy2Parser

    sequence = "MAVLKMAVLK"
    significant_only = tmp_path / "high.csv"
    significant_only.write_text(
        HEADER + "\nP,MAVLK,GBPL,1,5,0.675\n", encoding="utf-8"
    )
    with_ambiguous = tmp_path / "low.csv"
    with_ambiguous.write_text(
        HEADER + "\nP,MAVLK,GBPL,1,5,0.675\nP,MAVLK,GBPL,6,10,0.45\n",
        encoding="utf-8",
    )

    parser = ArchCandy2Parser(score_mode="highest", threshold=SIGNIFICANCE_THRESHOLD)
    high = parser.parse(significant_only, protein_id="P", sequence=sequence)
    low = parser.parse(with_ambiguous, protein_id="P", sequence=sequence)

    assert list(low.binary) == list(high.binary)
    # ... and the previously-blank half now carries its arch's confidence.
    assert max(low.scores[5:]) == pytest.approx(0.45)
    assert max(high.scores[5:]) == 0.0
