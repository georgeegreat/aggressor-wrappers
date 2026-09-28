"""Cross-Beta-Pred 2.0: anonymous REST API, and its 0-based index.

The refusal this replaces was based on the page form's reCAPTCHA widget. The
API asks for no token, and the earlier "no API exists" probe used the wrong
spelling (`crossBetaPred`); the route is `crossbetaPred`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aggressor_wrappers.runners.crossbeta_v2 import CrossBeta2Runner, parse_result
from aggressor_wrappers.runners.registry import get_runner

FIXTURE = Path(__file__).parent / "fixtures" / "crossbeta_v2_result_abeta42.json"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


@pytest.mark.skipif(not FIXTURE.is_file(), reason="crossbeta2 fixture missing")
def test_index_is_zero_based_and_is_shifted():
    """Every other predictor in the panel is 1-based; this one is not."""
    parsed = parse_result(json.loads(FIXTURE.read_text()), sequence=ABETA42)
    assert parsed["numbers"][0] == 1
    assert parsed["numbers"][-1] == len(ABETA42)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="crossbeta2 fixture missing")
def test_per_residue_score_is_mean_confidence_not_score_list():
    """score_list holds the 15 window values; mean_confidence is their average."""
    parsed = parse_result(json.loads(FIXTURE.read_text()), sequence=ABETA42)
    assert parsed["window"] == 15
    assert parsed["scores"][37] == pytest.approx(0.8536, abs=1e-3)
    assert max(parsed["scores"]) == parsed["scores"][37]


@pytest.mark.skipif(not FIXTURE.is_file(), reason="crossbeta2 fixture missing")
def test_its_own_regions_are_the_default_call(tmp_path):
    """AR_list embeds Cross-Beta's windowing; a fixed cut on the smoothed
    confidence is a different quantity, so it is opt-in."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">ABETA42\n{ABETA42}\n")
    result = CrossBeta2Runner().run(fasta=fasta, raw_json=FIXTURE)
    assert result.metadata["binarised_from"] == "AR_list"
    # On Abeta42 the tool calls the WHOLE peptide: a 15-residue window cannot
    # resolve a nucleating hexapeptide, which is the documented limitation.
    assert sum(result.binary) == len(ABETA42)

    by_threshold = CrossBeta2Runner(use_tool_regions=False, confidence_threshold=0.8).run(
        fasta=fasta, raw_json=FIXTURE
    )
    assert sum(by_threshold.binary) < len(ABETA42)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="crossbeta2 fixture missing")
def test_sequence_mismatch_is_refused():
    with pytest.raises(ValueError, match="different sequence"):
        parse_result(json.loads(FIXTURE.read_text()), sequence="M" * 42)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="crossbeta2 fixture missing")
def test_length_mismatch_is_refused():
    with pytest.raises(ValueError, match="42 residues"):
        parse_result(json.loads(FIXTURE.read_text()), sequence=ABETA42[:20])


def test_multi_sequence_job_is_refused(tmp_path):
    """prot_name is always 'sequence_query', so a batch cannot be demultiplexed."""
    fasta = tmp_path / "two.fasta"
    fasta.write_text(">A\nMKVLA\n>B\nMQQWW\n")
    with pytest.raises(ValueError, match="one sequence per job"):
        CrossBeta2Runner().execute_batch(fasta, tmp_path)


def test_registry_web_arm_is_the_rest_api():
    assert isinstance(get_runner("crossbeta2"), CrossBeta2Runner)
