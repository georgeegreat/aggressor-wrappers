"""AGGRESCAN submission: the line ending, and the three ways it says no.

All of this was measured against the live CGI, not inferred:

  * ``>t\\nACDEFGHIKLMNPQRSTVWY\\n``   -> 860-byte error page
  * ``>t\\r\\nACDEFGHIKLMNPQRSTVWY\\r\\n`` -> 15 643-byte result table

Same bytes otherwise. The CGI splits FASTA records on CRLF because an HTML
``<textarea>`` is required by the HTML specification to submit CRLF, so every
submission the authors saw had it; a client sending bare LF hands the script one
unsplittable line. The page then reports ``UNIDENTIFIED ERROR`` and blames a
copy-paste from MS Word, which sends a programmatic caller looking in the wrong
place entirely.

Also measured, with CRLF: lower case is accepted; 3, 5, 11, 20, 180, 600 and
2000-residue sequences all return tables; a header with dots
(``>6087.XP_002162002.2``) is fine; 60-column wrapping is fine. Only characters
outside the 20 standard amino acids are refused, and responses came back in
0.26-1.6 s, so none of this is a timeout.
"""

from __future__ import annotations

import io
import urllib.parse
from pathlib import Path

import pytest

from aggressor_wrappers.core.net import PermanentToolError
from aggressor_wrappers.runners import aggrescan as mod
from aggressor_wrappers.runners.aggrescan import (
    STANDARD_RESIDUES,
    AggrescanRunner,
    _crlf,
    parse_result_page,
    validate_sequence,
)

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

_SHELL = (
    '<HTML><HEAD><TITLE>Prediction of "hot spots" of aggregation in '
    "disease-linked polypeptides</TITLE></HEAD><BODY>"
    "<style>.color1 {{background-color: cbc5c5;}}</style>{body}</BODY></html>"
)

# Verbatim bodies from the live service.
LF_ERROR = _SHELL.format(
    body="<P><BR>UNIDENTIFIED ERROR\n<P><BR>Aggrescan has identified that your "
    "sequences are in fasta format, but found an error that can not be identified"
)
NO_HEADER_ERROR = _SHELL.format(
    body="<P><BR>Please use fasta format <a href='#'>click here</a> to see what it "
    "means<P>Aggrescan is a multisequence analyzer, so it needs the \">\" to "
    "identify the name, the beginning and the end of the sequence."
)
BAD_CHAR_ERROR = _SHELL.format(
    body='<P><BR>E R R O R! som characters of your sequence are not allowed\n'
    '<P><BR>\n&gt;t<BR>\nACDEFGHIKL<FONT COLOR="red">X</FONT>MNPQRSTVWY\n<BR>'
)


# ----------------------------------------------------------------- line ending


@pytest.mark.parametrize(
    "raw",
    [">t\nACD\n", ">t\r\nACD\r\n", ">t\rACD\r", ">t\r\nACD\n"],
)
def test_every_line_ending_normalises_to_crlf(raw):
    assert _crlf(raw) == ">t\r\nACD\r\n"


def test_normalisation_is_idempotent():
    once = _crlf(">t\nACD\n")
    assert _crlf(once) == once


def test_the_request_body_actually_carries_crlf(monkeypatch, tmp_path):
    """The regression that mattered: it is the wire bytes, not the local string."""
    captured = {}

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):  # noqa: ARG001
        captured["body"] = request.data.decode()
        return _Response(b"<html/>")

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    AggrescanRunner()._post(f">ABETA42\n{ABETA42}\n")

    sent = urllib.parse.parse_qs(captured["body"])["sequence"][0]
    assert sent == f">ABETA42\r\n{ABETA42}\r\n"
    assert "\n" not in sent.replace("\r\n", "")


# --------------------------------------------------------------- the refusals


def test_bare_lf_refusal_is_named_and_permanent():
    with pytest.raises(PermanentToolError, match="CRLF"):
        parse_result_page(LF_ERROR)


def test_missing_header_refusal_is_named():
    with pytest.raises(PermanentToolError, match="no '>' header"):
        parse_result_page(NO_HEADER_ERROR)


def test_disallowed_character_refusal_names_the_character():
    """The page red-flags the offending characters, so report them, not 'som'."""
    with pytest.raises(PermanentToolError, match=r"20 standard amino acids -- X"):
        parse_result_page(BAD_CHAR_ERROR)


def test_refusals_are_permanent_so_the_scheduler_skips_rather_than_retries():
    from aggressor_wrappers.core.net import classify

    for page in (LF_ERROR, NO_HEADER_ERROR, BAD_CHAR_ERROR):
        try:
            parse_result_page(page)
        except PermanentToolError as exc:
            assert classify(exc) == "permanent"
        else:  # pragma: no cover
            pytest.fail("expected a refusal")


def test_a_genuine_layout_change_still_reads_as_one():
    with pytest.raises(RuntimeError, match="data columns"):
        parse_result_page("<html><body><table><tr><td>hello</td></tr></table></body></html>")


# ------------------------------------------------------------- pre-flight screen


def test_standard_alphabet_is_the_20_amino_acids():
    assert len(STANDARD_RESIDUES) == 20
    assert not STANDARD_RESIDUES & set("BJOUXZ*-")


@pytest.mark.parametrize("sequence", [ABETA42, ABETA42.lower(), "ACD"])
def test_accepted_sequences_pass_the_screen(sequence):
    """Lower case is accepted by the service -- verified -- so do not reject it."""
    assert validate_sequence(sequence) == []


def test_screen_reports_position_and_character():
    assert validate_sequence("ACDXEFG*HI-K") == [(4, "X"), (8, "*"), (11, "-")]


def test_the_fasta_reader_is_the_first_line_of_defence(tmp_path):
    """A non-standard residue never reaches AGGRESCAN at all.

    ``read_fasta`` enforces the same 20-letter alphabet for every predictor, so
    an ortholog set carrying an X fails at load with its own message. Worth
    pinning, because it means a disallowed character was NEVER a candidate
    explanation for a failure seen mid-sweep -- which is what left the line
    ending as the only one standing.
    """
    from aggressor_wrappers.core.fasta import read_fasta

    fasta = tmp_path / "x.fasta"
    fasta.write_text(">ORTHO\nACDXEFGHIK\n")
    with pytest.raises(ValueError, match=r"Non-standard amino acids.*X"):
        read_fasta(fasta)


def test_execute_refuses_locally_without_spending_a_request(monkeypatch, tmp_path):
    """The screen still guards a caller that hands execute() a sequence directly.

    Redundant with the FASTA reader on the pipeline path, and deliberately so:
    the reader can be bypassed, and one avoided round trip per bad sequence is
    the difference between a named local error and 372 remote ones.
    """
    def explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("the request should never leave")

    monkeypatch.setattr(mod.urllib.request, "urlopen", explode)
    monkeypatch.setattr(mod, "read_first_sequence", lambda path: ("ORTHO", "ACDXEFGHIK"))
    fasta = tmp_path / "x.fasta"
    fasta.write_text(">ORTHO\nACDEFGHIK\n")
    with pytest.raises(PermanentToolError, match=r"X@4"):
        AggrescanRunner().execute(fasta, tmp_path)
