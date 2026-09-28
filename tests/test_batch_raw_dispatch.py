"""Regressions for the two failures seen on the 372-protein FXR1 sweep.

Both were silent misroutings rather than crashes in the predictor code itself,
and both reported an error that pointed away from the real cause -- which is why
they get tests rather than a fix alone.

  1. `[sched] FAILED foldamyloid: Provide raw_csv when --skip-run is set`
     from a run that never set --skip-run. The batch parse step had a hardcoded
     runner_key -> raw-file-keyword map; runners absent from it received
     raw_csv=None *and* skip_run=True, so they raised their own skip-run guard.

  2. `[sched] FAILED aggrescan: ... 0 column blocks ... The page layout changed.`
     from a layout that had not changed. The CGI path was joined RELATIVELY to
     base_url, so it only resolved when base_url happened to be the /bioinf/
     root; the natural value (the tool's own page) produced a 404 whose HTML has
     no <small> blocks, which the parser then reported as a layout change.
"""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import (
    BatchLayout,
    _predictor_tag,
    _parse_and_write_runner_batch,
    _RAW_KEYWORDS,
    _raw_kwarg,
)
from aggressor_wrappers.runners.aggrescan import _CGI_PATH, AggrescanRunner
from aggressor_wrappers.runners.registry import get_runner

# Runners whose backend is unconditionally available in a test environment, i.e.
# no Rscript, no vendored jar, no local binary on PATH.
WEB_RUNNER_KEYS = (
    "appnn",
    "waltz",
    "pasta",
    "crossbeta",
    "archcandy",
    "aggreprot",
    "foldamyloid",
    "aggrescan",
)


@pytest.mark.parametrize("runner_key", WEB_RUNNER_KEYS)
def test_every_runner_declares_a_raw_keyword(runner_key):
    """The batch parse step can hand a pre-fetched file to every runner.

    The old hardcoded map made "was this predictor remembered?" a property of an
    unrelated function. Resolving against the signature makes it a property of
    the runner, so a newly added predictor cannot regress this by omission.
    """
    runner = get_runner(runner_key)
    kwargs = _raw_kwarg(runner, "RAW")
    assert list(kwargs.values()) == ["RAW"]
    (keyword,) = kwargs
    assert keyword in _RAW_KEYWORDS


@pytest.mark.parametrize(
    ("runner_key", "expected"),
    [
        ("foldamyloid", "raw_csv"),   # regression: absent from the old map
        ("aggrescan", "raw_csv"),     # regression: absent from the old map
        ("waltz", "raw_txt"),
        ("pasta", "raw_profile"),
        ("crossbeta", "raw_json"),
        ("appnn", "raw_csv"),
    ],
)
def test_raw_keyword_matches_the_runners_own_signature(runner_key, expected):
    """Pin the keyword each runner actually declares, not the one we assume.

    A runner renaming its parameter should fail here, where the cause is
    visible, rather than deep in a batch run as a skip-run complaint.
    """
    assert _raw_kwarg(get_runner(runner_key), "RAW") == {expected: "RAW"}


def test_raw_kwarg_rejects_a_runner_with_no_raw_parameter():
    class NoRawRunner:
        def run(self, fasta_path, out_dir):  # noqa: ARG002
            raise AssertionError("not called")

    with pytest.raises(TypeError, match="declares none of"):
        _raw_kwarg(NoRawRunner(), "RAW")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://andromeda.uab.cat/bioinf/",                      # the old working value
        "http://andromeda.uab.cat/bioinf/aggrescan/",            # the tool's own page
        "http://andromeda.uab.cat/bioinf/aggrescan/index.html",  # a page, not a directory
        "http://andromeda.uab.cat/",                             # host root
    ],
)
def test_aggrescan_cgi_url_is_independent_of_the_base_url_path(base_url):
    """Any base_url on the right host reaches the same CGI.

    The CGI lives one level ABOVE the tool page, so a relative join silently
    produced .../aggrescan/cgi-bin/aap/aap_ov.pl -- a 404 page that parses as
    zero data blocks. Anchoring to scheme+host removes the trap.
    """
    runner = AggrescanRunner(base_url=base_url)
    assert runner.cgi_url == "http://andromeda.uab.cat" + _CGI_PATH


def test_aggrescan_cgi_url_keeps_the_scheme_and_host_it_was_given():
    """Anchoring must not hardcode the host: a mirror or https stays honoured."""
    runner = AggrescanRunner(base_url="https://mirror.example.org/bioinf/aggrescan/")
    parts = urllib.parse.urlsplit(runner.cgi_url)
    assert (parts.scheme, parts.netloc, parts.path) == (
        "https",
        "mirror.example.org",
        _CGI_PATH,
    )


def test_aggrescan_cgi_path_is_overridable():
    runner = AggrescanRunner(
        base_url="http://andromeda.uab.cat/bioinf/aggrescan/",
        cgi_path="/bioinf/cgi-bin/aap/aap_other.pl",
    )
    assert runner.cgi_url.endswith("/aap_other.pl")


# --------------------------------------------------------------------------- #
# End-to-end through the function that actually failed on the FXR1 sweep.
# The unit tests above pin the keyword; this pins the consequence -- a parsed
# CSV on disk -- so a future refactor of the dispatch cannot pass the keyword
# and still lose the file.
# --------------------------------------------------------------------------- #

FIXTURES = Path(__file__).parent / "fixtures"
ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


@pytest.mark.parametrize(
    ("runner_key", "fixture"),
    [
        ("foldamyloid", "foldamyloid_abeta42.csv"),
        ("aggrescan", "aggrescan_abeta42.csv"),
    ],
)
def test_batch_parse_step_reaches_disk_without_a_skip_run_complaint(
    runner_key, fixture, tmp_path
):
    """Reproduces `FAILED foldamyloid: Provide raw_csv when --skip-run is set`.

    Before the fix this raised from the runner's own skip-run guard, because the
    dispatch handed it raw_csv=None together with skip_run=True. The run had not
    set --skip-run at all, so the message named a flag the operator never used.
    """
    raw = FIXTURES / fixture
    if not raw.is_file():
        pytest.skip(f"{fixture} fixture missing")

    layout = BatchLayout.create(tmp_path / "results")
    layout.ensure([runner_key])
    (layout.fasta_split_dir / "ABETA42.fasta").write_text(f">ABETA42\n{ABETA42}\n")

    batch_work = layout.predictor_work_dir(runner_key) / "batch_1"
    batch_work.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    _parse_and_write_runner_batch(
        get_runner(runner_key),
        runner_key,
        ["ABETA42"],
        {"ABETA42": raw},
        layout,
        batch_work,
        batch_label="1",
        skip_run=False,
        save_raw_files=None,
        emit=logs.append,
    )

    tag = _predictor_tag(runner_key)
    written = layout.predictor_parsed_dir(runner_key) / f"ABETA42_{tag}.csv"
    assert written.is_file(), logs
    assert written.read_text().count("\n") > len(ABETA42) // 2


# --------------------------------------------------------------------------- #
# A permanent refusal belongs to ONE submission, not to the predictor.
# Before this, a single sequence the service would not accept aborted the whole
# predictor: on a 372-protein sweep, 371 usable results were thrown away
# because of the 372nd.
# --------------------------------------------------------------------------- #

from aggressor_wrappers.batch.pipeline import _report_refusals  # noqa: E402
from aggressor_wrappers.core.net import PermanentToolError  # noqa: E402


def test_partial_refusal_lets_the_rest_of_the_sweep_stand():
    logs: list[str] = []
    _report_refusals(
        "AGGRESCAN",
        [("7/372", ["ORTHO_7"], PermanentToolError("refused"))],
        completed=371,
        emit=logs.append,
    )
    joined = "\n".join(logs)
    assert "ORTHO_7" in joined
    assert "371 batch(es) completed" in joined


def test_total_refusal_is_raised_so_the_predictor_reads_as_skipped():
    """Nothing got through: that is about the predictor, not about one protein.

    Raising keeps it out of the merged table entirely rather than contributing
    an empty column that downstream code would read as 'AGGRESCAN abstained'.
    """
    with pytest.raises(PermanentToolError, match="every submission was refused"):
        _report_refusals(
            "AGGRESCAN",
            [
                ("1/2", ["A"], PermanentToolError("CRLF")),
                ("2/2", ["B"], PermanentToolError("CRLF")),
            ],
            completed=0,
            emit=lambda _: None,
        )


def test_no_refusals_says_nothing():
    logs: list[str] = []
    _report_refusals("AGGRESCAN", [], completed=3, emit=logs.append)
    assert logs == []
