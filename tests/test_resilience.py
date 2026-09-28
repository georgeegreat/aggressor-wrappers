"""Transient vs permanent failure: retry the first, report the second."""

from __future__ import annotations

import ssl
import urllib.error

import pytest

from aggressor_wrappers.batch.scheduler import PredictorOutcome, SchedulerReport
from aggressor_wrappers.core.net import PermanentToolError, classify, retry_call


@pytest.mark.parametrize(
    "error,expected",
    [
        (ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"), "transient"),
        (RuntimeError("Remote end closed connection without response"), "transient"),
        (ConnectionResetError("reset by peer"), "transient"),
        (urllib.error.HTTPError("u", 503, "busy", None, None), "transient"),
        (urllib.error.HTTPError("u", 429, "slow down", None, None), "transient"),
        (urllib.error.HTTPError("u", 404, "gone", None, None), "permanent"),
        (PermanentToolError("reCAPTCHA"), "permanent"),
    ],
)
def test_classification_matches_the_failures_seen_in_a_real_sweep(error, expected):
    """Every case here was observed in one 372-protein run."""
    assert classify(error) == expected


def test_transient_failure_is_retried_and_can_succeed():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")
        return "ok"

    assert retry_call(flaky, base_delay=0, attempts=4) == "ok"
    assert calls["n"] == 3


def test_permanent_failure_is_not_retried():
    """Retrying a CAPTCHA gate wastes time and hammers someone's server."""
    calls = {"n": 0}

    def gated():
        calls["n"] += 1
        raise PermanentToolError("reCAPTCHA")

    with pytest.raises(PermanentToolError):
        retry_call(gated, base_delay=0, attempts=5)
    assert calls["n"] == 1


def test_retry_gives_up_and_reraises_the_last_error():
    def always():
        raise ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")

    with pytest.raises(ssl.SSLError):
        retry_call(always, base_delay=0, attempts=2)


def test_scheduler_separates_a_gated_tool_from_a_dropped_connection():
    report = SchedulerReport(outcomes=[
        PredictorOutcome("appnn", True),
        PredictorOutcome("crossbeta", False, PermanentToolError("reCAPTCHA")),
        PredictorOutcome("waltz", False, ssl.SSLError("UNEXPECTED_EOF_WHILE_READING")),
    ])
    assert [o.key for o in report.succeeded] == ["appnn"]
    assert [o.key for o in report.failed] == ["waltz"]
    assert [o.key for o in report.skipped] == ["crossbeta"]


def test_all_skipped_raises_a_configuration_message_not_a_crash():
    report = SchedulerReport(outcomes=[
        PredictorOutcome("crossbeta", False, PermanentToolError("reCAPTCHA")),
    ])
    with pytest.raises(RuntimeError, match="skipped for permanent reasons"):
        report.raise_if_all_failed()


def test_a_genuine_all_failure_still_raises_the_original_error():
    report = SchedulerReport(outcomes=[
        PredictorOutcome("waltz", False, ssl.SSLError("boom")),
    ])
    with pytest.raises(RuntimeError, match="all 1 predictor"):
        report.raise_if_all_failed()
