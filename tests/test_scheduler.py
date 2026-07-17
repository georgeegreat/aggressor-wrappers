"""Predictor-level concurrency for multifasta runs."""

from __future__ import annotations

import threading
import time

import pytest

from aggressor_wrappers.batch.scheduler import run_predictors_concurrently

KEYS = ["pasta", "waltz", "appnn", "crossbeta"]

# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def test_predictors_run_concurrently():
    """Predictors overlap: wall time tracks max(T), not sum(T)."""
    delay = 0.25
    t0 = time.perf_counter()
    report = run_predictors_concurrently(
        KEYS, lambda _k: time.sleep(delay), max_workers=4, emit=lambda _m: None
    )
    elapsed = time.perf_counter() - t0
    assert len(report.succeeded) == len(KEYS)
    # sequential would need >= 4 * delay; allow generous slack for CI jitter
    assert elapsed < delay * len(KEYS) * 0.75


def test_max_observed_concurrency_respects_cap():
    live = 0
    peak = 0
    lock = threading.Lock()

    def run_one(_key):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1

    run_predictors_concurrently(KEYS, run_one, max_workers=2, emit=lambda _m: None)
    assert peak <= 2


def test_single_worker_is_sequential_fallback():
    report = run_predictors_concurrently(
        KEYS, lambda _k: None, max_workers=1, emit=lambda _m: None
    )
    assert len(report.succeeded) == len(KEYS)


def test_one_failing_predictor_does_not_abort_panel():
    def flaky(key):
        if key == "crossbeta":
            raise RuntimeError("web service 503")

    report = run_predictors_concurrently(
        KEYS, flaky, max_workers=4, emit=lambda _m: None
    )
    assert [o.key for o in report.failed] == ["crossbeta"]
    assert len(report.succeeded) == len(KEYS) - 1
    report.raise_if_all_failed()  # must not raise: some succeeded


def test_all_failing_raises():
    def dead(_key):
        raise RuntimeError("boom")

    report = run_predictors_concurrently(
        KEYS, dead, max_workers=2, emit=lambda _m: None
    )
    with pytest.raises(RuntimeError, match="all 4 predictor"):
        report.raise_if_all_failed()


