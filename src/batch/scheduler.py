"""Predictor-level concurrency for multifasta runs.

The original pipeline iterated predictors sequentially::

    for runner_key in runner_keys:
        _run_runner_batches(runner_key, items, ...)

so total wall time was the *sum* over predictors, and intra-predictor
parallelism existed only for three hardcoded runners (``path``, ``archcandy``,
``crossbeta``). Because the predictors in the panel are mutually independent —
each writes to its own ``work/`` and ``parsed/`` directory and shares no mutable
state — the panel is embarrassingly parallel along the predictor axis, and the
achievable wall time is the *maximum* over predictors rather than the sum.

That axis matters disproportionately here because most runners are long-polling
web services (AggreProt ``timeout_seconds = 1800``, Cross-Beta and ArchCandy
poll remote jobs). Those tasks are I/O-bound, so they overlap almost perfectly
on threads: while AggreProt blocks on its poll loop, PASTA and WALTZ can run.

Two invariants are preserved deliberately:

* **Per-service rate limits.** Each runner keeps its own ``parallel_jobs`` /
  ``sequences_per_run`` caps from ``config.cfg`` (AggreProt allows 3 sequences
  per job, ArchCandy 1). This scheduler adds an *outer* pool over predictors and
  never touches those inner caps, so no external service sees more concurrency
  than it did before.
* **Failure isolation.** One predictor raising (a dead web service, a missing
  licence) must not abort the panel; the exception is captured, reported, and
  the remaining predictors still produce parsed output.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

LogFn = Callable[[str], None]


@dataclass
class PredictorOutcome:
    """Result of running one predictor across the whole multifasta."""

    key: str
    ok: bool
    error: BaseException | None = None

    @property
    def skipped(self) -> bool:
        """A permanent refusal, not a run failure.

        Cross-Beta-Pred's CAPTCHA gate and a transient TLS drop both used to
        print as `FAILED`, which made a recoverable network blip and a
        deliberate, documented refusal to automate a gated form indistinguishable
        in the log — and counted both towards "N failed".
        """
        from aggressor_wrappers.core.net import classify

        return self.error is not None and classify(self.error) == "permanent"


@dataclass
class SchedulerReport:
    """Aggregate outcome of a concurrent predictor sweep."""

    outcomes: list[PredictorOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[PredictorOutcome]:
        """Predictors that failed for a reason that might not recur."""
        return [o for o in self.outcomes if not o.ok and not o.skipped]

    @property
    def skipped(self) -> list[PredictorOutcome]:
        return [o for o in self.outcomes if not o.ok and o.skipped]

    @property
    def succeeded(self) -> list[PredictorOutcome]:
        return [o for o in self.outcomes if o.ok]

    def raise_if_all_failed(self) -> None:
        """Raise only if nothing ran AND something actually failed.

        A panel where every configured predictor was skipped for a permanent,
        documented reason is a misconfiguration to report, not a crash; a panel
        where they all failed on the network is worth raising.
        """
        if self.outcomes and not self.succeeded and not self.failed:
            names = ", ".join(o.key for o in self.skipped)
            raise RuntimeError(
                f"no predictor produced output; all {len(self.skipped)} were "
                f"skipped for permanent reasons ({names}). Check [pipeline] "
                f"predictors and each runner's backend setting."
            )
        if self.outcomes and not self.succeeded:
            first = self.failed[0]
            raise RuntimeError(
                f"all {len(self.failed)} predictor(s) failed; first error from "
                f"{first.key!r}: {first.error}"
            ) from first.error


def serialise_log(emit: LogFn) -> LogFn:
    """Wrap a log function so concurrent predictors don't interleave mid-line."""
    lock = threading.Lock()

    def _emit(message: str) -> None:
        with lock:
            emit(message)

    return _emit


def run_predictors_concurrently(
    runner_keys: list[str],
    run_one: Callable[[str], None],
    *,
    max_workers: int = 4,
    emit: LogFn,
) -> SchedulerReport:
    """Run ``run_one(key)`` for every predictor, up to ``max_workers`` at once.

    ``run_one`` is expected to be the existing per-predictor batch routine, which
    already bounds its own internal concurrency. Exceptions are captured per
    predictor rather than propagated, so a single failing tool degrades the panel
    instead of destroying it.
    """
    report = SchedulerReport()
    if not runner_keys:
        return report

    workers = max(1, min(int(max_workers), len(runner_keys)))
    if workers == 1:
        emit(f"[sched] predictor_jobs=1 — running {len(runner_keys)} predictor(s) sequentially")
        for key in runner_keys:
            report.outcomes.append(_guarded(key, run_one, emit))
        return report

    emit(
        f"[sched] launching {len(runner_keys)} predictor(s) concurrently "
        f"(predictor_jobs={workers}); per-runner parallel_jobs caps still apply"
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pred") as executor:
        futures = {executor.submit(_guarded, key, run_one, emit): key for key in runner_keys}
        for future in as_completed(futures):
            report.outcomes.append(future.result())

    # Both classes are reported; only the counts differ, so a reader can see at
    # a glance whether a run needs retrying or reconfiguring.
    for outcome in report.skipped:
        emit(f"[sched] SKIPPED {outcome.key}: {outcome.error}")
    for outcome in report.failed:
        emit(f"[sched] FAILED {outcome.key}: {outcome.error}")
    summary = f"[sched] {len(report.succeeded)}/{len(runner_keys)} predictor(s) completed"
    if report.failed:
        summary += f"; {len(report.failed)} failed"
    if report.skipped:
        summary += f"; {len(report.skipped)} skipped (permanent)"
    emit(summary)
    return report


def _guarded(key: str, run_one: Callable[[str], None], emit: LogFn) -> PredictorOutcome:
    try:
        run_one(key)
    except BaseException as exc:  # noqa: BLE001 - isolation is the point
        emit(f"[{key}] error: {exc}")
        return PredictorOutcome(key=key, ok=False, error=exc)
    return PredictorOutcome(key=key, ok=True)
