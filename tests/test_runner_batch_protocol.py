"""Every runner the scheduler's GENERIC path handles must implement its protocol.

`_run_generic_batch` calls ``execute_batch`` and ``discover_outputs`` on whatever
runner it is handed, but ``BasePredictorRunner`` marks only ``run`` abstract. So a
runner can be registered, pass its availability check, get scheduled, and then
fail with ``AttributeError`` after the panel is already running -- which is how
AmyloGram aborted partway through an orthologue sweep.

The comment in `_run_generic_batch` records the same failure for FoldAmyloid and
AGGRESCAN, and says the fix was removing the scheduler's `if runner_key == ...`
ladder. That removed the need to EDIT the scheduler per predictor; it did not make
the protocol checkable. This does, at collection time rather than at hour three of
a sweep.

Scope matters here, and a first draft of this file got it wrong by demanding the
protocol of every class in the registry. Two groups legitimately do not need it:
predictors the ladder still routes to a dedicated function, and alias keys such as
``aggreprot_web`` or ``archcandy_local``, which name an implementation for a caller
pinning one and are not schedulable predictor keys at all.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY
from aggressor_wrappers.runners.registry import RUNNER_REGISTRY

#: Methods `_run_generic_batch` calls on a runner it was handed.
BATCH_PROTOCOL = ("execute_batch", "discover_outputs")

#: Predictor keys the scheduler's ladder still routes to a dedicated function,
#: which implements the batching those tools need (concurrency, single-invocation
#: R, licensed dependencies) instead of the generic two-method protocol.
#: `test_the_dedicated_paths_still_exist` keeps this list honest.
DEDICATED_SCHEDULER_PATHS = frozenset(
    {"appnn", "waltz", "pasta", "archcandy", "crossbeta", "aggreprot", "path"}
)


def _generic_path_runners() -> dict[str, type]:
    """Schedulable predictor keys that fall through to `_run_generic_batch`."""
    return {
        key: RUNNER_REGISTRY[key]
        for key in PREDICTOR_REGISTRY
        if key not in DEDICATED_SCHEDULER_PATHS and key in RUNNER_REGISTRY
    }


@pytest.mark.parametrize("key,cls", sorted(_generic_path_runners().items()))
def test_runner_implements_the_batch_protocol(key, cls):
    missing = [name for name in BATCH_PROTOCOL if not hasattr(cls, name)]
    assert not missing, (
        f"{key} ({cls.__name__}) reaches _run_generic_batch but lacks {missing}. "
        f"That fails at RUN time on a full panel rather than here. Implement them "
        f"over the runner's own per-protein logic."
    )


@pytest.mark.parametrize("key,cls", sorted(_generic_path_runners().items()))
def test_execute_batch_takes_a_fasta_and_a_work_directory(key, cls):
    """Signature, not only presence: the scheduler passes both positionally."""
    parameters = [
        name
        for name, p in inspect.signature(cls.execute_batch).parameters.items()
        if name != "self" and p.kind is not p.VAR_KEYWORD
    ]
    assert len(parameters) >= 2, (
        f"{key}.execute_batch{inspect.signature(cls.execute_batch)} cannot accept "
        f"(fasta, work_dir) as the scheduler calls it"
    )


def test_the_dedicated_paths_still_exist():
    """Guard the exemption list against the dangerous direction of drift.

    If a predictor is dropped from the ladder it starts using the generic path,
    and staying on this list would exempt it from the only check that covers it.
    """
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "batch" / "pipeline.py"
    ).read_text(encoding="utf-8")
    for key in sorted(DEDICATED_SCHEDULER_PATHS):
        assert f'runner_key == "{key}"' in source, (
            f"{key} is listed as having a dedicated scheduler path, but "
            f"pipeline.py no longer dispatches on it -- so it now reaches "
            f"_run_generic_batch and must implement the batch protocol. Remove it "
            f"from DEDICATED_SCHEDULER_PATHS."
        )


def test_the_generic_set_is_not_empty():
    """A vacuous parametrisation would make this file pass while testing nothing."""
    assert _generic_path_runners(), "no runner reaches the generic path"


def test_amylogram_specifically_now_has_it():
    """The regression this file was written for."""
    from aggressor_wrappers.runners.amylogram import AmyloGramRunner

    assert hasattr(AmyloGramRunner, "execute_batch")
    assert hasattr(AmyloGramRunner, "discover_outputs")
