"""Predictor runner registry."""

from __future__ import annotations

from typing import Any

from aggressor_wrappers.core.config import load_config, predictor_options, runner_options
from aggressor_wrappers.core.schema import resolve_predictor_key
from aggressor_wrappers.runners.appnn import APPNNRunner
from aggressor_wrappers.runners.archcandy import ArchCandyRunner
from aggressor_wrappers.runners.base import BasePredictorRunner
from aggressor_wrappers.runners.crossbeta import CrossBetaRunner
from aggressor_wrappers.runners.path import PATHRunner
from aggressor_wrappers.runners.pasta import PASTARunner
from aggressor_wrappers.runners.waltz import WALTZRunner

RUNNER_REGISTRY: dict[str, type[BasePredictorRunner]] = {
    "path": PATHRunner,
    "appnn": APPNNRunner,
    "waltz": WALTZRunner,
    "pasta": PASTARunner,
    "archcandy": ArchCandyRunner,
    "crossbeta": CrossBetaRunner,
}

# Pipeline-only keys from [runners.*]; not passed to runner constructors.
_RUNNER_BATCH_KEYS = frozenset({"parallel_jobs", "sequences_per_run"})


def get_runner(
    name: str,
    *,
    config_path: str | None = None,
    **overrides: Any,
) -> BasePredictorRunner:
    key = resolve_predictor_key(name)
    if key not in RUNNER_REGISTRY:
        raise KeyError(f"No runner registered for {name!r}. Known: {sorted(RUNNER_REGISTRY)}")

    cfg = load_config(config_path)
    options = {
        k: v for k, v in runner_options(key, cfg).items() if k not in _RUNNER_BATCH_KEYS
    }
    if key == "pasta":
        options.setdefault(
            "energy_threshold",
            predictor_options(key, cfg).get("energy_threshold"),
        )
    if key == "archcandy":
        options.setdefault(
            "score_mode",
            predictor_options(key, cfg).get("score_mode", "cumulative"),
        )
    if key == "crossbeta":
        options.setdefault(
            "confidence_threshold",
            predictor_options(key, cfg).get("confidence_threshold", 0.54),
        )
    options.update(overrides)
    return RUNNER_REGISTRY[key](**options)


def list_runners() -> list[str]:
    return sorted(RUNNER_REGISTRY)
