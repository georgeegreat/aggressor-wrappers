"""Predictor runner registry.

Two things happen here that used to be scattered or missing.

**Backend dispatch.** Several predictors exist as both a web service and a local
tool. ``[runners.<key>] backend`` selects between them, and the choice is made
*before* any network call (see :mod:`aggressor_wrappers.core.backend`), so a dead
web service costs nothing when the local tool is present. Previously the
``backend`` key was written into config.cfg but never consumed: it reached the
web runner constructors as an unexpected keyword and every predictor carrying it
died at construction. That is why a panel configured with nine predictors
returned only WALTZ -- the one runner whose config section had no ``backend``
line.

**Option filtering.** A ``[runners.<key>]`` section describes the predictor, not
one implementation of it, so it necessarily carries keys that only the web
runner understands (``base_url``, ``poll_interval_seconds``) alongside keys only
the local runner understands (``jar_path``, ``repo_path``). Options are therefore
matched against the selected class's signature; anything left over is dropped
with a warning rather than raising, so adding a key for one backend cannot break
the other, while a genuine typo is still reported instead of silently ignored.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

from aggressor_wrappers.core.backend import LocalTool, select_backend
from aggressor_wrappers.core.config import load_config, predictor_options, runner_options
from aggressor_wrappers.core.schema import resolve_predictor_key
from aggressor_wrappers.runners.aggreprot import AggreProtRunner
from aggressor_wrappers.runners.aggreprot_web import AggreProtWebRunner
from aggressor_wrappers.runners.aggrescan import AggrescanRunner
from aggressor_wrappers.runners.amylodeep import AmyloDeepRunner
from aggressor_wrappers.runners.amylogram import AmyloGramRunner
from aggressor_wrappers.runners.appnn import APPNNRunner
from aggressor_wrappers.runners.archcandy_v2 import ArchCandy2Runner
from aggressor_wrappers.runners.archcandy_local import ArchCandyLocalRunner
from aggressor_wrappers.runners.base import BasePredictorRunner
from aggressor_wrappers.runners.crossbeta_v2 import CrossBeta2Runner
from aggressor_wrappers.runners.crossbeta_local import CrossBetaLocalRunner
from aggressor_wrappers.runners.foldamyloid import FoldAmyloidRunner
from aggressor_wrappers.runners.pasta import PASTARunner
from aggressor_wrappers.runners.path import PATHRunner
from aggressor_wrappers.runners.tango import TANGORunner
from aggressor_wrappers.runners.waltz import WALTZRunner

RUNNER_REGISTRY: dict[str, type[BasePredictorRunner]] = {
    "path": PATHRunner,
    "appnn": APPNNRunner,
    "waltz": WALTZRunner,
    "pasta": PASTARunner,
    # Both keys name the PREDICTOR, and the backend policy below picks the
    # implementation. The 1.x web runners they used to name are retired to
    # legacy/runners/ -- ArchCandy's route no longer exists and Cross-Beta's form
    # is CAPTCHA-gated, so neither could fetch anything. Reparsing historical
    # output from either is unaffected: that is the parser, not the runner.
    "archcandy": ArchCandy2Runner,
    "crossbeta": CrossBeta2Runner,
    "aggreprot": AggreProtRunner,
    "aggrescan": AggrescanRunner,
    "aggreprot_web": AggreProtWebRunner,
    "tango": TANGORunner,
    "amylogram": AmyloGramRunner,
    "amylodeep": AmyloDeepRunner,
    "foldamyloid": FoldAmyloidRunner,
    # Explicit local classes stay addressable by name, for callers that want to
    # pin an implementation instead of going through the backend policy.
    # The digit is the TOOL's version, not a file revision: `archcandy2` is
    # ArchCandy 2.0. Module files are named archcandy_v2.py / crossbeta_v2.py so
    # that is unambiguous on disk; the config keys keep the short spelling.
    "archcandy_local": ArchCandyLocalRunner,
    "archcandy2": ArchCandy2Runner,
    "crossbeta_local": CrossBetaLocalRunner,
    "crossbeta2": CrossBeta2Runner,
}

#: Predictors with a single implementation and no _BACKENDS entry, whose one
#: implementation is a web service. Everything else without a policy is local.
#: Reported by resolve_backend() in a pre-flight check, so a wrong answer here
#: tells an operator a web tool will run offline.
_WEB_ONLY = frozenset(
    {"waltz", "pasta", "aggreprot", "aggreprot_web", "aggrescan", "foldamyloid"}
)

# Pipeline-only keys from [runners.*]; not passed to runner constructors.
_RUNNER_BATCH_KEYS = frozenset({"parallel_jobs", "sequences_per_run"})

# Keys consumed by the dispatcher itself.
_BACKEND_KEYS = frozenset({"backend"})


def _crossbeta_local_tool(options: dict[str, Any]) -> LocalTool | None:
    """Locate a cross-beta-predictor checkout.

    CrossBetaLocalRunner runs ``<repo_path>/<SCRIPT_NAME>``, so the presence
    test is the script, not the directory: an empty or half-cloned directory
    would otherwise be reported as an available local backend and the failure
    would surface only once the subprocess ran.
    """
    repo = options.get("repo_path")
    if not repo:
        return None
    from aggressor_wrappers.runners.crossbeta_local import SCRIPT_NAME

    return LocalTool(kind="executable", path=str(Path(repo).expanduser() / SCRIPT_NAME))


def _jar_tool(option_name: str):
    def factory(options: dict[str, Any]) -> LocalTool | None:
        value = options.get(option_name)
        return LocalTool(kind="jar", path=str(value)) if value else None

    return factory


def _binary_tool(option_name: str, default: str | None = None):
    def factory(options: dict[str, Any]) -> LocalTool | None:
        value = options.get(option_name, default)
        if not value:
            return None
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return LocalTool(kind="binary", path=str(candidate))
        return LocalTool(kind="binary", executable=str(value))

    return factory


def _executable_tool(option_name: str, default: str):
    def factory(options: dict[str, Any]) -> LocalTool | None:
        return LocalTool(kind="executable", executable=str(options.get(option_name, default)))

    return factory


#: predictor key -> (web class | None, local class | None, local-tool locator)
_BACKENDS: dict[str, tuple[type | None, type | None, Any]] = {
    # The web arm is ArchCandy 2.0. The 1.x runner is retired to
    # legacy/runners/archcandy_v1_deprecated.py: it targets
    # index.php?route=tools&tool=7, which the rebuilt site no longer serves, so
    # it could only ever return a dead link.
    "archcandy": (ArchCandy2Runner, ArchCandyLocalRunner, _jar_tool("jar_path")),
    "archcandy2": (ArchCandy2Runner, None, None),
    # Web arm is the Cross-Beta-Pred 2.0 REST API (anonymous, no CAPTCHA).
    "crossbeta": (CrossBeta2Runner, CrossBetaLocalRunner, _crossbeta_local_tool),
    "crossbeta2": (CrossBeta2Runner, None, None),
    # TANGO is licensed and has no web service: local only.
    "tango": (None, TANGORunner, _binary_tool("binary_path", "tango")),
    # AmyloGram and APPNN are R packages driven through Rscript. No REST API
    # exists for either (AmyloGram's public interface is a Shiny app), so
    # has_web is False and backend=web is rejected rather than silently ignored.
    "amylogram": (None, AmyloGramRunner, _executable_tool("rscript", "Rscript")),
    "amylodeep": (None, AmyloDeepRunner, _executable_tool("executable", "amylodeep")),
}


def _construct(cls: type, options: dict[str, Any], *, key: str) -> BasePredictorRunner:
    """Instantiate ``cls`` with the options it actually declares.

    Options are filtered against the constructor signature rather than passed
    wholesale. Runner classes that swallow extras with ``**_ignored`` are
    filtered too, so an unrecognised key is reported once here instead of
    disappearing into a different class depending on which backend won.
    """
    parameters = inspect.signature(cls.__init__).parameters
    accepted = {name for name in parameters if name != "self"}
    accepted.discard("_ignored")
    # Keys the OTHER backend of this predictor understands are expected in a
    # shared [runners.*] section and are not worth a warning; only a key that no
    # implementation of this predictor accepts is likely a typo.
    known_to_any = set(accepted)
    for candidate in _BACKENDS.get(key, (None, None, None))[:2]:
        if candidate is not None:
            known_to_any |= {
                n for n in inspect.signature(candidate.__init__).parameters if n != "self"
            }
    known_to_any.discard("_ignored")
    dropped = sorted(set(options) - known_to_any - _BACKEND_KEYS)
    if dropped:
        warnings.warn(
            f"[runners.{key}] option(s) {dropped} are not accepted by "
            f"{cls.__name__} and were ignored. If this is the backend policy "
            f"working as intended (e.g. jar_path while backend resolved to web) "
            f"no action is needed; otherwise check the spelling.",
            stacklevel=3,
        )
    return cls(**{name: value for name, value in options.items() if name in accepted})


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
    if key == "path":
        options.setdefault(
            "threshold_percentile",
            predictor_options(key, cfg).get("threshold_percentile", 75.0),
        )
    if key == "appnn":
        options.setdefault(
            "score_threshold",
            predictor_options(key, cfg).get("score_threshold", 0.5),
        )
    if key == "pasta":
        options.setdefault(
            "energy_threshold",
            predictor_options(key, cfg).get("energy_threshold"),
        )
    if key == "archcandy":
        # Authoritative source is [predictors.archcandy]; a leftover value in
        # [runners.archcandy] is a duplicate of a modelling choice and is
        # reported rather than silently preferred.
        parser_mode = predictor_options(key, cfg).get("score_mode")
        runner_mode = options.get("score_mode")
        if parser_mode and runner_mode and parser_mode != runner_mode:
            warnings.warn(
                f"score_mode is set in BOTH [predictors.archcandy] "
                f"({parser_mode!r}) and [runners.archcandy] ({runner_mode!r}). "
                f"Using {parser_mode!r} so that aggressor-run and "
                f"aggressor-parse agree; delete the [runners.archcandy] copy.",
                stacklevel=2,
            )
        if parser_mode:
            options["score_mode"] = parser_mode
        else:
            options.setdefault("score_mode", runner_mode or "highest")
    if key == "crossbeta":
        options.setdefault(
            "confidence_threshold",
            predictor_options(key, cfg).get("confidence_threshold", 0.54),
        )
    if key == "aggreprot":
        options.setdefault(
            "aggregation_threshold",
            predictor_options(key, cfg).get("aggregation_threshold", 0.25),
        )
    if key in ("tango", "amylogram", "amylodeep"):
        options.setdefault(
            "threshold",
            predictor_options(key, cfg).get("threshold", 0.5),
        )
    options.update(overrides)

    cls = _resolve_class(key, options)
    options = _apply_backend_specific_options(key, cls, options)
    return _construct(cls, options, key=key)


def _apply_backend_specific_options(
    key: str, cls: type, options: dict[str, Any]
) -> dict[str, Any]:
    """Resolve options whose meaning depends on which backend won.

    ArchCandy is one predictor with two thresholds, because it is two programs:
    the web service is 2.0 and the downloadable JAR is 1.0, whose published
    cutoff is 0.56. A single ``threshold`` key cannot hold both, so the config
    carries ``local_threshold`` alongside it -- but nothing read it, and no
    runner declares it, so it was filtered out with a warning and the local
    backend silently ran at the WEB threshold. Nobody saw it because
    ``backend = auto`` finds no JAR on the analysis machine and resolves to web;
    it would have surfaced as a quiet 0.01 shift the first time the JAR existed.
    """
    if key != "archcandy":
        return options
    local_threshold = options.pop("local_threshold", None)
    if local_threshold is None:
        return options
    from aggressor_wrappers.runners.archcandy_local import ArchCandyLocalRunner

    if issubclass(cls, ArchCandyLocalRunner):
        options["threshold"] = float(local_threshold)
    return options


def _resolve_class(key: str, options: dict[str, Any]) -> type:
    """Pick the web or local implementation for ``key`` per the backend policy."""
    spec = _BACKENDS.get(key)
    if spec is None:
        return RUNNER_REGISTRY[key]
    web_cls, local_cls, tool_factory = spec
    local_tool = tool_factory(options) if (tool_factory and local_cls) else None
    resolved, _local_path, _reason = select_backend(
        key,
        local_tool=local_tool,
        options=options,
        has_web=web_cls is not None,
    )
    if resolved == "local":
        if local_cls is None:
            raise RuntimeError(f"{key}: no local implementation is registered")
        return local_cls
    if web_cls is None:
        raise RuntimeError(f"{key}: no web implementation is registered")
    return web_cls


def resolve_backend(key: str, *, config_path: str | None = None) -> tuple[str, str]:
    """Report which backend ``key`` would use, and why, without constructing it.

    Intended for a pre-flight check before a long panel run: it performs no
    network access and starts no subprocess.
    """
    key = resolve_predictor_key(key)
    cfg = load_config(config_path)
    options = {
        k: v for k, v in runner_options(key, cfg).items() if k not in _RUNNER_BATCH_KEYS
    }
    spec = _BACKENDS.get(key)
    if spec is None:
        return (
            "web" if key in _WEB_ONLY else "local"
        ), "no backend policy (single implementation)"
    web_cls, local_cls, tool_factory = spec
    local_tool = tool_factory(options) if (tool_factory and local_cls) else None
    resolved, _path, reason = select_backend(
        key, local_tool=local_tool, options=options, has_web=web_cls is not None
    )
    return resolved, reason


def list_runners() -> list[str]:
    return sorted(RUNNER_REGISTRY)
