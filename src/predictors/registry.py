"""Predictor parser registry."""

from __future__ import annotations

from typing import Any

from aggressor_wrappers.core.config import load_config, predictor_options
from aggressor_wrappers.core.schema import resolve_predictor_key
from aggressor_wrappers.predictors.aggreprot import AggreProtParser
from aggressor_wrappers.predictors.aggrescan import AggrescanParser
from aggressor_wrappers.predictors.appnn import APPNNParser
from aggressor_wrappers.predictors.amylodeep import AmyloDeepParser
from aggressor_wrappers.predictors.archcandy import ArchCandyParser
from aggressor_wrappers.predictors.archcandy_v2 import ArchCandy2Parser
from aggressor_wrappers.predictors.base import BasePredictorParser
from aggressor_wrappers.predictors.crossbeta import CrossBetaParser
from aggressor_wrappers.predictors.foldamyloid import FoldAmyloidParser
from aggressor_wrappers.predictors.pasta import PASTAParser
from aggressor_wrappers.predictors.path import PATHParser
from aggressor_wrappers.predictors.waltz import WALTZParser

PARSER_REGISTRY: dict[str, type[BasePredictorParser]] = {
    "path": PATHParser,
    "appnn": APPNNParser,
    "waltz": WALTZParser,
    "pasta": PASTAParser,
    "aggreprot": AggreProtParser,
    "archcandy": ArchCandyParser,
    "archcandy2": ArchCandy2Parser,
    "crossbeta": CrossBetaParser,
    "foldamyloid": FoldAmyloidParser,
    "aggrescan": AggrescanParser,
    # Opt-in: registered so a pLM-based tool CAN be scored and attributed,
    # not so it votes by default. config.cfg decides the panel.
    "amylodeep": AmyloDeepParser,
}


def get_parser(
    name: str,
    *,
    config_path: str | None = None,
    **overrides: Any,
) -> BasePredictorParser:
    key = resolve_predictor_key(name)
    if key not in PARSER_REGISTRY:
        raise KeyError(f"No parser registered for {name!r}")

    cfg = load_config(config_path)
    options = predictor_options(key, cfg)
    options.update(overrides)
    return PARSER_REGISTRY[key](**options)


def list_parsers() -> list[str]:
    return sorted(PARSER_REGISTRY)
