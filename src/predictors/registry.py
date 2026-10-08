"""Predictor parser registry."""

from __future__ import annotations

from typing import Any

from aggressor_wrappers.core.config import load_config, predictor_options
from aggressor_wrappers.core.schema import resolve_predictor_key
from aggressor_wrappers.predictors.aggreprot import AggreProtParser
from aggressor_wrappers.predictors.aggrescan import AggrescanParser
from aggressor_wrappers.predictors.appnn import APPNNParser
from aggressor_wrappers.predictors.amylodeep import AmyloDeepParser
from aggressor_wrappers.predictors.amylogram import AmyloGramParser
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
    # AmyloGram's projection module and runner both predate this entry; without
    # it `batch.pipeline._merge_protein` gated the predictor out of every merged
    # table while the runner kept producing correct parsed CSVs for it. A tool
    # that fails loudly costs an afternoon; one that disappears costs the
    # consensus denominator, silently.
    "amylogram": AmyloGramParser,
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
    """Construct the parser for ``name``.

    An EXACT registry key wins over alias resolution. ``archcandy2`` is both a
    registry key (``ArchCandy2Parser``, which thresholds on the 2.0 score) and an
    alias of ``archcandy`` in ``PREDICTOR_ALIASES`` (so that filename inference
    and spec lookup land on the shared spec). Resolving the alias first made the
    2.0 parser unreachable through this function: asking for ``archcandy2``
    returned the 1.x parser, which applies no score threshold at all. Checking
    the literal name first keeps both behaviours -- the alias still does its job
    for spec lookup, and an explicit request for a versioned parser gets that
    version.
    """
    key = name if name in PARSER_REGISTRY else resolve_predictor_key(name)
    if key not in PARSER_REGISTRY:
        raise KeyError(
            f"No parser registered for {name!r}; known: {sorted(PARSER_REGISTRY)}"
        )

    cfg = load_config(config_path)
    # Options are keyed by the canonical predictor, so a versioned parser still
    # reads its family's [predictors.<name>] section.
    options = predictor_options(resolve_predictor_key(key), cfg)
    options.update(overrides)
    return PARSER_REGISTRY[key](**options)


def list_parsers() -> list[str]:
    return sorted(PARSER_REGISTRY)
