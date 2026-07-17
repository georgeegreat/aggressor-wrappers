"""Metascore computation — dispatches to ``core.metascore_plugins``."""

from __future__ import annotations

import pandas as pd

from aggressor_wrappers.core.config import AppConfig, load_config
from aggressor_wrappers.core.metascore_plugins import compute_metascore


def metascore_table(
    wide_df: pd.DataFrame,
    *,
    column_name: str = "metascore",
    config: AppConfig | None = None,
    method: str | None = None,
) -> pd.DataFrame:
    """Attach a metascore column to a wide merge table."""
    out = wide_df.copy()
    out[column_name] = compute_metascore(out, config=config, method=method)
    return out
