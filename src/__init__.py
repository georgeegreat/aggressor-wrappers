"""AGGRESSOR predictor wrappers — unified parsing and aggregation."""

__version__ = "0.3.1"

from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.core.merge import merge_predictor_tables, write_merge_csv
from aggressor_wrappers.core.schema import PredictorResult, get_predictor_spec

__all__ = [
    "__version__",
    "PredictorResult",
    "get_predictor_spec",
    "load_config",
    "merge_predictor_tables",
    "write_merge_csv",
]
