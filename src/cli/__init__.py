"""CLI entry points."""

from aggressor_wrappers.cli.app import main as app_main
from aggressor_wrappers.cli.batch import main as batch_main
from aggressor_wrappers.cli.merge import main as merge_main
from aggressor_wrappers.cli.parse import main as parse_main
from aggressor_wrappers.cli.run import main as run_main
from aggressor_wrappers.cli.widemerge import main as widemerge_main

__all__ = ["app_main", "batch_main", "merge_main", "parse_main", "run_main", "widemerge_main"]
