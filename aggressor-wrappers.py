#!/usr/bin/env python3
"""
AGGRESSOR wrappers — multifasta pipeline entry point.

Equivalent to:
  aggressor-wrappers FASTA -o OUTPUT
  python -m aggressor_wrappers FASTA -o OUTPUT
  python aggressor-wrappers.py FASTA -o OUTPUT

Runs all predictors listed in config.cfg ``[pipeline] predictors`` (default: path,appnn),
with per-runner batching from ``[runners.*]``.

Examples:
  python aggressor-wrappers.py vendor/PATH/test.fasta -o output_dir
  python aggressor-wrappers.py proteins.fasta -o output_dir --predictors path,appnn
  python aggressor-wrappers.py proteins.fasta -o output_dir --skip-run --save-raw-files ./raw/
"""

from __future__ import annotations

import sys

from aggressor_wrappers.cli.batch import run_pipeline

if __name__ == "__main__":
    raise SystemExit(run_pipeline(sys.argv[1:], prog="aggressor-wrappers.py"))
