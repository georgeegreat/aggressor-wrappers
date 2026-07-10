#!/usr/bin/env python3
"""
Unified CLI for notebook parsers (ArchCandy, Cross-Beta, PASTA, AggreProt, WALTZ).

Replaces the per-function calls in ``arch_cross_pasta_aggreprot_waltz_parser.ipynb``.
Requires: ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import sys

from aggressor_wrappers.cli.parse import main as parse_main


PREDICTORS = ("archcandy", "crossbeta", "pasta", "aggreprot", "waltz")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictor", choices=PREDICTORS)
    parser.add_argument("--input", required=True, help="Raw predictor output file")
    parser.add_argument("-o", "--output", required=True, help="Standard CSV output")
    parser.add_argument("--fasta", help="FASTA with target sequence")
    parser.add_argument("--sequence", help="Amino-acid sequence (if no FASTA)")
    parser.add_argument("--protein-id", default="protein")
    parser.add_argument("--threshold", type=float, help="Predictor-specific threshold")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    forwarded = [
        args.predictor,
        "--input",
        args.input,
        "-o",
        args.output,
        "--protein-id",
        args.protein_id,
    ]
    if args.fasta:
        forwarded.extend(["--fasta", args.fasta])
    if args.sequence:
        forwarded.extend(["--sequence", args.sequence])
    if args.threshold is not None:
        forwarded.extend(["--threshold", str(args.threshold)])
    return parse_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
