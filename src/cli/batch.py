"""Multifasta pipeline CLI — ``aggressor-wrappers FASTA -o OUTPUT``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.core.config import default_pipeline_predictors, load_config
from aggressor_wrappers.predictors.registry import list_parsers
from aggressor_wrappers.runners.registry import list_runners

_SUBCOMMANDS = frozenset({"parse", "merge", "run", "widemerge", "batch"})


def is_subcommand(argv: list[str]) -> bool:
    return bool(argv) and argv[0] in _SUBCOMMANDS


def build_parser(*, prog: str = "aggressor-wrappers") -> argparse.ArgumentParser:
    available = sorted(set(list_runners()) | set(list_parsers()))
    default_predictors = ",".join(default_pipeline_predictors())
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run amyloid predictors for every sequence in a multifasta file, "
            "parse outputs, and write one wide merged CSV per protein. "
            "Predictor list and per-tool batching come from config.cfg by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_PIPELINE_EPILOG.format(default_predictors=default_predictors),
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="Input multifasta file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output root (creates per-predictor parsed/ and merged/ subdirs)",
    )
    parser.add_argument(
        "--predictors",
        help=(
            f"Comma-separated predictors (default from config.cfg: {default_predictors}). "
            f"Available: {', '.join(available)}"
        ),
    )
    parser.add_argument(
        "--config",
        help="Path to config.cfg (default: package config.cfg or AGGRESSOR_WRAPPERS_CONFIG)",
    )
    parser.add_argument(
        "--save-raw-files",
        type=Path,
        metavar="DIR",
        help=(
            "Optional directory to archive raw predictor outputs "
            "({protein_id}/{predictor}.*). Also used as input for parse-only tools."
        ),
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cache/ after run (default: remove cache/)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help=(
            "Do not execute external tools; parse existing raw from "
            "{PREDICTOR}/work/ or --save-raw-files archive"
        ),
    )
    return parser


def run_pipeline(argv: list[str] | None, *, prog: str = "aggressor-wrappers") -> int:
    try:
        args = build_parser(prog=prog).parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    if not args.fasta.is_file():
        print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
        return 2

    try:
        if args.predictors:
            predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
        else:
            predictors = default_pipeline_predictors(load_config(args.config))
        merged = run_multifasta_pipeline(
            args.fasta,
            args.output,
            predictors=predictors,
            config_path=args.config,
            skip_run=args.skip_run,
            save_raw_files=args.save_raw_files,
            keep_cache=args.keep_cache,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Finished {len(merged)} protein(s). Merged tables:")
    for protein_id, path in merged.items():
        print(f"  {protein_id}\t{path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``aggressor-wrappers`` and ``aggressor-wrappers.py``."""
    return run_pipeline(argv, prog="aggressor-wrappers")


if __name__ == "__main__":
    raise SystemExit(main())


_PIPELINE_EPILOG = """
primary usage:
  aggressor-wrappers proteins.fasta -o output_dir/
  python aggressor-wrappers.py proteins.fasta -o output_dir/
  python -m aggressor_wrappers proteins.fasta -o output_dir/

advanced (standalone tools):
  aggressor-parse waltz --input WaltzJob.txt --fasta protein.fasta -o out.csv
  aggressor-run path --fasta protein.fasta -o PATH.csv
  aggressor-merge parsed/*.csv -o merged.csv --fasta protein.fasta

output layout:
  output_dir/PATH/parsed/{{id}}_PATH.csv
  output_dir/APPNN/parsed/{{id}}_APPNN.csv
  output_dir/waltz/parsed/{{id}}_waltz.csv
  output_dir/pasta/parsed/{{id}}_pasta.csv
  output_dir/ArchCandy/parsed/{{id}}_ArchCandy.csv
  output_dir/merged/{{id}}_merged.csv

config.cfg defaults:
  [pipeline] predictors = {default_predictors}
  [runners.path] parallel_jobs = 2, sequences_per_run = 1
  [runners.appnn] all sequences in one R call
  [runners.waltz] sequences_per_run = 10 (web service)
  [runners.pasta] npair = 22, amount = -2.8, sequences_per_run = 10
  [runners.archcandy] parallel_jobs = 2, sequences_per_run = 1 (web service)
"""
