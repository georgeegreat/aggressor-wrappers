"""Top-level CLI: ``aggressor-wrappers`` and ``python -m aggressor_wrappers``."""

from __future__ import annotations

import argparse
import sys

from aggressor_wrappers import __version__
from aggressor_wrappers.cli.batch import is_subcommand, run_pipeline


def build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-wrappers",
        description=(
            "AGGRESSOR companion package: run amyloidogenicity predictors on a multifasta, "
            "or dispatch to specialised subcommands (parse, merge, run, widemerge)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ROOT_EPILOG,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"aggressor-wrappers {__version__}",
    )
    return parser


def print_root_help() -> None:
    build_root_parser().print_help()


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])

    if not argv or argv == ["-h"] or argv == ["--help"]:
        print_root_help()
        return 0

    if argv[0] == "--version":
        build_root_parser().parse_args(argv)
        return 0

    if is_subcommand(argv):
        command, *rest = argv
        if command == "parse":
            from aggressor_wrappers.cli.parse import main as parse_main

            return parse_main(rest)
        if command == "merge":
            from aggressor_wrappers.cli.merge import main as merge_main

            return merge_main(rest)
        if command == "run":
            from aggressor_wrappers.cli.run import main as run_main

            return run_main(rest)
        if command == "widemerge":
            from aggressor_wrappers.cli.widemerge import main as widemerge_main

            return widemerge_main(rest)
        if command == "batch":
            return run_pipeline(rest, prog="aggressor-wrappers batch")

    return run_pipeline(argv, prog="aggressor-wrappers")


_ROOT_EPILOG = """
primary (multifasta pipeline — settings from config.cfg):
  aggressor-wrappers proteins.fasta -o output_dir/
  python -m aggressor_wrappers proteins.fasta -o output_dir/
  python aggressor-wrappers.py proteins.fasta -o output_dir/

subcommands (also as standalone entry points):
  parse      aggressor-parse       Raw predictor file → standard per-residue CSV
  merge      aggressor-merge       Standard CSVs → wide merged table
  run        aggressor-run         Execute PATH/APPNN → standard CSV
  widemerge  aggressor-widemerge   Merge + optional BHT reference validation
  batch      (alias)               Same as primary usage above

examples:
  aggressor-wrappers vendor/PATH/test.fasta -o output_dir
  python -m aggressor_wrappers parse --help
  aggressor-parse waltz --input WaltzJob.txt --fasta protein.fasta -o out.csv
  aggressor-merge parsed/*.csv -o merged.csv --fasta protein.fasta

configuration:
  config.cfg                    [pipeline] predictors, [runners.*] batching
  AGGRESSOR_WRAPPERS_CONFIG      environment variable for config path
"""


if __name__ == "__main__":
    raise SystemExit(main())
