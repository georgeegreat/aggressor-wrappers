"""``aggressor-merge`` — combine standard CSVs into a wide table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aggressor_wrappers.core.inputs import expand_csv_inputs, guess_predictor_from_filename
from aggressor_wrappers.core.merge import write_merge_csv
from aggressor_wrappers.core.schema import get_predictor_spec, read_standard_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-merge",
        description=(
            "Merge standard per-predictor CSV files into one wide table: "
            "position, aa_name, {predictor}_score, {predictor}_bin, ..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_MERGE_EPILOG,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Standard CSV files (or directories containing them)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--protein-id",
        default="protein",
        help="Protein identifier stored in metadata",
    )
    parser.add_argument(
        "--fasta",
        help="Optional FASTA to validate / supply sequence",
    )
    parser.add_argument(
        "--predictor",
        action="append",
        dest="predictors",
        help="Predictor key for the next input (auto-detected from filename if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    try:
        input_paths = expand_csv_inputs(args.inputs)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.predictors and len(args.predictors) != len(input_paths):
        print("Error: --predictor count must match number of input files", file=sys.stderr)
        return 2

    specs = []
    for key, path in zip(args.predictors or [None] * len(input_paths), input_paths, strict=True):
        try:
            specs.append(get_predictor_spec(key or guess_predictor_from_filename(path)))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    sequence = None
    if args.fasta:
        from aggressor_wrappers.core.fasta import read_first_sequence

        _, sequence = read_first_sequence(args.fasta)

    results = [
        read_standard_csv(path, spec, protein_id=args.protein_id, sequence=sequence)
        for path, spec in zip(input_paths, specs, strict=True)
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_merge_csv(results, output)

    print(f"Merged {len(results)} predictors → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


_MERGE_EPILOG = """
examples:
  aggressor-merge RPS2_PATH.csv RPS2_APPNN.csv -o RPS2_merged.csv --fasta RPS2.fasta
  aggressor-merge parsed/*.csv -o merged.csv --fasta protein.fasta

output columns:
  position, aa_name, {predictor}_score, {predictor}_bin, ...

notes:
  Inputs must describe the same protein sequence (--fasta validates this).
  Predictor type is inferred from filenames; use --predictor to force (once per file).
"""
