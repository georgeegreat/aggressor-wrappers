"""``aggressor-widemerge`` — merge standard CSVs and optionally validate vs BHT reference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.core.inputs import expand_csv_inputs, guess_predictor_from_filename
from aggressor_wrappers.core.merge import merge_predictor_tables
from aggressor_wrappers.core.schema import get_predictor_spec, read_standard_csv
from aggressor_wrappers.core.validate import compare_score_columns, load_reference_table
from aggressor_wrappers.paths import bht_reference_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-widemerge",
        description=(
            "Merge standard per-predictor CSV files into one wide table and "
            "optionally compare score columns against a BHT reference table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_WIDEMERGE_EPILOG,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Standard CSV files (or combine with --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory with per-predictor standard CSV files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output wide CSV path",
    )
    parser.add_argument(
        "--fasta",
        help="FASTA for sequence validation",
    )
    parser.add_argument(
        "--protein-id",
        default="protein",
        help="Protein identifier stored in metadata",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="BHT reference table (*_all.csv or wide CSV) for score regression check",
    )
    parser.add_argument(
        "--reference-name",
        default="RPS2_human_all.csv",
        help="Default filename when --check-bht-reference is set",
    )
    parser.add_argument(
        "--check-bht-reference",
        action="store_true",
        help="Compare against BHT_amyloid/all/{--reference-name}",
    )
    parser.add_argument(
        "--predictor",
        action="append",
        dest="predictors",
        help="Predictor key for the next input (auto-detected if omitted)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for reference comparison",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-9,
        help="Absolute tolerance for reference comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    if args.input_dir:
        try:
            inputs = expand_csv_inputs([args.input_dir])
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    elif args.inputs:
        try:
            inputs = expand_csv_inputs(args.inputs)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        print("Error: provide --input-dir or positional CSV paths", file=sys.stderr)
        return 2

    if args.predictors and len(args.predictors) != len(inputs):
        print("Error: --predictor count must match number of input files", file=sys.stderr)
        return 2

    sequence = None
    if args.fasta:
        _, sequence = read_first_sequence(args.fasta)

    results = []
    for path, predictor_key in zip(
        inputs,
        args.predictors or [None] * len(inputs),
        strict=True,
    ):
        try:
            key = predictor_key or guess_predictor_from_filename(path)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        spec = get_predictor_spec(key)
        results.append(
            read_standard_csv(
                path,
                spec,
                protein_id=args.protein_id,
                sequence=sequence,
            )
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_predictor_tables(results)
    merged.to_csv(output, index=False)
    print(f"Merged {len(results)} predictors → {output}")

    reference_path = args.reference
    if reference_path is None and args.check_bht_reference:
        reference_path = bht_reference_root() / "all" / f"{args.protein_id}_all.csv"
        if not reference_path.is_file():
            alt = bht_reference_root() / "all" / args.reference_name
            reference_path = alt if alt.is_file() else reference_path

    if reference_path is None:
        return 0

    ref_path = Path(reference_path)
    if not ref_path.is_file():
        print(f"Warning: reference not found: {ref_path}", file=sys.stderr)
        return 0

    reference = load_reference_table(ref_path)
    mismatches = compare_score_columns(
        merged,
        reference,
        rtol=args.rtol,
        atol=args.atol,
    )
    if mismatches:
        print("Reference comparison FAILED:", file=sys.stderr)
        for col, delta in sorted(mismatches.items()):
            if delta != delta:
                print(f"  {col}: missing in merged output", file=sys.stderr)
            else:
                print(f"  {col}: max |Δ| = {delta:.6g}", file=sys.stderr)
        return 1

    compared = [c for c in reference.columns if c.endswith("_score") and c in merged.columns]
    print(f"Reference OK ({len(compared)} score columns vs {ref_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


_WIDEMERGE_EPILOG = """
examples:
  aggressor-widemerge --input-dir parsed/ --fasta RPS2.fasta -o merged/RPS2_merged.csv
  aggressor-widemerge parsed/*.csv --fasta RPS2.fasta -o out.csv \\
      --reference ../all/RPS2_human_all.csv

  aggressor-widemerge --input-dir parsed/ -o out.csv --check-bht-reference \\
      --reference-name RPS2_human_all.csv

difference vs aggressor-merge:
  aggressor-merge          fast merge only
  aggressor-widemerge     merge + optional BHT reference score validation

notes:
  Inputs must describe the same protein sequence (--fasta validates this).
  Predictor type is inferred from filenames; use --predictor to force (once per file).
"""
