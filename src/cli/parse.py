"""``aggressor-parse`` — convert raw predictor output to standard CSV."""

from __future__ import annotations

import argparse
import sys

from aggressor_wrappers.core.cache import clear_cache_dir, store_raw_cache
from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.predictors.registry import get_parser, list_parsers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-parse",
        description=(
            "Parse raw amyloid predictor output into a standard per-residue CSV "
            "(columns: position, aa_name, {Tool}_score, {Tool}_bin)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_PARSE_EPILOG,
    )
    parser.add_argument(
        "predictor",
        choices=list_parsers(),
        metavar="PREDICTOR",
        help="predictor to parse (%(choices)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output standard CSV path",
    )
    parser.add_argument(
        "--config",
        help="Path to config.cfg (default: package config.cfg or AGGRESSOR_WRAPPERS_CONFIG)",
    )
    parser.add_argument(
        "--protein-id",
        help="Protein identifier (default: first FASTA header)",
    )
    parser.add_argument(
        "--fasta",
        help="FASTA file supplying the amino-acid sequence",
    )
    parser.add_argument(
        "--sequence",
        help="Raw amino-acid sequence (alternative to --fasta)",
    )
    parser.add_argument(
        "--input",
        dest="source",
        help="Primary input file (predictor-specific raw output)",
    )
    parser.add_argument(
        "--results",
        help="PATH results.csv (alias for --input when predictor=path)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override config threshold for this run",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cache/{protein_id}/{predictor}/ after run (default: remove cache/)",
    )
    parser.add_argument(
        "--cache-dir",
        help="Override cache root directory from config",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    source = args.source or args.results
    if not source:
        print("Error: provide --input or --results", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    sequence, protein_id = _resolve_sequence_and_id(args)
    parser_kwargs = _parser_kwargs(args)
    predictor = get_parser(args.predictor, config_path=args.config, **parser_kwargs)

    result = predictor.parse(
        source,
        protein_id=protein_id,
        sequence=sequence,
    )
    result.to_csv(args.output)

    if args.keep_cache:
        cached = store_raw_cache(
            protein_id,
            args.predictor,
            source,
            config=cfg,
            cache_root=args.cache_dir,
            force=True,
        )
        if cached:
            print(f"Cached raw input → {cached}")
    else:
        removed = clear_cache_dir(cfg, args.cache_dir)
        if removed:
            print(f"[cleanup] removed {removed}")

    print(f"Wrote {args.output} ({result.length} residues, {result.spec.display_name})")
    return 0


def _resolve_sequence_and_id(args: argparse.Namespace) -> tuple[str, str]:
    if args.fasta:
        protein_id, sequence = read_first_sequence(args.fasta)
        if args.protein_id:
            protein_id = args.protein_id
        return sequence, protein_id

    if args.sequence:
        protein_id = args.protein_id or "protein"
        return args.sequence.upper().replace(" ", ""), protein_id

    raise SystemExit("Error: provide --fasta or --sequence")


def _parser_kwargs(args: argparse.Namespace) -> dict:
    if args.threshold is None:
        return {}

    mapping = {
        "path": "threshold_percentile",
        "pasta": "energy_threshold",
        "aggreprot": "aggregation_threshold",
        "crossbeta": "confidence_threshold",
        "appnn": "score_threshold",
    }
    key = mapping.get(args.predictor)
    return {key: args.threshold} if key else {}


if __name__ == "__main__":
    raise SystemExit(main())


_PARSE_EPILOG = """
predictors and raw inputs:
  path        PATH results.csv (--results or --input) + FASTA
  appnn       CSV from appnn_converter.R + FASTA
  waltz       WALTZ detailed text output (WaltzJob_*.txt) + FASTA
  pasta       PASTA per-residue energy profile (.dat, one value per line) + FASTA
  aggreprot   AggreProt CSV export + FASTA
  archcandy   region CSV (Start, Stop, Score) + FASTA
  crossbeta   CRBM JSON (AA_list / mean_confidence) + FASTA

examples:
  aggressor-parse path --results results.csv --fasta RPS2.fasta -o RPS2_PATH.csv
  aggressor-parse waltz --input WaltzJob.txt --fasta APP.fasta -o APP_waltz.csv
  aggressor-parse crossbeta --input result.json --fasta protein.fasta -o out.csv

  python -m aggressor_wrappers parse waltz --input WaltzJob.txt --fasta APP.fasta -o out.csv

notes:
  --fasta or --sequence is required.
  Raw inputs are cached only with --keep-cache (cache/ is removed by default after run).
  Thresholds default from config.cfg; override with --threshold.
"""

