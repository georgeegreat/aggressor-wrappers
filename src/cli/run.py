"""``aggressor-run`` — execute predictors and write standard CSV."""

from __future__ import annotations

import argparse

from aggressor_wrappers.core.cache import clear_cache_dir, store_raw_cache
from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.runners.registry import get_runner, list_runners


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-run",
        description=(
            "Run amyloid predictors (PATH, APPNN, WALTZ, PASTA, ArchCandy, Cross-Beta) and write standard "
            "per-residue CSV (position, aa_name, {Tool}_score, {Tool}_bin)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RUN_EPILOG,
    )
    parser.add_argument(
        "predictor",
        choices=list_runners(),
        metavar="PREDICTOR",
        help="predictor to run (%(choices)s)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output standard CSV path",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Input FASTA (single protein recommended for PATH)",
    )
    parser.add_argument(
        "--config",
        help="Path to config.cfg",
    )
    parser.add_argument(
        "--protein-id",
        help="Protein identifier (default: first FASTA header)",
    )
    parser.add_argument(
        "--work-dir",
        help="Working directory for tool output (default: FASTA parent dir for APPNN)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Parse existing raw output only (requires --results or --input)",
    )
    parser.add_argument(
        "--results",
        help="PATH results.csv (with --skip-run for path)",
    )
    parser.add_argument(
        "--input",
        dest="raw_input",
        help="Raw predictor output (APPNN CSV, WALTZ txt, PASTA profile; --skip-run)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep cache/{protein_id}/{predictor}/ after run (default: remove cache/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override parser threshold for this run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    cfg = load_config(args.config)
    runner_kwargs = _runner_kwargs(args)
    runner = get_runner(args.predictor, config_path=args.config, **runner_kwargs)

    run_kwargs: dict = {"fasta": args.fasta, "skip_run": args.skip_run}
    if args.protein_id:
        run_kwargs["protein_id"] = args.protein_id
    if args.work_dir:
        run_kwargs["work_dir"] = args.work_dir
    if args.predictor == "path" and args.results:
        run_kwargs["results_csv"] = args.results
    if args.predictor == "appnn" and args.raw_input:
        run_kwargs["raw_csv"] = args.raw_input
    if args.predictor == "waltz" and args.raw_input:
        run_kwargs["raw_txt"] = args.raw_input
    if args.predictor == "pasta" and args.raw_input:
        run_kwargs["raw_profile"] = args.raw_input
    if args.predictor == "archcandy" and args.raw_input:
        run_kwargs["raw_csv"] = args.raw_input
    if args.predictor == "crossbeta" and args.raw_input:
        run_kwargs["raw_json"] = args.raw_input

    result = runner.run(**run_kwargs)
    result.to_csv(args.output)

    if args.keep_cache:
        raw_source = args.results or args.raw_input or getattr(runner, "last_raw_path", None)
        if raw_source:
            cached = store_raw_cache(
                result.protein_id,
                args.predictor,
                raw_source,
                config=cfg,
                force=True,
            )
            if cached:
                print(f"Cached raw input → {cached}")
    else:
        removed = clear_cache_dir(cfg)
        if removed:
            print(f"[cleanup] removed {removed}")

    print(f"Wrote {args.output} ({result.length} residues, {result.spec.display_name})")
    return 0


def _runner_kwargs(args: argparse.Namespace) -> dict:
    if args.threshold is None:
        return {}

    mapping = {
        "path": {"threshold_percentile": args.threshold},
        "appnn": {"score_threshold": args.threshold},
        "crossbeta": {"confidence_threshold": args.threshold},
    }
    return mapping.get(args.predictor, {})


if __name__ == "__main__":
    raise SystemExit(main())


_RUN_EPILOG = """
examples:
  aggressor-run appnn --fasta RPS2.fasta -o RPS2_APPNN.csv
  aggressor-run path --fasta RPS2.fasta -o RPS2_PATH.csv --work-dir ./path_out
  aggressor-run waltz --fasta APP.fasta -o APP_waltz.csv
  aggressor-run pasta --fasta APP.fasta -o APP_pasta.csv

  aggressor-run archcandy --fasta APP.fasta -o APP_ArchCandy.csv
  aggressor-run crossbeta --fasta RPL27.fasta -o RPL27_crossbeta.csv

  # Parse precomputed raw files (no external tool execution):
  aggressor-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
  aggressor-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv
  aggressor-run waltz --skip-run --input WaltzJob.txt --fasta APP.fasta -o out.csv
  aggressor-run pasta --skip-run --input APP_pasta.dat --fasta APP.fasta -o out.csv
  aggressor-run archcandy --skip-run --input APP_archcandy.csv --fasta APP.fasta -o out.csv
  aggressor-run crossbeta --skip-run --input RPL27_crossbeta.json --fasta RPL27.fasta -o out.csv

configuration ([runners.*] in config.cfg):
  path.script / path.python     vendor/PATH/path1.1.py (Modeller + PyRosetta)
  appnn.converter_script        legacy/appnn_converter.R
  waltz.base_url                https://waltz.switchlab.org/
  pasta.base_url                http://old.protein.bio.unipd.it/pasta2/
  archcandy.base_url            https://bioinfo.crbm.cnrs.fr/
  crossbeta.base_url            https://bioinfo.crbm.cnrs.fr/

notes:
  PATH threading is slow — configure timeout_seconds or use --skip-run for tests.
  APPNN requires R with the 'appnn' CRAN package installed.
  WALTZ, PASTA, ArchCandy, and Cross-Beta submit jobs to public web servers (no local install).
  ArchCandy and Cross-Beta accept one sequence per job (see [runners.archcandy] / [runners.crossbeta]).
"""
