"""``aggressor-export-amyloscope`` -- merged tables to an amyloscope run directory.

The handoff between the two packages already existed as
``core.amyloscope_export.export_run``, tested but reachable only from Python. A
function nobody can invoke is an integration that works in pytest and not on a
desk, so this exposes it.

What crosses the boundary is deliberately narrow: per-tool ``Number, Residue,
Score, APR`` tracks plus a ready ``config.yaml``. ``APR`` is the call each parser
made with that tool's own threshold *and direction* (PASTA's energy is inverted),
so amyloscope applies its tiers to evidence that is already scale-free and
polarity-correct, and no per-tool threshold has to be restated on the far side.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from aggressor_wrappers.core.amyloscope_export import export_run
from aggressor_wrappers.core.fasta import read_fasta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggressor-export-amyloscope",
        description=(
            "Export a batch run's merged tables as an amyloscope run directory "
            "(per-predictor tracks plus config.yaml)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "merged_dir",
        type=Path,
        help=(
            "Directory of merged wide tables. This is <output>/merged/ from a "
            "batch run; a run's top-level output directory is also accepted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Directory to write the amyloscope run into (created if absent)",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        help=(
            "The multifasta the run used. Sequences are otherwise reconstructed "
            "from the merged tables' aa_name column, which is absent from a table "
            "whose predictors all omit residue identities."
        ),
    )
    parser.add_argument(
        "--domains", type=Path,
        help=(
            "Per-species domain annotation for the orthologues: hmmscan "
            "--domtblout, an InterProScan TSV, or a table with protein_id, name, "
            "start, stop[, category]. The format is detected from the file. "
            "Annotations are used in each orthologue's OWN numbering and are never "
            "projected from a reference, so run the scan on the same multifasta the "
            "panel ran on:\n"
            "  hmmscan --domtblout domains.tbl --cut_ga Pfam-A.hmm proteins.faa"
        ),
    )
    parser.add_argument(
        "--domain-format", choices=["hmmscan", "interproscan", "table"],
        help="Override format detection for --domains.",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help=(
            "Remove an existing tracks/ directory first. Without this the writer "
            "adds to whatever is there, so tracks from an earlier export -- written "
            "by an older version, with different columns -- mix with the new ones "
            "and fail inside amyloscope rather than here."
        ),
    )
    parser.add_argument(
        "--name",
        default="aggressor-wrappers panel",
        help="Panel name recorded in the generated config.yaml",
    )
    return parser


def _resolve_merged_dir(path: Path) -> Path:
    """Accept either <output>/ or <output>/merged/.

    Pointing at the run directory is the obvious mistake to make, and the
    failure it produces ("No merged CSVs under ...") names a directory that does
    contain them one level down. Cheaper to resolve than to explain.
    """
    if path.is_dir() and not any(path.glob("*.csv")) and (path / "merged").is_dir():
        return path / "merged"
    return path


def main(argv: list[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code in (0, None) else int(code)

    merged_dir = _resolve_merged_dir(args.merged_dir)
    if not merged_dir.is_dir():
        print(f"Error: not a directory: {merged_dir}", file=sys.stderr)
        return 2

    sequences = None
    if args.fasta:
        if not args.fasta.is_file():
            print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
            return 2
        sequences = read_fasta(args.fasta)

    tracks = args.output / "tracks"
    if tracks.is_dir() and any(tracks.iterdir()):
        if args.clean:
            shutil.rmtree(tracks)
            print(f"Removed existing tracks under {tracks}", file=sys.stderr)
        else:
            print(
                f"Error: {tracks} already contains tracks. This writer adds to a "
                f"directory rather than replacing it, so an earlier export's files "
                f"would mix with these -- which is how a track with no APR column "
                f"reaches amyloscope and fails there instead of here. Pass --clean, "
                f"or choose an empty output directory.",
                file=sys.stderr,
            )
            return 2

    domain_set = None
    if args.domains:
        from aggressor_wrappers.core.domains import DomainSet, load_domains

        if not args.domains.is_file():
            print(f"Error: domain file not found: {args.domains}", file=sys.stderr)
            return 2
        try:
            by_protein = load_domains(args.domains, fmt=args.domain_format)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        domain_set = DomainSet(by_protein=by_protein, source=str(args.domains))
        counts = domain_set.coverage()
        print(
            f"Domains: {counts['total_hits']} hit(s) over "
            f"{counts['proteins_with_domains']} protein(s) from {args.domains}",
            file=sys.stderr,
        )
        if counts["proteins_with_domains"] == 0:
            print(
                "Warning: no protein in the domain file matched. The ids must be the "
                "FASTA headers the panel ran on -- an EggNOG set is keyed "
                "taxid.proteinid, and a scan run on re-headered sequences will not "
                "line up.",
                file=sys.stderr,
            )

    try:
        config = export_run(
            merged_dir, args.output, sequences=sequences, name=args.name,
            domains=domain_set,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    track_root = args.output / "tracks"
    tracks = sorted(track_root.glob("*")) if track_root.is_dir() else []
    proteins = len(list(merged_dir.glob("*.csv")))
    print(
        f"Exported {proteins} protein(s) across {len(tracks)} predictor(s) -> {args.output}"
    )
    # `config` is POSITIONAL in amyloscope's CLI (p_run.add_argument("config")),
    # so the -c form printed before was not a command that runs.
    print(f"Run it with:  amyloscope run {config}")
    print(f"Check it first:  amyloscope validate {config}")
    return 0



_EPILOG = """
examples:
  aggressor-export-amyloscope output/merged -o amyloscope_run/
  aggressor-export-amyloscope output/ --fasta proteins.fasta -o amyloscope_run/
  aggressor-wrappers export-amyloscope output/merged -o amyloscope_run/

what is written:
  <out>/tracks/<predictor>/<protein_id>.csv   Number,Residue,Score,APR
  <out>/config.yaml                           ready for `amyloscope run`

why APR rather than a threshold:
  Each parser binarised with its tool's own cut AND direction, so the flag is
  scale-free and polarity-correct. amyloscope reads it with
  detection: {method: flag, column: APR}, and applies its own tiers on top.
"""

# Defined BELOW build_parser() but it must be BOUND before main() runs, so the
# __main__ guard goes last. With the guard above this string, `python -m
# aggressor_wrappers.cli.export_amyloscope` raised "NameError: name '_EPILOG'
# is not defined": running a module as __main__ executes it top to bottom, so
# main() was called before this assignment. The console script was unaffected
# -- an entry point IMPORTS the module (running it to the end) and then calls
# main() -- which is why the failure only appeared via `python -m`.
if __name__ == "__main__":
    raise SystemExit(main())
