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
        "--domains", type=Path, action="append", metavar="FILE",
        help=(
            "Per-species domain annotation for the orthologues: hmmscan "
            "--domtblout, an InterProScan TSV, or a table with protein_id, name, "
            "start, stop[, category]. The format is detected per file. Repeatable, "
            "so a curated table and a scan covering the proteins it does not reach "
            "can be combined without editing either -- the formats differ and their "
            "headers would not survive being concatenated. Annotations are used in "
            "each orthologue's OWN numbering and are never projected from a "
            "reference, so run the scan on the same multifasta the panel ran on:\n"
            "  hmmscan --domtblout domains.tbl --cut_ga Pfam-A.hmm proteins.faa"
        ),
    )
    parser.add_argument(
        "--domain-format", choices=["hmmscan", "interproscan", "table"],
        help="Override format detection for every --domains file.",
    )
    parser.add_argument(
        "--domain-overlaps",
        choices=("auto", "best", "all"),
        default="auto",
        help=(
            "How overlapping domains on one protein are reduced. 'auto' (default) "
            "follows the source: a hand-written table keeps every domain, because "
            "nesting in a curated set is deliberate, while a scan keeps the best "
            "hit per overlapping group, because several Pfam models over one "
            "region are redundancy. 'best' or 'all' forces one policy"
        ),
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
        "--primary-domain-category",
        help=(
            "The domain category the enrichment test is directed at, e.g. "
            "rna_binding. amyloscope reads it as the PRE-DECLARED claim; "
            "without it the enrichment test is one of several run over every "
            "category, which is a search rather than a hypothesis. It must be a "
            "category the --domains annotation actually uses"
        ),
    )
    parser.add_argument(
        "--hypothesis",
        help=(
            "Free text surfaced in amyloscope's statistics report, or @FILE to "
            "read it from a file. State the directional claim the primary "
            "category operationalises"
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
        from aggressor_wrappers.core.domains import (
            DEFAULT_KEEP,
            DomainSet,
            detect_format,
            load_domains,
        )

        merged_hits: dict[str, list] = {}
        formats: list[str] = []
        for path in args.domains:
            if not path.is_file():
                print(f"Error: domain file not found: {path}", file=sys.stderr)
                return 2
            try:
                by_protein = load_domains(path, fmt=args.domain_format)
                fmt = args.domain_format or detect_format(path)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
            formats.append(fmt)
            hits_here = sum(len(v) for v in by_protein.values())
            print(
                f"Domains: {hits_here} hit(s) over {len(by_protein)} protein(s) "
                f"from {path} ({fmt})",
                file=sys.stderr,
            )
            for protein_id, hits in by_protein.items():
                merged_hits.setdefault(protein_id, []).extend(hits)

        # The overlap policy is a property of the source, so a mixed set has no
        # single right answer: a scan's redundancy would be kept to preserve a
        # table's nesting. Mixed sources therefore keep everything and say so --
        # the alternative is dropping curated domains to tidy a scan.
        sources = sorted(set(formats))
        if args.domain_overlaps != "auto":
            keep = args.domain_overlaps
        elif len(sources) == 1:
            keep = DEFAULT_KEEP.get(sources[0], "best")
        else:
            keep = "all"
            print(
                f"Domains: {len(args.domains)} files of differing formats "
                f"({', '.join(sources)}); keeping every overlapping domain, since "
                f"one policy cannot be right for both. Pass --domain-overlaps to "
                f"force one.",
                file=sys.stderr,
            )

        domain_set = DomainSet(
            by_protein=merged_hits,
            source=", ".join(str(p) for p in args.domains),
            keep=keep,
        )
        counts = domain_set.coverage()
        print(
            f"Domains: {counts['total_hits']} hit(s) over "
            f"{counts['proteins_with_domains']} protein(s) in all "
            f"(overlapping domains: keep {keep})",
            file=sys.stderr,
        )
        if counts["proteins_with_domains"] == 0:
            print(
                "Warning: no protein in the domain file(s) matched. The ids must be "
                "the FASTA headers the panel ran on -- an EggNOG set is keyed "
                "taxid.proteinid, and a scan run on re-headered sequences will not "
                "line up.",
                file=sys.stderr,
            )

    # Checked here, from the DomainSet, so a mistyped category costs nothing:
    # export_run writes every track before it could notice.
    if args.primary_domain_category:
        if domain_set is None:
            print(
                "Error: --primary-domain-category needs --domains: the category "
                "has to be one the annotation uses.",
                file=sys.stderr,
            )
            return 2
        present = domain_set.categories()
        if args.primary_domain_category not in present:
            print(
                f"Error: --primary-domain-category "
                f"{args.primary_domain_category!r} is not a category "
                f"{', '.join(str(p) for p in args.domains)} uses. "
                f"Present: {', '.join(sorted(present))}",
                file=sys.stderr,
            )
            return 2

    hypothesis = args.hypothesis
    if hypothesis and hypothesis.startswith("@"):
        hypothesis_path = Path(hypothesis[1:])
        if not hypothesis_path.is_file():
            print(f"Error: hypothesis file not found: {hypothesis_path}", file=sys.stderr)
            return 2
        hypothesis = hypothesis_path.read_text(encoding="utf-8")

    try:
        config = export_run(
            merged_dir, args.output, sequences=sequences, name=args.name,
            domains=domain_set,
            hypothesis=hypothesis,
            primary_domain_category=args.primary_domain_category,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
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
