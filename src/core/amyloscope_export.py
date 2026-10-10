"""Export merged predictor tables as amyloscope inputs.

The two packages are complementary rather than overlapping, and the cleanest
integration keeps them that way:

* **aggressor-wrappers** is the *execution* layer — it drives the external
  predictors (web services, JARs, subprocesses), parses their heterogeneous
  native outputs, and binarises each with that tool's own threshold *and*
  direction.
* **amyloscope** is the *analysis* layer — detection rules, tiered
  cross-predictor consensus, APR calling, domain mapping, statistics, figures.

So the metascore does not need to be reimplemented inside this package at all.
What is needed is a handoff, and the natural one is the ``{predictor}_bin``
columns of the merged wide table: those calls were produced by each parser with
the correct rule and direction (``binary_from_scores(..., greater_or_equal=…)``,
which is ``False`` for PASTA), so they are scale-free and polarity-correct
evidence. Exporting them as a per-tool ``APR`` flag lets amyloscope apply its own
tiers (unanimous / strong / moderate), its own coverage rules, and its own
figures, on top of this package's execution.

Output layout, one file per (predictor, protein)::

    <out>/tracks/<predictor>/<protein_id>.csv     Number,Residue,Score,APR
    <out>/config.yaml                             ready to `amyloscope run`

``Score`` is carried through unchanged (native scale, useful for plotting the
per-tool profile), while ``APR`` is the binarised call amyloscope consumes via
``detection: {method: flag, column: APR, flag_true_values: [true]}`` — which is
why no per-tool threshold or polarity has to be restated in the amyloscope
config.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY, get_predictor_spec

#: This package's predictor key -> the name amyloscope knows the tool by.
#:
#: Three separate things were wrong before, and they compound. The config was
#: written with ``spec.display_name`` ("Aggrescan", "PATH"), while amyloscope keys
#: MECHANISM_FAMILIES and its adapter registry on lowercase canonical names -- so a
#: tool loaded under a display name is absent from the family composition that
#: decides whether a tier may move. Two keys also simply differ between the
#: packages: PASTA 2.0 is ``pasta`` here and ``pasta2`` there.
#:
#: An entry mapping to None means amyloscope has no counterpart. Those are REPORTED
#: and skipped rather than written with a name nothing recognises, because a config
#: naming an unknown tool fails late, per protein, after the export looked fine.
AMYLOSCOPE_TOOL_NAMES: dict[str, str | None] = {
    "aggrescan": "aggrescan",
    "foldamyloid": "foldamyloid",
    "appnn": "appnn",
    "waltz": "waltz",
    "tango": "tango",
    "amylogram": "amylogram",
    "amylodeep": "amylodeep",
    "aggreprot": "aggreprot",
    "archcandy": "archcandy",
    "crossbeta": "crossbeta",
    # PASTA 2.0: different spelling on each side.
    "pasta": "pasta2",
    # PATH has no adapter and no mechanism family in amyloscope. Carrying it would
    # name a tool the analysis layer cannot place, so it is skipped loudly.
    "path": None,
}

#: The adapter that reads a track this package wrote. One adapter, because every
#: track has the same normalised shape -- Number, Residue, Score, APR -- and the
#: per-tool threshold and direction were applied before it was written.
NORMALISED_ADAPTER = "aggressor"


DEFAULT_TIERS = [
    ("unanimous", 1.0, "#B2182B"),
    ("strong", 0.75, "#2166AC"),
    ("moderate", 0.5, "#FDAE61"),
]


def export_protein(
    merged_csv: Path,
    protein_id: str,
    out_root: Path,
    *,
    sequence: str | None = None,
) -> dict[str, Path]:
    """Split one merged wide table into per-predictor amyloscope tracks."""
    df = pd.read_csv(merged_csv)
    if "position" not in df.columns:
        raise ValueError(f"{merged_csv} has no 'position' column")
    if sequence and len(df) != len(sequence):
        # The config records the sequence; the tracks record the merged table. When
        # they disagree, amyloscope rejects the track per protein, per tool, deep in
        # a run -- 372 proteins later. Both numbers are known HERE, so this is where
        # it is caught.
        raise ValueError(
            f"{merged_csv}: {len(df)} rows for a {len(sequence)}-residue sequence "
            f"({protein_id}). The merged table and the FASTA describe different "
            f"proteins. Most often the FASTA has a duplicate header -- read_fasta "
            f"keeps the LAST record, so a repeated id silently replaces one sequence "
            f"with another of a different length -- or the merged table came from a "
            f"run against a different FASTA. Check with:\n"
            f"    grep -c '^>' <fasta>   # records\n"
            f"    grep '^>' <fasta> | sort | uniq -d   # repeated ids"
        )
    residues = df["aa_name"] if "aa_name" in df.columns else pd.Series([""] * len(df))

    written: dict[str, Path] = {}
    for key in PREDICTOR_REGISTRY:
        spec = get_predictor_spec(key)
        if spec.score_column not in df.columns:
            continue
        track = pd.DataFrame(
            {
                "Number": df["position"].astype(int),
                "Residue": residues,
                "Score": df[spec.score_column].astype(float),
            }
        )
        if spec.bin_column in df.columns:
            track["APR"] = (
                df[spec.bin_column].fillna(0).astype(float).clip(0, 1).astype(bool)
            )
        else:
            # No binarised call for this tool: leave APR false rather than
            # inventing a threshold here. amyloscope can still score it from
            # `Score` with an explicit `above`/`below` rule if desired.
            track["APR"] = False

        dest = out_root / "tracks" / key / f"{protein_id}.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        track.to_csv(dest, index=False)
        written[key] = dest
    return written


def build_amyloscope_config(
    proteins: dict[str, dict],
    predictors: list[str],
    out_root: Path,
    *,
    name: str = "aggressor-wrappers panel",
    min_region_length: int = 5,
    coverage_fraction: float = 0.5,
) -> Path:
    """Emit a ready-to-run amyloscope config for the exported tracks.

    Every tool is declared with the same ``flag``-on-``APR`` detection rule,
    because the binarisation (threshold *and* direction) already happened
    upstream in this package. That keeps a single source of truth for each
    predictor's calling rule instead of restating PASTA's inverted cutoff in a
    second config file, where it could silently drift.
    """
    lines: list[str] = []
    lines.append(f"name: {name}")
    lines.append(f"output_dir: {out_root / 'amyloscope_out'}")
    lines.append("")
    lines.append("proteins:")
    for pid, meta in proteins.items():
        lines.append(f"  - id: {pid}")
        if meta.get("display_name"):
            lines.append(f"    display_name: {meta['display_name']}")
        if meta.get("length"):
            lines.append(f"    length: {meta['length']}")
        if meta.get("sequence"):
            lines.append(f"    sequence: {meta['sequence']}")
        domains = meta.get("domains") or []
        if domains:
            lines.append("    domains:")
            for domain in domains:
                # The name is quoted: Pfam and InterPro descriptions contain colons
                # and commas ("KH domain, type 1"), which unquoted would make the
                # YAML parse as something else or fail outright.
                lines.append(
                    f'      - {{name: "{domain["name"]}", '
                    f'start: {domain["start"]}, stop: {domain["stop"]}, '
                    f'category: {domain["category"]}}}'
                )
    lines.append("")
    lines.append("tools:")
    unmapped: list[str] = []
    for key in predictors:
        name = AMYLOSCOPE_TOOL_NAMES.get(key, key)
        if name is None:
            unmapped.append(key)
            continue
        lines.append(f"  - name: {name}")
        lines.append(f"    adapter: {NORMALISED_ADAPTER}")
        lines.append(f"    path: {out_root / 'tracks' / key}/{{protein}}.csv")
        lines.append(
            "    detection: {method: flag, column: APR, flag_true_values: [true]}"
        )
    if unmapped:
        warnings.warn(
            f"amyloscope has no adapter or mechanism family for {unmapped}, so "
            f"{'they are' if len(unmapped) > 1 else 'it is'} absent from the "
            f"generated config. The exported tracks remain on disk under tracks/. "
            f"A fractional consensus counts the tools it loads, so the panel size "
            f"on that side is smaller than the one that ran here -- which changes "
            f"every tier.",
            RuntimeWarning,
            stacklevel=2,
        )
    lines.append("")
    lines.append("consensus:")
    lines.append("  tiers:")
    for tier, frac, colour in DEFAULT_TIERS:
        lines.append(
            f'    - {{name: {tier}, min_fraction: {frac}, color: "{colour}"}}'
        )
    lines.append(f"  min_region_length: {min_region_length}")
    lines.append(f"  coverage_fraction: {coverage_fraction}")
    # 'available' rather than 'configured': the consensus floor should track the
    # predictors that actually produced tracks. With a fixed denominator, a dead
    # web service (or a missing Modeller licence for PATH) silently raises the bar
    # and can make a tier mathematically unreachable for a protein.
    lines.append("  denominator: available")
    lines.append("")
    lines.append("viz:")
    lines.append("  language: en")
    lines.append("  detailed: true")
    lines.append("")

    dest = out_root / "config.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


#: Suffix the batch pipeline appends when it writes a merged table.
MERGED_SUFFIX = "_merged"


def protein_id_from_merged(path: Path) -> str:
    """Recover the protein id from a merged table's filename.

    ``batch/pipeline.py`` writes ``<merged>/<protein_id>_merged.csv``, so a bare
    ``Path.stem`` yields ``RPS2_merged`` -- which then becomes the track
    directory name, the key in the amyloscope config, and the label on every
    figure. Stripping the suffix is not cosmetic: amyloscope matches tracks to
    proteins by this id, so a mismatched one silently drops the protein from the
    analysis rather than failing.

    A stem without the suffix is returned unchanged, so a hand-assembled
    directory of ``<protein_id>.csv`` still works.
    """
    stem = Path(path).stem
    return stem[: -len(MERGED_SUFFIX)] if stem.endswith(MERGED_SUFFIX) else stem


def export_run(
    merged_dir: Path,
    out_root: Path,
    *,
    sequences: dict[str, str] | None = None,
    name: str = "aggressor-wrappers panel",
    domains=None,
) -> Path:
    """Export a whole batch run (``<output>/merged/*.csv``) for amyloscope."""
    merged_dir = Path(merged_dir)
    out_root = Path(out_root)
    csvs = sorted(merged_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No merged CSVs under {merged_dir}")

    proteins: dict[str, dict] = {}
    predictors: set[str] = set()
    for csv in csvs:
        pid = protein_id_from_merged(csv)
        written = export_protein(csv, pid, out_root)
        predictors.update(written)
        df = pd.read_csv(csv)
        meta: dict = {"length": int(df["position"].max())}
        if sequences and pid in sequences:
            meta["sequence"] = sequences[pid]
        elif "aa_name" in df.columns:
            meta["sequence"] = "".join(df["aa_name"].astype(str))

        # Domains are annotated PER SPECIES, in that orthologue's own numbering, so
        # they are attached per protein and clipped against that protein's own
        # length -- never projected from a reference. A hit past the end means the
        # annotation and the sequence are different proteins, and DomainSet drops
        # it rather than shortening it to fit.
        if domains is not None:
            length = len(meta["sequence"]) if meta.get("sequence") else None
            hits = domains.for_protein(pid, sequence_length=length)
            if hits:
                meta["domains"] = [
                    {
                        "name": hit.name,
                        "start": hit.start,
                        "stop": hit.stop,
                        "category": hit.category,
                    }
                    for hit in hits
                ]
        proteins[pid] = meta

    ordered = [k for k in PREDICTOR_REGISTRY if k in predictors]
    config = build_amyloscope_config(proteins, ordered, out_root, name=name)
    verify_export(out_root, ordered, proteins, config=config)
    return config


#: Columns amyloscope's `aggressor` adapter requires of every track.
REQUIRED_TRACK_COLUMNS = ("Number", "Residue", "Score", "APR")


def verify_export(
    out_root: Path,
    predictors: list[str],
    proteins: dict[str, dict],
    *,
    config: Path | None = None,
) -> None:
    """Read back what was just written and check it against what was promised.

    "Exported 372 proteins across 9 predictors" was a count of files created, not a
    statement that they can be loaded. The run that prompted this printed exactly
    that and then failed inside amyloscope on a track with no APR column -- a track
    left over from an EARLIER export, in a directory the writer happily added to.

    Three things are checked, each of which was wrong in a real run: every declared
    track exists, every track carries the columns the config tells amyloscope to
    read, and every track's length matches the sequence the config records for that
    protein. Any failure names the file, because the alternative is finding out one
    protein at a time.
    """
    problems: list[str] = []
    for key in predictors:
        if AMYLOSCOPE_TOOL_NAMES.get(key, key) is None:
            continue                      # deliberately absent from the config
        for protein_id, meta in proteins.items():
            track = out_root / "tracks" / key / f"{protein_id}.csv"
            if not track.is_file():
                problems.append(f"{track}: declared in the config but absent")
                continue
            header = pd.read_csv(track, nrows=0).columns.tolist()
            missing = [c for c in REQUIRED_TRACK_COLUMNS if c not in header]
            if missing:
                problems.append(
                    f"{track}: missing {missing} (has {header}). A track written by "
                    f"an older export, or by hand, will not load."
                )
                continue
            sequence = meta.get("sequence")
            if sequence:
                rows = sum(1 for _ in open(track, encoding="utf-8")) - 1
                if rows != len(sequence):
                    problems.append(
                        f"{track}: {rows} rows against a {len(sequence)}-residue "
                        f"configured sequence"
                    )
    if problems:
        shown = "\n  ".join(problems[:10])
        more = (
            f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else ""
        )
        raise ValueError(
            f"The export is not loadable by amyloscope ({len(problems)} "
            f"problem(s)):\n  {shown}{more}\n\n"
            f"A tracks/ directory that already held files from an earlier export is "
            f"the usual cause: this writer adds to it rather than replacing it, so "
            f"old and new tracks mix. Export into an empty directory, or pass "
            f"--clean."
        )
