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

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY, get_predictor_spec

TRACK_COLUMNS = ("Number", "Residue", "Score", "APR")

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


def parse_aggressor_track(path: Path, sequence: str = "") -> pd.DataFrame:
    """Parse a per-predictor track CSV written by :func:`export_protein`."""
    del sequence  # sequence is carried in config.yaml for amyloscope proteins
    df = pd.read_csv(path)
    missing = [col for col in TRACK_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path}: expected columns {TRACK_COLUMNS}, missing {missing}")
    out = df[list(TRACK_COLUMNS)].copy()
    out["Number"] = out["Number"].astype(int)
    out["Score"] = out["Score"].astype(float)
    out["Residue"] = out["Residue"].astype(str)
    return out.sort_values("Number").reset_index(drop=True)


def register_amyloscope_adapter() -> bool:
    """
    Register the ``aggressor`` track parser with amyloscope if installed.

    amyloscope does not ship this adapter yet; call once before ``amyloscope run``
    on configs emitted by :func:`export_run`.
    """
    try:
        from amyloscope.io.adapters import register_adapter
    except ImportError:
        return False

    @register_adapter("aggressor")
    def _parse_aggressor(path: Path, sequence: str = "") -> pd.DataFrame:
        return parse_aggressor_track(path, sequence)

    return True


def _protein_id_from_merged_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_merged"):
        return stem[: -len("_merged")]
    return stem


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
    lines.append("")
    lines.append("tools:")
    for key in predictors:
        spec = get_predictor_spec(key)
        lines.append(f"  - name: {spec.display_name}")
        lines.append("    adapter: aggressor")
        lines.append(f"    path: {out_root / 'tracks' / key}/{{protein}}.csv")
        lines.append(
            "    detection: {method: flag, column: APR, flag_true_values: [true]}"
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


def export_run(
    merged_dir: Path,
    out_root: Path,
    *,
    sequences: dict[str, str] | None = None,
    name: str = "aggressor-wrappers panel",
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
        protein_id = _protein_id_from_merged_path(csv)
        written = export_protein(csv, protein_id, out_root)
        predictors.update(written)
        df = pd.read_csv(csv)
        meta: dict = {"length": int(df["position"].max())}
        if sequences and protein_id in sequences:
            meta["sequence"] = sequences[protein_id]
        elif "aa_name" in df.columns:
            meta["sequence"] = "".join(df["aa_name"].astype(str))
        proteins[protein_id] = meta

    register_amyloscope_adapter()

    ordered = [k for k in PREDICTOR_REGISTRY if k in predictors]
    return build_amyloscope_config(proteins, ordered, out_root, name=name)
