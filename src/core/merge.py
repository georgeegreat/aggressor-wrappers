"""Merge per-predictor tables into unified outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aggressor_wrappers.core.schema import PREDICTOR_REGISTRY, PredictorResult, PredictorSpec


def merge_predictor_tables(
    results: list[PredictorResult],
    *,
    sort_by: str = "registry",
    include_aux: bool = False,
) -> pd.DataFrame:
    """
    Merge multiple ``PredictorResult`` objects on ``position``.

    Returns a wide table with columns:
    ``position``, ``aa_name``, ``{predictor}_score``, ``{predictor}_bin``, ...
    """
    if not results:
        raise ValueError("At least one PredictorResult is required")

    lengths = {r.length for r in results}
    if len(lengths) != 1:
        raise ValueError(f"Sequence length mismatch across predictors: {lengths}")

    sequences = {r.sequence for r in results}
    if len(sequences) != 1:
        raise ValueError("Sequences differ across predictors; cannot merge")

    if sort_by == "registry":
        order = {spec.key: idx for idx, spec in enumerate(PREDICTOR_REGISTRY.values())}
        results = sorted(results, key=lambda r: order.get(r.spec.key, 999))
    else:
        results = sorted(results, key=lambda r: r.spec.key)

    seen_keys = {r.spec.key for r in results}
    if len(seen_keys) != len(results):
        raise ValueError("Duplicate predictor in merge input")

    frames: list[pd.DataFrame] = []
    for index, result in enumerate(results):
        df = result.to_dataframe(include_aux=include_aux)
        if index == 0:
            frames.append(df[["position", "aa_name"]])
        keep = [result.spec.score_column, result.spec.bin_column]
        if include_aux:
            # Auxiliary per-residue channels the tool produced alongside its score
            # (AggreProt's sasa/transmembrane, ArchCandy's arch topology). Namespaced
            # by the parser, so they cannot collide across predictors.
            keep += [
                c for c in df.columns
                if c.startswith(f"{result.spec.key}_") and c not in keep
            ]
        frames.append(df[keep])

    return pd.concat(frames, axis=1)


def merge_standard_csv_files(
    csv_paths: list[str | Path],
    specs: list[PredictorSpec],
    *,
    protein_id: str = "protein",
    sequence: str | None = None,
) -> pd.DataFrame:
    """Load standard CSV files and merge them."""
    from aggressor_wrappers.core.schema import read_standard_csv

    if len(csv_paths) != len(specs):
        raise ValueError("csv_paths and specs must have the same length")

    results: list[PredictorResult] = []
    resolved_sequence = sequence

    for csv_path, spec in zip(csv_paths, specs, strict=True):
        result = read_standard_csv(
            csv_path,
            spec,
            protein_id=protein_id,
            sequence=resolved_sequence,
        )
        if resolved_sequence is None:
            resolved_sequence = result.sequence
        elif result.sequence != resolved_sequence:
            raise ValueError(
                f"{csv_path}: sequence length/content differs from earlier inputs"
            )
        results.append(result)

    return merge_predictor_tables(results)


def write_merge_csv(
    results: list[PredictorResult],
    output_path: str | Path,
) -> Path:
    """Write merged wide table (position, aa_name, all predictor columns)."""
    output_path = Path(output_path)
    merge_predictor_tables(results).to_csv(output_path, index=False)
    return output_path


# Backward-compatible alias
write_standard_merge_csv = write_merge_csv
