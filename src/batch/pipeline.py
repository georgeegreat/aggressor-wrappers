"""Orchestrate multifasta runs across predictors with progress logging."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from aggressor_wrappers.core.config import load_config, runner_batch_config
from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.merge import merge_predictor_tables
from aggressor_wrappers.core.schema import get_predictor_spec, read_standard_csv
from aggressor_wrappers.predictors.registry import get_parser, list_parsers
from aggressor_wrappers.runners.registry import get_runner, list_runners

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class BatchLayout:
    output_dir: Path
    merged_dir: Path
    fasta_split_dir: Path

    @classmethod
    def create(cls, output_dir: Path) -> BatchLayout:
        return cls(
            output_dir=output_dir,
            merged_dir=output_dir / "merged",
            fasta_split_dir=output_dir / ".tmp" / "fasta_split",
        )

    def predictor_dir(self, runner_key: str) -> Path:
        """Per-predictor root, e.g. ``{output}/PATH`` or ``{output}/APPNN``."""
        return self.output_dir / _predictor_tag(runner_key)

    def predictor_parsed_dir(self, runner_key: str) -> Path:
        return self.predictor_dir(runner_key) / "parsed"

    def predictor_work_dir(self, runner_key: str) -> Path:
        return self.predictor_dir(runner_key) / "work"

    def ensure(self, runner_keys: Iterable[str] | None = None) -> None:
        paths = [self.output_dir, self.merged_dir, self.fasta_split_dir]
        for key in runner_keys or ():
            paths.extend(
                (
                    self.predictor_dir(key),
                    self.predictor_parsed_dir(key),
                    self.predictor_work_dir(key),
                )
            )
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)


def chunk_items(items: list[tuple[str, str]], chunk_size: int) -> list[list[tuple[str, str]]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def write_fasta_records(records: dict[str, str], path: Path) -> None:
    lines: list[str] = []
    for protein_id, sequence in records.items():
        lines.append(f">{protein_id}")
        lines.append(sequence)
    path.write_text("\n".join(lines) + "\n")


def split_multifasta(multifasta: Path, layout: BatchLayout) -> dict[str, str]:
    sequences = read_fasta(multifasta)
    for protein_id, sequence in sequences.items():
        write_fasta_records({protein_id: sequence}, layout.fasta_split_dir / f"{protein_id}.fasta")
    return sequences


def _default_log(message: str) -> None:
    print(message, flush=True)


def _predictor_tag(key: str) -> str:
    return get_predictor_spec(key).score_column.replace("_score", "")


def run_multifasta_pipeline(
    multifasta: Path,
    output_dir: Path,
    *,
    predictors: Iterable[str] | None = None,
    config_path: str | None = None,
    skip_run: bool = False,
    save_raw_files: Path | None = None,
    keep_cache: bool = False,
    log: LogFn | None = None,
) -> dict[str, Path]:
    """
    Run available predictors for every sequence in ``multifasta``.

    Per-runner batching is controlled by ``[runners.*]`` in config.cfg
    (``parallel_jobs``, ``sequences_per_run``).

    Returns ``{protein_id: merged_csv_path}``.
    """
    emit = log or _default_log
    load_config(config_path)
    layout = BatchLayout.create(output_dir)

    requested = _normalise_predictors(predictors, config_path)
    _validate_predictors(requested)
    runner_keys = [key for key in requested if key in list_runners()]
    parse_only_keys = [key for key in requested if key in list_parsers() and key not in list_runners()]

    layout.ensure(runner_keys)
    if save_raw_files is not None:
        save_raw_files.mkdir(parents=True, exist_ok=True)

    emit(f"[setup] loading {multifasta}")
    sequences = split_multifasta(multifasta, layout)
    protein_ids = list(sequences.keys())
    emit(f"[setup] {len(protein_ids)} protein(s); output → {layout.output_dir}")

    if skip_run:
        work_roots = ", ".join(str(layout.predictor_work_dir(k)) for k in runner_keys)
        emit(f"[setup] --skip-run: parsing raw files from {work_roots}")

    items = list(sequences.items())

    for runner_key in runner_keys:
        _run_runner_batches(
            runner_key,
            items,
            layout,
            config_path=config_path,
            skip_run=skip_run,
            save_raw_files=save_raw_files,
            emit=emit,
        )

    if parse_only_keys:
        _run_parse_only(
            parse_only_keys,
            protein_ids,
            layout,
            save_raw_files=save_raw_files,
            config_path=config_path,
            emit=emit,
        )

    merged_paths: dict[str, Path] = {}
    for protein_id in protein_ids:
        merged_paths[protein_id] = _merge_protein(
            protein_id,
            layout,
            requested,
            emit=emit,
        )

    emit(f"[done] merged tables for {len(merged_paths)} protein(s) under {layout.merged_dir}")

    if not skip_run:
        _cleanup_temp_dirs(layout, runner_keys, emit=emit)

    if not keep_cache:
        _cleanup_cache_dir(config_path, emit=emit)

    return merged_paths


def _normalise_predictors(
    predictors: Iterable[str] | None,
    config_path: str | None = None,
) -> list[str]:
    if predictors is None:
        cfg = load_config(config_path)
        return list(cfg.pipeline.predictors)
    keys: list[str] = []
    for item in predictors:
        for part in item.split(","):
            part = part.strip().lower()
            if part:
                keys.append(part)
    if not keys:
        raise ValueError("No predictors selected")
    return keys


def _validate_predictors(requested: list[str]) -> None:
    known = set(list_runners()) | set(list_parsers())
    unknown = [key for key in requested if key not in known]
    if unknown:
        available = ", ".join(sorted(known))
        raise ValueError(f"Unknown predictor(s): {', '.join(unknown)}. Available: {available}")


def _chunk_size_for_runner(runner_key: str, item_count: int, config_path: str | None) -> int:
    batch_cfg = runner_batch_config(runner_key, load_config(config_path))
    if batch_cfg.sequences_per_run == 0:
        return item_count
    return batch_cfg.sequences_per_run


def _run_runner_batches(
    runner_key: str,
    items: list[tuple[str, str]],
    layout: BatchLayout,
    *,
    config_path: str | None,
    skip_run: bool,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    runner = get_runner(runner_key, config_path=config_path)
    tag = _predictor_tag(runner_key)
    batch_cfg = runner_batch_config(runner_key, load_config(config_path))
    chunk_size = _chunk_size_for_runner(runner_key, len(items), config_path)
    batches = chunk_items(items, chunk_size)
    total_batches = len(batches)

    if runner_key == "path" and batch_cfg.parallel_jobs > 1 and not skip_run:
        emit(
            f"[{tag}] parallel_jobs={batch_cfg.parallel_jobs}, "
            f"sequences_per_run={batch_cfg.sequences_per_run or 'all'}"
        )
        _run_path_parallel(
            runner,
            batches,
            layout,
            parallel_jobs=batch_cfg.parallel_jobs,
            save_raw_files=save_raw_files,
            emit=emit,
        )
        return

    if runner_key == "archcandy" and batch_cfg.parallel_jobs > 1 and not skip_run:
        emit(
            f"[{tag}] parallel_jobs={batch_cfg.parallel_jobs}, "
            f"sequences_per_run={batch_cfg.sequences_per_run or 'all'}"
        )
        _run_archcandy_parallel(
            runner,
            batches,
            layout,
            parallel_jobs=batch_cfg.parallel_jobs,
            save_raw_files=save_raw_files,
            emit=emit,
        )
        return

    for batch_index, batch in enumerate(batches, start=1):
        batch_label = f"{batch_index}/{total_batches}"
        batch_ids = [protein_id for protein_id, _ in batch]
        work_root = layout.predictor_work_dir(runner_key)
        batch_fasta = work_root / f"batch_{batch_index}.fasta"
        write_fasta_records(dict(batch), batch_fasta)
        batch_work = work_root / f"batch_{batch_index}"

        if runner_key == "appnn":
            raw_map = _run_appnn_batch(
                runner,
                batch_fasta,
                batch_work,
                batch_ids,
                skip_run=skip_run,
                save_raw_files=save_raw_files,
                batch_label=batch_label,
                emit=emit,
            )
        elif runner_key == "waltz":
            raw_map = _run_waltz_batch(
                runner,
                batch_fasta,
                batch_work,
                batch_ids,
                skip_run=skip_run,
                save_raw_files=save_raw_files,
                batch_label=batch_label,
                emit=emit,
            )
        elif runner_key == "pasta":
            raw_map = _run_pasta_batch(
                runner,
                batch_fasta,
                batch_work,
                batch_ids,
                skip_run=skip_run,
                save_raw_files=save_raw_files,
                batch_label=batch_label,
                emit=emit,
            )
        elif runner_key == "archcandy":
            raw_map = _run_archcandy_batch(
                runner,
                batch_fasta,
                batch_work,
                batch_ids,
                skip_run=skip_run,
                save_raw_files=save_raw_files,
                batch_label=batch_label,
                emit=emit,
            )
        elif runner_key == "path":
            raw_map = _run_path_batch(
                runner,
                batch_fasta,
                batch_work,
                batch_ids,
                skip_run=skip_run,
                save_raw_files=save_raw_files,
                batch_label=batch_label,
                emit=emit,
            )
        else:
            raise RuntimeError(f"Batch runner not implemented for {runner_key!r}")

        _parse_and_write_runner_batch(
            runner,
            runner_key,
            batch_ids,
            raw_map,
            layout,
            batch_work,
            batch_label=batch_label,
            skip_run=skip_run,
            save_raw_files=save_raw_files,
            emit=emit,
        )


def _run_archcandy_parallel(
    runner,
    batches: list[list[tuple[str, str]]],
    layout: BatchLayout,
    *,
    parallel_jobs: int,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    tag = _predictor_tag("archcandy")
    total = len(batches)
    work_root = layout.predictor_work_dir("archcandy")

    def _job(batch_index: int, batch: list[tuple[str, str]]) -> None:
        if len(batch) != 1:
            raise RuntimeError(
                f"[{tag}] job {batch_index}/{total}: expected one sequence per job, "
                f"got {len(batch)} (check [runners.archcandy] sequences_per_run)"
            )
        batch_label = f"{batch_index}/{total}"
        protein_id = batch[0][0]
        batch_work = work_root / protein_id
        single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
        emit(f"[{tag}] job {batch_label}: submitting {protein_id} …")
        raw_csv = runner.execute(single_fasta, batch_work)
        emit(f"[{tag}] job {batch_label}: raw results → {raw_csv}")
        result = runner.run(
            fasta=single_fasta,
            protein_id=protein_id,
            work_dir=batch_work,
            skip_run=True,
            raw_csv=raw_csv,
        )
        out_csv = layout.predictor_parsed_dir("archcandy") / f"{protein_id}_{tag}.csv"
        result.to_csv(out_csv)
        emit(f"[{tag}] job {batch_label}: wrote {out_csv}")
        if save_raw_files is not None:
            archived = _archive_raw_file(
                raw_csv,
                save_raw_files,
                protein_id=protein_id,
                predictor_key="archcandy",
            )
            emit(f"[{tag}] archived raw → {archived}")

    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = [
            executor.submit(_job, batch_index, batch)
            for batch_index, batch in enumerate(batches, start=1)
        ]
        for future in as_completed(futures):
            future.result()


def _run_path_parallel(
    runner,
    batches: list[list[tuple[str, str]]],
    layout: BatchLayout,
    *,
    parallel_jobs: int,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    tag = _predictor_tag("path")
    total = len(batches)
    work_root = layout.predictor_work_dir("path")

    def _job(batch_index: int, batch: list[tuple[str, str]]) -> None:
        batch_label = f"{batch_index}/{total}"
        protein_id = batch[0][0]
        batch_work = work_root / protein_id
        single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
        emit(f"[{tag}] job {batch_label}: generating output for {protein_id} …")
        results_csv = runner.execute(single_fasta, batch_work)
        emit(f"[{tag}] job {batch_label}: raw results → {results_csv}")
        result = runner.run(
            fasta=single_fasta,
            protein_id=protein_id,
            work_dir=batch_work,
            skip_run=True,
            results_csv=results_csv,
        )
        out_csv = layout.predictor_parsed_dir("path") / f"{protein_id}_{tag}.csv"
        result.to_csv(out_csv)
        emit(f"[{tag}] job {batch_label}: wrote {out_csv}")
        if save_raw_files is not None:
            archived = _archive_raw_file(
                results_csv,
                save_raw_files,
                protein_id=protein_id,
                predictor_key="path",
            )
            emit(f"[{tag}] archived raw → {archived}")

    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = [
            executor.submit(_job, batch_index, batch)
            for batch_index, batch in enumerate(batches, start=1)
        ]
        for future in as_completed(futures):
            future.result()


def _parse_and_write_runner_batch(
    runner,
    runner_key: str,
    batch_ids: list[str],
    raw_map: dict[str, Path],
    layout: BatchLayout,
    batch_work: Path,
    *,
    batch_label: str,
    skip_run: bool,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    tag = _predictor_tag(runner_key)
    for protein_id in batch_ids:
        raw_csv = raw_map[protein_id]
        single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
        emit(f"[{tag}] batch {batch_label}: parsing {protein_id} …")
        result = runner.run(
            fasta=single_fasta,
            protein_id=protein_id,
            work_dir=batch_work,
            skip_run=True,
            results_csv=raw_csv if runner_key == "path" else None,
            raw_csv=raw_csv if runner_key in {"appnn", "archcandy"} else None,
            raw_txt=raw_csv if runner_key == "waltz" else None,
            raw_profile=raw_csv if runner_key == "pasta" else None,
        )
        out_csv = layout.predictor_parsed_dir(runner_key) / f"{protein_id}_{tag}.csv"
        result.to_csv(out_csv)
        emit(f"[{tag}] batch {batch_label}: wrote {out_csv}")

        if not skip_run and save_raw_files is not None:
            archived = _archive_raw_file(
                raw_csv,
                save_raw_files,
                protein_id=protein_id,
                predictor_key=runner_key,
            )
            emit(f"[{tag}] archived raw → {archived}")


def _run_appnn_batch(
    runner,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
) -> dict[str, Path]:
    if skip_run:
        emit(f"[APPNN] batch {batch_label}: loading raw from {batch_work} …")
        output_dir = batch_work / runner.output_dir
        if output_dir.is_dir():
            raw_map = runner.discover_outputs(output_dir, protein_ids)
            missing = [pid for pid in protein_ids if pid not in raw_map]
            if not missing:
                return raw_map
        if save_raw_files is not None:
            raw_map = _archived_appnn_outputs(save_raw_files, protein_ids)
            if raw_map:
                emit(f"[APPNN] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[APPNN] batch {batch_label}: expected {output_dir} or archived raw "
            f"under {save_raw_files} (--skip-run)"
        )
    else:
        emit(f"[APPNN] batch {batch_label}: generating output for {len(protein_ids)} sequence(s) …")
        output_dir = runner.execute_batch(batch_fasta, batch_work)
        emit(f"[APPNN] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[APPNN] batch {batch_label}: missing CSV for: {', '.join(missing)}"
        )
    return raw_map


def _run_waltz_batch(
    runner,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
) -> dict[str, Path]:
    if skip_run:
        emit(f"[WALTZ] batch {batch_label}: loading raw from {batch_work} …")
        raw_map = runner.discover_outputs(batch_work, protein_ids)
        missing = [pid for pid in protein_ids if pid not in raw_map]
        if not missing:
            return raw_map
        if save_raw_files is not None:
            raw_map = _archived_waltz_outputs(save_raw_files, protein_ids)
            if raw_map:
                emit(f"[WALTZ] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[WALTZ] batch {batch_label}: expected per-protein txt under {batch_work} "
            f"or archived raw under {save_raw_files} (--skip-run)"
        )

    emit(f"[WALTZ] batch {batch_label}: submitting {len(protein_ids)} sequence(s) to web service …")
    output_dir = runner.execute_batch(batch_fasta, batch_work)
    emit(f"[WALTZ] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[WALTZ] batch {batch_label}: missing txt for: {', '.join(missing)}"
        )
    return raw_map


def _archived_waltz_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "waltz")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _archived_waltz_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "waltz")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _run_pasta_batch(
    runner,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
) -> dict[str, Path]:
    if skip_run:
        emit(f"[PASTA] batch {batch_label}: loading raw from {batch_work} …")
        raw_map = runner.discover_outputs(batch_work, protein_ids)
        missing = [pid for pid in protein_ids if pid not in raw_map]
        if not missing:
            return raw_map
        if save_raw_files is not None:
            raw_map = _archived_pasta_outputs(save_raw_files, protein_ids)
            if raw_map:
                emit(f"[PASTA] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[PASTA] batch {batch_label}: expected per-protein profiles under {batch_work} "
            f"or archived raw under {save_raw_files} (--skip-run)"
        )

    emit(f"[PASTA] batch {batch_label}: submitting {len(protein_ids)} sequence(s) to web service …")
    output_dir = runner.execute_batch(batch_fasta, batch_work)
    emit(f"[PASTA] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[PASTA] batch {batch_label}: missing profile for: {', '.join(missing)}"
        )
    return raw_map


def _archived_pasta_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "pasta")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _archived_pasta_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "pasta")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _run_archcandy_batch(
    runner,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
) -> dict[str, Path]:
    if skip_run:
        emit(f"[ArchCandy] batch {batch_label}: loading raw from {batch_work} …")
        raw_map = _discover_archcandy_raw(batch_work, protein_ids)
        if raw_map:
            return raw_map
        if save_raw_files is not None:
            raw_map = _archived_archcandy_outputs(save_raw_files, protein_ids)
            if raw_map:
                emit(f"[ArchCandy] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[ArchCandy] batch {batch_label}: expected per-protein CSV under {batch_work}, "
            f"per-protein work dirs, or archived raw under {save_raw_files} (--skip-run)"
        )

    if len(protein_ids) != 1:
        raise RuntimeError(
            f"[ArchCandy] batch {batch_label}: expected one sequence per job, got {len(protein_ids)}"
        )
    emit(f"[ArchCandy] batch {batch_label}: submitting {protein_ids[0]} …")
    output_dir = runner.execute_batch(batch_fasta, batch_work)
    emit(f"[ArchCandy] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[ArchCandy] batch {batch_label}: missing CSV for: {', '.join(missing)}"
        )
    return raw_map


def _discover_archcandy_raw(work_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    """Locate ``{id}_archcandy.csv`` under batch or per-protein work directories."""
    mapping: dict[str, Path] = {}
    search_roots = [work_root]
    parent = work_root.parent
    if parent.is_dir() and parent != work_root:
        search_roots.append(parent)
        for child in sorted(parent.iterdir()):
            if child.is_dir():
                search_roots.append(child)
    for protein_id in protein_ids:
        filename = f"{protein_id}_archcandy.csv"
        for root in search_roots:
            candidate = root / filename
            if candidate.is_file():
                mapping[protein_id] = candidate
                break
    return mapping if len(mapping) == len(protein_ids) else {}


def _archived_archcandy_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "archcandy")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _run_path_batch(
    runner,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
) -> dict[str, Path]:
    if skip_run:
        emit(f"[PATH] batch {batch_label}: loading raw from {batch_work} …")
        results_csv = _find_path_results_csv(batch_work, batch_fasta, protein_ids, save_raw_files)
        if results_csv is None:
            raise FileNotFoundError(
                f"[PATH] batch {batch_label}: expected results under {batch_work}, "
                f"per-protein work dirs, or archived raw under {save_raw_files} (--skip-run)"
            )
        if results_csv.parent != batch_work:
            emit(f"[PATH] batch {batch_label}: loaded raw from {results_csv}")
    else:
        emit(f"[PATH] batch {batch_label}: generating output for {len(protein_ids)} sequence(s) …")
        results_csv = runner.execute(batch_fasta, batch_work)
        emit(f"[PATH] batch {batch_label}: raw results → {results_csv}")

    return {protein_id: results_csv for protein_id in protein_ids}


def _find_path_results_csv(
    batch_work: Path,
    batch_fasta: Path,
    protein_ids: list[str],
    save_raw_files: Path | None,
) -> Path | None:
    """Locate PATH ``results.csv`` for sequential or per-protein work layouts."""
    candidates = [
        batch_work / "results.csv",
        batch_work / f"{batch_fasta.stem}_results.csv",
    ]
    if len(protein_ids) == 1:
        candidates.extend(
            [
                batch_work.parent / protein_ids[0] / "results.csv",
                batch_work / protein_ids[0] / "results.csv",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path

    if save_raw_files is not None and protein_ids:
        archived = _find_archived_raw(save_raw_files, protein_ids[0], "path")
        if archived is not None:
            return archived
    return None


def _run_parse_only(
    predictor_keys: list[str],
    protein_ids: list[str],
    layout: BatchLayout,
    *,
    save_raw_files: Path | None,
    config_path: str | None,
    emit: LogFn,
) -> None:
    search_roots = _raw_search_roots(layout, save_raw_files)
    for predictor_key in predictor_keys:
        tag = _predictor_tag(predictor_key)
        parser = get_parser(predictor_key, config_path=config_path)
        for protein_id in protein_ids:
            raw_path = _find_raw_file(search_roots, protein_id, predictor_key)
            if raw_path is None:
                roots = ", ".join(str(root) for root in search_roots)
                emit(f"[{tag}] skip {protein_id}: no raw file in {roots}")
                continue
            single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
            emit(f"[{tag}] parsing {protein_id} from {raw_path} …")
            sequence = read_fasta(single_fasta)[protein_id]
            result = parser.parse(
                raw_path,
                protein_id=protein_id,
                sequence=sequence,
            )
            out_csv = layout.predictor_parsed_dir(predictor_key) / f"{protein_id}_{tag}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(out_csv)
            emit(f"[{tag}] wrote {out_csv}")


def _raw_search_roots(layout: BatchLayout, save_raw_files: Path | None) -> list[Path]:
    roots: list[Path] = []
    if save_raw_files is not None:
        roots.append(save_raw_files)
    for child in sorted(layout.output_dir.iterdir()) if layout.output_dir.is_dir() else []:
        work_dir = child / "work"
        if work_dir.is_dir():
            roots.append(work_dir)
    return roots


def _cleanup_cache_dir(config_path: str | None, *, emit: LogFn) -> None:
    from aggressor_wrappers.core.cache import clear_cache_dir

    cfg = load_config(config_path)
    removed = clear_cache_dir(cfg)
    if removed:
        emit(f"[cleanup] removed {removed}")


def _cleanup_temp_dirs(
    layout: BatchLayout,
    runner_keys: Iterable[str],
    *,
    emit: LogFn,
) -> None:
    """Remove per-predictor work dirs and the fasta_split scratch area."""
    for runner_key in runner_keys:
        work_root = layout.predictor_work_dir(runner_key)
        if work_root.is_dir():
            shutil.rmtree(work_root, ignore_errors=True)
            emit(f"[cleanup] removed {work_root}")

    tmp_root = layout.fasta_split_dir.parent
    if layout.fasta_split_dir.is_dir():
        shutil.rmtree(layout.fasta_split_dir, ignore_errors=True)
    if tmp_root.is_dir() and not any(tmp_root.iterdir()):
        shutil.rmtree(tmp_root, ignore_errors=True)
        emit(f"[cleanup] removed {tmp_root}")


def _find_archived_raw(archive_root: Path, protein_id: str, predictor_key: str) -> Path | None:
    """Return ``{archive}/{protein_id}/{predictor}.*`` if present."""
    direct = archive_root / protein_id
    if not direct.is_dir():
        return None
    matches = sorted(direct.glob(f"{predictor_key}.*"))
    return matches[0] if matches else None


def _archived_appnn_outputs(archive_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, "appnn")
        if path is not None:
            mapping[protein_id] = path
    return mapping if len(mapping) == len(protein_ids) else {}


def _archive_raw_file(
    source: Path,
    archive_root: Path,
    *,
    protein_id: str,
    predictor_key: str,
) -> Path:
    ext = source.suffix or ".bin"
    dest = archive_root / protein_id / f"{predictor_key}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _find_raw_file(
    search_roots: list[Path],
    protein_id: str,
    predictor_key: str,
) -> Path | None:
    """Search ``{root}/{protein_id}/{predictor}.*`` and flat patterns under each root."""
    for raw_root in search_roots:
        direct = raw_root / protein_id
        if direct.is_dir():
            for path in sorted(direct.glob(f"{predictor_key}.*")):
                return path
            for path in sorted(direct.glob("*")):
                if path.is_file() and predictor_key in path.stem.lower():
                    return path

        for path in sorted(raw_root.glob(f"**/{protein_id}/{predictor_key}.*")):
            if path.is_file():
                return path
        for path in sorted(raw_root.glob(f"{protein_id}*{predictor_key}*")):
            if path.is_file():
                return path
        for path in sorted(raw_root.glob(f"{protein_id}_*")):
            if path.is_file() and predictor_key in path.stem.lower():
                return path
    return None


def _merge_protein(
    protein_id: str,
    layout: BatchLayout,
    requested_predictors: list[str],
    *,
    emit: LogFn,
) -> Path:
    single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
    sequence = read_fasta(single_fasta)[protein_id]

    results = []
    for predictor_key in requested_predictors:
        if predictor_key not in list_parsers():
            continue
        spec = get_predictor_spec(predictor_key)
        tag = _predictor_tag(predictor_key)
        parsed_dir = layout.predictor_parsed_dir(predictor_key)
        csv_path = parsed_dir / f"{protein_id}_{tag}.csv"
        if not csv_path.is_file():
            emit(f"[merge] skip {protein_id}: missing {csv_path.name}")
            continue
        emit(f"[merge] {protein_id}: loading {csv_path.name}")
        results.append(
            read_standard_csv(
                csv_path,
                spec,
                protein_id=protein_id,
                sequence=sequence,
            )
        )

    if not results:
        raise RuntimeError(f"[merge] {protein_id}: no parsed CSV files to merge")

    emit(f"[merge] {protein_id}: building wide table ({len(results)} predictor(s)) …")
    wide = merge_predictor_tables(results)
    merged_path = layout.merged_dir / f"{protein_id}_merged.csv"
    wide.to_csv(merged_path, index=False)
    emit(f"[merge] {protein_id}: wrote {merged_path}")
    return merged_path
