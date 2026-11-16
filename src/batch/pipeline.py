"""Orchestrate multifasta runs across predictors with progress logging.

Batch orchestration lives here; cross-cutting concerns are split out deliberately:
* ``batch.resume`` — checkpoint validation (reuse ``read_standard_csv``)
* ``batch.logging`` — stdout tee to ``{output_dir.name}.log`` (CLI wiring)
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from aggressor_wrappers.batch.logging import LogFn, default_log
from aggressor_wrappers.batch.resume import partition_items_for_resume
from aggressor_wrappers.batch.scheduler import (
    run_predictors_concurrently,
    serialise_log,
)
from aggressor_wrappers.core.config import load_config, runner_batch_config
from aggressor_wrappers.core.fasta import read_fasta
from aggressor_wrappers.core.merge import merge_predictor_tables
from aggressor_wrappers.core.net import PermanentToolError
from aggressor_wrappers.core.schema import get_predictor_spec, read_standard_csv, resolve_predictor_key
from aggressor_wrappers.predictors.registry import get_parser, list_parsers
from aggressor_wrappers.runners.registry import get_runner, list_runners


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


def _resume_enabled(*, resume: bool, skip_run: bool) -> bool:
    """
    Whether to skip predictors whose ``parsed/`` output already matches the FASTA.

    Disabled with ``--skip-run``: that mode always re-parses from raw ``work/``
    files (e.g. after a parser fix), so reusing existing parsed tables would
    defeat the purpose.
    """
    return resume and not skip_run


def _default_log(message: str) -> None:
    """Backward-compatible alias; prefer ``batch.logging.default_log``."""
    default_log(message)


def _predictor_tag(key: str) -> str:
    return get_predictor_spec(key).score_column.replace("_score", "")


def run_multifasta_pipeline(
    multifasta: Path,
    output_dir: Path,
    *,
    predictors: Iterable[str] | None = None,
    config_path: str | None = None,
    skip_run: bool = False,
    resume: bool = True,
    save_raw_files: Path | None = None,
    keep_cache: bool = False,
    merge_existing: bool = False,
    log: LogFn | None = None,
) -> dict[str, Path]:
    """
    Run available predictors for every sequence in ``multifasta``.

    Per-runner batching is controlled by ``[runners.*]`` in config.cfg
    (``parallel_jobs``, ``sequences_per_run``).

    When ``resume`` is true (default) and ``skip_run`` is false, existing
    ``{PREDICTOR}/parsed/{id}_*.csv`` files are validated against the input
    FASTA (length and ``aa_name``) and skipped when they already match.

    ``merge_existing`` widens the MERGE step only: every registered predictor
    that already has a parsed table in this output tree enters the merged
    table, not only the ones asked for in this invocation. It exists because
    adding one tool to a finished panel otherwise meant re-running all of
    them. Each predictor it adds is named in the log, because a merged table
    is the denominator of every consensus tier computed from it and a panel
    that changes width without saying so is the worst kind of wrong.

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
        for key in runner_keys:
            emit(
                f"[setup] --skip-run: {_predictor_tag(key)} raw from "
                f"{layout.predictor_work_dir(key)}/batch_N or the drop zone "
                f"{layout.predictor_dir(key)}"
            )
    elif _resume_enabled(resume=resume, skip_run=skip_run):
        emit("[setup] resume enabled: will reuse valid parsed CSVs from prior runs")

    items = list(sequences.items())

    # Predictors are mutually independent (each owns its work/ and parsed/ dirs),
    # so they are launched concurrently: wall time becomes max(T_predictor) rather
    # than sum(T_predictor). Each runner's own parallel_jobs / sequences_per_run
    # caps still bound its internal concurrency, so external web services see no
    # more load than before. See batch/scheduler.py.
    predictor_jobs = _predictor_jobs(config_path)
    safe_emit = serialise_log(emit)
    report = run_predictors_concurrently(
        runner_keys,
        lambda key: _run_runner_batches(
            key,
            items,
            layout,
            config_path=config_path,
            skip_run=skip_run,
            resume=resume,
            save_raw_files=save_raw_files,
            emit=safe_emit,
        ),
        max_workers=predictor_jobs,
        emit=safe_emit,
    )
    report.raise_if_all_failed()

    if parse_only_keys:
        _run_parse_only(
            parse_only_keys,
            protein_ids,
            layout,
            sequences=sequences,
            save_raw_files=save_raw_files,
            config_path=config_path,
            resume=_resume_enabled(resume=resume, skip_run=skip_run),
            emit=emit,
        )

    to_merge = requested
    if merge_existing:
        to_merge = _widen_with_existing_parsed(layout, requested, emit=emit)

    merged_paths: dict[str, Path] = {}
    unmerged: list[tuple[str, BaseException]] = []
    for protein_id in protein_ids:
        try:
            merged_paths[protein_id] = _merge_protein(
                protein_id,
                layout,
                to_merge,
                emit=emit,
            )
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            # A protein no predictor could produce output for is a gap in the
            # panel, not a reason to discard the proteins that worked. This
            # raised, so a single protein whose every tool had failed took the
            # whole merge step with it -- including proteins already parsed by
            # every predictor in the panel.
            unmerged.append((protein_id, exc))
            emit(f"[merge] {protein_id}: NO merged table -- {exc}")

    if unmerged:
        emit(
            f"[merge] {len(merged_paths)} of {len(protein_ids)} protein(s) "
            f"merged; no table for: "
            + ", ".join(protein_id for protein_id, _ in unmerged)
        )
    if not merged_paths:
        raise RuntimeError(
            f"No protein produced a merged table. {len(unmerged)} failed; "
            f"first was {unmerged[0][0]}: {unmerged[0][1]}"
            if unmerged
            else "No protein produced a merged table."
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
            part = part.strip()
            if not part:
                continue
            try:
                keys.append(resolve_predictor_key(part))
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
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
    resume: bool,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    runner = get_runner(runner_key, config_path=config_path, parse_only=skip_run)
    tag = _predictor_tag(runner_key)

    partition = partition_items_for_resume(
        items,
        parsed_dir=layout.predictor_parsed_dir(runner_key),
        runner_key=runner_key,
        tag=tag,
        resume=_resume_enabled(resume=resume, skip_run=skip_run),
        emit=emit,
    )
    if not partition.pending:
        emit(f"[{tag}] all {len(items)} protein(s) already parsed; skipping runner")
        return

    pending_items = partition.pending
    batch_cfg = runner_batch_config(runner_key, load_config(config_path))
    chunk_size = _chunk_size_for_runner(runner_key, len(pending_items), config_path)
    batches = chunk_items(pending_items, chunk_size)
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

    if runner_key == "crossbeta" and batch_cfg.parallel_jobs > 1 and not skip_run:
        emit(
            f"[{tag}] parallel_jobs={batch_cfg.parallel_jobs}, "
            f"sequences_per_run={batch_cfg.sequences_per_run or 'all'}"
        )
        _run_crossbeta_parallel(
            runner,
            batches,
            layout,
            parallel_jobs=batch_cfg.parallel_jobs,
            save_raw_files=save_raw_files,
            emit=emit,
        )
        return

    # A permanent refusal is a property of ONE submission, not of the predictor:
    # a single sequence carrying an X should cost that protein, not the other
    # 371. Transient errors still propagate, because retrying is the right
    # response to those and the scheduler owns it.
    refused: list[tuple[str, list[str], BaseException]] = []
    unreadable: list[tuple[str, BaseException]] = []
    parsed_total = 0
    completed = 0

    for batch_index, batch in enumerate(batches, start=1):
        batch_label = f"{batch_index}/{total_batches}"
        batch_ids = [protein_id for protein_id, _ in batch]
        work_root = layout.predictor_work_dir(runner_key)
        try:
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
                    drop_zone=layout.predictor_dir(runner_key),
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
            elif runner_key == "crossbeta":
                raw_map = _run_crossbeta_batch(
                    runner,
                    batch_fasta,
                    batch_work,
                    batch_ids,
                    skip_run=skip_run,
                    save_raw_files=save_raw_files,
                    batch_label=batch_label,
                    emit=emit,
                )
            elif runner_key == "aggreprot":
                raw_map = _run_aggreprot_batch(
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
                raw_map = _run_generic_batch(
                    runner,
                    runner_key,
                    batch_fasta,
                    batch_work,
                    batch_ids,
                    skip_run=skip_run,
                    save_raw_files=save_raw_files,
                    batch_label=batch_label,
                    emit=emit,
                    drop_zone=layout.predictor_dir(runner_key),
                )

            parsed_here, failed_here = _parse_and_write_runner_batch(
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
            parsed_total += parsed_here
            unreadable.extend(failed_here)
        except PermanentToolError as exc:
            refused.append((batch_label, batch_ids, exc))
            continue
        completed += 1

    _report_refusals(tag, refused, completed=completed, emit=emit)

    if unreadable:
        emit(
            f"[{tag}] {parsed_total} protein(s) parsed, {len(unreadable)} raw "
            f"file(s) unreadable across the run: "
            + "; ".join(f"{pid}: {exc}" for pid, exc in unreadable)
        )
    if parsed_total == 0 and unreadable:
        # Nothing readable anywhere is a question about the predictor, not
        # about any one file: a broken parser, or a directory holding another
        # tool's output. Said here, once, with the first error attached.
        protein_id, exc = unreadable[0]
        emit(
            f"[{tag}] WARNING -- no raw file for this predictor could be read "
            f"({len(unreadable)} tried). First failure, {protein_id}: {exc}. "
            f"The predictor will be absent from every merged table."
        )


def _report_refusals(
    tag: str,
    refused: list[tuple[str, list[str], BaseException]],
    *,
    completed: int,
    emit: LogFn,
) -> None:
    """Summarise per-submission refusals; re-raise only if none got through.

    If some batches produced output the predictor did its job for those
    proteins, so the run continues and the affected ids are named in the log --
    they will be visible downstream as a smaller consensus denominator for those
    proteins. If NOTHING got through the refusal is about the predictor or the
    whole input, and it is raised so the scheduler records the predictor as
    skipped rather than as silently empty.
    """
    if not refused:
        return
    ids = [protein_id for _, batch_ids, _ in refused for protein_id in batch_ids]
    for batch_label, batch_ids, exc in refused:
        emit(f"[{tag}] batch {batch_label} REFUSED for {', '.join(batch_ids)}: {exc}")
    if completed:
        emit(
            f"[{tag}] {len(ids)} sequence(s) refused, {completed} batch(es) "
            f"completed; continuing without: {', '.join(ids)}"
        )
        return
    raise PermanentToolError(
        f"{tag}: every submission was refused ({len(ids)} sequence(s)). "
        f"First reason: {refused[0][2]}"
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


def _run_crossbeta_parallel(
    runner,
    batches: list[list[tuple[str, str]]],
    layout: BatchLayout,
    *,
    parallel_jobs: int,
    save_raw_files: Path | None,
    emit: LogFn,
) -> None:
    tag = _predictor_tag("crossbeta")
    total = len(batches)
    work_root = layout.predictor_work_dir("crossbeta")

    def _job(batch_index: int, batch: list[tuple[str, str]]) -> None:
        if len(batch) != 1:
            raise RuntimeError(
                f"[{tag}] job {batch_index}/{total}: expected one sequence per job, "
                f"got {len(batch)} (check [runners.crossbeta] sequences_per_run)"
            )
        batch_label = f"{batch_index}/{total}"
        protein_id = batch[0][0]
        batch_work = work_root / protein_id
        single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
        emit(f"[{tag}] job {batch_label}: submitting {protein_id} …")
        raw_json = runner.execute(single_fasta, batch_work)
        emit(f"[{tag}] job {batch_label}: raw results → {raw_json}")
        result = runner.run(
            fasta=single_fasta,
            protein_id=protein_id,
            work_dir=batch_work,
            skip_run=True,
            raw_json=raw_json,
        )
        out_csv = layout.predictor_parsed_dir("crossbeta") / f"{protein_id}_{tag}.csv"
        result.to_csv(out_csv)
        emit(f"[{tag}] job {batch_label}: wrote {out_csv}")
        if save_raw_files is not None:
            archived = _archive_raw_file(
                raw_json,
                save_raw_files,
                protein_id=protein_id,
                predictor_key="crossbeta",
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


#: Keyword names runners use for "parse this file instead of running".
#: Ordered by specificity so a runner declaring several still gets the right one.
_RAW_KEYWORDS = (
    "results_csv",
    "raw_profile",
    "raw_json",
    "raw_txt",
    "raw_csv",
    "raw_path",
)


def _raw_kwarg(runner, raw_path) -> dict:
    """Pass the raw file under the keyword this runner actually declares.

    This replaced a hardcoded map of runner_key -> keyword. Any runner missing
    from that map received `raw_csv=None` together with `skip_run=True`, so it
    raised its own "provide raw_csv when --skip-run is set" — from a pipeline the
    caller had NOT put in skip-run mode, which made the message actively
    misleading. FoldAmyloid and AGGRESCAN failed exactly this way on a full
    panel, having been written correctly and simply not added to the list.

    Resolving against the signature means adding a predictor no longer requires
    editing this function, and a runner that declares no raw keyword at all is
    reported here rather than at its own error site.
    """
    import inspect

    parameters = inspect.signature(runner.run).parameters
    for name in _RAW_KEYWORDS:
        if name in parameters:
            return {name: raw_path}
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return {"raw_csv": raw_path}
    raise TypeError(
        f"{type(runner).__name__}.run() declares none of {list(_RAW_KEYWORDS)}, "
        f"so a pre-fetched raw file cannot be handed to it for parsing."
    )


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
) -> tuple[int, list[tuple[str, BaseException]]]:
    """Parse one batch; returns ``(parsed_count, [(protein, error), ...])``."""
    tag = _predictor_tag(runner_key)
    # raw_map may legitimately cover only PART of the batch: under --skip-run
    # the raw files come from wherever they happen to be, and a drop zone
    # populated by hand on another machine can hold six of seven proteins.
    # Indexing raw_map directly turned that into a bare KeyError naming a
    # protein, with nothing to say which predictor or which directory was
    # short. The proteins that do have raw are parsed; the rest are named.
    without_raw = [pid for pid in batch_ids if pid not in raw_map]
    if without_raw:
        emit(
            f"[{tag}] batch {batch_label}: no raw output for "
            f"{', '.join(without_raw)} -- parsing the remaining "
            f"{len(batch_ids) - len(without_raw)} of {len(batch_ids)}"
        )
    # A file that will not parse is a property of THAT file, not of the
    # predictor. The scheduler already treats a refused submission that way --
    # "a single sequence carrying an X should cost that protein, not the other
    # 371" -- and the parse step did not: one unreadable file raised out of the
    # loop and took every protein after it in the batch with it. A seven-protein
    # WALTZ batch lost RPS13 to a length mismatch in RPS12, with nothing in the
    # log naming either, because the five before RPS12 had already been written
    # and the run simply stopped.
    failures: list[tuple[str, BaseException]] = []
    parsed = 0
    for protein_id in batch_ids:
        if protein_id not in raw_map:
            continue
        raw_csv = raw_map[protein_id]
        single_fasta = layout.fasta_split_dir / f"{protein_id}.fasta"
        emit(f"[{tag}] batch {batch_label}: parsing {protein_id} …")
        try:
            result = runner.run(
                fasta=single_fasta,
                protein_id=protein_id,
                work_dir=batch_work,
                skip_run=True,
                **_raw_kwarg(runner, raw_csv),
            )
        except Exception as exc:                      # noqa: BLE001
            failures.append((protein_id, exc))
            emit(
                f"[{tag}] batch {batch_label}: {protein_id} FAILED to parse "
                f"{raw_csv} -- {type(exc).__name__}: {exc}"
            )
            continue
        out_csv = layout.predictor_parsed_dir(runner_key) / f"{protein_id}_{tag}.csv"
        result.to_csv(out_csv)
        parsed += 1
        emit(f"[{tag}] batch {batch_label}: wrote {out_csv}")

        if not skip_run and save_raw_files is not None:
            archived = _archive_raw_file(
                raw_csv,
                save_raw_files,
                protein_id=protein_id,
                predictor_key=runner_key,
            )
            emit(f"[{tag}] archived raw → {archived}")

    if failures:
        emit(
            f"[{tag}] batch {batch_label}: {len(failures)} file(s) unreadable, "
            f"{parsed} parsed. Unreadable: "
            + "; ".join(f"{pid} ({type(exc).__name__})" for pid, exc in failures)
        )
    # Reported, never raised. "Every file in this batch failed" sounds like a
    # systematic problem worth stopping for, and is not: with
    # sequences_per_run = 1 a batch IS one protein, so that condition is true
    # of every single unreadable file, and raising on it stopped FoldAmyloid
    # at the second of seven proteins. Whether a predictor is systematically
    # broken is a question about the whole run, and _run_runner_batches answers
    # it once, below.
    return parsed, failures


def _skip_run_directories(
    batch_work: Path, drop_zone: Path | None
) -> list[tuple[str, Path]]:
    """Where raw output may be under ``--skip-run``, in priority order.

    The batch partition is an artefact of THIS run's chunk size, not of where
    files are: with ``sequences_per_run = 1`` a seven-protein panel has seven
    ``batch_N`` directories, and someone adding files by hand puts all of them
    in one. Searching only this batch's own directory then found the first
    protein and none of the rest -- six of seven FoldAmyloid files sat in
    ``work/batch_1`` while batches 2-7 looked in directories that did not
    exist.

    ``work/`` and its ``batch_*`` children are searched; ``parsed/`` is not,
    and must not be. A parsed table is named ``{id}_{tag}.csv``, which is
    exactly the canonical raw name for some predictors -- FoldAmyloid among
    them -- so reaching into it would feed a parser its own output.
    """
    work_root = batch_work.parent
    candidates: list[tuple[str, Path]] = [("this batch", batch_work)]
    for sibling in sorted(work_root.glob("batch_*")):
        if sibling.is_dir() and sibling != batch_work:
            candidates.append(("another batch", sibling))
    candidates.append(("the work directory", work_root))
    if drop_zone is not None:
        candidates.append(("the drop zone", drop_zone))

    seen: set[Path] = set()
    ordered: list[tuple[str, Path]] = []
    for where, directory in candidates:
        resolved = Path(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append((where, resolved))
    return ordered


def _run_generic_batch(
    runner,
    runner_key: str,
    batch_fasta: Path,
    batch_work: Path,
    protein_ids: list[str],
    *,
    skip_run: bool,
    save_raw_files: Path | None,
    batch_label: str,
    emit: LogFn,
    drop_zone: Path | None = None,
) -> dict[str, Path]:
    """Batch path for any runner exposing ``execute_batch`` + ``discover_outputs``.

    This replaces the tail of an ``if runner_key == ...`` ladder that raised
    "Batch runner not implemented" for anything not explicitly listed. Adding a
    predictor therefore required editing the scheduler as well as writing the
    runner, and forgetting the second edit failed at RUN time on a full panel —
    which is how FoldAmyloid and AGGRESCAN, both of which implement the batch
    protocol correctly, aborted a 372-protein sweep.

    ``execute_batch`` implementations differ in whether they return the work
    directory or the single file they wrote, so the result is normalised to a
    directory before discovery rather than each runner being made to conform.
    """
    label = getattr(getattr(runner, "spec", None), "display_name", None) or runner_key

    if skip_run:
        # Three places raw output can be, searched in order of how specific
        # they are, and MERGED rather than tried in turn: a panel is often
        # part-local and part-imported, and the first non-empty hit used to win
        # outright, so three TANGO files left in work/ by an aborted run hid the
        # four that had been obtained elsewhere.
        #
        #   1. {PREDICTOR}/work/batch_N/  -- what a run on this machine left.
        #   2. {PREDICTOR}/              -- the drop zone: where output from
        #      another machine is put by hand. TANGO ships a Linux x86_64 ELF,
        #      so on an arm64 Mac it cannot be executed AT ALL and its output
        #      can only ever arrive this way; the same holds for any tool run
        #      on a cluster. Searching only work/batch_N meant the files had to
        #      be filed under a directory named after a batch index the user
        #      does not choose and cannot predict.
        #   3. --save-raw-files archive   -- {root}/{protein}/{predictor}.*
        sources = _skip_run_directories(batch_work, drop_zone)
        searched = ", ".join(str(path) for _, path in sources)
        emit(f"[{label}] batch {batch_label}: loading raw from {searched} …")

        raw_map: dict[str, Path] = {}
        for where, directory in sources:
            if not directory.is_dir():
                continue
            missing = [pid for pid in protein_ids if pid not in raw_map]
            found = runner.discover_outputs(directory, missing)
            if found:
                emit(
                    f"[{label}] batch {batch_label}: {len(found)} raw file(s) "
                    f"from the {where} ({directory})"
                )
                raw_map.update(found)

        if len(raw_map) < len(protein_ids) and save_raw_files is not None:
            archived = _archived_runner_outputs(
                save_raw_files, protein_ids, runner_key
            )
            # _archived_runner_outputs is all-or-nothing by design, so it only
            # contributes when it can cover the whole batch.
            for protein_id, path in archived.items():
                raw_map.setdefault(protein_id, path)

        if not raw_map:
            raise FileNotFoundError(
                f"[{label}] batch {batch_label}: no raw output for any of "
                f"{', '.join(protein_ids)}. Looked in {searched}"
                + (f" and {save_raw_files}" if save_raw_files else "")
                + ". With --skip-run, raw files obtained elsewhere go in the "
                f"drop zone {drop_zone}, named as this runner expects."
            )
        return raw_map

    emit(f"[{label}] batch {batch_label}: submitting {len(protein_ids)} sequence(s) …")
    produced = Path(runner.execute_batch(batch_fasta, batch_work))
    output_dir = produced if produced.is_dir() else produced.parent
    emit(f"[{label}] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[{label}] batch {batch_label}: missing output for: {', '.join(missing)}"
        )
    return raw_map


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
            raw_map = _archived_runner_outputs(save_raw_files, protein_ids, "appnn")
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
    drop_zone: Path | None = None,
) -> dict[str, Path]:
    if skip_run:
        # Same three places, in the same order, as the generic path: a run's own
        # work/, then the drop zone, then the archive, merged rather than tried
        # in turn. WALTZ needs it more than most -- the service has been down
        # for long stretches, so the only copy of a prediction is often a file
        # saved from the server months ago and dropped in by hand.
        sources = _skip_run_directories(batch_work, drop_zone)
        searched = ", ".join(str(path) for _, path in sources)
        emit(f"[WALTZ] batch {batch_label}: loading raw from {searched} …")

        raw_map: dict[str, Path] = {}
        for where, directory in sources:
            if not directory.is_dir():
                continue
            outstanding = [pid for pid in protein_ids if pid not in raw_map]
            found = runner.discover_outputs(directory, outstanding)
            if found:
                emit(
                    f"[WALTZ] batch {batch_label}: {len(found)} raw file(s) from "
                    f"the {where} ({directory})"
                )
                raw_map.update(found)

        if len(raw_map) < len(protein_ids) and save_raw_files is not None:
            archived = _archived_runner_outputs(save_raw_files, protein_ids, "waltz")
            for protein_id, path in archived.items():
                raw_map.setdefault(protein_id, path)

        if raw_map:
            return raw_map
        raise FileNotFoundError(
            f"[WALTZ] batch {batch_label}: no raw output for any of "
            f"{', '.join(protein_ids)}. Looked in {searched}"
            + (f" and {save_raw_files}" if save_raw_files else "")
            + ". With --skip-run, files obtained elsewhere go in the drop zone "
            f"{drop_zone}; WALTZ's own .dat download keeps its server name."
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
            raw_map = _archived_runner_outputs(save_raw_files, protein_ids, "pasta")
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
            raw_map = _archived_runner_outputs(save_raw_files, protein_ids, "archcandy")
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


def _run_crossbeta_batch(
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
        emit(f"[Cross-Beta] batch {batch_label}: loading raw from {batch_work} …")
        raw_map = _discover_crossbeta_raw(batch_work, protein_ids)
        if raw_map:
            return raw_map
        if save_raw_files is not None:
            raw_map = _archived_runner_outputs(save_raw_files, protein_ids, "crossbeta")
            if raw_map:
                emit(f"[Cross-Beta] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[Cross-Beta] batch {batch_label}: expected per-protein JSON under {batch_work}, "
            f"per-protein work dirs, or archived raw under {save_raw_files} (--skip-run)"
        )

    if len(protein_ids) != 1:
        raise RuntimeError(
            f"[Cross-Beta] batch {batch_label}: expected one sequence per job, got {len(protein_ids)}"
        )
    emit(f"[Cross-Beta] batch {batch_label}: submitting {protein_ids[0]} …")
    output_dir = runner.execute_batch(batch_fasta, batch_work)
    emit(f"[Cross-Beta] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[Cross-Beta] batch {batch_label}: missing JSON for: {', '.join(missing)}"
        )
    return raw_map


def _discover_crossbeta_raw(work_root: Path, protein_ids: list[str]) -> dict[str, Path]:
    """Locate ``{id}_crossbeta.json`` under batch or per-protein work directories."""
    mapping: dict[str, Path] = {}
    search_roots = [work_root]
    parent = work_root.parent
    if parent.is_dir() and parent != work_root:
        search_roots.append(parent)
        for child in sorted(parent.iterdir()):
            if child.is_dir():
                search_roots.append(child)
    for protein_id in protein_ids:
        filename = f"{protein_id}_crossbeta.json"
        for root in search_roots:
            candidate = root / filename
            if candidate.is_file():
                mapping[protein_id] = candidate
                break
    return mapping if len(mapping) == len(protein_ids) else {}


def _run_aggreprot_batch(
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
        emit(f"[AggreProt] batch {batch_label}: loading raw from {batch_work} …")
        raw_map = runner.discover_outputs(batch_work, protein_ids)
        missing = [pid for pid in protein_ids if pid not in raw_map]
        if not missing:
            return raw_map
        if save_raw_files is not None:
            raw_map = _archived_runner_outputs(save_raw_files, protein_ids, "aggreprot")
            if raw_map:
                emit(f"[AggreProt] batch {batch_label}: loaded raw from {save_raw_files}")
                return raw_map
        raise FileNotFoundError(
            f"[AggreProt] batch {batch_label}: expected per-protein CSV under {batch_work} "
            f"or archived raw under {save_raw_files} (--skip-run)"
        )

    emit(
        f"[AggreProt] batch {batch_label}: submitting {len(protein_ids)} sequence(s) "
        f"to web service …"
    )
    output_dir = runner.execute_batch(batch_fasta, batch_work)
    emit(f"[AggreProt] batch {batch_label}: raw files in {output_dir}")

    raw_map = runner.discover_outputs(output_dir, protein_ids)
    missing = [pid for pid in protein_ids if pid not in raw_map]
    if missing:
        raise FileNotFoundError(
            f"[AggreProt] batch {batch_label}: missing CSV for: {', '.join(missing)}"
        )
    return raw_map


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
    sequences: dict[str, str],
    save_raw_files: Path | None,
    config_path: str | None,
    resume: bool,
    emit: LogFn,
) -> None:
    search_roots = _raw_search_roots(layout, save_raw_files)
    for predictor_key in predictor_keys:
        tag = _predictor_tag(predictor_key)
        parser = get_parser(predictor_key, config_path=config_path)
        parsed_dir = layout.predictor_parsed_dir(predictor_key)
        items = [(protein_id, sequences[protein_id]) for protein_id in protein_ids]
        partition = partition_items_for_resume(
            items,
            parsed_dir=parsed_dir,
            runner_key=predictor_key,
            tag=tag,
            resume=resume,
            emit=emit,
        )
        pending_ids = {protein_id for protein_id, _ in partition.pending}
        for protein_id in protein_ids:
            if protein_id not in pending_ids:
                continue
            raw_path = _find_raw_file(search_roots, protein_id, predictor_key)
            if raw_path is None:
                roots = ", ".join(str(root) for root in search_roots)
                emit(f"[{tag}] skip {protein_id}: no raw file in {roots}")
                continue
            sequence = sequences[protein_id]
            emit(f"[{tag}] parsing {protein_id} from {raw_path} …")
            result = parser.parse(
                raw_path,
                protein_id=protein_id,
                sequence=sequence,
            )
            out_csv = parsed_dir / f"{protein_id}_{tag}.csv"
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


def _archived_runner_outputs(
    archive_root: Path,
    protein_ids: list[str],
    predictor_key: str,
) -> dict[str, Path]:
    """Load a full batch of archived raw files for ``--skip-run`` recovery."""
    mapping: dict[str, Path] = {}
    for protein_id in protein_ids:
        path = _find_archived_raw(archive_root, protein_id, predictor_key)
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


def _widen_with_existing_parsed(
    layout: BatchLayout,
    requested: list[str],
    *,
    emit: LogFn,
) -> list[str]:
    """Add predictors that already have parsed tables in this output tree.

    Requested predictors keep their order and stay first, so the column order
    of a merged table does not shuffle between runs. Anything found beyond them
    is appended in a stable order and NAMED, one line per predictor: this
    function changes the width of every merged table it touches, which changes
    what a fractional consensus is a fraction of.
    """
    # Dedup by TAG, not by registry key. Several keys can share one tag and so
    # one parsed file: archcandy and archcandy2 are two parsers (1.x cumulative
    # vs 2.0, different published cutoffs) writing and reading
    # {id}_ArchCandy.csv. Widening by key added that single file twice and the
    # merge refused the batch with "Duplicate predictor in merge input" -- after
    # every tool had already been parsed.
    covered = {_predictor_tag(key) for key in requested}
    by_tag: dict[str, list[str]] = {}
    for key in list_parsers():
        if key in requested:
            continue
        tag = _predictor_tag(key)
        if tag in covered:
            continue
        parsed_dir = layout.predictor_parsed_dir(key)
        if not parsed_dir.is_dir() or not sorted(parsed_dir.glob(f"*_{tag}.csv")):
            continue
        by_tag.setdefault(tag, []).append(key)

    extra: list[str] = []
    for tag in sorted(by_tag):
        candidates = by_tag[tag]
        # The canonical key for the family wins. Which VERSION wrote the file is
        # not recorded in it, so the choice is stated rather than assumed.
        canonical = [k for k in candidates if resolve_predictor_key(k) == k]
        key = canonical[0] if canonical else candidates[0]
        parsed_dir = layout.predictor_parsed_dir(key)
        tables = sorted(parsed_dir.glob(f"*_{tag}.csv"))
        if len(candidates) > 1:
            emit(
                f"[merge] --merge-existing: {tag} tables could belong to any of "
                f"{', '.join(repr(c) for c in candidates)} -- they share one "
                f"filename and the file does not record which parser wrote it. "
                f"Merging as {key!r}; name the other explicitly in --predictors "
                f"to override."
            )
        emit(
            f"[merge] --merge-existing: adding {key!r} "
            f"({len(tables)} parsed table(s) under {parsed_dir})"
        )
        extra.append(key)

    if not extra:
        emit("[merge] --merge-existing: no other predictor has parsed tables here")
        return list(requested)

    widened = list(requested) + extra
    emit(
        f"[merge] --merge-existing: merging {len(widened)} predictor(s): "
        f"{', '.join(widened)}"
    )
    return widened


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
    unmergeable: list[str] = []
    for predictor_key in requested_predictors:
        if predictor_key not in list_parsers():
            # A configured predictor that cannot be merged is REPORTED, never
            # dropped in silence. AmyloGram had a runner and a projection module
            # but no registry entry, so it ran on every protein, wrote correct
            # parsed tables, and was skipped here without a line in the log: the
            # consensus denominator fell from 10 to 9 and nothing said so. A
            # predictor that fails loudly costs an afternoon; one that
            # disappears costs the denominator of every fractional tier
            # downstream, in a run nobody re-reads.
            unmergeable.append(predictor_key)
            emit(
                f"[merge] {protein_id}: WARNING -- {predictor_key!r} was "
                f"requested but has no registered parser, so it cannot enter "
                f"the merged table. Registered: {', '.join(list_parsers())}"
            )
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

    if unmergeable:
        # Stated once more at the point the denominator is actually fixed, with
        # the two numbers side by side. Anything that reads a fractional tier
        # from this table is reading n/len(results), and that is the number a
        # Methods section has to quote.
        emit(
            f"[merge] {protein_id}: consensus denominator is {len(results)}, "
            f"not {len(requested_predictors)} -- "
            f"{', '.join(unmergeable)} could not be merged"
        )

    emit(f"[merge] {protein_id}: building wide table ({len(results)} predictor(s)) …")
    wide = merge_predictor_tables(results)
    merged_path = layout.merged_dir / f"{protein_id}_merged.csv"
    wide.to_csv(merged_path, index=False)
    emit(f"[merge] {protein_id}: wrote {merged_path}")
    return merged_path


def _predictor_jobs(config_path: str | None) -> int:
    """How many predictors to run concurrently (``[pipeline] predictor_jobs``).

    Defaults to the number of predictors requested, capped at 4: the runners are
    dominated by remote polling rather than local CPU, so oversubscription is
    cheap, but a cap keeps log output and local subprocess pressure sane.
    Set to 1 to restore the original strictly-sequential behaviour.
    """
    cfg = load_config(config_path)
    raw = getattr(cfg, "pipeline", None)
    value = None
    if isinstance(raw, dict):
        value = raw.get("predictor_jobs")
    else:
        value = getattr(raw, "predictor_jobs", None)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 4
