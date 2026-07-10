"""Pipeline progress logging — stdout tee and ``{output_dir.name}.log``."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

LogFn = Callable[[str], None]


def pipeline_log_path(output_dir: Path) -> Path:
    """
    Log file co-located with pipeline output.

    Named after the output directory (``output_dir/output_dir.name.log``) so
    multiple runs into sibling folders (e.g. ``e2e_rpl27/``, ``e2e_app/``)
    produce distinct, easy-to-find logs without a separate ``--log`` flag.
    """
    return output_dir / f"{output_dir.name}.log"


def default_log(message: str) -> None:
    """Stdout-only sink for tests and programmatic callers without a log file."""
    print(message, flush=True)


@contextmanager
def pipeline_log_sink(output_dir: Path) -> Iterator[LogFn]:
    """
    Yield an ``emit`` callable that mirrors every pipeline line to stdout and disk.

    Opens the log in append mode so interrupted runs resumed later keep a single
    chronological transcript (matches the default resume behaviour).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = pipeline_log_path(output_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_path.open("a", encoding="utf-8") as handle:
        def emit(message: str) -> None:
            print(message, flush=True)
            handle.write(message + "\n")
            handle.flush()

        emit(f"[setup] log → {log_path} (run started {stamp})")
        yield emit
