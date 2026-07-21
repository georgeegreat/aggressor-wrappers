"""Automatic web/local backend selection for predictor runners.

Several predictors exist as both a web service and a local tool (ArchCandy is a
JAR, Cross-Beta and AMYPred-FRL are Python, APPNN/AmyloGram are R). The web
services are the fragile axis: a single outage makes a run fail, and — worse —
the failure is discovered only *after* the submission has blocked on a poll loop,
so wall time is wasted before the pipeline gives up.

This module lets a runner decide, *before* any network call, whether to run the
local tool or the web service, according to an explicit policy:

    backend = auto   (default) local if its command/JAR is resolvable, else web
    backend = local  force local; error if the local tool is not found
    backend = web    force web  (the previous behaviour)

Resolution is deliberately cheap and side-effect-free (it checks for an
executable/JAR on PATH or at a configured path); it never probes the network to
"see if the web is up", because that probe is exactly the slow, flaky thing the
fallback exists to avoid. If you want the web, ask for it.

The policy is read from ``[runners.<key>] backend`` in ``config.cfg`` (or the
``AGGRESSOR_<KEY>_BACKEND`` environment variable), so no code change is needed to
flip a predictor between web and local.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal["auto", "local", "web"]
Resolved = Literal["local", "web"]


class BackendUnavailable(RuntimeError):
    """Raised when ``backend = local`` is forced but the local tool is absent."""


@dataclass(frozen=True)
class LocalTool:
    """How to locate a local predictor executable.

    Exactly one of ``executable`` (found on PATH) or ``path`` (an explicit file,
    e.g. an ArchCandy JAR or a compiled TANGO binary) identifies the tool.
    """

    kind: Literal["executable", "jar", "binary", "rscript_pkg"]
    executable: str | None = None       # e.g. "python3", "Rscript", "java"
    path: str | None = None             # e.g. "/opt/ArchCandy.jar"
    r_package: str | None = None        # e.g. "AmyloGram" (checked via Rscript)

    def resolve(self) -> Path | None:
        """Return the concrete path if the tool is available, else None."""
        if self.path:
            p = Path(self.path).expanduser()
            return p if p.exists() else None
        if self.executable:
            found = shutil.which(self.executable)
            return Path(found) if found else None
        return None


def _env_key(key: str) -> str:
    return f"AGGRESSOR_{key.upper()}_BACKEND"


def read_backend(key: str, options: dict | None = None) -> Backend:
    """Resolve the requested backend policy for a predictor key.

    Precedence: env var > runner options > default ('auto').
    """
    env = os.environ.get(_env_key(key))
    if env:
        env = env.strip().lower()
    value = env or (options or {}).get("backend") or "auto"
    value = str(value).strip().lower()
    if value not in ("auto", "local", "web"):
        raise ValueError(
            f"[runners.{key}] backend must be auto|local|web; got {value!r}"
        )
    return value  # type: ignore[return-value]


def select_backend(
    key: str,
    *,
    local_tool: LocalTool | None,
    options: dict | None = None,
    has_web: bool = True,
) -> tuple[Resolved, Path | None, str]:
    """Choose local vs web for a predictor, before any network activity.

    Returns ``(resolved, local_path, reason)`` where ``resolved`` is
    ``"local"`` or ``"web"``, ``local_path`` is the located tool (or None), and
    ``reason`` is a short human-readable explanation for logging.

    Raises :class:`BackendUnavailable` if ``local`` is forced but the tool is
    missing, and ``RuntimeError`` if ``web`` is forced but this predictor has no
    web backend.
    """
    policy = read_backend(key, options)
    local_path = local_tool.resolve() if local_tool else None

    if policy == "web":
        if not has_web:
            raise RuntimeError(f"{key}: backend=web requested but no web backend exists")
        return "web", None, "backend=web (forced)"

    if policy == "local":
        if local_path is None:
            raise BackendUnavailable(
                f"{key}: backend=local requested but the local tool was not found "
                f"({_describe(local_tool)}). Install it, set its path in "
                f"[runners.{key}], or use backend=auto/web."
            )
        return "local", local_path, "backend=local (forced)"

    # auto: prefer local when present (no network dependency), else fall to web
    if local_path is not None:
        return "local", local_path, f"backend=auto -> local ({local_path})"
    if has_web:
        return "web", None, "backend=auto -> web (local tool not found)"
    raise BackendUnavailable(
        f"{key}: backend=auto but neither the local tool "
        f"({_describe(local_tool)}) nor a web backend is available"
    )


def _describe(tool: LocalTool | None) -> str:
    if tool is None:
        return "no local tool configured"
    if tool.path:
        return f"expected file at {tool.path}"
    if tool.executable:
        return f"{tool.executable!r} on PATH"
    return "unspecified"
