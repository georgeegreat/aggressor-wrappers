"""Backend selection: web/local fallback with no network probing."""

from __future__ import annotations

import pytest

from aggressor_wrappers.core.backend import (
    BackendUnavailable,
    LocalTool,
    read_backend,
    select_backend,
)

PRESENT = LocalTool(kind="executable", executable="python3")  # always on PATH in CI
ABSENT = LocalTool(kind="jar", path="/nonexistent/tool.jar")


def test_auto_prefers_local_when_present():
    resolved, path, _ = select_backend("crossbeta", local_tool=PRESENT)
    assert resolved == "local"
    assert path is not None


def test_auto_falls_back_to_web_when_local_absent():
    resolved, path, _ = select_backend("crossbeta", local_tool=ABSENT, has_web=True)
    assert resolved == "web"
    assert path is None


def test_web_forced_ignores_available_local():
    resolved, _, _ = select_backend(
        "crossbeta", local_tool=PRESENT, options={"backend": "web"}
    )
    assert resolved == "web"


def test_local_forced_but_missing_raises_not_silently_web():
    with pytest.raises(BackendUnavailable):
        select_backend("archcandy", local_tool=ABSENT, options={"backend": "local"})


def test_web_forced_but_no_web_backend_raises():
    with pytest.raises(RuntimeError, match="no web backend"):
        select_backend("tango", local_tool=PRESENT, options={"backend": "web"}, has_web=False)


def test_auto_with_no_local_and_no_web_raises():
    with pytest.raises(BackendUnavailable):
        select_backend("x", local_tool=ABSENT, has_web=False)


def test_env_var_overrides_options():
    import os

    os.environ["AGGRESSOR_CROSSBETA_BACKEND"] = "web"
    try:
        assert read_backend("crossbeta", {"backend": "local"}) == "web"
    finally:
        del os.environ["AGGRESSOR_CROSSBETA_BACKEND"]


def test_invalid_backend_value_rejected():
    with pytest.raises(ValueError, match="auto|local|web"):
        read_backend("crossbeta", {"backend": "nonsense"})


def test_localtool_resolve_jar_path(tmp_path):
    jar = tmp_path / "ArchCandy.jar"
    jar.write_bytes(b"PK\x03\x04")  # a file that exists
    tool = LocalTool(kind="jar", path=str(jar))
    assert tool.resolve() == jar
    assert LocalTool(kind="jar", path=str(tmp_path / "missing.jar")).resolve() is None
