"""Tests for pipeline log tee."""

from __future__ import annotations

from pathlib import Path

from aggressor_wrappers.batch.logging import pipeline_log_path, pipeline_log_sink


def test_pipeline_log_path_uses_output_dir_name(tmp_path: Path) -> None:
    out = tmp_path / "e2e_rpl27_pipeline"
    assert pipeline_log_path(out) == out / "e2e_rpl27_pipeline.log"


def test_pipeline_log_sink_appends(tmp_path: Path) -> None:
    out = tmp_path / "run_a"
    with pipeline_log_sink(out) as emit:
        emit("line one")
    with pipeline_log_sink(out) as emit:
        emit("line two")

    text = (out / "run_a.log").read_text()
    assert "line one" in text
    assert "line two" in text
    assert text.index("line one") < text.index("line two")
