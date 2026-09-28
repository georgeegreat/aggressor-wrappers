"""ArchCandy runner and batch integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import run_multifasta_pipeline
from aggressor_wrappers.runners.archcandy_v2 import ArchCandy2Runner
from aggressor_wrappers.runners.registry import get_runner, list_runners

FIXTURES = Path(__file__).parent / "fixtures"
ARCHCANDY_CSV = FIXTURES / "archcandy_app.csv"
APP_SEQUENCE = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def test_archcandy_runner_registered() -> None:
    assert "archcandy" in list_runners()


def test_get_runner_archcandy_from_config() -> None:
    """get_runner wires config.cfg into the runner it selects.

    The assertion reads the configured value rather than hardcoding one. The
    previous version pinned 0.4 while config.cfg had moved to 0.57, so the test
    failed for a reason unrelated to the behaviour it was written to protect --
    and it masked the two defects that actually stopped ArchCandy running: the
    unparsed inline comment in `threshold = 0.57 # 0.4`, and `backend` reaching
    the constructor as an unexpected keyword.
    """
    from aggressor_wrappers.core.config import load_config, predictor_options, runner_options

    cfg = load_config()
    configured = runner_options("archcandy", cfg)
    runner = get_runner("archcandy")
    # The web arm is ArchCandy 2.0: the BiSMM site was rebuilt and the 1.x
    # route (index.php?route=tools&tool=7) no longer exists, so the 1.x runner is
    # retired to legacy/runners/ and must not be importable from the package.
    assert isinstance(runner, ArchCandy2Runner)  # no local JAR here -> web backend
    assert runner.threshold == pytest.approx(float(configured["threshold"]))
    assert runner.transmembrane is bool(configured["transmembrane"])
    # score_mode must come from [predictors.archcandy] so that the runner and
    # the parse CLI score the same ArchCandy output identically.
    assert runner.score_mode == predictor_options("archcandy", cfg)["score_mode"]
    assert "score_mode" not in configured, (
        "score_mode is duplicated in [runners.archcandy]; it belongs only in "
        "[predictors.archcandy]"
    )


def test_config_threshold_is_numeric_not_a_commented_string() -> None:
    """A trailing `# comment` must not become part of a numeric option.

    configparser keeps everything after '=' unless inline comments are declared,
    so `threshold = 0.57 # 0.4` silently yields the string '0.57 # 0.4'. It then
    travels as far as float() inside the runner before failing, which is why the
    symptom looked like a broken predictor rather than a broken config line.
    """
    from aggressor_wrappers.core.config import load_config, runner_options

    for key in ("archcandy", "crossbeta", "waltz", "pasta", "tango"):
        for name, value in runner_options(key, load_config()).items():
            assert not (isinstance(value, str) and "#" in value), (
                f"[runners.{key}] {name} kept its inline comment: {value!r}"
            )


def test_batch_pipeline_skip_run_archcandy(tmp_path: Path) -> None:
    if not ARCHCANDY_CSV.is_file():
        pytest.skip("archcandy fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(f">APP\n{APP_SEQUENCE}\n")

    out = tmp_path / "results"
    work = out / "ArchCandy" / "work" / "batch_1"
    work.mkdir(parents=True)
    (work / "APP_archcandy.csv").write_text(ARCHCANDY_CSV.read_text())

    logs: list[str] = []
    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["archcandy"],
        skip_run=True,
        log=logs.append,
    )

    assert set(merged) == {"APP"}
    assert (out / "ArchCandy" / "parsed" / "APP_ArchCandy.csv").is_file()
    assert any("[ArchCandy]" in line for line in logs)


def test_batch_pipeline_skip_run_archcandy_parallel_work_layout(tmp_path: Path) -> None:
    """--skip-run finds raw CSV under per-protein work dirs (parallel_jobs layout)."""
    if not ARCHCANDY_CSV.is_file():
        pytest.skip("archcandy fixture missing")

    multifasta = tmp_path / "proteins.fasta"
    multifasta.write_text(f">APP\n{APP_SEQUENCE}\n")

    out = tmp_path / "results"
    work = out / "ArchCandy" / "work" / "APP"
    work.mkdir(parents=True)
    (work / "APP_archcandy.csv").write_text(ARCHCANDY_CSV.read_text())

    merged = run_multifasta_pipeline(
        multifasta,
        out,
        predictors=["archcandy"],
        skip_run=True,
    )

    assert set(merged) == {"APP"}
    assert (out / "ArchCandy" / "parsed" / "APP_ArchCandy.csv").is_file()


def test_the_retired_1x_runner_is_out_of_the_package():
    """It could only ever return a dead link, so it must not be reachable.

    Reparsing historical ArchCandy 1.0 output does not need it: that is
    predictors/archcandy.py, asserted here so the retirement cannot be mistaken
    for losing the ability to read old files.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aggressor_wrappers.runners.archcandy")

    from aggressor_wrappers.predictors.registry import get_parser

    assert type(get_parser("archcandy")).__name__ == "ArchCandyParser"


def test_the_retired_key_errors_instead_of_switching_tool_version():
    """`archcandy_legacy` must not silently resolve to ArchCandy 2.0.

    Aliasing it would answer a request for one tool version with another, which
    is worse than failing: the output would be scored on 2.0's scale while the
    config still named 1.x.
    """
    with pytest.raises(KeyError, match="archcandy_legacy"):
        get_runner("archcandy_legacy")

