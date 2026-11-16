"""CLI --help smoke tests."""

from __future__ import annotations

import subprocess
import sys


def test_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "aggressor_wrappers", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "aggressor-parse" in result.stdout
    assert "aggressor-merge" in result.stdout
    assert "aggressor-widemerge" in result.stdout
    assert "proteins.fasta -o output_dir" in result.stdout
    assert "parse" in result.stdout and "merge" in result.stdout


def test_app_primary_pipeline_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["vendor/PATH/test.fasta", "-o", "output_dir", "--help"]) == 0


def test_app_dispatch_batch_alias() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["batch", "--help"]) == 0


def test_parse_help() -> None:
    from aggressor_wrappers.cli.parse import main

    assert main(["--help"]) == 0


def test_merge_help() -> None:
    from aggressor_wrappers.cli.merge import main

    assert main(["--help"]) == 0


def test_app_main_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["--help"]) == 0
    assert main([]) == 0


def test_app_dispatch_parse_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["parse", "--help"]) == 0


def test_app_dispatch_run_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["run", "--help"]) == 0


def test_widemerge_help() -> None:
    from aggressor_wrappers.cli.widemerge import main

    assert main(["--help"]) == 0


def test_app_dispatch_widemerge_help() -> None:
    from aggressor_wrappers.cli.app import main

    assert main(["widemerge", "--help"]) == 0


# --------------------------------------------------------------------------- #
# --threshold must not be accepted and then ignored.
#
# Each parser names its threshold differently, and the mapping from the flag to
# the keyword covered five predictors out of thirteen. For the other eight the
# flag parsed, produced an empty kwargs dict, and the parser ran at its default:
# a table binarised at 5.0 while the command line said 10.0, with nothing in the
# output or the log to distinguish it from one binarised at 10.0.
# --------------------------------------------------------------------------- #


def test_threshold_reaches_the_parser_for_tango(tmp_path) -> None:
    from aggressor_wrappers.cli.parse import _parser_kwargs
    import argparse

    args = argparse.Namespace(predictor="tango", threshold=10.0)
    assert _parser_kwargs(args) == {"threshold": 10.0}


def test_threshold_is_refused_where_it_would_be_inert() -> None:
    """AGGRESCAN accepts a threshold that does nothing while use_tool_flag is on."""
    from aggressor_wrappers.cli.parse import _parser_kwargs
    import argparse
    import pytest as _pytest

    args = argparse.Namespace(predictor="aggrescan", threshold=0.5)
    with _pytest.raises(ValueError) as excinfo:
        _parser_kwargs(args)
    message = str(excinfo.value)
    assert "aggrescan" in message
    # The message has to say what to do instead, not just refuse.
    assert "use_tool_flag" in message


def test_threshold_absent_means_no_overrides() -> None:
    from aggressor_wrappers.cli.parse import _parser_kwargs
    import argparse

    assert _parser_kwargs(argparse.Namespace(predictor="tango", threshold=None)) == {}


def test_every_mapped_threshold_keyword_is_a_real_parser_argument() -> None:
    """A keyword that no parser declares would raise TypeError at construction.

    The mapping is hand-written, so the invariant is checked rather than trusted:
    every entry must name a parameter its parser's __init__ actually accepts.
    """
    import inspect

    from aggressor_wrappers.cli.parse import _THRESHOLD_KWARG
    from aggressor_wrappers.predictors.registry import PARSER_REGISTRY

    for predictor, keyword in _THRESHOLD_KWARG.items():
        assert predictor in PARSER_REGISTRY, predictor
        parameters = inspect.signature(PARSER_REGISTRY[predictor].__init__).parameters
        assert keyword in parameters, f"{predictor}: {keyword}"


def test_refused_threshold_exits_two_rather_than_writing_a_table(tmp_path) -> None:
    from aggressor_wrappers.cli.parse import main

    fasta = tmp_path / "p.fasta"
    fasta.write_text(">p\nMAVLK\n", encoding="utf-8")
    raw = tmp_path / "p.txt"
    raw.write_text("res\taa\n", encoding="utf-8")
    out = tmp_path / "out.csv"

    code = main(
        [
            "aggrescan",
            "--input",
            str(raw),
            "--fasta",
            str(fasta),
            "-o",
            str(out),
            "--threshold",
            "0.5",
        ]
    )
    assert code == 2
    assert not out.exists()
