"""Raw output obtained on another machine must be reusable where a user puts it.

``--skip-run`` searched exactly one directory: ``{PREDICTOR}/work/batch_N/``. The
batch index is chosen by the scheduler from the configured chunk size, not by the
user, so filing imported output correctly meant predicting it. For TANGO that is
not an edge case: the distributed binary is a Linux x86_64 ELF, so on an arm64 Mac
it cannot be executed AT ALL and its output can only ever arrive from elsewhere.

The drop zone is the predictor directory itself. These tests pin the three
behaviours that make it usable: it is searched, it is MERGED with whatever a local
run left behind rather than losing to it, and a batch it covers only in part is
reported instead of raising KeyError on a protein name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aggressor_wrappers.batch.pipeline import (
    BatchLayout,
    _parse_and_write_runner_batch,
    _run_generic_batch,
)


class _Runner:
    """Minimal stand-in exposing the batch protocol's discovery half."""

    class spec:  # noqa: D106 - mimics PredictorSpec.display_name only
        display_name = "TANGO"

    def __init__(self) -> None:
        self.executed = False

    def discover_outputs(
        self, output_dir: Path, protein_ids: list[str]
    ) -> dict[str, Path]:
        found = {}
        for pid in protein_ids:
            candidate = Path(output_dir) / f"{pid}.txt"
            if candidate.exists():
                found[pid] = candidate
        return found

    def execute_batch(self, batch_fasta: Path, batch_work: Path) -> Path:
        self.executed = True
        raise AssertionError("execute_batch must not run under --skip-run")


def _raw(directory: Path, *protein_ids: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pid in protein_ids:
        (directory / f"{pid}.txt").write_text("res\taa\n01\tM\n", encoding="utf-8")


def _skip_run(
    tmp_path: Path,
    protein_ids: list[str],
    *,
    save_raw_files: Path | None = None,
) -> tuple[dict[str, Path], list[str]]:
    log: list[str] = []
    runner = _Runner()
    raw_map = _run_generic_batch(
        runner,
        "tango",
        tmp_path / "tango" / "work" / "batch_1.fasta",
        tmp_path / "tango" / "work" / "batch_1",
        protein_ids,
        skip_run=True,
        save_raw_files=save_raw_files,
        batch_label="1/1",
        emit=log.append,
        drop_zone=tmp_path / "tango",
    )
    assert runner.executed is False
    return raw_map, log


def test_drop_zone_is_searched(tmp_path):
    """Files in {PREDICTOR}/ are found without a work/batch_N/ directory."""
    _raw(tmp_path / "tango", "RPS2", "RPS28")

    raw_map, log = _skip_run(tmp_path, ["RPS2", "RPS28"])

    assert set(raw_map) == {"RPS2", "RPS28"}
    assert raw_map["RPS2"] == tmp_path / "tango" / "RPS2.txt"
    assert any("drop zone" in line for line in log)


def test_work_and_drop_zone_are_merged_not_tried_in_turn(tmp_path):
    """A part-local, part-imported panel must yield every protein.

    The first non-empty hit used to win outright, so one file left behind by an
    aborted local run hid every imported file in the drop zone.
    """
    _raw(tmp_path / "tango" / "work" / "batch_1", "RPS2")
    _raw(tmp_path / "tango", "RPS28", "RPL27")

    raw_map, _ = _skip_run(tmp_path, ["RPS2", "RPS28", "RPL27"])

    assert set(raw_map) == {"RPS2", "RPS28", "RPL27"}
    # The local run's file wins for the protein it covers: it was produced by
    # the configuration this output directory records.
    assert raw_map["RPS2"].parent.name == "batch_1"
    assert raw_map["RPL27"].parent.name == "tango"


def test_nothing_anywhere_names_the_drop_zone(tmp_path):
    """The error has to say where to put the files, not only where it looked."""
    (tmp_path / "tango").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        _skip_run(tmp_path, ["RPS2"])

    message = str(excinfo.value)
    assert "drop zone" in message
    assert str(tmp_path / "tango") in message


def test_partial_batch_is_reported_not_a_keyerror(tmp_path, monkeypatch):
    """Six of seven proteins must parse, with the seventh named.

    ``_parse_and_write_runner_batch`` indexed raw_map for every id in the batch,
    so a drop zone short by one file raised KeyError('RPL27') -- which names
    neither the predictor nor the directory that was short.
    """
    layout = BatchLayout.create(tmp_path / "out")
    layout.ensure(["tango"])
    (layout.fasta_split_dir / "RPS2.fasta").write_text(">RPS2\nMAVLK\n", encoding="utf-8")

    parsed: list[str] = []

    class _ParsingRunner(_Runner):
        def run(self, *, fasta, protein_id, work_dir, skip_run, raw_csv, **kwargs):
            parsed.append(protein_id)

            class _Result:
                def to_csv(self, path):
                    Path(path).write_text("position,aa_name\n1,M\n", encoding="utf-8")

            return _Result()

    log: list[str] = []
    _parse_and_write_runner_batch(
        _ParsingRunner(),
        "tango",
        ["RPS2", "RPL27"],
        {"RPS2": tmp_path / "RPS2.txt"},
        layout,
        tmp_path / "work",
        batch_label="1/1",
        skip_run=True,
        save_raw_files=None,
        emit=log.append,
    )

    assert parsed == ["RPS2"]
    assert any("RPL27" in line and "no raw output" in line for line in log)
    assert any("1 of 2" in line for line in log)


# --------------------------------------------------------------------------- #
# --merge-existing: fold one more tool into a finished panel.
#
# The merge step took exactly the predictors named in this invocation, so adding
# a tool to an output tree that already held eight meant either re-running all
# nine or overwriting eight-tool merged tables with a one-tool table. Raw files
# are not always still there to re-parse from (several runners clean work/), so
# "re-run everything" is not always available either.
# --------------------------------------------------------------------------- #


def _parsed(layout, key: str, protein_ids: list[str], *, tag: str | None = None) -> None:
    from aggressor_wrappers.batch.pipeline import _predictor_tag

    tag = tag or _predictor_tag(key)
    directory = layout.predictor_parsed_dir(key)
    directory.mkdir(parents=True, exist_ok=True)
    for pid in protein_ids:
        (directory / f"{pid}_{tag}.csv").write_text(
            f"position,aa_name,{tag}_score,{tag}_bin\n1,M,0.1,0\n", encoding="utf-8"
        )


def test_widening_finds_predictors_already_parsed(tmp_path):
    from aggressor_wrappers.batch.pipeline import (
        BatchLayout,
        _widen_with_existing_parsed,
    )

    layout = BatchLayout.create(tmp_path / "out")
    _parsed(layout, "aggrescan", ["RPS2"])
    _parsed(layout, "pasta", ["RPS2"])

    log: list[str] = []
    widened = _widen_with_existing_parsed(layout, ["tango"], emit=log.append)

    # Requested predictors stay first, so column order does not shuffle.
    assert widened[0] == "tango"
    assert set(widened) == {"tango", "aggrescan", "pasta"}
    # Every addition is named: it changes what a fractional consensus divides by.
    assert any("aggrescan" in line for line in log)
    assert any("pasta" in line for line in log)


def test_one_parsed_file_is_never_merged_twice(tmp_path):
    """archcandy and archcandy2 are two parsers sharing one filename.

    Widening by registry key added {id}_ArchCandy.csv under both keys and the
    merge refused the whole batch with "Duplicate predictor in merge input" --
    after every tool had already been parsed.
    """
    from aggressor_wrappers.batch.pipeline import (
        BatchLayout,
        _predictor_tag,
        _widen_with_existing_parsed,
    )

    layout = BatchLayout.create(tmp_path / "out")
    _parsed(layout, "archcandy", ["RPS2"])

    log: list[str] = []
    widened = _widen_with_existing_parsed(layout, ["tango"], emit=log.append)

    tags = [_predictor_tag(key) for key in widened]
    assert len(tags) == len(set(tags)), tags
    assert widened.count("archcandy") + widened.count("archcandy2") == 1
    # The ambiguity is stated rather than resolved silently.
    assert any("share one filename" in line for line in log)


def test_a_requested_predictor_is_not_added_again(tmp_path):
    from aggressor_wrappers.batch.pipeline import (
        BatchLayout,
        _widen_with_existing_parsed,
    )

    layout = BatchLayout.create(tmp_path / "out")
    _parsed(layout, "pasta", ["RPS2"])

    log: list[str] = []
    widened = _widen_with_existing_parsed(layout, ["pasta"], emit=log.append)

    assert widened == ["pasta"]
    assert any("no other predictor" in line for line in log)


def test_empty_parsed_directory_adds_nothing(tmp_path):
    from aggressor_wrappers.batch.pipeline import (
        BatchLayout,
        _widen_with_existing_parsed,
    )

    layout = BatchLayout.create(tmp_path / "out")
    layout.predictor_parsed_dir("pasta").mkdir(parents=True)

    widened = _widen_with_existing_parsed(layout, ["tango"], emit=lambda _: None)

    assert widened == ["tango"]


def test_skip_run_does_not_require_the_executable():
    """--skip-run must work where the tool cannot run at all.

    TANGO ships a Linux x86_64 ELF, so on an arm64 Mac the binary can never
    execute and its output can only ever be obtained elsewhere -- yet
    backend = local refused at CONSTRUCTION with "the local tool was not found",
    before any file was looked at, which made --skip-run unreachable for exactly
    the tool it exists for.
    """
    from aggressor_wrappers.runners.registry import _resolve_class

    options = {"backend": "local", "path": "/nonexistent/tango_x86_64_release"}

    with pytest.raises(Exception):
        _resolve_class("tango", dict(options))

    cls = _resolve_class("tango", dict(options), parse_only=True)
    # The declared policy still chooses the class: raw output from a binary is
    # not shaped like raw output from the same tool's web service.
    assert "Local" in cls.__name__ or cls.__name__ == "TANGORunner"


# --------------------------------------------------------------------------- #
# One unreadable file must cost one protein.
#
# A seven-protein WALTZ batch lost RPS13 to a length mismatch in RPS12: the
# parse loop raised out of itself, the five proteins before RPS12 had already
# been written, and the one after never got its turn. Nothing in the log named
# either protein. The scheduler already treats a refused SUBMISSION this way --
# "a single sequence carrying an X should cost that protein, not the other 371"
# -- and the parse step did not.
# --------------------------------------------------------------------------- #


class _SelectivelyBrokenRunner(_Runner):
    """Parses everything except the proteins named in ``broken``."""

    def __init__(self, broken: set[str]) -> None:
        super().__init__()
        self.broken = broken
        self.attempted: list[str] = []

    def run(self, *, fasta, protein_id, work_dir, skip_run, raw_csv, **kwargs):
        self.attempted.append(protein_id)
        if protein_id in self.broken:
            raise ValueError(f"{raw_csv}: not this protein's output")

        class _Result:
            def to_csv(self, path):
                Path(path).write_text("position,aa_name\n1,M\n", encoding="utf-8")

        return _Result()


def _layout_with_fastas(tmp_path, protein_ids):
    from aggressor_wrappers.batch.pipeline import BatchLayout

    layout = BatchLayout.create(tmp_path / "out")
    layout.ensure(["tango"])
    for pid in protein_ids:
        (layout.fasta_split_dir / f"{pid}.fasta").write_text(
            f">{pid}\nMAVLK\n", encoding="utf-8"
        )
    return layout


def test_a_bad_file_does_not_cost_the_proteins_after_it(tmp_path):
    ids = ["RPS2", "RPS28", "RPL27", "RPS12", "RPS13"]
    layout = _layout_with_fastas(tmp_path, ids)
    runner = _SelectivelyBrokenRunner({"RPS12"})
    log: list[str] = []

    parsed, failures = _parse_and_write_runner_batch(
        runner,
        "tango",
        ids,
        {pid: tmp_path / f"{pid}.txt" for pid in ids},
        layout,
        tmp_path / "work",
        batch_label="1/1",
        skip_run=True,
        save_raw_files=None,
        emit=log.append,
    )

    assert runner.attempted == ids          # RPS13 was reached
    assert parsed == 4
    assert [pid for pid, _ in failures] == ["RPS12"]
    written = {p.name for p in layout.predictor_parsed_dir("tango").glob("*.csv")}
    assert "RPS13_tango.csv" in written
    assert "RPS12_tango.csv" not in written


def test_the_failure_names_the_protein_and_the_file(tmp_path):
    layout = _layout_with_fastas(tmp_path, ["RPS12"])
    log: list[str] = []

    _parse_and_write_runner_batch(
        _SelectivelyBrokenRunner({"RPS12"}),
        "tango",
        ["RPS12"],
        {"RPS12": tmp_path / "WaltzJob_1_RPS12.dat"},
        layout,
        tmp_path / "work",
        batch_label="1/1",
        skip_run=True,
        save_raw_files=None,
        emit=log.append,
    )

    joined = "\n".join(log)
    assert "RPS12 FAILED to parse" in joined
    assert "WaltzJob_1_RPS12.dat" in joined
    assert "ValueError" in joined


def test_a_single_protein_batch_that_fails_does_not_raise(tmp_path):
    """With sequences_per_run = 1 a batch IS one protein.

    Raising when every file in a batch failed therefore fired on every single
    unreadable file, and stopped FoldAmyloid at the second of seven proteins.
    """
    layout = _layout_with_fastas(tmp_path, ["RPS28"])

    parsed, failures = _parse_and_write_runner_batch(
        _SelectivelyBrokenRunner({"RPS28"}),
        "tango",
        ["RPS28"],
        {"RPS28": tmp_path / "RPS28.csv"},
        layout,
        tmp_path / "work",
        batch_label="2/7",
        skip_run=True,
        save_raw_files=None,
        emit=lambda _: None,
    )

    assert parsed == 0
    assert len(failures) == 1


# --------------------------------------------------------------------------- #
# Where --skip-run looks.
# --------------------------------------------------------------------------- #


def test_sibling_batch_directories_are_searched(tmp_path):
    """The batch partition is an artefact of this run's chunk size.

    Someone adding files by hand puts all of them in one directory. With
    sequences_per_run = 1 the scheduler then looked in batch_2 .. batch_7,
    which did not exist, and found six of seven files nowhere.
    """
    from aggressor_wrappers.batch.pipeline import _skip_run_directories

    work = tmp_path / "foldamyloid" / "work"
    for n in (1, 2, 3):
        (work / f"batch_{n}").mkdir(parents=True)

    found = _skip_run_directories(work / "batch_2", tmp_path / "foldamyloid")
    names = [path.name for _, path in found]

    assert names[0] == "batch_2"                 # this batch first
    assert "batch_1" in names and "batch_3" in names
    assert "work" in names
    assert names[-1] == "foldamyloid"            # the drop zone last


def test_the_parsed_directory_is_never_searched(tmp_path):
    """A parsed table is named {id}_{tag}.csv.

    That is exactly the canonical RAW name for some predictors -- FoldAmyloid
    among them -- so reaching into parsed/ would feed a parser its own output.
    """
    from aggressor_wrappers.batch.pipeline import _skip_run_directories

    root = tmp_path / "foldamyloid"
    (root / "work" / "batch_1").mkdir(parents=True)
    (root / "parsed").mkdir()

    found = _skip_run_directories(root / "work" / "batch_1", root)

    assert all(path.name != "parsed" for _, path in found)


def test_no_directory_is_searched_twice(tmp_path):
    from aggressor_wrappers.batch.pipeline import _skip_run_directories

    work = tmp_path / "tango" / "work"
    (work / "batch_1").mkdir(parents=True)

    found = _skip_run_directories(work / "batch_1", tmp_path / "tango")
    paths = [path for _, path in found]

    assert len(paths) == len(set(paths))
