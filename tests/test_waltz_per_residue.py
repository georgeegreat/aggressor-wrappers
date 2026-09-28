"""WALTZ: only text_long_graph yields the per-residue .dat files."""

from __future__ import annotations

import io
import zipfile

import pytest

from aggressor_wrappers.runners.waltz import WALTZRunner


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# Abeta42: every residue 0.0 except 16-21 (KLVFFA) at 97.993311 — the live
# value returned by the service at threshold 92.
ABETA_DAT = "\n".join(
    f"{i}\t{97.993311 if 16 <= i <= 21 else 0.0:.6f}" for i in range(1, 43)
).encode()


def test_dat_members_are_indexed_by_submission_order_not_accession():
    assert WALTZRunner._dat_index("WaltzJob_1789996606_1.dat") == 1
    assert WALTZRunner._dat_index("WaltzJob_1789996606_12.dat") == 12
    assert WALTZRunner._dat_index("WaltzJob_1789996606.txt") is None


def test_extract_members_filters_by_suffix():
    blob = _zip({
        "WaltzJob_1_1.dat": ABETA_DAT,
        "WaltzJob_1_1.png": b"\x89PNG",
        "WaltzJob_1.txt": b">ABETA42\n",
    })
    dats = WALTZRunner._extract_members(blob, ".dat")
    assert [name for name, _ in dats] == ["WaltzJob_1_1.dat"]
    assert dats[0][1] == ABETA_DAT


def test_per_residue_mapping_uses_the_submitted_fasta_order(tmp_path, monkeypatch):
    """Archive members carry an index, so the FASTA order is the only key.

    Getting this wrong assigns one protein's track to another silently — the
    failure that has already occurred once in this project with FoldAmyloid.
    """
    fasta = tmp_path / "panel.fasta"
    fasta.write_text(">FIRST\nMKVLAAG\n>SECOND\nMQQWWA\n")
    blob = _zip({
        "WaltzJob_9_1.dat": b"1\t0.000000\n",
        "WaltzJob_9_2.dat": b"1\t97.000000\n",
        "WaltzJob_9.txt": b">FIRST\n",
    })
    runner = WALTZRunner()
    monkeypatch.setattr(runner, "_submit_zip", lambda *a, **k: blob)
    written = runner.fetch_per_residue(fasta, tmp_path)
    assert set(written) == {"FIRST", "SECOND"}
    assert written["FIRST"].read_bytes() == b"1\t0.000000\n"
    assert written["SECOND"].read_bytes() == b"1\t97.000000\n"


def test_absent_dat_members_are_an_error_not_an_empty_result(tmp_path, monkeypatch):
    """text_long returns only a region table; failing loudly names the cause."""
    fasta = tmp_path / "a.fasta"
    fasta.write_text(">ABETA42\nDAEFRHDSGY\n")
    runner = WALTZRunner()
    monkeypatch.setattr(runner, "_submit_zip", lambda *a, **k: _zip({"j.txt": b">ABETA42\n"}))
    with pytest.raises(RuntimeError, match="no .dat members"):
        runner.fetch_per_residue(fasta, tmp_path)
