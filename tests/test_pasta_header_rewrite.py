"""PASTA renames FASTA headers before naming its outputs.

Member names below are copied from a real ``batch.tar.gz`` returned by
old.protein.bio.unipd.it for a five-sequence job whose accessions carried dots
and a hyphen.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from aggressor_wrappers.runners.pasta import (
    PASTARunner,
    check_member_key_collisions,
    pasta_member_key,
)

# submitted accession -> the name PASTA actually used
OBSERVED = {
    "6087.XP_002162002.2": "6087XP_0021620022",
    "6183.Smp_016050.1": "6183Smp_0160501",
    "7029.ACYPI000203-PA": "7029ACYPI000203PA",
    "7227.FBpp0070001": "7227FBpp0070001",
    "9606.ENSP00000001": "9606ENSP00000001",
}


def test_key_matches_what_the_service_did():
    """Dots and hyphens are deleted; underscores and digits survive."""
    for submitted, produced in OBSERVED.items():
        assert pasta_member_key(submitted) == produced


def _archive(tmp_path: Path) -> Path:
    """A tar shaped like the real one: both profile variants, plus noise."""
    path = tmp_path / "batch.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for index, name in enumerate(OBSERVED.values(), start=1):
            for suffix, payload in (
                (".fasta.seq.aggr_profile.dat", f"agg-{index}\n"),
                (".fasta.seq.aggr_profile.dat.free_energy", f"free-{index}\n"),
                (".fasta.seq.pairing_mat.dat", "noise\n"),
                (".fasta.espritz", "noise\n"),
            ):
                data = payload.encode()
                info = tarfile.TarInfo(f"predictions/{name}{suffix}")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


def test_profiles_are_found_despite_the_rewrite(tmp_path):
    """The failure this reproduces: every accession has a dot, so nothing matched."""
    runner = PASTARunner()
    out = tmp_path / "out"
    out.mkdir()
    mapping = runner._extract_profiles(_archive(tmp_path), list(OBSERVED), out)
    assert set(mapping) == set(OBSERVED)


def test_free_energy_is_the_default_profile(tmp_path):
    """PASTA scores a pairing FREE ENERGY; that is the track to threshold.

    The archive carries both variants and the runner used to take the other one
    while amyloscope's config read .free_energy — the two disagreed about which
    number was PASTA's datum.
    """
    out = tmp_path / "out"
    out.mkdir()
    runner = PASTARunner()
    assert runner.profile_kind == "free_energy"
    mapping = runner._extract_profiles(_archive(tmp_path), ["6087.XP_002162002.2"], out)
    assert mapping["6087.XP_002162002.2"].read_text().startswith("free-")

    other = tmp_path / "other"
    other.mkdir()
    mapping = PASTARunner(profile_kind="aggregation")._extract_profiles(
        _archive(tmp_path), ["6087.XP_002162002.2"], other
    )
    assert mapping["6087.XP_002162002.2"].read_text().startswith("agg-")


def test_result_order_is_not_assumed(tmp_path):
    """PASTA returned 6183 before 6087 though 6087 was submitted first."""
    out = tmp_path / "out"
    out.mkdir()
    reversed_ids = list(reversed(list(OBSERVED)))
    mapping = PASTARunner()._extract_profiles(_archive(tmp_path), reversed_ids, out)
    for protein_id in reversed_ids:
        index = list(OBSERVED).index(protein_id) + 1
        assert mapping[protein_id].read_text().strip() == f"free-{index}"


def test_lossy_rewrite_collisions_are_refused():
    check_member_key_collisions(list(OBSERVED))  # the real batch is fine
    with pytest.raises(ValueError, match="both become"):
        check_member_key_collisions(["FXR1.A", "FXR1A"])


def test_bad_profile_kind_is_rejected():
    with pytest.raises(ValueError, match="profile_kind"):
        PASTARunner(profile_kind="energy")
