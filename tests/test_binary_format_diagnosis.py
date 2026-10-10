"""Errno 8 has two causes with opposite remedies, so the runner must say which.

"[Errno 8] Exec format error: '/path/tango_x86_64_release'" names the file and not
the reason. A Linux ELF on macOS needs a different build; a Mach-O x86-64 on Apple
Silicon needs one softwareupdate command. The filename cannot distinguish them --
TANGO's distributed names carry the architecture and not the OS -- but the first
twenty bytes can.
"""

from __future__ import annotations

from unittest import mock

import pytest

import aggressor_wrappers.runners.tango as tango_module
from aggressor_wrappers.runners.tango import (
    describe_binary,
    exec_format_report,
    runnable_here,
)


def _elf(machine: int, *, width: int = 2) -> bytes:
    """An ELF header to spec: 16-byte e_ident, 2-byte e_type, then e_machine.

    Written out because the first version of this test put e_machine at offset 17
    and read back "unknown architecture" for a valid x86-64 header -- which looked
    like a bug in the parser rather than in the fixture.
    """
    ident = bytes([0x7F]) + b"ELF" + bytes([width, 1, 1]) + bytes(9)
    assert len(ident) == 16
    return ident + bytes(2) + machine.to_bytes(2, "little")


def _macho(cpu: int) -> bytes:
    return bytes.fromhex("cffaedfe") + cpu.to_bytes(4, "little") + bytes(12)


@pytest.mark.parametrize(
    "payload,expected",
    [
        (_elf(0x3E), "Linux/ELF 64-bit x86-64"),
        (_elf(0xB7), "Linux/ELF 64-bit aarch64"),
        (_elf(0x03, width=1), "Linux/ELF 32-bit i386"),
        (_macho(0x01000007), "macOS/Mach-O x86_64"),
        (_macho(0x0100000C), "macOS/Mach-O arm64"),
        (bytes.fromhex("cafebabe") + bytes(16), "universal (fat)"),
        (b"#!/usr/bin/env python3\n", "script"),
        (b"MZ" + bytes(18), "Windows PE"),
        (b"PK\x03\x04" + bytes(16), "unrecognised"),
    ],
)
def test_format_is_read_from_the_magic_bytes(tmp_path, payload, expected):
    binary = tmp_path / "tool"
    binary.write_bytes(payload)
    assert expected in describe_binary(binary)


def test_a_short_or_missing_file_is_reported_not_crashed(tmp_path):
    short = tmp_path / "short"
    short.write_bytes(b"ab")
    assert "too short" in describe_binary(short)
    assert "unreadable" in describe_binary(tmp_path / "absent")


@pytest.mark.parametrize(
    "host,kind_payload,runnable",
    [
        ("Darwin", _elf(0x3E), False),
        ("Darwin", _macho(0x01000007), True),
        ("Linux", _elf(0x3E), True),
        ("Linux", _macho(0x0100000C), False),
        ("Linux", bytes.fromhex("cafebabe") + bytes(16), False),
        ("Windows", _elf(0x3E), False),
    ],
)
def test_runnable_here_depends_on_the_host(tmp_path, host, kind_payload, runnable):
    binary = tmp_path / "tool"
    binary.write_bytes(kind_payload)
    with mock.patch.object(tango_module.platform, "system", return_value=host):
        assert runnable_here(describe_binary(binary)) is runnable


def test_the_remedy_matches_the_host_and_the_file(tmp_path):
    """An ELF on macOS: a different build. Not Rosetta, which cannot help."""
    binary = tmp_path / "tango_x86_64_release"
    binary.write_bytes(_elf(0x3E))
    with mock.patch.object(tango_module.platform, "system", return_value="Darwin"), \
         mock.patch.object(tango_module.platform, "machine", return_value="arm64"):
        report = exec_format_report(binary)
    assert "Linux/ELF" in report and "Darwin arm64 host" in report
    assert "cannot execute that format at all" in report
    assert "will not help" in report, "Rosetta must be ruled OUT, not suggested"
    assert "install-rosetta" not in report


def test_an_intel_mac_binary_on_apple_silicon_gets_the_rosetta_remedy(tmp_path):
    binary = tmp_path / "tool"
    binary.write_bytes(_macho(0x01000007))
    with mock.patch.object(tango_module.platform, "system", return_value="Darwin"), \
         mock.patch.object(tango_module.platform, "machine", return_value="arm64"):
        report = exec_format_report(binary)
    assert "install-rosetta" in report


def test_a_linux_host_is_not_told_about_macos(tmp_path):
    """The first version keyed its advice on the FILE alone and told a Linux host
    that macOS cannot run ELF binaries -- true, and useless."""
    binary = tmp_path / "tool"
    binary.write_bytes(_elf(0x3E))
    with mock.patch.object(tango_module.platform, "system", return_value="Linux"), \
         mock.patch.object(tango_module.platform, "machine", return_value="x86_64"):
        report = exec_format_report(binary)
    assert "Rosetta" not in report
    assert "runnable on this host" in report


def test_availability_refuses_a_binary_this_host_cannot_exec(tmp_path):
    """Before the panel starts, not at hour three of a sweep."""
    from aggressor_wrappers.runners.tango import TANGORunner

    binary = tmp_path / "tango_x86_64_release"
    binary.write_bytes(_elf(0x3E))
    binary.chmod(0o755)
    runner = TANGORunner(binary_path=str(binary))
    with mock.patch.object(tango_module.platform, "system", return_value="Darwin"), \
         mock.patch.object(tango_module.platform, "machine", return_value="arm64"):
        with pytest.raises(FileNotFoundError, match="cannot execute that format"):
            runner.require_available()


def test_a_runnable_binary_passes_availability(tmp_path):
    from aggressor_wrappers.runners.tango import TANGORunner

    binary = tmp_path / "tango"
    binary.write_bytes(_macho(0x0100000C))
    binary.chmod(0o755)
    runner = TANGORunner(binary_path=str(binary))
    with mock.patch.object(tango_module.platform, "system", return_value="Darwin"), \
         mock.patch.object(tango_module.platform, "machine", return_value="arm64"):
        runner.require_available()   # must not raise
