#!/usr/bin/env python3
"""Fail if Python source contains Cyrillic characters outside strings/comments.

Motivation, from a real bug in this repository::

    from .сore.schema import PredictorResult      # 'с' is U+0441, Cyrillic es

That line is indistinguishable by eye from ``from .core.schema import ...`` but
produces ``ModuleNotFoundError: No module named '...predictors.сore'``. It is an
easy slip on a Russian keyboard layout, it survives code review because the
glyphs are identical, and it breaks the import graph for every downstream module.

Cyrillic text is legitimate in *strings* (Russian log messages, docstrings) and
in *comments*, so only code is checked: the file is tokenised and any Cyrillic
character appearing in a NAME, OP, or NUMBER token is reported. That keeps the
guard useful without forbidding Russian prose.

Exit code 1 on any hit, so it can be wired into CI or a pre-commit hook.
"""

from __future__ import annotations

import io
import sys
import token
import tokenize
import unicodedata
from pathlib import Path

# Cyrillic block; the confusable subset (а, с, е, о, р, х, …) all live here.
CYRILLIC = range(0x0400, 0x0500)

# Tokens that become identifiers/syntax — Cyrillic here is always a bug.
CODE_TOKENS = {token.NAME, token.OP, token.NUMBER}

ROOTS = ("src", "tests", "scripts")


def offending_tokens(path: Path):
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return  # not our job to report syntax errors
    for tok in tokens:
        if tok.type not in CODE_TOKENS:
            continue
        bad = [ch for ch in tok.string if ord(ch) in CYRILLIC]
        if bad:
            yield tok, bad


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures = 0
    for base in ROOTS:
        for path in sorted((root / base).rglob("*.py")):
            for tok, bad in offending_tokens(path):
                failures += 1
                names = ", ".join(
                    f"{ch!r} (U+{ord(ch):04X} {unicodedata.name(ch, '?')})" for ch in bad
                )
                rel = path.relative_to(root)
                print(f"{rel}:{tok.start[0]}: Cyrillic character in code: {names}")
                print(f"    {tok.line.rstrip()}")
                print(f"    token: {tok.string!r}")
    if failures:
        print(
            f"\n{failures} Cyrillic character(s) found in code tokens. These are "
            f"visually identical to Latin letters but are different characters, and "
            f"will break imports or create silently-unreachable names.",
            file=sys.stderr,
        )
        return 1
    print("no Cyrillic homoglyphs in code tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
