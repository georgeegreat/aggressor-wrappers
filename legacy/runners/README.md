# Superseded runners

Runners kept for reference after the service they drove changed or disappeared.
Nothing here is imported by the package: `pyproject.toml` packages only `src/`,
so these files are not installed and cannot be reached from `runners/registry.py`.

Naming: `<tool>_v<n>_deprecated.py`, where the version is the **tool's**, not the
file's. A file here is superseded, erroneous, or points at a dead route; the
header of each says which, and when.

| file | why it is here |
|---|---|
| `archcandy_v1_deprecated.py` | drove `index.php?route=tools&tool=7`, which the rebuilt bioinfo.crbm.cnrs.fr no longer serves — a dead link, not a bug |
| `crossbeta_v1_deprecated.py` | reCAPTCHA-gated web form; refused to run by design, and no registry key could construct it |

The live equivalents are `src/runners/archcandy_v2.py` and
`src/runners/crossbeta_v2.py`. **Parsers are not affected**: reading historical
output from either tool still goes through `src/predictors/archcandy.py` and
`src/predictors/crossbeta.py`, which remain live.
