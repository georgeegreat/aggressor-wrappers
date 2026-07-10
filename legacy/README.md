# Legacy reference scripts (from BHT_amyloid)

Frozen copies of hackathon scripts kept inside the package for traceability and
runner integration. **Do not import these modules at runtime** — use
`aggressor_wrappers.predictors` and `aggressor_wrappers.runners` instead.

| File | Original role | Package replacement |
|------|---------------|---------------------|
| `path_converter.py` | PATH batch + APR export | `predictors/path.py`, `runners/path.py` |
| `appnn_converter.R` | APPNN R runner | `runners/appnn.py` (invokes this script) |
| `pasta_pipeline.py` | PASTA 2.0 Selenium + tar extract | `runners/pasta.py` (HTTP runner) |
| `arch_cross_pasta_aggreprot_waltz_parser.ipynb` | Notebook parsers | `predictors/*.py` |
| `parse_predictor.py` | Thin CLI shim | `aggressor-parse` |
| `api/aggreprot.py` | Selenium runner (obsolete URL) | `runners/aggreprot.py` |
| `api/cross_candy.py` | CRBM web runner | Phase 3 (planned) |
| `api/PASTA 2.0.py` | PASTA Selenium prototype | Superseded by `runners/pasta.py` |

Source of truth for edits: update aggressor_wrappers first; refresh legacy copies when the
reference script changes intentionally.
