# Changelog

All notable changes to `aggressor-wrappers` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is `0`, minor bumps may change default behaviour;
such changes are always listed under **Changed** with a migration note.

## [Unreleased]

### Proposed — needs a decision before release

- **`PATH` removed from the default predictor set.** `PATH` is the only
  structure-based predictor and the only one requiring two licensed
  dependencies (Modeller + PyRosetta), yet it currently runs by default
  (`predictors = path,appnn,...`), which makes the whole pipeline uninstallable
  without both licences and unusable in CI. Proposal: keep the wrapper, move
  `PATH` behind an optional extra (`pip install .[path]`) and out of the
  defaults. It supplies the panel's only orthogonal (structural) evidence, so it
  is worth keeping as an opt-in rather than deleting.
  *Not applied — this changes default output and is the maintainers' call.*

## [0.4.0] — 2026-07-11


First release with a changelog. Three themes: predictors now run concurrently,
the metascore is pluggable and no longer silently mis-combines its inputs, and
the wrappers stop discarding each instrument's distinctive output.

### Added

- **Predictor-level concurrency** (`batch/scheduler.py`). Predictors are mutually
  independent — each owns its `work/` and `parsed/` directories — so they are now
  launched concurrently instead of one after another. Wall time becomes
  `max(T_predictor)` rather than `sum(T_predictor)`; on a runner-cost profile
  taken from `config.cfg` this is 4.40 s → 1.20 s. Each runner keeps its own
  `[runners.*] parallel_jobs` / `sequences_per_run` caps, so **no external
  service sees more concurrency than before** (AggreProt's 3-sequences-per-job
  limit, ArchCandy's 1, and so on are untouched).
- **Failure isolation.** A predictor that raises (dead web service, missing
  licence) no longer aborts the panel; the remaining predictors still produce
  parsed output, and the run only fails if *every* predictor failed.
- **`[pipeline] predictor_jobs`** config key (default `4`). Set to `1` to restore
  the previous strictly-sequential behaviour exactly.
- **Pluggable metascore registry** (`core/metascore_plugins.py`).
  `[metascore] method` now dispatches through a registry, so an external
  consensus can be plugged in without editing this package:

  ```python
  @register_metascore("my_consensus")
  def my_consensus(wide_df, *, config, weights=None): ...
  ```

- **`zscore_consensus`** metascore method — standardises each predictor's scores
  and applies an explicit polarity before weighting, so the configured weights
  finally express confidence rather than absorbing a unit conversion.
- **`fractional_consensus`** metascore method — combines the per-tool
  `{predictor}_bin` columns rather than raw scores. Because those calls were
  produced by each parser with that tool's own threshold *and* direction, this
  combiner is scale-free and polarity-correct by construction.
- **Declared predictor polarity** (`HIGHER_IS_AMYLOIDOGENIC`). PASTA is inverted
  (lower energy = more amyloidogenic); this was known at binarisation time but
  absent from `PredictorSpec`. It belongs in `schema.py` and should be folded in.
- **Auxiliary output channels** on `PredictorResult`:
  - `aux` — extra per-residue columns an instrument produced alongside its score
    (length-validated against the sequence);
  - `regions` — an instrument's native region-level records, which have no
    per-residue representation at all.
- **`to_dataframe(include_aux=True)`** and
  **`merge_predictor_tables(..., include_aux=True)`** to emit those channels,
  namespaced `{predictor}_{name}` so they cannot collide across tools.
- **amyloscope bridge** (`core/amyloscope_export.py`). Exports merged tables as
  amyloscope tracks plus a ready-to-run config, so this package can stay the
  *execution* layer while amyloscope does tiered consensus, APR calling, domain
  mapping and figures. The handoff is the `{predictor}_bin` columns, which are
  already scale-free and polarity-correct.
- Tests: `test_scheduler.py`, `test_metascore_plugins.py`, `test_fidelity.py`,
  `test_amyloscope_export.py` (21 new tests; suite now 104 passed, 9 skipped).
- CI, the homoglyph guard, the tag/version check, .gitattributes

### Fixed

- **AggreProt no longer discards `sasa` and `transmembrane`.** The parser
  explicitly dropped both. Solvent accessibility is not redundant with the
  aggregation score — it is what determines whether an aggregation-prone segment
  is sterically *available* to pair — and transmembrane hydrophobicity is a
  known false-positive source for hydrophobicity-driven predictors. Both are now
  retained as `aux` channels.
- **ArchCandy no longer discards β-arch topology.** ArchCandy predicts discrete
  β-arches, each carrying a topology string (`GBPL`, `BLLPBL`, …) describing the
  structural motif. That string — the tool's distinguishing output — was never
  read. Arches are now retained in `regions`, with a per-residue best-arch `aux`
  channel.
- **ArchCandy warns when `score_mode = "cumulative"` leaves its native scale.**
  Cumulative mode *sums* the scores of overlapping arches, so the per-residue
  value becomes confidence × multiplicity: unbounded above and no longer
  comparable to ArchCandy's own threshold. Concretely, three overlapping arches
  each scoring `0.15` — every one well below the `0.4` cutoff — sum to `0.45` and
  *clear* it, while a single arch at `0.39` does not. A `UserWarning` now fires
  whenever summing inflates a residue above the best single arch, and names
  `score_mode = "highest"` as the fix. **The default is unchanged**, because
  changing it would alter every number produced so far; see *Deprecated*.
- the Cyrillic с import in waltz.py

### Changed

- **Predictors now run concurrently by default** (`predictor_jobs = 4`). Parsed
  output is identical; only scheduling changes. Log lines from different
  predictors may now interleave (each is prefixed with its predictor tag, and
  writes are serialised so lines are never torn). Set `predictor_jobs = 1` to
  restore the previous behaviour.
- **The version is now single-sourced** from `src/__init__.py`; `pyproject.toml`
  declares `dynamic = ["version"]`. It was previously declared in *both* files
  (`0.3.2` in each), which drifts silently and breaks tag-vs-package
  verification at release time.
- **BREAKING**: PATH removed from the default predictor set (with the re-enable instructions)

### Deprecated

- **`[metascore] method = weighted_sum`.** Retained, and byte-identical, so
  existing results stay reproducible — but it is not recommended, for two
  reasons that follow from the code and config alone:
  1. *Scale incommensurability.* It sums **raw** scores across incomparable
     scales (WALTZ/PATH on 0–100, APPNN/CrossBeta/ArchCandy on 0–1, Aggrescan on
     ≈±1, PASTA on a free energy of ≈ −8–0). With the shipped
     `predictor_specificity` preset, WALTZ spans ~22 metascore units while
     ArchCandy spans ~0.05 — so a predictor's influence is set by its *units*,
     not its weight. Consequently the three shipped presets rank residues
     near-identically (Spearman ρ = 0.997–1.000): the weight tuning they exist to
     express is swamped by scale.
  2. *Polarity inversion.* PASTA's score is a pairing free energy where *lower* is
     more amyloidogenic, but it is added with a positive weight — so strong PASTA
     evidence *lowers* the metascore. In a controlled simulation, **removing
     PASTA from the panel improves discrimination** (AUC 0.717 → 0.732), while
     correcting scale and polarity recovers it (0.910).

  Prefer `fractional_consensus` (scale-free, polarity-correct by construction) or
  `zscore_consensus`.
- **`[predictors.archcandy] score_mode = "cumulative"`** — still the default for
  reproducibility, but `"highest"` is the defensible setting for consensus use.

### Notes / not yet done

- **PASTA β-pairing partners are still discarded.** `npair` is already sent to the
  service, so the pairings are computed and downloaded — then filtered out of the
  archive by `_extract_profiles`, which keeps only `.aggr_profile.dat`. Pairing
  partners and their parallel/antiparallel orientation are PASTA's distinctive
  structural output. This is the largest remaining recoverable loss and needs a
  real PASTA archive to write the parser against.
- **PATH's `molpdf` / `dope` / `ga341`** are collapsed into a single scalar; each
  could become an `aux` channel with the same pattern used for AggreProt.
- Raw predictor output is deleted after parsing (`work/` is removed unless
  `--save-raw-files`). Combined with the losses above, discarded information is
  **unrecoverable without re-running the predictor** — a full re-queue for the web
  services, and re-running Modeller + PyRosetta for PATH. Consider defaulting
  `--save-raw-files` on, or keeping the raw archives in the cache.

## [0.3.2] — 2026-07-10

Baseline as inherited (no changelog kept before this release). Reconstructed
from the commit history:

### Added

- Resume support for interrupted multifasta runs, validating parsed CSVs against
  the input FASTA (length and `aa_name`) rather than trusting timestamps.
- AggreProt predictor (runner + parser).
- Cross-Beta predictor (runner + parser).
- ArchCandy predictor (runner + parser).
- PATH, APPNN, WALTZ, PASTA wrappers; config-driven thresholds and weights;
  caching, golden tests, and the `parse` / `merge` / `run` / `widemerge` CLI.

[Unreleased]: https://github.com/OWNER/aggressor-wrappers/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/OWNER/aggressor-wrappers/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/OWNER/aggressor-wrappers/releases/tag/v0.3.2
