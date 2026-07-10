# aggressor-wrappers (aka Wrappers for AGGRESSOR)

Installable Python package that normalises per-residue outputs from amyloidogenicity  
predictors into one schema (`position`, `aa_name`, `{Tool}_score`, `{Tool}_bin`) and  
merges them into wide CSV tables for metascores and downstream analysis.

## Install

1. Clone this repo

```bash
git clone https://github.com/georgeegreat/aggressor-wrappers.git
```

1. Before installation it is recommended to create a separate conda environment. For this, run setup_conda_env.sh in root directory:

```bash
bash ./setup_conda_env.sh
```

or manually (execute in root directory):

```bash
conda env create -f environment.yml # Python 3.11 + Modeller; pip deps from requirements.txt
conda activate AGGRESSOR
pip install -e ".[test]"
```

The environment is named AGGRESSOR and includes numpy, pandas, biopython, scikit-learn, modeller, pyrosetta and some R packages. 

Python-only deps are declared in `pyproject.toml`; Installation of external tools are described below

### Verify

```bash
which python pytest aggressor-parse aggressor-run
python -c "import aggressor_wrappers, Bio, sklearn; print('OK')"
python -m pytest
```



### External tools


| Tool          | Install                                                                                                                                         | Used by                         |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Modeller**  | `conda install -c salilab modeller` (in `environment.yml`)                                                                                      | PATH (`vendor/PATH/path1.1.py`) |
| **PyRosetta** | `pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release` or [pyrosetta.org](https://www.pyrosetta.org/) | PATH                            |
| **APPNN** (R) | R + CRAN package `appnn` (+ `dplyr`, `tidyr`, `readr`, `stringr`, `purrr`); runner calls `legacy/appnn_converter.R`                             | APPNN                           |
| **WALTZ**     | Public web service — no local install ([waltz.switchlab.org](https://waltz.switchlab.org/))                                                    | WALTZ                           |
| **PASTA 2.0** | Public web service — no local install ([old.protein.bio.unipd.it/pasta2/](http://old.protein.bio.unipd.it/pasta2/))                            | PASTA                           |
| **ArchCandy** | Public web service — no local install ([bioinfo.crbm.cnrs.fr/archCandy-input](https://bioinfo.crbm.cnrs.fr/archCandy-input))                  | ArchCandy                       |


**APPNN (R) install** — if system site-library is not writable, packages go to the user library automatically:

```bash
Rscript -e 'install.packages(c("appnn","dplyr","tidyr","readr","stringr","purrr"), repos="https://cloud.r-project.org")'
Rscript -e 'library(appnn); cat("appnn OK\n")'
```

For how to install and use Rscript and other R packages, please use [https://cran.r-project.org/](https://cran.r-project.org/)

**Modeller license** (required for PATH live runs):

```bash
export KEY_MODELLER="YOUR_LICENSE_KEY"    # add to ~/.bashrc
```

Edit `$CONDA_PREFIX/lib/modeller-10.8/modlib/modeller/config.py`:

```python
install_dir = r'/path/to/anaconda3/envs/AGGRESSOR/lib/modeller-10.8'
license = r'YOUR_LICENSE_KEY'
```

Tested pip/conda versions (env `AGGRESSOR`, Python 3.11): numpy 2.4.6, pandas 3.0.3,
biopython 1.87, scikit-learn 1.9.0, modeller 10.8, pyrosetta 2026.3+releasequarterly.

### Usage

The main command to use aggressor-wrappers is aggressor-wrappers :)
For this to work, you will only need to provide .fasta file (with any numbers of entries, at least 1) and path to output directory.

```bash
# Primary: multifasta → all predictors from config.cfg
aggressor-wrappers proteins.fasta -o output_dir/
```

Note that all parameters here are taken by default. Many of those can be finetuned for your specific task through editing **config.cfg** file without editing the code or adding additional flags (more explanation for config see below).

If you want to explore distinct command inside aggressor-wrappers, use:

```bash
aggressor-wrappers --help          # overview of all commands
```

which will show you all commands that can be used. For each of them, simply run

```bash
aggressor-parse --help             # parse one predictor
aggressor-merge --help             # merge standard CSVs
aggressor-run --help               # run PATH / APPNN / WALTZ / PASTA / ArchCandy
aggressor-widemerge --help        # merge + optional BHT reference check

# or alternatively
python -m aggressor_wrappers parse --help
python -m aggressor_wrappers merge --help
python -m aggressor_wrappers run --help
python -m aggressor_wrappers widemerge --help
```

to view its respective arguments

## Configuration (`config.cfg`)

All tunable weights and thresholds live in one `config.cfg` file. You can use different configs, but in the current version you will need to override path to existing one:

```bash
export AGGRESSOR_WRAPPERS_CONFIG=/path/to/config.cfg
aggressor-parse waltz ... --config /path/to/config.cfg
```



### Metascore weight presets

Three presets in `config.cfg` (7 predictors; AggreProt excluded for now):


| Preset                  | Use case                                              |
| ----------------------- | ----------------------------------------------------- |
| `functional_amyloids`   | WALTZ/PATH/APPNN/PASTA emphasis                       |
| `pathogenic_amyloids`   | Cross-Beta/APPNN emphasis                             |
| `predictor_specificity` | **default** — conservative, tool-specificity oriented |


Active preset is listed in `[metascore].preset` and also can be changed.

The weights for predictors can also be calibrated through least-squares fit against standart tables:

```bash
python scripts/calibrate_weights.py
  --merged-csv /path/to/merged.csv
  --metascore-csv /path/to/reference/metascore/table.csv
```

After calibration is complete, paste the printed snippet into a new preset table under `[metascore.presets.*]`.

### `[pipeline]`

Default predictors for the main multifasta command (`aggressor-wrappers FASTA -o DIR`):

```ini
[pipeline]
predictors = path,appnn,waltz,pasta,archcandy
```

Override per run with `--predictors path,appnn` (or any subset).

### `[runners.waltz]` / `[runners.pasta]`

Web runners (HTTP, no Selenium). Both batch up to `sequences_per_run` sequences per job (default 10).

```ini
[runners.waltz]
base_url = https://waltz.switchlab.org/
threshold = 92
output_format = text_long
timeout_seconds = 180
sequences_per_run = 10

[runners.pasta]
base_url = http://old.protein.bio.unipd.it/pasta2/
npair = 22              # Top pairing energies
amount = -2.8           # Energy threshold (PEU)
thresdrop = val1        # Custom thresholds
poll_interval_seconds = 10
timeout_seconds = 3600
sequences_per_run = 10
```

`[predictors.pasta] energy_threshold = -2.8` controls `pasta_bin` in parsed CSV (independent of server `amount`).

### `[runners.archcandy]`

ArchCandy accepts **one sequence per API job** (default `sequences_per_run = 1`). The runner
uses the CRBM REST API (no Selenium). Per-residue scores are derived from the region CSV
(`Start`, `Stop`, `Score`) by summing overlapping region scores — matching the web UI
*cumulative score* tab.

```ini
[runners.archcandy]
base_url = https://bioinfo.crbm.cnrs.fr/
threshold = 0.4
transmembrane = false
parallel_jobs = 2
sequences_per_run = 1
verify_ssl = false      # CRBM certificate may fail strict verify
```

`[predictors.archcandy] score_mode = cumulative` (default) or `highest` (max region score).

### `[predictors.*]`

Parser thresholds (binarisation cutoffs). CLI `--threshold` overrides for a single run.

### `[runners.path]` / `[runners.appnn]`

PATH uses a modified version of [upstream PATH](https://github.com/KubaWojciechowski/PATH.git)
bundled as `vendor/PATH/path1.1.py`. Run PATH only inside the `AGGRESSOR` conda env
(Modeller + PyRosetta). APPNN invokes `legacy/appnn_converter.R` via `Rscript`.

```ini
[runners.path]
script =
python = python3
parallel_jobs = 2
sequences_per_run = 1
```

`parallel_jobs` — concurrent PATH subprocesses in batch mode.  
`sequences_per_run` — sequences per invocation (`0` = all at once, used by APPNN).

Overrides: `AGGRESSOR_PATH_SCRIPT`, `AGGRESSOR_PATH_PYTHON`.

APPNN defaults to `legacy/appnn_converter.R` (`Rscript` on `PATH`).

#### Vendored PATH (vs [upstream](https://github.com/KubaWojciechowski/PATH))

Wojciechowski & Kotulska, *Sci Rep* **10**, 7721 (2020). Local tweaks in `path1.1.py`:
auto-detect Modeller binary, `--modeller` flag, sklearn pickle compat, skip finished
Modeller runs, clearer empty-results error.

---



### `[cache]`


| Key       | Default | Meaning                                               |
| --------- | ------- | ----------------------------------------------------- |
| `root`    | `cache` | Base directory for raw copies                         |
| `enabled` | `false` | Set `true` or pass CLI `--keep-cache` to retain cache |


Layout: `cache/{protein_id}/{predictor}/raw.{ext}` — removed after each run unless `--keep-cache`.

---



## Pipeline

```
FASTA / raw predictor output
        ↓  aggressor-run / aggressor-wrappers (PATH, APPNN, WALTZ, PASTA, ArchCandy) or aggressor-parse
standard CSV per predictor   (+ optional raw cache)
        ↓  aggressor-merge  (or aggressor-widemerge with --reference)
wide CSV (position, aa_name, all predictors)
        ↓  metascore (phase 5)
metascores/*_metascore.csv
```

---



## Canonical output format

All tools write the same per-residue schema:


| Column         | Description                              |
| -------------- | ---------------------------------------- |
| `position`     | 1-based residue index                    |
| `aa_name`      | One-letter amino acid                    |
| `{Tool}_score` | Continuous score (see predictor section) |
| `{Tool}_bin`   | 0/1 amyloid call                         |


Merged tables add columns from each predictor while keeping `position` and `aa_name`.

---



## CLI



### `aggressor-parse`

```bash
aggressor-parse waltz --input WaltzJob.txt --fasta APP.fasta -o APP_waltz.csv
aggressor-parse pasta --input APP_pasta.dat --fasta APP.fasta -o APP_pasta.csv
aggressor-parse path --results results.csv --fasta RPS2.fasta -o RPS2_PATH.csv
```


| Flag                     | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| `--fasta` / `--sequence` | Sequence source (required)                |
| `--input` / `--results`  | Raw file (`--results` = PATH alias)       |
| `--config`               | CFG config path (`config.cfg`)            |
| `--threshold`            | Override binarisation threshold           |
| `--keep-cache`           | Keep `cache/` after run (default: remove) |
| `--cache-dir`            | Override `[cache].root`                   |




### `aggressor-run`

```bash
aggressor-run appnn --fasta RPS2.fasta -o RPS2_APPNN.csv
aggressor-run path --fasta RPS2.fasta -o RPS2_PATH.csv --work-dir ./path_work
aggressor-run waltz --fasta APP.fasta -o APP_waltz.csv
aggressor-run pasta --fasta APP.fasta -o APP_pasta.csv
aggressor-run archcandy --fasta APP.fasta -o APP_ArchCandy.csv

# Parse existing raw output without running the tool:
aggressor-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
aggressor-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv
aggressor-run waltz --skip-run --input WaltzJob.txt --fasta APP.fasta -o out.csv
aggressor-run pasta --skip-run --input APP_pasta.dat --fasta APP.fasta -o out.csv
aggressor-run archcandy --skip-run --input APP_archcandy.csv --fasta APP.fasta -o out.csv
```

PATH threading is slow — use `--skip-run` in CI or when `results.csv` already exists.
WALTZ/PASTA/ArchCandy require network access to public web servers.

### `aggressor-wrappers` (multifasta pipeline)

Primary entry point for AGGRESSOR: pass a multifasta and output directory. Predictors,
batching, and thresholds come from `config.cfg` unless overridden on the CLI.

```bash
conda activate AGGRESSOR
aggressor-wrappers vendor/PATH/test.fasta -o output_dir
# equivalent:
python -m aggressor_wrappers vendor/PATH/test.fasta -o output_dir
python aggressor-wrappers.py vendor/PATH/test.fasta -o output_dir
```

**Output layout** (`-o` / `--output`):

```
output_dir/
├── PATH/parsed/{protein_id}_PATH.csv
├── APPNN/parsed/{protein_id}_APPNN.csv
├── waltz/parsed/{protein_id}_waltz.csv
├── pasta/parsed/{protein_id}_pasta.csv
├── ArchCandy/parsed/{protein_id}_ArchCandy.csv
├── merged/{protein_id}_merged.csv
└── .tmp/                  # removed after successful run (fasta split scratch)
```

Per-predictor `work/` directories (Modeller/PyRosetta/R/web job temp files) are **removed
automatically** after a successful run. Use `--save-raw-files DIR` to archive raw
`{protein_id}/{predictor}.*` copies (needed for `--skip-run` after work cleanup).

Per-predictor batching is configured in `config.cfg` (not a CLI flag):

```ini
[runners.path]
parallel_jobs = 2        # concurrent PATH subprocesses
sequences_per_run = 1    # one protein per subprocess

[runners.appnn]
# defaults: all sequences in one R call (fast)

[runners.waltz]
sequences_per_run = 10   # web job batch size

[runners.pasta]
sequences_per_run = 10   # web job batch size

[runners.archcandy]
parallel_jobs = 2        # concurrent API jobs (one sequence each)
sequences_per_run = 1
```


| Flag               | Purpose                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| `-o` / `--output`  | Output root (per-predictor `parsed/` + `merged/`)                         |
| `--predictors`     | Comma-separated list (default from `[pipeline]` in config.cfg)            |
| `--save-raw-files` | Archive raw tool outputs per protein (enables `--skip-run` after cleanup) |
| `--keep-cache`     | Keep `cache/` after run (default: removed)                                |
| `--skip-run`       | Parse only; read raw from `{PREDICTOR}/work/` or `--save-raw-files`       |


**Smoke test** (3 short proteins; PATH is slow, web predictors add network time — ArchCandy ~4 s/protein):

```bash
aggressor-wrappers vendor/PATH/test.fasta -o output_dir
ls output_dir/*/parsed output_dir/merged
```



### `aggressor-merge`

```bash
aggressor-merge parsed/*.csv -o merged.csv --fasta RPS2.fasta
```


| Flag          | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `--predictor` | Force type per input file (if filename ambiguous) |
| `--fasta`     | Validate identical sequence across inputs         |




### `aggressor-widemerge`

Same merge as `aggressor-merge`, plus optional regression check against a BHT
`*_all.csv` reference table (score columns only).

```bash
aggressor-widemerge --input-dir parsed/ --fasta RPS2.fasta -o merged/RPS2_merged.csv
aggressor-widemerge parsed/*.csv --fasta RPS2.fasta -o merged.csv \
    --reference ../all/RPS2_human_all.csv
aggressor-widemerge --input-dir parsed/ -o merged.csv --check-bht-reference \
    --reference-name RPS2_human_all.csv
```


| Flag                    | Purpose                                                 |
| ----------------------- | ------------------------------------------------------- |
| `--input-dir`           | Directory of standard CSVs (or pass paths as arguments) |
| `--reference`           | BHT wide table for score-column comparison              |
| `--check-bht-reference` | Default reference under `BHT_amyloid/all/`              |
| `--rtol` / `--atol`     | Tolerance for reference comparison                      |


Also available as `python -m aggressor_wrappers widemerge …`.

---



## Predictors: raw output → canonical mapping

Each parser implements `parse(source, protein_id, sequence) → PredictorResult`.
Thresholds default from `config.cfg`.

### `path` — PATH threading


|               |                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Raw input** | PATH `results.csv` (`seq`, `dope`, …)                                                                                |
| **Algorithm** | Best (min) DOPE per hexapeptide → normalise to [0,1] with inversion → sliding window (6 aa) mean → per-residue score |
| `PATH_score`  | Normalised per-residue score (0 = no hexapeptide coverage)                                                           |
| `PATH_bin`    | 1 if score ≥ global percentile (default 75th of all hexapeptide scores)                                              |
| **CLI**       | `aggressor-run path …` or `aggressor-parse path --results results.csv …`                                             |




### `appnn` — APPNN (R package output)


|                      |                                                                                 |
| -------------------- | ------------------------------------------------------------------------------- |
| **Raw input**        | CSV from `legacy/appnn_converter.R` (`APPNN_parsed/{id}_APPNN.csv`)             |
| **Expected columns** | `aminoacid_position`, `aminoacid_score`, optional `aminoacid`, `hotspot_region` |
| `APPNN_score`        | Per-residue APPNN score                                                         |
| `APPNN_bin`          | 1 if `hotspot_region==1` or score ≥ 0.5                                         |
| **CLI**              | `aggressor-parse appnn --input APP_APPNN.csv --fasta protein.fasta -o out.csv`  |




### `waltz` — WALTZ web service


|               |                                                                                  |
| ------------- | -------------------------------------------------------------------------------- |
| **Raw input** | WALTZ detailed text output (`WaltzJob_*.txt` from web ZIP)                       |
| **Algorithm** | Region average scores expanded to per-residue positions (0 outside regions)        |
| `waltz_score` | Average score per residue in predicted region (0 if absent)                        |
| `waltz_bin`   | 1 if score ≠ 0                                                                   |
| **CLI**       | `aggressor-run waltz --fasta protein.fasta -o out.csv` or `aggressor-parse …`    |
| **Runner**    | HTTP submit to [waltz.switchlab.org](https://waltz.switchlab.org/) (`text_long`)   |




### `pasta` — PASTA 2.0 energy profile


|               |                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------- |
| **Raw input** | `*.fasta.seq.aggr_profile.dat` — one numeric energy per line (no header)                    |
| `pasta_score` | Raw PASTA aggregation energy (negative = more amyloid-prone)                                |
| `pasta_bin`   | 1 if energy < −2.8 (default from `[predictors.pasta]`)                                      |
| **CLI**       | `aggressor-run pasta --fasta protein.fasta -o out.csv` or `aggressor-parse …`               |
| **Runner**    | HTTP multipart upload to [PASTA 2.0](http://old.protein.bio.unipd.it/pasta2/), `batch.tar.gz` |




### `aggreprot` — AggreProt export


|                   |                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------- |
| **Raw input**     | CSV with header row + columns `position`, `aggregation`, …                         |
| `aggreprot_score` | `aggregation` column                                                               |
| `aggreprot_bin`   | 1 if aggregation ≥ 0.25                                                            |
| **CLI**           | `aggressor-parse aggreprot --input aggreprot.csv --fasta protein.fasta -o out.csv` |




### `archcandy` — ArchCandy web service


|                   |                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **Raw input**     | Region CSV from API: `ID`, `Sequence`, `Arch`, `Start`, `Stop`, `Score`                              |
| **Algorithm**     | Default `cumulative`: per-residue score = sum of `Score` for all regions covering that position (matches web *cumulative score* tab). Alt: `highest` = max region score |
| `ArchCandy_score` | Cumulative (or max) region score; 0 outside any region                                               |
| `ArchCandy_bin`   | 1 if residue in any predicted region                                                                 |
| **CLI**           | `aggressor-run archcandy --fasta protein.fasta -o out.csv` or `aggressor-parse …`                    |
| **Runner**        | HTTP REST API at [bioinfo.crbm.cnrs.fr](https://bioinfo.crbm.cnrs.fr/archCandy-input); one sequence per job |




### `crossbeta` — Cross-Beta predictor (CRBM JSON)


|                              |                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Raw input**                | JSON from CRBM datastore: `{id: [{AA_list: [{index, amino_acid, mean_confidence}]}]}` |
| `cross-beta-predictor_score` | `mean_confidence` per residue (`index` is 0-based in JSON → +1 for position)          |
| `cross-beta-predictor_bin`   | 1 if mean_confidence ≥ 0.5                                                            |




### `aggrescan` — not implemented (phase 4)

Registered in schema for column names in merge only. Parser TBD.

---



## Python API

```python
from aggressor_wrappers.core.config import load_config
from aggressor_wrappers.core.cache import store_raw_cache
from aggressor_wrappers.core.merge import merge_predictor_tables, write_merge_csv
from aggressor_wrappers.core.metascore import compute_weighted_metascore
from aggressor_wrappers.predictors.registry import get_parser
from aggressor_wrappers.runners.registry import get_runner

cfg = load_config()
waltz = get_parser("waltz").parse("WaltzJob.txt", protein_id="APP", sequence=seq)
pasta = get_runner("pasta").run(fasta="RPS2.fasta", skip_run=True, raw_profile="…")

wide = merge_predictor_tables([waltz, pasta])
meta = compute_weighted_metascore(wide, config=cfg)
write_merge_csv([waltz, pasta], "merged.csv")
```

---



## Package layout

The **repository directory** is `aggressor_wrappers/`; Python sources live in `src/`
and are imported as `aggressor_wrappers` (see `package-dir` in `pyproject.toml`).

```
aggressor_wrappers/              ← repo / project root (clone this folder)
├── config.cfg                weights, thresholds, cache, runners
├── environment.yml           conda env `AGGRESSOR` (Python + Modeller)
├── requirements.txt          pip runtime dependencies
├── requirements-dev.txt      pytest, ruff, build
├── aggressor-wrappers.py     multifasta pipeline (same as `aggressor-wrappers` CLI)
├── vendor/PATH/              vendored PATH (path1.1.py, templates, models)
├── legacy/                   frozen BHT reference scripts
├── setup_conda_env.sh          create/update conda env AGGRESSOR
├── scripts/
│   └── calibrate_weights.py
├── src/                      importable package `aggressor_wrappers`
│   ├── batch/                multifasta pipeline
│   ├── core/                 schema, config, cache, fasta, merge, validate, metascore
│   ├── predictors/           parser modules (all predictors)
│   ├── runners/              PATH / APPNN / WALTZ / PASTA / ArchCandy execution
│   └── cli/                  parse, merge, run, batch, widemerge, app
└── tests/                    unit + golden tests (fixtures only)
```

---



## Tests

```bash
python -m pytest
```

!!Do not rely on a system-wide `pytest` from `apt` — it uses a different Python than your conda/venv!!

- Unit parsers: `tests/test_parsers.py`
- Web runners: `tests/test_waltz_runner.py`, `tests/test_pasta_runner.py`, `tests/test_archcandy_runner.py`
- Cache: `tests/test_cache.py`
- Batch: `tests/test_batch.py`
- Golden merge roundtrip vs `BHT_amyloid/all/RPS2_human_all.csv`
- Golden Cross-Beta vs `RPL27 and RPL36/Cross-beta predictor/RPL27.json`

---



## Manual workflow (parse-only tools)

For AggreProt and Cross-Beta (no live runner yet), run the main pipeline for
PATH/APPNN/WALTZ/PASTA/ArchCandy, then per-protein `aggressor-parse` and merge:

```bash
aggressor-parse crossbeta --input raw/${ID}.json --fasta fasta_split/${ID}.fasta \
  -o output_dir/crossbeta/parsed/${ID}_crossbeta.csv
aggressor-merge output_dir/*/parsed/${ID}_*.csv --fasta fasta_split/${ID}.fasta \
  -o output_dir/merged/${ID}_merged.csv
```

Or merge with BHT reference check: `aggressor-widemerge --reference …`.

---



## Development roadmap (v0.3.1)


| Phase | Status      | Notes                                                                               |
| ----- | ----------- | ----------------------------------------------------------------------------------- |
| **0** | done        | 7 parsers, merge, cache, config, golden tests                                       |
| **1** | done        | PATH/APPNN runners, multifasta batch, `aggressor-widemerge`                          |
| **2** | done        | WALTZ web runner (HTTP)                                                               |
| **3** | in progress | PASTA + ArchCandy web runners done; AggreProt, Cross-Beta next (`legacy/api/`)      |
| **4** | planned     | Aggrescan parser + runner                                                           |
| **5** | planned     | `aggressor-metascore` CLI, CI on GitHub Actions                                     |


**Validated:** `aggressor-wrappers vendor/PATH/test.fasta -o output_dir` with PATH +
APPNN + WALTZ + PASTA + ArchCandy in env `AGGRESSOR` (Modeller + PyRosetta + R `appnn` + network).
ArchCandy: one REST job per protein (~4 s each via CRBM API).

---

