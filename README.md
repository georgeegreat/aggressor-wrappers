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
| **Cross-Beta** | Public web service — no local install ([bioinfo.crbm.cnrs.fr/crossBetaPred/](https://bioinfo.crbm.cnrs.fr/crossBetaPred/))                      | Cross-Beta                      |
| **AggreProt** | Public web service — no local install ([loschmidt.chemi.muni.cz/aggreprot/](https://loschmidt.chemi.muni.cz/aggreprot/))                          | AggreProt                       |


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
aggressor-run --help               # run PATH / APPNN / WALTZ / PASTA / ArchCandy / Cross-Beta / AggreProt
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

Three presets in `config.cfg` (metascore weights for 7 tools; AggreProt merged but not weighted):


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
predictors = path,appnn,waltz,pasta,archcandy,crossbeta,aggreprot
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

### `[runners.crossbeta]`

Cross-Beta accepts **one sequence per API job**. Submission uses `threshold=0.54` and
`window_size=auto` (CRBM web defaults). Per-residue binarisation for parsed CSV uses
`[predictors.crossbeta] confidence_threshold = 0.54` — matching legacy BHT tables
(not the older 0.5 documented on some CRBM pages).

```ini
[runners.crossbeta]
base_url = https://bioinfo.crbm.cnrs.fr/
threshold = 0.54
window_size = auto
parallel_jobs = 2
sequences_per_run = 1
verify_ssl = false
```

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
        ↓  aggressor-run / aggressor-wrappers (PATH, APPNN, WALTZ, PASTA, ArchCandy, Cross-Beta) or aggressor-parse
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
aggressor-run crossbeta --fasta RPL27.fasta -o RPL27_crossbeta.csv

# Parse existing raw output without running the tool:
aggressor-run path --skip-run --results results.csv --fasta RPS2.fasta -o out.csv
aggressor-run appnn --skip-run --input APPNN_parsed/RPS2_APPNN.csv --fasta RPS2.fasta -o out.csv
aggressor-run waltz --skip-run --input WaltzJob.txt --fasta APP.fasta -o out.csv
aggressor-run pasta --skip-run --input APP_pasta.dat --fasta APP.fasta -o out.csv
aggressor-run archcandy --skip-run --input APP_archcandy.csv --fasta APP.fasta -o out.csv
```

PATH threading is slow — use `--skip-run` in CI or when `results.csv` already exists.
WALTZ/PASTA/ArchCandy/Cross-Beta require network access to public web servers.

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
├── {output_dir_name}.log   # full terminal transcript (append on resume)
├── PATH/parsed/{protein_id}_PATH.csv
├── APPNN/parsed/{protein_id}_APPNN.csv
├── waltz/parsed/{protein_id}_waltz.csv
├── pasta/parsed/{protein_id}_pasta.csv
├── ArchCandy/parsed/{protein_id}_ArchCandy.csv
├── cross-beta-predictor/parsed/{protein_id}_cross-beta-predictor.csv
├── aggreprot/parsed/{protein_id}_aggreprot.csv
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

[runners.crossbeta]
parallel_jobs = 2
sequences_per_run = 1
```


| Flag               | Purpose                                                                   |
| ------------------ | ------------------------------------------------------------------------- |
| `-o` / `--output`  | Output root (per-predictor `parsed/` + `merged/`)                         |
| `--predictors`     | Comma-separated list (default from `[pipeline]` in config.cfg)            |
| `--save-raw-files` | Archive raw tool outputs per protein (enables `--skip-run` after cleanup) |
| `--keep-cache`     | Keep `cache/` after run (default: removed)                                |
| `--skip-run`       | Parse only; read raw from `{PREDICTOR}/work/batch_N/`, the drop zone `{PREDICTOR}/`, or `--save-raw-files` |
| `--merge-existing` | At the merge step, also include predictors that already have `parsed/` tables here |
| `--no-resume`      | Re-run all predictors even when valid `parsed/` CSVs already exist        |

On repeat runs into the same `-o` directory, the pipeline **resumes by default**:
for each predictor it checks `{PREDICTOR}/parsed/{id}_*.csv` against the input
FASTA (row count and `aa_name` sequence). Valid outputs are reused; missing or
stale files are re-computed. Use `--no-resume` to force a full re-run.
Resume is disabled with `--skip-run` (that mode always re-parses from raw
files). Progress lines are tee'd to `{output_dir_name}.log` in append mode.

### Reusing output obtained on another machine

Raw output a tool produced somewhere else — on a cluster, or on a machine whose
architecture the binary actually runs on — goes in the **drop zone**: the
predictor's own directory, `{output}/{PREDICTOR}/`, named as that runner's
`discover_outputs` expects (`{protein_id}.txt` for TANGO). `--skip-run` searches
`{PREDICTOR}/work/batch_N/` first, then the drop zone, then the
`--save-raw-files` archive, and **merges** what it finds rather than taking the
first non-empty hit: a panel that is part-local and part-imported works.

Files are matched on their **whole stem**, never a prefix, under a short list
of names per predictor: what this package writes (`{protein}_waltz.txt`,
`{protein}_foldamyloid.csv`), the server's own download name
(`WaltzJob_<job>_{protein}.dat`), and a bare rename (`{protein}.dat`,
`{protein}.csv`). Prefix matching would hand `RPL36` the file belonging to
`RPL36a`, which is the worst failure available here because it produces a
complete, plausible, wrong answer. Two files claiming the same protein are
refused by name rather than resolved by sort order.

`--skip-run` does not require the tool to be installed. TANGO ships a Linux
x86_64 ELF, so on an arm64 Mac the binary can never execute and its output can
only ever arrive this way; `backend = local` no longer refuses at construction
when the file is absent. The declared backend still chooses the parser, because
it is what says which output format to expect.

Adding one tool to a panel that has already run:

```bash
# TANGO output from elsewhere, dropped in output_dir/tango/{RPS2,RPS28,...}.txt
aggressor-wrappers rms_last.faa -o output_dir --predictors tango \
  --skip-run --merge-existing
```

`--merge-existing` widens the **merge step only**: every registered predictor
with parsed tables under `{PREDICTOR}/parsed/` in that output directory enters
the merged table, each one named in the log. Without it, `--predictors tango`
rebuilds `merged/` from TANGO alone and the other eight columns are lost. The
widening deduplicates by output tag, so `{id}_ArchCandy.csv` — shared by the
`archcandy` and `archcandy2` parsers — is never merged twice; when a tag is
ambiguous the choice is stated in the log.


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


### `aggressor-export-amyloscope`

Turns a batch run's merged tables into an amyloscope run directory: one
normalised track per predictor per protein, plus a `config.yaml` that is ready
for `amyloscope run`.

```bash
aggressor-export-amyloscope output_dir/merged -o amyloscope_run/ \
    --fasta rms_last.faa \
    --domains domains.tsv \
    --primary-domain-category rna_binding \
    --hypothesis @hypothesis.txt \
    --clean --name "RPs panel"

amyloscope validate amyloscope_run/config.yaml
amyloscope run      amyloscope_run/config.yaml
```

Every tool is declared with the same `flag`-on-`APR` detection rule: each
parser already binarised with its tool's own cut **and direction**, so the APR
column is scale-free and polarity-correct and PASTA's inverted cutoff is not
restated in a second config file where it could drift. The config is written
with `adapter: aggressor` throughout, which is the adapter amyloscope uses for
these normalised tracks — not the per-tool adapters a hand-written config needs.


| Flag                         | Purpose                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| `--fasta`                    | The multifasta the run used; otherwise sequences come from `aa_name`  |
| `--domains`                  | hmmscan `--domtblout`, an InterProScan TSV, or a `protein_id, name, start, stop[, category]` table. **Repeatable** |
| `--domain-format`            | Override format detection                                             |
| `--domain-overlaps`          | `auto` (default), `best` or `all` — see below                         |
| `--primary-domain-category`  | The category the enrichment test is directed at, e.g. `rna_binding`   |
| `--hypothesis`               | Free text, or `@FILE`, surfaced in amyloscope's statistics report      |
| `--clean`                    | Remove an existing `tracks/` first                                     |
| `--name`                     | Panel name recorded in the config                                     |

**Domains are per species.** Annotations are used in each orthologue's own
numbering and never projected from a reference, so run the scan on the same
multifasta the panel ran on. A hit past the end of a sequence is dropped, not
clipped: it means the annotation and the sequence are different proteins.

`--domains` can be given more than once, so a curated table and a scan covering
the proteins it does not reach combine without editing either file — the formats
differ and their headers would not survive `cat`. Mixed formats keep every
overlapping domain, because one policy cannot be right for both; pass
`--domain-overlaps` to force one.

**Overlap policy follows the source.** `auto` keeps every domain from a
hand-written table, because nesting in a curated set is the annotation — an
RNA-binding domain inside a structured core is two facts about those residues —
and keeps the best hit per overlapping group from a scan, because several Pfam
models over one region are redundancy. Forcing `best` on a curated table deletes
the inner domains, which for a hypothesis about RNA-binding domains deletes the
hypothesis.

**The claim is pre-declared.** `--primary-domain-category` is what makes the
enrichment test directional; without it amyloscope runs the test over every
category with none nominated, which is a search rather than a hypothesis. The
category must be one the annotation actually uses, and the export refuses it
before writing anything if it is not.

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
| **Raw input** | Either download: the detailed text (`>header` sections, `start-end<TAB>?<TAB>score`) **or** the per-residue `.dat` (`position<TAB>score`, one line per residue). The format is detected from the content, never from the extension |
| **Algorithm** | Region average scores expanded to per-residue positions (0 outside regions)        |
| `waltz_score` | Average score per residue in predicted region (0 if absent)                        |
| `waltz_bin`   | 1 if score ≠ 0                                                                   |
| **CLI**       | `aggressor-run waltz --fasta protein.fasta -o out.csv` or `aggressor-parse …`    |
| **Runner**    | HTTP submit to [waltz.switchlab.org](https://waltz.switchlab.org/) (`text_long`)   |

The per-residue `.dat` carries **no protein header**, so the only thing tying it
to a protein is its filename. Its length is therefore checked against the query
sequence and a file that does not cover every residue exactly once is refused:
a file one residue short, or one belonging to a different chain of similar
length, otherwise parses perfectly and attributes someone else's segments.




### `pasta` — PASTA 2.0 energy profile


|               |                                                                                             |
| ------------- | ------------------------------------------------------------------------------------------- |
| **Raw input** | `*.fasta.seq.aggr_profile.dat` — one numeric energy per line (no header)                    |
| `pasta_score` | Raw PASTA aggregation energy (negative = more amyloid-prone)                                |
| `pasta_bin`   | 1 if energy < −2.8 (default from `[predictors.pasta]`)                                      |
| **CLI**       | `aggressor-run pasta --fasta protein.fasta -o out.csv` or `aggressor-parse …`               |
| **Runner**    | HTTP multipart upload to [PASTA 2.0](http://old.protein.bio.unipd.it/pasta2/), `batch.tar.gz` |




### `aggreprot` — AggreProt web service


|                   |                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------- |
| **Raw input**     | CSV export from API (`position`, `aggregation`, …)                                 |
| `aggreprot_score` | `aggregation` column                                                               |
| `aggreprot_bin`   | 1 if aggregation ≥ threshold (default 0.25; live RPL27 BHT bins match at 0.24)   |
| **CLI**           | `aggressor-run aggreprot --fasta protein.fasta -o out.csv`                         |
| **Parse only**    | `aggressor-parse aggreprot --input aggreprot.csv --fasta protein.fasta -o out.csv` |
| **Runner**        | REST API at [loschmidt AggreProt](https://loschmidt.chemi.muni.cz/aggreprot/) — up to 3 sequences/job, unique headers, no structure |




### `archcandy` — ArchCandy web service


|                   |                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **Raw input**     | Region CSV from API: `ID`, `Sequence`, `Arch`, `Start`, `Stop`, `Score`                              |
| **Algorithm**     | `highest` (the configured default): a residue takes the best single arch covering it, so scores stay on ArchCandy's own [0, 1] scale. Alt: `cumulative` = sum over overlapping arches, which matches the web *cumulative score* tab but can exceed 1.0 |
| `ArchCandy_score` | Best (or cumulative) arch score; 0 outside any region                                                |
| `ArchCandy_bin`   | 1 where the BEST single arch covering the residue is at or above `threshold` — not merely "inside any arch", since an arch the tool itself classes as non-significant is not evidence |
| **CLI**           | `aggressor-run archcandy --fasta protein.fasta -o out.csv` or `aggressor-parse …`                    |
| **Runner**        | HTTP REST API at [bioinfo.crbm.cnrs.fr](https://bioinfo.crbm.cnrs.fr/archCandy-input); one sequence per job |

**Two thresholds, and they are not the same number.** ArchCandy 2.0 takes a
threshold in the submission payload and returns only arches at or above it, so
`threshold` used to serve as both the server-side filter and the calling cut.
At the 0.57 significance boundary that means the raw CSV cannot show anything
below 0.57 — and a protein with no significant arch comes back with a
header and no rows, which the parser writes as a profile of zeros. In the
merged table, in an amyloscope track and in every figure, that is
indistinguishable from a tool that scored the protein and found nothing. On a
seven-protein ribosomal panel this happened to six of them, and ArchCandy went
on counting in the consensus denominator for all six.

`submit_threshold` (default 0.40, the service's own recall-oriented default) is
what the service filters at; `threshold` (0.57) is what binarises. Under
`score_mode = highest` the calls are **identical** either way — a residue takes
the best single arch covering it, and an arch below the calling cut cannot flip
a residue on — while the score column gains the ambiguous band, so a silent
protein is visibly silent rather than merely empty. The runner also says so on
stderr when the service returns no candidates at all, and records both
thresholds and the row count in the job sidecar, which outlives the `work/`
directory the pipeline deletes.




### `crossbeta` — Cross-Beta web service


|                              |                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| **Raw input**                | JSON from CRBM API (`result` file): list or legacy dict with `AA_list` / `mean_confidence` |
| `cross-beta-predictor_score` | `mean_confidence` per residue (`index` is 0-based in JSON → +1 for position)          |
| `cross-beta-predictor_bin`   | 1 if mean_confidence ≥ 0.54 (legacy BHT default; override in config.cfg)            |
| **CLI**                      | `aggressor-run crossbeta --fasta protein.fasta -o out.csv` or `aggressor-parse …`   |
| **Runner**                   | HTTP REST API at [bioinfo.crbm.cnrs.fr/crossBetaPred/](https://bioinfo.crbm.cnrs.fr/crossBetaPred/); one sequence per job |




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
│   ├── runners/              PATH / APPNN / WALTZ / PASTA / ArchCandy / Cross-Beta execution
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
- Web runners: `tests/test_waltz_runner.py`, `tests/test_pasta_runner.py`, `tests/test_archcandy_runner.py`, `tests/test_crossbeta_runner.py`
- Cache: `tests/test_cache.py`
- Batch: `tests/test_batch.py`
- Golden merge roundtrip vs `BHT_amyloid/all/RPS2_human_all.csv`
- Golden Cross-Beta vs `RPL27 and RPL36/Cross-beta predictor/RPL27.json`

---



## Manual workflow (parse-only tools)

For AggreProt (or any parse-only step), run the main pipeline for
PATH/APPNN/WALTZ/PASTA/ArchCandy/Cross-Beta, then per-protein `aggressor-parse` and merge.

AggreProt can also run standalone or in the batch pipeline:

```bash
aggressor-run aggreprot --fasta RPL27.fasta -o RPL27_aggreprot.csv
aggressor-wrappers proteins.fasta -o results/ --predictors aggreprot
```

Example merge after manual raw files:
aggressor-parse crossbeta --input raw/${ID}.json --fasta fasta_split/${ID}.fasta \
  -o output_dir/cross-beta-predictor/parsed/${ID}_cross-beta-predictor.csv
aggressor-merge output_dir/*/parsed/${ID}_*.csv --fasta fasta_split/${ID}.fasta \
  -o output_dir/merged/${ID}_merged.csv
```

Or merge with BHT reference check: `aggressor-widemerge --reference …`.

---



## Development roadmap (v0.3.2)


| Phase | Status      | Notes                                                                               |
| ----- | ----------- | ----------------------------------------------------------------------------------- |
| **0** | done        | 8 parsers, merge, cache, config, golden tests                                       |
| **1** | done        | PATH/APPNN runners, multifasta batch, `aggressor-widemerge`                          |
| **2** | done        | WALTZ web runner (HTTP)                                                               |
| **3** | done        | PASTA + ArchCandy + Cross-Beta + AggreProt web runners                              |
| **4** | planned     | Aggrescan parser + runner                                                           |
| **5** | planned     | `aggressor-metascore` CLI, CI on GitHub Actions                                     |


**Validated:** `aggressor-wrappers vendor/PATH/test.fasta -o output_dir` with PATH +
APPNN + WALTZ + PASTA + ArchCandy + Cross-Beta in env `AGGRESSOR`.
Cross-Beta: `threshold=0.54`, `window_size=auto`; parser `confidence_threshold=0.54` (BHT legacy).

---

