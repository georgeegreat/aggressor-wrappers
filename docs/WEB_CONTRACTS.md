# Recovered web-service contracts

None of the amyloid predictors wrapped here publishes an API. Each contract
below was recovered by observing the service's own browser traffic and is
reproduced field-for-field, including parameters whose meaning the page does not
explain. Where a parameter is not understood it is sent verbatim rather than
reinterpreted, because guessing at an undocumented field is how a wrapper stops
matching the tool it claims to reproduce without anyone noticing.

Each entry records: the endpoint, the request, the reply shape, and a
**validation case** — a sequence whose answer is known from the experimental
literature, so that a silent change in the service surfaces as a failed test
rather than as a quietly wrong panel.

---

## FoldAmyloid — `bioinfo.protres.ru` — CAPTURED, WRAPPED, TESTED

**Endpoint** `POST http://bioinfo.protres.ru/fold-amyloid/worker.php`
(`application/x-www-form-urlencoded`)

| field | value | meaning |
|-------|-------|---------|
| `cmd` | `do` | fixed |
| `p1`  | FASTA text | header + sequence |
| `p2`  | `A 19.89\|C 23.52\|…\|X 20.73\|` | **the residue scale — this is the model** |
| `p3`  | `5` | averaging frame |
| `p4`  | `3` | sent unconditionally by the page; no form control changes it |
| `p5`  | `21.4` | threshold |
| `p6`  | `21.4` | threshold, repeated |

**Reply** JSON with keys `res, msg, seq, tbl, dat, nam, pro, mrg, mrg2, tmp, zag`.

- `tbl` — base64 HTML: **the per-residue table**, `Num Res Fold Value`, one row
  per residue, flagged rows wrapped in `<span class='F'>` and carrying `f` in
  the Fold column. *This is the field the wrapper needs.*
- `seq` — base64 HTML: the coloured sequence view only. Easy to mistake for the
  data because it appears first.
- `tmp` — job token; plots at `tmp/<token>.png` (short) and `tmp/<token>L.png`
  (long). The UI's "Short result" / "Long result" tabs only swap these two
  images — both tables are already in the single reply.

**Mechanism.** Score = expected number of contacts within 8 Å, averaged over a
5-residue frame; residues above 21.4 are flagged (Garbuzynskiy, Lobanov &
Galzitskaya, 2010, *Bioinformatics* 26:326). The premise is packing density, not
hydrophobicity: a cross-beta spine requires interdigitated side chains, so the
segments able to form one are the densely contacting ones. Because the scale is
a *request parameter*, changing `p2` changes what is being predicted — it is
pinned in `runners/foldamyloid.py` rather than taken from the server.

**Validation case.** Aβ(1–42) `DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA`
returns exactly two regions: **16–21 `KLVFFA`** — the segment shown to be
necessary and sufficient for fibril formation (Tjernberg et al., 1996, *JBC*
271:8545) — and **32–36 `IGLMV`**. Profile maximum at F19 (24.708). Locked in
`tests/test_foldamyloid_aggrescan.py` against a verbatim capture of the reply
(`tests/fixtures/foldamyloid_worker_abeta42.json`).

---

## ArchCandy 2.0 — `bioinfo.crbm.cnrs.fr` — CAPTURED, WRAPPED, TESTED

The BiSMM site was rebuilt. `index.php?route=tools&tool=7` is gone; the tool now
lives at `/archCandy` and exposes a real JSON API.

```
POST /api/tools/archCandy/jobs                       (application/json)
     {"sequence": "<FASTA>", "threshold": 0.4, "transmembrane": false}
  -> 201 {"jobId": "<uuid>", "status": "QUEUED", "publicToken": "<hex>"}

GET  /api/tools/archCandy/jobs/{jobId}?token={publicToken}
  -> {"status": "QUEUED|RUNNING|DONE", "expiresAt": ..., "exitCode": 0,
      "errorMessage": null}

GET  /api/tools/archCandy/jobs/{jobId}/files/csv?token={publicToken}
  -> ID,Sequence,Arch,Start,Stop,Score
GET  .../files/input?token=...       -> the submitted FASTA
GET  .../files/selectseq?token=...   -> start,end,type
```

**No account is needed.** The job API is anonymous: a `POST` with
`credentials: "omit"` and no cookies at all returns `201` with a job. The site's
Login / Sign-up (and its 2-step email code) exist for *retrieving your own job
history* — the "Jobs submitted" / "Retrieve Jobs" panels — not for submission.
No credentials are handled by this wrapper.

Only three file endpoints exist: `csv`, `input`, `selectseq`. The UI's
**Table** view is the `csv` endpoint, and it is the same content as the
standalone build's output in a different column vocabulary —
`Number, Digram, Score, Arc_type, Position` (local 1.0) versus
`ID, Sequence, Arch, Start, Stop, Score` (web 2.0). The **Download results**
button is `selectseq` (`start,end,type`), which is the *selected-region* export
and is empty unless regions were picked in SeqView — it is not the arch table.

`publicToken` is required on every follow-up call and results **expire**
(`expiresAt`, a few hours out), so raw output must be downloaded, not
bookmarked.

**Score calibration, quoted from the tool's own documentation:** a prediction is
*non-significant* below **0.40**, *ambiguous* between **0.40 and 0.57**, and
*significant* above **0.57**. The web default is 0.40. This is a three-band
scale, not a single cutoff, and it is the authority for any threshold choice in
this project.

Columns differ from the standalone 1.0 build (`Number,Digram,Score,Arc_type,
Position`) — separate parsers, `ArchCandy2Parser` and `ArchCandyParser`.

**Validation case.** Aβ(1–42) at 0.40 returns 14 β-arch candidates, the best
being `QKLVFFAEDVGSNKGAIIGLMV` at **15–36, score 0.723** — one arch spanning both
hydrophobic segments (KLVFFA, IGLMV) with the bend between them. Under
`cumulative` the same residues reach 6.681, which is not an ArchCandy score.

---

## AGGRESCAN — `andromeda.uab.cat` — CAPTURED, WRAPPED, TESTED

Moved from `bioinf.uab.es`. Synchronous CGI — no job, no polling.

```
POST http://andromeda.uab.cat/bioinf/cgi-bin/aap/aap_ov.pl
     (application/x-www-form-urlencoded)   sequence=<FASTA, CRLF line endings>
  -> 200, the result page itself
```

Confirmed against the live entry page: one `<form method="post">`, one
`<textarea name="sequence">`, one `submit!` button, no other fields, and the
`action` is the absolute CGI path above — not a path under `/aggrescan/`.

**The line ending is load-bearing.** The CGI splits FASTA records on **CRLF**,
because the HTML specification requires a `<textarea>` to normalise its value to
CRLF on submission, so every request the authors ever saw had it. A client that
sends bare LF hands the script one unsplittable line and gets:

```
UNIDENTIFIED ERROR
Aggrescan has identified that your sequences are in fasta format, but found an
error that can not be identified. You should check your original document from
were you copy-pasted your sequences, especially if its a MS Word.
```

The MS-Word advice is a red herring for a programmatic caller and cost a
372-protein sweep. Measured, same bytes otherwise:

| body | reply |
|---|---|
| `>t\nACDEFGHIKLMNPQRSTVWY\n` | 860 B, `UNIDENTIFIED ERROR` |
| `>t\r\nACDEFGHIKLMNPQRSTVWY\r\n` | **15 643 B, full result table** |

**Three distinct refusals**, all returned as HTTP 200 inside the normal result
template, so none can be told from a result by the page shell:

| trigger | page says | notes |
|---|---|---|
| bare LF, or a header with no sequence | `UNIDENTIFIED ERROR` | the line-ending bug |
| no `>` header | `Please use fasta format …` | "needs the `>` to identify the name, the beginning and the end" |
| character outside the 20 standard AAs | `E R R O R! som characters of your sequence are not allowed` | echoes the sequence with the offenders in `<FONT COLOR="red">`, so they can be named |

All three are raised as `PermanentToolError`, so the scheduler skips rather than
retries.

**What is NOT a problem**, each verified with CRLF: lower case (accepted);
length — 3, 5, 11, 20, 180, 600 and 2000 residues all return tables; 60-column
wrapping; a dotted header such as `>6087.XP_002162002.2`. Responses came back in
0.26–1.6 s, so no timeout is involved. Only `B, J, O, U, X, Z, *, -` are
refused, and `read_fasta` already rejects those for every predictor before a
request is made.

Validation of the fix against the live service: Aβ(1–42) submitted with CRLF
returns a3vSA **0.064**, nHS **2**, Na4vSS **6.4**, Hot Spots **17–22** and
**30–42** by both NHSA and the page's red highlighting.

**There is no per-residue download.** The server's only file is
`/bioinf/aap/<job>/aap_<job>.txt.xls`, holding the nine global descriptors
(a3vSA, nHS, NnHS, AAT, THSA, TA, AATr, THSAr, Na4vSS) and nothing positional.
The per-residue profile exists only in the result HTML. This is why the
historical per-residue CSVs in this project carry stray non-breaking spaces —
0xCA is a Mac Roman nbsp, carried out of the HTML by a copy-paste.

**The profile table is anchored on its header row.** It is one `<tr>` of six
`<td>` cells — one cell per column, values separated by `<br>` — immediately
after the header row whose cells read `#  AA  a4v  HSA  NHSA  a4vAHS`, and it
runs to the end of that row. Anchoring there is not cosmetic: a real 334-residue
page opens **42** `<small>` tags and closes **29**, and contains a literal
`<smal>` typo, so any `<small>`-delimited scan reads across cell boundaries in a
way that depends on where the imbalance falls. The header row is the one
landmark the page states explicitly. Hot Spot residues are additionally printed
in `<FONT COLOR="red">`, which the parser cross-checks against NHSA.

**Hot Spots come from `NHSA`, not from a re-derived rule and not from `HSA`.**
AGGRESCAN describes a Hot Spot as a region held above the Hot Spot Threshold for
a minimum run (Conchillo-Solé et al., 2007, *BMC Bioinformatics* 8:65), but
reimplementing that description does not reproduce the tool's own calls.
`NHSA > 0` does, exactly:

| capture | reported nHS | `NHSA > 0` | a4v > −0.02, run ≥ 5 |
|---|---|---|---|
| Aβ(1–42), 42 aa | 2 | 2 — 17–22, 30–42 | 2 (identical) |
| CTSV_Homo_sapiens, 334 aa | 11 | **11 (identical)** | **12** |

On CTSV the threshold rule invents a Hot Spot at 105–111 and extends three more
by 3–4 residues each (230–232, 245–248, 332–334): **101 hot residues against the
server's 84 — 30.2 % breadth reported where the tool itself calls 25.1 %.** The
error is one-directional (it never under-calls), so it inflates AGGRESCAN's
apparent breadth and manufactures precisely the kind of shoulder that consensus
clustering is then asked to adjudicate. `NHSA > 0` also matches the page's own
red highlighting residue for residue (84/84).

`HSA` is not the mask either: on Aβ42 the isolated **Y10** carries
`HSA = 0.122` and `a4vAHS = 0.102` while `NHSA = 0` — the shared area is
assigned *before* the run-length requirement is applied, so only NHSA reflects
the final call.

**The summary is two columns, and naive adjacency misreads it.** Every label
sits in one column and every value in the other, so "the first number after
*Number of Hot Spots (nHS)*" is **100**, picked off the next label (*Normalized
nHS for 100 residues*). Labels and values are collected separately and paired by
order; if the runs differ in length nothing is returned, because an unavailable
cross-check beats a check against a constant.

**Validation cases.** Aβ(1–42): nHS = 2 — **17–22 `LVFFAE`** and **30–42
`AIIGLMVGGVVIA`**; a3vSA 0.064, Na4vSS 6.4; profile maximum at F19 (1.289).
CTSV_Homo_sapiens: nHS = 11 — 1–18, 75–79, 138–144, 182–187, 205–210, 223–229,
238–244, 259–265, 277–286, 298–302, 326–331; a3vSA −0.080, Na4vSS −8.2.
Verbatim capture kept at `tests/fixtures/aggrescan_result_ctsv.html`.

---

## Cross-Beta-Pred 2.0 — `bioinfo.crbm.cnrs.fr` — CAPTURED, WRAPPED, TESTED

**Correcting an earlier entry in this file.** It previously said the web arm
could not be wrapped because the page form carries `g-recaptcha-response`. That
was wrong on both counts:

* the **API asks for no token** — `POST` with `credentials: "omit"`, no cookie,
  returns `201`, exactly like ArchCandy 2.0. The reCAPTCHA is a front-end widget
  on the page, not a gate on the service, so using the API circumvents nothing;
* the probe that "proved" no API existed used `crossBetaPred`. The route is
  **`crossbetaPred`**, lower-case *b*. Every attempt 404'd and the absence looked
  real. The bundle's own tool→path map is the authority.

```
POST /api/tools/crossbetaPred/jobs                 (application/json)
     {"sequence": "<FASTA>", "threshold": 0.5, "windowSize": "auto"}
  -> 201 {"jobId", "status": "QUEUED", "publicToken"}

GET  /api/tools/crossbetaPred/jobs/{jobId}?token={publicToken}
GET  /api/tools/crossbetaPred/jobs/{jobId}/files/result?token={publicToken}
GET  /api/tools/crossbetaPred/jobs/{jobId}/files/input?token={publicToken}
```

`files/result` — **`result`, not `csv`; there is no CSV member** — returns

```json
[{"prot_name": "sequence_query", "All_sequence_pred": 0.7548,
  "AA_list": [{"index": 0, "amino_acid": "D",
               "score_list": [...15 values...], "mean_confidence": 0.5763}],
  "mean_list": [{"D": 0.5763}], "AR_list": [[1, 42]]}]
```

Three details that silently corrupt a track if missed:

* **`index` is 0-based.** Every other predictor in this panel is 1-based; an
  off-by-one here shifts every Cross-Beta call by one residue and nothing
  downstream can detect it.
* the per-residue score is **`mean_confidence`**, not `score`; `score_list`
  holds the 15 window values it averages.
* **`prot_name` is always the literal `"sequence_query"`**, never the submitted
  accession, so a multi-sequence job cannot be demultiplexed — one sequence per
  job, protein id supplied by the caller.

**Validation case, and the model's limitation made visible.** Aβ(1–42):
`score_list` has 15 entries — the training window — and `AR_list` is
**`[[1, 42]]`**, the entire peptide as one aggregation region. The profile ramps
monotonically to 0.854 at residue 38 rather than peaking on KLVFFA. Cross-Beta
detects extended aggregation-prone stretches and cannot resolve a nucleating
hexapeptide; this is why it is the panel's broadest caller (42 % of the
ribosomal panel) and why its support for a region boundary carries less weight
than a narrow caller's.

The standalone checkout remains the `backend = local` arm and is unchanged.

---

## PASTA 2.0 — `old.protein.bio.unipd.it` — CAPTURED, FIXED, VERIFIED

Alive at the original URL; the form posts to `Pasta2.jsp`.

```
POST /pasta2/Pasta2.jsp          (multipart/form-data)
     sequence   = <FASTA text>          <- the query goes HERE
     npair      = 20 (site default; this panel uses 22)
     amount     = -5 (site default; this panel uses -2.8)
     thresdrop  = val1 Custom | val2 Peptides | val3 Region (90%/85% spec)
     [emailaddress / emailnotice deliberately omitted]
  -> 302 to /pasta2/work/pid_<id>/pasta.html
     result:    /pasta2/work/pid_<id>//batch.tar.gz
     per query: /pasta2/work/pid_<id>/batch/<name>.fasta_pasta.html
```

**The bug.** The runner uploaded the FASTA as `mutantfastafile` — on the live
form that is the file input of the *mutant* panel, guarded by the `mutantcheck`
checkbox and meant for comparing a variant against a wild type. With no
`sequence` field the query was empty, so the service ran normally and returned
nothing useful. Fixed to post `sequence`.

**Header rewrite — the multi-sequence trap.** PASTA strips every non-word
character from each FASTA header before naming its output files:

| submitted | archive member |
|---|---|
| `6087.XP_002162002.2` | `6087XP_0021620022` |
| `6183.Smp_016050.1` | `6183Smp_0160501` |
| `7029.ACYPI000203-PA` | `7029ACYPI000203PA` |

Matching members against the original accession therefore fails for *every*
identifier containing a dot, which is every STRING-style accession — a
372-protein panel reported "missing profile" for all ten sequences of its first
batch while the service had analysed them correctly. Results also come back in a
different order than submitted, so position is never a safe key either. The
rewrite is lossy, so a batch where two accessions collapse onto one name is
refused rather than silently mis-assigned.

**Two profiles, not one.** The archive carries both
`<name>.fasta.seq.aggr_profile.dat` and `…​.dat.free_energy`. PASTA scores a
pairing **free energy** (more negative = more stable cross-β), so `.free_energy`
is the track a `below` threshold applies to — and what amyloscope already reads.
The runner now extracts it by default; `profile_kind = aggregation` selects the
other.

**Validation case.** Aβ(1–42) with `npair=22, amount=-2.8`: 22 amyloid
predictions, best pairing energy **−8.860653**, 28.57 % disorder, 57.14 %
β-strand, 42.86 % coil.

---

## AggreProt — `loschmidt.chemi.muni.cz` — CAPTURED, WRAPPED, TESTED

**Submission is a page flow, not an API call, and the API *looks* like it works.**
`POST /aggreprot/api/jobs` only **allocates a six-character id** — it accepts any
body, including none, and returns an id every time — while the data is carried by
a separate `PUT /aggreprot/api/jobs/{id}` whose payload the front end builds.
Probing the POST therefore returns plausible ids while creating empty jobs.

Page flow:

1. `/aggreprot/` — paste FASTA into the sequence textarea → **Next**
2. structure selection, one radio group **per sequence**,
   `inputStructureSource<ACCESSION>`:
   `WWPDB` (+ `inputStructureWwpdb<ACCESSION>` for the id) · `AFDB` · `FILE` ·
   `""` = **No structure** → **Next**
3. **Job Summary** → **Run job** (this issues the POST then the PUT)
4. waiting page, auto-advances to results

Retrieval needs none of it:

```
GET /aggreprot/api/jobs/{id}          (anonymous — no token, no cookie)
 -> {"id","status":"DONE","title","created",
     "proteins":[{"name","struct",
                  "series":{"size","positions","aminoAcids",
                            "aggreprot","sasa","transmembrane"}}]}
```

`Download results (CSV)` is a plain `<a href>`, not a Blob:

```
GET /aggreprot/api/jobs/{id}.csv?download=true
```

so it is fetchable and nothing reaches a downloads folder. **That CSV is the
primary retrieval path**, because its layout

```
Protein 1,ABETA42,,,,
position,struct_position,amino_acid,aggregation,sasa,transmembrane
1,,D,0.07315478920936577,,0.0
```

is exactly what amyloscope's `aggreprot` adapter already reads — going through
the JSON would re-serialise the same numbers into that shape and add a
translation step that can drift. Verified: both routes give identical scores to
1e-12. The JSON is kept as a fallback and for job metadata.

A multi-protein job repeats the `Protein N,<accession>` banner and the column
header before each block, so the file is **several tables concatenated**, not one
table with a decorative first line. A single `header=1` read — what the original
adapter did — silently keeps only the first protein. `split_report_csv()` emits
one well-formed CSV per protein.

`sasa` is `null` for every residue unless a structure was chosen on step 2 —
hence the `pdb_id` config key. That channel is not decorative: solvent
accessibility is what separates a buried aggregation-prone segment from an
exposed one, which is the masking question the ribosomal panel is built around.

AggreProt is an ensemble of deep CNNs trained on labelled (non-)amyloidogenic
**hexapeptides**, and its own documentation says it was "designed to detect
short, amyloid-related and biologically relevant APRs, no longer than 50
residues", performance not guaranteed beyond. That is the opposite bias to
Cross-Beta's 15-residue windows and explains why it behaves as a narrow caller
(14.2 % breadth on the ribosomal panel).

**Validation case.** Aβ(1–42), no structure, threshold 0.25: **12–21
`VHHQKLVFFA`** and **29–42 `GAIIGLMVGGVVIA`**; profile maximum **0.8464 at L34**.
Compare FoldAmyloid (16–21, 32–36) and AGGRESCAN (17–22, 30–42) on the same
sequence — three mechanistically distinct models converging on the two
hydrophobic segments.

---

## Still to capture

These runners exist and now *construct* correctly (see below), but their wire
contracts have not been re-verified against the live services:

| predictor | host | status |
|-----------|------|--------|
| WALTZ | `waltz.switchlab.org` | the only runner that worked before this pass; contract not re-captured |
| ST-ARCH, TAPASS | `bioinfo.crbm.cnrs.fr` | **not wanted.** Both are built on other panel members — ST-ARCH is trained on β-arch data produced by ArchCandy, TAPASS combines ArchCandy/TANGO/PASTA. A consensus count treats agreement as independent evidence, so a tool derived from another member inflates the denominator instead of adding to it (the same objection as amyloid_predict, which re-expresses TANGO + Waltz) |

Capture procedure that worked for FoldAmyloid, for reuse:

1. Open the tool's page in a browser you can script.
2. Install an `XMLHttpRequest.send` / `fetch` recorder before submitting.
3. Submit the validation sequence through the page's own controls, so the
   framework fills its hidden fields for you.
4. Read the recorded request body — that is the contract. Do not reconstruct it
   from the form markup; these pages mirror visible inputs into differently
   named hidden fields (`MemoFasta__lines`, `EdLevel__text`, …) that never
   appear in the POST.
5. Save the raw reply as a fixture and write the validation case as a test.

**Egress note.** These hosts are blocked by organisation policy from the cloud
sandbox and from the local Linux workspace, so captures must run in a browser on
a machine with ordinary internet access.
