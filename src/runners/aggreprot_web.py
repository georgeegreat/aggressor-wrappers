"""AggreProt web runner — loschmidt.chemi.muni.cz, driven through the page flow.

AggreProt does not expose a usable submission API. It has a REST surface, but
job creation is split in two: ``POST /aggreprot/api/jobs`` only **allocates an
id** — it accepts any body, including none, and returns a six-character id — and
the data is supplied by a separate ``PUT /aggreprot/api/jobs/{id}`` whose payload
the front end constructs. Probing the POST therefore looks like success while
creating empty jobs, which is exactly the trap a wrapper written against an
assumed API falls into.

So submission is driven through the page, and only retrieval is done here. That
split is deliberate rather than a compromise: the submit step is three clicks
that must go through the UI, while everything afterwards — polling, parsing,
projecting onto residues — is deterministic and belongs in tested Python.

Page flow (mapped from the live site)::

    1. /aggreprot/            paste FASTA into the sequence textarea, "Next"
    2. structure selection    one radio group PER SEQUENCE, named
                              inputStructureSource<ACCESSION>, with values
                                WWPDB  -> also fill inputStructureWwpdb<ACCESSION>
                                AFDB   -> AlphaFold DB id
                                FILE   -> upload a structure
                                ""     -> "No structure"        (the default here)
                              then "Next"
    3. Job Summary            "Run job"   -> POST then PUT, yielding the job id
    4. waiting page           auto-advances to results

Retrieval needs none of that:

    GET /aggreprot/api/jobs/{id}      (anonymous, no token, no cookie)
    -> {"id", "status": "DONE"|..., "title", "created",
        "proteins": [{"name", "struct",
                      "series": {"size", "positions", "aminoAcids",
                                 "aggreprot", "sasa", "transmembrane"}}]}

"Download results (CSV)" is a plain link, not a Blob::

    GET /aggreprot/api/jobs/{id}.csv?download=true

so it is fetchable and nothing has to land in a downloads folder. That CSV is
the PRIMARY retrieval path here, not the JSON, for a reason that matters
downstream: its layout is

    Protein 1,<accession>,,,,
    position,struct_position,amino_acid,aggregation,sasa,transmembrane
    1,,D,0.0731...,,0.0

which is exactly what amyloscope's ``aggreprot`` adapter already reads. Going
through the JSON would mean re-serialising the same numbers into that shape,
adding a translation step that can drift. The JSON carries identical values and
is kept as a fallback and for job metadata.

A multi-protein job repeats the ``Protein N,<accession>`` banner before each
block, so the file is split per protein before parsing — a single ``header=1``
read, which is what the original adapter did, silently truncates such a file to
its first protein.

Mechanistically, AggreProt is an ensemble of deep convolutional networks trained
on labelled amyloidogenic and non-amyloidogenic **hexapeptides**, and its own
documentation states it was "designed to detect short, amyloid-related and
biologically relevant APRs, no longer than 50 residues", with performance not
guaranteed on longer ones. That is the opposite bias to Cross-Beta's 15-residue
windows, and it is why AggreProt behaves as a narrow caller in a consensus panel
— which in turn is what makes its support for an extended region informative.

``sasa`` is null unless a structure was supplied, which is the practical reason
to set ``pdb_id`` in config: solvent accessibility is what distinguishes a
buried aggregation-prone segment from an exposed one, and without it that
channel is simply absent.

Validation case: Abeta(1-42) with no structure, at the configured 0.25 cutoff,
gives 12-21 and 29-42; profile maximum 0.8464 at L34.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from aggressor_wrappers import __version__
from aggressor_wrappers.core.fasta import read_first_sequence
from aggressor_wrappers.core.schema import (
    PredictorResult,
    binary_from_scores,
    get_predictor_spec,
)
from aggressor_wrappers.runners.base import BasePredictorRunner

_DEFAULT_BASE_URL = "https://loschmidt.chemi.muni.cz/aggreprot/"
_USER_AGENT = f"aggressor-wrappers/{__version__}"
_TERMINAL = frozenset({"DONE", "ERROR", "FAILED"})

#: Radio values of the per-sequence structure selector on step 2.
STRUCTURE_SOURCES = {
    "pdb": "WWPDB",
    "alphafold": "AFDB",
    "file": "FILE",
    "none": "",
}


def submission_recipe(
    records: dict[str, str], *, pdb_ids: dict[str, str] | None = None
) -> list[dict]:
    """The click-by-click steps for one submission, as data.

    Returned rather than executed because this package does not drive a browser.
    Emitting it as a structure keeps the flow versioned and reviewable next to
    the parser, and lets a caller replay it with whatever automation it has.
    """
    pdb_ids = pdb_ids or {}
    steps: list[dict] = [
        {"step": 1, "action": "fill", "selector": "textarea",
         "value": "".join(f">{k}\n{v}\n" for k, v in records.items())},
        {"step": 1, "action": "click", "text": "Next"},
    ]
    for accession in records:
        pdb = pdb_ids.get(accession)
        if pdb:
            steps.append({"step": 2, "action": "select_radio",
                          "name": f"inputStructureSource{accession}",
                          "value": STRUCTURE_SOURCES["pdb"]})
            steps.append({"step": 2, "action": "fill",
                          "name": f"inputStructureWwpdb{accession}", "value": pdb})
        else:
            steps.append({"step": 2, "action": "select_radio",
                          "name": f"inputStructureSource{accession}",
                          "value": STRUCTURE_SOURCES["none"],
                          "note": "the 'No structure' option; sasa will be null"})
    steps += [
        {"step": 2, "action": "click", "text": "Next"},
        {"step": 3, "action": "click", "text": "Run job",
         "note": "POST /api/jobs allocates the id, PUT /api/jobs/{id} submits"},
        {"step": 4, "action": "read_job_id",
         "note": "from the results URL, or GET /api/jobs and take the newest"},
    ]
    return steps


_BANNER = re.compile(r"^Protein\s+\d+\s*,\s*([^,]*)", re.IGNORECASE)


def split_report_csv(text: str) -> dict[str, str]:
    """Split a multi-protein AggreProt report into ``{accession: csv_text}``.

    Each protein's block is preceded by a ``Protein N,<accession>`` banner and
    its own column header. Returning per-protein CSVs — each a well-formed file
    with one header row — means every downstream consumer reads a normal table
    and no one has to know about the banner convention.
    """
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        banner = _BANNER.match(line)
        if banner:
            current = banner.group(1).strip() or f"protein_{len(blocks) + 1}"
            blocks[current] = []
            continue
        if current is None:
            # A file with no banner at all is a single-protein export saved by
            # hand; treat it as one block rather than refusing it.
            current = "protein_1"
            blocks.setdefault(current, [])
        blocks[current].append(line)
    if not blocks:
        raise ValueError("AggreProt report CSV contained no data rows")
    return {name: "\n".join(rows) + "\n" for name, rows in blocks.items()}


def parse_job(document: dict, *, protein_name: str | None = None) -> dict:
    """Pull one protein's series out of a job document."""
    proteins = document.get("proteins") or []
    if not proteins:
        raise ValueError(f"AggreProt job {document.get('id')!r} has no proteins")
    if protein_name is None:
        protein = proteins[0]
    else:
        matches = [p for p in proteins if str(p.get("name")) == protein_name]
        if not matches:
            names = [p.get("name") for p in proteins]
            raise ValueError(
                f"AggreProt job {document.get('id')!r} has no protein "
                f"{protein_name!r}; it has {names}"
            )
        protein = matches[0]
    series = protein.get("series") or {}
    for key in ("aminoAcids", "aggreprot"):
        if key not in series:
            raise ValueError(
                f"AggreProt series lacks {key!r}; keys are {sorted(series)}"
            )
    return {"name": protein.get("name"), "struct": protein.get("struct"), **series}


class AggreProtWebRunner(BasePredictorRunner):
    """Retrieve and parse an AggreProt job; submission goes through the page."""

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        aggregation_threshold: float = 0.25,
        pdb_id: str | None = None,
        prefer_csv: bool = True,
        poll_interval_seconds: float = 5.0,
        timeout_seconds: int = 3600,
        **_ignored,
    ) -> None:
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/") + "/"
        self.aggregation_threshold = float(aggregation_threshold)
        # A PDB id is per protein in general; a single value here covers the
        # common one-protein-per-config case and is read by submission_recipe.
        self.pdb_id = pdb_id or None
        # Retrieve through the report CSV by default: it is the layout the rest
        # of this project already consumes. Set False to read the job JSON,
        # which carries the same numbers plus job metadata.
        self.prefer_csv = bool(prefer_csv)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.timeout_seconds = int(timeout_seconds)
        self.last_raw_path: Path | None = None

    # ------------------------------------------------------------------ #
    def _get(self, path: str) -> str:
        request = urllib.request.Request(
            urllib.parse.urljoin(self.base_url, path.lstrip("/")),
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")

    def fetch_job(self, job_id: str) -> dict:
        """Poll ``GET api/jobs/{id}`` until the job leaves a running state."""
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            document = json.loads(self._get(f"api/jobs/{job_id}"))
            status = str(document.get("status", "")).upper()
            if status in _TERMINAL:
                if status != "DONE":
                    raise RuntimeError(f"AggreProt job {job_id} ended {status}")
                return document
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"AggreProt job {job_id} still {status} after {self.timeout_seconds}s"
                )
            time.sleep(self.poll_interval_seconds)

    def fetch_report_csv(self, job_id: str) -> str:
        """Fetch what the "Download results (CSV)" button links to."""
        return self._get(f"api/jobs/{job_id}.csv?download=true")

    def save_report(self, job_id: str, work_dir: str | Path) -> dict[str, Path]:
        """Write one ``{accession}_aggreprot.csv`` per protein in the job.

        The files are byte-compatible with the exports already in this project,
        so a job fetched here and a file downloaded by hand from the results
        page are interchangeable inputs.
        """
        cwd = Path(work_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for accession, block in split_report_csv(self.fetch_report_csv(job_id)).items():
            dest = cwd / f"{accession}_aggreprot.csv"
            dest.write_text(block, encoding="utf-8")
            written[accession] = dest
        self.last_raw_path = next(iter(written.values()), None)
        return written

    def recipe(self, fasta: str | Path) -> list[dict]:
        """Submission steps for the sequences in ``fasta``."""
        from aggressor_wrappers.core.fasta import read_fasta

        records = read_fasta(Path(fasta))
        pdb_ids = {next(iter(records)): self.pdb_id} if self.pdb_id else {}
        return submission_recipe(records, pdb_ids=pdb_ids)

    def run(
        self,
        *,
        fasta: str | Path,
        protein_id: str | None = None,
        work_dir: str | Path | None = None,
        job_id: str | None = None,
        raw_csv: str | Path | None = None,
        raw_json: str | Path | None = None,
        **kwargs,
    ) -> PredictorResult:
        fasta_path = Path(fasta)
        resolved_id, sequence = read_first_sequence(fasta_path)
        protein_id = protein_id or resolved_id

        if raw_csv is not None:
            from aggressor_wrappers.predictors.aggreprot import AggreProtParser

            self.last_raw_path = Path(raw_csv)
            return AggreProtParser(
                aggregation_threshold=self.aggregation_threshold
            ).parse(raw_csv, protein_id=protein_id, sequence=sequence)

        if raw_json is not None:
            document = json.loads(Path(raw_json).read_text())
            self.last_raw_path = Path(raw_json)
        elif job_id and self.prefer_csv:
            files = self.save_report(job_id, work_dir or fasta_path.parent)
            from aggressor_wrappers.predictors.aggreprot import AggreProtParser

            path = files.get(protein_id) or next(iter(files.values()))
            return AggreProtParser(
                aggregation_threshold=self.aggregation_threshold
            ).parse(path, protein_id=protein_id, sequence=sequence)
        elif job_id:
            document = self.fetch_job(job_id)
            if work_dir:
                dest = Path(work_dir) / f"{protein_id}_aggreprot.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(document, indent=1))
                self.last_raw_path = dest
        else:
            raise ValueError(
                "AggreProt needs job_id= (or raw_json=). Submission is a browser "
                "step: POST /api/jobs only allocates an id and the PUT that "
                "carries the data is built by the page. Use .recipe(fasta) for "
                "the click sequence, then pass the resulting job id back here."
            )

        series = parse_job(document, protein_name=protein_id if len(
            document.get("proteins") or []) > 1 else None)
        observed = "".join(series["aminoAcids"])
        if observed.upper() != sequence.upper():
            raise ValueError(
                f"AggreProt job returned a different sequence than was queried "
                f"for {protein_id} ({len(observed)} vs {len(sequence)} aa); "
                f"refusing to attribute the result."
            )

        scores = [float(v) for v in series["aggreprot"]]
        binary = binary_from_scores(scores, threshold=self.aggregation_threshold)
        aux: dict[str, list] = {}
        for key in ("sasa", "transmembrane"):
            values = series.get(key)
            if values and any(v is not None for v in values):
                aux[key] = [float(v) if v is not None else float("nan") for v in values]
        return PredictorResult(
            protein_id=protein_id,
            sequence=sequence,
            spec=get_predictor_spec("aggreprot"),
            scores=scores,
            binary=binary,
            metadata={
                "aggregation_threshold": self.aggregation_threshold,
                "job_id": document.get("id"),
                "job_created": document.get("created"),
                "structure": series.get("struct"),
                "has_sasa": "sasa" in aux,
                "source": str(self.last_raw_path) if self.last_raw_path else None,
            },
            aux=aux,
        )
