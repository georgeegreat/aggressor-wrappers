"""Per-species domain annotation, read from sequence rather than looked up by id.

The choice this module encodes is annotate-per-species, not project-from-reference,
and the reason is stronger than specificity alone: a reference domain carried onto
an orthologue by alignment is an inference about that orthologue, and it inherits
every gap and every misalignment on the way. Insertions in a divergent lineage
shift coordinates even in a well-conserved protein, and the shift is invisible once
the interval has been written down as if it were an annotation.

**Why from sequence rather than from a database.** An EggNOG orthologue set is keyed
``taxid.proteinid``, and those ids come from Ensembl, RefSeq, FlyBase and others --
not UniProt. InterPro's and UniProt's APIs take accessions, so a lookup route needs
id mapping first, and across an opisthokont set a large fraction will not map, map
to several accessions, or map to an isoform of a different length. Scanning the
sequence has none of those failure modes: the coordinates come back in the
numbering of the sequence that was scanned, which is the numbering the predictor
tracks already use.

Two readers are provided because two tools are worth supporting. ``hmmscan``
against Pfam-A is the light option (Pfam-A.hmm is ~1.5 GB against InterProScan's
tens of gigabytes) and covers what matters for most families. InterProScan's TSV is
read as well, for anyone who already runs it. Both land in the same shape, so the
rest of the pipeline cannot tell which produced a given interval -- except through
the ``source`` field, which is recorded precisely so it can.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Pfam/InterPro accession or name -> amyloscope's family-agnostic category.
#:
#: Matched as a lowercase SUBSTRING of the domain name, longest pattern first, so
#: "KH_1" and "KH_dom" both reach rna_binding without enumerating every variant.
#: Deliberately small: the categories exist to colour a figure consistently across
#: protein families, not to re-describe Pfam's hierarchy.
CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    # Nucleic-acid binding
    ("kh", "rna_binding"),
    ("rrm", "rna_binding"),
    ("dsrm", "rna_binding"),
    ("dsrbd", "rna_binding"),
    ("zf-ccch", "rna_binding"),
    ("pumilio", "rna_binding"),
    ("sam_", "rna_binding"),
    ("agenet", "rna_binding"),
    ("tudor", "rna_binding"),
    ("piwi", "rna_binding"),
    ("hmg", "dna_binding"),
    ("myb", "dna_binding"),
    ("homeobox", "dna_binding"),
    ("zf-c2h2", "dna_binding"),
    ("bzip", "dna_binding"),
    # Low-complexity and disorder-associated
    ("rgg", "low_complexity"),
    ("gar", "low_complexity"),
    ("prion", "low_complexity"),
    ("q-rich", "low_complexity"),
    ("n-rich", "low_complexity"),
    ("coiled", "linker"),
    # Catalysis and folded cores
    ("kinase", "catalytic"),
    ("peptidase", "peptidase"),
    ("hydrolase", "catalytic"),
    ("atpase", "catalytic"),
    ("helicase", "catalytic"),
    ("ww", "structured_core"),
    ("sh3", "structured_core"),
    ("pdz", "structured_core"),
    ("ankyrin", "structured_core"),
    ("wd40", "structured_core"),
    # Trafficking and control
    ("nls", "localization"),
    ("nes", "localization"),
    ("signal", "localization"),
    ("transmembrane", "localization"),
    ("ubiquitin", "regulatory"),
    ("brct", "regulatory"),
)


def categorise(name: str, *, default: str = "default") -> str:
    """Map a domain name onto amyloscope's category vocabulary.

    Unknown names get ``default`` rather than a guess. A wrong category silently
    recolours a figure and invites a reading the data does not support, which is
    worse than a grey box that says nothing.
    """
    lowered = name.strip().lower()
    for pattern, category in sorted(
        CATEGORY_PATTERNS, key=lambda item: -len(item[0])
    ):
        if pattern in lowered:
            return category
    return default


@dataclass(frozen=True)
class DomainHit:
    """One annotated interval on one protein, in that protein's own numbering."""

    protein_id: str
    name: str
    start: int            # 1-based, inclusive
    stop: int             # 1-based, inclusive
    category: str = "default"
    accession: str | None = None
    score: float | None = None
    evalue: float | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.start < 1 or self.stop < self.start:
            raise ValueError(
                f"{self.protein_id}/{self.name}: invalid interval "
                f"{self.start}-{self.stop} (1-based, inclusive, stop >= start)"
            )


def read_hmmscan_domtblout(
    path: str | Path, *, max_evalue: float = 1e-5
) -> dict[str, list[DomainHit]]:
    """Parse ``hmmscan --domtblout``.

    The ENVELOPE coordinates (columns 20-21) are used rather than the alignment
    coordinates (18-19). HMMER's alignment bounds are where the model matched
    confidently; the envelope is where the domain is inferred to lie, which is the
    extent a reader of a domain figure means. Using the alignment instead
    systematically under-draws every domain by a few residues at each end.

    Whitespace-delimited with a free-text description last, so the row is split on
    the first 22 fields only -- a description containing spaces would otherwise
    shift every column after it.
    """
    hits: dict[str, list[DomainHit]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=22)
            if len(fields) < 22:
                continue
            name, accession, protein_id = fields[0], fields[1], fields[3]
            try:
                evalue = float(fields[12])      # i-Evalue, this domain
                score = float(fields[13])
                start, stop = int(fields[19]), int(fields[20])
            except ValueError:
                continue
            if evalue > max_evalue:
                continue
            hits.setdefault(protein_id, []).append(
                DomainHit(
                    protein_id=protein_id,
                    name=name,
                    start=start,
                    stop=stop,
                    category=categorise(name),
                    accession=accession if accession != "-" else None,
                    score=score,
                    evalue=evalue,
                    source="hmmscan/Pfam",
                )
            )
    return hits


def read_interproscan_tsv(
    path: str | Path, *, max_evalue: float = 1e-5
) -> dict[str, list[DomainHit]]:
    """Parse InterProScan's TSV output.

    Columns: protein, md5, length, analysis, signature accession, signature
    description, start, stop, score, status, date, then optionally InterPro
    accession and description. The InterPro description is preferred for the name
    when present, because a member-database accession such as ``PF00013`` means
    nothing on a figure axis.
    """
    hits: dict[str, list[DomainHit]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 9 or row[0].startswith("#"):
                continue
            protein_id = row[0]
            signature_accession, signature_description = row[4], row[5]
            interpro_description = row[12] if len(row) > 12 else ""
            name = (
                interpro_description
                or signature_description
                or signature_accession
            )
            try:
                start, stop = int(row[6]), int(row[7])
            except ValueError:
                continue
            try:
                evalue = float(row[8])
            except ValueError:
                evalue = None            # some analyses report "-"
            if evalue is not None and evalue > max_evalue:
                continue
            hits.setdefault(protein_id, []).append(
                DomainHit(
                    protein_id=protein_id,
                    name=name,
                    start=start,
                    stop=stop,
                    category=categorise(name),
                    accession=row[11] if len(row) > 11 and row[11] else signature_accession,
                    evalue=evalue,
                    source=f"InterProScan/{row[3]}",
                )
            )
    return hits


def read_domain_tsv(path: str | Path) -> dict[str, list[DomainHit]]:
    """Parse a hand-written table: protein_id, name, start, stop[, category].

    The escape hatch. Any annotation route that is not one of the two above -- a
    literature table, a curated set, output from a tool with no reader here -- can
    be supplied this way without pretending to be a scan.
    """
    hits: dict[str, list[DomainHit]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        required = {"protein_id", "name", "start", "stop"}
        missing = required - {c.strip().lower() for c in (reader.fieldnames or [])}
        if missing:
            raise ValueError(
                f"{path}: domain table missing column(s) {sorted(missing)}; "
                f"required: protein_id, name, start, stop (category optional)"
            )
        for row in reader:
            lowered = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            protein_id = lowered["protein_id"]
            if not protein_id:
                continue
            name = lowered["name"]
            hits.setdefault(protein_id, []).append(
                DomainHit(
                    protein_id=protein_id,
                    name=name,
                    start=int(lowered["start"]),
                    stop=int(lowered["stop"]),
                    category=lowered.get("category") or categorise(name),
                    source="supplied table",
                )
            )
    return hits


READERS = {
    "hmmscan": read_hmmscan_domtblout,
    "interproscan": read_interproscan_tsv,
    "table": read_domain_tsv,
}


def detect_format(path: str | Path) -> str:
    """Identify which reader a file needs, from its own content."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    head = [line for line in text.splitlines() if line.strip()][:40]
    if any(line.startswith("# target name") for line in head):
        return "hmmscan"
    if any("hmmscan" in line or "hmmsearch" in line for line in head if line.startswith("#")):
        return "hmmscan"
    first_data = next((line for line in head if not line.startswith("#")), "")
    columns = first_data.split("\t")
    if len(columns) >= 9 and re.fullmatch(r"[0-9a-f]{32}", columns[1].strip() or "x"):
        return "interproscan"       # the md5 column is unmistakable
    if "protein_id" in first_data.lower():
        return "table"
    raise ValueError(
        f"{path}: cannot tell which domain format this is. Expected hmmscan "
        f"--domtblout (a '# target name' header), InterProScan TSV (an md5 in "
        f"column 2), or a table with a protein_id header."
    )


def load_domains(
    path: str | Path, *, fmt: str | None = None
) -> dict[str, list[DomainHit]]:
    """Read domains from any supported file, detecting the format when not told."""
    chosen = fmt or detect_format(path)
    if chosen not in READERS:
        raise ValueError(f"unknown domain format {chosen!r}; known: {sorted(READERS)}")
    return READERS[chosen](path)


def resolve_overlaps(
    hits: list[DomainHit], *, keep: str = "best"
) -> list[DomainHit]:
    """Reduce overlapping hits on one protein to a drawable set.

    Pfam reports several models for one region -- KH_1 and KH_dom over the same
    residues -- and a figure that draws both stacks two boxes where the protein has
    one domain. ``best`` keeps the hit with the lowest E-value in each overlapping
    group; ``all`` keeps everything, for a caller that wants to show the ambiguity.

    Ties are broken by the wider interval and then by name, so the result does not
    depend on input order -- a figure that changes between runs on identical data
    is worse than either choice.
    """
    if keep == "all":
        return sorted(hits, key=lambda h: (h.start, h.stop, h.name))
    if keep != "best":
        raise ValueError("keep must be 'best' or 'all'")

    ordered = sorted(
        hits,
        key=lambda h: (
            h.evalue if h.evalue is not None else float("inf"),
            -(h.stop - h.start),
            h.name,
        ),
    )
    chosen: list[DomainHit] = []
    for hit in ordered:
        if any(hit.start <= other.stop and other.start <= hit.stop for other in chosen):
            continue
        chosen.append(hit)
    return sorted(chosen, key=lambda h: (h.start, h.stop, h.name))


@dataclass
class DomainSet:
    """Domains for a whole orthologue set, with the bookkeeping to report on them."""

    by_protein: dict[str, list[DomainHit]] = field(default_factory=dict)
    source: str = "unknown"

    def for_protein(
        self, protein_id: str, *, sequence_length: int | None = None, keep: str = "best"
    ) -> list[DomainHit]:
        """Drawable domains for one protein, clipped to its own length.

        A hit past the end of the sequence means the annotation and the sequence are
        not the same protein -- a stale scan, or an isoform. It is DROPPED rather
        than clipped, because silently shortening a domain to fit produces a
        plausible-looking box in the wrong place.
        """
        hits = self.by_protein.get(protein_id, [])
        if sequence_length is not None:
            hits = [h for h in hits if h.stop <= sequence_length]
        return resolve_overlaps(hits, keep=keep)

    def coverage(self) -> dict[str, int]:
        """How many proteins have at least one domain, and how many hits in all."""
        return {
            "proteins_with_domains": sum(1 for v in self.by_protein.values() if v),
            "total_hits": sum(len(v) for v in self.by_protein.values()),
        }
