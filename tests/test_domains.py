"""Per-species domain annotation, read from sequence output rather than looked up.

The decision these tests pin: domains are annotated on each orthologue and used in
THAT orthologue's numbering. A reference domain projected by alignment is an
inference that inherits every gap on the way, and once written as an interval the
shift is invisible. Reading a scan of the orthologue itself has no such step.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.core.domains import (
    DomainHit,
    DomainSet,
    categorise,
    detect_format,
    load_domains,
    read_domain_tsv,
    read_hmmscan_domtblout,
    read_interproscan_tsv,
    resolve_overlaps,
)

# A real hmmscan --domtblout header plus two rows. Column 20/21 are the ENVELOPE
# bounds (ali 18/19 are narrower), and the free-text description runs to the end.
DOMTBLOUT = """\
# target name        accession   tlen query name           accession   qlen   E-value  score  bias   #  of  c-Evalue  i-Evalue  score  bias  from    to  from    to  from    to  acc description of target
#------------------- ---------- ----- -------------------- ---------- ----- --------- ------ ----- --- --- --------- --------- ------ ----- ----- ----- ----- ----- ----- ----- ---- ---------------------
KH_1                 PF00013.31    65 9606.ENSP00000250113 -            621   1.2e-18   66.1   0.1   1   2   1.1e-22   2.3e-19   69.0   0.0     2    64   214   278   210   282 0.95 KH domain
KH_1                 PF00013.31    65 9606.ENSP00000250113 -            621   1.2e-18   66.1   0.1   2   2   4.4e-20   9.1e-17   61.2   0.0     1    65   289   352   286   356 0.96 KH domain
Agenet               PF05641.16    70 9606.ENSP00000250113 -            621   3.1e-09   36.4   0.0   1   1   8.8e-13   1.8e-09   37.2   0.0     3    68    12    76     8    80 0.91 Agenet domain
"""


def test_envelope_coordinates_are_used_not_alignment(tmp_path):
    """HMMER's alignment bounds are where the model matched confidently; the
    envelope is where the domain is inferred to lie, which is what a domain figure
    means. Using the alignment under-draws every domain at both ends."""
    path = tmp_path / "d.tbl"
    path.write_text(DOMTBLOUT, encoding="utf-8")
    hits = read_hmmscan_domtblout(path)["9606.ENSP00000250113"]

    first = next(h for h in hits if h.start == 210)
    assert (first.start, first.stop) == (210, 282), "envelope, not ali (214-278)"


def test_each_domain_instance_is_its_own_hit(tmp_path):
    """FXR1 has two KH domains. Collapsing repeats of one model into a single
    interval would merge them into one box spanning the linker between them."""
    path = tmp_path / "d.tbl"
    path.write_text(DOMTBLOUT, encoding="utf-8")
    hits = read_hmmscan_domtblout(path)["9606.ENSP00000250113"]
    kh = sorted((h for h in hits if h.name == "KH_1"), key=lambda h: h.start)
    assert len(kh) == 2
    assert [(h.start, h.stop) for h in kh] == [(210, 282), (286, 356)]


def test_a_description_with_spaces_does_not_shift_the_columns(tmp_path):
    """The last field is free text. Splitting the whole line would put 'KH' in a
    numeric column and silently drop or corrupt the interval."""
    path = tmp_path / "d.tbl"
    path.write_text(DOMTBLOUT, encoding="utf-8")
    hits = read_hmmscan_domtblout(path)["9606.ENSP00000250113"]
    assert all(isinstance(h.start, int) and h.start > 0 for h in hits)
    assert {h.name for h in hits} == {"KH_1", "Agenet"}


def test_the_evalue_cut_is_applied(tmp_path):
    path = tmp_path / "d.tbl"
    path.write_text(DOMTBLOUT, encoding="utf-8")
    strict = read_hmmscan_domtblout(path, max_evalue=1e-18)
    assert all(h.evalue <= 1e-18 for h in strict["9606.ENSP00000250113"])
    assert len(strict["9606.ENSP00000250113"]) < 3


# --- categories -------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("KH_1", "rna_binding"),
        ("KH_dom", "rna_binding"),
        ("Agenet", "rna_binding"),
        ("Tudor", "rna_binding"),
        ("RRM_1", "rna_binding"),
        ("RGG_box", "low_complexity"),
        ("Pkinase", "catalytic"),
        ("Peptidase_C1", "peptidase"),
        ("HMG_box", "dna_binding"),
        ("Totally_Unknown_Fold", "default"),
    ],
)
def test_names_map_onto_the_family_agnostic_vocabulary(name, expected):
    assert categorise(name) == expected


def test_an_unrecognised_name_is_not_guessed():
    """A wrong category recolours a figure and invites a reading the data does not
    support, which is worse than a grey box that claims nothing."""
    assert categorise("Zzz_novel_module") == "default"


# --- overlap resolution -----------------------------------------------------


def _hit(name, start, stop, evalue=1e-10):
    return DomainHit("P", name, start, stop, categorise(name), evalue=evalue)


def test_overlapping_models_of_one_domain_collapse_to_the_best(tmp_path):
    """Pfam reports KH_1 and KH_dom over the same residues. Drawing both stacks two
    boxes where the protein has one domain."""
    hits = [_hit("KH_1", 210, 282, 1e-18), _hit("KH_dom", 212, 280, 1e-9)]
    kept = resolve_overlaps(hits)
    assert len(kept) == 1
    assert kept[0].name == "KH_1", "the lower E-value wins"


def test_non_overlapping_repeats_are_both_kept():
    hits = [_hit("KH_1", 210, 282), _hit("KH_1", 286, 356)]
    assert len(resolve_overlaps(hits)) == 2


def test_keep_all_preserves_the_ambiguity():
    hits = [_hit("KH_1", 210, 282, 1e-18), _hit("KH_dom", 212, 280, 1e-9)]
    assert len(resolve_overlaps(hits, keep="all")) == 2


def test_resolution_does_not_depend_on_input_order():
    """A figure that changes between runs on identical data is worse than either
    choice, so ties break on width then name rather than on arrival."""
    a = _hit("KH_1", 10, 50, 1e-10)
    b = _hit("KH_2", 20, 60, 1e-10)
    assert resolve_overlaps([a, b]) == resolve_overlaps([b, a])


# --- clipping, which is a refusal not a trim --------------------------------


def test_a_hit_past_the_sequence_end_is_dropped_not_clipped():
    """A hit beyond the end means the annotation and the sequence are different
    proteins -- a stale scan or another isoform. Shortening it to fit would produce
    a plausible box in the wrong place."""
    domains = DomainSet({"P": [_hit("KH_1", 10, 50), _hit("KH_1", 600, 660)]})
    kept = domains.for_protein("P", sequence_length=100)
    assert [(h.start, h.stop) for h in kept] == [(10, 50)]


def test_no_length_given_means_no_clipping():
    domains = DomainSet({"P": [_hit("KH_1", 600, 660)]})
    assert len(domains.for_protein("P")) == 1


def test_an_unannotated_protein_is_empty_not_an_error():
    assert DomainSet({}).for_protein("absent") == []


def test_invalid_intervals_are_refused_at_construction():
    with pytest.raises(ValueError, match="invalid interval"):
        DomainHit("P", "KH", 50, 10)
    with pytest.raises(ValueError, match="invalid interval"):
        DomainHit("P", "KH", 0, 10)


# --- the other two readers, and format detection ----------------------------

INTERPROSCAN = (
    "9606.ENSP00000250113\td41d8cd98f00b204e9800998ecf8427e\t621\tPfam\t"
    "PF00013\tKH domain\t214\t278\t1.2e-18\tT\t02-10-2026\tIPR004088\t"
    "KH domain, type 1\n"
)


def test_interproscan_prefers_the_interpro_description(tmp_path):
    """PF00013 means nothing on a figure axis."""
    path = tmp_path / "ips.tsv"
    path.write_text(INTERPROSCAN, encoding="utf-8")
    hit = read_interproscan_tsv(path)["9606.ENSP00000250113"][0]
    assert hit.name == "KH domain, type 1"
    assert hit.category == "rna_binding"
    assert (hit.start, hit.stop) == (214, 278)


def test_a_supplied_table_is_accepted_as_its_own_route(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text(
        "protein_id\tname\tstart\tstop\tcategory\n"
        "9606.ENSP1\tKH_1\t210\t282\trna_binding\n",
        encoding="utf-8",
    )
    hit = read_domain_tsv(path)["9606.ENSP1"][0]
    assert (hit.name, hit.start, hit.stop) == ("KH_1", 210, 282)
    assert hit.source == "supplied table"


def test_a_table_missing_a_column_says_which(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("protein_id\tname\tstart\n9606.X\tKH\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column\\(s\\) \\['stop'\\]"):
        read_domain_tsv(path)


@pytest.mark.parametrize(
    "content,expected",
    [(DOMTBLOUT, "hmmscan"), (INTERPROSCAN, "interproscan"),
     ("protein_id,name,start,stop\nP,KH,1,10\n", "table")],
)
def test_the_format_is_detected_from_the_content(tmp_path, content, expected):
    path = tmp_path / "f"
    path.write_text(content, encoding="utf-8")
    assert detect_format(path) == expected
    assert load_domains(path)


def test_an_unidentifiable_file_says_what_was_expected(tmp_path):
    path = tmp_path / "f"
    path.write_text("some,unrelated,csv\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot tell which domain format"):
        detect_format(path)


def test_coverage_reports_how_much_of_the_set_is_annotated():
    domains = DomainSet({"A": [_hit("KH_1", 1, 10)], "B": [], "C": [_hit("KH_1", 1, 5)]})
    assert domains.coverage() == {"proteins_with_domains": 2, "total_hits": 2}
