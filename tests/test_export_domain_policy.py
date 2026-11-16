"""Overlap policy follows the annotation's source, and the claim is exported.

Two defects, found on a real seven-protein panel.

``DomainSet.for_protein`` resolved overlaps with ``keep="best"`` for every
source. That is right for a scan -- Pfam reports KH_1 and KH_dom over the same
residues and drawing both stacks two boxes where the protein has one domain --
and wrong for a curated table, where nesting is the annotation: RPS2's
dsRNA-binding-like domain (96-167) sits inside its uS5 core (51-268), and
keeping only the core deleted the rna_binding domain the whole hypothesis is
about. Both rna_binding domains in the panel disappeared this way.

``export_run`` then wrote no ``primary_domain_category``, so amyloscope ran the
enrichment test over every category with none declared primary -- a search
wearing a hypothesis's clothes.
"""

from __future__ import annotations

import pytest

from aggressor_wrappers.core.domains import (
    DEFAULT_KEEP,
    DomainHit,
    DomainSet,
    read_domain_tsv,
)

#: RPS2's real annotation: three domains, two of them nested inside the core.
TABLE = """protein_id\tname\tstart\tstop\tcategory
RPS2\tBasic Arg/Gly-rich repeat segment\t35\t55\tdisordered
RPS2\tuS5 ribosomal core domain\t51\t268\tstructured_core
RPS2\tdsRNA-binding-like domain\t96\t167\trna_binding
"""


def _table(tmp_path):
    path = tmp_path / "domains.tsv"
    path.write_text(TABLE, encoding="utf-8")
    return path


def test_a_curated_table_keeps_nested_domains(tmp_path):
    hits = read_domain_tsv(_table(tmp_path))
    domains = DomainSet(by_protein=hits, source="table", keep=DEFAULT_KEEP["table"])

    kept = domains.for_protein("RPS2", sequence_length=293)

    assert [h.name for h in kept] == [
        "Basic Arg/Gly-rich repeat segment",
        "uS5 ribosomal core domain",
        "dsRNA-binding-like domain",
    ]
    assert "rna_binding" in {h.category for h in kept}


def test_a_scan_still_collapses_model_redundancy(tmp_path):
    hits = read_domain_tsv(_table(tmp_path))
    domains = DomainSet(by_protein=hits, source="scan", keep=DEFAULT_KEEP["hmmscan"])

    kept = domains.for_protein("RPS2", sequence_length=293)

    assert len(kept) == 1
    assert kept[0].name == "uS5 ribosomal core domain"


def test_an_explicit_keep_overrides_the_set_policy(tmp_path):
    hits = read_domain_tsv(_table(tmp_path))
    domains = DomainSet(by_protein=hits, source="table", keep="all")

    assert len(domains.for_protein("RPS2", keep="best")) == 1


def test_the_default_policy_is_unchanged_for_a_set_that_declares_none():
    """A DomainSet built without `keep` behaves as it always did."""
    domains = DomainSet(
        by_protein={
            "P": [
                DomainHit("P", "outer", 1, 100, "structured_core", "x", evalue=1e-30),
                DomainHit("P", "inner", 10, 20, "rna_binding", "x", evalue=1e-10),
            ]
        }
    )
    assert len(domains.for_protein("P")) == 1


def test_categories_reports_what_a_primary_claim_can_name(tmp_path):
    hits = read_domain_tsv(_table(tmp_path))
    domains = DomainSet(by_protein=hits, source="table")

    assert domains.categories() == {"disordered", "structured_core", "rna_binding"}


# --------------------------------------------------------------------------- #
# The exported config has to carry the pre-declared claim.
# --------------------------------------------------------------------------- #


def _panel(tmp_path):
    from aggressor_wrappers.core.domains import read_domain_tsv

    hits = read_domain_tsv(_table(tmp_path))
    return DomainSet(by_protein=hits, source="table", keep="all")


def test_config_carries_hypothesis_and_primary_category(tmp_path):
    from aggressor_wrappers.core.amyloscope_export import build_amyloscope_config

    proteins = {
        "RPS2": {
            "length": 5,
            "sequence": "MAVLK",
            "domains": [
                {"name": "dsRNA-binding-like domain", "start": 1, "stop": 5,
                 "category": "rna_binding"},
            ],
        }
    }
    config = build_amyloscope_config(
        proteins,
        ["tango"],
        tmp_path / "out",
        name="panel",
        hypothesis="APRs coincide with rRNA-contacting surfaces.",
        primary_domain_category="rna_binding",
    )
    text = config.read_text(encoding="utf-8")

    assert "primary_domain_category: rna_binding" in text
    assert "hypothesis: >" in text
    assert "rRNA-contacting" in text
    # The panel name must survive the tools loop, which used to rebind it.
    assert text.splitlines()[0] == "name: panel"


def test_a_category_no_domain_carries_is_refused(tmp_path):
    from aggressor_wrappers.core.amyloscope_export import build_amyloscope_config
    import aggressor_wrappers.core.amyloscope_export as module

    proteins = {
        "RPS2": {
            "length": 5,
            "sequence": "MAVLK",
            "domains": [
                {"name": "core", "start": 1, "stop": 5, "category": "structured_core"},
            ],
        }
    }
    # build_amyloscope_config writes whatever it is given; the refusal belongs to
    # export_run and to the CLI, which check before anything is written.
    merged = tmp_path / "merged"
    merged.mkdir()
    with pytest.raises(FileNotFoundError):
        module.export_run(merged, tmp_path / "out")

    config = build_amyloscope_config(
        proteins, ["tango"], tmp_path / "out2", primary_domain_category="rna_binding"
    )
    assert "primary_domain_category: rna_binding" in config.read_text(encoding="utf-8")


def test_a_hard_wrapped_hyphen_is_not_split(tmp_path):
    """"aggregation-\\ncompetent" must not become "aggregation- competent"."""
    from aggressor_wrappers.core.amyloscope_export import _wrap_block

    folded = " ".join(_wrap_block("the same segments are aggregation-\ncompetent here"))

    assert "aggregation-competent" in folded
    assert "aggregation- competent" not in folded


def test_paragraph_breaks_survive_folding():
    from aggressor_wrappers.core.amyloscope_export import _wrap_block

    lines = _wrap_block("first claim.\n\nsecond claim.")

    assert "" in lines
    assert lines[0].startswith("first")
    assert lines[-1].startswith("second")


# --------------------------------------------------------------------------- #
# A category the primary claim can actually name.
# --------------------------------------------------------------------------- #


def test_a_spelled_out_binding_name_is_categorised():
    """InterPro says what a domain binds in words more often than in a token.

    "uS15/NS1, RNA-binding domain superfamily" carries no family token, so it
    fell through to "default" -- and a --primary-domain-category of rna_binding
    then missed the one domain it was declared for.
    """
    from aggressor_wrappers.core.domains import categorise

    assert categorise("uS15/NS1, RNA-binding domain superfamily") == "rna_binding"
    assert categorise("Double-stranded RNA binding motif") == "rna_binding"
    assert categorise("Winged helix DNA-binding domain") == "dna_binding"


def test_family_token_names_still_categorise_the_same_way():
    from aggressor_wrappers.core.domains import categorise

    assert categorise("KH domain, type 1") == "rna_binding"
    assert categorise("dsRNA-binding-like domain") == "rna_binding"
    assert categorise("Homeobox domain") == "dna_binding"
    assert categorise("Prion-like Q/N-rich domain") == "low_complexity"


def test_an_unknown_name_is_still_not_guessed():
    from aggressor_wrappers.core.domains import categorise

    assert categorise("Small ribosomal subunit protein uS15") == "default"


# --------------------------------------------------------------------------- #
# --domains is repeatable.
#
# A curated table covering five of seven proteins and a scan covering the other
# two cannot be concatenated: the formats differ, and the second file's header
# becomes a data row whose start column is the word "start".
# --------------------------------------------------------------------------- #


def _run_export(tmp_path, argv):
    from aggressor_wrappers.cli.export_amyloscope import main

    return main(argv)


def _merged(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "RPS2_merged.csv").write_text(
        "position,aa_name,tango_score,tango_bin\n"
        "1,M,0.0,0\n2,A,97.2,1\n3,V,96.1,1\n4,L,0.0,0\n5,K,0.0,0\n",
        encoding="utf-8",
    )
    return merged


def test_two_annotation_files_are_combined(tmp_path):
    first = tmp_path / "curated.tsv"
    first.write_text(
        "protein_id\tname\tstart\tstop\tcategory\n"
        "RPS2\tdsRNA-binding-like domain\t1\t3\trna_binding\n",
        encoding="utf-8",
    )
    second = tmp_path / "extra.tsv"
    second.write_text(
        "protein_id\tname\tstart\tstop\tcategory\n"
        "RPS2\tC-terminal tail\t4\t5\tterminal\n",
        encoding="utf-8",
    )
    out = tmp_path / "export"

    code = _run_export(
        tmp_path,
        [
            str(_merged(tmp_path)),
            "-o", str(out),
            "--domains", str(first),
            "--domains", str(second),
            "--primary-domain-category", "rna_binding",
        ],
    )

    assert code == 0
    config = (out / "config.yaml").read_text(encoding="utf-8")
    assert "dsRNA-binding-like domain" in config
    assert "C-terminal tail" in config
    assert "primary_domain_category: rna_binding" in config


def test_a_missing_second_file_is_reported_by_name(tmp_path):
    first = tmp_path / "curated.tsv"
    first.write_text(
        "protein_id\tname\tstart\tstop\tcategory\n"
        "RPS2\tdsRNA-binding-like domain\t1\t3\trna_binding\n",
        encoding="utf-8",
    )
    out = tmp_path / "export"

    code = _run_export(
        tmp_path,
        [
            str(_merged(tmp_path)),
            "-o", str(out),
            "--domains", str(first),
            "--domains", str(tmp_path / "absent.tsv"),
        ],
    )

    assert code == 2
    assert not (out / "config.yaml").exists()
