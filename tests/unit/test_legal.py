"""Unit tests for the legal-admissibility layer.

The core contract: classify_columns reports a coverage state and, where no rule
pack covers the (use_case, jurisdiction), it returns `uncovered` honestly rather
than fabricating a verdict.
"""

from vfairness.legal import (
    LEGAL_DATA_REVISION,
    classify_columns,
    load_rules,
    map_domain_to_use_case,
)

REQUIRED_KEYS = {
    "coverage",
    "useCase",
    "jurisdiction",
    "jurisdictionId",
    "sourceRevision",
    "findings",
    "summary",
    "unmappedColumns",
}


def test_revision_is_a_nonempty_string():
    assert isinstance(LEGAL_DATA_REVISION, str)
    assert LEGAL_DATA_REVISION


def test_unknown_domain_maps_to_generic():
    assert map_domain_to_use_case(None) == "generic"
    assert map_domain_to_use_case("zzzqqq") == "generic"


def test_missing_jurisdiction_is_uncovered_not_fabricated():
    result = classify_columns(["age", "gender"], "lending", None)
    assert result["coverage"] == "uncovered"
    assert REQUIRED_KEYS.issubset(result)
    assert isinstance(result["findings"], list)
    # R-9 (2026-09-09): this used to pin a FOUR-key summary, which pinned the
    # defect. A column whose legal status was never determined (no rule-pack
    # entry) is a fourth state, counted under ``unknown``; with no pack for the
    # pair, both mapped columns land there rather than in no counter at all.
    assert set(result["summary"]) == {
        "forbidden",
        "restricted",
        "monitoringOnly",
        "allowed",
        "unknown",
    }
    assert result["summary"]["unknown"] == 2


def test_unmappable_column_is_reported_unmapped():
    result = classify_columns(["zzz_nonsense_column_42"], "lending", None)
    assert "zzz_nonsense_column_42" in result["unmappedColumns"]


def test_load_rules_known_pack_loads_and_missing_pack_is_none():
    pack = load_rules("lending", "us-federal")
    assert pack is not None
    assert "rules" in pack
    assert load_rules("no_such_use_case", "no_such_jurisdiction") is None


def test_classify_columns_degrades_instead_of_raising():
    result = classify_columns(["x"], "qzx_unknown", "qzx_nowhere")
    assert result["coverage"] in {"covered", "partial", "uncovered"}
