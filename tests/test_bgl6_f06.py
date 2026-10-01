"""BGL6 AUDIT of batch F06: the overturns, as executable evidence.

Written by the audit6 auditor on 2026-09-28 while attacking the BGL5 fixes for
``operations/reporting/compliance.py``, ``preprocessing/bias_detection/detector.py``,
``preprocessing/feature_engineering/correlation.py`` and
``multi_agent/groupthink.py``.

Every test below asserts the CORRECT behaviour. While a defect was live the test
recording it was marked ``xfail(strict=True)``, the convention the BGL4 audit
files use: the file is green while the defect is live and turns LOUD the moment
one is fixed, so a marker cannot quietly outlive its defect.

ALL TEN RECORDS ARE CLOSED, 2026-09-29. ``RECORDED_DEFECTS_STILL_OPEN`` is empty,
no ``xfail`` marker remains, and every test here is now a PIN asserting the
corrected behaviour, each one sabotage-checked (the source line it guards was
broken, the pin was shown RED, and the source was restored and proved
byte-identical with ``diff -q``). Each docstring opens with "CLOSED 2026-09-29",
states the fix and the re-measured numbers, and keeps the original record verbatim
below a "THE ORIGINAL RECORD follows." line, so the before-state is not lost.

The fixes touched ``operations/reporting/compliance.py``,
``preprocessing/feature_engineering/correlation.py``, ``rendering/adapters.py``
and ``multi_agent/groupthink.py``. Nothing in the original recording pass modified
a source file; the fixes came afterwards, in the commits the docstrings describe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ===========================================================================
# 1. compliance.generate_iso42001_evidence_map: the content test does not
#    recurse, so ONE level of nesting restores the whole defect.
# ===========================================================================

_NESTED_PLACEHOLDER_WIZARD = {
    "system_profile": {"purpose": {"note": None}},
    "bias_results": [{"finding": {"detail": None}}],
    "metric_results": {"demographic_parity": {"value": None}},
    "risk_register": [
        {
            "risk": {"label": None},
            "treatment": {"plan": None},
            "residual_risk": {"score": None},
        }
    ],
    "monitoring_config": {"enabled": {"flag": None}},
    "intervention_results": [{"intervention": {"kind": None}}],
    "dpia": {"completed": {"state": None}},
    "annex_iv": {"completed": {"state": None}},
    "model_card": {"completed": {"state": None}},
}

_FULL_WIZARD = {
    "system_profile": {"purpose": "credit scoring"},
    "bias_results": [{"finding": "disparity"}],
    "metric_results": {"demographic_parity": 0.12},
    "risk_register": [
        {"risk": "proxy discrimination", "treatment": "reweighting", "residual_risk": 4}
    ],
    "monitoring_config": {"enabled": True},
    "intervention_results": [{"intervention": "threshold shift"}],
    "dpia": {"completed": True},
    "annex_iv": {"completed": True},
    "model_card": {"completed": True},
}


#: HOW MANY CLAIMS THE SECOND-ROUND AUDIT RECORDED IN THIS FILE, on 2026-09-28.
#:
#: A HISTORICAL FACT, and it must not move. The register used to take this count by
#: counting the test functions in the file, which was right on the day the audit
#: landed and wrong from the first fix onwards: inverting a witness into a pin
#: renames it, and a fix arrives with its own over-correction control, so closing
#: records made the audit look BIGGER. It had grown from 59 claims to 69 by the time
#: anybody added them up, on a page whose whole subject is not misstating what was
#: measured.
#:
#: Recovered from the audit's own baseline commit e6a5780, which is where every one
#: of these numbers comes from.
RECORDED_CLAIMS_AT_AUDIT = 11

#: THE TESTS IN THIS FILE THAT STILL RECORD AN OPEN DEFECT.
#:
#: A name is removed from this list in the SAME commit that fixes its defect and
#: inverts the test into a pin, so the two cannot drift. It is declared here rather
#: than inferred from pass/fail because a test that RECORDS a defect passes while the
#: defect is live, which is indistinguishable by execution from a pin that passes
#: because the defect is gone.
#:
#: Read by scripts/bgl6_register.py, published as counts in
#: docs/bgl6-audit-register.json and in QUALITY_AND_HARDENING.md, and checked by
#: tests/test_bgl6_register_is_honest.py, which refuses a name that is not a test
#: function in this module.
RECORDED_DEFECTS_STILL_OPEN: list[str] = []


def test_a_nested_contentless_wizard_does_not_earn_the_full_wizard_iso_score():
    """CLOSED 2026-09-29. ``_iso_section_has_content`` recurses now.

    THE FIX, in ``operations/reporting/compliance.py``: a new
    ``_iso_value_has_content`` walks a nested value to its leaves (a non-None
    scalar is content at any depth, a container is content only when something
    inside it is, with a depth cap of 12 so a self-referential wizard cannot
    raise RecursionError inside a report generator). The mapping branch of
    ``_iso_section_has_content`` calls it instead of skipping only None, blank and
    EMPTY containers, and ``_iso_risk_entries_with`` calls it for the same reason:
    a ``treatment`` holding ``{"plan": None}`` used to count as a recorded
    treatment plan.

    RE-MEASURED 2026-09-29 by execution:

      nested placeholder -> coverage_percent 0.0, covered 0, partial 0, gaps 13
      full wizard        -> coverage_percent 92.3, covered 11, partial 2, gaps 0

    so the healthy input is untouched (92.3 before the fix, 92.3 after) and only
    the placeholder moved. Over-correction controls below assert the full
    wizard's real number, plus a nested measured zero and a nested ``False``,
    which are evidence and must still count.

    SABOTAGE 2026-09-29: replacing the recursive mapping branch with the old
    ``return True`` on any non-empty container turns this test RED with
    "a wizard whose every leaf is None earned 92.3% of ISO 42001 with 11
    controls covered and 0 gaps". Source restored and proved byte-identical
    with ``diff -q``.

    THE ORIGINAL RECORD follows.

    ``_iso_section_has_content`` stops at the first level of a mapping.

    Measured 2026-09-28: every leaf of the wizard above is ``None``, so not one
    section carries evidence, and ``generate_iso42001_evidence_map`` returns
    coverage_percent 92.3 with 11 covered, 2 partial, 0 gaps and A.4.3 reading
    "Treatment plans recorded for 1 of 1 risk entries." The whole result is
    IDENTICAL (apart from ``map_id`` and ``generated_at``) to the one the fix's
    own over-correction control asserts for ``_FULL_WIZARD``. The mapping branch
    returns True for any value that is a non-empty container, while the list
    branch recurses, so one level of nesting is enough to restore the defect the
    fix was written to end.
    """
    from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

    result = generate_iso42001_evidence_map(_NESTED_PLACEHOLDER_WIZARD)
    assert result["coverage_percent"] < 25.0, (
        f"a wizard whose every leaf is None earned {result['coverage_percent']}% of "
        f"ISO 42001 with {result['covered']} controls covered and {result['gaps']} gaps"
    )
    assert result["covered"] == 0, result["covered"]

    # The risk-register half of the fix, which ``has_risks`` hides on the wizard
    # above: an entry with a REAL risk and a NESTED PLACEHOLDER treatment. Before
    # the fix A.4.3 read "covered / Treatment plans recorded for 1 of 1 risk
    # entries."; it is PARTIAL now, with the count that earns the word.
    nested_treatment = generate_iso42001_evidence_map(
        {
            "risk_register": [
                {
                    "risk": "proxy discrimination",
                    "treatment": {"plan": None},
                    "residual_risk": {"score": None},
                }
            ]
        }
    )
    by_id = {c["control_id"]: c for c in nested_treatment["controls"]}
    assert by_id["A.4.3"]["coverage_status"] == "partial", by_id["A.4.3"]
    assert by_id["A.4.4"]["coverage_status"] == "partial", by_id["A.4.4"]
    assert "only 0 carry a treatment plan" in by_id["A.4.3"]["evidence_description"]

    # OVER-CORRECTION CONTROL: a blanket refusal would pass every line above.
    # The full wizard's real number, unchanged by the fix.
    full = generate_iso42001_evidence_map(_FULL_WIZARD)
    assert full["coverage_percent"] == 92.3, full["coverage_percent"]
    assert (full["covered"], full["partial"], full["gaps"]) == (11, 2, 0), full

    # A measured zero and a measured False are EVIDENCE, nested or not.
    assert (
        generate_iso42001_evidence_map({"metric_results": {"dp": {"value": 0.0}}})["covered"] >= 1
    )
    assert (
        generate_iso42001_evidence_map({"monitoring_config": {"enabled": {"flag": False}}})[
            "covered"
        ]
        >= 1
    )
    healthy_register = generate_iso42001_evidence_map(
        {
            "risk_register": [
                {"risk": "proxy discrimination", "treatment": "reweighting", "residual_risk": 4}
            ]
        }
    )
    healthy_by_id = {c["control_id"]: c for c in healthy_register["controls"]}
    assert healthy_by_id["A.4.3"]["coverage_status"] == "covered", healthy_by_id["A.4.3"]
    assert (
        healthy_by_id["A.4.3"]["evidence_description"]
        == "Treatment plans recorded for 1 of 1 risk entries."
    )


def test_the_nested_placeholder_map_is_not_identical_to_the_full_wizard_map():
    """CLOSED 2026-09-29. The two results now differ in every headline field.

    Same fix as the test above. RE-MEASURED 2026-09-29: the placeholder map
    reports 0.0% / 0 covered / 13 gaps against the full wizard's 92.3% / 11
    covered / 0 gaps, so a reader can act on the difference.

    SABOTAGE 2026-09-29: with the old non-recursive mapping branch restored the
    two payloads are equal again and this test fails on
    ``assert placeholder != full``. Source restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    The two results differ in no field a reader could act on.
    """
    from vfairness.operations.reporting.compliance import generate_iso42001_evidence_map

    def _strip(d):
        return {k: v for k, v in d.items() if k not in ("map_id", "generated_at")}

    placeholder = _strip(generate_iso42001_evidence_map(_NESTED_PLACEHOLDER_WIZARD))
    full = _strip(generate_iso42001_evidence_map(_FULL_WIZARD))
    assert placeholder != full
    assert placeholder["coverage_percent"] == 0.0, placeholder["coverage_percent"]
    assert full["coverage_percent"] == 92.3, full["coverage_percent"]


# ===========================================================================
# 2. compliance.build_assurance_verdict: the scope-limitation machinery is fed
#    only by assessable=False, and an unreadable `gap` on an ASSESSABLE row
#    deletes every disparate-impact finding in the run.
# ===========================================================================

_REAL_GAP = {
    "attribute": "race",
    "assessable": True,
    "gap": 0.23,
    "significant": True,
    "worstGroup": "Black",
    "referenceGroup": "White",
}
_UNMEASURED_GAP = {
    "attribute": "age",
    "assessable": True,
    "gap": None,
    "significant": True,
    "worstGroup": "50+",
    "referenceGroup": "30-",
}


def test_an_unmeasurable_gap_does_not_erase_a_measured_adverse_finding():
    """CLOSED 2026-09-29. The ranking reads the gap through a three-state reader.

    THE FIX, in ``operations/reporting/compliance.py`` ``build_assurance_verdict``:
    the disparate-impact block no longer sorts the raw field. It reads each
    assessable row's gap through ``_as_measured_number`` (this file's own reader:
    np.float32 passes, NaN, inf, bool and a numeric string are refused), ranks
    only the rows with a measured magnitude, and hands the rest to a NEW
    per-variable partial-failure block that names them as a scope limitation.
    ``gap = float(v.get("gap") or 0.0)`` is gone with it: an unreadable magnitude
    used to become 0.0 and get skipped by ``gap < 0.05`` as if measured clean.
    The disclosure sits in its OWN try block, because the two swallowing each
    other is the precise mechanism of this defect.

    RE-MEASURED 2026-09-29 by execution:

      [race 0.23]            -> Adverse, blocks True, 1 finding, 0 unassessed
      [race 0.23, age None]  -> Adverse, blocks True, 2 findings, 1 unassessed
      [age None]             -> Qualified (scope limitation), 'age' named
      [race np.float32(0.23)]-> Adverse, 1 finding, 0 unassessed

    The last line is the over-correction control: a reader that refused
    everything it could not prove would push the float32 row into unassessed and
    lose the 23-point finding, which is the same harm in the other direction.

    SABOTAGE 2026-09-29, twice, one per half of the fix, source restored and
    proved byte-identical with ``diff -q`` after each:

    1. ``key=lambda v: v.get("gap", 0.0)`` put back on the unsplit list, with
       ``float(v.get("gap") or 0.0)``. RED: "a row whose gap was never measured
       turned the verdict into 'Qualified' with 1 finding(s) and 1 unassessed",
       ``assert 'Qualified' == 'Adverse'``. The 23-point finding is deleted by
       the swallowed TypeError exactly as recorded, and the verdict is only
       'Qualified' rather than 'Unqualified' because the disclosure block still
       runs: the two halves are INDEPENDENT, which is the point of the separate
       try.
    2. the disclosure ``add(...)`` short-circuited. RED on
       ``assert ['disparate_impact'] == ['disparate_impact',
       'disparate_impact_bias_unassessed']``.

    THE ORIGINAL RECORD follows.

    One ``gap: None`` row silently deletes a 23-point significant gap.

    Measured 2026-09-28. ``build_assurance_verdict(per_variable=[_REAL_GAP])``
    gives overall 'Adverse', blocksDeployment True, 1 finding. Adding ONE row
    whose gap was not computed gives overall 'Unqualified', blocksDeployment
    False, findings [], unassessed [], "no material fairness defect found on the
    assessed attributes". The ranking's ``sorted(..., key=lambda v:
    v.get("gap", 0.0))`` raises TypeError on None, and the ``except Exception``
    around the whole loop swallows it, so the comment in this same function
    ("EVERY disparate-impact finding in the run, the 23-point gap included, left
    the verdict silently") describes a failure that is still live through the
    sort key. A sibling consumer of the same field guards it with
    ``isinstance(raw_gap, (int, float))`` (pulse/orchestrator.py:5036).
    """
    from vfairness.operations.reporting.compliance import build_assurance_verdict

    alone = build_assurance_verdict(per_variable=[_REAL_GAP], domain="hiring", jurisdiction="US")
    assert alone["overall"] == "Adverse"

    with_unmeasured = build_assurance_verdict(
        per_variable=[_REAL_GAP, _UNMEASURED_GAP], domain="hiring", jurisdiction="US"
    )
    assert with_unmeasured["overall"] == "Adverse", (
        f"a row whose gap was never measured turned the verdict into "
        f"{with_unmeasured['overall']!r} with {len(with_unmeasured['findings'])} finding(s) "
        f"and {len(with_unmeasured['unassessed'])} unassessed: "
        f"{with_unmeasured['oneLineVerdict']}"
    )
    # The measured finding SURVIVES, and the unmeasurable row is disclosed
    # rather than deleted or promoted.
    assert with_unmeasured["blocksDeployment"] is True
    assert [f["type"] for f in with_unmeasured["findings"]] == [
        "disparate_impact",
        "disparate_impact_bias_unassessed",
    ], [f["type"] for f in with_unmeasured["findings"]]
    assert [u["attribute"] for u in with_unmeasured["unassessed"]] == ["age"]
    assert "23 pts less" in with_unmeasured["findings"][0]["evidence"]

    # OVER-CORRECTION CONTROL. np.float32 is a MEASUREMENT. A reader that
    # refused what it could not prove would move this row to unassessed and lose
    # the 23-point finding, the same harm mirrored.
    f32 = build_assurance_verdict(
        per_variable=[dict(_REAL_GAP, gap=np.float32(0.23))], domain="hiring", jurisdiction="US"
    )
    assert f32["overall"] == "Adverse", f32["oneLineVerdict"]
    assert len(f32["unassessed"]) == 0, f32["unassessed"]
    assert "23 pts less" in f32["findings"][0]["evidence"], f32["findings"][0]["evidence"]


def test_an_assessable_attribute_with_no_measured_gap_is_named_somewhere():
    """CLOSED 2026-09-29. A partial failure is now a named scope limitation.

    Same fix as the test above: a new per-variable partial-failure block adds a
    ``disparate_impact_bias_unassessed`` finding with ``assessed=False``, which
    the R-10 scope-limitation machinery already routes into ``unassessed`` and
    into ``oneLineVerdict``.

    RE-MEASURED 2026-09-29: ``per_variable=[{age, assessable True, gap None}]``
    returns overall 'Qualified' with oneLineVerdict "Qualified opinion (scope
    limitation): no material fairness defect found on the assessed attributes,
    but disparate impact bias could not be assessed for 'age' (unassessed). This
    opinion does not cover what was not assessed; collect the missing data and
    re-run.", one entry in ``unassessed`` naming 'age', and blocksDeployment
    False (a scope limitation is an "except for" matter, not a defect).

    Over-correction control: an assessable row with a MEASURED 0.01 gap stays
    Unqualified with an EMPTY unassessed list, so this is not a blanket "name
    every attribute" change.

    SABOTAGE 2026-09-29: deleting the new block's ``add(...)`` call turns this
    test RED with "'age' appears nowhere in the payload: Unqualified opinion: no
    material fairness defect found on the assessed attributes. Keep monitoring."
    Source restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    An attribute nothing measured is neither a finding nor a scope limitation.

    Measured 2026-09-28: ``per_variable=[{'attribute': 'age', 'assessable':
    True, 'gap': None}]`` alone returns overall 'Unqualified', unassessed [],
    findings [], "no material fairness defect found on the assessed attributes.
    Keep monitoring." The fix covers only ``assessable=False``, so a partial
    failure (assessable row, unmeasurable magnitude) is still invisible.
    """
    from vfairness.operations.reporting.compliance import build_assurance_verdict

    verdict = build_assurance_verdict(per_variable=[_UNMEASURED_GAP])
    named = "age" in json.dumps(verdict)
    assert named, f"'age' appears nowhere in the payload: {verdict['oneLineVerdict']}"
    assert verdict["overall"] == "Qualified", verdict["oneLineVerdict"]
    assert verdict["blocksDeployment"] is False
    assert [u["attribute"] for u in verdict["unassessed"]] == ["age"], verdict["unassessed"]
    assert "could not be assessed for 'age'" in verdict["oneLineVerdict"]
    assert "was NOT measured" in verdict["unassessed"][0]["evidence"]

    # OVER-CORRECTION CONTROL: a MEASURED small gap is a clean assessed result,
    # not a scope limitation.
    clean = build_assurance_verdict(per_variable=[dict(_UNMEASURED_GAP, gap=0.01)])
    assert clean["overall"] == "Unqualified", clean["oneLineVerdict"]
    assert clean["unassessed"] == [], clean["unassessed"]


# ===========================================================================
# 3. compliance.compute_adverse_action_reasons: the new mirror guard is keyed
#    on isinstance(val, (int, float)), which REJECTS np.float32.
# ===========================================================================

_SHAP = {
    "debt_ratio": -0.40,
    "income": -0.30,
    "zip_code": -0.25,
    "credit_score": -0.20,
    "age_proxy": -0.15,
    "employment": -0.10,
}
_NAMES_WITHOUT_DEBT = ["income", "zip_code", "credit_score", "age_proxy", "employment"]


def test_a_float32_adverse_attribution_that_was_never_ranked_is_disclosed():
    """CLOSED 2026-09-29. The guard reads the value, it does not typecheck it.

    THE FIX, in ``operations/reporting/compliance.py``
    ``compute_adverse_action_reasons``: the mirror guard's
    ``isinstance(val, (int, float)) and not isinstance(val, bool)`` chain is
    replaced by ``_as_measured_number``, this module's own reader, which accepts
    any numeric type ``float()`` accepts (np.float32, np.int64) and refuses None,
    bool, np.bool_, a numeric-looking string, NaN and infinity.

    RE-MEASURED 2026-09-29 by execution, same scenario, three dtypes:

      python float -> complete False, not_ranked ['debt_ratio'], 1 warning
      np.float32   -> complete False, not_ranked ['debt_ratio'], 1 warning
      np.float64   -> complete False, not_ranked ['debt_ratio'], 1 warning

    Over-correction controls, both asserted below: the HEALTHY float32 notice
    (nothing omitted from feature_names) is complete True, not_ranked [], 0
    warnings, codes RC01 debt_ratio / RC02 income / RC03 credit_score / RC04
    employment, so the strongest driver is back in the top slot and the proxy is
    out of it; and ``np.bool_(True)`` plus the string "-0.9" are still NOT
    counted as adverse attributions, so the wider reader did not become a
    blanket accept.

    SABOTAGE 2026-09-29: putting the isinstance chain back turns this test RED
    with "complete=True, not_ranked=[], codes=[('RC01', 'income'), ('RC02',
    'credit_score'), ('RC03', 'employment'), ('RC04', 'zip_code')]". Source
    restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    np.float32 SHAP values walk straight past the new guard.

    Measured 2026-09-28 with the row's own scenario and the SAME numbers cast to
    ``np.float32`` (what the shap library returns for tree models, and the dtype
    of any float32 array):

      python float  -> complete False, adverse_attributions_not_ranked
                       ['debt_ratio'], 1 warning
      np.float32    -> complete True,  adverse_attributions_not_ranked [],
                       0 warnings, RC04 = zip_code (a PROXY) in the slot the
                       omitted strongest driver vacated

    which is byte for byte the BEFORE state the fix records. The guard uses
    ``isinstance(val, (int, float))``; np.float64 subclasses float and passes,
    np.float32 does not. The forward direction in the same function coerces
    through ``float()`` and handles both, and ``correlation._is_finite_number``
    carries a docstring about exactly this trap.
    """
    import warnings as _warnings

    from vfairness.operations.reporting.compliance import compute_adverse_action_reasons

    shap32 = {k: np.float32(v) for k, v in _SHAP.items()}
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        notice = compute_adverse_action_reasons(
            shap32, _NAMES_WITHOUT_DEBT, ["zip_code", "age_proxy"]
        )
    assert notice.adverse_attributions_not_ranked == ["debt_ratio"], (
        f"complete={notice.complete}, not_ranked={notice.adverse_attributions_not_ranked}, "
        f"codes={[(e['code'], e['feature']) for e in notice]}"
    )
    assert notice.complete is False
    assert any("absent from feature_names" in str(c.message) for c in caught), [
        str(c.message) for c in caught
    ]

    # OVER-CORRECTION CONTROL 1: the HEALTHY float32 notice, with its real codes.
    # A guard that refused every float32 would report this one incomplete too.
    with _warnings.catch_warnings(record=True) as caught_ok:
        _warnings.simplefilter("always")
        healthy = compute_adverse_action_reasons(shap32, list(_SHAP), ["zip_code", "age_proxy"])
    assert healthy.complete is True
    assert healthy.adverse_attributions_not_ranked == []
    assert len(caught_ok) == 0, [str(c.message) for c in caught_ok]
    assert [(e["code"], e["feature"]) for e in healthy] == [
        ("RC01", "debt_ratio"),
        ("RC02", "income"),
        ("RC03", "credit_score"),
        ("RC04", "employment"),
    ], [(e["code"], e["feature"]) for e in healthy]

    # OVER-CORRECTION CONTROL 2: a flag and a numeric-looking STRING are not
    # attributions, so the wider reader is not a blanket accept.
    loose = compute_adverse_action_reasons(
        {"income": -0.3, "flag": np.bool_(True), "note": "-0.9"}, ["income"], []
    )
    assert loose.adverse_attributions_not_ranked == []
    assert loose.complete is True


# ===========================================================================
# 4. compliance.compute_signed_test_log: `reproducibility_info` is RETURNED in
#    the document and left outside the seal, and it duplicates two sealed
#    fields, so the two halves can be desynchronised without breaking the hash.
# ===========================================================================


def _rehash(doc: dict) -> str:
    """The verification the docstring documents: rebuild the payload and compare."""
    payload = json.dumps(
        {
            "test_timestamp": doc["test_timestamp"],
            "test_timestamp_source": doc["test_timestamp_source"],
            "log_built_at": doc["log_built_at"],
            "data_hash": doc["data_hash"],
            "lib_version": doc["library_version"],
            "metrics_snapshot": doc["metrics_snapshot"],
            "summary": doc["test_results_summary"],
            "intervention_history": doc["intervention_history"],
            # ADDED 2026-09-28 with the fix. The seal now covers the
            # reproducibility block, so the documented verification recipe
            # includes it; leaving it out here would be verifying against the old
            # rule and would fail on an untouched document.
            "reproducibility_info": doc["reproducibility_info"],
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# FIXED 2026-09-28, so the strict xfail is gone rather than left to xpass. The
# reproducibility block is inside the sealed payload now and _rehash above verifies
# against the recipe the docstring states.
def test_the_seal_covers_the_reproducibility_block_the_log_returns():
    """The same root cause survives in a second returned field.

    Measured 2026-09-28: erasing ``intervention_history`` now breaks the seal
    (the fix), but replacing ``reproducibility_info['data_hash']`` with a
    different 64-character hash, or ``reproducibility_info['library_version']``
    with '9.9.9', or the reproducibility note with "Any result may be
    substituted freely.", leaves ``_rehash(doc) == doc['content_hash']`` True.
    The block duplicates two fields that ARE sealed, so a reader who takes the
    dataset identity from the reproducibility block is reading an unsealed,
    forgeable copy inside a document the docstring calls "a tamper-evident
    record of the test execution".
    """
    from vfairness.operations.reporting.compliance import compute_signed_test_log

    log = compute_signed_test_log(
        {
            "metrics": [
                {"name": "demographic_parity", "value": 0.03, "threshold": 0.1, "passed": True}
            ],
            "overall_pass": True,
            "timestamp": "2024-01-15T09:00:00+00:00",
        },
        "d" * 64,
        "0.1.0",
        intervention_history=[{"intervention": "reweighting"}],
    )
    assert _rehash(log) == log["content_hash"], "the documented verification must pass untouched"

    forged = copy.deepcopy(log)
    forged["reproducibility_info"]["data_hash"] = "0" * 64
    assert _rehash(forged) != log["content_hash"], (
        "reproducibility_info.data_hash was replaced and the seal still verifies; "
        f"top-level data_hash={forged['data_hash'][:8]}, "
        f"reproducibility_info.data_hash={forged['reproducibility_info']['data_hash'][:8]}"
    )


def test_a_declared_timestamp_that_is_not_a_string_is_not_reported_as_undeclared():
    """CLOSED 2026-09-29. The source field has three states, not two.

    THE FIX, in ``operations/reporting/compliance.py`` ``compute_signed_test_log``:
    ``isinstance(declared, str)`` sent every non-string declaration to the else
    branch, which reports 'log_build_time', documented as "no timestamp was
    declared". Now a ``datetime`` or ``date`` is READ through ``.isoformat()``
    (guarded, so a broken isoformat cannot raise inside a report builder), and a
    declaration this function genuinely cannot read gets its own third state
    ``declared_but_unreadable`` plus a warning naming the refused value. An epoch
    NUMBER lands there deliberately rather than being converted: seconds and
    milliseconds cannot be told apart from the value, and guessing the unit would
    seal a fabricated date into a provenance record.

    RE-MEASURED 2026-09-29 by execution, every input shape:

      absent / None / blank -> log_build_time,          0 warnings
      ISO string            -> declared_by_caller,      '2024-01-15T09:00:00+00:00'
      datetime(2024,1,15,9) -> declared_by_caller,      '2024-01-15T09:00:00'
      date(2024,1,15)       -> declared_by_caller,      '2024-01-15'
      epoch int 1705309200  -> declared_but_unreadable, 1 warning
      ['2024']              -> declared_but_unreadable, 1 warning

    The first line is the over-correction control: a caller that declared nothing
    must still read 'log_build_time' and must NOT be accused of an unreadable
    declaration. It is asserted below.

    SABOTAGE 2026-09-29: restoring the single ``isinstance(declared, str)`` guard
    turns this test RED with "a declared datetime was reported as undeclared:
    test_timestamp=2026-09-29T07:08:32.078764+00:00", ``assert 'log_build_time'
    != 'log_build_time'``. Source restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    ``test_timestamp_source`` states a falsehood for a datetime declaration.

    Measured 2026-09-28: ``test_results['timestamp'] = datetime(2024, 1, 15, 9,
    0, 0)`` (and an epoch int) seals ``test_timestamp`` = the build time with
    ``test_timestamp_source`` = 'log_build_time', whose documented meaning is
    "no timestamp was declared". The caller declared one. The guard is
    ``isinstance(declared, str)``.
    """
    import datetime as _dt
    import warnings as _warnings

    from vfairness.operations.reporting.compliance import compute_signed_test_log

    log = compute_signed_test_log(
        {"metrics": [], "overall_pass": None, "timestamp": _dt.datetime(2024, 1, 15, 9, 0, 0)},
        "d" * 64,
        "0.1.0",
    )
    assert log["test_timestamp_source"] != "log_build_time", (
        f"a declared datetime was reported as undeclared: test_timestamp={log['test_timestamp']}"
    )
    # The declaration is READ, not merely acknowledged.
    assert log["test_timestamp_source"] == "declared_by_caller"
    assert log["test_timestamp"] == "2024-01-15T09:00:00"
    assert log["log_built_at"] != log["test_timestamp"]

    # An epoch number is a declaration this function cannot read. The third
    # state, with the value named in a warning, never folded into either of the
    # other two.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        epoch = compute_signed_test_log(
            {"metrics": [], "overall_pass": None, "timestamp": 1705309200}, "d" * 64, "0.1.0"
        )
    assert epoch["test_timestamp_source"] == "declared_but_unreadable"
    assert any("declared_but_unreadable" in str(c.message) for c in caught), [
        str(c.message) for c in caught
    ]
    assert any("1705309200" in str(c.message) for c in caught)

    # OVER-CORRECTION CONTROL: a caller that declared NOTHING still reads
    # 'log_build_time' and is not accused of an unreadable declaration.
    with _warnings.catch_warnings(record=True) as quiet:
        _warnings.simplefilter("always")
        none_declared = compute_signed_test_log(
            {"metrics": [], "overall_pass": None}, "d" * 64, "0.1.0"
        )
    assert none_declared["test_timestamp_source"] == "log_build_time"
    assert none_declared["test_timestamp"] == none_declared["log_built_at"]
    assert len(quiet) == 0, [str(c.message) for c in quiet]

    # And an ISO string is still read verbatim.
    iso = compute_signed_test_log(
        {"metrics": [], "overall_pass": None, "timestamp": "2024-01-15T09:00:00+00:00"},
        "d" * 64,
        "0.1.0",
    )
    assert iso["test_timestamp_source"] == "declared_by_caller"
    assert iso["test_timestamp"] == "2024-01-15T09:00:00+00:00"

    # The new state is INSIDE the seal, so it cannot be edited after the fact.
    forged = copy.deepcopy(epoch)
    forged["test_timestamp_source"] = "declared_by_caller"
    assert _rehash(forged) != epoch["content_hash"]


# ===========================================================================
# 5. correlation.analyze_intersectional_correlations: coverage 'complete' is
#    still published over ZERO measured features.
# ===========================================================================


def test_intersectional_coverage_is_not_complete_when_nothing_was_measurable():
    """CLOSED 2026-09-29. No numeric feature is a could-not-check, not a pass.

    THE FIX, in ``preprocessing/feature_engineering/correlation.py``
    ``analyze_intersectional_correlations``: ``not numeric_features`` now decides
    coverage alongside ``not valid_groups``, because it is the same kind of fact
    (nothing downstream can measure anything), and a new
    ``features_not_numeric`` key records every requested feature column that
    cannot be read as a number, with its dtype. That key is what
    ``features_not_assessed`` and ``features_not_compared`` could never carry:
    a non-numeric feature never reaches either loop, so both lists were empty
    for want of anything to record and the coverage rule read as satisfied.
    A warning says it out loud, since the reassuring half of the old two-state
    answer was total silence. The group-level warning's condition was rewritten
    as the equivalent ``groups_not_assessed or not valid_groups`` so it does not
    fire here and print "0 were NOT examined".

    RE-MEASURED 2026-09-29 by execution on the frame below:

      feature_columns=['city'] -> coverage 'not_assessed', bgd {},
                                  features_not_numeric ['city'], 1 warning
      feature_columns=[]       -> coverage 'not_assessed', 1 warning
      feature_columns=['x']    -> coverage 'complete', bgd ['x'], 0 warnings
      ['x', 'city']            -> coverage 'complete', bgd ['x'],
                                  features_not_numeric ['city'], 0 warnings

    The last two lines are the over-correction control, asserted below: a frame
    that WAS measurable still earns 'complete' with its real comparison, so this
    is not a blanket downgrade. A non-numeric feature beside a numeric one is
    listed rather than counted against coverage, because the analysis never
    claimed to measure it.

    SABOTAGE 2026-09-29: dropping ``or not numeric_features`` from the coverage
    decision turns this test RED with "coverage 'complete' over
    between_group_differences {} with features_not_compared []". Source restored,
    ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    A frame with no numeric feature is 'complete' with an empty comparison.

    Measured 2026-09-28 on 80 rows across two examined groups where the only
    requested feature is categorical: coverage 'complete',
    between_group_differences {}, features_not_assessed [],
    features_not_compared [], zero warnings. Identical for
    ``feature_columns=[]`` and for a frame whose non-protected columns are all
    categorical. The fix moved the coverage decision after both loops, but with
    no numeric feature the loops record nothing, so the docstring's definition
    ("every numeric feature measured in each of them and compared across them")
    is satisfied VACUOUSLY, which is the reassuring half of a two-state answer.
    """
    from vfairness.preprocessing.feature_engineering.correlation import (
        analyze_intersectional_correlations,
    )

    n = 40
    frame = pd.DataFrame(
        {
            "gender": ["m"] * n + ["f"] * n,
            "city": ["zurich"] * (2 * n),
            "x": np.arange(2 * n, dtype=float),
        }
    )
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = analyze_intersectional_correlations(
            frame, protected_attributes=["gender"], feature_columns=["city"]
        )
    assert result["coverage"] != "complete", (
        f"coverage {result['coverage']!r} over between_group_differences "
        f"{result['between_group_differences']} with features_not_compared "
        f"{result['features_not_compared']}"
    )
    assert result["coverage"] == "not_assessed", result["coverage"]
    assert result["between_group_differences"] == {}
    assert [e["feature"] for e in result["features_not_numeric"]] == ["city"]
    assert "object" in result["features_not_numeric"][0]["dtype"]
    assert any(
        "none of the 1 requested feature column(s) is numeric" in str(c.message) for c in caught
    ), [str(c.message) for c in caught]

    empty = analyze_intersectional_correlations(
        frame, protected_attributes=["gender"], feature_columns=[]
    )
    assert empty["coverage"] == "not_assessed", empty["coverage"]

    # OVER-CORRECTION CONTROL: the same frame WITH its numeric feature still
    # earns 'complete', with the real comparison behind it. A blanket downgrade
    # would pass every line above and fail here.
    with _warnings.catch_warnings(record=True) as quiet:
        _warnings.simplefilter("always")
        healthy = analyze_intersectional_correlations(
            frame, protected_attributes=["gender"], feature_columns=["x"]
        )
    assert healthy["coverage"] == "complete", healthy["coverage"]
    assert list(healthy["between_group_differences"]) == ["x"]
    assert healthy["between_group_differences"]["x"]["max_difference"] == 40.0
    assert healthy["features_not_numeric"] == []
    assert len(quiet) == 0, [str(c.message) for c in quiet]

    # A non-numeric feature BESIDE a numeric one is listed, not counted against
    # coverage: this analysis never claimed to measure it.
    mixed = analyze_intersectional_correlations(
        frame, protected_attributes=["gender"], feature_columns=["x", "city"]
    )
    assert mixed["coverage"] == "complete", mixed["coverage"]
    assert [e["feature"] for e in mixed["features_not_numeric"]] == ["city"]


# ===========================================================================
# 6. detector.BiasAuditReport.to_svg: the 'unrecorded' canvas withholds the
#    headline and still publishes four measured-looking module tiles, and its
#    reason sentence is untrue about the report it describes.
# ===========================================================================


def _unrecorded_report():
    from vfairness.preprocessing.bias_detection.detector import AUDIT_MODULES, BiasAuditReport

    return BiasAuditReport(
        timestamp="2026-09-28T00:00:00",
        dataset_info={"n_rows": 500},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
        metadata={},
        modules_run=list(AUDIT_MODULES),
    )


def test_the_unrecorded_canvas_withholds_the_four_module_tiles():
    """CLOSED 2026-09-29. The tiles need a RECORDED assessment, not just a run.

    THE FIX, in ``rendering/adapters.py`` ``bias_audit_to_svg``: ``_ran()`` keyed
    the tile score on ``module in ran and not assessed_nothing``, and
    ``assessed_nothing`` listed only "none" and "ran_but_assessed_nothing". Under
    "unrecorded" the report lists all four modules, so all four tiles turned an
    empty finding list into a measured 0.00 under a headline reading NOT
    ASSESSABLE. A correct headline over fabricated tiles is still a fabricated
    measurement: the reader reads both.

    A NEW ``assessment_recorded`` flag decides the tiles. It is False for
    "unrecorded" as well, and it is asked of the report's own
    ``assessment_coverage()`` rather than inferred from the execution word,
    because ``execution_coverage()`` returns "partial" BEFORE it consults the
    assessment half: measured 2026-09-29, ``modules_run=['historical']`` with no
    assessment record rendered one '>0.00<' tile for the same reason, one state
    over. ``assessed_nothing`` is left exactly as it was, because the headline
    sentence names that state specifically.

    RE-MEASURED 2026-09-29 by execution:

      modules_run = all four, no assessment record -> coverage 'unrecorded',
        tiles 0 (was 4)
      modules_run = None                           -> coverage 'unrecorded',
        tiles 0 (unchanged: `ran` was already empty)
      modules_run = ['historical'], no assessment  -> coverage 'partial',
        tiles 0 (was 1)

    The over-correction control is the whole existing suite for this canvas: a
    real ``full_audit`` report records both halves, so its tiles are untouched.
    885 tests across test_bias_audit_svg_coverage.py, test_bias_audit_ran_nothing.py,
    test_explain_*.py, test_desc_*.py, test_assessability_chain.py and
    test_rendering_assessable.py pass unchanged, and this file asserts a measured
    tile below.

    This also closes the RESIDUAL that
    ``tests/test_bgl4_preprocessing_2.py::test_the_canvas_no_longer_bands_minimal_without_a_could_not_check_anywhere``
    recorded and handed back as needs_another_batch ("One word added to that
    tuple closes it"), whose ``assert svg.count(">0.00<") == 4`` is inverted to
    ``== 0`` in the same commit.

    SABOTAGE 2026-09-29: restoring ``return module in ran and not
    assessed_nothing`` turns this test RED with "4 measured-looking module
    tiles". Source restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    The residual the batch recorded, measured: four 0.00 tiles survive.

    ``execution_coverage()`` is 'unrecorded' and the headline reads NOT
    ASSESSABLE, but ``rendering/adapters.py`` keys its tile suppression on
    ``coverage in ("none", "ran_but_assessed_nothing")``, so the canvas still
    carries four ``>0.00<`` tiles. The batch's own pin for the 2-row report
    asserts ZERO such tiles, so by its own standard these four are fabricated
    measurements on a canvas that assessed nothing.
    """
    svg = _unrecorded_report().to_svg()
    assert svg.count(">0.00<") == 0, f"{svg.count('>0.00<')} measured-looking module tiles"

    # The same fact one state over: a PARTIAL run with no assessment record.
    from vfairness.preprocessing.bias_detection.detector import AUDIT_MODULES, BiasAuditReport

    partial = BiasAuditReport(
        timestamp="2026-09-28T00:00:00",
        dataset_info={"n_rows": 500},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
        metadata={},
        modules_run=["historical"],
    )
    assert partial.execution_coverage() == "partial"
    assert partial.to_svg().count(">0.00<") == 0, partial.to_svg().count(">0.00<")

    # OVER-CORRECTION CONTROL: a report that DOES record an assessment still
    # renders its measured tiles, so this is not a blanket withholding. A 0.00
    # tile is a measurement when the record establishes one.
    measured = BiasAuditReport(
        timestamp="2026-09-28T00:00:00",
        dataset_info={"n_rows": 500},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
        metadata={},
        modules_run=list(AUDIT_MODULES),
        attribute_assessed={"gender": True},
    )
    assert measured.execution_coverage() == "complete"
    assert measured.to_svg().count(">0.00<") == 4, measured.to_svg().count(">0.00<")


def test_the_unrecorded_canvas_does_not_claim_the_modules_are_unrecorded():
    """CLOSED 2026-09-29. The canvas prints the true one of the two silences.

    THE FIX, in ``rendering/adapters.py``: "unrecorded" has TWO routes, which
    ``detector.empty_is_not_a_measurement`` already treats as different silences
    in its own comment ("Saying 'does not record which modules executed' about a
    report that lists all four of them would be false"). The renderer had one
    ``else`` branch for both. A new ``elif modules_recorded`` prints the true
    sentence for a report that DOES list its modules, and the old sentence stays
    for a report whose ``modules_run`` is genuinely absent.

    RE-MEASURED 2026-09-29 by execution, reading the rendered canvas text:

      modules_run = all four -> "every audit module executed and this report does
        not record whether any requested protected attribute was assessed, so
        this canvas cannot tell a dataset every module cleared from an audit that
        had nothing to look at."
      modules_run = None     -> "no module of this audit returned a finding, and
        this report does not record which modules executed, ..." (unchanged, and
        asserted below as the over-correction control: the old sentence must
        still appear where it IS true)

    SABOTAGE 2026-09-29: deleting the ``elif modules_recorded`` branch turns this
    test RED, the assertion printing the visible canvas text containing "does not
    record which modules executed". Source restored, ``diff -q`` clean.

    THE ORIGINAL RECORD follows.

    The reason printed on the canvas contradicts the report.

    Measured 2026-09-28: the report records ``modules_run`` = all four audit
    modules, and the canvas explanation reads "no module of this audit returned
    a finding, and this report does not record which modules executed". That
    sentence is the ``else`` branch of ``adapters.py`` and it is false for this
    report. ``empty_is_not_a_measurement()`` was corrected to say the true
    thing ("every audit module executed and this report does not record whether
    any requested protected attribute was assessed"); the surface a reader
    looks at was not.
    """
    svg = _unrecorded_report().to_svg()
    visible = " ".join(t.strip() for t in re.findall(r">([^<>]+)<", svg) if t.strip())
    assert "does not record which modules executed" not in visible, visible[:400]
    # The TRUE sentence, the one empty_is_not_a_measurement() was corrected to.
    assert "does not record whether any requested protected attribute was assessed" in visible, (
        visible[:400]
    )

    # OVER-CORRECTION CONTROL: the old sentence must still be printed where it IS
    # true, for a report whose modules_run is genuinely absent. A blanket rewrite
    # would pass every line above.
    from vfairness.preprocessing.bias_detection.detector import BiasAuditReport

    no_record = BiasAuditReport(
        timestamp="2026-09-28T00:00:00",
        dataset_info={"n_rows": 500},
        protected_attributes=["gender"],
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
        metadata={},
        modules_run=None,
    )
    assert no_record.execution_coverage() == "unrecorded"
    visible_none = " ".join(
        t.strip() for t in re.findall(r">([^<>]+)<", no_record.to_svg()) if t.strip()
    )
    assert "does not record which modules executed" in visible_none, visible_none[:400]


# ===========================================================================
# 7. groupthink.detect_coalitions: the reason the refusal gives is false for a
#    NEGATIVE threshold, which is also the only half of the refused range the
#    row's stated before-value does not reproduce.
# ===========================================================================


def test_the_coalition_refusal_reason_is_true_of_the_threshold_it_refuses():
    """CLOSED 2026-09-29. One reason per half, each true of the half it explains.

    THE FIX, in ``multi_agent/groupthink.py`` ``detect_coalitions``: the REFUSAL
    is kept (two existing pins in test_bgl4_agents_multi_1.py and
    test_bgl5_agents_multi.py require -0.5 to raise, matching "finite agreement
    score in [0, 1]", and the BGL-5 note explains why the guard exists), and only
    the REASON is corrected. Above the range and for a non-finite value the
    message keeps "every agent would come back as its own singleton", which is
    true there. Below zero it now says the true thing instead: agreement never
    falls below -1.0, so a negative threshold is cleared by EVERY measured pair
    and the result would be one coalition holding every agent, a grouping the
    data could not have changed. The docstring's "[0, 1]" claim about the MATRIX
    is corrected to [-1, 1] in the same change, because this class's own
    ``_cosine_similarity`` returns -1.0 for opposing vectors; the THRESHOLD stays
    required in [0, 1], since a coalition means positive agreement.

    This is a defect in the MESSAGE, not in the guard. Executed 2026-09-29 across
    the whole refused range, comparing each message against what the membership
    test would actually do:

      1.5   fires=False  singleton claim=True   one-coalition claim=False
      nan   fires=False  singleton claim=True   one-coalition claim=False
      inf   fires=False  singleton claim=True   one-coalition claim=False
      -0.5  fires=True   singleton claim=False  one-coalition claim=True
      -1.0  fires=True   singleton claim=False  one-coalition claim=True

    Over-correction control, asserted below: every in-scale threshold still
    groups this matrix exactly as measured, 0.9 -> [{0, 1}], 0.0 -> [{0, 1}],
    1.0 -> [{0}, {1}].

    SABOTAGE 2026-09-29: collapsing the two halves back to the single "its own
    singleton" reason turns this test RED on
    ``assert "its own singleton" not in message``. Source restored, ``diff -q``
    clean.

    THE ORIGINAL RECORD follows.

    "every agent would come back as its own singleton" is false at -0.5.

    Measured 2026-09-28 by running the pre-fix comparison on the row's own
    matrix ``[[1.0, 0.95], [0.95, 1.0]]``: at 1.5, nan and inf the membership
    test fires for no pair and the BFS returns ``[{0}, {1}]`` (the row is
    right); at -0.5 and -1.0 it fires for EVERY pair and the BFS returns
    ``[{0, 1}]``, one coalition, not two singletons. The refusal message states
    the singleton outcome as the reason for refusing, so it asserts something
    untrue of half the range it refuses, and the sibling unit fixed in the same
    batch (``get_high_correlations``) deliberately does NOT refuse a negative
    threshold: it returns the real numbers with a warning that the list is not
    a selection. The agreement scale is also not [0, 1]: this class's own
    ``_cosine_similarity`` returns -1.0 for opposing vectors, measured as
    ``echo_chamber_score -1.0``.
    """
    from vfairness.multi_agent.groupthink import GroupthinkDetector

    matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
    detector = GroupthinkDetector()

    # BELOW the scale. The membership test WOULD fire for the one measured pair,
    # so the singleton reason is false here and must not be given.
    for below in (-0.5, -1.0):
        with pytest.raises(ValueError) as excinfo:
            detector.detect_coalitions(matrix, threshold=below)
        message = str(excinfo.value)
        assert "finite agreement score in [0, 1]" in message, message
        fires = bool(matrix[0, 1] >= below)
        assert fires, below
        assert "its own singleton" not in message, message
        assert "one coalition holding every agent" in message, message

    # ABOVE the scale, and non-finite. No pair can clear it, so the singleton
    # reason IS the true one and must be kept. Asserting both directions in the
    # same test, because a message rewritten for every threshold alike would be
    # wrong in the other half.
    for outside in (1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError) as excinfo:
            detector.detect_coalitions(matrix, threshold=outside)
        message = str(excinfo.value)
        assert "its own singleton" in message, message
        assert "one coalition holding every agent" not in message, message

    # OVER-CORRECTION CONTROL: the guard refuses only what it should, and every
    # in-scale threshold still groups this matrix by its real numbers.
    assert detector.detect_coalitions(matrix) == [{0, 1}]
    assert detector.detect_coalitions(matrix, threshold=0.0) == [{0, 1}]
    assert detector.detect_coalitions(matrix, threshold=1.0) == [{0}, {1}]
