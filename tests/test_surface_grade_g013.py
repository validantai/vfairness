"""Surface grading, batch g013: vfairness.operations.pulse.regulatory.

Every pin here fires on a value nobody measured being published as if it
were a measurement, and every pin is paired with a HEALTHY-DATA CONTROL
asserting the real number is still measured exactly, so a fix that simply
refuses everything fails just as loudly as the original defect.

Five defects were proved by execution and closed in this file's subject:

* ``base_rates_differ`` returned False (which recommend.py renders as "no
  material base-rate difference was confirmed on this data") for data whose
  real spread was 1.00, whenever no group cleared the undocumented 10-row
  floor, whenever every label was missing, and whenever no protected column
  reached it at all. It also let one missing label poison a whole group's
  mean to NaN, and a NaN spread compares False against every threshold.
* ``build_art10`` wrote "Column-role screening and data-quality checks ran
  before any metric was computed" into an EU AI Act Art. 10 draft from the
  orchestrator's all-empty COLLAPSE fallback, byte-identical to a screen
  that ran and flagged nothing.
* ``build_ll144`` returned applicable=True with an empty impact-ratio table
  and the full methodology notes when nothing had been screened.
* ``build_reference_groups`` read an ABSENT referenceOverrideHonored key as
  "not honored" and stated that the requested group "was not found among
  this attribute's groups" about a row using that very group.
* ``uncovered_admissibility`` reported four counters of 0 for a run in which
  no column was classified at all.

Plus ``build_targets`` dropping a real np.float32 ratio on the floor
(isinstance(v, (int, float)) rejects it) while accepting a bool.
"""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.pulse.regulatory import (
    _add_months,
    base_rates_differ,
    build_art10,
    build_ll144,
    build_recheck,
    build_reference_groups,
    build_targets,
    clamp_target_ratio,
    empty_exports,
    historical_context,
    iso24027_crosswalk,
    legal_admissibility,
    parse_pulse_options,
    sanitize_suite,
    uncovered_admissibility,
)

# ── fixtures ────────────────────────────────────────────────────────────────


def _two_group_frame(n_per_group: int) -> pd.DataFrame:
    return pd.DataFrame({"grp": ["a"] * n_per_group + ["b"] * n_per_group})


def _split_labels(n_per_group: int, rate_a: float, rate_b: float) -> np.ndarray:
    """Exact base rates, set by construction and not sampled."""
    a_pos = int(round(n_per_group * rate_a))
    b_pos = int(round(n_per_group * rate_b))
    return np.concatenate(
        [
            np.ones(a_pos),
            np.zeros(n_per_group - a_pos),
            np.ones(b_pos),
            np.zeros(n_per_group - b_pos),
        ]
    )


def _call(fn, *a, **k):
    """Run with warnings ON and hand back (value, [warning texts])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*a, **k)
    return value, [str(w.message) for w in caught]


# ── base_rates_differ: refusal pins ─────────────────────────────────────────


def test_base_rates_differ_refuses_when_no_group_reaches_the_size_floor():
    """REFUSAL PIN. The biggest possible spread, reported as "no difference".

    Five rows per group, base rates 1.00 and 0.00: the largest divergence
    that can exist. Every group is under the function's own 10-labelled-row
    floor, so nothing was compared. The answer must be None (not measured),
    never False, because False is rendered downstream as "no material
    base-rate difference was confirmed on this data".
    """
    work = _two_group_frame(5)
    y_true = _split_labels(5, 1.0, 0.0)

    verdict, msgs = _call(base_rates_differ, work, ["grp"], y_true)

    assert verdict is None
    assert any("could not be compared" in m for m in msgs)
    assert any("10 labelled rows" in m for m in msgs)


def test_base_rates_differ_refuses_when_every_label_is_missing():
    """REFUSAL PIN. An all-NaN ground-truth column answers nothing.

    NaN >= threshold is False, so the old code walked straight past an
    entirely unlabelled column and reported "no difference".
    """
    work = _two_group_frame(60)

    verdict, msgs = _call(base_rates_differ, work, ["grp"], np.full(120, np.nan))

    assert verdict is None
    assert any("could not be compared" in m for m in msgs)


@pytest.mark.parametrize(
    "usable,label",
    [([], "no protected attribute was passed"), (["absent"], "the column is not in the frame")],
)
def test_base_rates_differ_refuses_when_no_column_was_read(usable, label):
    """REFUSAL PIN. No column read is not the same fact as no difference."""
    work = _two_group_frame(60)
    y_true = _split_labels(60, 0.9, 0.1)

    verdict, _ = _call(base_rates_differ, work, usable, y_true)

    assert verdict is None, label


# ── base_rates_differ: over-correction controls ─────────────────────────────


def test_base_rates_differ_still_finds_a_real_divergence():
    """CONTROL. Exact rates, computed independently: 0.80 and 0.20.

    Spread 0.60, far above the 0.05 threshold. A refusal here would delete
    the impossibility acknowledgment that this flag exists to trigger.
    """
    work = _two_group_frame(50)
    y_true = _split_labels(50, 0.8, 0.2)
    assert y_true[:50].mean() == pytest.approx(0.80)
    assert y_true[50:].mean() == pytest.approx(0.20)

    verdict, msgs = _call(base_rates_differ, work, ["grp"], y_true)

    assert verdict is True
    assert msgs == []


def test_base_rates_differ_still_reports_false_for_measured_equality():
    """CONTROL. False must survive where it is TRUE.

    Both groups measured at exactly 0.50 over 50 labelled rows each: the
    spread is 0.00, which is a measurement, not a gap in the evidence.
    """
    work = _two_group_frame(50)
    y_true = _split_labels(50, 0.5, 0.5)

    verdict, msgs = _call(base_rates_differ, work, ["grp"], y_true)

    assert verdict is False
    assert msgs == []


def test_base_rates_differ_measures_over_labelled_rows_only():
    """CONTROL. One missing label must not erase the group's base rate.

    Group "a" is 0.80 over its 49 labelled rows and "b" is 0.20 over its
    49: a mean() over the raw slice returns NaN for both, and NaN loses
    every comparison. The real spread is 0.60 and must still be found.
    """
    work = _two_group_frame(50)
    y_true = _split_labels(50, 0.8, 0.2)
    y_true[0] = np.nan  # a positive row in "a"
    y_true[50] = np.nan  # a positive row in "b"

    verdict, _ = _call(base_rates_differ, work, ["grp"], y_true)

    assert verdict is True
    assert np.isnan(y_true[:50].mean()), "the fixture must really poison the raw mean"


def test_base_rates_differ_threshold_is_honoured_in_both_directions():
    """CONTROL. The comparison itself still discriminates.

    0.54 vs 0.50 is a measured 0.04 spread: under the 0.05 threshold, so
    False; over a 0.02 threshold, so True. Same data, opposite answers.
    """
    work = _two_group_frame(50)
    y_true = _split_labels(50, 0.54, 0.50)

    assert base_rates_differ(work, ["grp"], y_true) is False
    assert base_rates_differ(work, ["grp"], y_true, threshold=0.02) is True


# ── build_art10: the unrun column-role screen ───────────────────────────────

_ART10_RUN = dict(
    jurisdiction="Germany",
    artifact_label="loans-2026Q1.csv",
    artifact_hash="abc123def",
    rows=1000,
    columns=12,
    data_quality={"headline": "3% missing in income"},
    verdict={"headline": "Disparity found.", "summary": "Selection rates differ by gender."},
    bias_findings=[{"plain": "Women selected at 0.42x the male rate."}],
    interventions=[{"title": "Reweighing", "how": "resample training data"}],
)

# What orchestrator.py hands build_art10 when the schema_roles stage collapses.
_COLLAPSED_SCHEMA = {
    "roles": [],
    "pii_leakage": [],
    "oracle_columns": [],
    "model_output": [],
    "proxy_candidates": [],
    "mismatches": [],
    "refuse": False,
    "refuse_reason": "",
}
# What it hands build_art10 when the screen RAN and flagged nothing: the
# per-column roles are the evidence the screen executed.
_CLEAN_SCHEMA = {
    "roles": [{"column": "income", "role": "job_relevant"}],
    "pii_leakage": [],
    "oracle_columns": [],
    "model_output": [],
}


def _gov(block) -> str:
    return [s for s in block["sections"] if s["id"] == "data_governance"][0]["body"]


def test_art10_does_not_claim_a_column_role_screen_that_never_ran():
    """REFUSAL PIN. A collapsed screen must not be drafted as a clean one.

    The governance section is regulator-facing prose. Built from the
    collapse fallback it asserted that column-role screening ran, and the
    body was byte-identical to a run where the screen genuinely found
    nothing. Three states, and this is the third.
    """
    blk = build_art10(domain="hiring", schema=_COLLAPSED_SCHEMA, **_ART10_RUN)
    body = _gov(blk)

    assert blk["columnRoleScreen"] == "unavailable"
    assert "Column-role screening and data-quality checks ran" not in body
    assert "returned no result" in body
    assert "unchecked gap, not a clean screen" in body
    # The data-quality half DID run and must survive the disclosure.
    assert "3% missing in income" in body


def test_art10_collapsed_and_clean_screens_are_not_the_same_document():
    """REFUSAL PIN. The two states must be distinguishable at the surface."""
    collapsed = build_art10(domain="hiring", schema=_COLLAPSED_SCHEMA, **_ART10_RUN)
    clean = build_art10(domain="hiring", schema=_CLEAN_SCHEMA, **_ART10_RUN)

    assert _gov(collapsed) != _gov(clean)
    assert collapsed["columnRoleScreen"] == "unavailable"
    assert clean["columnRoleScreen"] == "reported"


def test_art10_control_a_screen_that_ran_and_flagged_nothing_still_says_so():
    """OVER-CORRECTION CONTROL. A clean screen keeps its clean sentence.

    A fix that reported "unavailable" whenever the flagged lists are empty
    would delete a real finding: the screen ran, and nothing was flagged.
    """
    blk = build_art10(
        domain="hiring",
        schema={**_CLEAN_SCHEMA, "roles": [{"column": "x"}]},
        **{**_ART10_RUN, "data_quality": {}},
    )
    body = _gov(blk)

    assert blk["columnRoleScreen"] == "reported"
    assert body.startswith("Column-role screening and data-quality checks ran")
    assert "No PII or oracle columns were flagged by the screen." in body


def test_art10_control_measured_screen_output_is_unchanged():
    """OVER-CORRECTION CONTROL. Detected columns are still named, in place."""
    blk = build_art10(
        domain="hiring",
        schema={"pii_leakage": ["email"], "oracle_columns": [], "model_output": ["pred"]},
        **_ART10_RUN,
    )
    body = _gov(blk)

    assert blk["columnRoleScreen"] == "reported"
    assert "identity/PII columns detected and excluded" in body
    assert "email" in body and "pred" in body
    assert "3% missing in income" in body
    # And the rest of the draft is untouched.
    assert [s["id"] for s in blk["sections"]] == [
        "data_sources",
        "data_governance",
        "bias_examination",
        "bias_mitigation",
    ]
    assert "1000 rows and 12 columns" in blk["sections"][0]["body"]


# ── build_ll144: an empty table is not a clean table ────────────────────────

_LL144_ARGS = dict(domain="hiring", jurisdiction="New York City", score=None)


def test_ll144_empty_table_says_nothing_was_screened():
    """REFUSAL PIN. applicable=True with no rows must disclose why.

    With no assessable protected attribute the table is empty while the
    notes recite the DCWP impact-ratio methodology, which reads as a
    screen that ran and found nothing to report.
    """
    df = pd.DataFrame({"gender": ["m"] * 5 + ["f"] * 5})
    y_pred = np.array([1.0, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    out = build_ll144(df, [], y_pred, df, [], **_LL144_ARGS)

    assert out["applicable"] is True
    assert out["impactRatios"] == []
    assert out["impactRatiosComputed"] is False
    assert any("no assessable protected attribute" in n for n in out["notes"])
    assert any("unrun screen, not a clean one" in n for n in out["notes"])


def test_ll144_empty_table_names_an_unscored_run():
    """REFUSAL PIN. No scored row is its own reason, and is named."""
    df = pd.DataFrame({"gender": ["m"] * 5 + ["f"] * 5})

    out = build_ll144(df, ["gender"], np.array([]), df, ["gender"], **_LL144_ARGS)

    assert out["impactRatiosComputed"] is False
    assert any("no scored decision row" in n for n in out["notes"])


def test_ll144_control_a_real_table_is_still_measured_exactly():
    """OVER-CORRECTION CONTROL. Exact impact ratios, computed by hand.

    m: 4 of 5 selected = 0.80. f: 1 of 5 = 0.20. Impact ratio 0.20/0.80 =
    0.25 for f and 1.00 for the best group.
    """
    df = pd.DataFrame({"gender": ["m"] * 5 + ["f"] * 5})
    y_pred = np.array([1.0, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    out = build_ll144(df, ["gender"], y_pred, df, ["gender"], **_LL144_ARGS)

    assert out["impactRatiosComputed"] is True
    rows = {r["group"]: r for r in out["impactRatios"]}
    assert rows["m"]["selectionRate"] == pytest.approx(0.80)
    assert rows["f"]["selectionRate"] == pytest.approx(0.20)
    assert rows["m"]["impactRatio"] == pytest.approx(1.00)
    assert rows["f"]["impactRatio"] == pytest.approx(0.25)
    assert not any("unrun screen" in n for n in out["notes"])


def test_ll144_control_inapplicable_is_a_third_state():
    """OVER-CORRECTION CONTROL. Not applicable is neither ran nor failed."""
    df = pd.DataFrame({"gender": ["m", "f"]})

    out = build_ll144(
        df, ["gender"], np.array([1.0, 0.0]), df, ["gender"], "lending", "New York City"
    )

    assert out["applicable"] is False
    assert out["impactRatiosComputed"] is None


# ── build_reference_groups: an unrecorded verdict is not a refusal ──────────


def test_reference_groups_absent_honored_flag_is_not_a_denial():
    """REFUSAL PIN. The engine recorded no verdict, so neither may the note.

    The old branch treated a missing referenceOverrideHonored key as False
    and published '"male" was requested but was not found among this
    attribute's groups' for a row whose reference group IS "male": a claim
    about the data from a check that never ran.
    """
    rows = [{"attribute": "gender", "assessable": True, "referenceGroup": "male"}]

    out = build_reference_groups({"gender": "male"}, rows)["gender"]

    assert out["honored"] is None
    assert "not found" not in out["note"]
    assert "recorded no verdict" in out["note"]
    assert out["used"] == "male"


def test_reference_groups_control_a_real_denial_still_reads_as_one():
    """OVER-CORRECTION CONTROL. An explicit False keeps the fallback note."""
    rows = [
        {
            "attribute": "gender",
            "assessable": True,
            "referenceGroup": "male",
            "referenceOverrideHonored": False,
        }
    ]

    out = build_reference_groups({"gender": "zzz"}, rows)["gender"]

    assert out["honored"] is False
    assert "was not found among this" in out["note"]
    assert 'default reference "male"' in out["note"]


def test_reference_groups_control_an_honoured_override_still_reads_as_one():
    """OVER-CORRECTION CONTROL. True keeps the honored note and the group."""
    rows = [
        {
            "attribute": "gender",
            "assessable": True,
            "referenceGroup": "male",
            "referenceOverrideHonored": True,
        }
    ]

    out = build_reference_groups({"gender": "male"}, rows)["gender"]

    assert out["honored"] is True
    assert "was used as the comparison baseline" in out["note"]


def test_reference_groups_control_unassessable_attribute_unchanged():
    """OVER-CORRECTION CONTROL. A silently-ignored request still surfaces."""
    out = build_reference_groups({"x": "a"}, [{"attribute": "x", "assessable": False}])["x"]

    assert out["used"] is None
    assert "was not assessable on this run" in out["note"]


# ── uncovered_admissibility: counters nobody counted ────────────────────────


def test_uncovered_admissibility_counters_are_null_not_zero():
    """REFUSAL PIN. No column was classified, so no counter has a value.

    "forbidden: 0" reads as "we checked every column and none is
    forbidden". This block is returned precisely when nothing was checked.
    """
    blk = uncovered_admissibility("Mars", error="legal admissibility stage collapsed")

    assert blk["coverage"] == "uncovered"
    assert set(blk["summary"]) == {
        "forbidden",
        "restricted",
        "monitoringOnly",
        "allowed",
        "unknown",
    }
    assert all(v is None for v in blk["summary"].values())
    assert "null rather than 0" in blk["summaryNote"]
    assert "legal admissibility stage collapsed" in blk["error"]


def test_uncovered_admissibility_control_a_real_classification_still_counts():
    """OVER-CORRECTION CONTROL. The rule pack's own integers are untouched.

    hiring x US maps to the curated us-federal recruitment pack: age is
    restricted (ADEA), gender is monitoring_only (Title VII EEO carve-out).
    Those are real counts and must stay integers.
    """
    real = legal_admissibility(["age", "gender"], "hiring", "US")

    assert real["coverage"] == "covered"
    assert real["summary"]["restricted"] == 1
    assert real["summary"]["monitoringOnly"] == 1
    assert real["summary"]["forbidden"] == 0  # measured zero: the pack was applied
    assert "summaryNote" not in real


# ── build_targets / clamp_target_ratio ──────────────────────────────────────


def test_build_targets_keeps_a_numpy_float32_ratio():
    """REFUSAL PIN (reverse direction). A real measurement must not be dropped.

    isinstance(v, (int, float)) is False for np.float32, so a measured
    0.29 ratio vanished from the table entirely, and an empty table reads
    as "nothing failed the target".
    """
    rows = [{"attribute": "g", "assessable": True, "fourFifthsRatio": np.float32(0.25)}]

    out = build_targets(rows, 0.8)

    assert [r["attribute"] for r in out["perAttribute"]] == ["g"]
    assert out["perAttribute"][0]["passes"] is False
    assert out["perAttribute"][0]["fourFifthsRatio"] == pytest.approx(0.25, abs=1e-6)
    assert out["ratioUnavailable"] == []


def test_build_targets_discloses_an_attribute_with_no_ratio():
    """REFUSAL PIN. An assessed attribute with no ratio must not vanish."""
    rows = [
        {"attribute": "g", "assessable": True, "fourFifthsRatio": 0.9},
        {"attribute": "h", "assessable": True, "fourFifthsRatio": None},
        {"attribute": "i", "assessable": True, "fourFifthsRatio": float("nan")},
    ]

    out = build_targets(rows, 0.8)

    assert [r["attribute"] for r in out["perAttribute"]] == ["g"]
    assert out["ratioUnavailable"] == ["h", "i"]
    assert "absence from the table above is an unrun test, not a pass" in out["note"]


def test_build_targets_refuses_a_bool_as_a_ratio():
    """REFUSAL PIN. float(True) == 1.0 would manufacture a perfect ratio."""
    rows = [{"attribute": "g", "assessable": True, "fourFifthsRatio": True}]

    out = build_targets(rows, 0.8)

    assert out["perAttribute"] == []
    assert out["ratioUnavailable"] == ["g"]
    # And the same coercion is refused on the target itself.
    assert clamp_target_ratio(True) == pytest.approx(0.8)
    assert "could not be parsed" in build_targets([], True)["note"]


def test_build_targets_control_measured_pass_fail_is_unchanged():
    """OVER-CORRECTION CONTROL. Real ratios still pass and fail correctly.

    0.29 < 0.90 fails; 0.95 >= 0.90 passes; the band clamp and its two
    disclosure sentences are untouched.
    """
    rows = [
        {"attribute": "gender", "assessable": True, "fourFifthsRatio": 0.29},
        {"attribute": "race", "assessable": True, "fourFifthsRatio": 0.95},
    ]

    out = build_targets(rows, 0.9)

    assert out["targetRatio"] == pytest.approx(0.9)
    by_attr = {r["attribute"]: r["passes"] for r in out["perAttribute"]}
    assert by_attr == {"gender": False, "race": True}
    assert out["ratioUnavailable"] == []
    assert "unrun test" not in out["note"]
    assert clamp_target_ratio(0.3) == pytest.approx(0.5)
    assert clamp_target_ratio(1.7) == pytest.approx(1.0)
    assert clamp_target_ratio("0.85") == pytest.approx(0.85)
    assert clamp_target_ratio(None) == pytest.approx(0.8)


# ── the items that produce no verdict: pinned as they are ───────────────────


def test_historical_context_is_a_catalog_lookup_with_an_honest_miss():
    """An unknown domain reports available=False, never borrowed history."""
    hit = historical_context("hiring")
    assert hit["available"] is True
    assert hit["domain"] == "hiring"
    assert any("Bertrand and Mullainathan" in c for c in hit["citations"])

    miss = historical_context("asteroid mining")
    assert miss["available"] is False
    assert "asteroid mining" in miss["reason"]
    assert "citations" not in miss and "patterns" not in miss


def test_empty_exports_carries_no_measured_looking_value():
    """The collapse block is null and false throughout, with the reason."""
    blk = empty_exports("The regulatory export stage could not be computed.")

    assert blk["legalContextAsOf"] is None
    assert blk["ll144"]["applicable"] is False
    assert blk["ll144"]["unknownDemographics"] is None
    assert blk["ll144"]["impactRatiosComputed"] is None
    assert blk["art10"]["applicable"] is False
    assert "could not be computed" in blk["ll144"]["reason"]


def test_iso24027_crosswalk_maps_only_what_was_computed():
    """A name-to-name crosswalk: no metric is claimed that did not run."""
    without_truth = {r["ourMetric"] for r in iso24027_crosswalk()}
    assert "equalized_odds_difference" not in without_truth
    with_truth = {r["ourMetric"] for r in iso24027_crosswalk(True, True, "binary")}
    assert {"equalized_odds_difference", "group_expected_calibration_error"} <= with_truth
    assert all("ISO/IEC TR 24027" in r["clause"] for r in iso24027_crosswalk(True, True))


def test_parse_pulse_options_refuses_an_unknown_harm_direction():
    """An unrecognised harm direction is None, not a guessed default."""
    assert parse_pulse_options({"harm_direction": "sideways"})["harm_direction"] is None
    assert parse_pulse_options({"harmDirection": "Punitive"})["harm_direction"] == "punitive"
    assert parse_pulse_options({"reference_group": ["not", "a", "dict"]})["reference_group"] == {}
    assert parse_pulse_options(None)["target_ratio"] is None


def test_sanitize_suite_drops_every_secret_bearing_key():
    """Negative case: the planted secrets must be gone, nested included."""
    cleaned = sanitize_suite(
        {
            "domain": "hiring",
            "auth_token": "sekrit",
            "llm_config": {"AUTH-TOKEN": "sekrit2", "endpoint_url": "http://example.invalid"},
            "rows": [{"api_key": "sekrit3", "keep": 1}],
        }
    )

    assert "auth_token" not in cleaned
    assert "AUTH-TOKEN" not in cleaned["llm_config"]
    assert "api_key" not in cleaned["rows"][0]
    assert "sekrit" not in str(cleaned)
    # The re-runnable parts survive.
    assert cleaned["domain"] == "hiring"
    assert cleaned["llm_config"]["endpoint_url"] == "http://example.invalid"
    assert cleaned["rows"][0]["keep"] == 1


# ── build_recheck: an unrun statute test is not a statute that did not match ─


def test_recheck_unknown_ll144_state_does_not_claim_no_statute_matched():
    """REFUSAL PIN. None means the screen did not run, and the basis says so.

    "No annual audit statute matched this run" is a finding about the law.
    When the regulatory export stage collapses, nobody made it.
    """
    rc = build_recheck({"domain": "hiring"}, None)

    assert "could not be determined" in rc["basis"]
    assert "NOT a finding that none applies" in rc["basis"]
    assert "No annual audit statute matched" not in rc["basis"]


def test_recheck_control_both_measured_states_are_unchanged():
    """OVER-CORRECTION CONTROL. True still means 12 months, False 6.

    Exact windows, computed independently from the returned assessedAt.
    """
    applies = build_recheck({"domain": "hiring"}, True)
    does_not = build_recheck({"domain": "hiring"}, False)

    assessed = datetime.fromisoformat(applies["assessedAt"].replace("Z", "+00:00"))
    assert datetime.fromisoformat(applies["validUntil"].replace("Z", "+00:00")) == _add_months(
        assessed, 12
    )
    assert "Local Law 144" in applies["basis"]

    assessed2 = datetime.fromisoformat(does_not["assessedAt"].replace("Z", "+00:00"))
    assert datetime.fromisoformat(does_not["validUntil"].replace("Z", "+00:00")) == _add_months(
        assessed2, 6
    )
    assert "No annual audit statute matched this run" in does_not["basis"]


def test_legal_admissibility_classifier_failure_returns_the_null_counter_block():
    """REFUSAL PIN. A classifier that raises must not yield zero counts.

    The rule-pack call is wrapped so a Pulse never crashes on it. The
    degraded block it falls back to must carry null counters and the
    error, so "0 forbidden columns" cannot be read off a run in which the
    classifier never answered.
    """
    import vfairness.legal as legal_module

    def _boom(*_a, **_k):
        raise RuntimeError("rule pack unreadable")

    original = legal_module.classify_columns
    legal_module.classify_columns = _boom
    try:
        blk = legal_admissibility(["age", "gender"], "hiring", "US")
    finally:
        legal_module.classify_columns = original

    assert blk["coverage"] == "uncovered"
    assert all(v is None for v in blk["summary"].values())
    assert "rule pack unreadable" in blk["error"]
    # CONTROL: with the real classifier back, the real counts return.
    assert legal_admissibility(["age", "gender"], "hiring", "US")["summary"]["restricted"] == 1


def test_build_targets_lists_an_attribute_that_was_never_assessed():
    """REFUSAL PIN. An unassessable attribute must not vanish from targets.

    A single-group or too-small attribute produces no four-fifths ratio at
    all. Dropping its name left targetRatio 0.80 beside an empty table,
    which reads as "every attribute cleared the target".
    """
    rows = [
        {"attribute": "gender", "assessable": False, "fourFifthsRatio": None},
        {"attribute": "race", "assessable": True, "fourFifthsRatio": 0.95},
    ]

    out = build_targets(rows, 0.9)

    assert [r["attribute"] for r in out["perAttribute"]] == ["race"]
    assert out["ratioUnavailable"] == ["gender"]
    assert "unrun test, not a pass" in out["note"]
