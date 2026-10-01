"""Insufficient-evidence correctness (VB-TEST-6).

An audit-evidence tool must NEVER let a small protected stratum become an
implicit pass. Two behaviours must hold together:

1. Groups below ``min_group_size`` are dropped from the disparity metrics
   (they are too noisy to compare), AND
2. every such dropped group is surfaced in the machine-readable
   ``assessment`` block with an explicit "insufficient_evidence" verdict,
   routed through the shared reliability tiers
   (invalid n<10, underpowered 10..29, caution 30..99, reliable n>=100).

These tests pin both halves so a future refactor cannot silently re-introduce
the drop-without-a-verdict path. They also re-assert the audited
``assessable`` / NOT ASSESSABLE behaviour so this lane does not regress it.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import (
    group_reliability,
    reliability_tier,
)
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)

# ── reliability tiers fire at the documented thresholds ─────────────────────


@pytest.mark.parametrize(
    "n,tier",
    [
        (0, "invalid"),
        (1, "invalid"),
        (9, "invalid"),
        (10, "underpowered"),
        (29, "underpowered"),
        (30, "caution"),
        (99, "caution"),
        (100, "reliable"),
        (250, "reliable"),
    ],
)
def test_reliability_tier_boundaries(n, tier):
    assert reliability_tier(n) == tier


def test_invalid_tier_is_not_interpretable():
    """n<10 groups must be flagged non-interpretable (rate must not be read)."""
    rel = group_reliability({"tiny": 4, "small": 20, "big": 500})
    assert rel["tiny"]["tier"] == "invalid"
    assert rel["tiny"]["interpretable"] is False
    assert rel["small"]["tier"] == "underpowered"
    assert rel["small"]["interpretable"] is True
    assert rel["big"]["tier"] == "reliable"


# ── classification report: excluded groups get an explicit verdict ──────────


def _clf(g, seed=7):
    rng = np.random.default_rng(seed)
    n = len(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return classification_fairness_report(
            rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n), np.asarray(g)
        )


def test_small_group_surfaces_insufficient_evidence_verdict():
    # Two assessable groups plus one below the size gate (n=5 -> invalid).
    g = ["A"] * 100 + ["B"] * 100 + ["C"] * 5
    rep = _clf(g)
    a = rep["assessment"]

    # Still assessable overall (>=2 valid groups), but the small stratum is
    # NOT silently gone: it is an explicit insufficient-evidence record.
    assert a["assessable"] is True
    recs = a["insufficient_evidence_groups"]
    assert [r["group"] for r in recs] == ["C"]
    rec = recs[0]
    assert rec["verdict"] == "insufficient_evidence"
    assert rec["n"] == 5
    assert rec["tier"] == "invalid"
    assert rec["reason"]

    # And the excluded group is discoverable in the assessment block itself,
    # not only in a transient warning or a separate data_info section.
    assert "C" in json.dumps(a)


def test_underpowered_and_invalid_groups_both_surface_with_correct_tiers():
    g = ["A"] * 100 + ["B"] * 100 + ["C"] * 5 + ["D"] * 15
    rep = _clf(g)
    recs = {r["group"]: r for r in rep["assessment"]["insufficient_evidence_groups"]}
    assert set(recs) == {"C", "D"}
    assert recs["C"]["tier"] == "invalid"  # n=5  < 10
    assert recs["D"]["tier"] == "underpowered"  # 10 <= 15 < 30
    for r in recs.values():
        assert r["verdict"] == "insufficient_evidence"


def test_no_insufficient_records_when_all_groups_large_enough():
    # No group is below the default gate -> no false-positive caveat records.
    rep = _clf(["A"] * 120 + ["B"] * 120)
    assert rep["assessment"]["insufficient_evidence_groups"] == []
    assert "insufficient evidence" not in rep["assessment"]["summary"]


def test_groups_at_the_gate_are_included_not_flagged():
    # n=30 == default min_group_size: included in the verdict, so it must NOT
    # appear as insufficient evidence. Its tier is 'caution' but that is a
    # read-with-a-wider-CI note, not an exclusion.
    rep = _clf(["A"] * 100 + ["B"] * 30)
    di = rep["data_info"]
    assert "B" in di["valid_groups"]
    assert "B" not in di["invalid_groups"]
    assert rep["assessment"]["insufficient_evidence_groups"] == []


def test_summary_mentions_excluded_groups():
    g = ["A"] * 100 + ["B"] * 100 + ["C"] * 7
    rep = _clf(g)
    s = rep["assessment"]["summary"]
    assert "insufficient evidence" in s
    assert "C" in s


# ── the key invariant: every dropped group is accounted for in the verdict ──


def test_no_group_is_dropped_without_a_surfaced_verdict():
    """Every group below the gate that GroupManager excludes MUST appear as an
    insufficient-evidence record. No silent drop, no silent NaN."""
    g = ["A"] * 100 + ["B"] * 80 + ["C"] * 3 + ["D"] * 9 + ["E"] * 12 + ["F"] * 29
    rep = _clf(g)
    excluded = set(rep["data_info"]["invalid_groups"])
    surfaced = {r["group"] for r in rep["assessment"]["insufficient_evidence_groups"]}
    # The two sets must match exactly: nothing excluded is missing from the
    # verdict, and nothing is invented that was not excluded.
    assert excluded == surfaced == {"C", "D", "E", "F"}


def test_metric_values_are_never_silent_nan_for_small_groups():
    """The disparity metrics must resolve to finite numbers computed over the
    valid groups, never a silent NaN that a consumer would read as 'no gap'."""
    g = ["A"] * 100 + ["B"] * 100 + ["C"] * 4
    rep = _clf(g)
    for name, value in rep["metrics"].items():
        assert value is not None, name
        assert np.isfinite(float(value)), name


# ── audited assessable / NOT ASSESSABLE behaviour must not regress ──────────


def test_single_valid_group_still_not_assessable():
    # Only one group clears the gate -> disparity is vacuous -> NOT ASSESSABLE.
    g = ["A"] * 100 + ["B"] * 5
    rep = _clf(g)
    a = rep["assessment"]
    assert a["assessable"] is False
    assert "NOT ASSESSABLE" in a["summary"]
    # The excluded group is STILL surfaced, even in the not-assessable case.
    assert [r["group"] for r in a["insufficient_evidence_groups"]] == ["B"]


def test_all_groups_too_small_is_not_assessable_and_all_surfaced():
    g = ["A"] * 8 + ["B"] * 6
    rep = _clf(g)
    a = rep["assessment"]
    assert a["assessable"] is False
    surfaced = {r["group"] for r in a["insufficient_evidence_groups"]}
    assert surfaced == {"A", "B"}


# ── regression report carries the same first-class verdict ──────────────────


def test_regression_report_surfaces_insufficient_evidence():
    rng = np.random.default_rng(3)
    g = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 8)
    n = len(g)
    y_true = rng.normal(0.0, 1.0, n)
    y_pred = y_true + rng.normal(0.0, 0.1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = regression_fairness_report(y_true, y_pred, g)
    a = rep["assessment"]
    recs = a["insufficient_evidence_groups"]
    assert [r["group"] for r in recs] == ["C"]
    assert recs[0]["verdict"] == "insufficient_evidence"
    assert recs[0]["tier"] == "invalid"  # n=8 < 10
    assert "insufficient evidence" in a["summary"]


def test_regression_no_insufficient_records_when_groups_large():
    rng = np.random.default_rng(4)
    g = np.array(["A"] * 120 + ["B"] * 120)
    n = len(g)
    y_true = rng.normal(0.0, 1.0, n)
    y_pred = y_true + rng.normal(0.0, 0.1, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = regression_fairness_report(y_true, y_pred, g)
    assert rep["assessment"]["insufficient_evidence_groups"] == []


# ── the drop is also audible (named warning), not only structured ───────────


def test_partial_drop_emits_named_warning():
    """The audited partial-drop warning must still fire and name the group, so
    the exclusion is visible both as a warning and as structured evidence."""
    rng = np.random.default_rng(5)
    g = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 5)
    n = len(g)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        classification_fairness_report(rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n), g)
    msgs = [str(w.message) for w in caught]
    assert any("min_group_size" in m and "C" in m for m in msgs)
