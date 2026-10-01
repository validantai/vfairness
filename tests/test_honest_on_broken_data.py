"""Check 2 of the beta gate: honest on broken data, for every measuring capability.

The harness is scripts/broken_data_check.py; read its docstring for how the
expected answer is counted from the data rather than chosen. This file holds it
to three things:

1. No capability it covers FABRICATES (a finite unflagged number where nothing
   was measurable) or OVER-REFUSES (a refusal where something was).
2. The harness itself can tell: a capability built to fabricate, one built to
   over-refuse, one that only WARNS, and one that reports a group it could not
   measure are each caught. A harness that passes everything looks exactly like
   one that found nothing wrong.
3. The capabilities it does not cover yet are named, so the gap cannot shrink or
   grow without this file changing.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import broken_data_check as B  # noqa: E402


@pytest.fixture(scope="module")
def result():
    return B.run()


def test_no_covered_capability_fabricates_or_over_refuses(result):
    bad = {
        lbl: [
            (r["world"], r["verdict"], r["detail"])
            for r in rows
            if r["verdict"] in ("FABRICATES", "OVER-REFUSES")
        ]
        for lbl, rows in result["rows"].items()
    }
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, bad


def test_every_not_judged_cell_carries_a_reason(result):
    for lbl, rows in result["rows"].items():
        for r in rows:
            if r["verdict"] == "NOT JUDGED":
                assert r.get("reason"), (lbl, r["world"])


# Empty since 2026-10-01: every evaluation capability has purpose-built inputs or is
# recorded in NOT_A_MEASUREMENT. A new registry entry fails here until it gets one.
NOT_YET_COVERED: set = set()


def test_the_uncovered_capabilities_are_exactly_the_recorded_ones(result):
    uncovered = {n for n, c in result["capabilities"].items() if c["verdict"] == "NOT COVERED"}
    assert uncovered == NOT_YET_COVERED, {
        "newly uncovered": sorted(uncovered - NOT_YET_COVERED),
        "now covered, remove from the list": sorted(NOT_YET_COVERED - uncovered),
    }


# === the harness must be able to disagree ===


def _verdicts(spec):
    worlds = {n: B._build(n) for n in B.WORLD_NAMES}
    return {n: B.judge(spec, worlds[n])["verdict"] for n in B.WORLD_NAMES}


def test_harness_catches_a_fabricator():
    v = _verdicts(B.Spec("fake", "fake", lambda w: 0.0))
    assert v["healthy"] == "PASS"
    assert v["single_group"] == "FABRICATES"
    assert v["empty"] == "FABRICATES"


def test_harness_catches_an_over_refuser():
    v = _verdicts(B.Spec("fake", "fake", lambda w: float("nan")))
    assert v["healthy"] == "OVER-REFUSES"
    assert v["single_group"] == "PASS"


def test_a_warning_is_not_a_refusal():
    def warns(w):
        warnings.warn("could not measure", UserWarning)
        return 0.0

    assert _verdicts(B.Spec("fake", "fake", warns))["single_group"] == "FABRICATES"


def test_a_flag_in_the_returned_object_is_a_refusal():
    def flagged(w):
        return {"value": 0.0, "measured": len(set(w.s.tolist())) >= 2}

    v = _verdicts(B.Spec("fake", "fake", flagged, lambda r: r["value"]))
    assert v["single_group"] == "PASS"
    assert v["healthy"] == "PASS"


def test_a_value_for_an_unmeasurable_group_is_caught():
    def every_group(w):
        return {g: 0.5 for g in set(w.s.tolist())}

    spec = B.Spec("fake", "fake", every_group, grouped_min=1, per_group=True)
    assert _verdicts(spec)["one_row_minority"] == "FABRICATES"


# === pins for defects check 2 found ===


def test_subgroup_audit_does_not_read_missing_predictions_as_no():
    """Found 2026-10-01: every prediction NaN gave rates 0.0 and a False verdict."""
    from vfairness.evaluation.vfairness_metrics import subgroup_robustness_audit

    w = B._build("all_nan_scores")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = subgroup_robustness_audit(w.p, {"g": w.s}, y_true=w.y)
    assert r.gerrymandering_detected is None
    assert np.isnan(r.worst_disparity)
    assert r.n_rows_missing_prediction == len(w.p)

    # Control: the healthy world is still measured exactly.
    h = B._build("healthy")
    r = subgroup_robustness_audit(h.p, {"g": h.s}, y_true=h.y)
    rate_a = h.p[h.s == "a"].mean()
    rate_b = h.p[h.s == "b"].mean()
    overall = h.p.mean()
    want = max((rate_a - overall, rate_b - overall), key=abs)
    assert r.gerrymandering_detected is False
    assert r.worst_disparity == pytest.approx(want, abs=1e-12)

    # Partly missing: a real breach is still reported, and the count is disclosed.
    p = h.p.astype(float)
    p[:5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = subgroup_robustness_audit(p, {"g": h.s}, disparity_threshold=0.05)
    assert r.gerrymandering_detected is True
    assert r.n_rows_missing_prediction == 5


def test_not_a_measurement_entries_really_return_no_number():
    """The exemption is re-earned on every run: if a number ever comes back, it fails."""
    from vfairness.xai.explainers.router import route_explainer

    assert set(B.NOT_A_MEASUREMENT) == {"route_explainer"}
    for model_type in ("tree", "linear", "neural", "black_box"):
        decision = route_explainer(model_type=model_type)
        values = [getattr(decision, f) for f in decision.__dataclass_fields__]
        assert not B._numbers(values), (model_type, decision)
