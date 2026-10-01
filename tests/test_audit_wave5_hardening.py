"""Audit wave 5: test-suite hardening pins.

Two groups of tests, both purely over public APIs (no src/ edits):

1. Direct render tests for the two SVG templates that exist on disk but are
   being wired into the registry in this same wave (intersectional_disparity,
   workflow_overview). They are exercised here directly so their coverage
   does not depend on the registry edit landing first.
2. The highest-value boundary pins the audit flagged as missing. Every
   expected value below was REPRODUCED against the current implementation on
   2026-08-02 before being pinned; these are regression pins of observed,
   documented behaviour at edges that the existing suites only crossed
   mid-band.
"""

from __future__ import annotations

import re
import warnings
import xml.dom.minidom as minidom

import numpy as np
import pytest

from vfairness.agents.correspondence import CorrespondenceTester
from vfairness.evaluation.vfairness_metrics import classification as C
from vfairness.evaluation.vfairness_metrics._statistics import (
    IntervalType,
    StatisticalResult,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_ci,
    cohens_d,
    interpret_effect_size,
    risk_ratio,
)

# ---------------------------------------------------------------------------
# 1. Direct render tests for the two newly wired templates
# ---------------------------------------------------------------------------

engine = pytest.importorskip(
    "vfairness.rendering.engine",
    reason="rendering engine requires jinja2",
)
if not engine.JINJA2_AVAILABLE:  # pragma: no cover - env without jinja2
    pytest.skip("jinja2 not installed", allow_module_level=True)


class _Num(float):
    """A float usable where an int is required (range())."""

    def __index__(self) -> int:
        return int(self)


_UNDEF_RE = re.compile(r"'([\w]+)' is undefined")

# Structural context per template: objects that are attribute-accessed or
# iterated cannot be auto-filled with a numeric sentinel, so they are given
# real shapes. Remaining scalars are auto-filled on UndefinedError, mirroring
# the strategy documented in tests/test_rendering.py.
_DIRECT_CONTEXT = {
    "intersectional_disparity": {
        "title": "Intersectional disparity",
        "subtitle": "sample",
        "timestamp": "2026-01-01",
        "explanation": "",
        "groups": [
            {
                "name": "A x young",
                "rate": 0.6,
                "n": 40,
                "rank": 1,
                "ground_truth": 0.55,
                "delta_pp": 0.0,
                "bar_color": "#334155",
                "dot_color": "#334155",
            },
            {
                "name": "B x old",
                "rate": 0.3,
                "n": 35,
                "rank": 2,
                "ground_truth": 0.5,
                "delta_pp": -30.0,
                "bar_color": "#7f1d1d",
                "dot_color": "#7f1d1d",
            },
        ],
        "insights": [{"bg": "#fef2f2", "color": "#7f1d1d", "text": "Gap of 30pp"}],
        "disadvantaged": {"group": "B x old", "rate": 0.3, "n": 35, "actual": 11, "predicted": 21},
        "privileged": {"group": "A x young", "rate": 0.6, "n": 40, "actual": 22, "predicted": 24},
    },
    "workflow_overview": {
        "title": "Workflow overview",
        "subtitle": "sample",
        "timestamp": "2026-01-01",
        "explanation": "",
        # `stages` and `lines` are {% set %} inside the template; only the
        # three lists below are context-driven.
        "integrations": [
            {"name": "CI gatehouse", "description": "gate", "color": "#334155", "status": "active"}
        ],
        "vcs_tools": [{"name": "git hooks", "description": "hooks"}],
        "test_types": [{"name": "unit metrics", "description": "metrics"}],
    },
}


def _render_direct(name: str) -> str:
    from jinja2.exceptions import UndefinedError

    data = dict(_DIRECT_CONTEXT[name])
    for _ in range(400):
        try:
            return engine.render_svg(name, dict(data))
        except UndefinedError as exc:
            m = _UNDEF_RE.search(str(exc))
            if not m or m.group(1) in data:
                raise
            data[m.group(1)] = _Num(1.0)
    raise AssertionError(f"{name}: render did not converge")


@pytest.mark.parametrize("name", sorted(_DIRECT_CONTEXT))
def test_new_template_renders_wellformed_svg(name):
    out = _render_direct(name)
    assert out and out.strip(), f"{name}: rendered empty output"
    stripped = out.lstrip()
    if stripped.startswith("<?xml"):
        stripped = stripped[stripped.index("?>") + 2 :].lstrip()
    assert stripped.startswith("<svg"), f"{name}: not an <svg document"
    minidom.parseString(out)  # raises on malformed XML
    assert "{{" not in out, f"{name}: leftover expression token"
    assert "{%" not in out, f"{name}: leftover statement token"


def test_new_templates_carry_their_data():
    """The rendered SVGs must contain the supplied content, proving the
    context actually flowed through (an empty-loop render would also be
    well-formed XML)."""
    inter = _render_direct("intersectional_disparity")
    assert "B x old" in inter
    assert "Gap of 30pp" in inter
    wf = _render_direct("workflow_overview")
    assert "CI gatehouse" in wf
    assert "git hooks" in wf


# ---------------------------------------------------------------------------
# 2. Highest-value missing boundary pins over public APIs
# ---------------------------------------------------------------------------


class TestMetricEdgeBehaviour:
    """Edges of the classification metrics' group handling."""

    def test_single_qualifying_group_is_not_assessable_not_zero(self):
        # T1: with only one group above min_group_size there is no pair to
        # compare, so the metric is UNMEASURABLE and must say so: NaN, not a
        # crash and not 0.0.
        #
        # This test previously pinned 0.0 and was named ..._returns_zero_...,
        # which encoded the release-blocking defect rather than guarding against
        # it: 0.0 is the value of PERFECT parity for a difference metric, so a
        # comparison that never happened was reported as flawless fairness and
        # every downstream consumer (verdict, report, SVG, CI gate) inherited
        # the false certificate. The real-world case is a minority group small
        # enough to be dropped by min_group_size and never selected at all: the
        # truth was a 0.56 gap and the engine returned 0.0. NaN routes to the
        # report's NOT_ASSESSABLE path instead, which is the honest answer.
        # The no-crash half of the original test is kept below.
        y = np.array([1, 0] * 40)
        sens = np.array(["A"] * 70 + ["B"] * 10)  # B < default 30 -> dropped
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dp = C.demographic_parity_difference(y, y, sens)
            eod = C.equalized_odds_difference(y, y, sens)
        assert np.isnan(dp), f"unmeasurable DP must be NaN, got {dp}"
        assert np.isnan(eod), f"unmeasurable equalized odds must be NaN, got {eod}"
        # Explicitly refuse the old sentinel: these must never read as parity.
        assert dp != 0.0
        assert eod != 0.0

    def test_min_group_size_boundary_is_inclusive(self):
        # T2: a group with EXACTLY min_group_size members is kept; one member
        # fewer drops it. Guards against a >= vs > regression.
        y_true = np.array([1, 0] * 40)
        y_pred = np.array([1] * 50 + [0] * 30)
        at = np.array(["A"] * 50 + ["B"] * 30)
        below = np.array(["A"] * 51 + ["B"] * 29)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kept = C.demographic_parity_difference(y_true, y_pred, at, min_group_size=30)
            dropped = C.demographic_parity_difference(y_true, y_pred, below, min_group_size=30)
        # Exactly 30 members: B is KEPT and the real 1.0 gap is measured.
        assert kept == pytest.approx(1.0)
        # 29 members: B is DROPPED, leaving one group and nothing to compare.
        # This used to assert 0.0, i.e. the boundary test proved the >= vs >
        # behaviour by contrasting a maximal disparity with a PERFECT-PARITY
        # reading of the same data. One member fewer must not flip a 1.0 gap
        # into a clean pass; it must flip it into "could not check".
        assert np.isnan(dropped), f"dropping the second group must be NaN, got {dropped}"
        assert dropped != 0.0

    def test_parity_ratio_with_zero_rate_group_is_zero_not_nan(self):
        # T3: one group selects nobody -> ratio min/max = 0/rate = 0.0.
        # A NaN or ZeroDivisionError here would poison downstream reports.
        y_true = np.concatenate([np.ones(20), np.zeros(20), np.ones(20), np.zeros(20)]).astype(int)
        y_pred = np.concatenate([np.ones(20), np.zeros(60)]).astype(int)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        r = C.demographic_parity_ratio(y_true, y_pred, sens, min_group_size=30)
        assert r == 0.0
        assert not np.isnan(r)


class TestStatisticsEdgeBehaviour:
    """Edges of the public statistics helpers."""

    def test_bonferroni_adjusted_p_is_capped_at_one(self):
        # T4: p * n can exceed 1; a probability must not.
        res = bonferroni_correction(np.array([0.5, 0.4, 0.9]))
        assert np.all(res.adjusted_p_values <= 1.0)
        assert res.n_rejected == 0

    def test_benjamini_hochberg_monotone_and_bounded(self):
        # T5: BH step-up adjusted p-values are nondecreasing in the sorted
        # p order and never exceed 1 (the monotonicity enforcement is the
        # classic off-by-one trap in hand-rolled BH implementations).
        p = np.array([0.01, 0.02, 0.9])
        res = benjamini_hochberg_correction(p)
        order = np.argsort(p)
        adj_sorted = res.adjusted_p_values[order]
        assert np.all(np.diff(adj_sorted) >= -1e-12)
        assert np.all(res.adjusted_p_values <= 1.0)
        # Reproduced exact values for this input: [0.03, 0.03, 0.9].
        assert res.adjusted_p_values == pytest.approx([0.03, 0.03, 0.9])

    def test_bootstrap_ci_is_deterministic_under_random_state(self):
        # T6: identical random_state must give bit-identical intervals;
        # otherwise CI runs are unreproducible and flaky by construction.
        rng = np.random.default_rng(3)
        data = rng.normal(0, 1, 80)
        r1 = bootstrap_ci(data, np.mean, n_bootstrap=400, random_state=11)
        r2 = bootstrap_ci(data, np.mean, n_bootstrap=400, random_state=11)
        assert r1.point_estimate == r2.point_estimate
        assert r1.lower_bound == r2.lower_bound
        assert r1.upper_bound == r2.upper_bound

    def test_interpret_effect_size_unknown_type_falls_back(self):
        # T7: an unknown effect_type must degrade to a plain numeric
        # disclosure, never raise or silently reuse Cohen's bands.
        assert interpret_effect_size(0.42, "eta_squared") == ("effect size = 0.420")

    def test_risk_ratio_zero_events_discloses_nan_ci(self):
        # T8: zero events in group 1 -> point estimate 0.0 with NaN CI
        # bounds (the log-based CI is undefined). Pins the current honest
        # disclosure; a silent positive CI here would be fabricated.
        rr, lower, upper = risk_ratio(0, 50, 10, 50)
        assert rr == 0.0
        assert np.isnan(lower) and np.isnan(upper)

    def test_cohens_d_sign_convention(self):
        # T9: positive d means group 1 is HIGHER; swapping groups negates d.
        # interpret_effect_size's direction wording depends on this.
        g_hi = np.array([2.0, 2.1, 1.9, 2.0])
        g_lo = np.array([1.0, 1.1, 0.9, 1.0])
        d = cohens_d(g_hi, g_lo)
        assert d > 0
        assert cohens_d(g_lo, g_hi) == pytest.approx(-d)

    def test_interval_touching_zero_is_not_significant(self):
        # T10: lower bound exactly 0.0 -> the interval CONTAINS zero and
        # must not be called significant (strict inequality semantics).
        res = StatisticalResult(
            point_estimate=0.1,
            lower_bound=0.0,
            upper_bound=0.2,
            interval_type=IntervalType.CONFIDENCE,
        )
        assert res.is_significant is False


class TestFourFifthsBoundary:
    def test_exactly_four_fifths_is_not_adverse_impact(self):
        # T11: the EEOC rule is "less than 4/5"; a ratio of exactly 0.8
        # passes, epsilon below fails. 0.4/0.5 is exact in binary floats,
        # so this pin is not tolerance-sensitive.
        tester = CorrespondenceTester()
        at = tester.four_fifths_rule(0.5, 0.4)
        assert at["ratio"] == pytest.approx(0.8)
        assert at["adverse_impact"] is False
        below = tester.four_fifths_rule(0.5, 0.395)
        assert below["adverse_impact"] is True
