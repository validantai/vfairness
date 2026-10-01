"""Regression tests pinning the Wave-2 fixes from the 2026-08 deep audit."""

from __future__ import annotations

import numpy as np
import pytest

# ── _statistics: Bayesian PPF numerically correct ────────────────────────────


def test_beta_ppf_matches_scipy_on_grid():
    scipy_stats = pytest.importorskip("scipy.stats")
    from vfairness.evaluation.vfairness_metrics._statistics import _beta_ppf

    for a, b in [(3, 17), (8, 4), (50, 50), (501, 501), (1, 1), (6, 16)]:
        for q in (0.025, 0.5, 0.975):
            assert _beta_ppf(q, a, b) == pytest.approx(
                float(scipy_stats.beta.ppf(q, a, b)), abs=1e-9
            ), (a, b, q)


def test_bayesian_proportion_ci_coverage_restored():
    from vfairness.evaluation.vfairness_metrics._statistics import (
        bayesian_proportion_ci,
    )

    rng = np.random.default_rng(11)
    hits = 0
    sims = 300
    for _ in range(sims):
        k = rng.binomial(20, 0.3)
        r = bayesian_proportion_ci(int(k), 20)
        hits += r.lower_bound <= 0.3 <= r.upper_bound
    # was ~0.55 with the broken homegrown PPF; must be >= nominal-ish now
    assert hits / sims >= 0.90


def test_wilson_score_interval_sane():
    from vfairness.evaluation.vfairness_metrics._statistics import (
        wilson_score_interval,
    )

    r = wilson_score_interval(30, 100)
    assert r.lower_bound < 0.3 < r.upper_bound
    assert 0.0 <= r.lower_bound and r.upper_bound <= 1.0
    assert r.method == "wilson_score"
    # empty group -> NaN result, no crash
    empty = wilson_score_interval(0, 0)
    assert np.isnan(empty.point_estimate)


def test_group_metrics_with_ci_large_groups_get_real_intervals():
    from vfairness.evaluation.vfairness_metrics.classification import (
        get_group_metrics_with_ci,
    )

    rng = np.random.default_rng(12)
    n = 400
    g = np.array(["A"] * 200 + ["B"] * 200)
    y_true = rng.binomial(1, 0.5, n)
    y_pred = rng.binomial(1, 0.5, n)
    out = get_group_metrics_with_ci(y_true, y_pred, g)
    for grp, metrics in out.items():
        pr = metrics["positive_rate"]
        assert not np.isnan(pr.lower_bound), f"{grp}: NaN lower bound"
        assert not np.isnan(pr.upper_bound), f"{grp}: NaN upper bound"
        assert pr.lower_bound <= pr.point_estimate <= pr.upper_bound


# ── reweighters: fit() no longer crashes ────────────────────────────────────


def test_all_reweighters_fit_and_guards_hold():
    from vfairness.post_processing.reweighting.reweighter import (
        CalibratedEqualizer,
        DistributionMatcher,
        PredictionReweighter,
        RejectionOptionClassifier,
    )

    rng = np.random.default_rng(7)
    y = rng.binomial(1, 0.5, 300)
    g = np.array(["A", "B"] * 150)
    p = np.clip(0.3 * y + rng.random(300) * 0.7, 0.01, 0.99)
    for cls in (
        RejectionOptionClassifier,
        CalibratedEqualizer,
        DistributionMatcher,
        PredictionReweighter,
    ):
        inst = cls()
        inst.fit(y_true=y, y_prob=p, sensitive_attr=g)
        assert inst.result_ is not None, cls.__name__
    # unfitted use must still raise
    for cls in (RejectionOptionClassifier, CalibratedEqualizer, DistributionMatcher):
        with pytest.raises(RuntimeError):
            cls().transform(p, g)


# ── report: degenerate data must not certify ────────────────────────────────


def test_single_group_report_is_marked_not_assessable():
    from vfairness.evaluation.vfairness_metrics.report import (
        classification_fairness_report,
    )

    y = np.array([1, 0, 1, 0] * 30)
    rep = classification_fairness_report(y, y, np.array(["A"] * 120))
    a = rep["assessment"]
    assert a["assessable"] is False
    assert "NOT ASSESSABLE" in a["summary"]


def test_two_group_report_still_assessable():
    from vfairness.evaluation.vfairness_metrics.report import (
        classification_fairness_report,
    )

    rng = np.random.default_rng(13)
    n = 200
    g = np.array(["A"] * 100 + ["B"] * 100)
    rep = classification_fairness_report(rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n), g)
    assert rep["assessment"]["assessable"] is True


# ── calibration: quantile binning on constant probabilities ─────────────────


def test_quantile_ece_not_zero_for_constant_probs():
    from vfairness.post_processing.calibration.metrics import (
        expected_calibration_error,
    )

    y = np.array([1] * 40 + [0] * 60)
    p = np.full(100, 0.7)
    for strategy in ("uniform", "quantile"):
        e = expected_calibration_error(y, p, strategy=strategy).overall_value
        assert e == pytest.approx(0.3), strategy


# ── explainer: enum severities counted; bands match the badge ───────────────


def test_explainer_counts_enum_severities():
    from enum import Enum
    from types import SimpleNamespace

    from vfairness.explainer import _explain_validation_result

    class Sev(Enum):
        ERROR = "error"
        WARNING = "warning"

    res = SimpleNamespace(
        passed=False,
        issues=[
            SimpleNamespace(severity=Sev.ERROR, message="x"),
            SimpleNamespace(severity=Sev.WARNING, message="y"),
        ],
    )
    rep = _explain_validation_result(res)
    assert "1 errors" in rep.summary and "1 warnings" in rep.summary


def test_bias_audit_severity_matches_badge_band_everywhere():
    from types import SimpleNamespace

    from vfairness.explainer import _explain_bias_audit, _risk_band

    band_to_sev = {"MINIMAL": "info", "LOW": "low", "MEDIUM": "medium", "HIGH": "high"}
    for score in (0.0, 0.12, 0.249, 0.25, 0.3, 0.499, 0.5, 0.6, 0.749, 0.75, 0.9, 1.0):
        report = SimpleNamespace(
            overall_risk_score=score,
            protected_attributes=["g"],
            critical_issues=[],
            historical_findings=[],
            representation_issues=[],
            statistical_disparities=[],
            proxy_variables=[],
            recommendations=[],
        )
        rep = _explain_bias_audit(report)
        band = _risk_band(score)
        assert f"({band})" in rep.summary
        # the overall-risk item's severity must match the badge band
        overall = rep.explanations[0]
        assert overall.severity == band_to_sev[band], (score, band)


# ── rendering: radar bands agree with the detailed report ───────────────────


def test_radar_bands_match_detailed_report_scale():
    import re

    src = open("src/vfairness/rendering/adapters_fairness.py", encoding="utf-8").read()
    block = src[src.index("Determine overall status") :]
    first_two = re.findall(r"score >= (0\.\d+)", block)[:2]
    assert first_two == ["0.8", "0.5"], (
        "radar Fair/Marginal bands must match the detailed report's 80/50"
    )
