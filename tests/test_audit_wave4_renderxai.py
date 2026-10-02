"""Audit wave 4 pins: rendering adapters + XAI misc.

Each test pins one adversarially confirmed audit finding:

1.  group_comparison: empty group_stats renders a neutral no-data state,
    and the disparity band label matches the value as displayed.
2.  reweighting: fairness_improvement uses the same formula for dict and
    object inputs (relative reduction of the demographic-parity gap).
3.  fairness_detailed_report: missing score renders an explicit N/A
    state, never '5000/100 FAIR'.
4.  reporting dashboard: ratio metrics (higher is better) breach when
    the value falls BELOW the threshold, not above.
5.  robustness: the overall badge scores only supplied components;
    absent permutation tests are excluded, not counted as zero.
6.  explain: displayed values never round across their band boundary.
7.  shap_fairness: >2 groups warns, naming the two groups compared.
8.  faithfulness: trapezoid integration works on numpy without
    np.trapezoid (declared floor is numpy>=1.21).
9.  dice: feasibility checks the observed [min, max] bounds, and
    _predict_label warns + returns None instead of swallowing errors.
10. schemas: Explanation carries an explicit scope field that
    to_db_row honours.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pytest

pytest.importorskip("jinja2")


# 1. group_comparison ────────────────────────────────────────────────────────


def test_group_comparison_empty_group_stats_is_neutral():
    from vfairness.rendering.adapters_fairness import group_comparison_to_svg

    svg = group_comparison_to_svg({"group_stats": {}})
    assert "High disparity" not in svg
    assert "severity: HIGH" not in svg
    assert "No group data" in svg


def test_group_comparison_label_matches_displayed_value():
    from vfairness.rendering.adapters_fairness import group_comparison_to_svg

    # max disparity = 0.0999: displayed at 3 decimals as '0.100', so the
    # band label must be Moderate (>= 0.1), never 'Low' beside '0.100'.
    svg = group_comparison_to_svg(
        {
            "group_stats": {
                "a": {"positive_rate": 0.5, "count": 10},
                "b": {"positive_rate": 0.4001, "count": 10},
            }
        }
    )
    assert "0.100" in svg
    assert "Moderate" in svg
    assert "Low" not in svg  # capital-L badge label; prose uses lowercase


def test_fr_group_comparison_none_disparity_is_info():
    from vfairness.rendering.explain import _fr_group_comparison

    finding, severity = _fr_group_comparison({"max_disparity": None})
    assert severity == "info"
    assert "No group data" in finding


# 2. reweighting semantics ───────────────────────────────────────────────────


def _reweighting_payload():
    return {
        "method": "equalized_odds",
        "original_fairness": {"demographic_parity_diff": 0.30},
        "adjusted_fairness": {"demographic_parity_diff": 0.15},
        "original_performance": {"accuracy": 0.90},
        "adjusted_performance": {"accuracy": 0.89},
        "calibration_metrics": {"ece_change": 0.0},
        "trade_off_score": 0.5,
    }


def test_reweighting_improvement_same_formula_dict_and_object():
    from dataclasses import dataclass, field

    from vfairness.rendering.adapters_post_processing import (
        reweighting_comparison_to_svg,
    )

    payload = _reweighting_payload()

    @dataclass
    class FakeResult:
        method: str = "equalized_odds"
        original_fairness: dict = field(default_factory=lambda: dict(payload["original_fairness"]))
        adjusted_fairness: dict = field(default_factory=lambda: dict(payload["adjusted_fairness"]))
        original_performance: dict = field(
            default_factory=lambda: dict(payload["original_performance"])
        )
        adjusted_performance: dict = field(
            default_factory=lambda: dict(payload["adjusted_performance"])
        )
        calibration_metrics: dict = field(
            default_factory=lambda: dict(payload["calibration_metrics"])
        )
        trade_off_score: float = 0.5

    svg_dict = reweighting_comparison_to_svg({"method_results": [payload]})
    svg_obj = reweighting_comparison_to_svg({"method_results": [FakeResult()]})
    # 0.30 -> 0.15 is a 50% relative reduction on BOTH input paths. The
    # object path used to show the absolute change (+15.0%) instead.
    assert "+50.0%" in svg_dict
    assert "+50.0%" in svg_obj
    assert "+15.0%" not in svg_obj


# 3. fairness_detailed_report N/A state ──────────────────────────────────────


def test_detailed_report_missing_score_renders_na():
    from vfairness.rendering.adapters_post_processing import (
        fairness_detailed_report_to_svg,
    )

    report = {"metrics": {"demographic_parity": {"value": 0.05, "threshold": 0.1}}}
    with pytest.warns(UserWarning, match="fairness_score"):
        svg = fairness_detailed_report_to_svg(report)
    assert "5000" not in svg
    assert "FAIR" not in svg.replace("FAIRNESS SCORE", "")
    assert "N/A" in svg


def test_detailed_report_present_score_unchanged():
    from vfairness.rendering.adapters_post_processing import (
        fairness_detailed_report_to_svg,
    )

    svg = fairness_detailed_report_to_svg(
        {
            "assessment": {"fairness_score": 0.9},
            "metrics": {"demographic_parity": {"value": 0.05, "threshold": 0.1}},
        }
    )
    assert "90" in svg
    assert "FAIR" in svg


# 4. reporting dashboard ratio direction ─────────────────────────────────────


def _report_with_metric(metric):
    return {
        "health_score": {"score": 90, "status": "green", "components": {}},
        "metrics": [metric],
        "tier": "OPERATIONAL",
        "sections": [],
        "alerts": [],
        "recommendations": [],
    }


def test_ratio_metric_above_threshold_is_not_a_breach():
    from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

    svg = reporting_dashboard_to_svg(
        _report_with_metric({"name": "disparate_impact_ratio", "value": 0.95, "threshold": 0.8})
    )
    assert "THRESHOLD BREACH REPORT" not in svg


def test_ratio_metric_below_threshold_is_a_breach():
    from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

    svg = reporting_dashboard_to_svg(
        _report_with_metric({"name": "disparate_impact_ratio", "value": 0.6, "threshold": 0.8})
    )
    assert "THRESHOLD BREACH REPORT" in svg


def test_difference_metric_direction_unchanged():
    from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg

    svg = reporting_dashboard_to_svg(
        _report_with_metric({"name": "demographic_parity_diff", "value": 0.15, "threshold": 0.1})
    )
    assert "THRESHOLD BREACH REPORT" in svg


def test_breach_and_margin_helpers_respect_direction():
    from vfairness.rendering.adapters_reporting import (
        _breach_pct,
        _is_breached,
        _margin_pct,
        _metric_color,
    )

    assert _is_breached(0.95, 0.8, higher_is_better=True) is False
    assert _is_breached(0.6, 0.8, higher_is_better=True) is True
    assert _is_breached(0.15, 0.1) is True
    assert _breach_pct(0.6, 0.8, higher_is_better=True) == 25.0
    assert _margin_pct(0.95, 0.8, higher_is_better=True) > 0
    assert _metric_color(0.95, 0.8, higher_is_better=True) != "#dc2626"
    assert _metric_color(0.6, 0.8, higher_is_better=True) == "#dc2626"


# 5. robustness overall score ────────────────────────────────────────────────


def test_robustness_perfect_sensitivity_without_permutation_passes():
    from vfairness.rendering.adapters_robustness import robustness_testing_to_svg

    svg = robustness_testing_to_svg(
        sensitivity_results=[
            {
                "perturbation_type": "noise",
                "robustness_score": 1.0,
                "is_robust": True,
                "max_deviation": 0.0,
            },
        ]
    )
    assert "MARGINAL" not in svg
    assert "PASS" in svg


def test_robustness_perfect_permutation_without_sensitivity_passes():
    from vfairness.rendering.adapters_robustness import robustness_testing_to_svg

    svg = robustness_testing_to_svg(
        permutation_results=[
            {
                "method": "demographic_parity",
                "observed_statistic": 0.01,
                "p_value": 0.7,
                "significant_at_05": False,
                "effect_direction": "neutral",
            },
        ]
    )
    assert "PASS" in svg


def test_robustness_both_components_still_weighted():
    from vfairness.rendering.adapters_robustness import robustness_testing_to_svg

    # sensitivity 1.0 (weight .6) + all permutation tests significant
    # (component 0, weight .4) = 0.6 overall = MARGINAL.
    svg = robustness_testing_to_svg(
        permutation_results=[
            {
                "method": "dp",
                "observed_statistic": 0.2,
                "p_value": 0.001,
                "significant_at_05": True,
                "effect_direction": "positive",
            },
        ],
        sensitivity_results=[
            {
                "perturbation_type": "noise",
                "robustness_score": 1.0,
                "is_robust": True,
                "max_deviation": 0.0,
            },
        ],
    )
    assert "MARGINAL" in svg


# 6. explain band-boundary rounding ──────────────────────────────────────────


def test_f_band_never_rounds_across_a_boundary():
    from vfairness.rendering.explain import _f_band

    # 0.2499 at 2 decimals reads '0.25', which sits in the next band up.
    assert _f_band(0.2499, (0.25, 0.50, 0.75)) == "0.2499"
    # An exact boundary value needs no extra precision.
    assert _f_band(0.25, (0.25, 0.50, 0.75)) == "0.25"
    assert _f_band(0.12, (0.25, 0.50, 0.75)) == "0.12"
    assert _f_band("junk", (0.25,)) == "?"


def test_bias_audit_finding_number_stays_in_badge_band():
    from vfairness.rendering.explain import _fr_bias_audit

    finding, severity = _fr_bias_audit(
        {"overall_score": 0.2499, "overall_label": "MINIMAL", "n_critical": 0}
    )
    assert "0.25 " not in finding
    assert "0.2499" in finding
    assert severity == "info"


def test_effect_sizes_finding_number_stays_in_interpretation_band():
    from vfairness.rendering.explain import _fr_effect_sizes

    finding, severity = _fr_effect_sizes(
        {"max_effect": 0.1999, "max_effect_interpretation": "Negligible", "avg_effect": 0.1}
    )
    assert "0.20 " not in finding
    assert "0.1999" in finding
    assert severity == "info"


# 7. shap_fairness multi-group warning ───────────────────────────────────────


def _decompose(group_labels):
    from vfairness.xai.decomposition.shap_fairness import (
        lundberg_fairness_decomposition,
    )

    shap_values = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.2], [0.15, 0.05]])
    return lundberg_fairness_decomposition(
        shap_values=shap_values,
        group_labels=np.asarray(group_labels),
        feature_names=["f0", "f1"],
        metric="demographic_parity",
        protected_attribute="ethnicity",
        subject_id="s",
        audit_artifact_id="a",
    )


def test_shap_decomposition_warns_naming_groups_when_more_than_two():
    with pytest.warns(UserWarning, match=r"'a'.*'b'"):
        _decompose(["a", "b", "c", "a"])


def test_shap_decomposition_two_groups_stays_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _decompose(["a", "b", "b", "a"])


# 8. faithfulness trapezoid fallback ─────────────────────────────────────────


def test_removal_curve_auc_without_np_trapezoid(monkeypatch):
    import vfairness.xai.diagnostics.faithfulness as faithfulness

    # keep a reference before hiding the name. numpy before 2.0 has no
    # np.trapezoid at all (only np.trapz), which is the very case the fallback
    # serves, so take whichever this numpy ships.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", trapezoid, raising=False)
    try:
        importlib.reload(faithfulness)
        auc = faithfulness.removal_curve_auc(
            predict_fn=lambda a: np.asarray([float(a.sum())]),
            x=np.array([1.0, 2.0, 3.0]),
            attributions=np.array([0.5, 0.2, 0.1]),
            background_mean=np.zeros(3),
        )
        assert auc > 0
    finally:
        monkeypatch.undo()
        importlib.reload(faithfulness)  # restore the module for other tests


# 9. dice feasibility bounds + prediction errors ─────────────────────────────


def _dice_class():
    from vfairness.xai.explainers.dice_adapter import DiceCounterfactualExplainer

    return DiceCounterfactualExplainer


def test_dice_within_observed_enforces_min_max():
    pd = pytest.importorskip("pandas")
    cls = _dice_class()
    frame = pd.DataFrame({"f0": [0.0, 1.0, 2.0], "cat": ["x", "y", "z"]})
    ranges = cls._feature_ranges(frame, ["f0", "cat"])
    assert ranges["f0"] == {"min": 0.0, "max": 2.0, "span": 2.0}
    assert cls._within_observed("f0", 1.5, ranges) is True
    assert cls._within_observed("f0", 0.0, ranges) is True  # inclusive bound
    assert cls._within_observed("f0", 999.0, ranges) is False
    assert cls._within_observed("f0", -0.1, ranges) is False
    assert cls._within_observed("cat", 1.0, ranges) is False  # no numeric bounds
    assert cls._within_observed("missing", 1.0, ranges) is False


def test_dice_feasibility_reflects_out_of_range_counterfactual():
    pd = pytest.importorskip("pandas")
    cls = _dice_class()
    adapter = object.__new__(cls)  # skip __init__ (needs dice-ml installed)
    frame = pd.DataFrame({"f0": [0.0, 1.0, 2.0], "f1": [0.0, 5.0, 10.0]})
    ranges = cls._feature_ranges(frame, ["f0", "f1"])
    inst = cls._to_instance(
        adapter,
        {"f0": 50.0, "f1": 5.0, "y": 1},  # f0 far outside [0, 2]
        {"f0": 1.0, "f1": 0.0},
        ["f0", "f1"],
        "y",
        ranges,
    )
    # One of two changes lands out of the observed range: feasibility 0.5,
    # not the constant 1.0 the span-only check produced.
    assert inst.feasibility == 0.5


def test_dice_predict_label_warns_and_returns_none_on_error():
    cls = _dice_class()

    class Boom:
        def predict(self, q):
            raise RuntimeError("real failure")

    with pytest.warns(RuntimeWarning, match="real failure"):
        assert cls._predict_label(Boom(), None, "sklearn") is None


# 10. schemas scope ──────────────────────────────────────────────────────────


def _explanation(**overrides):
    from vfairness.xai.schemas import Attribution, Explanation

    kwargs = dict(
        method="lime",
        instance_id="subj#row-0",
        subject_id="subj",
        model_hash="m",
        data_hash="d",
        base_value=0.0,
        prediction=1.0,
        attributions=[Attribution(feature="f", contribution=0.1)],
        units="raw",
    )
    kwargs.update(overrides)
    return Explanation(**kwargs)


def test_explanation_explicit_scope_is_written():
    row = _explanation(scope="global").to_db_row(owner="o")
    assert row["scope"] == "global"  # even though instance_id is set


def test_explanation_scope_fallback_heuristic_preserved():
    assert _explanation().to_db_row(owner="o")["scope"] == "local"
    assert _explanation(instance_id="").to_db_row(owner="o")["scope"] == "global"
