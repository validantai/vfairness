"""Regression tests pinning the Wave-3 fixes from the 2026-08 deep audit,
plus boundary pins for previously mutation-transparent thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── _statistics: NaN interval must not be significant ───────────────────────


def test_nan_interval_is_not_significant():
    from vfairness.evaluation.vfairness_metrics._statistics import (
        IntervalType,
        StatisticalResult,
    )

    r = StatisticalResult(
        point_estimate=0.4,
        lower_bound=float("nan"),
        upper_bound=float("nan"),
        interval_type=IntervalType.CONFIDENCE,
    )
    assert r.is_significant is False
    ok = StatisticalResult(
        point_estimate=0.4,
        lower_bound=0.1,
        upper_bound=0.7,
        interval_type=IntervalType.CONFIDENCE,
    )
    assert ok.is_significant is True


# ── agents: bias source must be the bias-ADDING stage ───────────────────────


def test_bias_source_is_the_adding_stage_not_the_reducing_one():
    from vfairness.agents import PipelineTracker

    tracker = PipelineTracker.__new__(PipelineTracker)

    class _R:
        def __init__(self, name, contribution):
            self.stage_name = name
            self.stage_contribution = contribution
            # READINESS-6, 2026-09-10. The stub carries bias_metrics with a
            # MEASURED p-value now, because identify_bias_source ranks only
            # among stages whose significance test could actually run. Without
            # it both stubs read as untestable and the function refuses, which
            # would be correct behaviour and would say nothing about the sign
            # logic this test exists to pin.
            self.bias_metrics = {"p_value": 0.01, "n_a": 40, "n_b": 40}

    results = [_R("adds_bias", 0.30), _R("reduces_bias_strongly", -0.90)]
    tracker.compute_cumulative = lambda: results
    assert tracker.identify_bias_source() == "adds_bias"


# ── xai: deep routing must select an implemented explainer ──────────────────


def test_deep_route_primary_is_implemented():
    from vfairness.xai.explainers.registry import _NOT_YET_IMPLEMENTED, _REGISTRY
    from vfairness.xai.explainers.router import route_explainer

    decision = route_explainer(model_type="deep")
    assert decision.primary in _REGISTRY, decision.primary
    assert decision.primary not in _NOT_YET_IMPLEMENTED


def test_decomposition_rejects_unsupported_metric():
    from vfairness.xai.decomposition.shap_fairness import (
        lundberg_fairness_decomposition,
    )

    shap_values = np.array([[0.2, -0.1], [0.1, 0.0], [-0.2, 0.1], [0.0, 0.2]])
    groups = np.array(["A", "A", "B", "B"])
    with pytest.raises(NotImplementedError):
        lundberg_fairness_decomposition(
            shap_values=shap_values,
            group_labels=groups,
            feature_names=["f0", "f1"],
            metric="equal_opportunity",
            protected_attribute="g",
            subject_id="s",
            audit_artifact_id="a",
        )


# ── rendering: intersectional spread + NaN badge + control chars ────────────


def test_intersectional_disparity_uses_max_minus_min():
    from vfairness.rendering import intersectional_analysis_to_svg

    data = {
        "x_attr": "G",
        "y_attr": "R",
        # extremes NOT adjacent in scan order: true spread 0.8, adjacent max 0.4
        "matrix": {"r1": {"a": 0.5, "b": 0.9}, "r2": {"a": 0.5, "b": 0.1}},
        "counts": {"r1": {"a": 10, "b": 10}, "r2": {"a": 10, "b": 10}},
    }
    svg = intersectional_analysis_to_svg(data, feature="rate", explanation="")
    assert "0.800" in svg or "0.80" in svg, "true spread (0.8) must be reported"


def test_nan_risk_score_does_not_render_minimal():
    from vfairness.rendering.engine import _risk_bg, _risk_color, _risk_label

    assert _risk_label(float("nan")) == "N/A"
    assert _risk_label(None) == "N/A"
    assert _risk_color(float("nan")) not in ("#059669",)
    assert _risk_bg(float("nan")) not in ("#d1fae5",)
    # real scores unchanged at the boundaries
    assert _risk_label(0.249) == "MINIMAL" and _risk_label(0.25) == "LOW"
    assert _risk_label(0.499) == "LOW" and _risk_label(0.5) == "MEDIUM"
    assert _risk_label(0.749) == "MEDIUM" and _risk_label(0.75) == "HIGH"


def test_render_svg_strips_xml_control_chars():
    import xml.dom.minidom as minidom

    from vfairness.rendering.engine import render_svg

    svg = render_svg("radar_chart", {"title": "bad \x02 ctrl \x1f chars", "status_text": "Fair"})
    assert "\x02" not in svg and "\x1f" not in svg
    minidom.parseString(svg)  # must be well-formed XML


# ── validation: NaN sensitive rows + y_true binary check ────────────────────


def test_intersectional_nan_rows_are_excluded():
    from vfairness.evaluation.vfairness_metrics._validation import (
        validate_inputs,
    )

    df = pd.DataFrame(
        {
            "g": ["m", "f", np.nan, "m", "f", "m"],
            "r": ["x", "y", "x", np.nan, "y", "x"],
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    y_true, y_pred, sens, y_prob, info = validate_inputs(
        y,
        y,
        df,
        task_type="classification",
        missing_strategy="exclude",
    )
    assert info["n_excluded"] == 2
    assert len(y_true) == 4
    # no NaN survives into the attribute frame
    assert not sens.isna().any().any()


def test_non_binary_y_true_is_rejected():
    from vfairness.evaluation.vfairness_metrics._validation import (
        validate_inputs,
    )

    y_true = np.array([1, 2, 1, 2])  # 1/2 encoding, not 0/1
    y_pred = np.array([1, 0, 1, 0])
    g = np.array(["a", "b", "a", "b"])
    with pytest.raises(ValueError):
        validate_inputs(y_true, y_pred, g, task_type="classification")


# ── representation: absent benchmark group + benchmark normalisation ────────


def test_absent_benchmark_group_is_flagged():
    from vfairness.preprocessing.bias_detection.representation import (
        detect_representation_bias,
    )

    df = pd.DataFrame({"gender": ["male"] * 100})  # all-male dataset
    results = detect_representation_bias(
        df,
        ["gender"],
        benchmarks={"gender": {"male": 0.5, "female": 0.5}},
    )
    r = results[0]
    under = {g["group"] for g in r.underrepresented_groups}
    assert "female" in under, "an absent expected group is maximal underrepresentation"


def test_benchmarks_not_summing_to_one_are_normalised():
    from vfairness.preprocessing.bias_detection.representation import (
        detect_representation_bias,
    )

    df = pd.DataFrame({"gender": ["male"] * 50 + ["female"] * 50})
    with pytest.warns(UserWarning, match="renormalis"):
        results = detect_representation_bias(
            df,
            ["gender"],
            # proportions sum to 2.0; balanced data must NOT be flagged
            benchmarks={"gender": {"male": 1.0, "female": 1.0}},
        )
    r = results[0]
    assert not r.underrepresented_groups, (
        "balanced data vs a (renormalised) balanced benchmark has no deficit"
    )


# ── detector: critical issues floor the risk band ────────────────────────────


def test_critical_findings_floor_the_risk_score():
    from types import SimpleNamespace

    from vfairness.preprocessing.bias_detection.detector import BiasDetector
    from vfairness.preprocessing.bias_detection.historical import (
        HistoricalRiskLevel,
    )

    det = BiasDetector.__new__(BiasDetector)
    crit = SimpleNamespace(risk_level=HistoricalRiskLevel.CRITICAL)
    low = SimpleNamespace(risk_level=HistoricalRiskLevel.LOW)
    # one critical among otherwise clean axes -> at least MEDIUM band
    score1 = det._calculate_overall_risk([crit, low], [], [], [])
    assert score1 >= 0.50
    # three criticals -> at least HIGH band (previously ~0.2, MINIMAL badge)
    score3 = det._calculate_overall_risk([crit, crit, crit], [], [], [])
    assert score3 >= 0.75


# ── exports: frozen API surface importable as documented ────────────────────


def test_frozen_symbols_importable_from_top_level():
    import vfairness

    for name in ("calibration_difference", "r2_parity_difference", "residual_bias"):
        assert hasattr(vfairness, name), name
        assert name in vfairness.__all__


# ── hardening: pulse tone machinery direct pins ──────────────────────────────


def test_pulse_tone_rank_and_worst_tone():
    from vfairness.operations.pulse.orchestrator import _tone_rank, _worst_tone

    assert _tone_rank("pass") < _tone_rank("warn") < _tone_rank("critical")
    assert _worst_tone("pass", "critical", "warn") == "critical"
    assert _worst_tone("pass", "pass") == "pass"


def test_pulse_disparity_tone_boundaries():
    from vfairness.operations.pulse.orchestrator import _disparity_tone

    # blatant confirmed disparity is critical; near-parity is pass
    bad = _disparity_tone(0.4, True, worst_rate=0.2, best_rate=0.5)
    good = _disparity_tone(0.99, False, worst_rate=0.59, best_rate=0.60)
    assert _tone_ok(bad, ("critical", "warn"))
    assert _tone_ok(good, ("pass",))


def _tone_ok(tone, allowed):
    return tone in allowed


# ── hardening: equalized odds FPR branch is exercised ────────────────────────


def test_equalized_odds_uses_fpr_branch():
    from vfairness.evaluation.vfairness_metrics.classification import (
        equalized_odds_difference,
    )

    # Equal TPR across groups (all positives predicted 1), unequal FPR:
    # group A: negatives all predicted 1 (FPR 1.0); group B: negatives all 0.
    y_true = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20)
    y_pred = np.concatenate(
        [
            np.ones(20),
            np.ones(20),  # group A: TPR 1.0, FPR 1.0
            np.ones(20),
            np.zeros(20),  # group B: TPR 1.0, FPR 0.0
        ]
    ).astype(int)
    g = np.array(["A"] * 40 + ["B"] * 40)
    val = equalized_odds_difference(y_true, y_pred, g, min_group_size=5)
    assert val == pytest.approx(1.0), (
        "equal TPRs with FPR gap 1.0 must yield equalized_odds 1.0 "
        "(the FPR branch, untested by the golden fixture, must count)"
    )
