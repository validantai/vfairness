"""Third-iteration audit, unit: ratio-sweep.

Pins the fix for the systemic ``"ratio" in <name>`` substring defect
(xai-render-vision-legal, CRITICAL class). The word "ratio" is a substring of
"cali[bratio]n_difference", so the shipped lower-is-better violation metrics
``calibration_difference`` / ``multicalibration`` / ``integrated_calibration_index``
were classified as higher-is-better RATIO metrics, inverting PASS/FAIL, colour,
severity, benchmark language and the "acceptable" band at every site.

A metric is a ratio iff its name ends with ``_ratio`` (demographic_parity_ratio,
disparate_impact_ratio, treatment_equality_ratio, ...). Each test below drives
BOTH the negative case (a value that used to give the wrong answer now gives the
right one) and a does-not-overcorrect case (a genuine ratio still uses the ratio
branch).
"""

import numpy as np
import pytest

THR = 0.05  # calibration threshold: lower is better
BIG = 0.9  # far above threshold -> must be a violation (FAIL/UNFAIR/red)
SMALL = 0.02  # below threshold -> must pass (FAIR/green)

CALIBRATION_KEYS = (
    "calibration_difference",
    "multicalibration",
    "integrated_calibration_index",
)
GENUINE_RATIO_KEYS = (
    "demographic_parity_ratio",
    "disparate_impact_ratio",
    "treatment_equality_ratio",
)


# ---------------------------------------------------------------------------
# Site 1: rendering/adapters.py::_metric_cards
# ---------------------------------------------------------------------------
def test_adapters_metric_cards_calibration_large_is_unfair():
    from vfairness.rendering.adapters import _metric_cards

    rep = {
        "metrics": {"calibration_difference": BIG},
        "thresholds_used": {"calibration_difference": THR},
    }
    card = _metric_cards(rep)[0]
    # Defect: was classified as a ratio -> 0.9 >= 0.05 -> "FAIR". Now UNFAIR.
    assert card["passed"] is False
    assert card["status"] == "UNFAIR"


def test_adapters_metric_cards_calibration_small_is_fair():
    from vfairness.rendering.adapters import _metric_cards

    rep = {
        "metrics": {"calibration_difference": SMALL},
        "thresholds_used": {"calibration_difference": THR},
    }
    card = _metric_cards(rep)[0]
    # Defect: was a ratio -> 0.02 >= 0.05 False -> "UNFAIR" (false FAIL). Now FAIR.
    assert card["passed"] is True
    assert card["status"] == "FAIR"


def test_adapters_metric_cards_genuine_ratio_not_overcorrected():
    from vfairness.rendering.adapters import _metric_cards

    rep = {
        "metrics": {"disparate_impact_ratio": 0.9},
        "thresholds_used": {"disparate_impact_ratio": 0.8},
    }
    card = _metric_cards(rep)[0]
    assert card["passed"] is True and card["status"] == "FAIR"  # 0.9 >= 0.8
    rep_fail = {
        "metrics": {"disparate_impact_ratio": 0.5},
        "thresholds_used": {"disparate_impact_ratio": 0.8},
    }
    card_fail = _metric_cards(rep_fail)[0]
    assert card_fail["passed"] is False and card_fail["status"] == "UNFAIR"  # 0.5 < 0.8


# ---------------------------------------------------------------------------
# Site 2: rendering/adapters_reporting.py::_is_ratio_metric
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", CALIBRATION_KEYS)
def test_adapters_reporting_calibration_is_not_ratio(key):
    from vfairness.rendering.adapters_reporting import _is_ratio_metric

    assert _is_ratio_metric(key) is False


@pytest.mark.parametrize("key", GENUINE_RATIO_KEYS)
def test_adapters_reporting_genuine_ratio_is_ratio(key):
    from vfairness.rendering.adapters_reporting import _is_ratio_metric

    assert _is_ratio_metric(key) is True


def test_adapters_reporting_disparate_impact_label_branches_preserved():
    """The explicit human/snake disparate-impact branches must stay."""
    from vfairness.rendering.adapters_reporting import _is_ratio_metric

    assert _is_ratio_metric("disparate impact") is True
    assert _is_ratio_metric("disparate_impact") is True


# ---------------------------------------------------------------------------
# Site 3: in_processing/analyzer.py::_first_violation
# ---------------------------------------------------------------------------
def test_first_violation_counts_calibration_difference():
    from vfairness.in_processing.analyzer import _first_violation

    # Defect: "ratio" substring made calibration_difference get skipped -> 0.0.
    assert _first_violation({"accuracy": 0.9, "calibration_difference": BIG}) == BIG
    assert _first_violation({"accuracy": 0.9, "multicalibration": 0.3}) == 0.3


def test_first_violation_still_skips_genuine_ratio():
    from vfairness.in_processing.analyzer import _first_violation

    # A 0.85 ratio is healthy, not a violation -> skipped. The sentinel for
    # "nothing violation-shaped here" is None since audit H-08 (2026-09-08);
    # what matters, then and now, is that it is not 0.85.
    assert _first_violation({"accuracy": 0.9, "disparate_impact_ratio": 0.85}) is None
    assert _first_violation({"accuracy": 0.9, "disparate_impact_ratio": 0.85}) != 0.85


# ---------------------------------------------------------------------------
# Site 4: evaluation/.../explainer.py::_evaluate_value + _get_benchmark_context
# ---------------------------------------------------------------------------
def _explainer():
    from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

    return FairExplAIner()


def test_explainer_calibration_large_is_severe_and_uses_difference_language():
    ex = _explainer()
    mdef = {"thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15}}
    text, severity = ex._evaluate_value("calibration_difference", BIG, THR, mdef)
    # Defect: ratio branch -> "Excellent ... near-perfect parity", severity info.
    assert severity in ("high", "critical")
    assert "difference" in text.lower()
    assert "ratio" not in text.lower()


def test_explainer_calibration_small_is_info():
    ex = _explainer()
    mdef = {"thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15}}
    _text, severity = ex._evaluate_value("calibration_difference", SMALL, THR, mdef)
    assert severity == "info"


def test_explainer_genuine_ratio_not_overcorrected():
    ex = _explainer()
    mdef = {"thresholds": {"excellent": 0.90, "acceptable": 0.80, "concerning": 0.70}}
    # Low ratio = severe disparity (ratio branch preserved).
    text_bad, sev_bad = ex._evaluate_value("disparate_impact_ratio", 0.5, 0.8, mdef)
    assert sev_bad == "critical" and "ratio" in text_bad.lower()
    # High ratio = healthy.
    _t, sev_good = ex._evaluate_value("disparate_impact_ratio", 0.95, 0.8, mdef)
    assert sev_good == "info"


def test_explainer_benchmark_context_direction():
    ex = _explainer()
    mdef = {"thresholds": {"excellent": 0.05, "acceptable": 0.10, "concerning": 0.15}}
    bench_diff = ex._get_benchmark_context("calibration_difference", BIG, mdef)
    # Difference metric -> "<=" direction, not the ">=" (80% rule) ratio phrasing.
    assert "≤" in bench_diff and "≥" not in bench_diff
    mdef_ratio = {"thresholds": {"excellent": 0.90, "acceptable": 0.80, "concerning": 0.70}}
    bench_ratio = ex._get_benchmark_context("disparate_impact_ratio", 0.9, mdef_ratio)
    assert "≥" in bench_ratio


# ---------------------------------------------------------------------------
# Site 5: evaluation/.../integrations.py::assert_fairness (line 716)
# ---------------------------------------------------------------------------
def _run_assert_fairness_with_metric(monkeypatch, metric_name, value, threshold):
    from vfairness.evaluation.vfairness_metrics import analyzer as anmod
    from vfairness.evaluation.vfairness_metrics.integrations import assert_fairness

    y_true = np.array([0, 1] * 100)
    y_pred = np.array([0, 1] * 100)
    sens = np.array(["A", "B"] * 100)

    def fake_compute(self, **kwargs):
        return {metric_name: value}

    monkeypatch.setattr(anmod.FairnessAnalyzer, "compute_all_metrics", fake_compute)
    return assert_fairness(
        y_true,
        y_pred,
        sens,
        metrics=[metric_name],
        thresholds={metric_name: threshold},
    )


def test_assert_fairness_calibration_large_raises(monkeypatch):
    from vfairness.evaluation.vfairness_metrics.integrations import FairnessAssertionError

    # Defect: ratio branch -> 0.9 < 0.05 False -> passed -> no raise. Now raises.
    with pytest.raises(FairnessAssertionError):
        _run_assert_fairness_with_metric(monkeypatch, "calibration_difference", BIG, THR)


def test_assert_fairness_calibration_small_passes(monkeypatch):
    result = _run_assert_fairness_with_metric(monkeypatch, "calibration_difference", SMALL, THR)
    assert result["calibration_difference"] == SMALL


def test_assert_fairness_genuine_ratio_not_overcorrected(monkeypatch):
    from vfairness.evaluation.vfairness_metrics.integrations import FairnessAssertionError

    # Ratio at/above minimum passes.
    _run_assert_fairness_with_metric(monkeypatch, "demographic_parity_ratio", 0.9, 0.8)
    # Ratio below minimum fails.
    with pytest.raises(FairnessAssertionError):
        _run_assert_fairness_with_metric(monkeypatch, "demographic_parity_ratio", 0.5, 0.8)


# ---------------------------------------------------------------------------
# Site 6a: evaluation/.../visualization.py::_add_metrics_panel colours (line 460)
# ---------------------------------------------------------------------------
def _panel_bar_color(metric, value, threshold):
    pytest.importorskip("plotly")  # plotly is an optional (viz) extra, absent in base CI
    from plotly.subplots import make_subplots

    from vfairness.evaluation.vfairness_metrics.visualization import (
        _add_metrics_panel,
        _get_palette,
    )

    pal = _get_palette("modern")
    fig = make_subplots(rows=1, cols=1)
    rep = {"metrics": {metric: value}, "thresholds_used": {metric: threshold}}
    _add_metrics_panel(fig, rep, pal, 1, 1)
    color = fig.data[0].marker.color
    # plotly stores a per-bar colour as a 1-tuple; unwrap it.
    if isinstance(color, (tuple, list)):
        color = color[0]
    return color, pal


def test_panel_calibration_large_is_danger():
    from vfairness.evaluation.vfairness_metrics.visualization import _get_palette

    pal = _get_palette("modern")
    color, _ = _panel_bar_color("calibration_difference", BIG, THR)
    assert color == pal["danger"]


def test_panel_calibration_small_is_success():
    from vfairness.evaluation.vfairness_metrics.visualization import _get_palette

    pal = _get_palette("modern")
    color, _ = _panel_bar_color("calibration_difference", SMALL, THR)
    assert color == pal["success"]


def test_panel_genuine_ratio_not_overcorrected():
    from vfairness.evaluation.vfairness_metrics.visualization import _get_palette

    pal = _get_palette("modern")
    color_ok, _ = _panel_bar_color("disparate_impact_ratio", 0.9, 0.8)
    assert color_ok == pal["success"]  # 0.9 >= 0.8
    color_bad, _ = _panel_bar_color("disparate_impact_ratio", 0.5, 0.8)
    assert color_bad == pal["danger"]  # 0.5 < 0.8


# ---------------------------------------------------------------------------
# Site 6b: visualization.py::_add_metrics_panel acceptable band (line 534)
# ---------------------------------------------------------------------------
def _panel_band_y0(metric, value, threshold):
    pytest.importorskip("plotly")  # plotly is an optional (viz) extra, absent in base CI
    from plotly.subplots import make_subplots

    from vfairness.evaluation.vfairness_metrics.visualization import (
        _add_metrics_panel,
        _get_palette,
    )

    pal = _get_palette("modern")
    fig = make_subplots(rows=1, cols=1)
    rep = {"metrics": {metric: value}, "thresholds_used": {metric: threshold}}
    _add_metrics_panel(fig, rep, pal, 1, 1)
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert rects, "expected an acceptable-region rectangle"
    return rects[0].y0, rects[0].y1


def test_panel_band_difference_metric_is_symmetric():
    # Difference metric band must be (-threshold, +threshold), not the ratio
    # band (threshold, >=1.0) that the substring defect produced.
    y0, y1 = _panel_band_y0("calibration_difference", BIG, THR)
    assert y0 == pytest.approx(-THR)
    assert y1 == pytest.approx(THR)


def test_panel_band_ratio_metric_starts_at_threshold():
    y0, _y1 = _panel_band_y0("disparate_impact_ratio", 0.9, 0.8)
    assert y0 == pytest.approx(0.8)  # higher-is-better band starts at the minimum


# ---------------------------------------------------------------------------
# Site 6c: visualization.py::plot_fairness_metrics matplotlib colours (line 1208)
# ---------------------------------------------------------------------------
def _mpl_bar_rgb(metric, value, threshold):
    pytest.importorskip("matplotlib")  # matplotlib is an optional (viz) extra, absent in base CI
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mc
    import matplotlib.pyplot as plt

    from vfairness.evaluation.vfairness_metrics.visualization import (
        _get_palette,
        plot_fairness_metrics,
    )

    pal = _get_palette("academic")
    rep = {"metrics": {metric: value}, "thresholds_used": {metric: threshold}}
    ax = plot_fairness_metrics(rep, show_ci=False, show_thresholds=False)
    rgb = ax.patches[0].get_facecolor()[:3]
    plt.close("all")
    return tuple(round(c, 4) for c in rgb), mc.to_rgb(pal["danger"]), mc.to_rgb(pal["success"])


def test_mpl_calibration_large_is_danger():
    rgb, danger, _success = _mpl_bar_rgb("calibration_difference", BIG, THR)
    assert rgb == tuple(round(c, 4) for c in danger)


def test_mpl_calibration_small_is_success():
    rgb, _danger, success = _mpl_bar_rgb("calibration_difference", SMALL, THR)
    assert rgb == tuple(round(c, 4) for c in success)


def test_mpl_genuine_ratio_not_overcorrected():
    rgb_ok, _d, success = _mpl_bar_rgb("disparate_impact_ratio", 0.9, 0.8)
    assert rgb_ok == tuple(round(c, 4) for c in success)
    rgb_bad, danger, _s = _mpl_bar_rgb("disparate_impact_ratio", 0.5, 0.8)
    assert rgb_bad == tuple(round(c, 4) for c in danger)


# ---------------------------------------------------------------------------
# Site 7: operations/cicd/gate.py::ModelFairnessGate (the deployment gate)
#
# The gate carried the OPPOSITE half of the same defect: it had no direction
# handling at all, just `abs(value) > threshold` commented "for difference
# metrics, lower is better". The calibration family was therefore graded
# correctly here by accident, while the whole ratio family was inverted, so this
# section drives both halves. Full four-row four-fifths table plus the
# fail-closed cases live in tests/test_gate_direction.py.
# ---------------------------------------------------------------------------
def _gate(metric, threshold):
    from vfairness.operations.cicd.gate import ModelFairnessGate

    return ModelFairnessGate(
        metrics=[metric],
        thresholds={metric: threshold},
        blocking_metrics=[metric],
    )


@pytest.mark.parametrize("key", CALIBRATION_KEYS)
def test_gate_calibration_stays_a_difference_metric(key):
    # Lower is better: BIG must block, SMALL must approve. If the ratio branch
    # ever reaches these names, both assertions invert.
    assert _gate(key, THR).evaluate_from_metrics({key: BIG}).approved is False
    assert _gate(key, THR).evaluate_from_metrics({key: SMALL}).approved is True


@pytest.mark.parametrize("key", GENUINE_RATIO_KEYS)
def test_gate_genuine_ratio_uses_the_ratio_branch(key):
    # Defect: `abs(0.5) > 0.8` is False, so a ratio at 0.5 (a clear four-fifths
    # violation) was APPROVED for deployment, while 0.9 was BLOCKED.
    assert _gate(key, 0.8).evaluate_from_metrics({key: 0.5}).approved is False
    assert _gate(key, 0.8).evaluate_from_metrics({key: 0.9}).approved is True


def test_gate_unknown_metric_direction_fails_closed():
    decision = _gate("mystery_metric", 0.8).evaluate_from_metrics({"mystery_metric": 0.5})
    assert decision.approved is False


# ---------------------------------------------------------------------------
# Site 8: operations/cicd/testing.py::FairnessTestSuite._test_metrics
# ---------------------------------------------------------------------------
def _suite_result(metric, value, threshold):
    from vfairness.operations.cicd.testing import FairnessTestSuite

    suite = FairnessTestSuite(
        protected_attributes=["group"],
        metrics=[metric],
        thresholds={metric: threshold},
        compute_metrics_fn=lambda y_true, y_pred, prot: {metric: value},
    )
    results = suite.test_predictions(
        np.array([0, 1] * 50),
        np.array([0, 1] * 50),
        np.array(["A", "B"] * 50),
        raise_on_failure=False,
    )
    assert len(results) == 1
    return results[0]


@pytest.mark.parametrize("key", CALIBRATION_KEYS)
def test_suite_calibration_stays_a_difference_metric(key):
    assert _suite_result(key, BIG, THR).status.value == "failed"
    assert _suite_result(key, SMALL, THR).status.value == "passed"


@pytest.mark.parametrize("key", GENUINE_RATIO_KEYS)
def test_suite_genuine_ratio_uses_the_ratio_branch(key):
    # Defect: `abs(0.5) <= 0.8` is True, so a four-fifths violation was reported
    # as a PASSED fairness test.
    assert _suite_result(key, 0.5, 0.8).status.value == "failed"
    assert _suite_result(key, 0.9, 0.8).status.value == "passed"


def test_suite_ratio_failure_message_says_below_not_exceeds():
    result = _suite_result("disparate_impact_ratio", 0.5, 0.8)
    assert "below the required minimum" in result.message
    assert "exceeds" not in result.message


def test_suite_unknown_metric_direction_is_skipped_not_passed():
    # Three states, never two: a metric whose direction we cannot resolve was
    # not checked, so it is neither PASSED nor FAILED.
    result = _suite_result("mystery_metric", 0.5, 0.8)
    assert result.status.value == "skipped"
    assert "no known better-direction" in result.message


def test_suite_raises_on_a_ratio_violation_when_asked_to():
    from vfairness.operations.cicd.testing import (
        FairnessAssertionError,
        FairnessTestSuite,
    )

    suite = FairnessTestSuite(
        protected_attributes=["group"],
        metrics=["disparate_impact_ratio"],
        thresholds={"disparate_impact_ratio": 0.8},
        compute_metrics_fn=lambda y_true, y_pred, prot: {"disparate_impact_ratio": 0.5},
    )
    with pytest.raises(FairnessAssertionError):
        suite.test_predictions(
            np.array([0, 1] * 50),
            np.array([0, 1] * 50),
            np.array(["A", "B"] * 50),
            raise_on_failure=True,
        )


# ---------------------------------------------------------------------------
# Site 9: rendering/adapters_post_processing.py::fairness_detailed_report_to_svg
# ---------------------------------------------------------------------------
def _detailed_report_svg(metric, value, threshold):
    from vfairness.rendering.adapters_post_processing import (
        fairness_detailed_report_to_svg,
    )

    return fairness_detailed_report_to_svg(
        {
            "assessment": {"fairness_score": 0.9},
            "metrics": {metric: {"value": value, "threshold": threshold}},
        }
    )


def test_detailed_report_genuine_ratio_badge_direction():
    # Defect: `abs(value) <= threshold` painted a green PASS badge for a ratio of
    # 0.0 (the protected group is never selected) and a red FAIL for 1.0.
    assert "PASS" not in _detailed_report_svg("disparate_impact_ratio", 0.0, 0.8)
    assert "PASS" in _detailed_report_svg("disparate_impact_ratio", 1.0, 0.8)


def test_detailed_report_calibration_stays_a_difference_metric():
    assert "PASS" not in _detailed_report_svg("calibration_difference", BIG, THR)
    assert "PASS" in _detailed_report_svg("calibration_difference", SMALL, THR)


def test_detailed_report_unknown_direction_is_not_a_silent_pass():
    # The SVG template has only PASS/FAIL badges, so the row fails closed AND
    # says COULD NOT CHECK in its own interpretation line.
    with pytest.warns(UserWarning, match="no known better-direction"):
        svg = _detailed_report_svg("mystery_metric", 0.05, 0.1)
    assert "PASS" not in svg
    assert "COULD NOT CHECK" in svg
