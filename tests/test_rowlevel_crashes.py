"""A partial ROW must be answered, not crashed on and not silently swallowed.

Wave 14 of the invented-analysis sweep, and the last class of hole in it. The
earlier waves closed the charts that ANSWERED a partial row dishonestly (a
default graded as a measurement, a sentinel painted green, an absent verdict
drawn as FAIL). These five adapters did something else with the same input: they
raised, or they quietly stopped being the chart they claim to be. Neither is one
of the three states. A traceback certifies nothing and cannot be read; a card
that looks like output but carries no verdict is worse, because it IS read.

DEFECT 1, a crash. ``tradeoff_analysis_to_svg`` reached for ``r["accuracy"]``
and raised ``KeyError('accuracy')`` for a sweep in which one configuration had
not finished, so a partial run produced no chart at all. Three lines below, the
marked optima took the OTHER wrong branch for the same absent field:
``best_fair.get("accuracy", 0)`` fed the axis range from a default (moving every
real point on the chart) and ``best_fair["accuracy"]`` then raised. Executed on
the tree at 927abdf.

DEFECT 2, a fabricated breach. ``training_analysis_report_to_svg`` built its
method rows by hand instead of through ``_method_state``, so the hardening that
closed this exact defect on ``training_report`` and ``method_comparison`` never
reached this page: ``MethodComparison("adversarial", 0.77, 0.05, None)``
rendered a red FAIL chip and the red row stripe for a method whose constraint
was never evaluated, beside a genuine PASS. Nothing distinguished the invented
failure from the measured one.

DEFECT 3, a silent degradation. ``threshold_optimization_to_svg`` answered a
render failure with a plain card headed with the report's own title, listing the
group thresholds and stating NONE of the three results the report exists for
(feasibility, disparity reduction, accuracy change). It never passed through
``inject_accessibility``, so it had no role, no title and no <desc> either. One
group supplied as ``{"name": "female"}`` is enough to reach it, and the only
announcement was a UserWarning, which is filtered by default in exactly the
batch pipelines that call this.

DEFECT 4, malformed output. ``experiment_recommendation.svg`` printed the
literal string "None" in the value column of the TRADE-OFFS table for a row
supplied as ``{"latency": None}``: the adapter normalises with ``str(v)``, so
the word reached the canvas, the <desc> and any export as though it were a
recorded finding.

DEFECT 5, more crashes, pre-existing and identical in both builds.
``alert_timeline_to_svg`` raised ``AttributeError`` on ``wm.alerts.items()`` for
a window whose alerts record is absent, taking down the eleven fully recorded
windows beside it; ``monitoring_dashboard_to_svg`` raised on the same record and
on a metric value it could not call ``abs()`` on; ``drift_report_to_svg`` read
``dr.metric`` inside the very branch whose job is to say that nothing was
measured.

The over-correction control is ``TestHealthyInputIsUnchanged`` at the bottom.
Beyond it, all eight healthy renders were rasterised against the committed tree
at 927abdf and compared byte for byte, outside pytest so that the
``pythonpath = ["src"]`` setting in pyproject could not silently load the new
code for both halves; the build actually loaded was asserted each way by its
``__file__``. All eight SVGs were byte-identical and all eight PNGs were
pixel-identical.
"""

from datetime import datetime

import pytest

from vfairness.rendering.adapters_experimentation import experiment_recommendation_to_svg
from vfairness.rendering.adapters_monitoring import (
    alert_timeline_to_svg,
    drift_report_to_svg,
    monitoring_dashboard_to_svg,
)
from vfairness.rendering.adapters_post_processing import threshold_optimization_to_svg
from vfairness.rendering.adapters_training import (
    _generate_fallback_svg,
    _method_state,
    tradeoff_analysis_to_svg,
    training_analysis_report_to_svg,
)

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

COULD_NOT_CHECK = "COULD NOT CHECK"
NOT_CHECKED = "NOT CHECKED"


def _plot_dot_count(svg: str) -> int:
    """Dots in the scatter panel.

    Keyed on the data point's own opacity, because the LEGEND also draws two
    r="5" circles and counting those would hide a missing point.
    """
    return svg.count('opacity="0.85"')


class _Rec:
    def __init__(self, method="reduction", priority="high"):
        self.recommended_method = method
        self.priority = priority
        self.rationale = "best trade-off"
        self.alternative_methods = []


class _Method:
    def __init__(self, name, accuracy, violation, satisfied, time=1.0):
        self.method_name = name
        self.accuracy = accuracy
        self.fairness_violation = violation
        self.constraint_satisfied = satisfied
        self.training_time = time


class _Report:
    """The attribute surface training_analysis_report_to_svg reads."""

    def __init__(self, methods, tradeoff=None):
        self.timestamp = "2026-08-28T12:00:00"
        self.data_info = {"n_samples": 1000, "n_groups": 2, "attribute_name": "gender"}
        self.task_type = "classification"
        self.baseline_metrics = {
            "accuracy": 0.81,
            "fairness_violation": 0.11,
            "constraint_satisfied": False,
        }
        self.fairness_analysis = {
            "constraint_type": "demographic_parity",
            "base_rate_disparity": 0.09,
            "group_statistics": {},
        }
        self.method_comparisons = methods
        self.recommendation = _Rec()
        self.tradeoff_analysis = tradeoff or {}
        self.critical_issues = []
        self.action_items = []


class _Window:
    """A monitoring window that reported only part of itself."""

    def __init__(
        self, alerts, metrics=None, timestamp=None, sample_count=100, rates=None, mmd=None
    ):
        self.batch_id = "b"
        self.timestamp = timestamp
        self.sample_count = sample_count
        self.metrics = metrics if metrics is not None else {"demographic_parity": 0.02}
        self.group_rates = rates if rates is not None else {}
        self.alerts = alerts
        self.mmd_scores = mmd if mmd is not None else {}
        self.any_alert = (
            bool(alerts) and any(alerts.values()) if isinstance(alerts, dict) else False
        )


class _Scale:
    def __init__(self, score=None):
        self.drift_score = score
        self.ks_statistic = None
        self.p_value = None
        self.mean_shift = None
        self.reference_mean = None
        self.current_mean = None
        self.drift_detected = None


class _Drift:
    """A drift result that carries no metric name."""

    def __init__(self, scales):
        self.scales = scales
        self.overall_drift_score = None
        self.drift_detected = None


HEALTHY_TRADEOFF = {
    "all_results": [
        {"accuracy": 0.90, "violation": 0.12, "lambda": 0.0, "satisfied": False},
        {"accuracy": 0.85, "violation": 0.03, "lambda": 1.0, "satisfied": True},
    ],
    "pareto_frontier": [{"accuracy": 0.90, "violation": 0.12}],
    "best_fair": {"accuracy": 0.85, "violation": 0.03},
    "best_accurate": {"accuracy": 0.90, "violation": 0.12},
}


# ── DEFECT 1: the trade-off scatter ────────────────────────────────────────


class TestTradeoffPartialRow:
    PARTIAL = {
        "all_results": [
            {"accuracy": 0.90, "violation": 0.12, "lambda": 0.0, "satisfied": False},
            {"lambda": 1.5},
        ]
    }

    def test_configuration_without_a_coordinate_does_not_raise(self):
        # KeyError('accuracy') at adapters_training.py, the whole render lost
        # because ONE row of the sweep had not finished.
        svg = tradeoff_analysis_to_svg(self.PARTIAL)
        assert svg.startswith("<svg") or "<svg" in svg

    def test_configuration_without_a_coordinate_gets_no_dot(self):
        svg = tradeoff_analysis_to_svg(self.PARTIAL)
        assert _plot_dot_count(svg) == 1, "the unfinished configuration was plotted"

    def test_configuration_without_a_verdict_gets_no_dot(self):
        # The dot's FILL is the constraint verdict, and the template's test is
        # two-state: None would be painted as "constraint violated" red.
        svg = tradeoff_analysis_to_svg(
            {
                "all_results": [
                    {"accuracy": 0.90, "violation": 0.12, "lambda": 0.0, "satisfied": False},
                    {"accuracy": 0.85, "violation": 0.03, "lambda": 1.0, "satisfied": None},
                ]
            }
        )
        assert _plot_dot_count(svg) == 1

    def test_the_withheld_count_is_on_the_canvas_in_the_headline_band(self):
        svg = tradeoff_analysis_to_svg(self.PARTIAL)
        assert "1 of 2 points not plotted" in svg
        # Headline rule (c): stated where the reader of the verdict sees it,
        # not at the foot. y=96 is this template's title line.
        headline = svg.split('y="96"', 1)[1][:200]
        assert "not plotted" in headline

    def test_a_marked_optimum_without_a_coordinate_is_not_drawn(self):
        svg = tradeoff_analysis_to_svg(
            {
                "all_results": [
                    {"accuracy": 0.90, "violation": 0.12, "lambda": 0.0, "satisfied": True}
                ],
                "best_fair": {"lambda": 2.0},
            }
        )
        assert "Best Fair" not in svg, "an optimum was labelled at a defaulted position"

    def test_a_marked_optimum_without_a_coordinate_does_not_move_the_axis(self):
        one_point = {
            "all_results": [{"accuracy": 0.90, "violation": 0.12, "lambda": 0.0, "satisfied": True}]
        }
        with_absent_optimum = dict(one_point, best_fair={"lambda": 2.0})
        # `.get("accuracy", 0)` used to push the x axis down to 0.00 for every
        # real point on the chart.
        assert "0.85" in tradeoff_analysis_to_svg(one_point)
        assert "0.85" in tradeoff_analysis_to_svg(with_absent_optimum)

    def test_an_infinite_coordinate_does_not_take_the_render_down(self):
        # OverflowError out of _axis_ranges (math.ceil(inf / 0.05)), raised
        # before the try block that could have explained it.
        svg = tradeoff_analysis_to_svg(
            {
                "all_results": [
                    {"accuracy": float("inf"), "violation": 0.1, "lambda": 0.0, "satisfied": True},
                    {"accuracy": 0.85, "violation": 0.03, "lambda": 1.0, "satisfied": True},
                ]
            }
        )
        assert _plot_dot_count(svg) == 1
        assert "1 of 2 points not plotted" in svg

    def test_rows_supplied_but_none_plottable_is_could_not_check(self):
        svg = tradeoff_analysis_to_svg({"all_results": [{"lambda": 0.1}, {"lambda": 0.2}]})
        assert COULD_NOT_CHECK in svg or NOT_CHECKED in svg
        assert "None of the 2 supplied trade-off row(s)" in svg

    def test_a_render_failure_is_written_out_as_could_not_check(self, tmp_path, monkeypatch):
        # NEVER "" from this branch: the caller that passed save_path kept the
        # PREVIOUS run's file and was told nothing.
        import vfairness.rendering.adapters_training as mod

        def _boom(*_a, **_k):
            raise ValueError("template exploded")

        monkeypatch.setattr(mod, "render_svg", _boom)
        out = tmp_path / "t.svg"
        out.write_text("PREVIOUS RUN")
        with pytest.warns(UserWarning):
            svg = tradeoff_analysis_to_svg(HEALTHY_TRADEOFF, save_path=str(out))
        assert svg != ""
        assert NOT_CHECKED in svg and COULD_NOT_CHECK in svg
        assert out.read_text() == svg


# ── DEFECT 2: the comprehensive training report ────────────────────────────


class TestTrainingAnalysisReportUngradedMethod:
    UNGRADED = [
        _Method("reduction", 0.79, 0.03, True, 12.0),
        _Method("adversarial", 0.77, 0.05, None, 30.0),
    ]

    def test_an_unevaluated_method_is_not_drawn_as_a_failure(self):
        svg = training_analysis_report_to_svg(_Report(self.UNGRADED))
        assert ">PASS<" in svg, "the graded method lost its verdict"
        assert ">FAIL<" not in svg, "a method whose constraint was never evaluated was FAILED"

    def test_an_unevaluated_method_gets_no_row_stripe(self):
        svg = training_analysis_report_to_svg(_Report(self.UNGRADED))
        # The failure stripe is the red row background of the methods table.
        assert svg.count('fill="#fef2f2" opacity="0.3"') == 0

    def test_the_ungraded_count_is_on_the_canvas_in_the_headline_band(self):
        svg = training_analysis_report_to_svg(_Report(self.UNGRADED))
        assert "1 of 2 method(s) ungraded" in svg
        subtitle = svg.split('y="122"', 1)[1][:300]
        assert "ungraded" in subtitle

    def test_a_measured_failure_is_still_drawn_as_a_failure(self):
        # The direction that must NOT be lost: False is a verdict.
        svg = training_analysis_report_to_svg(
            _Report([_Method("adversarial", 0.77, 0.09, False, 30.0)])
        )
        assert ">FAIL<" in svg

    def test_a_tradeoff_row_without_a_coordinate_is_not_plotted(self):
        graded_only = _Report(
            self.UNGRADED[:1],
            tradeoff={
                "all_results": [
                    {"accuracy": 0.90, "violation": 0.12, "satisfied": False},
                    {"lambda": 1.0},
                ]
            },
        )
        svg = training_analysis_report_to_svg(graded_only)
        assert "1 trade-off point(s) not plotted" in svg

    def test_a_tradeoff_row_without_a_verdict_is_not_painted_red(self):
        # `.get("satisfied", False)` painted it in the "unfair" colour.
        svg = training_analysis_report_to_svg(
            _Report(
                self.UNGRADED[:1],
                tradeoff={"all_results": [{"accuracy": 0.90, "violation": 0.12}]},
            )
        )
        assert "1 trade-off point(s) not plotted" in svg

    def test_the_fallback_card_survives_a_method_with_no_accuracy(self):
        # `{...:.3f}` on None raised inside the very branch whose job is to
        # survive a template failure, and "✗" stamped a fabricated failure.
        data = {
            "title": "t",
            "methods": [_method_state(_Method("adversarial", None, None, None))],
            "recommendation": {"method": "N/A", "rationale": ""},
        }
        card = _generate_fallback_svg(data)
        assert "✗" not in card
        assert "not evaluated" in card
        assert "not measured" in card


# ── DEFECT 3: the threshold-optimisation fallback ──────────────────────────


class TestThresholdOptimisationFallback:
    BARE_GROUP = {
        "timestamp": "2026-08-28 12:00",
        "groups": [{"name": "female"}, {"name": "male", "threshold": 0.62}],
        "original_disparity": 0.13,
        "optimized_disparity": 0.02,
        "is_feasible": True,
    }

    def test_a_render_failure_says_could_not_check_on_the_canvas(self):
        with pytest.warns(UserWarning):
            svg = threshold_optimization_to_svg(self.BARE_GROUP)
        assert NOT_CHECKED in svg, "the fallback still reads as the report"
        assert COULD_NOT_CHECK in svg, "the accessible description claimed nothing was wrong"

    def test_the_fallback_states_no_result_it_did_not_draw(self):
        with pytest.warns(UserWarning):
            svg = threshold_optimization_to_svg(self.BARE_GROUP)
        assert "FEASIBLE" not in svg
        assert "Group Thresholds:" not in svg

    def test_the_caller_that_passed_save_path_gets_the_same_answer(self, tmp_path):
        out = tmp_path / "thr.svg"
        out.write_text("PREVIOUS RUN")
        with pytest.warns(UserWarning):
            svg = threshold_optimization_to_svg(self.BARE_GROUP, save_path=str(out))
        assert out.read_text() == svg
        assert NOT_CHECKED in out.read_text()


# ── DEFECT 4: the trade-off row that carries no value ──────────────────────


class TestExperimentRecommendationTradeoffs:
    def test_the_literal_word_none_is_not_a_trade_off(self):
        svg = experiment_recommendation_to_svg(
            {
                "decision": "DEPLOY_TREATMENT",
                "confidence": 0.9,
                "reasoning": ["effect is large"],
                "trade_offs": {"latency": None, "cost": "+3%"},
            }
        )
        assert ">None<" not in svg
        assert ">not reported<" in svg
        assert ">+3%<" in svg, "the reported trade-off lost its value"

    def test_an_empty_value_is_reported_as_absent(self):
        svg = experiment_recommendation_to_svg(
            {"decision": "KEEP_CONTROL", "confidence": 0.5, "trade_offs": {"latency": ""}}
        )
        assert ">not reported<" in svg


# ── DEFECT 5: the monitoring adapters ──────────────────────────────────────


class TestMonitoringPartialRow:
    GOOD = _Window(
        {"demographic_parity": True},
        timestamp=datetime(2026, 8, 26, 9, 0),
        rates={"gender": {"f": 0.3}},
        mmd={"gender": 0.04},
    )

    def test_a_window_with_no_alert_record_does_not_take_the_timeline_down(self):
        bad = _Window(None, timestamp=datetime(2026, 8, 27, 9, 0))
        svg = alert_timeline_to_svg([self.GOOD, bad])
        assert "NOT MONITORED" in svg
        assert "ALERT" in svg, "the fully recorded window lost its verdict"

    def test_a_window_with_no_alert_record_is_not_counted_as_clean(self):
        bad = _Window(None, timestamp=datetime(2026, 8, 27, 9, 0))
        svg = alert_timeline_to_svg([self.GOOD, bad])
        assert "1 with alerts · 0 clean · 1 not monitored" in svg

    def test_a_window_with_no_timestamp_or_sample_count_prints_no_none(self):
        # Beside a recorded window, so the table is drawn: a history in which
        # EVERY window is unmonitored is could-not-check as a whole.
        svg = alert_timeline_to_svg([self.GOOD, _Window({}, timestamp=None, sample_count=None)])
        assert ">None<" not in svg
        assert "not reported" in svg

    def test_an_absent_history_is_not_a_crash(self):
        svg = alert_timeline_to_svg(None)
        assert COULD_NOT_CHECK in svg or NOT_CHECKED in svg

    def test_a_dashboard_window_with_no_alert_record_does_not_raise(self):
        svg = monitoring_dashboard_to_svg(_Window(None, timestamp=datetime(2026, 8, 28, 9, 0)))
        assert "NOT CHECKED" in svg

    def test_a_metric_with_no_value_gets_no_number(self):
        svg = monitoring_dashboard_to_svg(
            _Window(
                {"demographic_parity": False},
                metrics={"demographic_parity": 0.02, "equal_opportunity": None},
                timestamp=datetime(2026, 8, 28, 9, 0),
            )
        )
        assert ">0.020<" in svg, "the measured metric lost its value"
        assert ">0.000<" not in svg, "an unreported metric was printed as zero"

    def test_a_group_rate_that_was_not_recorded_is_not_zero_percent(self):
        svg = monitoring_dashboard_to_svg(
            _Window(
                {"demographic_parity": False},
                timestamp=datetime(2026, 8, 28, 9, 0),
                rates={"gender": {"f": 0.31, "m": None}},
            )
        )
        assert ">31.0%<" in svg
        assert ">0.0%<" not in svg

    def test_a_drift_result_with_no_metric_name_does_not_raise(self):
        # The AttributeError was inside the could-not-check branch itself.
        svg = drift_report_to_svg(_Drift({}))
        assert COULD_NOT_CHECK in svg or NOT_CHECKED in svg
        svg2 = drift_report_to_svg(_Drift({"short": _Scale(None)}))
        assert COULD_NOT_CHECK in svg2 or NOT_CHECKED in svg2


# ── The over-correction control ────────────────────────────────────────────


class TestHealthyInputIsUnchanged:
    """Fully reported input keeps every number, verdict and colour it had."""

    def test_a_fully_reported_sweep_plots_every_configuration(self):
        svg = tradeoff_analysis_to_svg(HEALTHY_TRADEOFF)
        assert _plot_dot_count(svg) == 2
        assert "Best Fair" in svg and "Best Accurate" in svg
        assert "not plotted" not in svg
        assert COULD_NOT_CHECK not in svg

    def test_a_fully_graded_report_keeps_its_table_and_subtitle(self):
        svg = training_analysis_report_to_svg(
            _Report(
                [
                    _Method("reduction", 0.79, 0.03, True, 12.0),
                    _Method("adversarial", 0.77, 0.09, False, 30.0),
                ],
                tradeoff=HEALTHY_TRADEOFF,
            )
        )
        assert ">PASS<" in svg and ">FAIL<" in svg
        assert "ungraded" not in svg and "not plotted" not in svg
        assert "demographic_parity constraint" in svg

    def test_a_complete_threshold_result_still_renders_the_chart(self):
        svg = threshold_optimization_to_svg(
            {
                "groups": [
                    {
                        "name": "female",
                        "threshold": 0.58,
                        "original_rate": 0.31,
                        "optimized_rate": 0.40,
                        "size": 400,
                    }
                ],
                "original_disparity": 0.13,
                "optimized_disparity": 0.02,
                "original_accuracy": 0.81,
                "optimized_accuracy": 0.79,
                "is_feasible": True,
                "constraint_type": "demographic_parity",
            }
        )
        assert "FEASIBLE" in svg
        assert NOT_CHECKED not in svg

    def test_a_complete_recommendation_keeps_both_trade_offs(self):
        svg = experiment_recommendation_to_svg(
            {
                "decision": "DEPLOY_TREATMENT",
                "confidence": 0.9,
                "trade_offs": {"latency": "+12ms", "cost": "+3%"},
            }
        )
        assert ">+12ms<" in svg and ">+3%<" in svg
        assert "not reported" not in svg

    def test_a_fully_recorded_window_is_still_clean_or_alerting(self):
        clean = _Window({"demographic_parity": False}, timestamp=datetime(2026, 8, 26, 9, 0))
        svg = alert_timeline_to_svg([clean, self.GOOD])
        assert "1 with alerts · 1 clean" in svg
        assert "not monitored" not in svg

    GOOD = _Window(
        {"demographic_parity": True},
        timestamp=datetime(2026, 8, 26, 9, 0),
        rates={"gender": {"f": 0.3}},
        mmd={"gender": 0.04},
    )

    def test_a_fully_recorded_dashboard_keeps_its_numbers(self):
        svg = monitoring_dashboard_to_svg(self.GOOD)
        assert ">0.020<" in svg
        assert ">30.0%<" in svg
        assert "n/a" not in svg
