"""Wave 15: the fabrication pattern across the feature-engineering, regression
and monitoring rendering files, scoped BY FILE rather than by reported site.

The pattern: a value that was never reported is replaced by a default (0, a
midpoint, a severity word, a boolean False) and the DEFAULT is then graded,
counted, coloured, plotted or sorted by as though it were a measurement.

Every test here pins one of the two halves of the rule:

  * a row that reported nothing gets no number, no badge, no colour, no plot
    point and no place in any count, sort or headline; and
  * a fabricated BREACH is exactly as wrong as a fabricated all-clear.

The CONTROL tests are load-bearing. A guard that withholds everything is not a
fix, it is a different fabrication, so each defect case is paired with a healthy
case asserting the chart still says what it always said.
"""

from types import SimpleNamespace

import pytest

from vfairness.rendering import (
    correlation_matrix_to_svg,
    drift_report_to_svg,
    intersectional_analysis_to_svg,
    intersectional_disparity_to_svg,
    monitoring_dashboard_to_svg,
    proxy_risk_to_svg,
    regression_fairness_to_svg,
    transformation_comparison_to_svg,
)

# ── shared fixtures ─────────────────────────────────────────────────────────

HEALTHY_MATRIX = {
    "income": {"income": 1.0, "zip": 0.72, "age": 0.10},
    "zip": {"income": 0.72, "zip": 1.0, "age": 0.05},
    "age": {"income": 0.10, "zip": 0.05, "age": 1.0},
}

GROUP_METRICS = {
    "Male": {"mae": 0.14, "rmse": 0.18, "r2": 0.82, "mean_residual": 0.03, "std_residual": 0.17},
    "Female": {"mae": 0.16, "rmse": 0.21, "r2": 0.79, "mean_residual": -0.04, "std_residual": 0.20},
}
ALL_PASSING = {
    "mae_parity": 0.02,
    "rmse_parity": 0.03,
    "mean_pred_diff": 0.01,
    "r2_parity": 0.04,
}


def _scale(**kw):
    base = dict(
        drift_score=0.1,
        ks_statistic=0.2,
        p_value=0.4,
        mean_shift=0.01,
        reference_mean=0.5,
        current_mean=0.51,
        drift_detected=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _drift(scales, **kw):
    base = dict(metric="dp_diff", overall_drift_score=0.1, drift_detected=False, scales=scales)
    base.update(kw)
    return SimpleNamespace(**base)


def _window(**kw):
    base = dict(
        metrics={"demographic_parity": 0.91},
        alerts={"demographic_parity": False},
        group_rates={},
        mmd_scores={},
        sample_count=500,
        timestamp=None,
        any_alert=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ── correlation_matrix: the crash, and the grid that contradicted itself ─────


class TestCorrelationMatrixCell:
    def test_a_pair_reported_as_none_does_not_take_the_render_down(self):
        """`float(corr_dict.get(rl, {}).get(cl, 0.0))` raised TypeError on a
        pair explicitly reported as None, so ONE unreportable pair cost the
        whole matrix. A crash is not the third state."""
        svg = correlation_matrix_to_svg(
            {"income": {"income": 1.0, "zip": None}, "zip": {"income": 0.72, "zip": 1.0}},
            threshold=0.5,
        )

        assert svg.startswith("\n") or svg.lstrip().startswith("<")
        assert "</svg>" in svg

    def test_an_absent_pair_is_not_drawn_as_a_measured_zero(self):
        svg = correlation_matrix_to_svg(
            {"income": {"income": 1.0, "zip": None}, "zip": {"income": 0.72, "zip": 1.0}},
            threshold=0.5,
        )

        assert ">N/A<" in svg
        # 0.00 is a correlation. The absent cell must not print one.
        assert ">0.00<" not in svg

    def test_the_grid_no_longer_contradicts_its_own_mirror_cell(self):
        """The reported defect: the missing (income, zip) cell read a measured
        0.00 while the mirror (zip, income) read 0.72, on the same grid."""
        svg = correlation_matrix_to_svg(
            {"income": {"income": 1.0}, "zip": {"income": 0.72, "zip": 1.0}}, threshold=0.5
        )

        assert ">0.72<" in svg
        assert ">N/A<" in svg
        assert ">0.00<" not in svg

    def test_an_uncomputed_pair_never_enters_the_high_correlation_count(self):
        both = correlation_matrix_to_svg(HEALTHY_MATRIX, threshold=0.5)
        one_missing = correlation_matrix_to_svg(
            {
                "income": {"income": 1.0, "zip": 0.72, "age": None},
                "zip": {"income": 0.72, "zip": 1.0, "age": 0.05},
                "age": {"income": None, "zip": 0.05, "age": 1.0},
            },
            threshold=0.5,
        )

        # The two real breaches (income/zip and its mirror) are still counted,
        # and the two absent cells raised the count by nothing.
        assert ">2</text>" in both
        assert ">2</text>" in one_missing

    def test_the_ungraded_count_is_on_the_canvas_in_the_headline_band(self):
        """Headline rule (c). The subtitle is drawn at y=122, above the badges."""
        svg = correlation_matrix_to_svg(
            {"income": {"income": 1.0}, "zip": {"income": 0.72, "zip": 1.0}}, threshold=0.5
        )

        assert "1 of 2 pair(s) not computed, so not tested." in svg
        headline_band = svg.split('y="122"')[1].split("</text>")[0]
        assert "not computed" in headline_band

    def test_a_ragged_matrix_no_longer_silently_loses_a_column(self):
        """`list(next(iter(corr.values())).keys())` took the columns of the
        FIRST row only, so a pair the caller really did report never reached the
        grid at all and left no trace of having been dropped."""
        svg = correlation_matrix_to_svg(
            {"income": {"income": 1.0}, "zip": {"income": 0.72, "zip": 1.0}}, threshold=0.5
        )

        assert "1 of 2 pair(s) not computed" in svg

    def test_a_matrix_whose_every_pair_is_blank_is_could_not_check(self):
        svg = correlation_matrix_to_svg(
            {"a": {"a": 1.0, "b": None}, "b": {"a": None, "b": 1.0}}, threshold=0.5
        )

        assert "COULD NOT CHECK" in svg
        assert "No high correlations detected" not in svg

    def test_an_empty_matrix_with_a_methods_dict_does_not_raise(self):
        """`next(iter(unique_methods))` on an empty set raised StopIteration,
        which killed the render before it could reach the could-not-check
        canvas that describes exactly this input."""
        svg = correlation_matrix_to_svg({}, methods={})

        assert "COULD NOT CHECK" in svg

    def test_a_cell_with_no_recorded_method_is_not_called_pearson(self):
        svg = correlation_matrix_to_svg(
            HEALTHY_MATRIX,
            methods={"income": {"income": "spearman", "zip": "spearman", "age": "spearman"}},
            threshold=0.5,
        )

        # Six off-diagonal cells were left unrecorded; three cells carried
        # spearman. Neither number may be inflated by the other.
        assert ">Spearman (3)<" in svg
        assert ">Not recorded (6)<" in svg
        # No legend entry claims Pearson. (The word itself also appears in the
        # static howToRead text, which is why this is anchored to the entry.)
        assert ">Pearson (" not in svg

    # CONTROLS: healthy input must render exactly as it always did.

    def test_a_fully_populated_matrix_still_counts_and_clears_normally(self):
        svg = correlation_matrix_to_svg(HEALTHY_MATRIX, threshold=0.5)

        assert ">0.72<" in svg
        assert "not computed" not in svg
        assert "COULD NOT CHECK" not in svg
        assert "high correlation" in svg

    def test_a_clean_matrix_still_earns_its_all_clear(self):
        svg = correlation_matrix_to_svg(HEALTHY_MATRIX, threshold=0.9)

        assert "No high correlations detected" in svg
        assert "COULD NOT CHECK" not in svg


# ── regression: the effect-size crash and its fabricated zero ────────────────


class TestRegressionEffectSizes:
    def test_a_none_effect_size_does_not_take_the_report_down(self):
        """`float(d.get("cohens_d", d.get("effect_size", 0)))` raised TypeError
        on a pair reporting None, losing every group and metric that HAD been
        measured beside it."""
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": None}]
        )

        assert "</svg>" in svg
        assert "M vs F" in svg

    def test_an_unreported_effect_size_is_not_plotted_on_the_zero_axis(self):
        """d = 0.00 in the PASS green, dotted ON the axis, is the strongest
        statement this panel can make about a pair, and it was produced by the
        ABSENCE of the comparison."""
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": None}]
        )

        assert "no effect size reported, not compared" in svg
        assert ">0.00<" not in svg
        assert 'cx="380"' not in svg

    def test_the_unmeasured_pair_is_named_in_the_headline_band(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": None}]
        )

        headline_band = svg.split('y="122"')[1].split("</text>")[0]
        assert "1 pair(s) not compared" in headline_band

    def test_an_unmeasured_pair_cannot_reach_the_large_effect_finding(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": None}]
        )

        assert "Large effect size detected" not in svg
        assert "reported no effect size" in svg

    def test_an_effect_size_object_missing_the_field_is_withheld_not_zeroed(self):
        class EffectSize:
            def to_dict(self):
                return {"pair": "A vs B"}

        svg = regression_fairness_to_svg(GROUP_METRICS, ALL_PASSING, [EffectSize()])

        assert "A vs B" in svg
        assert "no effect size reported, not compared" in svg

    # CONTROLS

    def test_a_measured_large_effect_is_still_plotted_and_still_flagged(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": 0.62}]
        )

        assert "Large effect size detected between groups" in svg
        assert ">0.62<" in svg
        assert "no effect size reported" not in svg

    def test_a_measured_zero_effect_is_still_plotted(self):
        """A d that really came out 0.00 is a measurement, and withholding it
        would be the opposite fabrication."""
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": 0.0}]
        )

        assert ">0.00<" in svg
        assert "no effect size reported" not in svg

    def test_a_fully_measured_report_still_reads_equitable(self):
        svg = regression_fairness_to_svg(
            GROUP_METRICS, ALL_PASSING, [{"pair": "M vs F", "cohens_d": 0.1}]
        )

        assert ">EQUITABLE<" in svg
        assert "not compared" not in svg


# ── transformation: the undefined ratio that was graded as a breach ──────────


class TestTransformationUndefinedReduction:
    # Five features that were never correlated with a protected attribute at
    # all, and one real proxy that the transformation halved. The five have no
    # proportional reduction to report: (0 - 0) / 0 is undefined. Under the old
    # `else 0` they entered the mean as five measured zeroes and pulled it to
    # 0.083, which bands as the red "Minimal", so a transformation that did
    # exactly what it was asked to do was graded a failure BY THE FEATURES IT
    # HAD NOTHING TO DO. The band over the graded subset alone is 0.5.
    _FIVE_FLAT_AND_ONE_REAL = (
        {"flat_a": 0.0, "flat_b": 0.0, "flat_c": 0.0, "flat_d": 0.0, "flat_e": 0.0, "zip": 0.8},
        {"flat_a": 0.0, "flat_b": 0.0, "flat_c": 0.0, "flat_d": 0.0, "flat_e": 0.0, "zip": 0.4},
    )

    def test_a_feature_that_started_at_zero_is_not_graded_zero_percent(self):
        """(0 - after) / 0 is undefined. The old `else 0` wrote that into the
        canvas AND into the mean behind the summary band, dragging the verdict
        towards the red "Minimal". A fabricated breach is no safer than a
        fabricated all-clear."""
        svg = transformation_comparison_to_svg(*self._FIVE_FLAT_AND_ONE_REAL)

        assert "Effective" in svg
        assert "Minimal" not in svg
        assert ">50%<" in svg

    def test_the_undefined_rows_are_counted_as_ungraded_on_the_canvas(self):
        svg = transformation_comparison_to_svg(*self._FIVE_FLAT_AND_ONE_REAL)

        assert "5 feature(s), not graded" in svg
        # and they are out of the improved denominator
        assert ">1 / 1<" in svg

    def test_a_table_of_nothing_but_undefined_rows_is_could_not_check(self):
        svg = transformation_comparison_to_svg({"flat": 0.0}, {"flat": 0.0})

        assert "COULD NOT CHECK" in svg
        assert "Minimal" not in svg

    # CONTROL

    def test_a_fully_measured_comparison_is_unchanged(self):
        svg = transformation_comparison_to_svg(
            {"zip": 0.72, "income": 0.45}, {"zip": 0.28, "income": 0.15}
        )

        assert "Effective" in svg
        assert ">2 / 2<" in svg
        assert "not graded" not in svg


# ── drift: the two-state verdict cell in the template ───────────────────────


class TestDriftVerdictThirdState:
    def test_a_scale_that_recorded_no_verdict_is_not_badged_stable(self):
        """`{% if s.drift_detected %}DRIFT{% else %}STABLE` is a two-state test,
        so a scale with no recorded verdict took the emerald STABLE pill: the
        most reassuring state the chart has, awarded for silence."""
        svg = drift_report_to_svg(_drift({"daily": _scale(drift_detected=None)}))

        assert "NOT RECORDED" in svg
        # Exactly one STABLE survives: the overall badge, whose verdict the
        # result object DID record. The scale's own pill is not it.
        assert svg.count(">STABLE<") == 1

    def test_a_report_with_no_overall_verdict_is_not_headlined_stable(self):
        svg = drift_report_to_svg(
            _drift({"daily": _scale()}, drift_detected=None),
        )

        assert "NOT RECORDED" in svg

    def test_the_unrecorded_scale_is_counted_in_the_headline_band(self):
        svg = drift_report_to_svg(_drift({"daily": _scale(drift_detected=None)}))

        headline_band = svg.split('y="122"')[1].split("</text>")[0]
        assert "did not fully report" in headline_band

    # CONTROLS

    def test_a_recorded_false_verdict_still_reads_stable(self):
        svg = drift_report_to_svg(_drift({"daily": _scale(drift_detected=False)}))

        # Both the overall badge and the scale's own pill, exactly as before.
        assert svg.count(">STABLE<") == 2
        assert "NOT RECORDED" not in svg

    def test_a_recorded_true_verdict_still_reads_drift(self):
        svg = drift_report_to_svg(
            _drift({"daily": _scale(drift_score=0.8, drift_detected=True)}, drift_detected=True)
        )

        assert ">DRIFT<" in svg
        assert ">DETECTED<" in svg
        assert "NOT RECORDED" not in svg


# ── monitoring dashboard: the OK badge handed out for silence ───────────────


class TestMonitoringStatusThirdState:
    def test_a_window_that_applied_no_guardrail_is_not_badged_ok(self):
        svg = monitoring_dashboard_to_svg(_window(metrics={"dp": 0.9}, alerts={}, any_alert=None))

        assert "NOT RECORDED" in svg
        assert ">OK<" not in svg
        assert "no guardrail applied" in svg

    def test_a_guardrail_result_recorded_as_none_is_not_an_ok_row(self):
        svg = monitoring_dashboard_to_svg(
            _window(metrics={"dp": 0.9}, alerts={"dp": None}, any_alert=None)
        )

        assert "NOT CHECKED" in svg
        assert ">OK<" not in svg

    def test_a_fired_alert_still_wins_even_with_no_overall_verdict(self):
        """The other direction: withholding the badge over a metric that
        visibly says ALERT would be the opposite fabrication."""
        svg = monitoring_dashboard_to_svg(
            _window(metrics={"dp": 0.4}, alerts={"dp": True}, any_alert=None)
        )

        assert ">ALERT<" in svg
        assert "NOT RECORDED" not in svg

    # CONTROLS

    def test_a_recorded_clean_window_still_reads_ok(self):
        svg = monitoring_dashboard_to_svg(_window())

        assert ">OK<" in svg
        assert "NOT RECORDED" not in svg
        assert "0 alerts" in svg

    def test_a_recorded_alerting_window_still_reads_alert(self):
        svg = monitoring_dashboard_to_svg(
            _window(metrics={"dp": 0.4}, alerts={"dp": True}, any_alert=True)
        )

        assert ">ALERT<" in svg
        assert "1 alert<" in svg


# ── intersectional disparity: the headline gap ──────────────────────────────


class TestIntersectionalDisparityGap:
    GROUPS = [
        {"group": "White Male", "positiveRate": 0.7},
        {"group": "Black Female", "positiveRate": 0.4},
    ]

    def _result(self, **kw):
        base = {
            "allGroups": self.GROUPS,
            "privilegedGroup": self.GROUPS[0],
            "disadvantagedGroup": self.GROUPS[1],
            "maxDisparity": 0.3,
            "disparitySeverity": "high",
        }
        base.update(kw)
        return base

    def test_a_non_numeric_gap_is_not_printed_as_repeated_text(self):
        """`max_disp * 100` on a string was Python string REPETITION, which put
        three hundred characters of "0.3" across the headline."""
        svg = intersectional_disparity_to_svg(self._result(maxDisparity="oops"))

        assert "oopsoops" not in svg
        assert "COULD NOT CHECK" in svg

    def test_a_gap_supplied_as_a_numeric_string_is_still_read(self):
        svg = intersectional_disparity_to_svg(self._result(maxDisparity="0.3"))

        assert "30.0" in svg
        assert "COULD NOT CHECK" not in svg

    def test_a_measured_single_attribute_zero_is_no_longer_suppressed(self):
        """The old expression tested the value for TRUTH, so a ground-truth
        disparity really measured at 0.0 (perfect single-attribute parity, the
        whole point of the comparison) was withheld as if never reported."""
        svg = intersectional_disparity_to_svg(self._result(maxGroundTruthDisparity=0.0))

        assert "0.0pp" in svg

    def test_a_finding_that_is_not_a_dict_does_not_take_the_render_down(self):
        svg = intersectional_disparity_to_svg(
            self._result(insights=["compound disadvantage"], findings=["not a dict"])
        )

        assert "compound disadvantage" in svg

    # CONTROL

    def test_a_fully_reported_result_is_unchanged(self):
        svg = intersectional_disparity_to_svg(self._result())

        assert "30.0" in svg
        assert ">HIGH<" in svg
        assert "COULD NOT CHECK" not in svg


# ── never crash on a partial row ────────────────────────────────────────────


class TestPartialInputNeverCrashesOrReturnsEmpty:
    @pytest.mark.parametrize(
        ("label", "call"),
        [
            ("correlation_matrix", lambda: correlation_matrix_to_svg(None)),
            ("correlation_matrix/methods", lambda: correlation_matrix_to_svg({}, methods={})),
            ("proxy_risk", lambda: proxy_risk_to_svg(None)),
            ("transformation", lambda: transformation_comparison_to_svg(None, None)),
            ("intersectional_analysis", lambda: intersectional_analysis_to_svg(None, "f")),
            (
                "intersectional_analysis/ragged",
                lambda: intersectional_analysis_to_svg({"matrix": {"a": None}}, "f"),
            ),
            ("intersectional_disparity", lambda: intersectional_disparity_to_svg(None)),
            (
                "regression/none-effect",
                lambda: regression_fairness_to_svg(
                    {"A": {"mae": 0.1}},
                    {"mae_parity": 0.02},
                    [{"pair": "A vs B", "cohens_d": None}],
                ),
            ),
            (
                "drift/none-verdict",
                lambda: drift_report_to_svg(_drift({"d": _scale(drift_detected=None)})),
            ),
            (
                "monitoring/none-alert",
                lambda: monitoring_dashboard_to_svg(_window(alerts={"dp": None})),
            ),
        ],
    )
    def test_it_renders_a_chart_rather_than_raising(self, label, call):
        svg = call()

        assert svg, f"{label} returned an empty string, which is not the third state"
        assert "</svg>" in svg, f"{label} returned something that is not an SVG document"
