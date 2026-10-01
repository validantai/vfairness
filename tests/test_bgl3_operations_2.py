"""BGL3 operations-2: monitoring and CI/CD gates, a verdict nobody measured.

Fourteen units across ``operations/cicd/precommit.py``,
``operations/cicd/testing.py`` and ``operations/monitoring/tracker.py``, each
executed on 2026-09-27 on an input where the quantity it reports genuinely does
not exist, and again on a measurable input. Six defects, each with a refusal pin
and at least one over-correction control asserting a real measured value exactly,
so a fix that simply refuses everything fails here too.

op2-1. ``TemporalFairnessAnalyzer.detect_trend`` answered ``("decreasing", nan)``
       whenever the OLS slope came out non-finite, because every comparison in
       the direction branch is False for a nan and the final ``else`` is
       "decreasing". For a lower-is-better metric, which is most of what this
       analyzer tracks, "decreasing" is the RECOVERING reading: the all-clear.
       Measured on 14 consecutive days of ``demographic_parity`` rising
       0.01 -> 0.14 with day 8 set to ``inf``: ``("decreasing", nan)``,
       ``get_metric_summary`` published ``trend_direction="decreasing"`` beside
       ``trend_slope=nan``, and the only notice was numpy's own "invalid value
       encountered in subtract". A ``NaT`` on the date axis gave the same
       ``("decreasing", nan)`` with NO warning at all, because ``denom`` is nan
       there and ``nan <= 0`` is False, so the existing 0/0 guard never fires.

op2-2. ``TemporalFairnessAnalyzer.get_explanation`` reported severity "info" for
       a report in which NO verdict was reached, because its escalation was
       keyed on the descriptive MEAN rather than on the two things it grades.
       Measured on a 4-day analyzer swinging 0.02 / 0.90 / 0.03 / 0.85 (under
       the 5-row trend floor and the 14-row weekday floor, so the trend was
       "not_assessed" and the weekday verdict None): severity "info", byte
       identical to a measured 20-day flat healthy series, because mean=0.4500
       counted as "something was measured".

op2-3. ``FairnessMonitor.compute_equal_opportunity`` returned a SILENT nan when
       fewer than two groups had a defined TPR, although its own docstring
       promised "``np.nan``, with a ``UserWarning``". Measured on 40 M rows all
       labelled 1 and 40 F rows all labelled 0, both over min_samples=30: nan
       and ZERO warnings, while ``compute_equalized_odds`` on the SAME frame
       warned that its TPR arm held one defined rate.

op2-4. ``FairnessMonitor.get_explanation`` reported severity "info" for a
       monitor that had never ingested a batch, the same severity a measured and
       compliant window gets, while a window that found no protected column
       already answered _UNKNOWN_SEV.

op2-5. The ``@fairness_test`` decorator did not raise on a SKIPPED result, and
       pytest judges the decorated test purely by whether it raises. Measured
       twice: one group in ``protected_attr`` (no between-group comparison
       exists) and metric='demogrpahic_parity_difference' (A TYPO) both returned
       a single SKIPPED result with no exception, so the fairness test went
       GREEN over a metric that was never compared to anything.
       ``assert_fairness`` has refused both since 2026-09-07.

op2-6. ``parametrize_fairness(protected_attributes=[])`` ran the test body 0
       times, returned ``[]`` and raised nothing, so the test went GREEN. An
       empty result made "no fairness problem was found" and "nothing was
       examined" the same output.

Judged CORRECT by execution, pinned here so they stay that way:
``check_fairness_config``, ``precommit.main``, ``FairnessTestSuite.get_summary``,
``FairnessTestSuite.test_predictions``, ``assert_fairness``,
``compute_equalized_odds``, ``update_and_check`` and ``mmd_gaussian``.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.precommit import check_fairness_config, main
from vfairness.operations.cicd.testing import (
    FairnessAssertionError,
    FairnessTestSuite,
    assert_fairness,
    fairness_test,
    parametrize_fairness,
)
from vfairness.operations.cicd.testing import (
    TestStatus as _TestStatus,
)
from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    TemporalFairnessAnalyzer,
    mmd_gaussian,
)

METRIC = "demographic_parity"  # lower is better, resolvable by name


def _caught(fn):
    """Run *fn* with warnings ON and record them.

    A refusal delivered in a warning is invisible when warnings are suppressed,
    which is how op2-3 survived: the value was right and nobody was told why.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(x.message) for x in w]


def _analyzer(values, start: str = "2025-01-06") -> TemporalFairnessAnalyzer:
    a = TemporalFairnessAnalyzer(lookback_days=400)
    base = pd.Timestamp(start)
    for i, v in enumerate(values):
        a.update_daily_metrics(base + pd.Timedelta(days=i), {METRIC: v})
    return a


# op2-1 detect_trend: a non-finite slope is not a direction


class TestTrendWithANonFiniteSlope:
    RISING = [0.01 * (i + 1) for i in range(14)]  # 0.01 .. 0.14

    def _with_inf(self) -> TemporalFairnessAnalyzer:
        values = list(self.RISING)
        values[7] = float("inf")
        return _analyzer(values)

    def test_it_refuses_instead_of_answering_decreasing(self):
        """REFUSAL. Measured before the fix: ("decreasing", nan), the recovering reading
        for a lower-is-better metric.
        """
        (direction, slope), warned = _caught(lambda: self._with_inf().detect_trend(METRIC))

        assert direction == "not_assessed", "a non-finite slope was graded as a direction"
        assert math.isnan(slope)
        assert any("non-finite" in w and "not_assessed" in w for w in warned), warned

    def test_a_broken_time_axis_is_refused_too_and_is_not_silent(self):
        """The sibling input, and the one that was refused by NOTHING: with a NaT on the
        date axis ``denom`` is nan, ``nan <= 0`` is False, so the existing 0/0 guard
        does not fire. Measured before the fix: ("decreasing", nan) with an EMPTY
        warning list.
        """
        analyzer = _analyzer(self.RISING)
        analyzer._daily_metrics.loc[3, "date"] = pd.NaT

        (direction, slope), warned = _caught(lambda: analyzer.detect_trend(METRIC))

        assert (direction, math.isnan(slope)) == ("not_assessed", True)
        assert any("date(s) are not finite" in w for w in warned), warned

    def test_the_summary_no_longer_publishes_the_fabricated_direction(self):
        """Measured before the fix: trend_direction "decreasing" beside trend_slope nan,
        which the docstring's own contract forbids.
        """
        summary, _ = _caught(lambda: self._with_inf().get_metric_summary(METRIC))

        assert summary["trend_direction"] == "not_assessed"
        assert math.isnan(summary["trend_slope"])

    def test_control_a_real_rising_series_keeps_its_exact_slope(self):
        """OVER-CORRECTION CONTROL. 0.01 per day, every day, is measurable and must still
        be measured.
        """
        (direction, slope), warned = _caught(lambda: _analyzer(self.RISING).detect_trend(METRIC))

        assert direction == "increasing"
        assert slope == pytest.approx(0.01, abs=1e-9)
        assert warned == []

    def test_control_a_real_flat_series_is_still_stable(self):
        """OVER-CORRECTION CONTROL. A genuinely flat series has a measured slope of 0.0 and
        "stable" is the true answer, not a refusal.
        """
        (direction, slope), warned = _caught(lambda: _analyzer([0.05] * 14).detect_trend(METRIC))

        assert direction == "stable"
        assert slope == pytest.approx(0.0, abs=1e-12)
        assert warned == []


# op2-2 TemporalFairnessAnalyzer.get_explanation: no verdict is not "info"


class TestTemporalReportWithNoVerdict:
    def test_a_report_that_graded_nothing_is_not_info(self):
        """REFUSAL. Measured before the fix: severity "info" for a 4-day analyzer swinging
        0.02 / 0.90 / 0.03 / 0.85, because mean=0.4500 counted as "something was
        measured". The mean is a statistic, not a verdict.
        """
        analyzer = _analyzer([0.02, 0.90, 0.03, 0.85])
        (direction, _), _ = _caught(lambda: analyzer.detect_trend(METRIC))
        (degraded, _), _ = _caught(lambda: analyzer.detect_weekly_degradation(METRIC))
        assert (direction, degraded) == ("not_assessed", None), "fixture no longer ungradable"

        report, _ = _caught(lambda: analyzer.get_explanation(METRIC))

        assert report.severity != "info", "a report that graded nothing read as clean"
        assert report.severity == "medium"  # _UNKNOWN_SEV, the house could-not-check
        assert "not_assessed" in report.summary

    def test_control_a_measured_stable_trend_is_still_info(self):
        """OVER-CORRECTION CONTROL. Eight flat days: the trend IS graded (stable) even
        though the weekday check still cannot be, so a verdict WAS reached and the
        report must stay "info".
        """
        analyzer = _analyzer([0.05] * 8)
        (direction, _), _ = _caught(lambda: analyzer.detect_trend(METRIC))
        (degraded, _), _ = _caught(lambda: analyzer.detect_weekly_degradation(METRIC))
        assert (direction, degraded) == ("stable", None)

        report, _ = _caught(lambda: analyzer.get_explanation(METRIC))

        assert report.severity == "info"

    def test_control_a_measured_degrading_trend_is_still_high(self):
        """OVER-CORRECTION CONTROL. 30 days climbing 0.02 -> 0.31 on a lower-is-better
        metric is a real finding and must stay escalated.
        """
        analyzer = _analyzer([0.02 + 0.01 * i for i in range(30)])

        report, _ = _caught(lambda: analyzer.get_explanation(METRIC))

        assert report.severity == "high"
        assert "increasing" in report.summary


# op2-3 compute_equal_opportunity: a TPR comparison that cannot be made


def _one_group_has_no_positives() -> pd.DataFrame:
    """40 M rows all labelled 1, 40 F rows all labelled 0.

    Both groups clear min_samples=30, so the undersampling guard does not fire;
    F has no positive label at all, so its TPR is undefined and no spread exists.
    """
    rows = [{"prediction": int(i % 2 == 0), "label": 1, "group_gender": "M"} for i in range(40)]
    rows += [{"prediction": int(i % 4 == 0), "label": 0, "group_gender": "F"} for i in range(40)]
    return pd.DataFrame(rows)


def _both_groups_measurable() -> pd.DataFrame:
    """40 rows per group, each with positives and negatives; TPR gap is 0.6."""
    rows = [
        {"prediction": int(i < 32), "label": int(i % 2 == 0), "group_gender": "M"}
        for i in range(40)
    ]
    rows += [
        {"prediction": int(i < 8), "label": int(i % 2 == 0), "group_gender": "F"} for i in range(40)
    ]
    return pd.DataFrame(rows)


class TestEqualOpportunityRefusalIsAudible:
    def test_it_names_the_group_with_no_positive_label(self):
        """REFUSAL. Measured before the fix: nan with an EMPTY warning list, while the
        docstring promised "np.nan, with a UserWarning".
        """
        monitor = FairnessMonitor()
        value, warned = _caught(
            lambda: monitor.compute_equal_opportunity(_one_group_has_no_positives(), "group_gender")
        )

        assert math.isnan(value), "the value was never the problem; it is the silence"
        assert len(warned) == 1, warned
        assert "1 of 2 group(s)" in warned[0]
        assert "'F'" in warned[0], "the group with no positives is not named"
        assert "0.0" in warned[0], "the warning does not say what 0.0 would have read as"

    def test_control_it_still_measures_the_exact_gap(self):
        """OVER-CORRECTION CONTROL. M TPR 1.0, F TPR 0.4, gap exactly 0.6."""
        monitor = FairnessMonitor()
        value, warned = _caught(
            lambda: monitor.compute_equal_opportunity(_both_groups_measurable(), "group_gender")
        )

        assert value == pytest.approx(0.6, abs=1e-9)
        assert warned == []

    def test_control_equalized_odds_keeps_its_own_refusal_and_its_own_number(self):
        """The sibling metric on both frames, so the two cannot drift apart."""
        monitor = FairnessMonitor()
        refused, warned = _caught(
            lambda: monitor.compute_equalized_odds(_one_group_has_no_positives(), "group_gender")
        )
        measured, clean = _caught(
            lambda: monitor.compute_equalized_odds(_both_groups_measurable(), "group_gender")
        )

        assert math.isnan(refused)
        assert any("TPR arm" in w for w in warned), warned
        assert measured == pytest.approx(0.6, abs=1e-9)
        assert clean == []


# op2-4 FairnessMonitor.get_explanation: an unmonitored monitor is not clean


def _clean_window() -> pd.DataFrame:
    rows = [{"prediction": int(i < 30), "label": 1, "group_gender": "M"} for i in range(60)]
    rows += [{"prediction": int(i < 30), "label": 1, "group_gender": "F"} for i in range(60)]
    return pd.DataFrame(rows)


def _breaching_window() -> pd.DataFrame:
    rows = [{"prediction": int(i < 54), "label": 1, "group_gender": "M"} for i in range(60)]
    rows += [{"prediction": int(i < 6), "label": 1, "group_gender": "F"} for i in range(60)]
    return pd.DataFrame(rows)


class TestMonitorExplanationWithNoHistory:
    def test_an_unmonitored_monitor_is_not_info(self):
        """REFUSAL. Measured before the fix: severity "info" and the bare summary "No data
        has been processed yet.", the same severity a measured and compliant window
        gets, while a window that merely found no protected column already answered
        "medium".
        """
        report, _ = _caught(lambda: FairnessMonitor().get_explanation())

        assert report.severity == "medium"  # _UNKNOWN_SEV
        assert report.explanations == []
        assert "NOTHING was compared" in report.summary

    def test_control_a_measured_clean_window_is_still_info(self):
        """OVER-CORRECTION CONTROL. Both groups selected at exactly 0.5, so disparate
        impact is 1.0 and parity 0.0: measured, compared, clean.
        """
        monitor = FairnessMonitor()
        snapshot, _ = _caught(lambda: monitor.update_and_check(_clean_window()))
        assert snapshot.metrics["disparate_impact_group_gender"] == pytest.approx(1.0)
        assert snapshot.any_alert is False

        report, _ = _caught(lambda: monitor.get_explanation())

        assert report.severity == "info"

    def test_control_a_breaching_window_is_still_high(self):
        """OVER-CORRECTION CONTROL. 0.9 against 0.1 is a ratio of 1/9."""
        monitor = FairnessMonitor()
        snapshot, _ = _caught(lambda: monitor.update_and_check(_breaching_window()))
        assert snapshot.metrics["disparate_impact_group_gender"] == pytest.approx(1 / 9)
        assert snapshot.any_alert is True

        report, _ = _caught(lambda: monitor.get_explanation())

        assert report.severity == "high"


# op2-5 the @fairness_test decorator: a green test over an unmeasured metric


_Y_TRUE = np.array([1, 0] * 10)
_Y_PRED = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1] * 2)
_ONE_GROUP = np.array(["A"] * 20)
_TWO_GROUPS = np.array(["A"] * 10 + ["B"] * 10)


class TestFairnessTestDecoratorFailsClosed:
    def test_one_group_is_refused_rather_than_passed(self):
        """REFUSAL. Measured before the fix: the decorated function RETURNED
        [FairnessTestResult(status=skipped, actual_value=None)] and raised nothing, so
        pytest recorded a pass over a comparison that does not exist. One group means
        there is no second rate to compare against.
        """

        @fairness_test(
            metric="demographic_parity_difference", threshold=0.1, protected_attribute="g"
        )
        def probe():
            return _Y_TRUE, _Y_PRED, _ONE_GROUP

        with pytest.raises(FairnessAssertionError) as excinfo:
            probe()
        assert "NOT MEASURABLE" in str(excinfo.value)

    def test_a_misspelt_metric_name_is_refused_rather_than_passed(self):
        """REFUSAL, and the sharper half: a green fairness gate over a metric that does not
        exist. Measured before the fix on healthy two-group data: status=skipped, "Could
        not compute metric 'demogrpahic_parity_difference'", no exception, test green.
        """

        @fairness_test(
            metric="demogrpahic_parity_difference", threshold=0.1, protected_attribute="g"
        )
        def probe():
            return _Y_TRUE, _Y_PRED, _TWO_GROUPS

        with pytest.raises(FairnessAssertionError) as excinfo:
            probe()
        assert "demogrpahic_parity_difference" in str(excinfo.value)
        assert "typo" in str(excinfo.value)

    def test_control_a_measured_clean_case_still_passes_with_its_value(self):
        """OVER-CORRECTION CONTROL. Identical rates in both groups: a real 0.0 gap,
        measured, compared, and allowed through.
        """
        y_true = np.array([1] * 10 + [0] * 10)
        y_pred = np.array([1, 0] * 10)

        @fairness_test(
            metric="demographic_parity_difference", threshold=0.1, protected_attribute="g"
        )
        def probe():
            return y_true, y_pred, _TWO_GROUPS

        results = probe()

        assert [r.status for r in results] == [_TestStatus.PASSED]
        assert results[0].actual_value == pytest.approx(0.0, abs=1e-12)

    def test_control_a_real_breach_still_raises_with_the_measured_gap(self):
        """OVER-CORRECTION CONTROL. Group A selected 10 of 10, group B 0 of 10: a measured
        gap of exactly 1.0, and the refusal must name that number rather than the
        wording reserved for a metric nobody could measure.
        """
        y_pred = np.array([1] * 10 + [0] * 10)

        @fairness_test(
            metric="demographic_parity_difference", threshold=0.1, protected_attribute="g"
        )
        def probe():
            return _Y_TRUE, y_pred, _TWO_GROUPS

        with pytest.raises(FairnessAssertionError) as excinfo:
            probe()
        assert excinfo.value.actual_value == pytest.approx(1.0)
        assert "NOT MEASURABLE" not in str(excinfo.value)


# op2-6 parametrize_fairness: an empty parametrization examined nothing


class TestEmptyParametrizationFailsClosed:
    def test_no_protected_attribute_is_refused_rather_than_returning_empty(self):
        """REFUSAL. Measured before the fix: the body was called 0 times, the wrapper
        returned [], nothing was raised, the test went GREEN.
        """
        calls = []

        @parametrize_fairness(
            protected_attributes=[],
            metrics=["demographic_parity_difference"],
            thresholds={"demographic_parity_difference": 0.1},
        )
        def probe(protected_attribute, metric, threshold):
            calls.append(protected_attribute)
            return "ran"

        with pytest.raises(FairnessAssertionError) as excinfo:
            probe()
        assert "NOT MEASURABLE" in str(excinfo.value)
        assert "0 protected attribute(s)" in str(excinfo.value)
        assert calls == []

    def test_control_a_real_parametrization_still_runs_the_body(self):
        """OVER-CORRECTION CONTROL. Two attributes, one metric, one bound: the body runs
        once per pair and gets the CONFIGURED 0.8 threshold, not an invented one.
        """
        calls = []

        @parametrize_fairness(
            protected_attributes=["gender", "race"],
            metrics=["disparate_impact_ratio"],
            thresholds={"disparate_impact_ratio": 0.8},
        )
        def probe(protected_attribute, metric, threshold):
            calls.append((protected_attribute, metric, threshold))
            return "ran"

        results = probe()

        assert calls == [
            ("gender", "disparate_impact_ratio", 0.8),
            ("race", "disparate_impact_ratio", 0.8),
        ]
        assert [r["passed"] for r in results] == [True, True]

    def test_control_a_metric_with_no_bound_still_fails_closed(self):
        """OVER-CORRECTION CONTROL for the neighbouring guard: an unbounded metric was
        never compared to anything and must not be handed the cap-shaped default 0.1.
        """

        @parametrize_fairness(
            protected_attributes=["gender"],
            metrics=["disparate_impact_ratio"],
            thresholds={},
        )
        def probe(protected_attribute, metric, threshold):
            return "ran"

        with pytest.raises(FairnessAssertionError) as excinfo:
            probe()
        assert math.isnan(excinfo.value.actual_value)


# Judged CORRECT by execution: the refusals that were already there


class TestTheGatesThatWereAlreadyHonest:
    def test_a_config_whose_metric_has_no_bound_is_refused(self, tmp_path):
        """The worst instance of this defect class in the repository was a gate writing
        Pass for a 0.90 gap it never compared, and this is the config checker that lets
        such a config exist. Executed, both directions.
        """
        good = tmp_path / "fairness.json"
        good.write_text(
            json.dumps(
                {
                    "metrics": ["demographic_parity_difference"],
                    "thresholds": {"demographic_parity_difference": 0.1},
                }
            )
        )
        unbounded = tmp_path / "bias.json"
        unbounded.write_text(
            json.dumps({"metrics": ["dp", "equalized_odds_difference"], "thresholds": {"dp": 0.1}})
        )
        empty = tmp_path / "gate.json"
        empty.write_text(json.dumps({"metrics": [], "thresholds": {}}))
        absent = tmp_path / "fairness-absent.json"

        assert check_fairness_config([str(good)]) == 0
        assert check_fairness_config([str(unbounded)]) == 1
        assert check_fairness_config([str(empty)]) == 1
        assert check_fairness_config([str(absent)]) == 1

    def test_main_without_a_subcommand_fails_closed(self, tmp_path):
        """pre-commit judges a hook purely by its exit code, so a misconfigured hook that
        checks nothing must not exit 0.
        """
        good = tmp_path / "fairness.json"
        good.write_text(json.dumps({"metrics": ["dp"], "thresholds": {"dp": 0.1}}))

        assert main([]) == 1
        assert main(["check-config", str(good)]) == 0

    def test_a_suite_that_measured_nothing_is_incomplete_not_passed(self):
        """One group: no between-group comparison exists, so the metric is SKIPPED with a
        reason and the batch headline is 'incomplete'.
        """
        suite = FairnessTestSuite(protected_attributes=["g"])
        results = suite.test_predictions(_Y_TRUE, _Y_PRED, _ONE_GROUP, "g", raise_on_failure=False)
        summary = suite.get_summary()

        assert [r.status for r in results] == [_TestStatus.SKIPPED]
        assert "Could not compute" in results[0].message
        assert (summary["status"], summary["passed"], summary["skipped"]) == ("incomplete", 0, 1)
        assert FairnessTestSuite(protected_attributes=["g"]).get_summary()["status"] == (
            "no_tests_run"
        )

    def test_assert_fairness_refuses_an_unmeasurable_metric(self):
        """The gate-shaped entry point, pinned beside the decorator above so the two cannot
        drift apart again.
        """
        with pytest.raises(FairnessAssertionError) as excinfo:
            assert_fairness(_Y_TRUE, _Y_PRED, _ONE_GROUP)
        assert "NOT MEASURABLE" in str(excinfo.value)

        y_true = np.array([1] * 10 + [0] * 10)
        assert assert_fairness(y_true, np.array([1, 0] * 10), _TWO_GROUPS) is None

    def test_a_window_with_no_protected_column_reports_none_not_false(self):
        monitor = FairnessMonitor(
            config=FairnessMonitorConfig(metrics_to_track=["disparate_impact"])
        )
        frame = pd.DataFrame({"prediction": [1, 0] * 60, "label": [1, 0] * 60})

        snapshot, warned = _caught(lambda: monitor.update_and_check(frame))

        assert snapshot.metrics == {} and snapshot.alerts == {}
        assert snapshot.any_alert is None
        assert snapshot.to_dict()["any_alert"] is None
        assert any("no protected-attribute column" in w for w in warned), warned

    def test_mmd_refuses_an_undefined_embedding_and_still_measures_a_shift(self):
        rng = np.random.default_rng(42)
        refused, warned = _caught(lambda: mmd_gaussian(np.array([]), rng.normal(0, 1, 50)))
        identical, clean = _caught(lambda: mmd_gaussian(np.arange(10.0), np.arange(10.0)))
        shifted, _ = _caught(lambda: mmd_gaussian(np.linspace(0, 1, 50), np.linspace(0.8, 1.8, 50)))

        assert math.isnan(refused)
        assert any("undefined" in w for w in warned), warned
        assert identical == pytest.approx(0.0, abs=1e-12)
        assert clean == []
        assert shifted > 0.05, "a real distribution shift is no longer measured"
