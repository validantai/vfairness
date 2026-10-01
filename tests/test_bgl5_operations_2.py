"""BGL-5: the seven overturned grades of batch A-operations-2, closed.

An independent auditor overturned seven grades in this batch on 2026-09-27, each
with a command and its output. This file is the second, independent pin: every
test below was written against a measured before and a measured after, and every
one of them was SABOTAGED (the fix reverted in a copy of the module source, the
test confirmed red, the source restored and confirmed byte-identical with
``diff``) so that none of them can be green for a reason other than the fix.

FOUR OF THE SEVEN SHARE ONE ROOT CAUSE. ``operations/monitoring/tracker.py``
never applied a measurement predicate to a PREDICTION. Every rate in it was built
with ``Series.mean()`` or ``(Series == 1).mean()``; pandas SKIPS a NaN in the
first and counts it as ``False`` in the second, so a row that was never scored was
read as a row the model declined. Measured before the fix, on 50 rows of group A
predicted 0.9 and 50 rows of group B predicted NaN with ``min_samples=30``:

    compute_disparate_impact    -> 1.0   (a PERFECT four-fifths ratio)  warnings []
    compute_demographic_parity  -> 0.0   (PERFECT parity)               warnings []

because ``max([0.9, nan])`` and ``min([0.9, nan])`` are both 0.9, so the
privileged group equalled the minimum. Renaming the groups so the NaN sorted
FIRST made the same data answer nan: the honesty rested on dict ordering. On a
frame where 20 of group B's 40 positive-label rows carried NaN, both
``compute_equal_opportunity`` and ``compute_equalized_odds`` returned exactly
``0.0`` while the TPR over B's SCORED rows was 1.0 against A's 0.5, a real gap of
0.5, and their pinned ``len(valid) < 2`` refusal was out of reach by construction
because both arms held two DEFINED rates.

That this is a missing guard and not a reading is settled inside this library:
``operations/cicd/monitor._compute_default_metrics`` refuses the identical input
on both arms, and its own source comment names the ``max``/``min`` order
dependence as the defect it fixed on 2026-09-17. It was never mirrored.

THE OTHER THREE.

  * ``get_alert_summary`` closed only the never-ran door. A monitor that RAN and
    compared nothing (no protected column, or every metric nan) still returned
    ``{}`` in silence, which ``if not monitor.get_alert_summary()`` reads as a
    clean bill.
  * ``generate_threshold_breach_report`` guarded with
    ``math.isfinite(float(v))``, and ``float(True)`` is 1.0, so handed ``True``
    it published "Breach Magnitude: 900.0% beyond threshold" and
    ``comparison_made: true``.
  * ``generate_alert_report`` fixed the section BODY and not the three TITLE
    sites of the identical ``.get('severity', 'ALERT')`` idiom, so a severity
    present holding ``None`` titled the page "Alert Report: None".

Every refusal here is paired with an OVER-CORRECTION CONTROL asserting the real
number healthy input still gets, because a fix that refuses everything passes
every refusal test and destroys the library.
"""

from __future__ import annotations

import decimal
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    _unscored_predictions,
)
from vfairness.operations.reporting.reports import (
    OutputFormat,
    ReportConfig,
    ReportGenerator,
    _assessed_rows,
    _severity_word,
)
from vfairness.operations.reporting.store import MetricsStore, StoredMetricRecord

NAN = float("nan")


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in rec]


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ===========================================================================
# 1 and 2. compute_disparate_impact / compute_demographic_parity
# ===========================================================================


def _one_group_unscored(unscored_first: bool = False) -> pd.DataFrame:
    """50 rows of group A predicted 0.9, 50 rows of group B never scored."""
    scored = [1] * 45 + [0] * 5
    unscored = [NAN] * 50
    predictions = unscored + scored if unscored_first else scored + unscored
    return pd.DataFrame(
        {
            "prediction": predictions,
            "label": [1] * 100,
            "group_gender": ["A"] * 50 + ["B"] * 50,
        }
    )


def _measurable_gap() -> pd.DataFrame:
    """A predicted 45 of 50, B 10 of 50: ratio 0.2/0.9, gap 0.7. Every row scored."""
    return pd.DataFrame(
        {
            "prediction": [1] * 45 + [0] * 5 + [1] * 10 + [0] * 40,
            "label": [1] * 100,
            "group_gender": ["A"] * 50 + ["B"] * 50,
        }
    )


class TestAnUndefinedGroupRateIsRefusedNotDropped:
    """The rate metrics, both directions, on both orderings of the same data."""

    @pytest.mark.parametrize("unscored_first", [False, True])
    def test_disparate_impact_refuses_and_names_the_group(self, unscored_first):
        """BEFORE: 1.0 with the NaN group second, nan with it first, warnings [] in
        both. AFTER: nan in both orderings, each with one warning naming the group
        and counting its unscored rows.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        value, warned = _caught(
            monitor.compute_disparate_impact,
            _one_group_unscored(unscored_first),
            "group_gender",
        )

        assert np.isnan(value), (
            f"an undefined group rate was published as disparate impact {value!r}; "
            f"1.0 clears the 0.8 four-fifths floor. warnings: {warned}"
        )
        assert len(warned) == 1, warned
        assert "could NOT be measured" in warned[0]
        assert "50 of 50 row(s) carry no prediction" in warned[0], warned[0]

    @pytest.mark.parametrize("unscored_first", [False, True])
    def test_demographic_parity_refuses_and_names_the_group(self, unscored_first):
        """BEFORE: 0.0, PERFECT parity, with the NaN group second. AFTER: nan with
        one warning, in both orderings.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        value, warned = _caught(
            monitor.compute_demographic_parity,
            _one_group_unscored(unscored_first),
            "group_gender",
        )

        assert np.isnan(value), (
            f"an undefined group rate was published as a parity gap of {value!r}, "
            f"which is perfect parity. warnings: {warned}"
        )
        assert len(warned) == 1, warned
        assert "could NOT be measured" in warned[0]

    def test_a_rate_over_the_scored_subset_is_not_the_groups_rate(self):
        """The partial case, which the ``min_samples`` floor lets through.

        BEFORE: group B held 30 rows of which 29 carried no prediction, and
        ``_group_positive_rates`` reported the single scored row as the group's
        rate (``Series.mean()`` skips NaN while the floor counts all 30), so the
        ratio was computed from one observation. AFTER: that group's rate is nan
        and the aggregate refuses.

        THIS IS THE ONLY TEST HERE THAT DISCRIMINATES THE ``_group_positive_rates``
        half of the fix, and the sabotage said so: reverting that method to the
        plain mean turned exactly one test red, this one. The FULLY unscored group
        is caught either way, because ``Series.mean()`` of an all-NaN group is
        already NaN, so the refusal above the max/min pair still fires. Only a
        PARTIALLY scored group separates the two guards.
        """
        frame = pd.DataFrame(
            {
                "prediction": [1] * 40 + [0] * 10 + [1] + [NAN] * 29 + [0] * 20,
                "label": [1] * 100,
                "group_gender": ["A"] * 50 + ["B"] * 50,
            }
        )
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        rates = monitor._group_positive_rates(frame, "group_gender", "prediction")

        assert rates["A"] == 0.8
        assert np.isnan(rates["B"]), rates
        value, warned = _caught(monitor.compute_disparate_impact, frame, "group_gender")
        assert np.isnan(value)
        assert any("29 of 50 row(s) carry no prediction" in w for w in warned), warned

    def test_control_a_real_disparity_is_still_measured_exactly(self):
        """OVER-CORRECTION CONTROL. A 45 of 50, B 10 of 50, every row scored:
        disparate impact 0.2/0.9 and a parity gap of exactly 0.7, no warning.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        ratio, ratio_warned = _caught(
            monitor.compute_disparate_impact, _measurable_gap(), "group_gender"
        )
        gap, gap_warned = _caught(
            monitor.compute_demographic_parity, _measurable_gap(), "group_gender"
        )

        assert ratio == pytest.approx(0.2 / 0.9), ratio
        assert gap == pytest.approx(0.7), gap
        assert ratio_warned == [] and gap_warned == []

    def test_control_a_boolean_prediction_column_is_still_a_measurement(self):
        """OVER-CORRECTION CONTROL, the one that stops this becoming a refusal
        machine. ``_triage.is_measured`` rejects a bool because a metric VALUE of
        True clamps to a perfect 1.0, but a PREDICTION of True is one of the two
        decisions a classifier makes and this library's own ``y_pred`` arrays
        carry it. Both groups selected 45 of 50 as booleans: ratio exactly 1.0,
        gap exactly 0.0, no warning.
        """
        frame = pd.DataFrame(
            {
                "prediction": ([True] * 45 + [False] * 5) * 2,
                "label": [1] * 100,
                "group_gender": ["A"] * 50 + ["B"] * 50,
            }
        )
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        ratio, warned = _caught(monitor.compute_disparate_impact, frame, "group_gender")
        gap, gap_warned = _caught(monitor.compute_demographic_parity, frame, "group_gender")

        assert ratio == 1.0 and gap == 0.0
        assert warned == [] and gap_warned == []

    def test_a_nullable_boolean_na_is_still_an_unscored_row(self):
        """pandas' NULLABLE boolean dtype holds pd.NA beside True and False, and a
        bare ``return 0`` for every boolean column would have hidden it. ``isna``
        counts it; a numpy bool column cannot hold one and still counts 0.
        """
        numpy_bools = pd.Series([True, False, True], dtype=bool)
        nullable = pd.Series([True, None, False], dtype="boolean")

        assert _unscored_predictions(numpy_bools) == 0
        assert _unscored_predictions(nullable) == 1

    def test_the_sibling_monitor_still_agrees(self):
        """The two monitors in this library now answer the same way.

        ``operations/cicd/monitor._compute_default_metrics`` has refused these rows
        with nan since 2026-09-17; the tracker published 1.0 and 0.0 for them. This
        is the cross-check that the disagreement is gone, in the direction of the
        module that was already right.
        """
        from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig

        y_pred = np.array([1.0] * 45 + [0.0] * 5 + [NAN] * 50)
        y_true = np.array([1] * 100)
        groups = np.array(["A"] * 50 + ["B"] * 50)
        sibling = BiasMonitor(baseline_metrics={}, config=MonitorConfig(metrics_to_monitor=[]))
        sibling_metrics = _quiet(sibling._compute_default_metrics, y_true, y_pred, groups)

        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        tracker_value = _quiet(
            monitor.compute_demographic_parity, _one_group_unscored(), "group_gender"
        )

        assert np.isnan(sibling_metrics["demographic_parity_difference"])
        assert np.isnan(tracker_value)


# ===========================================================================
# 3 and 4. compute_equal_opportunity / compute_equalized_odds
# ===========================================================================


def _half_of_one_group_unscored() -> pd.DataFrame:
    """A's TPR is genuinely 0.5; B has 20 scored positive rows, all selected.

    Over the rows that were actually scored the TPR gap is 0.5. Counting the 20
    unscored rows as negative predictions drags B's reported rate to exactly A's.
    """
    rows = []
    rows += [("A", 1, 1)] * 20 + [("A", 1, 0)] * 20
    rows += [("A", 0, 1)] * 10 + [("A", 0, 0)] * 30
    rows += [("B", 1, 1)] * 20 + [("B", 1, NAN)] * 20
    rows += [("B", 0, 1)] * 10 + [("B", 0, 0)] * 30
    return pd.DataFrame(rows, columns=["group_gender", "label", "prediction"])


def _both_arms_measurable() -> pd.DataFrame:
    """A TPR 0.5, B TPR 1.0, both FPRs 0.25: a gap of exactly 0.5 on both metrics."""
    rows = []
    rows += [("A", 1, 1)] * 20 + [("A", 1, 0)] * 20
    rows += [("A", 0, 1)] * 10 + [("A", 0, 0)] * 30
    rows += [("B", 1, 1)] * 40
    rows += [("B", 0, 1)] * 10 + [("B", 0, 0)] * 30
    return pd.DataFrame(rows, columns=["group_gender", "label", "prediction"])


def _one_group_has_no_positives() -> pd.DataFrame:
    """40 M rows all labelled 1, 40 F rows all labelled 0, no NaN anywhere.

    F has no positive label, so its TPR does not EXIST: a different state from a
    TPR that exists and was not measured, and the refusal it already had must stay
    exactly one warning.
    """
    rows = [("M", 1, 1)] * 20 + [("M", 1, 0)] * 20 + [("F", 0, 1)] * 20 + [("F", 0, 0)] * 20
    return pd.DataFrame(rows, columns=["group_gender", "label", "prediction"])


class TestAnUnscoredPositiveRowIsNotANegativePrediction:
    def test_equal_opportunity_refuses_and_counts_the_unscored_rows(self):
        """BEFORE: np.float64(0.0), PERFECT equal opportunity, warnings []. The TPR
        over B's scored rows is 1.0 against A's 0.5. AFTER: nan with one warning
        naming "B: 20 of 40 positive-label row(s)".
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        value, warned = _caught(
            monitor.compute_equal_opportunity, _half_of_one_group_unscored(), "group_gender"
        )

        assert np.isnan(value), f"published {value!r} as perfect equal opportunity: {warned}"
        assert len(warned) == 1, warned
        assert "never scored" in warned[0]
        assert "B: 20 of 40 positive-label row(s)" in warned[0], warned[0]

    def test_equalized_odds_refuses_above_the_rate_count_dispatch(self):
        """BEFORE: np.float64(0.0), warnings []. BOTH arms held two DEFINED rates,
        so the ``len(valid) < 2`` refusal this unit was graded PROVEN on could not
        fire: an unscored row does not remove a rate, it silently moves one. AFTER:
        nan with one warning, raised ABOVE that dispatch.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        value, warned = _caught(
            monitor.compute_equalized_odds, _half_of_one_group_unscored(), "group_gender"
        )

        assert np.isnan(value), f"published {value!r} as perfect equalized odds: {warned}"
        assert len(warned) == 1, warned
        assert "never scored" in warned[0]
        assert "0 of 40 negative-label row(s)" in warned[0], warned[0]

    def test_control_both_metrics_still_measure_the_exact_gap(self):
        """OVER-CORRECTION CONTROL. A TPR 0.5 against B TPR 1.0, every row scored:
        both metrics measure exactly 0.5 and say nothing.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        opportunity, opportunity_warned = _caught(
            monitor.compute_equal_opportunity, _both_arms_measurable(), "group_gender"
        )
        odds, odds_warned = _caught(
            monitor.compute_equalized_odds, _both_arms_measurable(), "group_gender"
        )

        assert opportunity == pytest.approx(0.5), opportunity
        assert odds == pytest.approx(0.5), odds
        assert opportunity_warned == [] and odds_warned == []

    def test_control_the_no_positive_label_refusal_keeps_its_own_single_warning(self):
        """OVER-CORRECTION CONTROL for the guard's PLACEMENT. The new refusal is a
        separate state and must not fire on, duplicate or reword the refusal this
        unit already had: a group with no positive label at all still produces nan
        and EXACTLY ONE warning, still naming 'F' and still saying what 0.0 would
        have read as. That is what ``test_bgl3_operations_2`` pins with
        ``len(warned) == 1``.
        """
        monitor = FairnessMonitor()
        value, warned = _caught(
            monitor.compute_equal_opportunity, _one_group_has_no_positives(), "group_gender"
        )

        assert np.isnan(value)
        assert len(warned) == 1, warned
        assert "1 of 2 group(s)" in warned[0]
        assert "'F'" in warned[0]
        assert "never scored" not in warned[0], warned[0]


class TestTheRefusalReachesTheSnapshotAndTheDashboard:
    """The disclosure at the surfaces a consumer reads, not only at the return."""

    def test_the_window_records_no_threshold_comparison_at_all(self):
        """BEFORE: metrics 1.0 and 0.0, alerts {di: False, dp: False}, any_alert
        False, to_dict()['any_alert'] False, get_explanation().severity 'info': the
        whole chain read as a measured, compliant window. AFTER: metrics nan, so
        the existing ``if not np.isnan(val)`` guard records NO comparison, alerts
        {}, any_alert None at the object and the JSON boundary, and severity
        'medium' (_UNKNOWN_SEV).
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        snapshot, warned = _caught(monitor.update_and_check, _one_group_unscored())

        assert np.isnan(snapshot.metrics["disparate_impact_group_gender"])
        assert np.isnan(snapshot.metrics["demographic_parity_group_gender"])
        assert snapshot.alerts == {}, snapshot.alerts
        assert snapshot.any_alert is None
        assert snapshot.to_dict()["any_alert"] is None
        assert np.isnan(snapshot.group_rates["group_gender"]["B"])
        assert len([w for w in warned if "could NOT be measured" in w]) == 2, warned
        assert monitor.get_explanation().severity == "medium"

    def test_control_a_measured_breaching_window_still_alerts(self):
        """OVER-CORRECTION CONTROL. 54 of 60 against 6 of 60, every row scored:
        ratio exactly 1/9, gap exactly 0.8, both alerts True, any_alert True, and
        the summary counts one breach per metric.
        """
        frame = pd.DataFrame(
            {
                "prediction": [1] * 54 + [0] * 6 + [1] * 6 + [0] * 54,
                "label": [1] * 120,
                "group_gender": ["A"] * 60 + ["B"] * 60,
            }
        )
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        snapshot, warned = _caught(monitor.update_and_check, frame)

        assert snapshot.metrics["disparate_impact_group_gender"] == pytest.approx(1 / 9)
        assert snapshot.metrics["demographic_parity_group_gender"] == pytest.approx(0.8)
        assert snapshot.alerts == {
            "disparate_impact_group_gender": True,
            "demographic_parity_group_gender": True,
        }
        assert snapshot.any_alert is True
        assert warned == []
        assert _caught(monitor.get_alert_summary) == (
            {"disparate_impact_group_gender": 1, "demographic_parity_group_gender": 1},
            [],
        )


# ===========================================================================
# 5. get_alert_summary: the two doors the never-ran fix left open
# ===========================================================================


def _compared_and_clean() -> pd.DataFrame:
    """Both groups selected at exactly 0.5: DI 1.0, DP 0.0, compared and clean."""
    per_group = [1] * 30 + [0] * 30
    return pd.DataFrame(
        {
            "prediction": per_group * 2,
            "label": per_group * 2,
            "group_gender": ["A"] * 60 + ["B"] * 60,
        }
    )


class TestAnEmptyAlertSummaryNamesWhichEmptyItIs:
    def test_door_one_a_monitor_that_never_ran_still_says_so(self):
        """Unchanged by this fix and re-pinned here, because the new guard sits
        directly below it and must not swallow it.
        """
        summary, warned = _caught(FairnessMonitor().get_alert_summary)

        assert summary == {}
        assert any("no window has been monitored" in w for w in warned), warned

    def test_door_two_a_window_with_no_protected_column_says_so(self):
        """BEFORE: ``{}`` with warnings []. The monitor had run one window,
        ``snapshot.alerts`` was {} and ``any_alert`` None, so nothing was compared,
        and ``if not monitor.get_alert_summary()`` read it as a clean bill. AFTER:
        ``{}`` plus "1 window(s) were monitored and nothing was compared to a
        threshold in any of them".
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        snapshot = _quiet(
            monitor.update_and_check,
            pd.DataFrame({"prediction": [1, 0] * 60, "label": [1, 0] * 60}),
        )
        assert snapshot.alerts == {} and snapshot.any_alert is None, "fixture drifted"

        summary, warned = _caught(monitor.get_alert_summary)

        assert summary == {}
        assert any("nothing was compared to a threshold" in w for w in warned), warned

    def test_door_three_an_all_nan_prediction_column_says_so(self):
        """BEFORE: ``{}`` with warnings [] for a window whose metrics were BOTH nan.
        The protected column was found and the groups were large enough; there was
        simply nothing to score. AFTER: the same warning as door two.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        frame = pd.DataFrame(
            {
                "prediction": [NAN] * 120,
                "label": [1, 0] * 60,
                "group_gender": ["A"] * 60 + ["B"] * 60,
            }
        )
        snapshot = _quiet(monitor.update_and_check, frame)
        assert all(np.isnan(v) for v in snapshot.metrics.values()), snapshot.metrics

        summary, warned = _caught(monitor.get_alert_summary)

        assert summary == {}
        assert any("nothing was compared to a threshold" in w for w in warned), warned

    def test_control_a_monitored_and_clean_history_stays_silent(self):
        """OVER-CORRECTION CONTROL, and the one the auditor proved a naive fix
        fails. "Monitored a window and nothing breached" is a real finding and must
        not be drowned in a caveat. Both groups selected at exactly 0.5: DI 1.0, DP
        0.0, both alerts recorded False, so the empty summary is a measurement.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        snapshot, ingest_warned = _caught(monitor.update_and_check, _compared_and_clean())
        assert snapshot.metrics["disparate_impact_group_gender"] == 1.0
        assert snapshot.metrics["demographic_parity_group_gender"] == 0.0
        assert snapshot.alerts == {
            "disparate_impact_group_gender": False,
            "demographic_parity_group_gender": False,
        }
        assert snapshot.any_alert is False
        assert ingest_warned == []

        summary, warned = _caught(monitor.get_alert_summary)

        assert summary == {}
        assert warned == [], warned

    def test_control_one_compared_window_beside_a_refused_one_stays_silent(self):
        """OVER-CORRECTION CONTROL on the quantifier, CORRECTED 2026-09-28.

        This asserted that a history holding one measured window and one
        unmeasurable window stays SILENT, on the reasoning that the history "has a
        verdict" and the refused window is disclosed at ingest. The BGL6 audit
        overturned that, and it is right. The guard asked whether ANY window
        compared anything, so nine unmeasurable windows beside one measured one were
        silent too: measured on one clean window followed by nine whose prediction
        column is all NaN, get_alert_summary() returned {} with no warnings, which is
        byte-identical to a ten-window fully monitored clean history. A caller
        writing `if not monitor.get_alert_summary()` read 90 per cent unmonitored as
        a clean bill. The ingest warning does not help: it fired nine windows ago,
        in a different call, and this method's result is what gets read.
        COVERAGE is the question, not total absence.

        The control's SUBJECT is unchanged and is what is asserted below: the
        disclosure must not fire when every window in the history did compare
        something. That is what over-correction would look like here, and the old
        silence assertion never tested it.
        """
        monitor = FairnessMonitor(config=FairnessMonitorConfig())
        _quiet(monitor.update_and_check, _compared_and_clean())
        _quiet(
            monitor.update_and_check,
            pd.DataFrame({"prediction": [1, 0] * 60, "label": [1, 0] * 60}),
        )

        summary, warned = _caught(monitor.get_alert_summary)

        assert summary == {}
        # One of the two windows carries no comparison, so the coverage is stated.
        assert len(warned) == 1, warned
        assert "1 of 2 window(s)" in warned[0], warned[0]
        assert "NOT a finding that no metric breached" in warned[0], warned[0]

        # AND THE CONTROL PROPER: a history in which EVERY window compared something
        # stays silent, so the disclosure is about coverage rather than about having
        # any unmeasured data anywhere.
        full = FairnessMonitor(config=FairnessMonitorConfig())
        _quiet(full.update_and_check, _compared_and_clean())
        _quiet(full.update_and_check, _compared_and_clean())
        full_summary, full_warned = _caught(full.get_alert_summary)
        assert full_summary == {}
        assert full_warned == [], full_warned


# ===========================================================================
# 6. generate_threshold_breach_report: a flag is not a breach
# ===========================================================================


def _generator() -> ReportGenerator:
    return ReportGenerator(MetricsStore(), config=ReportConfig(time_window=timedelta(days=7)))


def _breach_section(report) -> str:
    return next(s for s in report.sections if s["title"] == "Threshold Breach")["content"]


class TestAThresholdBreachNeedsAComparisonThatHappened:
    @pytest.mark.parametrize("flag", [True, False, np.bool_(True)])
    def test_a_flag_does_not_publish_a_breach_magnitude(self, flag):
        """BEFORE, handed ``True``: "Current Value: 1.0000 | Threshold: 0.1000 |
        Breach Magnitude: 900.0% beyond threshold", the recommendation "The
        demographic_parity_difference metric has breached its 0.10 threshold." and
        JSON ``"current_value": true, "breach_pct": 900.0, "comparison_made":
        true``. ``_is_comparable`` was ``math.isfinite(float(v))`` and
        ``float(True)`` is 1.0. AFTER: "Current Value: not measured (a yes/no flag,
        not a measurement)", COULD NOT CHECK, no breach recommendation,
        ``breach_pct`` null and ``comparison_made`` false.
        """
        report = _quiet(
            _generator().generate_threshold_breach_report,
            "demographic_parity_difference",
            flag,
            0.10,
            OutputFormat.MARKDOWN,
        )
        section = _breach_section(report)

        assert "COULD NOT CHECK" in section, section
        assert "a yes/no flag, not a measurement" in section, section
        assert "% beyond threshold" not in section, section
        assert not [r for r in report.recommendations if "has breached" in r]
        # The rendered document, not only the section list a caller could ignore.
        assert "COULD NOT CHECK" in report.content

        payload = json.loads(
            _quiet(
                _generator().generate_threshold_breach_report,
                "demographic_parity_difference",
                flag,
                0.10,
                OutputFormat.JSON,
            ).content
        )
        assert payload["comparison_made"] is False
        assert payload["breach_pct"] is None

    def test_a_numeric_string_is_refused_rather_than_crashing_the_method(self):
        """BEFORE: ``float("0.45")`` is finite so the guard passed, and the
        arithmetic then used the RAW value: "TypeError: unsupported operand type(s)
        for -: 'str' and 'float'" out of the public method. AFTER: the predicate and
        the arithmetic agree, and the page refuses and names the reason. A report
        does not invent a parse of its own input.
        """
        report = _quiet(
            _generator().generate_threshold_breach_report,
            "dp",
            "0.45",
            0.10,
            OutputFormat.MARKDOWN,
        )
        section = _breach_section(report)

        assert "COULD NOT CHECK" in section, section
        assert "not a number (str)" in section, section

    def test_control_a_real_breach_still_states_its_magnitude(self):
        """OVER-CORRECTION CONTROL. 0.45 against a 0.10 threshold: 350.0% beyond,
        the breach recommendation, and ``comparison_made`` true.
        """
        report = _quiet(
            _generator().generate_threshold_breach_report,
            "demographic_parity_difference",
            0.45,
            0.10,
            OutputFormat.MARKDOWN,
        )

        assert "Breach Magnitude: 350.0% beyond threshold" in _breach_section(report)
        assert "Current Value: 0.4500" in _breach_section(report)
        assert [r for r in report.recommendations if "has breached its 0.10 threshold" in r]

        payload = json.loads(
            _quiet(
                _generator().generate_threshold_breach_report,
                "demographic_parity_difference",
                0.45,
                0.10,
                OutputFormat.JSON,
            ).content
        )
        assert payload["comparison_made"] is True
        assert payload["breach_pct"] == pytest.approx(350.0)

    @pytest.mark.parametrize(
        "value,expected",
        [(np.float32(0.45), 349.99998807907104), (decimal.Decimal("0.45"), 350.0)],
    )
    def test_control_a_numpy_or_decimal_measurement_is_still_compared(self, value, expected):
        """OVER-CORRECTION CONTROL against the OPPOSITE error, a measurement
        discarded as a could-not-check. ``isinstance(v, (int, float))`` would have
        rejected both of these; ``np.float32`` is the READINESS-6 type and a
        ``Decimal`` is what a database NUMERIC column delivers. Both are real,
        finite numbers, both are compared, and the arithmetic runs on floats bound
        after the guard rather than on the raw value: ``Decimal("0.45") - 0.10``
        raised TypeError out of this method until that binding was added.
        """
        payload = json.loads(
            _quiet(
                _generator().generate_threshold_breach_report,
                "dp",
                value,
                0.10,
                OutputFormat.JSON,
            ).content
        )

        assert payload["comparison_made"] is True
        assert payload["breach_pct"] == pytest.approx(expected)

    def test_control_a_zero_threshold_keeps_the_wording_r2_gave_it(self):
        """OVER-CORRECTION CONTROL. 0.0 is FINITE: the comparison is real and only
        the percentage beyond it is undefined, so this must NOT become a
        could-not-check.
        """
        report = _quiet(
            _generator().generate_threshold_breach_report,
            "dp",
            0.45,
            0.0,
            OutputFormat.MARKDOWN,
        )
        section = _breach_section(report)

        assert "not measurable (the threshold is 0" in section, section
        assert "COULD NOT CHECK" not in section, section
        assert [r for r in report.recommendations if "has breached" in r]


# ===========================================================================
# 7. generate_alert_report: the title is a surface too
# ===========================================================================


class TestAnUnscoredSeverityIsNotTitledNone:
    @pytest.mark.parametrize("severity", [None, "", "   "])
    def test_every_surface_says_unscored(self, severity):
        """BEFORE, on ``{'severity': None, ...}``: ``GeneratedReport.title`` was
        "Alert Report: None", the markdown rendered "# Alert Report: None" and the
        HTML "<title>Alert Report: None</title>", while the BODY of the same object
        correctly said "Severity: UNSCORED (not determined)".
        ``d.get('severity', 'ALERT')`` stood at three title sites and does not fire
        for a key PRESENT holding None. AFTER: one ``_severity_word`` call feeds all
        four surfaces.
        """
        alert = {
            "severity": severity,
            "metric_name": "demographic_parity_difference",
            "message": "drift observed",
        }
        html = _quiet(_generator().generate_alert_report, dict(alert), OutputFormat.HTML)
        markdown = _quiet(_generator().generate_alert_report, dict(alert), OutputFormat.MARKDOWN)

        assert html.title == "Alert Report: UNSCORED (not determined)", html.title
        assert "Alert Report: None" not in html.content
        assert "<title>Alert Report: UNSCORED (not determined)</title>" in html.content
        assert markdown.content.splitlines()[0] == "# Alert Report: UNSCORED (not determined)"
        assert "Severity: UNSCORED (not determined)" in markdown.sections[0]["content"]

    def test_an_absent_severity_key_is_the_same_could_not_check(self):
        """BEFORE: "Alert Report: ALERT", a default that ASSERTS there is an
        alert-worthy severity here, which is exactly what was not determined. An
        absent key and a null one are one state.
        """
        report = _quiet(
            _generator().generate_alert_report,
            {"metric_name": "dp", "message": "x"},
            OutputFormat.MARKDOWN,
        )

        assert report.title == "Alert Report: UNSCORED (not determined)", report.title

    def test_control_a_recorded_severity_is_printed_exactly_as_recorded(self):
        """OVER-CORRECTION CONTROL. A severity that WAS determined reaches every one
        of the four surfaces unaltered, and the affected-groups line still names
        exactly the groups the alert names.
        """
        alert = {
            "severity": "CRITICAL",
            "metric_name": "demographic_parity_difference",
            "message": "drift observed",
            "affected_groups": ["gender_female", "race_black"],
        }
        html = _quiet(_generator().generate_alert_report, dict(alert), OutputFormat.HTML)
        markdown = _quiet(_generator().generate_alert_report, dict(alert), OutputFormat.MARKDOWN)

        assert html.title == "Alert Report: CRITICAL"
        assert "<title>Alert Report: CRITICAL</title>" in html.content
        assert markdown.content.splitlines()[0] == "# Alert Report: CRITICAL"
        assert "Severity: CRITICAL" in markdown.sections[0]["content"]
        assert "Affected Groups: gender_female, race_black" in markdown.sections[0]["content"]

    def test_a_flag_in_a_statistic_slot_is_not_printed_as_1_0000(self):
        """The same flag defect on the other three numbers this page prints.

        ``_format_stat`` coerced with ``float(value)``, and ``float(True)`` is 1.0:
        measured 2026-09-27, ``_format_stat(True)`` returned "1.0000". On this page
        that is a priority score of 1.0000 and a drift score of 1.0000, the two
        figures an operator triages by, invented out of a yes/no. AFTER: "not
        measured". A numeric STRING still renders, because this renderer takes rows
        from JSON and CSV where "0.5" is a serialised measurement.
        """
        alert = {
            "severity": "HIGH",
            "metric_name": "dp",
            "message": "x",
            "drift_score": True,
            "priority_score": np.bool_(True),
        }
        report = _quiet(_generator().generate_alert_report, dict(alert), OutputFormat.MARKDOWN)
        details = report.sections[0]["content"]

        assert "Drift Score: not measured" in details, details
        assert "Priority Score: not measured" in details, details
        assert "1.0000" not in details, details

        measured = _quiet(
            _generator().generate_alert_report,
            {"severity": "HIGH", "metric_name": "dp", "message": "x", "drift_score": "0.4200"},
            OutputFormat.MARKDOWN,
        )
        assert "Drift Score: 0.4200" in measured.sections[0]["content"]

    def test_the_word_itself_is_the_one_function(self):
        """One reading for all four surfaces, so the next fix cannot reach three
        sites out of four again.
        """
        assert _severity_word(None) == "UNSCORED (not determined)"
        assert _severity_word("") == "UNSCORED (not determined)"
        assert _severity_word("HIGH") == "HIGH"


# ===========================================================================
# 8. _assessed_rows: the same defect running BACKWARDS
# ===========================================================================


def _store_holding(value) -> MetricsStore:
    store = MetricsStore()
    now = datetime.now()
    for hour in range(3):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(hours=hour + 1),
                source="monitor",
                metric_name="demographic_parity_difference",
                value=value,
                group="overall",
                group_size=500,
                alert=True,
            )
        )
    return store


class TestAMeasurementIsNotReportedAsACouldNotCheck:
    @pytest.mark.parametrize("value", [0.91, np.float32(0.91), decimal.Decimal("0.91")])
    def test_a_real_finite_value_is_graded(self, value):
        """BEFORE, with ``value=Decimal("0.91")`` and ``alert=True``: 0 of 3 graded
        rows and the metric named as never compared, so the report said "3 of 3
        measurement(s) were never compared to a threshold, or could not be measured
        at all" about three rows that were both. AFTER: 3 graded rows, no name.

        This is the READINESS-6 direction: a measurement reported as a
        could-not-check, which reads as caution while discarding the evidence.

        Only the DECIMAL parametrisation discriminates the fix, which the sabotage
        established and reading the code did not: with the old
        ``isinstance(v, (int, float))`` restored, [value2] failed and [value1],
        np.float32, stayed GREEN, because ``Series.apply`` boxes a float dtype's
        elements as Python floats (measured: ``.apply(lambda v: type(v).__name__)``
        on a float32 Series answers 'float' while ``.iloc[0]`` is a np.float32).
        np.float32 is kept as a control on exactly that assumption: the day pandas
        stops boxing, this row is what notices.
        """
        graded, not_assessed = _assessed_rows(_store_holding(value).get_metrics())

        assert len(graded) == 3, f"reclassified as could-not-check: {not_assessed}"
        assert not_assessed == []

    @pytest.mark.parametrize("value", [NAN, float("inf")])
    def test_control_a_value_that_cannot_be_graded_is_still_named(self, value):
        """OVER-CORRECTION CONTROL in the other direction. NaN was already excluded
        and stays excluded; ``inf`` is excluded too, because ``inf > threshold`` is
        True for EVERY threshold, so it is not a comparison either.
        """
        graded, not_assessed = _assessed_rows(_store_holding(value).get_metrics())

        assert len(graded) == 0
        assert not_assessed == ["demographic_parity_difference"]
