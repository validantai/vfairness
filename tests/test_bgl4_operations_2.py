"""BGL-4 AUDIT of batch A-operations-2: the overturns, each one executed.

Written by an AUDITOR, not by a fixer. Every test below asserted the behaviour
the graded unit SHOULD have and was marked ``xfail(strict=True)`` with the output
measured on 2026-09-27 against the code of that day.

**CLOSED the same day, BGL5.** The fixes landed, so every one of those eight
markers XPASSed, which under ``xfail_strict`` is a failure: that is the marker
doing its job rather than outliving the defect. Each one is now a plain
assertion of the CORRECTED behaviour, keeping its original subject and the
measured before/after in its docstring, and three corroborations that documented
the defect as it stood (the order dependence, the snapshot that published the
all-clear, the guard that crashed on a numeric string) assert the corrected
behaviour instead. The fixes themselves live in
``src/vfairness/operations/monitoring/tracker.py`` and
``src/vfairness/operations/reporting/reports.py``; the second, independent pin
with the sabotage record is ``tests/test_bgl5_operations_2.py``.

Five grades were overturned. Four of them share ONE root cause with two faces:
a prediction that was never made is silently counted as a prediction of 0, and a
group rate that is undefined is silently dropped out of a ``max``/``min`` pair.

  1. FairnessMonitor.compute_disparate_impact   graded SEMI-PROVEN -> 1.0
  2. FairnessMonitor.compute_demographic_parity graded SEMI-PROVEN -> 0.0
  3. FairnessMonitor.compute_equal_opportunity  graded PROVEN      -> 0.0
  4. FairnessMonitor.compute_equalized_odds     graded PROVEN      -> 0.0
  5. FairnessMonitor.get_alert_summary          graded PROVEN      -> silent {}
  6. ReportGenerator.generate_threshold_breach_report graded PROVEN -> 900%
  7. ReportGenerator.generate_alert_report      graded PROVEN      -> "None"

The contrast that settles 1 to 4 as defects rather than as design choices: the
OTHER monitor in this same library, ``operations.cicd.monitor``, refuses the
identical input. Its ``_compute_default_metrics`` answers nan on both arms, and
its own source comment explains exactly why, naming the order dependence of
Python's ``max``/``min`` around NaN. That guard was never mirrored into
``operations.monitoring.tracker``.
"""

from __future__ import annotations

import decimal
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig
from vfairness.operations.monitoring.tracker import FairnessMonitor, FairnessMonitorConfig
from vfairness.operations.reporting.reports import (
    OutputFormat,
    ReportConfig,
    ReportGenerator,
    _assessed_rows,
)
from vfairness.operations.reporting.store import MetricsStore

NAN = float("nan")


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in rec]


# ===========================================================================
# 1 and 2. A group whose predicted-positive rate is UNDEFINED
# ===========================================================================


def _one_group_unscored(first_group_unscored: bool = False) -> pd.DataFrame:
    """50 rows of group A predicted 0.9, 50 rows of group B never scored.

    This is the fixture ``operations/cicd/monitor.py`` names in its own source
    comment for the sibling defect it fixed on 2026-09-17 ("100 rows of group A
    predicted 1.0 and 100 rows of group B predicted NaN").
    """
    scored = [1] * 45 + [0] * 5
    unscored = [NAN] * 50
    if first_group_unscored:
        return pd.DataFrame(
            {
                "prediction": unscored + scored,
                "label": [1] * 100,
                "group_gender": ["A"] * 50 + ["B"] * 50,
            }
        )
    return pd.DataFrame(
        {
            "prediction": scored + unscored,
            "label": [1] * 100,
            "group_gender": ["A"] * 50 + ["B"] * 50,
        }
    )


def test_a_group_with_no_defined_rate_is_not_a_disparate_impact_of_one():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: compute_disparate_impact returned 1.0, the cleanest value on
    the four-fifths scale, with ZERO warnings, for a window in which group B's
    positive rate is nan. _group_positive_rates yielded {'A': 0.9, 'B': nan};
    max() kept 0.9 because nan > 0.9 is False and min() kept 0.9 because
    nan < 0.9 is False, so privileged == min and the ratio was exactly 1.0.
    MEASURED AFTER: nan, and one warning naming group B and counting its 50
    unscored rows of 50.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    value, warned = _caught(monitor.compute_disparate_impact, _one_group_unscored(), "group_gender")
    assert np.isnan(value), (
        f"an undefined group rate was published as disparate impact {value!r}, "
        f"which passes the 0.8 four-fifths floor; warnings: {warned}"
    )
    assert any("could NOT be measured" in w and "B (50 of 50" in w for w in warned), warned


def test_a_group_with_no_defined_rate_is_not_perfect_demographic_parity():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: compute_demographic_parity returned 0.0, PERFECT PARITY,
    with zero warnings, for the same window, because max(rates) - min(rates) was
    0.9 - 0.9: both reductions skip the nan. MEASURED AFTER: nan with the same
    one warning naming group B.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    value, warned = _caught(
        monitor.compute_demographic_parity, _one_group_unscored(), "group_gender"
    )
    assert np.isnan(value), (
        f"an undefined group rate was published as a demographic parity gap of "
        f"{value!r}, which is perfect parity; warnings: {warned}"
    )
    assert any("could NOT be measured" in w and "B (50 of 50" in w for w in warned), warned


def test_the_answer_no_longer_depends_on_which_group_numpy_groups_first():
    """The honesty of both metrics above rested on dict ORDER, not on a guard.

    SAME SUBJECT, corrected direction. This test used to pass by asserting the
    order dependence, and it was the proof that the two results above were an
    accident of ``max``/``min`` rather than a decision: the unmeasurable group
    FIRST was refused, the same data with it SECOND answered 1.0 and 0.0.

    MEASURED BEFORE: nan / nan with the group first, 1.0 / 0.0 with it second.
    MEASURED AFTER: nan / nan in BOTH orders, each with its warning, so no
    reordering of the input can change the verdict.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    for label, frame in (
        ("unmeasurable group first", _one_group_unscored(first_group_unscored=True)),
        ("unmeasurable group second", _one_group_unscored()),
    ):
        ratio, ratio_warned = _caught(monitor.compute_disparate_impact, frame, "group_gender")
        gap, gap_warned = _caught(monitor.compute_demographic_parity, frame, "group_gender")
        assert np.isnan(ratio), f"{label}: disparate impact {ratio!r}, warnings {ratio_warned}"
        assert np.isnan(gap), f"{label}: parity gap {gap!r}, warnings {gap_warned}"
        assert any("could NOT be measured" in w for w in ratio_warned), (label, ratio_warned)
        assert any("could NOT be measured" in w for w in gap_warned), (label, gap_warned)


def test_the_other_monitor_in_this_library_refuses_the_same_input():
    """``operations.cicd.monitor`` answers nan for the identical rows.

    Not an xfail. This is the evidence that the tracker behaviour above is a
    defect and not this library's intended reading of an unscored row.
    """
    y_pred = np.array([1.0] * 45 + [0.0] * 5 + [NAN] * 50)
    y_true = np.array([1] * 100)
    groups = np.array(["A"] * 50 + ["B"] * 50)
    monitor = BiasMonitor(baseline_metrics={}, config=MonitorConfig(metrics_to_monitor=[]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = monitor._compute_default_metrics(y_true, y_pred, groups)
    assert np.isnan(metrics["demographic_parity_difference"])
    assert np.isnan(metrics["equalized_odds_difference"])


# ===========================================================================
# 3 and 4. Positive-label rows that carry NO prediction
# ===========================================================================


def _half_of_one_group_unscored() -> pd.DataFrame:
    """Group A TPR 0.5 genuinely; group B has 20 scored rows, all positive.

    Over the rows that were actually scored the TPR gap is 0.5. Counting the 20
    unscored rows as negative predictions drags B's reported rate to exactly
    A's, and the metric prints perfect parity.
    """
    rows = []
    rows += [("A", 1, 1)] * 20 + [("A", 1, 0)] * 20
    rows += [("A", 0, 1)] * 10 + [("A", 0, 0)] * 30
    rows += [("B", 1, 1)] * 20 + [("B", 1, NAN)] * 20
    rows += [("B", 0, 1)] * 10 + [("B", 0, 0)] * 30
    return pd.DataFrame(rows, columns=["group_gender", "label", "prediction"])


def test_unscored_positive_label_rows_are_not_perfect_equal_opportunity():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: compute_equal_opportunity returned np.float64(0.0), perfect
    equal opportunity, with zero warnings. (pos[pred] == 1).mean() counts a row
    with NO prediction as a row that was not selected, so B read 20/40 = 0.5
    against A's genuine 0.5 while the TPR over B's SCORED rows is 1.0.
    MEASURED AFTER: nan with one warning, "1 group(s) of 'group_gender' hold
    positive-label rows that were never scored ... B: 20 of 40 positive-label
    row(s)".
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    value, warned = _caught(
        monitor.compute_equal_opportunity, _half_of_one_group_unscored(), "group_gender"
    )
    assert np.isnan(value) or value == pytest.approx(0.5), (
        f"equal opportunity was published as {value!r} over a window where half of "
        f"one group's positive-label rows were never scored; warnings: {warned}"
    )
    assert any("never scored" in w and "B: 20 of 40" in w for w in warned), warned


def test_unscored_positive_label_rows_are_not_perfect_equalized_odds():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: compute_equalized_odds returned np.float64(0.0) with zero
    warnings on the same frame. Both arms held two DEFINED rates, so the
    len(valid) < 2 refusal the grade names could not fire; the unmeasured rows
    entered the rates themselves rather than removing one. MEASURED AFTER: nan
    with one warning naming "B: 20 of 40 positive-label row(s) and 0 of 40
    negative-label row(s) carry no prediction", raised ABOVE the two len(valid)
    dispatches.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    value, warned = _caught(
        monitor.compute_equalized_odds, _half_of_one_group_unscored(), "group_gender"
    )
    assert np.isnan(value) or value == pytest.approx(0.5), (
        f"equalized odds was published as {value!r} over a window where half of one "
        f"group's positive-label rows were never scored; warnings: {warned}"
    )
    assert any("never scored" in w and "B: 20 of 40 positive-label" in w for w in warned), warned


def test_the_monitor_snapshot_no_longer_publishes_the_fabricated_all_clear():
    """The consequence at the surface a dashboard reads.

    SAME SUBJECT, corrected direction. This used to document what
    ``update_and_check`` published: metrics 1.0 and 0.0, alerts
    {di: False, dp: False}, any_alert False and get_explanation().severity
    "info", the whole chain reading clean. Its own ``if not np.isnan(val)`` guard
    was correct and load-bearing all along; it was handed a finite 1.0 and could
    not know the 1.0 was invented, which is why the fix belongs in the metric and
    not here.

    MEASURED AFTER: metrics both nan, so that guard now fires and records NO
    threshold comparison, alerts {}, any_alert None at the object AND the JSON
    boundary, group_rates {'A': 0.9, 'B': nan}, and the explanation severity
    "medium" (_UNKNOWN_SEV, could-not-check) instead of "info".
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    snapshot, warned = _caught(monitor.update_and_check, _one_group_unscored())
    assert np.isnan(snapshot.metrics["disparate_impact_group_gender"])
    assert np.isnan(snapshot.metrics["demographic_parity_group_gender"])
    assert snapshot.alerts == {}, snapshot.alerts
    assert snapshot.any_alert is None
    assert snapshot.to_dict()["any_alert"] is None
    assert np.isnan(snapshot.group_rates["group_gender"]["B"])
    assert snapshot.group_rates["group_gender"]["A"] == 0.9
    assert len([w for w in warned if "could NOT be measured" in w]) == 2, warned
    assert monitor.get_explanation().severity == "medium"


# ===========================================================================
# 5. get_alert_summary: the second door into "nothing was compared"
# ===========================================================================


def test_a_monitor_that_ran_but_compared_nothing_also_says_so():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: a monitor that RAN one window and compared NOTHING returned
    {} in silence. The earlier fix covered only the never-ran door (no
    ``_history``); this door is the one the method's own docstring names, and it
    is just as distinguishable: no snapshot carries any alert determination.
    MEASURED AFTER: {} plus "1 window(s) were monitored and nothing was compared
    to a threshold in any of them, so the empty result is could-not-check".
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig())
    frame = pd.DataFrame({"prediction": [1, 0] * 60, "label": [1, 0] * 60})
    snapshot, _ = _caught(monitor.update_and_check, frame)
    assert snapshot.alerts == {} and snapshot.any_alert is None, "fixture drifted"

    summary, warned = _caught(monitor.get_alert_summary)
    assert summary == {}
    assert any("nothing was compared" in m for m in warned), (
        "a monitor that compared nothing returned an empty summary in silence, and "
        "`if not monitor.get_alert_summary()` reads that as a clean bill"
    )


def test_the_named_control_for_that_grade_is_itself_a_compared_nothing_window():
    """The pin named as evidence would have REFUSED the fix above.

    Not an xfail, and still green: the frame is rebuilt here exactly as that
    control built it, so this remains the standing proof of WHY the fixture had
    to change rather than the guard.

    ``test_control_a_monitor_that_did_run_and_found_nothing_stays_silent`` built
    its "monitored and clean" fixture with the columns ``gender`` / ``y_true`` /
    ``y_pred``, while the configured ``prediction_col`` is ``"prediction"``. The
    monitor itself warns that every metric is NaN and that NO threshold
    comparison was made, and the control then asserted that ``get_alert_summary``
    must stay silent about exactly that state. Applied in memory on 2026-09-27,
    a correct fix made that control FAIL with "a real monitored-and-clean result
    was drowned in a could-not-check caveat".

    CLOSED BGL5, 2026-09-27: the FIXTURE was corrected, not the guard, and not
    the control's subject. ``_clean_batch`` now feeds ``group_gender`` /
    ``label`` / ``prediction`` with both groups selected at exactly 0.5, so the
    control is a genuinely compared-and-clean window (disparate impact 1.0,
    parity gap 0.0, both alerts False, any_alert False) and it asserts that, so
    it cannot silently become a compared-nothing window again.
    """
    rng = np.random.default_rng(20260925)
    control_fixture = pd.DataFrame(
        {
            "gender": np.where(rng.random(400) < 0.5, "F", "M"),
            "y_true": (rng.random(400) < 0.5).astype(int),
            "y_pred": (rng.random(400) < 0.5).astype(int),
        }
    )
    monitor = FairnessMonitor()
    snapshot, warned = _caught(monitor.update_and_check, control_fixture)

    assert any("NO threshold comparison was made" in m for m in warned), warned
    assert all(np.isnan(v) for v in snapshot.metrics.values())
    assert snapshot.alerts == {}
    assert snapshot.any_alert is None


# ===========================================================================
# 6 and 7. The reporting surfaces
# ===========================================================================


def _generator() -> ReportGenerator:
    return ReportGenerator(MetricsStore(), config=ReportConfig(time_window=timedelta(days=7)))


def test_a_flag_does_not_establish_a_threshold_breach():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: handed True, the page published 'Current Value: 1.0000',
    'Breach Magnitude: 900.0% beyond threshold', the recommendation 'has breached
    its 0.10 threshold' and JSON '"current_value": true, "breach_pct": 900.0,
    "comparison_made": true'. _is_comparable was math.isfinite(float(v)), and
    float(True) is 1.0; the library's canonical predicate
    vfairness._triage.is_measured rejects a bool for this exact reason.
    MEASURED AFTER: 'Current Value: not measured (a yes/no flag, not a
    measurement)', 'Breach Magnitude: COULD NOT CHECK ...', no 'has breached'
    recommendation, and JSON '"breach_pct": null, "comparison_made": false'.
    """
    generator = _generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = generator.generate_threshold_breach_report(
            "demographic_parity_difference", True, 0.10, OutputFormat.MARKDOWN
        )
    breach_section = next(s for s in report.sections if s["title"] == "Threshold Breach")
    assert "COULD NOT CHECK" in breach_section["content"], breach_section["content"]
    assert "900.0%" not in breach_section["content"], breach_section["content"]
    assert not [r for r in report.recommendations if "has breached" in r], report.recommendations


def test_a_numeric_string_no_longer_passes_the_guard_and_crashes_the_report():
    """Corroboration for the grade above: ``_is_comparable`` was not the predicate.

    SAME SUBJECT, corrected direction. It coerced with ``float(value)`` to
    answer, and the arithmetic that follows uses the RAW value, so a value the
    guard accepted could still be one the method cannot subtract.

    MEASURED BEFORE: ``TypeError: unsupported operand type(s) for -: 'str' and
    'float'`` raised out of the public method at reports.py:1290. MEASURED AFTER:
    no exception; the predicate and the arithmetic agree, the report refuses and
    names the reason. A report does not invent a parse of its own input.
    """
    generator = _generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = generator.generate_threshold_breach_report(
            "dp", "0.45", 0.10, OutputFormat.MARKDOWN
        )
    breach_section = next(s for s in report.sections if s["title"] == "Threshold Breach")
    assert "COULD NOT CHECK" in breach_section["content"], breach_section["content"]
    assert "not a number (str)" in breach_section["content"], breach_section["content"]
    assert not [r for r in report.recommendations if "has breached" in r], report.recommendations


def test_an_alert_with_no_severity_does_not_title_the_page_none():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE: the report TITLE was built by the same wrong idiom the grade
    says it fixed. The body correctly said 'Severity: UNSCORED (not determined)'
    while ``d.get('severity', 'ALERT')`` at three title sites does not fire for a
    key PRESENT holding None, so GeneratedReport.title, the markdown H1 and the
    HTML <title> all read 'Alert Report: None'. MEASURED AFTER: all four surfaces
    read 'UNSCORED (not determined)', from one ``_severity_word`` call.
    """
    generator = _generator()
    alert = {
        "severity": None,
        "metric_name": "demographic_parity_difference",
        "message": "drift observed",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = generator.generate_alert_report(dict(alert), OutputFormat.HTML)
    assert "None" not in report.title, report.title
    assert report.title == "Alert Report: UNSCORED (not determined)", report.title
    assert "<title>Alert Report: None</title>" not in report.content
    assert "<title>Alert Report: UNSCORED (not determined)</title>" in report.content

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        markdown = generator.generate_alert_report(dict(alert), OutputFormat.MARKDOWN)
    assert markdown.content.splitlines()[0] == "# Alert Report: UNSCORED (not determined)"
    assert "Severity: UNSCORED (not determined)" in markdown.sections[0]["content"]


def test_a_decimal_measurement_is_not_reported_as_never_compared():
    """CLOSED BGL5, 2026-09-27. Was xfail(strict=True).

    MEASURED BEFORE, the READINESS-6 defect running BACKWARDS in the function
    grades 23 and 24 rest on: _assessed_rows tested measurability with
    isinstance(v, (int, float)), which is False for decimal.Decimal, so three rows
    carrying alert=True and a real finite value were reported as 'never compared
    to a threshold, or could not be measured at all'. _triage names Decimal
    explicitly as a value that reaches a metric from a database NUMERIC column.
    MEASURED AFTER: 3 graded rows and an empty not_assessed list, because the
    predicate is now ``_triage.is_measured`` itself.
    """
    from datetime import datetime

    from vfairness.operations.reporting.store import StoredMetricRecord

    store = MetricsStore()
    now = datetime.now()
    for i in range(3):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(hours=i + 1),
                source="monitor",
                metric_name="demographic_parity_difference",
                value=decimal.Decimal("0.91"),
                group="overall",
                group_size=500,
                alert=True,
            )
        )
    frame = store.get_metrics()
    graded, not_assessed = _assessed_rows(frame)
    assert len(graded) == 3, (
        f"{3 - len(graded)} of 3 measured, breaching rows were reclassified as "
        f"could-not-check and named as {not_assessed}"
    )
    assert not_assessed == [], not_assessed
