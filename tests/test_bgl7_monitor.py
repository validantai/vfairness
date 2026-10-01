"""BGL7 monitor lane: five overturned grades on ``FairnessMonitor``.

An independent audit returned 12 of 12 OVERTURNED across the CI/CD gate and the
monitors; the five pinned here are the monitor half. A monitor decides whether a
live model has drifted, so a false clean reading here is a disparity nobody is
told about.

All five are the same defect class: a NEUTRAL value (0.0, 1.0, {}) substituted
for one that could not be measured, then published as a measurement.

    1. compute_equal_opportunity   0.0  warnings []   partial LABEL loss
    2. compute_equalized_odds      0.0  warnings []   partial LABEL loss, both arms
    3. compute_disparate_impact    1.0  warnings []   a SCORE column
    4. compute_demographic_parity  0.0  warnings []   a SCORE column
    5. get_alert_summary           {}   warnings []   a metric never compared

Each is pinned with its measured before-state, and each has an OVER-CORRECTION
CONTROL asserting the healthy case's REAL number, because a monitor that refuses
everything reports no drift ever and passes every refusal test.

SABOTAGE RECORD, measured 2026-09-30. An unsabotaged guard is indistinguishable
from one that cannot fail, so each half of each fix was disabled in turn and this
file plus ``test_bgl6_f03.py`` and ``test_bgl5_operations_2.py`` were run:

    OMP_NUM_THREADS=1 ./.venv/bin/python -m pytest -q -p no:randomly \
      tests/test_bgl7_monitor.py tests/test_bgl6_f03.py tests/test_bgl5_operations_2.py

Green baseline: 61 passed. Each sabotage was reverted from a byte-identical copy
and ``diff -q`` confirmed silent afterwards.

  1. ``if unlabelled:`` -> ``if False:`` in ``compute_equal_opportunity``
     1 failed, 60 passed
       test_partial_label_loss_is_not_perfect_equal_opportunity
  2. ``if unlabelled:`` -> ``if False:`` in ``compute_equalized_odds``
     1 failed, 60 passed
       test_partial_label_loss_is_not_perfect_equalized_odds
  3. ``if _unthresholded_scores(values) > 0:`` -> ``if False:`` in ``_selected_share``
     4 failed, 57 passed
       test_a_score_column_is_not_a_perfect_disparate_impact
       test_the_monitor_snapshot_publishes_could_not_check_for_a_score_column
       test_control_the_four_builtins_now_agree_about_one_frame
       test_bgl6_f03::test_overturn_equal_opportunity_reports_perfect_parity_for_a_score_column
  4. ``_selected_share(sub[pred_col])`` -> ``float(sub[pred_col].mean())`` in
     ``_group_positive_rates``, which is the DELEGATION itself rather than the
     predicate, so it has to be sabotaged separately: with ``_selected_share``
     fully intact the two rate metrics go back to scoring a score column.
     5 failed, 56 passed
       test_a_score_column_is_not_a_perfect_disparate_impact
       test_the_monitor_snapshot_publishes_could_not_check_for_a_score_column
       test_control_the_four_builtins_now_agree_about_one_frame
       test_bgl6_f03::test_overturn_a_mostly_unmonitored_history_still_returns_a_silent_empty
       test_bgl5_operations_2::test_a_rate_over_the_scored_subset_is_not_the_groups_rate
  5. ``uncompared = sorted(...)`` -> ``uncompared = []`` in ``get_alert_summary``
     2 failed, 59 passed
       test_a_metric_never_compared_in_any_window_is_disclosed
       test_a_metric_compared_in_only_some_windows_is_disclosed_too

No control failed under any sabotage, which is the other half of the reading: the
controls hold the REAL numbers and a disabled refusal does not make them red.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    _unlabelled_rows,
    _unthresholded_scores,
)

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


def _monitor(**kwargs) -> FairnessMonitor:
    return FairnessMonitor(config=FairnessMonitorConfig(min_samples=30, **kwargs))


# ===========================================================================
# 1. compute_equal_opportunity: the unmeasurable LABEL
# ===========================================================================


def _half_of_one_group_unlabelled() -> pd.DataFrame:
    """A: 40 positive-label rows, 20 selected, TPR 0.500.

    B: 40 genuinely positive rows. 20 carry ``label = 1`` and 10 of those are
    selected; the other 20 carry ``label = NaN`` and ARE selected. So the TPR over
    B's LABELLED rows is 10/20 = 0.500, exactly A's, and B's real TPR over all 40
    of its positive rows is 30/40 = 0.750.
    """
    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": 1.0, "prediction": 1 if i < 20 else 0})
    for i in range(20):
        rows.append({"group_gender": "B", "label": 1.0, "prediction": 1 if i < 10 else 0})
    for _ in range(20):
        rows.append({"group_gender": "B", "label": NAN, "prediction": 1})
    return pd.DataFrame(rows)


def test_partial_label_loss_is_not_perfect_equal_opportunity():
    """MEASURED BEFORE (2026-09-30, min_samples=30):

        compute_equal_opportunity -> 0.0   warnings []

    which is the textbook PERFECT value, against a real gap of 0.250.

    The unscored-prediction fix guards the unmeasurable PREDICTION; the
    unmeasurable LABEL was unguarded. ``pos = grp[grp[label] == 1]`` is False for
    a NaN label exactly as ``(pos[pred] == 1)`` was False for a NaN prediction,
    and ``_missing_column_nan`` only checks that the label COLUMN exists. TOTAL
    label loss WAS refused, because every group then has an empty positive arm and
    the ``len(valid) < 2`` door fires; PARTIAL label loss passed. That is the
    coverage-guard-fires-only-on-zero shape.

    MEASURED AFTER: nan, with one warning naming group B and counting its 20
    unlabelled rows of 40.
    """
    value, warned = _caught(
        _monitor().compute_equal_opportunity, _half_of_one_group_unlabelled(), "group_gender"
    )

    assert np.isnan(value), (
        f"equal opportunity was published as {value!r} over a window where half of "
        f"group B's positive rows carry no label, so its TPR was computed over the "
        f"labelled half and matched group A exactly; warnings: {warned}"
    )
    assert any("no usable ground-truth label" in w and "B: 20 of 40 row(s)" in w for w in warned), (
        warned
    )


def test_control_a_real_equal_opportunity_gap_is_still_measured_exactly():
    """OVER-CORRECTION CONTROL. The same shape with every label present: A 20 of
    40 selected (TPR 0.500) and B 30 of 40 (TPR 0.750), which is the real gap the
    fixture above hides. Must be exactly 0.25 and silent, or the fix has turned a
    monitor into a machine that reports no drift ever.
    """
    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": 1, "prediction": 1 if i < 20 else 0})
    for i in range(40):
        rows.append({"group_gender": "B", "label": 1, "prediction": 1 if i < 30 else 0})
    frame = pd.DataFrame(rows)

    value, warned = _caught(_monitor().compute_equal_opportunity, frame, "group_gender")

    assert value == pytest.approx(0.25), value
    assert warned == [], warned


# ===========================================================================
# 2. compute_equalized_odds: the same door, both arms
# ===========================================================================


def _half_of_one_group_unlabelled_both_arms() -> pd.DataFrame:
    """A: TPR 20/40 = 0.500, FPR 10/40 = 0.250.

    B: 40 genuinely positive rows of which 20 carry ``label = NaN`` and ARE
    selected (TPR over the labelled 20 is 10/20 = 0.500, real TPR 30/40 = 0.750),
    plus 40 negative-label rows with FPR 10/40 = 0.250.

    BOTH ARMS therefore hold two DEFINED rates, which is what put the
    ``len(valid_tprs) < 2`` refusal out of reach by construction.
    """
    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": 1.0, "prediction": 1 if i < 20 else 0})
    for i in range(40):
        rows.append({"group_gender": "A", "label": 0.0, "prediction": 1 if i < 10 else 0})
    for i in range(20):
        rows.append({"group_gender": "B", "label": 1.0, "prediction": 1 if i < 10 else 0})
    for _ in range(20):
        rows.append({"group_gender": "B", "label": NAN, "prediction": 1})
    for i in range(40):
        rows.append({"group_gender": "B", "label": 0.0, "prediction": 1 if i < 10 else 0})
    return pd.DataFrame(rows)


def test_partial_label_loss_is_not_perfect_equalized_odds():
    """MEASURED BEFORE (2026-09-30, min_samples=30):

        compute_equalized_odds -> 0.0   warnings []

    PERFECT equalized odds, against a real gap of max(0.250 TPR, 0.0 FPR) = 0.250.

    Both arms held two DEFINED rates on this frame, so the ``len(valid_tprs) < 2``
    refusal could never fire: nothing was REMOVED from the TPR arm, a rate was
    silently MOVED, from 0.750 to 0.500. That is why the guard has to be the label
    predicate itself and has to sit above both dispatches.

    MEASURED AFTER: nan, with one warning naming group B and its 20 unlabelled
    rows of 80.
    """
    value, warned = _caught(
        _monitor().compute_equalized_odds,
        _half_of_one_group_unlabelled_both_arms(),
        "group_gender",
    )

    assert np.isnan(value), (
        f"equalized odds was published as {value!r} over a window where 20 of group "
        f"B's 40 positive rows carry no label; warnings: {warned}"
    )
    assert any("no usable ground-truth label" in w and "B: 20 of 80 row(s)" in w for w in warned), (
        warned
    )


def test_control_a_real_equalized_odds_gap_is_still_measured_exactly():
    """OVER-CORRECTION CONTROL. The same shape fully labelled: A TPR 0.500 /
    FPR 0.250, B TPR 0.750 / FPR 0.250, so the answer is exactly max(0.25, 0.0).
    """
    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": 1, "prediction": 1 if i < 20 else 0})
    for i in range(40):
        rows.append({"group_gender": "A", "label": 0, "prediction": 1 if i < 10 else 0})
    for i in range(40):
        rows.append({"group_gender": "B", "label": 1, "prediction": 1 if i < 30 else 0})
    for i in range(40):
        rows.append({"group_gender": "B", "label": 0, "prediction": 1 if i < 10 else 0})
    frame = pd.DataFrame(rows)

    value, warned = _caught(_monitor().compute_equalized_odds, frame, "group_gender")

    assert value == pytest.approx(0.25), value
    assert warned == [], warned


def test_control_a_boolean_label_column_is_still_a_label():
    """OVER-CORRECTION CONTROL on the label predicate. ``True`` IS the positive
    class and ``grp[label] == 1`` matches it, so a boolean label column must be
    measured, not refused. Only a NULLABLE boolean's pd.NA is an absent label.
    """
    numpy_bools = pd.Series([True, False, True], dtype=bool)
    nullable = pd.Series([True, None, False], dtype="boolean")

    assert _unlabelled_rows(numpy_bools) == 0
    assert _unlabelled_rows(nullable) == 1
    # A label of 2 is in neither arm of a binary-label metric, so it is unusable
    # for the same reason a NaN is: FairnessMonitorConfig.label_col says 0/1.
    assert _unlabelled_rows(pd.Series([0, 1, 2, 1])) == 1

    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": True, "prediction": 1 if i < 20 else 0})
    for i in range(40):
        rows.append({"group_gender": "B", "label": True, "prediction": 1 if i < 30 else 0})
    frame = pd.DataFrame(rows)
    value, warned = _caught(_monitor().compute_equal_opportunity, frame, "group_gender")

    assert value == pytest.approx(0.25), value
    assert warned == [], warned


# ===========================================================================
# 3 and 4. A SCORE column has no positive-prediction rate either
# ===========================================================================


def _one_group_all_mid_scores() -> pd.DataFrame:
    """50 group-A rows scoring 1.0 and 0.0 half and half, 50 group-B rows all 0.5.

    At any decision threshold t in [0.5, 1) group A selects 0.5 of its rows and
    group B selects 0.0, so the real ratio is 0.0 and the real gap 0.5.
    """
    rows = []
    for i in range(50):
        rows.append({"group_gender": "A", "label": 1, "prediction": 1.0 if i < 25 else 0.0})
    for _ in range(50):
        rows.append({"group_gender": "B", "label": 1, "prediction": 0.5})
    return pd.DataFrame(rows)


def test_a_score_column_is_not_a_perfect_disparate_impact():
    """MEASURED BEFORE (2026-09-30, min_samples=30):

        _group_positive_rates       -> {'A': 0.5, 'B': 0.5}
        compute_disparate_impact    -> 1.0   warnings []
        compute_demographic_parity  -> 0.0   warnings []
        update_and_check            -> alerts {both False}, any_alert False
        _selected_share(B's column) -> nan   (the IDENTICAL column, refused)

    A PERFECT four-fifths ratio and PERFECT parity over a population nobody chose.
    ``prediction_col`` is documented as "binary 0/1 or scores", a previous wave
    decided a selected-share is UNDEFINED for a score column, and it put that
    refusal on ``_selected_share`` (which only the two label-dependent metrics
    reach) and not on ``_group_positive_rates`` (which these two reach). So one
    frame made the four built-ins disagree about whether the model discriminates.

    MEASURED AFTER: nan from both, each with one warning naming group B and saying
    its rows hold a SCORE rather than a 0/1 decision.
    """
    frame = _one_group_all_mid_scores()
    monitor = _monitor()

    ratio, ratio_warned = _caught(monitor.compute_disparate_impact, frame, "group_gender")
    gap, gap_warned = _caught(monitor.compute_demographic_parity, frame, "group_gender")

    assert np.isnan(ratio), (
        f"a mean score was published as disparate impact {ratio!r}, which passes the "
        f"0.8 four-fifths floor; warnings: {ratio_warned}"
    )
    assert np.isnan(gap), (
        f"a mean score was published as a parity gap of {gap!r}, which is perfect "
        f"parity; warnings: {gap_warned}"
    )
    for warned in (ratio_warned, gap_warned):
        assert any("could NOT be measured" in w and "hold a SCORE" in w for w in warned), warned
        assert any("B (50 of 50 row(s) hold a SCORE" in w for w in warned), warned


def test_the_monitor_snapshot_publishes_could_not_check_for_a_score_column():
    """The surface a dashboard reads. Both metrics nan, so the ``not np.isnan``
    guard records NO threshold comparison: alerts {}, any_alert None at the object
    AND the JSON boundary, and group B's published rate is nan rather than 0.5.
    """
    monitor = _monitor()
    snapshot, warned = _caught(monitor.update_and_check, _one_group_all_mid_scores())

    assert np.isnan(snapshot.metrics["disparate_impact_group_gender"])
    assert np.isnan(snapshot.metrics["demographic_parity_group_gender"])
    assert snapshot.alerts == {}, snapshot.alerts
    assert snapshot.any_alert is None
    assert snapshot.to_dict()["any_alert"] is None
    assert np.isnan(snapshot.group_rates["group_gender"]["B"])
    assert len([w for w in warned if "could NOT be measured" in w]) == 2, warned


def test_control_a_binary_and_a_boolean_column_are_still_measured_exactly():
    """OVER-CORRECTION CONTROL, the one that stops this becoming a refusal machine.

    A 45 of 50 against B 10 of 50, every row a 0/1 decision: ratio exactly 0.2/0.9
    and gap exactly 0.7, no warning. The same data as booleans: identical. And the
    predicate itself must count zero score rows for both.
    """
    binary = pd.DataFrame(
        {
            "prediction": [1] * 45 + [0] * 5 + [1] * 10 + [0] * 40,
            "label": [1] * 100,
            "group_gender": ["A"] * 50 + ["B"] * 50,
        }
    )
    booleans = binary.assign(prediction=binary["prediction"].astype(bool))
    monitor = _monitor()

    assert _unthresholded_scores(binary["prediction"]) == 0
    assert _unthresholded_scores(booleans["prediction"]) == 0
    # A float column holding only 0.0 and 1.0 is a decision column too.
    assert _unthresholded_scores(pd.Series([1.0, 0.0, 1.0])) == 0
    assert _unthresholded_scores(pd.Series([0.9, 0.0])) == 1
    # An unscored row is the OTHER state and is not counted here.
    assert _unthresholded_scores(pd.Series([1.0, NAN, 0.0])) == 0

    for frame in (binary, booleans):
        ratio, ratio_warned = _caught(monitor.compute_disparate_impact, frame, "group_gender")
        gap, gap_warned = _caught(monitor.compute_demographic_parity, frame, "group_gender")
        assert ratio == pytest.approx(0.2 / 0.9), (frame["prediction"].dtype, ratio)
        assert gap == pytest.approx(0.7), (frame["prediction"].dtype, gap)
        assert ratio_warned == [] and gap_warned == []


def test_control_the_four_builtins_now_agree_about_one_frame():
    """The cross-check the BGL6 wave named as the point of its own fix: on a frame
    whose predictions are scores, all four built-ins must answer could-not-check.
    Two of them already did; these two published 1.0 and 0.0 beside them.
    """
    frame = pd.DataFrame(
        {
            "group_gender": ["A"] * 80 + ["B"] * 80,
            "label": ([1] * 40 + [0] * 40) * 2,
            "prediction": ([0.9] * 40 + [0.0] * 40) + ([0.1] * 40 + [0.0] * 40),
        }
    )
    monitor = _monitor()

    assert np.isnan(_quiet(monitor.compute_disparate_impact, frame, "group_gender"))
    assert np.isnan(_quiet(monitor.compute_demographic_parity, frame, "group_gender"))
    assert np.isnan(_quiet(monitor.compute_equal_opportunity, frame, "group_gender"))
    assert np.isnan(_quiet(monitor.compute_equalized_odds, frame, "group_gender"))


# ===========================================================================
# 5. get_alert_summary: the guard counts WINDOWS, the result is keyed by METRIC
# ===========================================================================


def _nobody_selected() -> pd.DataFrame:
    """Both groups present and above min_samples, nobody selected.

    Both positive rates are 0.0, so the privileged rate is 0 and
    ``compute_disparate_impact`` refuses (a ratio needs a non-zero denominator)
    while ``compute_demographic_parity`` measures 0.0 - 0.0 = 0.0 and IS compared
    to its threshold. So the window carries a comparison and disparate impact is
    not part of it.
    """
    return pd.DataFrame(
        {
            "group_gender": ["A"] * 40 + ["B"] * 40,
            "label": [1] * 80,
            "prediction": [0] * 80,
        }
    )


def _compared_and_clean() -> pd.DataFrame:
    """Both groups selected at exactly 0.5: DI 1.0, DP 0.0, both compared, clean."""
    return pd.DataFrame(
        {
            "group_gender": ["A"] * 40 + ["B"] * 40,
            "label": [1] * 80,
            "prediction": ([1] * 20 + [0] * 20) * 2,
        }
    )


def test_a_metric_never_compared_in_any_window_is_disclosed():
    """MEASURED BEFORE (2026-09-30), three windows of ``_nobody_selected``:

        per window: metrics {disparate_impact: nan, demographic_parity: 0.0}
                    alerts  {demographic_parity_group_gender: False}
        windows carrying a comparison: 3 of 3
        get_alert_summary() -> {}   warnings []

    BYTE-IDENTICAL to a three-window fully compared clean history, which also
    returns {} in silence. The four-fifths rule was never applied to a single
    window and ``if not monitor.get_alert_summary()`` read that as a clean bill on
    it.

    Every guard on this method counted WINDOWS (``if not any(snap.alerts)``,
    ``compared = sum(... if snap.alerts)``) while the value it returns is keyed by
    METRIC, and a window carries a comparison as soon as ONE of its metrics was
    compared. So no door could see a metric that was never compared in any window.

    MEASURED AFTER: {} with one warning naming ``disparate_impact_group_gender``,
    computed in 3 window(s) and compared in 0.
    """
    monitor = _monitor(window_size=10_000)
    for _ in range(3):
        snapshot = _quiet(monitor.update_and_check, _nobody_selected())
    assert np.isnan(snapshot.metrics["disparate_impact_group_gender"]), "fixture drifted"
    assert snapshot.alerts == {"demographic_parity_group_gender": False}, snapshot.alerts
    assert sum(1 for s in monitor._history if s.alerts) == 3, "fixture drifted"

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert len(warned) == 1, warned
    assert "disparate_impact_group_gender" in warned[0], warned[0]
    assert "computed in 3 window(s) and compared to a threshold in 0" in warned[0], warned[0]
    assert "NOT a finding that no metric breached" in warned[0], warned[0]
    # And the disclosure must not read as "no window was monitored", which is a
    # different state with its own door.
    assert "demographic_parity_group_gender was computed" not in warned[0], warned[0]


def test_a_metric_compared_in_only_some_windows_is_disclosed_too():
    """The partial-coverage case at metric granularity: one window in which
    disparate impact WAS compared beside two in which it was not. Before, the
    per-window guard saw 3 of 3 covered and said nothing about the metric.
    """
    # window_size=80 so each batch REPLACES the previous one in the sliding
    # window. At 10_000 the window is cumulative and window 2 holds the clean rows
    # as well, which keeps both rates non-zero and disparate impact measurable in
    # every window: the fixture would then prove nothing, silently.
    monitor = _monitor(window_size=80)
    _quiet(monitor.update_and_check, _compared_and_clean())
    for _ in range(2):
        _quiet(monitor.update_and_check, _nobody_selected())
    judged = sum(1 for s in monitor._history if "disparate_impact_group_gender" in s.alerts)
    assert judged == 1, "fixture drifted: disparate impact was compared in {judged} windows"

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert len(warned) == 1, warned
    assert "computed in 3 window(s) and compared to a threshold in 1" in warned[0], warned[0]


def test_control_a_fully_compared_clean_history_stays_silent():
    """OVER-CORRECTION CONTROL. "Every metric was compared in every window and
    nothing breached" is a real finding and must not be drowned in a caveat, or the
    disclosure becomes noise that a reader learns to skip.
    """
    monitor = _monitor(window_size=10_000)
    for _ in range(3):
        snapshot, ingest_warned = _caught(monitor.update_and_check, _compared_and_clean())
        assert snapshot.metrics["disparate_impact_group_gender"] == 1.0, snapshot.metrics
        assert snapshot.metrics["demographic_parity_group_gender"] == 0.0, snapshot.metrics
        assert snapshot.alerts == {
            "disparate_impact_group_gender": False,
            "demographic_parity_group_gender": False,
        }, snapshot.alerts
        assert snapshot.any_alert is False
        assert ingest_warned == [], ingest_warned

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert warned == [], warned


def test_control_a_real_breach_is_still_counted_per_window_and_silent():
    """OVER-CORRECTION CONTROL. 36 of 40 against 4 of 40 every window: ratio 1/9,
    gap 0.8, both metrics compared and breaching in all three windows, so the
    summary counts three breaches per metric and says nothing else.
    """
    breaching = pd.DataFrame(
        {
            "group_gender": ["A"] * 40 + ["B"] * 40,
            "label": [1] * 80,
            "prediction": [1] * 36 + [0] * 4 + [1] * 4 + [0] * 36,
        }
    )
    monitor = _monitor(window_size=80, alert_cooldown_seconds=0.0)
    for _ in range(3):
        snapshot, ingest_warned = _caught(monitor.update_and_check, breaching)
        assert snapshot.metrics["disparate_impact_group_gender"] == pytest.approx(1 / 9)
        assert snapshot.metrics["demographic_parity_group_gender"] == pytest.approx(0.8)
        assert snapshot.any_alert is True
        assert ingest_warned == [], ingest_warned

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {
        "disparate_impact_group_gender": 3,
        "demographic_parity_group_gender": 3,
    }, summary
    assert warned == [], warned


# ===========================================================================
# THE SIBLING: BiasMonitor had the identical label hole
# ===========================================================================
#
# Found by asking what the sibling of defects 1 and 2 was, which is what the
# whole wave is about. operations/cicd/monitor.py is the module whose own source
# comment claims it was right about the NaN PREDICTION before the tracker was;
# it was wrong about the NaN LABEL in the same function, and the tracker's five
# fixes would have left it standing.
#
# SABOTAGE (2026-09-30): `if unlabelled:` -> `if False:` in
# _compute_default_metrics turned exactly the pin below RED (1 failed, and the
# control stayed green).


def _sibling_arrays():
    """group A: 40 label=1 rows, 20 predicted 1, TPR 0.500.

    group B: 20 label=1 rows with 10 predicted 1, plus 20 rows whose label is NaN
    and which ARE predicted 1. TPR over B's LABELLED rows 10/20 = 0.500, exactly
    A's; B's real TPR over all 40 of its positive rows is 30/40 = 0.750.
    """
    y_pred = np.array([1.0] * 20 + [0.0] * 20 + [1.0] * 10 + [0.0] * 10 + [1.0] * 20)
    y_true = np.array([1.0] * 40 + [1.0] * 20 + [NAN] * 20)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, groups


def _bias_monitor():
    from vfairness.operations.cicd.monitor import BiasMonitor, MonitorConfig

    return BiasMonitor(baseline_metrics={}, config=MonitorConfig(metrics_to_monitor=[]))


def test_the_sibling_monitor_had_the_same_unmeasurable_label_hole():
    """MEASURED BEFORE (2026-09-30):

        {'demographic_parity_difference': 0.25,
         'equalized_odds_difference': 0.0}   warnings []

    PERFECT equalized odds against a real gap of 0.250, published in the same dict
    as a demographic-parity reading that IS a measurement, so the two arms
    disagreed about one batch.

    ``any(r != r for r in tpr_rates)`` is aimed at a group with NO positive label,
    which leaves an UNDEFINED TPR. A group with SOME unusable labels leaves a
    perfectly DEFINED one, the rate over its labelled rows, so that guard could
    not see it: ``y_true == 1`` is False for a NaN label exactly as ``y_pred == 1``
    was False for a NaN prediction in the arm above it.

    MEASURED AFTER: nan, with one warning naming group B and counting its 20
    unlabelled rows of 40.
    """
    y_true, y_pred, groups = _sibling_arrays()
    metrics, warned = _caught(_bias_monitor()._compute_default_metrics, y_true, y_pred, groups)

    assert np.isnan(metrics["equalized_odds_difference"]), (
        f"equalized odds was published as {metrics['equalized_odds_difference']!r} "
        f"over a batch where half of group B's positive rows carry no label; "
        f"warnings: {warned}"
    )
    assert any("no usable ground-truth label" in w and "B: 20 of 40 row(s)" in w for w in warned), (
        warned
    )
    # The demographic-parity arm does not read the label column, so its 0.25 IS a
    # measurement and must not be collateral damage.
    assert metrics["demographic_parity_difference"] == pytest.approx(0.25)


def test_control_the_sibling_still_measures_a_real_tpr_gap_exactly():
    """OVER-CORRECTION CONTROL. The same shape fully labelled: A TPR 0.500 and B
    TPR 0.750, so equalized_odds_difference must be exactly 0.25 and silent.
    """
    y_pred = np.array([1.0] * 20 + [0.0] * 20 + [1.0] * 30 + [0.0] * 10)
    y_true = np.array([1.0] * 80)
    groups = np.array(["A"] * 40 + ["B"] * 40)

    metrics, warned = _caught(_bias_monitor()._compute_default_metrics, y_true, y_pred, groups)

    assert metrics["equalized_odds_difference"] == pytest.approx(0.25), metrics
    assert warned == [], warned
    # And an integer label column, and a boolean one, are labels rather than junk.
    for labels in (np.array([1] * 80), np.array([True] * 80)):
        out = _quiet(_bias_monitor()._compute_default_metrics, labels, y_pred, groups)
        assert out["equalized_odds_difference"] == pytest.approx(0.25), (labels.dtype, out)


def test_control_the_sibling_still_refuses_a_group_with_no_positive_label():
    """OVER-CORRECTION CONTROL on the guard this one sits above: a group with no
    positive label at all still yields nan through the ORIGINAL door, so the new
    refusal has not shadowed it.
    """
    y_pred = np.array([1.0] * 20 + [0.0] * 20 + [1.0] * 30 + [0.0] * 10)
    y_true = np.array([1.0] * 40 + [0.0] * 40)
    groups = np.array(["A"] * 40 + ["B"] * 40)

    metrics, warned = _caught(_bias_monitor()._compute_default_metrics, y_true, y_pred, groups)

    assert np.isnan(metrics["equalized_odds_difference"]), metrics
    assert not any("no usable ground-truth label" in w for w in warned), warned
