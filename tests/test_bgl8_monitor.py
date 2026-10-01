"""BGL8 monitor lane: the five BGL7 grades overturned a SECOND time.

The BGL7 wave closed the unmeasurable LABEL door on the two label metrics, the
SCORE-column door on the two rate metrics and the per-metric-key coverage door on
``get_alert_summary``. Every one of the auditor's own recipes for those five is
green at this commit (re-run and recorded in the lane report), so this file pins
the doors that were STILL open beside them.

    1/2/3/4  a TEXT-TYPED decision column. ``_unscored_predictions`` and
             ``_unthresholded_scores`` both read the prediction column through a
             NUMERIC coercion and answer "every row holds a clean 0/1 decision"
             about ``"1"``/``"0"``, and then ``(values == 1).mean()`` compared the
             RAW objects, where ``"1" == 1`` is False, so every row of the column
             counted as a row the model declined and the share was exactly 0.0.
             Measured 2026-09-30 on 80 rows per group, group A selecting EVERY one
             of its rows and group B selecting NONE, which is maximal disparity on
             all four built-ins:

                 compute_equal_opportunity   0.0  warnings []   truth 1.0
                 compute_equalized_odds      0.0  warnings []   truth 1.0
                 compute_demographic_parity  0.0  warnings []   truth 1.0
                 compute_disparate_impact    nan  (truth 0.0), and its refusal
                     reason read "no group was selected at all" while group A had
                     selected all 80 of its rows
                 update_and_check            demographic_parity 0.0, alert False,
                     any_alert False: a measured clean bill

             Three perfect scores from a total denial of selection to one group.
             The guard and the arithmetic looked at one column through different
             eyes, which is the BGL7 divergence lesson one level down.

    3        two SILENT refusals in ``compute_disparate_impact``. The independent
             audit named ``if privileged == 0: return np.nan`` as the silent
             refusal that reaches ``get_alert_summary``: measured 2026-09-30 on
             two groups in which nobody is selected, nan with ZERO warnings, so
             three such windows made 3 of 3 look fully compared while the
             four-fifths rule had never once been applied. ``if len(rates) < 2``
             was silent too.

    4        the same silent ``len(rates) < 2`` in ``compute_demographic_parity``,
             shared with 3 through one helper rather than repeated.

    5        ``get_alert_summary`` measured coverage over the metric keys the
             windows PRODUCED. A metric named in ``config.metrics_to_track`` that
             was never computed in any window had no key to be missing from and
             was invisible to all four earlier doors. Measured 2026-09-30, three
             windows with all four built-ins requested and no label column in the
             frame: ``{}`` with ZERO warnings, byte-identical to a history in
             which all four were computed, compared and clean. With a one-letter
             typo in the metric name there was no disclosure at ANY surface, at
             ingest or here.

Every pin is paired with an OVER-CORRECTION CONTROL asserting the healthy case's
REAL number, because a monitor that refuses everything reports no drift ever.

SABOTAGE RECORD, measured 2026-09-30. An unsabotaged guard is indistinguishable
from one that cannot fail, so every half of the fix was disabled in turn and this
command was run:

    OMP_NUM_THREADS=1 ./.venv/bin/python -m pytest -q -p no:randomly \
      tests/test_bgl8_monitor.py tests/test_bgl7_monitor.py tests/test_bgl6_f03.py \
      tests/test_bgl5_operations_2.py tests/test_bgl3_operations_2.py \
      tests/test_readiness_tracker.py

Green baseline: 133 passed. Each sabotage was greppable and the marker count was
asserted BEFORE the result was read, because a sabotage that failed to apply reads
exactly like a guard that passed. After every one the file was restored and
``diff -q`` against a pre-sabotage copy exited 0, with the sha256 unchanged.

  S1. ``(numeric == 1.0).mean()`` -> ``(values == 1).mean()`` in ``_selected_share``,
      which is the coercion itself.
      8 failed, 125 passed
        test_the_two_predicates_and_the_rate_now_read_the_same_column_the_same_way
        test_a_text_typed_decision_column_is_not_perfect_equal_opportunity
        test_a_text_typed_decision_column_is_not_perfect_equalized_odds
        test_a_text_typed_decision_column_is_not_a_perfect_four_fifths_ratio
        test_a_text_typed_decision_column_is_not_perfect_demographic_parity
        test_a_text_typed_column_recovers_a_fraction_not_only_a_flag
        test_the_snapshot_now_alerts_on_the_frame_it_published_as_clean
        test_control_a_breach_is_still_counted_per_window_and_the_period_is_silent
      The last of those is a control that RIDES on the text frame, so it is red for
      the right reason. No control over an integer, boolean, score or unreadable
      column went red under this sabotage, which is the other half of the reading.
  S2. an early ``return np.nan`` inserted above the ``privileged == 0`` warning, so
      the refusal survives and only the disclosure is gone, exactly as before.
      1 failed, 132 passed
        test_a_ratio_with_no_denominator_says_so_out_loud
  S3. ``if len(rates) >= 2:`` -> ``if len(rates) >= 0:`` in ``_warn_too_few_rates``,
      which makes it always answer False and never warn.
      2 failed, 131 passed
        test_one_group_is_not_a_comparison_and_the_refusal_is_audible
        test_a_missing_column_is_named_rather_than_answered_with_a_bare_nan
  S4. ``never_computed = []`` appended after the comprehension in
      ``get_alert_summary``.
      2 failed, 131 passed
        test_a_requested_metric_never_computed_in_any_window_is_disclosed
        test_a_metric_name_this_monitor_never_computes_is_disclosed
"""

import math
import warnings

import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
    _selected_share,
    _unscored_predictions,
    _unthresholded_scores,
)


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in rec]


def _monitor(**kwargs) -> FairnessMonitor:
    return FairnessMonitor(config=FairnessMonitorConfig(min_samples=30, **kwargs))


ALL_FOUR = [
    "disparate_impact",
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
]


# ===========================================================================
# The text-typed decision column, which is all four metrics at once
# ===========================================================================


def _text_predictions_total_denial() -> pd.DataFrame:
    """Group A selects EVERY row, group B selects NONE, predictions are strings.

    80 rows per group, 40 positive-label and 40 negative-label each, so every arm
    holds 40 rows and no floor is anywhere near firing. The truth on this frame:

        A: selected share 1.0, TPR 1.0, FPR 1.0
        B: selected share 0.0, TPR 0.0, FPR 0.0

        disparate_impact   0.0 / 1.0 = 0.0
        demographic_parity 1.0 - 0.0 = 1.0
        equal_opportunity  1.0 - 0.0 = 1.0
        equalized_odds     max(1.0, 1.0) = 1.0
    """
    rows = []
    for label in (1, 0):
        for _ in range(40):
            rows.append({"group_gender": "A", "label": label, "prediction": "1"})
        for _ in range(40):
            rows.append({"group_gender": "B", "label": label, "prediction": "0"})
    return pd.DataFrame(rows)


def _text_predictions_partial() -> pd.DataFrame:
    """The same text typing with a FRACTION on each side, not a flag.

    A: 40 rows, 30 selected -> 0.75.   B: 40 rows, 10 selected -> 0.25.
    demographic_parity 0.50, disparate_impact 0.25 / 0.75 = 1/3.
    """
    rows = []
    for i in range(40):
        rows.append({"group_gender": "A", "label": i % 2, "prediction": "1" if i < 30 else "0"})
    for i in range(40):
        rows.append({"group_gender": "B", "label": i % 2, "prediction": "1" if i < 10 else "0"})
    return pd.DataFrame(rows)


def test_the_two_predicates_and_the_rate_now_read_the_same_column_the_same_way():
    """The divergence itself, at the helper. Both predicates accepted the column
    as a clean 0/1 decision column (0 unscored, 0 unthresholded) and the share
    then came out 0.0 for a column in which every row is selected.
    """
    text = _text_predictions_total_denial()
    col_a = text[text["group_gender"] == "A"]["prediction"]

    assert _unscored_predictions(col_a) == 0, "the predicate says every row IS scored"
    assert _unthresholded_scores(col_a) == 0, "the predicate says every row is a 0/1 decision"
    assert _selected_share(col_a) == pytest.approx(1.0), "and the rate must agree with them"

    col_b = text[text["group_gender"] == "B"]["prediction"]
    assert _selected_share(col_b) == pytest.approx(0.0)


def test_a_text_typed_decision_column_is_not_perfect_equal_opportunity():
    """Measured before: 0.0, warnings [], against a real TPR gap of 1.0."""
    value, warned = _caught(
        _monitor().compute_equal_opportunity, _text_predictions_total_denial(), "group_gender"
    )

    assert value == pytest.approx(1.0), "a total denial of selection is not perfect parity"
    assert warned == [], "this is a MEASUREMENT recovered, not a refusal"


def test_a_text_typed_decision_column_is_not_perfect_equalized_odds():
    """Measured before: 0.0, warnings [], on both arms at once."""
    value, warned = _caught(
        _monitor().compute_equalized_odds, _text_predictions_total_denial(), "group_gender"
    )

    assert value == pytest.approx(1.0)
    assert warned == []


def test_a_text_typed_decision_column_is_not_a_perfect_four_fifths_ratio():
    """Measured before: nan, and the reason given was wrong. 0.0 is the ratio, and
    0.0 is far below the 0.8 four-fifths floor.
    """
    value, warned = _caught(
        _monitor().compute_disparate_impact, _text_predictions_total_denial(), "group_gender"
    )

    assert value == pytest.approx(0.0)
    assert warned == []


def test_a_text_typed_decision_column_is_not_perfect_demographic_parity():
    """Measured before: 0.0, warnings []."""
    value, warned = _caught(
        _monitor().compute_demographic_parity, _text_predictions_total_denial(), "group_gender"
    )

    assert value == pytest.approx(1.0)
    assert warned == []


def test_a_text_typed_column_recovers_a_fraction_not_only_a_flag():
    """The fix must read the column, not merely answer 1.0 where it once said 0.0.

    A 0.75 against a 0.25 is a 0.50 gap and a ratio of exactly one third, and
    neither number is reachable by any all-or-nothing reading.
    """
    frame = _text_predictions_partial()
    monitor = _monitor()

    rates = monitor._group_positive_rates(frame, "group_gender", "prediction")
    assert rates == {"A": pytest.approx(0.75), "B": pytest.approx(0.25)}

    parity, w1 = _caught(monitor.compute_demographic_parity, frame, "group_gender")
    ratio, w2 = _caught(monitor.compute_disparate_impact, frame, "group_gender")

    assert parity == pytest.approx(0.5)
    assert ratio == pytest.approx(1.0 / 3.0)
    assert w1 == [] and w2 == []


def test_the_snapshot_now_alerts_on_the_frame_it_published_as_clean():
    """THE CONSUMER. update_and_check published demographic_parity 0.0 with
    alerts {...: False} and any_alert False on this frame: a measured clean bill
    for a model that selects one group entirely and the other never.
    """
    monitor = _monitor(metrics_to_track=ALL_FOUR)
    snap, warned = _caught(monitor.update_and_check, _text_predictions_total_denial())

    assert snap.metrics["disparate_impact_group_gender"] == pytest.approx(0.0)
    assert snap.metrics["demographic_parity_group_gender"] == pytest.approx(1.0)
    assert snap.metrics["equal_opportunity_group_gender"] == pytest.approx(1.0)
    assert snap.metrics["equalized_odds_group_gender"] == pytest.approx(1.0)
    assert snap.alerts == {
        "disparate_impact_group_gender": True,
        "demographic_parity_group_gender": True,
        "equalized_odds_group_gender": True,
        "equal_opportunity_group_gender": True,
    }
    assert snap.any_alert is True
    assert warned == []


# --- over-correction controls for the coercion -----------------------------


def _int_predictions_real_gap() -> pd.DataFrame:
    """Integer predictions, 40 rows per group, TPR gap exactly 0.6.

    The frame ``test_bgl3_operations_2`` uses as its own over-correction control:
    M predicted positive for 32 of 40 rows and F for 8 of 40, labels alternating,
    so each arm holds 20 rows, M's TPR is 1.0 and F's is 0.4.
    """
    rows = [
        {"prediction": int(i < 32), "label": int(i % 2 == 0), "group_gender": "M"}
        for i in range(40)
    ]
    rows += [
        {"prediction": int(i < 8), "label": int(i % 2 == 0), "group_gender": "F"} for i in range(40)
    ]
    return pd.DataFrame(rows)


def test_control_an_integer_column_still_measures_its_exact_numbers():
    """OVER-CORRECTION CONTROL. Exact values, and silence, on every built-in.

    A coercion that changed any rate that WAS a measurement fails here: the gap is
    0.6, the parity spread 0.8 - 0.2 = 0.6, the ratio 0.2 / 0.8 = 0.25.
    """
    frame = _int_predictions_real_gap()
    monitor = _monitor()

    eo, w1 = _caught(monitor.compute_equal_opportunity, frame, "group_gender")
    odds, w2 = _caught(monitor.compute_equalized_odds, frame, "group_gender")
    parity, w3 = _caught(monitor.compute_demographic_parity, frame, "group_gender")
    ratio, w4 = _caught(monitor.compute_disparate_impact, frame, "group_gender")

    assert eo == pytest.approx(0.6, abs=1e-9)
    assert odds == pytest.approx(0.6, abs=1e-9)
    assert parity == pytest.approx(0.6, abs=1e-9)
    assert ratio == pytest.approx(0.25, abs=1e-9)
    assert w1 == [] and w2 == [] and w3 == [] and w4 == []


def test_control_a_boolean_column_is_still_a_decision_column():
    """OVER-CORRECTION CONTROL. The library's own ``y_pred`` arrays carry booleans
    and ``_unscored_predictions`` parts from ``is_measured`` deliberately to keep
    them. 30 of 40 True against 10 of 40 True is a 0.5 spread and a 0.25/0.75
    ratio, exactly as the integer frame gives.
    """
    rows = [{"group_gender": "A", "prediction": i < 30, "label": i % 2} for i in range(40)]
    rows += [{"group_gender": "B", "prediction": i < 10, "label": i % 2} for i in range(40)]
    frame = pd.DataFrame(rows)
    assert frame["prediction"].dtype == bool

    monitor = _monitor()
    parity, w1 = _caught(monitor.compute_demographic_parity, frame, "group_gender")
    ratio, w2 = _caught(monitor.compute_disparate_impact, frame, "group_gender")

    assert parity == pytest.approx(0.5)
    assert ratio == pytest.approx(1.0 / 3.0)
    assert w1 == [] and w2 == []


def test_control_the_earlier_refusals_still_refuse_a_score_column():
    """OVER-CORRECTION CONTROL for the BGL7 door beside this one. A genuine SCORE
    column must still be refused: reading it numerically is exactly what the
    coercion now does, and it must not turn a score into a decision.
    """
    rows = [
        {"group_gender": "A", "prediction": 1.0 if i < 25 else 0.0, "label": i % 2}
        for i in range(50)
    ]
    rows += [{"group_gender": "B", "prediction": 0.5, "label": i % 2} for i in range(50)]
    frame = pd.DataFrame(rows)

    value, warned = _caught(_monitor().compute_disparate_impact, frame, "group_gender")

    assert math.isnan(value)
    assert any("SCORE rather than a" in w for w in warned), warned


def test_control_an_unreadable_text_column_is_still_refused():
    """OVER-CORRECTION CONTROL the other way. ``errors="coerce"`` must not be able
    to mint a 0 out of a value nothing can read: the unscored predicate refuses
    "yes"/"no" above the coercion, so the rate is nan, not 0.0.
    """
    rows = [{"group_gender": "A", "prediction": "yes", "label": i % 2} for i in range(40)]
    rows += [{"group_gender": "B", "prediction": "no", "label": i % 2} for i in range(40)]
    frame = pd.DataFrame(rows)

    value, warned = _caught(_monitor().compute_demographic_parity, frame, "group_gender")

    assert math.isnan(value)
    assert any("carry no prediction" in w for w in warned), warned


# ===========================================================================
# 3. compute_disparate_impact: the two SILENT refusals
# ===========================================================================


def _nobody_is_selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_gender": ["A"] * 50 + ["B"] * 50,
            "prediction": [0] * 100,
            "label": [1, 0] * 50,
        }
    )


def _only_one_group() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_gender": ["A"] * 50,
            "prediction": [1, 0] * 25,
            "label": [1, 0] * 25,
        }
    )


def test_a_ratio_with_no_denominator_says_so_out_loud():
    """Measured before: nan with ZERO warnings, which the audit named as the
    silent refusal reaching get_alert_summary. The value was never the problem.
    """
    value, warned = _caught(
        _monitor().compute_disparate_impact, _nobody_is_selected(), "group_gender"
    )

    assert math.isnan(value)
    assert len(warned) == 1, warned
    assert "is 0" in warned[0]
    assert "NOT compared" in warned[0], "the warning must say the threshold was not applied"


def test_one_group_is_not_a_comparison_and_the_refusal_is_audible():
    """Measured before: nan with ZERO warnings in BOTH rate metrics, while the
    sibling compute_equal_opportunity has warned on the same shape since BGL3.
    """
    frame = _only_one_group()
    monitor = _monitor()

    ratio, w1 = _caught(monitor.compute_disparate_impact, frame, "group_gender")
    parity, w2 = _caught(monitor.compute_demographic_parity, frame, "group_gender")

    assert math.isnan(ratio) and math.isnan(parity)
    assert len(w1) == 1 and len(w2) == 1, (w1, w2)
    assert "only 1 group(s)" in w1[0] and "only 1 group(s)" in w2[0]
    assert "'A'" in w1[0], "the group that was found is named"
    assert "compute_disparate_impact" in w1[0]
    assert "compute_demographic_parity" in w2[0]


def test_a_missing_column_is_named_rather_than_answered_with_a_bare_nan():
    """The other half of the same silent door: ``_group_positive_rates`` returns an
    empty dict for a column that is not in the frame, and the refusal said nothing.
    """
    frame = _nobody_is_selected().drop(columns=["group_gender"])

    value, warned = _caught(_monitor().compute_disparate_impact, frame, "group_gender")

    assert math.isnan(value)
    assert len(warned) == 1, warned
    assert "group_gender" in warned[0]
    assert "NOT among" in warned[0]


def test_control_a_measurable_pair_is_still_measured_and_silent():
    """OVER-CORRECTION CONTROL. A guard that refuses whenever a rate is low, or
    that warns on every window, fails here: 0.2 / 0.8 is 0.25 exactly.
    """
    frame = _int_predictions_real_gap()
    ratio, warned = _caught(_monitor().compute_disparate_impact, frame, "group_gender")

    assert ratio == pytest.approx(0.25, abs=1e-9)
    assert warned == []


def test_control_a_zero_rate_beside_a_non_zero_one_is_still_a_measured_zero():
    """OVER-CORRECTION CONTROL. ``privileged == 0`` is about the DENOMINATOR only.
    A group never selected beside a group half selected is a real ratio of 0.0 and
    the strongest reading a window can carry: it must not become a refusal.
    """
    rows = [{"group_gender": "A", "prediction": int(i < 25), "label": i % 2} for i in range(50)]
    rows += [{"group_gender": "B", "prediction": 0, "label": i % 2} for i in range(50)]
    frame = pd.DataFrame(rows)

    ratio, w1 = _caught(_monitor().compute_disparate_impact, frame, "group_gender")
    parity, w2 = _caught(_monitor().compute_demographic_parity, frame, "group_gender")

    assert ratio == pytest.approx(0.0)
    assert parity == pytest.approx(0.5)
    assert w1 == [] and w2 == []


# ===========================================================================
# 5. get_alert_summary: coverage over the REQUESTED set
# ===========================================================================


def _batch(with_label: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "group_gender": ["A"] * 50 + ["B"] * 50,
            "prediction": ([1] * 25 + [0] * 25) * 2,
        }
    )
    if with_label:
        frame["label"] = ([1, 0] * 25) * 2
    return frame


def _period(monitor: FairnessMonitor, frame: pd.DataFrame, rounds: int = 3) -> FairnessMonitor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(rounds):
            monitor.update_and_check(frame.copy())
    return monitor


def test_a_requested_metric_never_computed_in_any_window_is_disclosed():
    """Measured before: ``{}`` with ZERO warnings over three windows in which the
    two label-dependent guardrails were never applied once, byte-identical to a
    fully monitored clean period. Doors two to five all count or key on what the
    windows PRODUCED, so a metric that produced nothing was invisible to them.
    """
    monitor = _period(_monitor(metrics_to_track=ALL_FOUR), _batch(with_label=False))

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert len(warned) == 1, warned
    assert "NEVER computed" in warned[0]
    assert "equalized_odds" in warned[0] and "equal_opportunity" in warned[0]
    assert "3 window(s)" in warned[0]


def test_a_metric_name_this_monitor_never_computes_is_disclosed():
    """The sharpest form, and the one with no disclosure at ANY surface before: a
    one-letter typo in a tracked metric name produced ZERO warnings at ingest and
    ZERO here, over three windows, and equal opportunity was never monitored.
    """
    monitor = _period(
        _monitor(metrics_to_track=["disparate_impact", "equal_oportunity"]),
        _batch(with_label=True),
    )

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert len(warned) == 1, warned
    assert "equal_oportunity" in warned[0], "the unrecognised name must be quoted back"
    assert "NEVER computed" in warned[0]


def test_control_a_fully_requested_fully_compared_clean_period_stays_silent():
    """OVER-CORRECTION CONTROL, and the one that makes the pins above mean
    something: an empty summary from a period in which every requested metric was
    computed and compared in every window is a real finding, and it must be
    silent. A guard that warns whenever the summary is empty fails here.
    """
    monitor = _period(_monitor(metrics_to_track=ALL_FOUR), _batch(with_label=True))

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {}
    assert warned == [], "compared and nothing breached is a finding, not a caveat"


def test_control_a_breach_is_still_counted_per_window_and_the_period_is_silent():
    """OVER-CORRECTION CONTROL. The counting half of the method still works: three
    windows of a maximally disparate frame with every requested metric computed and
    compared give a count of 3 for each of the four keys and no coverage caveat.
    """
    monitor = _period(_monitor(metrics_to_track=ALL_FOUR), _text_predictions_total_denial())

    summary, warned = _caught(monitor.get_alert_summary)

    assert summary == {
        "disparate_impact_group_gender": 3,
        "demographic_parity_group_gender": 3,
        "equalized_odds_group_gender": 3,
        "equal_opportunity_group_gender": 3,
    }
    assert warned == []
