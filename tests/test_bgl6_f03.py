"""BGL6 audit, batch F03: overturns of the A-operations / A-agents_multi fixes.

Every test in this file was RUN before it was written and its measured output is
quoted in its docstring. Each one attacks a fix that was recorded as PROVEN with a
named pin and a reproduced sabotage. The sabotages all reproduce; what does not
hold is the SCOPE of five of the fixes and the reach of one disclosure.

Written 2026-09-28. Python is .venv/bin/python from the vfairness directory.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.gate import (
    FairnessReportCard,
    GateConfig,
    ModelFairnessGate,
)
from vfairness.operations.monitoring.tracker import (
    FairnessMonitor,
    FairnessMonitorConfig,
)

DP = "demographic_parity_difference"


#: HOW MANY CLAIMS THE SECOND-ROUND AUDIT RECORDED IN THIS FILE, on 2026-09-28.
#:
#: A HISTORICAL FACT, and it must not move. The register used to take this count by
#: counting the test functions in the file, which was right on the day the audit
#: landed and wrong from the first fix onwards: inverting a witness into a pin
#: renames it, and a fix arrives with its own over-correction control, so closing
#: records made the audit look BIGGER. It had grown from 59 claims to 69 by the time
#: anybody added them up, on a page whose whole subject is not misstating what was
#: measured.
#:
#: Recovered from the audit's own baseline commit e6a5780, which is where every one
#: of these numbers comes from.
RECORDED_CLAIMS_AT_AUDIT = 8

#: THE TESTS IN THIS FILE THAT STILL RECORD AN OPEN DEFECT.
#:
#: A name is removed from this list in the SAME commit that fixes its defect and
#: inverts the test into a pin, so the two cannot drift. It is declared here rather
#: than inferred from pass/fail because a test that RECORDS a defect passes while the
#: defect is live, which is indistinguishable by execution from a pin that passes
#: because the defect is gone.
#:
#: Read by scripts/bgl6_register.py, published as counts in
#: docs/bgl6-audit-register.json and in QUALITY_AND_HARDENING.md, and checked by
#: tests/test_bgl6_register_is_honest.py, which refuses a name that is not a test
#: function in this module.
RECORDED_DEFECTS_STILL_OPEN: list[str] = []


def _capture(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in caught]


# 1 and 2. The BASELINE operand was guarded. The MARGIN operand was not.


def test_overturn_a_nan_improvement_margin_switches_off_the_requirement():
    """``improvement < margin`` has TWO operands and only one was guarded.

    The fix graded PROVEN for ModelFairnessGate.evaluate and
    .evaluate_from_metrics states its own root cause as "``improvement < margin``
    and ``degradation > margin`` were both False for NaN and one unmeasurable
    baseline switched off both configured requirements". The guard added covers
    the BASELINE. The MARGIN is the other operand of the same two comparisons and
    has no guard anywhere, so the identical sentence is still true of it.

    Measured 2026-09-28 on HEAD, threshold 0.95 so only the improvement
    requirement can fail, baseline 0.01 against a value of 0.9 (a degradation of
    0.89 in the metric's own declared direction), require_improvement=True,
    improvement_margin=float('nan'):

        approved=True  status=approved  'APPROVED - All fairness requirements met'
        warnings=[]  blocking=[]  improvement=-0.89  github conclusion 'success'

    The same call with improvement_margin=0.01 blocks with 'demographic_parity_
    difference did not improve by required margin', so the margin IS the operand
    that decides it. ``check_threshold`` in this same library refuses an
    unmeasurable THRESHOLD for exactly this reason ("An unmeasurable value, an
    unmeasurable threshold, an unknown direction or a DEGENERATE bound all yield
    COULD_NOT_CHECK"); the gate's two bounds never reach that rule.
    """
    gate = ModelFairnessGate(
        config=GateConfig(
            metrics=[DP],
            thresholds={DP: 0.95},
            require_improvement=True,
            improvement_margin=float("nan"),
        )
    )
    decision = gate.evaluate_from_metrics({DP: 0.9}, baseline_metrics={DP: 0.01})
    assert not decision.approved or decision.warnings, (
        f"the gate reported {decision.summary!r} with warnings={decision.warnings} "
        f"and github conclusion "
        f"{gate.create_github_check(decision)['conclusion']!r} for a required "
        f"improvement whose margin is NaN, so `improvement < margin` was False and "
        f"the configured requirement was never checked. improvement published as "
        f"{decision.metric_evaluations[0].improvement!r}."
    )


def test_overturn_a_nan_degradation_margin_switches_off_the_bound():
    """The degradation half, and here the published ``improvement`` collapses too.

    Measured 2026-09-28 on HEAD, allow_degradation_margin=float('nan'), baseline
    0.01 against 0.9:

        approved=True  status=approved  'APPROVED - All fairness requirements met'
        warnings=[]  improvement=None  github conclusion 'success'

    With allow_degradation_margin=0.0 the same call blocks with 'degraded beyond
    allowed margin'. ``improvement`` is None, which this module's own comment
    reserves for "no degradation margin set": the comment above that very block
    says "without this the row published None both for 'no degradation margin
    set' and for 'the margin could not be checked'", which is what it does when
    the margin is the unmeasurable operand.
    """
    gate = ModelFairnessGate(
        config=GateConfig(
            metrics=[DP],
            thresholds={DP: 0.95},
            require_improvement=False,
            allow_degradation_margin=float("nan"),
        )
    )
    decision = gate.evaluate_from_metrics({DP: 0.9}, baseline_metrics={DP: 0.01})
    assert not decision.approved or decision.warnings, (
        f"{decision.summary!r} with warnings={decision.warnings} for a degradation "
        f"bound of NaN; improvement published as "
        f"{decision.metric_evaluations[0].improvement!r}, which is this dataclass's "
        f"'no comparison was required'."
    )


def test_overturn_the_raw_data_entry_point_has_the_same_margin_hole():
    """The same hole through evaluate(), the entry point a CI job calls with data.

    Measured 2026-09-28: 100 + 100 genuinely fair rows, a measured baseline of
    0.01, improvement_margin=float('nan'), require_improvement=True ->
    approved=True, status=approved, 'APPROVED - All fairness requirements met',
    warnings=[], github conclusion 'success'.
    """
    y = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)
    protected = np.array(["a"] * 100 + ["b"] * 100)
    gate = ModelFairnessGate(
        config=GateConfig(
            metrics=[DP],
            thresholds={DP: 0.95},
            require_improvement=True,
            improvement_margin=float("nan"),
        )
    )
    decision = gate.evaluate(y, y, protected, baseline_metrics={DP: 0.01})
    assert not decision.approved or decision.warnings, (
        f"{decision.summary!r} with warnings={decision.warnings}; the requirement "
        f"was configured and `improvement < nan` is False, so it was never checked."
    )


# 5. small_sample_check_ran exists, and the two surfaces a reviewer reads drop it.


def test_overturn_the_report_card_cannot_tell_never_checked_from_checked_clean():
    """The three-state field never reaches the PR comment.

    GateDecision.to_dict emits small_sample_check_ran and to_markdown_report
    renders a "Small-Sample Check: NOT RUN" section, both verified. The
    FairnessReportCard, which gate.py's own comment calls the surface "where a
    reviewer decides to merge", prints neither, so the card for an
    evaluate_from_metrics decision (check never ran) is byte-identical to the
    card for an evaluate() decision on data whose groups were all large enough.

    Measured 2026-09-28 on the same number 0.0 through both entry points:

        FairnessReportCard(...).to_markdown() identical: True
        'NOT RUN' / 'Small-Sample Check' / 'did not happen' / 'not checked'
            in the never-checked card: False, False, False, False
        rendering.adapters_workflow.report_card_to_svg identical
            (timestamp normalised): True

    That is the same measurement the field's own docstring records as the defect
    ("identical: True"), still true of the two surfaces a person looks at, and it
    is the silence the sibling row in this batch grades as a defect for the
    hierarchical half of the card.
    """
    y = np.array([1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)
    protected = np.array(["a"] * 100 + ["b"] * 100)
    gate = ModelFairnessGate(config=GateConfig(metrics=[DP], thresholds={DP: 0.5}))
    checked = gate.evaluate(y, y, protected)
    never_checked = gate.evaluate_from_metrics({DP: 0.0})
    assert checked.small_sample_check_ran is True
    assert never_checked.small_sample_check_ran is False
    card_checked = FairnessReportCard(checked, model_name="m").to_markdown()
    card_never = FairnessReportCard(never_checked, model_name="m").to_markdown()
    assert card_checked != card_never, (
        "the report card is byte-identical for a decision whose small-sample check "
        "RAN and found nothing and one where it never ran. The card is what the "
        "pull request shows.\n" + card_never
    )


# 7. The pin named as evidence for evaluate() never calls evaluate().


def test_the_unknown_direction_guard_in_evaluate_was_never_pinned():
    """CLOSED 2026-09-29. The named evidence now reaches evaluate() too.

    THE FIX was in the evidence, not in the gate: the guard was already correct in
    evaluate(), and nothing exercised it. tests/test_bgl5_gate_improvement_direction.py
    now parametrises every one of its six tests over ENTRIES = both doors, and its
    helper hands the evaluate() branch a compute_metrics_fn, without which the row
    under test does not exist and the helper raises StopIteration. That absence is
    why the parametrisation was a one-element list while the docstring claimed both.

    Re-measured 2026-09-29: reinstating abs(baseline_value) - abs(value) at BOTH
    evaluate()-side sites now prints '7 failed, 15 passed', every failure an
    [evaluate-...] case. The same sabotage printed '11 passed' before this change.
    The file went from 11 tests to 22.

    THE ORIGINAL RECORD follows.

    tests/test_bgl5_gate_improvement_direction.py is the named evidence for
    ModelFairnessGate.evaluate and its sabotage record says "Four, all red: the
    original substitution in each half". Every ``_decide`` call in that file
    passes entry="from_metrics", and its only parametrisation of that argument is
    the one-element list ``["from_metrics"]``, so it never calls evaluate().

    Measured 2026-09-28: reinstating ``if improvement is None: improvement =
    abs(baseline_value) - abs(value)`` at BOTH sites inside evaluate() and running
    that file printed '11 passed'. The same reinstatement inside
    evaluate_from_metrics printed '7 failed, 4 passed'. Under the evaluate-side
    sabotage the method published improvement=0.09999999999999998 and
    -0.09999999999999998, the exact numbers the row quotes as the BEFORE, with
    'COULD NOT BE EVALUATED' absent from the message and the warnings.
    """
    undeclared = "a_metric_nobody_declared"
    rng = np.random.default_rng(0)
    n = 120
    for value, baseline in ((0.2, 0.3), (0.3, 0.2)):
        gate = ModelFairnessGate(
            config=GateConfig(
                metrics=[undeclared],
                thresholds={undeclared: 0.9},
                require_improvement=True,
                improvement_margin=0.01,
            ),
            compute_metrics_fn=lambda a, b, c, _v=value: {undeclared: _v},
        )
        decision = gate.evaluate(
            rng.integers(0, 2, n),
            rng.integers(0, 2, n),
            np.array(["a"] * 60 + ["b"] * 60),
            baseline_metrics={undeclared: baseline},
        )
        row = decision.metric_evaluations[0]
        assert row.improvement != row.improvement, (
            f"evaluate() published improvement={row.improvement!r} for a metric whose "
            f"better direction is not declared"
        )
        text = f"{row.message} {' '.join(decision.warnings)}"
        assert "COULD NOT BE EVALUATED" in text, text


# 8 and 9. The PREDICTION operand was guarded. Group MEMBERSHIP was not.


def test_overturn_rows_with_no_protected_attribute_are_dropped_in_silence():
    """The fabricated pair 1.0 / 0.0 is still reachable, with zero warnings.

    compute_disparate_impact and compute_demographic_parity now refuse a group
    whose positive-prediction rate could not be measured, verified. Neither
    method applies any predicate to the GROUP column: ``df.groupby(group_col)``
    drops rows whose protected attribute is missing, and ``_undersampled_groups``
    groups the same way, so those rows are invisible to the min_samples
    disclosure too.

    Measured 2026-09-28 on 150 rows, 50 group A (all selected), 50 group B (all
    selected) and 50 rows carrying NO group value (none selected):

        compute_disparate_impact   -> 1.0  warnings 0
        compute_demographic_parity -> 0.0  warnings 0

    That is exactly the pair the fix's own BEFORE section quotes ("returned 1.0
    with warnings []. A PERFECT four-fifths ratio"), over a population nobody
    chose: a third of the window. If those 50 rows were a third group the ratio
    would be 0.0. ``_warn_undersampled``'s own wording applies verbatim: "an
    aggregate over the surviving groups only is a LOWER BOUND on the disparity".
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig(min_samples=30))
    frame = pd.DataFrame(
        {
            "group_gender": ["A"] * 50 + ["B"] * 50 + [np.nan] * 50,
            "prediction": [1] * 50 + [1] * 50 + [0] * 50,
            "label": [1] * 150,
        }
    )
    ratio, ratio_warnings = _capture(monitor.compute_disparate_impact, frame, "group_gender")
    gap, gap_warnings = _capture(monitor.compute_demographic_parity, frame, "group_gender")
    assert (ratio != ratio) or ratio_warnings, (
        f"compute_disparate_impact returned {ratio!r} with warnings={ratio_warnings} "
        f"on a window where 50 of 150 rows carry no protected attribute at all"
    )
    assert (gap != gap) or gap_warnings, (
        f"compute_demographic_parity returned {gap!r} with warnings={gap_warnings} "
        f"on the same window"
    )


# 10 and 11. "never 0.0, which reads as perfect equal opportunity" is not held
# for a score column, which the config documents as supported input.


def test_overturn_equal_opportunity_reports_perfect_parity_for_a_score_column():
    """Exactly 0.0 with zero warnings for a maximally discriminating model.

    FairnessMonitorConfig documents prediction_col as "binary 0/1 or scores", and
    compute_disparate_impact reads a score column as a rate (mean of the column).
    ``_selected_share``, which the two label-dependent metrics use, is
    ``(values == 1).mean()``, so NO row of a score column is ever counted as
    selected, both true-positive rates are 0.0 and the spread is 0.0.

    Measured 2026-09-28 on 80 rows per group, every row scored, group A's
    positive-label rows all scoring 0.9 and group B's all scoring 0.1 (a maximal
    TPR gap at any threshold in (0.1, 0.9]):

        compute_equal_opportunity -> 0.0   warnings []
        compute_equalized_odds    -> 0.0   warnings []
        compute_disparate_impact  -> 0.111...   (the same frame, no warnings)
        compute_demographic_parity -> 0.800...  (the same frame, no warnings)

    So on one frame the four built-ins disagree about whether the model
    discriminates, and the two this fix covers return the textbook perfect value.
    The fix's own sentence, "Returning nan (could not measure), never 0.0, which
    reads as perfect equal opportunity", does not hold here: ``_selected_share``
    refuses only a non-finite prediction, and 0.1 is finite.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig(min_samples=30))
    frame = pd.DataFrame(
        {
            "group_gender": ["A"] * 80 + ["B"] * 80,
            "label": ([1] * 40 + [0] * 40) * 2,
            "prediction": ([0.9] * 40 + [0.0] * 40) + ([0.1] * 40 + [0.0] * 40),
        }
    )
    eo, eo_warnings = _capture(monitor.compute_equal_opportunity, frame, "group_gender")
    eodds, eodds_warnings = _capture(monitor.compute_equalized_odds, frame, "group_gender")
    assert (eo != eo) or eo_warnings or eo == pytest.approx(1.0), (
        f"compute_equal_opportunity returned {eo!r} with warnings={eo_warnings} on a "
        f"frame where group A's positive-label rows all score 0.9 and group B's all "
        f"score 0.1"
    )
    assert (eodds != eodds) or eodds_warnings or eodds == pytest.approx(1.0), (
        f"compute_equalized_odds returned {eodds!r} with warnings={eodds_warnings} on "
        f"the same frame"
    )


# 12. The empty-alert-summary guard is keyed on TOTAL absence of comparison.


def test_overturn_a_mostly_unmonitored_history_still_returns_a_silent_empty():
    """Nine of ten windows compared nothing and the empty result stays silent.

    The guard is ``if not any(snap.alerts for snap in self._history)``, so it
    fires only when NO window in the history carries a threshold comparison. A
    history in which one window compared and nine could not is the partial case,
    and the fix's own control (test_control_one_compared_window_beside_a_refused_
    one_stays_silent) pins it as silent.

    Measured 2026-09-28: one window with both groups selected at exactly 0.5
    (disparate impact 1.0, both alerts False) followed by nine windows whose
    prediction column is all NaN (metrics nan, alerts {}, any_alert None):

        windows in history: 10
        windows carrying ANY threshold comparison: 1
        get_alert_summary() -> {}   warnings []

    A ten-window fully monitored clean history returns the same {} with the same
    silence, so ``if not monitor.get_alert_summary()`` reads a period in which
    90 per cent of windows were never compared as a clean bill. That is the
    reading the method's own docstring forbids.
    """
    monitor = FairnessMonitor(config=FairnessMonitorConfig(min_samples=30))
    clean = pd.DataFrame(
        {
            "group_gender": ["A"] * 60 + ["B"] * 60,
            "prediction": ([1] * 30 + [0] * 30) * 2,
            "label": [1] * 120,
        }
    )
    blind = pd.DataFrame(
        {
            "group_gender": ["A"] * 60 + ["B"] * 60,
            "prediction": [np.nan] * 120,
            "label": [1] * 120,
        }
    )
    _capture(monitor.update_and_check, clean)
    for _ in range(9):
        _capture(monitor.update_and_check, blind)
    compared = sum(1 for snap in monitor._history if snap.alerts)
    assert compared == 1
    summary, caught = _capture(monitor.get_alert_summary)
    assert summary or caught, (
        f"get_alert_summary() returned {summary!r} with warnings={caught} for a "
        f"history of {len(monitor._history)} window(s) of which only {compared} "
        f"carries any threshold comparison at all"
    )
