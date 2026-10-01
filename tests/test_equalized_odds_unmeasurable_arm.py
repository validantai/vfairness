"""EqualizedOdds must not certify compliance from a half-measured metric.

Finding (xai/in-proc audit, 2026-08-27, HIGH). Equalized odds is the max of the
TPR disparity and the FPR disparity (Hardt, Price & Srebro 2016); both arms are
load-bearing. The per-arm code was already honest: ``_defined_spread`` returns
NaN when fewer than two groups have a defined rate. One line later the arms were
combined with ``_defined([tpr_violation, fpr_violation])``, which DROPPED the
NaN arm and took the max over the survivor, so with one arm unmeasurable the
reported violation was a LOWER BOUND. On 8 rows where one group has no positive
labels that produced ``overall_violation=0.0``, ``is_satisfied=True`` and
``insufficient_data=False`` while ``details['tpr_violation']`` was ``nan`` in
the same dict.

Two things are pinned here, and the second matters as much as the first.

1. An unmeasurable arm makes the verdict could-not-check.
2. **The siblings agree.** ``EqualOpportunityConstraint`` measures the very same
   TPR quantity and always got this right; the bug was that the two disagreed,
   and the permissive one was wrong. Pinning only EqualizedOdds' new value would
   let them drift apart again, so the agreement itself is asserted.

The both-arms-defined cases are the over-correction control: they prove the fix
propagates NaN only where an arm is genuinely unmeasurable, rather than
blanket-NaNing the constraint into uselessness.

SECOND FINDING (publish-readiness audit, B4, 2026-08-28). Everything described
above is real, but the fixture it runs on could not reach the case that
mattered, and two further defects lived behind it.

``A`` below has exactly TWO groups. Make one group's rate undefined and only ONE
group is left, so ``_defined_spread`` refused for the trivial reason that no
pair survived. The interesting case is THREE groups with one unmeasurable and
TWO survivors that still form a pair: there the survivors' spread WAS computed
and reported as the violation, a LOWER BOUND presented as a measurement. So this
file called itself HIGH and named the invariant correctly while being unable to
fire on it.

1. ``_defined_spread`` dropped the unmeasurable group. Measured on
   A (TPR 0.90) | B (TPR 0.88) | C (no positive labels), ``EqualizedOddsConstraint``
   (tolerance 0.05) reported ``overall_violation=0.02``, ``is_satisfied=True``
   and ``insufficient_data=False`` while ``group_tprs['C']`` was NaN in that
   same result. The drop is where the lower bound is created; the arm
   combination fixed earlier was only the last step.
2. ``evaluation...classification.equalized_odds_difference`` had the same drop,
   so it returned 0.0200 on that data while ``equal_opportunity_difference``,
   its OWN TPR leg over the SAME rates, correctly returned NaN. One report
   filed equalized odds under passed_metrics and equal opportunity under
   not_assessable simultaneously.

AND NOTE WHAT THIS MEANS FOR THE INVARIANT ABOVE. At three groups the two
constraints AGREED: both dropped C and both reported 0.02 with
``insufficient_data=False``. Sibling agreement is therefore NECESSARY BUT NOT
SUFFICIENT, and asserting it alone would not have caught B4. The tests below
pin the VALUE (could-not-check) as well as the agreement, at three groups, and
across both the constraint layer and the metric layer that must not diverge
from it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    equal_opportunity_difference,
    equalized_odds_difference,
)
from vfairness.in_processing.constraints.base import (
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
)

# 4 rows per group. Group "b" varies per scenario to make an arm unmeasurable.
A = np.array(["a"] * 4 + ["b"] * 4)


def _is_nan(x: float) -> bool:
    return x != x


# --- the defect: an unmeasurable arm must not be dropped ----------------------


def test_unmeasurable_tpr_arm_is_could_not_check_not_a_pass():
    """Group b has no positive labels, so its TPR is undefined."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # perfect; no real disparity
    cv = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)

    assert _is_nan(cv.details["tpr_violation"]), "precondition: the TPR arm is unmeasurable"
    assert cv.details["fpr_violation"] == 0.0, "precondition: the FPR arm IS measurable"
    assert _is_nan(cv.overall_violation), "the unmeasurable arm must propagate, not be dropped"
    assert cv.is_satisfied is False, "could-not-check must not read as satisfied"
    assert cv.details["insufficient_data"] is True


def test_unmeasurable_fpr_arm_is_could_not_check_not_a_pass():
    """Group b has no negative labels, so its FPR is undefined."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    cv = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)

    assert _is_nan(cv.details["fpr_violation"]), "precondition: the FPR arm is unmeasurable"
    assert cv.details["tpr_violation"] == 0.0, "precondition: the TPR arm IS measurable"
    assert _is_nan(cv.overall_violation)
    assert cv.is_satisfied is False
    assert cv.details["insufficient_data"] is True


# --- the sibling agreement, which is the actual invariant ---------------------


def test_siblings_agree_on_insufficient_data_when_the_tpr_arm_is_unmeasurable():
    """Two constraints measuring the same TPR must reach the same verdict.

    Divergence IS the bug. EqualOpportunity already reported this correctly
    while EqualizedOdds, which contains that same TPR, reported the data as
    sufficient. Asserting only EqualizedOdds' new value would let them drift
    apart again, so the agreement is pinned directly.
    """
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])

    eo = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)
    eop = EqualOpportunityConstraint().compute_violation(y_pred, y_true, A)

    # Precondition: both genuinely see the same undefined group TPR.
    assert _is_nan(eo.details["group_tprs"]["b"])
    assert _is_nan(eop.details["group_tprs"]["b"])

    assert eo.details["insufficient_data"] == eop.details["insufficient_data"] is True, (
        "EqualizedOdds and EqualOpportunity measure the same TPR and must agree on "
        f"whether it was measurable; got {eo.details['insufficient_data']} vs "
        f"{eop.details['insufficient_data']}"
    )
    assert eo.is_satisfied == eop.is_satisfied is False
    assert _is_nan(eo.overall_violation) and _is_nan(eop.overall_violation)


def test_siblings_agree_when_the_data_is_sufficient():
    """The agreement must not be an artifact of both always saying 'insufficient'."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # group b's TPR collapses to 0

    eo = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)
    eop = EqualOpportunityConstraint().compute_violation(y_pred, y_true, A)

    assert eo.details["insufficient_data"] == eop.details["insufficient_data"] is False
    assert not _is_nan(eo.overall_violation) and not _is_nan(eop.overall_violation)
    # A real TPR gap of 1.0 is present and both must see it as a violation.
    assert eo.overall_violation == pytest.approx(1.0)
    assert eo.is_satisfied is False


# --- over-correction control: measurable data must be untouched ---------------


def test_both_arms_defined_with_a_real_disparity_is_unchanged():
    """Control. The fix must not blanket-NaN a fully measurable constraint."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    cv = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)

    assert cv.details["tpr_violation"] == pytest.approx(1.0)
    assert cv.details["fpr_violation"] == pytest.approx(0.0)
    assert cv.overall_violation == pytest.approx(1.0), "max over BOTH arms"
    assert cv.is_satisfied is False
    assert cv.details["insufficient_data"] is False


def test_both_arms_defined_with_perfect_predictions_still_passes():
    """Control. A genuinely satisfied constraint must still report satisfied."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    cv = EqualizedOddsConstraint().compute_violation(y_pred, y_true, A)

    assert cv.details["tpr_violation"] == pytest.approx(0.0)
    assert cv.details["fpr_violation"] == pytest.approx(0.0)
    assert cv.overall_violation == pytest.approx(0.0)
    assert cv.is_satisfied is True, "the fix must not turn a real PASS into a refusal"
    assert cv.details["insufficient_data"] is False


def test_both_arms_unmeasurable_is_still_could_not_check():
    """A single group leaves no spread to measure on either arm."""
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    one_group = np.array(["a"] * 8)
    cv = EqualizedOddsConstraint().compute_violation(y_pred, y_true, one_group)

    assert _is_nan(cv.overall_violation)
    assert cv.is_satisfied is False
    assert cv.details["insufficient_data"] is True


# =============================================================================
# B4: THREE groups, so survivors REMAIN after the unmeasurable one is excluded.
#
# This is the case the two-group fixture above can never construct, and it is
# where a dropped group turns into a lower bound reported as a measurement.
# =============================================================================


def _arm(n_pos: int, n_neg: int, tpr: float, fpr: float):
    """Exact y_true / y_pred for ONE group with the requested TPR and FPR.

    Exact rather than sampled: a fixture for a fail-closed guard must not be
    able to drift across a tolerance boundary on a different numpy version.
    """
    n_tp, n_fp = round(tpr * n_pos), round(fpr * n_neg)
    y_true = np.array([1] * n_pos + [0] * n_neg, dtype=int)
    y_pred = np.array(
        [1] * n_tp + [0] * (n_pos - n_tp) + [1] * n_fp + [0] * (n_neg - n_fp), dtype=int
    )
    return y_true, y_pred


def _three_groups(c_pos: int, c_neg: int, c_tpr: float, c_fpr: float):
    """The audit's fixture. A and B are always fully measured; C is the variable.

    A: TPR 0.90 / FPR 0.10 (n=200) | B: TPR 0.88 / FPR 0.11 (n=200) | C: n=100.

    Every group clears the default ``min_group_size=30``, so nothing here is
    about the size gate: C is excluded only for having an undefined RATE. The
    A-vs-B TPR spread is 0.02, well inside a 0.05 tolerance, which is what made
    dropping C read as a PASS rather than as an obvious error.
    """
    parts = [
        _arm(100, 100, 0.90, 0.10),
        _arm(100, 100, 0.88, 0.11),
        _arm(c_pos, c_neg, c_tpr, c_fpr),
    ]
    y_true = np.concatenate([p[0] for p in parts])
    y_pred = np.concatenate([p[1] for p in parts])
    sens = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * (c_pos + c_neg))
    return y_true, y_pred, sens


# C has NO positive labels at all: its TPR is undefined, its FPR is fine.
UNMEASURABLE_C = dict(c_pos=0, c_neg=100, c_tpr=0.0, c_fpr=0.10)
# C is an ordinary, fully measured group. Chosen so that this fixture yields the
# very same 0.02 the DROP used to report above: same number, one measured and
# one fabricated, so the fix has to tell them apart by more than the value.
MEASURABLE_C = dict(c_pos=50, c_neg=50, c_tpr=0.90, c_fpr=0.10)
# C has NO negative labels: only the FPR leg is undefined.
NO_NEGATIVES_C = dict(c_pos=100, c_neg=0, c_tpr=0.90, c_fpr=0.0)


def _eod(y_true, y_pred, sens) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return equalized_odds_difference(y_true, y_pred, sens)


def _eop(y_true, y_pred, sens) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return equal_opportunity_difference(y_true, y_pred, sens)


# --- the constraint layer (_defined_spread) -----------------------------------


def test_constraint_three_groups_drops_nobody_and_refuses():
    """The measured B4 case: violation 0.02, is_satisfied True, C invisible."""
    y_true, y_pred, sens = _three_groups(**UNMEASURABLE_C)
    cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

    # Preconditions. TWO groups survive with defined TPRs, so unlike every
    # two-group case above there IS still a pair to spread over. That is the
    # whole point of this fixture.
    assert _is_nan(cv.details["group_tprs"]["C"]), "precondition: C's TPR is undefined"
    assert cv.details["group_tprs"]["A"] == pytest.approx(0.90)
    assert cv.details["group_tprs"]["B"] == pytest.approx(0.88)

    assert _is_nan(cv.details["tpr_violation"]), (
        "the TPR arm must be could-not-check: it used to report 0.02, the A-vs-B "
        "spread, with C silently excluded"
    )
    assert _is_nan(cv.overall_violation)
    assert cv.overall_violation != pytest.approx(0.02), "0.02 is a LOWER bound, not the violation"
    assert cv.is_satisfied is False, (
        "0.02 <= 0.05 issued a compliance PASS over a group the constraint could not see"
    )
    assert cv.details["insufficient_data"] is True


def test_constraint_three_groups_siblings_agree_and_are_both_could_not_check():
    """Agreement alone is not enough here: at three groups both were wrong.

    The earlier finding in this file was a DISAGREEMENT between these two.
    B4 is different and nastier: both dropped C, both reported 0.02, both said
    insufficient_data=False, so they agreed with each other and disagreed with
    reality. The value is therefore asserted alongside the agreement.
    """
    y_true, y_pred, sens = _three_groups(**UNMEASURABLE_C)
    eo = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)
    eop = EqualOpportunityConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

    assert eo.details["insufficient_data"] == eop.details["insufficient_data"] is True
    assert eo.is_satisfied == eop.is_satisfied is False
    assert _is_nan(eo.overall_violation) and _is_nan(eop.overall_violation)


def test_constraint_three_groups_all_measurable_is_unchanged():
    """OVER-CORRECTION CONTROL. Same 0.02, but here every group was measured."""
    y_true, y_pred, sens = _three_groups(**MEASURABLE_C)
    cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y_true, sens)

    assert cv.details["group_tprs"]["C"] == pytest.approx(0.90), "precondition: C IS measurable"
    assert cv.details["tpr_violation"] == pytest.approx(0.02)
    assert cv.details["fpr_violation"] == pytest.approx(0.01)
    assert cv.overall_violation == pytest.approx(0.02), "max over BOTH arms, all groups present"
    assert cv.is_satisfied is True, "a genuinely satisfied constraint must still pass"
    assert cv.details["insufficient_data"] is False


# --- the metric layer (equalized_odds_difference) -----------------------------


def test_metric_three_groups_refuses_instead_of_reporting_the_survivors_spread():
    """equalized_odds_difference must not answer with the A-vs-B spread."""
    y_true, y_pred, sens = _three_groups(**UNMEASURABLE_C)

    # Precondition: over A and B ALONE the metric is perfectly well defined and
    # equals 0.02. That is exactly the number the drop used to return for the
    # three-group data, i.e. it answered a question nobody asked.
    ab = sens != "C"
    assert _eod(y_true[ab], y_pred[ab], sens[ab]) == pytest.approx(0.02)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eod = equalized_odds_difference(y_true, y_pred, sens)

    assert _is_nan(eod), (
        "C has no positive labels, so the true equalized-odds gap is >= 0.02 and "
        f"unbounded above (C's TPR could be anything); got {eod}"
    )
    assert eod != pytest.approx(0.02)

    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "TPR is undefined for group(s)" in m and "'C'" in m and "equalized_odds_difference" in m
        for m in messages
    ), f"must warn in the same shape as equal_opportunity_difference; got {messages}"


@pytest.mark.parametrize(
    "case",
    [UNMEASURABLE_C, MEASURABLE_C],
    ids=["C has no positive labels", "C fully measured"],
)
def test_metric_siblings_are_assessable_under_the_same_conditions(case):
    """The TPR leg of equalized odds IS equal opportunity, over the same rates.

    Both fixtures have every FPR defined, so the FPR leg can never be the reason
    for a refusal and the two must be assessable together, in both directions.
    Before B4 the first case split them: NaN from equal_opportunity_difference,
    0.0200 from equalized_odds_difference, which let one report certify
    equalized odds while filing its own TPR leg as not assessable.
    """
    y_true, y_pred, sens = _three_groups(**case)
    eop = _eop(y_true, y_pred, sens)
    eod = _eod(y_true, y_pred, sens)

    assert _is_nan(eop) == _is_nan(eod), (
        "equal_opportunity_difference and the TPR leg of equalized_odds_difference "
        f"must be assessable under the same conditions; got {eop} vs {eod}"
    )


def test_metric_only_the_fpr_leg_undefined_refuses_equalized_odds_alone():
    """The one legitimate asymmetry, pinned so it is not mistaken for the bug.

    Equal opportunity does not CONTAIN the FPR leg, so when only that leg is
    undefined equal opportunity stays measurable and equalized odds refuses.
    That is the two metrics measuring different things, not diverging on the
    same thing.
    """
    y_true, y_pred, sens = _three_groups(**NO_NEGATIVES_C)
    assert not _is_nan(_eop(y_true, y_pred, sens)), "the TPR leg is fully defined here"
    assert _is_nan(_eod(y_true, y_pred, sens)), "C has no negative labels: its FPR is undefined"


def test_metric_three_groups_all_measurable_is_unchanged():
    """OVER-CORRECTION CONTROL for the metric layer."""
    y_true, y_pred, sens = _three_groups(**MEASURABLE_C)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eod = equalized_odds_difference(y_true, y_pred, sens)

    assert eod == pytest.approx(0.02), "TPR spread 0.02 beats FPR spread 0.01"
    assert not any("returning NaN" in str(w.message) for w in caught), (
        "healthy input must not warn about unmeasurable groups"
    )
    assert _eop(y_true, y_pred, sens) == pytest.approx(0.02)
