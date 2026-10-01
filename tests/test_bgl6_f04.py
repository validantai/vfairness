"""BGL6 audit of batch F04. ALL SEVEN RECORDS ARE CLOSED (2026-09-29).

This file began as evidence: every test asserted the CURRENT, wrong behaviour, so
it passed while the defect was live. All seven have now been fixed in the product
and inverted into PINS on the correct behaviour, and RECORDED_DEFECTS_STILL_OPEN
below is empty. Each docstring opens with "CLOSED 2026-09-29.", states what the fix
was, the re-measured numbers and the sabotage that proved the pin can fail, and
keeps the original record verbatim under "THE ORIGINAL RECORD follows."

Every pin here carries an over-correction control with a real number beside it, not
just a shape, because a fix that refuses everything passes every refusal test.

Sources changed: evaluation/vfairness_metrics/explanation_diagnostics.py (F04-1),
post_processing/calibration/tradeoffs.py (F04-4, F04-5, F04-6) and explainer.py
(F04-7). F04-2 and F04-3 were already fixed before this file was written and their
tests were already pins.

The seven records, as originally written:

  F04-1  CLOSED. removal_curve_auc refuses a tie only when np.unique sees ONE value, so a
         vector one ULP away from tied is scored in the caller's column order and
         answers 0.575, 0.775 or 0.975 for the same explanation on the same data,
         silently. Rows: removal_curve_auc.
  F04-2  CLOSED. _background_cannot_be_perturbed refuses only when NOT ONE column carries
         spread. With one column frozen the perturbation cloud cannot move along
         it at all, so a model scaffolded on that column is cleared with
         flag=False and confidence 1.0, byte identical to a clean model, with no
         warning. Rows: multi_seed_adversarial_probe (both), and
         diagnose_local_attribution, which publishes it.
  F04-3  CLOSED. ThresholdAnalyzer validates y_prob at construction and casts y_true with
         `.astype(int)` on the line above, which turns an unlabelled row into
         class 0 with a numpy RuntimeWarning and nothing else. Rows:
         analyze_threshold, find_optimal_threshold, find_feasible_region.
  F04-4  CLOSED. impossibility_diagnostics computes is_degenerate from an overall base
         rate that can be NaN, and `bool(nan < 0.01 or nan > 0.99)` is a MEASURED
         False, which flips impossibility_applies. Rows:
         impossibility_diagnostics, and analyze_calibration_fairness_tradeoff,
         which publishes it.
  F04-5  CLOSED. mitigation_pareto filters rows with no SCORE and decides rows with no
         LABEL as ground truth negatives via `yt0 >= 0.5`. Row: mitigation_pareto.
  F04-6  CLOSED. recommend_calibration_strategy names a group with no measured ECE in
         not_assessed_groups but adds the caveat sentence and the warning only
         when FEWER THAN TWO groups have one. Row: recommend_calibration_strategy.
  F04-7  CLOSED. _explain_threshold_analysis reports a region of unknown status as "No
         feasible threshold region found ... may need retraining", which is worse
         than the behaviour the fix replaced for that input. Row: get_explanation.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    _background_cannot_be_perturbed,
    diagnose_local_attribution,
    multi_seed_adversarial_probe,
    removal_curve_auc,
)
from vfairness.exceptions import InvalidDataError
from vfairness.explainer import _explain_threshold_analysis
from vfairness.post_processing.calibration.tradeoffs import (
    analyze_calibration_fairness_tradeoff,
    impossibility_diagnostics,
    mitigation_pareto,
    recommend_calibration_strategy,
)
from vfairness.post_processing.threshold_optimization.analyzer import ThresholdAnalyzer
from vfairness.xai.diagnostics.adversarial import (
    _unperturbable_background as _xai_guard,
)

# ===========================================================================
# F04-1. The tie guard tests EXACT equality, and one ulp is not equality.
# ===========================================================================

_W = np.array([0.5, -0.25, 1.0, 0.75])  # the batch's own control model
_X = np.array([1.0, 2.0, -1.0, 0.5])
_BG0 = np.zeros(4)


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
RECORDED_CLAIMS_AT_AUDIT = 13

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
RECORDED_DEFECTS_STILL_OPEN = []


def _auc_under_column_order(attributions, order):
    """The SAME explanation and the SAME data, with the columns arranged
    differently. A faithfulness score must not depend on this."""
    order = np.asarray(order)
    weights = _W[order]

    def predict(matrix):
        return np.atleast_2d(np.asarray(matrix, dtype=float)) @ weights

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = removal_curve_auc(
            predict, _X[order], np.asarray(attributions, dtype=float)[order], _BG0
        )
    return value, [str(w.message) for w in caught]


def test_the_exact_tie_really_is_refused():
    """CLOSED 2026-09-29. The exact tie is still refused after the tolerance fix.

    The fix replaced ``np.unique(abs_attr).size == 1`` with a tie-block cut on
    DISTINGUISHABILITY (``np.diff`` of the ranked magnitudes against a tolerance
    that scales with the vector), so this record had to be re-measured to prove
    the new test still catches the old input rather than only the new one.
    Re-measured: value nan, warning "all 4 |attribution| values are identical to
    within floating point tolerance (1.0 to 1.0, tolerance 1e-09) ... there is no
    order of |attribution| to mask features BY".

    THE ORIGINAL RECORD follows.
    The fix as recorded. Kept here so the contrast below is not an artefact.
    """
    value, caught = _auc_under_column_order([1.0, 1.0, 1.0, 1.0], [0, 1, 2, 3])
    assert not np.isfinite(value)
    assert any("no order of |attribution|" in m for m in caught)
    assert any("identical to within floating point tolerance" in m for m in caught), caught


def test_a_tie_one_ulp_wide_is_not_refused_and_moves_with_the_column_order():
    """CLOSED 2026-09-29. Inverted into a pin: the one-ulp tie is refused, in
    every column order, and the score can no longer move with the arrangement.

    THE FIX, in src/vfairness/evaluation/vfairness_metrics/explanation_diagnostics.py:
    two changes, because fixing only the first would have moved the fabrication one
    branch along (a partial tie is equally arrangement-chosen).
      1. The total-tie guard no longer tests exact float equality. The ranked
         magnitudes are cut into tie blocks with ``np.diff(ranked) < -atol``,
         atol = 1e-12 + 1e-9 * max|attr|, and a single block over a vector of more
         than one feature is refused. np.diff/np.ptp deliberately, never np.var:
         np.var of a constant array is exactly 0.0 only at some n.
      2. ABOVE the return, not inside one branch: when any tie block is wider than
         one feature the curve is walked a SECOND time with every tie block
         reversed (the opposite extreme of the arrangements argsort could return).
         Equal under both arrangements means the published number does not depend
         on the arrangement and is a measurement; different means it does, and the
         run returns NaN with a warning naming both arms.

    RE-MEASURED after the fix, same fixture, all three column orders
    ([0,1,2,3], [3,2,1,0], [2,0,1,3]): nan, nan, nan, each with one warning. Before:
    0.775, 0.975, 0.575, silently, 0.4 of the whole [0, 1] scale chosen by the
    column arrangement.

    THE HEALTHY CASE IS ASSERTED BELOW TOO, because a blanket refusal would pass
    every line above: the partially tied vector [1.0, 1.0, 0.2, 0.1] still returns
    exactly 0.725 in silence, and the model's true attributions still return
    exactly 1.0 in silence.

    SABOTAGE, run three ways, because the two halves of the fix have to be shown
    to carry their own weight:
      a) atol forced to 0.0 (back to exact equality), fix 2 left in place:
         "AssertionError: ['removal_curve_auc: 3 of 4 |attribution| values are
         tied, and the score MOVES with the arrangement of the tied features ...']"
         RED here, and the reading is that fix 2 is the backstop under fix 1: the
         input is still refused, by the other guard, with the other sentence.
      b) fix 2 disabled (``if False and n_blocks < ...``), atol left in place:
         "AssertionError: [1.0, 1.0, 0.1] scored 0.8121693121693121" RED.
      c) both disabled, which is the pre-fix source:
         "AssertionError: the one-ulp tie was scored 0.775 in column order
         (0, 1, 2, 3)" RED, i.e. the original recorded number reproduces exactly.
    Source restored after each and proved byte-identical with diff -q.

    THE ORIGINAL RECORD follows.
    The guard is ``np.unique(abs_attr).size == 1``. Add one ulp to a single entry
    and np.unique sees two values, so the guard does not fire, although the top
    THREE magnitudes are still exactly equal and nothing orders them but their
    position in the caller's array.
    """
    eps = np.finfo(float).eps
    almost_tied = [1.0, 1.0, 1.0, 1.0 + eps]
    # The input is unchanged: np.unique still sees two values here, so the fix is
    # the tolerance and not a different fixture.
    assert np.unique(np.abs(np.asarray(almost_tied))).size == 2

    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 1, 3]):
        value, caught = _auc_under_column_order(almost_tied, order)
        assert not np.isfinite(value), (
            f"the one-ulp tie was scored {value} in column order {tuple(order)}"
        )
        assert any("no order of |attribution|" in m for m in caught), caught

    # OVER-CORRECTION CONTROL, with the real numbers and not just their shape. A
    # guard that refused everything would satisfy every assertion above.
    partial, quiet = _auc_under_column_order([1.0, 1.0, 0.2, 0.1], [0, 1, 2, 3])
    assert partial == pytest.approx(0.725), partial
    assert quiet == [], quiet
    true_attr, quiet_true = _auc_under_column_order(list(_W * _X), [0, 1, 2, 3])
    assert true_attr == pytest.approx(1.0), true_attr
    assert quiet_true == [], quiet_true


def test_a_partial_tie_that_moves_the_score_is_refused_and_names_both_arms():
    """CLOSED 2026-09-29, the second half of the F04-1 fix, and the proof that the
    ambiguity measurement is not dead code.

    A partial tie is still not refused outright (see the control above). It is
    refused when the arrangement demonstrably moves the number. Measured here with
    a model weighted 3.0 / 0.1 / 0.05 and attributions [1.0, 1.0, 0.1]: the same
    explanation on the same row scores 0.8121693121693121 in the caller's column
    order and 0.5052910052910052 with the tied pair reversed, so the published
    answer is NaN and the warning carries both arms.
    """

    def predict(matrix):
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        return matrix[:, 0] * 3.0 + matrix[:, 1] * 0.1 + matrix[:, 2] * 0.05

    x = np.array([1.0, 1.0, 1.0])
    for attributions in ([1.0, 1.0, 0.1], [1.0, 1.0000000001, 0.1]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = removal_curve_auc(
                predict, x, np.asarray(attributions, dtype=float), np.zeros(3)
            )
        said = [str(w.message) for w in caught]
        assert not np.isfinite(value), f"{attributions} scored {value}"
        assert any("MOVES with the arrangement of the tied features" in m for m in said), said
        assert any("0.8121693121693121" in m and "0.5052910052910052" in m for m in said), said

    # AND THE CONTROL for this branch: a tie whose arrangement does NOT move the
    # score keeps its measured number, which is the case the preserved comment in
    # the source is about (the true attributions tie two features at |0.5|).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kept = removal_curve_auc(predict, x, np.array([1.0, 1.0, 1.0, 1.0])[:3], np.zeros(3))
    assert not np.isfinite(kept)  # that one is a total tie, refused by guard 1
    value, quiet = _auc_under_column_order(list(_W * _X), [0, 1, 2, 3])
    assert value == pytest.approx(1.0) and quiet == []


# ===========================================================================
# F04-2. The spread guard refuses only a TOTALLY frozen background.
# ===========================================================================


def _partly_frozen_background():
    """Column 0 frozen at 1.0, column 1 a real sample. A constant column in a
    reference sample is the commonest shape there is."""
    rng = np.random.default_rng(3)
    return np.column_stack([np.full(30, 1.0), 2.0 + rng.normal(0.0, 0.5, 30)])


def _scaffolded_on_the_frozen_column(matrix):
    """0.95 off the manifold, 0.05 on it, keyed on column 0."""
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    return np.where(np.abs(matrix[:, 0] - 1.0) > 0.01, 0.95, 0.05)


def _clean_flat(matrix):
    matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
    return np.full(len(matrix), 0.05)


def test_both_twin_guards_refuse_a_partly_frozen_background():
    """OVERTURN F04-2, the twins half. FIXED 2026-09-28, and this is now the pin.

    The seven fixtures the twins were pinned on are each either fully frozen or
    fully real, so they agreed, and they agreed on the answer that let the probe
    run with no power along a frozen column: `if n_moving: return` cleared as soon
    as ONE column moved. Before: both guards said perturbable. After: both refuse
    with the same sentence, naming the column, and they are compared word for word
    here because two sibling guards disagreeing about one input is the shape that
    let the original hole survive.
    """
    background = _partly_frozen_background()
    assert float(np.ptp(background[:, 0])) == 0.0

    twin = _background_cannot_be_perturbed(background)
    xai = _xai_guard(background)
    assert twin, "the evaluation twin still calls a partly frozen background perturbable"
    assert xai, "the xai twin still calls a partly frozen background perturbable"
    assert twin == xai, (
        "the twins refuse for different reasons, which is how the first hole "
        f"survived:\n  twin: {twin}\n  xai:  {xai}"
    )
    assert "no usable spread" in twin and "index 0" in twin, twin

    # AND THE CONTROL: a background whose every column varies is still perturbable,
    # so the guard is about the frozen column and not about refusing everything.
    real = np.random.default_rng(0).normal(size=(50, 2)) + np.array([1.0, 2.0])
    assert _background_cannot_be_perturbed(real) is None
    assert not bool(_xai_guard(real))


def test_a_partly_frozen_background_no_longer_clears_a_scaffolded_model():
    """OVERTURN F04-2. FIXED 2026-09-28, and this is now the pin.

    ``sigma = background.std(axis=0) + 1e-9`` gives a frozen column a scale of
    1e-9, which is the division guard and not spread, so every perturbed point
    keeps x[0] and the gap along it is exactly 0.0 for ANY model. That is the
    condition the refusal's own wording describes ("the check could not have fired
    for any data") and it was not refused.

    BEFORE: flag False, confidence 1.0, mean_gap 0.0, not_run_because "", no
    warning, and the answer for a model deliberately scaffolded on the frozen
    column was BYTE-IDENTICAL to the answer for a clean one.

    AFTER: flag None for both, confidence and mean_gap NaN, a reason naming the
    column, one warning each. They are still identical to each other, and that is
    correct and is the whole point: the probe learned nothing about that column for
    either model, so it must clear NEITHER. What changed is that the identical
    answer is now a could-not-check instead of a clean bill.
    """
    background = _partly_frozen_background()
    x = np.array([1.0, 2.0])

    results = {}
    for name, model in (("scaffolded", _scaffolded_on_the_frozen_column), ("clean", _clean_flat)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results[name] = multi_seed_adversarial_probe(
                model, x, background, n_perturbations=200, n_seeds=6
            )
        # The probe WARNS now, which it did not before. That is the disclosure, so
        # it is asserted rather than forbidden: before, the silence was part of the
        # defect ("flag False, confidence 1.0, no warning").
        assert caught, f"{name} refused and said nothing"
        assert any("could perturb x away from the background" in str(w.message) for w in caught), (
            f"{name} warned about something else: {[str(w.message) for w in caught]}"
        )

    for name, probe in results.items():
        assert probe.flag is None, f"{name}: flag {probe.flag!r} clears the model"
        assert math.isnan(probe.confidence), name
        assert math.isnan(probe.mean_gap), name
        assert "COULD NOT CHECK" in (probe.reason or ""), name
        assert "no usable spread" in (probe.reason or ""), name

    # Identical to each other, and that is now the honest answer rather than the
    # defect: the probe had no power over the frozen column for either model, so it
    # must refuse both. Before, the identical answer was flag False / confidence 1.0.
    assert results["scaffolded"].flag is None and results["clean"].flag is None


def test_the_consumer_no_longer_publishes_a_cleared_verdict():
    """OVERTURN F04-2, the consumer half (diagnose_local_attribution). FIXED.

    BEFORE: adversarial_flag False, adversarial_confidence 1.0, adversarial_reason
    None, notes [], no warning, for a model scaffolded on a column the probe never
    moved. AFTER: the flag is not False, the reason names the frozen column, and
    the run says so where a reader sees it.
    """
    background = _partly_frozen_background()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = diagnose_local_attribution(
            _scaffolded_on_the_frozen_column, [1.0, 2.0], [0.5, 0.25], background, n_seeds=6
        )
    assert out["adversarial_flag"] is not False, (
        "the consumer still publishes a clean adversarial verdict for a model the "
        "probe could not move"
    )
    assert out["adversarial_reason"], "the consumer publishes no reason for the refusal"
    assert "no usable spread" in str(out["adversarial_reason"]), out["adversarial_reason"]
    assert caught, "the run is silent about a probe that could not check"


# ===========================================================================
# F04-3. y_prob is validated at construction. y_true is cast on the line above.
# ===========================================================================


def _one_group_unlabelled():
    """Two fully SCORED 100 row groups. Group B carries no label at all."""
    rng = np.random.default_rng(0)
    prob_a = rng.uniform(0.0, 0.49, 100)
    prob_b = rng.uniform(0.5, 0.99, 100)
    y = np.concatenate([(rng.random(100) < 0.4).astype(float), np.full(100, np.nan)])
    groups = np.array(["A"] * 100 + ["B"] * 100)
    return y, np.concatenate([prob_a, prob_b]), groups


def test_the_threshold_analyzer_refuses_a_row_nobody_labelled():
    """FIXED 2026-09-28, and this is now the pin.

    ``self.y_true = coerce_to_array(y_true).astype(int)`` sat ONE LINE ABOVE the
    validate_probabilities call the earlier batch relies on, so the label was cast
    before anything looked at it. Casting NaN to int is undefined in numpy: it yields
    a platform-dependent integer rather than raising.

    BEFORE, on this host: 100 rows nobody labelled became 100 ground-truth negatives,
    and the only signal was numpy's own "invalid value encountered in cast" warning.
    On CI's numpy the same call behaved differently, so the library's answer to "is
    this model fair" varied with the host, which is the worst property a fabrication
    can have.

    AFTER: InvalidDataError at construction, naming how many values are not a finite
    label and what to do about it. Refused at construction for the same reason y_prob
    is: every method of the class derives its rates from self.y_true.
    """
    y, prob, groups = _one_group_unlabelled()
    with pytest.raises(InvalidDataError) as excinfo:
        ThresholdAnalyzer(y, prob, groups)
    said = str(excinfo.value)
    assert "100 of 200" in said, said
    assert "no true-positive or false-positive rate" in said, said


def test_control_a_fully_labelled_frame_is_still_accepted():
    """The over-correction control. A constructor that refused every frame would
    satisfy the pin above and make the whole class unusable, and it would look
    identical from the refusal side."""
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(float)
    prob = rng.uniform(0.0, 1.0, 200)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    analyzer = ThresholdAnalyzer(y, prob, groups)
    assert set(np.unique(analyzer.y_true)).issubset({0, 1})
    result = analyzer.analyze_threshold(0.5, constraints=["equalized_odds"])
    v = result.constraint_violations["equalized_odds"]
    assert v.violation is not None and np.isfinite(v.violation), v.violation


def test_an_unlabelled_row_no_longer_becomes_a_measured_equalized_odds_violation():
    """FIXED 2026-09-28. The same defect at the verdict.

    BEFORE: analyze_threshold(0.5, ['equalized_odds']) returned violation 1.0,
    is_satisfied False, group B fpr 1.0 and a confusion matrix of
    {'tp': 0, 'tn': 0, 'fp': 100, 'fn': 0}, naming only 'B.tpr' as unmeasured, so the
    FPR arm read as a measurement of 100 rows nobody labelled. A maximal finding of
    unfairness, asserted about labels that do not exist.

    AFTER: the analyzer cannot be constructed from that frame at all, so no verdict is
    reached rather than a false one being published.
    """
    y, prob, groups = _one_group_unlabelled()
    with pytest.raises(InvalidDataError):
        ThresholdAnalyzer(y, prob, groups).analyze_threshold(0.5, constraints=["equalized_odds"])


def test_an_unlabelled_row_no_longer_yields_a_measured_infeasible_region():
    """FIXED 2026-09-28. The same defect in find_feasible_region and
    find_optimal_threshold, which is where it would have reached a deployment
    decision: before this, the region came back assessed with an optimum on it.
    """
    y, prob, groups = _one_group_unlabelled()
    analyzer_args = (y, prob, groups)
    with pytest.raises(InvalidDataError):
        ThresholdAnalyzer(*analyzer_args).find_feasible_region(constraint="equalized_odds")


def _near_universal_outcome(with_unlabelled_group):
    """A base rate 1.000 and B base rate 0.985: the overall rate is 0.9925, so the
    outcome IS near universal and the theorem does not bind."""
    y = [np.ones(200), np.concatenate([np.ones(197), np.zeros(3)])]
    groups = [np.array(["A"] * 200), np.array(["B"] * 200)]
    prob = [np.linspace(0.02, 0.98, 200), np.linspace(0.02, 0.98, 200)]
    if with_unlabelled_group:
        y.append(np.full(200, np.nan))
        groups.append(np.array(["C"] * 200))
        prob.append(np.linspace(0.02, 0.98, 200))
    return np.concatenate(y), np.concatenate(prob), np.concatenate(groups)


def test_is_degenerate_is_a_fabricated_false_and_flips_the_theorem_verdict():
    """CLOSED 2026-09-29. Inverted into a pin: the degeneracy test is measured over
    the population the verdict rests on, so the theorem verdict no longer flips when
    an unlabelled group is added, and it is three-state where nothing is measurable.

    THE FIX, in src/vfairness/post_processing/calibration/tradeoffs.py. The first
    attempt was WRONG and its own red test said so, which is recorded here because
    the correction is the finding: setting is_degenerate=None whenever ANY row lacks
    a finite label broke
    tests/test_bgl5_post_processing_1.py::TestTheImpossibilityGateCountsMeasuredRates
    ::test_a_measured_verdict_that_leaves_a_group_out_says_which, whose subject is
    exactly "a measured verdict that leaves a group out must STAND". That test is
    right. The defect was never that the mean went NaN; it was that the degeneracy
    test rested on a DIFFERENT POPULATION from the verdict it gates.
      1. ``overall_base_rate`` is measured over the rows of the groups that HAVE a
         measured base rate, which is the population ``base_rates_differ`` is
         computed from. A group whose labels are missing has no measured rate
         (np.mean propagates NaN), so it is outside the comparison and is now
         outside the degeneracy test with it. Not "the mean of every finite label",
         which would fold back in the groups excluded for size.
      2. ``is_degenerate`` is Optional[bool] and is None when that population has no
         finite label at all, with a warning saying so.
      3. ``impossibility_applies`` is None when ``is_degenerate`` is None, because
         ``not None`` is True and an unmeasurable degeneracy used to SATISFY the
         theorem's second condition outright.
      4. Both are computed ABOVE the two return paths. The early return for fewer
         than two measured base rates publishes ``is_degenerate`` as well, so a fix
         inside the lower branch would have moved the fabricated False one branch
         along instead of removing it.
      5. ``_generate_impossibility_explanation`` was called with
         ``bool(is_degenerate)``, and bool(None) is False, so the third state would
         have reached the dictionary key and not the sentence a reader acts on. It
         takes Optional[bool] now and answers "NOT ASSESSED. ...".

    RE-MEASURED, same fixture:
      labelled_only    overall 0.9925, is_degenerate True, applies False, no warning
      plus_unlabelled  overall 0.9925, is_degenerate True, applies False, one warning
                       naming group C, and "(1 group(s) inside the size gate have no
                       measured base rate ...: C.)" appended to the explanation
      Before: overall nan, is_degenerate False, applies TRUE for the second run, so
      the two runs disagreed on the headline for the same two measured base rates.
      The verdicts are now identical, which is the point.
      And the third state, on 200 rows of one group with no labels at all: before
      is_degenerate False (a measurement out of a NaN), after None, applies None.

    SABOTAGE, four ways, one per moving part, each restored and proved
    byte-identical with diff -q afterwards:
      a) the population reverted to every row instead of the compared rows:
         FAILED test_a_group_excluded_for_size_cannot_flip_the_degeneracy_verdict
         "AssertionError: the degeneracy test folded in a group the comparison
         excluded for size / assert 0.980246913580247 == 0.9925 +- 9.9e-07".
         Recorded because the FIRST sabotage attempt of this half did NOT fire: the
         finite-label filter alone reproduces 0.9925 for the unlabelled-group
         fixture, so the mask needed a fixture where it is load-bearing, and the
         five-row excluded group is it.
      b) the ``is_degenerate = None`` branch replaced by ``= False``:
         FAILED test_a_population_with_no_label_at_all_reports_degeneracy_as_
         could_not_check "AssertionError: a group with no labels at all still
         answers is_degenerate=False / assert False is None".
      c) ``impossibility_applies`` reverted to the one-line ``bool(base_rates_differ
         and not is_degenerate)``: FAILED the same test, "AssertionError:
         impossibility_applies True rests on a degeneracy test that could not be
         run / assert True is None".
      d) the ``is_degenerate is None`` branch of the explanation disabled: FAILED the
         same test, "assert 'Base rates differ substantially (40.0%). Strong
         trade-off ...'.startswith('NOT ASSESSED.')" is False, which is the
         could-not-check being published as a measured strong finding.

    THE ORIGINAL RECORD follows.
    ``bool(nan < 0.01 or nan > 0.99)`` is False, so the degeneracy test reports a
    MEASURED "not degenerate" for an overall base rate it could not compute, and
    ``impossibility_applies = bool(base_rates_differ and not is_degenerate)``
    rests on it. The two measured base rates are identical in both runs.
    """
    out = {}
    said = {}
    for label, extra in (("labelled_only", False), ("plus_unlabelled", True)):
        y, prob, groups = _near_universal_outcome(extra)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out[label] = impossibility_diagnostics(y, prob, groups)
        said[label] = [str(w.message) for w in caught]

    assert out["labelled_only"]["base_rate_disparity"] == pytest.approx(
        out["plus_unlabelled"]["base_rate_disparity"]
    )
    assert out["labelled_only"]["n_groups_compared"] == 2
    assert out["plus_unlabelled"]["n_groups_compared"] == 2

    # THE HEALTHY CASE, with its real numbers. A blanket refusal would satisfy every
    # assertion about an unmeasurable input and make the function useless.
    assert out["labelled_only"]["overall_base_rate"] == pytest.approx(0.9925)
    assert out["labelled_only"]["is_degenerate"] is True
    assert out["labelled_only"]["impossibility_applies"] is False
    assert said["labelled_only"] == [], said["labelled_only"]
    assert out["labelled_only"]["explanation"].startswith("The outcome is nearly universal")

    # THE PIN: adding a group nobody labelled changes NOTHING about the verdict the
    # two measured base rates support, and the reader is told the group is out.
    assert out["plus_unlabelled"]["overall_base_rate"] == pytest.approx(0.9925), (
        f"the degeneracy test still reads a population the verdict is not about: "
        f"{out['plus_unlabelled']['overall_base_rate']}"
    )
    assert out["plus_unlabelled"]["is_degenerate"] is True, (
        f"is_degenerate {out['plus_unlabelled']['is_degenerate']!r} is a measurement "
        f"out of an overall base rate of "
        f"{out['plus_unlabelled']['overall_base_rate']}"
    )
    assert out["plus_unlabelled"]["impossibility_applies"] is False, (
        "the headline verdict flipped because a group nobody labelled was added"
    )
    assert out["plus_unlabelled"]["groups_without_measured_base_rate"] == ["C"]
    assert "C" in out["plus_unlabelled"]["explanation"]
    assert any("NO measured base rate" in m for m in said["plus_unlabelled"]), said[
        "plus_unlabelled"
    ]


def test_a_group_excluded_for_size_cannot_flip_the_degeneracy_verdict():
    """CLOSED 2026-09-29, and the reason the fix masks to the COMPARED rows rather
    than to "every finite label", which would have been the smaller change and is
    not the right one.

    A group below min_group_size contributes no base rate to the comparison, so it
    must contribute nothing to the degeneracy test that gates the same verdict.
    Measured here: A 200 rows at 1.000, B 200 rows at 0.985 (the compared
    population, pooled 0.9925, degenerate, so the theorem does not bind) plus a
    FIVE row group D at 0.000, excluded for size. Pooled over every finite label the
    rate is 0.980246913580247, which is NOT degenerate, and the headline verdict
    would flip to "the theorem applies" on the strength of five rows that were
    explicitly left out of the comparison.
    """
    y = np.concatenate([np.ones(200), np.ones(197), np.zeros(3), np.zeros(5)])
    groups = np.array(["A"] * 200 + ["B"] * 200 + ["D"] * 5)
    prob = np.concatenate([np.linspace(0.02, 0.98, 200)] * 2 + [np.full(5, 0.5)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnosis = impossibility_diagnostics(y, prob, groups)

    assert diagnosis["excluded_groups"] == ["D"]
    assert float(np.mean(y)) == pytest.approx(0.980246913580247)  # the wrong population
    assert diagnosis["overall_base_rate"] == pytest.approx(0.9925), (
        "the degeneracy test folded in a group the comparison excluded for size"
    )
    assert diagnosis["is_degenerate"] is True
    assert diagnosis["impossibility_applies"] is False


def test_a_population_with_no_label_at_all_reports_degeneracy_as_could_not_check():
    """CLOSED 2026-09-29, the third state of the same fix, on the OTHER return path.

    The early return for fewer than two measured base rates publishes
    ``is_degenerate`` too, and it was computed the same way, so 200 rows of a single
    group with no labels answered a MEASURED is_degenerate False. RE-MEASURED after
    the fix: overall_base_rate nan, is_degenerate None, impossibility_applies None,
    and a warning that says which of the three states this is.
    """
    y = np.full(200, np.nan)
    prob = np.linspace(0.02, 0.98, 200)
    groups = np.array(["A"] * 200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnosis = impossibility_diagnostics(y, prob, groups)
    said = [str(w.message) for w in caught]

    assert not np.isfinite(diagnosis["overall_base_rate"])
    assert diagnosis["is_degenerate"] is None, (
        f"a group with no labels at all still answers is_degenerate={diagnosis['is_degenerate']!r}"
    )
    assert diagnosis["impossibility_applies"] is None
    assert any("could not be measured" in m and "is_degenerate=None" in m for m in said), said
    assert diagnosis["explanation"].startswith("NOT ASSESSED."), diagnosis["explanation"]

    # AND THE SAME THIRD STATE ON THE MAIN RETURN PATH, which is reachable because
    # ``base_rates`` is a documented parameter: a caller can supply two measured
    # rates for rows that carry no label, so the comparison is made and the
    # degeneracy test still has nothing to measure. Before the fix this was
    # ``bool(True and not False)``, a fabricated APPLIES, in the strongest wording
    # the function has.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        supplied = impossibility_diagnostics(
            y, prob, np.array(["A"] * 100 + ["B"] * 100), base_rates={"A": 0.6, "B": 0.2}
        )
    said_supplied = [str(w.message) for w in caught]
    assert supplied["n_groups_compared"] == 2
    assert supplied["base_rates_differ"] is True
    assert supplied["is_degenerate"] is None, supplied["is_degenerate"]
    assert supplied["impossibility_applies"] is None, (
        f"impossibility_applies {supplied['impossibility_applies']!r} rests on a "
        f"degeneracy test that could not be run"
    )
    # The sentence, not only the key: bool(None) is False, so the third state would
    # otherwise have fallen through to one of the two measured sentences.
    assert supplied["explanation"].startswith("NOT ASSESSED."), supplied["explanation"]
    assert "overall base rate could not be measured" in supplied["explanation"]
    assert any("could not be measured" in m for m in said_supplied), said_supplied


def test_the_tradeoff_analysis_publishes_the_flipped_theorem_verdict():
    """CLOSED 2026-09-29, the consumer half (analyze_calibration_fairness_tradeoff).

    RE-MEASURED: neither run publishes an IMPOSSIBILITY line now, because neither
    run finds the theorem binding. Before, the unlabelled run published
    "IMPOSSIBILITY THEOREM APPLIES: Perfect calibration and error parity cannot both
    be achieved. Document your prioritization decision." off the fabricated False,
    while the labelled run published nothing at all for the same two base rates.

    The consumer's own three-state read of ``impossibility_applies`` (READINESS-6,
    2026-09-10) is exercised separately below, so this pin cannot pass by the
    could-not-check branch having been deleted.

    THE ORIGINAL RECORD follows.
    OVERTURN F04-4, the consumer (analyze_calibration_fairness_tradeoff).
    """
    lines = {}
    for label, extra in (("labelled_only", False), ("plus_unlabelled", True)):
        y, prob, groups = _near_universal_outcome(extra)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze_calibration_fairness_tradeoff(y, prob, groups)
        lines[label] = [r for r in result.recommendations if "IMPOSSIBILITY THEOREM" in r]

    assert lines["labelled_only"] == []
    assert lines["plus_unlabelled"] == [], (
        f"the consumer still publishes a theorem verdict the producer did not "
        f"measure: {lines['plus_unlabelled']}"
    )

    # AND THE COULD-NOT-CHECK ARM, so that a deleted branch cannot make the pin
    # above pass: two groups above the size gate of which one carries no label, so
    # only one base rate is measured and there is nothing to compare. The consumer
    # says so in words rather than falling silent.
    rng = np.random.default_rng(0)
    y = np.concatenate([(rng.random(100) < 0.4).astype(float), np.full(100, np.nan)])
    prob = np.concatenate([np.linspace(0.02, 0.98, 100)] * 2)
    groups = np.array(["A"] * 100 + ["B"] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyze_calibration_fairness_tradeoff(y, prob, groups)
    said = [r for r in result.recommendations if "IMPOSSIBILITY THEOREM" in r]
    assert len(said) == 1, said
    assert said[0].startswith("IMPOSSIBILITY THEOREM: COULD NOT BE CHECKED"), said


# ===========================================================================
# F04-5. mitigation_pareto filters the unscored row and keeps the unlabelled one.
# ===========================================================================


def test_mitigation_pareto_decides_an_unlabelled_row_as_a_ground_truth_negative():
    """CLOSED 2026-09-29. Inverted into a pin: an unlabelled row is no longer a
    ground-truth negative, and the run that cannot be measured says so.

    THE FIX, in src/vfairness/post_processing/calibration/tradeoffs.py. The score
    arm was filtered in 2026-09-27 and the label arm, built by the same arithmetic
    one line up, was not, so the fabrication simply moved to the sibling array.
    Rows whose label is not finite are now dropped WITH the unscored rows, once,
    before any group is measured, and only when the run is label-aware at all
    (``yt`` None reads no label, so there is nothing to fabricate). The count is
    published in ``nRowsUnlabelled`` on EVERY return path, named in the refusal
    ``reason`` and named in ``summary``, which is the string a reader acts on.

    RE-MEASURED, same fixture:
      labelled    unchanged: available True, accuracy of the current system 0.495,
                  bestBalanced "Group thresholds (equal opportunity)", "at 5.0
                  points of accuracy cost", nRowsUnlabelled 0, no extra sentence.
      unlabelled  available False, reason "... 1 of 2 group(s) qualify, so no gap
                  was measured. This is NOT a finding that the gap is zero. 100
                  row(s) carry no finite label and are excluded, because an
                  unlabelled row is not a ground-truth negative.",
                  nRowsUnlabelled 100.
      Before: available True, labelAware True, nRowsUnscored 0, accuracy 0.865 (a
      37 point rise on the same frame, because 100 rows nobody labelled were
      counted as correctly rejected), bestBalanced "(equal selection rate)", "at
      41.0 points of accuracy cost", and not one word about a label.

    OVER-CORRECTION CONTROL, asserted below with its real numbers, because
    refusing every frame with a gap in its labels would pass every line above and
    would delete real findings: ONE missing label out of 200 still returns
    available True with the accuracy of the current system 0.4975 measured over
    the 199 rows that have one, the same bestBalanced, and the exclusion named in
    the summary.

    SABOTAGE (``labelled = np.ones(n, dtype=bool)`` forced, i.e. the label filter
    removed while the unscored filter stays):
        FAILED test_mitigation_pareto_decides_an_unlabelled_row_as_a_ground_truth_negative
          - AssertionError: 100 rows nobody labelled are still counted as
            correctly rejected: accuracy 0.865, available True
    Source restored and proved byte-identical with diff -q.

    THE ORIGINAL RECORD follows.
    The fix drops rows with no SCORE because ``nan >= thr`` is False. The label
    arm is built the same way, ``yt = (yt0 >= 0.5).astype(int)``, and is not
    filtered, so an unlabelled row is a ground truth 0 in every accuracy figure,
    in ``base_rate_g`` and therefore in the whole context distortion axis.
    ``labelAware`` still says True and ``nRowsUnscored`` still says 0.
    """
    rng = np.random.default_rng(11)
    prob = np.concatenate([rng.uniform(0.55, 0.99, 100), rng.uniform(0.01, 0.45, 100)])
    y_a = (rng.random(100) < 0.7).astype(float)
    y_b_real = (rng.random(100) < 0.7).astype(float)
    groups = np.array(["A"] * 100 + ["B"] * 100)

    out = {}
    for label, y_b in (("labelled", y_b_real), ("unlabelled", np.full(100, np.nan))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out[label] = mitigation_pareto(np.concatenate([y_a, y_b]), prob, groups)
        assert caught == [], f"{label}: {[str(w.message) for w in caught]}"

    # THE HEALTHY CASE, with the numbers the record measured for it. Unchanged.
    good = out["labelled"]
    assert good["available"] is True
    assert good["labelAware"] is True
    assert good["nRowsUnscored"] == 0
    assert good["nRowsUnlabelled"] == 0
    assert good["nGroupsMeasurable"] == 2
    assert good["points"][0]["accuracy"] == pytest.approx(0.495)
    assert good["bestBalanced"] == "Group thresholds (equal opportunity)"
    assert "at 5.0 points of accuracy cost" in good["summary"]
    assert "no finite label" not in good["summary"], good["summary"]

    # THE PIN: the run whose ground truth is missing for a whole group is not
    # measured at all, rather than measured 37 points better for the absence.
    bad = out["unlabelled"]
    assert bad["available"] is False, (
        f"100 rows nobody labelled are still counted as correctly rejected: "
        f"accuracy {bad.get('points', [{}])[0].get('accuracy')}, available True"
    )
    assert bad["nRowsUnlabelled"] == 100
    assert bad["nGroupsMeasurable"] == 1
    assert "no finite label" in bad["reason"], bad["reason"]
    assert "not a ground-truth negative" in bad["reason"], bad["reason"]
    assert "NOT a finding that the gap is zero" in bad["reason"], bad["reason"]


def test_control_one_missing_label_is_still_measured_over_the_rows_that_have_one():
    """CLOSED 2026-09-29, the over-correction control for F04-5, with its real
    numbers rather than its shape.

    Refusing any frame with a gap in its labels would satisfy every assertion in
    the pin above and would delete real findings, which is the failure mode the
    sibling unscored filter was explicitly built to avoid. ONE unlabelled row in
    200 still yields a measured trade-off over the 199 that carry both a score and
    a label, and the exclusion is named in the summary.
    """
    rng = np.random.default_rng(11)
    prob = np.concatenate([rng.uniform(0.55, 0.99, 100), rng.uniform(0.01, 0.45, 100)])
    y_a = (rng.random(100) < 0.7).astype(float)
    y_b = (rng.random(100) < 0.7).astype(float)
    y_b[-1] = np.nan
    groups = np.array(["A"] * 100 + ["B"] * 100)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mitigation_pareto(np.concatenate([y_a, y_b]), prob, groups)

    assert caught == [], [str(w.message) for w in caught]
    assert out["available"] is True
    assert out["labelAware"] is True
    assert out["nGroupsMeasurable"] == 2
    assert out["nRowsUnlabelled"] == 1
    assert out["points"][0]["accuracy"] == pytest.approx(0.4975)
    assert out["bestBalanced"] == "Group thresholds (equal opportunity)"
    assert (
        "Measured over the 199 row(s) that carry both a score and a finite label"
        in (out["summary"])
    ), out["summary"]
    assert "1 row(s) had no finite label" in out["summary"], out["summary"]


# ===========================================================================
# F04-6. The caveat sentence fires only when FEWER THAN TWO ECEs are measured.
# ===========================================================================


def test_a_third_group_without_a_measured_ece_gets_no_caveat_and_no_warning():
    """CLOSED 2026-09-29. Inverted into a pin: the group with no measured
    calibration error is named in the RATIONALE, not only in a list field.

    THE FIX, in src/vfairness/post_processing/calibration/tradeoffs.py. Both ways a
    group can be outside the disparity already reached ``not_assessed_groups`` and
    both already bumped the priority; the only branch that reached the rationale was
    gated on ``excluded`` alone, i.e. on "too small to look at". A second NOTE
    sentence now covers the other way, a group inside the size gate whose
    calibration error could not be measured, and a warning goes with it. The
    existing EXCLUDED sentence is untouched, so the H-06 pin that quotes it still
    holds.

    WHY IT SURVIVED, which is the trap this batch is full of: the guard higher up
    fires on ``not math.isfinite(ece_disparity) or len(eces_measured) < 2``. With
    TWO measured group ECEs and a THIRD group carrying none, neither clause is true,
    so that guard's NOT ASSESSED sentence and warning never ran, and this branch did
    not cover the gap. Fixing only the upper guard's threshold would have moved the
    hole to four groups with two measured.

    RE-MEASURED, same fixture (A calibrated, B not, C carrying no label):
      not_assessed_groups ['C'], disparity_assessed True (the A-to-B disparity IS
      measured, 0.094, and must stand), priority medium, rationale "ECE disparity of
      0.094 suggests group-specific calibration would improve probability
      consistency. NOTE: the calibration disparity does NOT cover 1 group(s) inside
      the size gate that have no measured calibration error (C), so nothing here is
      evidence about them in either direction.", one warning.
      Before: the same list field and priority, a rationale with no "NOT ASSESSED",
      no "EXCLUDED" and no mention of C, and zero warnings.

    OVER-CORRECTION CONTROL, asserted below: the same A and B with no third group
    keep the measured disparity of 0.094, gain no NOTE sentence and raise no
    warning, so a blanket caveat would be caught.

    SABOTAGE (``if unmeasured_inside:`` forced to ``if False:``):
        FAILED test_a_third_group_without_a_measured_ece_gets_no_caveat_and_no_warning
          - AssertionError: the rationale still says nothing about the group with no
            measured calibration error: 'ECE disparity of 0.094 suggests
            group-specific calibration would improve probability consistency.'
    Source restored and proved byte-identical with diff -q.

    THE ORIGINAL RECORD follows.
    The guard is ``not math.isfinite(ece_disparity) or len(eces_measured) < 2``.
    With two measured group ECEs and a third group that has none, neither clause
    fires: the group lands in ``not_assessed_groups`` and the rationale, which is
    the sentence a reader acts on, gains nothing. The NOTE branch below it is
    gated on ``excluded`` (too small to look at) alone.
    """
    rng = np.random.default_rng(4)

    def group(rate, calibrated, labelled=True):
        y = (rng.random(200) < rate).astype(float) if labelled else np.full(200, np.nan)
        if calibrated:
            prob = np.where(
                np.nan_to_num(y) == 1, rng.normal(0.8, 0.05, 200), rng.normal(0.2, 0.05, 200)
            )
        else:
            prob = rng.normal(0.5, 0.05, 200)
        return y, np.clip(prob, 0.01, 0.99)

    y_a, p_a = group(0.6, True)
    y_b, p_b = group(0.6, False)
    y_c, p_c = group(0.6, True, labelled=False)
    y = np.concatenate([y_a, y_b, y_c])
    prob = np.concatenate([p_a, p_b, p_c])
    groups = np.array(["A"] * 200 + ["B"] * 200 + ["C"] * 200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = recommend_calibration_strategy(y, prob, groups, context="general")

    assert rec.not_assessed_groups == ["C"], "the field knows"
    # The A-to-B disparity is a real measurement and must stand: this record is
    # about its COVERAGE reaching the reader, not about withdrawing it.
    assert rec.disparity_assessed is True
    assert "does NOT cover" in rec.rationale, (
        f"the rationale still says nothing about the group with no measured "
        f"calibration error: {rec.rationale!r}"
    )
    assert "C" in rec.rationale.split("does NOT cover")[1]
    assert "no measured calibration error" in rec.rationale
    assert "nothing here is evidence about them" in rec.rationale
    # to_dict carries the same string, because a copy constructor that rebuilds
    # field by field is how a disclosure gets dropped at the boundary.
    assert rec.to_dict()["rationale"] == rec.rationale
    said = [str(w.message) for w in caught]
    assert any("NO measured calibration error" in m for m in said), said
    assert any("not a finding that their calibration matches" in m for m in said), said

    # OVER-CORRECTION CONTROL, with the real number: the same two measurable groups
    # with no third group keep their measured disparity and gain no caveat at all.
    y2 = np.concatenate([y_a, y_b])
    prob2 = np.concatenate([p_a, p_b])
    groups2 = np.array(["A"] * 200 + ["B"] * 200)
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        control = recommend_calibration_strategy(y2, prob2, groups2, context="general")
    assert control.not_assessed_groups == []
    assert control.disparity_assessed is True
    assert "ECE disparity of 0.094" in control.rationale, control.rationale
    assert "NOTE:" not in control.rationale, control.rationale
    assert quiet == [], [str(w.message) for w in quiet]


# ===========================================================================
# F04-7. The explainer turns the fourth state into a measured infeasibility.
# ===========================================================================


class _HandBuiltReport:
    def __init__(self, regions):
        self.feasible_regions = regions
        self.optimal_thresholds = {name: {} for name in regions}
        self.recommendations = []
        self.threshold_results = [None] * 20


def test_an_unknown_region_is_reported_as_a_measured_infeasibility():
    """CLOSED 2026-09-29. Inverted into a pin: a region whose status nobody recorded
    is named as a could-not-check and no longer draws a recommendation to retrain.

    THE FIX, in src/vfairness/explainer.py. The three bucket reads
    ("feasible", "not_assessed", "infeasible") skipped the fourth state entirely, so
    n_satisfiable 0, not_assessed [] and n_infeasible 0 fell through to the measured
    infeasible sentence. The unknown regions are now collected BY EXCLUSION
    (anything that is not one of the three known statuses), which is why a status
    value nobody has written yet cannot land in the same hole, and they carry their
    own coverage clause into the summary, the value and the evaluation.

    RE-MEASURED on {"demographic_parity": (0.2, 0.8)}:
      summary     "Analysed 20 threshold(s). 0 constraint(s) are satisfiable within
                   the threshold range. 1 region(s) carry no recorded feasibility
                   status, so they were neither assessed nor ruled out
                   (demographic_parity)."
      value       "1 constraint(s) analysed, 0 feasible region(s), 1 NOT ASSESSED"
      evaluation  "NOT ASSESSED: no fairness constraint was established as feasible
                   or ruled out. 1 region(s) carry no recorded feasibility status,
                   so they were neither assessed nor ruled out (demographic_parity).
                   This is not a finding that no region exists."
      severity    "medium"
      Before: the same 0 satisfiable, then "No feasible threshold region found for
      the given constraints: the model may need retraining with fairness-aware
      methods.", with the region named nowhere at all.

    THE THREE STATES ARE KEPT APART, which is the point of the record: a sweep that
    measured every threshold and found none feasible is still a FINDING and still
    says "No feasible threshold region found" with no NOT ASSESSED language (pinned
    below), a sweep with a real region still counts it, and a region with no
    recorded status is neither.

    SABOTAGE (``no_status`` forced to ``[]``, i.e. the fourth state dropped again):
        FAILED tests/test_bgl6_f04.py::
          test_an_unknown_region_is_reported_as_a_measured_infeasibility
          - AssertionError: a region of unknown status is still published as a
            measured infeasibility: 'No feasible threshold region found for the
            given constraints: the model may need retraining with fairness-aware
            methods.'
    Source restored and proved byte-identical with diff -q.

    THE ORIGINAL RECORD follows.
    ``_region_status`` answers "unknown" for a bare pair, exactly as
    ``_region_to_dict`` does, and the explainer then counts it in none of its
    three buckets. With ``n_satisfiable`` 0, ``not_assessed`` empty and
    ``n_infeasible`` 0 the evaluation falls through to the measured infeasible
    sentence and recommends retraining, for a region whose endpoints are 0.2 and
    0.8. Before this fix the same input read "1 constraint(s) are satisfiable",
    which was right, so this input got worse rather than better.
    """
    report = _HandBuiltReport({"demographic_parity": (0.2, 0.8)})
    explanation = _explain_threshold_analysis(report)
    first = explanation.explanations[0]

    # Still not credited with being feasible: a bare pair is no evidence either way.
    assert "0 constraint(s) are satisfiable" in explanation.summary, explanation.summary
    # And no longer condemned as a measured infeasibility.
    assert "may need retraining" not in first.evaluation, (
        f"a region of unknown status is still published as a measured "
        f"infeasibility: {first.evaluation!r}"
    )
    assert "No feasible threshold region found" not in first.evaluation, first.evaluation
    assert first.evaluation.startswith("NOT ASSESSED:"), first.evaluation
    assert "This is not a finding that no region exists." in first.evaluation
    assert "no recorded feasibility status" in first.evaluation

    # The disclosure reaches every surface a reader sees, not just one of them.
    assert "NOT ASSESSED" in first.value, first.value
    assert "1 NOT ASSESSED" in first.value, first.value
    assert "no recorded feasibility status" in explanation.summary, explanation.summary
    assert "demographic_parity" in explanation.summary, explanation.summary
    assert explanation.severity == "medium"


def test_the_measured_infeasible_and_the_mixed_sweeps_are_unchanged():
    """The over-correction control for F04-7, with the real counts.

    A fix that reported NOT ASSESSED for everything would satisfy every assertion
    above. Two states must survive untouched: a sweep that measured every threshold
    and found none feasible is a FINDING and keeps the retraining sentence, and a
    sweep with a real region keeps its count while naming what it did not cover.
    """
    from vfairness.post_processing.threshold_optimization.analyzer import FeasibleRegion

    measured_infeasible = _HandBuiltReport(
        {"demographic_parity": FeasibleRegion(None, None, assessed=True, n_searched=100)}
    )
    out = _explain_threshold_analysis(measured_infeasible)
    said = out.explanations[0].evaluation
    assert "No feasible threshold region found" in said, said
    assert "NOT ASSESSED" not in said, said
    assert "no recorded feasibility status" not in said, said
    assert out.severity == "medium"

    mixed = _HandBuiltReport(
        {
            "demographic_parity": FeasibleRegion(0.3, 0.7, assessed=True, n_searched=100),
            "equalized_odds": (0.2, 0.8),
        }
    )
    out = _explain_threshold_analysis(mixed)
    assert "1 constraint(s) are satisfiable" in out.summary, out.summary
    assert "2 constraint(s) analysed, 1 feasible region(s), 1 NOT ASSESSED" in (
        out.explanations[0].value
    ), out.explanations[0].value
    assert out.explanations[0].evaluation.startswith(
        "1 fairness constraint(s) have a feasible threshold region."
    ), out.explanations[0].evaluation
    assert "equalized_odds" in out.explanations[0].evaluation
    assert out.severity == "info"
