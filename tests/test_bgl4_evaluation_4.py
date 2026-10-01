"""BGL4 AUDIT of batch A-evaluation-4: the overturns, as executable evidence.

Every test here BEGAN as a CHARACTERISATION of the behaviour of the day, recorded
by an auditor, not a fix and not a requirement. Each one asserted what the code
did on an input class the graded pin never tried, and each docstring names the
honest answer a fix should produce.

INVERTED 2026-09-27 (BGL5, batch A-evaluation-4). All eleven overturns were
closed, so every assertion below now states the CORRECTED behaviour instead of
the defect, while the docstrings keep the measured before, so the record of what
was wrong survives the fix. The fuller pins, with the over-correction controls
that prove healthy input still gets its real number, are in
tests/test_bgl5_evaluation_4.py.

Audited 2026-09-27. No file under src/ was modified to produce the original
measurements.
"""

from __future__ import annotations

import io
import math
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics.attribution import FeatureAttributionExplainer
from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    diagnose_local_attribution,
    multi_seed_adversarial_probe,
    removal_curve_auc,
)
from vfairness.evaluation.vfairness_metrics.intersectional import (
    generate_structured_findings,
    get_group_rankings,
    identify_privileged_groups,
)
from vfairness.evaluation.vfairness_metrics.ranking import get_ranking_group_metrics
from vfairness.evaluation.vfairness_metrics.report import print_report

_W = np.array([0.5, -0.25, 1.0, 0.75])
_X = np.array([1.0, 2.0, -1.0, 0.5])


def _margin(rows, weights=_W):
    return np.atleast_2d(np.asarray(rows, dtype=float)) @ weights


# ===========================================================================
# 1. removal_curve_auc: the guard covers NON-FINITE attributions only, and an
#    ALL-TIED vector defines no order either.
# ===========================================================================


def test_overturn_tied_attributions_still_score_the_callers_column_order():
    """OVERTURN of removal_curve_auc PROVEN.

    ``np.argsort(-np.abs(attr))`` returns the IDENTITY permutation for a
    constant vector exactly as it does for an all-NaN one, so the curve masks
    features in the caller's own column order. The guard tests finiteness only.

    0.725 is the same number the guard's own comment records as the
    fabrication it was written to remove, and it grades "strong".

    Honest answer: NaN plus a warning naming the tie, which is what this
    package's own ``ranking._undefined_order_reason`` returns for "all N
    ranking scores are identical, so no ranking order is defined".

    CLOSED (BGL5): that is what it returns now. Inverted from ``== 0.725``.
    """
    with pytest.warns(UserWarning, match="all 4 \\|attribution\\| values are identical"):
        all_zero = removal_curve_auc(_margin, _X, np.zeros(4), np.zeros(4))
    with pytest.warns(UserWarning, match="all 4 \\|attribution\\| values are identical"):
        all_tied = removal_curve_auc(_margin, _X, np.ones(4), np.zeros(4))
    assert all_zero != 0.725
    assert all_tied != 0.725
    assert math.isnan(all_zero) and math.isnan(all_tied)


def test_overturn_the_tied_score_changes_with_the_column_order_alone():
    """OVERTURN evidence, part two: the number is a function of the ARRANGEMENT.

    Same model, same tied explanation, columns permuted (and x permuted with
    them, so the data is identical): 0.725 becomes 0.975. Nothing about the
    model or the explanation changed.

    CLOSED (BGL5): both arrangements refuse, so there is no number left to move
    with the column order. Inverted from ``== 0.725`` / ``== 0.975``.
    """
    perm = [3, 2, 1, 0]
    with pytest.warns(UserWarning, match="no order of \\|attribution\\|"):
        forward = removal_curve_auc(_margin, _X, np.ones(4), np.zeros(4))
    with pytest.warns(UserWarning, match="no order of \\|attribution\\|"):
        reversed_columns = removal_curve_auc(
            lambda rows: _margin(rows, _W[perm]), _X[perm], np.ones(4), np.zeros(4)
        )
    assert math.isnan(forward) and math.isnan(reversed_columns)
    assert forward != 0.725
    assert reversed_columns != 0.975


# ===========================================================================
# 2. global_importance: the guard demands 2 rows, and 2 rows do not make a
#    permutation exist.
# ===========================================================================


def _weighted_explainer():
    weights = np.array([2.0, 0.5, 0.1])
    X = np.random.default_rng(7).normal(0, 1, (200, 3))
    ex = FeatureAttributionExplainer(
        predict=lambda A: np.asarray(A, dtype=float) @ weights, feature_names=["f0", "f1", "f2"]
    )
    return ex, X


def test_overturn_two_rows_can_still_report_importance_exactly_zero():
    """OVERTURN of global_importance PROVEN.

    The guard is ``if Xarr.shape[0] < 2``, so n=2 is accepted. With two rows a
    column has exactly TWO arrangements and ``rng.shuffle`` lands on the
    IDENTITY half the time, so a repeat measures nothing and its drop is
    exactly 0.0. At n_repeats=1, random_state=0 every feature comes back 0.0,
    which is the same "no feature drove this model" the guard was written to
    remove at n=1; at n_repeats=3, random_state=0 the DOMINANT feature (weight
    2.0) reports 0.0 while the two weak ones report real numbers.

    The same campaign fixed this exact reasoning in llm/counterfactual.py on
    the same day: "a shuffle that lands on the ORIGINAL order tests nothing".

    Honest answer: count the permutations that were actually distinct, and
    report NaN for a feature whose column never moved.

    CLOSED (BGL5): ``_column_rearranged`` counts them per repeat and per feature,
    so a column no shuffle moved reports NaN. Inverted from ``== 0.0``.
    """
    ex, X = _weighted_explainer()
    with pytest.warns(UserWarning, match="never actually permuted"):
        one = ex.global_importance(X[:2], n_repeats=1, random_state=0)
    with pytest.warns(UserWarning, match="never actually permuted"):
        three = ex.global_importance(X[:2], n_repeats=3, random_state=0)
    assert all(math.isnan(c.importance) for c in one.contributions)
    assert [c.importance for c in one.contributions] != [0.0, 0.0, 0.0]
    by_name = {c.feature: c.importance for c in three.contributions}
    assert math.isnan(by_name["f0"])  # the dominant feature, 0.0 before
    assert by_name["f1"] > 0.0 and by_name["f2"] > 0.0  # these WERE permuted
    # And with 200 rows the same model gives f0 about 2.23, so 0.0 was not the truth.
    assert {c.feature: c.importance for c in ex.global_importance(X).contributions}["f0"] > 2.0


def test_overturn_zero_repeats_refuses_only_through_a_numpy_accident():
    """OVERTURN evidence for global_importance, the out-of-reach parameter.

    ``n_repeats=0`` makes the loop that the importance is averaged over empty.
    The value is NaN, which is honest, but the only disclosure is numpy's own
    "Mean of empty slice" RuntimeWarning, and on the scored path ``notes`` is
    empty. ``test_zero_rows_refuses_by_statement_not_by_numpy_accident`` in the
    graded pin names that exact failure mode for a different input.

    CLOSED (BGL5): the budget is checked above the scored/label-free dispatch, so
    the NaN now arrives with a statement instead of a RuntimeWarning. Inverted
    from ``notes == []`` and "no warning mentions n_repeats".
    """
    ex, X = _weighted_explainer()
    y = np.asarray(X, dtype=float) @ np.array([2.0, 0.5, 0.1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scored = ex.global_importance(X, y=y, n_repeats=0)
    assert all(math.isnan(c.importance) for c in scored.contributions)
    assert scored.notes and "n_repeats is 0" in scored.notes[0]
    assert any(w.category is UserWarning for w in caught)
    assert any("n_repeats" in str(w.message) for w in caught)


# ===========================================================================
# 3. explain_decision / shapley_matrix: the refusal belongs to the FIXTURE's
#    model, not to the unit.
# ===========================================================================


def _imputing_explainer():
    """The canonical sklearn shape: a pipeline that fills missing values.

    SimpleImputer + LinearRegression, HistGradientBoosting, or any wrapped API
    client answers a row carrying NaN instead of propagating it. The module
    docstring names "a fitted sklearn estimator" as a supported predict.
    """
    weights = np.array([2.0, 0.5, 0.1])
    X = np.random.default_rng(7).normal(0, 1, (200, 3))
    means = X.mean(axis=0)

    def predict(A):
        A = np.asarray(A, dtype=float)
        return np.where(np.isfinite(A), A, means) @ weights

    return FeatureAttributionExplainer(predict=predict, feature_names=["f0", "f1", "f2"]), X


def test_overturn_explain_decision_never_checks_its_background():
    """OVERTURN of explain_decision PROVEN.

    ``_occlusion_local`` contains no check on ``background``. The pinned NaN
    refusal is produced by the FIXTURE's model propagating NaN through a matrix
    multiply, so with an imputing model the same unusable background (50 rows of
    NaN, and separately an empty one) yields signed, directional, confident
    contributions that are indistinguishable from the real 200-row run.

    Honest answer: refuse when the background cannot form a baseline, in the
    unit, the way ``_MIN_PROBE_BACKGROUND`` does one module along.

    CLOSED (BGL5): ``explain_decision`` now checks the baseline it derives from
    the background, above the shap/occlusion dispatch. Inverted from "finite
    importances within 2 percent of the real run".
    """
    ex, X = _imputing_explainer()
    with pytest.warns(UserWarning, match="baseline this explanation is measured AGAINST"):
        nan_bg = ex.explain_decision(X[0], np.full((50, 3), np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the REAL background is still silent
        real = ex.explain_decision(X[0], X)
    assert all(math.isnan(c.importance) for c in nan_bg.contributions)
    assert [c.direction for c in nan_bg.contributions] == ["not_assessed"] * 3
    assert [c.direction for c in real.contributions] == ["increase", "increase", "decrease"]
    # No longer indistinguishable from the numbers measured over 200 real rows.
    for a, b in zip(nan_bg.contributions, real.contributions):
        assert math.isnan(a.importance) and not math.isnan(b.importance)
    assert math.isnan(nan_bg.base_value)


def test_overturn_shapley_matrix_with_a_non_positive_permutation_budget():
    """OVERTURN of shapley_matrix PROVEN, the out-of-reach parameter.

    ``n_permutations=-1`` skips the sampling loop entirely, ``phi /= -1.0``
    leaves every contribution exactly -0.0, and ``predictions`` (which the dict
    labels as predictions) comes back as the base value for every row. The
    completeness property the docstring advertises then appears to hold,
    because both sides were built from the same zeros. Silent.

    The model's real predictions for those two rows are 0.1244 and -2.1077.

    Honest answer: refuse a budget below 1, the way this batch's own
    ``get_invalid_groups`` fix refuses ``min_group_size <= 0``.

    CLOSED (BGL5): it refuses, and the faked completeness goes with it, because
    NaN values cannot be summed against a NaN base value to look like an axiom.
    Inverted from the all-(-0.0) matrix.
    """
    ex, X = _weighted_explainer()
    with pytest.warns(UserWarning, match="not one ordering was sampled"):
        out = ex.shapley_matrix(X[:2], background=X, n_permutations=-1)
    assert out["values"] != [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]]
    assert all(math.isnan(v) for row in out["values"] for v in row)
    assert all(math.isnan(p) for p in out["predictions"])
    assert math.isnan(out["base_value"])
    assert out["n_permutations"] == -1


def test_overturn_shapley_matrix_empty_background_refusal_is_the_models():
    """OVERTURN evidence, part two: the same empty background with an imputing
    model gives a finite base_value and finite contributions, disclosed by
    nothing but numpy's "Mean of empty slice".

    CLOSED (BGL5): the reference is checked before it is used, and it is not even
    computed for an empty background, so numpy's empty-slice warning is gone too.
    Inverted from "finite base_value and finite values, RuntimeWarning only"."""
    ex, X = _imputing_explainer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = ex.shapley_matrix(X[:2], background=np.empty((0, 3)), n_permutations=5)
    assert math.isnan(out["base_value"])
    assert all(math.isnan(v) for row in out["values"] for v in row)
    assert any(w.category is UserWarning for w in caught)
    assert any("reference does not exist" in str(w.message) for w in caught)


# ===========================================================================
# 4. multi_seed_adversarial_probe and its consumer: a background with no
#    spread collapses the perturbation cloud onto x.
# ===========================================================================


def _scaffolded_model(x):
    """Slack et al. scaffolding, as blatant as it gets: 0.95 off the manifold,
    0.05 on it. This is precisely what the probe exists to flag."""

    def predict(rows):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        p = np.where(np.linalg.norm(rows - x, axis=1) > 0.5, 0.95, 0.05)
        return np.column_stack([1.0 - p, p])

    return predict


def test_overturn_a_background_with_no_spread_clears_a_scaffolded_model():
    """OVERTURN of multi_seed_adversarial_probe PROVEN.

    ``sigma = background.std(axis=0) + 1e-9`` is 1e-9 when the background has
    no spread, so ``rng.normal(loc=x, scale=sigma)`` returns copies of x and
    the nearest neighbours are x as well: the gap is 0.0 by construction for
    ANY model, and the check cannot fire for any data. Thirty identical rows
    clear the consumer's row count while carrying nothing to compare against.

    The result is ``flag=False, confidence=1.0``, which the function's own
    comment names as the thing that must never be produced ("asserts the
    explainer is clean"), and it is byte-identical to the clean model's.

    Honest answer: refuse a background whose spread is zero, exactly as the
    all-NaN-gap branch refuses.

    CLOSED (BGL5): ``_background_cannot_be_perturbed`` asks the question with
    ``np.ptp`` above the seed loop. Inverted from ``flag is False`` with
    ``confidence == 1.0``.
    """
    x = np.array([1.0, 2.0])
    flat_background = np.tile(x, (30, 1))
    with pytest.warns(UserWarning, match="could perturb x away from the background"):
        scaffolded = multi_seed_adversarial_probe(
            _scaffolded_model(x), x, flat_background, n_perturbations=200, n_seeds=6
        )
    assert scaffolded.flag is None
    assert scaffolded.flag is not False
    assert math.isnan(scaffolded.confidence)
    assert math.isnan(scaffolded.mean_gap)
    assert math.isnan(scaffolded.fired_fraction)
    assert scaffolded.reason.startswith("COULD NOT CHECK")


def test_overturn_the_consumer_publishes_that_clearance_as_a_verdict():
    """OVERTURN of diagnose_local_attribution PROVEN, on a different input from
    the graded one.

    ``_MIN_PROBE_BACKGROUND`` counts ROWS, so 30 identical rows pass the gate
    that exists to say "there is nothing to compare against". The tri-state
    fix the grade covers is not reached, and the dict carries
    ``adversarial_flag False`` with ``adversarial_reason None`` and no note:
    the verdict "the probe ran and found no scaffolding", for a scaffolded
    model the probe could not have flagged.

    CLOSED (BGL5): the probe itself refuses this background now, and the consumer
    routes ``flag is None`` through ``_record_probe_not_run`` carrying the
    probe's OWN cause rather than its old guess about the seeds. Inverted from
    ``flag is False`` with an empty ``notes``.
    """
    x = np.array([1.0, 2.0])
    with pytest.warns(UserWarning, match="adversarial probe did not run"):
        out = diagnose_local_attribution(
            _scaffolded_model(x), [1.0, 2.0], [0.5, 0.25], np.tile(x, (30, 1)), n_seeds=6
        )
    assert out["adversarial_flag"] is None
    assert out["adversarial_flag"] is not False
    assert out["adversarial_confidence"] is None
    assert out["adversarial_reason"].startswith("COULD NOT CHECK")
    assert any("adversarial probe did not run" in note for note in out["notes"])


# ===========================================================================
# 5. generate_structured_findings: the new refusal entry states a reason that
#    was not measured.
# ===========================================================================


def _one_analysable_cell(sizes):
    rng = np.random.default_rng(3)
    sens, labels = [], []
    for name, n in sizes.items():
        sens += [name] * n
        labels.append(rng.binomial(1, 0.5, n))
    y = np.concatenate(labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return identify_privileged_groups(y, y.copy(), np.array(sens), min_group_size=30)


def test_overturn_the_refusal_entry_blames_a_size_gate_that_did_not_fire():
    """OVERTURN of generate_structured_findings PROVEN.

    ``all_groups`` is also empty when exactly ONE cell was analysable, because
    a disparity needs two. The new entry then says "none of the 1 cell(s) seen
    met the minimum group size" for a 100-row cell that met a gate of 30, and
    contradicts its own ``metric_values`` in the same dict
    (``n_cells_excluded_small: 0``). With a partial drop it says none of 3 met
    the gate when one did. ``data_treatment['n_cells_included']``, which the
    function already reads for ``n_total_cells_seen``, holds the true reason.

    The COULD NOT CHECK state itself does survive, so this is narrower than the
    defect it replaced. A disclosure that misdescribes its own cause is still a
    statement nobody measured: this batch's own ``get_group_rankings``
    judgement rejected a sibling helper for exactly that.

    Honest answer: distinguish "no cell met the gate" from "only one cell was
    analysable, so no comparison exists".

    CLOSED (BGL5): the cause is read from ``n_cells_included`` and
    ``n_cells_analysed`` reports it. Inverted from "none of the N cell(s) seen met
    the minimum group size" on both fixtures.
    """
    solo = _one_analysable_cell({"A": 100})
    assert solo["all_groups"] == []
    assert solo["data_treatment"]["n_cells_included"] == 1
    assert solo["data_treatment"]["n_cells_excluded_small"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entry = generate_structured_findings(solo)[0]
    assert entry["type"] == "no_cells_analysed"
    assert "none of the 1 cell(s) seen met the minimum group size" not in entry["description"]
    assert "only 1 of the 1 cell(s) seen met the minimum group size" in entry["description"]
    assert entry["metric_values"]["n_cells_excluded_small"] == 0  # so one DID meet it
    assert entry["metric_values"]["n_cells_analysed"] == 1  # and it is counted
    assert entry["groups"] == []

    partial = _one_analysable_cell({"A": 100, "B": 10, "C": 10})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        partial_entry = generate_structured_findings(partial)[0]
    assert (
        "none of the 3 cell(s) seen met the minimum group size" not in partial_entry["description"]
    )
    assert (
        "only 1 of the 3 cell(s) seen met the minimum group size" in (partial_entry["description"])
    )
    assert partial_entry["metric_values"]["n_cells_excluded_small"] == 2


# ===========================================================================
# 6. get_group_rankings: a group whose METRIC is undefined is still ranked.
# ===========================================================================


def test_overturn_a_group_with_an_undefined_metric_still_gets_a_rank():
    """OVERTURN of get_group_rankings PROVEN.

    The fix covers the size gate. A cell whose metric has an empty denominator
    (no actual negatives, so no FPR) is not dropped: it is given a ``value`` of
    NaN and a ``rank``, sorted to the end, in silence. Group A here has 60
    rows, every label positive, so its FPR was never measured, and it is
    returned as rank 2 of 2 in a descending FPR ranking, the position a reader
    calls "least often wrongly flagged".

    ``ranking._undefined_order_reason``, in the same package, calls this exact
    behaviour the defect: "an item whose score is missing is silently ranked
    LAST, a position nobody measured".

    Honest answer: no rank number for an unmeasured cell, and a warning naming
    it, the way the size-gate drop is named.

    CLOSED (BGL5): ranks number the measured cells only, and
    ``_warn_unmeasured_ranking_cells`` names the rest. Inverted from
    ``rank == 2`` and ``"unmeasured" not in`` the row.
    """
    sens = np.array(["A"] * 60 + ["B"] * 60)
    y_true = np.concatenate([np.ones(60, dtype=int), np.zeros(60, dtype=int)])
    y_pred = np.concatenate([np.ones(60, dtype=int), np.array([1, 0] * 30)])
    with pytest.warns(UserWarning, match="NO rank and a NaN fpr"):
        rankings = get_group_rankings(y_true, y_pred, sens, metric="fpr", min_group_size=30)
    assert [r["group"] for r in rankings] == ["B", "A"]
    assert rankings[0]["value"] == 0.5 and rankings[0]["rank"] == 1
    assert math.isnan(rankings[1]["value"])
    assert rankings[1]["rank"] is None  # no invented position for an unmeasured cell
    assert rankings[1]["rank"] != 2
    assert "unmeasured" in rankings[1]


# ===========================================================================
# 7. get_ranking_group_metrics: the tie guard is all or nothing.
# ===========================================================================


def test_overturn_one_untied_item_defeats_the_tie_guard():
    """OVERTURN of get_ranking_group_metrics PROVEN.

    ``_undefined_order_reason`` refuses a column only when ``np.unique(arr).size
    == 1``. Add ONE distinct score and 11 of 12 positions come straight from the
    caller's row order again. The same 12 (score, group) pairs, rearranged, give
    avg_position A 2.5 or A 5.0, in silence.

    These are the numbers the guard's own CRITICAL comment cites as the defect
    (exposure_parity_ratio 0.5411 FAIL against 0.8187 PASS on the same
    non-data).

    Honest answer: disclose the tied block, or give tied items their mid-rank.

    CLOSED (BGL5): tied items get their mid-rank, so the two arrangements measure
    the SAME thing (A 5.0, B 6.0) and neither of the old pair of numbers survives.
    Inverted from (2.5, 8.5) against (5.0, 6.0).
    """
    scores = np.array([0.9] + [0.5] * 11)
    groups = np.array(["A"] * 6 + ["B"] * 6)
    order = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the measurement itself is unchanged: silent
        blocked = get_ranking_group_metrics(scores, groups, min_group_size=5)
        interleaved = get_ranking_group_metrics(scores[order], groups[order], min_group_size=5)
    assert (blocked["A"]["avg_position"], blocked["B"]["avg_position"]) != (2.5, 8.5)
    assert (blocked["A"]["avg_position"], blocked["B"]["avg_position"]) == (5.0, 6.0)
    assert blocked == interleaved


# ===========================================================================
# 8. get_invalid_groups: the vacuity guard is off by one.
# ===========================================================================


def test_overturn_min_group_size_one_is_equally_vacuous_and_silent():
    """OVERTURN of GroupManager.get_invalid_groups PROVEN.

    Every group is built from ``np.unique``, so no group can ever hold fewer
    than one row, on the simple path or the intersectional one. At
    ``min_group_size=1`` the check therefore cannot fire for ANY data, which is
    the definition the fix itself gives, and the guard tests ``<= 0``, so
    nothing is said. ``min_group_size=1`` is passed by production call sites in
    post_processing/calibration, operations/cicd and _statistics.

    Honest answer: warn for ``min_group_size <= 1``.

    CLOSED (BGL5): the bound is ``<= 1`` and the message says why one row is the
    floor. Inverted from ``caught == []``.
    """
    groups = np.array(["a"] * 40 + ["b"] * 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vacuous = GroupManager(groups, min_group_size=1).get_invalid_groups()
    assert vacuous == []
    assert caught != []  # min_group_size=1 discloses exactly as 0 does
    assert any("vacuous" in str(w.message) for w in caught)
    with warnings.catch_warnings(record=True) as caught_zero:
        warnings.simplefilter("always")
        GroupManager(groups, min_group_size=0).get_invalid_groups()
    assert any("vacuous" in str(w.message) for w in caught_zero)
    # And 2 is the first threshold that can flag anything, so 1 is out of reach.
    assert GroupManager(np.array(["a"] * 40 + ["b"]), min_group_size=2).get_invalid_groups() == [
        "b"
    ]


# ===========================================================================
# 9. print_report: an absent fairness_score prints 0.0 percent.
# ===========================================================================


def test_overturn_an_absent_fairness_score_prints_the_worst_verdict():
    """OVERTURN of print_report SEMI-PROVEN.

    ``assessment.get("fairness_score", 0)`` substitutes the number 0 when the
    KEY IS ABSENT, and 0 is a real value on the scale, so a report carrying no
    score prints "Fairness Score: 0.0%", the worst verdict there is. Every
    other absent field in the same function prints "unknown" or "N/A", and the
    rule is spelled out fifty lines above: "An absent field prints 'unknown',
    never 0: claiming zero exclusions for a report that never recorded any is a
    false provenance statement, not a harmless default."

    Honest answer: the same "N/A (could not check)" line the None case gets.

    CLOSED (BGL5): an absent key is read with a sentinel, so it prints its own
    could-not-check line. Inverted from ``"Fairness Score: 0.0%" in printed``.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_report(
            {
                "task_type": "classification",
                "assessment": {"summary": "no score was ever recorded"},
                "metrics": {},
                "data_info": {},
            }
        )
    printed = buffer.getvalue()
    assert "Fairness Score: 0.0%" not in printed
    assert "could not check" in printed
    assert "carries no fairness_score field" in printed
    # The None case, for contrast, is handled correctly.
    buffer_none = io.StringIO()
    with redirect_stdout(buffer_none):
        print_report(
            {
                "task_type": "classification",
                "assessment": {"fairness_score": None, "summary": "NOT ASSESSABLE"},
                "metrics": {},
                "data_info": {},
            }
        )
    assert "could not check" in buffer_none.getvalue()
