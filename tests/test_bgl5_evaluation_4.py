"""BGL5 batch A-evaluation-4: the eleven overturns, closed and pinned.

An independent auditor attacked the eleven PROVEN/SEMI-PROVEN grades in batch
A-evaluation-4 and overturned every one of them, each with a command and its
output. This file is the other half of the fix: for each overturn, a pin that
goes RED when the defect is put back, and an OVER-CORRECTION CONTROL asserting
that healthy input still gets its real measured number, by value.

One mechanism runs through five of them, and it is worth naming once: a
degeneracy guard that tests for EXACT NON-EXISTENCE (is it finite, are there two
rows, is the threshold zero, is every score identical) while the real degenerate
case is an ARGSORT IDENTITY PERMUTATION, or an n so small that one arrangement IS
the identity. A threshold on n cannot make a permutation exist. The fixes test
the thing the measurement actually needs:

  * ``removal_curve_auc``: is there an ORDER of |attribution| at all, not just
    numbers (np.unique on the raw magnitudes).
  * ``global_importance``: did the shuffle actually REARRANGE the column, per
    repeat, per feature (value comparison, so a constant column falls out too).
  * ``_rankings_to_positions``: tied scores get the MID-RANK of the slots their
    block fills, so no position comes from the caller's row order.
  * ``multi_seed_adversarial_probe``: does the background carry any SPREAD for
    the perturbation cloud (np.ptp, never a variance test).
  * ``get_invalid_groups``: the vacuity bound is min_group_size <= 1, because
    every group is built from the levels that occur and so holds at least one
    row.

Every "before" number in the docstrings below was measured on this repo before
the fix, and every "after" number was measured after it. Written 2026-09-27.
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
from vfairness.evaluation.vfairness_metrics.ranking import (
    exposure_parity_ratio,
    get_ranking_group_metrics,
    normalized_discounted_kl_divergence,
)
from vfairness.evaluation.vfairness_metrics.report import print_report

_W = np.array([0.5, -0.25, 1.0, 0.75])
_X4 = np.array([1.0, 2.0, -1.0, 0.5])
_WEIGHTS3 = np.array([2.0, 0.5, 0.1])


def _margin(rows, weights=_W):
    return np.atleast_2d(np.asarray(rows, dtype=float)) @ weights


def _weighted_explainer():
    """A linear model dominated by its first feature, and 200 real rows."""
    X = np.random.default_rng(7).normal(0, 1, (200, 3))
    explainer = FeatureAttributionExplainer(
        predict=lambda A: np.asarray(A, dtype=float) @ _WEIGHTS3,
        feature_names=["f0", "f1", "f2"],
    )
    return explainer, X


def _imputing_explainer():
    """The canonical sklearn shape: a predict that FILLS missing values.

    SimpleImputer + LinearRegression, HistGradientBoosting or any wrapped API
    client answers a row carrying NaN instead of propagating it, which is how the
    auditor showed that two pinned "refusals" belonged to the fixture's model
    rather than to the unit.
    """
    X = np.random.default_rng(7).normal(0, 1, (200, 3))
    means = X.mean(axis=0)

    def predict(A):
        A = np.asarray(A, dtype=float)
        return np.where(np.isfinite(A), A, means) @ _WEIGHTS3

    return FeatureAttributionExplainer(predict=predict, feature_names=["f0", "f1", "f2"]), X


# ===========================================================================
# 1. GroupManager.get_invalid_groups: the vacuity bound was off by one.
# ===========================================================================


@pytest.mark.parametrize("threshold", [0, 1, 0.5])
def test_a_threshold_that_can_never_flag_a_group_is_disclosed(threshold):
    """OVERTURN CLOSED. Every group comes from ``np.unique``, so none can hold
    fewer than one row: at min_group_size=1 the check cannot fire for ANY data,
    which is the definition of vacuous the fix itself gives, and the bound tested
    ``<= 0``.

    Before, on ``['a'] * 40 + ['b'] * 2``: t=0 -> [] with the vacuity warning,
    t=1 -> [] with ZERO warnings, t=2 -> [] (nothing is below 2). After: t=1
    discloses like t=0. min_group_size=1 is passed by production call sites in
    post_processing/calibration, operations/cicd/gate and _statistics.
    """
    groups = np.array(["a"] * 40 + ["b"] * 2)
    with pytest.warns(UserWarning, match="no group can ever hold fewer than 1 row"):
        assert GroupManager(groups, min_group_size=threshold).get_invalid_groups() == []


def test_control_a_reachable_threshold_still_names_the_small_groups_in_silence():
    """OVER-CORRECTION CONTROL: 2 is the smallest threshold that can flag
    anything, and it must stay silent and still name ``b``."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert GroupManager(
            np.array(["a"] * 40 + ["b"]), min_group_size=2
        ).get_invalid_groups() == ["b"]
        assert GroupManager(
            np.array(["a"] * 40 + ["b"] * 29), min_group_size=30
        ).get_invalid_groups() == ["b"]
        assert (
            GroupManager(np.array(["a"] * 40 + ["b"] * 30), min_group_size=30).get_invalid_groups()
            == []
        )


# ===========================================================================
# 2. global_importance: two rows do not make a permutation exist.
# ===========================================================================


def test_a_shuffle_that_lands_on_the_original_order_measures_nothing():
    """OVERTURN CLOSED. The guard was ``if Xarr.shape[0] < 2``, so n=2 passed it,
    and a two-row column has exactly two arrangements: ``rng.shuffle`` returns the
    identity about half the time and that repeat's drop is 0.0 by construction.

    Before (model y = 2*x0 + 0.5*x1 + 0.1*x2, X = rng(7).normal(0, 1, (200, 3))):
        global_importance(X[:2], n_repeats=1, random_state=0)
            -> [('f0', 0.0), ('f1', 0.0), ('f2', 0.0)], ZERO warnings
        global_importance(X[:2], n_repeats=3, random_state=0)
            -> f0 0.0 (the DOMINANT feature, weight 2.0), f1 0.2511, f2 0.0239
    After: NaN for every feature no shuffle moved, plus a warning naming them.
    """
    explainer, X = _weighted_explainer()
    with pytest.warns(UserWarning, match="3 of 3 feature\\(s\\) were never actually permuted"):
        one = explainer.global_importance(X[:2], n_repeats=1, random_state=0)
    assert all(math.isnan(c.importance) for c in one.contributions)
    assert all(c.direction == "not_assessed" for c in one.contributions)
    assert any("never actually permuted" in note for note in one.notes)

    with pytest.warns(UserWarning, match="1 of 3 feature\\(s\\) were never actually permuted"):
        three = explainer.global_importance(X[:2], n_repeats=3, random_state=0)
    by_name = {c.feature: c.importance for c in three.contributions}
    assert math.isnan(by_name["f0"])  # the dominant feature, and 0.0 before
    assert by_name["f0"] != 0.0
    # The two that WERE permuted keep a measured number: this is not a blanket
    # refusal of the whole result.
    assert by_name["f1"] == pytest.approx(0.3767, abs=1e-4)
    assert by_name["f2"] == pytest.approx(0.0718, abs=1e-4)


def test_a_constant_column_is_not_a_feature_that_does_not_matter():
    """OVERTURN CLOSED, the same mechanism at full n: every arrangement of a
    constant column is value-identical, so no shuffle can move it.

    Before, f1 held at 4.2 over all 200 rows, n_repeats=3:
    [('f0', 2.2265), ('f1', 0.0), ('f2', 0.1037)] in SILENCE. After: f1 NaN with
    a warning, f0 and f2 unchanged measurements.
    """
    explainer, X = _weighted_explainer()
    Xc = X.copy()
    Xc[:, 1] = 4.2
    with pytest.warns(UserWarning, match="never actually permuted"):
        result = explainer.global_importance(Xc, n_repeats=3)
    by_name = {c.feature: c.importance for c in result.contributions}
    assert math.isnan(by_name["f1"])
    assert by_name["f0"] == pytest.approx(2.2362, abs=1e-4)
    assert by_name["f2"] == pytest.approx(0.0979, abs=1e-4)


def test_the_scored_path_carries_the_same_check_and_is_reached():
    """THE GUARD HAS TO SIT WHERE A CALLER ARRIVES, on BOTH arms.

    The label-free arm is the default call and was the one the auditor measured,
    but the labelled arm runs the same loop and the constant-target refusal above
    it does NOT catch a constant COLUMN. Executed here so the branch in
    ``_scored_permutation`` is proved reachable rather than assumed: a constant f1
    with a real target reports f1 NaN 'not_assessed' with signed_value None and a
    warning, while f0 keeps its measured 2.0278 and f2 its 0.0040. At n=2 with two
    distinct targets (so the constant-target branch does not fire either) all
    three come back NaN.
    """
    explainer, X = _weighted_explainer()
    Xc = X.copy()
    Xc[:, 1] = 4.2
    with pytest.warns(UserWarning, match="1 of 3 feature\\(s\\) were never actually permuted"):
        scored = explainer.global_importance(Xc, y=Xc @ _WEIGHTS3, n_repeats=3)
    by_name = {c.feature: c for c in scored.contributions}
    assert math.isnan(by_name["f1"].importance)
    assert by_name["f1"].direction == "not_assessed"
    assert by_name["f1"].signed_value is None
    assert by_name["f0"].importance == pytest.approx(2.0278, abs=1e-4)
    assert by_name["f2"].importance == pytest.approx(0.0040, abs=1e-4)
    assert any("never actually permuted" in note for note in scored.notes)

    with pytest.warns(UserWarning, match="3 of 3 feature\\(s\\) were never actually permuted"):
        two_rows = explainer.global_importance(
            X[:2], y=X[:2] @ _WEIGHTS3, n_repeats=1, random_state=0
        )
    assert all(math.isnan(c.importance) for c in two_rows.contributions)


def test_zero_repeats_is_refused_by_statement_on_both_paths():
    """OVERTURN CLOSED. ``n_repeats=0`` makes the loop the importance is averaged
    over EMPTY. Before: NaN with ``notes == []`` and the only disclosure numpy's
    own "Mean of empty slice" plus "invalid value encountered in scalar divide".
    After: the same NaN, stated, above the scored/label-free dispatch so both
    paths answer alike.
    """
    explainer, X = _weighted_explainer()
    y = np.asarray(X, dtype=float) @ _WEIGHTS3
    for kwargs in ({}, {"y": y}):
        with pytest.warns(UserWarning, match="not one shuffle was performed"):
            result = explainer.global_importance(X, n_repeats=0, **kwargs)
        assert all(math.isnan(c.importance) for c in result.contributions)
        assert result.notes and "n_repeats is 0" in result.notes[0]


def test_control_global_importance_still_ranks_a_real_dataset():
    """OVER-CORRECTION CONTROL with the actual numbers: 200 rows are untouched by
    the new per-repeat check, so the label-free path still reports f0 2.2362 >
    f1 0.4927 > f2 0.0979 and the scored path still signs f0 at 1.9425. Both
    silent under ``simplefilter('error')``, which is what proves no healthy call
    acquired a warning.
    """
    explainer, X = _weighted_explainer()
    y = np.asarray(X, dtype=float) @ _WEIGHTS3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        label_free = explainer.global_importance(X, n_repeats=3)
        scored = explainer.global_importance(X, y=y, n_repeats=3)
    imps = {c.feature: c.importance for c in label_free.contributions}
    assert imps["f0"] == pytest.approx(2.2362, abs=1e-4)
    assert imps["f1"] == pytest.approx(0.4927, abs=1e-4)
    assert imps["f2"] == pytest.approx(0.0979, abs=1e-4)
    top = scored.top(1)[0]
    assert top.feature == "f0"
    assert top.importance == pytest.approx(1.9425, abs=1e-4)
    assert top.direction == "increase"


# ===========================================================================
# 3. explain_decision: the unit never checked its background at all.
# ===========================================================================


def test_an_unusable_background_is_refused_by_the_unit_not_by_the_model():
    """OVERTURN CLOSED. ``_occlusion_local`` derives everything from
    ``np.median(background, axis=0)`` and checked nothing, so the pinned all-NaN
    refusal came from the FIXTURE's model propagating NaN. With an imputing
    predict the same unusable background answered confidently.

    Before (imputing predict, 50 rows of NaN):
        [('f0', 0.5031, 'increase'), ('f1', 0.1899, 'increase'),
         ('f2', 0.0204, 'decrease')], base_value -0.5481, notes [], and SILENT
        under warnings.simplefilter("error"); the real 200-row background gives
        0.4985 / 0.1746 / 0.0201, within 2 percent, so nothing distinguished a
        baseline formed from 200 rows from one formed from none. An empty (0, 3)
        background produced the identical output.
    After: every importance NaN, direction 'not_assessed', base_value NaN, a
    note and a warning. The individual's own prediction (0.1244) does not depend
    on the background and is still reported.
    """
    explainer, X = _imputing_explainer()
    for background in (np.full((50, 3), np.nan), np.empty((0, 3))):
        with pytest.warns(UserWarning, match="baseline this explanation is measured AGAINST"):
            result = explainer.explain_decision(X[0], background)
        assert all(math.isnan(c.importance) for c in result.contributions)
        assert all(c.direction == "not_assessed" for c in result.contributions)
        assert [c.signed_value for c in result.contributions] == [None, None, None]
        assert math.isnan(result.base_value)
        assert result.prediction == pytest.approx(0.1244, abs=1e-4)


def test_a_background_missing_one_column_withdraws_only_that_column():
    """OVERTURN CLOSED, the partial case: a baseline that exists for two of three
    features is not a reason to refuse the two. Measured after the fix, the same
    200-row background with column 1 set to NaN: f0 0.4985 'increase' and f2
    0.0201 'decrease' survive, f1 is NaN 'not_assessed', base_value is NaN
    because the baseline ROW does not exist.
    """
    explainer, X = _imputing_explainer()
    background = X.copy()
    background[:, 1] = np.nan
    with pytest.warns(UserWarning, match="does not exist for 1 of 3 feature"):
        result = explainer.explain_decision(X[0], background)
    by_name = {c.feature: c for c in result.contributions}
    assert math.isnan(by_name["f1"].importance)
    assert by_name["f1"].direction == "not_assessed"
    assert by_name["f0"].importance == pytest.approx(0.4985, abs=1e-4)
    assert by_name["f0"].direction == "increase"
    assert by_name["f2"].importance == pytest.approx(0.0201, abs=1e-4)
    assert math.isnan(result.base_value)


def test_control_a_real_background_still_explains_and_a_zero_shift_is_still_zero():
    """OVER-CORRECTION CONTROL, both halves.

    A real 200-row background still measures f0 0.4985 increase, f1 0.1746
    increase, f2 0.0201 decrease with base_value -0.528461, silently. And a
    background whose median EQUALS the row is the true-estimand case: the
    prediction shift really is 0.0 and 'neutral' is a measurement, so it must
    survive the new guard untouched.
    """
    explainer, X = _weighted_explainer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        real = explainer.explain_decision(X[0], X)
        same = explainer.explain_decision(X[0], np.tile(X[0], (20, 1)))
    assert [c.direction for c in real.contributions] == ["increase", "increase", "decrease"]
    assert [round(c.importance, 4) for c in real.contributions] == [0.4985, 0.1746, 0.0201]
    assert real.base_value == pytest.approx(-0.528461, abs=1e-6)
    assert [c.importance for c in same.contributions] == [0.0, 0.0, 0.0]
    assert [c.direction for c in same.contributions] == ["neutral", "neutral", "neutral"]


# ===========================================================================
# 4. shapley_matrix: a permutation budget below 1, and a reference that does
#    not exist.
# ===========================================================================


@pytest.mark.parametrize("budget", [-1, 0, -5])
def test_a_permutation_budget_below_one_publishes_nothing(budget):
    """OVERTURN CLOSED. ``range(int(-1))`` is empty, so the sampling loop never
    runs and ``phi /= -1.0`` leaves -0.0 everywhere.

    Before, X = rng(7).normal(0, 1, (200, 3)) against predict = X @ [2, 0.5, 0.1],
    n_permutations=-1: values [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]],
    base_value -0.5481, "predictions" [-0.5480792970402042,
    -0.5480792970402042] (the base value twice, while the real predictions are
    0.1244 and -2.1077), ZERO warnings under simplefilter("error"). The
    completeness axiom appeared to hold because both sides were built from the
    same zeros. n_permutations=0 returned NaN through numpy's "invalid value
    encountered in divide" and nothing else.
    After: NaN values, NaN base_value, NaN predictions, a note and a warning.
    """
    explainer, X = _weighted_explainer()
    with pytest.warns(UserWarning, match="not one ordering was sampled"):
        out = explainer.shapley_matrix(X[:2], background=X, n_permutations=budget)
    assert all(math.isnan(v) for row in out["values"] for v in row)
    assert not any(v == 0.0 for row in out["values"] for v in row)
    assert math.isnan(out["base_value"])
    assert all(math.isnan(p) for p in out["predictions"])
    assert out["notes"] and "n_permutations is" in out["notes"][0]


def test_a_background_that_cannot_form_a_reference_is_refused_by_the_unit():
    """OVERTURN CLOSED. Before, with an imputing predict,
    shapley_matrix(X[:2], background=np.empty((0, 3)), n_permutations=5) returned
    base_value 0.0 and finite values [[0.0025, 0.1494, -0.0274], [-1.7812,
    -0.2273, -0.0992]]: the all-NaN refusal that was pinned for an empty
    background belonged to the test's model. After: NaN throughout, with the
    reason stated, and numpy's own "Mean of empty slice" no longer the only
    disclosure (the reference is not even computed).
    """
    explainer, X = _imputing_explainer()
    with pytest.warns(UserWarning, match="that reference does not exist"):
        empty = explainer.shapley_matrix(X[:2], background=np.empty((0, 3)), n_permutations=5)
    assert math.isnan(empty["base_value"])
    assert all(math.isnan(v) for row in empty["values"] for v in row)
    with pytest.warns(UserWarning, match="that reference does not exist"):
        all_nan = explainer.shapley_matrix(
            X[:2], background=np.full((5, 3), np.nan), n_permutations=5
        )
    assert all(math.isnan(v) for row in all_nan["values"] for v in row)


def test_control_one_unusable_reference_column_does_not_withdraw_the_others():
    """OVER-CORRECTION CONTROL, and it is here because a test in the suite caught
    the first version of this fix refusing too much.

    A 400-row dataset whose 4th feature is all-NaN, decomposed against a model
    that never reads it (the fixture from tests/test_bgl_stage2_s2g01.py
    ::test_a_constant_and_an_all_nan_feature_are_not_cleared_of_being_proxies):
    refusing the whole matrix over that one column destroyed three genuine
    measurements, including the real proxy feature, and broke a pin whose subject
    is that the proxy is still detected. So a PARTIAL reference is disclosed and
    the numbers stand: base_value 0.4968, the proxy's contribution -0.2968 for the
    first row, finite predictions, and one warning naming nan_feat. Only a
    reference that does not exist at all is refused.
    """
    rng = np.random.default_rng(7)
    n = 400
    protected = np.array(["a"] * 200 + ["b"] * 200)
    X = np.column_stack(
        [
            (protected == "b").astype(float) + rng.normal(0, 0.05, n),
            rng.normal(0, 1, n),
            np.full(n, 3.0),
            np.full(n, np.nan),
        ]
    )
    explainer = FeatureAttributionExplainer(
        predict=lambda Z: np.clip(0.2 + 0.6 * np.asarray(Z, dtype=float)[:, 0], 0, 1),
        feature_names=["proxy", "neutral", "const", "nan_feat"],
    )
    with pytest.warns(UserWarning, match="not a finite number for 1 of 4 feature"):
        out = explainer.shapley_matrix(X[:3], background=X, n_permutations=5)
    assert out["base_value"] == pytest.approx(0.4968, abs=1e-4)
    assert out["values"][0][0] == pytest.approx(-0.2968, abs=1e-4)
    assert all(np.isfinite(p) for p in out["predictions"])
    assert not any(math.isnan(v) for row in out["values"] for v in row)


def test_control_shapley_matrix_still_closes_on_the_prediction():
    """OVER-CORRECTION CONTROL: 3 rows against the 200-row background at
    n_permutations=8 still give base_value -0.5481 and contributions that sum to
    each row's real prediction to 1e-9 (checked against the model's closed form
    X @ [2.0, 0.5, 0.1]), with an empty ``notes`` and no warning at all.
    """
    explainer, X = _weighted_explainer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = explainer.shapley_matrix(X[:3], background=X, n_permutations=8)
    assert out["base_value"] == pytest.approx(-0.5481, abs=1e-4)
    assert out["notes"] == []
    for i, row in enumerate(out["values"]):
        assert out["base_value"] + sum(row) == pytest.approx(float(X[i] @ _WEIGHTS3), abs=1e-9)


# ===========================================================================
# 5. removal_curve_auc: a TIED attribution vector defines no order either.
# ===========================================================================


@pytest.mark.parametrize("tied", [0.0, 1.0, -3.0])
def test_a_tied_attribution_vector_has_no_order_to_mask_by(tied):
    """OVERTURN CLOSED. ``np.argsort(-np.abs(attr))`` returns the IDENTITY
    permutation for a CONSTANT vector exactly as it does for an all-NaN one, and
    the old guard tested finiteness only.

    Before, model w = [0.5, -0.25, 1.0, 0.75], x = [1, 2, -1, 0.5], zero
    background: [0.0]*4 -> 0.725, [1.0]*4 -> 0.725, [-3.0]*4 -> 0.725, all with
    ZERO warnings, and 0.725 grades "strong" in ``_grade_faithfulness``. It was a
    function of the ARRANGEMENT and nothing else: permuting the columns to
    [3, 2, 1, 0] (with x permuted alongside, so the data is identical) gave
    0.975. After: NaN with a warning naming the tie, the same answer this
    package's own ``ranking._undefined_order_reason`` gives.
    """
    with pytest.warns(UserWarning, match="all 4 \\|attribution\\| values are identical"):
        auc = removal_curve_auc(_margin, _X4, np.full(4, tied), np.zeros(4))
    assert math.isnan(auc)
    assert auc != 0.725


def test_the_tied_score_can_no_longer_move_with_the_column_order():
    """OVERTURN CLOSED, part two: the two arrangements that measured 0.725 and
    0.975 now both refuse, so the number cannot flip with the row order because
    there is no number.
    """
    perm = [3, 2, 1, 0]
    with pytest.warns(UserWarning, match="no order of \\|attribution\\|"):
        forward = removal_curve_auc(_margin, _X4, np.ones(4), np.zeros(4))
    with pytest.warns(UserWarning, match="no order of \\|attribution\\|"):
        reversed_columns = removal_curve_auc(
            lambda rows: _margin(rows, _W[perm]), _X4[perm], np.ones(4), np.zeros(4)
        )
    assert math.isnan(forward) and math.isnan(reversed_columns)


def test_control_removal_curve_auc_still_scores_every_vector_that_has_an_order():
    """OVER-CORRECTION CONTROL, four ways, all silent.

    The true attributions of the linear model still score 1.0 (they tie two
    features at |0.5| and must NOT be refused: one distinct magnitude is an
    order); the 0.25/0.25 model still scores exactly 0.5; a model that ignores
    its input still returns a MEASURED 0.0 on an untied vector; and a
    single-feature explanation has exactly one arrangement, which is not a
    missing order, so it still scores 0.5.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert removal_curve_auc(_margin, _X4, _W * _X4, np.zeros(4)) == 1.0
        assert (
            removal_curve_auc(
                lambda arr: np.array([0.25 * arr.ravel()[0] + 0.25 * arr.ravel()[1]]),
                np.array([1.0, 1.0]),
                np.array([2.0, 1.0]),
                np.zeros(2),
            )
            == 0.5
        )
        measured_zero = removal_curve_auc(
            lambda arr: np.array([0.5]), np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.zeros(2)
        )
        single = removal_curve_auc(
            lambda arr: np.atleast_2d(np.asarray(arr, dtype=float)) @ np.array([2.0]),
            np.array([1.0]),
            np.array([3.0]),
            np.zeros(1),
        )
    assert measured_zero == 0.0 and not math.isnan(measured_zero)
    assert single == 0.5
    # A partial tie still scores: one distinct magnitude orders what it separates.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert removal_curve_auc(_margin, _X4, np.array([1.0, 1.0, 0.2, 0.1]), np.zeros(4)) == 0.725


# ===========================================================================
# 6. multi_seed_adversarial_probe: a background with no spread makes every gap
#    0.0 by construction, and its consumer published that as a verdict.
# ===========================================================================


def _scaffolded_model(x):
    """Slack et al. scaffolding, as blatant as it gets: 0.95 off the manifold,
    0.05 on it. Exactly what the probe exists to flag."""

    def predict(rows):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        p = np.where(np.linalg.norm(rows - x, axis=1) > 0.5, 0.95, 0.05)
        return np.column_stack([1.0 - p, p])

    return predict


def _flat_model(rows):
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    return np.column_stack([np.full(len(rows), 0.7), np.full(len(rows), 0.3)])


def test_a_background_with_no_spread_cannot_clear_a_scaffolded_model():
    """OVERTURN CLOSED. ``sigma = background.std(axis=0) + 1e-9`` is 1e-9 when
    the background has no spread, so ``rng.normal(loc=x, scale=sigma)`` returns
    copies of x, x's nearest neighbours ARE x, and the gap is exactly 0.0 for ANY
    model: the check could not fire for any data.

    Before, x = [1, 2], background = np.tile(x, (30, 1)), n_perturbations=200,
    n_seeds=6, under simplefilter("error"): the SCAFFOLDED model returned
    flag=False, confidence=1.0, mean_gap=0.0, fired_fraction=0.0, reason='',
    0 warnings, byte-identical to the clean model's answer. The pair
    (flag False, confidence 1.0) is what the function's own comment names as the
    thing that must never be produced. With a ONE-row background the same
    fabrication ran the other way: flag=True, confidence=1.0, mean_gap=0.9.
    After: flag None, NaN statistics, a COULD NOT CHECK reason, a warning.
    """
    x = np.array([1.0, 2.0])
    for background in (np.tile(x, (30, 1)), x.reshape(1, -1)):
        for model in (_scaffolded_model(x), _flat_model):
            with pytest.warns(UserWarning, match="could perturb x away from the background"):
                result = multi_seed_adversarial_probe(
                    model, x, background, n_perturbations=200, n_seeds=6
                )
            assert result.flag is None
            assert result.flag is not False
            assert math.isnan(result.confidence)
            assert math.isnan(result.mean_gap)
            assert result.reason.startswith("COULD NOT CHECK")
            # Asserted on the SUBSTANCE, not on a quoted phrase. This read
            # `"peak-to-peak spread" in result.not_run_because` and went red later
            # the same day when the guard was tightened from `spread > 0.0` to a
            # floor relative to each column's own magnitude, because the sentence
            # no longer says "peak-to-peak". The behaviour it is about did not
            # change: the probe still refuses and still says why. A pin located by
            # quoting prose is a pin on the prose.
            #
            # The tightening was not cosmetic. The exact test cleared a background
            # one part in 1e12 from constant, whose per-column spread measures
            # 4.0e-12 against magnitudes of order 1, so a scaffolded model still
            # got the clean verdict. The guard in xai/diagnostics/adversarial.py
            # had already been written the stricter way, and the two disagreed.
            because = result.not_run_because
            assert "spread" in because, because
            assert "floor" in because, because
            assert "0.0 for every model" in because, because


def test_the_consumer_publishes_the_refusal_and_states_the_real_cause():
    """OVERTURN CLOSED for ``diagnose_local_attribution``.

    ``_MIN_PROBE_BACKGROUND`` counts ROWS, so 30 identical rows cleared the gate
    that exists to say "there is nothing to compare against". Before, on the
    scaffolded model over that background:
        {'adversarial_flag': False, 'adversarial_confidence': 1.0,
         'adversarial_reason': None, 'notes': []}, 0 warnings
    which is the verdict "the probe ran and found no scaffolding" for a model
    whose prediction jumps from 0.05 to 0.95 off the manifold.

    After: flag None, confidence None, and the reason is the PROBE's own, naming
    the spread. That second half matters on its own: this consumer used to write
    "none of its N seed(s) produced a finite gap" for every refusal, and here
    every seed DID produce a finite gap of exactly 0.0, so the old sentence would
    have been a cause nobody measured.
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
    assert "spread" in out["adversarial_reason"]
    assert "seed(s) produced a finite gap" not in out["adversarial_reason"]
    assert any("adversarial probe did not run" in note for note in out["notes"])


def test_control_a_real_background_still_reaches_a_verdict_both_ways():
    """OVER-CORRECTION CONTROL. A 30-row background with real spread still gets a
    MEASURED False (the probe ran and cleared the model): confidence 1.0,
    mean_gap 0.0, reason '', and through the consumer flag False with
    confidence 1.0, reason None, stability 0.0075 and no notes, all silent. The
    all-NaN-gap refusal keeps its own seed-counting reason, unchanged.
    """
    x = np.array([1.0, 2.0])
    background = np.random.default_rng(0).normal(size=(30, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probe = multi_seed_adversarial_probe(_flat_model, x, background, n_seeds=4)
        out = diagnose_local_attribution(
            _flat_model,
            [1.0, 2.0],
            [0.5, 0.25],
            np.random.default_rng(2).normal(size=(30, 2)),
            reruns=[[0.5, 0.25], [0.52, 0.24]],
            n_seeds=3,
        )
    assert probe.flag is False
    assert probe.confidence == 1.0
    assert probe.mean_gap == 0.0
    assert probe.reason == ""
    assert probe.not_run_because == ""
    assert out["adversarial_flag"] is False
    assert out["adversarial_confidence"] == 1.0
    assert out["adversarial_reason"] is None
    assert out["stability"] == 0.0075
    assert out["notes"] == []

    def nan_model(rows):
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        return np.full((len(rows), 2), np.nan)

    with pytest.warns(UserWarning, match="only 0 of 4 seed"):
        outage = multi_seed_adversarial_probe(nan_model, x, background, n_seeds=4)
    assert outage.flag is None
    assert "produced a finite gap" in outage.not_run_because


# ===========================================================================
# 7. generate_structured_findings: the refusal entry blamed a gate that had not
#    fired.
# ===========================================================================


def _analysis(sizes):
    rng = np.random.default_rng(3)
    sens, labels = [], []
    for name, n in sizes.items():
        sens += [name] * n
        labels.append(rng.binomial(1, 0.5, n))
    y = np.concatenate(labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return identify_privileged_groups(y, y.copy(), np.array(sens), min_group_size=30)


def test_the_refusal_entry_states_the_cause_that_was_measured():
    """OVERTURN CLOSED. ``all_groups`` is empty whenever fewer than TWO cells are
    analysable, not only when none is, so the entry was also the answer when a
    cell met the gate, and it blamed the gate anyway.

    Before, one 100-row cell at min_group_size=30 (data_treatment
    n_total_cells_seen 1, n_cells_included 1, n_cells_excluded_small 0):
        description "COULD NOT CHECK: none of the 1 cell(s) seen met the minimum
        group size..." with metric_values {'n_cells_analysed': 0, ...}
    which contradicted its own metric_values in the same dict. With a partial drop
    (A 100, B 10, C 10) it said "none of the 3 cell(s) seen met the minimum group
    size" while n_cells_excluded_small was 2.
    After: the cause comes from n_cells_included, which the function already reads
    the same dict for, and n_cells_analysed reports the measured count.
    """
    solo = _analysis({"A": 100})
    assert solo["all_groups"] == []
    assert solo["data_treatment"]["n_cells_included"] == 1
    with pytest.warns(UserWarning, match="only 1 of the 1 cell"):
        entry = generate_structured_findings(solo)[0]
    assert entry["type"] == "no_cells_analysed"
    assert entry["description"].startswith("COULD NOT CHECK")
    assert "a disparity is a COMPARISON" in entry["description"]
    assert "none of the 1 cell(s) seen met the minimum group size" not in entry["description"]
    assert entry["metric_values"]["n_cells_analysed"] == 1
    assert entry["metric_values"]["n_cells_excluded_small"] == 0
    assert entry["p_value"] is None and entry["statistically_significant"] is None

    partial = _analysis({"A": 100, "B": 10, "C": 10})
    with pytest.warns(UserWarning, match="only 1 of the 3 cell"):
        partial_entry = generate_structured_findings(partial)[0]
    assert partial_entry["metric_values"]["n_cells_analysed"] == 1
    assert partial_entry["metric_values"]["n_cells_excluded_small"] == 2
    assert sorted(partial_entry["groups"]) == ["B", "C"]


def test_control_the_size_gate_cause_survives_where_it_is_true():
    """OVER-CORRECTION CONTROL for the cause itself: with two 10-row cells at
    min_group_size=30 NOTHING met the gate, so the original sentence is the
    correct one and must still be produced, with n_cells_analysed 0 and both
    cells named. A hand-built analysis that carries no inventory says THAT,
    rather than blaming a gate nobody recorded.
    """
    nothing = _analysis({"A": 10, "B": 10})
    with pytest.warns(UserWarning, match="not one cell met the size gate"):
        entry = generate_structured_findings(nothing)[0]
    assert entry["metric_values"]["n_cells_analysed"] == 0
    assert entry["metric_values"]["n_cells_excluded_small"] == 2
    assert sorted(entry["groups"]) == ["A", "B"]
    assert "none of the 2 cell(s) seen met the minimum group size" in entry["description"]

    with pytest.warns(UserWarning, match="reports no cell inventory"):
        bare = generate_structured_findings({"all_groups": []})[0]
    assert "not known" in bare["description"]


def test_control_real_cells_still_produce_real_findings():
    """OVER-CORRECTION CONTROL: 200 rows in two 100-row cells (A predicted
    positive at 80 percent against B at 20 percent) still produce five findings
    and no ``no_cells_analysed`` entry.
    """
    rng = np.random.default_rng(3)
    sensitive = np.array(["A"] * 100 + ["B"] * 100)
    y_true = rng.binomial(1, 0.5, 200)
    y_pred = np.concatenate([rng.binomial(1, 0.8, 100), rng.binomial(1, 0.2, 100)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        findings = generate_structured_findings(
            identify_privileged_groups(y_true, y_pred, sensitive)
        )
    types = [f["type"] for f in findings]
    assert "no_cells_analysed" not in types
    assert len(findings) == 5
    assert any(t in ("over_prediction", "under_prediction") for t in types)


# ===========================================================================
# 8. get_group_rankings: a cell whose metric has an empty denominator was given
#    a rank number.
# ===========================================================================


def test_a_cell_whose_metric_was_never_measured_gets_no_rank():
    """OVERTURN CLOSED. A cell that PASSES the size gate but whose metric has an
    empty denominator was sorted to the end and numbered anyway.

    Before, A 60 rows with every label positive (no actual negatives, so no FPR
    exists) beside B 60 rows with a measured FPR, min_group_size=30, under
    simplefilter("error"):
        metric='fpr' -> [('B', 0.5, rank 1), ('A', nan, rank 2)], ZERO warnings,
        and no field on the row saying it was not measured. Rank 2 of 2 in a
        descending FPR ranking is the position a reader calls "least often
        wrongly flagged". metric='tpr' -> [('A', 1.0, 1), ('B', nan, 2)].
    After: rank None, an 'unmeasured' clause naming the empty denominator, and a
    warning. The sibling module calls the old behaviour the defect in so many
    words (ranking._undefined_order_reason).
    """
    sens = np.array(["A"] * 60 + ["B"] * 60)
    y_true = np.concatenate([np.ones(60, dtype=int), np.zeros(60, dtype=int)])
    y_pred = np.concatenate([np.ones(60, dtype=int), np.array([1, 0] * 30)])
    with pytest.warns(UserWarning, match="NO rank and a NaN fpr"):
        rankings = get_group_rankings(y_true, y_pred, sens, metric="fpr", min_group_size=30)
    assert [r["group"] for r in rankings] == ["B", "A"]
    assert rankings[0]["value"] == pytest.approx(0.5)
    assert rankings[0]["rank"] == 1
    assert math.isnan(rankings[1]["value"])
    assert rankings[1]["rank"] is None
    assert "empty denominator" in rankings[1]["unmeasured"]
    # The group is still in the list with its size: only the position is withheld.
    assert rankings[1]["size"] == 60

    with pytest.warns(UserWarning, match="NO rank and a NaN tpr"):
        by_tpr = get_group_rankings(y_true, y_pred, sens, metric="tpr", min_group_size=30)
    assert by_tpr[0]["group"] == "A" and by_tpr[0]["rank"] == 1
    assert by_tpr[1]["group"] == "B" and by_tpr[1]["rank"] is None


def test_control_a_complete_ranking_is_still_numbered_and_silent():
    """OVER-CORRECTION CONTROL with the actual values: two measured cells still
    come back ranked 1 and 2 with their real rates, silently, and the size-gate
    disclosure for a dropped group is unchanged.
    """
    sensitive = np.array(["A"] * 60 + ["B"] * 40)
    y_true = np.ones(100, dtype=int)
    y_pred = np.concatenate([np.ones(60, dtype=int), np.zeros(40, dtype=int)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rankings = get_group_rankings(y_true, y_pred, sensitive, min_group_size=30)
    assert [(r["group"], r["value"], r["rank"]) for r in rankings] == [
        ("A", 1.0, 1),
        ("B", 0.0, 2),
    ]
    assert all("unmeasured" not in r for r in rankings)

    with pytest.warns(UserWarning, match="ABSENT from the returned ranking"):
        dropped = get_group_rankings(
            np.ones(65, dtype=int),
            np.concatenate([np.ones(60, dtype=int), np.zeros(5, dtype=int)]),
            np.array(["A"] * 60 + ["B"] * 5),
            min_group_size=30,
        )
    assert [r["group"] for r in dropped] == ["A"]


# ===========================================================================
# 9. get_ranking_group_metrics: the tie guard was all or nothing.
# ===========================================================================


def test_one_untied_item_no_longer_hands_eleven_positions_to_the_row_order():
    """OVERTURN CLOSED. ``_undefined_order_reason`` refuses a score column only
    when EVERY score is identical, so ONE distinct score restored the
    unconditional stable argsort, and a stable argsort of a tied block is the
    caller's ROW ORDER.

    Before, scores = [0.9] + [0.5] * 11 with groups ['A']*6 + ['B']*6 at
    min_group_size=5, under simplefilter("error"):
        A-block first  -> avg_position A 2.500, B 8.500,
                          avg_exposure 0.5508 / 0.2980, ratio 0.5411 (a FAIL)
        interleaved    -> avg_position A 5.000, B 6.000,
                          avg_exposure 0.4667 / 0.3821, ratio 0.8187 (a PASS)
        0 warnings either way, on the same 12 (score, group) pairs.
    After: tied scores share the MID-RANK of the slots their block fills, so both
    arrangements measure avg_position A 5.0, B 6.0 and ratio 0.75. The metric is
    no longer a function of the order the rows arrived in.
    """
    scores = np.array([0.9] + [0.5] * 11)
    groups = np.array(["A"] * 6 + ["B"] * 6)
    order = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        blocked = get_ranking_group_metrics(scores, groups, min_group_size=5)
        interleaved = get_ranking_group_metrics(scores[order], groups[order], min_group_size=5)
        ratio_blocked = exposure_parity_ratio(scores, groups, min_group_size=5)
        ratio_interleaved = exposure_parity_ratio(scores[order], groups[order], min_group_size=5)
        ndkl_blocked = normalized_discounted_kl_divergence(scores, groups, min_group_size=5)
        ndkl_interleaved = normalized_discounted_kl_divergence(
            scores[order], groups[order], min_group_size=5
        )
    assert blocked == interleaved  # the arrangement cannot move any cell any more
    assert blocked["A"]["avg_position"] == 5.0
    assert blocked["B"]["avg_position"] == 6.0
    assert (blocked["A"]["avg_position"], blocked["B"]["avg_position"]) != (2.5, 8.5)
    assert ratio_blocked == ratio_interleaved == pytest.approx(0.75)
    assert ratio_blocked != pytest.approx(0.5410755479122771)
    assert ndkl_blocked == ndkl_interleaved == pytest.approx(0.136105, abs=1e-6)


def test_control_an_untied_ranking_measures_exactly_what_it_always_did():
    """OVER-CORRECTION CONTROL with the pre-fix numbers, which must not move: a
    real permutation of 12 gives A avg_position 2.5 against B 8.5, avg_exposure
    0.5507777176645691 against 0.2980123553632302, exposure_parity_ratio
    0.5410755479122771 and ndkl 0.48112668216872206 (all measured before the fix
    as well), silently. An untied float score column is likewise unchanged.
    """
    groups = np.array(["A"] * 6 + ["B"] * 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = get_ranking_group_metrics(np.arange(12), groups)
        ratio = exposure_parity_ratio(np.arange(12), groups)
        ndkl = normalized_discounted_kl_divergence(np.arange(12), groups)
        scores = np.random.default_rng(4).normal(size=12)
        untied = get_ranking_group_metrics(scores, groups)
        untied_ndkl = normalized_discounted_kl_divergence(scores, groups)
    assert out["A"]["avg_position"] == 2.5
    assert out["B"]["avg_position"] == 8.5
    assert out["A"]["avg_exposure"] == pytest.approx(0.5507777176645691, abs=1e-15)
    assert out["B"]["avg_exposure"] == pytest.approx(0.2980123553632302, abs=1e-15)
    assert out["A"]["min_position"] == 0.0 and out["A"]["max_position"] == 5.0
    assert ratio == pytest.approx(0.5410755479122771, abs=1e-15)
    assert ndkl == pytest.approx(0.48112668216872206, abs=1e-15)
    assert untied["A"]["avg_position"] == pytest.approx(5.833333333333333)
    assert untied_ndkl == pytest.approx(0.1484182414522164, abs=1e-15)


def test_control_an_all_tied_column_is_still_refused():
    """OVER-CORRECTION CONTROL in the other direction: mid-ranking must not turn
    the all-tied column into a measurement. Every item would share one position
    and every group would look identical, i.e. perfect parity over nothing, so
    the existing refusal has to stay in front of it.
    """
    groups = np.array(["A"] * 6 + ["B"] * 6)
    with pytest.warns(UserWarning, match="all 12 ranking scores are identical"):
        tied = get_ranking_group_metrics(np.full(12, 0.5), groups)
    assert math.isnan(tied["A"]["avg_position"])
    assert math.isnan(tied["B"]["avg_exposure"])
    assert tied["A"]["count"] == 6  # the group inventory WAS measured
    with pytest.warns(UserWarning, match="12 of 12 ranking scores are not finite"):
        unscored = get_ranking_group_metrics(np.full(12, np.nan), groups)
    assert math.isnan(unscored["B"]["avg_position"])


# ===========================================================================
# 10. print_report: an ABSENT fairness_score printed the worst verdict.
# ===========================================================================


def _printed(report):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_report(report)
    return buffer.getvalue()


def test_an_absent_fairness_score_is_a_could_not_check_not_zero_percent():
    """OVERTURN CLOSED. ``assessment.get("fairness_score", 0)`` substitutes the
    NUMBER 0 when the key is ABSENT, and 0 is a real value on that scale.

    Before:
        print_report({'task_type': 'classification',
                      'assessment': {'summary': 'no score was ever recorded'},
                      'metrics': {'demographic_parity_difference': nan},
                      'data_info': {}})
        printed "  Fairness Score: 0.0%", the WORST verdict there is, for a
        report carrying no score at all, while every other absent field in the
        same function printed "unknown" or "N/A". The rule is written out fifty
        lines above the line that broke it.
    After: "Fairness Score: N/A (could not check; this report carries no
    fairness_score field)".
    """
    printed = _printed(
        {
            "task_type": "classification",
            "assessment": {"summary": "no score was ever recorded"},
            "metrics": {"demographic_parity_difference": float("nan")},
            "data_info": {},
        }
    )
    assert "Fairness Score: 0.0%" not in printed
    assert "could not check" in printed
    assert "carries no fairness_score field" in printed


def test_control_print_report_still_prints_every_score_it_was_given():
    """OVER-CORRECTION CONTROL with the actual strings: a measured 0.0 is a real
    verdict and must still print as 0.0 percent, 1.0 as 100.0 percent, 0.75 as
    75.0 percent, and a PRESENT None keeps its own distinct could-not-check line
    (no metric was assessable), which is a different sentence from the absent
    key.
    """
    for score, expected in ((0.0, "0.0%"), (1.0, "100.0%"), (0.75, "75.0%")):
        printed = _printed(
            {
                "task_type": "classification",
                "assessment": {"fairness_score": score, "summary": "s"},
                "metrics": {},
                "data_info": {},
            }
        )
        assert f"Fairness Score: {expected}" in printed
        assert "could not check" not in printed
    present_none = _printed(
        {
            "task_type": "classification",
            "assessment": {"fairness_score": None, "summary": "NOT ASSESSABLE"},
            "metrics": {},
            "data_info": {},
        }
    )
    assert "no metric was assessable" in present_none
    assert "carries no fairness_score field" not in present_none


def test_the_per_metric_not_assessable_reasons_reach_the_reader():
    """The disclosure has to arrive at the surface a person reads. The assessment
    block carried ``not_assessable_metrics`` with a reason per metric and
    print_report rendered NONE of it, so the printed artifact said only "N
    metric(s) not assessable". Now each one is listed under a heading that says
    it is not a pass.
    """
    printed = _printed(
        {
            "task_type": "classification",
            "assessment": {
                "fairness_score": None,
                "summary": "NOT ASSESSABLE",
                "not_assessable_metrics": [
                    {
                        "metric": "demographic_parity_difference",
                        "value": float("nan"),
                        "status": "NOT_ASSESSABLE",
                        "reason": "1 valid group(s) after filtering (need at least 2)",
                    }
                ],
            },
            "metrics": {},
            "data_info": {},
        }
    )
    assert "Not Assessable (excluded from the score, NOT a pass):" in printed
    assert "demographic_parity_difference: 1 valid group(s) after filtering" in printed
