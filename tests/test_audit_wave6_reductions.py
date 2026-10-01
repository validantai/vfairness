"""
Wave-6 audit pins: reductions unit (second-iteration audit 2026-08-22).

Findings pinned here (negative case first in each block, then a
does-not-overcorrect case):

    1. [CRITICAL] ExponentiatedGradient's cost-sensitive reduction was label-
       and sign-blind (one uniform weight per group), so the flagship
       in-processing algorithm could not enforce any constraint. Fixed per
       Agarwal et al. (2018), Sec. 3: signed per-coordinate multipliers enter
       the cost of predicting positive and the base learner trains on
       relabeled/reweighted samples.
    2. [HIGH] GridSearch swept ONE shared unsigned lambda, producing
       byte-identical classifiers for balanced groups. Fixed: signed
       per-group multipliers along the baseline disparity direction.
    3. [HIGH] find_feasible_thresholds (equalized odds) tested the TPR/FPR
       marginals independently, marking thresholds feasible that no partner
       threshold could jointly match. Fixed: pairwise joint (TPR, FPR)
       Chebyshev check against every other group's operating points.
    4. [HIGH] FairRegressor mean_parity used uniform per-group sample
       reweighting, a provable no-op on group prediction means, and crashed
       with negative sample weights for tolerance > 1. Fixed: post-fit
       per-group prediction offsets (band of half-width tolerance/2 around
       the pooled mean); reweighting removed.
"""

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LinearRegression, LogisticRegression

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.in_processing.constraints.base import (
    DemographicParityConstraint,
    EqualizedOddsConstraint,
)
from vfairness.in_processing.constraints.reductions import (
    ExponentiatedGradient,
    GridSearch,
    _cost_sensitive_relabel,
)
from vfairness.in_processing.wrappers.sklearn_wrappers import FairRegressor
from vfairness.post_processing.threshold_optimization.constraints import (
    find_feasible_thresholds,
)


def _reviewer_biased_data(n=2000, seed=7):
    """The audit reviewer's construction: group-shifted feature drives a
    large DP violation for a plain logistic regression."""
    r = np.random.default_rng(seed)
    g = (r.random(n) < 0.5).astype(int)
    x1 = r.normal(0, 1, n)
    x2 = r.normal(0, 1, n) + 1.2 * g
    logit = 0.8 * x1 + 1.0 * x2 - 0.6
    y = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return np.column_stack([x1, x2]), y, g.astype(str)


def _separated_biased_data(n=4000, seed=0):
    """The audit verifier's construction: strongly group-separated feature."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    x0 = rng.normal(0.8 * (2 * g - 1), 1.0)
    x1 = rng.normal(0, 1, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-2 * x0)))
    return np.column_stack([x0, x1]), y, g.astype(str)


# ---------------------------------------------------------------------------
# Finding 1: ExponentiatedGradient enforces its constraint
# ---------------------------------------------------------------------------


class TestExponentiatedGradientEnforces:
    def test_costs_are_label_and_sign_aware(self):
        # NEGATIVE CASE: the recorded defect was one unique, label-blind cost
        # per group (cost_y1 == cost_y0, n_unique == 1), so lambda's sign
        # never reached the costs. The reduction must now flip labels in
        # opposite directions for opposite-signed multipliers.
        y = np.array([1, 0, 1, 0])
        s = np.array(["a", "a", "b", "b"])
        gm = GroupManager(s)
        coords = [(g, "all") for g in gm.groups]
        phi = {("a", "all"): 2.0, ("b", "all"): -2.0}
        base_w = np.full(4, 0.25)

        z, w = _cost_sensitive_relabel(y, gm, coords, phi, base_w)

        # phi sums to zero, so net_i = (1 - 2*y_i) + phi_g * n/n_g:
        # group a (phi=+2): positives get net +3 -> relabeled 0
        # group b (phi=-2): negatives get net -3 -> relabeled 1
        assert z[0] == 0, "positive in penalized group must flip to 0"
        assert z[3] == 1, "negative in boosted group must flip to 1"
        # Weights must differ between y=1 and y=0 inside one group
        # (label-blindness was the recorded defect).
        assert not np.isclose(w[0], w[1])

    def test_costs_neutral_lambda_recovers_plain_problem(self):
        # Does not overcorrect: zero multipliers reproduce the original
        # labels with uniform weights.
        y = np.array([1, 0, 1, 0, 1, 0])
        s = np.array(["a", "a", "a", "b", "b", "b"])
        gm = GroupManager(s)
        coords = [(g, "all") for g in gm.groups]
        phi = {c: 0.0 for c in coords}
        z, w = _cost_sensitive_relabel(y, gm, coords, phi, np.full(6, 1 / 6))
        np.testing.assert_array_equal(z, y)
        # Weights are normalized to MEAN 1 (sum n): sklearn estimators scale
        # the data-fit term by the weights but not the regularization
        # penalty, so sum-1 weights would cripple regularized fits.
        assert np.allclose(w, 1.0)

    def test_eg_dp_measurably_reduces_disparity_reviewer_data(self):
        # NEGATIVE CASE: recorded run showed 0.3489 -> 0.3470 (no effect).
        X, y, s = _reviewer_biased_data()
        dp = DemographicParityConstraint(tolerance=0.05)
        base = LogisticRegression(max_iter=500).fit(X, y)
        v_base = dp.compute_violation(base.predict(X), y, s).overall_violation
        assert v_base > 0.3, "data must be strongly biased for this pin"

        eg = ExponentiatedGradient(LogisticRegression(max_iter=500), dp, max_iterations=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg.fit(X, y, sensitive_attr=s)
        v_eg = dp.compute_violation(eg.predict(X), y, s).overall_violation
        assert v_eg < 0.5 * v_base, (
            f"EG must at least halve the DP violation (unconstrained {v_base:.4f}, EG {v_eg:.4f})"
        )

    def test_eg_dp_satisfies_constraint_on_separable_bias(self):
        # On the verifier's strongly separated data the old code plateaued at
        # 5.4x the tolerance (0.2700 vs 0.05). The fixed reduction must
        # actually satisfy the constraint. eta is a saddle-point step size
        # and data-dependent as in any exponentiated-gradient method; 1.0
        # keeps the multiplier trajectory sampling the transition region on
        # this data. The pin is that the algorithm CAN enforce, which the
        # old label-blind costs could not at any setting.
        X, y, s = _separated_biased_data()
        dp = DemographicParityConstraint(tolerance=0.05)
        eg = ExponentiatedGradient(LogisticRegression(max_iter=500), dp, max_iterations=50, eta=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eg.fit(X, y, sensitive_attr=s)
        v = dp.compute_violation(eg.predict(X), y, s)
        assert v.is_satisfied, f"violation {v.overall_violation:.4f} > 0.05"
        assert bool(res.fairness_metrics["constraint_satisfied"]) is True

    def test_eg_equalized_odds_does_not_worsen(self):
        # NEGATIVE CASE: recorded run made equalized odds WORSE
        # (0.2804 -> 0.2944).
        X, y, s = _reviewer_biased_data()
        eo = EqualizedOddsConstraint(tolerance=0.05)
        base = LogisticRegression(max_iter=500).fit(X, y)
        v_base = eo.compute_violation(base.predict(X), y, s).overall_violation

        eg = ExponentiatedGradient(LogisticRegression(max_iter=500), eo, max_iterations=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg.fit(X, y, sensitive_attr=s)
        v_eg = eo.compute_violation(eg.predict(X), y, s).overall_violation
        assert v_eg < 0.5 * v_base, (
            f"EG must at least halve the equalized-odds violation "
            f"(unconstrained {v_base:.4f}, EG {v_eg:.4f})"
        )

    def test_eg_does_not_overcorrect_fair_data(self):
        # On data with no group signal the unconstrained fit already
        # satisfies DP; EG must keep (near-)baseline accuracy rather than
        # trade it away.
        rng = np.random.default_rng(3)
        n = 1500
        s = (np.arange(n) % 2).astype(str)
        x = rng.normal(0, 1, (n, 2))
        y = (x[:, 0] + 0.3 * rng.normal(0, 1, n) > 0).astype(int)
        # Tolerance 0.1: the baseline group-rate gap is pure sampling noise
        # (a few percent), so the unconstrained fit is unambiguously fair.
        dp = DemographicParityConstraint(tolerance=0.1)
        base = LogisticRegression(max_iter=500).fit(x, y)
        base_acc = np.mean(base.predict(x) == y)
        assert dp.compute_violation(base.predict(x), y, s).is_satisfied

        eg = ExponentiatedGradient(LogisticRegression(max_iter=500), dp, max_iterations=25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg.fit(x, y, sensitive_attr=s)
        y_pred = eg.predict(x)
        v = dp.compute_violation(y_pred, y, s)
        assert v.is_satisfied
        assert np.mean(y_pred == y) >= base_acc - 0.03


# ---------------------------------------------------------------------------
# Finding 2: GridSearch sweeps signed multipliers, explores, and improves
# ---------------------------------------------------------------------------


class TestGridSearchSignedSweep:
    def test_balanced_groups_sweep_is_not_a_noop(self):
        # NEGATIVE CASE: with exactly balanced groups the old shared-lambda
        # weights normalized to the uniform vector for EVERY lambda, so all
        # 20 grid classifiers were byte-identical to the unconstrained fit.
        n = 4000
        rng = np.random.default_rng(1)
        g_int = np.arange(n) % 2  # exactly balanced, alternating
        x0 = rng.normal(0.8 * (2 * g_int - 1), 1.0)
        x1 = rng.normal(0, 1, n)
        y = rng.binomial(1, 1 / (1 + np.exp(-2 * x0)))
        X = np.column_stack([x0, x1])
        s = g_int.astype(str)

        gs = GridSearch(
            LogisticRegression(max_iter=500),
            "demographic_parity",
            n_lambda_values=20,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gs.fit(X, y, sensitive_attr=s)

        prediction_sets = {clf.predict(X).tobytes() for clf in res.classifiers}
        assert len(prediction_sets) > 1, "grid classifiers must differ across lambda values"

        dp = gs._constraint
        v_base = dp.compute_violation(
            LogisticRegression(max_iter=500).fit(X, y).predict(X), y, s
        ).overall_violation
        assert v_base > 0.3
        assert res.final_violation < v_base, (
            f"sweep must improve on the unconstrained baseline "
            f"(baseline {v_base:.4f}, best {res.final_violation:.4f})"
        )

    def test_biased_data_sweep_reaches_satisfaction(self):
        # NEGATIVE CASE: the recorded run never satisfied the constraint on
        # biased data (best 0.3472 vs baseline 0.3489). With signed
        # multipliers the sweep must find a satisfying classifier here.
        X, y, s = _reviewer_biased_data()
        gs = GridSearch(
            LogisticRegression(max_iter=500),
            DemographicParityConstraint(tolerance=0.05),
            n_lambda_values=20,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gs.fit(X, y, sensitive_attr=s)
        assert res.optimization_result.converged is True
        assert res.final_violation <= 0.05

    def test_fair_data_returns_accurate_satisfied_classifier(self):
        # Does not overcorrect: when the unconstrained fit is already fair,
        # the sweep must return a satisfied classifier at (near-)baseline
        # accuracy.
        rng = np.random.default_rng(5)
        n = 1200
        s = (np.arange(n) % 2).astype(str)
        X = rng.normal(0, 1, (n, 2))
        y = (X[:, 0] > 0).astype(int)
        base_clf = LogisticRegression(max_iter=500).fit(X, y)
        base_acc = np.mean(base_clf.predict(X) == y)

        # Tolerance 0.1: the baseline group-rate gap is pure sampling noise,
        # so lambda=0 already satisfies and must win on accuracy.
        constraint = DemographicParityConstraint(tolerance=0.1)
        assert constraint.compute_violation(base_clf.predict(X), y, s).is_satisfied
        gs = GridSearch(
            LogisticRegression(max_iter=500),
            constraint,
            n_lambda_values=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = gs.fit(X, y, sensitive_attr=s)
        assert res.optimization_result.converged is True
        assert res.accuracy >= base_acc - 0.02


# ---------------------------------------------------------------------------
# Finding 3: find_feasible_thresholds (equalized odds) joint feasibility
# ---------------------------------------------------------------------------


def _best_joint_violation(y, p, s, group, threshold, thresholds):
    """Brute-force best achievable joint eq-odds violation (canonical
    max(|dTPR|, |dFPR|)) over all partner thresholds of the other group."""
    groups = np.unique(s)
    other = [g for g in groups if str(g) != str(group)][0]

    def rates(gv, t):
        mask = s == gv
        y_t, y_p = y[mask], (p[mask] >= t).astype(int)
        pos, neg = y_t == 1, y_t == 0
        tpr = np.mean(y_p[pos]) if pos.sum() else 0.0
        fpr = np.mean(y_p[neg]) if neg.sum() else 0.0
        return tpr, fpr

    tpr0, fpr0 = rates(group, threshold)
    best = np.inf
    for t in thresholds:
        tpr1, fpr1 = rates(other, t)
        best = min(best, max(abs(tpr0 - tpr1), abs(fpr0 - fpr1)))
    return best


class TestFeasibleThresholdsJoint:
    def _degenerate_data(self, n=1000, seed=11):
        # The verifier's construction: group 1 has a CONSTANT score, so its
        # only operating points are (0, 0) and (1, 1).
        rng = np.random.default_rng(seed)
        g = np.arange(n) % 2
        y = rng.binomial(1, 0.5, n)
        p = np.where(
            g == 1, 0.5, np.clip(0.5 + 0.4 * (2 * y - 1) + 0.15 * rng.normal(size=n), 0.01, 0.99)
        )
        return y, p, g.astype(str)

    def test_degenerate_partner_prunes_jointly_unmatchable(self):
        # NEGATIVE CASE: the recorded run reported 50/50 thresholds feasible
        # for both groups while a reported-feasible group-0 threshold had a
        # best achievable joint violation 17x the tolerance.
        y, p, s = self._degenerate_data()
        tolerance = 0.05
        n_thresholds = 50
        feasible = find_feasible_thresholds(
            y,
            p,
            s,
            constraint="equalized_odds",
            tolerance=tolerance,
            n_thresholds=n_thresholds,
        )
        assert len(feasible["0"]) < n_thresholds, (
            "mid-range group-0 thresholds cannot joint-match a constant-score "
            "partner and must be pruned"
        )
        # Every threshold still reported feasible must ACTUALLY have a
        # partner within tolerance (the semantic the audit demanded).
        grid = np.linspace(0.01, 0.99, n_thresholds)
        for t in feasible["0"]:
            best = _best_joint_violation(y, p, s, "0", t, grid)
            assert best <= tolerance + 1e-12, (
                f"threshold {t:.3f} reported feasible but best joint violation is {best:.4f}"
            )

    def test_identical_distributions_remain_fully_feasible(self):
        # Does not overcorrect: with IDENTICAL scores in both groups every
        # threshold has an exact partner (distance zero at the same
        # threshold), so all thresholds stay feasible.
        rng = np.random.default_rng(4)
        half = 400
        scores = rng.random(half)
        labels = (scores + 0.2 * rng.normal(size=half) > 0.5).astype(int)
        y = np.concatenate([labels, labels])
        p = np.concatenate([scores, scores])
        s = np.array(["A"] * half + ["B"] * half)
        feasible = find_feasible_thresholds(
            y, p, s, constraint="equalized_odds", tolerance=0.05, n_thresholds=50
        )
        assert len(feasible["A"]) == 50
        assert len(feasible["B"]) == 50


# ---------------------------------------------------------------------------
# Finding 4: FairRegressor mean_parity has a real effect and never crashes
# ---------------------------------------------------------------------------


class TestFairRegressorMeanParity:
    def _regression_data(self, n=2000, seed=0):
        # The verifier's construction: y = 2x + 3g + noise, X contains the
        # group dummy, so the unconstrained prediction mean gap is ~3.
        rng = np.random.default_rng(seed)
        g = rng.integers(0, 2, n)
        x = rng.normal(0, 1, n)
        X = np.column_stack([x, g])
        y = 2 * x + 3 * g + rng.normal(0, 0.5, n)
        return X, y, g, g.astype(str)

    @staticmethod
    def _mean_gap(pred, g):
        return abs(pred[g == 1].mean() - pred[g == 0].mean())

    def test_mean_parity_measurably_reduces_gap(self):
        # NEGATIVE CASE: the recorded run left the gap identical to 6
        # decimals (2.900822 -> 2.900822).
        X, y, g, s = self._regression_data()
        base_gap = self._mean_gap(LinearRegression().fit(X, y).predict(X), g)
        assert base_gap > 2.0

        reg = FairRegressor(LinearRegression(), fairness_constraint="mean_parity", tolerance=0.5)
        reg.fit(X, y, sensitive_attr=s)
        adjusted = reg.predict_with_sensitive_attr(X, s)
        gap = self._mean_gap(adjusted, g)
        assert gap <= 0.5 + 1e-9, f"adjusted mean gap {gap:.4f} exceeds tolerance"

    def test_tolerance_above_one_does_not_crash(self):
        # NEGATIVE CASE: the recorded run raised
        # "ValueError: Negative values in data passed to `sample_weight`"
        # for tolerance > 1 with |group diff| > 1.
        X, y, g, s = self._regression_data()
        reg = FairRegressor(LinearRegression(), fairness_constraint="mean_parity", tolerance=3.0)
        reg.fit(X, y, sensitive_attr=s)  # must not raise
        adjusted = reg.predict_with_sensitive_attr(X, s)
        assert self._mean_gap(adjusted, g) <= 3.0 + 1e-9

    def test_plain_predict_warns_and_is_unadjusted(self):
        # Honesty pin: predict() without the attribute cannot apply the
        # offsets; it must SAY so and return the base predictions.
        X, y, _, s = self._regression_data()
        reg = FairRegressor(LinearRegression(), fairness_constraint="mean_parity", tolerance=0.5)
        reg.fit(X, y, sensitive_attr=s)
        with pytest.warns(UserWarning, match="UNADJUSTED"):
            plain = reg.predict(X)
        np.testing.assert_allclose(plain, reg._inner_model.predict(X))

    def test_no_overcorrection_when_already_fair(self):
        # Does not overcorrect: no group effect in the data means offsets
        # stay ~0, predictions are untouched and predict() does not warn.
        rng = np.random.default_rng(9)
        n = 1000
        g = np.arange(n) % 2
        x = rng.normal(0, 1, n)
        X = x.reshape(-1, 1)
        y = 2 * x + rng.normal(0, 0.5, n)
        s = g.astype(str)

        reg = FairRegressor(LinearRegression(), fairness_constraint="mean_parity", tolerance=0.5)
        reg.fit(X, y, sensitive_attr=s)
        assert all(abs(v) < 1e-9 for v in reg.group_offsets_.values())
        adjusted = reg.predict_with_sensitive_attr(X, s)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test
            plain = reg.predict(X)
        np.testing.assert_allclose(adjusted, plain)

    def test_removed_reweighting_method_is_refused(self):
        # The provably ineffective method must fail loudly, not silently fit.
        X, y, _, s = self._regression_data(n=100)
        reg = FairRegressor(
            LinearRegression(),
            fairness_constraint="mean_parity",
            method="reweighting",
        )
        with pytest.raises(ValueError, match="reweighting"):
            reg.fit(X, y, sensitive_attr=s)
