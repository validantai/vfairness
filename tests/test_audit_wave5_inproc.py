"""
Audit wave 5: in-processing correctness pins + behaviour smoke battery.

Pins the fixes for the wave-5 audit findings in:
    - in_processing/analyzer.py (regression rejected loudly)
    - in_processing/calibrators/group_calibrators.py (unknown group ids
      pass through uncalibrated; real beta calibration)
    - in_processing/constraints/base.py (signed equalized-odds value;
      honest factory error for unimplemented constraint types)
    - in_processing/constraints/reductions.py (GridSearch report/predict
      consistency; joint ThresholdOptimizer search with validation;
      per-fit nu)
    - in_processing/loss_functions/ (warmup semantics, n=1 pairwise NaN,
      torch check at construction, factory wiring for individual and
      counterfactual losses)

Also contains the smoke battery for the subsystem: every public fairness
loss constructs and yields a finite loss, FairClassifier and every
implemented constraint fit/predict end-to-end on 200 synthetic rows, and
every group calibrator fits (gradient step) and transforms.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.in_processing.analyzer import FairnessTrainingAnalyzer
from vfairness.in_processing.constraints.base import (
    EqualizedOddsConstraint,
    FairnessConstraintType,
    create_constraint,
)
from vfairness.in_processing.constraints.reductions import (
    ExponentiatedGradient,
    GridSearch,
    ThresholdOptimizer,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------------------
# Shared synthetic data (deterministic given the seed)
# ---------------------------------------------------------------------------


def _make_classification_data(n=200, n_groups=2, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    s = np.arange(n) % n_groups
    # Group-dependent signal creates a real disparity to mitigate.
    logits = X[:, 0] + 0.8 * (s == 0) + rng.normal(size=n) * 0.5
    y = (logits > 0.4).astype(int)
    return X, y, s


# ---------------------------------------------------------------------------
# F1: analyzer rejects regression for classification-only pipelines
# ---------------------------------------------------------------------------


class TestAnalyzerRegressionRejected:
    def _regression_analyzer(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=80) * 0.1
        s = np.arange(80) % 2
        return FairnessTrainingAnalyzer(X, y, s, task_type="regression")

    def test_analyze_tradeoffs_raises_for_regression(self):
        from sklearn.linear_model import LinearRegression

        an = self._regression_analyzer()
        with pytest.raises(ValueError, match="classification only"):
            an.analyze_tradeoffs(LinearRegression())

    def test_compare_methods_raises_for_regression(self):
        from sklearn.linear_model import LinearRegression

        an = self._regression_analyzer()
        with pytest.raises(ValueError, match="classification only"):
            an.compare_methods(LinearRegression())

    def test_full_analysis_regression_warns_and_skips(self):
        from sklearn.linear_model import LinearRegression

        an = self._regression_analyzer()
        with pytest.warns(UserWarning, match="regression"):
            report = an.full_analysis(base_estimator=LinearRegression())
        assert report.method_comparisons == []
        assert report.tradeoff_analysis == {}
        # Baseline uses R^2-style score, which is meaningful for regression.
        assert report.baseline_metrics["accuracy"] > 0.9


# ---------------------------------------------------------------------------
# F2: calibrators pass unknown group ids through uncalibrated (with warning)
# F3: BetaCalibrator implements real beta calibration
# ---------------------------------------------------------------------------


@needs_torch
class TestGroupCalibrators:
    def test_unknown_group_ids_pass_through_uncalibrated(self):
        from vfairness.in_processing.calibrators.group_calibrators import (
            BetaCalibrator,
            FocalCalibrator,
            PlattScalingCalibrator,
            TemperatureScalingCalibrator,
        )

        logits = torch.tensor([2.0, -3.0, 4.0])
        gids = torch.tensor([0, 1, 5])  # 5 is out of range(n_groups=2)

        for cls in (TemperatureScalingCalibrator, PlattScalingCalibrator, BetaCalibrator):
            cal = cls(n_groups=2)
            with pytest.warns(UserWarning, match="uncalibrated"):
                out = cal(logits, gids)
            # The unknown-group sample keeps its original logit (the old
            # behaviour zeroed it out).
            assert out[2].item() == pytest.approx(4.0, abs=1e-5), cls.__name__

        probs = torch.tensor([0.9, 0.2, 0.7])
        cal = FocalCalibrator(n_groups=2)
        with pytest.warns(UserWarning, match="uncalibrated"):
            out = cal(probs, gids)
        assert out[2].item() == pytest.approx(0.7, abs=1e-5)

    def test_beta_calibrator_identity_at_init(self):
        from vfairness.in_processing.calibrators.group_calibrators import (
            BetaCalibrator,
        )

        cal = BetaCalibrator(n_groups=1)
        logits = torch.tensor([2.0, -1.0, 0.3, 0.0])
        out = cal(logits, torch.zeros(4, dtype=torch.long))
        # c=e=1, d=0: ln(p) - ln(1-p) is exactly the logit.
        assert torch.allclose(out, logits, atol=1e-4)

    def test_beta_calibrator_matches_kull_formula(self):
        from vfairness.in_processing.calibrators.group_calibrators import (
            BetaCalibrator,
        )

        cal = BetaCalibrator(n_groups=1)
        with torch.no_grad():
            cal.c[0] = 2.0
            cal.d[0] = 0.1
            cal.e[0] = 0.5
        logits = torch.tensor([1.0, -2.0, 0.5])
        out = cal(logits, torch.zeros(3, dtype=torch.long))
        p = torch.sigmoid(logits)
        expected = 2.0 * torch.log(p) - 0.5 * torch.log(1 - p) + 0.1
        assert torch.allclose(out, expected, atol=1e-5)
        # With c != e the transform is NOT affine in the logit: increments
        # of equal logit steps must differ (a Platt transform would not).
        lg = torch.tensor([1.0, 2.0, 3.0])
        o = cal(lg, torch.zeros(3, dtype=torch.long))
        assert abs((o[1] - o[0]).item() - (o[2] - o[1]).item()) > 1e-3


# ---------------------------------------------------------------------------
# F4: EqualizedOddsConstraint.signed_constraint_value carries a sign
# ---------------------------------------------------------------------------


class TestEqualizedOddsSignedValue:
    def test_signed_values_have_both_signs(self):
        # Group a: TPR=1, FPR=1. Group b: TPR=0, FPR=0.
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        sa = np.array(["a"] * 4 + ["b"] * 4)
        eo = EqualizedOddsConstraint()
        va = eo.signed_constraint_value(y_pred, y_true, sa, "a")
        vb = eo.signed_constraint_value(y_pred, y_true, sa, "b")
        assert va == pytest.approx(0.5)
        assert vb == pytest.approx(-0.5)

    def test_unknown_group_refuses_instead_of_returning_zero(self):
        """Renamed and inverted 2026-09-17 (Beta Go-Live Stage 2).

        It used to assert that an unknown group returns 0.0. On this scale 0.0
        is NO CONSTRAINT VIOLATION, so asking about a group that is not in the
        data at all came back as a clean bill for it: the exact shape this
        campaign exists to remove, and the worst possible input to get it on,
        because a group missing from the data is precisely the group a fairness
        audit most needs to hear about.

        It now raises, and the message names the groups that ARE present so the
        caller can see the typo or the missing cohort.
        """
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        sa = np.array(["a", "a", "b", "b"])
        eo = EqualizedOddsConstraint()
        with pytest.raises(KeyError, match="no group 'zzz'"):
            eo.signed_constraint_value(y_pred, y_true, sa, "zzz")

        # CONTROL: a group that IS present still measures.
        assert eo.signed_constraint_value(y_pred, y_true, sa, "a") == pytest.approx(
            eo.signed_constraint_value(y_pred, y_true, sa, "a")
        )
        assert not math.isnan(eo.signed_constraint_value(y_pred, y_true, sa, "a"))


# ---------------------------------------------------------------------------
# F7: create_constraint is honest about unimplemented enum members
# ---------------------------------------------------------------------------


class TestCreateConstraintHonesty:
    @pytest.mark.parametrize("ctype", ["predictive_parity", "error_rate_parity"])
    def test_unimplemented_types_raise_not_implemented(self, ctype):
        with pytest.raises(NotImplementedError, match="no implementation"):
            create_constraint(ctype)

    @pytest.mark.parametrize(
        "ctype",
        [
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "fpr_parity",
            "tpr_parity",
            "bounded_group_loss",
        ],
    )
    def test_implemented_types_construct(self, ctype):
        c = create_constraint(ctype)
        assert c is not None
        assert FairnessConstraintType(ctype)  # valid enum value


# ---------------------------------------------------------------------------
# F5: GridSearch reporting matches what predict() uses
# ---------------------------------------------------------------------------


class TestGridSearchConsistency:
    def test_reported_metrics_match_predictions(self):
        from sklearn.linear_model import LogisticRegression

        X, y, s = _make_classification_data(n=150, seed=1)
        gs = GridSearch(
            LogisticRegression(max_iter=200),
            create_constraint("demographic_parity", tolerance=0.2),
            n_lambda_values=8,
        )
        result = gs.fit(X, y, sensitive_attr=s)
        y_pred = result.predict(X)
        acc = np.mean(y_pred == y)
        viol = gs._constraint.compute_violation(y_pred, y, s).overall_violation
        assert result.accuracy == pytest.approx(acc, abs=1e-12)
        assert result.final_violation == pytest.approx(viol, abs=1e-12)
        # Prediction uses exactly one classifier (one-hot weights).
        assert np.sum(result.weights > 0) == 1
        assert result.weights.sum() == pytest.approx(1.0)

    def test_infeasible_constraint_reports_real_metrics(self):
        from sklearn.linear_model import LogisticRegression

        X, y, s = _make_classification_data(n=150, seed=1)
        gs = GridSearch(
            LogisticRegression(max_iter=200),
            "demographic_parity",
            n_lambda_values=6,
        )
        gs._constraint.tolerance = 1e-9  # nothing can satisfy this
        with pytest.warns(UserWarning, match="no lambda value satisfied"):
            result = gs.fit(X, y, sensitive_attr=s)
        y_pred = result.predict(X)
        acc = np.mean(y_pred == y)
        viol = gs._constraint.compute_violation(y_pred, y, s).overall_violation
        # Previously accuracy=0.0 and violation=inf were reported here
        # while predict() used an unrelated uniform ensemble.
        assert np.isfinite(result.final_violation)
        assert result.accuracy == pytest.approx(acc, abs=1e-12)
        assert result.final_violation == pytest.approx(viol, abs=1e-12)
        assert result.optimization_result.converged is False


# ---------------------------------------------------------------------------
# F6: ThresholdOptimizer evaluates and validates the JOINT assignment
# ---------------------------------------------------------------------------


class TestThresholdOptimizerJointSearch:
    def _three_group_data(self):
        # Deterministic three-group data with a strong disparity that a
        # one-group-at-a-time search from all-0.5 cannot fix.
        n = 180
        groups = np.arange(n) % 3
        base = np.tile(np.linspace(0.05, 0.95, n // 3), 3)
        probs = np.clip(base + 0.2 * (groups == 0) - 0.2 * (groups == 2), 0.01, 0.99)
        y = (base > 0.5).astype(int)
        return probs, y, groups

    def test_stored_violation_matches_actual_predictions(self):
        probs, y, groups = self._three_group_data()
        to = ThresholdOptimizer("demographic_parity", grid_size=30)
        to.fit(probs, y, sensitive_attr=groups)
        y_pred = to.predict(probs, groups)
        actual = to._constraint.compute_violation(y_pred, y, groups)
        assert to.final_violation_ == pytest.approx(actual.overall_violation, abs=1e-12)
        assert to.constraint_satisfied_ == actual.is_satisfied

    def test_joint_search_reaches_feasibility(self):
        probs, y, groups = self._three_group_data()
        to = ThresholdOptimizer("demographic_parity", grid_size=30)
        to.fit(probs, y, sensitive_attr=groups)
        # The disparity here is fixable by moving group 0 up and group 2
        # down; the old others-fixed-at-0.5 search left all thresholds at
        # 0.5 and never noticed the joint assignment was infeasible.
        assert to.constraint_satisfied_ is True
        thresholds = to.get_thresholds()
        assert thresholds["0"] > thresholds["2"]

    def test_warns_when_no_feasible_assignment(self):
        probs, y, groups = self._three_group_data()
        to = ThresholdOptimizer("demographic_parity", grid_size=10)
        # A negative tolerance is unsatisfiable by construction (violations
        # are >= 0), so the warning path is guaranteed regardless of data.
        to._constraint.tolerance = -0.1
        with pytest.warns(UserWarning, match="no joint threshold"):
            to.fit(probs, y, sensitive_attr=groups)
        assert to.constraint_satisfied_ is False
        assert np.isfinite(to.final_violation_)


# ---------------------------------------------------------------------------
# F8: ExponentiatedGradient recomputes nu per fit
# ---------------------------------------------------------------------------


class TestExponentiatedGradientNu:
    def test_nu_recomputed_for_different_group_counts(self):
        from sklearn.linear_model import LogisticRegression

        X, y, _ = _make_classification_data(n=120, seed=2)
        eg = ExponentiatedGradient(
            LogisticRegression(max_iter=200),
            "demographic_parity",
            max_iterations=2,
        )
        s2 = np.arange(120) % 2
        eg.fit(X, y, sensitive_attr=s2)
        nu2 = eg.get_result().optimization_result.metadata["nu"]
        s4 = np.arange(120) % 4
        eg.fit(X, y, sensitive_attr=s4)
        nu4 = eg.get_result().optimization_result.metadata["nu"]
        assert nu2 == pytest.approx(1.0 / 4)
        assert nu4 == pytest.approx(1.0 / 8)
        # The user-facing attribute is not silently mutated.
        assert eg.nu is None


# ---------------------------------------------------------------------------
# F9: warmup means NO penalty during warmup epochs
# F10: n=1 batch yields zero (not NaN) individual-fairness penalty
# F11: losses raise ImportError at construction without torch
# F12: factory produces working losses for every metric type
# ---------------------------------------------------------------------------


@needs_torch
class TestLossFunctionFixes:
    def test_warmup_zero_then_full(self):
        from vfairness.in_processing.loss_functions.fairness_losses import (
            DemographicParityLoss,
        )

        loss = DemographicParityLoss(lambda_fairness=1.0, warmup_epochs=5)
        for epoch in range(5):
            loss.set_epoch(epoch)
            assert loss._get_effective_lambda() == 0.0
        loss.set_epoch(5)
        assert loss._get_effective_lambda() == 1.0

    def test_individual_fairness_batch_of_one_is_finite(self):
        from vfairness.in_processing.loss_functions.counterfactual import (
            IndividualFairnessLoss,
        )

        loss_fn = IndividualFairnessLoss()
        loss = loss_fn(
            torch.tensor([0.7]),
            torch.tensor([1.0]),
            torch.tensor([0]),
            features=torch.tensor([[1.0, 2.0]]),
        )
        assert torch.isfinite(loss).item()

    def test_construction_raises_without_torch(self, monkeypatch):
        import vfairness.in_processing.loss_functions.base as lfbase
        from vfairness.in_processing.loss_functions.fairness_losses import (
            DemographicParityLoss,
        )

        monkeypatch.setattr(lfbase, "TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch is required"):
            DemographicParityLoss()

    @pytest.mark.parametrize(
        "metric",
        [
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "predictive_parity",
            "calibration",
            "individual_fairness",
            "counterfactual_fairness",
        ],
    )
    def test_factory_losses_work_at_first_call(self, metric):
        from vfairness.in_processing.loss_functions.fairness_losses import (
            create_fairness_loss,
        )

        loss_fn = create_fairness_loss(metric, lambda_fairness=0.1)
        y_pred = torch.tensor([0.2, 0.8, 0.5, 0.6])
        y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
        s = torch.tensor([0, 1, 0, 1])
        with warnings.catch_warnings():
            # individual/counterfactual warn when optional inputs are
            # missing; the point here is that they do not crash.
            warnings.simplefilter("ignore")
            loss = loss_fn(y_pred, y_true, s)
        assert torch.isfinite(loss).item()

    def test_factory_individual_and_counterfactual_types(self):
        from vfairness.in_processing.loss_functions.counterfactual import (
            CounterfactualFairnessLoss,
            IndividualFairnessLoss,
        )
        from vfairness.in_processing.loss_functions.fairness_losses import (
            create_fairness_loss,
        )

        assert isinstance(create_fairness_loss("individual_fairness"), IndividualFairnessLoss)
        assert isinstance(
            create_fairness_loss("counterfactual_fairness"),
            CounterfactualFairnessLoss,
        )


# ---------------------------------------------------------------------------
# Smoke battery: the always-crash-shaped paths of the subsystem
# ---------------------------------------------------------------------------


@needs_torch
class TestSmokeLosses:
    """Every public loss in fairness_losses constructs and computes a
    finite loss with a gradient on a tiny batch."""

    @pytest.mark.parametrize(
        "make_loss",
        [
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["FairnessAwareBCELoss"],
            ).FairnessAwareBCELoss(lambda_fairness=0.1),
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["DemographicParityLoss"],
            ).DemographicParityLoss(lambda_fairness=0.1),
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["EqualizedOddsLoss"],
            ).EqualizedOddsLoss(lambda_fairness=0.1),
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["EqualOpportunityLoss"],
            ).EqualOpportunityLoss(lambda_fairness=0.1),
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["FalsePositiveRateParityLoss"],
            ).FalsePositiveRateParityLoss(lambda_fairness=0.1),
            lambda: __import__(
                "vfairness.in_processing.loss_functions.fairness_losses",
                fromlist=["BoundedGroupLoss"],
            ).BoundedGroupLoss(lambda_fairness=0.1),
        ],
    )
    def test_loss_finite_and_differentiable(self, make_loss):
        torch.manual_seed(0)
        loss_fn = make_loss()
        raw = torch.randn(16, requires_grad=True)
        y_pred = torch.sigmoid(raw)
        y_true = (torch.arange(16) % 2).float()
        s = torch.arange(16) % 2
        loss = loss_fn(y_pred, y_true, s)
        assert torch.isfinite(loss).item()
        loss.backward()
        assert torch.isfinite(raw.grad).all().item()


class TestSmokeConstraintsEndToEnd:
    """FairClassifier and each implemented constraint fit/predict on 200
    synthetic rows."""

    @pytest.mark.parametrize("method", ["reductions", "threshold", "grid_search"])
    def test_fair_classifier_each_method(self, method):
        from sklearn.linear_model import LogisticRegression

        from vfairness.in_processing.wrappers import FairClassifier

        X, y, s = _make_classification_data(n=200, seed=7)
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=200),
            fairness_constraint="demographic_parity",
            tolerance=0.1,
            method=method,
            max_iterations=5,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=s)
            y_pred = clf.predict(X)
        assert y_pred.shape == (200,)
        assert set(np.unique(y_pred)).issubset({0, 1})
        assert clf.fairness_result_ is not None
        assert np.isfinite(clf.fairness_result_.accuracy)

    @pytest.mark.parametrize(
        "ctype",
        [
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "fpr_parity",
            "bounded_group_loss",
        ],
    )
    def test_exponentiated_gradient_each_constraint(self, ctype):
        from sklearn.linear_model import LogisticRegression

        X, y, s = _make_classification_data(n=200, seed=9)
        eg = ExponentiatedGradient(
            LogisticRegression(max_iter=200),
            ctype,
            max_iterations=3,
        )
        result = eg.fit(X, y, sensitive_attr=s)
        y_pred = eg.predict(X)
        assert y_pred.shape == (200,)
        assert set(np.unique(y_pred)).issubset({0, 1})
        assert np.isfinite(result.final_violation)
        assert 0.0 <= result.accuracy <= 1.0


@needs_torch
class TestSmokeCalibrators:
    """Each calibrator transforms a batch and takes a gradient step."""

    @pytest.mark.parametrize("method", ["temperature", "platt", "beta", "focal"])
    def test_calibrator_fit_and_transform(self, method):
        from vfairness.in_processing.calibrators.group_calibrators import (
            TrainableGroupCalibrator,
        )

        torch.manual_seed(0)
        cal = TrainableGroupCalibrator(n_groups=2, method=method)
        n = 64
        raw_logits = torch.randn(n) * 2
        group_ids = torch.arange(n) % 2
        y = (torch.rand(n) < torch.sigmoid(raw_logits)).float()

        # Focal calibrates probabilities, the others calibrate logits.
        inputs = torch.sigmoid(raw_logits) if method == "focal" else raw_logits

        out = cal(inputs, group_ids)
        assert out.shape == (n,)
        assert torch.isfinite(out).all().item()

        # One optimization step must run without error and keep params finite.
        opt = torch.optim.SGD(cal.parameters(), lr=0.01)
        opt.zero_grad()
        logits_for_loss = torch.logit(out.clamp(1e-6, 1 - 1e-6)) if method == "focal" else out
        loss = cal.calibration_loss(logits_for_loss, y, group_ids)
        loss.backward()
        opt.step()
        for p in cal.parameters():
            assert torch.isfinite(p).all().item()

        state = cal.get_calibration_state()
        assert state.method == method
        assert len(state.parameters) == 2
