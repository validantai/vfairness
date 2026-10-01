"""
Audit 6, lane 3: post-processing and in-processing honesty pins.

Two defect classes, each reproduced by execution before its fix (2026-09-09).

(1) A value that was never measured was replaced by a neutral default and
    reported as a measurement.
      CAL-DISP  calibration_disparity with fewer than two adequate groups
                returned ece/mce/brier disparity 0.0 (perfect parity on
                every scale) and has_significant_disparity False.
      C-09 tail ThresholdResult.is_feasible was typed bool while the
                ConstraintViolation.is_satisfied it is built from can be
                None (could not check), and summary() rendered None as a
                failure glyph.
    The fix is three states: NaN / None for unmeasured, warned by name,
    verdicts None rather than False, and the summary says "not assessed".

(2) A documented parameter or option that was accepted and did nothing.
      F6   DistributionMatcher(method=...)              stored, never read
      F7   sample_weight in four reweighter fit()s      accepted, never read
      F12  PredictionReweighter(preserve_ranking=...)   stored, never read,
                                                        and untrue at the clip
      F13  CalibratedEqualizer(preserve_calibration=)   stored, never read
      F10  GroupThresholdOptimizer(grid_search=False)   no gradient path exists
      F17  TemperatureScaling(init_temperature=...)     bounded minimiser has
                                                        no starting point
      F19  fallback_strategy='borrow'                   ran 'global' silently
      F4   CausalFairnessLoss direct_effect /
           path_specific / mediator_indices              all aliased total_effect
      F5   ConditionalIndependenceRegularizer(
           conditional_on=...)                           always the label
      F14  FairnessAwareBCELoss(temperature=...)        stored, never applied
      F16  generate_recommendation(tradeoff_analysis=)  never read
    The honest fix is a small implementation only where it is verifiable by
    execution (F7 for PredictionReweighter, F14), otherwise a loud refusal at
    construction naming what IS implemented plus a corrected docstring.
    Removed parameters (F12, F13, F17, F16) raise TypeError like any unknown
    keyword.

Every pin carries an over-correction control: the healthy path is unchanged
(checked byte-for-byte against the HEAD module for the reweighters and
TemperatureScaling while the fix was made).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import warnings

import numpy as np
import pytest

from vfairness.in_processing.analyzer import FairnessTrainingAnalyzer, MethodComparison
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.post_processing.calibration.group_calibrator import (
    IMPLEMENTED_FALLBACK_STRATEGIES,
    GroupCalibrator,
)
from vfairness.post_processing.calibration.methods import TemperatureScaling
from vfairness.post_processing.calibration.metrics import calibration_disparity
from vfairness.post_processing.reweighting.reweighter import (
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
)
from vfairness.post_processing.threshold_optimization.constraints import ConstraintViolation
from vfairness.post_processing.threshold_optimization.optimizer import (
    FairnessConstraintType,
    GroupThresholdOptimizer,
    ThresholdResult,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only on a core-only install
    HAS_TORCH = False

_ENFORCED = os.environ.get("VFAIRNESS_REQUIRE_BACKENDS") == "1"


@pytest.fixture
def torch_mod():
    """torch, or a NAMED failure when the environment claims the full suite runs."""
    if HAS_TORCH:
        return torch
    if _ENFORCED:
        pytest.fail(
            "torch is not installed; VFAIRNESS_REQUIRE_BACKENDS=1 makes that a "
            "failure, not a skip. Install the 'training' extra."
        )
    pytest.skip("torch not installed")


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    p = np.clip(rng.beta(2, 2, n) * 0.6 + 0.2 * y, 0.01, 0.99)
    g = np.where(rng.random(n) < 0.5, "A", "B")
    return y, p, g


@pytest.fixture
def small_group_data(binary_data):
    """One 280-sample group and one 20-sample group (below min_group_size=50)."""
    y, p, g = binary_data
    small = np.full_like(g, "A")
    small[:20] = "B"
    return y, p, small


# ===========================================================================
# CAL-DISP: a disparity needs two groups; with fewer it is NaN, not 0.0
# ===========================================================================


class TestCalDisp:
    def test_one_group_is_nan_and_verdict_is_none(self, binary_data):
        y, p, _ = binary_data
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = calibration_disparity(y, p, np.array(["A"] * len(y)))
        assert r.n_groups == 1
        assert np.isnan(r.ece_disparity)
        assert np.isnan(r.mce_disparity)
        assert np.isnan(r.brier_disparity)
        assert r.has_significant_disparity is None
        assert r.to_dict()["has_significant_disparity"] is None
        messages = [str(w.message) for w in caught]
        assert any("NaN" in m and "'A'" in m for m in messages), messages

    def test_two_groups_unchanged(self, binary_data):
        """Over-correction control: a real comparison is still a number."""
        y, p, g = binary_data
        r = calibration_disparity(y, p, g)
        assert r.n_groups == 2
        assert np.isfinite(r.ece_disparity)
        assert np.isfinite(r.mce_disparity)
        assert np.isfinite(r.brier_disparity)
        eces = list(r.group_ece.values())
        assert r.ece_disparity == pytest.approx(max(eces) - min(eces))
        assert isinstance(r.has_significant_disparity, bool)
        assert r.has_significant_disparity == bool(r.ece_disparity > 0.05)

    def test_report_carries_the_third_state(self, binary_data):
        y, p, g = binary_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one = CalibrationAnalyzer(y, p, np.array(["A"] * len(y))).full_analysis()
            two = CalibrationAnalyzer(y, p, g).full_analysis()
        assert one.has_significant_disparity is None
        lines = one.summary().splitlines()
        assert "Significant Disparity: Not assessable (not measured)" in lines
        assert "ECE Disparity: not measured (fewer than two groups)" in lines
        assert "Significant Disparity: No" not in lines
        assert "Significant Disparity: Yes" not in lines
        # control: the two-group report still renders a measured verdict
        assert two.has_significant_disparity in (True, False)
        two_lines = two.summary().splitlines()
        assert ("Significant Disparity: Yes" in two_lines) or (
            "Significant Disparity: No" in two_lines
        )


# ===========================================================================
# ThresholdResult.is_feasible: could-not-check is neither True nor False
# ===========================================================================


def _violation(is_satisfied, violation=0.01):
    return ConstraintViolation(
        constraint_type=FairnessConstraintType.DEMOGRAPHIC_PARITY,
        violation=violation,
        group_metrics={},
        group_violations={},
        is_satisfied=is_satisfied,
        tolerance=0.05,
    )


def _result(verdict, violation):
    return ThresholdResult(
        global_threshold=0.5,
        group_thresholds={},
        constraint_violations=[violation],
        performance_metrics={"accuracy": 0.8},
        is_feasible=verdict,
    )


class TestThresholdResultThreeStates:
    def test_none_is_not_assessed(self):
        unknown = _violation(None, violation=float("nan"))
        r = _result(unknown.is_satisfied, unknown)
        assert r.is_feasible is None
        assert r.to_dict()["is_feasible"] is None
        text = r.summary()
        assert "Feasible: not assessed" in text
        assert "Feasible: False" not in text
        assert "✗" not in text
        assert "? demographic_parity" in text

    @pytest.mark.parametrize("verdict,glyph", [(True, "✓"), (False, "✗")])
    def test_measured_verdicts_unchanged(self, verdict, glyph):
        """Over-correction control: measured pass/fail renders as before."""
        r = _result(verdict, _violation(verdict))
        assert r.is_feasible is verdict
        text = r.summary()
        assert f"Feasible: {verdict}" in text
        assert f"{glyph} demographic_parity" in text
        assert "not assessed" not in text

    def test_optimizer_still_reports_a_bool_on_healthy_data(self, binary_data):
        y, p, g = binary_data
        opt = GroupThresholdOptimizer(n_thresholds=10).fit(y_true=y, y_prob=p, sensitive_attr=g)
        assert isinstance(opt.result_.is_feasible, bool)


# ===========================================================================
# F6: DistributionMatcher.method
# ===========================================================================


class TestF6DistributionMatcherMethod:
    def test_histogram_refused_by_name(self):
        with pytest.raises(NotImplementedError, match="histogram.*only method='quantile'"):
            DistributionMatcher(method="histogram")

    def test_unknown_refused(self):
        with pytest.raises(ValueError, match="totally_bogus"):
            DistributionMatcher(method="totally_bogus")

    def test_quantile_control(self, binary_data):
        y, p, g = binary_data
        dm = DistributionMatcher(method="quantile").fit(y_true=y, y_prob=p, sensitive_attr=g)
        out = dm.transform(p, g)
        ref = dm.reference_group
        # the reference group is untouched, the other group is mapped onto it
        assert np.array_equal(out[g == ref], p[g == ref])
        assert not np.array_equal(out[g != ref], p[g != ref])
        assert out.min() >= 0 and out.max() <= 1


# ===========================================================================
# F7: sample_weight accepted and ignored
# ===========================================================================


@pytest.mark.parametrize(
    "cls", [RejectionOptionClassifier, CalibratedEqualizer, DistributionMatcher]
)
class TestF7RefusingReweighters:
    def test_non_uniform_weights_refused_by_name(self, cls, binary_data):
        y, p, g = binary_data
        sw = np.where(g == "A", 0.01, 100.0)
        with pytest.raises(NotImplementedError) as exc:
            cls().fit(y_true=y, y_prob=p, sensitive_attr=g, sample_weight=sw)
        assert cls.__name__ in str(exc.value)
        assert "PredictionReweighter" in str(exc.value)

    def test_none_and_uniform_weights_identical(self, cls, binary_data):
        """Over-correction control: None and a uniform vector are the same fit."""
        y, p, g = binary_data
        a = cls().fit(y_true=y, y_prob=p, sensitive_attr=g).transform(p, g)
        b = (
            cls()
            .fit(y_true=y, y_prob=p, sensitive_attr=g, sample_weight=np.full(len(y), 3.0))
            .transform(p, g)
        )
        assert _sha(a) == _sha(b)

    def test_bad_weight_vectors_refused(self, cls, binary_data):
        y, p, g = binary_data
        with pytest.raises(ValueError, match="entries"):
            cls().fit(y_true=y, y_prob=p, sensitive_attr=g, sample_weight=np.ones(len(y) - 1))
        with pytest.raises(ValueError, match="non-negative"):
            cls().fit(
                y_true=y, y_prob=p, sensitive_attr=g, sample_weight=np.where(g == "A", -1.0, 1.0)
            )


class TestF7PredictionReweighterHonoursWeights:
    def test_unweighted_path_is_the_plain_mean(self, binary_data):
        """Over-correction control: sample_weight=None is exactly the old fit."""
        y, p, g = binary_data
        pr = PredictionReweighter().fit(y_true=y, y_prob=p, sensitive_attr=g)
        target = float(np.mean(p))
        assert pr.target_rate_ == target
        for grp in ("A", "B"):
            assert pr.adjustments_[grp] == target / np.mean(p[g == grp])

    def test_uniform_weights_match_unweighted(self, binary_data):
        y, p, g = binary_data
        a = PredictionReweighter().fit(y_true=y, y_prob=p, sensitive_attr=g)
        b = PredictionReweighter().fit(
            y_true=y, y_prob=p, sensitive_attr=g, sample_weight=np.full(len(y), 2.5)
        )
        # np.average is sum(w*a)/sum(w), so equality holds to rounding, not bits
        for grp in ("A", "B"):
            assert b.adjustments_[grp] == pytest.approx(a.adjustments_[grp], rel=1e-12)

    def test_weights_move_the_fit_in_the_expected_direction(self, binary_data):
        y, p, g = binary_data
        base = PredictionReweighter(target_rate=0.5).fit(y_true=y, y_prob=p, sensitive_attr=g)
        # up-weight the high-probability half of group A: A's weighted mean
        # rises, so the multiplicative factor it needs to hit 0.5 falls
        w = np.where((g == "A") & (p > np.median(p[g == "A"])), 10.0, 1.0)
        weighted = PredictionReweighter(target_rate=0.5).fit(
            y_true=y, y_prob=p, sensitive_attr=g, sample_weight=w
        )
        assert weighted.adjustments_["A"] < base.adjustments_["A"]
        assert weighted.adjustments_["B"] == pytest.approx(base.adjustments_["B"], rel=1e-12)
        expected_a = 0.5 / np.average(p[g == "A"], weights=w[g == "A"])
        assert weighted.adjustments_["A"] == pytest.approx(expected_a, rel=1e-12)

    def test_weighted_default_target_rate(self, binary_data):
        y, p, g = binary_data
        w = np.where(p > np.median(p), 10.0, 1.0)
        pr = PredictionReweighter().fit(y_true=y, y_prob=p, sensitive_attr=g, sample_weight=w)
        assert pr.target_rate_ == pytest.approx(float(np.average(p, weights=w)), rel=1e-12)
        assert pr.target_rate_ != pytest.approx(float(np.mean(p)), rel=1e-6)

    def test_zero_weight_group_refused(self, binary_data):
        y, p, g = binary_data
        with pytest.raises(ValueError, match="zero for every sample in group"):
            PredictionReweighter().fit(
                y_true=y, y_prob=p, sensitive_attr=g, sample_weight=np.where(g == "A", 0.0, 1.0)
            )


# ===========================================================================
# F12: preserve_ranking removed; the clip's rank loss is reported instead
# ===========================================================================


class TestF12PreserveRanking:
    def test_parameter_is_gone(self):
        assert "preserve_ranking" not in inspect.signature(PredictionReweighter).parameters
        with pytest.raises(TypeError, match="preserve_ranking"):
            PredictionReweighter(preserve_ranking=True)

    def test_n_clipped_is_measured_not_promised(self, binary_data):
        from scipy.stats import spearmanr

        y, _, g = binary_data
        rng = np.random.default_rng(1)
        # group A low, group B high: a multiplicative pull towards a high
        # target saturates part of group A at 1.0
        p = np.where(g == "A", rng.beta(2, 5, len(y)), rng.beta(6, 2, len(y)))
        pr = PredictionReweighter(method="multiplicative", target_rate=0.7).fit(
            y_true=y, y_prob=p, sensitive_attr=g
        )
        out = pr.transform(p, g)
        unclipped = p * np.array([pr.adjustments_[grp] for grp in g])
        n_clipped = int(((unclipped < 0) | (unclipped > 1)).sum())
        assert n_clipped > 0
        assert pr.result_.calibration_impact["n_clipped"] == n_clipped
        assert np.array_equal(out, np.clip(unclipped, 0, 1))
        # the ranking claim the flag made was false: ties at 1.0 break it
        rho = spearmanr(p[g == "A"], out[g == "A"]).correlation
        assert rho < 1.0

    def test_no_saturation_control(self, binary_data):
        """Over-correction control: a mild fit clips nothing and keeps ranks."""
        from scipy.stats import spearmanr

        y, p, g = binary_data
        pr = PredictionReweighter().fit(y_true=y, y_prob=p, sensitive_attr=g)
        out = pr.transform(p, g)
        assert pr.result_.calibration_impact["n_clipped"] == 0
        for grp in ("A", "B"):
            assert spearmanr(p[g == grp], out[g == grp]).correlation == pytest.approx(1.0)


# ===========================================================================
# F13: preserve_calibration removed
# ===========================================================================


class TestF13PreserveCalibration:
    def test_parameter_is_gone(self):
        assert "preserve_calibration" not in inspect.signature(CalibratedEqualizer).parameters
        with pytest.raises(TypeError, match="preserve_calibration"):
            CalibratedEqualizer(preserve_calibration=True)

    def test_quantile_mapping_control(self, binary_data):
        y, p, g = binary_data
        eq = CalibratedEqualizer(n_quantiles=50).fit(y_true=y, y_prob=p, sensitive_attr=g)
        out = eq.transform(p, g)
        assert out.min() >= 0 and out.max() <= 1
        assert "distribution_shift" in eq.result_.calibration_impact
        assert eq.result_.adjusted_metrics["mean_disparity"] < (
            eq.result_.original_metrics["mean_disparity"] + 1e-12
        )


# ===========================================================================
# F10: grid_search=False promised a path that does not exist
# ===========================================================================


class TestF10GridSearch:
    def test_false_refused_by_name(self):
        with pytest.raises(NotImplementedError, match="only implements grid search"):
            GroupThresholdOptimizer(grid_search=False)

    def test_true_and_default_identical(self, binary_data):
        y, p, g = binary_data
        a = GroupThresholdOptimizer(n_thresholds=10).fit(y_true=y, y_prob=p, sensitive_attr=g)
        b = GroupThresholdOptimizer(n_thresholds=10, grid_search=True).fit(
            y_true=y, y_prob=p, sensitive_attr=g
        )
        assert a.result_.group_thresholds == b.result_.group_thresholds

    def test_objective_refusal_untouched(self):
        """The lead's objective validation still fires (not weakened by F10)."""
        with pytest.raises(ValueError, match="cannot compute objective"):
            GroupThresholdOptimizer(objective="expected_cost")


# ===========================================================================
# F17: init_temperature removed
# ===========================================================================


class TestF17InitTemperature:
    def test_parameter_is_gone(self):
        assert set(inspect.signature(TemperatureScaling).parameters) == {"max_iter", "tol"}
        with pytest.raises(TypeError, match="init_temperature"):
            TemperatureScaling(init_temperature=2.0)

    def test_fit_control(self, binary_data):
        y, p, _ = binary_data
        ts = TemperatureScaling().fit(y, p)
        assert 0.01 <= ts.temperature_ <= 100.0
        assert ts.fit_result.parameters["temperature"] == ts.temperature_
        eps = 1e-10
        cal = ts.transform(p)

        def nll(q):
            return -np.mean(
                y * np.log(np.clip(q, eps, 1)) + (1 - y) * np.log(np.clip(1 - q, eps, 1))
            )

        assert nll(cal) <= nll(p) + 1e-12


# ===========================================================================
# F19: 'borrow' refused at both entry points
# ===========================================================================


class TestF19Borrow:
    def test_group_calibrator_refuses_borrow(self):
        with pytest.raises(NotImplementedError) as exc:
            GroupCalibrator(fallback_strategy="borrow")
        assert "'global'" in str(exc.value) and "'none'" in str(exc.value)
        assert IMPLEMENTED_FALLBACK_STRATEGIES == ("global", "none")

    def test_group_calibrator_refuses_unknown(self):
        with pytest.raises(ValueError, match="garbage"):
            GroupCalibrator(fallback_strategy="garbage")

    def test_analyzer_refuses_borrow(self, small_group_data):
        y, p, small = small_group_data
        an = CalibrationAnalyzer(y, p, small, min_group_size=50)
        with pytest.raises(NotImplementedError, match="borrow"):
            an.fit_calibrator(method="platt", fallback_strategy="borrow")
        assert an._group_calibrator is None

    @pytest.mark.parametrize("strategy", ["global", "none"])
    def test_implemented_strategies_control(self, strategy, small_group_data):
        y, p, small = small_group_data
        gc = GroupCalibrator(method="platt", min_group_size=50, fallback_strategy=strategy).fit(
            y_true=y, y_prob=p, protected_attr=small
        )
        assert gc.fit_result_.fallback_groups == ["B"]
        if strategy == "global":
            assert gc.calibrators_["B"] is gc.global_calibrator_
        else:
            assert gc.calibrators_["B"] is None
        out = gc.transform(p, small)
        assert out.shape == p.shape


# ===========================================================================
# F4: CausalFairnessLoss criteria that all computed the same thing
# ===========================================================================


class TestF4CausalFairnessLoss:
    @pytest.mark.parametrize("criterion", ["direct_effect", "path_specific"])
    def test_unimplemented_criteria_refused(self, torch_mod, criterion):
        from vfairness.in_processing.loss_functions.counterfactual import CausalFairnessLoss

        with pytest.raises(NotImplementedError, match=f"{criterion}.*Only 'total_effect'"):
            CausalFairnessLoss(causal_criterion=criterion)

    def test_mediators_refused(self, torch_mod):
        from vfairness.in_processing.loss_functions.counterfactual import CausalFairnessLoss

        with pytest.raises(NotImplementedError, match="mediator_indices"):
            CausalFairnessLoss(mediator_indices=[0, 1, 2])
        with pytest.raises(ValueError, match="garbage"):
            CausalFairnessLoss(causal_criterion="garbage")

    def test_total_effect_control(self, torch_mod):
        from vfairness.in_processing.loss_functions.counterfactual import CausalFairnessLoss

        torch_mod.manual_seed(0)
        yp = torch_mod.sigmoid(torch_mod.randn(200))
        yt = (torch_mod.rand(200) > 0.5).float()
        sa = (torch_mod.rand(200) > 0.5).long()
        loss = CausalFairnessLoss(lambda_fairness=0.3, mediator_indices=[])(yp, yt, sa)
        bce = torch_mod.nn.functional.binary_cross_entropy(yp, yt)
        expected = bce + 0.3 * abs(yp[sa == 0].mean() - yp[sa == 1].mean())
        assert float(loss.detach()) == pytest.approx(float(expected), rel=1e-6)

    def test_group_mean_counterfactual_expand_fix(self, torch_mod):
        """mypy call-overload at expand(mask.sum(), -1): same result with an int."""
        from vfairness.in_processing.loss_functions.counterfactual import (
            CounterfactualFairnessLoss,
        )

        torch_mod.manual_seed(1)
        feats = torch_mod.randn(40, 5)
        sa = torch_mod.tensor([0, 1] * 20)
        cf = CounterfactualFairnessLoss(
            counterfactual_strategy="group_mean"
        )._group_mean_counterfactual(feats, sa)
        assert cf.shape == feats.shape
        for grp, other in ((0, 1), (1, 0)):
            expected = feats[sa == other].mean(dim=0)
            assert torch_mod.allclose(cf[sa == grp], expected.expand(20, -1))


# ===========================================================================
# F5: conditional_on always meant the label
# ===========================================================================


class TestF5ConditionalOn:
    @pytest.mark.parametrize("option", ["prediction", "both"])
    def test_unimplemented_refused(self, torch_mod, option):
        from vfairness.in_processing.regularizers.fairness_regularizers import (
            ConditionalIndependenceRegularizer,
        )

        with pytest.raises(NotImplementedError, match=f"{option}.*conditional_on='label'"):
            ConditionalIndependenceRegularizer(conditional_on=option)
        with pytest.raises(ValueError, match="garbage"):
            ConditionalIndependenceRegularizer(conditional_on="garbage")

    def test_label_control(self, torch_mod):
        from vfairness.in_processing.regularizers.fairness_regularizers import (
            ConditionalIndependenceRegularizer,
        )

        torch_mod.manual_seed(0)
        yp = torch_mod.sigmoid(torch_mod.randn(200))
        yt = (torch_mod.rand(200) > 0.5).float()
        sa = (torch_mod.rand(200) > 0.5).long()
        penalty = ConditionalIndependenceRegularizer(strength=0.4, conditional_on="label")(
            yp, sa, yt
        )
        terms = []
        for label in (0.0, 1.0):
            label_mean = yp[yt == label].mean()
            for grp in (0, 1):
                terms.append(abs(yp[(yt == label) & (sa == grp)].mean() - label_mean))
        expected = 0.4 * torch_mod.stack(terms).mean()
        assert float(penalty.detach()) == pytest.approx(float(expected), rel=1e-6)


# ===========================================================================
# F14: temperature now reaches the soft rates
# ===========================================================================


def _fixed_batch(torch_mod):
    torch_mod.manual_seed(0)
    yp = torch_mod.sigmoid(torch_mod.randn(200))
    yt = (torch_mod.rand(200) > 0.5).float()
    sa = (torch_mod.rand(200) > 0.5).long()
    return yp, yt, sa


class TestF14Temperature:
    def test_temperatures_now_differ(self, torch_mod):
        from vfairness.in_processing.loss_functions.fairness_losses import FairnessAwareBCELoss

        yp, yt, sa = _fixed_batch(torch_mod)
        losses = {
            t: float(FairnessAwareBCELoss(temperature=t, lambda_fairness=1.0)(yp, yt, sa).detach())
            for t in (0.01, 1.0, 100.0)
        }
        assert len(set(losses.values())) == 3, losses

    def test_t_equal_one_is_the_old_loss(self, torch_mod):
        """Over-correction control: T=1 skips the tempering entirely.

        The seed-0 batch below round-trips sigmoid(logit(p)) exactly, so on
        its own it cannot tell the `if self.temperature != 1.0` skip from a
        variant that always tempers. An adversarial reviewer removed that skip
        on 2026-09-09: all nine pins in this class stayed green while the
        variant differed from the old loss in 15 of 1000 batches. So the
        second half checks the identity on inputs the logit round trip
        provably does NOT preserve, against a reference the loss object's
        temperature cannot move.
        """
        from vfairness.in_processing.loss_functions.fairness_losses import (
            FairnessAwareBCELoss,
            _temper_predictions,
        )

        yp, yt, sa = _fixed_batch(torch_mod)
        default = FairnessAwareBCELoss(lambda_fairness=1.0)(yp, yt, sa)
        explicit = FairnessAwareBCELoss(lambda_fairness=1.0, temperature=1.0)(yp, yt, sa)
        assert float(default.detach()) == float(explicit.detach())
        bce = torch_mod.nn.functional.binary_cross_entropy(yp, yt)
        overall = yp.mean()
        dp = sum(abs(yp[sa == k].mean() - overall) for k in (0, 1)) / 2
        assert float(default.detach()) == pytest.approx(float(bce + dp), rel=1e-6)

        # Saturated predictions: sigmoid(randn * 30) is exactly 0.0 or 1.0 for
        # most entries, which _temper_predictions has to clamp, so tempering
        # at T=1 is measurably not the identity on them. Each metric is
        # compared against its own penalty computed from the RAW predictions;
        # the private penalty methods never temper, so the reference stays put
        # even if _compute_fairness_penalty starts tempering unconditionally.
        #
        # The labels are the hard decision on purpose. With random labels the
        # BCE term on this batch is ~25, and adding a ~1e-7 penalty difference
        # to 25.0 in float32 rounds it away: the always-tempering variant then
        # passes again. Keeping BCE near 0.01 keeps the difference visible in
        # the total, and the penalty is asserted on its own as well.
        raw_penalty = {
            "demographic_parity": lambda f, p, t, s: f._demographic_parity_penalty(p, s),
            "equalized_odds": lambda f, p, t, s: f._equalized_odds_penalty(p, t, s),
            "equal_opportunity": lambda f, p, t, s: f._equal_opportunity_penalty(p, t, s),
            "predictive_parity": lambda f, p, t, s: f._predictive_parity_penalty(p, t, s),
            "calibration": lambda f, p, t, s: f._calibration_penalty(p, t, s),
        }
        for seed in range(12):
            torch_mod.manual_seed(seed)
            p = torch_mod.sigmoid(torch_mod.randn(64) * 30)
            t = (p > 0.5).float()
            s = (torch_mod.rand(64) > 0.5).long()
            assert not bool((p == _temper_predictions(p, 1.0)).all()), (
                f"seed {seed} round-trips exactly, so it can pin nothing"
            )
            assert float(torch_mod.nn.functional.binary_cross_entropy(p, t)) < 0.1, seed
            for metric, ref_fn in raw_penalty.items():
                loss_fn = FairnessAwareBCELoss(
                    fairness_metric=metric, lambda_fairness=1.0, temperature=1.0
                )
                ref_penalty = ref_fn(loss_fn, p, t, s)
                assert float(loss_fn._compute_fairness_penalty(p, t, s).detach()) == float(
                    ref_penalty.detach()
                ), (metric, seed)
                got = float(loss_fn(p, t, s).detach())
                ref = float(
                    (torch_mod.nn.functional.binary_cross_entropy(p, t) + ref_penalty).detach()
                )
                assert got == ref, (metric, seed, got, ref)

    def test_limits(self, torch_mod):
        from vfairness.in_processing.loss_functions.fairness_losses import FairnessAwareBCELoss

        yp, yt, sa = _fixed_batch(torch_mod)
        bce = float(torch_mod.nn.functional.binary_cross_entropy(yp, yt))
        # T -> 0: the soft rates become the hard rates at 0.5
        cold = float(
            FairnessAwareBCELoss(lambda_fairness=1.0, temperature=0.01)(yp, yt, sa).detach()
        )
        hard = (yp > 0.5).float()
        hard_dp = sum(abs(hard[sa == k].mean() - hard.mean()) for k in (0, 1)) / 2
        assert cold - bce == pytest.approx(float(hard_dp), abs=5e-3)
        # T -> inf: every prediction flattens to 0.5, so the penalty vanishes
        hot = float(
            FairnessAwareBCELoss(lambda_fairness=1.0, temperature=100.0)(yp, yt, sa).detach()
        )
        assert hot - bce == pytest.approx(0.0, abs=1e-3)

    @pytest.mark.parametrize(
        "metric",
        [
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "predictive_parity",
            "calibration",
        ],
    )
    def test_reaches_every_metric_and_stays_differentiable(self, torch_mod, metric):
        from vfairness.in_processing.loss_functions.fairness_losses import FairnessAwareBCELoss

        yp, yt, sa = _fixed_batch(torch_mod)
        warm = float(
            FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=1.0, temperature=1.0)(
                yp, yt, sa
            ).detach()
        )
        leaf = yp.clone().requires_grad_(True)
        loss = FairnessAwareBCELoss(fairness_metric=metric, lambda_fairness=1.0, temperature=0.3)(
            leaf, yt, sa
        )
        assert float(loss.detach()) != warm
        loss.backward()
        assert leaf.grad is not None and bool(torch_mod.isfinite(leaf.grad).all())

    def test_non_positive_refused(self, torch_mod):
        from vfairness.in_processing.loss_functions.fairness_losses import FairnessAwareBCELoss

        with pytest.raises(ValueError, match="temperature must be > 0"):
            FairnessAwareBCELoss(temperature=0.0)

    def test_guard_covers_the_arithmetic_not_only_the_sign(self, torch_mod):
        """`> 0` was not enough: both ends of it produced silent nonsense.

        inf was accepted and removed the fairness penalty entirely (penalty
        exactly 0.0, zero gradient, i.e. plain BCE with nothing saying so),
        and any T that underflows the compute dtype divided by zero, leaving
        a plausible forward loss over NaN gradients (float32: fine at 1e-45,
        200 NaNs of 200 at 1e-46).
        """
        from vfairness.in_processing.loss_functions.fairness_losses import (
            TEMPERATURE_MIN,
            FairnessAwareBCELoss,
        )

        for bad in (
            float("inf"),
            float("-inf"),
            float("nan"),
            1e-300,
            1e-46,
            1e-45,
            TEMPERATURE_MIN / 2,
        ):
            with pytest.raises(ValueError, match="temperature must be > 0"):
                FairnessAwareBCELoss(temperature=bad)

        # Over-correction control: the band the guard admits still builds and
        # still differentiates, so this refuses nothing that worked.
        yp, yt, sa = _fixed_batch(torch_mod)
        for good in (TEMPERATURE_MIN, 1e-4, 0.01, 0.3, 1.0, 10.0, 100.0):
            leaf = yp.clone().requires_grad_(True)
            loss = FairnessAwareBCELoss(lambda_fairness=1.0, temperature=good)(leaf, yt, sa)
            assert bool(torch_mod.isfinite(loss.detach())), good
            loss.backward()
            assert leaf.grad is not None and bool(torch_mod.isfinite(leaf.grad).all()), good

    @pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
    def test_clamp_is_a_real_bound_in_every_dtype(self, torch_mod, dtype_name):
        """The docstring's "clamped away from 0 and 1" has to be true.

        With the old hardcoded eps=1e-7, 1.0 - eps rounded back to exactly 1.0
        in float16 and bfloat16, so the upper bound bounded nothing, logit(1.0)
        was inf and the gradient came back NaN.
        """
        from vfairness.in_processing.loss_functions.fairness_losses import _temper_predictions

        dtype = getattr(torch_mod, dtype_name)
        eps = torch_mod.finfo(dtype).eps
        assert float(torch_mod.tensor(1.0 - eps, dtype=dtype)) < 1.0
        # why the dtype-aware eps was needed, pinned so it cannot regress
        assert (float(torch_mod.tensor(1.0 - 1e-7, dtype=dtype)) == 1.0) is (
            dtype_name in ("float16", "bfloat16")
        )

        p = torch_mod.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=dtype, requires_grad=True)
        out = _temper_predictions(p, 0.5)
        assert bool(torch_mod.isfinite(out).all())
        out.sum().backward()
        assert p.grad is not None and bool(torch_mod.isfinite(p.grad).all())


# ===========================================================================
# F16: tradeoff_analysis was never read
# ===========================================================================


class TestF16TradeoffAnalysis:
    def test_parameter_is_gone(self, binary_data):
        y, _, g = binary_data
        an = FairnessTrainingAnalyzer(np.random.default_rng(0).normal(size=(len(y), 3)), y, g)
        assert "tradeoff_analysis" not in inspect.signature(an.generate_recommendation).parameters
        comps = [MethodComparison("m1", 0.8, 0.02, True)]
        with pytest.raises(TypeError, match="tradeoff_analysis"):
            an.generate_recommendation(comps, tradeoff_analysis={})
        with pytest.raises(TypeError):
            an.generate_recommendation(comps, {})

    def test_recommendation_and_full_analysis_control(self, binary_data):
        y, _, g = binary_data
        an = FairnessTrainingAnalyzer(np.random.default_rng(0).normal(size=(len(y), 3)), y, g)
        comps = [
            MethodComparison("m1", 0.80, 0.02, True),
            MethodComparison("m2", 0.90, 0.20, False),
            MethodComparison("m3", 0.85, 0.03, True),
        ]
        rec = an.generate_recommendation(comps)
        assert rec.recommended_method == "m3"
        assert rec.alternative_methods == ["m1"]
        # the internal call site after the removal still runs end to end
        report = an.full_analysis(base_estimator=None)
        assert report.recommendation.recommended_method == "Unconstrained"
