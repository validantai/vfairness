"""Beta Go-Live Stage 2, group s2g02: three states in the fairness LOSSES.

Ten findings, every one reproduced at the public entry before the fix, all in
``src/vfairness/in_processing/loss_functions/``. They share one shape: the
penalty TENSOR must stay a finite zero (it feeds a backward pass, and one NaN
poisons every gradient in the graph), and the code took that as licence to
report the same zero as a measurement. ``base.forward`` copies it into
``LossComponents.fairness_loss``, where 0.0 is the score a perfectly fair model
earns, and ``end_epoch`` averages it into ``TrainingMetrics.avg_fairness_loss``.

Measured before the fix, at the public entry:

    1. adversarial_debiasing   constant attribute -> fairness_loss 0.625611,
       and five seeds on IDENTICAL data gave 0.625611 / 0.710383 / 0.744421 /
       0.698632 / 0.695603, a spread of 0.119 straddling the healthy 0.694993:
       the number reported the adversary's random initialisation.
    2. fair_representation     no reconstruction pair -> regularization_loss
       0.0 and batch_metrics['reconstruction_loss'] 0.0, byte identical to a
       genuinely PERFECT reconstruction.
    3. causal_fairness         one-group batch -> 0.0, epoch average halved
       from 0.400000 to 0.200000; and three groups (0.5 / 0.5 / 1.0) -> 0.0,
       a maximal disparity reported as none.
    4. counterfactual_fairness one group -> exactly 0.0, because the
       "counterfactual" of x was x.
    5. individual_fairness     features omitted -> 0.0 where the real penalty
       was 0.514286; nine measured batches plus one unmeasurable reported
       0.462857.
    6. bounded_group_loss      one group -> 0.0, field for field identical to
       two groups whose losses are genuinely equal.
    7. fpr_parity              one group / no negatives / one measurable group
       -> 0.0 each, epoch average 0.100000 for a single measured 0.300000.
    8. create_fairness_loss    all five types, single group -> 0.000000.
    9. equal_opportunity       a group with no positives -> 0.0, identical to
       a truly fair batch; three groups reported the two-group lower bound.
   10. equalized_odds          an arm nobody could compare contributed 0.0 AND
       the total was still divided by the full group count: a measured 0.4 TPR
       disparity was reported as 0.2.

Every class carries a CONTROL: healthy data still measures, and measures the
same number as before. A fix that makes everything refuse is worse than the
defect it replaces.

Module-local fixtures on purpose: thirteen other agents are editing this
checkout, so no shared fixture, conftest or existing test file is touched.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")

from vfairness.in_processing import (  # noqa: E402
    AdversarialDebiasingLoss,
    BoundedGroupLoss,
    CausalFairnessLoss,
    CounterfactualFairnessLoss,
    EqualizedOddsLoss,
    EqualOpportunityLoss,
    FairRepresentationLoss,
    FalsePositiveRateParityLoss,
    IndividualFairnessLoss,
    create_fairness_loss,
)


def _is_nan(value) -> bool:
    """NaN is the only value that is not equal to itself."""
    return value != value


def _components(loss_fn, *args, **kwargs):
    """Call a loss at its public entry and return (total, components, warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total, comps = loss_fn(*args, return_components=True, **kwargs)
    return total, comps, [str(w.message) for w in caught]


def _assert_refused(comps, reason: str, messages) -> None:
    """The three-state contract, asserted the way a CALLER would read it."""
    assert _is_nan(comps.fairness_loss), (
        f"an unmeasured penalty must be NaN, not {comps.fairness_loss!r}"
    )
    assert comps.batch_metrics["fairness_penalty_assessed"] is False
    assert comps.batch_metrics["fairness_unassessable_reason"] == reason
    assert _is_nan(comps.to_dict()["fairness_loss"])
    assert any("NOT MEASURED" in m for m in messages), messages


def _assert_measured(comps) -> None:
    assert comps.batch_metrics["fairness_penalty_assessed"] is True
    assert comps.batch_metrics["fairness_unassessable_reason"] is None
    assert not _is_nan(comps.fairness_loss)


# ---------------------------------------------------------------------------
# 1. AdversarialDebiasingLoss
# ---------------------------------------------------------------------------


def _adversarial_batch(seed: int, n_groups: int, n: int = 64):
    torch.manual_seed(seed)
    y_pred = torch.rand(n, requires_grad=True)
    y_true = (torch.rand(n) > 0.5).float()
    sensitive = torch.zeros(n) if n_groups == 1 else (torch.arange(n) % 2).float()
    return y_pred, y_true, sensitive


class TestAdversarialDebiasingLoss:
    def test_a_constant_attribute_is_not_a_leakage_measurement(self):
        y_pred, y_true, sensitive = _adversarial_batch(0, n_groups=1)
        # the fixture really does exercise the branch under test
        assert int(torch.unique(sensitive).numel()) == 1

        _, comps, messages = _components(AdversarialDebiasingLoss(), y_pred, y_true, sensitive)

        _assert_refused(comps, "single_group", messages)
        assert comps.batch_metrics["fairness_groups_total"] == 1
        # and nothing was added to the total loss
        assert comps.total_loss == pytest.approx(comps.task_loss)

    def test_the_seed_sweep_that_exposed_it_no_longer_moves(self):
        """Five seeds, IDENTICAL single-group data. Before: 0.625611 ..
        0.744421, a spread of 0.119 that straddled the healthy value."""
        values = []
        for seed in range(5):
            y_pred, y_true, sensitive = _adversarial_batch(seed, n_groups=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, comps = AdversarialDebiasingLoss()(
                    y_pred, y_true, sensitive, return_components=True
                )
            values.append(comps.fairness_loss)
        assert all(_is_nan(v) for v in values), values

    def test_an_untrained_adversary_reports_its_initialisation_as_nothing(self):
        y_pred, y_true, sensitive = _adversarial_batch(0, n_groups=2)
        _, comps, messages = _components(AdversarialDebiasingLoss(), y_pred, y_true, sensitive)

        _assert_refused(comps, "adversary_at_initialization", messages)
        assert comps.batch_metrics["adversary_at_initialization"] is True
        # The MITIGATION is untouched: the penalty the optimizer saw is kept,
        # and it still reaches y_pred as a gradient.
        assert comps.batch_metrics["fairness_loss_unassessed_value"] > 0.0
        assert comps.total_loss > comps.task_loss

    def test_control_a_trained_adversary_on_two_groups_still_measures(self):
        y_pred, y_true, sensitive = _adversarial_batch(0, n_groups=2)
        loss_fn = AdversarialDebiasingLoss()
        loss_fn.update_adversary(y_pred.detach(), sensitive)

        total, comps, messages = _components(loss_fn, y_pred, y_true, sensitive)

        _assert_measured(comps)
        assert comps.fairness_loss > 0.0
        assert comps.batch_metrics["fairness_groups_compared"] == 2
        assert not any("NOT MEASURED" in m for m in messages), messages
        # still differentiable, and the fairness term still moves y_pred
        total.backward()
        assert y_pred.grad is not None
        assert bool(torch.isfinite(y_pred.grad).all())
        assert float(y_pred.grad.abs().max()) > 0.0

    def test_the_refused_batch_is_still_finite_and_differentiable(self):
        y_pred, y_true, sensitive = _adversarial_batch(0, n_groups=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total = AdversarialDebiasingLoss()(y_pred, y_true, sensitive)
        assert bool(torch.isfinite(total))
        total.backward()
        assert bool(torch.isfinite(y_pred.grad).all())


# ---------------------------------------------------------------------------
# 2. FairRepresentationLoss
# ---------------------------------------------------------------------------


def _representation_batch(n: int = 32):
    torch.manual_seed(0)
    y_pred = torch.sigmoid(torch.randn(n, 1))
    y_true = torch.randint(0, 2, (n, 1)).float()
    sensitive = torch.randint(0, 2, (n,))
    x = torch.randn(n, 5)
    return y_pred, y_true, sensitive, x


class TestFairRepresentationLoss:
    def test_a_reconstruction_nobody_computed_is_not_a_perfect_one(self):
        y_pred, y_true, sensitive, _ = _representation_batch()
        loss_fn = FairRepresentationLoss(lambda_fairness=1.0, alpha_reconstruction=0.5)

        _, comps, messages = _components(loss_fn, y_pred, y_true, sensitive)

        assert _is_nan(comps.regularization_loss)
        assert comps.batch_metrics["reconstruction_loss"] is None
        assert comps.batch_metrics["reconstruction_assessed"] is False
        assert _is_nan(comps.to_dict()["regularization_loss"])
        assert any("reconstruction term is NOT MEASURED" in m for m in messages), messages

    def test_control_a_genuinely_perfect_reconstruction_is_a_measured_zero(self):
        y_pred, y_true, sensitive, x = _representation_batch()
        loss_fn = FairRepresentationLoss(lambda_fairness=1.0, alpha_reconstruction=0.5)

        _, perfect, _ = _components(
            loss_fn, y_pred, y_true, sensitive, x_reconstructed=x.clone(), x_original=x
        )
        _, real, _ = _components(
            loss_fn, y_pred, y_true, sensitive, x_reconstructed=x + 1.0, x_original=x
        )

        assert perfect.regularization_loss == 0.0
        assert perfect.batch_metrics["reconstruction_loss"] == 0.0
        assert perfect.batch_metrics["reconstruction_assessed"] is True
        assert real.regularization_loss == pytest.approx(1.0)
        assert real.batch_metrics["reconstruction_assessed"] is True

    def test_half_a_reconstruction_pair_is_refused_not_scored_zero(self):
        y_pred, y_true, sensitive, x = _representation_batch()
        loss_fn = FairRepresentationLoss()
        with pytest.raises(ValueError, match="must be supplied together"):
            loss_fn(y_pred, y_true, sensitive, x_reconstructed=x.clone())

    def test_a_substituted_representation_says_so(self):
        y_pred, y_true, sensitive, _ = _representation_batch()
        loss_fn = FairRepresentationLoss()
        _, comps, messages = _components(loss_fn, y_pred, y_true, sensitive)
        assert comps.batch_metrics["representation_supplied"] is False
        assert any("adversary is being run on the task predictions" in m for m in messages)


# ---------------------------------------------------------------------------
# 3. CausalFairnessLoss
# ---------------------------------------------------------------------------


def _causal_batch(n: int = 30):
    y_true = torch.cat([torch.ones(n), torch.zeros(n)])
    y_pred = torch.cat([torch.full((n,), 0.30), torch.full((n,), 0.70)])
    return y_pred, y_true


class TestCausalFairnessLoss:
    def test_a_one_group_batch_is_not_a_zero_causal_effect(self):
        n = 30
        y_pred, y_true = _causal_batch(n)
        loss_fn = CausalFairnessLoss(lambda_fairness=1.0, track_metrics=True)

        _, one_group, messages = _components(loss_fn, y_pred, y_true, torch.zeros(2 * n))
        _assert_refused(one_group, "single_group", messages)

        _, two_group, _ = _components(
            loss_fn, y_pred, y_true, torch.cat([torch.zeros(n), torch.ones(n)])
        )
        _assert_measured(two_group)
        assert two_group.fairness_loss == pytest.approx(0.4, abs=1e-6)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = loss_fn.end_epoch()
        # was 0.2: the unmeasurable batch halved the epoch's causal effect
        assert epoch.avg_fairness_loss == pytest.approx(0.4, abs=1e-6)
        assert epoch.convergence_info["n_batches_fairness_assessed"] == 1
        assert epoch.convergence_info["n_batches_fairness_unassessable"] == 1
        assert epoch.convergence_info["fairness_unassessable_reasons"] == {"single_group": 1}

    def test_a_third_group_at_the_ceiling_is_no_longer_invisible(self):
        """|mean[0] - mean[1]| compared the first two groups and no others."""
        n = 30
        y_pred = torch.cat([torch.full((n,), 0.5), torch.full((n,), 0.5), torch.full((n,), 1.0)])
        y_true = torch.cat([torch.ones(n), torch.zeros(n), torch.ones(n)])
        sensitive = torch.cat([torch.zeros(n), torch.ones(n), torch.full((n,), 2.0)])
        assert int(torch.unique(sensitive).numel()) == 3

        _, comps, _ = _components(
            CausalFairnessLoss(lambda_fairness=1.0), y_pred, y_true, sensitive
        )

        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.5, abs=1e-6)  # was 0.0
        assert comps.batch_metrics["fairness_groups_compared"] == 3

    def test_control_two_groups_are_measured_exactly_as_before(self):
        n = 30
        y_pred, y_true = _causal_batch(n)
        sensitive = torch.cat([torch.zeros(n), torch.ones(n)])
        _, comps, messages = _components(
            CausalFairnessLoss(lambda_fairness=1.0), y_pred, y_true, sensitive
        )
        assert comps.fairness_loss == pytest.approx(0.4, abs=1e-6)
        assert not any("NOT MEASURED" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 4. CounterfactualFairnessLoss
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


def _counterfactual_batch(n_groups: int, n: int = 200):
    torch.manual_seed(0)
    model = _TinyModel()
    features = torch.randn(n, 4)
    sensitive = (
        torch.zeros(n, dtype=torch.long) if n_groups == 1 else (torch.arange(n) % 2).to(torch.long)
    )
    y_true = (torch.rand(n) > 0.5).float()
    return model, features, sensitive, y_true


class TestCounterfactualFairnessLoss:
    @pytest.mark.parametrize("strategy", ["group_mean", "group_swap", "adversarial"])
    def test_the_counterfactual_of_x_being_x_is_not_perfect_fairness(self, strategy):
        model, features, sensitive, y_true = _counterfactual_batch(n_groups=1)
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0, counterfactual_strategy=strategy)
        # the generator refuses rather than cloning
        assert loss_fn._generate_counterfactuals(features, sensitive) is None

        _, comps, messages = _components(
            loss_fn, model(features), y_true, sensitive, features=features, model=model
        )

        _assert_refused(comps, "single_group", messages)
        assert comps.batch_metrics["counterfactual_strategy"] == strategy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = loss_fn.end_epoch()
        assert _is_nan(epoch.avg_fairness_loss)
        assert epoch.to_dict()["convergence_info"]["n_batches_fairness_unassessable"] == 1

    def test_a_custom_generator_that_returns_the_factual_is_refused(self):
        """The second, independent guard. The built-in strategies refuse at
        the generator (they return None for a single group), so reverting
        this check alone leaves that one standing; a CUSTOM generator is the
        input that reaches it, on a healthy TWO-group batch."""
        model, features, sensitive, y_true = _counterfactual_batch(n_groups=2)
        assert int(torch.unique(sensitive).numel()) == 2
        loss_fn = CounterfactualFairnessLoss(
            lambda_fairness=1.0,
            counterfactual_strategy="custom",
            counterfactual_generator=lambda feats, sens: feats.clone(),
        )
        _, comps, messages = _components(
            loss_fn, model(features), y_true, sensitive, features=features, model=model
        )
        _assert_refused(comps, "counterfactual_equals_factual", messages)

    def test_no_counterfactual_and_no_model_is_refused_too(self):
        model, features, sensitive, y_true = _counterfactual_batch(n_groups=2)
        _, comps, messages = _components(
            CounterfactualFairnessLoss(), model(features), y_true, sensitive
        )
        _assert_refused(comps, "no_counterfactual_supplied", messages)

    def test_control_two_groups_still_measure_the_same_number(self):
        model, features, sensitive, y_true = _counterfactual_batch(n_groups=2)
        loss_fn = CounterfactualFairnessLoss(lambda_fairness=1.0)
        generated = loss_fn._generate_counterfactuals(features, sensitive)
        assert generated is not None
        assert not bool(torch.equal(generated, features))

        _, comps, messages = _components(
            loss_fn, model(features), y_true, sensitive, features=features, model=model
        )
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.021326439455151558, rel=1e-6)
        assert not any("NOT MEASURED" in m for m in messages), messages


# ---------------------------------------------------------------------------
# 5. IndividualFairnessLoss
# ---------------------------------------------------------------------------


def _individual_batch():
    """Eight IDENTICAL individuals, scored 0.95 four times and 0.05 four."""
    features = torch.ones(8, 3)
    y_pred = torch.tensor([0.95] * 4 + [0.05] * 4)
    y_true = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    sensitive = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return features, y_pred, y_true, sensitive


class TestIndividualFairnessLoss:
    def test_features_omitted_is_not_perfect_individual_fairness(self):
        features, y_pred, y_true, sensitive = _individual_batch()
        _, comps, messages = _components(IndividualFairnessLoss(), y_pred, y_true, sensitive)
        _assert_refused(comps, "features_not_provided", messages)

    def test_a_single_sample_is_refused_and_no_longer_silent(self):
        features, y_pred, y_true, sensitive = _individual_batch()
        _, comps, messages = _components(
            IndividualFairnessLoss(),
            y_pred[:1],
            y_true[:1],
            sensitive[:1],
            features=features[:1],
        )
        _assert_refused(comps, "fewer_than_two_samples", messages)
        assert comps.batch_metrics["n_samples"] == 1

    def test_control_with_features_the_violation_is_measured_as_before(self):
        features, y_pred, y_true, sensitive = _individual_batch()
        _, comps, messages = _components(
            IndividualFairnessLoss(), y_pred, y_true, sensitive, features=features
        )
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.514285683631897, rel=1e-6)
        assert comps.batch_metrics["n_samples"] == 8
        assert not any("NOT MEASURED" in m for m in messages), messages

    def test_a_mixed_epoch_averages_only_what_it_measured(self):
        """Nine measured batches plus one unmeasurable reported 0.462857
        against a true 0.514286: a manufactured 10 percent improvement."""
        features, y_pred, y_true, sensitive = _individual_batch()
        loss_fn = IndividualFairnessLoss()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(9):
                loss_fn(y_pred, y_true, sensitive, features=features, return_components=True)
            loss_fn(y_pred, y_true, sensitive, return_components=True)
            epoch = loss_fn.end_epoch()

        assert epoch.avg_fairness_loss == pytest.approx(0.514285683631897, rel=1e-6)
        info = epoch.to_dict()["convergence_info"]
        assert info["n_batches"] == 10
        assert info["n_batches_fairness_assessed"] == 9
        assert info["fairness_unassessable_reasons"] == {"features_not_provided": 1}


# ---------------------------------------------------------------------------
# 6. BoundedGroupLoss
# ---------------------------------------------------------------------------


def _assert_bounded_healthy_control():
    torch.manual_seed(0)
    n = 64
    y_pred = torch.rand(n)
    y_true = (torch.rand(n) > 0.5).float()
    sensitive = (torch.arange(n) % 2).float()

    # Independent two-group identity: sample variance is (a - b)^2 / 2.
    # Use native BCE primitives, not vfairness or a recorded backend's scalar:
    # float32 reductions vary slightly across CPU/PyTorch implementations.
    a, b = [
        torch.nn.functional.binary_cross_entropy(y_pred[sensitive == g], y_true[sensitive == g])
        for g in (0, 1)
    ]
    variance = (a - b).square() / 2
    ratio_excess = (torch.maximum(a, b) / (torch.minimum(a, b) + 1e-8) - 1.5).clamp_min(0)
    expected = float(variance + ratio_excess)
    assert expected > 0.0, "the healthy fixture must detect a zero-penalty regression"

    _, comps, messages = _components(
        BoundedGroupLoss(), y_pred.clone().requires_grad_(True), y_true, sensitive
    )
    assert comps.fairness_loss == pytest.approx(expected, rel=1e-6), "healthy numerical control"
    _assert_measured(comps)
    assert not any("NOT MEASURED" in m for m in messages), messages


class TestBoundedGroupLoss:
    def test_one_group_is_not_perfectly_equal_group_losses(self):
        torch.manual_seed(0)
        n = 64
        y_pred = torch.rand(n)
        y_true = (torch.rand(n) > 0.5).float()

        _, comps, messages = _components(
            BoundedGroupLoss(), y_pred.clone().requires_grad_(True), y_true, torch.zeros(n)
        )
        _assert_refused(comps, "single_group", messages)

    def test_a_measured_zero_and_an_unmeasured_one_are_distinguishable(self):
        """The two records used to be identical field for field."""
        y_pred = torch.cat([torch.full((16,), 0.8), torch.full((16,), 0.2)] * 2)
        y_true = torch.cat([torch.ones(16), torch.zeros(16)] * 2)
        equal_groups = torch.cat([torch.zeros(32), torch.ones(32)])

        _, measured, _ = _components(
            BoundedGroupLoss(), y_pred.clone().requires_grad_(True), y_true, equal_groups
        )
        _, refused, messages = _components(
            BoundedGroupLoss(),
            y_pred.clone().requires_grad_(True),
            y_true,
            torch.zeros(64),
        )

        assert measured.fairness_loss == pytest.approx(0.0, abs=1e-9)
        _assert_measured(measured)
        _assert_refused(refused, "single_group", messages)
        assert measured.to_dict() != refused.to_dict()

    def test_control_two_unequal_groups_still_measure_the_same_number(self):
        _assert_bounded_healthy_control()


# ---------------------------------------------------------------------------
# 7. FalsePositiveRateParityLoss
# ---------------------------------------------------------------------------


def _fpr_batches(n: int = 30):
    y_pred = torch.cat([torch.full((n,), 0.8), torch.full((n,), 0.2)])
    sensitive = torch.cat([torch.zeros(n), torch.ones(n)])
    return y_pred, sensitive


class TestFalsePositiveRateParityLoss:
    def test_the_three_unmeasurable_batches_refuse_and_the_epoch_excludes_them(self):
        n = 30
        y_pred, sensitive = _fpr_batches(n)
        loss_fn = FalsePositiveRateParityLoss(lambda_fairness=0.5, track_metrics=True)

        _, measured, _ = _components(loss_fn, y_pred, torch.zeros(2 * n), sensitive)
        _, one_group, m1 = _components(loss_fn, y_pred, torch.zeros(2 * n), torch.zeros(2 * n))
        _, no_negatives, m2 = _components(loss_fn, y_pred, torch.ones(2 * n), sensitive)

        assert measured.fairness_loss == pytest.approx(0.3, abs=1e-6)
        _assert_measured(measured)
        _assert_refused(one_group, "single_group", m1)
        _assert_refused(no_negatives, "fewer_than_two_groups_with_negative_labels", m2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = loss_fn.end_epoch()
        # was 0.100: two batches that measured nothing were arithmetic in it
        assert epoch.avg_fairness_loss == pytest.approx(0.3, abs=1e-6)
        assert epoch.convergence_info["n_batches_fairness_assessed"] == 1

    def test_one_measurable_group_is_a_tautology_not_a_zero(self):
        """With one group, overall_fpr IS that group's rate: |f - f| = 0."""
        n = 30
        y_pred, sensitive = _fpr_batches(n)
        y_true = torch.cat([torch.zeros(n), torch.ones(n)])  # group 1 has no negatives
        _, comps, messages = _components(
            FalsePositiveRateParityLoss(lambda_fairness=0.5), y_pred, y_true, sensitive
        )
        _assert_refused(comps, "fewer_than_two_groups_with_negative_labels", messages)
        assert comps.batch_metrics["fairness_groups_compared"] == 1
        assert comps.batch_metrics["fairness_unmeasurable_groups"] == [1.0]


# ---------------------------------------------------------------------------
# 8. create_fairness_loss (the factory hands back the same disclosure)
# ---------------------------------------------------------------------------


_FACTORY_HEALTHY = (
    "demographic_parity",
    "equalized_odds",
    "equal_opportunity",
    "predictive_parity",
    "calibration",
)


def _two_group_factory_reference(metric, y_pred, y_true, sensitive):
    """An independent two-group identity, without production loss helpers.

    The overall rate lies between the two group rates, so their mean absolute
    distance from it is half their gap, even when group sizes differ. Equalized
    odds averages that gap over its two arms. Compute native float32 group
    statistics: pinning a tiny gap to another backend's rounded means magnifies
    harmless reduction differences beyond the unchanged 1e-6 relative tolerance.
    """
    assert torch.unique(sensitive).tolist() == [0, 1]

    def half_gap(values, selected=None):
        if selected is None:
            selected = torch.ones_like(sensitive, dtype=torch.bool)
        groups = [values[(sensitive == g) & selected] for g in (0, 1)]
        assert all(group.numel() > 0 for group in groups)
        return (groups[0].mean() - groups[1].mean()).abs() / 2

    if metric == "demographic_parity":
        return float(half_gap(y_pred))
    if metric == "equal_opportunity":
        return float(half_gap(y_pred, y_true == 1))
    if metric == "equalized_odds":
        return float((half_gap(y_pred, y_true == 1) + half_gap(y_pred, y_true == 0)) / 2)
    if metric == "calibration":
        return float(half_gap((y_pred - y_true).abs()))
    if metric == "predictive_parity":
        precisions = []
        for g in (0, 1):
            p, y = y_pred[sensitive == g], y_true[sensitive == g]
            assert float(p.sum()) > 0.0
            precisions.append((p * y).sum() / (p.sum() + 1e-8))
        return float((precisions[0] - precisions[1]).abs() / 2)
    raise AssertionError(f"no independent reference for {metric}")


def _assert_factory_healthy_control(metric):
    torch.manual_seed(0)
    y_pred = torch.rand(200)
    y_true = (torch.rand(200) > 0.5).float()
    sensitive = (torch.arange(200) % 2).to(torch.long)
    expected = _two_group_factory_reference(metric, y_pred, y_true, sensitive)
    assert expected > 0.0, "the healthy fixture must detect a zero-penalty regression"

    loss_fn = create_fairness_loss(metric, lambda_fairness=0.1)
    _, comps, messages = _components(loss_fn, y_pred, y_true, sensitive)

    _assert_measured(comps)
    assert comps.fairness_loss == pytest.approx(expected, rel=1e-6), "healthy numerical control"
    assert not any("NOT MEASURED" in m for m in messages), messages


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("demographic_parity", 1 / 32),
        ("equal_opportunity", 1 / 16),
        ("equalized_odds", 5 / 48),
        ("calibration", 1 / 8),
        ("predictive_parity", 17 / 72),
    ],
)
def test_two_group_reference_agrees_with_hand_calculated_fixture(metric, expected):
    # Group means: 1/2, 9/16; TPR: 3/4, 5/8; FPR: 1/4, 13/24;
    # absolute errors: 1/4, 1/2; soft precision: 3/4, 5/18.
    # These inputs are exactly representable, and the unequal label counts
    # also check the half-gap identity when the overall rate is not a midpoint.
    y_pred = torch.tensor([0.875, 0.625, 0.375, 0.125, 0.625, 0.875, 0.5, 0.25])
    y_true = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    sensitive = torch.tensor([0] * 4 + [1] * 4)
    actual = _two_group_factory_reference(metric, y_pred, y_true, sensitive)
    assert actual == pytest.approx(expected, rel=1e-6)


class TestCreateFairnessLoss:
    @pytest.mark.parametrize("metric", sorted(_FACTORY_HEALTHY))
    def test_every_factory_type_refuses_a_single_group_batch(self, metric):
        torch.manual_seed(0)
        y_pred = torch.rand(200)
        y_true = (torch.rand(200) > 0.5).float()
        sensitive = torch.zeros(200, dtype=torch.long)

        loss_fn = create_fairness_loss(metric, lambda_fairness=0.1)
        _, comps, messages = _components(loss_fn, y_pred, y_true, sensitive)

        _assert_refused(comps, "single_group", messages)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert _is_nan(loss_fn.end_epoch().avg_fairness_loss)

    @pytest.mark.parametrize("metric", sorted(_FACTORY_HEALTHY))
    def test_control_two_groups_measure_exactly_what_they_measured_before(self, metric):
        _assert_factory_healthy_control(metric)


@pytest.mark.parametrize("metric", ("bounded_group_loss", *_FACTORY_HEALTHY))
@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0], ids=["zero", "halved", "doubled"])
def test_healthy_numerical_controls_reject_wrong_penalties(monkeypatch, metric, factor):
    """The same controls must fail if a measured penalty is zeroed or rescaled."""
    loss_type = (
        BoundedGroupLoss if metric == "bounded_group_loss" else type(create_fairness_loss(metric))
    )
    original = loss_type._compute_fairness_penalty

    def wrong_penalty(self, *args, **kwargs):
        return original(self, *args, **kwargs) * factor

    monkeypatch.setattr(loss_type, "_compute_fairness_penalty", wrong_penalty)
    with pytest.raises(AssertionError, match="healthy numerical control"):
        if metric == "bounded_group_loss":
            _assert_bounded_healthy_control()
        else:
            _assert_factory_healthy_control(metric)


# ---------------------------------------------------------------------------
# 9. EqualOpportunityLoss
# ---------------------------------------------------------------------------


def _tensors(y_pred, y_true, sensitive):
    return (
        torch.tensor(y_pred),
        torch.tensor(y_true, dtype=torch.float32),
        torch.tensor(sensitive),
    )


class TestEqualOpportunityLoss:
    def test_a_group_with_no_positives_cannot_be_compared(self):
        args = _tensors([0.2] * 10 + [0.9] * 10, [1] * 10 + [0] * 10, [0] * 10 + [1] * 10)
        _, comps, messages = _components(EqualOpportunityLoss(lambda_fairness=0.1), *args)
        _assert_refused(comps, "fewer_than_two_groups_with_positive_labels", messages)
        assert comps.batch_metrics["fairness_groups_compared"] == 1

    def test_control_a_truly_fair_batch_is_a_measured_zero(self):
        args = _tensors([0.5] * 20, [1] * 20, [0] * 10 + [1] * 10)
        _, comps, messages = _components(EqualOpportunityLoss(lambda_fairness=0.1), *args)
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.0, abs=1e-9)
        assert comps.batch_metrics["fairness_groups_compared"] == 2
        assert not any("NOT MEASURED" in m for m in messages), messages

    def test_a_dropped_third_group_is_disclosed_as_partial_coverage(self):
        """A(tpr .9) B(tpr .1) C(no positives) gave 0.400000, identical to the
        same batch with C deleted, and nothing said C was left out."""
        args = _tensors(
            [0.9] * 10 + [0.1] * 10 + [0.5] * 10,
            [1] * 20 + [0] * 10,
            [0] * 10 + [1] * 10 + [2] * 10,
        )
        _, comps, messages = _components(EqualOpportunityLoss(lambda_fairness=0.1), *args)

        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.4, abs=1e-6)
        assert comps.batch_metrics["fairness_groups_total"] == 3
        assert comps.batch_metrics["fairness_groups_compared"] == 2
        assert comps.batch_metrics["fairness_unmeasurable_groups"] == [2]
        assert comps.batch_metrics["fairness_penalty_partial"] is True
        assert any("no positive labels" in m for m in messages), messages

    def test_control_the_two_group_batch_measures_the_same_number(self):
        args = _tensors([0.9] * 10 + [0.1] * 10, [1] * 20, [0] * 10 + [1] * 10)
        _, comps, _ = _components(EqualOpportunityLoss(lambda_fairness=0.1), *args)
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.4, abs=1e-6)
        assert comps.batch_metrics.get("fairness_penalty_partial") is None


# ---------------------------------------------------------------------------
# 10. EqualizedOddsLoss
# ---------------------------------------------------------------------------


class TestEqualizedOddsLoss:
    def test_an_arm_nobody_could_compare_is_not_a_satisfied_arm(self):
        args = _tensors([0.9] * 5 + [0.5] * 15, [1] * 5 + [0] * 15, [0] * 10 + [1] * 10)
        _, comps, messages = _components(EqualizedOddsLoss(lambda_fairness=0.1), *args)

        _assert_refused(comps, "tpr_arm_not_comparable", messages)
        # the arm that WAS measured is kept rather than deleted
        assert comps.batch_metrics["equalized_odds_fpr_disparity"] == pytest.approx(0.0, abs=1e-9)
        assert comps.batch_metrics["equalized_odds_tpr_disparity"] is None
        assert comps.batch_metrics["equalized_odds_arms_compared"] == ["fpr"]

    def test_a_missing_arm_no_longer_halves_the_arm_that_was_measured(self):
        """Every label is 1, so no FPR exists anywhere. The measured per-group
        TPR disparity is 0.400000; the old code reported 0.200000 because the
        sum was still divided by the full group count across BOTH arms."""
        args = _tensors([0.9] * 10 + [0.1] * 10, [1] * 20, [0] * 10 + [1] * 10)
        _, comps, messages = _components(EqualizedOddsLoss(lambda_fairness=0.1), *args)

        _assert_refused(comps, "fpr_arm_not_comparable", messages)
        assert comps.batch_metrics["equalized_odds_tpr_disparity"] == pytest.approx(0.4, abs=1e-6)
        assert comps.batch_metrics["equalized_odds_fpr_disparity"] is None

    def test_control_a_truly_fair_batch_and_a_healthy_one_still_measure(self):
        fair = _tensors([0.5] * 20, [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5, [0] * 10 + [1] * 10)
        _, comps, messages = _components(EqualizedOddsLoss(lambda_fairness=0.1), *fair)
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.0, abs=1e-9)
        assert comps.batch_metrics["equalized_odds_arms_compared"] == ["fpr", "tpr"]
        assert not any("NOT MEASURED" in m for m in messages), messages

        healthy = _tensors(
            [0.9] * 5 + [0.8] * 5 + [0.1] * 5 + [0.2] * 5,
            [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5,
            [0] * 10 + [1] * 10,
        )
        _, comps, _ = _components(EqualizedOddsLoss(lambda_fairness=0.1), *healthy)
        _assert_measured(comps)
        assert comps.fairness_loss == pytest.approx(0.3499999940395355, rel=1e-6)


# ---------------------------------------------------------------------------
# Over-correction control across the whole group: a refusal must never break
# the training loop it sits in.
# ---------------------------------------------------------------------------


class TestRefusalsDoNotPoisonTraining:
    @pytest.mark.parametrize(
        "make_loss",
        [
            lambda: BoundedGroupLoss(lambda_fairness=1.0),
            lambda: EqualOpportunityLoss(lambda_fairness=1.0),
            lambda: EqualizedOddsLoss(lambda_fairness=1.0),
            lambda: FalsePositiveRateParityLoss(lambda_fairness=1.0),
            lambda: CausalFairnessLoss(lambda_fairness=1.0),
            lambda: create_fairness_loss("demographic_parity", lambda_fairness=1.0),
        ],
    )
    def test_a_refused_batch_still_returns_a_finite_differentiable_loss(self, make_loss):
        torch.manual_seed(0)
        raw = torch.randn(16, requires_grad=True)
        y_pred = torch.sigmoid(raw)
        y_true = (torch.arange(16) % 2).float()
        sensitive = torch.zeros(16)  # one group: nothing is measurable

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            total = make_loss()(y_pred, y_true, sensitive)

        assert bool(torch.isfinite(total)), "the steering tensor must stay finite"
        total.backward()
        assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())
