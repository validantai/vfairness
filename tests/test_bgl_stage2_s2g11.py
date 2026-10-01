"""Stage 2 group s2g11: the regularizers and constraints must not report 0.0 for "I did not look".

Every regularizer in ``vfairness.in_processing.regularizers`` scores dependence
on a scale where **0.0 is the clean end**. HSIC "is zero iff the variables are
independent"; a Pearson correlation of 0.0 is exactly "predictions carry no
information about the sensitive attribute"; a group-parity penalty of 0.0 is
"every group agrees with the population". So a 0.0 returned because nothing was
measurable is not a missing value, it is the strongest possible positive
finding, asserted from no data.

The eight findings this file pins, with the value each one actually returned
before the fix (run at the public entry, never at a helper):

    hilbert_schmidt_regularizer   n=0 and n=1 -> dependence_measure 0.0, no warning
    correlation_penalty           empty / single group / constant scores /
                                  one label -> dependence_measure 0.0, no warning
    fairness_regularization       single group -> 0.0 with group_penalties
                                  {'g0': 0.0}; empty batch -> 0.0; 'eop' with no
                                  positive label anywhere -> 0.0
    conditional_independence_reg  single group -> 0.0; empty -> 0.0
    equal_opportunity_constraint  is_satisfied(single group) -> False, identical
                                  to a measured breach
    fpr_parity_constraint         same
    fair_classifier              single group -> fairness_violation 0.0,
                                  constraint_satisfied True, score -0.0
    create_constraint            'demographic_parity' single group -> 0.0,
                                  is_satisfied True, could_not_evaluate False

The last four were closed by Stage 1 (commit 2b0fa20) and are pinned here
because nothing in the suite held them: the sabotage log in the group report
records what each of these tests does when the fix is reverted.

Each capability gets BOTH halves: the refusal, and a CONTROL on healthy data
that must still produce the real measurement. A fix that makes everything
refuse passes any test that only checks the degenerate case.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _call(fn, *args, **kwargs):
    """Run a regularizer and return (penalty, metrics, warning messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        penalty, metrics = fn(*args, **kwargs)
    return penalty, metrics, [str(w.message) for w in caught]


def _refusal_warning(messages):
    return [m for m in messages if "could not check" in m]


def _assert_zero_grad_penalty(penalty):
    """The penalty tensor must stay a differentiable 0.0 so backprop survives.

    This is the half of the fix that must NOT change: a short final batch
    (DataLoader with drop_last=False) or a single-group shard has to keep
    training, and a zero contribution is the right optimisation behaviour when
    there is nothing to push on. Only the REPORT becomes NaN.
    """
    assert float(penalty.detach()) == 0.0
    assert penalty.requires_grad or penalty.grad_fn is not None
    # and it really is usable: this raises if the graph is broken
    penalty.backward()


# ===========================================================================
# Finding 1 - hilbert_schmidt_regularizer
# public entry: HilbertSchmidtRegularizer.forward(..., return_metrics=True)
# ===========================================================================


class TestHilbertSchmidtRegularizer:
    @pytest.mark.parametrize("n", [0, 1])
    def test_fewer_than_two_rows_is_not_independence(self, n):
        """BEFORE: dependence_measure 0.0 with zero warnings. AFTER: NaN."""
        from vfairness.in_processing import HilbertSchmidtRegularizer

        reg = HilbertSchmidtRegularizer()
        y_pred = torch.rand(n, requires_grad=True)
        sensitive = (torch.rand(n) > 0.5).float()

        penalty, metrics, messages = _call(reg, y_pred, sensitive, return_metrics=True)

        assert math.isnan(metrics.dependence_measure), (
            "HSIC over fewer than two samples has no estimate; 0.0 here reads as "
            "the class docstring's 'zero iff the variables are independent'"
        )
        assert metrics.measured is False
        assert metrics.to_dict()["measured"] is False
        assert math.isnan(metrics.to_dict()["dependence_measure"])
        # the fixture really did reach the branch under test, not some other one
        assert metrics.metadata["not_assessed"] == "n_samples_lt_2"
        assert metrics.metadata["n_samples"] == n
        assert _refusal_warning(messages), "the refusal must be audible, not only in the field"
        _assert_zero_grad_penalty(penalty)

    def test_control_healthy_batch_still_measures(self):
        from vfairness.in_processing import HilbertSchmidtRegularizer

        torch.manual_seed(0)
        reg = HilbertSchmidtRegularizer()
        y_pred = torch.rand(100, requires_grad=True)
        sensitive = (torch.rand(100) > 0.5).float()

        _penalty, metrics, messages = _call(reg, y_pred, sensitive, return_metrics=True)

        assert metrics.measured is True
        assert math.isfinite(metrics.dependence_measure)
        assert "not_assessed" not in metrics.metadata
        assert metrics.metadata["kernel"] == "rbf"
        assert not _refusal_warning(messages), "healthy data must not refuse"

    def test_control_dependent_batch_scores_higher_than_independent(self):
        """The measurement still discriminates: HSIC must rank a dependent
        batch above an independent one, or 'it measures' means nothing."""
        from vfairness.in_processing import HilbertSchmidtRegularizer

        torch.manual_seed(0)
        sensitive = (torch.rand(400) > 0.5).float()
        dependent = sensitive * 0.8 + 0.1  # prediction IS the group
        independent = torch.rand(400)

        _p, dep_metrics = HilbertSchmidtRegularizer()(dependent, sensitive, return_metrics=True)
        _p, ind_metrics = HilbertSchmidtRegularizer()(independent, sensitive, return_metrics=True)

        assert dep_metrics.dependence_measure > 10 * ind_metrics.dependence_measure


# ===========================================================================
# Finding 2 - correlation_penalty
# public entry: CorrelationPenalty.forward(..., return_metrics=True)
# ===========================================================================


def _correlation_cases():
    return {
        "empty": (torch.tensor([]), torch.tensor([]), "fewer_than_two_finite_rows"),
        "single_group": (torch.rand(30), torch.zeros(30), "single_group"),
        "constant_scores": (
            torch.full((30,), 0.7),
            (torch.arange(30) % 2).float(),
            "constant_predictions",
        ),
        "one_label_only": (torch.rand(30), torch.ones(30), "single_group"),
    }


class TestCorrelationPenalty:
    @pytest.mark.parametrize("label", sorted(_correlation_cases()))
    def test_undefined_correlation_is_not_zero_correlation(self, label):
        """BEFORE: 0.0 (or -4.3e-10) with zero warnings. AFTER: NaN."""
        from vfairness.in_processing import CorrelationPenalty

        y_pred, sensitive, expected_reason = _correlation_cases()[label]
        y_pred = y_pred.clone().requires_grad_(True)

        penalty, metrics, messages = _call(
            CorrelationPenalty(strength=0.1), y_pred, sensitive, return_metrics=True
        )

        assert math.isnan(metrics.dependence_measure), (
            "an undefined 0/0 Pearson must not arrive as 0.0, the value that "
            "means 'predictions are perfectly independent of the group'"
        )
        assert metrics.measured is False
        assert metrics.metadata["not_assessed"] == expected_reason
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    def test_the_naive_guard_would_have_missed_the_constant_vector(self):
        """Standing trap 1, asserted rather than assumed.

        ``(centred ** 2).sum() == 0`` is the guard that looks right for a
        constant prediction vector and is not: the accumulated statistic is
        ~1e-10, not exactly zero. This test fails if that ever stops being
        true, which would mean the fix is being held up by luck.
        """
        v = torch.full((30,), 0.7)
        centred_sq_sum = float(((v - v.mean()) ** 2).sum())
        assert centred_sq_sum != 0.0, "the naive guard would now work by accident"
        assert centred_sq_sum < 1e-6

        from vfairness.in_processing.regularizers.fairness_regularizers import _has_spread

        assert _has_spread(v) is False
        assert _has_spread(torch.tensor([0.1, 0.2, 0.2])) is True

    def test_control_perfect_correlation_still_reads_one(self):
        from vfairness.in_processing import CorrelationPenalty

        x = torch.arange(30).float().requires_grad_(True)
        _penalty, metrics, messages = _call(
            CorrelationPenalty(strength=0.1), x, torch.arange(30).float(), return_metrics=True
        )

        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(1.0, abs=1e-5)
        assert not _refusal_warning(messages)

    def test_control_random_batch_still_measures(self):
        from vfairness.in_processing import CorrelationPenalty

        torch.manual_seed(0)
        y_pred = torch.rand(200, requires_grad=True)
        sensitive = (torch.rand(200) > 0.5).float()

        _penalty, metrics, messages = _call(
            CorrelationPenalty(strength=0.1), y_pred, sensitive, return_metrics=True
        )

        assert metrics.measured is True
        assert math.isfinite(metrics.dependence_measure)
        assert abs(metrics.dependence_measure) < 1.0
        assert not _refusal_warning(messages)

    def test_history_carries_the_third_state(self):
        """The history is a public surface (get_history), and the fabricated
        0.0 was appended to it. A consumer reading only the history must still
        be able to tell a refusal from a measurement."""
        from vfairness.in_processing import CorrelationPenalty

        reg = CorrelationPenalty(strength=0.1, track_metrics=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg(torch.rand(30), torch.zeros(30))  # single group: refuses
            reg(torch.arange(30).float(), torch.arange(30).float())  # measures

        history = reg.get_history()
        assert len(history) == 2
        assert history[0].measured is False and math.isnan(history[0].dependence_measure)
        assert history[1].measured is True and history[1].dependence_measure == pytest.approx(
            1.0, abs=1e-5
        )


# ===========================================================================
# Finding 3 - fairness_regularization (GroupFairnessRegularizer)
# public entry: GroupFairnessRegularizer.forward(..., return_metrics=True)
# ===========================================================================


class TestGroupFairnessRegularizer:
    def test_single_group_is_not_perfect_parity(self):
        """The worst of the four: BEFORE this returned dependence_measure 0.0
        AND a populated ``group_penalties={'g0': 0.0}``, so it did not even
        leave the empty-dict tell the other degenerate inputs leave."""
        from vfairness.in_processing import GroupFairnessRegularizer

        reg = GroupFairnessRegularizer(strength=1.0, fairness_metric="dp")
        y_pred = torch.rand(30, requires_grad=True)

        penalty, metrics, messages = _call(
            reg, y_pred, torch.zeros(30, dtype=torch.long), return_metrics=True
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert metrics.group_penalties == {}, (
            "the agreeing {'g0': 0.0} was the reading a plot consumer took as a finding"
        )
        assert metrics.metadata["not_assessed"] == "fewer_than_two_groups"
        assert metrics.metadata["n_groups"] == 1
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    @pytest.mark.parametrize("metric", ["dp", "eo", "eop", "fpr", "ppv"])
    def test_the_guard_sits_above_the_metric_dispatch(self, metric):
        """All five metrics are between-group comparisons and share the
        precondition. Fixing only the filed branch would leave four live."""
        from vfairness.in_processing import GroupFairnessRegularizer

        reg = GroupFairnessRegularizer(strength=1.0, fairness_metric=metric)
        y_pred = torch.rand(30, requires_grad=True)
        y_true = (torch.arange(30) % 2).long()

        _penalty, metrics, messages = _call(
            reg,
            y_pred,
            torch.zeros(30, dtype=torch.long),
            y_true=y_true,
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure), f"{metric} still fabricates"
        assert metrics.measured is False
        assert metrics.metadata["fairness_metric"] == metric
        assert _refusal_warning(messages)

    def test_empty_batch_refuses(self):
        from vfairness.in_processing import GroupFairnessRegularizer

        empty_f = torch.tensor([], dtype=torch.float32, requires_grad=True)
        empty_i = torch.tensor([], dtype=torch.long)

        _penalty, metrics, messages = _call(
            GroupFairnessRegularizer(1.0, "dp"), empty_f, empty_i, return_metrics=True
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert metrics.metadata["n_rows"] == 0
        assert _refusal_warning(messages)

    def test_no_conditioning_set_refuses(self):
        """Two groups present, but no positive label anywhere, so no group's
        TPR exists. BEFORE: 0.0 through the same field that carries a real
        measurement, with zero warnings."""
        from vfairness.in_processing import GroupFairnessRegularizer

        torch.manual_seed(0)
        reg = GroupFairnessRegularizer(strength=1.0, fairness_metric="eop")
        y_pred = torch.rand(20, requires_grad=True)
        sensitive = torch.tensor([0] * 10 + [1] * 10)

        penalty, metrics, messages = _call(
            reg,
            y_pred,
            sensitive,
            y_true=torch.zeros(20, dtype=torch.long),
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert metrics.metadata["not_assessed"] == "no_comparable_group_pair"
        assert sorted(metrics.metadata["excluded_groups"]) == ["g0", "g1"]
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    def test_one_defined_group_is_not_a_comparison(self):
        """A group compared against itself. With only g0 holding a positive
        label, ``overall_tpr`` IS g0's TPR, so the penalty was 0.0 by
        arithmetic vacuity and ``group_penalties={'g0': 0.0}`` read as a
        measured agreement."""
        from vfairness.in_processing import GroupFairnessRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(20, requires_grad=True)
        sensitive = torch.tensor([0] * 10 + [1] * 10)
        y_true = torch.cat([torch.ones(10, dtype=torch.long), torch.zeros(10, dtype=torch.long)])

        _penalty, metrics, messages = _call(
            GroupFairnessRegularizer(1.0, "eop"),
            y_pred,
            sensitive,
            y_true=y_true,
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert _refusal_warning(messages)

    def test_a_vacuous_arm_no_longer_dilutes_the_measured_arm(self):
        """Standing trap 3, in its reporting form: the fabricated cell moved
        the number in the direction that reads FAIRER.

        With a measurable FPR arm and an unmeasurable TPR arm, 'eo' used to
        average over [0.0, fpr_0, fpr_1]. The vacuous 0.0 pulled the reported
        disparity DOWN. The arm is now dropped and disclosed instead.
        """
        from vfairness.in_processing import GroupFairnessRegularizer

        torch.manual_seed(3)
        y_pred = torch.rand(20)
        sensitive = torch.tensor([0] * 10 + [1] * 10)
        # g0: 5 positives + 5 negatives, g1: 0 positives + 10 negatives
        y_true = torch.tensor([1] * 5 + [0] * 5 + [0] * 10)

        _penalty, metrics, messages = _call(
            GroupFairnessRegularizer(1.0, "eo"),
            y_pred,
            sensitive,
            y_true=y_true,
            return_metrics=True,
        )

        negatives = y_true == 0
        overall_fpr = y_pred[negatives].mean()
        f0 = float(abs(y_pred[(sensitive == 0) & negatives].mean() - overall_fpr))
        f1 = float(abs(y_pred[(sensitive == 1) & negatives].mean() - overall_fpr))
        honest = (f0 + f1) / 2
        diluted = (0.0 + f0 + f1) / 3

        # it still MEASURES (over-refusing would be a worse defect)
        assert metrics.measured is True
        assert metrics.dependence_measure == pytest.approx(honest, rel=1e-5)
        assert metrics.dependence_measure > diluted
        # and the partial coverage is disclosed rather than hidden
        assert sorted(metrics.metadata["excluded_groups"]) == ["g0_tpr", "g1_tpr"]
        assert metrics.metadata["partial_coverage"] is True
        assert any("LOWER BOUND" in m for m in messages)

    @pytest.mark.parametrize("metric", ["dp", "eo", "eop", "fpr", "ppv"])
    def test_control_healthy_two_group_batch_still_measures(self, metric):
        from vfairness.in_processing import GroupFairnessRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(40, requires_grad=True)
        sensitive = torch.tensor([0] * 20 + [1] * 20)
        y_true = torch.tensor([1, 0] * 20)

        _penalty, metrics, messages = _call(
            GroupFairnessRegularizer(1.0, metric),
            y_pred,
            sensitive,
            y_true=y_true,
            return_metrics=True,
        )

        assert metrics.measured is True
        assert math.isfinite(metrics.dependence_measure)
        assert metrics.group_penalties, "a healthy batch must report per-group values"
        assert "not_assessed" not in metrics.metadata
        assert not _refusal_warning(messages)

    def test_control_dp_value_is_the_real_arithmetic(self):
        from vfairness.in_processing import GroupFairnessRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(40)
        sensitive = torch.tensor([0] * 20 + [1] * 20)

        _penalty, metrics = GroupFairnessRegularizer(1.0, "dp")(
            y_pred, sensitive, return_metrics=True
        )

        overall = y_pred.mean()
        expected = float(
            torch.stack(
                [
                    torch.abs(y_pred[sensitive == 0].mean() - overall),
                    torch.abs(y_pred[sensitive == 1].mean() - overall),
                ]
            ).mean()
        )
        assert metrics.dependence_measure == pytest.approx(expected, rel=1e-6)
        assert metrics.dependence_measure > 0.0


# ===========================================================================
# Finding 4 - conditional_independence_regularizer
# public entry: ConditionalIndependenceRegularizer.forward(..., return_metrics=True)
# ===========================================================================


class TestConditionalIndependenceRegularizer:
    def test_single_group_is_not_conditional_independence(self):
        """BEFORE: dependence_measure 0.0, metadata {}, zero warnings, through
        the same field that carries the healthy 0.296."""
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        torch.manual_seed(0)
        reg = ConditionalIndependenceRegularizer(strength=1.0)
        y_pred = torch.rand(200, requires_grad=True)

        penalty, metrics, messages = _call(
            reg,
            y_pred,
            torch.zeros(200, dtype=torch.long),
            y_true=torch.randint(0, 2, (200,)).float(),
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert metrics.metadata["not_assessed"] == "single_group"
        assert metrics.metadata["n_groups"] == 1
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    def test_single_group_without_y_true_also_refuses(self):
        """The guard sits ABOVE the y_true branch.

        Without it, this input takes the statistical-parity fallback and picks
        up THAT class's own single-group ``RegularizerMetrics(0.0, 0.0)``: the
        same fabrication, reached through a different door.
        """
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        reg = ConditionalIndependenceRegularizer(strength=1.0)
        y_pred = torch.rand(200, requires_grad=True)

        _penalty, metrics, messages = _call(
            reg, y_pred, torch.zeros(200, dtype=torch.long), return_metrics=True
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert _refusal_warning(messages)

    def test_empty_batch_refuses(self):
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        empty_f = torch.tensor([], requires_grad=True)

        _penalty, metrics, messages = _call(
            ConditionalIndependenceRegularizer(strength=1.0),
            empty_f,
            torch.tensor([], dtype=torch.long),
            y_true=torch.tensor([]),
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert metrics.metadata["not_assessed"] == "no_rows"
        assert _refusal_warning(messages)

    @pytest.mark.parametrize(
        "label,y_true",
        [
            ("nan_labels", torch.full((200,), float("nan"))),
            ("no_labels_at_all", torch.tensor([])),
        ],
    )
    def test_no_populated_group_label_cell_refuses(self, label, y_true):
        """The SIBLING three lines below the guard above, and the one my first
        pin missed: two groups ARE present, so the top guard passes, but no
        (group, label) stratum has any rows in it, so ``all_penalties`` is
        empty. A label column of NaNs reaches it -- ``y_true == nan`` is
        all-False for every unique value -- and BEFORE the fix it returned
        dependence_measure 0.0 with no warning, exactly like the single-group
        case one branch up.
        """
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(200, requires_grad=True)
        sensitive = (torch.rand(200) > 0.5).long()

        penalty, metrics, messages = _call(
            ConditionalIndependenceRegularizer(strength=1.0),
            y_pred,
            sensitive,
            y_true=y_true,
            return_metrics=True,
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        # the fixture really did reach THIS branch, not the guard above it
        assert metrics.metadata["not_assessed"] == "no_populated_group_label_cell"
        assert metrics.metadata["n_groups"] == 2
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    def test_control_two_groups_still_measures(self):
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(200, requires_grad=True)
        sensitive = (torch.rand(200) > 0.5).long()
        y_true = torch.randint(0, 2, (200,)).float()

        _penalty, metrics, messages = _call(
            ConditionalIndependenceRegularizer(strength=1.0),
            y_pred,
            sensitive,
            y_true=y_true,
            return_metrics=True,
        )

        assert metrics.measured is True
        assert math.isfinite(metrics.dependence_measure)
        assert metrics.dependence_measure > 0.0
        assert len(metrics.group_penalties) == 4
        assert not _refusal_warning(messages)

    def test_control_fallback_names_the_substitution(self):
        """With two groups and no y_true the fallback is legitimate, but the
        number it returns is a STATISTICAL PARITY gap under a conditional
        independence call, so the object says so."""
        from vfairness.in_processing import ConditionalIndependenceRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(200, requires_grad=True)
        sensitive = (torch.rand(200) > 0.5).long()

        _penalty, metrics, messages = _call(
            ConditionalIndependenceRegularizer(strength=1.0),
            y_pred,
            sensitive,
            return_metrics=True,
        )

        assert metrics.measured is True
        assert math.isfinite(metrics.dependence_measure)
        assert metrics.metadata["fallback"] == "statistical_parity_no_y_true"
        assert any("statistical parity" in m.lower() for m in messages)


# ===========================================================================
# Adjacent sibling found while fixing the four above (reported, not filed)
# ===========================================================================


class TestStatisticalParityRegularizerSibling:
    def test_single_group_is_not_perfect_parity(self):
        from vfairness.in_processing import StatisticalParityRegularizer

        reg = StatisticalParityRegularizer(strength=0.1)
        y_pred = torch.rand(30, requires_grad=True)

        penalty, metrics, messages = _call(
            reg, y_pred, torch.zeros(30, dtype=torch.long), return_metrics=True
        )

        assert math.isnan(metrics.dependence_measure)
        assert metrics.measured is False
        assert _refusal_warning(messages)
        _assert_zero_grad_penalty(penalty)

    def test_control_two_groups_still_measures(self):
        from vfairness.in_processing import StatisticalParityRegularizer

        torch.manual_seed(0)
        y_pred = torch.rand(40, requires_grad=True)
        sensitive = torch.tensor([0] * 20 + [1] * 20)

        _penalty, metrics, messages = _call(
            StatisticalParityRegularizer(strength=0.1), y_pred, sensitive, return_metrics=True
        )

        assert metrics.measured is True
        assert metrics.dependence_measure > 0.0
        assert not _refusal_warning(messages)


# ===========================================================================
# Findings 5 and 6 - is_satisfied on BaseFairnessConstraint
# public entry: <Constraint>().is_satisfied(y_pred, y_true, sensitive_attr)
# Closed by Stage 1; unpinned until now.
# ===========================================================================


def _single_group_inputs(n=100):
    rng = np.random.default_rng(0)
    y_pred = (rng.uniform(size=n) > 0.5).astype(int)
    y_true = (rng.uniform(size=n) > 0.5).astype(int)
    return y_pred, y_true, np.array(["A"] * n)


class TestConstraintIsSatisfiedThirdState:
    def test_equal_opportunity_single_group_is_none_not_false(self):
        """BEFORE: ``False`` -- byte-identical to a measured breach -- while the
        same call's ConstraintViolation already said could_not_evaluate=True."""
        from vfairness.in_processing import EqualOpportunityConstraint

        y_pred, y_true, sensitive = _single_group_inputs()
        constraint = EqualOpportunityConstraint()

        verdict = constraint.is_satisfied(y_pred, y_true, sensitive)
        violation = constraint.compute_violation(y_pred, y_true, sensitive)

        assert verdict is None, "could-not-check must not read as a measured breach"
        assert violation.could_not_evaluate is True
        assert math.isnan(violation.overall_violation)

    def test_equal_opportunity_empty_is_none(self):
        from vfairness.in_processing import EqualOpportunityConstraint

        empty_i = np.array([], dtype=int)
        assert EqualOpportunityConstraint().is_satisfied(empty_i, empty_i, np.array([])) is None

    def test_fpr_parity_no_negatives_is_none_not_false(self):
        """20 rows, 2 groups, every y_true == 1, so an FPR has no negatives to
        be computed over. BEFORE: ``False``, the same value a real 1.0 breach
        returns."""
        from vfairness.in_processing.constraints import FalsePositiveRateParityConstraint

        constraint = FalsePositiveRateParityConstraint(tolerance=0.05)
        groups = np.array(["a"] * 10 + ["b"] * 10)
        y_pred = np.random.default_rng(0).integers(0, 2, 20)

        violation = constraint.compute_violation(y_pred, np.ones(20, dtype=int), groups)

        assert math.isnan(violation.overall_violation)
        assert violation.could_not_evaluate is True
        assert constraint.is_satisfied(y_pred, np.ones(20, dtype=int), groups) is None

    def test_control_a_measured_breach_is_still_false(self):
        """The refusal must not swallow the finding. A real disparity has to
        keep returning False, or the fix has turned a detector off."""
        from vfairness.in_processing import EqualOpportunityConstraint
        from vfairness.in_processing.constraints import FalsePositiveRateParityConstraint

        groups = np.array(["A"] * 50 + ["B"] * 50)
        y_pred = np.concatenate([np.ones(50, dtype=int), np.zeros(50, dtype=int)])

        eop = EqualOpportunityConstraint()
        assert eop.is_satisfied(y_pred, np.ones(100, dtype=int), groups) is False
        assert eop.compute_violation(y_pred, np.ones(100, dtype=int), groups).overall_violation == (
            pytest.approx(1.0)
        )

        fpr = FalsePositiveRateParityConstraint(tolerance=0.05)
        assert fpr.is_satisfied(y_pred, np.zeros(100, dtype=int), groups) is False

    def test_control_a_measured_pass_is_still_true(self):
        """And the other end: identical groups must still return True, or
        everything refuses and the test above proves nothing."""
        from vfairness.in_processing import EqualOpportunityConstraint

        groups = np.array(["A", "B"] * 50)
        y_pred = np.tile([1, 1, 0, 0], 25)
        y_true = np.ones(100, dtype=int)

        assert EqualOpportunityConstraint(tolerance=0.05).is_satisfied(y_pred, y_true, groups) is (
            True
        )


# ===========================================================================
# Findings 7 and 8 - demographic parity must not certify a single group
# public entries: create_constraint(...).compute_violation, FairClassifier.fit
# Closed by Stage 1; unpinned until now.
# ===========================================================================


class TestDemographicParitySingleGroup:
    def test_create_constraint_does_not_certify_one_group(self):
        """BEFORE: overall_violation 0.0, is_satisfied True,
        could_not_evaluate False -- a compliance verdict over a max-minus-min
        on a ONE element list, while the three sibling constraint types on the
        same call already refused."""
        from vfairness.in_processing import create_constraint

        rng = np.random.default_rng(2)
        n = 200
        y_true = rng.binomial(1, 0.5, n)
        y_pred = rng.binomial(1, 0.5, n)
        sensitive = np.array(["A"] * n)

        violation = create_constraint("demographic_parity").compute_violation(
            y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive
        )

        assert math.isnan(violation.overall_violation)
        assert violation.could_not_evaluate is True
        assert violation.details["insufficient_data"] is True
        assert violation.is_satisfied is False, "fail closed, never a false PASS"
        assert violation.to_dict()["could_not_evaluate"] is True

    def test_the_siblings_agree_on_the_same_input(self):
        """The cross-check that made this finding unambiguous: on the SAME
        call three other constraint types refuse, and the evaluation-side twin
        metric returns NaN. demographic_parity must not be the odd one out."""
        from vfairness import demographic_parity_difference
        from vfairness.in_processing import create_constraint

        rng = np.random.default_rng(2)
        n = 200
        y_true = rng.binomial(1, 0.5, n)
        y_pred = rng.binomial(1, 0.5, n)
        sensitive = np.array(["A"] * n)

        for name in ("demographic_parity", "equalized_odds", "equal_opportunity", "fpr_parity"):
            violation = create_constraint(name).compute_violation(
                y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive
            )
            assert violation.could_not_evaluate is True, f"{name} certifies a single group"

        assert math.isnan(demographic_parity_difference(y_true, y_pred, sensitive))

    def test_control_two_groups_still_measured(self):
        from vfairness.in_processing import create_constraint

        rng = np.random.default_rng(2)
        n = 200
        y_true = rng.binomial(1, 0.5, n)
        y_pred = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
        sensitive = np.array(["A"] * 100 + ["B"] * 100)

        violation = create_constraint("demographic_parity").compute_violation(
            y_true=y_true, y_pred=y_pred, sensitive_attr=sensitive
        )

        assert violation.could_not_evaluate is False
        assert violation.overall_violation == pytest.approx(1.0)
        assert violation.is_satisfied is False

    def test_fair_classifier_does_not_score_minus_zero(self):
        """BEFORE: fairness_violation 0.0, constraint_satisfied True,
        insufficient_data False, and ``score(metric='fairness')`` -> -0.0, the
        BEST attainable fairness score, with zero warnings."""
        from sklearn.linear_model import LogisticRegression

        from vfairness import FairClassifier

        rng = np.random.default_rng(6)
        n = 60
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
        sensitive = np.array(["a"] * n)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = FairClassifier(LogisticRegression(), "demographic_parity", tolerance=0.05).fit(
                X, y, sensitive_attr=sensitive
            )
            result = clf.fairness_result_.to_dict()
            score = clf.score(X, y, sensitive_attr=sensitive, metric="fairness")
        messages = [str(w.message) for w in caught]

        assert result["fairness_violation"] is None
        assert result["constraint_satisfied"] is None
        assert result["fairness_metrics"]["insufficient_data"] is True
        assert math.isnan(score), "-0.0 is the best attainable score, not the absence of one"
        assert messages, "an unscored fit must say so"

    def test_control_two_groups_still_scores(self):
        from sklearn.linear_model import LogisticRegression

        from vfairness import FairClassifier

        rng = np.random.default_rng(6)
        n = 120
        X = rng.normal(0, 1, (n, 3))
        y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
        sensitive = np.array(["a"] * 60 + ["b"] * 60)

        clf = FairClassifier(LogisticRegression(), "demographic_parity", tolerance=0.05).fit(
            X, y, sensitive_attr=sensitive
        )
        result = clf.fairness_result_.to_dict()
        score = clf.score(X, y, sensitive_attr=sensitive, metric="fairness")

        assert result["fairness_violation"] is not None
        assert math.isfinite(result["fairness_violation"])
        assert result["constraint_satisfied"] in (True, False)
        assert result["fairness_metrics"]["insufficient_data"] is False
        assert math.isfinite(score)
