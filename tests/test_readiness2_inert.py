"""Readiness wave 2: four documented options that did nothing, and the gate
weakness that let them survive five reading passes.

An INERT OPTION is the same defect as a fabricated verdict, one step earlier: a
value the caller chose, recorded on the object, never read, and then reported
back as though it had been honoured. The result carries the caller's label
while the code did something else.

Each finding below was reproduced by execution before it was touched, and each
pin here comes with an over-correction control, because a refusal that also
refuses the legitimate call is worse than the inert option it replaced.

    F22  BaseFairnessConstraint(bound_type=...)          REFUSED
    F23  CalibrationAwareTrainer(calibration_frequency=)  REMOVED
    F24  counterfactual_augment(mode='balanced')          IMPLEMENTED
    F25  reject_swapped_labels_and_scores(y_prob=...)     IMPLEMENTED

The gate weakness itself is pinned next door, in
the fabricated-verdict gate (a private test module the public export drops).
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._validation import (
    reject_swapped_labels_and_scores,
)
from vfairness.exceptions import ConfigurationError
from vfairness.in_processing.constraints.base import (
    BoundedGroupLossConstraint,
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FalsePositiveRateParityConstraint,
    create_constraint,
)
from vfairness.preprocessing.feature_engineering.data_balancing import (
    CounterfactualAugmenter,
    counterfactual_augment,
)

DIFFERENCE_BOUND_CONSTRAINTS = [
    DemographicParityConstraint,
    EqualizedOddsConstraint,
    EqualOpportunityConstraint,
    FalsePositiveRateParityConstraint,
]


def _rate_split(n: int = 400):
    """Two groups with positive rates 0.70 and 0.20 by construction."""
    rng = np.random.default_rng(20260910)
    sens = rng.choice(["a", "b"], n)
    y_pred = (rng.random(n) < np.where(sens == "a", 0.70, 0.20)).astype(int)
    y_true = (rng.random(n) < 0.4).astype(int)
    return y_pred, y_true, sens


# ---------------------------------------------------------------------------
# F22. bound_type selected between two DIFFERENT bounds and selected nothing.
#
# Measured 2026-09-10 on the data above: bound_type='difference' and
# bound_type='ratio' both returned overall_violation 0.527124 and the same
# is_satisfied, and a __getattribute__ spy recorded 0 reads of self.bound_type
# across compute_violation, signed_constraint_value and is_satisfied.
# 'banana' was accepted and stored. The documented ratio bound would have read
# 0.276082 on that same data, a different scale entirely, so the label 'ratio'
# was carried on an answer computed as a difference.
# ---------------------------------------------------------------------------


class TestBoundTypeIsRefusedNotIgnored:
    @pytest.mark.parametrize("cls", DIFFERENCE_BOUND_CONSTRAINTS, ids=lambda c: c.__name__)
    def test_a_ratio_bound_is_refused_by_name(self, cls):
        with pytest.raises(NotImplementedError) as excinfo:
            cls(tolerance=0.05, bound_type="ratio")
        msg = str(excinfo.value)
        assert cls.__name__ in msg, msg
        assert "'ratio'" in msg, msg
        # It must say what it DOES implement, not only what it refuses.
        assert "|g_a - g_b| <= tolerance" in msg, msg

    def test_an_unknown_bound_is_a_configuration_error(self):
        with pytest.raises(ConfigurationError, match="not a bound this library knows"):
            DemographicParityConstraint(tolerance=0.05, bound_type="banana")

    def test_the_factory_passes_the_refusal_through(self):
        """create_constraint forwards **kwargs, so the inert option was
        reachable from the documented factory too."""
        with pytest.raises(NotImplementedError, match="does not implement the 'ratio' bound"):
            create_constraint("equalized_odds", tolerance=0.05, bound_type="ratio")

    def test_the_value_is_read_at_construction(self):
        """The point of the fix: bound_type reaches a branch. If it were still
        write-only, every value would construct."""
        assert DemographicParityConstraint.supported_bound_types == ("difference",)
        assert BoundedGroupLossConstraint.supported_bound_types == ("ratio",)


class TestControlTheDifferenceBoundIsUntouched:
    """Over-correction control. The refusal must not move a single number on
    the path that was always correct."""

    def test_the_measured_violation_is_the_same_number_as_before_the_fix(self):
        y_pred, y_true, sens = _rate_split()
        violation = DemographicParityConstraint(tolerance=0.05).compute_violation(
            y_pred, y_true, sens
        )
        # Measured before the fix and after it: 0.527124, is_satisfied False.
        # This class returns np.bool_ here, so the conversion is a TYPE change
        # and nothing else: there is no third state on this field to flatten.
        assert violation.overall_violation == pytest.approx(0.527124, abs=1e-6)
        assert bool(violation.is_satisfied) is False

    @pytest.mark.parametrize("cls", DIFFERENCE_BOUND_CONSTRAINTS, ids=lambda c: c.__name__)
    def test_the_default_and_the_explicit_difference_both_construct(self, cls):
        assert cls(tolerance=0.05).bound_type == "difference"
        assert cls(tolerance=0.05, bound_type="difference").bound_type == "difference"

    def test_bounded_group_loss_keeps_its_ratio_bound(self):
        """It is the one class whose bound really is a ratio one, and it must
        still build and still measure."""
        y_pred, y_true, sens = _rate_split()
        constraint = BoundedGroupLossConstraint(tolerance=0.1)
        assert constraint.bound_type == "ratio"
        violation = constraint.compute_violation(y_pred, y_true, sens)
        assert violation.overall_violation == pytest.approx(0.012772, abs=1e-6)
        assert bool(violation.is_satisfied) is False

    def test_the_factory_still_builds_every_supported_type(self):
        for name in (
            "demographic_parity",
            "equalized_odds",
            "equal_opportunity",
            "fpr_parity",
            "bounded_group_loss",
        ):
            assert create_constraint(name, tolerance=0.05) is not None


# ---------------------------------------------------------------------------
# F23. calibration_frequency named a schedule that nothing could run.
#
# Measured 2026-09-10 over 8 optimisation steps with a fixed seed: frequency 1,
# 7 and 99 produced identical loss traces (final total_loss 0.6608137488) and
# bit-identical model weights, and a spy recorded 0 reads. The class holds no
# training loop and no step counter, so it was REMOVED, not implemented.
# ---------------------------------------------------------------------------


def _trainer_module():
    pytest.importorskip("torch")
    from vfairness.in_processing.calibrators import group_calibrators

    return group_calibrators


class TestCalibrationFrequencyIsGone:
    def test_the_keyword_is_refused_loudly(self):
        gc = _trainer_module()
        import torch.nn as nn

        model, calibrator = nn.Linear(4, 1), gc.TrainableGroupCalibrator(n_groups=2)
        with pytest.raises(TypeError, match="calibration_frequency"):
            gc.CalibrationAwareTrainer(model, calibrator, calibration_frequency=7)

    def test_a_third_positional_argument_is_refused_loudly(self):
        """The removal must not be SILENT. A caller who passed the frequency
        positionally would otherwise have it land on calibration_epochs."""
        gc = _trainer_module()
        import torch.nn as nn

        model, calibrator = nn.Linear(4, 1), gc.TrainableGroupCalibrator(n_groups=2)
        with pytest.raises(TypeError, match="positional"):
            gc.CalibrationAwareTrainer(model, calibrator, 7)

    def test_nothing_named_a_frequency_survives_on_the_object(self):
        gc = _trainer_module()
        import torch.nn as nn

        trainer = gc.CalibrationAwareTrainer(
            nn.Linear(4, 1), gc.TrainableGroupCalibrator(n_groups=2)
        )
        # List what it HAS rather than asking whether one name is absent: a
        # getattr miss on a wrong name looks exactly like a real absence.
        assert sorted(vars(trainer)) == ["calibration_epochs", "calibrator", "model"]


class TestControlTheTrainerStillTrains:
    def _trace(self, gc, steps: int = 8, seed: int = 7, **kwargs):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        model = nn.Linear(4, 1)
        calibrator = gc.TrainableGroupCalibrator(n_groups=2)
        trainer = gc.CalibrationAwareTrainer(model, calibrator, **kwargs)
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(calibrator.parameters()), lr=0.1
        )
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(seed)
        losses = []
        for _ in range(steps):
            x = torch.randn(32, 4, generator=generator)
            y = (torch.rand(32, 1, generator=generator) < 0.4).float()
            group_ids = (torch.rand(32, generator=generator) < 0.5).long()
            losses.append(trainer.train_step(x, y, group_ids, optimizer, loss_fn)["total_loss"])
        return losses

    def test_the_documented_construction_and_the_step_are_unchanged(self):
        gc = _trainer_module()
        losses = self._trace(gc)
        assert len(losses) == 8
        # The number the inert parameter produced for EVERY frequency, so the
        # removal is proven to have changed nothing about the training path.
        assert losses[-1] == pytest.approx(0.6608137488, abs=1e-5)

    def test_calibration_epochs_is_still_settable_and_still_read(self):
        gc = _trainer_module()
        trainer = gc.CalibrationAwareTrainer(
            gc.nn.Linear(4, 1),
            gc.TrainableGroupCalibrator(n_groups=2),
            calibration_epochs=9,
        )
        assert trainer.calibration_epochs == 9
        # And it does not disturb the optimisation step.
        assert self._trace(gc, calibration_epochs=9) == self._trace(gc)


# ---------------------------------------------------------------------------
# F24. mode='balanced' inverted the imbalance instead of removing it.
#
# Recorded 2026-08-22 as [MEDIUM][CONFIRMED], still live on 2026-09-10.
# Measured before the fix: A=60/B=30/C=10 gave {C: 100, A: 60, B: 40}, the
# rarest level inflated ten-fold into the largest. Multi-attribute cells
# {(A,M):45,(A,F):25,(B,M):22,(B,F):8} gave a maximum deviation from a uniform
# split of 0.2500 against 0.2000 before, strictly worse than doing nothing, and
# mode was never read in that branch at all.
# ---------------------------------------------------------------------------


def _max_deviation_from_uniform(counts) -> float:
    total = sum(counts.values())
    k = len(counts)
    return max(abs(v / total - 1 / k) for v in counts.values())


def _three_levels() -> pd.DataFrame:
    return pd.DataFrame({"race": ["A"] * 60 + ["B"] * 30 + ["C"] * 10, "y": [0, 1] * 50})


def _two_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race": ["A"] * 70 + ["B"] * 30,
            "gender": ["M"] * 45 + ["F"] * 25 + ["M"] * 22 + ["F"] * 8,
        }
    )


class TestBalancedActuallyBalances:
    def test_the_rarest_level_does_not_become_the_largest(self):
        df = _three_levels()
        before = Counter(df["race"])
        after = Counter(counterfactual_augment(df, ["race"], mode="balanced")["race"])
        assert before.most_common()[-1][0] == "C", "fixture drifted"
        # The defect's signature: C, the rarest, ending up on top.
        assert after["C"] < after.most_common(1)[0][1] or after["C"] == after["A"]
        assert after == Counter({"A": 70, "C": 70, "B": 60})
        assert _max_deviation_from_uniform(after) == pytest.approx(0.0333, abs=5e-4)
        assert _max_deviation_from_uniform(before) == pytest.approx(0.2667, abs=5e-4)

    def test_the_multi_attribute_branch_balances_too(self):
        df = _two_attributes()
        before = Counter(map(tuple, df[["race", "gender"]].values))
        out = counterfactual_augment(df, ["race", "gender"], mode="balanced")
        after = Counter(map(tuple, out[["race", "gender"]].values))
        assert after == Counter({("A", "F"): 51, ("A", "M"): 50, ("B", "M"): 50, ("B", "F"): 49})
        assert _max_deviation_from_uniform(before) == pytest.approx(0.2000, abs=5e-4)
        assert _max_deviation_from_uniform(after) == pytest.approx(0.0050, abs=5e-4)

    def test_mode_is_read_in_the_multi_attribute_branch(self):
        """It used to return observed[0] whatever mode said, so the two modes
        were byte-identical. They must now differ, and differ in the direction
        the docstring describes."""
        df = _two_attributes()
        balanced = counterfactual_augment(df, ["race", "gender"], mode="balanced")
        other = counterfactual_augment(df, ["race", "gender"], mode="first_alternative")
        assert not balanced.equals(other)
        other_counts = Counter(map(tuple, other[["race", "gender"]].values))
        # 'first alternative in sorted order' balances nothing, and says so.
        assert other_counts == Counter(
            {("A", "F"): 100, ("A", "M"): 70, ("B", "M"): 22, ("B", "F"): 8}
        )
        assert _max_deviation_from_uniform(other_counts) > _max_deviation_from_uniform(
            Counter(map(tuple, balanced[["race", "gender"]].values))
        )

    def test_the_audit_case_from_2026_08_22(self):
        df = pd.DataFrame({"race": ["A"] * 80 + ["B"] * 15 + ["C"] * 5})
        after = Counter(counterfactual_augment(df, ["race"], mode="balanced")["race"])
        assert after == Counter({"A": 80, "C": 65, "B": 55})  # was {C: 100, A: 80, B: 20}


class TestControlAugmentationStillAugments:
    """Over-correction control: the two-level case was always correct and must
    stay bit-for-bit correct, and nothing may invent or drop a group."""

    def test_two_levels_are_unchanged(self):
        df = pd.DataFrame({"grp": ["A"] * 80 + ["B"] * 20})
        after = Counter(counterfactual_augment(df, ["grp"], mode="balanced")["grp"])
        assert after == Counter({"A": 100, "B": 100})

    def test_a_single_level_produces_no_twins(self):
        df = pd.DataFrame({"grp": ["A"] * 10})
        assert len(counterfactual_augment(df, ["grp"], mode="balanced")) == 10

    def test_every_row_still_gets_a_twin_and_the_target_survives(self):
        df = _three_levels()
        out = counterfactual_augment(df, ["race"], target="y", mode="balanced")
        assert len(out) == 2 * len(df)
        assert Counter(out["y"]) == Counter({0: 100, 1: 100})
        assert set(out["race"]) == set(df["race"])

    def test_the_result_does_not_depend_on_random_state(self):
        """The docstring says the function is deterministic and the seed is not
        read. That claim is pinned rather than trusted."""
        df = _three_levels()
        assert counterfactual_augment(df, ["race"], random_state=1).equals(
            counterfactual_augment(df, ["race"], random_state=999)
        )

    def test_the_transformer_hook_still_returns_aligned_rows(self):
        df = pd.DataFrame(
            {"f1": np.arange(100, dtype=float), "grp": ["A"] * 60 + ["B"] * 30 + ["C"] * 10}
        )
        y = np.tile([0, 1], 50)
        augmenter = CounterfactualAugmenter(protected_attributes=["grp"]).fit(df, y)
        X_res, y_res = augmenter.get_resampled_data(df, y)
        assert len(X_res) == len(y_res) == 200
        assert Counter(y_res) == Counter({0: 100, 1: 100})
        assert set(X_res["grp"]) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# F25. reject_swapped_labels_and_scores(y_true, y_prob) never read y_prob.
#
# Measured 2026-09-10: y_prob='not an array' was accepted and returned None,
# and a hard 0/1 column in the y_true slot returned None with no warning at
# all, which is indistinguishable from a checked pass. y_prob now decides
# between "checked, the shapes agree" and "COULD NOT CHECK".
# ---------------------------------------------------------------------------

LABELS = np.array([0, 1, 0, 1, 1, 0])
SCORES = np.array([0.11, 0.72, 0.4, 0.9, 0.33, 0.5])
HARD = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])


def _warnings_from(y_true, y_prob):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reject_swapped_labels_and_scores(y_true, y_prob, caller="Probe")
    return [str(w.message) for w in caught]


class TestTheRefusalItAlreadyMadeStillWorks:
    """The guard against a reversed fit() is the reason this function exists.
    Reading y_prob must not weaken it."""

    def test_scores_in_the_label_slot_are_still_refused(self):
        with pytest.raises(ValueError, match="y_true must be labels"):
            reject_swapped_labels_and_scores(SCORES, LABELS, caller="Probe")

    def test_the_refusal_still_names_the_swap_and_the_way_out(self):
        with pytest.raises(ValueError) as excinfo:
            reject_swapped_labels_and_scores(SCORES, LABELS, caller="Probe")
        msg = str(excinfo.value)
        assert "Probe" in msg and "reversed call" in msg and "non-integral" in msg

    def test_the_refusal_reaches_the_public_fitters(self):
        from vfairness.post_processing.threshold_optimization.optimizer import (
            GroupThresholdOptimizer,
        )

        rng = np.random.default_rng(20260910)
        n = 200
        sens = rng.choice(["a", "b"], n)
        y_prob = rng.random(n)
        y_true = (rng.random(n) < 0.4).astype(int)
        with pytest.raises(ValueError, match="y_true must be labels"):
            GroupThresholdOptimizer().fit(y_true=y_prob, y_prob=y_true, sensitive_attr=sens)


class TestCouldNotCheckIsReportedRatherThanPassed:
    def test_an_unreadable_y_prob_is_reported(self):
        """It used to be accepted in silence: this is the proof y_prob is read
        at all."""
        messages = _warnings_from(LABELS, "not an array")
        assert len(messages) == 1, messages
        assert "could not be checked" in messages[0]
        assert "could not be read as numbers" in messages[0]

    def test_a_hard_zero_one_score_column_is_reported(self):
        messages = _warnings_from(LABELS, HARD)
        assert len(messages) == 1, messages
        assert "could not be checked" in messages[0]
        assert "hard 0/1 prediction column" in messages[0]
        assert "not a clean bill of health" in messages[0]

    def test_the_swap_that_cannot_be_detected_is_at_least_declared(self):
        """Hard predictions in the label slot are genuinely indistinguishable
        from labels, so this can never be a refusal. It must not be silence
        either."""
        messages = _warnings_from(HARD, LABELS)
        assert len(messages) == 1, messages
        assert "could not be checked" in messages[0]

    def test_an_all_missing_label_column_is_reported(self):
        messages = _warnings_from(np.array([np.nan, np.nan]), SCORES)
        assert len(messages) == 1, messages
        assert "no finite values" in messages[0]

    def test_the_message_does_not_collide_with_the_resolution_warning(self):
        """The optimizer's own warning about a two-valued score column is
        matched on 'distinct value' by tests/test_audit6_wave2_argument_order.py.
        Two warnings on the same call must stay tellable apart."""
        for messages in (_warnings_from(LABELS, HARD), _warnings_from(LABELS, None)):
            assert all("distinct value" not in m for m in messages), messages


class TestControlAMeasurableOrientationIsSilent:
    """Over-correction control. A guard that warns on every call teaches
    readers to ignore it. These are the shapes real labels and real scores
    arrive in, and every one of them must pass without a word."""

    @pytest.mark.parametrize(
        "y_true, y_prob, label",
        [
            (LABELS, SCORES, "int labels"),
            (LABELS.astype(float), SCORES, "float labels holding 0.0 and 1.0"),
            (list(LABELS), list(SCORES), "python lists"),
            (np.array([0, 1, 2, 1, 0, 2]), SCORES, "multiclass labels"),
            (pd.Series(LABELS), pd.Series(SCORES), "pandas series"),
        ],
    )
    def test_labels_with_a_continuous_score_column_are_silent(self, y_true, y_prob, label):
        assert _warnings_from(y_true, y_prob) == [], label

    def test_a_hole_in_the_labels_is_not_read_as_a_swap_or_as_unmeasurable(self):
        holed = LABELS.astype(float)
        holed[3] = np.nan
        assert _warnings_from(holed, SCORES) == []

    def test_the_public_fitters_are_silent_on_the_shape_they_are_for(self):
        from vfairness.post_processing.threshold_optimization.optimizer import (
            GroupThresholdOptimizer,
        )

        rng = np.random.default_rng(20260910)
        n = 200
        sens = rng.choice(["a", "b"], n)
        y_prob = rng.random(n)
        y_true = (rng.random(n) < 0.4).astype(int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer = GroupThresholdOptimizer()
            optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        assert [str(w.message) for w in caught if "could not be checked" in str(w.message)] == []
        # And the fit still produced measured thresholds, not a refusal.
        assert len(optimizer.result_.group_thresholds) == 2
