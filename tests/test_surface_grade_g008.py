"""Surface grade G008: the reweighting reweighters must not publish a value nobody measured.

Two defects were reproduced by execution at the public API before the fix, and
both are pinned below beside a healthy-data control, because a fix that makes
everything refuse is worse than the defect it replaces.

  RejectionOptionClassifier   `min()` returns the FIRST minimal key in iteration
  .fit / .transform / .predict
                              order, and the positive rates are keyed by
                              np.unique(), i.e. sorted group names. So when the
                              lowest rate is SHARED, the "unprivileged group"
                              was chosen by the alphabet. Measured on two groups
                              whose positive rate was 0.500 each: with labels
                              ('a', 'b') group a's rate went to 1.000 and b's to
                              0.000; on byte-identical scores relabelled
                              ('z', 'b'), b went to 1.000 and z to 0.000. A
                              measured parity of 0.00 became a disparity of
                              1.00, result_.group_adjustments published
                              unprivileged_group='a', and nothing warned.

  ReweightingResult.summary   The same run recorded disparity_reduction=-1.0
                              correctly and rendered it, under the heading
                              "Fairness Improvement", as
                              "disparity_reduction: 1.0000 ↓": `abs(change)`
                              threw the sign away and the arrow was picked from
                              it. The reader of the text surface saw a full unit
                              of improvement where the fit had made the gap
                              maximally worse.

The remaining classes in the module were executed on the degenerate inputs their
claims become undefined on and refused honestly there; the tests below hold them
at that behaviour.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.post_processing.reweighting.reweighter import (
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
    ReweightingMethod,
    ReweightingResult,
)


def _caught(fn):
    """Run fn() with every warning recorded; return (value, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _rate(values, mask, threshold=0.5):
    """Positive rate over the DECIDED rows of a group, computed here, not read back."""
    vals = np.asarray(values, dtype=float)[mask]
    decided = np.isfinite(vals)
    return float(np.mean(vals[decided] >= threshold)) if decided.any() else float("nan")


def _tied_rates(n_per_group=40):
    """Two groups whose positive rate at 0.5 is EXACTLY equal, scores in the critical region."""
    half = n_per_group // 2
    a = np.concatenate([np.full(half, 0.45), np.full(half, 0.55)])
    b = np.concatenate([np.full(half, 0.46), np.full(half, 0.56)])
    y_prob = np.concatenate([a, b])
    y_true = np.concatenate([np.zeros(half, int), np.ones(half, int)] * 2)
    return y_true, y_prob


def _healthy_gap(n_per_group=100, seed=5):
    """Two groups with a real, findable positive-rate gap and scores that straddle 0.5."""
    rng = np.random.default_rng(seed)
    sens = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = rng.integers(0, 2, 2 * n_per_group)
    y_prob = np.concatenate(
        [rng.uniform(0.45, 0.90, n_per_group), rng.uniform(0.15, 0.55, n_per_group)]
    )
    return y_true, y_prob, sens


# ===========================================================================
# RejectionOptionClassifier: the unprivileged group must come from the data
# ===========================================================================


class TestRejectionOptionUnprivilegedIsMeasured:
    def test_equal_positive_rates_do_not_name_an_unprivileged_group(self):
        """BEFORE: 0.500/0.500 became 1.000/0.000 and unprivileged_group='a'."""
        y_true, y_prob = _tied_rates()
        sens = np.array(["a"] * 40 + ["b"] * 40)

        roc, messages = _caught(
            lambda: RejectionOptionClassifier().fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        assert roc.detected_unprivileged_ is None
        assert roc.detected_unprivileged_groups_ == []
        assert roc.assessable_ is False
        assert any("same positive rate" in m for m in messages)

        adjusted, _ = _caught(lambda: roc.transform(y_prob, sens))
        assert np.array_equal(np.asarray(adjusted, dtype=float), y_prob)

        decisions, _ = _caught(lambda: roc.predict(y_prob, sens))
        for group in ("a", "b"):
            mask = sens == group
            # The input rate really is 0.5 in both groups; it must still be 0.5.
            assert _rate(y_prob, mask) == pytest.approx(0.5)
            assert _rate(decisions, mask) == pytest.approx(0.5)

        # The measured parity survives: a real 0.0 over two really scored
        # groups is evidence and must not be discarded as could-not-check.
        assert roc.result_.original_metrics["disparity"] == pytest.approx(0.0)
        assert roc.result_.adjusted_metrics["disparity"] == pytest.approx(0.0)
        assert not math.isnan(roc.result_.original_metrics["disparity"])
        assert roc.result_.group_adjustments["unprivileged_group"] is None

    def test_the_outcome_does_not_depend_on_the_group_names(self):
        """BEFORE: relabelling a -> z swapped which group was pushed to 0.000."""
        y_true, y_prob = _tied_rates()
        outcomes = {}
        for first in ("a", "z"):
            sens = np.array([first] * 40 + ["b"] * 40)
            roc, _ = _caught(
                lambda: RejectionOptionClassifier().fit(
                    y_true=y_true, y_prob=y_prob, sensitive_attr=sens
                )
            )
            decisions, _ = _caught(lambda: roc.predict(y_prob, sens))
            outcomes[first] = (
                _rate(decisions, sens == first),
                _rate(decisions, sens == "b"),
            )
        assert outcomes["a"] == outcomes["z"]
        assert outcomes["a"] == (pytest.approx(0.5), pytest.approx(0.5))

    def test_a_group_tied_for_worst_is_not_pushed_down(self):
        """BEFORE: a and b both at 0.500 -> a went to 1.000 and b to 0.000."""
        y_true, y_prob_two = _tied_rates()
        sens = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
        y_prob = np.concatenate([y_prob_two, np.full(40, 0.55)])
        y_true3 = np.concatenate([y_true, np.ones(40, int)])

        roc, messages = _caught(
            lambda: RejectionOptionClassifier().fit(
                y_true=y_true3, y_prob=y_prob, sensitive_attr=sens
            )
        )
        # The real gap (0.5 against 1.0) is NOT discarded: the fit proceeds.
        assert roc.assessable_ is True
        assert roc.detected_unprivileged_groups_ == ["a", "b"]
        assert roc.detected_unprivileged_ is None
        assert any("share the lowest positive rate" in m for m in messages)

        decisions, _ = _caught(lambda: roc.predict(y_prob, sens))
        rate_a = _rate(decisions, sens == "a")
        rate_b = _rate(decisions, sens == "b")
        assert rate_a == rate_b
        assert rate_a >= _rate(y_prob, sens == "a")

    def test_control_a_real_gap_is_still_detected_and_corrected(self):
        """CONTROL. One group really is worst off: it must be named and helped."""
        y_true, y_prob, sens = _healthy_gap()
        roc, messages = _caught(
            lambda: RejectionOptionClassifier().fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        assert messages == []
        assert roc.detected_unprivileged_ == "B"
        assert roc.detected_unprivileged_groups_ == ["B"]
        assert roc.assessable_ is True

        before_a = _rate(y_prob, sens == "A")
        before_b = _rate(y_prob, sens == "B")
        assert before_a - before_b > 0.5  # the fixture really does carry a gap

        decisions, _ = _caught(lambda: roc.predict(y_prob, sens))
        after_a = _rate(decisions, sens == "A")
        after_b = _rate(decisions, sens == "B")
        assert after_b > before_b
        assert after_a < before_a
        assert (after_a - after_b) < (before_a - before_b)
        # Independently recomputed, not read back from result_.
        assert roc.result_.original_metrics["disparity"] == pytest.approx(before_a - before_b)
        assert roc.result_.fairness_improvement["disparity_reduction"] > 0.2


# ===========================================================================
# ReweightingResult.summary: the sign of an improvement is the whole finding
# ===========================================================================


def _result_with(change: float) -> ReweightingResult:
    return ReweightingResult(
        method=ReweightingMethod.REJECTION_OPTION,
        group_adjustments={"unprivileged_group": "a"},
        original_metrics={"disparity": 0.0},
        adjusted_metrics={"disparity": 1.0},
        fairness_improvement={"disparity_reduction": change},
        calibration_impact={"n_modified": 40},
    )


class TestSummaryDoesNotInvertTheFinding:
    def test_a_regression_is_not_rendered_as_an_improvement(self):
        """BEFORE: -1.0 rendered as 'disparity_reduction: 1.0000 ↓'."""
        text = _result_with(-1.0).summary()
        assert "disparity_reduction: 1.0000 ↓" not in text
        assert "1.0000 ↓" not in text
        assert "-1.0000" in text
        assert "REGRESSION" in text

    def test_control_a_real_improvement_still_reads_as_one(self):
        """CONTROL. A measured reduction must not be re-rendered as a regression."""
        text = _result_with(0.95).summary()
        assert "+0.9500" in text
        assert "improvement" in text
        assert "REGRESSION" not in text
        assert "NOT MEASURED" not in text

    def test_control_an_unmeasurable_change_is_still_neither(self):
        text = _result_with(float("nan")).summary()
        assert "NOT MEASURED (no comparison was possible)" in text
        assert "REGRESSION" not in text
        assert "improvement" not in text

    def test_the_end_to_end_regression_reaches_the_reader(self):
        """The two defects met here: a created disparity, rendered as progress."""
        y_true, y_prob = _tied_rates()
        sens = np.array(["a"] * 40 + ["b"] * 40)
        roc, _ = _caught(
            lambda: RejectionOptionClassifier(unprivileged_group="a").fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        # The caller NAMED the group, so the fit proceeds; it makes the gap
        # worse, and the text surface has to say so.
        assert roc.result_.fairness_improvement["disparity_reduction"] < 0
        text = roc.result_.summary()
        assert "REGRESSION" in text
        assert "1.0000 ↓" not in text


# ===========================================================================
# The rest of the batch: three states on the inputs where the claim is undefined
# ===========================================================================


class TestReweightersRefuseWhatTheyCannotMeasure:
    def test_prediction_reweighter_all_nan_is_not_a_neutral_factor(self):
        y_true = np.random.default_rng(0).integers(0, 2, 60)
        y_prob = np.full(60, np.nan)
        sens = np.array(["a"] * 30 + ["b"] * 30)

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert all(math.isnan(v) for v in rw.adjustments_.values())
        assert 1.0 not in set(rw.adjustments_.values())
        assert math.isnan(rw.result_.original_metrics["disparity"])
        assert sorted(rw.result_.unfittable_groups) == ["a", "b"]
        assert any("not 1.0" in m for m in messages)

        adjusted, _ = _caught(lambda: rw.transform(y_prob, sens))
        assert np.all(np.isnan(adjusted))

        decisions, decide_messages = _caught(lambda: rw.predict(y_prob, sens))
        assert decisions.dtype == np.float64
        assert np.all(np.isnan(decisions))
        assert any("no decision was made" in m for m in decide_messages)

    def test_control_one_unscored_group_does_not_discard_the_other_two(self):
        """CONTROL for the reverse defect: a measured gap must survive a hole."""
        rng = np.random.default_rng(4)
        sens = np.array(["A"] * 40 + ["B"] * 40 + ["C"])
        y_prob = np.concatenate(
            [rng.uniform(0.60, 0.95, 40), rng.uniform(0.05, 0.40, 40), [np.nan]]
        )
        y_true = rng.integers(0, 2, 81)

        rw, messages = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        # A and B are fully scored and one full unit apart: that is a finding.
        assert rw.result_.original_metrics["disparity"] == pytest.approx(1.0)
        assert rw.result_.groups_not_compared == ["C"]
        assert math.isnan(rw.adjustments_["C"])
        assert np.isfinite(rw.adjustments_["A"]) and np.isfinite(rw.adjustments_["B"])
        assert any("spread over ['A', 'B'] only" in m for m in messages)
        # The factor is the one arithmetic says it is, computed here.
        assert rw.adjustments_["A"] == pytest.approx(
            float(np.mean(y_prob[:80])) / float(np.mean(y_prob[:40]))
        )

    def test_fit_transform_is_fit_then_transform_and_carries_the_refusal(self):
        y_true, y_prob, sens = _healthy_gap()
        by_hand, _ = _caught(
            lambda: (
                PredictionReweighter()
                .fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
                .transform(y_prob, sens)
            )
        )
        in_one, _ = _caught(lambda: PredictionReweighter().fit_transform(y_true, y_prob, sens))
        assert np.array_equal(in_one, by_hand)
        assert np.all(np.isfinite(in_one))
        # and on an input nothing can be fitted from, it is NaN, not the input
        nan_prob = np.full(60, np.nan)
        refused, _ = _caught(
            lambda: PredictionReweighter().fit_transform(
                np.zeros(60, int), nan_prob, np.array(["a"] * 30 + ["b"] * 30)
            )
        )
        assert np.all(np.isnan(refused))

    def test_calibrated_equalizer_does_not_map_a_group_it_never_fitted(self):
        rng = np.random.default_rng(7)
        sens = np.array(["a"] * 60 + ["b"] * 60)
        y_prob = np.concatenate([rng.uniform(0.4, 0.9, 60), rng.uniform(0.1, 0.6, 60)])
        y_true = rng.integers(0, 2, 120)

        eq, _ = _caught(
            lambda: CalibratedEqualizer().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        # CONTROL: a fitted group really is mapped onto the pooled distribution.
        fitted, _ = _caught(lambda: eq.transform(y_prob, sens))
        assert np.all(np.isfinite(fitted))
        assert float(np.mean(np.abs(fitted - y_prob))) > 0.0
        assert np.isfinite(eq.result_.original_metrics["mean_disparity"])

        unseen, messages = _caught(lambda: eq.transform(y_prob[:5], np.array(["c"] * 5)))
        assert np.all(np.isnan(unseen))
        assert any("not present at fit time" in m for m in messages)

    def test_distribution_matcher_refuses_a_grid_it_cannot_estimate(self):
        rng = np.random.default_rng(8)
        sens = np.array(["a"] * 60 + ["b"] * 5)
        y_prob = np.concatenate([rng.uniform(0.3, 0.9, 60), rng.uniform(0.1, 0.4, 5)])
        y_true = rng.integers(0, 2, 65)

        dm, messages = _caught(
            lambda: DistributionMatcher().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert "b" in dm.unfittable_groups_
        assert any("fewer than min_group_size" in m for m in messages)
        adjusted, _ = _caught(lambda: dm.transform(y_prob, sens))
        assert np.all(np.isnan(adjusted[60:]))
        # CONTROL: the reference group IS the target distribution, untouched.
        assert np.array_equal(adjusted[:60], np.clip(y_prob[:60], 0, 1))

    def test_control_distribution_matcher_matches_two_big_groups(self):
        """CONTROL. With both grids estimable the mapping really moves scores."""
        rng = np.random.default_rng(12)
        sens = np.array(["a"] * 80 + ["b"] * 80)
        y_prob = np.concatenate([rng.uniform(0.5, 0.95, 80), rng.uniform(0.05, 0.5, 80)])
        y_true = rng.integers(0, 2, 160)
        dm, messages = _caught(
            lambda: DistributionMatcher().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert messages == []
        assert dm.unfittable_groups_ == {}
        before = abs(float(np.mean(y_prob[:80])) - float(np.mean(y_prob[80:])))
        adjusted, _ = _caught(lambda: dm.transform(y_prob, sens))
        after = abs(float(np.mean(adjusted[:80])) - float(np.mean(adjusted[80:])))
        assert before > 0.4
        assert after < before


class TestReweightingResultToDict:
    def test_to_dict_carries_every_field_including_the_disclosures(self):
        """A field-by-field rebuild is where a later disclosure gets dropped."""
        import dataclasses

        result = ReweightingResult(
            method=ReweightingMethod.ADDITIVE,
            group_adjustments={"a": float("nan")},
            original_metrics={"disparity": float("nan")},
            adjusted_metrics={"disparity": float("nan")},
            fairness_improvement={"disparity_reduction": float("nan")},
            calibration_impact={"mean_shift": 0.0},
            requested_constraint="equalized_odds",
            honoured_constraint="demographic_parity",
            groups_not_compared=["a"],
            unfittable_groups={"a": "no finite score"},
        )
        as_dict = result.to_dict()
        for field in dataclasses.fields(ReweightingResult):
            assert field.name in as_dict, f"{field.name} is dropped by to_dict()"
        assert as_dict["groups_not_compared"] == ["a"]
        assert as_dict["unfittable_groups"] == {"a": "no finite score"}
        assert as_dict["honoured_constraint"] == "demographic_parity"
        assert as_dict["requested_constraint"] == "equalized_odds"
        assert as_dict["method"] == "additive"


# ===========================================================================
# PredictionReweighter: a multiplicative factor that cannot exist is not 1.0
# ===========================================================================


class TestZeroMeanGroupIsNotANeutralFactor:
    def _zero_mean_frame(self):
        rng = np.random.default_rng(2)
        sens = np.array(["A"] * 60 + ["B"] * 60)
        y_prob = np.concatenate([np.zeros(60), rng.uniform(0.4, 0.9, 60)])
        y_true = rng.integers(0, 2, 120)
        return y_true, y_prob, sens

    def test_a_zero_mean_group_is_refused_not_given_the_identity(self):
        """BEFORE: adjustments_['A'] == 1.0 and disparity_reduction == 0.8333."""
        y_true, y_prob, sens = self._zero_mean_frame()
        rw, messages = _caught(
            lambda: PredictionReweighter(method="multiplicative").fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        assert not (rw.adjustments_["A"] == 1.0)
        assert math.isnan(rw.adjustments_["A"])
        assert "A" in (rw.result_.unfittable_groups or {})
        assert any("no multiplicative factor can move them to the target" in m for m in messages)
        assert "no multiplicative factor" in rw.result_.unfittable_groups["A"]

        # The "improvement" that came from rejecting everybody is gone.
        assert math.isnan(rw.result_.adjusted_metrics["disparity"])
        assert math.isnan(rw.result_.fairness_improvement["disparity_reduction"])

        # The REAL finding is untouched: A is measurably worse off, and the gap
        # is the one arithmetic gives, computed here.
        expected_gap = _rate(y_prob, sens == "B") - _rate(y_prob, sens == "A")
        assert expected_gap > 0.5
        assert rw.result_.original_metrics["disparity"] == pytest.approx(expected_gap)

        adjusted, _ = _caught(lambda: rw.transform(y_prob, sens))
        assert np.all(np.isnan(adjusted[:60]))
        assert np.all(np.isfinite(adjusted[60:]))

    def test_control_additive_reaches_the_target_from_zero_and_still_fits(self):
        """CONTROL. The offset CAN move a zero-mean group, so it must not refuse."""
        y_true, y_prob, sens = self._zero_mean_frame()
        rw, messages = _caught(
            lambda: PredictionReweighter(method="additive").fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        assert messages == []
        expected_offset = float(np.mean(y_prob)) - 0.0
        assert rw.adjustments_["A"] == pytest.approx(expected_offset)
        assert rw.result_.unfittable_groups == {}
        assert np.isfinite(rw.result_.adjusted_metrics["disparity"])
        adjusted, _ = _caught(lambda: rw.transform(y_prob, sens))
        assert np.all(np.isfinite(adjusted))

    def test_control_a_zero_target_really_is_reached_by_the_identity(self):
        """CONTROL. mean 0 and target 0 means the group IS at the target: 1.0 is a fit."""
        sens = np.array(["A"] * 30 + ["B"] * 30)
        y_prob = np.zeros(60)
        y_true = np.zeros(60, int)
        rw, messages = _caught(
            lambda: PredictionReweighter(method="multiplicative").fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        assert rw.target_rate_ == 0.0
        assert rw.adjustments_ == {"A": 1.0, "B": 1.0}
        assert rw.result_.unfittable_groups == {}
        assert not any("multiplicative factor" in m for m in messages)
        # both groups really do sit at the same (zero) rate: a measured 0.0
        assert rw.result_.original_metrics["disparity"] == 0.0


# ===========================================================================
# A group that was never fitted is not a reweighted group
# ===========================================================================


class TestAGroupAbsentAtFitTimeIsRefused:
    def _fitted_pair(self):
        rng = np.random.default_rng(1)
        sens = np.array(["a"] * 50 + ["b"] * 50)
        y_prob = np.concatenate([rng.uniform(0.5, 0.9, 50), rng.uniform(0.1, 0.5, 50)])
        y_true = rng.integers(0, 2, 100)
        return y_true, y_prob, sens

    NEW_SCORES = np.array([0.30, 0.45, 0.55, 0.70])

    def test_prediction_reweighter_does_not_pass_an_unfitted_group_through(self):
        """BEFORE: transform(scores, ['c']*4) returned the input and predict said [0,0,1,1]."""
        y_true, y_prob, sens = self._fitted_pair()
        rw, _ = _caught(
            lambda: PredictionReweighter().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        unseen = np.array(["c"] * 4)
        adjusted, messages = _caught(lambda: rw.transform(self.NEW_SCORES, unseen))
        assert np.all(np.isnan(adjusted))
        assert not np.array_equal(adjusted, self.NEW_SCORES)
        assert any("not present at fit time" in m for m in messages)

        decisions, decide_messages = _caught(lambda: rw.predict(self.NEW_SCORES, unseen))
        assert decisions.dtype == np.float64
        assert np.all(np.isnan(decisions))
        assert any("no decision was made" in m for m in decide_messages)

        # CONTROL: a group that WAS fitted is still adjusted and still decided.
        fitted, control_messages = _caught(
            lambda: rw.transform(self.NEW_SCORES, np.array(["b"] * 4))
        )
        assert control_messages == []
        assert np.all(np.isfinite(fitted))
        assert fitted == pytest.approx(np.clip(self.NEW_SCORES * rw.adjustments_["b"], 0, 1))
        control_decisions, _ = _caught(lambda: rw.predict(self.NEW_SCORES, np.array(["b"] * 4)))
        assert control_decisions.dtype == np.int64

    def test_rejection_option_does_not_push_down_a_group_it_never_saw(self):
        """BEFORE: an unseen 'c' row went 0.55 -> 0.49, flipping its decision to 0."""
        y_true, y_prob, sens = self._fitted_pair()
        roc, _ = _caught(
            lambda: RejectionOptionClassifier().fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=sens
            )
        )
        adjusted, messages = _caught(lambda: roc.transform(self.NEW_SCORES, np.array(["c"] * 4)))
        assert np.all(np.isnan(adjusted))
        assert 0.49 not in set(np.asarray(adjusted, dtype=float).tolist())
        assert any("not present at fit time" in m for m in messages)

        decisions, decide_messages = _caught(
            lambda: roc.predict(self.NEW_SCORES, np.array(["c"] * 4))
        )
        assert decisions.dtype == np.float64
        assert np.all(np.isnan(decisions))
        assert any("no decision was made" in m for m in decide_messages)

        # CONTROL: the groups it DID measure are still moved, in the right
        # direction: 'b' is the unprivileged one, so its near-boundary row goes
        # up, and 'a' (privileged) goes down.
        up, _ = _caught(lambda: roc.transform(self.NEW_SCORES, np.array(["b"] * 4)))
        down, _ = _caught(lambda: roc.transform(self.NEW_SCORES, np.array(["a"] * 4)))
        assert roc.detected_unprivileged_ == "b"
        assert up[1] == pytest.approx(0.51)
        assert down[2] == pytest.approx(0.49)


class TestCalibratedEqualizerFitRefusesAnUnestimableGrid:
    def test_a_group_below_min_group_size_is_not_given_a_quantile_map(self):
        rng = np.random.default_rng(21)
        sens = np.array(["a"] * 60 + ["b"] * 5)
        y_prob = np.concatenate([rng.uniform(0.3, 0.9, 60), rng.uniform(0.05, 0.2, 5)])
        y_true = rng.integers(0, 2, 65)

        eq, messages = _caught(
            lambda: CalibratedEqualizer().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert "b" in eq.unfittable_groups_
        assert "b" not in eq.quantile_maps_
        assert any("fewer than min_group_size" in m for m in messages)
        assert "NOT FITTED" in eq.result_.group_adjustments["b"]
        adjusted, _ = _caught(lambda: eq.transform(y_prob, sens))
        assert np.all(np.isnan(adjusted[60:]))
        assert np.all(np.isfinite(adjusted[:60]))
        assert math.isnan(eq.result_.adjusted_metrics["mean_disparity"])

    def test_control_a_group_at_min_group_size_is_still_mapped(self):
        """CONTROL. 30 scored rows is the documented floor, not a refusal."""
        rng = np.random.default_rng(22)
        sens = np.array(["a"] * 60 + ["b"] * 30)
        y_prob = np.concatenate([rng.uniform(0.3, 0.9, 60), rng.uniform(0.05, 0.5, 30)])
        y_true = rng.integers(0, 2, 90)
        eq, messages = _caught(
            lambda: CalibratedEqualizer().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        )
        assert messages == []
        assert eq.unfittable_groups_ == {}
        assert set(eq.quantile_maps_) == {"a", "b"}
        adjusted, _ = _caught(lambda: eq.transform(y_prob, sens))
        assert np.all(np.isfinite(adjusted))
        assert np.isfinite(eq.result_.adjusted_metrics["mean_disparity"])
