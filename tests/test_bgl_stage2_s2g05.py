"""Beta Go-Live stage 2, group 05: the reweighters must not report a mitigation
they could not make, nor a decision they never scored.

Every value quoted below was READ OFF the public entry before the fix, with the
repro script filed in the group's JSON.

  DistributionMatcher.fit / .fit_transform
      single group: original_metrics {'mean_disparity': 0.0}, adjusted 0.0,
      fairness_improvement {'disparity_reduction': 0.0}, ZERO warnings, from
      max(means) - min(means) over a ONE-element dict. 199 A + 1 B:
      disparity_reduction -0.40733990746826054, i.e. the mitigation recorded
      itself as having made the gap about nine times worse, silently, off a
      percentile grid built from one repeated value. n = 2 (one row per group):
      fit_transform returned array([0.2, 0.2]) beside a reported
      disparity_reduction of 0.6000000000000001.

  CalibratedEqualizer.fit
      single group: the same structural 0.0 trio, and summary() printed
      "disparity_reduction: 0.0000 ↑". Separately, fit() never cleared
      quantile_maps_, so after a fit on ['a', 'b'] and a refit on an a-only
      frame, 'b' was still there, byte-identical to the previous dataset's
      grid, and transform() applied it to later 'b' rows: 0.394 -> 0.224 on the
      refitted object against 0.394 on a fresh one. result_.group_adjustments
      IS rebuilt per fit and looked clean, which is what hid it.

  CalibratedEqualizer.transform
      transform(p, ['c'] * 5) on an equalizer fitted for 'a' and 'b' returned
      an array byte-identical to np.clip(input, 0, 1), with no warning and no
      marker: the identity map substituted for an adjustment never fitted. The
      same four scores decided as fitted 'b' gave [0, 1, 1, 1] and as unseen
      'c' gave [0, 0, 0, 1].

  CalibratedEqualizer.fit_transform
      n = 2 -> array([0.6, 0.6]) with disparity_reduction 0.19999999999999996.
      Production shape 29 + 1: the single minority row scoring 0.05 came back
      as 0.9470751842323551, EXACTLY the pooled maximum, and the run reported
      a 0.2619759832594277 "disparity reduction" for it. That is the neutered
      mitigation shape: the fairness number improves BECAUSE the map is
      degenerate.

  CalibratedEqualizer.predict / RejectionOptionClassifier.predict
      `np.nan >= 0.5` is False, so predict() on 60 all-NaN probabilities
      returned (array([0]), array([60])): sixty confident rejections from a
      model output that does not exist, while transform() honestly returned 60
      NaNs one line earlier. With a healthy fit and five NaN holes at scoring
      time, the NaN rows were the same int 0 as five measured-low rows.

  RejectionOptionClassifier.fit / .fit_transform
      single group: detected_unprivileged_ = 'a' from min() over a ONE-key
      dict, disparity 0.0 -> 0.0, reduction 0.0, no warnings; fit_transform
      rewrote 6 of 60 scores to threshold + 0.01 and flipped 6 of 60 decisions
      on the strength of a gap that could not exist.

  create_reweighter
      'multiplicative', 'additive' and 'rejection_option' each reported
      disparity 0.0 and disparity_reduction 0.0 on a single-cohort frame, with
      no group count anywhere in to_dict().

Three states, never two: measured / failed / could-not-check. Each test pins the
could-not-check at the PUBLIC entry, and every class has a CONTROL on healthy
data in the same file, because a fix that makes everything refuse is a worse
defect than the one it replaces.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from vfairness import (
    CalibratedEqualizer,
    DistributionMatcher,
    RejectionOptionClassifier,
    create_reweighter,
)

FACTORY_METHODS = [
    "multiplicative",
    "additive",
    "rejection_option",
    "distribution_matching",
    "calibrated_equalization",
]


def _caught(fn):
    """Run fn() capturing every warning; return (value, [messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn()
    return value, [str(w.message) for w in caught]


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _healthy(n_per_group: int = 100, seed: int = 11) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two groups, both well above min_group_size, with a real score gap."""
    rng = np.random.default_rng(seed)
    sens = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    y_true = rng.integers(0, 2, 2 * n_per_group)
    y_prob = np.concatenate(
        [
            rng.uniform(0.55, 0.95, n_per_group),
            rng.uniform(0.05, 0.45, n_per_group),
        ]
    )
    return y_true, y_prob, sens


def _single_group(n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0, 1, n)
    y_true = (rng.uniform(size=n) < y_prob).astype(int)
    return y_true, y_prob, np.array(["A"] * n)


def _disparity(metrics: Dict[str, Any]) -> float:
    """The disparity field, whichever of the two names this class uses."""
    for key in ("disparity", "mean_disparity"):
        if key in metrics:
            return float(metrics[key])
    raise AssertionError(f"no disparity field in {metrics!r}")


# ===========================================================================
# DistributionMatcher.fit and .fit_transform
# ===========================================================================


class TestDistributionMatcherThreeStates:
    def test_single_group_disparity_is_not_perfect_parity(self):
        """BEFORE: 0.0 / 0.0 / 0.0 and not one warning."""
        y_true, y_prob, sens = _single_group()
        assert len(set(sens)) == 1, "fixture must reach the fewer-than-two-groups branch"

        m = DistributionMatcher()
        _, messages = _caught(lambda: m.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert math.isnan(_disparity(m.result_.original_metrics))
        assert math.isnan(_disparity(m.result_.adjusted_metrics))
        assert math.isnan(m.result_.fairness_improvement["disparity_reduction"])
        assert any("no between-group disparity exists" in msg for msg in messages), messages
        assert "0.0000 ↑" not in m.result_.summary()
        assert "NOT MEASURED" in m.result_.summary()

    def test_fit_transform_reaches_the_same_fields_by_its_own_path(self):
        """fit_transform goes through BaseReweighter.fit_transform, not fit()
        directly, so it needs its own pin: a fit-only test would go green while
        this path stayed broken."""
        y_true, y_prob, sens = _single_group()

        m = DistributionMatcher()
        out, messages = _caught(lambda: m.fit_transform(y_true, y_prob, sens))

        assert math.isnan(m.result_.fairness_improvement["disparity_reduction"])
        assert math.isnan(_disparity(m.result_.original_metrics))
        assert any("no between-group disparity exists" in msg for msg in messages), messages
        # the single group IS the reference, so its own scores come back as they
        # went in; what must not come back is a disparity figure for them
        assert np.array_equal(out, np.clip(y_prob, 0, 1))

    def test_one_row_group_is_refused_not_matched(self):
        """BEFORE: disparity_reduction -0.40733990746826054 off a grid built
        from a single repeated value, silently."""
        rng = np.random.default_rng(0)
        y_prob = rng.uniform(0, 1, 200)
        y_true = (rng.uniform(size=200) < y_prob).astype(int)
        sens = np.array(["A"] * 199 + ["B"])
        assert int((sens == "B").sum()) == 1, "fixture must reach the group-size floor"

        m = DistributionMatcher()
        _, messages = _caught(lambda: m.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert "B" in m.unfittable_groups_
        assert "B" not in m.group_distributions_
        assert math.isnan(m.result_.fairness_improvement["disparity_reduction"])
        assert math.isnan(_disparity(m.result_.adjusted_metrics))
        # the ORIGINAL spread is a fact about the scores that were read, so it
        # stays a measurement: only what the mitigation claims is refused
        assert np.isfinite(_disparity(m.result_.original_metrics))
        assert any("min_group_size" in msg and "'B'" in msg for msg in messages), messages

        adjusted = _quiet(lambda: m.transform(y_prob, sens))
        assert np.isnan(adjusted[sens == "B"]).all()
        assert np.isfinite(adjusted[sens == "A"]).all()

    def test_n_equals_two_is_not_a_matched_distribution(self):
        """BEFORE: fit_transform -> array([0.2, 0.2]) with a reported
        disparity_reduction of 0.6000000000000001."""
        m = DistributionMatcher()
        out, messages = _caught(
            lambda: m.fit_transform(np.array([0, 1]), np.array([0.2, 0.8]), np.array(["A", "B"]))
        )

        assert not np.array_equal(np.nan_to_num(out, nan=-1.0), np.array([0.2, 0.2]))
        assert math.isnan(m.result_.fairness_improvement["disparity_reduction"])
        assert any("min_group_size" in msg for msg in messages), messages

    def test_refit_does_not_keep_the_previous_reference_or_grids(self):
        y_true, y_prob, sens = _healthy()
        m = DistributionMatcher()
        _quiet(lambda: m.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        assert set(m.group_distributions_) == {"A", "B"}

        only_b = sens == "B"
        _quiet(
            lambda: m.fit(y_true=y_true[only_b], y_prob=y_prob[only_b], sensitive_attr=sens[only_b])
        )
        assert set(m.group_distributions_) == {"B"}, m.group_distributions_
        assert m.reference_group == "B", m.reference_group

    def test_control_healthy_data_still_matches_and_measures(self):
        """CONTROL. Over-correction check: a real gap is still measured and
        still closed, with no warning at all."""
        y_true, y_prob, sens = _healthy()

        m = DistributionMatcher()
        out, messages = _caught(lambda: m.fit_transform(y_true, y_prob, sens))

        assert messages == []
        assert np.isfinite(out).all()
        original = _disparity(m.result_.original_metrics)
        adjusted = _disparity(m.result_.adjusted_metrics)
        assert np.isfinite(original) and np.isfinite(adjusted)
        assert original > 0.3, original
        assert adjusted < original
        assert m.result_.fairness_improvement["disparity_reduction"] > 0.0
        assert "NOT MEASURED" not in m.result_.summary()


# ===========================================================================
# CalibratedEqualizer.fit, .transform, .fit_transform, .predict
# ===========================================================================


class TestCalibratedEqualizerThreeStates:
    def test_single_group_disparity_is_not_perfect_parity(self):
        """BEFORE: 0.0 / 0.0 / 0.0 and 'disparity_reduction: 0.0000 ↑'."""
        rng = np.random.default_rng(1)
        n = 60
        y_true, y_prob = rng.integers(0, 2, n), rng.uniform(0, 1, n)
        sens = np.array(["a"] * n)

        eq = CalibratedEqualizer()
        _, messages = _caught(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert math.isnan(_disparity(eq.result_.original_metrics))
        assert math.isnan(_disparity(eq.result_.adjusted_metrics))
        assert math.isnan(eq.result_.fairness_improvement["disparity_reduction"])
        assert any("no between-group disparity exists" in msg for msg in messages), messages
        summary = eq.result_.summary()
        assert "disparity_reduction: 0.0000 ↑" not in summary
        assert "NOT MEASURED" in summary

    def test_refit_drops_the_previous_dataset_s_quantile_map(self):
        """BEFORE: 'b' survived a refit on an a-only frame, byte-identical to
        the previous dataset's grid, and was APPLIED to later 'b' rows."""
        rng = np.random.default_rng(5)
        y2, p2 = rng.integers(0, 2, 80), rng.uniform(0, 1, 80)
        y3, p3 = rng.integers(0, 2, 40), rng.uniform(0, 1, 40)

        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit(y_true=y2, y_prob=p2, sensitive_attr=np.array(["a", "b"] * 40)))
        assert set(eq.quantile_maps_) == {"a", "b"}, "fixture must fit both groups first"

        _quiet(lambda: eq.fit(y_true=y3, y_prob=p3, sensitive_attr=np.array(["a"] * 40)))
        assert set(eq.quantile_maps_) == {"a"}, eq.quantile_maps_

        rows = np.array([0.394, 0.244, 0.118])
        out, messages = _caught(lambda: eq.transform(rows, np.array(["b", "b", "b"])))
        assert np.isnan(out).all(), out
        assert any("'b'" in msg and "not present at fit time" in msg for msg in messages), messages

    def test_unseen_group_is_not_returned_as_an_adjusted_score(self):
        """BEFORE: byte-identical to np.clip(input, 0, 1), ZERO warnings, and
        the same four scores decided [0,1,1,1] as 'b' and [0,0,0,1] as 'c'."""
        rng = np.random.default_rng(3)
        n = 80
        y_true, y_prob = rng.integers(0, 2, n), rng.uniform(0, 1, n)
        eq = CalibratedEqualizer()
        _quiet(
            lambda: eq.fit(
                y_true=y_true, y_prob=y_prob, sensitive_attr=np.array(["a", "b"] * (n // 2))
            )
        )
        assert "c" not in eq.quantile_maps_, "fixture must reach the unfitted-group branch"

        probe = np.array([0.05, 0.22, 0.51, 0.77, 0.96])
        out, messages = _caught(lambda: eq.transform(probe, np.array(["c"] * 5)))

        assert not np.array_equal(out, np.clip(probe, 0, 1))
        assert np.isnan(out).all(), out
        assert any("'c'" in msg for msg in messages), messages

        # and no decision is manufactured for those rows either
        decisions = _quiet(lambda: eq.predict(probe, np.array(["c"] * 5)))
        assert np.isnan(np.asarray(decisions, dtype=float)).all(), decisions

    def test_mixed_array_does_not_hide_the_unmapped_rows(self):
        """BEFORE: ['a','b','a','c','c'] came back with rows 3 and 4 untouched
        and indistinguishable by value from the adjusted ones."""
        y_true, y_prob, sens = _healthy()
        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        probe_sens = np.array(["A", "B", "A", "c", "c"])
        probe = np.array([0.6, 0.2, 0.7, 0.05, 0.22])
        out = _quiet(lambda: eq.transform(probe, probe_sens))

        assert np.isfinite(out[probe_sens != "c"]).all()
        assert np.isnan(out[probe_sens == "c"]).all()

    def test_production_shape_minority_row_is_not_rewritten_to_the_pooled_max(self):
        """BEFORE: the 0.05 minority row came back as 0.9470751842323551, the
        pooled maximum, with a reported 0.26 'disparity reduction'."""
        rng = np.random.default_rng(7)
        y_prob = np.concatenate([rng.uniform(0.3, 0.95, 29), [0.05]])
        sens = np.array(["maj"] * 29 + ["min"])
        assert int((sens == "min").sum()) == 1, "fixture must reach the group-size floor"

        eq = CalibratedEqualizer()
        out, messages = _caught(lambda: eq.fit_transform(rng.integers(0, 2, 30), y_prob, sens))

        assert math.isnan(float(out[-1])), out[-1]
        assert float(out[-1]) != pytest.approx(float(y_prob.max()))
        assert math.isnan(eq.result_.fairness_improvement["disparity_reduction"])
        assert any("min_group_size" in msg for msg in messages), messages

    def test_n_equals_two_is_not_an_equalized_distribution(self):
        """BEFORE: fit_transform([1,0],[0.6,0.4],['A','B']) -> array([0.6, 0.6])
        with disparity_reduction 0.19999999999999996."""
        eq = CalibratedEqualizer()
        out, messages = _caught(
            lambda: eq.fit_transform(np.array([1, 0]), np.array([0.6, 0.4]), np.array(["A", "B"]))
        )
        assert np.isnan(out).all(), out
        assert math.isnan(eq.result_.fairness_improvement["disparity_reduction"])
        assert messages, "a refusal this total must not be silent"

    def test_fit_transform_twice_leaves_no_stale_group(self):
        y_true, y_prob, sens = _healthy()
        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit_transform(y_true, y_prob, sens))
        assert set(eq.quantile_maps_) == {"A", "B"}

        only_a = sens == "A"
        _quiet(lambda: eq.fit_transform(y_true[only_a], y_prob[only_a], sens[only_a]))
        assert set(eq.quantile_maps_) == {"A"}, eq.quantile_maps_

    def test_all_nan_scores_are_not_sixty_rejections(self):
        """BEFORE: predict on 60 all-NaN probabilities returned
        (array([0]), array([60]))."""
        rng = np.random.default_rng(2)
        n = 60
        sens = np.array(["a", "b"] * (n // 2))
        y_prob = np.full(n, np.nan)

        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit(y_true=rng.integers(0, 2, n), y_prob=y_prob, sensitive_attr=sens))
        decisions, messages = _caught(lambda: eq.predict(y_prob, sens))

        decided = np.asarray(decisions, dtype=float)
        assert np.isnan(decided).all(), np.unique(decided, return_counts=True)
        assert not np.array_equal(decided, np.zeros(n))
        assert any("no finite adjusted score" in msg for msg in messages), messages

    def test_nan_holes_at_scoring_time_are_not_measured_rejections(self):
        """BEFORE: a healthy fit plus five NaN holes gave [0,0,0,0,0], the same
        array as five measured-low rows, with result_ entirely clean."""
        y_true, y_prob, sens = _healthy()
        eq = CalibratedEqualizer()
        _quiet(lambda: eq.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        holed = y_prob.copy()
        holed[:5] = np.nan
        decisions, messages = _caught(lambda: eq.predict(holed, sens))

        decided = np.asarray(decisions, dtype=float)
        assert np.isnan(decided[:5]).all(), decided[:5]
        assert np.isfinite(decided[5:]).all()
        assert set(np.unique(decided[5:])) <= {0.0, 1.0}
        assert any("5 of 200" in msg for msg in messages), messages

    def test_control_healthy_data_still_equalizes_measures_and_decides(self):
        """CONTROL. The whole point of the class still works, silently."""
        y_true, y_prob, sens = _healthy()

        eq = CalibratedEqualizer()
        out, messages = _caught(lambda: eq.fit_transform(y_true, y_prob, sens))

        assert messages == []
        assert np.isfinite(out).all()
        original = _disparity(eq.result_.original_metrics)
        adjusted = _disparity(eq.result_.adjusted_metrics)
        assert original > 0.3, original
        assert adjusted < original
        assert eq.result_.fairness_improvement["disparity_reduction"] > 0.0

        decisions = eq.predict(y_prob, sens)
        assert decisions.dtype.kind == "i", decisions.dtype
        assert set(np.unique(decisions)) <= {0, 1}


# ===========================================================================
# RejectionOptionClassifier.fit, .fit_transform, .predict
# ===========================================================================


class TestRejectionOptionThreeStates:
    def test_single_group_names_no_unprivileged_and_measures_no_gap(self):
        """BEFORE: detected_unprivileged_ = 'a' from min() over one key,
        disparity 0.0 -> 0.0, reduction 0.0, ZERO warnings."""
        rng = np.random.default_rng(11)
        y_prob = rng.uniform(0, 1, 60)
        y_true = (rng.uniform(0, 1, 60) < y_prob).astype(int)
        sens = np.array(["a"] * 60)

        roc = RejectionOptionClassifier()
        _, messages = _caught(lambda: roc.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert roc.detected_unprivileged_ is None
        assert roc.assessable_ is False
        assert math.isnan(_disparity(roc.result_.original_metrics))
        assert math.isnan(_disparity(roc.result_.adjusted_metrics))
        assert math.isnan(roc.result_.fairness_improvement["disparity_reduction"])
        assert roc.result_.calibration_impact["n_modified"] == 0
        assert any("no between-group gap exists to correct" in msg for msg in messages), messages
        assert "disparity_reduction: 0.0000 ↑" not in roc.result_.summary()

    def test_single_group_fit_transform_rewrites_nothing(self):
        """BEFORE: 6 of 60 scores rewritten to threshold + 0.01 and 6 of 60
        decisions flipped, on a disparity that could not exist."""
        rng = np.random.default_rng(11)
        y_prob = rng.uniform(0, 1, 60)
        y_true = (rng.uniform(0, 1, 60) < y_prob).astype(int)
        sens = np.array(["a"] * 60)
        in_region = int(((y_prob >= 0.4) & (y_prob <= 0.6)).sum())
        assert in_region > 0, "fixture must put rows in the critical region"

        roc = RejectionOptionClassifier()
        out, messages = _caught(lambda: roc.fit_transform(y_true, y_prob, sens))

        assert np.array_equal(out, y_prob), np.flatnonzero(out != y_prob)
        assert roc.result_.calibration_impact["n_in_critical_region"] == in_region
        assert math.isnan(roc.result_.fairness_improvement["disparity_reduction"])
        assert any("No row is moved" in msg for msg in messages), messages

    def test_all_nan_scores_are_not_sixty_denials(self):
        """BEFORE: transform kept 60 NaN and predict returned int64 [0] x 60,
        sum 0, with no error and no warning."""
        rng = np.random.default_rng(11)
        y_true = (rng.uniform(0, 1, 60) < 0.5).astype(int)
        y_prob = np.full(60, np.nan)
        sens = np.array(["a"] * 30 + ["b"] * 30)

        roc = RejectionOptionClassifier()
        _quiet(lambda: roc.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        assert np.isnan(_quiet(lambda: roc.transform(y_prob, sens))).sum() == 60

        decisions, messages = _caught(lambda: roc.predict(y_prob, sens))
        decided = np.asarray(decisions, dtype=float)
        assert np.isnan(decided).all(), np.unique(decided, return_counts=True)
        assert decided.dtype.kind == "f"
        assert any("no finite adjusted score" in msg for msg in messages), messages
        assert math.isnan(_disparity(roc.result_.original_metrics))

    def test_an_unprivileged_group_absent_from_the_data_is_refused(self):
        """Naming a group that is not there made every row privileged, so every
        in-region row was pushed DOWN: the opposite of the correction asked
        for."""
        y_true, y_prob, sens = _healthy()
        with pytest.raises(ValueError, match="does not appear in sensitive_attr"):
            _quiet(
                lambda: RejectionOptionClassifier(unprivileged_group="Z").fit(
                    y_true=y_true, y_prob=y_prob, sensitive_attr=sens
                )
            )

    def test_control_healthy_data_still_flips_and_measures(self):
        """CONTROL. A real gap is still detected, corrected and reported."""
        y_true, y_prob, sens = _healthy()

        roc = RejectionOptionClassifier()
        out, messages = _caught(lambda: roc.fit_transform(y_true, y_prob, sens))

        assert messages == []
        assert roc.assessable_ is True
        assert roc.detected_unprivileged_ == "B"
        assert roc.result_.calibration_impact["n_modified"] > 0
        assert not np.array_equal(out, y_prob)
        assert np.isfinite(_disparity(roc.result_.original_metrics))
        assert _disparity(roc.result_.original_metrics) == pytest.approx(1.0)
        assert roc.result_.fairness_improvement["disparity_reduction"] > 0.0

        decisions = roc.predict(y_prob, sens)
        assert decisions.dtype.kind == "i"
        assert set(np.unique(decisions)) <= {0, 1}


# ===========================================================================
# create_reweighter: the factory is a public entry of its own
# ===========================================================================


class TestCreateReweighterThreeStates:
    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_single_cohort_is_never_reported_as_perfect_parity(self, method):
        """BEFORE: multiplicative, additive and rejection_option each reported
        disparity 0.0 and disparity_reduction 0.0 on a filtered single-cohort
        frame, which is ordinary input, not a contrived one."""
        rng = np.random.default_rng(11)
        y_prob = rng.uniform(0.05, 0.95, 300)
        y_true = rng.binomial(1, y_prob)
        sens = np.array(["A"] * 300)

        rw = _quiet(lambda: create_reweighter(method))
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert math.isnan(_disparity(rw.result_.original_metrics)), rw.result_.original_metrics
        assert math.isnan(_disparity(rw.result_.adjusted_metrics))
        assert math.isnan(rw.result_.fairness_improvement["disparity_reduction"])
        assert "0.0000 ↑" not in rw.result_.summary()
        assert messages, f"{method}: a structural non-measurement must not be silent"

    @pytest.mark.parametrize("method", FACTORY_METHODS)
    def test_control_healthy_data_still_measures_through_the_factory(self, method):
        """CONTROL. All five methods still measure and close a real gap."""
        y_true, y_prob, sens = _healthy()

        rw = create_reweighter(method)
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))

        assert messages == [], f"{method}: {messages}"
        original = _disparity(rw.result_.original_metrics)
        adjusted = _disparity(rw.result_.adjusted_metrics)
        assert np.isfinite(original) and np.isfinite(adjusted), rw.result_.to_dict()
        assert original > 0.3, (method, original)
        assert rw.result_.fairness_improvement["disparity_reduction"] > 0.0


# ===========================================================================
# The refusal must not have eaten the findings that sit under it
# ===========================================================================


def test_a_measured_zero_gap_is_still_a_measured_zero():
    """Over-correction control at the top level. Two groups drawn from the SAME
    distribution really do have (almost) no gap, and that must stay a number:
    the fix replaced unmeasurable 0.0s, not every 0.0."""
    rng = np.random.default_rng(9)
    sens = np.array(["A", "B"] * 100)
    y_prob = rng.uniform(0.2, 0.8, 200)
    y_true = rng.integers(0, 2, 200)

    results: List[Tuple[str, float]] = []
    for method in FACTORY_METHODS:
        rw = create_reweighter(method)
        _, messages = _caught(lambda: rw.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens))
        assert messages == [], (method, messages)
        original = _disparity(rw.result_.original_metrics)
        assert np.isfinite(original), (method, rw.result_.to_dict())
        results.append((method, original))

    assert all(value < 0.2 for _, value in results), results
