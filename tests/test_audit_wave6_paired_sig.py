"""Wave-6 audit pin: paired_metric_significance must actually be paired.

Finding (HIGH, confirmed twice): the delta bootstrap drew the before-indices
and after-indices INDEPENDENTLY every iteration, which is the
two-independent-samples bootstrap, not the paired bootstrap of
Efron & Tibshirani (1993, ch. 8). On row-aligned before/after data (the
documented primary use case) the two sides are strongly positively
correlated, so independent resampling roughly doubled the delta sd, widened
the delta CI, and inflated the p-value: a real 12pp improvement on n=400 was
detected 0/30 times (proper paired bootstrap: 27/30) and reported as
'consistent with sampling noise'.

The fix resamples ONE index vector per replicate and evaluates both sides on
it whenever the sides are row-aligned (auto when lengths match, forced via
paired=True, disabled via paired=False), and reports the mode that ran in the
returned dict under 'resampling'.
"""

import numpy as np
import pytest

from vfairness.preprocessing.feature_engineering.significance import (
    paired_metric_significance,
)


def _make_aligned_case(seed):
    """n=400, groups A/B 200 each; after = SAME rows with ~24 of B's zeros flipped.

    Errors are strongly correlated between the sides (~94% of predictions are
    unchanged) and the true delta is a real but small ~-0.12 on the
    demographic parity difference.
    """
    rng = np.random.default_rng(seed)
    n = 400
    sens = np.array(["A"] * 200 + ["B"] * 200)
    y_true = rng.integers(0, 2, size=n)
    y_pred_before = np.concatenate(
        [
            (rng.random(200) < 0.5).astype(int),
            (rng.random(200) < 0.3).astype(int),
        ]
    )
    y_pred_after = y_pred_before.copy()
    b_zeros = np.where((sens == "B") & (y_pred_before == 0))[0]
    flip = rng.choice(b_zeros, size=min(24, len(b_zeros)), replace=False)
    y_pred_after[flip] = 1
    return y_true, y_pred_before, y_pred_after, sens


class TestPairedDetectsRealChange:
    """Negative case first: the exact scenario that used to be missed."""

    def test_real_small_difference_on_correlated_models_is_detected(self):
        # Before the fix this case gave p=0.062, significant=False,
        # delta_ci straddling 0, despite a deterministic -0.12 true delta.
        y_true, ypb, ypa, sens = _make_aligned_case(seed=7)
        res = paired_metric_significance(y_true, ypb, sens, y_true, ypa, sens)
        assert res["resampling"] == "paired"
        assert res["delta"] == pytest.approx(-0.12, abs=1e-9)
        assert res["p_value"] < 0.05
        assert res["significant"] is True
        # The delta CI must exclude 0 (the old independent CI did not).
        assert res["delta_ci"][1] < 0.0
        assert "statistically significant" in res["interpretation"]
        assert "sampling noise" not in res["interpretation"]

    def test_power_over_replicates(self):
        # The recorded 27/30-style check: across seeded replicates of the same
        # real improvement, the paired test must detect it a large majority of
        # the time. The pre-fix implementation detected 0-2 of 30.
        detections = 0
        n_reps = 10
        for rep in range(n_reps):
            y_true, ypb, ypa, sens = _make_aligned_case(seed=100 + rep)
            res = paired_metric_significance(
                y_true, ypb, sens, y_true, ypa, sens, n_bootstrap=400, random_state=rep
            )
            if res["significant"]:
                detections += 1
        assert detections >= 7, "paired bootstrap lost its power: %d/%d" % (detections, n_reps)


class TestDoesNotOvercorrect:
    """Two identical models must NOT be flagged."""

    def test_identical_models_not_significant(self):
        y_true, ypb, _ypa, sens = _make_aligned_case(seed=11)
        res = paired_metric_significance(y_true, ypb, sens, y_true, ypb.copy(), sens)
        assert res["resampling"] == "paired"
        assert res["delta"] == pytest.approx(0.0, abs=1e-12)
        assert res["p_value"] == pytest.approx(1.0)
        assert res["significant"] is False
        assert "sampling noise" in res["interpretation"]

    def test_null_false_positive_rate_stays_controlled(self):
        # Independent draws from the SAME distribution (no real effect, and no
        # pairing structure): the test must not start flagging noise.
        false_positives = 0
        n_reps = 10
        for rep in range(n_reps):
            rng = np.random.default_rng(500 + rep)
            n = 400
            sens = np.array(["A"] * 200 + ["B"] * 200)
            y_true = rng.integers(0, 2, size=n)
            ypb = (rng.random(n) < 0.4).astype(int)
            ypa = (rng.random(n) < 0.4).astype(int)
            res = paired_metric_significance(
                y_true, ypb, sens, y_true, ypa, sens, n_bootstrap=400, random_state=rep
            )
            if res["significant"]:
                false_positives += 1
        assert false_positives <= 2, "null rejections: %d/%d" % (false_positives, n_reps)


class TestModeSelection:
    """The paired argument and the auto-detection contract."""

    def test_unequal_lengths_fall_back_to_independent(self):
        y_true, ypb, ypa, sens = _make_aligned_case(seed=3)
        res = paired_metric_significance(
            y_true[:300],
            ypb[:300],
            sens[:300],
            y_true,
            ypa,
            sens,
            n_bootstrap=100,
        )
        assert res["resampling"] == "independent"

    def test_paired_true_with_unequal_lengths_raises(self):
        y_true, ypb, ypa, sens = _make_aligned_case(seed=3)
        with pytest.raises(ValueError, match="equal length"):
            paired_metric_significance(
                y_true[:300],
                ypb[:300],
                sens[:300],
                y_true,
                ypa,
                sens,
                paired=True,
            )

    def test_paired_false_forces_independent_on_aligned_data(self):
        y_true, ypb, ypa, sens = _make_aligned_case(seed=3)
        res = paired_metric_significance(
            y_true, ypb, sens, y_true, ypa, sens, n_bootstrap=100, paired=False
        )
        assert res["resampling"] == "independent"

    def test_seeded_reproducibility(self):
        y_true, ypb, ypa, sens = _make_aligned_case(seed=5)
        r1 = paired_metric_significance(
            y_true, ypb, sens, y_true, ypa, sens, n_bootstrap=200, random_state=9
        )
        r2 = paired_metric_significance(
            y_true, ypb, sens, y_true, ypa, sens, n_bootstrap=200, random_state=9
        )
        assert r1["p_value"] == r2["p_value"]
        assert r1["delta_ci"] == r2["delta_ci"]
