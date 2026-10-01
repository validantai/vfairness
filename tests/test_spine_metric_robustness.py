"""Statistical-robustness + reference-parity tests for the spec-v2 metrics, held to the
engine's existing bar: hand-derived GOLDEN values (independent ground truth), cross-library
reference parity (fairlearn / sklearn / statsmodels), confidence-interval behaviour (bracketing
the point estimate + widening with smaller n), and the three-state verdict from the CI band.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import _fairness_verdict
from vfairness.evaluation.vfairness_metrics.classification import (
    auroc_parity,
    conditional_demographic_disparity,
    disparate_impact_ratio,
    fpr_parity_difference,
    fpr_parity_difference_with_ci,
    negative_predictive_value_difference,
)
from vfairness.post_processing.calibration.metrics import (
    calibration_slope,
    integrated_calibration_index_with_ci,
)

ABS = 1e-9


def _hand_fixture():
    """Explicit per-group confusion matrices so every expected value is an exact fraction.
    Group A: SR .60, FPR .50, NPV 12/16=.75.  Group B: SR .35, FPR .125, NPV 21/26."""
    rows = []

    def add_group(name, tp, fn, fp, tn):
        rows.extend([(name, 1, 1)] * tp)
        rows.extend([(name, 1, 0)] * fn)
        rows.extend([(name, 0, 1)] * fp)
        rows.extend([(name, 0, 0)] * tn)

    add_group("A", tp=12, fn=4, fp=12, tn=12)  # 40 rows
    add_group("B", tp=11, fn=5, fp=3, tn=21)  # 40 rows
    sens = np.array([r[0] for r in rows])
    y_true = np.array([r[1] for r in rows])
    y_pred = np.array([r[2] for r in rows])
    return y_true, y_pred, sens


class TestGoldenValues:
    def test_disparate_impact_ratio_hand_value(self):
        yt, yp, s = _hand_fixture()
        # min/max selection rate = 0.35 / 0.60 = 7/12
        assert disparate_impact_ratio(yt, yp, s) == pytest.approx(7 / 12, abs=ABS)

    def test_negative_predictive_value_difference_hand_value(self):
        yt, yp, s = _hand_fixture()
        # |21/26 - 3/4| = 3/52
        assert negative_predictive_value_difference(yt, yp, s) == pytest.approx(3 / 52, abs=ABS)

    def test_fpr_parity_difference_hand_value(self):
        yt, yp, s = _hand_fixture()
        # |0.50 - 0.125| = 0.375
        assert fpr_parity_difference(yt, yp, s) == pytest.approx(0.375, abs=ABS)

    def test_conditional_demographic_disparity_hand_value(self):
        # stratum X: both groups 50% selected (no within-stratum disparity)
        # stratum Y: A always selected, B never -> full within-stratum disparity
        a = np.array(["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10)
        strata = np.array(["X"] * 20 + ["Y"] * 20)
        yp = np.array(([1, 0] * 5) + ([1, 0] * 5) + [1] * 10 + [0] * 10)
        # CDD = (20/40)*0 + (20/40)*1.0 = 0.5
        assert conditional_demographic_disparity(
            yp, a, strata, min_group_size=1, min_cell=1
        ) == pytest.approx(0.5, abs=ABS)


class TestReferenceParity:
    def test_disparate_impact_ratio_matches_fairlearn(self):
        flm = pytest.importorskip("fairlearn.metrics")
        yt, yp, s = _hand_fixture()
        vf = disparate_impact_ratio(yt, yp, s, min_group_size=1)
        fl = flm.demographic_parity_ratio(yt, yp, sensitive_features=s)
        assert vf == pytest.approx(fl, abs=1e-9)

    def test_auroc_parity_matches_sklearn_per_group(self):
        skm = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(3)
        n = 2000
        a = rng.choice(["M", "F"], n)
        yt = (rng.uniform(0, 1, n) < 0.5).astype(int)
        score = np.where(a == "M", yt * 0.6 + rng.uniform(0, 0.4, n), rng.uniform(0, 1, n))
        aucs = [skm.roc_auc_score(yt[a == g], score[a == g]) for g in ("M", "F")]
        ref = max(aucs) - min(aucs)
        assert auroc_parity(yt, score, a, min_group_size=1) == pytest.approx(ref, abs=1e-9)

    def test_calibration_slope_matches_statsmodels_logit(self):
        sm = pytest.importorskip("statsmodels.api")
        rng = np.random.default_rng(4)
        n = 5000
        p_true = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p_true).astype(int)
        # over-confident predictions -> recalibration slope < 1
        p = np.clip((p_true - 0.5) * 1.6 + 0.5, 1e-4, 1 - 1e-4)
        lp = np.log(p / (1 - p))
        res = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
        ref_slope = float(res.params[1])
        assert calibration_slope(y, p).overall_value == pytest.approx(ref_slope, abs=0.02)


class TestConfidenceIntervalBehaviour:
    def test_ci_brackets_point_and_widens_with_small_n(self):
        rng = np.random.default_rng(5)
        n = 3000
        a = rng.choice(["M", "F"], n)
        yp = np.where(a == "M", rng.uniform(0, 1, n) < 0.55, rng.uniform(0, 1, n) < 0.45).astype(
            int
        )
        yt = (rng.uniform(0, 1, n) < 0.5).astype(int)
        big = fpr_parity_difference_with_ci(yt, yp, a, random_state=0, n_bootstrap=800)
        assert big.lower_bound <= big.point_estimate + 1e-9
        assert big.point_estimate <= big.upper_bound + 1e-9
        sub = rng.choice(n, 100, replace=False)
        small = fpr_parity_difference_with_ci(
            yt[sub], yp[sub], a[sub], random_state=0, n_bootstrap=800, min_group_size=10
        )
        assert small.interval_width > big.interval_width

    def test_ici_disparity_ci_is_finite_and_nonnegative(self):
        rng = np.random.default_rng(7)
        n = 2000
        a = rng.choice(["M", "F"], n)
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        r = integrated_calibration_index_with_ci(y, p, a, random_state=0, n_bootstrap=150)
        assert np.isfinite(r.lower_bound) and np.isfinite(r.upper_bound)
        assert r.lower_bound >= 0.0 - 1e-9


class TestThreeStateVerdictFromCI:
    def test_verdict_logic_fair_unfair_insufficient(self):
        # The metric must be NAMED: _fairness_verdict resolves the fair band from
        # the metric's direction and has no default. It used to assume
        # lower-is-better for an unnamed call, which is a guessed direction and
        # inverts the whole ratio family (see tests/test_verdict_defaults.py).
        name = "fpr_parity_difference"
        # whole CI below threshold -> fair
        assert _fairness_verdict(0.02, 0.01, 0.04, 0.10, metric_name=name) == "fair"
        # whole CI above threshold -> unfair
        assert _fairness_verdict(0.30, 0.25, 0.35, 0.10, metric_name=name) == "unfair"
        # CI straddles the threshold (wide small-sample) -> insufficient
        assert (
            _fairness_verdict(0.09, 0.02, 0.20, 0.10, metric_name=name) == "insufficient_evidence"
        )

    def test_a_new_metric_ci_feeds_the_verdict(self):
        # A clearly-biased large-n case must read as a confident unfair via the CI, not a guess.
        rng = np.random.default_rng(8)
        n = 4000
        a = rng.choice(["M", "F"], n)
        yt = (rng.uniform(0, 1, n) < 0.5).astype(int)
        # force a large FPR gap: M predicts positive on negatives far more often
        yp = np.where(
            (yt == 0) & (a == "M"),
            (rng.uniform(0, 1, n) < 0.7),
            np.where((yt == 0) & (a == "F"), (rng.uniform(0, 1, n) < 0.1), yt),
        ).astype(int)
        r = fpr_parity_difference_with_ci(yt, yp, a, random_state=0, n_bootstrap=800)
        verdict = _fairness_verdict(
            r.point_estimate,
            r.lower_bound,
            r.upper_bound,
            0.05,
            metric_name="fpr_parity_difference",
        )
        assert verdict == "unfair"
