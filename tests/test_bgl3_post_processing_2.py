"""BGL3 batch post_processing-2: does a post-processing unit refuse honestly?

Fourteen units across calibration diagnostics and prediction reweighting, each
executed on an input where the quantity it reports does not exist. Every
docstring below carries the value measured BEFORE the fix, taken from a run of
the real function, not from reading it.

Findings, all executed on 2026-09-27:

  1. ece_confidence_intervals(n_bootstrap=0) returned, for both groups of 100,
     ci_lower == ci_upper == ece, ci_width 0.0, se 0.0 and unstable False, with
     no warning. Zero resamples, reported as a perfectly precise interval.
  2. ece_confidence_intervals(n_bootstrap=1) returned group A's "95% interval"
     as [0.20745, 0.20745], width 0.0, unstable False, around a point estimate
     of 0.15737 that the interval does not even contain.
  3. cv_calibration_stability left a group measurable on ONE fold with
     std_ece 0.0, cv_coefficient 0.0, degradation_flag False: zero fold-to-fold
     variation from a single fold. On ZERO folds it reported
     degradation_flag False with every figure None.
  4. per_group_brier_decomposition / ece_confidence_intervals /
     cv_calibration_stability / sufficiency_test dropped a 2-row group with NO
     warning and no trace in the result. sufficiency_test answered
     passes=True, status='assessed' over the surviving two groups.
  5. BaseReweighter.predict(threshold=nan) returned an int64 array of 120
     zeros, no warning: a confident rejection of every row.
  6. RejectionOptionClassifier(threshold=nan).fit published original
     disparity 0.0, adjusted 0.0, disparity_reduction 0.0 on data whose A/B
     positive-rate gap is exactly 1.00, and warned that "every measurable
     group ['A', 'B'] has the same positive rate (0)".
  7. RejectionOptionClassifier(theta=nan) published critical_region (nan, nan)
     with n_in_critical_region 0 over 120 rows.
  8. CalibratedEqualizer(n_quantiles=0) rewrote all 120 scores to the single
     value 0.0518126 and published adjusted mean_disparity 0.0 with
     disparity_reduction 0.5426, unfittable_groups {} and no warning.
  9. DistributionMatcher with n_bins set to 0 collapsed group B's 60 scores
     onto one value and published disparity_reduction 0.3725.

Every test here was sabotage-checked: the fix was re-broken, the test observed
red, the fix restored, the test observed green. The sabotage is recorded in the
batch report.
"""

import warnings

import numpy as np
import pytest

from vfairness.post_processing.calibration.metrics import (
    cv_calibration_stability,
    ece_confidence_intervals,
    per_group_brier_decomposition,
    sufficiency_test,
)
from vfairness.post_processing.reweighting.analyzer import ReweightingAnalyzer
from vfairness.post_processing.reweighting.reweighter import (
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
    _compute_group_positive_rates,
)


def _two_groups(n_per_group=100, seed=0):
    """Healthy calibration data: two groups of n_per_group, scores in (0, 1)."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_group
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _two_groups_plus_tiny_third(seed=0):
    """The same, with a 2-row group C that every min_group_size=30 drops."""
    y, p, g = _two_groups(seed=seed)
    rng = np.random.default_rng(seed + 1)
    p3 = np.concatenate([p, rng.uniform(0.05, 0.95, 2)])
    y3 = np.concatenate([y, np.array([0, 1])])
    g3 = np.concatenate([g, np.array(["C", "C"])])
    return y3, p3, g3


def _gap_data(seed=7):
    """120 rows whose A/B positive-rate gap at 0.5 is exactly 1.00.

    Group A scores in [0.60, 0.95] and group B in [0.05, 0.40], so every A row
    is accepted and no B row is, and the group mean gap is 0.5426.
    """
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.uniform(0.60, 0.95, 60), rng.uniform(0.05, 0.40, 60)])
    g = np.array(["A"] * 60 + ["B"] * 60)
    y = (rng.uniform(size=120) < p).astype(int)
    return y, p, g


def _critical_region_data():
    """A real gap that the ROC critical region can actually correct.

    Group A sits just above the 0.5 threshold and group B just below it, so
    every row is inside the default critical region (0.4, 0.6): A's positive
    rate is 1.0 and B's is 0.0 before any correction.
    """
    p = np.concatenate([np.linspace(0.52, 0.59, 40), np.linspace(0.41, 0.48, 40)])
    g = np.array(["A"] * 40 + ["B"] * 40)
    y = np.concatenate([np.ones(40, dtype=int), np.zeros(40, dtype=int)])
    return y, p, g


# ===========================================================================
# cv_calibration_stability
# ===========================================================================


class TestCvCalibrationStability:
    def test_zero_measurable_folds_is_not_a_stable_group(self):
        """Measured before: with min_group_size=1 and a 1-row group D, the entry
        was {'fold_eces': [], 'mean_ece': None, 'std_ece': None,
        'cv_coefficient': None, 'degradation_flag': False} and NOT ONE warning.
        degradation_flag False is the reading "this group's calibration holds
        across folds", issued over zero folds."""
        y, p, g = _two_groups()
        y = np.concatenate([y, [1]])
        p = np.concatenate([p, [0.5]])
        g = np.concatenate([g, ["D"]])

        with pytest.warns(UserWarning, match="fold-to-fold spread does not exist"):
            out = cv_calibration_stability(y, p, g, min_group_size=1)

        d = out["D"]
        assert d["n_folds_measured"] == 0
        assert d["degradation_flag"] is None
        assert d["std_ece"] is None
        assert d["cv_coefficient"] is None
        assert d["not_assessed"] and "0 of the 5 fold(s)" in d["not_assessed"]

    def test_one_fold_is_not_zero_fold_to_fold_variation(self):
        """Measured before, 29 + 29 + 2 rows, n_folds=2, min_group_size=2,
        random_state=0: group C came back {'fold_eces': [0.1678580849868459],
        'mean_ece': 0.1678580849868459, 'std_ece': 0.0, 'cv_coefficient': 0.0,
        'degradation_flag': False}. np.std of one value is exactly 0.0, so a
        single fold reported perfect stability. The fold's own ECE is a real
        measurement and is kept; the dispersion over it is refused."""
        rng = np.random.default_rng(11)
        p = rng.uniform(0.05, 0.95, 60)
        y = (rng.uniform(size=60) < p).astype(int)
        g = np.array(["A"] * 29 + ["B"] * 29 + ["C"] * 2)

        with pytest.warns(UserWarning, match="a single fold"):
            out = cv_calibration_stability(y, p, g, n_folds=2, min_group_size=2, random_state=0)

        c = out["C"]
        assert c["n_folds_measured"] == 1
        assert c["mean_ece"] == pytest.approx(0.1678580849868459)
        assert c["std_ece"] is None
        assert c["cv_coefficient"] is None
        assert c["degradation_flag"] is None

    def test_an_undersized_group_is_named_not_dropped(self):
        """Measured before: with A=100, B=100 and a 2-row C the result held only
        'A', 'B' and '__overall__', with zero warnings. The only trace of C was
        its absence from a dict nobody compares against the input."""
        y, p, g = _two_groups_plus_tiny_third()

        with pytest.warns(UserWarning, match="Groups excluded from the cross-validated"):
            out = cv_calibration_stability(y, p, g)

        assert "C" in out
        assert out["C"]["degradation_flag"] is None
        assert "fewer than min_group_size=30" in out["C"]["not_assessed"]

    def test_control_a_healthy_group_is_still_measured_on_every_fold(self):
        """A unit that refuses everything is as wrong as one that answers
        everything. Measured after the fix, unchanged from before it: group A
        keeps 5 fold ECEs, mean 0.23324528301954897, std 0.04093219768437972,
        degradation_flag True."""
        y, p, g = _two_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = cv_calibration_stability(y, p, g)

        a = out["A"]
        assert a["n_folds_measured"] == 5
        assert a["mean_ece"] == pytest.approx(0.23324528301954897)
        assert a["std_ece"] == pytest.approx(0.04093219768437972)
        assert a["degradation_flag"] is True
        assert a["not_assessed"] is None


# ===========================================================================
# ece_confidence_intervals
# ===========================================================================


class TestEceConfidenceIntervals:
    def test_zero_resamples_is_refused(self):
        """Measured before: n_bootstrap=0 returned, for both groups of 100,
        ci_lower == ci_upper == ece (0.1573694620635577 for A), ci_width 0.0,
        se 0.0 and unstable False, with no warning at all. The resample loop
        never executed. Same refusal, same reason, as sufficiency_test's
        n_permutations=0."""
        y, p, g = _two_groups()
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            ece_confidence_intervals(y, p, g, n_bootstrap=0)

    def test_a_single_resample_is_not_a_zero_width_interval(self):
        """Measured before: n_bootstrap=1 gave group A ci_lower == ci_upper ==
        0.20745146968427158, ci_width 0.0, se 0.0, unstable False, around a
        point estimate of 0.1573694620635577 which that "95% interval" does not
        contain. np.percentile of one value returns it at every percentile."""
        y, p, g = _two_groups()
        with pytest.warns(UserWarning, match="no percentile interval could be formed"):
            out = ece_confidence_intervals(y, p, g, n_bootstrap=1)

        a = out["A"]
        assert a["ece"] == pytest.approx(0.1573694620635577)
        assert a["n_bootstrap_effective"] == 1
        assert np.isnan(a["ci_lower"]) and np.isnan(a["ci_upper"])
        assert np.isnan(a["ci_width"]) and np.isnan(a["se"])
        assert a["unstable"] is None
        assert "no percentile interval" in a["not_assessed"]

    def test_an_undersized_group_is_named_not_dropped(self):
        """Measured before: the 2-row group C was absent from the result, with
        no warning, while A and B reported intervals."""
        y, p, g = _two_groups_plus_tiny_third()
        with pytest.warns(UserWarning, match="Groups excluded from the per-group ECE interval"):
            out = ece_confidence_intervals(y, p, g, n_bootstrap=50)

        assert out["C"]["ece"] is None
        assert out["C"]["unstable"] is None
        assert out["C"]["n_bootstrap_effective"] == 0
        assert "fewer than min_group_size=30" in out["C"]["not_assessed"]

    def test_control_a_real_interval_is_still_measured(self):
        """Unchanged by the fix: 50 resamples on group A give ci_lower
        0.11589460845323353, ci_upper 0.2734240523892867, width
        0.15752944393605317, se 0.04178458354563346, unstable True."""
        y, p, g = _two_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = ece_confidence_intervals(y, p, g, n_bootstrap=50)

        a = out["A"]
        assert a["n_bootstrap_effective"] == 50
        assert a["ci_lower"] == pytest.approx(0.11589460845323353)
        assert a["ci_upper"] == pytest.approx(0.2734240523892867)
        assert a["se"] == pytest.approx(0.04178458354563346)
        assert a["unstable"] is True
        assert a["not_assessed"] is None


# ===========================================================================
# per_group_brier_decomposition
# ===========================================================================


class TestPerGroupBrierDecomposition:
    def test_an_undersized_group_is_named_not_dropped(self):
        """Measured before: A=100, B=100 and a 2-row C returned keys
        ['A', 'B', '__overall__'] with zero warnings, while C's two rows WERE
        inside the '__overall__' figures (n_samples 202). The group was in the
        pooled number and absent from the comparison."""
        y, p, g = _two_groups_plus_tiny_third()
        with pytest.warns(UserWarning, match="Groups excluded from the per-group Brier"):
            out = per_group_brier_decomposition(y, p, g)

        assert out["C"]["brier_score"] is None
        assert out["C"]["n_samples"] == 2
        assert "fewer than min_group_size=30" in out["C"]["not_assessed"]
        assert out["__overall__"]["n_samples"] == 202

    def test_control_a_healthy_group_is_still_decomposed(self):
        """Unchanged by the fix: group A keeps brier_score
        0.18394395212985765, reliability 0.036857012474060376, resolution
        0.10121691086691087 and not_assessed None."""
        y, p, g = _two_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = per_group_brier_decomposition(y, p, g)

        assert out["A"]["brier_score"] == pytest.approx(0.18394395212985765)
        assert out["A"]["reliability"] == pytest.approx(0.036857012474060376)
        assert out["A"]["not_assessed"] is None


# ===========================================================================
# sufficiency_test
# ===========================================================================


class TestSufficiencyTest:
    def test_the_verdict_names_the_group_it_does_not_cover(self):
        """Measured before: with a 2-row C beside A=100 and B=100 the overall
        block read passes=True, status='assessed', n_bins_tested=9, and held
        nothing at all about C, with no warning. A pass over two of three
        groups must say which groups it covers."""
        y, p, g = _two_groups_plus_tiny_third()
        with pytest.warns(UserWarning, match="Groups excluded from the sufficiency"):
            overall = sufficiency_test(y, p, g)["overall"]

        assert overall["excluded_groups"] == ["C"]
        assert overall["groups_tested"] == ["A", "B"]
        assert overall["passes"] is True
        assert overall["status"] == "assessed"

    def test_zero_testable_bins_is_still_not_a_pass(self):
        """A correct refusal, re-pinned because this batch depends on it: one
        group means no contingency table in any bin, and the verdict is
        passes=None / status='not_assessable' with n_bins_tested 0. It is not
        True: zero evidence is could-not-check."""
        y, p, _ = _two_groups()
        overall = sufficiency_test(y, p, np.array(["A"] * len(y)))["overall"]
        assert overall["passes"] is None
        assert overall["status"] == "not_assessable"
        assert overall["n_bins_tested"] == 0
        assert "COULD NOT CHECK" in overall["detectability_note"]

    def test_control_a_healthy_verdict_is_still_assessed(self):
        """Unchanged by the fix: two groups of 100 give n_bins_tested 9,
        min_bin_p_value 0.08897301170181363, passes True, excluded_groups []."""
        y, p, g = _two_groups()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            overall = sufficiency_test(y, p, g)["overall"]

        assert overall["n_bins_tested"] == 9
        assert overall["min_bin_p_value"] == pytest.approx(0.08897301170181363)
        assert overall["passes"] is True
        assert overall["excluded_groups"] == []


# ===========================================================================
# BaseReweighter.predict and RejectionOptionClassifier
# ===========================================================================


class TestADecisionThresholdThatIsNotANumber:
    def test_predict_refuses_a_nan_threshold(self):
        """Measured before: PredictionReweighter fitted on 120 rows, then
        predict(p, g, threshold=nan), returned array of dtype int64 whose only
        unique value was 0, over all 120 rows, with NO warning. np.nan >= nan is
        False, so every applicant was rejected with the confidence of a measured
        decision. The NaN-SCORE side of this comparison was already three-state;
        the threshold side was not."""
        y, p, g = _gap_data()
        rw = PredictionReweighter(method="multiplicative")
        rw.fit(y_true=y, y_prob=p, sensitive_attr=g)
        with pytest.raises(ValueError, match="threshold=nan is not a number"):
            rw.predict(p, g, threshold=float("nan"))

    def test_rejection_option_predict_refuses_a_nan_threshold(self):
        """The same hole through RejectionOptionClassifier.predict, which has
        its own override: measured before, dtype int64 with unique value 0."""
        y, p, g = _gap_data()
        roc = RejectionOptionClassifier()
        roc.fit(y_true=y, y_prob=p, sensitive_attr=g)
        with pytest.raises(ValueError, match="threshold=nan is not a number"):
            roc.predict(p, g, threshold=float("nan"))

    def test_rejection_option_fit_refuses_a_nan_threshold(self):
        """Measured before: RejectionOptionClassifier(threshold=nan).fit on data
        whose A/B positive-rate gap is exactly 1.00 published original
        {'disparity': 0.0}, adjusted {'disparity': 0.0} and
        {'disparity_reduction': 0.0}, and warned that "every measurable group
        ['A', 'B'] has the same positive rate (0)". Both the number and the
        warning describe a comparison that never happened."""
        with pytest.raises(ValueError, match="threshold=nan is not a number"):
            RejectionOptionClassifier(threshold=float("nan"))

    def test_rejection_option_refuses_a_nan_critical_region_width(self):
        """Measured before: RejectionOptionClassifier(theta=nan).fit published
        critical_region (nan, nan) with n_in_critical_region 0 and
        disparity_reduction 0.0 over 120 rows. No row can be inside a NaN
        region, so that count is a fact about the data nobody established."""
        with pytest.raises(ValueError, match="theta=nan is not a number"):
            RejectionOptionClassifier(theta=float("nan"))

    def test_the_rate_helper_itself_refuses_a_nan_threshold(self):
        """Pinned directly, because both public callers now refuse the argument
        earlier and a guard nothing can reach is a guard nobody can trust.
        _compute_group_positive_rates is the function that turned a NaN
        threshold into {'A': 0.0, 'B': 0.0} on the 1.00-gap data above, and it
        is imported by analyzer.py as well as used inside this module."""
        _, p, g = _gap_data()
        with pytest.raises(ValueError, match="threshold=nan is not a number"):
            _compute_group_positive_rates(p, g, threshold=float("nan"))

    def test_control_the_rate_helper_still_measures_a_real_gap(self):
        """The same helper at threshold=0.5 returns the measured rates: A 1.0,
        B 0.0, the 1.00 gap every fabrication above hid."""
        _, p, g = _gap_data()
        rates = _compute_group_positive_rates(p, g, threshold=0.5)
        assert rates["A"] == pytest.approx(1.0)
        assert rates["B"] == pytest.approx(0.0)

    def test_control_predict_still_decides_on_a_real_threshold(self):
        """Unchanged by the fix: the same fit and data at threshold=0.5 returns
        an int64 array of 120 decisions, 64 of them positive."""
        y, p, g = _gap_data()
        rw = PredictionReweighter(method="multiplicative")
        rw.fit(y_true=y, y_prob=p, sensitive_attr=g)
        out = rw.predict(p, g, threshold=0.5)
        assert out.dtype.kind == "i"
        assert out.size == 120
        assert int(out.sum()) == 64

    def test_control_predict_still_refuses_an_unscored_row(self):
        """The three-state behaviour on the SCORE side must survive the new
        threshold guard: a NaN score is NaN in the decision vector, not 0."""
        y, p, g = _gap_data()
        rw = PredictionReweighter(method="multiplicative")
        rw.fit(y_true=y, y_prob=p, sensitive_attr=g)
        p_holed = p.copy()
        p_holed[0] = np.nan
        with pytest.warns(UserWarning, match="no finite adjusted"):
            out = rw.predict(p_holed, g, threshold=0.5)
        assert np.isnan(out[0])
        assert out.dtype.kind == "f"

    def test_control_rejection_option_still_moves_rows_and_measures_the_gap(self):
        """The refusals must not neuter the mitigation. On 40 + 40 rows all
        inside the critical region, with A at positive rate 1.0 and B at 0.0,
        the fit names B unprivileged, pushes every B row to threshold + 0.01 and
        every A row to threshold - 0.01, and both disparities come back as
        finite measured numbers.

        On this deliberately extreme fixture the correction OVERSHOOTS: the
        rates swap, so the measured reduction is 1.0 - 1.0 = 0.0. That zero is a
        real measurement of an over-correction, arrived at from two rates that
        were both observed, which is exactly what the NaN cases in this class
        are not."""
        y, p, g = _critical_region_data()
        roc = RejectionOptionClassifier()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roc.fit(y_true=y, y_prob=p, sensitive_attr=g)
            adjusted = roc.transform(p, g)

        assert roc.detected_unprivileged_ == "B"
        assert np.allclose(adjusted[40:], 0.51)
        assert np.allclose(adjusted[:40], 0.49)
        assert roc.result_.original_metrics["disparity"] == pytest.approx(1.0)
        assert roc.result_.adjusted_metrics["disparity"] == pytest.approx(1.0)
        assert roc.result_.calibration_impact["n_modified"] == 80
        assert roc.result_.unfittable_groups == {}


# ===========================================================================
# CalibratedEqualizer and DistributionMatcher grids
# ===========================================================================


class TestAQuantileGridTooSmallToBeAMapping:
    def test_calibrated_equalizer_refuses_a_one_point_grid(self):
        """Measured before: CalibratedEqualizer(n_quantiles=0) on 120 rows
        carrying a group mean gap of 0.5425700945559788 rewrote all 120 scores
        to the single value 0.0518126, then published adjusted mean_disparity
        0.0, disparity_reduction 0.5425700945559788, unfittable_groups {} and
        not one warning. np.linspace(0, 100, 1) is ONE percentile and
        np.interp against it returns one constant for every input: the
        mitigation threw the scores away and reported perfect parity."""
        with pytest.raises(ValueError, match="n_quantiles must be at least 1"):
            CalibratedEqualizer(n_quantiles=0)

    def test_calibrated_equalizer_fit_rechecks_the_grid_it_builds(self):
        """n_quantiles is a plain attribute, so the constructor guard alone can
        be walked around. fit() builds the grid and refuses there too."""
        y, p, g = _gap_data()
        eq = CalibratedEqualizer()
        eq.n_quantiles = 0
        with pytest.raises(ValueError, match="n_quantiles must be at least 1"):
            eq.fit(y_true=y, y_prob=p, sensitive_attr=g)

    def test_distribution_matcher_fit_rechecks_the_grid_it_builds(self):
        """Measured before, with n_bins set to 0 on the same 120 rows: group B's
        60 scores collapsed onto a single value and the fit published adjusted
        mean_disparity 0.1700627344933272 with disparity_reduction
        0.3725073600626516, no warning."""
        y, p, g = _gap_data()
        m = DistributionMatcher()
        m.n_bins = 0
        with pytest.raises(ValueError, match="n_bins must be at least 1"):
            m.fit(y_true=y, y_prob=p, sensitive_attr=g)

    def test_control_the_default_equalizer_still_equalises(self):
        """Unchanged by the fix: the default 100 quantiles take the group mean
        gap from 0.5425700945559788 to 0.0003635790179041809, a measured
        reduction of 0.5422065155380746 with no group refused."""
        y, p, g = _gap_data()
        eq = CalibratedEqualizer()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eq.fit(y_true=y, y_prob=p, sensitive_attr=g)

        assert eq.result_.original_metrics["mean_disparity"] == pytest.approx(0.5425700945559788)
        assert eq.result_.adjusted_metrics["mean_disparity"] == pytest.approx(0.0003635790179041809)
        assert eq.result_.unfittable_groups == {}
        assert len(eq.quantile_maps_["A"]) == 101

    def test_control_transform_still_maps_a_fitted_group_and_holds_a_hole_open(self):
        """CalibratedEqualizer.transform on a fitted group maps its scores, and
        a row whose own score is NaN comes back NaN rather than at the pooled
        maximum (which is what np.interp returns for an unfittable constant
        grid, measured at 0.9470751842323551 in BGL-S2)."""
        y, p, g = _gap_data()
        eq = CalibratedEqualizer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eq.fit(y_true=y, y_prob=p, sensitive_attr=g)
            p_holed = p.copy()
            p_holed[0] = np.nan
            adjusted = eq.transform(p_holed, g)

        assert np.isnan(adjusted[0])
        assert np.isfinite(adjusted[1:]).all()
        assert not np.allclose(adjusted[60:], p[60:])

    def test_control_distribution_matcher_still_refuses_a_constant_group(self):
        """A correct refusal this batch depends on: group B with one distinct
        score across 60 rows gets no grid, its rows come back NaN, and the
        reduction is NaN rather than a number off an undefined grid."""
        y, p, g = _gap_data()
        p_const = p.copy()
        p_const[60:] = 0.3
        m = DistributionMatcher()
        with pytest.warns(UserWarning, match="single distinct score"):
            m.fit(y_true=y, y_prob=p_const, sensitive_attr=g)
            adjusted = m.transform(p_const, g)

        assert "B" in m.unfittable_groups_
        assert np.isnan(adjusted[60:]).all()
        assert np.isnan(m.result_.fairness_improvement["disparity_reduction"])

    def test_control_distribution_matcher_still_matches_two_healthy_groups(self):
        """Unchanged by the fix: the default grid takes the group mean gap from
        0.5425700945559788 to 0.00011892399046253832."""
        y, p, g = _gap_data()
        m = DistributionMatcher()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.fit(y_true=y, y_prob=p, sensitive_attr=g)

        assert m.result_.adjusted_metrics["mean_disparity"] == pytest.approx(0.00011892399046253832)
        assert m.result_.unfittable_groups == {}


# ===========================================================================
# ReweightingAnalyzer: the four units that were already honest
# ===========================================================================


class TestTheReweightingAnalyzerRefusesWhatItCannotCompare:
    def test_analyze_method_reports_nan_when_no_row_is_scored(self):
        """Executed on 200 rows whose every probability is NaN: the result
        carries demographic_parity_diff NaN, group_rates {'A': nan, 'B': nan},
        accuracy NaN with n_decided 0 of 200, both ECEs NaN with 0 rows used,
        and trade_off_score NaN, beside 16 warnings. Not one 0.0."""
        y, _, g = _two_groups()
        analyzer = ReweightingAnalyzer(y, np.full(len(y), np.nan), g)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = analyzer.analyze_method("multiplicative")

        assert np.isnan(result.original_fairness["demographic_parity_diff"])
        assert np.isnan(result.original_performance["accuracy"])
        assert result.original_performance["n_decided"] == 0
        assert result.calibration_metrics["original_ece_rows_used"] == 0
        assert np.isnan(result.trade_off_score)
        assert len(caught) > 0

    def test_compare_methods_carries_the_nan_rather_than_a_zero(self):
        """One group means no between-group gap exists. Executed: both methods
        come back with demographic_parity_diff NaN and trade_off_score NaN, so a
        caller reading the comparison cannot mistake either for a measured
        parity."""
        y, p, _ = _two_groups()
        one_group = np.array(["A"] * len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = ReweightingAnalyzer(y, p, one_group)
            results = analyzer.compare_methods(["multiplicative", "calibrated"])

        assert set(results) == {"multiplicative", "calibrated"}
        for name, result in results.items():
            assert np.isnan(result.original_fairness["demographic_parity_diff"]), name
            assert np.isnan(result.trade_off_score), name

    def test_full_analysis_ranks_nothing_when_nothing_is_measurable(self):
        """Executed on one group: best_method is the string
        'None (all methods failed)', method_results is empty, and all five
        methods are in failed_methods with a NotMeasurable reason. No method is
        recommended off a 0.000 improvement."""
        y, p, _ = _two_groups()
        one_group = np.array(["A"] * len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = ReweightingAnalyzer(y, p, one_group).full_analysis()

        assert report.method_results == []
        assert report.best_method == "None (all methods failed)"
        assert len(report.failed_methods) == 5
        assert all("NotMeasurable" in reason for reason in report.failed_methods.values())
        assert "No methods could be analyzed successfully." in report.recommendations

    def test_the_report_summary_says_nothing_was_compared(self):
        """ReweightingAnalysisReport.summary() on that report: 'Analyzed 0
        reweighting methods', 'Recommended method: None (all methods failed)'
        and a WARNING block naming all five failures. The reader is never
        handed a method name as a recommendation."""
        y, p, _ = _two_groups()
        one_group = np.array(["A"] * len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = ReweightingAnalyzer(y, p, one_group).full_analysis().summary()

        assert "Analyzed 0 reweighting methods" in text
        assert "Recommended method: None (all methods failed)" in text
        assert "WARNING: 5 method(s) failed" in text

    def test_get_explanation_does_not_narrate_a_measurement(self):
        """The educational surface must not outrun the code. Executed on one
        group: summary reads 'Compared 0 reweighting method(s). Best trade-off
        achieved by: None (all methods failed).' and the recommendations carry
        'No methods could be analyzed successfully.'"""
        y, p, _ = _two_groups()
        one_group = np.array(["A"] * len(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explanation = ReweightingAnalyzer(y, p, one_group).get_explanation()

        payload = explanation.to_dict()
        assert "Compared 0 reweighting method(s)" in payload["summary"]
        assert "None (all methods failed)" in payload["summary"]
        assert "No methods could be analyzed successfully." in payload["recommendations"]

    def test_control_a_healthy_frame_is_still_ranked_and_explained(self):
        """Two groups with a real 1.00 gap: all five methods are analysed, none
        fails, a real method is recommended, and the explanation names it."""
        y, p, g = _two_groups(seed=3)
        rng = np.random.default_rng(3)
        p = np.concatenate([rng.uniform(0.55, 0.95, 100), rng.uniform(0.05, 0.45, 100)])
        y = (rng.uniform(size=200) < p).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = ReweightingAnalyzer(y, p, g)
            report = analyzer.full_analysis()
            explanation = analyzer.get_explanation(report)

        assert len(report.method_results) == 5
        assert report.failed_methods == {}
        assert report.best_method in {
            "multiplicative",
            "additive",
            "rejection_option",
            "calibrated",
            "distribution_matching",
        }
        assert (
            f"Best trade-off achieved by: {report.best_method}" in explanation.to_dict()["summary"]
        )
