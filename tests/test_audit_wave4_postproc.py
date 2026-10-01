"""Audit wave 4 pins: post-processing calibration and reweighting.

Each test pins one adversarially confirmed audit finding so the fix
cannot silently regress.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.post_processing.reweighting.analyzer as rw_mod
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.post_processing.calibration.group_calibrator import (
    IntersectionalCalibrator,
)
from vfairness.post_processing.calibration.metrics import (
    CalibrationMetricResult,
    brier_score,
    brier_score_decomposition,
    calibration_curve,
    calibration_disparity,
    expected_calibration_error,
    maximum_calibration_error,
)
from vfairness.post_processing.calibration.tradeoffs import (
    ParetoPoint,
    TradeoffAnalysisResult,
)
from vfairness.post_processing.reweighting.analyzer import (
    ReweightingAnalyzer,
    _compute_calibration_error,
)

RNG = np.random.default_rng(42)


def _binary_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = rng.uniform(0.01, 0.99, n)
    g = np.where(rng.uniform(size=n) < 0.5, "a", "b")
    return y, p, g


# ── F1: analyzer memoization must respect varying arguments ────────────────


class TestAnalyzerCacheKeys:
    def test_evaluate_ece_strategy_not_ignored(self):
        y, p, g = _binary_data()
        an = CalibrationAnalyzer(y, p, g)
        e_u = an.evaluate_ece(strategy="uniform")
        e_q = an.evaluate_ece(strategy="quantile")
        assert e_q.metadata["strategy"] == "quantile"
        assert e_u.metadata["strategy"] == "uniform"
        # value must match a fresh computation, not the cached uniform one
        fresh = CalibrationAnalyzer(y, p, g).evaluate_ece(strategy="quantile")
        assert e_q.overall_value == pytest.approx(fresh.overall_value)

    def test_evaluate_mce_and_brier_decomp_strategy_not_ignored(self):
        y, p, g = _binary_data()
        an = CalibrationAnalyzer(y, p, g)
        m_u = an.evaluate_mce(strategy="uniform")
        m_q = an.evaluate_mce(strategy="quantile")
        assert m_q.metadata["strategy"] == "quantile"
        assert m_u.metadata["strategy"] == "uniform"

        d_u = an.decompose_brier(strategy="uniform")
        d_q = an.decompose_brier(strategy="quantile")
        fresh_q = CalibrationAnalyzer(y, p, g).decompose_brier(strategy="quantile")
        assert d_q.reliability == pytest.approx(fresh_q.reliability)
        # cached uniform result still returned for uniform calls
        assert an.decompose_brier(strategy="uniform") is d_u

    def test_repeat_call_still_memoized(self):
        y, p, g = _binary_data()
        an = CalibrationAnalyzer(y, p, g)
        first = an.evaluate_ece(strategy="quantile")
        assert an.evaluate_ece(strategy="quantile") is first

    def test_clear_cache_resets_keyed_caches(self):
        y, p, g = _binary_data()
        an = CalibrationAnalyzer(y, p, g)
        first = an.evaluate_ece(strategy="uniform")
        an.clear_cache()
        assert an.evaluate_ece(strategy="uniform") is not first

    def test_get_recommendation_context_not_ignored(self):
        y, p, g = _binary_data()
        an = CalibrationAnalyzer(y, p, g)
        r_gen = an.get_recommendation(context="general")
        r_health = an.get_recommendation(context="healthcare")
        # distinct cache slots: repeated calls return their own objects
        assert an.get_recommendation(context="general") is r_gen
        assert an.get_recommendation(context="healthcare") is r_health


# ── F2: intersectional group keys must not collide on '_' ──────────────────


class TestIntersectionalKeyCollision:
    def test_underscore_values_get_distinct_calibrators(self):
        rng = np.random.default_rng(1)
        n = 200
        y = rng.integers(0, 2, n)
        p = rng.uniform(0.01, 0.99, n)
        # ('x_y', 'z') and ('x', 'y_z') used to both map to key 'x_y_z'
        attrs = pd.DataFrame(
            {
                "attr1": ["x_y"] * 100 + ["x"] * 100,
                "attr2": ["z"] * 100 + ["y_z"] * 100,
            }
        )
        ic = IntersectionalCalibrator(method="platt", min_group_size=30)
        ic.fit(y_true=y, y_prob=p, protected_attrs=attrs)
        assert len(ic.calibrators_) == 2

    def test_keys_without_underscores_unchanged(self):
        rng = np.random.default_rng(2)
        n = 120
        y = rng.integers(0, 2, n)
        p = rng.uniform(0.01, 0.99, n)
        attrs = pd.DataFrame(
            {
                "gender": ["male"] * 60 + ["female"] * 60,
                "race": ["white"] * 60 + ["black"] * 60,
            }
        )
        ic = IntersectionalCalibrator(method="platt", min_group_size=30)
        ic.fit(y_true=y, y_prob=p, protected_attrs=attrs)
        assert set(ic.calibrators_) == {"male_white", "female_black"}
        # lookup helper uses the same escaping and still finds the group
        assert (
            ic.get_calibrator_for_group({"gender": "male", "race": "white"})
            is ic.calibrators_["male_white"]
        )

    def test_hierarchical_borrowing_with_underscore_values(self):
        rng = np.random.default_rng(3)
        # parent group 'new_york' is large, one intersection is tiny and
        # must borrow the parent's calibrator (split('_') used to mangle
        # the value into 'new' / 'york' and miss the parent)
        n_big, n_tiny = 100, 5
        y = rng.integers(0, 2, n_big + n_tiny)
        p = rng.uniform(0.01, 0.99, n_big + n_tiny)
        attrs = pd.DataFrame(
            {
                "city": ["new_york"] * (n_big + n_tiny),
                "grp": ["a"] * n_big + ["b"] * n_tiny,
            }
        )
        ic = IntersectionalCalibrator(
            method="platt", min_group_size=30, borrowing_strategy="hierarchical"
        )
        ic.fit(y_true=y, y_prob=p, protected_attrs=attrs)
        tiny_key = ic._create_group_key(pd.Series({"city": "new_york", "grp": "b"}))
        assert ic.calibrators_[tiny_key] is ic.parent_calibrators_["city"]["new_york"]
        # transform runs end to end on the same attribute frame
        out = ic.transform(p, attrs)
        assert out.shape == p.shape
        assert np.all((out >= 0) & (out <= 1))


# ── F3: y_true must be validated binary ─────────────────────────────────────


class TestBinaryLabelValidation:
    @pytest.mark.parametrize(
        "fn",
        [
            expected_calibration_error,
            maximum_calibration_error,
            brier_score,
            brier_score_decomposition,
            calibration_curve,
        ],
    )
    def test_non_binary_labels_rejected(self, fn):
        y_bad = np.array([0, 5, 3, 2] * 25)
        p = np.linspace(0.01, 0.99, 100)
        with pytest.raises(ValueError, match="y_true"):
            fn(y_bad, p)

    def test_binary_labels_still_accepted(self):
        y, p, _ = _binary_data()
        result = expected_calibration_error(y, p)
        assert 0.0 <= result.overall_value <= 1.0


# ── F4: is_well_calibrated is metric-aware ──────────────────────────────────


class TestIsWellCalibratedMetricAware:
    def test_ece_threshold_unchanged(self):
        assert CalibrationMetricResult("expected_calibration_error", 0.04).is_well_calibrated
        assert not CalibrationMetricResult("expected_calibration_error", 0.06).is_well_calibrated

    def test_mce_uses_looser_worst_case_threshold(self):
        # 0.08 is a fine worst-case bin error; the old ECE cutoff called it bad
        assert CalibrationMetricResult("maximum_calibration_error", 0.08).is_well_calibrated
        assert not CalibrationMetricResult("maximum_calibration_error", 0.12).is_well_calibrated

    def test_brier_uses_climatology_threshold(self):
        # 0.20 beats the uninformative p=0.5 forecast (Brier 0.25)
        assert CalibrationMetricResult("brier_score", 0.20).is_well_calibrated
        assert not CalibrationMetricResult("brier_score", 0.30).is_well_calibrated

    def test_unknown_metric_is_not_graded_at_all(self):
        # REVISED 2026-09-10 (readiness-6). This pinned the ECE-threshold
        # FALLBACK: an unrecognised name was graded 0.04 -> True / 0.06 -> False
        # by ``value < 0.05``, a hardcoded lower-is-better rule with an
        # ECE-shaped bound and no third state. What this class exists to pin is
        # that the check is METRIC-AWARE (the three tests above), and the
        # fallback was the opposite of metric-aware: it answered for a metric it
        # had never heard of. Handed a higher-is-better name the same fallback
        # inverted outright -- measured, ``worst_group_accuracy`` at 0.04 read
        # True and at 0.99 read False -- and ``to_dict()`` serialised that.
        #
        # There are three states now, so an unrecognised name gets None: not a
        # pass, not a failure, and never a number nobody chose applied to a
        # metric nobody registered.
        assert CalibrationMetricResult("some_new_metric", 0.04).is_well_calibrated is None
        assert CalibrationMetricResult("some_new_metric", 0.06).is_well_calibrated is None
        # ... and it is None in the serialised form too, which is where the
        # inverted verdict used to leave the building.
        assert (
            CalibrationMetricResult("some_new_metric", 0.04).to_dict()["is_well_calibrated"] is None
        )

    def test_a_higher_is_better_name_is_never_graded_by_an_error_bound(self):
        # The inversion the fallback produced, kept as its own NEGATIVE case.
        for name in ("worst_group_accuracy", "disparate_impact_ratio", "exposure_parity_ratio"):
            assert CalibrationMetricResult(name, 0.04).is_well_calibrated is None
            assert CalibrationMetricResult(name, 0.99).is_well_calibrated is None

    def test_an_unmeasurable_value_is_could_not_check_not_a_failure(self):
        # ``nan < 0.05`` is False, so an ECE that could not be computed read as
        # "not well calibrated": fail-closed, but still a verdict nobody measured.
        assert (
            CalibrationMetricResult("expected_calibration_error", float("nan")).is_well_calibrated
            is None
        )
        assert CalibrationMetricResult("calibration_slope", float("nan")).is_well_calibrated is None


# ── F5: excluded small groups are disclosed ─────────────────────────────────


class TestDisparityExcludedGroups:
    def _data_with_tiny_group(self):
        rng = np.random.default_rng(4)
        g = np.array(["big1"] * 180 + ["big2"] * 180 + ["tiny"] * 40)
        y = rng.integers(0, 2, 400)
        p = rng.uniform(0.01, 0.99, 400)
        # tiny group is horribly miscalibrated but below min_group_size
        p[360:] = 0.95
        y[360:] = 0
        return y, p, g

    def test_excluded_group_reported_and_warned(self):
        y, p, g = self._data_with_tiny_group()
        with pytest.warns(UserWarning, match="excluded from calibration disparity"):
            d = calibration_disparity(y, p, g, min_group_size=50)
        assert d.excluded_groups == ["tiny"]
        assert d.to_dict()["excluded_groups"] == ["tiny"]
        assert any("NOT ASSESSED" in r for r in d.recommendations)

    def test_no_exclusions_means_empty_field(self):
        y, p, g = _binary_data()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = calibration_disparity(y, p, g, min_group_size=30)
        assert d.excluded_groups == []

    def test_single_valid_group_path_carries_exclusions(self):
        rng = np.random.default_rng(5)
        g = np.array(["big"] * 100 + ["tiny"] * 10)
        y = rng.integers(0, 2, 110)
        p = rng.uniform(0.01, 0.99, 110)
        with pytest.warns(UserWarning):
            d = calibration_disparity(y, p, g, min_group_size=50)
        assert d.excluded_groups == ["tiny"]
        assert d.n_groups == 1


# ── F6: reweighting ECE top bin includes probability 1.0 ────────────────────


class TestReweightingEceTopBin:
    def test_probability_one_counted(self):
        y = np.zeros(100)
        p = np.ones(100)
        # every prediction is p=1.0 and wrong: true ECE is 1.0, the open
        # upper bin edge used to drop all samples and report 0.0
        assert _compute_calibration_error(y, p) == pytest.approx(1.0)

    def test_perfect_confident_predictions_stay_zero(self):
        y = np.ones(50)
        p = np.ones(50)
        assert _compute_calibration_error(y, p) == pytest.approx(0.0)

    def test_interior_values_unaffected(self):
        y, p, _ = _binary_data()
        ece_ref = expected_calibration_error(y, p).overall_value
        assert _compute_calibration_error(y, p) == pytest.approx(ece_ref)


# ── F7: permanent method failures disclosed in the report ───────────────────


class _BoomReweighter:
    def fit(self, *args, **kwargs):
        raise RuntimeError("boom")

    def transform(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("boom")


class TestReweightingFailureDisclosure:
    def test_full_analysis_records_failed_methods(self, monkeypatch):
        y, p, g = _binary_data()
        monkeypatch.setattr(rw_mod, "PredictionReweighter", lambda **kw: _BoomReweighter())
        ra = ReweightingAnalyzer(y, p, g)
        with pytest.warns(UserWarning, match="Failed to analyze multiplicative"):
            report = ra.full_analysis()
        assert set(report.failed_methods) == {"multiplicative", "additive"}
        assert "RuntimeError: boom" in report.failed_methods["multiplicative"]
        # still recommends from survivors, but discloses the gap everywhere
        assert report.best_method not in report.failed_methods
        assert any("NOT EVALUATED" in r for r in report.recommendations)
        assert report.to_dict()["failed_methods"] == report.failed_methods
        assert "failed and" in report.summary()

    def test_no_failures_means_empty_field(self):
        y, p, g = _binary_data()
        report = ReweightingAnalyzer(y, p, g).full_analysis()
        assert report.failed_methods == {}
        assert not any("NOT EVALUATED" in r for r in report.recommendations)

    def test_compare_methods_warns_not_prints(self, monkeypatch, capsys):
        y, p, g = _binary_data()
        monkeypatch.setattr(rw_mod, "PredictionReweighter", lambda **kw: _BoomReweighter())
        ra = ReweightingAnalyzer(y, p, g)
        with pytest.warns(UserWarning, match="Failed to analyze"):
            results = ra.compare_methods(["multiplicative", "calibrated"])
        assert "multiplicative" not in results
        assert "calibrated" in results
        assert capsys.readouterr().out == ""


# ── F8: tradeoff_severity band boundary pins (no code change) ───────────────


class TestTradeoffSeverityBands:
    @pytest.mark.parametrize(
        "disparity,expected",
        [
            (0.0, "minimal"),
            (0.0499, "minimal"),
            (0.05, "moderate"),
            (0.1499, "moderate"),
            (0.15, "severe"),
            (0.5, "severe"),
        ],
    )
    def test_bands(self, disparity, expected):
        pt = ParetoPoint(calibration_error=0.1, fairness_violation=0.1)
        result = TradeoffAnalysisResult(
            pareto_points=[pt],
            current_point=pt,
            impossibility_diagnosis={},
            recommendations=[],
            base_rate_disparity=disparity,
        )
        assert result.tradeoff_severity == expected
        assert result.to_dict()["tradeoff_severity"] == expected
