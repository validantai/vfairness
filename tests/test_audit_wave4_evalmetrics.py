"""Audit wave 4 regression tests: evaluation metrics core.

Pins the fixes for the adversarially confirmed audit findings in
_statistics.py, _validation.py, analyzer.py, classification.py,
integrations.py and intersectional.py (unit: Evaluation metrics core).
"""

import json
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._validation import (
    validate_probabilities,
)
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.classification import (
    demographic_parity_difference,
    equal_opportunity_difference,
    selection_rate_disparity_matrix,
)
from vfairness.evaluation.vfairness_metrics.intersectional import (
    generate_structured_findings,
    identify_privileged_groups,
)


class TestEmptyDataCIDegradesGracefully:
    """_statistics.compute_metric_with_ci: all rows excluded must not crash."""

    def test_all_rows_excluded_include_ci_no_crash(self):
        analyzer = FairnessAnalyzer(
            np.array([np.nan, np.nan]),
            np.array([1.0, 0.0]),
            np.array(["a", "b"]),
            task_type="classification",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyzer.compute_all_metrics(include_ci=True)
        res = results["demographic_parity_difference"]
        assert np.isnan(res.confidence_interval[0])
        assert np.isnan(res.confidence_interval[1])

    def test_empty_data_ci_warns(self):
        from vfairness.evaluation.vfairness_metrics.classification import (
            demographic_parity_difference_with_ci,
        )

        with pytest.warns(UserWarning, match="No data remains"):
            result = demographic_parity_difference_with_ci(
                np.array([np.nan, np.nan]),
                np.array([1.0, 0.0]),
                np.array(["a", "b"]),
                n_bootstrap=20,
            )
        assert result.method == "unavailable_no_data"
        assert result.sample_size == 0


class TestValidateProbabilitiesRejectsNaN:
    """_validation.validate_probabilities: NaN must raise, not pass."""

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            validate_probabilities(np.array([0.2, np.nan, 0.7]))

    def test_valid_probs_still_pass(self):
        validate_probabilities(np.array([0.0, 0.5, 1.0]))

    def test_out_of_range_still_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            validate_probabilities(np.array([0.2, 1.5]))


class TestErrorRateEffectSizes:
    """analyzer: effect_size must reflect the quantity the metric measures.

    Construction: both groups select at rate 0.5 (selection-rate effect is
    exactly 0) but group A has TPR=1/FPR=0 and group B has TPR=0/FPR=1, so
    equal_opportunity_difference and equalized_odds_difference are both 1.0.
    The old selection-rate effect reported 'negligible' here.
    """

    @pytest.fixture
    def analyzer(self):
        yt = np.concatenate(
            [
                np.array([1] * 100 + [0] * 100),  # group A truth
                np.array([1] * 100 + [0] * 100),  # group B truth
            ]
        )
        yp = np.concatenate(
            [
                np.array([1] * 100 + [0] * 100),  # A: perfect
                np.array([0] * 100 + [1] * 100),  # B: inverted
            ]
        )
        grp = np.array(["A"] * 200 + ["B"] * 200)
        return FairnessAnalyzer(yt, yp, grp, task_type="classification")

    def test_equal_opportunity_effect_matches_tpr_gap(self, analyzer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.equal_opportunity_difference(
                include_ci=True, n_bootstrap=30, random_state=0
            )
        assert r.value == pytest.approx(1.0)
        # Cohen's h on TPR gap (1.0 vs 0.0) = pi, a LARGE effect
        assert r.effect_size == pytest.approx(np.pi)
        assert "large" in r.effect_interpretation
        assert "TPR" in r.effect_interpretation

    def test_equalized_odds_effect_matches_error_rate_gap(self, analyzer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.equalized_odds_difference(include_ci=True, n_bootstrap=30, random_state=0)
        assert r.value == pytest.approx(1.0)
        assert r.effect_size == pytest.approx(np.pi)
        assert "large" in r.effect_interpretation

    def test_demographic_parity_effect_unchanged(self, analyzer):
        # DP measures the selection rate, so its effect stays the
        # selection-rate effect and must be ~0 here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.demographic_parity_difference(
                include_ci=True, n_bootstrap=30, random_state=0
            )
        assert r.effect_size == pytest.approx(0.0, abs=1e-12)


class TestJsonSerializableBoundary:
    """analyzer: to_dict()/get_report() must be json.dumps-able."""

    @pytest.fixture
    def analyzer(self):
        rng = np.random.default_rng(7)
        n = 120
        yt = rng.integers(0, 2, n)
        yp = rng.integers(0, 2, n)
        # include NaN labels so n_excluded > 0 (was np.int64 before)
        yt = yt.astype(float)
        yt[:3] = np.nan
        grp = np.array(["x", "y"] * (n // 2))
        return FairnessAnalyzer(yt, yp, grp, task_type="classification")

    def test_metric_result_to_dict_json(self, analyzer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.demographic_parity_difference(
                include_ci=True, n_bootstrap=30, random_state=0
            )
        d = r.to_dict()
        json.dumps(d)  # must not raise
        assert isinstance(d["is_fair"], bool)
        assert all(isinstance(v, int) for v in d["group_sizes"].values())

    def test_get_report_json(self, analyzer):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = analyzer.get_report()
        json.dumps(report)  # must not raise (n_excluded was np.int64)


class TestEqualOpportunityUndefinedTPR:
    """classification: zero positive labels in a group -> NaN + warning."""

    def test_returns_nan_with_warning(self):
        # Group A has NO positive labels: TPR undefined
        yt = np.array([0] * 50 + [1] * 25 + [0] * 25)
        yp = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        g = np.array(["A"] * 50 + ["B"] * 50)
        with pytest.warns(UserWarning, match="TPR is undefined"):
            v = equal_opportunity_difference(yt, yp, g)
        assert np.isnan(v)

    def test_downstream_is_fair_does_not_crash(self):
        yt = np.array([0] * 50 + [1] * 25 + [0] * 25)
        yp = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        g = np.array(["A"] * 50 + ["B"] * 50)
        analyzer = FairnessAnalyzer(yt, yp, g, task_type="classification")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.equal_opportunity_difference(
                include_ci=True, n_bootstrap=20, random_state=0
            )
        assert np.isnan(r.value)
        # NaN cannot demonstrate fairness; must be a plain False, not a crash
        assert bool(r.is_fair) is False
        json.dumps(r.to_dict())

    def test_defined_tpr_unchanged(self):
        yt = np.array([1] * 25 + [0] * 25 + [1] * 25 + [0] * 25)
        yp = np.array([1] * 25 + [0] * 25 + [0] * 25 + [0] * 25)
        g = np.array(["A"] * 50 + ["B"] * 50)
        v = equal_opportunity_difference(yt, yp, g)
        assert v == pytest.approx(1.0)


class TestPartialGroupDropWarns:
    """classification: dropping SOME groups must warn, naming them."""

    def test_warns_and_names_dropped_group(self):
        yt = np.array([0, 1] * 40)
        yp = np.array([0, 1] * 40)
        g = np.array(["big"] * 70 + ["tiny"] * 10)
        with pytest.warns(UserWarning, match="tiny"):
            demographic_parity_difference(yt, yp, g)

    def test_no_warning_when_all_groups_survive(self):
        yt = np.array([0, 1] * 40)
        yp = np.array([0, 1] * 40)
        g = np.array(["a"] * 40 + ["b"] * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            demographic_parity_difference(yt, yp, g)


class TestSelectionRateDisparityMatrixInputs:
    """classification.selection_rate_disparity_matrix input handling."""

    def test_unrecognized_labels_raise(self):
        with pytest.raises(ValueError, match="ja"):
            selection_rate_disparity_matrix(
                np.array(["Ja", "Nein", "Ja", "Nein"] * 10),
                np.array(["m", "f"] * 20),
            )

    def test_recognized_labels_still_work(self):
        res = selection_rate_disparity_matrix(
            np.array(["Yes", "No"] * 20),
            np.array(["m", "f"] * 20),
        )
        assert res["rates"]["m"]["rate"] == pytest.approx(1.0)
        assert res["rates"]["f"]["rate"] == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            selection_rate_disparity_matrix(np.array([1, 0, 1]), np.array(["m", "f"]))

    def test_confidence_level_honored(self):
        yp = np.array([1, 1, 1, 0] * 25)
        sens = np.array(["m", "f"] * 50)
        r99 = selection_rate_disparity_matrix(yp, sens, confidence_level=0.99)
        r95 = selection_rate_disparity_matrix(yp, sens, confidence_level=0.95)
        # a 99% interval must be strictly wider than a 95% one
        w99 = r99["rates"]["m"]["ci_high"] - r99["rates"]["m"]["ci_low"]
        w95 = r95["rates"]["m"]["ci_high"] - r95["rates"]["m"]["ci_low"]
        assert w99 > w95


class TestIntegrationsThresholdsMatchReport:
    """integrations.assert_fairness defaults must equal report defaults."""

    def test_defaults_agree_for_shared_metrics(self):
        import inspect

        from vfairness.evaluation.vfairness_metrics import integrations, report

        # Extract each module's default dict by executing just enough of the
        # source: cheap and robust against re-formatting, no live model run.
        def _defaults(func_src: str) -> dict:
            start = func_src.index("default_thresholds = {")
            end = func_src.index("}", start)
            block = func_src[start : end + 1].replace("default_thresholds =", "")
            return eval(block)  # noqa: S307  # literal dict from our own source

        integ = _defaults(inspect.getsource(integrations.assert_fairness))
        rep = _defaults(inspect.getsource(report.classification_fairness_report))
        shared = set(integ) & set(rep)
        assert shared, "expected overlapping default threshold keys"
        for key in shared:
            assert integ[key] == rep[key], (
                f"assert_fairness default for {key} ({integ[key]}) disagrees "
                f"with classification_fairness_report ({rep[key]})"
            )


class TestFPRBaselineWeightedByNegatives:
    """intersectional: overall FPR baseline weighted by actual negatives."""

    def test_overall_fpr_uses_negative_counts(self):
        # A: 90 pos / 10 neg, every negative flagged  -> FPR 1.0
        # B: 10 pos / 90 neg, no negative flagged     -> FPR 0.0
        # Negative-weighted overall FPR = (10*1 + 90*0) / 100 = 0.10
        # Size-weighted (the old bug) would report 0.50.
        yt = np.array([1] * 90 + [0] * 10 + [1] * 10 + [0] * 90)
        yp = np.array([1] * 90 + [1] * 10 + [1] * 10 + [0] * 90)
        g = np.array(["A"] * 100 + ["B"] * 100)
        inter = identify_privileged_groups(yt, yp, g)
        findings = generate_structured_findings(inter)
        fpr_findings = [f for f in findings if f["type"] == "fpr_disparity"]
        assert fpr_findings, "expected FPR disparity findings"
        for f in fpr_findings:
            assert f["metric_values"]["overall_fpr"] == pytest.approx(0.10)
