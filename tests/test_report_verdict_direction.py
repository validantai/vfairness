"""The report and analyzer verdict layers must grade each metric in ITS OWN direction.

Two defects motivated these tests, both of them false-FAIR:

1. ``report.py`` had TWO threshold verdict sites that disagreed. The classification
   loop special-cased exactly ONE name, ``demographic_parity_ratio``, for the
   higher-is-better branch; the regression loop had no ratio branch at all. Every
   other higher-is-better metric was therefore graded as if LOWER were better
   inside the report itself, so a ``disparate_impact_ratio`` of 0.00 (the protected
   group is never selected) was recorded as PASS and 1.00 (perfect parity) as FAIL.

2. ``analyzer._fairness_verdict`` assumed the fair band was ``[0, threshold]`` for
   every metric, in the point-estimate fallback AND in the confidence-interval
   reasoning (``hi <= threshold`` -> fair, ``lo > threshold`` -> unfair). For the
   ratio family that inverts the entire three-state verdict.

Every assertion here is a NEGATIVE case as well as a positive one: it is not enough
that a good value reads fair, the bad value must be REFUSED.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import _fairness_verdict
from vfairness.evaluation.vfairness_metrics.report import (
    _threshold_entry,
    classification_fairness_report,
    print_report,
    regression_fairness_report,
)


class TestReportThresholdDirection:
    """The shared report-layer verdict, used by BOTH report surfaces."""

    def test_disparate_impact_ratio_zero_is_not_a_pass(self):
        """0.00 means the protected group is NEVER selected. It cannot be a PASS."""
        entry = _threshold_entry("disparate_impact_ratio", 0.00, 0.80)
        assert entry["status"] == "FAIL", entry

    def test_disparate_impact_ratio_one_is_a_pass(self):
        """1.00 is exact parity. It must not be graded as a violation."""
        entry = _threshold_entry("disparate_impact_ratio", 1.00, 0.80)
        assert entry["status"] == "PASS", entry

    def test_disparate_impact_ratio_on_the_bound_passes(self):
        """The four-fifths rule is satisfied AT 0.80, not only above it."""
        assert _threshold_entry("disparate_impact_ratio", 0.80, 0.80)["status"] == "PASS"
        assert _threshold_entry("disparate_impact_ratio", 0.7999, 0.80)["status"] == "FAIL"

    @pytest.mark.parametrize(
        "name",
        [
            "demographic_parity_ratio",
            "equal_opportunity_ratio",
            "disparate_impact",
            "worst_group_accuracy",
        ],
    )
    def test_whole_higher_is_better_family_not_just_one_name(self, name):
        """The fix must not be a second special case for one name."""
        assert _threshold_entry(name, 0.10, 0.80)["status"] == "FAIL", name
        assert _threshold_entry(name, 0.95, 0.80)["status"] == "PASS", name

    def test_mirrored_difference_metric_control(self):
        """The lower-is-better family keeps its own direction: 0.00 passes, 1.00 fails."""
        assert _threshold_entry("demographic_parity_difference", 0.00, 0.10)["status"] == "PASS"
        assert _threshold_entry("demographic_parity_difference", 1.00, 0.10)["status"] == "FAIL"

    def test_calibration_is_not_a_ratio(self):
        """'ratio' is a substring of 'cali[bratio]n_difference'. A large gap must FAIL."""
        assert _threshold_entry("calibration_difference", 0.90, 0.05)["status"] == "FAIL"
        assert _threshold_entry("multicalibration", 0.90, 0.03)["status"] == "FAIL"

    def test_unknown_direction_fails_closed(self):
        """No known better-direction means could-not-check, never a pass."""
        entry = _threshold_entry("some_bespoke_client_metric", 0.01, 0.10)
        assert entry["status"] == "NOT_ASSESSABLE", entry
        assert entry["status"] != "PASS"
        assert "could not check" in entry["reason"].lower()
        # A big value is not silently a FAIL either: it is the same third state.
        assert _threshold_entry("some_bespoke_client_metric", 99.0, 0.10)["status"] == (
            "NOT_ASSESSABLE"
        )

    def test_nan_value_is_not_a_fairness_violation(self):
        """An unmeasured metric is could-not-check, not a FAIL and not a PASS."""
        nan = float("nan")
        assert _threshold_entry("demographic_parity_difference", nan, 0.10)["status"] == (
            "NOT_ASSESSABLE"
        )
        assert _threshold_entry("disparate_impact_ratio", nan, 0.80)["status"] == "NOT_ASSESSABLE"
        assert _threshold_entry("disparate_impact_ratio", 0.90, nan)["status"] == "NOT_ASSESSABLE"


class TestReportEndToEndDirection:
    """The direction rule reaching a real generated report."""

    @staticmethod
    def _one_group_never_selected():
        """Group B is never selected: DP ratio 0.0, DP difference 1.0."""
        rng = np.random.RandomState(0)
        n = 200
        groups = np.array(["A"] * n + ["B"] * n)
        y_pred = np.array([1] * n + [0] * n)
        y_true = rng.randint(0, 2, size=2 * n)
        return y_true, y_pred, groups

    def test_zero_ratio_is_reported_as_failed_not_passed(self):
        y_true, y_pred, groups = self._one_group_never_selected()
        report = classification_fairness_report(y_true, y_pred, groups)
        assessment = report["assessment"]
        assert report["metrics"]["demographic_parity_ratio"] == pytest.approx(0.0)
        failed = {m["metric"] for m in assessment["failed_metrics"]}
        passed = {m["metric"] for m in assessment["passed_metrics"]}
        assert "demographic_parity_ratio" in failed
        assert "demographic_parity_ratio" not in passed

    def test_parity_ratio_of_one_is_reported_as_passed(self):
        rng = np.random.RandomState(1)
        n = 200
        groups = np.array(["A"] * n + ["B"] * n)
        # Identical selection pattern in both groups: ratio 1.0, difference 0.0.
        pattern = rng.randint(0, 2, size=n)
        y_pred = np.concatenate([pattern, pattern])
        y_true = np.concatenate([pattern, pattern])
        report = classification_fairness_report(y_true, y_pred, groups)
        assessment = report["assessment"]
        assert report["metrics"]["demographic_parity_ratio"] == pytest.approx(1.0)
        passed = {m["metric"] for m in assessment["passed_metrics"]}
        assert "demographic_parity_ratio" in passed

    def test_regression_report_grades_a_ratio_metric_the_right_way(self):
        """The regression site had NO ratio branch, so it inherited the bug.

        The regression report's own metric set is all differences, so the site is
        exercised through the shared entry point it now uses.
        """
        assert _threshold_entry("mae_parity_ratio", 0.10, 0.80)["status"] == "FAIL"
        assert _threshold_entry("mae_parity_ratio", 0.95, 0.80)["status"] == "PASS"
        # And the regression report itself still grades its difference metrics
        # as lower-is-better.
        rng = np.random.RandomState(2)
        n = 200
        groups = np.array(["A"] * n + ["B"] * n)
        y_true = rng.normal(0, 1, size=2 * n)
        y_pred = y_true.copy()
        y_pred[n:] += 5.0  # group B is badly predicted
        report = regression_fairness_report(y_true, y_pred, groups)
        failed = {m["metric"] for m in report["assessment"]["failed_metrics"]}
        assert "mean_prediction_difference" in failed


class TestFairnessVerdictDirection:
    """The analyzer's three-state confidence-interval verdict."""

    def test_ratio_ci_entirely_below_threshold_is_unfair(self):
        """A ratio whose whole interval sits below 0.80 is a refusal, not a pass."""
        assert (
            _fairness_verdict(0.05, 0.01, 0.10, 0.80, metric_name="disparate_impact_ratio")
            == "unfair"
        )

    def test_ratio_ci_entirely_above_threshold_is_fair(self):
        assert (
            _fairness_verdict(0.95, 0.90, 0.99, 0.80, metric_name="disparate_impact_ratio")
            == "fair"
        )

    def test_ratio_ci_straddling_threshold_is_insufficient_evidence(self):
        assert (
            _fairness_verdict(0.82, 0.70, 0.95, 0.80, metric_name="disparate_impact_ratio")
            == "insufficient_evidence"
        )

    def test_ratio_point_estimate_fallback_is_direction_correct(self):
        """No usable interval: the fallback must also read the ratio the right way."""
        nan = float("nan")
        assert (
            _fairness_verdict(0.05, nan, nan, 0.80, metric_name="demographic_parity_ratio")
            == "unfair"
        )
        assert (
            _fairness_verdict(0.95, nan, nan, 0.80, metric_name="demographic_parity_ratio")
            == "fair"
        )

    def test_difference_metric_control_is_unchanged(self):
        """The lower-is-better family keeps the band [0, threshold]."""
        assert (
            _fairness_verdict(0.02, 0.01, 0.04, 0.10, metric_name="demographic_parity_difference")
            == "fair"
        )
        assert (
            _fairness_verdict(0.30, 0.25, 0.35, 0.10, metric_name="demographic_parity_difference")
            == "unfair"
        )
        assert (
            _fairness_verdict(0.09, 0.02, 0.20, 0.10, metric_name="demographic_parity_difference")
            == "insufficient_evidence"
        )

    def test_unknown_direction_verdict_fails_closed(self):
        """A named metric with no known direction can never read 'fair'."""
        assert (
            _fairness_verdict(0.01, 0.00, 0.02, 0.10, metric_name="some_bespoke_client_metric")
            == "insufficient_evidence"
        )

    def test_the_legacy_four_argument_form_is_now_refused(self):
        """The unnamed form assumed lower-is-better, i.e. it GUESSED a direction.

        This test used to assert the opposite: that a four-argument call kept
        reading as a non-negative difference, so existing callers were "not
        silently regraded". That compatibility shim was itself the third
        appearance of the guessed-direction defect, latent only because every
        in-repo caller happened to name its metric. ``metric_name`` is required
        as of 2026-08-27; the full pin lives in tests/test_verdict_defaults.py.
        """
        with pytest.raises(TypeError):
            _fairness_verdict(0.02, 0.01, 0.04, 0.10)
        name = "demographic_parity_difference"
        assert _fairness_verdict(0.02, 0.01, 0.04, 0.10, metric_name=name) == "fair"
        assert _fairness_verdict(0.30, 0.25, 0.35, 0.10, metric_name=name) == "unfair"
        assert (
            _fairness_verdict(0.09, 0.02, 0.20, 0.10, metric_name=name) == "insufficient_evidence"
        )


class TestNoneFairnessScoreIsToleratedByReportConsumers:
    """fairness_score is Optional[float] and is None on a not-assessable run."""

    def test_print_report_does_not_crash_on_none_score(self, capsys):
        print_report(
            {
                "task_type": "classification",
                "metrics": {"demographic_parity_difference": float("nan")},
                "assessment": {
                    "fairness_score": None,
                    "assessable": False,
                    "passed_metrics": [],
                    "failed_metrics": [],
                    "not_assessable_metrics": [],
                    "summary": "NOT ASSESSABLE",
                },
                "data_info": {},
            }
        )
        out = capsys.readouterr().out
        assert "could not check" in out.lower()
        # It must NOT print a number for a score that does not exist.
        assert "Fairness Score: 0" not in out
        assert "Fairness Score: 100" not in out

    def test_explainer_summary_does_not_crash_on_none_score(self):
        from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

        explainer = FairExplAIner(task_type="classification")
        summary = explainer._generate_summary(
            {
                "assessment": {
                    "fairness_score": None,
                    "assessable": False,
                    "passed_metrics": [],
                    "failed_metrics": [],
                    "not_assessable_metrics": [],
                }
            },
            {"metrics": {}},
        )
        assert "NOT AVAILABLE" in summary
        assert "COULD NOT CHECK" in summary

    def test_not_assessable_run_reports_none_score_without_crashing(self):
        """The real end-to-end path that produces None."""
        y_true = np.array([0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 1] * 10)
        groups = np.array(["only_one_group"] * 40)
        report = classification_fairness_report(y_true, y_pred, groups)
        assert report["assessment"]["fairness_score"] is None
        assert report["assessment"]["assessable"] is False
        print_report(report)  # must not raise


class TestExplainerDirection:
    """The explainer must not call a backwards reading 'Excellent!'."""

    def test_ratio_of_zero_is_critical_not_excellent(self):
        from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

        explainer = FairExplAIner(task_type="classification")
        explanation = explainer.explain_metric("demographic_parity_ratio", 0.0)
        assert explanation.severity == "critical", explanation.evaluation
        assert "Excellent" not in explanation.evaluation

    def test_unknown_direction_fairness_metric_is_not_graded(self):
        from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

        explainer = FairExplAIner(task_type="classification")
        # Inject a fairness metric definition whose name carries no direction signal.
        explainer.metrics_definitions = dict(explainer.metrics_definitions)
        explainer.metrics_definitions["bespoke_client_score"] = {
            "definition": "A bespoke client metric.",
            "thresholds": {"excellent": 0.05, "acceptable": 0.10},
        }
        explanation = explainer.explain_metric("bespoke_client_score", 0.01)
        assert "could not check" in explanation.evaluation.lower()
        assert "Excellent" not in explanation.evaluation
