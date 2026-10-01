"""Three producers that computed their own could-not-check state and then
routed around it.

Readiness wave 2, lane ``threestate``. Each finding here is the same defect:
a value nobody measured is substituted with a neutral default and then graded,
counted or reported as if it were a measurement. In all three the honest
machinery was ALREADY PRESENT, one branch or one function away, and the common
path did not reach it.

1. ``FairnessExperiment.detect_heterogeneous_effects`` assigned ``het_p = 1.0``
   on the two branches that mean the heterogeneity test COULD NOT RUN. 1.0 is
   the most non-significant p there is, it is finite, and the three-state gate
   below it tests ``np.isfinite``, so ``heterogeneity_detected`` came back False
   on every surface: repr, ``to_dict``, ``get_summary``, the report section and
   the rendered chart, with a fabricated p-value and no warning.

2. ``ThresholdAnalyzer`` consumed ``ConstraintViolation.is_satisfied``, which is
   ``Optional[bool]`` precisely so a constraint that cannot be evaluated is
   neither satisfied nor violated, through bare truthiness tests. A sweep in
   which 100 of 100 thresholds were unmeasurable produced ``is_feasible=False``
   and the report recommended group-specific thresholds on the strength of zero
   measurements.

3. ``analyze_calibration_fairness_tradeoff`` returned fabricated neutral
   measurements when fewer than two groups were valid, while its own sibling
   ``impossibility_diagnostics`` computed a full NOT ASSESSED verdict for the
   identical condition and was never called on that path.

Every finding is pinned twice: a REFUSAL PIN that the could-not-check state
reaches the caller, and an OVER-CORRECTION CONTROL asserting the measured
numbers a real finding still produces. The controls exist because turning every
verdict into None would satisfy the refusal pins and destroy the library.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.operations.experimentation.experiment as experiment_module
from vfairness.operations.experimentation.experiment import (
    ExperimentConfig,
    FairnessExperiment,
)
from vfairness.post_processing.calibration.tradeoffs import (
    analyze_calibration_fairness_tradeoff,
    impossibility_diagnostics,
)
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalyzer,
)
from vfairness.post_processing.threshold_optimization.constraints import (
    compute_constraint_violation,
)


def _arms(pairs):
    """Build one experiment arm from ``[(group_name, [outcomes]), ...]``."""
    groups, outcomes = [], []
    for name, values in pairs:
        groups += [name] * len(values)
        outcomes += list(values)
    return pd.DataFrame({"g": groups, "y": outcomes})


# A spread of outcomes with a real, non-zero variance in every cell, so the
# bootstrap standard errors are positive and the heterogeneity test can run.
SPREAD = [0.0, 1.0, 2.0, 3.0] * 15


# Finding 1. The heterogeneity test that could not run.


class TestHeterogeneityThatCouldNotRun:
    def test_one_analysable_intersection_is_not_reported_as_homogeneous(self):
        """A single intersection means there is no between-group comparison to
        make. The verdict must be None, not the False that ``het_p = 1.0``
        produced."""
        control = _arms([("A", SPREAD)])
        treatment = _arms([("A", [v + 0.5 for v in SPREAD])])
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1234, n_bootstrap=200),
        )

        with pytest.warns(UserWarning, match="heterogeneity test could not be computed"):
            result = experiment.detect_heterogeneous_effects()

        assert result.n_intersections == 1
        assert result.heterogeneity_detected is None
        assert math.isnan(result.heterogeneity_p_value)
        assert "not assessed" in repr(result)
        assert result.to_dict()["heterogeneity_detected"] is None
        assert math.isnan(result.to_dict()["heterogeneity_p_value"])
        assert experiment.get_summary()["heterogeneity_detected"] is None

    def test_intersections_with_no_bootstrap_spread_cannot_run_the_test_either(self):
        """The worse half of the finding: two intersections whose effects are
        1.0 and 4.0, a fourfold difference, reported ``heterogeneity=no`` with
        ``p=1.0000`` because Cochran's Q needs a positive standard error and
        neither cell had one."""
        control = _arms([("A", [1.0] * 40), ("B", [1.0] * 40)])
        treatment = _arms([("A", [2.0] * 40), ("B", [5.0] * 40)])
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1234, n_bootstrap=100),
        )

        with pytest.warns(UserWarning, match="heterogeneity test could not be computed"):
            result = experiment.detect_heterogeneous_effects()

        # The per-intersection effects WERE measured; only the test across them
        # could not run. Both facts have to survive.
        assert [e.effect for e in result.intersection_effects] == [1.0, 4.0]
        assert result.n_intersections == 2
        assert result.heterogeneity_detected is None
        assert math.isnan(result.heterogeneity_p_value)

    def test_control_a_real_heterogeneity_still_reports_its_measured_p_value(self):
        """Over-correction control. Effects of 0.5 and 3.0 across two well
        sampled intersections are genuine heterogeneity, and the numbers must
        come back unchanged."""
        control = _arms([("A", SPREAD), ("B", SPREAD)])
        treatment = _arms([("A", [v + 0.5 for v in SPREAD]), ("B", [v + 3.0 for v in SPREAD])])
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1234, n_bootstrap=300),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = experiment.detect_heterogeneous_effects()

        assert [e.effect for e in result.intersection_effects] == [0.5, 3.0]
        assert result.overall_effect == 1.75
        assert result.heterogeneity_detected is True
        assert result.heterogeneity_p_value == pytest.approx(6.10823128030915e-19, rel=1e-3)
        assert "not assessed" not in repr(result)

    def test_control_a_measured_homogeneity_is_still_reported_as_false(self):
        """The other half of the control. A test that RAN and found no spread
        must keep saying False, which is a real finding about the treatment,
        rather than being swept into the new None state."""
        control = _arms([("A", SPREAD), ("B", SPREAD)])
        treatment = _arms([("A", [v + 0.5 for v in SPREAD]), ("B", [v + 0.5 for v in SPREAD])])
        experiment = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1234, n_bootstrap=300),
        )

        result = experiment.detect_heterogeneous_effects()

        assert [e.effect for e in result.intersection_effects] == [0.5, 0.5]
        assert result.heterogeneity_detected is False
        assert math.isfinite(result.heterogeneity_p_value)
        assert result.heterogeneity_p_value == pytest.approx(1.0, abs=1e-9)


class TestOverallPValueWithoutScipy:
    """Sibling of finding 1, found in the same function.

    The scipy-free fallback for the OVERALL t-test answered ``1.0`` when both
    arms are constant, which is the condition under which no test can be run at
    all. The scipy branch answers 0.0 on the identical arrays, so the two
    branches of one computation disagreed completely.
    """

    def test_two_constant_arms_have_no_overall_p_value(self, monkeypatch):
        monkeypatch.setattr(experiment_module, "_HAS_SCIPY", False)
        control = _arms([("A", [1.0] * 40), ("B", [1.0] * 40)])
        treatment = _arms([("A", [2.0] * 40), ("B", [2.0] * 40)])

        result = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1, n_bootstrap=50),
        ).detect_heterogeneous_effects()

        assert result.overall_effect == 1.0
        assert math.isnan(result.overall_p_value)

    def test_control_measurable_arms_keep_their_measured_p_value(self, monkeypatch):
        monkeypatch.setattr(experiment_module, "_HAS_SCIPY", False)
        control = _arms([("A", SPREAD), ("B", SPREAD)])
        treatment = _arms([("A", [v + 0.2 for v in SPREAD]), ("B", [v + 0.4 for v in SPREAD])])

        result = FairnessExperiment(
            control,
            treatment,
            ["g"],
            "y",
            config=ExperimentConfig(random_state=1234, n_bootstrap=50),
        ).detect_heterogeneous_effects()

        assert result.overall_effect == pytest.approx(0.3, abs=1e-12)
        assert result.overall_p_value == pytest.approx(0.03886089702130502, rel=1e-9)


# Finding 2. The feasibility sweep in which nothing could be checked.


# Group B carries no positive label, so equal opportunity (a TPR comparison) has
# an empty denominator for B at EVERY threshold: 100 percent could-not-check.
NOT_ASSESSED_Y = np.array([1] * 10 + [0] * 10 + [0] * 20)
NOT_ASSESSED_P = np.array([0.9] * 10 + [0.2] * 10 + [0.7] * 10 + [0.3] * 10)
NOT_ASSESSED_G = np.array(["A"] * 20 + ["B"] * 20)

# Every subject is a true positive, group A always scores 1.0 and group B always
# scores 0.0, so equal opportunity is MEASURED at every threshold and satisfied
# at none. This is a real finding and must keep reporting as one.
INFEASIBLE_Y = np.array([1] * 20)
INFEASIBLE_P = np.array([1.0] * 10 + [0.0] * 10)
INFEASIBLE_G = np.array(["A"] * 10 + ["B"] * 10)


class TestFeasibilityThatCouldNotBeChecked:
    def test_the_producer_itself_says_could_not_check(self):
        """The premise. Everything below is about a consumer discarding this."""
        violation = compute_constraint_violation(
            NOT_ASSESSED_Y,
            (NOT_ASSESSED_P >= 0.5).astype(int),
            NOT_ASSESSED_G,
            constraint="equal_opportunity",
            tolerance=0.05,
        )
        assert violation.is_satisfied is None
        assert math.isnan(violation.violation)

    def test_a_sweep_of_unmeasurable_thresholds_is_not_reported_as_infeasible(self):
        analyzer = ThresholdAnalyzer(NOT_ASSESSED_Y, NOT_ASSESSED_P, NOT_ASSESSED_G)

        with pytest.warns(UserWarning, match="none of the 25 searched"):
            optimum = analyzer.find_optimal_threshold(
                constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
            )

        assert optimum["is_feasible"] is None
        assert optimum["optimal_threshold"] is None
        assert optimum["n_thresholds_searched"] == 25
        assert optimum["n_thresholds_not_assessed"] == 25

    def test_the_empty_region_carries_the_state_the_pair_cannot_express(self):
        """The return type had to be widened: ``(None, None)`` alone has two
        states and the question has three."""
        analyzer = ThresholdAnalyzer(NOT_ASSESSED_Y, NOT_ASSESSED_P, NOT_ASSESSED_G)

        with pytest.warns(UserWarning, match="none of the 25 searched"):
            region = analyzer.find_feasible_region(
                constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
            )

        assert region.assessed is False
        assert region.status == "not_assessed"
        assert region.n_not_assessed == 25
        assert region.n_searched == 25
        # Still a two element pair, so every existing endpoint reader keeps
        # working; the third state rides alongside rather than replacing it.
        assert tuple(region) == (None, None)
        low, high = region
        assert low is None and high is None

    def test_the_report_does_not_recommend_group_thresholds_from_no_measurement(self):
        analyzer = ThresholdAnalyzer(NOT_ASSESSED_Y, NOT_ASSESSED_P, NOT_ASSESSED_G)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = analyzer.full_analysis(n_thresholds=4, constraints=["equal_opportunity"])

        joined = " ".join(report.recommendations)
        assert "No single threshold can satisfy" not in joined
        assert "NOT ASSESSED" in joined
        assert "equal_opportunity" in joined
        assert report.optimal_thresholds["equal_opportunity"]["accuracy"] is None
        summary = report.summary()
        assert "equal_opportunity: NOT ASSESSED" in summary
        assert "No feasible region found" not in summary

    def test_control_a_measured_infeasibility_reports_exactly_as_before(self):
        """Over-correction control. Equal opportunity is measurable at every one
        of the 25 thresholds here and satisfied at none of them, so the verdict
        is a measured False with a violation of exactly 1.0.

        READINESS-6, 2026-09-10. This used `simplefilter("error")` and asserted
        that a fully measured case warns about NOTHING. That claim was slightly
        too broad for this fixture, and the fix that exposed it was right.
        ``INFEASIBLE_Y`` is all-positive ON PURPOSE, because equal opportunity is
        a statement about the true-positive rate and needs positive labels. With
        no negative label anywhere, ``balanced_accuracy`` has no value on this
        data, and ``_compute_performance_metrics`` now says so instead of
        returning a fabricated 0.0.

        So the control keeps its strictness and states what it is strict ABOUT:
        the feasibility path must stay silent, and the ONLY warning permitted is
        the one naming that single genuinely-undefined metric. Anything else
        fails, including a return of the old silence, which would mean the
        fabricated zero is back.
        """
        analyzer = ThresholdAnalyzer(INFEASIBLE_Y, INFEASIBLE_P, INFEASIBLE_G)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            optimum = analyzer.find_optimal_threshold(
                constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
            )
            region = analyzer.find_feasible_region(
                constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
            )

        messages = [str(w.message) for w in caught]
        unexpected = [
            m for m in messages if "balanced_accuracy (the data holds no negative label)" not in m
        ]
        assert not unexpected, f"the feasibility path warned about something new: {unexpected}"
        assert messages, (
            "no warning at all. balanced_accuracy is undefined on this all-positive "
            "fixture, so silence here means the fabricated 0.0 is back."
        )

        assert optimum["is_feasible"] is False
        assert optimum["optimal_threshold"] == pytest.approx(0.01)
        assert optimum["violation"].is_satisfied is False
        assert optimum["violation"].violation == pytest.approx(1.0)
        assert optimum["n_thresholds_not_assessed"] == 0
        assert region == (None, None)
        assert region.assessed is True
        assert region.status == "infeasible"

    def test_control_the_measured_infeasibility_keeps_its_recommendation(self):
        analyzer = ThresholdAnalyzer(INFEASIBLE_Y, INFEASIBLE_P, INFEASIBLE_G)

        report = analyzer.full_analysis(n_thresholds=4, constraints=["equal_opportunity"])

        joined = " ".join(report.recommendations)
        assert "No single threshold can satisfy: equal_opportunity" in joined
        assert "NOT ASSESSED" not in joined
        assert "equal_opportunity: No feasible region found" in report.summary()

    def test_control_a_measured_feasibility_still_reports_its_endpoints(self):
        y_true = np.array([1, 0] * 20)
        y_prob = np.array([0.6, 0.4] * 20)
        groups = np.array(["A"] * 20 + ["B"] * 20)
        analyzer = ThresholdAnalyzer(y_true, y_prob, groups)

        optimum = analyzer.find_optimal_threshold(
            constraint="demographic_parity", tolerance=0.05, n_thresholds=25
        )
        region = analyzer.find_feasible_region(
            constraint="demographic_parity", tolerance=0.05, n_thresholds=25
        )

        assert optimum["is_feasible"] is True
        assert optimum["optimal_threshold"] == pytest.approx(0.41833333333333333)
        assert optimum["violation"].violation == pytest.approx(0.0)
        assert region.status == "feasible"
        assert region[0] == pytest.approx(0.01)
        assert region[1] == pytest.approx(0.99)
        assert region.n_not_assessed == 0
        assert isinstance(region, FeasibleRegion)


# Finding 3. The trade-off block that answered a question it never asked.


class TestTradeoffThatWasNeverMeasured:
    def test_one_group_produces_no_neutral_measurements(self):
        y_true = np.array([1, 0] * 30)
        y_prob = np.array([0.8, 0.3] * 30)
        groups = np.array(["A"] * 60)

        with pytest.warns(UserWarning, match="fewer than 2 groups"):
            result = analyze_calibration_fairness_tradeoff(y_true, y_prob, groups)

        # The three all-clears that used to sit in one reader-facing block.
        assert math.isnan(result.base_rate_disparity)
        assert result.tradeoff_severity == "not assessed"
        assert result.impossibility_diagnosis["impossibility_applies"] is None
        assert math.isnan(result.current_point.calibration_error)
        assert math.isnan(result.current_point.fairness_violation)
        assert math.isnan(result.to_dict()["base_rate_disparity"])
        assert result.to_dict()["tradeoff_severity"] == "not assessed"

    def test_it_answers_with_its_own_sibling_rather_than_a_second_opinion(self):
        """The fix is a CALL, not a reimplementation. Two functions answering
        the same question differently is how this started, so the two answers
        are pinned equal on identical input."""
        y_true = np.array([1, 0] * 30)
        y_prob = np.array([0.8, 0.3] * 30)
        groups = np.array(["A"] * 60)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = analyze_calibration_fairness_tradeoff(y_true, y_prob, groups)
            sibling = impossibility_diagnostics(y_true, y_prob, groups)

        diagnosis = result.impossibility_diagnosis
        assert diagnosis["base_rates_differ"] is sibling["base_rates_differ"] is None
        assert diagnosis["impossibility_applies"] is sibling["impossibility_applies"] is None
        assert diagnosis["n_groups_compared"] == sibling["n_groups_compared"] == 1
        assert diagnosis["explanation"] == sibling["explanation"]
        assert diagnosis["explanation"].startswith("NOT ASSESSED.")
        assert math.isnan(diagnosis["base_rate_disparity"])
        assert any("NOT ASSESSED" in r for r in result.recommendations)

    def test_control_a_real_tradeoff_reports_its_measured_numbers(self):
        """Over-correction control. Base rates of 0.8 and 0.1 across two groups
        of 60 are a genuine, severe trade-off and the impossibility theorem
        genuinely binds."""
        y_true = np.array([1] * 48 + [0] * 12 + [1] * 6 + [0] * 54)
        y_prob = np.array([0.8] * 60 + [0.2] * 60)
        groups = np.array(["A"] * 60 + ["B"] * 60)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = analyze_calibration_fairness_tradeoff(
                y_true, y_prob, groups, fairness_metric="demographic_parity"
            )

        assert result.base_rate_disparity == pytest.approx(0.7)
        assert result.tradeoff_severity == "severe"
        assert result.impossibility_diagnosis["impossibility_applies"] is True
        assert result.impossibility_diagnosis["base_rates"]["A"] == pytest.approx(0.8)
        assert result.impossibility_diagnosis["base_rates"]["B"] == pytest.approx(0.1)
        assert result.current_point.fairness_violation == pytest.approx(1.0)
        assert result.current_point.calibration_error == pytest.approx(0.05, abs=1e-9)
        assert any("SIGNIFICANT TRADE-OFF" in r for r in result.recommendations)

    def test_control_a_measured_minimal_tradeoff_is_still_called_minimal(self):
        """The other half. A disparity that WAS measured and came out near zero
        keeps its band, rather than being swallowed by the new third state."""
        y_true = np.array([1] * 30 + [0] * 30 + [1] * 30 + [0] * 30)
        y_prob = np.array([0.7, 0.3] * 60)
        groups = np.array(["A"] * 60 + ["B"] * 60)

        result = analyze_calibration_fairness_tradeoff(y_true, y_prob, groups)

        assert result.base_rate_disparity == pytest.approx(0.0)
        assert result.tradeoff_severity == "minimal"
        assert result.impossibility_diagnosis["base_rates_differ"] is False
        assert result.impossibility_diagnosis["impossibility_applies"] is False


class TestFairnessViolationWithOneMeasurableRate:
    """Sibling of finding 3, found in the same function.

    ``fairness_violation`` fell back to 0.0, which is PERFECT PARITY, whenever
    fewer than two groups had a measurable rate at a threshold. The same file's
    ``calibration_vs_error_parity`` already answers NaN for that condition.
    """

    def test_a_disparity_with_one_measurable_rate_is_not_perfect_parity(self):
        # Group B carries no negative label, so it has no false positive rate.
        y_true = np.array([1, 0] * 30 + [1] * 60)
        y_prob = np.array([0.8, 0.3] * 30 + [0.8, 0.3] * 30)
        groups = np.array(["A"] * 60 + ["B"] * 60)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = analyze_calibration_fairness_tradeoff(
                y_true, y_prob, groups, fairness_metric="fpr_parity"
            )

        measured = result.metrics_at_thresholds[0.5]["group_metrics"]
        assert measured["A"] == pytest.approx(0.0)
        assert math.isnan(measured["B"])
        assert math.isnan(result.current_point.fairness_violation)
        assert any("NOT ASSESSED: fpr_parity" in r for r in result.recommendations)

    def test_control_two_measurable_rates_still_report_the_measured_gap(self):
        y_true = np.array([1, 0] * 30 + [1, 0] * 30)
        y_prob = np.array([0.8] * 60 + [0.3] * 60)
        groups = np.array(["A"] * 60 + ["B"] * 60)

        result = analyze_calibration_fairness_tradeoff(
            y_true, y_prob, groups, fairness_metric="fpr_parity"
        )

        measured = result.metrics_at_thresholds[0.5]["group_metrics"]
        assert measured["A"] == pytest.approx(1.0)
        assert measured["B"] == pytest.approx(0.0)
        assert result.current_point.fairness_violation == pytest.approx(1.0)
        assert not any("NOT ASSESSED" in r for r in result.recommendations)
        assert any("FAIRNESS CONCERN" in r for r in result.recommendations)
