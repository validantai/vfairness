"""Behavioural tests for :class:`ThresholdAnalyzer`.

Closes the coverage half of register finding #22 for
``post_processing/threshold_optimization/analyzer.py``, which sat at 14.5
percent: every function body below ``__init__`` was unexecuted even though
``ThresholdAnalyzer`` is re-exported from ``vfairness.__all__`` and its output
is what a practitioner uses to pick an operating point.

The class decides a gate ("is there a threshold that satisfies this fairness
constraint"), so the negative case matters more than the happy path: when no
threshold can satisfy the constraint the analyzer must SAY so, not hand back a
number that quietly violates it.
"""

import math

import numpy as np
import pytest

from vfairness.exceptions import InvalidDataError
from vfairness.post_processing.threshold_optimization.analyzer import (
    PERFORMANCE_OBJECTIVES,
    ThresholdAnalysisReport,
    ThresholdAnalyzer,
)
from vfairness.post_processing.threshold_optimization.constraints import (
    compute_constraint_violation,
)


def _biased_scores(n=400, seed=7):
    """Scores that are systematically higher for group A than for group B."""
    rng = np.random.default_rng(seed)
    groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    probs = np.concatenate([rng.uniform(0.5, 1.0, n // 2), rng.uniform(0.0, 0.5, n // 2)])
    labels = (rng.uniform(size=n) < probs).astype(int)
    return labels, probs, groups


# A dataset where equal opportunity is unsatisfiable at every searched
# threshold: every subject is a true positive, group A always scores 1.0 and
# group B always scores 0.0, so any cut in (0, 1] separates the groups
# completely.
IMPOSSIBLE_Y = np.array([1] * 20)
IMPOSSIBLE_P = np.array([1.0] * 10 + [0.0] * 10)
IMPOSSIBLE_G = np.array(["A"] * 10 + ["B"] * 10)


class TestSingleThresholdAnalysis:
    def test_confusion_matrix_partitions_the_whole_sample(self):
        labels, probs, groups = _biased_scores()
        result = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.5)

        assert sum(result.confusion_matrix.values()) == len(labels)
        assert result.threshold == 0.5

    def test_group_confusion_matrices_add_up_to_the_overall_one(self):
        labels, probs, groups = _biased_scores()
        result = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.5)

        for cell in ("tp", "tn", "fp", "fn"):
            per_group = sum(m[cell] for m in result.group_confusion_matrices.values())
            assert per_group == result.confusion_matrix[cell]

    def test_a_score_exactly_on_the_threshold_counts_as_positive(self):
        """The rule is ``y_prob >= threshold``; the boundary case decides real
        borderline applicants, so it is pinned by value."""
        analyzer = ThresholdAnalyzer(
            np.array([1, 0, 1, 0]),
            np.array([0.5, 0.5, 0.2, 0.9]),
            np.array(["A", "A", "B", "B"]),
        )
        result = analyzer.analyze_threshold(0.5)

        assert result.confusion_matrix == {"tp": 1, "tn": 0, "fp": 2, "fn": 1}

    def test_a_threshold_below_every_score_accepts_everyone(self):
        labels, probs, groups = _biased_scores()
        result = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.0)

        assert result.confusion_matrix["fn"] == 0
        assert result.confusion_matrix["tn"] == 0
        assert result.performance_metrics["recall"] == pytest.approx(1.0)
        # Accepting everyone is perfectly parity-fair, and the analyzer must
        # report exactly that rather than an accuracy-driven verdict.
        assert result.constraint_violations["demographic_parity"].violation == pytest.approx(0.0)

    def test_a_threshold_above_every_score_rejects_everyone(self):
        labels, probs, groups = _biased_scores()
        result = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(1.01)

        assert result.confusion_matrix["tp"] == 0
        assert result.confusion_matrix["fp"] == 0
        # READINESS-6, 2026-09-10. This asserted precision == 0.0, which was the
        # fabricated value: with tp + fp == 0 the model made NO positive call, so
        # precision has no value at all. 0.0 says something quite different and
        # much stronger, that every positive call it made was wrong.
        assert math.isnan(result.performance_metrics["precision"])
        # f1_score stays a MEASURED zero, and that distinction is the point.
        # Recall is 0.0 here and measured: positives exist and the model found
        # none of them. That is a complete failure on the positive class and a
        # real finding, so F1 must not be turned into a could-not-check by the
        # undefined precision beside it.
        assert result.performance_metrics["recall"] == 0.0
        assert result.performance_metrics["f1_score"] == 0.0

    def test_sample_weights_change_the_reported_accuracy(self):
        labels = np.array([1, 0, 1, 0])
        probs = np.array([0.5, 0.5, 0.2, 0.9])
        groups = np.array(["A", "A", "B", "B"])
        unweighted = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.5)
        weighted = ThresholdAnalyzer(
            labels, probs, groups, sample_weight=np.array([1.0, 1.0, 1.0, 10.0])
        ).analyze_threshold(0.5)

        assert unweighted.performance_metrics["accuracy"] == pytest.approx(0.25)
        # The heavily weighted row is a false positive, so accuracy must fall.
        assert weighted.performance_metrics["accuracy"] < 0.1

    def test_every_group_gets_its_own_metrics_block(self):
        labels, probs, groups = _biased_scores()
        result = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.5)

        assert set(result.group_metrics) == {"A", "B"}
        assert set(result.group_confusion_matrices) == {"A", "B"}
        assert result.to_dict()["constraint_violations"]["demographic_parity"]["is_satisfied"] in (
            True,
            False,
        )


class TestThresholdRange:
    def test_the_default_grid_is_ordered_and_the_right_size(self):
        labels, probs, groups = _biased_scores(n=100)
        results = ThresholdAnalyzer(labels, probs, groups).analyze_threshold_range(
            n_thresholds=5, constraints=["demographic_parity"]
        )

        assert len(results) == 5
        thresholds = [r.threshold for r in results]
        assert thresholds == sorted(thresholds)
        assert thresholds[0] == pytest.approx(0.05)
        assert thresholds[-1] == pytest.approx(0.95)

    def test_positive_predictions_fall_as_the_threshold_rises(self):
        labels, probs, groups = _biased_scores(n=200)
        results = ThresholdAnalyzer(labels, probs, groups).analyze_threshold_range(
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], constraints=["demographic_parity"]
        )

        accepted = [r.confusion_matrix["tp"] + r.confusion_matrix["fp"] for r in results]
        assert accepted == sorted(accepted, reverse=True)


class TestGateDecisions:
    def test_an_unsatisfiable_constraint_yields_no_feasible_region(self):
        analyzer = ThresholdAnalyzer(IMPOSSIBLE_Y, IMPOSSIBLE_P, IMPOSSIBLE_G)

        assert analyzer.find_feasible_region(
            constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
        ) == (None, None)

    def test_an_unsatisfiable_constraint_is_never_reported_as_feasible(self):
        """Negative case: the optimum still returns a threshold, so the caller
        depends on ``is_feasible`` and on the violation being marked unsatisfied."""
        analyzer = ThresholdAnalyzer(IMPOSSIBLE_Y, IMPOSSIBLE_P, IMPOSSIBLE_G)

        optimum = analyzer.find_optimal_threshold(
            constraint="equal_opportunity", tolerance=0.05, n_thresholds=25
        )

        assert optimum["is_feasible"] is False
        assert optimum["violation"].is_satisfied is False
        assert optimum["violation"].violation == pytest.approx(1.0)

    def test_a_reported_feasible_optimum_really_satisfies_the_constraint(self):
        """Re-derive the verdict from the returned threshold instead of trusting
        the flag that came back with it."""
        labels, probs, groups = _biased_scores()
        analyzer = ThresholdAnalyzer(labels, probs, groups)

        optimum = analyzer.find_optimal_threshold(
            constraint="demographic_parity", tolerance=0.05, n_thresholds=40
        )
        assert optimum["is_feasible"] is True

        recomputed = compute_constraint_violation(
            analyzer.y_true,
            (analyzer.y_prob >= optimum["optimal_threshold"]).astype(int),
            analyzer.sensitive_attr,
            constraint="demographic_parity",
            tolerance=0.05,
        )
        assert recomputed.is_satisfied is True
        assert recomputed.violation == pytest.approx(optimum["violation"].violation)

    def test_the_feasible_optimum_maximises_the_objective_over_the_grid(self):
        labels, probs, groups = _biased_scores()
        analyzer = ThresholdAnalyzer(labels, probs, groups)

        optimum = analyzer.find_optimal_threshold(
            constraint="demographic_parity", objective="accuracy", tolerance=0.2, n_thresholds=30
        )

        best_feasible_accuracy = -1.0
        for threshold in np.linspace(0.01, 0.99, 30):
            predictions = (analyzer.y_prob >= threshold).astype(int)
            violation = compute_constraint_violation(
                analyzer.y_true,
                predictions,
                analyzer.sensitive_attr,
                constraint="demographic_parity",
                tolerance=0.2,
            )
            if violation.is_satisfied:
                accuracy = float(np.mean(analyzer.y_true == predictions))
                best_feasible_accuracy = max(best_feasible_accuracy, accuracy)

        assert optimum["metrics"]["accuracy"] == pytest.approx(best_feasible_accuracy)

    def test_the_feasible_region_endpoints_are_themselves_feasible(self):
        labels, probs, groups = _biased_scores()
        analyzer = ThresholdAnalyzer(labels, probs, groups)

        low, high = analyzer.find_feasible_region(
            constraint="demographic_parity", tolerance=0.05, n_thresholds=40
        )
        assert low is not None and high is not None
        assert low <= high

        for endpoint in (low, high):
            violation = compute_constraint_violation(
                analyzer.y_true,
                (analyzer.y_prob >= endpoint).astype(int),
                analyzer.sensitive_attr,
                constraint="demographic_parity",
                tolerance=0.05,
            )
            assert violation.is_satisfied is True


class TestFullAnalysisReport:
    def test_an_unsatisfiable_constraint_is_named_in_the_recommendations(self):
        analyzer = ThresholdAnalyzer(IMPOSSIBLE_Y, IMPOSSIBLE_P, IMPOSSIBLE_G)

        report = analyzer.full_analysis(n_thresholds=4, constraints=["equal_opportunity"])

        assert report.feasible_regions["equal_opportunity"] == (None, None)
        joined = " ".join(report.recommendations)
        assert "No single threshold can satisfy" in joined
        assert "equal_opportunity" in joined

    def test_the_summary_states_the_missing_region_rather_than_a_number(self):
        analyzer = ThresholdAnalyzer(IMPOSSIBLE_Y, IMPOSSIBLE_P, IMPOSSIBLE_G)

        summary = analyzer.full_analysis(
            n_thresholds=4, constraints=["equal_opportunity"]
        ).summary()

        assert "equal_opportunity: No feasible region found" in summary
        assert "Analyzed 4 threshold values" in summary

    def test_the_report_records_the_data_it_was_built_from(self):
        labels, probs, groups = _biased_scores(n=120)

        report = ThresholdAnalyzer(labels, probs, groups).full_analysis(
            n_thresholds=3, constraints=["demographic_parity"], tolerance=0.07
        )

        assert isinstance(report, ThresholdAnalysisReport)
        assert report.metadata == {
            "n_samples": 120,
            "n_groups": 2,
            "groups": ["A", "B"],
            "tolerance": 0.07,
        }
        assert len(report.threshold_results) == 3
        assert set(report.optimal_thresholds["demographic_parity"]) == {
            "accuracy",
            "f1_score",
            "balanced_accuracy",
        }

    def test_the_report_serialises_without_losing_the_verdicts(self):
        labels, probs, groups = _biased_scores(n=120)
        report = ThresholdAnalyzer(labels, probs, groups).full_analysis(
            n_thresholds=2, constraints=["demographic_parity"]
        )

        as_dict = report.to_dict()

        assert len(as_dict["threshold_results"]) == 2
        first = as_dict["threshold_results"][0]["constraint_violations"]["demographic_parity"]
        assert set(first) >= {"violation", "is_satisfied", "tolerance"}
        assert isinstance(first["is_satisfied"], bool)


class TestRefusals:
    def test_mismatched_input_lengths_are_refused_at_construction(self):
        with pytest.raises(InvalidDataError, match="Inconsistent array lengths"):
            ThresholdAnalyzer(
                np.array([1, 0, 1, 0]), np.array([0.5, 0.5]), np.array(["A", "A", "B", "B"])
            )

    def test_an_unknown_constraint_name_is_refused_rather_than_ignored(self):
        labels, probs, groups = _biased_scores(n=40)

        with pytest.raises(ValueError, match="not a valid FairnessConstraintType"):
            ThresholdAnalyzer(labels, probs, groups).analyze_threshold(
                0.5, constraints=["nonexistent_constraint"]
            )

    def test_an_unknown_objective_name_is_refused_rather_than_silently_swapped(self):
        labels, probs, groups = _biased_scores(n=80)
        analyzer = ThresholdAnalyzer(labels, probs, groups)

        with pytest.raises(ValueError, match="not a performance objective"):
            analyzer.find_optimal_threshold(
                constraint="demographic_parity", objective="f1", n_thresholds=10
            )

    def test_the_refusal_names_the_objectives_that_would_have_worked(self):
        labels, probs, groups = _biased_scores(n=40)

        with pytest.raises(ValueError) as excinfo:
            ThresholdAnalyzer(labels, probs, groups).find_optimal_threshold(
                objective="f1", n_thresholds=4
            )

        assert "f1_score" in str(excinfo.value)

    def test_the_advertised_objectives_are_exactly_the_ones_computed(self):
        """The list the refusal is built from must not drift from the metrics.

        A name in the table that the metric function does not return would sail
        past the refusal and then be looked up, so the two sets are pinned equal
        rather than trusted to stay in step.
        """
        labels, probs, groups = _biased_scores(n=40)
        computed = ThresholdAnalyzer(labels, probs, groups).analyze_threshold(0.5)

        assert set(computed.performance_metrics) == set(PERFORMANCE_OBJECTIVES)

    def test_every_advertised_objective_is_still_accepted(self):
        """The control on the refusal above: it must refuse the wrong names ONLY.

        A guard that refused everything would pass the two tests above and make
        the analyzer useless, so each advertised objective is exercised here.
        """
        labels, probs, groups = _biased_scores(n=60)
        analyzer = ThresholdAnalyzer(labels, probs, groups)

        for objective in sorted(PERFORMANCE_OBJECTIVES):
            result = analyzer.find_optimal_threshold(objective=objective, n_thresholds=5)

            assert 0.0 < result["optimal_threshold"] < 1.0
            assert objective in result["metrics"]
