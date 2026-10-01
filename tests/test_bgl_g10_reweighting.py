"""G10 grading pins: post_processing.reweighting (analyzer records + reweighter base).

REWEIGHTING IS A MITIGATION, so the failure mode is inverted: a reweighter that has
become an identity map, or a comparison in which nothing helped, produces a BETTER
looking report, not a worse one. No pin here judges the unit by the fairness number
it prints; each asks whether the mapping still did anything.

Two defects pinned, both reproduced by execution:

D8 THE REPORT RECOMMENDED A MITIGATION THAT MITIGATED NOTHING. When not one method
   reduced the disparity, ``_generate_recommendations`` still printed "For maximum
   fairness improvement, use 'multiplicative' (reduces disparity by 0.000)." and
   "Overall recommendation: 'multiplicative' provides the best trade-off between
   fairness, accuracy, and calibration." ``max`` always returns a winner, and the
   winner of a comparison between methods that all changed nothing is not a
   recommendation. The template is worse in the negative direction, which ``max``
   over negatives reaches: a run measured the same day gave rejection_option a
   fairness_improvement of -0.49, and as the least-bad row it would have printed
   "reduces disparity by -0.490" for a method that WIDENED the gap.
D9 THE COULD-NOT-CHECK WAS WRITTEN FOUR LINES UP AND THE NEXT BLOCK UNDID IT.
   ``ReweightingResult.summary()`` prints "NOT MEASURED (no comparison was
   possible)" for a non-finite fairness change and then rendered the identical
   condition as the value ``nan`` under "Calibration Impact:".
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.post_processing.reweighting.analyzer import (
    ReweightingAnalysisReport,
    ReweightingAnalyzer,
    ReweightingImpactResult,
)
from vfairness.post_processing.reweighting.reweighter import (
    BaseReweighter,
    PredictionReweighter,
    ReweightingMethod,
    ReweightingResult,
)

N = 400
SENS = np.array(["a"] * (N // 2) + ["b"] * (N // 2))


class _Identity(BaseReweighter):
    """A reweighter that has become an identity map: the mapping no longer depends
    on the group at all, which is the shape that reports success for free."""

    def fit(self, *, y_true, y_prob, sensitive_attr, sample_weight=None):
        self.is_fitted = True
        return self

    def transform(self, y_prob, sensitive_attr):
        return np.asarray(y_prob, dtype=float)


def _biased_frame(seed: int = 1):
    rng = np.random.default_rng(seed)
    y_prob = np.clip(rng.normal(np.where(SENS == "a", 0.75, 0.35), 0.12, N), 0.01, 0.99)
    y_true = (rng.uniform(size=N) < y_prob).astype(int)
    return y_prob, y_true


def _nothing_to_reduce():
    """Deterministic: identical score/label pattern in both groups, so the
    demographic-parity gap is exactly 0.000 and no method can reduce it."""
    return np.tile([0.2, 0.8], N // 2), np.tile([0, 1], N // 2)


def _report(y_prob, y_true):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = ReweightingAnalyzer(
            y_true=y_true, y_prob=y_prob, sensitive_attr=SENS
        ).full_analysis()
    return report, [str(w.message) for w in caught]


# ------------------------------------------------ D8: a recommendation with no effect


def test_a_run_where_nothing_reduced_the_disparity_says_so_in_the_recommendations():
    report, messages = _report(*_nothing_to_reduce())

    assert report.comparison_summary, "the reproduction needs surviving methods"
    best = max(v["fairness_improvement"] for v in report.comparison_summary.values())
    assert best <= 0, "the fixture must leave nothing to reduce"

    joined = " ".join(report.recommendations)
    assert "NO METHOD REDUCED THE DISPARITY" in joined, report.recommendations
    assert "NO OVERALL RECOMMENDATION" in joined
    assert "For maximum fairness improvement" not in joined
    assert "provides the best trade-off" not in joined
    assert any("NOT ONE" in m for m in messages), messages


def test_the_disclosure_is_reachable_from_the_object_not_only_from_the_prose():
    """``best_method`` is a plain str and a caller that reads it and applies that
    method would have applied a no-op, so the flag rides beside it."""
    report, _ = _report(*_nothing_to_reduce())
    assert report.metadata["no_method_reduced_disparity"] is True
    assert report.metadata["best_fairness_improvement"] == pytest.approx(0.0)
    assert report.metadata["original_demographic_parity_diff"] == pytest.approx(0.0)
    # best_method still names the winner of the comparison that WAS made.
    assert report.best_method in report.comparison_summary
    assert report.to_dict()["metadata"]["no_method_reduced_disparity"] is True


def test_the_human_readable_summary_carries_it_next_to_the_recommended_method():
    report, _ = _report(*_nothing_to_reduce())
    lines = report.summary().splitlines()
    recommended = next(i for i, line in enumerate(lines) if line.startswith("Recommended method"))
    assert "NOT A MITIGATION" in lines[recommended + 1], lines[recommended : recommended + 3]
    assert "not a method that improved fairness" in lines[recommended + 1]


def test_control_a_run_where_a_method_really_helps_keeps_its_recommendation():
    """CONTROL for D8. A report that always says NO METHOD would pass every pin
    above and destroy the unit."""
    report, messages = _report(*_biased_frame())
    joined = " ".join(report.recommendations)
    assert "For maximum fairness improvement" in joined, report.recommendations
    assert "provides the best trade-off" in joined
    assert "NO METHOD REDUCED THE DISPARITY" not in joined
    assert report.metadata["no_method_reduced_disparity"] is False
    assert report.metadata["best_fairness_improvement"] > 0
    assert not any("NOT ONE" in m for m in messages)
    assert "NOT A MITIGATION" not in report.summary()


def test_the_flag_is_always_present_so_an_absent_key_is_never_read_as_good_news():
    for frame in (_nothing_to_reduce(), _biased_frame()):
        report, _ = _report(*frame)
        assert "no_method_reduced_disparity" in report.metadata
        assert isinstance(report.metadata["no_method_reduced_disparity"], bool)


def test_a_hand_built_report_without_the_key_does_not_claim_a_mitigation_either():
    """``is True`` rather than a truthy test: absence is not the same claim as
    False, and a report assembled by hand carries neither."""
    blank = ReweightingAnalysisReport(
        method_results=[], best_method="none", comparison_summary={}, recommendations=[]
    )
    assert "NOT A MITIGATION" not in blank.summary()
    assert blank.metadata == {}


# ------------------------------------------------------ an identity map is not helpful


def test_an_identity_map_reports_no_improvement_rather_than_a_better_number():
    """The mitigation-specific check: the mapping no longer depends on the group,
    so the adjusted gap must equal the original one exactly and the trade-off
    score must be 0.0. A reweighter is never judged by its fairness number here."""
    y_prob, y_true = _biased_frame()
    analyzer = ReweightingAnalyzer(y_true=y_true, y_prob=y_prob, sensitive_attr=SENS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyzer.analyze_method(_Identity())

    original = result.original_fairness["demographic_parity_diff"]
    adjusted = result.adjusted_fairness["demographic_parity_diff"]
    assert original > 0.5, "the fixture must carry a real disparity to leave in place"
    assert adjusted == pytest.approx(original), "an identity map cannot change the gap"
    assert result.trade_off_score == pytest.approx(0.0)
    assert result.original_fairness["group_rates"] == result.adjusted_fairness["group_rates"]
    assert result.method == "_Identity"


def test_control_a_real_reweighter_changes_the_mapping_per_group():
    """CONTROL: the group-conditional mapping must still vary, or the pin above
    would pass for a library in which nothing works."""
    y_prob, y_true = _biased_frame()
    reweighter = PredictionReweighter(method="multiplicative")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adjusted = reweighter.fit_transform(y_true, y_prob, SENS)
    factors = reweighter.result_.group_adjustments
    assert set(factors) == {"a", "b"}
    assert factors["a"] != pytest.approx(factors["b"]), "the mapping must depend on the group"
    assert np.ptp(adjusted - y_prob) > 1e-6, "the scores must actually move"
    assert reweighter.result_.fairness_improvement["disparity_reduction"] > 0


# ----------------------------------------------- D9: the record's own text surface


def test_a_calibration_impact_that_was_not_measured_does_not_print_as_a_value():
    """D9. The fairness block above it already says NOT MEASURED for exactly this
    condition; the two states must read alike wherever they appear."""
    result = ReweightingResult(
        method=ReweightingMethod.MULTIPLICATIVE,
        group_adjustments={"a": 1.2, "b": float("nan")},
        original_metrics={"disparity": 0.6},
        adjusted_metrics={"disparity": float("nan")},
        fairness_improvement={"disparity_reduction": float("nan")},
        calibration_impact={
            "ece_change": float("nan"),
            "original_ece": float("inf"),
            "mean_shift": 0.0,
            "n_clipped": 3,
        },
        requested_constraint="equalized_odds",
        honoured_constraint="demographic_parity",
        groups_not_compared=["b"],
        unfittable_groups={"b": "no scored rows"},
    )
    text = result.summary()
    calibration_block = text.split("Calibration Impact:")[1]
    assert "nan" not in calibration_block, calibration_block
    assert "inf" not in calibration_block
    assert calibration_block.count("NOT MEASURED (no comparison was possible)") == 2
    # CONTROL inside the same block: a measured 0.0 and a real count survive.
    assert "mean_shift: 0.0000" in calibration_block
    assert "n_clipped: 3.0000" in calibration_block
    # And the rest of the record's three-state rendering is still intact.
    assert "disparity_reduction: NOT MEASURED (no comparison was possible)" in text
    assert "b: NOT FITTED" in text
    assert "equalized_odds was NOT enforced" in text
    assert "Groups NOT compared:" in text


def test_a_regression_is_printed_as_a_regression_and_not_as_an_improvement():
    """The sign carries the whole finding: a fit that moved the gap the wrong way
    must not render under an 'improvement' arrow."""
    worse = ReweightingResult(
        method=ReweightingMethod.MULTIPLICATIVE,
        group_adjustments={"a": 1.0},
        original_metrics={},
        adjusted_metrics={},
        fairness_improvement={"disparity_reduction": -1.0},
        calibration_impact={},
    )
    assert "disparity_reduction: -1.0000 (REGRESSION" in worse.summary()
    better = ReweightingResult(
        method=ReweightingMethod.MULTIPLICATIVE,
        group_adjustments={"a": 1.0},
        original_metrics={},
        adjusted_metrics={},
        fairness_improvement={"disparity_reduction": 0.25},
        calibration_impact={},
    )
    assert "disparity_reduction: +0.2500 (improvement)" in better.summary()
    flat = ReweightingResult(
        method=ReweightingMethod.MULTIPLICATIVE,
        group_adjustments={"a": 1.0},
        original_metrics={},
        adjusted_metrics={},
        fairness_improvement={"disparity_reduction": 0.0},
        calibration_impact={},
    )
    assert "disparity_reduction: 0.0000 (no change)" in flat.summary()


# ---------------------------------------------------------------------- the records


def test_reweighting_result_dict_carries_every_field_it_declares():
    result = ReweightingResult(
        method=ReweightingMethod.MULTIPLICATIVE,
        group_adjustments={"a": 1.0},
        original_metrics={"x": 1.0},
        adjusted_metrics={"x": 0.5},
        fairness_improvement={"disparity_reduction": 0.5},
        calibration_impact={"ece_change": 0.01},
    )
    payload = result.to_dict()
    assert set(payload) == set(result.__dataclass_fields__)
    assert payload["method"] == "multiplicative", "the enum must serialise to its value"
    assert payload["groups_not_compared"] is None and payload["unfittable_groups"] is None


def test_impact_result_dict_carries_every_field_it_declares():
    result = ReweightingImpactResult(
        method="m",
        original_fairness={"demographic_parity_diff": 0.4},
        adjusted_fairness={"demographic_parity_diff": 0.1},
        original_performance={"accuracy": 0.8},
        adjusted_performance={"accuracy": 0.79},
        calibration_metrics={"ece_change": 0.0},
        trade_off_score=0.295,
    )
    assert set(result.to_dict()) == set(result.__dataclass_fields__)
    assert result.to_dict()["trade_off_score"] == pytest.approx(0.295)


def test_report_dict_carries_every_field_and_defaults_are_per_instance():
    report = ReweightingAnalysisReport(
        method_results=[], best_method="none", comparison_summary={}, recommendations=[]
    )
    assert set(report.to_dict()) == set(report.__dataclass_fields__)
    ReweightingAnalysisReport([], "n", {}, []).metadata["leak"] = 1
    ReweightingAnalysisReport([], "n", {}, []).failed_methods["leak"] = "x"
    assert ReweightingAnalysisReport([], "n", {}, []).metadata == {}
    assert ReweightingAnalysisReport([], "n", {}, []).failed_methods == {}


def test_a_method_whose_effect_could_not_be_measured_is_excluded_not_scored_zero():
    """A single-group frame makes every demographic-parity gap NaN, so no method
    has a measurable effect and none of them may be ranked."""
    y_prob, y_true = _biased_frame()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = ReweightingAnalyzer(
            y_true=y_true, y_prob=y_prob, sensitive_attr=np.array(["a"] * N)
        ).full_analysis()
    assert report.method_results == []
    assert report.comparison_summary == {}
    assert report.best_method == "None (all methods failed)"
    assert report.failed_methods, "every method must be recorded as not evaluated"
    assert "No methods could be analyzed successfully." in report.recommendations
    assert any("NOT EVALUATED" in r for r in report.recommendations)
    # Nothing here may claim a mitigation, and nothing may claim one failed to
    # help either: there was no comparison at all.
    assert report.metadata["no_method_reduced_disparity"] is False
    assert not math.isfinite(report.metadata["best_fairness_improvement"])
    assert any("Cannot measure" in str(m.message) for m in caught)


def test_the_base_reweighter_cannot_be_instantiated_and_records_what_it_honours():
    with pytest.raises(TypeError, match="abstract"):
        BaseReweighter()
    with pytest.warns(UserWarning):
        reweighter = PredictionReweighter(method="multiplicative", constraint="equalized_odds")
    assert reweighter.constraint == "equalized_odds", "what the caller asked for"
    assert reweighter.honoured_constraint == "demographic_parity", "what is enforced"
    assert reweighter.is_fitted is False and reweighter.result_ is None
