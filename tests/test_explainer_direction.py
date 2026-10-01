"""The explainer may not decide what a metric IS by looking inside its name.

``FairExplAIner`` turns a number into prose a human acts on, so every claim it
makes has to be one the code established. Three of its claims were decided by a
containment test on the metric name (``"demographic_parity" in metric_name`` and
two siblings), which is the bug class recorded in
``evaluation.vfairness_metrics._metric_direction``: "ratio" is a substring of
"cali[bratio]n_difference", and in the same shape "equal_opportunity" is a
substring of "un[equal_opportunity]_cost". The repo guard
``tests/test_verdict_defaults.py::TestNoSubstringDirectionTestAnywhereInThePackage``
was RED on those three lines.

This module pins the four claims that matter on the explainer surface:

1. a ratio metric is read in the ratio direction (higher is better);
2. the calibration family is NOT read as a ratio, so a large miscalibration
   cannot come out as a pass;
3. a metric whose better-direction cannot be resolved gets could-not-check
   wording, never a verdict and never reassurance;
4. a metric that was never computed does not receive the advice written for a
   PASSING metric;
5. a metric the library holds NO definition for is presented as could-not-check,
   at a severity a graded benign metric never gets, and never in the shape of a
   metric that was checked.

Each test was checked by SABOTAGE: the exact defect was put back and the test
was watched go red. A guard nobody has seen fail is not a guard.
"""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explainer import (
    CLASSIFICATION_METRICS,
    FairExplAIner,
    _metric_family,
)

_PASSING_METRIC_ADVICE = "Continue monitoring this metric"


@pytest.fixture
def explainer():
    return FairExplAIner()


# ── 1. a ratio metric is graded in the ratio direction ─────────────────────


class TestRatioMetricsGetRatioDirectionProse:
    def test_a_low_ratio_is_the_severe_end_not_the_good_end(self, explainer):
        """0.20 on a four-fifths ratio is the WORST case, not a small number."""
        explanation = explainer.explain_metric("demographic_parity_ratio", 0.20)
        assert explanation.severity == "critical"
        assert "ratio" in explanation.evaluation.lower()
        assert "Excellent" not in explanation.evaluation

    def test_a_high_ratio_is_the_good_end(self, explainer):
        explanation = explainer.explain_metric("demographic_parity_ratio", 0.95)
        assert explanation.severity == "info"
        assert "near-perfect parity" in explanation.evaluation

    def test_the_benchmarks_are_stated_as_a_required_minimum(self, explainer):
        """A ratio passes at or ABOVE its bound, so the benchmark reads '>='."""
        context = explainer.explain_metric("demographic_parity_ratio", 0.95).benchmark_context
        assert "≥" in context
        assert "≤" not in context


# ── 2. calibration is not a ratio ──────────────────────────────────────────


class TestCalibrationIsNotReadAsARatio:
    """The original incident, on this surface.

    ``"ratio" in "calibration_difference"`` is True. Under that test a
    calibration gap of 0.45 against a 0.05 bound was read as a ratio well above
    the four-fifths floor, i.e. as a PASS.
    """

    def test_a_large_calibration_gap_is_severe(self, explainer):
        explanation = explainer.explain_metric("calibration_difference", 0.45)
        assert explanation.severity == "critical"
        assert "difference" in explanation.evaluation.lower()
        assert "near-perfect parity" not in explanation.evaluation

    def test_its_benchmarks_are_stated_as_a_maximum(self, explainer):
        context = explainer.explain_metric("calibration_difference", 0.45).benchmark_context
        assert "≤" in context
        assert "≥" not in context

    def test_it_is_not_given_the_advice_of_another_family(self, explainer):
        """Calibration is not the demographic-parity family, however it is spelled."""
        recommendation = explainer.explain_metric("calibration_difference", 0.45).recommendation
        assert "demographic parity is a priority" not in recommendation


# ── 3. an unresolvable metric gets could-not-check, not a verdict ──────────


def _explainer_with_a_directionless_metric():
    """A definition whose NAME carries no direction signal at all.

    Every metric shipped in ``CLASSIFICATION_METRICS`` resolves today, so the
    unresolvable case is reached the way a client reaches it: by adding their own
    metric definition to the explainer.
    """
    explainer = FairExplAIner()
    explainer.metrics_definitions = dict(CLASSIFICATION_METRICS)
    explainer.metrics_definitions["client_bespoke_score"] = {
        "definition": "A client-specific score.",
        "interpretation_guide": "See the client's documentation.",
        "thresholds": {"excellent": 0.03, "acceptable": 0.05, "concerning": 0.08},
    }
    return explainer


class TestUnknownDirectionIsCouldNotCheck:
    def test_the_evaluation_says_it_was_not_graded(self):
        explanation = _explainer_with_a_directionless_metric().explain_metric(
            "client_bespoke_score", 0.42
        )
        assert "could not check" in explanation.evaluation.lower()
        assert "not graded against a threshold" in explanation.evaluation
        for verdict_word in ("Excellent", "Acceptable", "Concerning", "Critical"):
            assert verdict_word not in explanation.evaluation

    def test_the_recommendation_does_not_reassure(self):
        explanation = _explainer_with_a_directionless_metric().explain_metric(
            "client_bespoke_score", 0.42
        )
        assert "NOT MEASURED" in explanation.recommendation
        assert _PASSING_METRIC_ADVICE not in explanation.recommendation
        assert "not as passing" in explanation.recommendation

    def test_no_family_advice_is_attached_to_a_name_we_cannot_place(self):
        """Contradictory name: a violation token AND the ratio suffix at once."""
        assert _metric_family("demographic_parity_gap_ratio") is None


# ── 4. nothing that was never computed gets the passing advice ─────────────


class TestNeverComputedGetsNoPassingAdvice:
    def test_a_nan_metric_is_not_told_to_keep_monitoring(self, explainer):
        explanation = explainer.explain_metric(
            "demographic_parity_difference",
            float("nan"),
            excluded_groups=["F"],
        )
        assert _PASSING_METRIC_ADVICE not in explanation.recommendation
        assert "NOT MEASURED" in explanation.recommendation

    def test_it_is_told_what_to_actually_do(self, explainer):
        """Could-not-check advice must be actionable, not reassuring."""
        recommendation = explainer.explain_metric(
            "demographic_parity_difference",
            np.nan,
            excluded_groups=["F"],
        ).recommendation
        assert "min_group_size" in recommendation
        assert "'F'" in recommendation
        assert "re-run" in recommendation

    def test_a_measured_pass_still_gets_the_passing_advice(self, explainer):
        """The negative control: the passing branch must not have been broken."""
        explanation = explainer.explain_metric("demographic_parity_difference", 0.01)
        assert explanation.severity == "info"
        assert _PASSING_METRIC_ADVICE in explanation.recommendation


# ── 5. family advice is selected on whole tokens ───────────────────────────


class TestFamilyAdviceIsSelectedOnWholeTokens:
    @pytest.mark.parametrize(
        "metric_name,family",
        [
            ("demographic_parity_difference", "demographic_parity"),
            ("demographic_parity_ratio", "demographic_parity"),
            ("equal_opportunity_difference", "equal_opportunity"),
            ("equalized_odds_difference", "equalized_odds"),
            # A per-column tracker key still belongs to its family.
            ("equal_opportunity_difference_gender", "equal_opportunity"),
            # Human label forms normalise onto the same family.
            ("Demographic Parity Difference", "demographic_parity"),
        ],
    )
    def test_the_real_families_still_resolve(self, metric_name, family):
        assert _metric_family(metric_name) == family

    @pytest.mark.parametrize(
        "metric_name",
        [
            # "equal_opportunity" is a SUBSTRING of "unequal_opportunity_cost",
            # exactly as "ratio" is a substring of "calibration_difference".
            "unequal_opportunity_cost",
            "calibration_difference",
            "predictive_parity_difference",
            "",
        ],
    )
    def test_a_name_that_merely_contains_a_family_token_is_not_that_family(self, metric_name):
        assert _metric_family(metric_name) is None

    def test_the_substring_lookalike_gets_no_equal_opportunity_advice(self, explainer):
        """End to end: the prose itself, not just the helper.

        Under the containment test this metric was told to reduce false
        negatives to improve equal opportunity, which is a statement about what
        the number measures that nothing had established.
        """
        recommendation = explainer._generate_recommendation(
            "unequal_opportunity_cost",
            value=0.40,
            threshold=0.10,
            severity="high",
        )
        assert "improve equal opportunity" not in recommendation
        assert "URGENT" in recommendation

    def test_a_real_equal_opportunity_gap_still_gets_its_advice(self, explainer):
        explanation = explainer.explain_metric("equal_opportunity_difference", 0.40)
        assert "improve equal opportunity" in explanation.recommendation


# ── 6. a violation magnitude is graded on its MAGNITUDE ────────────────────


class TestASignedGapIsGradedOnItsMagnitude:
    """The sign hole that wave 4 closed in ``_fairness_verdict``, on this surface.

    A lower-is-better metric is a violation MAGNITUDE, so the band has to be
    symmetric about zero. Graded on the raw value instead, every negative gap
    slid under the "excellent" bound: ``explain_metric('mean_prediction_difference',
    -0.45)`` returned "Excellent! The difference of -0.4500 indicates
    near-perfect fairness across groups.", severity "info", with the advice
    written for a PASSING metric attached. The disparity is the same size as
    ``+0.45``, which the same call grades "Critical".
    """

    def test_a_large_negative_gap_is_not_excellent(self):
        explainer = FairExplAIner(task_type="regression")
        explanation = explainer.explain_metric("mean_prediction_difference", -0.45)
        assert "Excellent" not in explanation.evaluation
        assert explanation.severity == "critical"
        assert _PASSING_METRIC_ADVICE not in explanation.recommendation

    def test_both_signs_of_one_gap_are_graded_alike(self):
        explainer = FairExplAIner(task_type="regression")
        negative = explainer.explain_metric("mean_prediction_difference", -0.45)
        positive = explainer.explain_metric("mean_prediction_difference", 0.45)
        assert negative.severity == positive.severity

    def test_the_signed_value_is_still_reported_as_measured(self):
        """Grade on the magnitude, but never rewrite the number itself."""
        explainer = FairExplAIner(task_type="regression")
        explanation = explainer.explain_metric("mean_prediction_difference", -0.45)
        assert "-0.4500" in explanation.evaluation

    def test_a_small_negative_gap_is_still_excellent(self):
        """The negative control: the magnitude band must stay a band."""
        explainer = FairExplAIner(task_type="regression")
        explanation = explainer.explain_metric("mean_prediction_difference", -0.001)
        assert explanation.severity == "info"


# ── 7. the no-definition fallback is could-not-check, not a quiet pass ─────


class TestTheUndefinedMetricFallbackReadsAsCouldNotCheck:
    """A metric the library holds no definition for was dressed as a checked one.

    ``explain_metric`` falls through to ``_generic_explanation`` whenever the
    name has no entry in ``metrics_definitions`` or ``STATISTICAL_MEASURES``.
    Executed on this repo before 2026-08-27,
    ``explain_metric('disparate_impact_ratio', 0.60)`` returned::

        evaluation:     'The computed value is 0.6000.'
        severity:       'info'
        recommendation: 'Review this metric in the context of your specific use case.'

    Nothing in that is false, which is exactly why it survived every review. The
    lie is the SHAPE: it is the same severity, the same fields and the same
    unbothered tone a genuinely graded, genuinely benign metric receives, so a
    reader cannot tell "we checked and it is fine" from "we have no idea what
    this metric is". Could-not-check is a THIRD state and has to look like one.
    """

    UNDEFINED = "disparate_impact_ratio"

    def test_the_evaluation_says_no_threshold_was_applied(self, explainer):
        evaluation = explainer.explain_metric(self.UNDEFINED, 0.60).evaluation
        assert "COULD NOT CHECK" in evaluation
        assert "no definition" in evaluation
        assert "not graded" in evaluation
        assert "neither a pass nor a fail" in evaluation

    def test_the_evaluation_makes_no_verdict_claim(self, explainer):
        evaluation = explainer.explain_metric(self.UNDEFINED, 0.60).evaluation
        for verdict_word in ("Excellent", "Acceptable", "Concerning", "Critical", "High concern"):
            assert verdict_word not in evaluation

    def test_the_severity_is_not_the_one_a_benign_graded_metric_gets(self, explainer):
        """The heart of it: 'info' is what a PASSING metric is given."""
        benign = explainer.explain_metric("demographic_parity_difference", 0.01)
        undefined = explainer.explain_metric(self.UNDEFINED, 0.60)
        assert benign.severity == "info"
        assert undefined.severity != benign.severity
        assert undefined.severity == "could_not_check"
        assert undefined.severity not in ("info", "low", "medium", "high", "critical")

    def test_the_recommendation_does_not_reassure_and_says_what_to_do(self, explainer):
        recommendation = explainer.explain_metric(self.UNDEFINED, 0.60).recommendation
        assert "NOT MEASURED" in recommendation
        assert _PASSING_METRIC_ADVICE not in recommendation
        assert "Review this metric in the context of your specific use case." not in recommendation
        assert "not as passing" in recommendation
        # The two routes out, both named.
        assert "add a definition" in recommendation
        assert "demographic_parity_difference" in recommendation

    def test_the_definition_field_admits_there_is_no_definition(self, explainer):
        explanation = explainer.explain_metric(self.UNDEFINED, 0.60)
        assert "NOT DEFINED" in explanation.definition
        # "Custom metric: <name>" implied the library knew it to be a custom
        # metric of the caller's. It knows only that it has never heard of it.
        assert explanation.definition != f"Custom metric: {self.UNDEFINED}"

    def test_a_resolvable_direction_is_used_to_say_something_true(self, explainer):
        """``disparate_impact_ratio`` resolves higher-is-better by NAME alone."""
        evaluation = explainer.explain_metric(self.UNDEFINED, 0.60).evaluation
        assert "parity ratio" in evaluation
        assert "larger value is better" in evaluation
        # ... and still refuses to turn that into a verdict.
        assert "reading of the NAME, not a measurement" in evaluation

    def test_an_unresolvable_direction_says_so_rather_than_guessing(self, explainer):
        evaluation = explainer.explain_metric("client_bespoke_score", 0.42).evaluation
        assert "no reliable direction signal" in evaluation
        assert "larger value is better" not in evaluation
        assert "smaller value is better" not in evaluation

    def test_a_lower_is_better_name_is_read_in_its_own_direction(self, explainer):
        evaluation = explainer.explain_metric("bespoke_vendor_gap", 0.42).evaluation
        assert "violation magnitude" in evaluation
        assert "smaller value is better" in evaluation
        assert "reading of the NAME, not a measurement" in evaluation

    def test_a_value_that_was_never_computed_is_not_printed_as_a_number(self, explainer):
        """NaN formatted as '.4f' reads 'nan', which looks like a measurement."""
        explanation = explainer.explain_metric(self.UNDEFINED, float("nan"))
        assert "NaN (never computed)" in explanation.evaluation
        assert explanation.severity == "could_not_check"

    def test_a_non_numeric_value_does_not_crash_the_report(self, explainer):
        """The old body ran ``f'{value:.4f}'`` unguarded on whatever it was given."""
        explanation = explainer.explain_metric(self.UNDEFINED, "not-a-number")
        assert "COULD NOT CHECK" in explanation.evaluation
        assert explanation.severity == "could_not_check"


class TestDefinedMetricsAreNumericallyUntouchedByTheFallbackFix:
    """The over-correction control.

    A library that answers could-not-check for everything passes every test in
    the class above and is useless. Every metric that HAS a definition must come
    out of ``explain_metric`` exactly as it did before, byte for byte.
    """

    # Captured by execution at 7c5fff8, before the fallback was touched.
    SEVERITY_SWEEP = {
        ("demographic_parity_difference", 0.01): "info",
        ("demographic_parity_difference", 0.06): "low",
        ("demographic_parity_difference", 0.12): "medium",
        ("demographic_parity_difference", 0.18): "high",
        ("demographic_parity_difference", 0.25): "critical",
        ("demographic_parity_ratio", 0.01): "critical",
        ("demographic_parity_ratio", 0.95): "info",
        ("equal_opportunity_difference", 0.01): "info",
        ("equal_opportunity_difference", 0.06): "medium",
        ("equal_opportunity_difference", 0.12): "high",
        ("equal_opportunity_difference", 0.18): "critical",
        ("equalized_odds_difference", 0.06): "low",
        ("equalized_odds_difference", 0.12): "medium",
        ("predictive_parity_difference", 0.06): "medium",
        ("calibration_difference", 0.01): "info",
        ("calibration_difference", 0.06): "medium",
        ("calibration_difference", 0.12): "critical",
        # Statistical measures reach explain_metric through STATISTICAL_MEASURES
        # and are graded by the same bands; pinned so the fallback edit cannot
        # reroute them into could-not-check.
        ("confidence_interval", 0.3): "critical",
        ("cohens_d", 0.3): "critical",
    }

    @pytest.mark.parametrize("key,expected", sorted(SEVERITY_SWEEP.items()))
    def test_every_defined_metric_keeps_its_exact_severity(self, explainer, key, expected):
        metric_name, value = key
        explanation = explainer.explain_metric(metric_name, value)
        assert explanation.severity == expected
        assert "COULD NOT CHECK" not in explanation.evaluation
        assert "NOT DEFINED" not in explanation.definition

    def test_a_defined_metric_still_carries_its_shipped_definition(self, explainer):
        explanation = explainer.explain_metric("demographic_parity_difference", 0.01)
        assert (
            explanation.definition
            == CLASSIFICATION_METRICS["demographic_parity_difference"]["definition"]
        )
        assert explanation.evaluation == (
            "Excellent! The difference of 0.0100 indicates near-perfect fairness across groups."
        )
        assert _PASSING_METRIC_ADVICE in explanation.recommendation

    def test_a_defined_metric_that_could_not_be_computed_says_could_not_check(self, explainer):
        """Was pinned to severity 'info' as out of scope for d160fcd, which
        introduced `could_not_check` and applied it only to the NOT-DEFINED
        branch. The fifth-iteration audit flagged the branches that wave left
        behind, and the principle is that wave's own, stated in its test name:
        'info' is what a PASSING metric is given.

        A metric the library knows but never COMPUTED is not a passing metric.
        Its recommendation was already honest ("NOT MEASURED"); only the
        severity beside it disagreed, which is what made a NaN row and a
        genuinely excellent row sort and colour identically.
        """
        explanation = explainer.explain_metric(
            "demographic_parity_difference", float("nan"), excluded_groups=["F"]
        )
        benign = explainer.explain_metric("demographic_parity_difference", 0.01)
        assert explanation.severity == "could_not_check"
        assert explanation.severity != benign.severity
        assert "COULD NOT CHECK" in explanation.evaluation
        assert "NOT MEASURED" in explanation.recommendation

    def test_a_defined_metric_with_an_unresolvable_direction_says_could_not_check(self):
        """Same reasoning as the test above, for the other branch d160fcd left
        at 'info'. This one was the sharper of the two: `_evaluate_value` and
        `explain_metric` returned DIFFERENT severities for the same metric and
        value, so the two surfaces disagreed with each other."""
        explainer = _explainer_with_a_directionless_metric()
        explanation = explainer.explain_metric("client_bespoke_score", 0.42)
        assert explanation.severity == "could_not_check"
        assert "could not check" in explanation.evaluation.lower()
        # The two surfaces now agree, which is the point.
        assert (
            explainer._evaluate_value("client_bespoke_score", 0.42, 0.1, {"thresholds": {}})[1]
            == "could_not_check"
        )
