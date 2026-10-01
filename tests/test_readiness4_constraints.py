"""R4C: an unevaluable constraint must never be reported, or DRAWN, as a FAIL.

THE DEFECT, reproduced by execution before it was fixed. ``ConstraintViolation``
records could-not-check honestly in ``details["insufficient_data"]`` (written at
constraints/base.py:540, :647 and :732). ``is_satisfied`` is deliberately False
in that state, which is the right fail-closed direction for something that
STEERS TRAINING, and every REPORTING consumer read that False as a measured
verdict. Measured on 240 rows whose third group had no positive labels, so the
TPR arm of equalized odds was undefined:

    producer   overall_violation=nan, is_satisfied=False, insufficient_data=True
    boundary   MethodComparison(fairness_violation=nan, constraint_satisfied=False)
    canvas     a red FAIL chip, a violation bar in the alert colour labelled "nan"
    <desc>     "0 of 1 method(s) satisfy the fairness constraint. (severity: HIGH)"
    report     critical issue "No training method achieved the fairness
               constraint" at severity HIGH
    advice     "Minimizes constraint violation (nan)", chosen by
               min(..., key=fairness_violation) over NaN, which never wins and
               never loses, so `min` simply returned the FIRST method
    sweep      all 11 configurations reported as the "pareto_frontier", because
               every comparison against NaN is False, so nothing was dominated

None of that failure was measured. Group C's TPR was undefined, so the disparity
was unknown, not large, and "unknown" drawn as a red bar is a fabricated
measurement in the most legible form there is.

WHAT THE FIX IS. The rendering layer already speaks the third state: it reads
None (``adapters_training._method_state`` gates the chip on
``satisfied is not None``), and NaN is not None, so NaN sailed through every one
of those gates. The boundary now translates once, in
``constraints.base._reported_constraint``, and hands on (None, None).

EVERY refusal pin below is paired with an OVER-CORRECTION CONTROL asserting
MEASURED values: a genuinely violated constraint must still report and draw
exactly as it does today, and a measured violation of 0.0 is a RESULT, not a
missing one.
"""

import math
import re
import warnings

import numpy as np
import pytest

from vfairness.in_processing.analyzer import (
    FairnessTrainingAnalyzer,
    MethodComparison,
    _measured,
)
from vfairness.in_processing.constraints.base import (
    EqualizedOddsConstraint,
    _reported_constraint,
)
from vfairness.in_processing.wrappers.sklearn_wrappers import FairClassifier
from vfairness.rendering.adapters_training import method_comparison_to_svg

# ---------------------------------------------------------------------------
# Fixtures. Group labels are dtype=object on purpose: a numpy "<U5" array
# TRUNCATES longer labels and would quietly void the fixture.
# ---------------------------------------------------------------------------

N_PER_GROUP = 20


def _unmeasurable():
    """A, B measurable; C has NO positive labels, so the TPR arm is undefined."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3 * N_PER_GROUP, 3))
    groups = np.array(["A"] * N_PER_GROUP + ["B"] * N_PER_GROUP + ["C"] * N_PER_GROUP, dtype=object)
    y = np.zeros(3 * N_PER_GROUP, dtype=int)
    y[:10] = 1  # A: 10 positives
    y[N_PER_GROUP : N_PER_GROUP + 10] = 1  # B: 10 positives
    # C: none at all
    y_pred = np.zeros(3 * N_PER_GROUP, dtype=int)
    y_pred[:9] = 1  # A TPR 0.9
    y_pred[N_PER_GROUP : N_PER_GROUP + 3] = 1  # B TPR 0.3
    return X, y, groups, y_pred


def _measurable_breach():
    """Every group has both labels, and the TPR spread is a real 0.60."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(3 * N_PER_GROUP, 3))
    groups = np.array(["A"] * N_PER_GROUP + ["B"] * N_PER_GROUP + ["C"] * N_PER_GROUP, dtype=object)
    y = np.zeros(3 * N_PER_GROUP, dtype=int)
    y_pred = np.zeros(3 * N_PER_GROUP, dtype=int)
    for i in range(3):
        start = i * N_PER_GROUP
        y[start : start + 10] = 1  # 10 positives, 10 negatives per group
    y_pred[0:9] = 1  # A TPR 0.9
    y_pred[N_PER_GROUP : N_PER_GROUP + 6] = 1  # B TPR 0.6
    y_pred[2 * N_PER_GROUP : 2 * N_PER_GROUP + 3] = 1  # C TPR 0.3
    return X, y, groups, y_pred


def _analyzer(X, y, groups, tolerance=0.05):
    return FairnessTrainingAnalyzer(
        X=X,
        y=y,
        sensitive_attr=groups,
        fairness_constraint="equalized_odds",
        tolerance=tolerance,
    )


# ===========================================================================
# The producer's own third state, and the one place it is read
# ===========================================================================


class TestTheThirdStateIsReadable:
    def test_the_producer_still_records_could_not_check(self):
        """The state was never missing; it was DISCARDED downstream."""
        _, y, groups, y_pred = _unmeasurable()
        cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y, groups)
        assert cv.details["insufficient_data"] is True
        assert math.isnan(cv.overall_violation)
        # fail-closed for the reduction, and NOT a measured breach
        assert cv.is_satisfied is False
        assert cv.could_not_evaluate is True
        assert cv.to_dict()["could_not_evaluate"] is True

    def test_reported_pair_refuses_to_invent_a_verdict(self):
        _, y, groups, y_pred = _unmeasurable()
        cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y, groups)
        assert _reported_constraint(cv) == (None, None)

    def test_control_a_measured_breach_keeps_its_number_and_its_verdict(self):
        """OVER-CORRECTION CONTROL. The measured path is untouched."""
        _, y, groups, y_pred = _measurable_breach()
        cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y, groups)
        assert cv.could_not_evaluate is False
        assert cv.to_dict()["could_not_evaluate"] is False
        magnitude, verdict = _reported_constraint(cv)
        assert magnitude == pytest.approx(0.6, abs=1e-9)
        assert verdict is False

    def test_control_a_measured_pass_is_still_a_pass(self):
        """OVER-CORRECTION CONTROL, the other direction."""
        _, y, groups, y_pred = _measurable_breach()
        cv = EqualizedOddsConstraint(tolerance=0.95).compute_violation(y_pred, y, groups)
        magnitude, verdict = _reported_constraint(cv)
        assert magnitude == pytest.approx(0.6, abs=1e-9)
        assert verdict is True

    def test_a_null_flag_is_not_read_as_measured(self):
        """`.get(key, False)` returns the STORED value when the key is present
        holding None, so a null flag would have read as measured. The NaN
        fallback answers absence and null alike."""
        _, y, groups, y_pred = _unmeasurable()
        cv = EqualizedOddsConstraint(tolerance=0.05).compute_violation(y_pred, y, groups)
        cv.details["insufficient_data"] = None
        assert cv.could_not_evaluate is True
        del cv.details["insufficient_data"]
        assert cv.could_not_evaluate is True

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, True),  # a measured zero disparity IS a measurement
            (0.31, True),
            (None, False),
            (float("nan"), False),
            (True, False),  # a bool is not a magnitude
        ],
    )
    def test_measured_knows_a_number_from_an_absence(self, value, expected):
        assert _measured(value) is expected


# ===========================================================================
# Consumer 1: FairnessTrainingAnalyzer.evaluate_baseline
# ===========================================================================


class TestEvaluateBaseline:
    def test_an_unevaluable_baseline_is_not_a_failed_baseline(self):
        X, y, groups, y_pred = _unmeasurable()
        mc = _analyzer(X, y, groups).evaluate_baseline(y_pred=y_pred)
        assert mc.constraint_satisfied is None, "None, not the fail-closed False"
        assert mc.fairness_violation is None, "None, not NaN and not 0.0"
        assert mc.parameters["constraint_evaluated"] is False
        # the accuracy really was measured and still is
        assert mc.accuracy == pytest.approx(float(np.mean(y_pred == y)))

    def test_control_a_measured_baseline_is_unchanged(self):
        """OVER-CORRECTION CONTROL."""
        X, y, groups, y_pred = _measurable_breach()
        mc = _analyzer(X, y, groups).evaluate_baseline(y_pred=y_pred)
        assert mc.constraint_satisfied is False
        assert mc.fairness_violation == pytest.approx(0.6, abs=1e-9)
        assert mc.parameters["constraint_evaluated"] is True


# ===========================================================================
# Consumer 2: the FairClassifier wrappers (all three fit paths)
# ===========================================================================


class TestFairClassifierResult:
    @pytest.mark.parametrize("method", ["reductions", "threshold", "grid_search"])
    def test_no_fit_path_reports_an_unevaluable_constraint_as_a_breach(self, method):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _unmeasurable()
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=200),
            fairness_constraint="equalized_odds",
            tolerance=0.05,
            method=method,
            max_iterations=3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=groups)
        result = clf.fairness_result_
        assert result.constraint_satisfied is None
        assert result.fairness_violation is None
        assert result.fairness_metrics["insufficient_data"] is True
        assert result.to_dict()["constraint_satisfied"] is None

    def test_control_a_measured_fit_still_reports_numbers(self):
        """OVER-CORRECTION CONTROL: on data where every rate is defined, the
        result carries a real float and a real bool."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _measurable_breach()
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=200),
            fairness_constraint="equalized_odds",
            tolerance=0.05,
            method="threshold",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=groups)
        result = clf.fairness_result_
        assert isinstance(result.constraint_satisfied, bool)
        assert isinstance(result.fairness_violation, float)
        assert math.isfinite(result.fairness_violation)
        assert result.fairness_metrics["insufficient_data"] is False

    def test_score_does_not_return_nan_in_silence(self):
        """A NaN score cannot be mistaken for a good score, but it CAN be
        mistaken for a measured one when nothing says otherwise."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _unmeasurable()
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=200),
            fairness_constraint="equalized_odds",
            tolerance=0.05,
            method="threshold",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=groups)
        with pytest.warns(UserWarning, match="could not be evaluated"):
            score = clf.score(X, y, sensitive_attr=groups, metric="fairness")
        assert math.isnan(score), "NaN, never 0.0, which is the BEST fairness score here"

    def test_control_a_measured_score_is_a_number_and_warns_about_nothing(self):
        """OVER-CORRECTION CONTROL."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _measurable_breach()
        clf = FairClassifier(
            base_estimator=LogisticRegression(max_iter=200),
            fairness_constraint="equalized_odds",
            tolerance=0.05,
            method="threshold",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X, y, sensitive_attr=groups)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = clf.score(X, y, sensitive_attr=groups, metric="fairness")
        assert math.isfinite(score)
        assert not [w for w in caught if "could not be evaluated" in str(w.message)]


# ===========================================================================
# Consumer 3: the report's own prose
# ===========================================================================


class TestReportSummary:
    def _built(self, X, y, groups, y_pred, comparisons):
        an = _analyzer(X, y, groups)
        baseline = an.evaluate_baseline(y_pred=y_pred)
        from vfairness.in_processing.analyzer import FairnessTrainingReport

        return FairnessTrainingReport(
            timestamp="2026-09-10T00:00:00",
            data_info={"n_samples": len(y), "n_groups": 3},
            task_type="classification",
            baseline_metrics={
                "accuracy": baseline.accuracy,
                "fairness_violation": baseline.fairness_violation,
                "constraint_satisfied": baseline.constraint_satisfied,
            },
            fairness_analysis={},
            method_comparisons=comparisons,
            recommendation=an.generate_recommendation(comparisons),
            tradeoff_analysis={},
            critical_issues=[],
            action_items=[],
        )

    def test_the_baseline_lines_say_not_measured_rather_than_zero_or_false(self):
        X, y, groups, y_pred = _unmeasurable()
        text = self._built(X, y, groups, y_pred, []).summary()
        assert "Fairness Violation: not measured (the constraint could not be evaluated)" in text
        assert "Constraint Satisfied: not assessed (the constraint could not be evaluated)" in text
        assert "Fairness Violation: 0.0000" not in text
        assert "Constraint Satisfied: False" not in text
        assert "nan" not in text.lower()

    def test_an_ungraded_method_row_gets_the_third_glyph_and_a_count(self):
        X, y, groups, y_pred = _unmeasurable()
        comps = [MethodComparison("Reductions", 0.84, None, None)]
        text = self._built(X, y, groups, y_pred, comps).summary()
        assert "[?]" in text
        assert "[✗]" not in text
        assert "1 of 1 method(s) reported no constraint result" in text

    def test_control_measured_rows_print_exactly_as_before(self):
        """OVER-CORRECTION CONTROL: a measured breach still reads
        `Violation=0.3100 [✗]`, and a measured pass still reads `[✓]`."""
        X, y, groups, y_pred = _measurable_breach()
        comps = [
            MethodComparison("Reductions", 0.84, 0.31, False),
            MethodComparison("Threshold", 0.80, 0.02, True),
        ]
        text = self._built(X, y, groups, y_pred, comps).summary()
        assert "Reductions: Acc=0.8400, Violation=0.3100 [✗]" in text
        assert "Threshold: Acc=0.8000, Violation=0.0200 [✓]" in text
        assert "[?]" not in text
        assert "reported no constraint result" not in text
        assert "Fairness Violation: 0.6000" in text
        assert "Constraint Satisfied: False" in text

    def test_control_a_measured_zero_violation_is_still_printed_as_a_number(self):
        """OVER-CORRECTION CONTROL, the one a careless fix breaks: 0.0 is a
        MEASUREMENT (perfect parity), not a missing value."""
        X, y, groups, y_pred = _measurable_breach()
        comps = [MethodComparison("Reductions", 0.84, 0.0, True)]
        text = self._built(X, y, groups, y_pred, comps).summary()
        assert "Violation=0.0000 [✓]" in text
        assert "not measured" not in text


# ===========================================================================
# Consumer 4: the recommendation, which RANKED unmeasured methods
# ===========================================================================


class TestRecommendation:
    def test_no_winner_is_named_when_nothing_was_evaluated(self):
        X, y, groups, _ = _unmeasurable()
        comps = [
            MethodComparison("Reductions", 0.84, None, None),
            MethodComparison("Threshold", 0.80, None, None),
        ]
        rec = _analyzer(X, y, groups).generate_recommendation(comps)
        assert rec.recommended_method == "N/A"
        assert "none was ranked" in rec.rationale
        assert "nan" not in rec.rationale.lower()
        assert rec.alternative_methods == []
        assert rec.expected_tradeoff == {}

    def test_a_mixed_run_ranks_only_the_measured_methods_and_says_so(self):
        X, y, groups, _ = _unmeasurable()
        comps = [
            MethodComparison("Ungraded", 0.99, None, None),
            MethodComparison("Reductions", 0.84, 0.31, False),
            MethodComparison("Threshold", 0.80, 0.12, False),
        ]
        rec = _analyzer(X, y, groups).generate_recommendation(comps)
        assert rec.recommended_method == "Threshold", "the smallest MEASURED violation"
        assert "Ungraded" not in rec.alternative_methods
        assert "1 of 3 method(s) reported no constraint result" in rec.rationale

    def test_control_an_all_measured_run_is_unchanged(self):
        """OVER-CORRECTION CONTROL: same winner, same alternatives, same
        rationale text as before the fix, with no ungraded clause appended."""
        X, y, groups, _ = _measurable_breach()
        comps = [
            MethodComparison("m1", 0.80, 0.02, True),
            MethodComparison("m2", 0.90, 0.20, False),
            MethodComparison("m3", 0.85, 0.03, True),
        ]
        rec = _analyzer(X, y, groups).generate_recommendation(comps)
        assert rec.recommended_method == "m3"
        assert rec.alternative_methods == ["m1"]
        assert rec.rationale == "Achieves fairness constraint with best accuracy (0.850)"
        assert rec.priority == "high"

    def test_control_the_minimising_branch_is_unchanged(self):
        """OVER-CORRECTION CONTROL for the second branch."""
        X, y, groups, _ = _measurable_breach()
        comps = [
            MethodComparison("m1", 0.80, 0.20, False),
            MethodComparison("m2", 0.90, 0.09, False),
        ]
        rec = _analyzer(X, y, groups).generate_recommendation(comps)
        assert rec.recommended_method == "m2"
        assert rec.rationale == "Minimizes constraint violation (0.090)"
        assert rec.priority == "medium"


# ===========================================================================
# Consumer 5: critical issues and action items
# ===========================================================================


class TestCriticalIssuesAndActions:
    def test_an_unevaluated_run_is_not_an_infeasible_constraint(self):
        X, y, groups, y_pred = _unmeasurable()
        an = _analyzer(X, y, groups)
        baseline = an.evaluate_baseline(y_pred=y_pred)
        metrics = {
            "accuracy": baseline.accuracy,
            "fairness_violation": baseline.fairness_violation,
            "constraint_satisfied": baseline.constraint_satisfied,
        }
        comps = [MethodComparison("Reductions", 0.84, None, None)]
        issues = an._identify_critical_issues(metrics, {}, comps)
        types = [i["type"] for i in issues]
        assert "Infeasible Constraint" not in types, (
            "nothing achieved the constraint because nothing was measured"
        )
        assert types.count("Constraint Not Evaluated") == 2, "the baseline and the methods"
        assert {i["severity"] for i in issues} == {"unmeasured"}
        assert not [i for i in issues if i["severity"] == "high"]

    def test_the_action_list_does_not_tell_the_reader_to_use_n_a(self):
        X, y, groups, y_pred = _unmeasurable()
        an = _analyzer(X, y, groups)
        baseline = an.evaluate_baseline(y_pred=y_pred)
        metrics = {
            "accuracy": baseline.accuracy,
            "fairness_violation": baseline.fairness_violation,
            "constraint_satisfied": baseline.constraint_satisfied,
        }
        rec = an.generate_recommendation([MethodComparison("Reductions", 0.84, None, None)])
        items = an._generate_action_items(metrics, {}, rec)
        assert not [i for i in items if "Use N/A" in i]
        assert [i for i in items if i.startswith("MEASURE:")]

    def test_control_a_measured_infeasible_run_still_raises_it_at_high(self):
        """OVER-CORRECTION CONTROL: a real, measured, universally failed
        constraint is still the HIGH finding it was."""
        X, y, groups, y_pred = _measurable_breach()
        an = _analyzer(X, y, groups)
        baseline = an.evaluate_baseline(y_pred=y_pred)
        metrics = {
            "accuracy": baseline.accuracy,
            "fairness_violation": baseline.fairness_violation,
            "constraint_satisfied": baseline.constraint_satisfied,
        }
        comps = [MethodComparison("Reductions", 0.84, 0.31, False)]
        issues = an._identify_critical_issues(metrics, {}, comps)
        types = [i["type"] for i in issues]
        assert "Infeasible Constraint" in types
        assert "Constraint Not Evaluated" not in types
        infeasible = [i for i in issues if i["type"] == "Infeasible Constraint"][0]
        assert infeasible["severity"] == "high"
        # and the measured baseline violation of 0.60 is still the HIGH finding
        assert "High Violation" in types
        assert [i for i in issues if i["type"] == "High Violation"][0]["severity"] == "high"

    def test_control_the_measured_action_items_are_unchanged(self):
        """OVER-CORRECTION CONTROL."""
        X, y, groups, y_pred = _measurable_breach()
        an = _analyzer(X, y, groups)
        baseline = an.evaluate_baseline(y_pred=y_pred)
        metrics = {
            "accuracy": baseline.accuracy,
            "fairness_violation": baseline.fairness_violation,
            "constraint_satisfied": baseline.constraint_satisfied,
        }
        rec = an.generate_recommendation([MethodComparison("Reductions", 0.84, 0.31, False)])
        items = an._generate_action_items(metrics, {}, rec)
        assert "IMPLEMENT: Use Reductions training method" in items
        assert "VALIDATE: Test fairness metrics on held-out data" in items
        assert not [i for i in items if i.startswith("MEASURE:")]


# ===========================================================================
# Consumer 6: the trade-off sweep, where unmeasured rows were RANKED OPTIMAL
# ===========================================================================


class TestTradeoffSweep:
    def test_an_unevaluable_configuration_is_not_pareto_optimal(self):
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _unmeasurable()
        an = _analyzer(X, y, groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ta = an.analyze_tradeoffs(LogisticRegression(max_iter=200), lambda_values=[0.0, 1.0])
        assert len(ta["all_results"]) == 2
        assert all(r["violation"] is None for r in ta["all_results"])
        assert all(r["satisfied"] is None for r in ta["all_results"])
        assert ta["pareto_frontier"] == [], "nothing can be optimal on an axis nobody measured"
        assert ta["best_fair"] is None
        assert ta["n_not_evaluated"] == 2
        # the accuracy axis WAS measured, so best_accurate is still reported
        assert ta["best_accurate"] is not None

    def test_control_a_measured_sweep_still_has_a_frontier(self):
        """OVER-CORRECTION CONTROL."""
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LogisticRegression

        X, y, groups, _ = _measurable_breach()
        an = _analyzer(X, y, groups, tolerance=0.9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ta = an.analyze_tradeoffs(LogisticRegression(max_iter=200), lambda_values=[0.0, 1.0])
        assert all(isinstance(r["violation"], float) for r in ta["all_results"])
        assert all(isinstance(r["satisfied"], bool) for r in ta["all_results"])
        assert ta["pareto_frontier"], "a measured sweep still ranks"
        assert ta["n_not_evaluated"] == 0


# ===========================================================================
# The drawn surface. This is the one the defect is named for.
# ===========================================================================


def _chips(svg):
    return re.findall(r">(PASS|FAIL|NOT CHECKED)<", svg)


def _desc(svg):
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    return match.group(1).strip() if match else ""


class TestTheChart:
    def test_an_unevaluable_constraint_is_not_drawn_as_a_failing_bar(self):
        svg = method_comparison_to_svg([MethodComparison("Reductions", 0.84, None, None)])
        assert "FAIL" not in svg, "a red FAIL chip for a constraint nobody evaluated"
        assert "nan" not in svg.lower()
        desc = _desc(svg)
        assert "COULD NOT CHECK" in desc.upper()
        assert "severity: HIGH" not in desc

    def test_a_nan_violation_is_no_longer_drawn_as_a_measurement(self):
        """WHY None AND NOT NaN, and what changed on 2026-09-30.

        This test used to assert the DEFECT, deliberately: the renderer's gates
        were `is not None`, NaN passed them, and the chart drew the literal
        string "nan" as the violation magnitude. It was kept as the evidence
        that the third state had to be None, because NaN could not be relied on
        to be refused.

        The rendering wave closed that door at the boundary itself, so the
        argument no longer needs a live fabrication to stand on. The SUBJECT is
        unchanged and is what is pinned below: a NaN magnitude must not be
        published as a number. Only the mechanism moved, from "None is the only
        shape the gates refuse" to "the boundary refuses every non-finite".

        The FAIL chip is correct and stays. `constraint_satisfied=False` is a
        claim the caller made, and the chart is faithful to it; what was wrong
        was drawing an unmeasured MAGNITUDE beside it as though measured.
        """
        svg = method_comparison_to_svg([MethodComparison("Reductions", 0.84, float("nan"), False)])
        assert "FAIL" in svg, "satisfied=False is the caller's claim and must still be drawn"
        assert "nan" not in svg.lower(), "the literal NaN reached the drawn surface"
        assert "not measured" in svg, (
            "an unmeasured magnitude must SAY SO where the number would be, not vanish: "
            "a FAIL chip with no number is indistinguishable from a measured breach"
        )

    def test_control_a_measured_breach_draws_exactly_as_it_did(self):
        """OVER-CORRECTION CONTROL: the red FAIL chip and the number are what a
        REAL violation is supposed to look like, and they are untouched."""
        svg = method_comparison_to_svg([MethodComparison("Reductions", 0.84, 0.31, False)])
        assert _chips(svg) == ["FAIL"]
        assert "0.310" in svg
        assert "NOT CHECKED" not in svg
        assert "1 of 1 method(s) satisfy" not in _desc(svg)

    def test_control_a_measured_pass_draws_as_a_pass(self):
        """OVER-CORRECTION CONTROL."""
        svg = method_comparison_to_svg([MethodComparison("Threshold", 0.80, 0.02, True)])
        assert _chips(svg) == ["PASS"]
        assert "0.020" in svg

    def test_a_mixed_table_counts_only_what_it_graded(self):
        svg = method_comparison_to_svg(
            [
                MethodComparison("Reductions", 0.84, 0.31, False),
                MethodComparison("Threshold", 0.80, 0.02, True),
                MethodComparison("Ungraded", 0.99, None, None),
            ]
        )
        chips = _chips(svg)
        assert chips.count("NOT CHECKED") == 1
        assert chips.count("FAIL") == 1
        assert chips.count("PASS") == 1
        desc = _desc(svg)
        assert "1 of 2 method(s) satisfy the fairness constraint" in desc, (
            "the ungraded row is out of BOTH the numerator and the denominator"
        )

    def test_the_whole_pipeline_end_to_end(self):
        """The defect as it actually reached a reader: analyser to canvas."""
        X, y, groups, y_pred = _unmeasurable()
        mc = _analyzer(X, y, groups).evaluate_baseline(y_pred=y_pred)
        svg = method_comparison_to_svg([mc])
        assert _chips(svg) == ["NOT CHECKED"]
        assert "nan" not in svg.lower()

    def test_control_the_whole_pipeline_on_measurable_data(self):
        """OVER-CORRECTION CONTROL for the end-to-end path."""
        X, y, groups, y_pred = _measurable_breach()
        mc = _analyzer(X, y, groups).evaluate_baseline(y_pred=y_pred)
        svg = method_comparison_to_svg([mc])
        assert _chips(svg) == ["FAIL"]
        assert "0.600" in svg
