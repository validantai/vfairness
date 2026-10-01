"""Precedence pins for :mod:`vfairness.evaluation.vfairness_metrics._metric_direction`.

The cali-BRATIO-n incident (CLAUDE.md, reconciled in 47f1e09f8) was a substring
direction test: ``"ratio" in name`` matched "cali[bratio]n_difference", so a
large miscalibration was graded against the four-fifths rule and read as a PASS.

The fix for that incident then reintroduced the SAME bug class with a different
token. ``is_ratio_metric`` ended with ``"disparate impact" in name or
"disparate_impact" in name``, evaluated before any lower-is-better rule, so:

    disparate_impact_ratio       -> HIGHER_IS_BETTER  (correct)
    disparate_impact_difference  -> HIGHER_IS_BETTER  (WRONG: a violation
                                    magnitude of 0.45 against a 0.10 bound was
                                    graded "at or above the minimum" = PASS)
    disparate_impact_gap         -> HIGHER_IS_BETTER  (WRONG, same shape)

This module pins the whole precedence table so the class cannot come back a
third time, and carries a source guard that fails if a bare substring direction
test is reintroduced into the module.
"""

import pathlib

import pytest

from vfairness.evaluation.vfairness_metrics._metric_direction import (
    HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS,
    MetricDirection,
    ThresholdOutcome,
    check_threshold,
    improvement_amount,
    is_ratio_metric,
    metric_direction,
    relax_threshold,
)

HIGHER = MetricDirection.HIGHER_IS_BETTER
LOWER = MetricDirection.LOWER_IS_BETTER
UNKNOWN = MetricDirection.UNKNOWN


# === the full precedence table ================================================
@pytest.mark.parametrize(
    "name,expected",
    [
        # The ratio family, by EXACT suffix.
        ("disparate_impact_ratio", HIGHER),
        ("disparate_impact_ratio_with_ci", HIGHER),
        ("demographic_parity_ratio", HIGHER),
        ("exposure_parity_ratio", HIGHER),
        # The bare disparate-impact name, and its human label forms, matched
        # exactly after normalisation (never as a substring).
        ("disparate_impact", HIGHER),
        ("Disparate Impact", HIGHER),
        ("DISPARATE IMPACT", HIGHER),
        ("disparate-impact", HIGHER),
        ("  disparate_impact  ", HIGHER),
        # THE REINTRODUCED DEFECT: a violation magnitude that merely CONTAINS
        # the disparate-impact token is lower-is-better, not a ratio.
        ("disparate_impact_difference", LOWER),
        ("disparate_impact_gap", LOWER),
        ("disparate_impact_diff", LOWER),
        ("disparate_impact_disparity", LOWER),
        ("disparate_impact_deviation", LOWER),
        ("disparate impact difference", LOWER),
        # THE ORIGINAL cali-BRATIO-n CASE: "ratio" is a substring of
        # "cali[bratio]n", and these are all violation magnitudes.
        ("calibration_difference", LOWER),
        ("calibration", LOWER),
        ("multicalibration", LOWER),
        ("multicalibration_with_ci", LOWER),
        ("integrated_calibration_index", LOWER),
        ("expected_calibration_error", LOWER),
        ("maximum_calibration_error", LOWER),
        ("calibration_disparity", LOWER),
        # Explicit higher-is-better name with no "_ratio" suffix.
        ("worst_group_accuracy", HIGHER),
        # Max between-group gaps that carry a "_parity" name.
        ("auroc_parity", LOWER),
        ("net_benefit_parity", LOWER),
        ("demographic_parity", LOWER),
        ("demographic_parity_difference", LOWER),
        ("equalized_odds_difference", LOWER),
        ("bounded_group_loss", LOWER),
        ("pricing_disparity", LOWER),
        # Unresolvable names fail closed. None of these may be guessed.
        ("", UNKNOWN),
        ("   ", UNKNOWN),
        ("calibration_slope", UNKNOWN),
        ("proxy_feature_score", UNKNOWN),
        ("shiny_new_metric_2027", UNKNOWN),
        # "ratio" is not the "_ratio" SUFFIX, and a disparate-impact PREFIX on
        # some other quantity is not the disparate-impact metric.
        ("ratio", UNKNOWN),
        ("disparate_impact_gender", UNKNOWN),
        ("disparate", UNKNOWN),
        ("impact", UNKNOWN),
        # Genuinely contradictory name: a ratio suffix AND a violation token.
        # Ambiguous, therefore could-not-check, never a guess in either
        # direction.
        ("calibration_error_ratio", UNKNOWN),
    ],
)
def test_metric_direction_precedence_table(name, expected):
    assert metric_direction(name) is expected, name


@pytest.mark.parametrize(
    "name",
    [
        "calibration",
        "calibration_difference",
        "multicalibration",
        "integrated_calibration_index",
        "disparate_impact_difference",
        "disparate_impact_gap",
        "disparate_impact_gender",
        "ratio",
    ],
)
def test_is_ratio_metric_refuses_substring_matches(name):
    assert is_ratio_metric(name) is False, name


@pytest.mark.parametrize(
    "name",
    [
        "disparate_impact_ratio",
        "disparate_impact_ratio_with_ci",
        "demographic_parity_ratio",
        "disparate_impact",
        "Disparate Impact",
    ],
)
def test_is_ratio_metric_accepts_the_exact_forms(name):
    assert is_ratio_metric(name) is True, name


def test_direction_sets_are_disjoint():
    """A name in both sets would make the verdict depend on rule order."""
    overlap = HIGHER_IS_BETTER_METRICS & LOWER_IS_BETTER_METRICS
    assert not overlap, f"a metric is claimed in both directions: {sorted(overlap)}"


# === the negative cases: the gate must REFUSE the bad input ===================
class TestFalseCertificateIsRefused:
    """Each of these graded a violation as a pass before the precedence fix."""

    def test_disparate_impact_difference_violation_fails(self):
        outcome, message = check_threshold("disparate_impact_difference", 0.45, 0.10)
        assert outcome is ThresholdOutcome.FAIL, message
        assert "exceeds threshold" in message

    def test_disparate_impact_gap_violation_fails(self):
        outcome, message = check_threshold("disparate_impact_gap", 0.90, 0.10)
        assert outcome is ThresholdOutcome.FAIL, message

    def test_calibration_difference_violation_fails(self):
        outcome, message = check_threshold("calibration_difference", 0.55, 0.05)
        assert outcome is ThresholdOutcome.FAIL, message

    def test_disparate_impact_difference_within_bound_passes(self):
        outcome, _ = check_threshold("disparate_impact_difference", 0.02, 0.10)
        assert outcome is ThresholdOutcome.PASS

    def test_disparate_impact_ratio_keeps_the_four_fifths_direction(self):
        assert check_threshold("disparate_impact_ratio", 0.0, 0.8)[0] is ThresholdOutcome.FAIL
        assert check_threshold("disparate_impact_ratio", 1.0, 0.8)[0] is ThresholdOutcome.PASS

    def test_unknown_metric_could_not_check_never_passes(self):
        outcome, message = check_threshold("shiny_new_metric_2027", 0.01, 0.10)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK
        assert "no known better-direction" in message

    def test_relax_moves_a_disparate_impact_difference_the_permissive_way(self):
        # Lower-is-better: relaxing means allowing a LARGER gap. Dividing would
        # make the "relaxed" intersectional bound stricter than the base bound.
        assert relax_threshold("disparate_impact_difference", 0.10, 2.0) == pytest.approx(0.20)
        # Higher-is-better: relaxing means accepting a SMALLER ratio.
        assert relax_threshold("disparate_impact_ratio", 0.80, 2.0) == pytest.approx(0.40)

    def test_improvement_of_a_disparate_impact_difference_is_a_shrinking_gap(self):
        # 0.30 -> 0.10 is a 0.20 improvement for a violation magnitude.
        assert improvement_amount("disparate_impact_difference", 0.10, 0.30) == pytest.approx(0.20)
        # 0.60 -> 0.90 is a 0.30 improvement for a parity ratio.
        assert improvement_amount("disparate_impact_ratio", 0.90, 0.60) == pytest.approx(0.30)
        # Unknown direction yields None so the caller decides, never a guess.
        assert improvement_amount("shiny_new_metric_2027", 0.10, 0.30) is None


# === a bound that cannot be breached grades nothing ===========================
class TestDegenerateBoundIsNotAPass:
    """Wave 4: a threshold no value can breach never graded anything.

    ``check_threshold`` compared the value against the bound and reported PASS
    when the comparison did not trip, without ever asking whether the bound
    COULD trip. On a higher-is-better metric the threshold is a required
    MINIMUM, and the ratio family is non-negative, so a minimum of 0.0 is
    satisfied by every possible value including the worst one.

    Executed before the fix, on this repo:

        check_threshold("disparate_impact_ratio", 0.00, 0.0)
            -> (ThresholdOutcome.PASS, "")
        ModelFairnessGate(thresholds={"disparate_impact_ratio": 0.0})
            .evaluate_from_metrics({"disparate_impact_ratio": 0.00})
            -> approved=True, GateStatus.APPROVED,
               create_github_check(...)["conclusion"] == "success"

    A protected group that is NEVER selected was approved for deployment and
    stamped green on the pull request. That is the same lie as the substring
    direction test in a third costume: a metric nobody actually graded is
    reported as a metric that passed.

    The renderer built this guard locally in
    ``rendering/adapters_fairness._metric_state`` and left a comment saying the
    hole was still open upstream. It is closed here now, so every caller of the
    shared check inherits it.
    """

    def test_a_zero_minimum_on_a_ratio_is_could_not_check_not_a_pass(self):
        outcome, message = check_threshold("disparate_impact_ratio", 0.00, 0.0)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK, message
        assert "no value can fall below" in message

    def test_the_worst_and_the_best_ratio_are_graded_alike_under_a_zero_floor(self):
        """The tell: a real bound separates them, a degenerate one cannot."""
        worst = check_threshold("disparate_impact_ratio", 0.00, 0.0)[0]
        best = check_threshold("disparate_impact_ratio", 1.00, 0.0)[0]
        assert worst is best is ThresholdOutcome.COULD_NOT_CHECK

    def test_a_negative_minimum_is_degenerate_too(self):
        outcome, _ = check_threshold("demographic_parity_ratio", 0.49, -1.0)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK

    def test_the_named_inflation_case_is_no_longer_a_passed_metric(self):
        """demographic_parity_ratio 0.49 under a 0.0 floor was a 'passed' metric.

        It was written into ``assessment['passed_metrics']`` and inflated the
        fairness score with a grade nobody had computed.
        """
        outcome, _ = check_threshold("demographic_parity_ratio", 0.49, 0.0)
        assert outcome is not ThresholdOutcome.PASS

    def test_zero_tolerance_on_a_difference_metric_still_grades(self):
        """The MIRROR case is not degenerate and must keep working.

        0.0 on a lower-is-better metric is a real zero-tolerance policy:
        abs(value) can exceed it, so the bound does grade.
        """
        assert check_threshold("demographic_parity_difference", 0.02, 0.0)[0] is (
            ThresholdOutcome.FAIL
        )
        assert check_threshold("demographic_parity_difference", 0.00, 0.0)[0] is (
            ThresholdOutcome.PASS
        )

    def test_a_negative_maximum_on_a_difference_metric_grades_nothing_either(self):
        """No magnitude is below a negative maximum, so nothing was graded.

        Fail-closed in both readings, but it is could-not-check, not a measured
        failure: the value was never compared against a bound that could hold.
        """
        outcome, message = check_threshold("demographic_parity_difference", 0.02, -0.10)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK, message

    def test_an_unknown_metric_keeps_its_own_reason(self):
        """The degenerate guard must not swallow the unknown-direction reason."""
        outcome, message = check_threshold("shiny_new_metric_2027", 0.01, 0.0)
        assert outcome is ThresholdOutcome.COULD_NOT_CHECK
        assert "no known better-direction" in message

    def test_a_real_floor_on_a_ratio_still_grades_both_ways(self):
        """Negative control: the guard must not disarm the four-fifths rule."""
        assert check_threshold("disparate_impact_ratio", 0.00, 0.80)[0] is ThresholdOutcome.FAIL
        assert check_threshold("disparate_impact_ratio", 1.00, 0.80)[0] is ThresholdOutcome.PASS


# === source guard: no bare substring direction test may return ================
def _scanner():
    """The ONE executable-source scanner, from tests/test_audit_final_release.py.

    Deliberately imported rather than copied. A second copy of the tokenizer
    could be weakened on its own (a naive version strips every STRING token,
    which makes the pin permanently green: that has already happened here once,
    which is why that module carries a positive control), and then two pins
    would disagree about what "the scanner" does.
    """
    try:
        from tests.test_audit_final_release import _executable_source, _normalize_tokens
    except ImportError:  # pragma: no cover - depends on how pytest set sys.path
        import importlib.util

        path = pathlib.Path(__file__).with_name("test_audit_final_release.py")
        spec = importlib.util.spec_from_file_location("_release_pin_scanner", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _executable_source = mod._executable_source
        _normalize_tokens = mod._normalize_tokens
    return _executable_source, _normalize_tokens


# Every substring direction test this module has ever been tempted to write, in
# the variable names it actually uses. A hit means a family token is being
# matched anywhere inside a name again, which is the cali-BRATIO-n bug class.
_DIRECTION_TOKENS = (
    "ratio",
    "disparate",
    "disparate impact",
    "disparate_impact",
    "parity",
    "gap",
    "difference",
    "diff",
    "disparity",
    "calibration",
)
_NAME_VARS = ("name", "n", "key", "metric_name", "metric", "metric_key")


def _substring_markers():
    for token in _DIRECTION_TOKENS:
        for var in _NAME_VARS:
            for quote in ('"', "'"):
                yield f"{quote}{token}{quote} in {var}"
                yield f"{quote}{token}{quote} not in {var}"


def _module_file():
    from vfairness.evaluation.vfairness_metrics import _metric_direction as M

    return pathlib.Path(M.__file__)


def test_no_substring_direction_test_in_this_module():
    """Executable code only, so the incident may still be EXPLAINED in prose.

    The module docstring quotes the bad pattern while explaining why the module
    refuses it. A guard whose only green states are "the bug is absent" or "the
    explanation is deleted" pressures the next author to delete the
    explanation, and that is how the cali-BRATIO-n understanding was lost the
    first time.
    """
    executable_source, normalize_tokens = _scanner()
    code = executable_source(_module_file())
    hits = [m for m in _substring_markers() if normalize_tokens(m) in code]
    assert not hits, (
        "a bare substring direction test is back in _metric_direction.py: "
        f"{hits}. Classify by exact name or exact suffix instead."
    )


def test_the_guard_still_fires_on_a_real_code_hit(tmp_path):
    """Positive control: the guard above must not be able to go silently green.

    The offender file contains the bug in executable form AND the same text in
    a docstring and a comment. Exactly the executable one must be reported.
    """
    executable_source, normalize_tokens = _scanner()

    offender = tmp_path / "offender.py"
    offender.write_text(
        '"""Docstring mentioning \'"disparate_impact" in name\' as prose only."""\n'
        '# comment mentioning "disparate_impact" in name\n'
        "def f(name):\n"
        '    return "disparate_impact" in name\n'
    )
    code = executable_source(offender)
    hits = [m for m in _substring_markers() if normalize_tokens(m) in code]
    assert hits, (
        "the guard no longer sees the bug in executable code, so it would be "
        f"permanently green. Tokenized code was: {code!r}"
    )

    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        '"""Explains why a naive \'"disparate_impact" in name\' test is wrong."""\n'
        "def f(name):\n"
        '    return name == "disparate_impact" or name.endswith("_ratio")\n'
    )
    innocent_code = executable_source(innocent)
    assert not [m for m in _substring_markers() if normalize_tokens(m) in innocent_code], (
        "the guard fires on prose; documenting the incident would fail CI"
    )
