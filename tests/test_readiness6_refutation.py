"""READINESS-6: the causal refutation suite could not fail at fairness scale.

`run_refutation_suite` runs four DoWhy refuters and reports
``robustness_score``, the fraction that passed. A refuter "passes" when the
refuted estimate stays near its null, within a tolerance.

Both tolerances carry an ABSOLUTE floor: 0.05 for the placebo and dummy-outcome
family, 0.1 for the near-original family. Fairness effects live at 0.01 to 0.05.
So the floor routinely exceeded the entire effect under test, and a tolerance
wider than the effect admits every outcome the refuter can produce.

Measured on this repo before the fix, at an observed effect of 0.03::

    placebo returns the ENTIRE original effect (0.03)  -> PASS
    data subset makes the effect VANISH (0.00)         -> PASS
    data subset REVERSES the sign (-0.03)              -> PASS

Each of those is the most complete refutation its test can produce, and
``robustness_score`` reported 1.0 for a causal estimate that nothing had tested.
At an observed effect of 0.30 all three correctly FAIL, which is why nobody saw
it: the suite works on large effects and stops working on exactly the small ones
this library exists to measure.

It is the degenerate-bound defect that `check_threshold` already guards for
thresholds, in a third costume: a bound that cannot be breached grades nothing.
"""

from __future__ import annotations

import pytest

from vfairness.operations.causal import refute as R
from vfairness.operations.causal.refute import _passed, _tolerance

# The four refuters, and the single most extreme refutation each can produce.
# "The strongest possible evidence against the estimate", per family.
WORST_CASE = [
    ("Placebo treatment", lambda observed: observed, "the placebo reproduced the ENTIRE effect"),
    ("Dummy outcome", lambda observed: observed, "the dummy outcome reproduced the ENTIRE effect"),
    ("Data subset", lambda observed: 0.0, "the effect VANISHED on a subset"),
    ("Random common cause", lambda observed: -observed, "the effect REVERSED SIGN"),
]

# The range fairness disparities actually occupy.
FAIRNESS_SCALE = [0.04, 0.03, 0.02, 0.01, 0.005]
# Effects large enough that the absolute floor is not the binding constraint.
LARGE_SCALE = [0.5, 0.3, 0.2]


@pytest.mark.parametrize("test_name,worst,description", WORST_CASE)
@pytest.mark.parametrize("observed", FAIRNESS_SCALE)
def test_a_refuter_that_cannot_fail_reports_could_not_check(
    test_name, worst, description, observed
):
    """Not a PASS. Three states, never two."""
    verdict = _passed(test_name, observed, worst(observed))
    assert verdict is None, (
        f"{test_name} at observed={observed}: {description} and the suite said "
        f"{'PASS' if verdict else 'FAIL'}. Tolerance is "
        f"{_tolerance(test_name, observed)}, wider than the effect."
    )


@pytest.mark.parametrize("test_name,worst,description", WORST_CASE)
@pytest.mark.parametrize("observed", LARGE_SCALE)
def test_the_same_refutations_are_still_refuted_where_the_test_has_room(
    test_name, worst, description, observed
):
    """OVER-CORRECTION CONTROL, and the reason this defect hid for so long.

    On a large effect every one of these is correctly refuted. A fix that turned
    these into could-not-checks would destroy the suite's whole purpose while
    looking more careful.
    """
    verdict = _passed(test_name, observed, worst(observed))
    assert verdict is False, f"{test_name} at observed={observed}: {description} was not refuted"


@pytest.mark.parametrize("observed", LARGE_SCALE)
def test_a_genuine_pass_is_still_a_pass(observed):
    """OVER-CORRECTION CONTROL."""
    assert _passed("Placebo treatment", observed, 0.0) is True
    assert _passed("Dummy outcome", observed, 0.001) is True
    assert _passed("Data subset", observed, observed) is True
    assert _passed("Random common cause", observed, observed * 1.01) is True


def test_the_boundary_grades_rather_than_refusing():
    """At tolerance EXACTLY equal to the effect the test still grades: it can
    detect a total vanishing and nothing weaker. Thin, but real, and calling it
    unmeasurable would discard a genuine refutation. The comparison is strictly
    `tolerance > abs(observed)`, and this pins that the `>` is not a `>=`."""
    # Placebo floor is 0.05, so observed 0.05 is exactly the boundary.
    assert _tolerance("Placebo treatment", 0.05) == 0.05
    assert _passed("Placebo treatment", 0.05, 0.05) is False, "the boundary stopped grading"
    assert _passed("Placebo treatment", 0.05, 0.0) is True


def test_an_unmeasurable_estimate_is_not_a_verdict():
    for bad in (float("nan"), float("inf"), None, True, "0.03"):
        assert _passed("Placebo treatment", bad, 0.0) is None, f"observed={bad!r}"
        assert _passed("Placebo treatment", 0.3, bad) is None, f"refuted={bad!r}"


# ------------------------------------------------------- end to end, no dowhy


class _FakeRefutation:
    def __init__(self, new_effect):
        self.new_effect = new_effect


def _fake_dowhy(observed, refuted_by_method):
    """A CausalModel stand-in. The point is to exercise run_refutation_suite's
    DENOMINATOR, which no unit test of `_passed` can reach."""

    class _Estimate:
        value = observed

    class _FakeCausalModel:
        def __init__(self, **kwargs):
            pass

        def identify_effect(self, **kwargs):
            return object()

        def estimate_effect(self, estimand, method_name=None):
            return _Estimate()

        def refute_estimate(self, estimand, base, method_name=None):
            return _FakeRefutation(refuted_by_method[method_name])

    return _FakeCausalModel


def test_could_not_check_refuters_leave_the_denominator(monkeypatch):
    """THE POINT OF THE WHOLE FIX. `attempted` used to be incremented BEFORE the
    verdict, so a test that could never have failed still counted as evidence.

    Here every refuter reproduces the worst case at a fairness-scale effect, so
    none of them graded anything, and robustness_score must be unavailable
    rather than a perfect 1.0.
    """
    import pandas as pd

    observed = 0.03
    monkeypatch.setattr(
        R,
        "_import_dowhy",
        lambda: _fake_dowhy(
            observed,
            {
                "placebo_treatment_refuter": observed,  # reproduced the whole effect
                "dummy_outcome_refuter": observed,  # reproduced the whole effect
                "data_subset_refuter": 0.0,  # effect vanished
                "random_common_cause": -observed,  # sign reversed
            },
        ),
    )
    data = pd.DataFrame({"t": [0, 1] * 50, "y": [0.0, 1.0] * 50})
    result = R.run_refutation_suite("graph[]", data, "t", "y")

    assert [o.passed for o in result.outcomes] == [None, None, None, None], [
        (o.test_name, o.passed) for o in result.outcomes
    ]
    assert result.robustness_score is None, (
        f"robustness_score is {result.robustness_score!r} over four refuters that "
        f"graded nothing. Before the fix this was 1.0."
    )
    assert any("No refuter could run" in w for w in result.warnings), result.warnings
    for o in result.outcomes:
        assert any("Could not check" in n for n in o.notes), o.notes
        assert any("wider than the effect" in n for n in o.notes), o.notes


def test_a_real_refutation_still_scores(monkeypatch):
    """OVER-CORRECTION CONTROL, end to end. A large effect where three refuters
    hold and one genuinely refutes must score 0.75, not None."""
    import pandas as pd

    observed = 0.30
    monkeypatch.setattr(
        R,
        "_import_dowhy",
        lambda: _fake_dowhy(
            observed,
            {
                "placebo_treatment_refuter": 0.0,  # holds
                "dummy_outcome_refuter": 0.0,  # holds
                "data_subset_refuter": observed,  # holds
                "random_common_cause": 0.0,  # REFUTED: effect vanished
            },
        ),
    )
    data = pd.DataFrame({"t": [0, 1] * 50, "y": [0.0, 1.0] * 50})
    result = R.run_refutation_suite("graph[]", data, "t", "y")

    verdicts = {o.test_name: o.passed for o in result.outcomes}
    assert verdicts == {
        "Placebo treatment": True,
        "Dummy outcome": True,
        "Data subset": True,
        "Random common cause": False,
    }, verdicts
    assert result.robustness_score == pytest.approx(0.75)


def test_a_partly_gradeable_suite_scores_on_what_it_graded(monkeypatch):
    """The mixed case, which is what real data will produce: the denominator is
    the refuters that could grade, and the shortfall is disclosed."""
    import pandas as pd

    observed = 0.06  # placebo tolerance 0.05 grades; subset tolerance 0.1 does not
    monkeypatch.setattr(
        R,
        "_import_dowhy",
        lambda: _fake_dowhy(
            observed,
            {
                "placebo_treatment_refuter": 0.0,
                "dummy_outcome_refuter": 0.0,
                "data_subset_refuter": observed,
                "random_common_cause": observed,
            },
        ),
    )
    data = pd.DataFrame({"t": [0, 1] * 50, "y": [0.0, 1.0] * 50})
    result = R.run_refutation_suite("graph[]", data, "t", "y")

    verdicts = {o.test_name: o.passed for o in result.outcomes}
    assert verdicts["Placebo treatment"] is True
    assert verdicts["Dummy outcome"] is True
    assert verdicts["Data subset"] is None, "the 0.1 floor exceeds a 0.06 effect"
    assert verdicts["Random common cause"] is None
    assert result.robustness_score == pytest.approx(1.0)
    # ... and the reader is told the score rests on two of four, not four.
    assert any("could not run and were" in w for w in result.warnings), result.warnings
