"""Audit 6, wave 3, finding S-02: a requested statistical test was ignored.

``run_disparity_tests(..., test_type=...)`` advertised a choice of five tests
('auto', 'ttest', 'anova', 'chi2', 'mannwhitney') and read the parameter
nowhere in its body. Whatever the caller asked for, the automatic data-driven
selection ran, and the answer came back under the name of the test that
actually ran. Measured before the fix, on one dataframe:

    test_type='mannwhitney'  ->  Independent t-test, t=-9.043721, p=6.74e-18
    test_type='ttest'        ->  Chi-squared test on a binary target
    test_type='not_a_test'   ->  Independent t-test

so a caller who believed they had run a rank-based non-parametric test had run
a parametric one, and would publish that number attributed to the wrong method.

There is no name-keyed test selector in this module: the branch is chosen from
the dtype and the group count, and each branch carries its own effect size and
confidence interval. Routing a caller-named test would mean writing new
statistics (a Mann-Whitney U with a rank-biserial effect size and its own CI,
a policy for a t-test across three or more groups, a binning policy for chi2 on
a continuous column). So the fix REFUSES: only 'auto' is honoured, anything
else raises, and the returned payload carries both what was requested
(``test_type_requested``) and what actually ran (``test_name``).

These tests pin, in order:
  1. the refusal, for each value the old docstring advertised,
  2. the refusal of an unknown value,
  3. that no name the docstring mentions is silently accepted,
  4. over-correction controls: the honoured 'auto' path still returns its
     real, measured statistics, p-values and effect sizes unchanged.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from vfairness.preprocessing.bias_detection.statistical import (
    _SUPPORTED_TEST_TYPES,
    run_disparity_tests,
)

# Every test name the old docstring advertised but the module cannot honour.
UNHONOURABLE_DOCUMENTED_TYPES = ["ttest", "anova", "chi2", "mannwhitney"]

# The two groups are deliberately UNBALANCED (60 vs 45). On equal-sized
# groups Welch's t statistic is algebraically identical to Student's, so a
# balanced fixture cannot tell the two apart on the statistic, and the
# p-value difference lands below pytest.approx's default absolute tolerance.
# Measured: a quiet swap to equal_var=False left a balanced fixture entirely
# green. Unequal group sizes make the statistic itself move.
_NM, _NF = 60, 45
_MALE_INCOME = [50000.0 + 137 * (i % 23) + 11 * i for i in range(_NM)]
_FEMALE_INCOME = [46000.0 + 149 * (i % 19) + 9 * i for i in range(_NF)]

_M = 40


@pytest.fixture
def two_group_df() -> pd.DataFrame:
    """Deterministic frame: one continuous target, one binary target."""
    return pd.DataFrame(
        {
            "gender": ["male"] * _NM + ["female"] * _NF,
            "income": _MALE_INCOME + _FEMALE_INCOME,
            "approved": [1] * 42 + [0] * 18 + [1] * 16 + [0] * 29,
        }
    )


@pytest.fixture
def three_group_df() -> pd.DataFrame:
    """Deterministic frame with three groups, so the ANOVA branch runs."""
    return pd.DataFrame(
        {
            "region": ["north"] * _M + ["south"] * _M + ["east"] * _M,
            "score": (
                [70.0 + 0.5 * (i % 17) + 0.25 * i for i in range(_M)]
                + [64.0 + 0.5 * (i % 13) + 0.25 * i for i in range(_M)]
                + [58.0 + 0.5 * (i % 11) + 0.25 * i for i in range(_M)]
            ),
        }
    )


# ---------------------------------------------------------------------------
# 1. The refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested", UNHONOURABLE_DOCUMENTED_TYPES)
def test_named_test_that_cannot_be_honoured_is_refused_on_continuous_target(
    two_group_df: pd.DataFrame, requested: str
) -> None:
    """A named test the module cannot run must raise, not return a t-test.

    Before the fix every one of these returned the Independent t-test result,
    t=-24.192, under its own name in ``test_name`` but with the caller
    believing the test they asked for had run.
    """
    with pytest.raises(ValueError) as excinfo:
        run_disparity_tests(two_group_df, "income", "gender", test_type=requested)

    message = str(excinfo.value)
    assert requested in message, "the refusal must name the value it refused"
    assert "auto" in message, "the refusal must name what is supported"


@pytest.mark.parametrize("requested", UNHONOURABLE_DOCUMENTED_TYPES)
def test_named_test_that_cannot_be_honoured_is_refused_on_binary_target(
    two_group_df: pd.DataFrame, requested: str
) -> None:
    """Same refusal on the categorical branch.

    This is the cross-type case: before the fix ``test_type='ttest'`` on the
    binary column returned a Chi-squared test, chi2=10.9848, p=9.1860e-04.
    """
    with pytest.raises(ValueError):
        run_disparity_tests(two_group_df, "approved", "gender", test_type=requested)


@pytest.mark.parametrize(
    "requested",
    ["not_a_test_at_all", "AUTO", "Auto", "", "ttest2", "welch", "kruskal"],
)
def test_unknown_test_type_is_refused(two_group_df: pd.DataFrame, requested: str) -> None:
    """An unknown value must raise rather than fall through to the default.

    Case matters: 'AUTO' is not 'auto'. Accepting it would be a silent
    normalisation, which is the same class of surprise as the original bug.
    """
    with pytest.raises(ValueError):
        run_disparity_tests(two_group_df, "income", "gender", test_type=requested)


def test_supported_set_is_exactly_auto() -> None:
    """Widening this set without implementing the routing reopens S-02."""
    assert _SUPPORTED_TEST_TYPES == ("auto",)


def test_docstring_advertises_nothing_it_cannot_honour(two_group_df: pd.DataFrame) -> None:
    """Every test name the docstring mentions is honoured or refused.

    The docstring is allowed to name 'ttest', 'anova', 'chi2' and
    'mannwhitney' while explaining that they raise. What it must never do is
    name one that is quietly accepted and answered with a different test.
    """
    doc = inspect.getdoc(run_disparity_tests) or ""
    assert doc, "run_disparity_tests must keep a docstring"

    mentioned = [name for name in ["auto", *UNHONOURABLE_DOCUMENTED_TYPES] if name in doc]
    assert "auto" in mentioned

    for name in mentioned:
        if name in _SUPPORTED_TEST_TYPES:
            result = run_disparity_tests(two_group_df, "income", "gender", test_type=name)
            assert result["test_type_requested"] == name
        else:
            with pytest.raises(ValueError):
                run_disparity_tests(two_group_df, "income", "gender", test_type=name)


# ---------------------------------------------------------------------------
# 2. Over-correction controls: the honoured path still measures
# ---------------------------------------------------------------------------


def test_auto_continuous_two_groups_returns_measured_ttest(
    two_group_df: pd.DataFrame,
) -> None:
    """The default path returns its real t-test numbers, unchanged by the fix.

    These literals were captured from the same fixture before the refusal was
    added, so a fix that refused everything, or that perturbed the measured
    branch, fails here.
    """
    result = run_disparity_tests(two_group_df, "income", "gender")

    assert result["test_name"] == "Independent t-test"
    assert result["test_type_requested"] == "auto"
    assert result["test_statistic"] == pytest.approx(-24.192461460438327, rel=1e-9, abs=0.0)
    # abs=0.0 is load-bearing: pytest.approx defaults to abs=1e-12, which on a
    # p-value of 1e-44 calls every possible value equal. Measured: a 14.6%
    # relative error passed under the default.
    assert result["pvalue"] == pytest.approx(2.7881203219290533e-44, rel=1e-9, abs=0.0)
    assert result["effect_size"] == pytest.approx(-4.770817738317193, rel=1e-9, abs=0.0)
    assert result["effect_size_type"] == "Cohen's d"
    assert result["significance"] == "highly_significant"
    assert result["effect_interpretation"] == "large"
    assert result["privileged_group"] == "male"
    assert result["disadvantaged_group"] == "female"
    assert result["disparity_magnitude"] == pytest.approx(4287.716666666667, rel=1e-12)
    assert result["confidence_interval"][0] == pytest.approx(-4635.111890429663, rel=1e-9)
    assert result["confidence_interval"][1] == pytest.approx(-3940.321442903671, rel=1e-9)
    assert result["group_statistics"]["male"]["mean"] == pytest.approx(51687.65, rel=1e-12)
    assert result["group_statistics"]["female"]["mean"] == pytest.approx(
        47399.933333333334, rel=1e-12
    )
    assert result["sample_sizes"] == {"female": 45, "male": 60}


def test_auto_binary_two_groups_returns_measured_chi_squared(
    two_group_df: pd.DataFrame,
) -> None:
    """The categorical branch still returns its real chi-squared numbers."""
    result = run_disparity_tests(two_group_df, "approved", "gender")

    assert result["test_name"] == "Chi-squared test"
    assert result["test_type_requested"] == "auto"
    assert result["test_statistic"] == pytest.approx(10.984845011005135, rel=1e-9, abs=0.0)
    assert result["pvalue"] == pytest.approx(0.0009185996422947638, rel=1e-9, abs=0.0)
    assert result["effect_size"] == pytest.approx(0.32344639669253117, rel=1e-9, abs=0.0)
    assert result["effect_size_type"] == "Cramér's V"
    assert result["privileged_group"] == "male"
    assert result["disadvantaged_group"] == "female"
    assert result["group_statistics"]["male"]["positive_rate"] == pytest.approx(0.7, rel=1e-12)
    assert result["group_statistics"]["female"]["positive_rate"] == pytest.approx(
        0.35555555555555557, rel=1e-12
    )


def test_auto_continuous_three_groups_returns_measured_anova(
    three_group_df: pd.DataFrame,
) -> None:
    """Three groups still route to the ANOVA branch with its real F and p."""
    result = run_disparity_tests(three_group_df, "score", "region")

    assert result["test_name"] == "One-way ANOVA"
    assert result["test_type_requested"] == "auto"
    assert result["test_statistic"] == pytest.approx(121.48651301104269, rel=1e-9, abs=0.0)
    assert result["pvalue"] == pytest.approx(2.799499015944359e-29, rel=1e-9, abs=0.0)
    assert result["effect_size"] == pytest.approx(0.6749756466674212, rel=1e-9, abs=0.0)
    assert result["effect_size_type"] == "Eta-squared"
    assert result["privileged_group"] == "north"
    assert result["disadvantaged_group"] == "east"
    assert result["disparity_magnitude"] == pytest.approx(13.2625, rel=1e-9, abs=0.0)
    assert result["confidence_interval"] == (None, None)


def test_default_argument_is_still_auto(two_group_df: pd.DataFrame) -> None:
    """Passing nothing and passing 'auto' agree on every value."""
    implicit = run_disparity_tests(two_group_df, "income", "gender")
    explicit = run_disparity_tests(two_group_df, "income", "gender", test_type="auto")
    assert implicit == explicit


# ---------------------------------------------------------------------------
# 3. Three states: measured / could-not-check / refused
# ---------------------------------------------------------------------------


def test_groups_too_small_reports_could_not_check_not_a_test_name() -> None:
    """No test ran, so ``test_name`` is None rather than a plausible label.

    This is the third state. It must not be collapsed into either a measured
    result or the refusal.
    """
    tiny = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})

    result = run_disparity_tests(tiny, "y", "g")

    assert result["error"] == "Could not run disparity test"
    assert result["target"] == "y"
    assert result["test_name"] is None
    assert result["test_type_requested"] == "auto"
    assert "pvalue" not in result, "a run that produced no p-value must not report one"


def test_refusal_precedes_the_could_not_check_path() -> None:
    """An unhonourable request raises even when the data could not be tested.

    Otherwise the refusal would be reachable only on data that happens to
    work, and a caller on thin data would get a quiet error dict for what is
    really an unsupported request.
    """
    tiny = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "y": [1.0, 2, 3, 4, 5, 6, 7, 8]})

    with pytest.raises(ValueError):
        run_disparity_tests(tiny, "y", "g", test_type="mannwhitney")
