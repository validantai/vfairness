"""Two answers to "are these the same" for the case that occurs most.

At temperature 0 an endpoint returns byte-identical text on repeated runs, so
two arms that are treated alike produce two CONSTANT EQUAL score arrays. That
is the commonest genuinely-fair shape this library sees, and two components
disagreed about it:

    llm_probe._pair_stats                (p=1.0, effect=0.0)   compared, no difference
    NonDeterminismAnalyzer.equivalence_test  'could_not_check'  nothing was established

``ttest_ind`` on two constant arrays answers NaN, which is why the second one
refused. But the refusal discards a real measurement: the system behaved
identically, reproducibly, on every observation. That is equivalence
established by OBSERVATION rather than inferred, and it is more than a t-test
could have shown.

The 1e-12 clamp is what hid it. It scales the SESOI down by the same 1e-12, so
the t-statistics come out around 0.245 whatever the data did and TOST reports
"equivalence not established" for two byte-identical arms.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.llm.nondeterminism import NonDeterminismAnalyzer
from vfairness.operations.pulse.llm_probe import _pair_stats


@pytest.fixture()
def analyzer() -> NonDeterminismAnalyzer:
    return NonDeterminismAnalyzer()


def test_two_constant_equal_arms_are_measured_not_refused(analyzer) -> None:
    a = np.full(5, 0.5)
    b = np.full(5, 0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here would mean it still refuses
        out = analyzer.equivalence_test(a, b)

    assert out["verdict"] == "fairness_confirmed"
    assert out["effect_size"] == 0.0
    # No p-value exists and none is invented. This is the distinction the whole
    # audit turns on: 1.0 here would be a sentinel, None is the absence of a
    # test that genuinely cannot be run.
    assert out["p_diff"] is None
    assert out["p_tost"] is None
    assert out["constant_arms"] is True
    assert "established by observation rather than inferred" in out["interpretation"]
    assert "says nothing about prompts that were not tested" in out["interpretation"]


def test_the_two_components_now_agree_on_that_case(analyzer) -> None:
    """The disagreement this file exists to close, asserted from both sides."""
    a = [0.5] * 5
    b = [0.5] * 5

    p, effect = _pair_stats(a, b)
    assert p == 1.0, "the probe reads two constant equal arms as compared, no difference"
    assert effect == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        verdict = analyzer.equivalence_test(np.asarray(a), np.asarray(b))["verdict"]

    # Neither says could-not-check. They agree the arms were compared and found
    # alike, which is what the data show.
    assert verdict == "fairness_confirmed"
    assert verdict != "could_not_check"


def test_a_reproducible_difference_leads_with_the_finding_not_with_the_gap(analyzer) -> None:
    """Two constant arms that DIFFER.

    The difference is certain and perfectly reproducible. Only its practical
    significance is ungraded, because Cohen's d needs a variance and there is
    none. 'undetermined' on its own would read as "nothing to see", so the
    sentence leads with the gap.
    """
    out = analyzer.equivalence_test(np.full(5, 0.9), np.full(5, 0.2))
    assert out["verdict"] == "undetermined"
    assert out["effect_size"] is None, "Cohen's d is undefined without variance and is not invented"
    assert out["mean_diff"] == pytest.approx(0.7)
    assert "they DIFFER" in out["interpretation"]
    assert "perfectly reproducible" in out["interpretation"]
    assert "ungraded real difference, not an absence of one" in out["interpretation"]


def test_a_raw_sesoi_grades_what_cohens_d_cannot(analyzer) -> None:
    big = analyzer.equivalence_test(np.full(5, 0.9), np.full(5, 0.2), sesoi_raw=0.05)
    assert big["verdict"] == "bias_detected"
    assert big["effect_size"] is None
    assert "no effect size is reported" in big["interpretation"]

    small = analyzer.equivalence_test(np.full(5, 0.51), np.full(5, 0.50), sesoi_raw=0.05)
    assert small["verdict"] == "below_practical_significance"
    assert "too small to matter" in small["interpretation"]


def test_the_varying_path_is_untouched(analyzer) -> None:
    """Over-correction control.

    The constant-arms branch must not capture data that HAS variance, or it
    would replace a real t-test with an observation-only reading.
    """
    out = analyzer.equivalence_test(
        np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
        np.array([0.11, 0.19, 0.31, 0.39, 0.52]),
    )
    assert out.get("constant_arms") is None
    assert out["p_diff"] is not None
    assert out["p_tost"] is not None
    assert isinstance(out["effect_size"], float)


def test_one_constant_arm_and_one_varying_is_not_the_constant_case(analyzer) -> None:
    """Only BOTH arms constant collapses the scale. One is an ordinary test."""
    out = analyzer.equivalence_test(np.full(5, 0.5), np.array([0.4, 0.5, 0.6, 0.5, 0.5]))
    assert out.get("constant_arms") is None
    assert out["p_diff"] is not None


# ---------------------------------------------------------------------------
# The guard itself, over values and sample sizes the first version missed
# ---------------------------------------------------------------------------
#
# The first version tested `np.var(a) == 0.0`. Variance is an ACCUMULATED
# statistic, so a constant array of a value that is not exactly representable in
# binary does not have exactly zero variance, and whether it does depends on n
# as well as the value:
#
#   np.var([0.9] * 5)  -> 0.0        np.var([0.9] * 20) -> 4.93e-32
#   np.var([0.5] * 5)  -> 0.0        np.var([0.5] * 20) -> 0.0
#
# Every fixture above used 0.5 and n=5, which is the one corner where the broken
# guard behaves. It passed for the wrong reason. Found by a peer session running
# the unequal case; these parameterised cases exist so no fixture can supply the
# condition under test again.

_CONSTANT_VALUES = [0.5, 0.25, 0.9, 0.6, 0.2, 0.1, 1.0 / 3.0, 0.51, 7.3, 1e-4]
_SIZES = [2, 5, 20, 51]


@pytest.mark.parametrize("value", _CONSTANT_VALUES)
@pytest.mark.parametrize("size", _SIZES)
def test_equal_constant_arms_are_recognised_at_every_value_and_size(
    analyzer, value: float, size: int
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.equivalence_test(np.full(size, value), np.full(size, value))
    assert out["constant_arms"] is True, f"value={value} size={size} fell through the guard"
    assert out["verdict"] == "fairness_confirmed"
    assert out["p_diff"] is None
    assert out["effect_size"] == 0.0


@pytest.mark.parametrize("value", _CONSTANT_VALUES)
@pytest.mark.parametrize("size", _SIZES)
def test_differing_constant_arms_never_fabricate_an_effect_size(
    analyzer, value: float, size: int
) -> None:
    """The failure the variance guard produced, at every value it missed.

    Falling through to the 1e-12 clamp gave Cohen's d of -4e11 with the text
    "This is a real, meaningful bias", from a comparison where d has no value at
    all, and a t-test p of exactly 0.0 on zero-variance data.
    """
    other = value + 0.4
    out = analyzer.equivalence_test(np.full(size, value), np.full(size, other))
    assert out["constant_arms"] is True, f"value={value} size={size} fell through the guard"
    assert out["effect_size"] is None, "Cohen's d has no value without variance"
    assert out["p_diff"] is None, "a t-test p of 0.0 on zero-variance data is not a measurement"
    assert out["mean_diff"] == pytest.approx(-0.4, abs=1e-9)
    assert "real, meaningful bias" not in out["interpretation"]


@pytest.mark.parametrize("size", _SIZES)
def test_a_tiny_real_spread_is_not_treated_as_constant(analyzer, size: int) -> None:
    """Over-correction control.

    ptp is exact, so a genuine difference of one ulp is variance and must take
    the ordinary path. A guard that swallowed near-constant data would replace
    a real test with an observation-only reading.
    """
    a = np.full(size, 0.9)
    a[0] = np.nextafter(0.9, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = analyzer.equivalence_test(a, np.full(size, 0.9))
    assert out.get("constant_arms") is None
    assert out["p_diff"] is not None
