"""S-01: the strongest possible evidence of bias must not read as no evidence.

THE DEFECT. ``compare_to_benchmark`` graded its chi-squared result with

    "statistically_significant": chi2_pval < 0.05 if chi2_pval else None,

``if chi2_pval`` is a FALSY test, not a None test. A p-value of exactly 0.0 is
the strongest evidence a chi-squared test can give that the observed
representation differs from the benchmark, and it is reached by ordinary data:
3000 male rows and 1 female row against a 50/50 benchmark gives
chi2 = 2997.001332889037 and p = 0.0, because the survival function underflows
the double range somewhere above chi2 = 1497. Being falsy, that 0.0 took the
else branch and the function reported None, which is the value it also reports
when the test DID NOT RUN AT ALL. The most extreme finding in the library and a
complete absence of measurement produced byte-identical output.

The second half of the same line: 0.05 was hardcoded while this module has a
real ``significance_level`` parameter on ``detect_representation_bias``. That
parameter never reaches this function, and this function takes no such
argument, so the alpha here genuinely is fixed. The fix is therefore not a new
knob (inventing one would be a different claims-vs-code defect); it is to name
the constant, report it back as ``significance_level``, and have the docstring
say plainly that it cannot be changed. The tests below hold the docstring and
the signature to each other so the honest statement cannot rot into a false one.

The same substitution of a neutral 0 for an absent measurement was found twice
more in this file, in ``calculate_representation_ratio``: an empty column and a
non-positive benchmark both produced a ratio of 0, and 0 grades as
``is_underrepresented: True``. A dataset with no rows was reported as
underrepresented, and so was a group holding half the dataset. Those are pinned
here too.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

pd = pytest.importorskip("pandas")

from vfairness.preprocessing.bias_detection.representation import (  # noqa: E402
    _BENCHMARK_SIGNIFICANCE_LEVEL,
    calculate_representation_ratio,
    compare_to_benchmark,
    detect_representation_bias,
)

BALANCED_BENCHMARK = {"male": 0.5, "female": 0.5}


def _gender_frame(n_male: int, n_female: int) -> "pd.DataFrame":
    return pd.DataFrame({"gender": ["male"] * n_male + ["female"] * n_female})


def _compare(df, benchmark=BALANCED_BENCHMARK):
    """Call the function without letting an unrelated warning fail the run."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return compare_to_benchmark(df, "gender", benchmark)


# ── the refusal pin: p == 0.0 is the strongest evidence, not a missing one ───


def test_p_value_of_exactly_zero_is_reported_as_significant():
    """3000 vs 1 against 50/50: p underflows to a hard 0.0 and must grade True."""
    out = _compare(_gender_frame(3000, 1))

    # The measured numbers, asserted as values rather than as "some float".
    assert out["sample_size"] == 3001
    assert out["chi_squared_statistic"] == pytest.approx(2997.001332889037)
    assert out["pvalue"] == 0.0
    assert isinstance(out["pvalue"], float)

    # The whole point: 0.0 is falsy, and the verdict must still be True.
    assert out["statistically_significant"] is True


def test_zero_p_value_is_distinguishable_from_a_test_that_never_ran():
    """The two states must not print the same thing."""
    strongest = _compare(_gender_frame(3000, 1))

    # Only one group is shared with the benchmark, so no chi-squared test is
    # possible: this is the genuine could-not-check.
    with pytest.warns(UserWarning, match="did not run"):
        no_test = compare_to_benchmark(
            pd.DataFrame({"gender": ["male"] * 100}), "gender", {"male": 1.0}
        )

    assert strongest["pvalue"] == 0.0
    assert no_test["pvalue"] is None
    assert strongest["statistically_significant"] is True
    assert no_test["statistically_significant"] is None
    assert strongest["statistically_significant"] is not no_test["statistically_significant"], (
        "the strongest evidence of bias and no measurement at all report the same value"
    )


# ── the could-not-check pin: None stays None, and says so out loud ──────────


def test_absent_p_value_stays_none_and_is_named_in_a_warning():
    with pytest.warns(UserWarning, match="UNMEASURED, not 'not significant'"):
        out = compare_to_benchmark(
            pd.DataFrame({"gender": ["male"] * 100}), "gender", {"male": 1.0}
        )

    assert out["chi_squared_statistic"] is None
    assert out["pvalue"] is None
    assert out["statistically_significant"] is None


# ── over-correction controls: the measured path still answers ──────────────


def test_a_clearly_non_significant_comparison_reports_false():
    """A dataset that matches its benchmark exactly: chi2 = 0.0, p = 1.0."""
    out = _compare(_gender_frame(50, 50))

    assert out["chi_squared_statistic"] == 0.0  # a measured perfect fit, not None
    assert out["pvalue"] == pytest.approx(1.0)
    assert out["statistically_significant"] is False


def test_a_significant_but_non_underflowing_p_value_reports_true():
    """999 vs 1: p = 1.33e-218, small but genuinely representable."""
    out = _compare(_gender_frame(999, 1))

    assert out["chi_squared_statistic"] == pytest.approx(996.004)
    assert out["pvalue"] == pytest.approx(1.3269482792741334e-218)
    assert out["pvalue"] > 0.0
    assert out["statistically_significant"] is True


def test_a_mildly_skewed_dataset_reports_false_not_true():
    """59/41 gives p = 0.0719, above alpha: the fix must not grade everything True."""
    out = _compare(_gender_frame(59, 41))

    assert out["chi_squared_statistic"] == pytest.approx(3.24)
    assert out["pvalue"] == pytest.approx(0.0718606382258515)
    assert out["statistically_significant"] is False


# ── the alpha half: fixed at 0.05, actually applied, honestly documented ────


def test_the_verdict_turns_on_the_reported_alpha():
    """60/40 and 59/41 straddle 0.05, and the verdict flips exactly there."""
    just_significant = _compare(_gender_frame(60, 40))
    just_not = _compare(_gender_frame(59, 41))

    assert just_significant["pvalue"] == pytest.approx(0.045500263896358445)
    assert just_not["pvalue"] == pytest.approx(0.0718606382258515)

    alpha = just_significant["significance_level"]
    assert alpha == 0.05
    assert alpha == _BENCHMARK_SIGNIFICANCE_LEVEL
    assert just_significant["pvalue"] < alpha < just_not["pvalue"]

    assert just_significant["statistically_significant"] is True
    assert just_not["statistically_significant"] is False


def test_the_alpha_is_reported_on_every_comparison():
    for df in (_gender_frame(50, 50), _gender_frame(3000, 1)):
        assert _compare(df)["significance_level"] == 0.05

    with pytest.warns(UserWarning):
        no_test = compare_to_benchmark(
            pd.DataFrame({"gender": ["male"] * 100}), "gender", {"male": 1.0}
        )
    assert no_test["significance_level"] == 0.05


def test_the_docstring_tells_the_truth_about_the_fixed_alpha():
    """The honest statement and the signature must stay in step.

    ``detect_representation_bias`` has a real ``significance_level``; this
    function does not, and its docstring says so. If someone ever adds the
    parameter, this test fails and forces the docstring to be corrected rather
    than left claiming an immovable alpha that has become movable.
    """
    sig = inspect.signature(compare_to_benchmark)
    # Collapse the docstring's own line wrapping so a phrase can be looked for
    # as one sentence rather than as whatever the formatter happened to split.
    doc = " ".join((inspect.getdoc(compare_to_benchmark) or "").split())

    assert "significance_level" in inspect.signature(detect_representation_bias).parameters

    if "significance_level" in sig.parameters:
        pytest.fail(
            "compare_to_benchmark now takes significance_level; the docstring "
            "must stop saying the alpha is fixed and the verdict must use the "
            "caller's value"
        )

    assert "FIXED alpha" in doc
    assert "0.05" in doc
    assert "no ``significance_level`` parameter" in doc
    assert str(_BENCHMARK_SIGNIFICANCE_LEVEL) == "0.05"


# ── siblings in the same file: a neutral 0 standing in for no measurement ───


def test_an_empty_column_yields_no_representation_verdict():
    """Zero rows measured nothing, so nothing is underrepresented."""
    df = pd.DataFrame({"gender": pd.Series([None, None, None], dtype=object)})

    with pytest.warns(UserWarning, match="UNMEASURED"):
        out = calculate_representation_ratio(df, "gender", "female", 0.5)

    assert out["actual_count"] == 0
    assert out["actual_proportion"] is None
    assert out["representation_ratio"] is None
    assert out["deficit_count"] is None
    assert out["is_underrepresented"] is None
    assert out["is_overrepresented"] is None


def test_a_non_positive_benchmark_yields_no_representation_verdict():
    """Half the dataset used to come back underrepresented off an undefined ratio."""
    df = _gender_frame(50, 50)

    with pytest.warns(UserWarning, match="not positive"):
        out = calculate_representation_ratio(df, "gender", "female", 0.0)

    assert out["actual_count"] == 50
    assert out["actual_proportion"] == pytest.approx(0.5)  # still measured
    assert out["representation_ratio"] is None
    assert out["is_underrepresented"] is None
    assert out["is_overrepresented"] is None


def test_a_measured_representation_ratio_still_grades():
    """Over-correction control for both siblings, asserted as measured values."""
    df = _gender_frame(50, 50)

    parity = calculate_representation_ratio(df, "gender", "female", 0.5)
    assert parity["actual_proportion"] == pytest.approx(0.5)
    assert parity["representation_ratio"] == pytest.approx(1.0)
    assert parity["deficit_count"] == 0
    assert parity["is_underrepresented"] is False
    assert parity["is_overrepresented"] is False

    # 50% observed against a 90% benchmark: a real, severe shortfall.
    shortfall = calculate_representation_ratio(df, "gender", "female", 0.9)
    assert shortfall["representation_ratio"] == pytest.approx(0.5 / 0.9)
    assert shortfall["deficit_count"] == 40
    assert shortfall["is_underrepresented"] is True
    assert shortfall["is_overrepresented"] is False

    # 50% observed against a 10% benchmark: a real overrepresentation.
    surplus = calculate_representation_ratio(df, "gender", "female", 0.1)
    assert surplus["representation_ratio"] == pytest.approx(5.0)
    assert surplus["is_underrepresented"] is False
    assert surplus["is_overrepresented"] is True


class TestDeficitCountAgreesWithItsDocstring:
    """The first S-01 fix documented ``deficit_count`` as None for a
    non-positive benchmark and left the code computing a number from it. The
    branch was unpinned in both directions, so either half could drift."""

    def test_a_non_positive_benchmark_yields_no_deficit(self):
        df = pd.DataFrame({"gender": ["m"] * 50 + ["f"] * 50})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = calculate_representation_ratio(df, "gender", "female", 0.0)
        assert out["representation_ratio"] is None
        assert out["deficit_count"] is None, out["deficit_count"]
        assert out["is_underrepresented"] is None
        # The one thing that IS measured here stays measured.
        assert out["actual_count"] == 0

    def test_control_a_real_benchmark_still_reports_a_real_deficit(self):
        df = pd.DataFrame({"gender": ["m"] * 90 + ["f"] * 10})
        out = calculate_representation_ratio(df, "gender", "f", 0.5)
        assert out["deficit_count"] == 40
        assert out["representation_ratio"] == pytest.approx(0.2)
        assert out["is_underrepresented"] is True
