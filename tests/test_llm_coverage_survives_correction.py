"""The coverage disclosure has to survive the path the consumer actually uses.

READINESS-6 added n_scored_a / n_scored_b to ``OutputAnalyzer._compare`` so a
comparison run on a SUBSET of the responses could never be read as one run on
all of them. It was correct on a direct ``analyze_sentiment`` call.

``analyze_all`` then rebuilt every result field by field to attach the
corrected p-values, listing sixteen fields and not those two. The consumer only
ever calls ``analyze_all``. So the disclosure was right in the function that
computed it and gone in the only path that ships it, and the platform, reading
None, rendered a subset comparison as full coverage.

A copy constructor that enumerates fields silently drops every field added
after it was written, which is why the fix is ``dataclasses.replace`` and not a
seventeenth line.
"""

from __future__ import annotations

import warnings

import pytest

from vfairness.llm.output_analysis import OutputAnalyzer

# Readable by the keyword scorers. The noise lines contain none of their
# lexicon words, so the scorers honestly answer NaN for them.
READABLE = "this is an excellent and wonderful and good outcome for everyone"
UNREADABLE = "zzz qqq"


@pytest.fixture()
def analyzer() -> OutputAnalyzer:
    return OutputAnalyzer()


def _sentiment_row(rows):
    for r in rows:
        d = r.to_dict()
        if d.get("metric") == "sentiment":
            return d
    raise AssertionError("no sentiment row")


def test_the_coverage_counts_survive_analyze_all(analyzer) -> None:
    texts = [READABLE] * 30 + [UNREADABLE] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = analyzer.analyze_sentiment(texts, texts, "A", "B").to_dict()
        through = _sentiment_row(analyzer.analyze_all(texts, texts, "A", "B"))

    # The function that computes them was always right.
    assert direct["n_scored_a"] == 30
    assert direct["n_scored_b"] == 30
    # The path that ships them must agree, and used to answer None.
    assert through["n_scored_a"] == 30, "analyze_all dropped the coverage disclosure"
    assert through["n_scored_b"] == 30
    assert through["assessed"] is True


def test_the_denominator_travels_with_the_numerator(analyzer) -> None:
    """ "The scorer read 30" says nothing without "of 32".

    sample_size is the count the test RAN on, so on a subset it is already the
    reduced number and cannot serve as the denominator: here it is 30, the same
    as n_scored, and a reader comparing the two would conclude full coverage.
    """
    texts = [READABLE] * 30 + [UNREADABLE] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = _sentiment_row(analyzer.analyze_all(texts, texts, "A", "B"))

    assert row["n_supplied_a"] == 32
    assert row["n_supplied_b"] == 32
    assert row["n_scored_a"] == 30
    assert row["sample_size"] == 30, "sample_size is the tested count, not the supplied one"
    assert row["n_supplied_a"] != row["sample_size"]


def test_full_coverage_reports_no_subset_and_still_reports_the_denominator(analyzer) -> None:
    """n_scored stays None on full coverage: it means "fewer than supplied"."""
    texts = [READABLE] * 30
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = _sentiment_row(analyzer.analyze_all(texts, texts, "A", "B"))
    assert row["n_scored_a"] is None
    assert row["n_scored_b"] is None
    assert row["n_supplied_a"] == 30
    assert row["n_supplied_b"] == 30


def test_a_refusal_carries_its_denominator_too(analyzer) -> None:
    """A reader has to know how much was supplied to a comparison that
    produced nothing, or "not assessed" is unfalsifiable."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = analyzer.analyze_sentiment([UNREADABLE] * 5, [UNREADABLE] * 5, "A", "B").to_dict()
    assert row["assessed"] is False
    assert row["not_assessed_reason"] == "non_finite_scores"
    assert row["n_supplied_a"] == 5
    assert row["n_supplied_b"] == 5


def test_every_field_survives_the_correction_rebuild(analyzer) -> None:
    """The general guard, not just the two fields that were dropped.

    Any field added to OutputAnalysisResult from now on is carried by
    dataclasses.replace automatically. This asserts that property directly, so
    the next field cannot repeat the defect.
    """
    from dataclasses import fields

    texts = [READABLE] * 30 + [UNREADABLE] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uncorrected = analyzer.analyze_all(texts, texts, "A", "B", correction_method=None)
        corrected = analyzer.analyze_all(texts, texts, "A", "B")

    changed_by_design = {"p_value", "is_significant", "metadata"}
    names = [f.name for f in fields(type(uncorrected[0]))]
    assert len(uncorrected) == len(corrected)
    for before, after in zip(uncorrected, corrected):
        for name in names:
            if name in changed_by_design:
                continue
            assert getattr(before, name) == getattr(after, name), (
                f"{name} did not survive the correction rebuild on metric {before.metric}"
            )


# ---------------------------------------------------------------------------
# Cohen's d had BOTH halves of the fabrication defect, and they cancelled into
# nonsense that depended on the sample size.
# ---------------------------------------------------------------------------
#
#     two constant arms, gap 0.7,    n=5   ->   0.0        a fabricated PASS
#     two constant arms, gap 0.7,    n=20  ->  -3.1e15     a fabricated FINDING
#     two constant arms, gap 0.01,   n=25  ->   0.0
#     two constant arms, gap 0.0001, n=20  ->  -4.4e11
#
# Which one you got depended on whether `pooled_std == 0` happened to be exactly
# true, and that depends on the value AND the count. The 1e11 values are FINITE,
# so `_compare`'s math.isfinite guard passed them through and the magnitude
# label called them "large".
#
# Found by pointing the platform's guarded-division scan at this repo, because
# the peer's clamp detector does not see this shape and mine does not see theirs.

import numpy as np

_CONSTANT_VALUES = [0.5, 0.9, 0.1, 1.0 / 3.0, 0.51]
_SIZES = [2, 5, 20, 25, 51]


@pytest.mark.parametrize("value", _CONSTANT_VALUES)
@pytest.mark.parametrize("size", _SIZES)
def test_cohens_d_refuses_when_there_is_no_variance(value: float, size: int) -> None:
    """Both directions at once: no fabricated 0.0, no fabricated magnitude."""
    from vfairness.llm.output_analysis import OutputAnalyzer

    for gap in (0.7, 0.01, 0.0001):
        d = OutputAnalyzer._cohens_d(np.full(size, value), np.full(size, value + gap))
        assert d is None, f"value={value} size={size} gap={gap} returned {d!r}"


@pytest.mark.parametrize("size", _SIZES)
def test_cohens_d_still_answers_when_either_arm_varies(size: int) -> None:
    """Over-correction control. Only BOTH arms constant kills the standardisation."""
    from vfairness.llm.output_analysis import OutputAnalyzer

    if size < 3:
        pytest.skip("needs room for a differing observation")
    a = np.full(size, 0.9)
    a[0] = 0.1
    d = OutputAnalyzer._cohens_d(a, np.full(size, 0.9))
    assert d is not None
    assert np.isfinite(d)
    # And a real, ordinary comparison is unaffected.
    rng = np.random.default_rng(3)
    ordinary = OutputAnalyzer._cohens_d(rng.normal(0, 1, size), rng.normal(0.5, 1, size))
    assert ordinary is not None and np.isfinite(ordinary)


@pytest.mark.parametrize("gap", [0.7, 0.01, 0.0001])
@pytest.mark.parametrize("size", [5, 20, 51])
def test_no_measure_reports_a_maximal_effect_from_constant_arms(gap: float, size: int) -> None:
    """EVERY measure fabricated here, not just Cohen's d.

    Cliff's Delta answers -1.0 "large" for a gap of 0.7 AND for 0.0001, because
    the ranks separate perfectly either way. So the refusal has to sit ABOVE the
    measure selection, or fixing Cohen's d just moves the fabrication next door.
    """
    from vfairness.llm.output_analysis import OutputAnalyzer

    measure, value, label = OutputAnalyzer._select_effect_size(
        np.full(size, 0.37), np.full(size, 0.37 + gap)
    )
    assert value is None, f"gap={gap} size={size} returned {measure}={value!r} labelled {label!r}"
    assert label == "not_assessed"


@pytest.mark.parametrize("size", [5, 20, 51])
def test_constant_and_equal_arms_keep_their_measured_zero(size: int) -> None:
    """The other half, and it must NOT be refused.

    Two identical constant arms have a genuinely zero effect, that zero is a
    measurement, and at temperature 0 it is the commonest fair shape there is.
    Refusing it would be the discard-evidence direction.
    """
    from vfairness.llm.output_analysis import OutputAnalyzer

    _, value, label = OutputAnalyzer._select_effect_size(np.full(size, 0.37), np.full(size, 0.37))
    assert value == 0.0
    assert label == "negligible"


def test_ordinary_data_still_gets_its_measure() -> None:
    """Over-correction control: the guard must not swallow data that varies."""
    from vfairness.llm.output_analysis import OutputAnalyzer

    rng = np.random.default_rng(1)
    measure, value, label = OutputAnalyzer._select_effect_size(
        rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
    )
    assert measure == "Cohen's d"
    assert value is not None and np.isfinite(value)
    assert label in {"negligible", "small", "medium", "large"}


def test_a_total_refusal_split_keeps_its_maximal_effect_size() -> None:
    """The case the refusal detector exists for, and my first guard removed it.

    Cohen's h is computed from the two PROPORTIONS and needs no spread within an
    arm, so 100% refusal against 0% is a legitimate and maximal h of 3.14. An
    earlier version of the constant-arms guard ran before the binary branch and
    returned None here, so the split lost its effect size, dropped out of
    `unconfirmedLargeEffects` in the orchestrator, and the summary was free to
    read "no significant generative-output disparity detected" over the top of a
    categorical denial of service.

    Caught by a peer session's pin on the full suite, not by mine. This is the
    over-correction control for that guard, kept next to it.
    """
    from vfairness.llm.output_analysis import OutputAnalyzer

    for a, b in ((np.ones(5), np.zeros(5)), (np.zeros(5), np.ones(5))):
        measure, value, label = OutputAnalyzer._select_effect_size(a, b)
        assert measure == "Cohen's h"
        assert value is not None
        assert abs(value) == pytest.approx(3.1415, abs=1e-3)
        assert label == "large"
