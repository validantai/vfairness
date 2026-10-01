"""Beta Go-Live stage 1, group g12: preprocessing statistical disparities.

Two critical defects, both reproduced at the public API
``vfairness.analyze_statistical_disparities`` before any fix:

1. ``_cohens_d`` returned a fabricated ``0.0`` whenever the pooled standard
   deviation was zero. Measured 2026-09-11 on 40 rows at 0.90 against 40 rows
   at 0.10 (a total separation)::

       effect_size 0.0, effect_interpretation NEGLIGIBLE,
       recommendations ['Statistically significant but small effect. Monitor
       but may not require immediate intervention.']

   while ``disparity_magnitude`` held 0.8 in the same object, unread. The same
   gap with sd=0.05 noise returns 15.68 / LARGE / "Investigate root causes."

2. A group smaller than ``min_group_size`` was dropped silently. Measured on
   200 majority rows all approved against 25 minority rows all rejected::

       analyze_statistical_disparities(...) -> []   with warnings == []

   byte-identical to the return for a genuinely clean 100-vs-100 frame. The
   same excluded frame at ``min_group_size=10`` reports the gap at p=9.7e-34.

Both fixes are three-state: measured / failed / could-not-check. The residual
on defect 2 is recorded in ``test_report_surface_still_grades_coverage_complete``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness import BiasDetector, analyze_statistical_disparities
from vfairness.preprocessing.bias_detection.statistical import (
    EffectSizeInterpretation,
    _eta_squared,
)

_UNMEASURED_TOKEN = "UNMEASURED effect size"
_EXCLUDED_TOKEN = "excluded 1 comparison(s) from testing"
_DOWNGRADE = "may not require immediate intervention"


def _user_warnings(fn):
    """Run *fn*, returning ``(result, [UserWarning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _constant_gap_frame():
    """Total 0.90-vs-0.10 separation with zero within-group spread.

    The 5-row 'c' group is load-bearing and is the reproducer's own shape: it
    puts a third distinct value in the column so ``nunique() == 3`` routes the
    test down the CONTINUOUS branch (t-test + Cohen's d), while itself falling
    under min_group_size so exactly two groups reach the test. Without it the
    column is binary and the categorical Cramer's V branch runs instead.
    """
    return pd.DataFrame(
        {
            "gender": ["a"] * 40 + ["b"] * 40 + ["c"] * 5,
            "score": [0.90] * 40 + [0.10] * 40 + [0.50] * 5,
        }
    )


def _noisy_gap_frame():
    """The SAME 0.8 gap, with real within-group spread. The control."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "gender": ["a"] * 40 + ["b"] * 40 + ["c"] * 5,
            "score": list(rng.normal(0.90, 0.05, 40))
            + list(rng.normal(0.10, 0.05, 40))
            + [0.50] * 5,
        }
    )


def _excluded_frame():
    """25 minority rows, all rejected: below the default min_group_size=30."""
    return pd.DataFrame({"g": ["a"] * 200 + ["b"] * 25, "approved": [1] * 200 + [0] * 25})


def _clean_frame():
    """100 vs 100, 50%/50%: genuinely tested and genuinely clean."""
    return pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100, "approved": ([1] * 50 + [0] * 50) * 2})


# --------------------------------------------------------------------------
# Defect 1: a zero pooled SD is an UNMEASURED effect, never a NEGLIGIBLE one
# --------------------------------------------------------------------------


def test_zero_pooled_sd_reports_not_measurable_not_negligible():
    """THREE STATES at the public entry: nan / not_measurable, never 0.0."""
    df = _constant_gap_frame()
    results, msgs = _user_warnings(
        lambda: analyze_statistical_disparities(df, ["gender"], outcome_columns=["score"])
    )

    assert len(results) == 1
    r = results[0]

    # State 3 -- could not measure. Not a number a reader can rank or plot.
    assert np.isnan(r.effect_size), f"effect_size {r.effect_size!r} is not NaN"
    assert r.effect_size != 0.0
    assert r.effect_interpretation is EffectSizeInterpretation.NOT_MEASURABLE
    assert r.effect_interpretation is not EffectSizeInterpretation.NEGLIGIBLE

    # The caller can tell it apart WITHOUT reading the source.
    assert any(_UNMEASURED_TOKEN in m for m in msgs), msgs

    # SECOND HOP. A bare NaN re-enters _interpret_effect_size (every band is a
    # chain of `<` and every comparison against NaN is False, so it falls to
    # the final `else`) and then _generate_disparity_recommendations, whose
    # "significant but small" arm is the sentence a reader actually acts on.
    joined = " ".join(r.recommendations)
    assert _DOWNGRADE not in joined, r.recommendations
    assert "UNMEASURED" in joined

    # RULE 1: removing the fabricated number must not remove the alarm. The
    # raw gap is still measured and the finding is still published.
    assert r.disparity_magnitude == pytest.approx(0.8)
    assert "Investigate root causes" in joined


def test_control_same_gap_with_spread_still_measures_a_large_effect():
    """CONTROL: a fix that makes everything refuse is a worse defect."""
    results, msgs = _user_warnings(
        lambda: analyze_statistical_disparities(
            _noisy_gap_frame(), ["gender"], outcome_columns=["score"]
        )
    )

    assert len(results) == 1
    r = results[0]
    assert np.isfinite(r.effect_size)
    assert abs(r.effect_size) > 0.8
    assert r.effect_interpretation is EffectSizeInterpretation.LARGE
    assert "Investigate root causes" in " ".join(r.recommendations)
    assert not any(_UNMEASURED_TOKEN in m for m in msgs), msgs


def test_control_three_constant_groups_still_measure_eta_squared():
    """CONTROL for the ANOVA sibling of the same dispatch.

    Zero WITHIN-group spread does not make eta-squared unmeasurable: all of
    the variance is between groups, and 1.0 is the honest answer. The guard
    must not swallow this one.
    """
    df = pd.DataFrame(
        {
            "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            "score": [0.9] * 40 + [0.5] * 40 + [0.1] * 40,
        }
    )
    results = analyze_statistical_disparities(df, ["g"], outcome_columns=["score"])
    assert len(results) == 1
    assert results[0].test_name == "One-way ANOVA"
    assert results[0].effect_size == pytest.approx(1.0)
    assert results[0].effect_interpretation is EffectSizeInterpretation.LARGE


def test_eta_squared_sibling_does_not_fabricate_zero():
    """Defence in depth for the other branch of the same dispatch.

    ``ss_total == 0`` is only reachable when every observation is identical,
    which also makes the ANOVA p-value NaN, so the result is filtered out as
    not_testable before the public entry returns. Pinned at the helper because
    fixing Cohen's d alone leaves its sibling free to answer 0.0 the moment a
    future caller does reach it.
    """
    flat = [np.array([0.5] * 10), np.array([0.5] * 10), np.array([0.5] * 10)]
    assert np.isnan(_eta_squared(flat))
    # ...and it still measures when there IS variance to partition.
    spread = [np.array([0.9] * 10), np.array([0.5] * 10), np.array([0.1] * 10)]
    assert _eta_squared(spread) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Defect 2: an untested group is not a clean group
# --------------------------------------------------------------------------


def test_undersized_group_is_disclosed_not_silently_dropped():
    """THREE STATES at the public entry: [] + a warning != [] on its own."""
    excluded, excl_msgs = _user_warnings(
        lambda: analyze_statistical_disparities(
            _excluded_frame(), ["g"], outcome_columns=["approved"]
        )
    )
    clean, clean_msgs = _user_warnings(
        lambda: analyze_statistical_disparities(_clean_frame(), ["g"], outcome_columns=["approved"])
    )

    # Both still return an empty finding list...
    assert excluded == []
    assert clean == []

    # ...but they are no longer the same answer. This is the whole defect:
    # before the fix both were `[]` with `warnings == []`.
    disclosure = [m for m in excl_msgs if _EXCLUDED_TOKEN in m]
    assert disclosure, excl_msgs
    assert "b (n=25)" in disclosure[0]
    assert "NOT TESTED" in disclosure[0]
    assert "NO test was run" in disclosure[0]
    assert "min_group_size=30" in disclosure[0]

    # CONTROL: a frame that really was tested and really is clean says nothing.
    assert not any(_EXCLUDED_TOKEN in m for m in clean_msgs), clean_msgs


def test_control_the_excluded_frame_is_a_real_finding_when_it_is_tested():
    """CONTROL / RULE 1: the disclosure is about a finding that exists."""
    results = analyze_statistical_disparities(
        _excluded_frame(), ["g"], outcome_columns=["approved"], min_group_size=10
    )
    assert len(results) == 1
    assert results[0].pvalue < 1e-30
    assert results[0].effect_interpretation is EffectSizeInterpretation.LARGE


def test_partial_exclusion_is_disclosed_even_when_a_test_still_runs():
    """A test that ran on 2 of 3 groups has not cleared the third."""
    df = pd.DataFrame(
        {
            "gender": ["a"] * 40 + ["b"] * 40 + ["c"] * 5,
            "score": [0.9] * 40 + [0.1] * 40 + [0.5] * 5,
        }
    )
    _, msgs = _user_warnings(
        lambda: analyze_statistical_disparities(df, ["gender"], outcome_columns=["score"])
    )
    disclosure = [m for m in msgs if _EXCLUDED_TOKEN in m]
    assert disclosure, msgs
    assert "c (n=5)" in disclosure[0]


def test_report_surface_still_grades_coverage_complete():
    """The disclosure reaches BiasDetector.full_audit(); the GRADE does not.

    Recorded, not asserted as correct. ``BiasAuditReport.execution_coverage()``
    grades module INVOCATION plus per-attribute observation counts, and both
    are genuinely satisfied here -- attribute 'g' carried 225 rows and all four
    modules ran -- so it still answers "complete" for an audit whose only
    disparity comparison was never run. Closing that half needs a third record
    on ``BiasAuditReport`` and a change to the grader, which live in
    ``detector.py``: outside the file set assigned to this group, and owned by
    another agent in this checkout. This test pins what IS closed here: the
    reader of full_audit() is told, in the same call, that the comparison was
    not run.
    """
    _, msgs = _user_warnings(
        lambda: BiasDetector(
            _excluded_frame(), protected_attributes=["g"], outcome_column="approved"
        ).full_audit()
    )
    assert any(_EXCLUDED_TOKEN in m for m in msgs), msgs
