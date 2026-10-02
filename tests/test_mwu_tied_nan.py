"""A fully tied Mann-Whitney comparison is p = 1.0; any other nan p is a non-test.

WHY. scipy 1.18 changed ``mannwhitneyu`` on fully tied input: every value in both
samples equal now returns ``pvalue=nan`` (the asymptotic path divides by a zero
tie-corrected variance), where scipy <= 1.17 returned 1.0. Measured 2026-10-02 on
scipy 1.18.1, ``mannwhitneyu([1.0] * 30, [1.0] * 31)``: ``pvalue=nan``.

Four call sites guarded identical arms with ``np.array_equal``, which is False as
soon as the two arms differ in LENGTH, so a constant metric over 30 and 31
observations fell through to scipy and came back nan. Each site then read that
nan differently, and none of them read it as what it is:

* ActionBiasAnalyzer.analyze_outcomes: p nan, is_significant None, no warning.
* CorrespondenceTester.analyze_outcomes (numeric path): p nan, is_significant
  None, and a warning that "the statistical test produced no p-value".
* PipelineTracker.compute_cumulative: p nan, so identify_bias_source silently
  dropped the stage from the ranking without naming it in any warning.
* OutputAnalyzer._compare: not assessed, "non-finite p-value" warning.

The exact permutation p for fully tied data is 1.0, a reading and not a stand-in:
every relabelling of one repeated value gives the same U. scipy's own
``method="exact"`` answers 1.0. The shared rule now lives in
``_mannwhitney_two_sided_p``; any OTHER non-finite p stays a could-not-check in
each site's own vocabulary, and now says so where it did not.

Sites audited and NOT changed, because a fully tied input cannot reach scipy there
or the nan is already refused: llm_probe._mw (both arms constant short-circuits
first; a nan p returns (None, None)), CounterfactualTester._test_metric (pooled
ptp == 0 short-circuits; a non-finite p is refused), the text_fairness and
counterfactual design-floor calls (only reached when the pooled data are NOT all
tied, so the extreme rearrangement is never fully tied), and
min_attainable_p_mannwhitney (fixed separated inputs; non-finite candidates are
discarded).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.agents.action_bias import ActionBiasAnalyzer
from vfairness.agents.correspondence import CorrespondenceTester
from vfairness.agents.pipeline_tracker import PipelineTracker
from vfairness.evaluation.vfairness_metrics._statistics import _mannwhitney_two_sided_p
from vfairness.llm.output_analysis import OutputAnalyzer


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# The shared rule


def test_all_tied_unequal_lengths_is_exact_p_one():
    assert _mannwhitney_two_sided_p([1.0] * 30, [1.0] * 31) == 1.0
    assert _mannwhitney_two_sided_p([0.5, 0.5], [0.5, 0.5, 0.5]) == 1.0


def test_a_non_finite_input_is_no_p_value_not_a_reading():
    assert _mannwhitney_two_sided_p([0.1, np.nan, 0.3], [0.2, 0.4, 0.6]) is None
    # Tied apart from one nan is NOT the tied case: the nan value is unknown.
    assert _mannwhitney_two_sided_p([1.0, 1.0, np.nan], [1.0, 1.0]) is None


def test_control_an_ordinary_comparison_is_scipys_own_answer():
    from scipy.stats import mannwhitneyu

    a, b = [0.1, 0.4, 0.35, 0.8, 0.2], [0.9, 0.7, 0.85, 0.6, 0.95]
    expected = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    assert _mannwhitney_two_sided_p(a, b) == pytest.approx(expected)
    assert _mannwhitney_two_sided_p(a, b) < 0.05
    # Perfect separation is a real finding, never folded into the tied rule.
    assert _mannwhitney_two_sided_p([0.0] * 10, [1.0] * 10) < 1e-3


# ActionBiasAnalyzer.analyze_outcomes


def _actions(values):
    return [{"action": "approve", "score": v} for v in values]


def test_action_bias_tied_unequal_lengths_is_a_measured_negative():
    result, _ = _caught(
        ActionBiasAnalyzer().analyze_outcomes, _actions([1.0] * 30), _actions([1.0] * 31), "score"
    )
    assert result.p_value == 1.0
    assert result.is_significant is False


def test_action_bias_a_nan_p_is_could_not_check_and_says_so():
    result, messages = _caught(
        ActionBiasAnalyzer().analyze_outcomes,
        _actions([0.2] * 29 + [np.nan]),
        _actions([0.8] * 30),
        "score",
    )
    assert math.isnan(result.p_value)
    assert result.is_significant is None
    assert any("significance was NOT tested" in m for m in messages), messages


def test_action_bias_control_a_real_difference_is_still_found():
    result, _ = _caught(
        ActionBiasAnalyzer().analyze_outcomes,
        _actions([0.2] * 15 + [0.3] * 15),
        _actions([0.8] * 15 + [0.9] * 15),
        "score",
    )
    assert result.is_significant is True
    assert result.p_value < 1e-6


# CorrespondenceTester.analyze_outcomes, numeric path


def test_correspondence_tied_numeric_unequal_lengths_is_a_measured_negative():
    result, messages = _caught(CorrespondenceTester().analyze_outcomes, [0.5] * 30, [0.5] * 31)
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert not any("produced no p-value" in m for m in messages), messages


def test_correspondence_control_a_real_numeric_difference_is_still_found():
    result, _ = _caught(
        CorrespondenceTester().analyze_outcomes, [0.2] * 15 + [0.3] * 15, [0.7] * 15 + [0.8] * 15
    )
    assert result.is_significant is True


# PipelineTracker


def test_pipeline_tied_stage_is_tested_and_ranked():
    tracker = PipelineTracker(["retrieval", "action"])
    tracker.record_stage("retrieval", np.array([1.0] * 30), np.array([1.0] * 31))
    tracker.record_stage("action", np.array([0.2] * 20 + [0.4] * 20), np.array([0.6] * 40))
    results, _ = _caught(tracker.compute_cumulative)
    by_name = {r.stage_name: r for r in results}
    assert by_name["retrieval"].bias_metrics["p_value"] == 1.0
    _source, messages = _caught(tracker.identify_bias_source)
    assert not any("retrieval" in m and "excluded" in m.lower() for m in messages), messages


def test_pipeline_a_nan_p_on_finite_data_is_named_not_silently_dropped(monkeypatch):
    """No finite input makes scipy 1.18 return nan except the tied case, so the
    other half of the rule is driven by making the shared helper answer None."""
    import vfairness.agents.pipeline_tracker as pt

    monkeypatch.setattr(pt, "_mannwhitney_two_sided_p", lambda a, b: None)
    tracker = PipelineTracker(["retrieval"])
    tracker.record_stage("retrieval", np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]))
    results, messages = _caught(tracker.compute_cumulative)
    assert math.isnan(results[0].bias_metrics["p_value"])
    assert any("retrieval" in m and "returned no p-value" in m for m in messages), messages


# OutputAnalyzer._compare


def test_output_analysis_tied_unequal_lengths_is_assessed_at_p_one():
    result, messages = _caught(
        OutputAnalyzer()._compare, np.array([1.0] * 30), np.array([1.0] * 31), "refusal", "a", "b"
    )
    assert result.p_value == 1.0
    assert result.is_significant is False
    assert result.not_assessed_reason is None
    assert not any("non-finite p-value" in m for m in messages), messages
