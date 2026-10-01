"""BGL4 AUDIT of batch A-evaluation-1: the overturns, as executable evidence.

This file was written by an AUDITOR. It changed nothing in the library, and the
fixer who closed the findings changed nothing here either except the four
assertions themselves, which now hold the corrected behaviour on the same inputs
(see the paragraph below). Each test below demonstrates one grade in
``/tmp/claude-501/bgl/audit_batches2/A-evaluation-1.json`` that did NOT survive
being attacked, and every number in a docstring was produced by running the unit
on this repo on 2026-09-27.

The defect demonstrations WERE ``xfail(strict=True)``, so that the day one of
them was fixed the strict xfail became a FAILURE and the finding could not be
closed silently. That day is 2026-09-27: all four are now CLOSED, the marks are
gone, and each one below asserts the corrected behaviour on the same subject and
the same input it was written for. The docstrings keep the pre-fix measurement,
because it is the evidence that the assertion above it is load-bearing.

The fixes, and the pins that hold them with their own sabotages, are in
``tests/test_bgl5_evaluation_1.py``:

    improvement_amount    an is_measured gate above the direction dispatch in
                          evaluation/vfairness_metrics/_metric_direction.py
    explain_metric        _for_grading, the guard above the _get_explainer
    get_explanations      dispatch, in evaluation/vfairness_metrics/analyzer.py
    preview_palette       _palette_preview_labels, so the figure itself names the
                          palette it drew, in .../visualization.py

THE COMMON ROOT of the first three: infinity is the OTHER HALF of the NaN hole,
and this repo already knows it. ``_triage.is_measured`` refuses ``inf`` by name,
``check_threshold`` was fixed for ``inf`` at READINESS-6 (2026-09-10), and
``FairExplAIner._finite_or_none`` says in its own docstring that "NaN/inf mean
nothing was measured". Three surfaces graded PROVEN on 2026-09-27 still graded an
infinity, and the tests that were the evidence for them only ever tried NaN. All
three refuse it now, and each one refuses it at ONE place that both of its call
sites reach, rather than at the site the auditor happened to probe.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._metric_direction import (
    check_threshold,
    improvement_amount,
)


def _two_group_analyzer() -> FairnessAnalyzer:
    rng = np.random.default_rng(0)
    y = (rng.random(80) < 0.5).astype(int)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y, y, groups)


# ===========================================================================
# 1. improvement_amount: graded PROVEN, and it converted an UNMEASURABLE value
#    into the most reassuring number there is. CLOSED 2026-09-27.
# ===========================================================================


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), True, "0.9"],
    ids=["inf", "minus_inf", "bool", "numeric_string"],
)
def test_improvement_amount_refuses_an_unmeasurable_value_instead_of_numbering_it(value):
    """Executed 2026-09-27, all four with a KNOWN direction so the None path is not involved:

        improvement_amount('disparate_impact_ratio', inf, 0.5)   -> inf
        improvement_amount('demographic_parity_difference', 0.5, inf) -> inf
        improvement_amount('disparate_impact_ratio', True, 0.5)  -> 0.5
        improvement_amount('disparate_impact_ratio', '0.9', 0.5) -> 0.4

    Every one of those inputs is refused by this repo's canonical predicate
    (``is_measured``) and by ``check_threshold`` in the SAME module, which returns
    COULD_NOT_CHECK for each. A caller enforcing an improvement bound reads
    ``+inf >= margin`` as "improved" and ``-inf`` evades the degradation check
    (``-(-inf) > margin`` is the only one that fires, and it fires on the strict
    side). ``operations/cicd/gate.py`` puts whatever comes back in
    ``MetricEvaluation.improvement``, which ``to_dict()`` publishes.

    The named evidence,
    ``test_improvement_amount_returns_none_rather_than_a_direction_it_guessed``,
    covers exactly two classes: an unknown direction (None) and NaN (NaN). This
    third class returned a plain number that no caller could tell from a
    measurement.

    CLOSED 2026-09-27 by an ``is_measured`` gate placed ABOVE the direction
    dispatch, so it holds for both branches: all four now return NaN, which the
    existing NaN convention of this function already marks as "nothing to
    compare", while ``None`` still means "no known better-direction" and stays
    distinguishable. Re-measured after the fix, in the same order: nan, nan, nan,
    nan. Sabotage and controls in ``tests/test_bgl5_evaluation_1.py``.
    """
    assert not is_measured(value), "fixture error: this input must be unmeasurable"
    assert check_threshold("disparate_impact_ratio", value, 0.8)[0].value == "could_not_check"
    got = improvement_amount("disparate_impact_ratio", value, 0.5)
    assert got is None or (isinstance(got, float) and math.isnan(got)), (
        f"an unmeasurable value produced the improvement {got!r}, which a caller "
        f"cannot tell from a measured one"
    )


# ===========================================================================
# 2. FairnessAnalyzer.explain_metric: graded PROVEN. An infinite ratio was graded
#    "Excellent", severity info, with the recommendation written for a PASS.
#    CLOSED 2026-09-27.
# ===========================================================================


def test_explain_metric_does_not_call_an_infinite_ratio_near_perfect_parity():
    """Executed 2026-09-27 at the public entry point:

        analyzer.explain_metric('demographic_parity_ratio', float('inf'))
          value       inf
          severity    'info'
          evaluation  'Excellent! The ratio of inf indicates near-perfect parity
                       between groups.'
          recommend.  'Continue monitoring this metric as part of regular fairness
                       audits. Document your fairness practices for compliance ...'
          warnings    none

    ``inf`` reaches a ratio from a denominator of zero, i.e. a group with no
    selections at all: the most extreme unfairness the data can express arriving
    as the most reassuring verdict the surface can write. The root line is
    ``explainer._could_not_check_reason``, which tests only ``np.isnan(value)``,
    while the same class's ``_finite_or_none`` refuses inf and says so.
    ``check_threshold('demographic_parity_ratio', inf, 0.8)`` is COULD_NOT_CHECK,
    so the report's own assessment block and its explanation block disagreed about
    the same value.

    CLOSED 2026-09-27. ``_could_not_check_reason`` lives in ``explainer.py``,
    which belongs to another batch, so the guard went where BOTH of this
    analyzer's explanation surfaces arrive: ``analyzer._for_grading``, above the
    ``_get_explainer()`` dispatch. Re-measured after the fix, same call: severity
    'could_not_check', evaluation "COULD NOT CHECK: demographic_parity_ratio could
    not be measured on this data (infinite: no comparison can grade it), so no
    threshold was applied to it. ...", value still inf, and the passing
    recommendation gone. The narrower fix in ``explainer.py`` is recorded for that
    batch; this one holds whether or not it lands.
    """
    analyzer = _two_group_analyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = analyzer.explain_metric("demographic_parity_ratio", float("inf"))
    assert explanation.severity == "could_not_check", (
        f"an infinite ratio was graded {explanation.severity!r}: {explanation.evaluation}"
    )
    assert "COULD NOT CHECK" in explanation.evaluation
    # The REASON, in the artifact, not only the state: an infinity is a different
    # repair from a NaN (a zero denominator, i.e. a group with no selections at
    # all) and the report's assessment block names it the same way.
    assert "infinite" in explanation.evaluation, explanation.evaluation
    assert "Continue monitoring" not in explanation.recommendation
    assert caught or True


# ===========================================================================
# 3. FairnessAnalyzer.get_explanations: graded PROVEN. For values that are
#    unmeasurable WITHOUT being NaN it asserted a critical disparity nobody
#    measured, and dropped the NOT GRADED clause its own pin exists for.
#    CLOSED 2026-09-27.
# ===========================================================================


def test_get_explanations_does_not_assert_a_disparity_it_never_compared():
    """Executed 2026-09-27 on a two-group regression frame with one infinite
    prediction (which ``validate_inputs`` admits in silence, so every metric
    comes back ``inf``):

        get_report()['assessment']  fairness_score None, four metrics
                                    NOT_ASSESSABLE, reason 'mae_parity_difference
                                    could not be measured on this data (infinite:
                                    no comparison can grade it)'
        get_explanations()['summary']
            'Overall Fairness Score: NOT AVAILABLE (could not check) (0 passed,
             0 failed, 4 not assessable) | CRITICAL: 4 metric(s) show critical
             disparities requiring immediate attention. | COULD NOT CHECK: ...'

    Two findings in one string. The CRITICAL clause was a verdict on four metrics
    the same run says were never compared to a threshold, and the 'NOT GRADED:
    ... Do not read their absence from the counts above as a pass' clause that
    ``test_get_explanations_summary_refuses_to_read_as_a_pass`` pins was ABSENT
    here, because that pin was only ever exercised with NaN.

    CLOSED 2026-09-27 by the SAME guard as explain_metric, above the same
    ``_get_explainer()`` dispatch: the summary is built from the severity of each
    per-metric card, so refusing the cards is what removes the verdict, and there
    is no second rule for the summary. Re-measured after the fix:
    'Overall Fairness Score: NOT AVAILABLE (could not check) (0 passed, 0 failed,
    4 not assessable) | NOT GRADED: 4 metric(s) were never compared to a
    threshold, so they are neither passing nor failing. Do not read their absence
    from the counts above as a pass. | COULD NOT CHECK: ...'
    """
    rng = np.random.default_rng(2)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    y = rng.normal(10, 2, 80)
    pred = y + rng.normal(0, 0.5, 80)
    pred[0] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = FairnessAnalyzer(y, pred, groups)
        summary = analyzer.get_explanations()["summary"]
    assert "CRITICAL" not in summary.upper(), (
        f"a critical disparity was asserted for metrics nobody compared: {summary}"
    )
    assert "NOT GRADED" in summary, f"the not-graded disclosure is missing: {summary}"


# ===========================================================================
# 4. preview_palette: graded PROVEN and FIXED. The warning was added; the
#    FIGURE still said it was showing a palette that does not exist.
#    CLOSED 2026-09-27.
# ===========================================================================


def test_preview_palette_figure_does_not_label_another_palette_with_the_asked_name():
    """Executed 2026-09-27, AFTER the fix:

        preview_palette('colorblind_safe_v2')
          warning  "_get_palette: 'colorblind_safe_v2' is not a known
                    visualization style ... Falling back to 'modern'"
          title    '<b>Colorblind_Safe_V2 Color Palette</b>'
          swatches the MODERN palette
          the figure itself contains none of 'not a known', 'fallback',
          'modern palette' or 'does not exist'

    The fix put the disclosure in a channel the artifact does not carry: a saved
    PNG, an exported HTML or a notebook with warnings filtered showed the modern
    palette titled with the requested name, exactly as before. The grade's own
    words for the defect ("it labels the fallback as the request") still described
    the figure.

    CLOSED 2026-09-27: the title is built from the palette that was DRAWN and the
    substitution travels in the figure. Re-measured, same call: title
    '<b>Modern Color Palette</b><br><span style=...>requested
    'colorblind_safe_v2', which is not a known style: these are the MODERN
    colours, so nothing here shows 'colorblind_safe_v2'</span>', and the warning
    is still emitted as well. The six documented styles are unchanged and silent.
    """
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from vfairness.evaluation.vfairness_metrics.visualization import PALETTES, preview_palette

    unknown = "colorblind_safe_v2"
    assert unknown not in PALETTES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = preview_palette(unknown)
    blob = (
        str(fig.to_dict())
        if hasattr(fig, "to_dict")
        else str([t.get_text() for t in getattr(fig, "texts", [])])
    )
    assert unknown not in blob.lower() or any(
        token in blob.lower() for token in ("not a known", "falling back", "does not exist")
    ), (
        "the figure is titled with the requested style and carries no disclosure "
        "that the colours belong to another one"
    )
    # And the headline names the palette the swatches ACTUALLY are, which is the
    # half a disclaimer alone would not fix.
    assert "modern color palette" in blob.lower(), blob[:300]


# ===========================================================================
# 5. compute_all_metrics: the pin could not see the defect it names. NOT a defect
#    in the library, a defect in the evidence, so this test PASSES and records
#    it. See the audit report for the executed sabotage.
#
#    The rule reproduced below is STILL the rule in
#    tests/test_analyzer_flagship_entry_points.py, which is not this batch's file
#    and was left alone. What replaces it as evidence is
#    tests/test_bgl5_evaluation_1.py::TestComputeAllMetricsRefusesEachMetricByName,
#    which asserts the measured/unmeasured partition PER METRIC and PER FRAME and
#    reads the disclosure for the names it carries, so the fabricated 0.0 below
#    reddens it.
# ===========================================================================


def test_the_flagship_refusal_rule_accepts_a_fabricated_zero():
    """``test_compute_all_metrics_refuses_where_nothing_is_measurable`` asserts
    ``msgs or not measured``: ANY warning satisfies it.

    Measured 2026-09-27: all four degenerate frames in that test emit at least
    one warning (single_group 1, one_label 3, n_equals_2 8, empty 3), and the
    single_group warning is about PRECISION, a different metric. Sabotaging
    ``classification.demographic_parity_difference`` so the fewer-than-two-groups
    branch returns 0.0 instead of NaN put ``{'demographic_parity_difference':
    0.0, ...}`` on a single-group frame, which is the canvas for perfect parity,
    and ``tests/test_analyzer_flagship_entry_points.py`` stayed 12 passed.

    This test reproduces the rule on that fabricated payload, so the hole is
    recorded rather than argued.
    """
    fabricated = {
        "demographic_parity_difference": 0.0,
        "demographic_parity_ratio": 1.0,
        "equalized_odds_difference": 0.0,
    }
    unrelated_warning = ["Precision is undefined for group(s) ['a'] ..."]
    numeric = [v for v in fabricated.values() if isinstance(v, (int, float))]
    measured = [v for v in numeric if not (isinstance(v, float) and math.isnan(v))]
    rule_holds = bool(unrelated_warning) or not measured
    assert rule_holds, "if this ever fails the rule has been strengthened, which is the fix"
    assert measured == [0.0, 1.0, 0.0], "every metric is a fabricated measurement here"
