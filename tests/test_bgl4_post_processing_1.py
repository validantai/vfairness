"""BGL4 audit of batch A-post_processing-1: the overturns, and their closure.

Every test here asserts the behaviour the graded unit SHOULD have. They were
written as ``xfail(strict=True)`` while the defects stood, so the shared suite
stayed green and an XPASS would say the defect was closed.

CLOSED on 2026-09-27 in the BGL5 fix wave: PP1-A1 through PP1-A6, PP1-A8 and
PP1-A9. Those eight tests are now plain assertions of the corrected behaviour,
with their subject and their measured BEFORE values kept verbatim, and the fix
comments in the source carry the same numbers. The pins with the sabotage
evidence and the over-correction controls are in
``tests/test_bgl5_post_processing_1.py``.

NOTHING IS OPEN in this batch any more. PP1-A7 was the last one and it closed on
2026-09-28: its count is computed in ``_explain_threshold_analysis`` in
``src/vfairness/explainer.py``, a file this batch did not own, so it was handed
back rather than edited from here. That file now buckets every region by
``_region_status()`` and counts only the feasible ones, and the pin with its
sabotage evidence is ``tests/test_bgl5_threshold_explanation_counts_evaluations.py``.

Eight grades in that batch were overturned on 2026-09-27. Each number quoted
below was printed by running the unit.

PP1-A1  ``analyze_calibration_fairness_tradeoff`` (graded PROVEN) computes
        ``base_rate_disparity = max(base_rates.values()) - min(base_rates.values())``.
        Python's builtin ``max``/``min`` do NOT propagate a NaN that is not the
        first element, so ``{'A': 0.5167, 'B': nan}`` collapses to 0.0, a FINITE
        number, and every ``math.isfinite`` guard downstream reads it as
        measured. On 240 rows where group B carries no label at all:
        base_rate_disparity 0.0, ``tradeoff_severity`` "minimal", and the first
        recommendation "GOOD NEWS: Base rates are similar across groups.
        Calibration and fairness can likely be improved together." No warning
        mentions the base-rate comparison. The fix that produced the PROVEN
        grade guards ``not math.isfinite(self.base_rate_disparity)``, and 0.0 is
        finite, so it never fires.

PP1-A2  ``calibration_vs_error_parity`` (graded PROVEN) builds every disparity
        over ``GroupManager(protected_attr, min_group_size=1)``, so a ONE-ROW
        group contributes a base rate of exactly 0.0 or 1.0. Two 100-row groups
        with a TPR gap of 1.00 answer conflict_exists False, because their
        base-rate disparity is 0.0. Append a SINGLE row of a third group with
        label 1 and the same call answers base_rate_disparity 0.5 and
        conflict_exists TRUE: one row flipped a measured "calibration and error
        parity are jointly achievable" into a measured "the impossibility
        theorem binds", with ``not_assessed`` [] and no warning about it. The
        sibling ``impossibility_diagnostics`` in this same file was given
        ``min_group_size=30`` for precisely this, and says so in its docstring.

PP1-A3  ``impossibility_diagnostics`` (graded PROVEN, fabricated False) has the
        same NaN-collapse. On 240 rows where group B holds no label:
        base_rates {'A': 0.55, 'B': nan}, base_rate_disparity 0.0,
        base_rates_differ False, impossibility_applies False,
        calibration_sacrifice_estimate 0.0, n_groups_compared 2, ZERO warnings,
        and the explanation "Base rates are approximately equal across groups.
        In this case, calibration and error rate parity can theoretically be
        achieved simultaneously." Its fewer-than-two-groups gate counts a NaN
        group as a measured one.

PP1-A4  ``mitigation_pareto`` (graded PROVEN) gates its fairness axis on ROW
        count, ``int(masks[g].sum()) >= 10``, not on scored-row count. Two
        100-row groups where group B was never scored (every y_prob NaN):
        available True, baselineGap 0.5, bestGap 0.17 and the summary "The
        selection-rate gap can be cut from 50 to 17 points via "Group thresholds
        (equal opportunity)" at 16.5 points of accuracy cost." Every number rests
        on ``NaN >= threshold`` being False, so group B was decided as 100%
        rejected at every threshold. The only warnings are numpy's "All-NaN slice
        encountered". The sibling ``exposure_parity_rerank`` in this same batch
        REFUSES exactly this input.

PP1-A5  ``recommend_calibration_strategy`` (graded PROVEN) guards
        ``not math.isfinite(ece_disparity)``, and ``calibration_disparity``
        computes ``max(ece_values) - min(ece_values)`` over group values that can
        hold a NaN, so it hands back a finite 0.0. On 400 rows where group B
        holds no label: group ECEs {'A': 0.1636, 'B': nan}, ece_disparity 0.0,
        excluded_groups [], strategy "monitor_only", priority "low",
        disparity_assessed TRUE, not_assessed_groups [], rationale "Calibration
        metrics are acceptable. Monitor for drift but no immediate action
        needed.", zero warnings. Group A's own ECE of 0.1636 is a real and poor
        measurement, reported as acceptable.

PP1-A6  ``ThresholdAnalyzer`` (find_feasible_region / find_optimal_threshold /
        analyze_threshold / analyze_threshold_range / full_analysis, all graded
        PROVEN) never validates y_prob and thresholds it with
        ``(self.y_prob >= t).astype(int)``, so an unscored row is a measured
        rejection. On 200 rows where group B was never scored: at threshold 0.5
        demographic_parity is_satisfied False, violation 0.5, group_metrics
        {'A': 0.5, 'B': 0.0}; find_feasible_region returns status "feasible",
        (0.941, 0.990), n_feasible 6 and n_not_assessed 0; find_optimal_threshold
        returns optimal_threshold 0.9405, is_feasible True,
        n_thresholds_not_assessed 0; and summary() prints
        "demographic_parity: [0.941, 0.990]" with "The feasible region for
        demographic_parity is narrow (0.05)". The counts actively assert that
        nothing was unmeasurable.

PP1-A7  ``ThresholdAnalyzer.get_explanation`` (graded SEMI-PROVEN) reaches
        ``_explain_threshold_analysis``, which reads
        ``feasible = report.feasible_regions`` and then reports ``len(feasible)``
        as the number of SATISFIABLE constraints. That dict holds one entry per
        constraint REQUESTED, including ``(None, None)`` regions with
        ``assessed=False``. On 200 rows of a single group, where all 100 searched
        thresholds were could-not-check, the summary reads "Analysed 20
        threshold(s). 3 constraint(s) are satisfiable within the threshold
        range." with severity "info", directly above its own recommendation
        "NOT ASSESSED: demographic_parity, equalized_odds, equal_opportunity
        could not be evaluated at any searched threshold".

PP1-A8  ``ThresholdAnalysisReport.to_dict`` puts the FeasibleRegion objects into
        the dict unchanged. A FeasibleRegion is a tuple subclass, so a
        NOT ASSESSED region and a MEASURED INFEASIBLE region both serialise to
        ``[null, null]`` and ``assessed``, ``contiguous``, ``n_feasible``,
        ``n_searched`` and ``n_not_assessed`` are all dropped. That is exactly
        the ambiguity the FeasibleRegion type was added to remove, reintroduced
        at the boundary a consumer reads.

PP1-A9  ``ReweightingAnalyzer`` coerces labels with
        ``coerce_to_array(y_true).astype(int)``, which turns a NaN label into an
        int (0 on this platform, INT_MIN on others) behind one numpy
        "invalid value encountered in cast" warning. On 160 rows where 40 carry
        no label, ``analyze_method`` reports accuracy 0.75 with n_decided 160 and
        n_rows 160, and original_ece 0.2 with original_ece_rows_used 160: the
        coverage fields assert that all 160 rows were graded. Accuracy over the
        120 rows that DO have a label is 1.0. ``_compute_accuracy`` and
        ``_compute_calibration_error`` were both hardened to exclude rows nothing
        SCORED; the row nothing LABELLED is not covered.
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from vfairness.exceptions import InvalidDataError
from vfairness.post_processing.calibration.tradeoffs import (
    analyze_calibration_fairness_tradeoff,
    calibration_vs_error_parity,
    impossibility_diagnostics,
    mitigation_pareto,
    recommend_calibration_strategy,
)
from vfairness.post_processing.reweighting.analyzer import ReweightingAnalyzer
from vfairness.post_processing.threshold_optimization.analyzer import (
    FeasibleRegion,
    ThresholdAnalyzer,
    _region_to_dict,
)


def _unlabelled_second_group(n_per_group: int = 120):
    """Group A labelled, group B carrying no ground truth at all."""
    labels_a = np.tile(np.array([1.0, 0.0, 1.0, 1.0, 0.0]), n_per_group // 5)
    y = np.concatenate([labels_a, np.full(n_per_group, np.nan)])
    p = np.concatenate([np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), n_per_group // 5)] * 2)
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _unscored_second_group(n_per_group: int = 100):
    """Group A scored across the grid, group B never scored."""
    scored = np.linspace(0.02, 0.98, n_per_group)
    p = np.concatenate([scored, np.full(n_per_group, np.nan)])
    y = np.concatenate([(scored >= 0.5).astype(int), np.tile(np.array([1, 0]), n_per_group // 2)])
    g = np.array(["A"] * n_per_group + ["B"] * n_per_group)
    return y, p, g


def _one_group_two_hundred():
    p = np.tile(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), 40)
    y = np.tile(np.array([0, 0, 1, 1, 1]), 40)
    return y, p, np.array(["A"] * 200)


def test_a_base_rate_nobody_measured_is_not_a_minimal_tradeoff():
    """PP1-A1, CLOSED. max()-min() over {'A': 0.5167, 'B': nan} was a finite 0.0,
    so the isfinite guard never fired and an unmeasured base-rate comparison
    graded 'minimal' with a GOOD NEWS recommendation. `_measured_spread` drops the
    unmeasured values before the spread and answers NaN below two of them."""
    y, p, g = _unlabelled_second_group()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = analyze_calibration_fairness_tradeoff(y, p, g)

    assert not math.isfinite(result.base_rate_disparity)
    assert result.tradeoff_severity == "not assessed"
    assert not [r for r in result.recommendations if "GOOD NEWS" in r]


def test_one_row_of_a_third_group_does_not_settle_the_impossibility():
    """PP1-A2, CLOSED. min_group_size=1 let a ONE-ROW group set
    base_rate_disparity to 0.5 and flip conflict_exists from a measured False to a
    measured True, with not_assessed empty. The gate is 30 now, matching the
    sibling impossibility_diagnostics, and the excluded group is named."""
    y = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] * 50 + [0] * 50)])
    p = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
    g = np.array(["A"] * 100 + ["B"] * 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        two_groups = calibration_vs_error_parity(y, p, g)
        with_one_row = calibration_vs_error_parity(
            np.concatenate([y, [1]]), np.concatenate([p, [0.9]]), np.concatenate([g, ["C"]])
        )

    assert two_groups["conflict_exists"] is False
    # One row may not turn that measured False into a measured True, and if the
    # row is used at all it has to be disclosed.
    assert with_one_row["conflict_exists"] is not True or "C" in str(with_one_row["not_assessed"])
    assert with_one_row["base_rate_disparity"] == pytest.approx(
        two_groups["base_rate_disparity"], abs=1e-12
    ) or "C" in str(with_one_row["not_assessed"])
    # Closed by exclusion rather than by disclosure inside not_assessed, so the
    # disclosure is asserted where it actually landed.
    assert with_one_row["excluded_groups"] == ["C"]


def test_a_group_with_no_measured_base_rate_is_not_an_equal_base_rate():
    """PP1-A3, CLOSED. The fewer-than-two-groups gate counted a NaN base rate as a
    measured one, so a group nobody labelled read as an equal base rate and the
    theorem was reported not to apply. The gate counts MEASURED rates now, and
    `groups_without_measured_base_rate` names the rest."""
    y, p, g = _unlabelled_second_group()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnosis = impossibility_diagnostics(y, p, g)

    assert not math.isfinite(diagnosis["base_rate_disparity"])
    assert diagnosis["base_rates_differ"] is None
    assert diagnosis["impossibility_applies"] is None
    assert "NOT ASSESSED" in diagnosis["explanation"]
    assert diagnosis["groups_without_measured_base_rate"] == ["B"]


def test_a_group_that_was_never_scored_is_not_a_selection_rate_gap():
    """PP1-A4, CLOSED. The two-group gate counted ROWS, not scored rows, so a
    group that was never scored was decided as 100% rejected and the
    selection-rate gap that produced was reported as measured and mitigable.
    Unscored rows leave the analysis before any group is measured."""
    _, p, g = _unscored_second_group()
    y = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = mitigation_pareto(y, p, g)

    assert result["available"] is False
    assert "score" in result["reason"] or "scored" in result["reason"]
    assert result["nRowsUnscored"] == 100


def test_an_unlabelled_group_is_not_acceptable_calibration():
    """PP1-A5, CLOSED. calibration_disparity handed back a finite 0.0 from group
    ECEs {'A': 0.1636, 'B': nan}, so the isfinite guard never fired and a real
    0.1636 read as 'Calibration metrics are acceptable'. The wrapper now also
    requires two MEASURED group ECEs, because two are needed to be between."""
    y, p, g = _unlabelled_second_group(n_per_group=200)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recommendation = recommend_calibration_strategy(y, p, g, context="lending")

    assert recommendation.disparity_assessed is False
    assert recommendation.priority != "low"
    assert "NOT ASSESSED" in recommendation.rationale
    assert recommendation.not_assessed_groups == ["B"]


def test_a_sweep_over_unscored_rows_is_not_a_measured_feasible_region():
    """PP1-A6, CLOSED. ThresholdAnalyzer cast an unscored row to a hard 0
    decision, so a group nobody scored produced a selection rate of 0.0, a
    measured feasible region (status "feasible", (0.9405, 0.990), n_feasible 6)
    and n_not_assessed 0, plus find_optimal_threshold's is_feasible True with
    n_thresholds_not_assessed 0.

    The SUBJECT is unchanged: no measured feasible region may come out of a sweep
    over rows nobody scored. The MECHANISM of the closure is a refusal at
    construction rather than a per-threshold third state, because each of
    analyze_threshold, find_optimal_threshold and find_feasible_region thresholds
    y_prob in its own body, so one validation above all three closes every one of
    them and no object exists to answer with. The measured-but-unmeasurable sweep
    (one group, every threshold could-not-check) keeps its three-state answer and
    is pinned in tests/test_bgl3_post_processing_1.py.
    """
    y, p, g = _unscored_second_group()

    with pytest.raises(InvalidDataError, match="100 NaN value"):
        ThresholdAnalyzer(y, p, g)


# PP1-A7 IS CLOSED (2026-09-28), so the strict xfail is gone rather than left to
# xpass. It was handed back from this batch because the count lives in
# src/vfairness/explainer.py, which the post_processing-1 batch did not own. The
# change asked for in the old xfail reason is the change that was made:
# _explain_threshold_analysis now buckets every region by _region_status() and
# counts only the feasible ones, so a constraint that was never assessed is named
# in its own clause instead of being reported as satisfiable. Its pin, sabotage
# and over-correction control are in
# tests/test_bgl5_threshold_explanation_counts_evaluations.py (7 tests).
def test_the_explanation_does_not_call_an_unassessed_constraint_satisfiable():
    y, p, g = _one_group_two_hundred()
    analyzer = ThresholdAnalyzer(y, p, g)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = analyzer.get_explanation()

    payload = explanation.to_dict()
    assert "3 constraint(s) are satisfiable" not in payload["summary"]
    assert "0 constraint(s) are satisfiable" in payload["summary"] or "NOT ASSESSED" in str(
        payload["summary"]
    )


def test_the_serialised_region_separates_not_assessed_from_infeasible():
    """PP1-A8, CLOSED. to_dict serialised a FeasibleRegion as a bare pair, so NOT
    ASSESSED and MEASURED INFEASIBLE both became [null, null] at the boundary a
    consumer reads. Each region is a dict carrying `status` now, and the attribute
    itself still behaves as the pair every existing caller unpacks."""
    y, p, g = _one_group_two_hundred()
    analyzer = ThresholdAnalyzer(y, p, g)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyzer.full_analysis(n_thresholds=5, constraints=["demographic_parity"])

    not_assessed = report.to_dict()["feasible_regions"]["demographic_parity"]
    measured_infeasible = _region_to_dict(FeasibleRegion(None, None, assessed=True, n_searched=100))

    assert json.dumps(not_assessed) != json.dumps(measured_infeasible)
    assert not_assessed["status"] == "not_assessed"
    assert measured_infeasible["status"] == "infeasible"


def test_a_row_with_no_label_is_not_graded():
    """PP1-A9, CLOSED. astype(int) turned a NaN label into a real label, so 40
    unlabelled rows were graded and n_decided / n_rows asserted full coverage of
    them. They are excluded now, and the count is published."""
    labelled_scores = np.concatenate([np.full(60, 0.9), np.full(60, 0.1)])
    p = np.concatenate([labelled_scores, np.full(40, 0.5)])
    y = np.concatenate([np.ones(60), np.zeros(60), np.full(40, np.nan)])
    g = np.array(["A"] * 60 + ["B"] * 60 + ["C"] * 40)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analyzer = ReweightingAnalyzer(y, p, g)
        result = analyzer.analyze_method("multiplicative")

    performance = result.original_performance
    # Accuracy over the rows that carry a label is 1.0, and the coverage count
    # may not claim the 40 unlabelled rows.
    assert performance["accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert performance["n_decided"] == 120
    assert performance["n_rows_unlabelled_excluded"] == 40
