"""End-to-end pins for "a comparison that never happened must never certify as fair".

The release-blocking defect this file guards:

    With the DEFAULT min_group_size=30, a 25-person minority group that is NEVER
    selected, against a majority selected at 56 percent, returned
    demographic_parity_difference 0.0 and demographic_parity_ratio 1.0. That is
    the exact INVERSE of the truth (0.56 and 0.0): maximal unfairness reported as
    perfect parity. A warning was emitted, but the VALUE was a false certificate,
    and the report, the verdict, the score and the prose all inherited it, ending
    at "fairness_score 1.0, all metrics within acceptable thresholds".

The convention is enforced here for ALL THREE task types. It was written for
classification, regression mirrored it, and RANKING did not obey it at all until
2026-08-28: every ranking metric returned the perfect sentinel for a dropped
group, none of them warned, and ``attention_weighted_rank_fairness`` also
attached ``is_fair=True`` to its sentinel, which the SVG adapter printed onto a
green PASS badge. Covering two task types is what let the third one drift, so the
ranking family is asserted here beside the other two rather than only in
``tests/test_ranking.py``.

The metric layer alone is not enough to pin, because the bug's damage was done
downstream. Every layer is asserted here:

    1. the metric value                      -> NaN, never 0.0 / 1.0
    2. the MetricResult verdict / is_fair    -> insufficient_evidence, not fair
    3. the report assessment block           -> NOT_ASSESSABLE, nothing "passed"
    4. the fairness_score                    -> None, not a number
    5. explain_fairness_report prose         -> could-not-check, not "all within"

and two controls prove the fix did not simply blanket-NaN the engine:

    A. min_group_size=20 keeps the same minority group and measures the REAL
       0.56 / 0.0 disparity, reporting it as UNFAIR.
    B. ordinary healthy two-group data is completely unchanged and still passes.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics import classification as C
from vfairness.evaluation.vfairness_metrics import ranking as K
from vfairness.evaluation.vfairness_metrics import regression as R
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer, MetricResult
from vfairness.evaluation.vfairness_metrics.explainer import explain_fairness_report
from vfairness.evaluation.vfairness_metrics.report import (
    classification_fairness_report,
    regression_fairness_report,
)

# The incident data. 100-person majority selected at 56 percent; 25-person
# minority (below the default min_group_size=30) never selected at all.
MAJ_N, MIN_N = 100, 25
Y_PRED = np.concatenate([np.array([1] * 56 + [0] * 44), np.zeros(MIN_N, dtype=int)])
Y_TRUE = np.concatenate([np.array([1] * 50 + [0] * 50), np.array([1] * 12 + [0] * 13)])
SENS = np.array(["maj"] * MAJ_N + ["min"] * MIN_N)

# The TRUE disparity, recoverable by lowering min_group_size to 20.
TRUE_DP_DIFFERENCE = 0.56
TRUE_DP_RATIO = 0.0


def _quiet(fn, *args, **kwargs):
    """Run fn with the dropped-group warnings silenced (they are asserted
    separately); the point of these tests is the VALUE, not the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Layer 1: the metric value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        C.demographic_parity_difference,
        C.demographic_parity_ratio,
        C.disparate_impact_ratio,
        C.equal_opportunity_difference,
        C.equalized_odds_difference,
        C.predictive_parity_difference,
        C.fpr_parity_difference,
        C.fnr_parity_difference,
        C.accuracy_parity_difference,
        C.negative_predictive_value_difference,
    ],
    ids=lambda f: f.__name__,
)
def test_dropped_minority_makes_every_classification_metric_unmeasurable(fn):
    got = _quiet(fn, Y_TRUE, Y_PRED, SENS)
    assert math.isnan(got), f"{fn.__name__} returned {got!r} for a comparison that never ran"
    # Spell out the refusal: neither "perfect" sentinel may come back here.
    assert got != 0.0, f"{fn.__name__} returned the perfect-difference sentinel 0.0"
    assert got != 1.0, f"{fn.__name__} returned the perfect-ratio sentinel 1.0"


def test_dropped_group_is_disclosed_as_a_warning_too():
    """The NaN is the machine-readable signal; the warning is the human one.
    Both must be present: the original defect emitted the warning and still
    returned a false value, so the warning alone was never sufficient."""
    with pytest.warns(UserWarning, match="min_group_size"):
        C.demographic_parity_difference(Y_TRUE, Y_PRED, SENS)


def test_single_group_selection_rate_matrix_has_no_adverse_impact_verdict():
    """A four-fifths screen over one group is not "no adverse impact", it is no
    screen at all."""
    out = _quiet(
        C.selection_rate_disparity_matrix,
        np.array([1, 0] * 30),
        np.array(["A"] * 60),
    )
    assert math.isnan(out["min_ratio"])
    assert math.isnan(out["max_difference"])


# ---------------------------------------------------------------------------
# Layer 2: MetricResult (value + verdict + assessability)
# ---------------------------------------------------------------------------


def test_metric_result_reports_insufficient_evidence_not_fair():
    analyzer = FairnessAnalyzer(Y_TRUE, Y_PRED, SENS, task_type="classification")
    for name in (
        "demographic_parity_difference",
        "equalized_odds_difference",
        "equal_opportunity_difference",
    ):
        result = _quiet(getattr(analyzer, name), include_ci=True, n_bootstrap=200, random_state=1)
        assert result.verdict == "insufficient_evidence", name
        assert result.is_fair is False, name
        assert result.assessable is False, name
        # A consumer reading only the serialized form must see it too.
        assert result.to_dict()["assessable"] is False, name
        assert result.to_dict()["is_fair"] is False, name


def test_metric_result_default_does_not_claim_fairness():
    """A MetricResult nobody computed must not read as a pass. is_fair used to
    default to True while verdict defaulted to "not_computed"."""
    blank = MetricResult(metric_name="x", value=float("nan"))
    assert blank.is_fair is False
    assert blank.verdict == "not_computed"
    assert blank.assessable is False


def test_metric_result_assessable_tracks_the_value():
    assert MetricResult(metric_name="x", value=0.03).assessable is True
    assert MetricResult(metric_name="x", value=float("nan")).assessable is False
    assert MetricResult(metric_name="x", value=float("inf")).assessable is False


# ---------------------------------------------------------------------------
# Layers 3 and 4: the report assessment block and the fairness score
# ---------------------------------------------------------------------------


def test_report_marks_every_metric_not_assessable_and_scores_nothing():
    report = _quiet(classification_fairness_report, Y_TRUE, Y_PRED, SENS)
    assessment = report["assessment"]

    assert assessment["assessable"] is False
    assert assessment["passed_metrics"] == [], "a vacuous metric must never be a pass"
    assert assessment["failed_metrics"] == []
    assert assessment["not_assessable_metrics"], "the metrics must be disclosed somewhere"
    assert {m["metric"] for m in assessment["not_assessable_metrics"]} == set(report["metrics"])
    assert all(m["status"] == "NOT_ASSESSABLE" for m in assessment["not_assessable_metrics"])

    # The score is the third state, not a number. 1.0 was the false certificate;
    # 0.0 would be the opposite lie (measured, and everything failed).
    assert assessment["fairness_score"] is None

    # The excluded group is named, not silently dropped.
    excluded = {g["group"] for g in assessment["insufficient_evidence_groups"]}
    assert "min" in excluded

    assert "NOT ASSESSABLE" in assessment["summary"]


def test_report_summary_never_claims_metrics_are_within_thresholds():
    report = _quiet(classification_fairness_report, Y_TRUE, Y_PRED, SENS)
    assert "within thresholds" not in report["assessment"]["summary"]


RANK_POSITIONS = np.arange(12)
RANK_GROUPS = np.array(["A"] * 8 + ["B"] * 4)

# What the ranking family reported before the fix, with the default
# min_group_size=5 dropping group B: the exact inverse of the finding.
RANK_SENTINELS = {
    "exposure_parity_difference": 0.0,
    "exposure_parity_ratio": 1.0,
    "normalized_discounted_kl_divergence": 0.0,
}


@pytest.mark.parametrize("name, sentinel", sorted(RANK_SENTINELS.items()))
def test_dropped_group_makes_every_ranking_metric_unmeasurable(name, sentinel):
    """The THIRD task type. Ranking obeyed none of this until 2026-08-28."""
    got = _quiet(getattr(K, name), RANK_POSITIONS, RANK_GROUPS)
    assert math.isnan(got), f"{name} returned {got!r} for a comparison that never ran"
    assert got != sentinel, f"{name} returned the perfect-parity sentinel {sentinel}"


def test_ranking_verdict_is_could_not_check_not_fair():
    """``attention_weighted_rank_fairness`` is the only metric in the library
    that carries its own is_fair, and it set it to True off a sentinel."""
    result = _quiet(K.attention_weighted_rank_fairness, RANK_POSITIONS, RANK_GROUPS)

    assert math.isnan(result.value)
    assert result.is_fair is None, "could-not-check must not collapse into a pass"
    assert result.is_fair is not True
    assert result.is_fair is not False, "nor into a fail: nothing was measured"


@pytest.mark.parametrize(
    "fn",
    [
        K.exposure_parity_difference,
        K.exposure_parity_ratio,
        K.normalized_discounted_kl_divergence,
        K.attention_weighted_rank_fairness,
        K.get_ranking_group_metrics,
    ],
    ids=lambda f: f.__name__,
)
def test_ranking_discloses_the_dropped_group_as_a_warning_too(fn):
    """Classification warned and still returned a false value; ranking did not
    even warn. Both halves are required, in all three families."""
    with pytest.warns(UserWarning, match="min_group_size"):
        fn(RANK_POSITIONS, RANK_GROUPS)


def test_ranking_canvas_does_not_certify_a_ranking_it_never_graded():
    """Layer 3 for the ranking family: the SVG an auditor actually reads.

    This is the surface the defect reached. The metric handed the adapter
    is_fair=True, the adapter's "the metric module's own verdict wins" branch
    turned it into a green PASS badge, and the page read FAIR for a ranking in
    which the dropped group received 12 percent of the attention.
    """
    pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")
    from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg

    result = _quiet(K.attention_weighted_rank_fairness, RANK_POSITIONS, RANK_GROUPS)
    group_metrics = _quiet(K.get_ranking_group_metrics, RANK_POSITIONS, RANK_GROUPS)
    svg = ranking_fairness_to_svg([result], group_metrics)

    assert ">FAIR<" not in svg, "a ranking nobody graded was certified as fair"
    assert ">NOT ASSESSED<" in svg
    assert "NOT MEASURED" in svg
    assert "All ranking fairness metrics pass" not in svg
    assert "certifies nothing about them" in svg
    # No fabricated number beside the withheld verdict, and no bare "nan" either.
    assert ">nan<" not in svg


def test_control_ranking_keeps_measuring_when_every_group_qualifies():
    """CONTROL for the ranking family, both directions of normal operation.

    Same 12 items with the gate lowered to 4: the real disparity comes back as
    a number and as a verdict of False, which is what proves the fix is not a
    blanket NaN.
    """
    assert _quiet(
        K.exposure_parity_difference, RANK_POSITIONS, RANK_GROUPS, min_group_size=4
    ) == pytest.approx(0.20936408399822987)
    assert _quiet(
        K.exposure_parity_ratio, RANK_POSITIONS, RANK_GROUPS, min_group_size=4
    ) == pytest.approx(0.5763430618481035)
    assert _quiet(
        K.normalized_discounted_kl_divergence, RANK_POSITIONS, RANK_GROUPS, min_group_size=4
    ) == pytest.approx(0.5782551230280996)

    measured = _quiet(
        K.attention_weighted_rank_fairness, RANK_POSITIONS, RANK_GROUPS, min_group_size=4
    )
    assert measured.value == pytest.approx(0.9411945920182281)
    assert measured.is_fair is False


def test_regression_report_mirrors_the_classification_behaviour():
    rng = np.random.default_rng(7)
    y_true = np.concatenate([rng.normal(10, 2, MAJ_N), rng.normal(40, 2, MIN_N)])
    y_pred = np.concatenate([rng.normal(10, 2, MAJ_N), rng.normal(10, 2, MIN_N)])
    report = _quiet(regression_fairness_report, y_true, y_pred, SENS)
    assessment = report["assessment"]
    assert assessment["assessable"] is False
    assert assessment["passed_metrics"] == []
    assert assessment["fairness_score"] is None
    for name in ("mae_parity_difference", "rmse_parity_difference", "mean_prediction_difference"):
        assert math.isnan(_quiet(getattr(R, name), y_true, y_pred, SENS)), name


# ---------------------------------------------------------------------------
# Layer 5: the prose a human actually reads
# ---------------------------------------------------------------------------


def test_explainer_says_could_not_check_instead_of_all_within_thresholds():
    report = _quiet(classification_fairness_report, Y_TRUE, Y_PRED, SENS)
    summary = _quiet(explain_fairness_report, report)["summary"]

    assert "COULD NOT CHECK" in summary
    assert "All metrics are within acceptable thresholds" not in summary
    # Nor the softened wording used for a genuine all-pass run.
    assert "within acceptable thresholds" not in summary
    # And it must not print a percentage score for a score that does not exist.
    assert "100.0%" not in summary


def test_explainer_on_empty_report_does_not_certify_fairness():
    """Degenerate input straight into the public convenience function."""
    empty = {
        "task_type": "classification",
        "metrics": {},
        "assessment": {
            "fairness_score": None,
            "assessable": False,
            "passed_metrics": [],
            "failed_metrics": [],
            "not_assessable_metrics": [],
            "insufficient_evidence_groups": [],
            "summary": "NOT ASSESSABLE",
        },
    }
    summary = explain_fairness_report(empty)["summary"]
    assert "COULD NOT CHECK" in summary
    assert "within acceptable thresholds" not in summary


# ---------------------------------------------------------------------------
# CONTROL A: the same data, with the minority group kept, still measures the
# real disparity. This is what proves the fix is not a blanket NaN.
# ---------------------------------------------------------------------------


def test_control_min_group_size_20_measures_the_true_disparity():
    assert _quiet(
        C.demographic_parity_difference, Y_TRUE, Y_PRED, SENS, min_group_size=20
    ) == pytest.approx(TRUE_DP_DIFFERENCE)
    assert _quiet(
        C.demographic_parity_ratio, Y_TRUE, Y_PRED, SENS, min_group_size=20
    ) == pytest.approx(TRUE_DP_RATIO)
    assert _quiet(
        C.disparate_impact_ratio, Y_TRUE, Y_PRED, SENS, min_group_size=20
    ) == pytest.approx(TRUE_DP_RATIO)


def test_control_min_group_size_20_reports_the_run_as_unfair():
    report = _quiet(classification_fairness_report, Y_TRUE, Y_PRED, SENS, min_group_size=20)
    assessment = report["assessment"]

    assert assessment["assessable"] is True
    assert isinstance(assessment["fairness_score"], float)
    failed = {m["metric"] for m in assessment["failed_metrics"]}
    assert "demographic_parity_difference" in failed
    assert "demographic_parity_ratio" in failed
    assert assessment["fairness_score"] < 1.0

    summary = _quiet(explain_fairness_report, report)["summary"]
    assert "COULD NOT CHECK" not in summary
    assert "demographic_parity_difference" in summary


def test_control_min_group_size_20_metric_result_is_unfair_not_unmeasurable():
    analyzer = FairnessAnalyzer(Y_TRUE, Y_PRED, SENS, task_type="classification", min_group_size=20)
    result = _quiet(
        analyzer.demographic_parity_difference, include_ci=True, n_bootstrap=200, random_state=1
    )
    assert result.assessable is True
    assert result.verdict == "unfair"
    assert result.is_fair is False
    assert result.value == pytest.approx(TRUE_DP_DIFFERENCE)


# ---------------------------------------------------------------------------
# CONTROL B: ordinary healthy data, every group above min_group_size, must
# behave EXACTLY as before. No over-correction.
# ---------------------------------------------------------------------------

FAIR_N = 60
FAIR_SENS = np.array(["A"] * FAIR_N + ["B"] * FAIR_N)
FAIR_Y_TRUE = np.array([1, 0] * (FAIR_N // 2) + [1, 0] * (FAIR_N // 2))
FAIR_Y_PRED = np.array([1, 0] * (FAIR_N // 2) + [1, 0] * (FAIR_N // 2))


def test_control_healthy_two_group_data_still_measures_zero_disparity():
    """A 0.0 that was actually MEASURED is still a 0.0. The convention only
    forbids the sentinel where no comparison happened."""
    assert C.demographic_parity_difference(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == 0.0
    assert C.demographic_parity_ratio(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == pytest.approx(1.0)
    assert C.equalized_odds_difference(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == 0.0
    assert C.equal_opportunity_difference(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == 0.0
    assert C.predictive_parity_difference(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == 0.0
    assert C.accuracy_parity_difference(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS) == 0.0


def test_control_healthy_two_group_data_still_scores_and_passes():
    report = classification_fairness_report(FAIR_Y_TRUE, FAIR_Y_PRED, FAIR_SENS)
    assessment = report["assessment"]

    assert assessment["assessable"] is True
    assert assessment["fairness_score"] == 1.0
    assert assessment["not_assessable_metrics"] == []
    assert {m["metric"] for m in assessment["passed_metrics"]} == set(report["metrics"])
    assert "metrics within thresholds" in assessment["summary"]

    summary = explain_fairness_report(report)["summary"]
    assert "COULD NOT CHECK" not in summary
    assert "within acceptable thresholds" in summary
    assert "100.0%" in summary


def test_control_healthy_data_measures_a_real_gap_as_a_failure():
    """The other direction of normal operation: a genuine disparity between two
    valid groups is still reported as a number and still fails."""
    y_pred = np.concatenate([np.ones(FAIR_N, dtype=int), np.zeros(FAIR_N, dtype=int)])
    got = C.demographic_parity_difference(FAIR_Y_TRUE, y_pred, FAIR_SENS)
    assert got == pytest.approx(1.0)
    report = classification_fairness_report(FAIR_Y_TRUE, y_pred, FAIR_SENS)
    assert report["assessment"]["assessable"] is True
    assert "demographic_parity_difference" in {
        m["metric"] for m in report["assessment"]["failed_metrics"]
    }
