"""Audit wave 5: explainer narrative coherence (label vs severity).

Pins seven fixes in ``vfairness.explainer`` / ``vfairness._bands``:

1. The dead ``_score_to_severity`` helper carried a divergent
   0.2/0.4/0.6/0.8 risk scale; it is gone, all risk severities come
   from the canonical 0.25/0.50/0.75 badge bands in ``_bands``.
2. Calibration ECE severity and prose derive from the same
   0.05/0.10/0.15 thresholds (an ECE of 0.045 said 'well calibrated'
   but was labelled 'low').
3. The ECE card keeps its own severity; the group-disparity bump only
   raises the report-level severity and the disparity card.
4. The training report's overall severity includes the
   method-comparison card in the worst-severity roll-up.
5. Baseline-fairness prose and severity share one |DP| > 0.08
   predicate (exactly 0.08 said 'exhibits unfairness' at 'info').
6. The drift report's overall severity rolls up per-scale drift cards.
7. The experiment 'Overall Treatment Effect' card is labelled from its
   own values, not the report-level worst-item severity.

All handler inputs are duck-typed SimpleNamespace objects, mirroring
how the handlers read real report dataclasses via getattr.
"""

import math
from types import SimpleNamespace as NS  # noqa: N814

import pytest

from vfairness import _bands, explainer
from vfairness.explainer import _SEVERITY_ORDER


def _rank(sev: str) -> int:
    return _SEVERITY_ORDER[sev]


# 1. Divergent 0.2/0.4/0.6/0.8 risk helper removed


def test_divergent_score_to_severity_helper_removed():
    assert not hasattr(explainer, "_score_to_severity")


def test_no_stale_02_04_06_08_risk_band_thresholds_in_source():
    # Cohen's d benchmarks legitimately mention 0.2/0.5/0.8; the stale risk
    # scale always appeared as the 0.4-and-0.6 pair, so that pair is the tripwire.
    import inspect

    src = inspect.getsource(explainer)
    for line in src.splitlines():
        code = line.split("#", 1)[0]
        assert not ("0.4" in code and "0.6" in code), line


# 2. Calibration: ECE severity derives from the prose thresholds


@pytest.mark.parametrize(
    "ece,expected_sev,expected_text",
    [
        (0.0, "info", "well calibrated"),
        (0.045, "info", "well calibrated"),  # the reported bug case
        (0.0499, "info", "well calibrated"),
        (0.05, "low", "Moderate miscalibration"),  # boundary flips both together
        (0.08, "low", "Moderate miscalibration"),
        (0.0999, "low", "Moderate miscalibration"),
        (0.10, "medium", "Significant miscalibration"),
        (0.1499, "medium", "Significant miscalibration"),
        (0.15, "high", "Significant miscalibration"),
        (0.30, "high", "Significant miscalibration"),
    ],
)
def test_calibration_text_and_severity_share_thresholds(ece, expected_sev, expected_text):
    rep = NS(
        overall_metrics={"ece": ece},
        has_significant_disparity=False,
        recommendations=[],
        n_groups=2,
    )
    out = explainer._explain_calibration(rep)
    card = out.explanations[0]
    assert card.metric_name == "Expected Calibration Error (ECE)"
    assert card.severity == expected_sev
    assert expected_text in card.evaluation
    assert out.severity == expected_sev


def test_calibration_ece_severity_bands_helper():
    assert _bands.CALIBRATION_ECE_THRESHOLDS == (0.05, 0.10, 0.15)
    assert _bands.calibration_ece_severity(0.049) == "info"
    assert _bands.calibration_ece_severity(0.05) == "low"
    assert _bands.calibration_ece_severity(0.10) == "medium"
    assert _bands.calibration_ece_severity(0.15) == "high"
    # NaN must not read as a pass upgrade nor crash
    assert _bands.calibration_ece_severity(math.nan) == "info"
    assert _bands.calibration_ece_severity(None) == "info"


# 3. Calibration: disparity bump stays off the ECE card


def test_disparity_bump_raises_report_not_ece_card():
    rep = NS(
        overall_metrics={"ece": 0.02},
        has_significant_disparity=True,
        recommendations=[],
        n_groups=2,
    )
    out = explainer._explain_calibration(rep)
    ece_card = out.explanations[0]
    disparity_card = [
        e for e in out.explanations if e.metric_name == "Calibration Disparity Across Groups"
    ][0]
    # The card describes its own well-calibrated value
    assert "well calibrated" in ece_card.evaluation
    assert ece_card.severity == "info"
    # The bump lives on the disparity card and the report level
    assert disparity_card.severity == "medium"
    assert out.severity == "medium"


def test_disparity_bump_never_lowers_a_bad_ece():
    rep = NS(
        overall_metrics={"ece": 0.2}, has_significant_disparity=True, recommendations=[], n_groups=2
    )
    out = explainer._explain_calibration(rep)
    assert out.explanations[0].severity == "high"
    assert out.severity == "high"


# 4. Training report: method-comparison severity in the roll-up


def test_training_report_rolls_up_method_comparison_severity():
    rep = NS(
        recommendation=None,
        method_comparisons=[NS(constraint_satisfied=False)],
        baseline_metrics={"demographic_parity_difference": 0.01},
        action_items=[],
        task_type="classification",
    )
    out = explainer._explain_training_report(rep)
    comp = [e for e in out.explanations if e.metric_name == "Training Method Comparison"][0]
    assert comp.severity == "medium"
    assert _rank(out.severity) >= _rank(comp.severity)


def test_training_report_severity_never_below_any_card():
    rep = NS(
        recommendation=None,
        method_comparisons=[NS(constraint_satisfied=True)],
        baseline_metrics={"demographic_parity_difference": 0.2},
        action_items=[],
        task_type="classification",
    )
    out = explainer._explain_training_report(rep)
    assert out.severity == "high"  # baseline high dominates comparison low
    for card in out.explanations:
        assert _rank(out.severity) >= _rank(card.severity)


# 5. Training report: baseline DP boundary coherence


@pytest.mark.parametrize(
    "dp,expected_sev,fair_text",
    [
        (0.0, "info", True),
        (0.08, "info", True),  # the reported boundary bug: was unfair-at-info
        (-0.08, "info", True),
        (0.0801, "medium", False),
        (-0.12, "medium", False),
        (0.15, "medium", False),  # high starts strictly above 0.15
        (0.1501, "high", False),
    ],
)
def test_baseline_dp_text_matches_severity(dp, expected_sev, fair_text):
    rep = NS(
        recommendation=None,
        method_comparisons=[],
        baseline_metrics={"demographic_parity_difference": dp},
        action_items=[],
        task_type="classification",
    )
    out = explainer._explain_training_report(rep)
    base = out.explanations[0]
    assert base.metric_name == "Baseline Fairness Assessment"
    assert base.severity == expected_sev
    if fair_text:
        assert "reasonably fair" in base.evaluation
        assert "exhibits unfairness" not in base.evaluation
    else:
        assert "exhibits unfairness" in base.evaluation


# 6. Drift report: per-scale severities in the roll-up


def _scale(detected=True):
    return NS(drift_detected=detected, ks_statistic=0.4, p_value=0.01, mean_shift=0.1)


def test_drift_report_rolls_up_scale_cards():
    res = NS(
        drift_detected=True, overall_drift_score=0.2, scales={"d1": _scale()}, worst_scale=None
    )
    out = explainer._explain_drift_result(res)
    main, scale_card = out.explanations[0], out.explanations[1]
    # Main card still describes the aggregate score band
    assert main.severity == "low"
    assert scale_card.severity == "medium"
    # Report label must not undercut the drifting scale card
    assert out.severity == "medium"


def test_drift_report_without_scale_drift_keeps_score_band():
    res = NS(
        drift_detected=True,
        overall_drift_score=0.2,
        scales={"d1": _scale(detected=False)},
        worst_scale=None,
    )
    out = explainer._explain_drift_result(res)
    assert len(out.explanations) == 1
    assert out.severity == "low"


def test_drift_report_high_score_stays_high():
    res = NS(
        drift_detected=True, overall_drift_score=0.7, scales={"d1": _scale()}, worst_scale=None
    )
    out = explainer._explain_drift_result(res)
    assert out.severity == "high"
    for card in out.explanations:
        assert _rank(out.severity) >= _rank(card.severity)


# 7. Experiment: overall-effect card severity from its own values


def _experiment(het, overall_p, effects):
    return NS(
        heterogeneity_detected=het,
        heterogeneity_p_value=0.01 if het else 0.8,
        overall_effect=0.05,
        overall_p_value=overall_p,
        intersection_effects=effects,
        n_intersections=max(2, len(effects)),
    )


def test_experiment_overall_card_not_borrowing_report_severity():
    harmed = NS(effect=-0.2, significant=True, intersection=("a",), effect_size_d=0.5, p_value=0.01)
    out = explainer._explain_experiment_result(_experiment(True, 0.4, [harmed]))
    card = [e for e in out.explanations if e.metric_name == "Overall Treatment Effect"][0]
    # Report-level worst is high (heterogeneity + harmed group)
    assert out.severity == "high"
    # ...but the overall effect itself is not significant: its card says so
    assert "not significant" in card.evaluation
    assert card.severity == "info"


def test_experiment_overall_card_significant_effect_is_low():
    out = explainer._explain_experiment_result(_experiment(False, 0.01, []))
    card = [e for e in out.explanations if e.metric_name == "Overall Treatment Effect"][0]
    assert "significant" in card.evaluation
    assert card.severity == "low"
    assert out.severity == "low"
