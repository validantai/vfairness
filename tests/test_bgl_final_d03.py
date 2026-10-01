"""BGL final pass, group d03: could-not-check survives all the way to the reader.

Two capabilities, one mechanism. A value nobody measured must never reach a
reader as a grade, and the layer that *does* measure honestly must not have its
honesty thrown away by the layer above it.

CAPABILITY 1 -- compute_effect_sizes (classification.py) and the surface that
renders it, FairExplAIner (explainer.py). The engine's own NaN guard was fixed
in an earlier wave; the reader-facing ladder was not, so an effect size nobody
measured was rendered as "Large effect size", severity "high", with the
recommendation "The effect is large and requires action".

CAPABILITY 2 -- intersectional_disparity_analysis (intersectional.py) and the
Pulse envelope that carries it (operations/pulse/orchestrator.py). The engine
skips unmeasurable cells and names them; the normaliser dropped the reason,
emitted bare NaNs and never fired its could-not-check warning, and the whole
payload could not be serialised with ``allow_nan=False``.

Every test here pairs a could-not-check assertion with a HEALTHY-DATA CONTROL
in the same file, because a guard that cannot fail looks identical to a guard
that passed.
"""

import json
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics import classification as classification_module
from vfairness.evaluation.vfairness_metrics._statistics import cohens_h, odds_ratio, risk_ratio
from vfairness.evaluation.vfairness_metrics.classification import compute_effect_sizes
from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner
from vfairness.evaluation.vfairness_metrics.intersectional import (
    intersectional_disparity_analysis,
)
from vfairness.operations.pulse.orchestrator import _normalise_intersectional

# ---------------------------------------------------------------------------
# Fixtures: real arrays, no hand-built dicts standing in for engine output.
# ---------------------------------------------------------------------------


def _nobody_selected():
    """80 rows, two groups, NOT ONE positive prediction anywhere.

    Both selection rates are 0, so the risk ratio is 0/0 and the odds ratio is
    0/0: genuinely undefined, not zero and not infinite.
    """
    y_true = np.array([1, 0] * 40)
    y_pred = np.zeros(80, dtype=int)
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sensitive


def _total_exclusion_of_b():
    """80 rows; group A receives 20 positives, group B receives none.

    This is the STRONGEST disparate impact reading there is, and it must stay a
    graded finding. It is here to stop the could-not-check guard from eating it.
    """
    y_true = np.array([1, 0] * 40)
    y_pred = np.array([1] * 20 + [0] * 20 + [0] * 40)
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sensitive


def _total_exclusion_of_a():
    """The mirror image: group A receives nothing, group B receives 20.

    The risk ratio is a measured 0.0 here, which used to raise
    ZeroDivisionError out of the explainer (1 / 0.0).
    """
    y_true = np.array([1, 0] * 40)
    y_pred = np.array([0] * 40 + [1] * 20 + [0] * 20)
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sensitive


def _healthy_two_groups():
    """80 rows with a real, measurable selection gap: A 75%, B 25%."""
    y_true = np.array([1, 0] * 40)
    y_pred = np.array([1] * 30 + [0] * 10 + [1] * 10 + [0] * 30)
    sensitive = np.array(["A"] * 40 + ["B"] * 40)
    return y_true, y_pred, sensitive


def _intersectional_cells(include_unmeasurable_cell: bool):
    """Two cells with sharply different, fully MEASURED false positive rates.

    A: 100 rows, 50 actual negatives, 45 of them flagged -> FPR 0.900
    B: 100 rows, 60 actual negatives,  2 of them flagged -> FPR 0.033
    C (optional): 50 rows, every label positive -> no actual negatives at all,
       so its FPR has an empty denominator and is not measurable.
    """
    y_true, y_pred, sensitive = [], [], []
    for i in range(100):
        negative = i < 50
        y_true.append(0 if negative else 1)
        y_pred.append(1 if (negative and i < 45) else (0 if negative else 1))
        sensitive.append("A")
    for i in range(100):
        negative = i < 60
        y_true.append(0 if negative else 1)
        y_pred.append(1 if (negative and i < 2) else (0 if negative else 1))
        sensitive.append("B")
    if include_unmeasurable_cell:
        for i in range(50):
            y_true.append(1)
            y_pred.append(1 if i < 25 else 0)
            sensitive.append("C")
    return np.array(y_true), np.array(y_pred), np.array(sensitive)


def _analyse(y_true, y_pred, sensitive):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return intersectional_disparity_analysis(y_true, y_pred, sensitive, min_group_size=30)


# ===========================================================================
# CAPABILITY 1: compute_effect_sizes -> FairExplAIner
# ===========================================================================


def test_an_unmeasurable_risk_ratio_is_not_rendered_as_a_large_effect():
    """PUBLIC ENTRY. explain_report on a real compute_effect_sizes result.

    BEFORE: severity "high", evaluation "Large difference: one group is nanx
    less likely.", recommendation "The effect is large and requires action."
    """
    y_true, y_pred, sensitive = _nobody_selected()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(y_true, y_pred, sensitive, min_group_size=30)

    # The engine really did hand up a could-not-check, not a number.
    assert np.isnan(effects["A_vs_B"]["risk_ratio"][0])

    report = FairExplAIner().explain_report({"effect_sizes": effects})
    rr = report["statistical"]["risk_ratio_A_vs_B"]

    assert rr["severity"] == "could_not_check", rr["severity"]
    assert "COULD NOT CHECK" in rr["evaluation"]
    assert "nanx" not in rr["evaluation"]
    assert "Large difference" not in rr["evaluation"]
    assert "requires action" not in rr["recommendation"]
    # Three states, never two: it is not a pass either.
    assert rr["severity"] != "info"


@pytest.mark.parametrize("effect_type", ["cohens_d", "risk_ratio", "odds_ratio"])
def test_every_effect_size_ladder_reports_the_third_state(effect_type):
    """The guard sits ABOVE the dispatch, so no sibling branch keeps the defect."""
    explanation = FairExplAIner().explain_effect_size(effect_type, float("nan"), "A", "B")

    assert explanation.severity == "could_not_check"
    assert "COULD NOT CHECK" in explanation.evaluation
    assert "nan" not in explanation.evaluation.replace("NaN", "")
    assert "requires action" not in explanation.recommendation


def test_the_guard_sits_above_the_dispatch_not_only_in_the_three_ladders():
    """An effect type with no ladder of its own must not become a PASS.

    The fourth branch of the dispatch is the `else`, which formats the value
    ("The hedges_g is nan.") and grades it severity "info": a could-not-check
    collapsed into the bottom of the scale, which is exactly what "info" means
    for a genuinely graded, genuinely benign metric. Only a guard ABOVE the
    selection covers it, so this test fails if the guard is pushed down into
    the three ladders where the earlier fixes lived.
    """
    explanation = FairExplAIner().explain_effect_size("hedges_g", float("nan"), "A", "B")

    assert explanation.severity == "could_not_check"
    assert "COULD NOT CHECK" in explanation.evaluation
    assert "The hedges_g is nan" not in explanation.evaluation

    # Control: the same unladdered effect type with a REAL value is still
    # DESCRIBED rather than refused, and the number survives.
    #
    # REVISED 2026-09-29, and the subject of this control is unchanged: the value
    # must still reach the reader. What changed is the severity it may carry.
    # `severity == "info"` was asserted here and it was the defect, not the fix:
    # 'info' is the state a graded, genuinely benign value receives, and this
    # value was never graded, because the branch that produced it is the one with
    # no ladder. Measured before the change, with the value the caller passed:
    #   explain_effect_size("p_value", 0.001, "A", "B")
    #     -> severity 'info', "Comparing A vs B: The p_value is 0.001.",
    #        recommendation "No action needed - the effect size is negligible."
    # A p of 0.001 graded as a benign pass, with the library's own definition text
    # attached. Six of the nine keys in STATISTICAL_MEASURES reached that branch.
    # So: could_not_check for the BAND, the value itself still printed, and the
    # recommendation says the value is real and only the yardstick is missing
    # (never "not measured", which would be the mirror defect).
    measured = FairExplAIner().explain_effect_size("hedges_g", 0.42, "A", "B")
    assert measured.severity == "could_not_check"
    assert measured.severity != "info", "an ungraded band is not a graded benign result"
    assert "0.420" in measured.evaluation, "the measured value was discarded, not just ungraded"
    assert "no interpretation bands" in measured.evaluation
    assert "not measured" not in measured.recommendation, (
        "the value WAS measured; only its band is missing"
    )


def test_an_ungraded_severity_never_inherits_the_urgent_recommendation():
    """`_effect_size_recommendation` ends in a catch-all `else`, so a severity
    it does not name is handed "The effect is large and requires action"."""
    explainer = FairExplAIner()
    text = explainer._effect_size_recommendation("risk_ratio", float("nan"), "could_not_check")
    assert "requires action" not in text
    assert "not measured" in text

    # ...and it is on the live path, not dead code: the public entry uses it.
    explanation = explainer.explain_effect_size("risk_ratio", float("nan"), "A", "B")
    assert explanation.recommendation == text

    # Control: a genuinely large measured effect still gets the urgent text.
    assert "requires action" in explainer._effect_size_recommendation("risk_ratio", 4.0, "high")


@pytest.mark.parametrize(
    "method_name",
    ["_interpret_cohens_d", "_interpret_risk_ratio", "_interpret_odds_ratio"],
)
def test_the_three_interpreters_refuse_nan_when_called_directly(method_name):
    """They are called directly too, so each carries the guard itself."""
    text, severity = getattr(FairExplAIner(), method_name)(float("nan"))
    assert severity == "could_not_check"
    assert "COULD NOT CHECK" in text


def test_a_nan_confidence_interval_is_not_printed_as_an_interval():
    explanation = FairExplAIner().explain_effect_size(
        "risk_ratio", 2.5, "A", "B", ci=(float("nan"), float("nan"))
    )
    assert "95% CI: [nan" not in explanation.evaluation
    assert "no 95% interval was estimable" in explanation.evaluation


# --- the findings the guard must NOT delete --------------------------------


def test_total_exclusion_of_one_group_is_still_a_graded_critical_finding():
    """An INFINITE ratio is a measurement, not a could-not-check.

    Group B receives nothing at all. Folding inf into the third state would
    delete the single strongest disparate impact reading in the run.
    """
    y_true, y_pred, sensitive = _total_exclusion_of_b()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(y_true, y_pred, sensitive, min_group_size=30)
    assert np.isinf(effects["A_vs_B"]["risk_ratio"][0])

    report = FairExplAIner().explain_report({"effect_sizes": effects})
    rr = report["statistical"]["risk_ratio_A_vs_B"]

    assert rr["severity"] == "critical", rr["severity"]
    assert rr["severity"] != "could_not_check"
    assert "Total exclusion" in rr["evaluation"]
    assert "infx" not in rr["evaluation"]


def test_the_mirror_image_of_total_exclusion_does_not_crash_the_explainer():
    """risk_ratio == 0.0 reached `factor = 1 / rr` and raised ZeroDivisionError.

    Found by re-reading the whole function after fixing the NaN above; the
    defect had moved to the sibling end of the same ladder.
    """
    y_true, y_pred, sensitive = _total_exclusion_of_a()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(y_true, y_pred, sensitive, min_group_size=30)
    assert effects["A_vs_B"]["risk_ratio"][0] == 0.0

    report = FairExplAIner().explain_report({"effect_sizes": effects})
    rr = report["statistical"]["risk_ratio_A_vs_B"]
    assert rr["severity"] == "critical"
    assert "Total exclusion" in rr["evaluation"]


# --- HEALTHY-DATA CONTROL --------------------------------------------------


def test_control_healthy_effect_sizes_are_still_graded_normally():
    """The guard must not turn a measurable disparity into a could-not-check."""
    y_true, y_pred, sensitive = _healthy_two_groups()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(y_true, y_pred, sensitive, min_group_size=30)

    pair = effects["A_vs_B"]
    assert np.isfinite(pair["cohens_h_positive_rate"])
    assert pair["interpretation"] == "large effect"
    assert pytest.approx(3.0, rel=1e-6) == pair["risk_ratio"][0]

    report = FairExplAIner().explain_report({"effect_sizes": effects})
    cohens = report["statistical"]["cohens_d_A_vs_B"]
    rr = report["statistical"]["risk_ratio_A_vs_B"]

    assert cohens["severity"] == "high"
    assert "Large effect size" in cohens["evaluation"]
    assert rr["severity"] == "high"
    assert "3.0x more likely" in rr["evaluation"]
    for entry in (cohens, rr):
        assert entry["severity"] != "could_not_check"
        assert "COULD NOT CHECK" not in entry["evaluation"]


def test_control_a_measured_zero_effect_is_still_graded_negligible():
    """0.0 is a MEASUREMENT and must not be swept into the third state."""
    text, severity = FairExplAIner()._interpret_cohens_d(0.0)
    assert severity == "info"
    assert "Negligible" in text

    explanation = FairExplAIner().explain_effect_size("cohens_d", 0.0, "A", "B")
    assert explanation.severity == "info"
    assert "COULD NOT CHECK" not in explanation.evaluation


# --- the engine-side guard the auditor asked to have PINNED ----------------


def test_cohens_h_returns_nan_for_an_unmeasured_proportion():
    """The clamp `min(1.0, nan)` used to answer pi/2, the top of the scale."""
    with pytest.warns(UserWarning, match="not measured"):
        assert np.isnan(cohens_h(float("nan"), 0.5))
    with pytest.warns(UserWarning, match="not measured"):
        assert np.isnan(cohens_h(0.5, float("nan")))
    # Control: a real pair of proportions still produces a real effect size.
    assert cohens_h(0.75, 0.25) == pytest.approx(1.0471975512, rel=1e-9)


def test_compute_effect_sizes_grades_no_effect_size_it_did_not_measure(monkeypatch):
    """The `not np.isfinite(d)` branch in compute_effect_sizes, PINNED.

    It is unreachable through real data (validate_inputs drops NaN predictions
    before any rate is taken), so it is reached the only honest way: by making
    the effect size function itself answer a could-not-check, exactly as
    cohens_h now does for an unmeasured proportion. Without this pin the branch
    can be deleted with the suite still green.
    """
    monkeypatch.setattr(classification_module, "cohens_h", lambda p1, p2: float("nan"))

    y_true, y_pred, sensitive = _healthy_two_groups()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        effects = compute_effect_sizes(y_true, y_pred, sensitive, min_group_size=30)

    pair = effects["A_vs_B"]
    assert np.isnan(pair["cohens_h_positive_rate"])
    assert "not interpretable" in pair["interpretation"]
    for graded in ("negligible effect", "small effect", "medium effect", "large effect"):
        assert pair["interpretation"] != graded


def test_impossible_count_pairs_are_refused_by_both_ratios():
    """11 events in 4 trials answered 13.75 with a CI, and an OR of -6.29."""
    from vfairness.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="valid count pair"):
        risk_ratio(11, 4, 2, 10)
    with pytest.raises(ConfigurationError, match="valid count pair"):
        odds_ratio(11, 4, 2, 10)
    # Control: a possible table is still measured.
    rr, low, high = risk_ratio(11, 40, 2, 10)
    assert rr == pytest.approx(1.375, rel=1e-9)
    assert low < rr < high


# ===========================================================================
# CAPABILITY 2: intersectional_disparity_analysis -> Pulse envelope
# ===========================================================================


def test_one_unmeasurable_cell_does_not_delete_the_measured_fpr_findings():
    """PUBLIC ENTRY. BEFORE: 3 fpr_disparity findings -> 0, silently."""
    baseline = _analyse(*_intersectional_cells(include_unmeasurable_cell=False))
    baseline_fpr = [f for f in baseline["findings"] if f["type"] == "fpr_disparity"]
    assert len(baseline_fpr) == 2, "fixture must produce measured FPR findings"

    result = _analyse(*_intersectional_cells(include_unmeasurable_cell=True))
    fpr_findings = [f for f in result["findings"] if f["type"] == "fpr_disparity"]
    groups = sorted(g for f in fpr_findings for g in f["groups"])

    assert groups == ["A", "B"], groups
    assert all(f["severity"] == "critical" for f in fpr_findings)
    assert fpr_findings[0]["metric_values"]["group_fpr"] == pytest.approx(0.9, abs=1e-3)

    # ...and the cell that could NOT be compared is NAMED, never silently absent.
    not_assessed = [f for f in result["findings"] if f["type"] == "fpr_not_assessed"]
    assert len(not_assessed) == 1
    assert not_assessed[0]["groups"] == ["C"]
    assert "COULD NOT CHECK" in not_assessed[0]["description"]
    assert not_assessed[0]["fpr_comparison_ran"] is True


def test_an_unmeasured_cell_figure_is_named_on_the_finding_that_carries_it():
    result = _analyse(*_intersectional_cells(include_unmeasurable_cell=True))
    carrying = [
        f
        for f in result["findings"]
        if "false_positive_rate" in f["metric_values"]
        and not np.isfinite(f["metric_values"]["false_positive_rate"])
    ]
    assert carrying, "expected a finding carrying the unmeasurable FPR"
    for finding in carrying:
        assert "false_positive_rate" in finding["unmeasured"]
        assert "denominator" in finding["unmeasured"]["false_positive_rate"]


def test_the_pulse_envelope_carries_could_not_check_as_null_with_a_reason():
    """The normaliser is a field-by-field rebuild and DROPPED the reason."""
    result = _analyse(*_intersectional_cells(include_unmeasurable_cell=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = _normalise_intersectional(result, label_free=False)

    unmeasured_findings = [f for f in payload["findings"] if f["unmeasured"]]
    assert unmeasured_findings, "the engine's unmeasured map must survive the boundary"
    for finding in unmeasured_findings:
        for field, reason in finding["unmeasured"].items():
            assert finding["metricValues"].get(field, "missing") is None
            assert isinstance(reason, str) and reason

    cell_c = [g for g in payload["allGroups"] if g["group"] == "C"]
    assert cell_c, "cell C must still be reported"
    assert cell_c[0]["falsePositiveRate"] is None
    assert "falsePositiveRate" in cell_c[0]["unmeasured"]


def test_the_pulse_envelope_survives_a_strict_json_boundary():
    """json.dumps(..., allow_nan=False) raised ValueError on the whole payload."""
    result = _analyse(*_intersectional_cells(include_unmeasurable_cell=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = _normalise_intersectional(result, label_free=False)
    encoded = json.loads(json.dumps(payload, allow_nan=False))
    assert encoded["findings"]


def test_a_relative_rate_with_an_empty_denominator_warns_and_emits_null():
    """BEFORE: relativeToOverall/relativeToBest went out as bare nan, no warning."""
    y_true = np.array([1] * 30 + [0] * 30 + [1] * 30 + [0] * 30)
    y_pred = np.zeros(120, dtype=int)  # nobody selected -> overall rate 0.0
    sensitive = np.array(["A"] * 60 + ["B"] * 60)
    result = _analyse(y_true, y_pred, sensitive)

    with pytest.warns(UserWarning, match="no measured relativeToOverall"):
        payload = _normalise_intersectional(result, label_free=False)

    for group in payload["allGroups"]:
        assert group["relativeToOverall"] is None
        assert group["relativeToBest"] is None
        assert group["relativeToOverall"] != 1.0  # never parity by default
        assert "relativeToOverall" in group["unmeasured"]
    json.dumps(payload, allow_nan=False)


# --- HEALTHY-DATA CONTROL --------------------------------------------------


def test_control_fully_measured_cells_carry_no_unmeasured_markers():
    """Every figure is real, so nothing is nulled and nothing is disclosed."""
    result = _analyse(*_intersectional_cells(include_unmeasurable_cell=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        payload = _normalise_intersectional(result, label_free=False)
    assert not [w for w in caught if "no measured relative" in str(w.message)]

    assert not [f for f in payload["findings"] if f["unmeasured"]]
    assert [f for f in payload["findings"] if f["type"] == "fpr_disparity"]
    for group in payload["allGroups"]:
        assert group["unmeasured"] == {}
        assert group["falsePositiveRate"] is not None
        assert np.isfinite(group["falsePositiveRate"])
        assert np.isfinite(group["relativeToOverall"])
        assert np.isfinite(group["relativeToBest"])
    json.dumps(payload, allow_nan=False)


def test_control_a_measured_zero_relative_rate_is_kept_not_nulled():
    """relative_to_best == 0.0 is TOTAL EXCLUSION, the worst reading there is.

    It must survive as 0.0. Nulling it, or defaulting it to 1.0, reports the
    single worst measurement as the single best.
    """
    y_true = np.array([1] * 30 + [0] * 30 + [1] * 30 + [0] * 30)
    # Cell A gets half its rows selected; cell B gets nothing at all.
    y_pred = np.array([1] * 30 + [0] * 30 + [0] * 60)
    sensitive = np.array(["A"] * 60 + ["B"] * 60)
    result = _analyse(y_true, y_pred, sensitive)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payload = _normalise_intersectional(result, label_free=False)

    cell_b = [g for g in payload["allGroups"] if g["group"] == "B"]
    assert cell_b
    assert cell_b[0]["relativeToBest"] == 0.0
    assert cell_b[0]["relativeToOverall"] == 0.0
    assert "relativeToBest" not in cell_b[0]["unmeasured"]


# ── 2026-09-29: the dispatch's `else` was grading the TYPE it could not grade ──
#
# The guard above the dispatch answers the could-not-check question for the VALUE.
# The `else` answered it for the TYPE by fabricating a verdict:
#   interpretation = f"The {effect_type} is {value:.3f}."; severity = "info"
# and `_effect_size_recommendation` then supplied "No action needed - the effect
# size is negligible." with the library's real definition text attached, which is
# what makes such a card read as authentic. Measured before the fix:
#
#   p_value 0.001           -> info, "No action needed - the effect size is negligible."
#   standard_error 12.5     -> info, same recommendation
#   confidence_interval 0.0 -> info, same
#   fdr_correction 0.99     -> info, same
#   credible_interval 0.5   -> info, same
#   bonferroni_correction   -> info, same
#
# Six of the NINE keys in this module's own STATISTICAL_MEASURES have no band
# ladder, so six of nine graded as a benign pass.

UNLADDERED = [
    ("p_value", 0.001),
    ("standard_error", 12.5),
    ("confidence_interval", 0.0),
    ("fdr_correction", 0.99),
    ("credible_interval", 0.5),
    ("bonferroni_correction", 0.0001),
]


@pytest.mark.parametrize("effect_type,value", UNLADDERED)
def test_a_type_with_no_band_ladder_is_not_graded_as_negligible(effect_type, value):
    from vfairness.evaluation.vfairness_metrics.explainer import (
        STATISTICAL_MEASURES,
        FairExplAIner,
    )

    # The premise, asserted rather than assumed: this key really is in the
    # library's own table and really has no ladder, so the test is about the
    # branch it claims to be about.
    assert effect_type in STATISTICAL_MEASURES
    assert "thresholds" not in STATISTICAL_MEASURES[effect_type]

    explanation = FairExplAIner().explain_effect_size(effect_type, value, "A", "B")
    assert explanation.severity == "could_not_check", (
        f"{effect_type}={value} came back {explanation.severity}"
    )
    assert explanation.severity != "info"
    assert "COULD NOT CHECK" in explanation.evaluation
    assert "No action needed" not in explanation.recommendation
    assert "negligible" not in explanation.recommendation
    # The measurement is NOT discarded: the mirror defect is a real value
    # reported as a could-not-check with the number thrown away.
    assert f"{value:.3f}" in explanation.evaluation
    assert "not measured" not in explanation.recommendation


def test_an_unknown_effect_type_fails_closed_the_same_way():
    """A type absent from STATISTICAL_MEASURES gets an empty definition and lands in
    the same branch; it must not read as a pass either."""
    from vfairness.evaluation.vfairness_metrics.explainer import (
        STATISTICAL_MEASURES,
        FairExplAIner,
    )

    assert "cliffs_delta" not in STATISTICAL_MEASURES
    explanation = FairExplAIner().explain_effect_size("cliffs_delta", 0.7, "A", "B")
    assert explanation.severity == "could_not_check"
    assert "0.700" in explanation.evaluation


@pytest.mark.parametrize(
    "effect_type,value,expected",
    [("cohens_d", 0.05, "info"), ("cohens_d", 0.9, "high"), ("risk_ratio", 3.0, "high")],
)
def test_control_the_three_graded_ladders_still_grade(effect_type, value, expected):
    """OVER-CORRECTION CONTROL. The fix must not turn a graded band into a refusal:
    cohens_d 0.05 IS a measured negligible effect and must stay 'info'."""
    from vfairness.evaluation.vfairness_metrics.explainer import FairExplAIner

    explanation = FairExplAIner().explain_effect_size(effect_type, value, "A", "B")
    assert explanation.severity == expected, explanation.evaluation
    assert "COULD NOT CHECK" not in explanation.evaluation
    if expected == "info":
        assert "No action needed" in explanation.recommendation
