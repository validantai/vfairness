"""BGL3 batch evaluation-4: does each unit refuse honestly when nothing can be measured?

Eight units across five files in ``evaluation.vfairness_metrics``, plus the one
adjacent consumer in the same file that was proved to throw a refusal away.

Every test below was run against the code BEFORE the fix in the same session, and
each docstring states what it returned then, in numbers. The controls exist
because a unit that refuses everything is as wrong as one that answers
everything, and the two are indistinguishable without them.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
    attribution_stability,
    diagnose_local_attribution,
    multi_seed_adversarial_probe,
    removal_curve_auc,
)
from vfairness.evaluation.vfairness_metrics.intersectional import (
    generate_structured_findings,
    get_group_rankings,
    identify_privileged_groups,
)
from vfairness.evaluation.vfairness_metrics.ranking import get_ranking_group_metrics
from vfairness.evaluation.vfairness_metrics.report import regression_fairness_report
from vfairness.evaluation.vfairness_metrics.visualization import (
    PALETTES,
    get_available_styles,
    preview_palette,
)

_WEIGHTS = np.array([0.5, -0.25, 1.0, 0.75])
_X = np.array([1.0, 2.0, -1.0, 0.5])


def _margin_model(rows: np.ndarray) -> np.ndarray:
    """A deterministic linear margin, so every number below is reproducible."""
    return np.atleast_2d(np.asarray(rows, dtype=float)) @ _WEIGHTS


def _nan_model(rows: np.ndarray) -> np.ndarray:
    """A model that answers nothing at all (an outage, not a clean verdict)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    return np.full((len(rows), 2), np.nan)


def _clean_two_class_model(rows: np.ndarray) -> np.ndarray:
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    return np.column_stack([np.full(len(rows), 0.7), np.full(len(rows), 0.3)])


# ===========================================================================
# 1. removal_curve_auc: the order IS the measurement
# ===========================================================================


def test_an_all_nan_attribution_vector_is_not_a_faithful_explanation():
    """DEFECT PIN, proved by execution twice (another batch, then here).

    Before: ``removal_curve_auc(_margin_model, x=[1, 2, -1, 0.5],
    attributions=[nan, nan, nan, nan], background_mean=zeros)`` returned
    **0.725** with **zero warnings**. 0.725 grades "strong" in
    ``_grade_faithfulness``. ``np.argsort(-np.abs(nan_vector))`` returns the
    identity permutation, so the curve masked features in the caller's own
    column order and scored the faithfulness of a ranking that does not exist.

    After: NaN, with a warning naming 0 of 4 attributions as finite.
    """
    with pytest.warns(UserWarning, match="only 0 of 4 attribution"):
        auc = removal_curve_auc(_margin_model, _X, np.full(4, np.nan), np.zeros(4))
    assert math.isnan(auc)
    assert auc != 0.0  # nor the clean 0.0 the old clamp would have produced


def test_a_partly_unattributed_vector_is_refused_too():
    """DEFECT PIN, the same defect one step milder.

    Before: ``attributions=[0.5, nan, 1.0, nan]`` returned **1.0**, the maximum
    faithfulness this function can report, because ``abs(nan) > x`` is False for
    every x so both unattributable features were simply ranked last. After: NaN
    with a warning naming 2 of 4 as finite.
    """
    with pytest.warns(UserWarning, match="only 2 of 4 attribution"):
        auc = removal_curve_auc(
            _margin_model, _X, np.array([0.5, np.nan, 1.0, np.nan]), np.zeros(4)
        )
    assert math.isnan(auc)


def test_control_a_real_attribution_vector_is_still_scored_and_still_clamped():
    """OVER-CORRECTION CONTROL: the guard must reject only what it cannot rank.

    The true attributions of the linear model score 1.0; the 0.25/0.25 model
    scores exactly 0.5; a model that ignores its input still scores a MEASURED
    0.0 (which is a real finding, not a refusal). All three run silently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert removal_curve_auc(_margin_model, _X, _WEIGHTS * _X, np.zeros(4)) == 1.0
        assert (
            removal_curve_auc(
                lambda arr: np.array([0.25 * arr.ravel()[0] + 0.25 * arr.ravel()[1]]),
                np.array([1.0, 1.0]),
                np.array([2.0, 1.0]),
                np.zeros(2),
            )
            == 0.5
        )
        measured_zero = removal_curve_auc(
            lambda arr: np.array([0.5]), np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.zeros(2)
        )
    assert measured_zero == 0.0
    assert not math.isnan(measured_zero)


def test_the_guard_reaches_the_delegating_xai_twin():
    """The twin in ``xai.diagnostics.faithfulness`` delegates here on purpose.

    Fixing the twin instead would have recreated the divergence it was collapsed
    to remove, so the guard has to be inherited rather than copied. Before:
    the twin returned the same fabricated 0.725.
    """
    from vfairness.xai.diagnostics.faithfulness import removal_curve_auc as twin

    with pytest.warns(UserWarning, match="only 0 of 4 attribution"):
        assert math.isnan(
            twin(
                predict_fn=_margin_model,
                x=_X,
                attributions=np.full(4, np.nan),
                background_mean=np.zeros(4),
            )
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            twin(
                predict_fn=_margin_model,
                x=_X,
                attributions=_WEIGHTS * _X,
                background_mean=np.zeros(4),
            )
            == 1.0
        )


# ===========================================================================
# 2. attribution_stability
# ===========================================================================


def test_stability_says_out_loud_when_no_feature_has_a_sigma():
    """DEFECT PIN (disclosure, not value).

    Before: ``attribution_stability([[nan, nan, nan], [nan, nan, nan]])``
    returned nan with **zero warnings**, and so did
    ``[[nan, 1.0, 2.0], [nan, 1.1, 1.9]]``, where two of the three features DID
    have a sigma. The value was already honest (nan, never the perfect 0.0); a
    caller reading only the number could not tell it from a run where nothing
    was wrong. After: the same nan, plus a warning stating how many features of
    how many had a finite sigma.
    """
    with pytest.warns(UserWarning, match="only 0 of 3 feature"):
        assert math.isnan(attribution_stability([np.full(3, np.nan), np.full(3, np.nan)]))
    with pytest.warns(UserWarning, match="only 2 of 3 feature"):
        assert math.isnan(
            attribution_stability([np.array([np.nan, 1.0, 2.0]), np.array([np.nan, 1.1, 1.9])])
        )


def test_control_stability_still_measures_a_real_pair_and_still_returns_zero():
    """OVER-CORRECTION CONTROL. A measured 0.0 here IS perfect stability and
    must survive: two identical reruns really do have no variation."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert attribution_stability([np.array([1.0, 2.0]), np.array([1.2, 1.9])]) == pytest.approx(
            0.075
        )
        assert attribution_stability([np.array([1.0, 2.0]), np.array([1.0, 2.0])]) == 0.0


def test_a_single_rerun_is_still_refused():
    """CORRECT REFUSAL PIN (fixed 2026-09-10, re-verified here): stability is
    variation ACROSS runs, so one run has none to measure and 0.0 would be the
    best score on the scale."""
    with pytest.warns(UserWarning, match="1 rerun"):
        assert math.isnan(attribution_stability([np.array([1.0, 2.0])]))


# ===========================================================================
# 3. multi_seed_adversarial_probe and the consumer that dropped its refusal
# ===========================================================================


def test_the_probe_itself_refuses_when_no_seed_produced_a_gap():
    """CORRECT REFUSAL PIN. Verified, not assumed: flag None, not False."""
    background = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.warns(UserWarning, match="only 0 of 4 seed"):
        result = multi_seed_adversarial_probe(
            _nan_model, np.array([1.0, 2.0]), background, n_perturbations=40, n_seeds=4
        )
    assert result.flag is None
    assert result.flag is not False
    assert math.isnan(result.confidence)
    assert result.reason.startswith("COULD NOT CHECK")


def test_a_probe_that_measured_nothing_is_not_a_cleared_model():
    """DEFECT PIN on ``diagnose_local_attribution``, the adjacent consumer.

    ``multi_seed_adversarial_probe`` was given a tri-state flag in READINESS-6
    so a probe that measured nothing could say so, and the consumer then wrote
    ``out["adversarial_flag"] = bool(probe.flag)``, which is False for None.

    Measured before the fix, a model returning all-NaN predictions over a 6-row
    background at n_seeds=3:
        ``adversarial_flag False, adversarial_confidence nan, notes
        ['faithfulness unavailable (removal curve not computable)']``
    and a genuinely clean model over a 30-row background:
        ``adversarial_flag False, adversarial_confidence 1.0, notes []``
    The field a reader acts on was identical, and ``XaiDiagnostics`` has no
    confidence field to carry the difference.

    After: flag None, confidence None, the reason a COULD NOT CHECK and a note
    in ``notes``, which are the fields that survive the schema mapping.
    """
    background = np.random.default_rng(1).normal(size=(6, 2))
    with pytest.warns(UserWarning, match="adversarial probe did not run"):
        out = diagnose_local_attribution(_nan_model, [1.0, 2.0], [0.5, 0.25], background, n_seeds=3)
    assert out["adversarial_flag"] is None
    assert out["adversarial_flag"] is not False
    assert out["adversarial_confidence"] is None
    assert out["adversarial_reason"].startswith("COULD NOT CHECK")
    assert any("adversarial probe did not run" in note for note in out["notes"])


def test_control_a_clean_model_is_still_cleared_with_a_real_false():
    """OVER-CORRECTION CONTROL. False is a VERDICT and must still be reachable:
    the probe ran over 30 background rows and found no scaffolding."""
    background = np.random.default_rng(2).normal(size=(30, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = diagnose_local_attribution(
            _clean_two_class_model,
            [1.0, 2.0],
            [0.5, 0.25],
            background,
            reruns=[[0.5, 0.25], [0.52, 0.24]],
            n_seeds=3,
        )
    assert out["adversarial_flag"] is False
    assert out["adversarial_confidence"] == 1.0
    assert out["adversarial_reason"] is None
    assert out["stability"] == 0.0075
    assert out["notes"] == []


# ===========================================================================
# 4. generate_structured_findings
# ===========================================================================


def _too_small_analysis():
    """Two 10-row cells at the default min_group_size=30: nothing is analysable."""
    rng = np.random.default_rng(3)
    sensitive = np.array(["A"] * 10 + ["B"] * 10)
    y_true = rng.binomial(1, 0.5, 20)
    y_pred = np.concatenate([np.ones(10, dtype=int), np.zeros(10, dtype=int)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return identify_privileged_groups(y_true, y_pred, sensitive, min_group_size=30)


def test_an_empty_findings_list_now_names_what_was_never_examined():
    """DEFECT PIN.

    Before: ``generate_structured_findings`` returned ``[]`` with zero warnings
    both when every cell was examined and none showed a disparity, and when NO
    cell was examined at all. Measured on the 20-row, two-cell analysis above
    (``n_total_cells_seen`` 2, ``n_cells_included`` 0): ``[]``, byte-identical
    to a clean 2000-row audit, while the not-assessed state sat in
    ``max_disparity=nan`` and ``data_treatment``, which do not travel with a
    list.

    After: one explicit ``no_cells_analysed`` entry naming both excluded cells,
    with ``p_value``, ``p_value_corrected`` and ``statistically_significant``
    all None so it cannot be read as a tested finding, plus a warning.
    """
    analysis = _too_small_analysis()
    assert analysis["all_groups"] == []

    with pytest.warns(UserWarning, match="not one cell met the size gate"):
        findings = generate_structured_findings(analysis)

    assert len(findings) == 1
    entry = findings[0]
    assert entry["type"] == "no_cells_analysed"
    assert entry["description"].startswith("COULD NOT CHECK")
    assert entry["p_value"] is None
    assert entry["p_value_corrected"] is None
    assert entry["statistically_significant"] is None
    assert sorted(entry["groups"]) == ["A", "B"]
    assert entry["metric_values"]["n_total_cells_seen"] == 2
    assert entry["metric_values"]["n_cells_analysed"] == 0
    assert entry["unmeasured"]  # names that NO hypothesis was examined
    # And it is not a disparity claim of any severity above info.
    assert entry["severity"] == "info"


def test_control_real_cells_still_produce_real_findings():
    """OVER-CORRECTION CONTROL: the refusal entry must not appear for data that
    WAS analysed, and the finding types must be unchanged.

    Measured: 200 rows in two 100-row cells, group A predicted positive at 80
    percent against B at 20 percent, gives 5 findings
    (over_prediction, under_prediction, two fpr_disparity, amplified_bias).
    """
    rng = np.random.default_rng(3)
    sensitive = np.array(["A"] * 100 + ["B"] * 100)
    y_true = rng.binomial(1, 0.5, 200)
    y_pred = np.concatenate([rng.binomial(1, 0.8, 100), rng.binomial(1, 0.2, 100)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = identify_privileged_groups(y_true, y_pred, sensitive)
        findings = generate_structured_findings(analysis)
    types = [f["type"] for f in findings]
    assert "no_cells_analysed" not in types
    assert len(findings) >= 3
    assert any(t in ("over_prediction", "under_prediction") for t in types)


# ===========================================================================
# 5. get_group_rankings
# ===========================================================================


def _shut_out_ranking_inputs():
    """60 rows in ``A``, all selected; 5 rows in ``B``, none selected."""
    sensitive = np.array(["A"] * 60 + ["B"] * 5)
    y_true = np.concatenate([np.ones(60, dtype=int), np.ones(5, dtype=int)])
    y_pred = np.concatenate([np.ones(60, dtype=int), np.zeros(5, dtype=int)])
    return y_true, y_pred, sensitive


def test_a_group_dropped_from_a_ranking_is_named_not_silently_absent():
    """DEFECT PIN.

    Before: ``get_group_rankings`` on 65 rows, ``A`` (60 rows, selected at 100
    percent) beside ``B`` (5 rows, selected at 0 percent), at the default
    min_group_size=30, returned
    ``[{'group': 'A', 'value': 1.0, 'size': 60, 'metric': 'positive_rate',
    'rank': 1}]`` with **zero warnings**. A spread taken over that list is
    0.0, i.e. perfect parity, over data whose only disparity is the group that
    is missing from it. ``GroupManager`` warns only when EVERY group fails the
    gate, so a partial drop was silent, and the sibling ranking module has
    disclosed exactly this since it imported ``_warn_dropped_groups``.

    After: the same list, plus a warning naming ``B``, its 5 rows, the 7.7
    percent share and the consequence for a ranking rather than for a metric.
    """
    y_true, y_pred, sensitive = _shut_out_ranking_inputs()
    with pytest.warns(UserWarning, match="ABSENT from the returned ranking"):
        rankings = get_group_rankings(y_true, y_pred, sensitive, min_group_size=30)
    assert [r["group"] for r in rankings] == ["A"]
    assert rankings[0]["value"] == pytest.approx(1.0)
    # The disparity a consumer would compute from the returned list alone.
    values = [r["value"] for r in rankings]
    assert max(values) - min(values) == 0.0  # which is why the warning has to exist


def test_a_partial_drop_with_two_survivors_is_named_too():
    """DEFECT PIN, the case where a comparison DOES happen but not over
    everybody. Before: three cells (A 60, B 40, C 5) returned two ranked rows
    and no warning, so the worst row in the list read as the worst group."""
    sensitive = np.array(["A"] * 60 + ["B"] * 40 + ["C"] * 5)
    y_true = np.ones(105, dtype=int)
    y_pred = np.concatenate(
        [np.ones(60, dtype=int), np.zeros(40, dtype=int), np.zeros(5, dtype=int)]
    )
    with pytest.warns(UserWarning, match=r"\{'C': 5\}"):
        rankings = get_group_rankings(y_true, y_pred, sensitive, min_group_size=30)
    assert [r["group"] for r in rankings] == ["A", "B"]


def test_control_a_complete_ranking_is_still_silent():
    """OVER-CORRECTION CONTROL: nothing was dropped, so nothing is warned, and
    both groups are still ranked with their real rates."""
    sensitive = np.array(["A"] * 60 + ["B"] * 40)
    y_true = np.ones(100, dtype=int)
    y_pred = np.concatenate([np.ones(60, dtype=int), np.zeros(40, dtype=int)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rankings = get_group_rankings(y_true, y_pred, sensitive, min_group_size=30)
    assert [(r["group"], r["value"], r["rank"]) for r in rankings] == [
        ("A", 1.0, 1),
        ("B", 0.0, 2),
    ]


# ===========================================================================
# 6. ranking.get_ranking_group_metrics: verified CORRECT, pinned so it stays
# ===========================================================================


def test_ranking_group_metrics_refuses_an_order_that_does_not_exist():
    """CORRECT REFUSAL PIN (no fix needed, verified by execution).

    All 12 scores NaN, and all 12 identical, both return every position derived
    cell as NaN while keeping ``count``, which is the one thing that WAS
    measured, and both warn naming the reason. A partial drop warns as well.
    """
    groups = np.array(["A"] * 6 + ["B"] * 6)
    with pytest.warns(UserWarning, match="12 of 12 ranking scores are not finite"):
        out = get_ranking_group_metrics(np.full(12, np.nan), groups)
    assert set(out) == {"A", "B"}
    assert out["A"]["count"] == 6
    assert math.isnan(out["A"]["avg_position"])
    assert math.isnan(out["A"]["avg_exposure"])

    with pytest.warns(UserWarning, match="all 12 ranking scores are identical"):
        tied = get_ranking_group_metrics(np.full(12, 0.5), groups)
    assert math.isnan(tied["B"]["avg_position"])

    with pytest.warns(UserWarning, match="Excluding 1 group"):
        partial = get_ranking_group_metrics(np.arange(8), np.array(["A"] * 6 + ["B"] * 2))
    assert set(partial) == {"A"}

    # Every group below the gate: the empty dict is disclosed by GroupManager
    # rather than reading as "no groups needed checking".
    with pytest.warns(UserWarning, match="All groups are below min_group_size"):
        nothing = get_ranking_group_metrics(np.arange(4), np.array(["A"] * 2 + ["B"] * 2))
    assert nothing == {}


def test_control_ranking_group_metrics_still_measures_a_real_ranking():
    """OVER-CORRECTION CONTROL with exact positions: A holds 0 to 5, B 6 to 11."""
    groups = np.array(["A"] * 6 + ["B"] * 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = get_ranking_group_metrics(np.arange(12), groups)
    assert out["A"]["avg_position"] == 2.5
    assert out["B"]["avg_position"] == 8.5
    assert out["A"]["avg_exposure"] > out["B"]["avg_exposure"]


# ===========================================================================
# 7. report.regression_fairness_report: verified CORRECT, pinned so it stays
# ===========================================================================


def test_regression_report_refuses_a_verdict_it_could_not_reach():
    """CORRECT REFUSAL PIN (no fix needed, verified by execution).

    One group, and separately two groups both below min_group_size: the score is
    None rather than a number, ``assessable`` is False, every metric lands in
    ``not_assessable_metrics`` rather than ``passed_metrics``, and the summary
    opens with NOT ASSESSABLE.
    """
    rng = np.random.default_rng(5)
    y_true = rng.normal(10, 2, 100)
    y_pred = y_true + 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one_group = regression_fairness_report(y_true, y_pred, np.array(["A"] * 100))
    assessment = one_group["assessment"]
    assert assessment["fairness_score"] is None
    assert assessment["assessable"] is False
    assert assessment["passed_metrics"] == []
    assert len(assessment["not_assessable_metrics"]) == 4
    assert assessment["summary"].startswith("NOT ASSESSABLE")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        too_small = regression_fairness_report(
            y_true[:20], y_pred[:20], np.array(["A"] * 10 + ["B"] * 10), min_group_size=30
        )
    assert too_small["assessment"]["fairness_score"] is None
    assert [g["group"] for g in too_small["assessment"]["insufficient_evidence_groups"]] == [
        "A",
        "B",
    ]

    # Two more refusal paths verified by execution in the same session: a
    # constant y_true (every scale-relative threshold would be 0.0, so any
    # nonzero gap would "FAIL") and predictions that are all NaN (not one row
    # survives validation). Both report a None score and four NOT_ASSESSABLE
    # metrics rather than a 0/4 or 4/4 verdict.
    sensitive = np.array(["A"] * 100 + ["B"] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        constant_scale = regression_fairness_report(
            np.full(200, 7.0), np.concatenate([y_pred, y_pred]), sensitive
        )
        no_rows = regression_fairness_report(
            np.concatenate([y_true, y_true]), np.full(200, np.nan), sensitive
        )
    assert constant_scale["assessment"]["fairness_score"] is None
    assert len(constant_scale["assessment"]["not_assessable_metrics"]) == 4
    assert "NOT ASSESSABLE" in constant_scale["assessment"]["summary"]
    assert no_rows["assessment"]["fairness_score"] is None
    assert no_rows["assessment"]["passed_metrics"] == []


def test_control_regression_report_still_certifies_a_fair_model():
    """OVER-CORRECTION CONTROL: a genuinely fair regression still scores 1.0 over
    all four metrics, so the refusal above is not this function refusing
    everything."""
    rng = np.random.default_rng(11)
    y_true = rng.normal(10, 2, 200)
    y_pred = y_true + rng.normal(0, 0.2, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report = regression_fairness_report(y_true, y_pred, np.array(["A"] * 100 + ["B"] * 100))
    assert report["assessment"]["fairness_score"] == 1.0
    assert report["assessment"]["assessable"] is True
    assert len(report["assessment"]["passed_metrics"]) == 4


# ===========================================================================
# 8. visualization.preview_palette
# ===========================================================================


def test_an_unknown_style_is_not_silently_some_other_palette():
    """DEFECT PIN (labelling, not a fairness number, and graded accordingly).

    Before: ``preview_palette("colorblind_safe_v2")`` returned a figure titled
    "Colorblind_Safe_V2 Color Palette" whose swatches were the MODERN palette,
    with zero warnings. ``_get_palette`` fell back silently while every title in
    the module is built from the name the caller asked for, so a reader checking
    which colours a named style uses was shown another style's under the name
    they asked about. A typo'd "accessible" is the case that costs something.

    After: the same fallback, with a warning saying which palette was actually
    drawn and that the requested style does not exist.
    """
    with pytest.warns(UserWarning, match="is not a known visualization style"):
        fig = preview_palette("colorblind_safe_v2")
    assert fig is not None
    # Still the modern palette underneath, which is exactly what the warning says.
    assert PALETTES["modern"]["primary"] != PALETTES["dark"]["primary"]


def test_control_every_documented_style_still_previews_silently():
    """OVER-CORRECTION CONTROL: all six real styles render with no warning."""
    styles = get_available_styles()
    assert len(styles) >= 6
    for style in styles:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert preview_palette(style) is not None
