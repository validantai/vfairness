"""BGL3 batch xai-2: does the explainability surface refuse honestly?

Eight units were examined by execution on 2026-09-27 (plus one handed over
already proven by a neighbouring batch). Six fabricated a clean answer where
nothing had been measured, and each fix below is pinned here with the number it
replaced. The two that were already honest are pinned too, because an unpinned
correct refusal is one refactor away from a fabricated all-clear, and because a
suite that only holds the fixes cannot tell a working explainer from one that
refuses everything.

What was measured before the fixes, in short:

* ``XaiDiagnostics(faithfulness=0.8, stability=0.02)`` recorded
  ``adversarial_flag: False``, the clean verdict, for a probe that never ran.
* ``FairnessDecomposition.assert_identity`` returned None for a residual of
  ``nan`` (``nan > tolerance`` is False), and ``to_db_row`` then emitted
  ``{'total_disparity': nan, 'per_feature': {'income': nan, ...}}``.
* ``LimeExplainer.explain_local`` on a 50-row background of one repeated row
  returned every attribution as exactly 0.0, fidelity 0.0, stability 0.0, no
  warning, for a model whose prediction moves from 0.9999997 to 3.2e-07 when
  feature 0 flips sign.
* the same adapter reported ``c = 0.0000`` for the only feature the model uses
  when that column was frozen in the background, against ``c = 0.5508`` with a
  background that varies in it.
* ``AnchorsExplainer.explain_local`` with a 300-row background frozen in the
  deciding feature returned ``rule=[]`` with precision 1.0 and coverage 1.0 and
  ``background_sufficient=True``, silently, against ``rule=['c > 0.72']`` with
  coverage 0.25 on a varying background.
* ``DiceCounterfactualExplainer.explain_local`` recorded ``prediction=0.0``, the
  declined class, for a model it never managed to ask, on a branch that emitted
  no warning at all.
* the same adapter scored a categorical-only recourse ``proximity=0.0`` and
  ``feasibility=1.0``, the best possible pair, with nothing measured.
* ``WorkerLoop.run`` logged "XAI worker stopped cleanly" after three polls that
  every one of them failed with
  ``RuntimeError('relation "pgmq.q_xai_jobs" does not exist')``.
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vfairness.xai.diagnostics.stability import attribution_stability  # noqa: E402
from vfairness.xai.schemas import (  # noqa: E402
    Attribution,
    Explanation,
    FairnessDecomposition,
    XaiDiagnostics,
)
from vfairness.xai.storage.audit_artifact import build_audit_artifact_bundle  # noqa: E402

KW = dict(subject_id="s", model_hash="m", data_hash="d", feature_names=["a", "b", "c", "d"])


# === schemas: the two states that used to read as clean ===================


def test_diagnostics_do_not_default_to_a_clean_adversarial_verdict() -> None:
    """DEFECT, fixed. Measured before the fix, a diagnostics record built
    without the flag came back as
    {'faithfulness': 0.8, 'stability': 0.02, 'adversarial_flag': False,
    'adversarial_reason': None, 'notes': []}, and write_assessment asdict()s
    that into the row: 'no OOD scaffolding detected' for a probe that may never
    have run. Three states, and the default is the third.
    """
    import dataclasses

    default = XaiDiagnostics(faithfulness=0.8, stability=0.02)
    assert default.adversarial_flag is None
    assert default.adversarial_flag is not False
    assert dataclasses.asdict(default)["adversarial_flag"] is None

    # NOT refusing everything: both real verdicts still travel.
    assert (
        XaiDiagnostics(faithfulness=0.8, stability=0.02, adversarial_flag=False).adversarial_flag
        is False
    )
    assert (
        XaiDiagnostics(faithfulness=0.8, stability=0.02, adversarial_flag=True).adversarial_flag
        is True
    )


def _decomposition(total: float, per_feature: dict[str, float]) -> FairnessDecomposition:
    return FairnessDecomposition(
        id="d-1",
        subject_id="s",
        metric="demographic_parity",
        protected_attribute="gender",
        total_disparity=total,
        per_feature=per_feature,
        proxy_scores={},
        flagged_proxies=[],
        audit_artifact_id="aa-1",
    )


def test_assert_identity_refuses_a_residual_that_is_not_a_number() -> None:
    """DEFECT, fixed. Measured before the fix: total_disparity=nan with
    per_feature={'income': nan, 'zipcode': 11.0} gave identity_residual=nan,
    `nan > 1e-06` is False, assert_identity returned None, and to_db_row wrote
    {'total_disparity': nan, 'per_feature': {'income': nan, 'zipcode': 11.0},
    'flagged_proxies': []}. The comment in xai.decomposition.shap_fairness names
    this exact hole; this is the public guard that had it.
    """
    bad = _decomposition(float("nan"), {"income": float("nan"), "zipcode": 11.0})
    assert math.isnan(bad.identity_residual)

    with pytest.raises(ValueError) as excinfo:
        bad.assert_identity()
    message = str(excinfo.value)
    assert "COULD NOT BE CHECKED" in message
    # It must name WHAT was not measured, not merely that something was not.
    assert "income" in message
    assert "zipcode" not in message

    # ...and the row cannot be written any more either.
    with pytest.raises(ValueError, match="COULD NOT BE CHECKED"):
        bad.to_db_row("auth0|probe")

    # One non-finite feature is enough, even with a finite total.
    with pytest.raises(ValueError, match="COULD NOT BE CHECKED"):
        _decomposition(0.08, {"a": float("nan"), "b": 0.03}).assert_identity()


def test_assert_identity_still_passes_a_real_identity_and_still_breaks_a_broken_one() -> None:
    """CONTROL for the refusal above: a guard that refused everything would
    pass the test above and be useless. A decomposition whose per-feature parts
    sum to the disparity is accepted and written; one that misses by 0.02 raises
    the original 'identity broken' error, which is a different message from the
    could-not-check one.
    """
    good = _decomposition(0.08, {"a": 0.05, "b": 0.03})
    assert good.identity_residual <= 1e-6
    assert good.assert_identity() is None
    assert good.to_db_row("auth0|probe")["total_disparity"] == 0.08

    with pytest.raises(ValueError, match="identity broken") as excinfo:
        _decomposition(0.10, {"a": 0.05, "b": 0.03}).assert_identity()
    assert "COULD NOT BE CHECKED" not in str(excinfo.value)


# === stability: already honest, now held there ============================


def test_attribution_stability_already_refuses_a_single_run() -> None:
    """CORRECT ALREADY, pinned. Stability is variation ACROSS runs, and 0.0
    sigma is the best score the scale has. Measured 2026-09-27: one rerun
    returns nan and warns '1 rerun(s) supplied ... Returning nan, not 0.0, which
    would read as perfectly stable'; zero reruns the same; two rerun vectors of
    zero length return nan as well. The value 0.0 appears only for two runs that
    genuinely agreed, which is a measurement.
    """
    for reruns in ([np.array([0.1, 0.2])], []):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = attribution_stability(reruns)
        assert math.isnan(value), f"{len(reruns)} rerun(s) returned {value!r}, not nan"
        assert any("nothing was measured" in str(w.message) for w in caught)
        assert any("not 0.0" in str(w.message) for w in caught)

    # Nothing to take a sigma over, feature-wise: still nan, never 0.0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(attribution_stability([np.array([]), np.array([])]))

    # The measuring half. 0.075 is the mean per-feature sigma of these two.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = attribution_stability([np.array([0.1, 0.2]), np.array([0.3, 0.1])])
    assert measured == pytest.approx(0.075, abs=1e-12)
    assert caught == []

    # Two identical runs ARE perfectly stable, and that zero must survive.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agreed = attribution_stability([np.array([0.1, 0.2]), np.array([0.1, 0.2])])
    assert agreed == pytest.approx(0.0, abs=1e-12)
    assert caught == []


# === LIME: a neighbourhood that does not exist ============================


@pytest.fixture(scope="module")
def lime_world():
    """A model whose only live coefficient is on feature 'c' (5.80 against
    <= 0.06 on the others), the row with the largest c, and three backgrounds:
    one that varies everywhere, one frozen in c, one with no spread at all.
    """
    pytest.importorskip("lime")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    y = (X[:, 2] > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=800).fit(X, y)
    x = X[int(np.argmax(X[:, 2]))]
    frozen = X.copy()
    frozen[:, 2] = x[2]
    return model, X, x, frozen


def _lime():
    from vfairness.xai.explainers.lime_adapter import LimeExplainer

    return LimeExplainer()


def _contributions(explanation) -> dict[str, float]:
    return {a.feature: a.contribution for a in explanation.attributions}


def test_lime_refuses_a_background_with_no_neighbourhood(lime_world) -> None:
    """DEFECT, fixed. Measured before the fix with a 50-row background of one
    repeated row (and again with a single-row background): every attribution
    came back exactly 0.0 with fidelity 0.0, stability 0.0 and base == prediction
    == 0.4919, no warning, while the model's own prediction moves from 0.9999997
    to 3.2e-07 when the deciding feature flips sign. LIME scales its
    perturbations by each background column's spread, so nothing was sampled and
    nothing was measured. This is NOT the SHAP case pinned in
    tests/test_xai_degenerate_attribution.py: a deviation-based attribution of
    an instance equal to its own background is a measured zero, while a
    surrogate regression with no design variation is no measurement at all.
    """
    model, _X, x, _frozen = lime_world
    for label, background in (
        ("50 identical rows", np.tile(x, (50, 1))),
        ("single row", x.reshape(1, -1)),
        ("a different constant", np.tile(np.full(4, 9.0), (40, 1))),
    ):
        with pytest.raises(ValueError) as excinfo:
            _lime().explain_local(
                model,
                x,
                background,
                instance_id="i",
                stability_reruns=2,
                num_samples=200,
                **KW,
            )
        message = str(excinfo.value)
        assert "COULD NOT MEASURE" in message, label
        assert "no feature influenced this decision" in message, label


def test_lime_names_the_feature_it_could_not_perturb(lime_world) -> None:
    """DEFECT, fixed. Measured before the fix: with the background frozen in
    'c', the one feature the model uses, LIME reported c = 0.0000 exactly and
    said nothing, against c = 0.5508 (the dominant attribution) on a background
    that varies in c. The model's prediction still moves from 1.000000 to
    0.000001 when c flips sign, so that zero describes the sampler, not the
    model. The 0.0 is lime's own output and stays; what is added is the explicit
    not-assessed list beside it.
    """
    model, _X, x, frozen = lime_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _lime().explain_local(
            model, x, frozen, instance_id="i", stability_reruns=2, num_samples=400, **KW
        )
    assert explanation.params["unperturbable_features"] == ["c"]
    assert _contributions(explanation)["c"] == 0.0
    messages = [str(w.message) for w in caught]
    assert any("no spread" in m and "'c'" in m for m in messages), messages
    assert any("NOT measured" in m for m in messages), messages


def test_lime_still_explains_a_neighbourhood_it_can_sample(lime_world) -> None:
    """CONTROL. The fix must not push LIME into refusing what it can answer.
    Measured after the fix, unchanged from before it: c is the largest
    attribution by magnitude, the not-assessed list is empty and no warning is
    raised.
    """
    model, X, x, _frozen = lime_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _lime().explain_local(
            model, x, X, instance_id="i", stability_reruns=2, num_samples=400, **KW
        )
    contributions = _contributions(explanation)
    assert explanation.params["unperturbable_features"] == []
    assert max(contributions, key=lambda k: abs(contributions[k])) == "c"
    assert abs(contributions["c"]) > 4 * max(abs(v) for k, v in contributions.items() if k != "c")
    assert explanation.fidelity is not None and explanation.fidelity > 0.0
    assert [str(w.message) for w in caught] == []

    # Two distinct rows are a neighbourhood, and the refusal must not reach them.
    two_rows = np.vstack([x, X[0]])
    assert _lime().explain_local(
        model, x, two_rows, instance_id="i", stability_reruns=2, num_samples=200, **KW
    )


def test_lime_global_answers_per_row_and_refuses_the_degenerate_matrix(lime_world) -> None:
    """DEFECT for the constant world, CORRECT for the empty one.

    Measured before the fix, explain_global over a constant matrix whose
    background is that same matrix returned one Explanation per row with every
    attribution 0.0 and fidelity 0.0; it now propagates the refusal from
    explain_local. The zero-row case is a different question and was judged on
    2026-09-25 (tests/test_xai_degenerate_attribution.py): explain_global emits
    one Explanation per ROW, so [] is literally what no rows means, and the
    caller supplied those rows. It was pinned there for SHAP and not for LIME;
    it is pinned here.
    """
    model, X, x, _frozen = lime_world
    assert (
        _lime().explain_global(
            model, np.empty((0, 4)), X, stability_reruns=2, num_samples=200, **KW
        )
        == []
    )

    constant = np.tile(x, (3, 1))
    with pytest.raises(ValueError, match="COULD NOT MEASURE"):
        _lime().explain_global(model, constant, constant, stability_reruns=2, num_samples=200, **KW)

    # The measuring half: three real rows, three explanations.
    out = _lime().explain_global(model, X[:3], X, stability_reruns=2, num_samples=200, **KW)
    assert len(out) == 3
    assert all(e.params["unperturbable_features"] == [] for e in out)


# === anchors: the empty rule that reads as a perfect explanation ==========


@pytest.fixture(scope="module")
def anchors_world(lime_world):
    pytest.importorskip("anchor")
    from vfairness.xai.explainers.anchors_adapter import AnchorsExplainer

    return AnchorsExplainer(), lime_world


def test_anchors_says_an_empty_rule_is_not_a_universal_explanation(anchors_world) -> None:
    """DEFECT, fixed. Measured before the fix with a 300-row background frozen
    in 'c', the only feature the model uses: rule=[], precision=1.0,
    coverage=1.0, n_background=300, background_sufficient=True and no warning of
    any kind, against rule=['c > 0.72'] with coverage 0.25 on a background that
    varies in c. The empty anchor's 1.0 / 1.0 belong to the whole perturbation
    space, and the adapter already disclosed exactly this failure mode below
    MIN_BACKGROUND_ROWS, where it had no chance of being the one that mattered.
    """
    explainer, (model, _X, x, frozen) = anchors_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = explainer.explain_local(model, x, frozen, instance_id="i", **KW)
    params = explanation.params
    assert params["rule"] == []
    assert params["background_sufficient"] is True
    assert params["unperturbable_features"] == ["c"]
    note = params["empty_rule_warning"]
    assert "EMPTY rule" in note
    assert "state no sufficient condition" in note
    assert "'c'" in note, note
    assert any("EMPTY rule" in str(w.message) for w in caught)

    # A rule CAN be found while another column is frozen, and then the statement
    # is the narrower one. Measured with 'a' frozen instead, a feature this model
    # does not use: rule=['c > 0.72'], unperturbable_features=['a'] and the
    # unperturbable note rather than the empty-rule one.
    frozen_elsewhere = _X.copy()
    frozen_elsewhere[:, 0] = x[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        other = explainer.explain_local(model, x, frozen_elsewhere, instance_id="i", **KW)
    assert other.params["rule"], other.params
    assert other.params["unperturbable_features"] == ["a"]
    assert "empty_rule_warning" not in other.params
    assert "NOT measured" in other.params["unperturbable_warning"]
    assert any("single value ['a']" in str(w.message) for w in caught)


def test_anchors_still_finds_a_real_rule(anchors_world) -> None:
    """CONTROL. With a background that varies, the same model and row yield a
    rule naming c, coverage well below 1.0, an empty not-assessed list and no
    empty-rule note. A disclosure that fired on every call would be noise.
    """
    explainer, (model, X, x, _frozen) = anchors_world
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = explainer.explain_local(model, x, X, instance_id="i", **KW)
    params = explanation.params
    assert params["rule"], "the search found no rule on a background it can sample"
    assert any("c" in condition for condition in params["rule"]), params["rule"]
    assert params["coverage"] < 1.0
    assert params["unperturbable_features"] == []
    assert "empty_rule_warning" not in params
    assert "unperturbable_warning" not in params
    assert not any("EMPTY rule" in str(w.message) for w in caught)


# === DiCE: a prediction nobody asked for, and a recourse nobody priced ====


@pytest.fixture(scope="module")
def dice_frames():
    pytest.importorskip("dice_ml")
    pd = pytest.importorskip("pandas")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    n = 300
    numeric = pd.DataFrame(
        {
            "income": rng.normal(50, 10, n).round(2),
            "age": rng.integers(20, 70, n).astype(float),
        }
    )
    numeric["y"] = ((numeric["income"] > 50) & (numeric["age"] < 55)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=500).fit(numeric[["income", "age"]], numeric["y"])

    categorical = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east"], 240),
            "band": rng.choice(["a", "b"], 240),
        }
    )
    return numeric, model, categorical


def _dice():
    from vfairness.xai.explainers.dice_adapter import DiceCounterfactualExplainer

    return DiceCounterfactualExplainer()


class _ProbaOnly:
    """A black box exposing predict_proba only: no .predict, not callable.

    DiCE's sklearn backend generates counterfactuals from predict_proba quite
    happily, so this reaches the end of explain_local; only the adapter's own
    original-prediction call has nothing to use.
    """

    def __init__(self, inner):
        self._inner = inner

    def predict_proba(self, frame):
        return self._inner.predict_proba(frame)


def test_dice_records_nan_not_the_declined_class_when_it_cannot_ask_the_model(
    dice_frames,
) -> None:
    """DEFECT, fixed. Measured before the fix with a predict_proba-only model:
    DiCE produced 2 counterfactuals and the returned explanation carried
    prediction=0.0, which to_db_row wrote as 0.0, with NO warning, because
    _predict_label's 'model exposes neither predict nor __call__' branch returns
    None silently. 0.0 is the DECLINED class, the very label a recourse
    explanation exists for, so it cannot also mean 'never asked'.
    """
    numeric, model, _categorical = dice_frames
    features = ["income", "age"]
    declined = numeric[numeric["y"] == 0].iloc[0]
    x = declined[features].to_numpy(dtype=float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _dice().explain_local(
            _ProbaOnly(model),
            x,
            dataframe=numeric,
            outcome_name="y",
            instance_id="i",
            feature_names=features,
            continuous_features=features,
            total_CFs=2,
            **{k: v for k, v in KW.items() if k != "feature_names"},
        )
    assert math.isnan(explanation.prediction)
    assert math.isnan(explanation.to_db_row("auth0|probe")["prediction"])
    assert "COULD NOT MEASURE" in explanation.params["prediction_reason"]
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught)
    # The counterfactuals themselves were produced and must not be thrown away.
    assert len(explanation.counterfactuals) >= 1


def test_dice_scores_a_recourse_it_could_not_price_as_not_measured(dice_frames) -> None:
    """DEFECT, fixed. Measured before the fix on an all-categorical frame
    (region, band) with a model keyed on region: both counterfactuals came back
    proximity=0.0 and feasibility=1.0 for a flip of region alone, silently. Those
    are the two best recourse scores there are (no distance at all, every change
    in-distribution) and nothing was measured: there was no numeric change to
    normalise or bound-check.
    """
    _numeric, _model, categorical = dice_frames
    pd = pytest.importorskip("pandas")

    class RegionModel:
        def predict(self, frame):
            frame = pd.DataFrame(frame)
            return (frame["region"] == "north").astype(int).to_numpy()

        def predict_proba(self, frame):
            positive = self.predict(frame).astype(float)
            return np.column_stack([1 - positive, positive])

    frame = categorical.copy()
    model = RegionModel()
    frame["y"] = model.predict(frame)
    features = ["region", "band"]
    declined = frame[frame["y"] == 0].iloc[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _dice().explain_local(
            model,
            np.array(declined[features].tolist(), dtype=object),
            dataframe=frame,
            outcome_name="y",
            instance_id="i",
            feature_names=features,
            continuous_features=[],
            total_CFs=2,
            **{k: v for k, v in KW.items() if k != "feature_names"},
        )
    assert explanation.counterfactuals, "DiCE found no counterfactual to judge"
    for counterfactual in explanation.counterfactuals:
        assert counterfactual.flipped_features, "a counterfactual that changes nothing"
        assert math.isnan(counterfactual.proximity)
        assert math.isnan(counterfactual.feasibility)
    assert any("COULD NOT BE MEASURED" in str(w.message) for w in caught)


def test_dice_still_prices_a_numeric_recourse(dice_frames) -> None:
    """CONTROL. The numeric frame still gets real numbers: the original
    prediction is measured (1.0 for this row), proximity is a finite normalised
    distance (0.3128 for the first counterfactual before and after the fix) and
    feasibility is the measured in-range fraction. Only base_value is NaN, and
    it says why: DiCE returns changes, not contributions against a reference
    output, so there was never a base value to report.
    """
    numeric, model, _categorical = dice_frames
    features = ["income", "age"]
    approved = numeric[numeric["y"] == 1].iloc[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        explanation = _dice().explain_local(
            model,
            approved[features].to_numpy(dtype=float),
            dataframe=numeric,
            outcome_name="y",
            instance_id="i",
            feature_names=features,
            continuous_features=features,
            total_CFs=2,
            **{k: v for k, v in KW.items() if k != "feature_names"},
        )
    assert explanation.prediction == 1.0
    assert "prediction_reason" not in explanation.params
    assert math.isnan(explanation.base_value)
    assert "none was measured" in explanation.params["base_value_reason"]
    assert explanation.counterfactuals
    for counterfactual in explanation.counterfactuals:
        assert math.isfinite(counterfactual.proximity)
        assert math.isfinite(counterfactual.feasibility)
        assert 0.0 <= counterfactual.feasibility <= 1.0
    assert not any("COULD NOT BE MEASURED" in str(w.message) for w in caught)


# === the audit artifact bundle: already honest ============================


def _explanation(contribution: float = 0.4) -> Explanation:
    return Explanation(
        method="shap.TreeExplainer",
        instance_id="s#0",
        subject_id="s",
        model_hash="m",
        data_hash="d",
        base_value=0.1,
        prediction=0.5,
        attributions=[Attribution(feature="a", contribution=contribution)],
        units="log-odds",
    )


_BUNDLE_KW = dict(subject_id="s", params={"p": 1}, library_versions={"shap": "0.52.0"}, seed=7)


def test_the_audit_bundle_digest_covers_the_content_and_excludes_only_the_clock() -> None:
    """CORRECT ALREADY, pinned. A content address that does not cover the
    content attests nothing, and one that covers the clock can never match a
    replay. Measured 2026-09-27: two calls on identical content agree
    (fe6c4c970533cfc4...), changing one attribution by 1e-06 changes the digest,
    changing only an Explanation.timestamp does not, and the excluded paths are
    declared inside the hashed payload itself.
    """
    first, sha = build_audit_artifact_bundle(explanations=[_explanation()], **_BUNDLE_KW)
    _again, sha_again = build_audit_artifact_bundle(explanations=[_explanation()], **_BUNDLE_KW)
    assert sha == sha_again

    _changed, sha_changed = build_audit_artifact_bundle(
        explanations=[_explanation(0.400001)], **_BUNDLE_KW
    )
    assert sha_changed != sha, "the digest does not cover the attributions it attests"

    reclocked = _explanation()
    reclocked.timestamp = "1999-01-01T00:00:00+00:00"
    _clock, sha_clock = build_audit_artifact_bundle(explanations=[reclocked], **_BUNDLE_KW)
    assert sha_clock == sha, "the digest covers the clock, so no replay can ever match it"

    assert first["digest_excludes"] == ["generated_at", "explanations[].timestamp"]
    assert first["content_sha256"] == sha
    assert first["explanations"][0]["timestamp"], "the clock must stay ON the bundle"

    _seedless, sha_seedless = build_audit_artifact_bundle(
        explanations=[_explanation()], **{**_BUNDLE_KW, "seed": None}
    )
    assert sha_seedless != sha, "an unseeded run addresses the same as a seeded one"


def test_the_audit_bundle_shows_an_empty_evidence_set_as_empty() -> None:
    """CORRECT ALREADY, pinned. Nothing here is unmeasurable, so there is
    nothing to refuse: the builder packs what it is given and both absences stay
    visible and machine-readable in the bundle (explanations [], decomposition
    None) rather than being dressed as content. The pin exists because a later
    change that dropped either key, or defaulted the decomposition to {}, would
    turn 'no evidence' into 'nothing to report'.
    """
    bundle, sha = build_audit_artifact_bundle(explanations=[], **_BUNDLE_KW)
    assert bundle["explanations"] == []
    assert bundle["decomposition"] is None
    assert len(sha) == 64
    assert bundle["content_sha256"] == sha


# === the worker loop: a clean stop it had not earned ======================


class _Response:
    def __init__(self, data):
        self.data = data


class _Rpc:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def execute(self):
        if self.behaviour == "raise":
            raise RuntimeError('relation "pgmq.q_xai_jobs" does not exist')
        return _Response([])


class _Client:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def rpc(self, *_a, **_k):
        return _Rpc(self.behaviour)


class _Writer:
    def __init__(self, behaviour):
        self._client = _Client(behaviour)


def _run_three_ticks(monkeypatch, behaviour: str):
    from vfairness.xai.worker import runner

    loop = runner.WorkerLoop(config=runner.WorkerConfig(poll_interval_s=0.0))
    monkeypatch.setattr(runner, "SupabaseWriter", lambda **_k: _Writer(behaviour))
    ticks = {"n": 0}

    def _fake_sleep(_seconds):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            loop.stop()

    monkeypatch.setattr(runner.time, "sleep", _fake_sleep)
    loop.run()
    return loop


def test_the_worker_does_not_report_a_clean_stop_it_never_earned(monkeypatch, caplog) -> None:
    """DEFECT, fixed. Measured before the fix with a stubbed client, three ticks
    each: a queue that answered 'empty' and a queue whose read raised
    RuntimeError('relation "pgmq.q_xai_jobs" does not exist') on EVERY tick both
    ended on the same line, 'XAI worker stopped cleanly'. _pop_one has to return
    None for both, and the per-tick warning was the only thing that knew the
    difference, so the one surface a supervisor reads said the opposite of it.
    """
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop = _run_three_ticks(monkeypatch, "raise")
    assert (loop._polls_answered, loop._polls_failed) == (0, 3)
    text = caplog.text
    assert "NOT ONE answered read" in text
    assert "stopped cleanly" not in text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_the_worker_still_reports_a_clean_stop_when_the_queue_really_was_empty(
    monkeypatch, caplog
) -> None:
    """CONTROL. An idle worker is the normal case and must still say so, or the
    alarm above is worth nothing. Measured after the fix: three answered polls,
    zero failures, 'XAI worker stopped cleanly' and no ERROR record.
    """
    caplog.set_level(logging.INFO, logger="vfairness.xai.worker")
    loop = _run_three_ticks(monkeypatch, "empty")
    assert (loop._polls_answered, loop._polls_failed) == (3, 0)
    assert "XAI worker stopped cleanly" in caplog.text
    assert "NOT ONE answered read" not in caplog.text
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)


def test_a_failed_poll_is_not_reported_as_an_empty_queue(monkeypatch, caplog) -> None:
    """DEFECT, fixed, at the poll itself. The warning already existed; what did
    not exist was any way for the loop, or a host program, to tell the two
    states apart. The counters are that way, and they are what run() reports on.
    """
    caplog.set_level(logging.WARNING, logger="vfairness.xai.worker")
    loop = _run_three_ticks(monkeypatch, "raise")
    assert loop._pop_one() is None  # the read still fails
    assert loop._polls_failed == 4
    assert "This is NOT an empty queue" in caplog.text
