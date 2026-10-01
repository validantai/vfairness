"""BGL3 batch xai-1: do the explainability units refuse honestly?

Five files, fourteen units, one question per unit: when the quantity it reports
cannot be computed, does it say so, or does it hand back the value that reads as
its clean answer?

Every number in these docstrings was MEASURED on the real unit, before and
after the fix, on this machine. Five defects were proven by execution:

1. ``slack_adversarial_probe`` against a model returning all-NaN predictions
   returned ``(False, '')`` with zero warnings, byte-identical to the same call
   on a flat clean model. ``gap > threshold`` is False for a NaN gap, so a draw
   that produced nothing fell through to the clean verdict. Its multi-seed twin
   in the same file took this fix as READINESS-6 on 2026-09-10; the single-draw
   entry point, exported in its own right, was left behind.
2. ``local_r_squared`` returned ``1.0``, the top of the fidelity scale, for an
   EMPTY local sample and for a single-point sample whose surrogate happened to
   match (and ``0.0`` when it did not). sklearn.metrics.r2_score raises on the
   first and returns nan with an UndefinedMetricWarning on the second.
3. ``TreeShapExplainer.explain_local`` on an ``x`` of shape (0, 3) returned an
   Explanation with ``prediction`` 0.41250, ``base_value`` 0.41250,
   ``attributions`` [] and no warning: a complete-looking explanation of an
   instance that was never examined.
4. ``_build_attributions`` paired names with values through a bare ``zip``,
   which stops at the shorter sequence in silence. With ``feature_names``
   ["a", "b"] for three features, TreeShap returned two attributions summing to
   0.50906 beside base_value 0.41250 and prediction 1.00000, so 0.07844 of the
   movement, 13 percent, sat on a deleted feature. Through
   ``IntegratedGradientsExplainer.explain_local`` the same mismatch returned two
   attributions summing to 0.18012 against a movement of 0.29050, leaving 38.0
   percent of it unexplained, while params reported ``completeness_ok=True``,
   ``attributions_complete=True`` and a residual of 3.7e-09: the axiom was
   checked against the decomposition the method COMPUTED, not the one it
   RETURNED.
5. ``SupabaseWriter.update_job_progress`` threw the response away. PostgREST
   returns an empty data list when the id filter matches nothing, so an update
   of a job row that does not exist returned None exactly like one that updated
   a row, with no warning. ``WorkerLoop._dispatch`` passes the pgmq ``msg_id``
   when the payload carries no ``job_id``, so the zero-row update is a live
   path.

The refusals that were already correct are pinned too, because a unit that
refuses everything is as wrong as one that answers everything, and only the
healthy case tells them apart.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

from vfairness.xai.diagnostics.adversarial import (
    multi_seed_adversarial_probe,
    slack_adversarial_probe,
)
from vfairness.xai.diagnostics.faithfulness import local_r_squared, removal_curve_auc
from vfairness.xai.schemas import XaiAssessment, XaiDiagnostics
from vfairness.xai.storage.supabase_writer import SupabaseWriter, SupabaseWriterError

# === shared fixtures ========================================================

_WEIGHTS = np.array([0.5, -0.3, 0.2, 0.1])


def _background(n: int = 200) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 4))


def _clean_predict(arr: np.ndarray) -> np.ndarray:
    """A flat, honest model: identical behaviour on and off the manifold."""
    arr = np.asarray(arr, dtype=float)
    p = 1.0 / (1.0 + np.exp(-(arr @ _WEIGHTS)))
    return np.column_stack([1.0 - p, p])


def _outage_predict(arr: np.ndarray) -> np.ndarray:
    """A model that answers nothing: every score non-finite."""
    arr = np.asarray(arr, dtype=float)
    return np.full((len(arr), 2), np.nan)


def _scaffolded_predict(arr: np.ndarray) -> np.ndarray:
    """Slack-style OOD scaffold: innocent on the x1 == x0 manifold, biased off it."""
    arr = np.asarray(arr, dtype=float)
    on_manifold = np.abs(arr[:, 1] - arr[:, 0]) < 0.5
    return on_manifold.astype(float)


def _manifold_background(n: int = 400) -> np.ndarray:
    rng = np.random.RandomState(7)
    a = rng.normal(0, 1, n)
    return np.column_stack([a, a])


def _forest(n_features: int = 3):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, n_features))
    y = (X[:, 0] + 0.5 * rng.normal(size=60) > 0).astype(int)
    return RandomForestClassifier(n_estimators=8, random_state=0).fit(X, y), X


# === 1. slack_adversarial_probe ============================================


def test_slack_probe_refuses_when_the_draw_produced_no_gap() -> None:
    """DEFECT CASE. Measured before the fix, on the same x and background:

        clean model   -> (False, '') and 0 warnings
        all-NaN model -> (False, '') and 0 warnings

    byte-identical, so no caller could tell an explainer the probe had cleared
    from one the probe never ran on. False here is a verdict of "no OOD
    scaffolding detected".
    """
    bg = _background()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flag, reason = slack_adversarial_probe(predict_fn=_outage_predict, x=bg[0], background=bg)
    assert flag is None, "an unmeasurable draw must not report the clean verdict"
    assert reason.startswith("COULD NOT CHECK")
    messages = [str(w.message) for w in caught]
    assert any("slack_adversarial_probe" in m and "could not check" in m for m in messages), (
        f"the refusal must name itself and its counts, got {messages}"
    )


def test_slack_probe_still_decides_when_the_draw_is_measurable() -> None:
    """NOT REFUSING EVERYTHING. Measured after the fix:

    scaffolded model -> (True, 'Slack probe: ... differs from background by 0.632 ...')
    clean model      -> (False, '')
    """
    scaffold_bg = _manifold_background()
    flag, reason = slack_adversarial_probe(
        predict_fn=_scaffolded_predict, x=scaffold_bg[0], background=scaffold_bg, threshold=0.3
    )
    assert flag is True and reason

    clean_bg = _background()
    flag, reason = slack_adversarial_probe(
        predict_fn=_clean_predict, x=clean_bg[0], background=clean_bg
    )
    assert flag is False and reason == ""


def test_multi_seed_probe_refusal_and_verdict_agree_with_the_single_draw() -> None:
    """CORRECT ALREADY (READINESS-6), pinned so the twins cannot diverge again.

    Measured: all-NaN model -> flag=None, confidence=nan, mean_gap=nan and one
    warning; clean model -> flag=False, confidence=1.0, mean_gap=0.010273.
    """
    bg = _background()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = multi_seed_adversarial_probe(
            predict_fn=_outage_predict, x=bg[0], background=bg, n_seeds=4
        )
    assert out.flag is None
    assert np.isnan(out.confidence) and np.isnan(out.mean_gap)
    assert out.reason.startswith("COULD NOT CHECK")
    assert len(caught) >= 1

    ok = multi_seed_adversarial_probe(predict_fn=_clean_predict, x=bg[0], background=bg, n_seeds=4)
    assert ok.flag is False and np.isfinite(ok.mean_gap) and ok.reason == ""


# === 2. local_r_squared =====================================================


@pytest.mark.parametrize(
    ("surrogate", "true", "before"),
    [
        (np.array([]), np.array([]), 1.0),
        (np.array([0.5]), np.array([0.5]), 1.0),
        (np.array([0.3]), np.array([0.5]), 0.0),
    ],
)
def test_local_r_squared_refuses_below_two_observations(surrogate, true, before) -> None:
    """DEFECT CASE. Measured before the fix, beside sklearn on the same input:

        n=0 (both empty)        ours 1.0   sklearn ValueError
        n=1, surrogate == true  ours 1.0   sklearn nan + UndefinedMetricWarning
        n=1, surrogate != true  ours 0.0   sklearn nan + UndefinedMetricWarning

    1.0 is perfect local fidelity and it is recorded on Explanation.fidelity,
    so a surrogate fitted on one point, or on none, was published as explaining
    the model exactly. R^2 is variance explained and one point has none.
    """
    assert before in (0.0, 1.0), "the pre-fix value was a real point on the scale"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = local_r_squared(surrogate_predictions=surrogate, true_predictions=true)
    assert np.isnan(got)
    assert any("local_r_squared" in str(w.message) for w in caught)


def test_local_r_squared_refuses_non_finite_predictions() -> None:
    """DEFECT CASE (disclosure). Measured before the fix: every true prediction
    NaN returned nan SILENTLY, with no warning naming the gap, so a caller
    reading only the value could not tell it from a computed nan.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = local_r_squared(
            surrogate_predictions=np.array([0.1, 0.2, 0.3]),
            true_predictions=np.array([np.nan, np.nan, np.nan]),
        )
    assert np.isnan(got)
    assert any("local_r_squared" in str(w.message) for w in caught)


def test_local_r_squared_still_scores_measurable_samples() -> None:
    """NOT REFUSING EVERYTHING, and the sklearn-parity subject pinned by
    tests/test_audit3_vision_render_tail.py is untouched at n=4.

    Measured after the fix: healthy 0.992 (sklearn 0.992), constant target with
    an arbitrary surrogate 0.0, constant target with a zero-residual surrogate
    1.0, each with no warning.
    """
    true = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sur = np.array([0.11, 0.19, 0.31, 0.39, 0.52])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = local_r_squared(surrogate_predictions=sur, true_predictions=true)
    assert got == pytest.approx(r2_score(true, sur))
    assert not caught

    const = np.array([0.5, 0.5, 0.5, 0.5])
    assert local_r_squared(
        surrogate_predictions=np.array([0.1, 0.9, 0.2, 0.7]), true_predictions=const
    ) == pytest.approx(0.0)
    assert local_r_squared(
        surrogate_predictions=const.copy(), true_predictions=const
    ) == pytest.approx(1.0)


# === 3. removal_curve_auc (the xai twin, which delegates) ===================


def test_removal_curve_auc_returns_nan_when_the_curve_could_not_be_computed() -> None:
    """CORRECT ALREADY. Measured: an all-NaN background_mean, which is what an
    empty background produces, gives nan, not the 0.0 that grades "weak".
    Healthy input on the same model gives 0.159484, inside the documented
    [0, 1] contract.
    """
    x = np.array([1.0, 2.0, -1.0, 0.5])
    attributions = _WEIGHTS * x

    def margin(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        return 1.0 / (1.0 + np.exp(-(arr @ _WEIGHTS)))

    unmeasurable = removal_curve_auc(
        predict_fn=margin, x=x, attributions=attributions, background_mean=np.full(4, np.nan)
    )
    assert np.isnan(unmeasurable)

    healthy = removal_curve_auc(
        predict_fn=margin, x=x, attributions=attributions, background_mean=np.zeros(4)
    )
    assert np.isfinite(healthy) and 0.0 <= healthy <= 1.0


# === 4. the SHAP adapters ===================================================


def test_tree_shap_refuses_an_instance_with_nothing_to_attribute() -> None:
    """DEFECT CASE. Measured before the fix, on x of shape (0, 3) against a
    fitted 8-tree forest: an Explanation with prediction 0.41250000000000003,
    base_value 0.41250000000000003, attributions [] and zero warnings, which a
    reader cannot tell from an explanation whose features all contributed
    nothing.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import TreeShapExplainer

    model, _ = _forest()
    with pytest.raises(ValueError, match="nothing to attribute"):
        TreeShapExplainer().explain_local(
            model,
            np.zeros((0, 3)),
            instance_id="i",
            subject_id="s",
            model_hash="m",
            data_hash="d",
        )


def test_shap_adapters_refuse_a_feature_name_count_mismatch() -> None:
    """DEFECT CASE. Measured before the fix with feature_names ["a", "b"] for
    three features:

        TreeShap   -> 2 attributions summing to 0.50906, base_value 0.41250,
                      prediction 1.00000; the dropped feature carried 0.07844,
                      13 percent of the movement, and nothing said so.
        KernelShap -> 2 attributions summing to -0.38750, base_value 0.39375,
                      prediction 0.0.

    In both, the returned decomposition no longer adds up to the returned
    prediction, which is the SHAP axiom the explanation exists to satisfy.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer, TreeShapExplainer

    model, X = _forest()
    kwargs = dict(instance_id="i", subject_id="s", model_hash="m", data_hash="d")
    with pytest.raises(ValueError, match="2 name.* for 3 attributed value"):
        TreeShapExplainer().explain_local(model, X[0], feature_names=["a", "b"], **kwargs)
    with pytest.raises(ValueError, match="2 name.* for 3 attributed value"):
        KernelShapExplainer().explain_local(
            model.predict_proba, X[0], X[:20], feature_names=["a", "b"], **kwargs
        )


def test_shap_adapters_still_explain_a_healthy_instance() -> None:
    """NOT REFUSING EVERYTHING. Measured after the fix, 3 features and matching
    names: TreeShap gives base_value 0.41250 with contributions
    (0.45858, 0.05048, 0.07844) and prediction 1.0, which is
    rf.predict_proba(x)[0, 1] exactly; KernelShap gives base_value 0.40000 with
    three finite contributions.
    """
    pytest.importorskip("shap")
    from vfairness.xai.explainers.shap_adapter import KernelShapExplainer, TreeShapExplainer

    model, X = _forest()
    kwargs = dict(instance_id="i", subject_id="s", model_hash="m", data_hash="d")

    tree = TreeShapExplainer().explain_local(model, X[0], feature_names=["a", "b", "c"], **kwargs)
    assert [a.feature for a in tree.attributions] == ["a", "b", "c"]
    assert all(np.isfinite(a.contribution) for a in tree.attributions)
    assert tree.prediction == pytest.approx(float(model.predict_proba(X[:1])[0, 1]))

    kernel = KernelShapExplainer().explain_local(
        model.predict_proba, X[0], X[:20], feature_names=["a", "b", "c"], **kwargs
    )
    assert len(kernel.attributions) == 3
    assert all(np.isfinite(a.contribution) for a in kernel.attributions)

    # A global run is one Explanation per row, so zero rows in is zero out. That
    # is a per-row mapping and not a verdict, which is why it is pinned as a
    # count rather than refused.
    rows = TreeShapExplainer().explain_global(
        model,
        X[:2],
        subject_id="s",
        model_hash="m",
        data_hash="d",
        feature_names=["a", "b", "c"],
    )
    assert len(rows) == 2
    assert (
        len(
            TreeShapExplainer().explain_global(
                model,
                np.zeros((0, 3)),
                subject_id="s",
                model_hash="m",
                data_hash="d",
                feature_names=["a", "b", "c"],
            )
        )
        == 0
    )


# === 5. the Integrated Gradients adapter ====================================


def _torch_linear(weights: list[float]):
    import torch
    import torch.nn as nn

    class Lin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(len(weights), 1, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(torch.tensor([weights], dtype=torch.float32))

        def forward(self, t):  # noqa: ANN001, ANN201
            return self.lin(t)

    return Lin()


def test_ig_refuses_a_feature_name_count_mismatch() -> None:
    """DEFECT CASE. Measured before the fix on a 3-feature linear torch model
    with feature_names ["a", "b"]: two attributions summing to 0.18012 while
    prediction minus base_value was 0.29050, so 38.0 percent of the movement
    was missing from what the caller received, and params still reported
    completeness_ok=True, attributions_complete=True and residual 3.7e-09 with
    zero warnings. The axiom had been checked against the full attribution
    vector rather than the truncated one that was returned.
    """
    pytest.importorskip("torch")
    pytest.importorskip("captum")
    from vfairness.xai.explainers.ig_adapter import IntegratedGradientsExplainer

    model = _torch_linear([1.0, -0.5, 0.25])
    background = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="2 name.* for 3 attributed value"):
        IntegratedGradientsExplainer().explain_local(
            model,
            background[0],
            background,
            instance_id="i",
            subject_id="s",
            model_hash="m",
            data_hash="d",
            feature_names=["a", "b"],
        )


def test_ig_still_explains_and_still_discloses_an_unattributable_feature() -> None:
    """NOT REFUSING EVERYTHING, plus the refusal that was already CORRECT.

    Measured after the fix, weights [1.0, -0.5, 0.25]: three attributions,
    completeness_ok True, residual 3.7e-09, scale 0.29050, no warnings. With
    the first weight NaN: contribution f0 NaN, unattributed_features ['f0'],
    attributions_complete False, completeness_ok None (not a pass),
    prediction_measured False, and two warnings naming both gaps.
    """
    pytest.importorskip("torch")
    pytest.importorskip("captum")
    from vfairness.xai.explainers.ig_adapter import IntegratedGradientsExplainer

    background = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)
    kwargs = dict(instance_id="i", subject_id="s", model_hash="m", data_hash="d")

    healthy = IntegratedGradientsExplainer().explain_local(
        _torch_linear([1.0, -0.5, 0.25]), background[0], background, **kwargs
    )
    assert len(healthy.attributions) == 3
    assert healthy.params["completeness_ok"] is True
    assert healthy.params["attributions_complete"] is True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        broken = IntegratedGradientsExplainer().explain_local(
            _torch_linear([float("nan"), -0.5, 0.25]), background[0], background, **kwargs
        )
    assert broken.params["unattributed_features"] == ["f0"]
    assert broken.params["attributions_complete"] is False
    assert broken.params["completeness_ok"] is None, "unchecked is not a pass"
    assert broken.params["prediction_measured"] is False
    assert len(caught) >= 2

    rows = IntegratedGradientsExplainer().explain_global(
        _torch_linear([1.0, -0.5, 0.25]),
        background[:2],
        background,
        subject_id="s",
        model_hash="m",
        data_hash="d",
    )
    assert len(rows) == 2


def test_completeness_scale_reports_nan_when_the_movement_is_not_finite() -> None:
    """CORRECT ALREADY. Measured: (0.8, 0.3) gives 0.5, a non-finite prediction
    gives nan rather than a denominator, and a model that did not move gives
    0.0, which collapses the caller's tolerance to residual <= 0.0 instead of
    dividing by zero.

    The INFINITE prediction is the case that does the work here. A NaN one
    returns nan through the arithmetic whether the guard is there or not, so
    asserting only that could never disagree with a version that had no guard
    (sabotage S10 stayed green on exactly that). An infinite movement without
    the guard returns inf, and a denominator of inf makes
    ``residual <= tol * scale`` True for every residual there is, which is a
    completeness PASS handed out over an unmeasurable model.
    """
    pytest.importorskip("torch")
    from vfairness.xai.explainers.ig_adapter import completeness_scale

    assert completeness_scale(0.8, 0.3, np.array([0.3, 0.2])) == pytest.approx(0.5)
    assert np.isnan(completeness_scale(float("inf"), 0.3, np.array([0.3, 0.2])))
    assert np.isnan(completeness_scale(float("nan"), 0.3, np.array([0.3, 0.2])))
    assert completeness_scale(0.5, 0.5, np.array([5.0, 0.0])) == 0.0


# === 6. SupabaseWriter ======================================================


class _FakeResponse:
    def __init__(self, data) -> None:  # noqa: ANN001
        self.data = data


class _FakeResponseWithoutData:
    """A response carrying no representation at all (return=minimal)."""


class _FakeQuery:
    def __init__(self, log: list, data, response_cls) -> None:  # noqa: ANN001
        self._log = log
        self._data = data
        self._response_cls = response_cls

    def insert(self, row):  # noqa: ANN001, ANN201
        self._log.append(("insert", row))
        return self

    def update(self, row):  # noqa: ANN001, ANN201
        self._log.append(("update", row))
        return self

    def eq(self, column, value):  # noqa: ANN001, ANN201
        self._log.append(("eq", column, value))
        return self

    def execute(self):  # noqa: ANN201
        self._log.append(("execute",))
        if self._response_cls is _FakeResponseWithoutData:
            return _FakeResponseWithoutData()
        return _FakeResponse(self._data)


class _FakeClient:
    def __init__(self, data, response_cls=_FakeResponse) -> None:  # noqa: ANN001
        self.log: list = []
        self._data = data
        self._response_cls = response_cls

    def table(self, name):  # noqa: ANN001, ANN201
        self.log.append(("table", name))
        return _FakeQuery(self.log, self._data, self._response_cls)


def _writer(data, response_cls=_FakeResponse) -> SupabaseWriter:
    """A writer around a stub client.

    ``__init__`` is bypassed on purpose: it imports supabase, which is an
    optional extra that is not installed in this environment, and the methods
    under test only ever touch ``self._client``.
    """
    writer = object.__new__(SupabaseWriter)
    writer._client = _FakeClient(data, response_cls)
    writer._url = "https://stub.invalid"
    writer._key = "stub"
    return writer


def test_update_job_progress_refuses_a_zero_row_update() -> None:
    """DEFECT CASE. Measured before the fix, the same call twice:

        id that exists  -> returned None, resp.data == [{"id": "job-1", ...}]
        id that does not -> returned None, resp.data == []

    identical to the caller, no warning, so a worker recording "failed" against
    a job row that is not there logged a clean handover.
    """
    with pytest.raises(SupabaseWriterError, match="matched no row"):
        _writer([]).update_job_progress("job-that-does-not-exist", status="running", progress=50)


def test_update_job_progress_confirms_a_real_update_and_flags_an_unconfirmable_one() -> None:
    """NOT REFUSING EVERYTHING, and the third state is distinguishable.

    Measured after the fix: one matched row returns 1; a response carrying no
    data attribute returns None and warns, rather than asserting success.
    """
    assert (
        _writer([{"id": "job-1", "status": "running"}]).update_job_progress(
            "job-1", status="running", progress=50
        )
        == 1
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _writer(None, _FakeResponseWithoutData).update_job_progress(
            "job-1", status="running", progress=50
        )
    assert out is None
    assert any("COULD NOT BE CONFIRMED" in str(w.message) for w in caught)


def _assessment() -> XaiAssessment:
    return XaiAssessment(
        id="",
        subject_id="s",
        created_at="",
        explainer="shap.TreeExplainer",
        scope="local",
        explanations=[],
        sp_shap_representatives=None,
        diagnostics=XaiDiagnostics(faithfulness=float("nan"), stability=None),
        audit_artifact_id="aa-1",
    )


def test_write_assessment_and_audit_artifact_already_refuse_a_zero_row_insert() -> None:
    """CORRECT ALREADY, pinned beside the update above because they share a
    class and only one of the three checked its response.

    Measured: both return the new id on a one-row insert and both raise
    SupabaseWriterError naming "0 rows written" on an empty response.
    """
    assert _writer([{"id": "as-1"}]).write_assessment("owner", _assessment()) == "as-1"
    with pytest.raises(SupabaseWriterError, match="0 rows written"):
        _writer([]).write_assessment("owner", _assessment())

    artifact = dict(
        owner="owner",
        subject_id="s",
        data_hash="d",
        model_hash="m",
        params={},
        library_versions={},
        seed=1,
        artifact_uri="s3://bucket/key",
    )
    assert _writer([{"id": "aa-1"}]).write_audit_artifact(**artifact) == "aa-1"
    with pytest.raises(SupabaseWriterError, match="0 rows written"):
        _writer([]).write_audit_artifact(**artifact)


def test_write_assessment_carries_a_not_assessed_diagnostic_through_unchanged() -> None:
    """CORRECT ALREADY. A faithfulness of nan is the could-not-check state, and
    the row must not launder it into a number. Measured: the row's diagnostics
    are {'faithfulness': nan, 'stability': None, 'adversarial_flag': False,
    'adversarial_reason': None, 'notes': []}, so the nan reaches the wire as the
    invalid JSON literal NaN and the insert fails loudly rather than silently
    becoming 0.0 or null.
    """
    writer = _writer([{"id": "as-1"}])
    writer.write_assessment("owner", _assessment())
    row = next(entry[1] for entry in writer._client.log if entry[0] == "insert")
    assert np.isnan(row["diagnostics"]["faithfulness"])
    assert row["diagnostics"]["stability"] is None
    assert row["owner"] == "owner"
