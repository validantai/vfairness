"""Beta Go-Live Stage 2, group s2g09: five confident values where nothing was
measured, each pinned at its PUBLIC entry point with a healthy-data control.

One test class per finding. Every class holds both halves, because a refusal
that fires on healthy data is a worse defect than the one it replaced and it
passes any test that only exercises the degenerate case:

* ``bias_audit``              BiasDetector.full_audit coverage over 2 rows
* ``multivariate_proxy_leakage`` chance-level AUC for a test that never ran
* ``anchors``                 a hardcoded base_value of 0.0 for a rule method
* ``integrated_gradients``    an unflagged NaN attribution, and an ungraded
                              completeness residual
* ``linear_explainer``        an unflagged NaN SHAP value, and a "prediction"
                              that was the attributions' own sum
"""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness import multivariate_proxy_leakage
from vfairness.preprocessing.bias_detection.detector import BiasDetector

# ---------------------------------------------------------------------------
# helpers


def _caught(fn, *a, **kw):
    """Call *fn* and return (result, [warning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*a, **kw)
    return out, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# 1. bias_audit -- BiasDetector.full_audit


def _two_row_frame() -> pd.DataFrame:
    return pd.DataFrame({"gender": ["M", "F"], "approved": [1, 0], "income": [50000, 52000]})


def _healthy_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "gender": rng.choice(["male", "female"], size=n),
            "approved": rng.integers(0, 2, n),
            "income": rng.normal(50_000, 5_000, n),
        }
    )


class TestBiasAuditCoverageOverTwoRows:
    """MEASURED before the fix, on the 2-row frame::

        overall_risk_score   : 0.0
        execution_coverage() : 'complete'
        assessment_coverage(): 'complete'
        recommendations      : ['No critical bias issues detected. Continue
                                monitoring and perform periodic audits.']

    while four sub-module warnings said UNASSESSED, including the detector's
    own "the overall risk score of 0.0 is NOT a measurement of low risk. Call
    BiasAuditReport.execution_coverage()" -- and that method answered
    'complete', the one word its docstring reserves for a real measurement.
    """

    def test_the_fixture_reaches_the_branch_it_is_meant_to(self):
        """The 2 rows really are non-null, so the old ``if n`` test passed.

        Without this the whole class could be green against an all-null column,
        which a different guard already caught.
        """
        report, _ = _caught(
            BiasDetector(
                _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        assert report.attribute_observations == {"gender": 2}
        assert set(report.modules_run) == {
            "historical",
            "representation",
            "disparities",
            "proxies",
        }

    def test_two_rows_are_not_complete_coverage(self):
        report, _ = _caught(
            BiasDetector(
                _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        assert report.execution_coverage() == "ran_but_assessed_nothing"
        assert report.assessment_coverage() == "none"
        assert report.unassessable_attributes() == ["gender"]
        assert report.attribute_assessed == {"gender": False}

    def test_the_recommendation_is_not_an_all_clear(self):
        report, _ = _caught(
            BiasDetector(
                _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        joined = " ".join(report.recommendations)
        assert "No critical bias issues detected" not in joined
        assert "none of them assessed anything" in joined
        assert "clears the data" in joined

    def test_the_serialised_report_carries_the_refusal(self):
        report, _ = _caught(
            BiasDetector(
                _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        exported = report.to_dict()
        assert exported["execution_coverage"] == "ran_but_assessed_nothing"
        assert exported["assessment_coverage"] == "none"
        assert exported["attribute_assessed"] == {"gender": False}
        # The count is still there beside it: the record is added, not swapped.
        assert exported["attribute_observations"] == {"gender": 2}

    def test_the_text_summary_says_it(self):
        report, _ = _caught(
            BiasDetector(
                _two_row_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        assert "none of them assessed anything" in report.summary()

    # ---- over-correction controls ----

    def test_a_populated_frame_is_still_complete(self):
        report, messages = _caught(
            BiasDetector(
                _healthy_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        assert report.execution_coverage() == "complete"
        assert report.assessment_coverage() == "complete"
        assert report.unassessable_attributes() == []
        assert report.attribute_assessed == {"gender": True}
        assert not any("NOT a measurement of low risk" in m for m in messages)

    def test_a_populated_frame_still_gets_the_all_clear_wording(self):
        report, _ = _caught(
            BiasDetector(
                _healthy_frame(), protected_attributes=["gender"], outcome_column="approved"
            ).full_audit
        )
        assert any("No critical bias issues detected" in r for r in report.recommendations)

    def test_a_real_disparity_is_still_found_and_scored(self):
        """The guard must not swallow a finding: rule 1 of the stage."""
        rng = np.random.default_rng(3)
        n = 600
        g = rng.choice(["male", "female"], size=n)
        approved = np.where(g == "male", rng.random(n) < 0.85, rng.random(n) < 0.15).astype(int)
        df = pd.DataFrame({"gender": g, "approved": approved, "income": rng.normal(5e4, 5e3, n)})
        report, _ = _caught(
            BiasDetector(df, protected_attributes=["gender"], outcome_column="approved").full_audit
        )
        assert report.execution_coverage() == "complete"
        assert report.overall_risk_score > 0.0
        assert report.disparity_findings

    def test_one_assessable_and_one_two_row_attribute_is_partial(self):
        df = _healthy_frame()
        # A second protected attribute that HAS non-null rows (so the old
        # ``if n`` test passed it) but never enough of them for any module to
        # reach a verdict.
        region = [None] * len(df)
        region[:5] = ["north", "north", "north", "south", "south"]
        df["region"] = pd.Series(region, dtype=object)
        report, _ = _caught(
            BiasDetector(
                df, protected_attributes=["gender", "region"], outcome_column="approved"
            ).full_audit
        )
        assert report.attribute_observations["region"] == 5  # it had rows
        assert report.assessment_coverage() == "partial"
        assert report.unassessable_attributes() == ["region"]
        assert report.attribute_assessed == {"gender": True, "region": False}

    def test_a_hand_built_report_falls_back_to_the_counts(self):
        from vfairness.preprocessing.bias_detection.detector import (
            AUDIT_MODULES,
            BiasAuditReport,
        )

        report = BiasAuditReport(
            timestamp="2026-09-16T00:00:00",
            dataset_info={},
            protected_attributes=["gender"],
            historical_findings=[],
            representation_findings=[],
            disparity_findings=[],
            proxy_findings=[],
            overall_risk_score=0.0,
            critical_issues=[],
            recommendations=[],
            modules_run=list(AUDIT_MODULES),
        )
        assert report.attribute_assessed is None
        assert report.assessment_coverage() == "unrecorded"
        assert report.unassessable_attributes() is None
        # "unrecorded" since 2026-09-27, and it used to be "complete".
        #
        # execution_coverage's OWN DOCSTRING has always defined "complete" as
        # "every module in AUDIT_MODULES ran AND at least one protected attribute
        # was actually assessed". The code only ever checked the first half, so a
        # report recording every module and no assessment half at all answered
        # "complete", and rendering.adapters keys its all-clear on exactly that
        # word: the canvas read "OVERALL RISK 0% MINIMAL" with four 0.00 tiles for
        # an audit over a column that was entirely NULL.
        #
        # The old reasoning was that an unrecorded second half should leave the
        # first half's answer alone. Answering "complete" IS a verdict about the
        # second half, so leaving it alone was never what that answer did.
        assert report.execution_coverage() == "unrecorded"


# ---------------------------------------------------------------------------
# 2. multivariate_proxy_leakage


def _single_group_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"group": ["a"] * n, "f1": rng.normal(size=n), "f2": rng.normal(size=n)})


def _no_leak_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "group": rng.choice(["a", "b"], size=n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )


def _leaky_frame(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(6)
    g = rng.choice(["a", "b"], size=n)
    return pd.DataFrame(
        {
            "leaky": g,
            "flat": ["z"] * n,
            "f1": (g == "a").astype(float) + rng.normal(scale=0.01, size=n),
            "f2": rng.normal(size=n),
        }
    )


class TestMultivariateProxyLeakageNotAssessed:
    """MEASURED before the fix: an empty frame, a 2-row frame and a
    single-group 200-row frame ALL returned the identical graded record
    ``{"auc": 0.5, "macro_auc": 0.5, "worst_group_auc": 0.5, "severity":
    "negligible", "systemic_leakage": false}`` with ZERO warnings, which is
    indistinguishable at every machine-read field from the genuine no-leak
    control (auc 0.5080, "negligible", false).
    """

    @pytest.mark.parametrize(
        "frame",
        [
            _single_group_frame(),
            pd.DataFrame({"group": ["a", "b"], "f1": [0.1, 0.2], "f2": [1.0, 2.0]}),
            pd.DataFrame({"group": pd.Series([], dtype=object), "f1": pd.Series([], dtype=float)}),
        ],
        ids=["single_group_200", "n_2", "empty"],
    )
    def test_an_unmeasurable_attribute_is_not_graded(self, frame):
        res, messages = _caught(multivariate_proxy_leakage, frame, ["group"])
        (rec,) = res
        assert rec.auc is None
        assert rec.macro_auc is None
        assert rec.worst_group_auc is None
        assert rec.systemic_leakage is None
        assert rec.severity == "not_assessed"
        assert rec.was_assessed() is False
        assert any("UNASSESSED" in m for m in messages)

    def test_the_serialised_record_is_null_not_chance(self):
        res, _ = _caught(multivariate_proxy_leakage, _single_group_frame(), ["group"])
        payload = json.loads(json.dumps(res[0].to_dict(), default=str))
        for key in ("auc", "macro_auc", "worst_group_auc", "systemic_leakage"):
            assert payload[key] is None, key
        assert payload["severity"] == "not_assessed"

    def test_sklearn_missing_is_not_a_clean_bill(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *a, **kw):
            if name.startswith("sklearn"):
                raise ImportError("blocked for test")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", blocked)
        res, messages = _caught(multivariate_proxy_leakage, _no_leak_frame(), ["group"])
        monkeypatch.undo()
        (rec,) = res
        assert rec.method == "unavailable"
        assert rec.auc is None and rec.systemic_leakage is None
        assert rec.severity == "not_assessed"
        assert any("scikit-learn is unavailable" in m for m in messages)

    def test_an_internal_error_is_not_a_clean_bill(self, monkeypatch):
        import vfairness.preprocessing.bias_detection.proxy as proxy_mod

        def boom(*a, **kw):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(proxy_mod.pd, "get_dummies", boom)
        res, messages = _caught(multivariate_proxy_leakage, _leaky_frame(), ["leaky"])
        (rec,) = res
        assert rec.method == "error"
        assert rec.auc is None and rec.systemic_leakage is None
        assert rec.severity == "not_assessed"
        assert any("UNASSESSED" in m for m in messages)

    def test_an_unmeasured_attribute_is_not_ranked_among_the_measured(self):
        res, _ = _caught(multivariate_proxy_leakage, _leaky_frame(), ["leaky", "flat"])
        assert [r.protected_attribute for r in res] == ["leaky", "flat"]
        assert res[0].auc is not None and res[0].severity == "severe"
        assert res[-1].auc is None and res[-1].severity == "not_assessed"

    # ---- over-correction controls ----

    def test_a_genuine_no_leak_attribute_still_measures_chance(self):
        res, messages = _caught(multivariate_proxy_leakage, _no_leak_frame(), ["group"])
        (rec,) = res
        assert rec.method == "hist_gradient_boosting"
        assert rec.was_assessed() is True
        assert rec.auc is not None and 0.4 < rec.auc < 0.65
        assert rec.severity == "negligible"
        assert rec.systemic_leakage is False  # False, not None: it WAS tested
        assert not messages

    def test_a_genuinely_leaky_attribute_is_still_severe(self):
        res, _ = _caught(multivariate_proxy_leakage, _leaky_frame(), ["leaky"])
        (rec,) = res
        assert rec.auc is not None and rec.auc > 0.9
        assert rec.systemic_leakage is True
        assert rec.severity == "severe"


# ---------------------------------------------------------------------------
# 3. anchors

anchor = pytest.importorskip("anchor", reason="anchor-exp is an optional extra")


def _anchor_setup():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return model, X


class TestAnchorsBaseValue:
    """MEASURED before the fix: ``base_value`` 0.0 on every input, healthy or
    not, travelling into ``to_db_row`` beside ``"fidelity": null`` and
    ``"stability": null`` -- the same record using null for two unmeasured
    quantities and a number for a third that was never measured at all.
    """

    def _explain(self, background):
        from vfairness.xai.explainers import AnchorsExplainer

        model, X = _anchor_setup()
        return _caught(
            AnchorsExplainer().explain_local,
            model,
            X[0],
            background,
            instance_id="i0",
            subject_id="s0",
            model_hash="mh",
            data_hash="dh",
            feature_names=["a", "b", "c", "d"],
        )

    def test_base_value_is_not_a_measured_zero(self):
        _, X = _anchor_setup()
        exp, messages = self._explain(X)
        assert math.isnan(exp.base_value)
        assert exp.params["base_value"] is None
        assert "no base value" in exp.params["base_value_reason"]
        assert any("base_value is NaN" in m for m in messages)

    def test_the_db_row_no_longer_shows_a_number_beside_two_nulls(self):
        _, X = _anchor_setup()
        exp, _ = self._explain(X)
        row = exp.to_db_row("owner")
        assert row["fidelity"] is None and row["stability"] is None
        assert math.isnan(row["base_value"])
        assert row["params"]["base_value"] is None

    def test_a_tiny_background_carries_its_own_provenance(self):
        _, X = _anchor_setup()
        exp, messages = self._explain(X[:2])
        assert exp.params["n_background"] == 2
        assert exp.params["background_sufficient"] is False
        assert "not reliable estimates" in exp.params["background_warning"]
        # The degenerate shape the caveat exists for: an EMPTY rule whose
        # precision and coverage are trivially 1.0.
        assert exp.params["rule"] == []
        assert any("background row(s)" in m for m in messages)

    # ---- over-correction control ----

    def test_a_healthy_background_still_produces_a_rule_and_no_caveat(self):
        _, X = _anchor_setup()
        exp, _ = self._explain(X)
        assert exp.params["background_sufficient"] is True
        assert "background_warning" not in exp.params
        assert exp.params["rule"]
        assert 0.0 < exp.params["precision"] <= 1.0
        assert 0.0 < exp.params["coverage"] <= 1.0


# ---------------------------------------------------------------------------
# 4. integrated_gradients

torch = pytest.importorskip("torch", reason="torch/captum are optional extras")
pytest.importorskip("captum", reason="torch/captum are optional extras")


def _ig_models():
    import torch.nn as nn

    class Saturated(nn.Module):
        def forward(self, x):
            return torch.tanh(x.sum(dim=1, keepdim=True) * 50.0)

    class Linear3(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(3, 1)
            with torch.no_grad():
                self.l.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
                self.l.bias.zero_()

        def forward(self, x):
            return self.l(x)

    return Saturated(), Linear3()


class TestIntegratedGradientsRefusals:
    """MEASURED before the fix:

    (1) with ``income=NaN`` in x, ``attributions=[('age',2.0),('income',nan),
        ('zip',2.0)]`` and ``prediction=nan``, emitted with ``warnings=[]`` and
        no flag; ranking by ``abs(contribution)`` puts NaN LAST, so the feature
        nothing could be said about is reported as the least influential.
    (2) a saturated tanh at ``n_steps=2`` returned three finite 0.0
        contributions with ``completeness_residual=1.0`` -- 100% of the
        prediction unexplained -- and nothing warned or graded it.
    """

    def _explain(self, model, x, **kw):
        from vfairness.xai.explainers import IntegratedGradientsExplainer

        return _caught(
            IntegratedGradientsExplainer().explain_local,
            model,
            x,
            instance_id="i",
            subject_id="s",
            model_hash="h",
            data_hash="d",
            feature_names=["age", "income", "zip"],
            **kw,
        )

    def test_a_nan_attribution_is_named_not_merely_emitted(self):
        _, linear = _ig_models()
        exp, messages = self._explain(linear, np.array([1.0, np.nan, 1.0], dtype=np.float32))
        assert exp.params["unattributed_features"] == ["income"]
        assert exp.params["attributions_complete"] is False
        assert exp.params["prediction_measured"] is False
        assert any("no attribution could be computed for income" in m for m in messages)
        # The NaN is still in the list: nothing was deleted, only disclosed.
        contributions = {a.feature: a.contribution for a in exp.attributions}
        assert math.isnan(contributions["income"])

    def test_an_unexplained_prediction_is_graded_not_just_recorded(self):
        saturated, _ = _ig_models()
        exp, messages = self._explain(
            saturated, np.array([1.0, 1.0, 1.0], dtype=np.float32), n_steps=2
        )
        assert exp.params["completeness_residual"] == pytest.approx(1.0)
        assert exp.params["completeness_ok"] is False
        assert any("completeness axiom does not hold" in m for m in messages)
        # The fixture really is the confident-value shape: three FINITE zeros.
        assert [a.contribution for a in exp.attributions] == [0.0, 0.0, 0.0]
        assert exp.prediction == pytest.approx(1.0)

    def test_an_uncomputable_residual_is_none_not_a_pass(self):
        _, linear = _ig_models()
        exp, _ = self._explain(linear, np.array([1.0, np.nan, 1.0], dtype=np.float32))
        assert exp.params["completeness_ok"] is None
        assert math.isnan(exp.params["completeness_residual"])

    # ---- over-correction controls ----

    def test_a_healthy_explanation_still_measures_and_passes(self):
        _, linear = _ig_models()
        exp, messages = self._explain(linear, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        assert [a.contribution for a in exp.attributions] == pytest.approx([1.0, 2.0, 3.0])
        assert exp.prediction == pytest.approx(6.0)
        assert exp.params["completeness_ok"] is True
        assert exp.params["attributions_complete"] is True
        assert exp.params["unattributed_features"] == []
        assert not messages

    def test_the_saturated_model_passes_once_it_is_given_enough_steps(self):
        """completeness_ok is a real check, not a blanket refusal for tanh."""
        saturated, _ = _ig_models()
        exp, _ = self._explain(
            saturated, np.array([0.005, 0.005, 0.005], dtype=np.float32), n_steps=400
        )
        assert exp.params["completeness_ok"] is True


# ---------------------------------------------------------------------------
# 5. linear_explainer

pytest.importorskip("shap", reason="shap is an optional extra")


def _linear_setup():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(1)
    bg = rng.normal(size=(400, 3))
    y = (bg @ [0.5, -3.0, 0.8] + 0.1 + rng.normal(scale=0.3, size=400) > 0).astype(int)
    return LogisticRegression(max_iter=500).fit(bg, y), bg


class TestLinearShapRefusals:
    """MEASURED before the fix, with ``income=NaN`` in x::

        attributions : [('age',0.4353),('income',nan),('zip',0.7720)]
        prediction   : nan          params: {}          warnings: []

    ``income`` is the model's strongest feature (coef -5.60, about 3.6x the
    next) and the library's own ranker ordered these ['zip','age','income'],
    reporting it LAST. ``prediction`` was ``base_value + sum(attributions)``,
    an identity that agrees with the attributions by construction.
    """

    def _explain(self, x, background, **kw):
        from vfairness.xai.explainers import LinearShapExplainer

        model, _ = _linear_setup()
        return _caught(
            LinearShapExplainer().explain_local,
            model,
            x,
            background=background,
            instance_id="i",
            subject_id="s",
            model_hash="h",
            data_hash="d",
            feature_names=["age", "income", "zip"],
            **kw,
        )

    def test_a_nan_shap_value_is_named(self):
        _, bg = _linear_setup()
        x = bg[0].copy()
        x[1] = np.nan
        exp, messages = self._explain(x, bg)
        assert exp.params["unattributed_features"] == ["income"]
        assert exp.params["attributions_complete"] is False
        assert any("no SHAP value could be computed for income" in m for m in messages)
        # The fixture exercises the ranking trap it is named for.
        contributions = np.array([a.contribution for a in exp.attributions], dtype=float)
        order = [["age", "income", "zip"][i] for i in np.argsort(-np.abs(contributions))]
        assert order[-1] == "income"

    def test_an_empty_background_refuses_on_every_attribution(self):
        _, bg = _linear_setup()
        exp, messages = self._explain(bg[0].copy(), bg[:0])
        assert exp.params["unattributed_features"] == ["age", "income", "zip"]
        assert exp.params["base_value_measured"] is False
        assert exp.params["local_accuracy_ok"] is None
        assert exp.params["local_accuracy_residual"] is None
        assert any("INCOMPLETE" in m for m in messages)

    def test_prediction_is_the_model_not_the_attributions_own_sum(self):
        """The distinguishing case, because on HEALTHY data the identity and
        the measurement coincide exactly (SHAP for a linear model is closed
        form), so healthy data can never tell them apart. An empty background
        separates them: the attributions and base_value are NaN while the model
        is perfectly scoreable, so ``base + sum(values)`` is NaN and the model's
        own output is not.
        """
        model, bg = _linear_setup()
        exp, _ = self._explain(bg[0].copy(), bg[:0])
        real = float(model.decision_function(bg[0].reshape(1, -1))[0])
        assert math.isnan(exp.params["local_accuracy_identity"])
        assert math.isfinite(exp.prediction)
        assert exp.prediction == pytest.approx(real)

    def test_local_accuracy_is_checked_not_only_recorded(self):
        model, bg = _linear_setup()
        exp, _ = self._explain(bg[0].copy(), bg)
        real = float(model.decision_function(bg[0].reshape(1, -1))[0])
        assert exp.prediction == pytest.approx(real)
        assert exp.params["local_accuracy_residual"] is not None
        assert exp.params["local_accuracy_ok"] is True

    def test_a_probability_request_is_not_silently_relabelled(self):
        _, bg = _linear_setup()
        exp, messages = self._explain(bg[0].copy(), bg, units="probability")
        assert exp.units == "log-odds"
        assert exp.params["units_requested"] == "probability"
        assert any("log-odds margin space" in m for m in messages)

    # ---- over-correction controls ----

    def test_a_healthy_explanation_still_ranks_the_dominant_feature_first(self):
        _, bg = _linear_setup()
        exp, messages = self._explain(bg[0].copy(), bg)
        contributions = np.array([a.contribution for a in exp.attributions], dtype=float)
        assert np.isfinite(contributions).all()
        order = [["age", "income", "zip"][i] for i in np.argsort(-np.abs(contributions))]
        assert order[0] == "income"
        assert exp.params["attributions_complete"] is True
        assert exp.params["unattributed_features"] == []
        assert not messages

    def test_local_accuracy_is_a_real_check_and_passes_on_healthy_data(self):
        _, bg = _linear_setup()
        exp, _ = self._explain(bg[1].copy(), bg)
        assert exp.params["local_accuracy_ok"] is True
        assert exp.params["local_accuracy_residual"] < exp.params["local_accuracy_tol"]
