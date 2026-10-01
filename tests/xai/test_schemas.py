"""
Contract-shape tests. Lock the to_db_row outputs so the Supabase writer
keeps emitting the column names the canonical assessment-layer tables
expect.
"""

from __future__ import annotations

from vfairness.xai.schemas import (
    Attribution,
    Explanation,
    FairnessDecomposition,
)


def _make_explanation() -> Explanation:
    return Explanation(
        method="shap.TreeExplainer",
        instance_id="applicant-1",
        subject_id="subject-loan-xgb-v3",
        model_hash="sha256:abc",
        data_hash="sha256:def",
        base_value=-0.4,
        prediction=-1.1,
        attributions=[
            Attribution(feature="credit_util", contribution=-0.4),
            Attribution(feature="income", contribution=-0.3),
        ],
        units="log-odds",
        library_versions={"shap": "0.46.0"},
    )


def test_explanation_to_db_row_keys() -> None:
    row = _make_explanation().to_db_row(owner="auth0|test")
    assert row["owner"] == "auth0|test"
    assert row["subject_id"] == "subject-loan-xgb-v3"
    assert row["scope"] == "local"
    assert row["instance_id"] == "applicant-1"
    assert row["units"] == "log-odds"
    assert row["view_provenance"] == ["xai"]
    assert isinstance(row["attributions"], list)
    assert row["attributions"][0]["feature"] == "credit_util"


def test_attributions_serialise_as_camelcase() -> None:
    """
    Lock the wire contract: nested JSONB payloads use camelCase so the
    React reader matches without a transformer. Bug found in review:
    asdict() was emitting snake_case `base_value`, silently invisible
    to the frontend Attribution interface.
    """
    from vfairness.xai.schemas import Attribution, Explanation

    e = Explanation(
        method="shap.TreeExplainer",
        instance_id="a-1",
        subject_id="s-1",
        model_hash="x",
        data_hash="y",
        base_value=-0.4,
        prediction=-1.0,
        attributions=[Attribution(feature="f1", contribution=0.2, base_value=-0.4)],
        units="log-odds",
    )
    row = e.to_db_row(owner="o")
    attr = row["attributions"][0]
    assert "baseValue" in attr, "Attribution.base_value must serialise as baseValue"
    assert "base_value" not in attr


def test_explanation_global_scope() -> None:
    e = _make_explanation()
    e.instance_id = ""
    row = e.to_db_row(owner="auth0|test")
    assert row["scope"] == "global"


def test_fairness_decomposition_asserts_identity_before_db_row() -> None:
    good = FairnessDecomposition(
        id="d-1",
        subject_id="subject-1",
        metric="demographic_parity",
        protected_attribute="gender",
        total_disparity=0.08,
        per_feature={"a": 0.05, "b": 0.03},
        proxy_scores={"a": 0.6, "b": 0.4},
        flagged_proxies=["a"],
        audit_artifact_id="audit-1",
    )
    row = good.to_db_row(owner="auth0|test")
    assert row["metric"] == "demographic_parity"
    assert row["total_disparity"] == 0.08
    assert row["per_feature"] == {"a": 0.05, "b": 0.03}
    assert "a" in row["flagged_proxies"]
