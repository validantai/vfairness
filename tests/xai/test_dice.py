"""
DiCE counterfactual adapter tests (backlog #P2-01).

Acceptance: a model + a recourse goal returns a CounterfactualExplanation
with N counterfactuals, each carrying ``proximity`` and ``feasibility``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dice_ml")
pd = pytest.importorskip("pandas")
from sklearn.linear_model import LogisticRegression

from vfairness.xai.explainers import (
    DiceCounterfactualExplainer,
    get_explainer,
    route_explainer,
)
from vfairness.xai.schemas import CounterfactualExplanation


def _toy_frame():
    rng = np.random.RandomState(7)
    n = 200
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    # Outcome driven by f0 + f1 so counterfactuals have a real lever.
    logit = 1.5 * f0 + 1.0 * f1 - 0.2
    y = (logit > 0).astype(int)
    return pd.DataFrame({"f0": f0, "f1": f1, "f2": f2, "approved": y})


def test_dice_returns_counterfactuals_with_proximity_and_feasibility() -> None:
    df = _toy_frame()
    feats = ["f0", "f1", "f2"]
    model = LogisticRegression(max_iter=1000).fit(df[feats].values, df["approved"].values)

    # Pick a query the model declines, so "opposite" is meaningful.
    declined = df[model.predict(df[feats].values) == 0].iloc[0]
    x = declined[feats].to_numpy(dtype=float)

    exp = DiceCounterfactualExplainer().explain_local(
        model,
        x,
        instance_id="subj-1#row-0",
        subject_id="subj-1",
        model_hash="m",
        data_hash="d",
        feature_names=feats,
        dataframe=df,
        outcome_name="approved",
        continuous_features=feats,
        total_CFs=4,
        seed=7,
    )

    assert isinstance(exp, CounterfactualExplanation)
    assert exp.method == "dice"
    assert len(exp.counterfactuals) >= 1
    for cf in exp.counterfactuals:
        assert isinstance(cf.proximity, float)
        assert isinstance(cf.feasibility, float)
        assert 0.0 <= cf.feasibility <= 1.0
        assert cf.proximity >= 0.0
        assert len(cf.flipped_features) >= 1
    assert "dice-ml" in exp.library_versions


def test_dice_to_db_row_carries_counterfactuals_camelcase() -> None:
    df = _toy_frame()
    feats = ["f0", "f1", "f2"]
    model = LogisticRegression(max_iter=1000).fit(df[feats].values, df["approved"].values)
    x = df[feats].iloc[0].to_numpy(dtype=float)

    exp = DiceCounterfactualExplainer().explain_local(
        model,
        x,
        instance_id="s#0",
        subject_id="s",
        model_hash="m",
        data_hash="d",
        feature_names=feats,
        dataframe=df,
        outcome_name="approved",
        continuous_features=feats,
        total_CFs=2,
        seed=7,
    )
    row = exp.to_db_row(owner="owner-1")
    cfs = row["params"]["counterfactuals"]
    assert isinstance(cfs, list) and len(cfs) >= 1
    # camelCase bridge: flipped_features -> flippedFeatures.
    assert "flippedFeatures" in cfs[0]


def test_dice_has_no_global_mode() -> None:
    with pytest.raises(NotImplementedError):
        DiceCounterfactualExplainer().explain_global(
            None,
            np.zeros((2, 3)),
            subject_id="s",
            model_hash="m",
            data_hash="d",
        )


def test_recourse_goal_routes_to_dice_and_registry_resolves() -> None:
    decision = route_explainer(model_type="blackbox", goal="actionable_recourse")
    assert decision.primary == "dice"
    assert isinstance(get_explainer(decision.primary), DiceCounterfactualExplainer)
