"""Tests for the local-first vfairness MCP tool logic (vfairness.mcp.tools).

These exercise the pure tool functions (no `mcp` SDK required) on a small
synthetic dataset with a planted disparity, asserting they return
JSON-serializable output via the same vfairness engine the platform uses.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vfairness.mcp import tools


@pytest.fixture()
def biased_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    race = rng.choice(["White", "Black", "Hispanic"], n, p=[0.6, 0.25, 0.15])
    sex = rng.choice(["M", "F"], n)
    # zip is a strong proxy for race
    zipc = np.where(race == "Black", rng.integers(10000, 10005, n), rng.integers(20000, 20010, n))
    approved = rng.integers(0, 2, n)
    # prediction is harsher for the Black group -> planted disparity
    prediction = np.where(race == "Black", rng.binomial(1, 0.3, n), rng.binomial(1, 0.6, n))
    return pd.DataFrame(
        {
            "race": race,
            "sex": sex,
            "zip": zipc.astype(str),
            "approved": approved,
            "prediction": prediction,
        }
    )


def _assert_json(obj) -> None:
    json.dumps(obj)  # raises if not serializable


def test_load_dataframe_inline_and_validation():
    df = tools.load_dataframe(csv="a,b\n1,2\n3,4\n")
    assert list(df.columns) == ["a", "b"]
    with pytest.raises(ValueError):
        tools.load_dataframe()


def test_measure_fairness(biased_df):
    out = tools.measure_fairness(biased_df, ["race", "sex"], "prediction", "approved")
    _assert_json(out)
    assert set(["summary", "per_attribute"]).issubset(out)
    assert "race" in out["per_attribute"] and "sex" in out["per_attribute"]


def test_measure_fairness_missing_column(biased_df):
    with pytest.raises(ValueError):
        tools.measure_fairness(biased_df, ["race"], "does_not_exist", "approved")


def test_triage_dataset(biased_df):
    out = tools.triage_dataset(biased_df, ["race", "sex"], "approved")
    _assert_json(out)
    assert "report" in out and "finding_counts" in out


def test_detect_proxies_is_offline(biased_df):
    out = tools.detect_proxies(biased_df, ["race"])
    _assert_json(out)
    assert "correlations" in out and "proxy_chains" in out
    # The semantic/embedding ranker must not be invoked (local-first, no model hub).
    assert "high_risk_features" not in out


def test_analyze_intersectional(biased_df):
    out = tools.analyze_intersectional(
        biased_df, ["race", "sex"], "prediction", "approved", min_group_size=10
    )
    _assert_json(out)
    assert "result" in out or out.get("skipped")


def test_analyze_intersectional_needs_two_attributes(biased_df):
    out = tools.analyze_intersectional(biased_df, ["race"], "prediction", "approved")
    assert out.get("skipped") is True


def test_suggest_mitigation(biased_df):
    out = tools.suggest_mitigation(biased_df, ["race", "sex"], "approved")
    _assert_json(out)
    assert "feature_analysis" in out


def test_explain_decision_ranks_drivers(biased_df):
    out = tools.explain_decision(biased_df, "prediction")
    _assert_json(out)
    assert isinstance(out["drivers"], list)
    # zip is engineered as a strong proxy/driver of the prediction in the fixture.
    feats = [d["feature"] for d in out["drivers"]]
    assert "zip" in feats


def test_audit_agent_action_disparity():
    rng = np.random.default_rng(11)
    n = 300
    group = rng.choice(["A", "B"], n)
    # The agent escalates group B far more often -> a planted action disparity.
    action = np.where(
        group == "B",
        rng.choice(["escalate", "resolve"], n, p=[0.7, 0.3]),
        rng.choice(["escalate", "resolve"], n, p=[0.2, 0.8]),
    )
    df = pd.DataFrame({"group": group, "action": action})
    out = tools.audit_agent(df, "group", "action")
    _assert_json(out)
    assert set(out["per_action"]) == {"escalate", "resolve"}


def test_reference_resources_nonempty():
    assert tools.glossary_markdown().strip()
    assert tools.jurisdictions_markdown().strip()


def test_jsonify_serializes_enums():
    """Regression: enum members must serialize to their value, not {}.
    (The enum metaclass is EnumType on Python 3.11+, not EnumMeta.)"""
    import enum

    class Severity(enum.Enum):
        HIGH = "high"

    assert tools.jsonify(Severity.HIGH) == "high"
    assert tools.jsonify({"severity": Severity.HIGH}) == {"severity": "high"}


def test_measure_fairness_with_string_labels():
    """Regression: string-labelled target/prediction columns must be encoded,
    not crash with 'could not convert string to float'."""
    rng = np.random.default_rng(3)
    n = 200
    race = rng.choice(["White", "Black"], n)
    df = pd.DataFrame(
        {
            "race": race,
            "approved": np.where(rng.random(n) < 0.5, "yes", "no"),
            "prediction": np.where((race == "White") & (rng.random(n) < 0.7), "approve", "deny"),
        }
    )
    out = tools.measure_fairness(df, ["race"], "prediction", "approved")
    _assert_json(out)
    metrics = out["per_attribute"]["race"].get("metrics", {})
    assert metrics.get("demographic_parity_difference") is not None
