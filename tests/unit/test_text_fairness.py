"""Text-classifier identity-term fairness test suite, Workstream E.

A biased mock toxicity classifier scores any text mentioning 'muslim' higher;
the analyzer must flag that group as worst with a significant, high-severity
gap. A fair classifier must not flag anyone.
"""

import numpy as np
import pytest

from vfairness.llm.text_fairness import TextFairnessAnalyzer, TextFairnessResult

GROUPS = {
    "christian": [f"a comment about christian people number {i}" for i in range(30)],
    "muslim": [f"a comment about muslim people number {i}" for i in range(30)],
    "jewish": [f"a comment about jewish people number {i}" for i in range(30)],
}


def _biased_score(texts):
    # Toxicity classifier that unfairly inflates 'muslim'-mentioning text.
    rng = np.random.default_rng(0)
    out = []
    for t in texts:
        base = 0.20 + rng.normal(0, 0.03)
        if "muslim" in t:
            base += 0.30
        out.append(min(max(base, 0.0), 1.0))
    return out


def _fair_score(texts):
    rng = np.random.default_rng(1)
    return [min(max(0.20 + rng.normal(0, 0.03), 0.0), 1.0) for _ in texts]


def test_requires_callable():
    with pytest.raises(TypeError):
        TextFairnessAnalyzer(score_fn=123)


def test_needs_two_groups():
    a = TextFairnessAnalyzer(_fair_score)
    with pytest.raises(ValueError):
        a.analyze({"only": ["one group"]})


def test_detects_identity_term_bias():
    a = TextFairnessAnalyzer(_biased_score)
    res = a.analyze(GROUPS, higher_is_worse=True)
    assert isinstance(res, TextFairnessResult)
    assert res.worst_group == "muslim"
    assert res.max_gap > 0.10
    assert res.p_value is not None and res.p_value < 0.05
    assert res.severity in ("high", "critical")


def test_fair_classifier_not_flagged():
    a = TextFairnessAnalyzer(_fair_score)
    res = a.analyze(GROUPS, higher_is_worse=True)
    assert res.severity in ("info", "low")


def test_score_count_mismatch_raises():
    a = TextFairnessAnalyzer(lambda texts: [0.1])  # wrong length
    with pytest.raises(ValueError):
        a.analyze(GROUPS)


def test_to_dict_json_friendly():
    import json

    a = TextFairnessAnalyzer(_biased_score)
    d = a.analyze(GROUPS).to_dict()
    expected = {
        "overall_mean",
        "worst_group",
        "max_gap",
        "p_value",
        "severity",
        "interpretation",
        "notes",
        "groups",
        # Added 2026-09-27. An identity term supplied with NO TEXT was silently
        # deleted from the design: three terms in, a groups list of two out, severity
        # 'info' and "No material identity-term bias detected". These two fields are
        # what makes that visible, and they have to survive to_dict or the disclosure
        # exists only on the object. tests/test_serialiser_honesty.py caught them
        # missing from the serialiser, which is the same defect one layer down.
        "groups_not_scored",
        "n_groups_supplied",
    }
    assert set(d) == expected
    assert all(set(g) == {"group", "n", "mean_score", "gap_vs_overall"} for g in d["groups"])
    json.dumps(d)
