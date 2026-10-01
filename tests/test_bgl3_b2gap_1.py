"""The last seven open defects on the release gate, and why they counted.

All seven are PREVIEW scope, which the beta release does not promise. They were still
in scope because the gate's B2 criterion counts every open defect that returns a
measurement, without exempting preview, and fails closed on any unit whose return type
nobody classified: "not having worked out what this returns is not evidence that it is
harmless." A preview feature is allowed to be unproven. It is not allowed to be
fabricating, because a reader who tries it gets a false clean bill whether or not the
surface is labelled preview.

Five are MCP tools, and that matters to how they were judged: the caller is a model
acting on what it is given, not a person who might find a suspicious zero odd.

WHAT WAS ACTUALLY WRONG. Two of the seven, and NOT the ones the ledger pointed at.
Four MCP analysis tools turned out to be exemplary already, each carrying a sentence
naming what it could not measure, and what was missing for them was only the pin. Their
recorded status said as much: "evidence insufficient", not "fabricating".

  1. jsonify mapped NaN AND infinity to the same null. In this library those mean
     opposite things. explainer.py carries the carve-out verbatim: risk_ratio and
     odds_ratio "return inf only for a table that WAS measured and in which one arm
     received no positive outcomes at all", reported as "Total exclusion: ... This is
     the strongest disparate impact reading available, not a missing measurement", at
     severity critical. So the most serious finding the ratio metrics produce reached
     an agent as null, reading exactly like a field nobody computed.
  2. attach_lightweight_regulatory swallowed every error with a bare
     `except Exception: pass`, its docstring saying so. On a compliance surface that
     left no difference between "the caller did not ask for admissibility" and "the
     attachment raised", which are opposite readings.

  And a smaller one: empty_exports(reason="") returned
  {"applicable": false, "reason": ""} for every framework, which reads as a considered
  finding that no regulation applies.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.mcp import tools as T
from vfairness.operations.pulse.regulatory import (
    attach_lightweight_regulatory,
    empty_exports,
)

# ------------------------------------------------------------------- jsonify


def test_a_total_exclusion_does_not_arrive_as_a_missing_value():
    """Infinity is a measurement here, and the most serious one available."""
    assert T.jsonify(float("inf")) == "Infinity"
    assert T.jsonify(float("-inf")) == "-Infinity"
    assert T.jsonify(np.float64("inf")) == "Infinity"


def test_an_unmeasured_value_arrives_as_null():
    """NaN means no measurement was made, and JSON has no NaN, so null is honest."""
    assert T.jsonify(float("nan")) is None
    assert T.jsonify(np.float64("nan")) is None


def test_the_two_are_distinguishable_in_one_document():
    """The whole point. Before the fix both were null and no consumer could tell."""
    out = T.jsonify(
        {"measured": 0.0, "not_measured": float("nan"), "total_exclusion": float("inf")}
    )
    assert out == {"measured": 0.0, "not_measured": None, "total_exclusion": "Infinity"}
    # And it is still parseable, which is the constraint that made null tempting.
    assert json.loads(json.dumps(out))["total_exclusion"] == "Infinity"


def test_a_measured_value_is_untouched():
    """Over-correction control. A measured 0.0 is perfect parity, a real reading, and
    must not become a string or a null."""
    assert T.jsonify(0.0) == 0.0
    assert T.jsonify(0.42) == 0.42
    assert T.jsonify(np.float64(0.42)) == 0.42
    assert T.jsonify({"a": 0.0, "b": 1.0}) == {"a": 0.0, "b": 1.0}
    nested = T.jsonify(np.array([0.1, np.nan, np.inf]))
    assert nested[0] == pytest.approx(0.1)
    assert nested[1] is None
    assert nested[2] == "Infinity"


# ------------------------------------------------- the four analysis tools

_N = 60


@pytest.fixture(scope="module")
def one_group():
    return pd.DataFrame(
        {
            "race": ["a"] * _N,
            "f1": np.linspace(0, 1, _N),
            "f2": np.arange(_N) * 1.0,
            "y": [0, 1] * (_N // 2),
            "pred": [0, 1] * (_N // 2),
            "score": np.linspace(0, 1, _N),
        }
    )


@pytest.fixture(scope="module")
def two_groups():
    n = 200
    rng = np.random.RandomState(0)
    race = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    f1 = np.where(race == "a", rng.normal(0.7, 0.1, n), rng.normal(0.3, 0.1, n))
    y = (f1 > 0.5).astype(int)
    return pd.DataFrame(
        {"race": race, "f1": f1, "f2": rng.normal(0, 1, n), "y": y, "pred": y, "score": f1}
    )


_DISCLOSES = (
    "assessed nothing",
    "not a clean bill",
    "incomplete",
    "could not",
    "not measured",
    "could not be computed",
    "not evidence",
)


def _says_it_could_not(summary: str) -> bool:
    low = summary.lower()
    return any(p in low for p in _DISCLOSES)


@pytest.mark.parametrize(
    "tool,args",
    [
        ("triage_dataset", lambda df: (df, ["race"], "y")),
        ("detect_proxies", lambda df: (df, ["race"])),
        ("suggest_mitigation", lambda df: (df, ["race"], "y", "score")),
    ],
)
def test_an_mcp_tool_says_on_its_own_summary_that_it_measured_nothing(tool, args, one_group):
    """One protected group means no comparison exists. The summary is the field an
    agent reads first, so the disclosure has to be there and not only in a warning
    the agent never sees."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = getattr(T, tool)(*args(one_group))
    assert isinstance(out, dict) and "summary" in out
    assert _says_it_could_not(out["summary"]), (
        f"{tool} produced a summary a model would read as a clean result: {out['summary']!r}"
    )


def test_explain_decision_names_the_features_it_could_not_assess(one_group):
    """A feature absent from a driver ranking is indistinguishable from a feature
    that drives nothing, and only one of those is a finding."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = T.explain_decision(one_group.assign(pred=1), "pred", ["f1", "f2"])
    assert _says_it_could_not(out["summary"]), out["summary"]
    assert out.get("features_not_measured"), (
        "the unassessable features must be listed, not merely missing from the ranking"
    )
    assert {f["feature"] for f in out["features_not_measured"]} == {"f1", "f2"}


def test_an_mcp_tool_on_measurable_data_does_not_cry_wolf(two_groups):
    """Over-correction control. Two groups of 100 with a real association: the tool
    must NOT report that it assessed nothing, or the disclosure above means nothing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = T.triage_dataset(two_groups, ["race"], "y")
    s = out["summary"].lower()
    assert "assessed nothing" not in s, out["summary"]
    assert "not a clean bill" not in s, out["summary"]


# ------------------------------------------------------------- regulatory


def test_a_failed_regulatory_attachment_leaves_a_trace():
    """It is additive so it must not raise, and it is compliance so it must not be
    silent. Before the fix, `except Exception: pass`."""

    class _Raises(dict):
        def setdefault(self, *a, **k):
            raise RuntimeError("synthetic failure")

    result = {"data": {"regulatoryExports": _Raises()}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = attach_lightweight_regulatory(result, {}, domain="lending", columns=["x"])
    note = dict(out["data"]["regulatoryExports"]).get("attachmentIncomplete")
    assert note and "RuntimeError" in note, (
        "a swallowed attachment failure must say which error it swallowed"
    )
    assert "NOT evaluated" in note
    assert any("INCOMPLETE" in str(w.message) for w in caught)


def test_a_successful_attachment_is_silent_and_complete():
    """Over-correction control."""
    result = {"data": {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = attach_lightweight_regulatory(result, {}, domain="lending", columns=["x"])
    exports = out["data"]["regulatoryExports"]
    assert "legalContextAsOf" in exports
    assert "attachmentIncomplete" not in exports
    assert not caught, [str(w.message) for w in caught]


def test_a_collapsed_export_block_without_a_reason_says_so():
    """applicable=False here means "could not evaluate", and only `reason` separates
    that from "no regulation applies"."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = empty_exports("")
    assert out["ll144"]["reason"].strip(), "a blank reason discloses nothing"
    assert "no reason was recorded" in out["ll144"]["reason"]
    assert out["art10"]["reason"].strip()
    assert caught, "the caller must be told it passed no reason"


def test_a_real_reason_is_passed_through_untouched():
    """Over-correction control."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = empty_exports("the calibration stage raised")
    assert out["ll144"]["reason"] == "the calibration stage raised"
    assert out["art10"]["reason"] == "the calibration stage raised"
    assert not caught


def test_every_framework_on_a_collapsed_block_is_marked_inapplicable_not_passing():
    """A collapse must not read as a pass on any framework."""
    out = empty_exports("the stage collapsed")
    assert out["ll144"]["applicable"] is False
    assert out["art10"]["applicable"] is False
    assert out["ll144"]["impactRatiosComputed"] is None, (
        "0 computed ratios would read as a measured absence of disparity"
    )
    assert out["ll144"]["impactRatios"] == [], (
        "a collapsed block must carry no ratios at all rather than a computed empty set"
    )
