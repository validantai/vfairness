"""A probe grade may not claim "measured on healthy input" unless it was.

WHY. scripts/grade_from_probe.py awards SEMI-PROVEN from probe evidence with no
agent involved, and its first clause exists to refuse evidence that cannot tell an
honest refusal from a call the probe got wrong. Its own docstring calls that clause
"the whole reason this file is short".

The clause could not fire. It rejected a healthy outcome of None, NOT_REACHED,
raised or hung, and the probe's describer maps a None RETURN to the outcome
"refused" and a NaN to "refused". So a unit that returned None on the GOOD world
was recorded as refusing it, passed the clause, and was published SEMI-PROVEN with
the reason string "measured on healthy input and refused on all 8 degenerate
worlds".

Measured 2026-09-27, found by an auditor attacking two of those rows: ALL 39 grades
the file had produced carried healthy outcome "refused", 37 with detail None and 2
with detail nan. Not one was measured on healthy input. The 316-to-29 filtering the
docstring describes was done entirely by the "raised" case, and "refused" went
through. Correcting the clause qualifies zero of the 39, and all 39 are withdrawn.

A refusal on the healthy world is not evidence of honesty, it is the absence of
evidence either way: nothing distinguishes "refuses what it cannot measure" from
"the probe's binder called it with nonsense" from "a void function returns None and
always did". That is a third state and it is recorded as NOT_EXAMINED.

Asked of classify() with fixtures rather than of today's evidence, because today's
evidence qualifies nothing, so every live assertion would pass over an empty set.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBE_GRADES = ROOT / "docs" / "surface-grading-from-probe.json"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_grade_from_probe_under_test", ROOT / "scripts" / "grade_from_probe.py"
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def gfp():
    return _mod()


def _worlds(mod, healthy_outcome, degenerate="refused"):
    runs = {"healthy": {"outcome": healthy_outcome, "detail": "None"}}
    for w in mod.DEGENERATE:
        runs[w] = {"outcome": degenerate, "detail": "None"}
    return runs


@pytest.mark.parametrize("outcome", ["refused", "raised", "hung", "NOT_REACHED", None])
def test_a_healthy_world_that_produced_no_value_is_not_examined(gfp, outcome):
    """The list of outcomes that are NOT a measurement. "refused" was missing."""
    state, reason = gfp.classify(_worlds(gfp, outcome))
    assert state == "NOT_EXAMINED", f"healthy={outcome!r} was graded {state}: {reason}"
    assert "did not produce a value" in reason


def test_the_over_correction_control_a_real_healthy_measurement_still_qualifies(gfp):
    """Without this, rejecting every healthy outcome would pass the test above.

    A clause that refuses everything is not a stricter clause, it is a broken one:
    it would silently withdraw every future probe grade and the file would read as
    "the probe found nothing" forever.
    """
    state, reason = gfp.classify(_worlds(gfp, "measured"))
    assert state == "SEMI-PROVEN", f"a measured healthy run was rejected: {reason}"
    assert "measured on healthy input" in reason


def test_a_neutral_healthy_value_still_qualifies(gfp):
    """A neutral value IS a value. 0.0 on healthy data is a measurement the probe
    took, however uninteresting, and the clause is about whether anything came
    back, not about whether it was interesting."""
    state, _reason = gfp.classify(_worlds(gfp, "neutral"))
    assert state == "SEMI-PROVEN"


def test_a_silent_neutral_on_a_degenerate_world_is_still_not_a_grade(gfp):
    """The other clauses must keep working after clause 1 was widened."""
    runs = _worlds(gfp, "measured")
    runs["single_group"] = {"outcome": "neutral", "detail": "0.0", "n_warnings": 0}
    state, reason = gfp.classify(runs)
    assert state == "NEEDS_JUDGEMENT", reason
    assert "single_group" in reason


def test_the_reason_string_and_the_clause_cannot_disagree(gfp):
    """The defect was not the clause alone: it was a reason string asserting
    something the clause had not established. Any state that claims a measured
    healthy run must have come through a healthy outcome that produced a value."""
    for outcome in ("refused", "raised", "hung", "NOT_REACHED", None):
        state, reason = gfp.classify(_worlds(gfp, outcome))
        assert "measured on healthy input" not in reason, (
            f"healthy={outcome!r} produced no value and the reason still claims a "
            f"measurement: {reason}"
        )
        assert state != "SEMI-PROVEN"


def test_the_withdrawal_is_recorded_rather_than_a_silent_drop():
    """39 grades vanishing with nothing saying why reads as a lost file."""
    d = json.loads(PROBE_GRADES.read_text(encoding="utf-8"))
    items = d.get("items") or {}
    withdrawn = d.get("_withdrawn") or {}
    if items:
        pytest.skip("the probe qualifies grades again; nothing to reconcile here")
    assert withdrawn, (
        "the probe grading file holds no grades and no withdrawal record, so the "
        "39 it used to publish cannot be reconciled against any surface quoting them"
    )
    for name, rec in withdrawn.items():
        assert rec.get("was"), f"{name} does not say what it was graded"
        assert rec.get("withdrawn_because"), f"{name} does not say why it was withdrawn"
