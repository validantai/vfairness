"""A record with no answer must not be scored against a manufactured word.

`str(rec.get("answer", ""))` does not fire for a PRESENT key holding None, which
is what a JSON payload with no answer looks like, so `{"answer": null}` became
the five-character string "None". That is truthy and non-blank, so the scorer's
own empty-answer guard passed it through and a hallucination rate was scored
against a word the model never said.

`str(x)` on an absent value MINTS CONTENT. It is the same door that put 'None'
into a demographic group holding a fifth of a population, and that produced a
redlining finding against a place that is not a place.

Found by the agent grading a neighbouring package, reported across its boundary
rather than reached into, and closed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.validity.task_handlers import _answer_of


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), np.float64("nan"), pd.NA, pd.NaT, np.ma.masked],
    ids=["None", "float nan", "np nan", "pd.NA", "pd.NaT", "masked"],
)
def test_no_absent_value_becomes_a_word(value):
    assert _answer_of({"answer": value}) == "", repr(value)


def test_an_absent_key_is_the_same_as_an_absent_value():
    assert _answer_of({}) == ""


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Paris", "Paris"),
        ("None of the above", "None of the above"),
        ("nan", "nan"),
        ("0", "0"),
        (0, "0"),
        (False, "False"),
    ],
    ids=["ordinary", "contains None", "the literal string nan", "zero str", "zero int", "bool"],
)
def test_control_a_real_answer_is_returned_untouched(value, expected):
    """OVER-CORRECTION CONTROL, and the third case is the deliberate one.

    A caller who literally sent the string "nan" sent a STRING, and manufacturing
    absence out of their data is the mirror of the defect above. Strings are
    returned before the absence-token set is consulted, on purpose. The sibling
    decision in this campaign went the same way for the literal 'None' as a
    demographic group: kept and disclosed, because it is a real category for some
    attributes.
    """
    assert _answer_of({"answer": value}) == expected


def test_a_manufactured_word_would_have_reached_the_scorer():
    """Non-vacuity: the old expression really did produce a five-character answer,
    so the pins above are not asserting something that was never possible."""
    rec = {"answer": None}
    assert str(rec.get("answer", "")) == "None"
    assert len(str(rec.get("answer", ""))) == 4
    assert _answer_of(rec) == ""


# ─────────────────────────────────────────────────────────────────────────────
# THE CALL SITE, not just the helper
# ─────────────────────────────────────────────────────────────────────────────
#
# Every test above calls _answer_of DIRECTLY, so reverting the CALL SITE back to
# `str(rec.get("answer", ""))` left all fourteen of them GREEN. That is asserting
# the ingredients rather than the behaviour, which is the error this campaign
# keeps punishing, committed in the pin for a fix against it. Measured 2026-09-30
# by sabotaging the call site and watching nothing happen.
#
# The helper also has redundant internal coverage (`is None` and `pd.isna` both
# answer for None), so single-line sabotages of the helper do not discriminate
# either. This test observes what the HANDLER passes to the scorer, which is the
# only thing a reader of the result depends on.


def test_the_handler_hands_the_scorer_no_manufactured_answer(monkeypatch):
    """Drives handle_validity_run and records what the scorer actually received."""
    from vfairness.operations.validity import task_handlers as th

    seen: list[str] = []

    class _Recorder:
        available = True
        name = "recorder"

        def score(self, answer, contexts, question=None, gold=None, language="en"):
            seen.append(answer)
            from vfairness.validity.groundedness import GroundednessResult

            return GroundednessResult(available=False, note="recorded by the pin")

    monkeypatch.setattr(th, "_build_scorer", lambda payload: _Recorder())

    th.handle_validity_run(
        {
            "records": [
                {"answer": None, "contexts": ["c"]},
                {"answer": "Paris", "contexts": ["c"]},
            ]
        }
    )

    assert seen == ["", "Paris"], (
        "the handler manufactured an answer for the record that carried none: "
        f"{seen!r}. 'None' here means str() minted a four-character word the "
        "model never said, and the scorer's empty-answer guard passes it."
    )
