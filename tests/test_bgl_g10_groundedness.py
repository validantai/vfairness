"""G10 grading pins: vfairness.validity.groundedness (judge ladder, fail-closed).

Every assertion here was reproduced by execution first. The two defects pinned:

D1 the secondary rung fields VG-002 (faithfulness) and VG-003 (context precision)
   reached the result through a bare ``float()``, so NaN, 47.0 and ``True`` all
   became published values while the module comment claimed a single central
   finite + [0, 1] guard for every rung. ``to_json()`` then emitted the bare token
   ``NaN``, which is not valid JSON, for an envelope the validity task handler
   publishes record by record.
D2 total loss of the SOURCES was not refused while total loss of the ANSWER was.
   ``score("The capital is Paris.", [])`` returned a PERFECT groundedness of 1.0
   against zero retrieved passages, so a retrieval outage aggregated to a clean
   VG-001 mean of 1.0 over a whole batch.

Controls accompany both: the healthy case must still carry its real numbers, or a
guard that refuses everything would pass every refusal test here.
"""

from __future__ import annotations

import json
import math
import warnings

import pytest

from vfairness.validity.aggregate import aggregate_validity
from vfairness.validity.groundedness import (
    GoldUnsupportedWarning,
    GroundednessJudge,
    GroundednessResult,
    GroundednessScorer,
    ValidityScorerUnavailableWarning,
    groundedness_scorer_status,
)

CONTEXT = ["Paris is the capital of France."]
ANSWER = "The capital of France is Paris."


class _Rung:
    """A rung that returns exactly the verdict it was constructed with."""

    def __init__(self, **verdict):
        self._verdict = verdict
        self.seen_contexts = None

    def judge_groundedness(self, *, answer, contexts, question, language):
        self.seen_contexts = list(contexts)
        return dict(self._verdict)


class _Raiser:
    def judge_groundedness(self, *, answer, contexts, question, language):
        raise RuntimeError("rung exploded")


def _score(rung, answer=ANSWER, contexts=CONTEXT, **kw):
    scorer = GroundednessScorer(sidecar=rung)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return scorer.score(answer, contexts, **kw)


# ---------------------------------------------------------------- D1: secondary fields


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        47.0,
        1.5,
        -0.25,
        True,
        False,
        "abc",
        "47",
        object(),
    ],
)
def test_unusable_faithfulness_is_not_measured_and_says_so(bad):
    """D1. An unusable VG-002 value must be None WITH a visible reason, never a number.

    ``False`` is in here deliberately: a boolean that happens to be falsy is still
    proof the rung did not follow the rubric, and ``float(True) == 1.0`` would
    otherwise read as a perfect score. ``"47"`` is in here because a value that
    PARSES is still refused when it lands outside the rubric's range.
    """
    result = _score(_Rung(groundedness=0.9, faithfulness=bad))
    assert result.available is True, "groundedness itself was fine and must still be published"
    assert result.value == pytest.approx(0.9)
    assert result.faithfulness is None, f"{bad!r} was published as a VG-002 measurement"
    assert "faithfulness (VG-002) NOT MEASURED" in result.note, (
        "the refusal must reach a reader; a refusal nobody can read is the same "
        "defect as a fabrication one layer up"
    )


@pytest.mark.parametrize("bad", [float("nan"), 47.0, True, "x"])
def test_unusable_context_precision_is_not_measured_and_says_so(bad):
    """D1, the VG-003 half. Same guard, same visibility requirement."""
    result = _score(_Rung(groundedness=0.9, context_precision=bad))
    assert result.context_precision is None
    assert "context_precision (VG-003) NOT MEASURED" in result.note


def test_secondary_fields_control_real_values_survive():
    """CONTROL for D1. A guard that refuses everything must fail this."""
    result = _score(_Rung(groundedness=0.9, faithfulness=0.82, context_precision=0.71, note="ok"))
    assert result.faithfulness == pytest.approx(0.82)
    assert result.context_precision == pytest.approx(0.71)
    assert result.note == "ok", "no refusal text belongs on a healthy verdict"
    # The boundary values of the rubric are legal scores, not out-of-range ones.
    edges = _score(_Rung(groundedness=0.5, faithfulness=0.0, context_precision=1.0))
    assert edges.faithfulness == pytest.approx(0.0)
    assert edges.context_precision == pytest.approx(1.0)
    # An in-range NUMERIC STRING is coerced, matching what the primary groundedness
    # guard already does with one. Refusing it here would discard real evidence, and
    # the range check still runs afterwards (see the "47" case above).
    stringy = _score(_Rung(groundedness="0.9", faithfulness="0.8"))
    assert stringy.value == pytest.approx(0.9)
    assert stringy.faithfulness == pytest.approx(0.8)


def test_absent_secondary_field_is_not_reported_as_a_refusal():
    """A rung that simply does not compute VG-002 has refused nothing."""
    result = _score(_Rung(groundedness=0.9))
    assert result.faithfulness is None and result.context_precision is None
    assert "NOT MEASURED" not in result.note
    assert result.note == ""


def test_result_serialises_to_strict_json_after_a_bad_rung_verdict():
    """D1's consumer-side proof. ``operations/validity/task_handlers`` publishes every
    record through ``to_dict()`` inside one ``json.dumps``; a NaN there is the bare
    token ``NaN``, so a strict parser rejects the whole envelope, not just one field.
    """
    result = _score(_Rung(groundedness=0.9, faithfulness=float("nan")))
    payload = result.to_json()

    def _reject_constant(name):
        raise AssertionError(f"non-JSON constant {name} published in the envelope")

    round_tripped = json.loads(payload, parse_constant=_reject_constant)
    assert round_tripped["faithfulness"] is None
    assert round_tripped["value"] == pytest.approx(0.9)


def test_note_key_present_holding_none_does_not_mint_the_string_none():
    """``.get(key, default)`` does NOT fire when the key is PRESENT holding None."""
    result = _score(_Rung(groundedness=0.9, note=None))
    assert result.note == "", f"minted note content: {result.note!r}"


# ---------------------------------------------------------------- D2: no sources at all


@pytest.mark.parametrize(
    "contexts",
    [[], [""], ["   "], ["\n\t"], None, 42, ["", "  "]],
)
def test_no_usable_context_is_refused_not_scored_perfect(contexts):
    """D2. Groundedness is undefined with no passage to be grounded in.

    The rung here would answer a perfect 1.0, so before the fix each of these
    published available=True, value=1.0, hallucination_rate=0.0, supported=True.
    """
    rung = _Rung(groundedness=1.0)
    result = _score(rung, contexts=contexts)
    assert result.available is False, "a perfect score against zero sources"
    assert result.value is None
    assert result.hallucination_rate is None
    assert result.supported is None
    assert "COULD NOT CHECK" in result.note and "no retrieved context" in result.note
    assert rung.seen_contexts is None, "the guard must sit ABOVE the rung dispatch"


def test_no_usable_context_refusal_applies_to_the_judge_rung_too():
    """PUT THE GUARD ABOVE THE DISPATCH: the precondition is shared by both rungs."""
    judge = _Rung(groundedness=1.0)
    scorer = GroundednessScorer(judge=judge)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scorer.score(ANSWER, [])
    assert result.available is False and result.value is None
    assert judge.seen_contexts is None


def test_no_usable_context_warns_so_a_batch_run_is_not_silent():
    scorer = GroundednessScorer(sidecar=_Rung(groundedness=1.0))
    with pytest.warns(ValidityScorerUnavailableWarning):
        scorer.score(ANSWER, [])


def test_retrieval_outage_batch_cannot_aggregate_to_a_clean_pass():
    """D2's consumer-side proof: 100 context-less records used to mean VG-001 = 1.0."""
    scorer = GroundednessScorer(sidecar=_Rung(groundedness=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch = [scorer.score(ANSWER, []) for _ in range(100)]
    aggregate = aggregate_validity(batch)
    assert aggregate["available"] is False
    assert aggregate["n_measured"] == 0
    assert "VG-001" not in aggregate


def test_context_guard_control_real_passages_still_measure():
    """CONTROL for D2, and a check that blanks are DROPPED rather than counted."""
    rung = _Rung(groundedness=0.25)
    result = _score(rung, contexts=["", "Paris is the capital of France.", "  "])
    assert result.available is True
    assert result.value == pytest.approx(0.25)
    assert result.hallucination_rate == pytest.approx(0.75)
    assert result.supported is False, "0.25 is below the 0.5 support line"
    assert rung.seen_contexts == ["Paris is the capital of France."]


def test_literal_none_and_nan_in_a_passage_are_real_text_not_absence():
    """DO NOT OVER-CORRECT: a passage may legitimately contain these words."""
    for text in ("None", "nan", "null", "None of the above", "<NA>"):
        rung = _Rung(groundedness=0.6)
        result = _score(rung, contexts=[text])
        assert result.available is True, f"{text!r} was treated as an absent passage"
        assert rung.seen_contexts == [text]


# ---------------------------------------------------------------- the ladder's floor


def test_unwired_scorer_refuses_rather_than_returning_zero_or_one():
    scorer = GroundednessScorer()
    assert scorer.available is False
    with pytest.warns(ValidityScorerUnavailableWarning):
        result = scorer.score(ANSWER, CONTEXT)
    assert result.available is False
    assert result.value is None and result.hallucination_rate is None
    assert result.scorer_tier == "none"
    assert "not a measurement" in result.note.lower()


@pytest.mark.parametrize(
    "verdict,expected_note",
    [
        ({}, "returned no score, refusing"),
        ({"groundedness": None}, "returned no score, refusing"),
        ({"groundedness": True}, "non-numeric (boolean) score, refusing"),
        ({"groundedness": float("nan")}, "non-finite score, refusing"),
        ({"groundedness": float("inf")}, "non-finite score, refusing"),
        ({"groundedness": "abc"}, "non-finite score, refusing"),
        ({"groundedness": 1.5}, "out-of-range score, refusing"),
        ({"groundedness": -0.5}, "out-of-range score, refusing"),
    ],
)
def test_ladder_refuses_every_shape_of_unusable_primary_verdict(verdict, expected_note):
    """The negative case for the ladder: saying "grounded" for good input proves
    nothing unless ungradeable input is REFUSED."""
    result = _score(_Rung(**verdict))
    assert result.available is False, f"{verdict!r} was published as a measurement"
    assert result.value is None
    assert expected_note in result.note


def test_ladder_refuses_when_the_rung_raises():
    scorer = GroundednessScorer(sidecar=_Raiser())
    with pytest.warns(ValidityScorerUnavailableWarning):
        result = scorer.score(ANSWER, CONTEXT)
    assert result.available is False and result.value is None
    assert "rung exploded" in result.note, "the reason must survive into the note"


def test_ladder_control_an_ungrounded_answer_is_reported_as_ungrounded():
    """CONTROL: the ladder must not refuse a real LOW score, only unusable ones."""
    result = _score(_Rung(groundedness=0.05, unsupported_spans=["invented citation"]))
    assert result.available is True
    assert result.value == pytest.approx(0.05)
    assert result.hallucination_rate == pytest.approx(0.95)
    assert result.supported is False
    assert result.unsupported_spans == ["invented citation"]


def test_sidecar_outranks_judge_and_the_tier_is_reported_truthfully():
    sidecar, judge = _Rung(groundedness=0.9), _Rung(groundedness=0.1)
    scorer = GroundednessScorer(judge=judge, sidecar=sidecar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = scorer.score(ANSWER, CONTEXT)
    assert result.scorer_tier == "owned_sidecar" and result.value == pytest.approx(0.9)
    assert judge.seen_contexts is None


def test_empty_answer_is_refused():
    for answer in ("", "   ", "\n"):
        result = _score(_Rung(groundedness=1.0), answer=answer)
        assert result.available is False and result.value is None
        assert result.note.startswith("empty answer")


# ---------------------------------------------------------------- gold refusal (R6-2)


def test_gold_is_refused_on_every_return_path_and_the_note_carries_it():
    for scorer, contexts in (
        (GroundednessScorer(), CONTEXT),
        (GroundednessScorer(sidecar=_Rung(groundedness=0.9)), CONTEXT),
        (GroundednessScorer(sidecar=_Rung(groundedness=1.0)), []),
        (GroundednessScorer(sidecar=_Raiser()), CONTEXT),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = scorer.score(ANSWER, contexts, gold="Paris.")
        assert "gold= REFUSED" in result.note


def test_gold_warns_once_per_scorer_and_never_changes_the_score():
    scorer = GroundednessScorer(sidecar=_Rung(groundedness=0.9))
    with pytest.warns(GoldUnsupportedWarning):
        first = scorer.score(ANSWER, CONTEXT, gold="Paris.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GoldUnsupportedWarning)
        second = scorer.score(ANSWER, CONTEXT, gold="COMPLETELY WRONG GOLD")
        plain = scorer.score(ANSWER, CONTEXT)
    assert first.value == second.value == plain.value == pytest.approx(0.9)


# ---------------------------------------------------------------- records and status


def test_result_default_is_a_refusal_not_a_clean_zero():
    """NOT A MEASUREMENT units still must not quietly mint a value."""
    blank = GroundednessResult()
    assert blank.available is False
    assert blank.value is None and blank.hallucination_rate is None
    assert blank.faithfulness is None and blank.context_precision is None
    assert blank.supported is None
    assert blank.scorer_tier == "none" and blank.note == ""
    assert blank.unsupported_spans == []
    payload = blank.to_dict()
    assert set(payload) >= {"value", "available", "note", "hallucination_rate"}
    assert payload["value"] is None and payload["available"] is False
    # Two instances must not share the mutable default.
    GroundednessResult().unsupported_spans.append("leak")
    assert GroundednessResult().unsupported_spans == []


def test_scorer_status_reports_the_three_states_truthfully():
    assert groundedness_scorer_status(GroundednessScorer()) == {
        "tier": "none",
        "quality": "unavailable",
        "available": False,
    }
    judge_only = groundedness_scorer_status(GroundednessScorer(judge=_Rung(groundedness=0.5)))
    assert judge_only == {"tier": "llm_judge", "quality": "interim", "available": True}
    both = groundedness_scorer_status(
        GroundednessScorer(judge=_Rung(groundedness=0.5), sidecar=_Rung(groundedness=0.9))
    )
    assert both == {"tier": "owned_sidecar", "quality": "production", "available": True}


def test_judge_protocol_is_runtime_checkable_and_discriminates():
    assert isinstance(_Rung(groundedness=0.5), GroundednessJudge)

    class NotAJudge:
        pass

    assert not isinstance(NotAJudge(), GroundednessJudge)


def test_warning_classes_are_runtime_warnings_and_keep_their_reason():
    for cls in (GoldUnsupportedWarning, ValidityScorerUnavailableWarning):
        assert issubclass(cls, RuntimeWarning)
        assert str(cls("the reason")) == "the reason"
    assert math.isfinite(1.0)  # keeps the math import honest for the NaN pins above
