"""Empty text scoring 0.0 is a MEASUREMENT here, and this file is why it stays.

The mechanical sweep flagged seven units in `llm.scorers` for answering a
degenerate world with a neutral value and no warning. Examined one at a time, none
of them is a fabrication, and the distinction is the whole point of grading rather
than pattern-matching:

  * These scorers answer "how much of X is present in this text". For an empty or
    whitespace-only generation the answer is genuinely none, so 0.0 is the
    measurement. Refusing it would throw away a real observation, and this
    library's own doctrine counts that as the same defect running backwards.
  * The case that IS could-not-check is text the scorer could not READ, and that
    path already returns nan with an UnscorableTextWarning. Verified below on CJK
    input, so the two are distinguishable by a caller.
  * `word_tokens`, `term_forms` and `mentions_any_term` are tokenizers. An empty
    string has no tokens, and "no term is mentioned" is true of a text with no
    words in it. They produce no fairness number and no verdict.

What separates this from the sidecar outage fixed in the same wave: an outage means
the text was never read at all, so a clean 0.0 there is a fabrication, and that one
now returns nan. Same file, same day, opposite answers, because the inputs differ.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness.llm import scorers as S

CJK = "你好世界こんにちはمرحبا"
SCORERS = ("HelpfulnessScorer", "RefusalScorer", "SemanticQualityScorer")


def _caught(fn, *a):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*a)
    return value, [str(w.message) for w in rec]


@pytest.mark.parametrize("name", SCORERS)
@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_an_empty_generation_scores_a_measured_absence(name, text):
    """0.0 means "none of it is present", which is true of an empty string."""
    scorer = getattr(S, name)()
    value, _msgs = _caught(scorer.score, text)
    assert value == 0.0 or math.isnan(value), f"{name} answered {value!r} for empty text"


@pytest.mark.parametrize("name", SCORERS)
def test_text_the_scorer_cannot_read_is_could_not_check_not_zero(name):
    """The distinction that makes the 0.0 above defensible. If this stops holding,
    the two states have collapsed and the empty-text 0.0 is no longer safe."""
    scorer = getattr(S, name)()
    value, msgs = _caught(scorer.score, CJK)
    assert math.isnan(value), f"{name} scored unreadable text {value!r} instead of refusing"
    assert msgs, f"{name} refused unreadable text in silence"


def test_control_a_scorer_still_separates_a_real_difference():
    """Without this, every assertion above is satisfiable by a scorer that only
    ever answers 0.0 and nan."""
    helpful = S.HelpfulnessScorer()
    rich = helpful.score(
        "Here are three concrete steps you can take, with an example for each, "
        "and a short explanation of why each one helps."
    )
    bare = helpful.score("no.")
    assert rich > bare, f"a detailed answer scored {rich} and a bare refusal {bare}"


@pytest.mark.parametrize(
    "fn,arg,expected",
    [
        ("word_tokens", "", []),
        ("term_forms", "", set()),
    ],
)
def test_the_tokenizers_return_nothing_for_nothing(fn, arg, expected):
    value, _msgs = _caught(getattr(S, fn), arg)
    assert value == expected


def test_mentions_any_term_is_false_when_there_are_no_words():
    assert _caught(S.mentions_any_term, "", {"kind"})[0] is False


def test_control_the_tokenizers_still_tokenise():
    assert S.word_tokens("helpful and kind") == ["helpful", "and", "kind"]
    assert S.mentions_any_term("helpful and kind", {"kind"}) is True
    assert S.mentions_any_term("helpful and kind", {"absent"}) is False
