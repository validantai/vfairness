"""SCORER-05 (readiness wave 5, 2026-09-10). The default sentiment scorer
returned a measured neutral for text it had never read.

``KeywordSentimentScorer`` carries a 30-word English lexicon and is
``DEFAULT_SENTIMENT_SCORER`` on a bare install, so it is what a user gets
without configuring anything. A text containing none of those 30 words scored
0.0, which on this scale is MEASURED neutral sentiment, and ``OutputAnalyzer``
then averaged it, tested it and reported it. Measured 2026-09-10 on six glowing
against six scathing sentences, none of which shares a lexicon word:

    group_a_value=0.0  group_b_value=0.0  delta=0.0  p_value=1.0
    is_significant=False  assessed=True  not_assessed_reason=None
    effect_size=0.0 "Cohen's h: negligible"

with no warning at all on the second analysis in the process, because the
placeholder warning latches on first use and the default scorer is a module
singleton.

The fix returns NaN for a text the lexicon cannot read, which the analyzer's
LF-06 path already knows how to refuse, and warns on EVERY call rather than
once. It is deliberately NOT the wave-4 sidecar decision reversed: that one
keeps 0.0 for an OUTAGE, which is a property of the whole run that
``scorer_status()`` and ``.available`` expose, while a lexicon miss is per text
and invisible. ``KeywordSentimentScorer``'s docstring records the reasoning.

Both pins come with an over-correction control asserting MEASURED values,
because a scorer that answers "could not check" to everything is the same
failure wearing the other mask. In particular a text whose positive and
negative hits CANCEL has been read, and its 0.0 must survive.
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.llm.output_analysis import OutputAnalyzer
from vfairness.llm.scorers import (
    DEFAULT_SENTIMENT_SCORER,
    KeywordSentimentScorer,
    PlaceholderScorerWarning,
    UnscorableTextWarning,
)

# ---------------------------------------------------------------------------
# Fixtures. Two corpora whose sentiment is not in doubt to a human reader and
# which share no word with the 30-word lexicon, plus the readable corpus the
# controls use.
# ---------------------------------------------------------------------------

GLOWING = [
    "I absolutely adore working with her; she is a delight every single day.",
    "Her presentation was superb and the whole room applauded her insight.",
    "He is a joy to mentor and consistently exceeds every expectation set.",
    "Everyone admires their dedication; the team thrives under their guidance.",
    "A remarkable colleague whose work I would happily endorse anywhere.",
    "She lifted the whole project and I am grateful she joined us.",
]

SCATHING = [
    "I loathe working with him; he is an ordeal every single day.",
    "His presentation was abysmal and the whole room winced at the errors.",
    "He is a chore to mentor and consistently falls short of every expectation.",
    "Nobody trusts their dedication; the team flounders under their guidance.",
    "An appalling colleague whose work I would never endorse anywhere.",
    "He sank the whole project and I regret that he joined us.",
]

READABLE_POSITIVE = [f"a wonderful and kind and helpful person number {i}" for i in range(30)]
READABLE_NEGATIVE = [f"a horrible and cruel and useless person number {i}" for i in range(30)]


def _fresh() -> KeywordSentimentScorer:
    """A new instance, so the placeholder warning has not latched yet."""
    return KeywordSentimentScorer()


def _scored(scorer, texts):
    """Score a batch with every warning recorded, not swallowed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = scorer.score_batch(list(texts))
    return values, caught


def _refusals(caught):
    return [w for w in caught if issubclass(w.category, UnscorableTextWarning)]


# ---------------------------------------------------------------------------
# PIN 1. Text the lexicon cannot read is a could-not-check, never a neutral.
# ---------------------------------------------------------------------------


def test_text_the_lexicon_cannot_read_is_refused_not_scored_neutral():
    """The scalar path. NaN plus a warning, never 0.0."""
    scorer = _fresh()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = scorer.score("I absolutely adore working with her.")

    assert math.isnan(value), (
        f"scored {value!r} for a text containing none of the 30 lexicon words; "
        "0.0 on this scale is measured neutral sentiment"
    )
    refusals = _refusals(caught)
    assert refusals, "the refusal was silent"
    assert "not 0.0" in str(refusals[0].message)


def test_a_blatant_sentiment_gap_is_not_reported_as_no_disparity():
    """The path a user actually gets: OutputAnalyzer on the default scorer.

    Every numeric field must be withheld and the reason named. Before the fix
    this returned delta=0.0, p_value=1.0, is_significant=False, assessed=True.
    """
    assert isinstance(DEFAULT_SENTIMENT_SCORER, KeywordSentimentScorer), (
        "this pin assumes the bare-install rung; a backend is installed here"
    )

    analyzer = OutputAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.analyze_sentiment(GLOWING, SCATHING, "glowing", "scathing")

    assert result.assessed is False, (
        f"assessed=True for delta={result.delta!r}, p={result.p_value!r} "
        "on twelve texts the scorer never read"
    )
    assert result.not_assessed_reason == "non_finite_scores"
    assert result.group_a_value is None
    assert result.group_b_value is None
    assert result.delta is None
    assert result.p_value is None
    assert result.is_significant is None
    assert result.effect_size is None
    assert result.effect_size_interpretation == "not_assessed"
    assert _refusals(caught), "the analyzer path scored twelve texts in silence"


def test_the_refusal_is_reported_on_every_call_not_once_per_process():
    """`_warn_once` latches per instance and the default scorer is a module
    singleton, so a latched refusal warning would be silent for the whole life
    of a long-running consumer after its first job. Two jobs, two warnings.

    The placeholder warning itself is deliberately still once per instance (two
    existing suites pin exactly-one, and it describes the configuration rather
    than the data); what may not latch is the per-text refusal.
    """
    scorer = _fresh()

    first, caught_first = _scored(scorer, GLOWING)
    second, caught_second = _scored(scorer, SCATHING)

    assert np.isnan(first).all() and np.isnan(second).all()
    assert len(_refusals(caught_first)) == 1
    assert len(_refusals(caught_second)) == 1, (
        "the second job was silent; a warning that fires once per process is "
        "close to no warning at all for a service"
    )
    assert f"{len(GLOWING)} of {len(GLOWING)}" in str(_refusals(caught_first)[0].message), (
        "the warning must name how many texts were refused"
    )
    # The placeholder warning is the one that legitimately latches.
    placeholders = [
        w
        for w in caught_second
        if issubclass(w.category, PlaceholderScorerWarning)
        and not issubclass(w.category, UnscorableTextWarning)
    ]
    assert placeholders == []


# ---------------------------------------------------------------------------
# OVER-CORRECTION CONTROLS. Everything the lexicon CAN read still measures, and
# measures the same numbers as before the fix.
# ---------------------------------------------------------------------------


def test_text_the_lexicon_can_read_scores_exactly_as_it_did_before():
    """Frozen numbers, including the two ends and a fractional mix."""
    scorer = _fresh()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = [
            scorer.score("She is a wonderful and talented engineer with an outstanding record."),
            scorer.score("Those people are lazy criminals and they always cause trouble."),
            scorer.score("good great terrible"),
            scorer.score("a good day"),
            scorer.score("a bad day"),
        ]

    assert got == pytest.approx([1.0, -1.0, 1.0 / 3.0, 1.0, -1.0])
    assert _refusals(caught) == [], "a readable text was refused"


def test_a_cancelling_text_still_measures_a_real_neutral_zero():
    """The value the refusal must NOT swallow. One positive word against one
    negative word HAS been read, and 0.0 is its measurement.
    """
    scorer = _fresh()
    # Scored in SEPARATE warning contexts. The empty string refuses now, so
    # sharing one context made its warning indistinguishable from a refusal of
    # the readable text, which is the thing this test is actually guarding.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = scorer.score("a good and bad day")
    with warnings.catch_warnings(record=True) as caught_empty:
        warnings.simplefilter("always")
        empty = scorer.score("")

    assert measured == 0.0
    assert not math.isnan(measured)
    # The EMPTY string moved 0.0 -> NaN on 2026-09-17, and this assertion used
    # to say the opposite. The carve-out reasoned that an empty string has no
    # sentiment to miss, but on a SIGNED scale 0.0 is measured NEUTRAL
    # sentiment, the exact midpoint, and it beats every negative score.
    # Measured at the public entry before the change, eight whitespace-only
    # generations against eight denigrating sentences came back delta=1.0,
    # p=0.000138, is_significant=True: a significant advantage for the group
    # that generated nothing at all. The regard sibling had already closed
    # exactly this.
    #
    # The subject of THIS test is untouched: a text carrying one positive and
    # one negative word HAS been read, and its 0.0 is a measurement the refusal
    # must not swallow. That is the assertion above.
    assert math.isnan(empty)
    assert _refusals(caught) == [], "a readable text was refused"
    assert _refusals(caught_empty), "the empty string refused SILENTLY"


def test_a_real_sentiment_difference_is_still_detected():
    """The whole point of the scorer. A measurable gap must still be measured,
    tested and called significant.
    """
    analyzer = OutputAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.analyze_sentiment(
            READABLE_POSITIVE, READABLE_NEGATIVE, "positive", "negative"
        )

    assert result.assessed is True
    assert result.not_assessed_reason is None
    assert result.group_a_value == pytest.approx(1.0)
    assert result.group_b_value == pytest.approx(-1.0)
    assert result.delta == pytest.approx(2.0)
    assert result.p_value is not None and result.p_value < 1e-10
    assert result.is_significant is True
    assert _refusals(caught) == [], "a fully readable corpus was refused"


def test_a_mixed_batch_keeps_every_measured_score_and_refuses_only_the_rest():
    """Positional integrity. The refusal must land on the texts that were not
    read and nowhere else, so a partially readable batch is not blanket-voided
    inside the scorer (the analyzer decides what to do with the NaNs).
    """
    scorer = _fresh()
    texts = [
        "a wonderful day",  # readable, +1
        "I absolutely adore this",  # unreadable
        "a horrible day",  # readable, -1
        "a good and bad day",  # readable, measured 0.0
    ]
    values, caught = _scored(scorer, texts)

    assert values[0] == pytest.approx(1.0)
    assert math.isnan(values[1])
    assert values[2] == pytest.approx(-1.0)
    assert values[3] == 0.0
    assert len(_refusals(caught)) == 1
    assert "1 of 4" in str(_refusals(caught)[0].message)


def test_score_batch_keeps_its_shape_and_dtype_contract():
    """The wave-4 API-compatibility argument, held to: a float ndarray of the
    same length, and an empty input still returns an empty float array.
    """
    scorer = _fresh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = scorer.score_batch(list(GLOWING))
        empty = scorer.score_batch([])

    assert isinstance(values, np.ndarray)
    assert values.shape == (len(GLOWING),)
    assert values.dtype == np.float64
    assert isinstance(empty, np.ndarray)
    assert empty.shape == (0,)
