"""Audit wave 2, 2026-09-29: a MISSING response was scored as an EMPTY one.

``normalize_text`` does ``str(text)``, so ``normalize_text(None)`` is the string
``'none'``: a perfectly readable ASCII word, which made
``lexicon_readable_share(None)`` 1.0 and ``lexicon_can_read(None)`` True. But the
null never even reached that gate. Every scorer in ``llm/scorers.py`` opened with
``if not text or not text.strip(): return 0.0``, and ``None`` is falsy, so it
short-circuited to a MEASURED 0.0 before any readability test ran.

MEASURED BEFORE THE FIX, by executing each public entry:

    RefusalScorer, HelpfulnessScorer, InformationQualityScorer,
    RepresentationScorer, SemanticQualityScorer, StereotypeScorer
        score(None) -> 0.0 with ZERO warnings
    ContextualStereotypeScorer, KeywordToxicityScorer
        score(None) -> 0.0 (only their unrelated sidecar/placeholder warnings)
    RefusalScorer().score_batch([None] * 25)
        -> 25 measured zeros, n_nan 0, ZERO warnings
    every scorer, score(float("nan"))
        -> AttributeError: 'float' object has no attribute 'strip'

So a model that returned NOTHING was published as "did not refuse", "not toxic"
and "carries no stereotype", and those zeros are averaged and significance tested
downstream by ``OutputAnalyzer`` and ``noise_floor_from_runs``.

THE DISTINCTION THIS FILE EXISTS TO PROTECT, in both directions. ``""`` returning
a measured 0.0 is a DELIBERATE, separately pinned decision (see
``StereotypeScorer._score_or_none``, ``tests/test_bgl2_scorer_batch_paths.py``,
``tests/test_b4_llm_scorer_readability_share.py``): an empty response is a real
response that contains no refusal. ``None`` is the ABSENCE of a response and is a
could-not-check. Collapsing the two is the defect; withdrawing the empty-string
0.0 would be the over-correction, so ``test_the_empty_string_keeps_its_measured_zero_silently``
is as load-bearing as the refusals above it.

EVERY PIN HERE WAS SABOTAGED: the defect was put back in the source, the named
test was confirmed RED, and the source was restored and diffed byte-identical.

THE OVER-CORRECTION CONTROL was also run OUTSIDE pytest, as 88 fresh
subprocesses (11 scorers x 8 inputs: '', '   ', a refusal, a helpful answer, a
non-refusal, a stereotype, a Japanese refusal, a French refusal) against the
pre-fix file and the post-fix file. Values and warning sets were byte-identical.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vfairness.llm.scorers as S

#: Every scorer whose ``score(None)`` answered a MEASURED 0.0 before this change.
#: These are the ones where the defect published a verdict; the name of each is
#: the before-state, so a scorer added to this file has to be measured, not
#: assumed.
_SCORERS_THAT_ANSWERED_A_MEASURED_ZERO = [
    "RefusalScorer",
    "HelpfulnessScorer",
    "InformationQualityScorer",
    "RepresentationScorer",
    "SemanticQualityScorer",
    "StereotypeScorer",
    "ContextualStereotypeScorer",
    "KeywordToxicityScorer",
]

#: The three that already answered NaN for ``None`` but raised AttributeError for
#: ``float("nan")``, and whose NaN was attributed to the wrong reason ("this text
#: is blank" rather than "there is no text").
_SCORERS_THAT_ALREADY_REFUSED_NONE = [
    "FramingScorer",
    "KeywordRegardScorer",
    "KeywordSentimentScorer",
]

_ALL = _SCORERS_THAT_ANSWERED_A_MEASURED_ZERO + _SCORERS_THAT_ALREADY_REFUSED_NONE

#: Shapes that are NOT a recorded response. ``float("nan")`` is in here because it
#: is TRUTHY: it did not short-circuit at all, it reached ``.strip()`` and raised.
_NOT_A_RESPONSE = [None, float("nan"), np.float64("nan"), 0, 0.0, [], {}]


def _scored(name, text):
    """``(value, warnings)`` from one scorer's public entry, warnings recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = getattr(S, name)().score(text)
    return value, list(caught)


def _no_response_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, S.UnscorableTextWarning)
        and "no recorded response" in str(w.message)
    ]


# ==========================================================================
# THE PREDICATE. One test for every shape, in one place, because eight scorers
# used to each have their own falsy test and they all got it wrong the same way.
# ==========================================================================


@pytest.mark.parametrize("value", _NOT_A_RESPONSE)
def test_the_predicate_names_every_shape_that_is_not_text(value):
    assert S.response_not_recorded(value) is True


@pytest.mark.parametrize("value", ["", "   ", "\n", "a", "0", "none", "None"])
def test_the_predicate_calls_every_string_a_response(value):
    """INCLUDING the empty one, and including the literal word 'none'.

    ``str(None)`` is ``'none'`` after normalisation, which is exactly how the
    defect minted a readable word out of nothing. A caller who really did receive
    the four characters "none" received a response, and it is scored.
    """
    assert S.response_not_recorded(value) is False


# ==========================================================================
# DEFECT 1. The measured zero.
# ==========================================================================


@pytest.mark.parametrize("name", _ALL)
def test_a_missing_response_is_a_could_not_check_not_a_zero(name):
    """Before: 0.0 for eight of these eleven, with no warning at all."""
    value, caught = _scored(name, None)
    assert math.isnan(value), (
        f"{name}.score(None) returned {value!r}. On these scales 0.0 is the "
        "MEASURED claim 'this response did not refuse' / 'is not toxic', for a "
        "response that does not exist."
    )
    assert _no_response_warnings(caught), (
        f"{name} refused silently. A NaN a reader never hears about is averaged "
        "out of existence one layer up; the warning is the third state made visible."
    )


@pytest.mark.parametrize("name", _ALL)
def test_a_float_nan_response_is_refused_and_does_not_raise(name):
    """Before: AttributeError: 'float' object has no attribute 'strip'.

    The fourth shape. ``float("nan")`` is truthy, so ``if not text`` did not catch
    it and ``text.strip()`` raised out of the public entry point. A crash is not a
    fabricated measurement, but it is the same missing distinction, and a caller
    reading a dataframe column of responses meets it first.
    """
    value, caught = _scored(name, float("nan"))
    assert math.isnan(value)
    assert _no_response_warnings(caught)


def test_the_batch_path_counts_and_names_the_missing_responses():
    """Before: 25 measured zeros, n_nan 0, ZERO warnings.

    This is the path every aggregate comparison consumes, so it is the one that
    mattered: those 25 zeros are averaged and significance tested.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = S.RefusalScorer().score_batch([None] * 25)
    assert int(np.count_nonzero(np.isnan(scores))) == 25, scores
    named = _no_response_warnings(caught)
    assert len(named) == 1, "one aggregate line per call, not 25 identical ones"
    assert "25 of 25" in str(named[0].message)


#: The two batch implementations in this module, which count the third state in
#: two DIFFERENT ways, so a pin over one of them says nothing about the other.
#: ``RefusalScorer.score_batch`` counts by REASON, so it cannot double count by
#: construction. The other nine subtract the missing count from the unreadable
#: one, and THAT is the arithmetic a pin has to be able to catch: sabotaging the
#: subtraction left the RefusalScorer case green, which is how this
#: parametrisation came to exist.
_BOTH_BATCH_IMPLEMENTATIONS = ["RefusalScorer", "HelpfulnessScorer", "StereotypeScorer"]


@pytest.mark.parametrize("name", _BOTH_BATCH_IMPLEMENTATIONS)
def test_the_batch_path_does_not_double_count_a_missing_response(name):
    """One item, one reason. A missing response is not also an unreadable one.

    Both come back as ``None``, so a naive count reports the same item twice, once
    under each reason, and a reader cannot then tell how much of each the batch
    met. Two missing, one CJK text no ASCII lexicon can read, one readable answer.
    """
    texts = [
        None,
        None,
        "そのリクエストには対応できません。",
        "Yes. First, open the file, then run the build, and check the log for errors.",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = getattr(S, name)().score_batch(texts)
    assert math.isnan(scores[0]) and math.isnan(scores[1])
    assert math.isnan(scores[2]), "the CJK text is still a could-not-check"
    assert not math.isnan(scores[3]), "the readable answer is still measured"
    missing = _no_response_warnings(caught)
    unreadable = [
        w
        for w in caught
        if issubclass(w.category, S.UnscorableTextWarning)
        and "produced no readable" in str(w.message)
    ]
    assert len(missing) == 1 and "2 of 4" in str(missing[0].message)
    assert len(unreadable) == 1 and "1 of 4" in str(unreadable[0].message), (
        f"{name} counted the 2 missing responses under the unreadable reason as "
        f"well: {[str(w.message)[:60] for w in unreadable]}"
    )


def test_an_identifier_nobody_supplied_has_no_words():
    """THE FIFTH SITE, found by grepping this module for the other ``str(x)``.

    Before: ``identifier_tokens(None) == ['none']`` and
    ``identifier_tokens(float('nan')) == ['nan']``, a real matchable word token
    invented out of an absent name. ``_names.name_token_list`` is the sibling that
    answers the same question for the rest of the package and was fixed for
    exactly this on 2026-09-27; this one was missed.

    NOT LIVE at its only in-repo caller (the pulse label-quality stage matches
    against ``_LQ_ANNOTATOR_TOKENS``, which holds no "none" and no "nan"), so this
    pin is about the function's contract rather than a published verdict.
    """
    assert S.identifier_tokens(None) == []
    assert S.identifier_tokens(float("nan")) == []
    # The controls: every identifier a caller actually wrote is unchanged, and the
    # STRING "None" is a name somebody wrote.
    assert S.identifier_tokens("encoder_output") == ["encoder", "output"]
    assert S.identifier_tokens("raterName") == ["rater", "name"]
    assert S.identifier_tokens("judge_1") == ["judge", "1"]
    assert S.identifier_tokens("None") == ["none"]
    assert S.identifier_tokens(0) == ["0"]


def test_a_missing_response_does_not_mint_a_readable_word():
    """``normalize_text(None)`` is 'none', so the readability gate read 1.0.

    Defence in depth for any caller that reaches the tokeniser directly. The
    normalisation itself is unchanged, deliberately: it is documented to accept
    anything and stringify it, and every gate that decides an OUTCOME is above it.
    """
    assert S.word_tokens(None) == []
    assert S.lexicon_readable_share(None) == 0.0
    assert S.lexicon_can_read(None) is False
    assert S.normalize_text(None) == "none", (
        "unchanged on purpose: this function's contract is 'fold anything to text'"
    )


# ==========================================================================
# OVER-CORRECTION CONTROLS. Mandatory. A gate that refused what it was unsure of
# would pass every assertion above and delete the library.
# ==========================================================================


@pytest.mark.parametrize("name", _SCORERS_THAT_ANSWERED_A_MEASURED_ZERO)
@pytest.mark.parametrize("text", ["", "   "])
def test_the_empty_string_keeps_its_measured_zero_silently(name, text):
    """A DELIBERATE, separately pinned decision, and this change must not touch it.

    An empty generation is a real response that contains no refusal, no toxic word
    and no stereotype. Measured before AND after: 0.0, and no unscorable warning.
    """
    value, caught = _scored(name, text)
    assert value == 0.0, f"{name}.score({text!r}) moved to {value!r}"
    assert not _no_response_warnings(caught), (
        f"{name} called an EMPTY response a MISSING one, which is the same "
        "collapse running backwards"
    )


@pytest.mark.parametrize("name", _SCORERS_THAT_ALREADY_REFUSED_NONE)
@pytest.mark.parametrize("text", ["", "   "])
def test_the_scorers_that_already_refused_blank_text_still_do(name, text):
    """These three refuse blank text for their own recorded reasons (2026-09-25).

    Their NaN is not touched here; only the REASON changes, from "this text is
    blank" to "there is no text", and only when there is in fact no text.
    """
    value, caught = _scored(name, text)
    assert math.isnan(value)
    assert not _no_response_warnings(caught)


def test_a_real_refusal_still_scores_one_and_readable_text_is_unchanged():
    """The numbers, not the shape. Each of these is the value measured before."""
    assert S.RefusalScorer().score("I'm sorry, I cannot help with that request.") == 1.0
    assert S.RefusalScorer().score("I can help with that.") == 0.0
    assert S.StereotypeScorer().score("Women are naturally more emotional than men.") > 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        helpful = S.HelpfulnessScorer().score(
            "Yes. First, open the file, then run the build, and check the log for errors."
        )
    assert helpful > 0.0 and not _no_response_warnings(caught)


def test_a_readable_batch_is_untouched_and_silent():
    """Twenty-five real refusals: 25 ones, no NaN, no missing-response warning."""
    texts = ["I cannot assist with that request."] * 25
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scores = S.RefusalScorer().score_batch(texts)
    assert np.unique(scores).tolist() == [1.0]
    assert not _no_response_warnings(caught)
