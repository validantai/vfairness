"""The six ``score_batch`` methods, which were graded PROVEN and never executed.

HOW THEY WERE FOUND. ``scripts/verify_grade_attribution.py`` asks, per grading row,
whether the test file the row NAMES actually reaches the unit's body lines. Six rows
in grading wave 1 named ``tests/test_surface_grade_g000.py`` for
``score_batch`` on the keyword scorers, carried a grade of PROVEN with no sabotage,
and were published as CHECKED. That file never calls ``score_batch``. The single-text
``score`` path was thoroughly pinned; the batch path beside it was not pinned at all.

That distinction matters more than it looks. Every aggregate comparison in the LLM
fairness surface goes through the batch path, so a batch that diverges from the
single path by one element is a disparity computed from values no caller could
reproduce one at a time.

WHAT WAS FOUND. ``FramingScorer`` returned **0.5**, the exact midpoint of its [0, 1]
scale, for empty and whitespace-only text, while a real text carrying no framing
marker two branches below correctly returned NaN. So "the doctor was competent and
kind" was refused and "" was scored. The carve-out justified itself "on the same
ground as ``KeywordSentimentScorer``'s empty-text carve-out", and that carve-out had
been withdrawn on 2026-09-17 by the argument that a midpoint is a measured value and
not an absence. The cross-reference outlived the thing it pointed at.

WHAT IS DELIBERATE AND IS PINNED HERE SO IT SURVIVES. ``KeywordToxicityScorer``
keeps a MEASURED 0.0 for a readable text containing no toxic word, because it is a
presence detector and absence of toxic words is evidence of absence. That is not the
same as the framing case and must not be "fixed" to NaN: doing so would refuse every
non-toxic response in the corpus. The two scorers differ on purpose, and the reason
lives in their docstrings.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.llm.scorers import (
    ContextualStereotypeScorer,
    FramingScorer,
    KeywordRegardScorer,
    KeywordSentimentScorer,
    KeywordToxicityScorer,
    RefusalScorer,
)

SCORERS = [
    KeywordToxicityScorer,
    KeywordSentimentScorer,
    KeywordRegardScorer,
    RefusalScorer,
    FramingScorer,
    ContextualStereotypeScorer,
]

# A corpus spanning every input class these scorers distinguish: scorable text at
# both poles, a refusal, text with no marker of any kind, non-ASCII text the
# lexicons cannot read, and blank text.
CORPUS = [
    "you are an idiot",
    "have a nice day",
    "he was clearly incompetent, obviously",
    "I cannot help with that request",
    "she was allowed to speak",
    # A text the STEREOTYPE scorer scores non-zero, and it is here because its
    # absence made one pin blind. Sabotaging each score_batch to return zeros turned
    # the agreement test red for five of the six scorers and left
    # ContextualStereotypeScorer green: every text above scores 0.0 for it, so a
    # constant-zero batch matched its honest output exactly. A corpus that cannot
    # produce a non-zero value for a scorer cannot detect that scorer answering zero
    # to everything.
    "women are naturally more nurturing",
    "zzzz qqqq",
    "El gato come pescado.",
    "",
    "   ",
]


@pytest.fixture(autouse=True)
def _restore_the_once_per_process_warning_flags():
    """Leave the module's warn-once flags exactly as they were found.

    ``_warn_sidecar_down`` fires ONCE PER PROCESS, by a module-level boolean. This
    file scores text with every scorer, which trips that flag, and
    ``tests/test_no_aggregator_fabricates_a_verdict.py::
    test_a_scorer_whose_sidecar_is_down_is_loud_about_it`` then found no warning to
    observe and failed. It passed on its own and failed in the full run, which is the
    signature of exactly this kind of shared state.

    The flag is process-global on purpose (a long run must not emit one warning per
    generation), so the fix belongs here: a test that consumes another test's
    once-per-process budget has to put it back.
    """
    from vfairness.llm import scorers as _scorers

    saved = {
        name: getattr(_scorers, name)
        for name in ("_sidecar_down_warned", "_sidecar_bad_reply_warned")
        if hasattr(_scorers, name)
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_scorers, name, value)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@pytest.mark.parametrize("cls", SCORERS, ids=[c.__name__ for c in SCORERS])
def test_the_batch_path_agrees_with_the_single_path_element_by_element(cls):
    """The batch path is what every aggregate comparison uses, and it had no pin.

    A divergence here produces a group disparity from values the caller cannot
    reproduce by scoring the same texts one at a time, and nothing downstream
    could tell.
    """
    scorer = cls()
    single = np.array([_quiet(scorer.score, text) for text in CORPUS], dtype=float)
    batch = np.asarray(_quiet(scorer.score_batch, CORPUS), dtype=float)

    assert batch.shape == single.shape, (
        f"{cls.__name__}.score_batch returned {batch.shape} values for {len(CORPUS)} texts"
    )
    np.testing.assert_array_equal(
        np.isnan(batch),
        np.isnan(single),
        err_msg=f"{cls.__name__}: batch and single disagree about WHICH texts are unscorable",
    )
    finite = ~np.isnan(single)
    np.testing.assert_allclose(
        batch[finite],
        single[finite],
        err_msg=f"{cls.__name__}: batch and single disagree on a scorable text",
    )


@pytest.mark.parametrize("cls", SCORERS, ids=[c.__name__ for c in SCORERS])
def test_an_empty_batch_returns_an_empty_result_and_not_a_zero(cls):
    """Nothing to score is not a score. A scalar 0.0 here would become a group
    mean of 0.0 for a group with no generations."""
    result = np.asarray(_quiet(cls().score_batch, []), dtype=float)
    assert result.shape == (0,), f"{cls.__name__} invented {result} for an empty batch"


# THE TWO FAMILIES ANSWER A BLANK GENERATION DIFFERENTLY, ON PURPOSE, and the first
# version of this file asserted that all six must refuse it. Three failed, and the
# test was wrong rather than the library: it also contradicted the toxicity assertion
# further down in this same file.
#
#   A MIDPOINT SCALE has a neutral value in the middle of its range, and that value
#   is a measurement. Sentiment's 0.0 sits between the poles and beats every negative
#   score; framing's 0.5 is the centre of [0, 1]. Returning it for a text that does
#   not exist makes two groups of blanks compare equal with assessed=True.
#
#   A PRESENCE DETECTOR has its neutral value at the END of its range, and reaching
#   it means the marker was looked for and not found. Absence of a toxic word, of a
#   refusal phrase, of a stereotype cue IS evidence of absence. Refusing these would
#   leave no comparison computable, because most responses are non-toxic.
#
# So the families are pinned separately, and the pair is pinned together below, since
# a deliberate inconsistency looks exactly like an accidental one to the next reader.
MIDPOINT_SCALES = [KeywordSentimentScorer, KeywordRegardScorer, FramingScorer]
PRESENCE_DETECTORS = [KeywordToxicityScorer, RefusalScorer, ContextualStereotypeScorer]


@pytest.mark.parametrize("cls", MIDPOINT_SCALES, ids=[c.__name__ for c in MIDPOINT_SCALES])
def test_a_midpoint_scale_refuses_blank_text_rather_than_reporting_its_centre(cls):
    result = np.asarray(_quiet(cls().score_batch, ["", "  ", "\t\n"]), dtype=float)

    assert np.all(np.isnan(result)), (
        f"{cls.__name__} scored blank generations {result}. On a scale with a neutral "
        f"centre a constant makes two groups of blanks compare equal with "
        f"assessed=True, which is the defect the marker-free branch of these scorers "
        f"was written to remove."
    )


@pytest.mark.parametrize("cls", PRESENCE_DETECTORS, ids=[c.__name__ for c in PRESENCE_DETECTORS])
def test_a_presence_detector_reports_a_measured_zero_for_a_readable_blank(cls):
    """DELIBERATE. Pinned so the refusal above is never generalised into this family:
    a detector that refused every text without its marker would refuse almost the
    whole corpus and leave nothing to compare."""
    result = np.asarray(_quiet(cls().score_batch, ["", "  "]), dtype=float)

    assert not np.any(np.isnan(result)), (
        f"{cls.__name__} started refusing text it had read. Absence of the marker it "
        f"detects is evidence of absence, and refusing it removes every comparison."
    )
    assert np.all(result == 0.0)


def test_the_framing_scorer_refuses_blank_text_as_it_refuses_marker_free_text():
    """The specific defect: 0.5 for "" while a real marker-free text got NaN."""
    scorer = FramingScorer()

    assert np.isnan(_quiet(scorer.score, "")), "empty text scored a framing midpoint"
    assert np.isnan(_quiet(scorer.score, "   ")), "whitespace scored a framing midpoint"
    assert np.isnan(_quiet(scorer.score, "the doctor was competent and kind")), (
        "the marker-free fixture no longer trips the refusal, so the pair above "
        "proves nothing about consistency"
    )


def test_control_the_framing_scorer_still_scores_text_that_carries_markers():
    """The over-correction control. A scorer that refused everything would pass
    every assertion above."""
    value = _quiet(FramingScorer().score, "he was clearly incompetent, obviously")
    assert not np.isnan(value), "framing refused a text carrying two certainty markers"
    assert 0.0 <= value <= 1.0


def test_toxicity_keeps_a_measured_zero_for_a_readable_non_toxic_text():
    """DELIBERATE, and pinned so the blank-text fix above is not generalised into it.

    This scorer is a presence detector: a text the tokeniser read, containing no
    toxic word, is a measured 0.0. Turning that into NaN would refuse every
    non-toxic response in a corpus and leave no toxicity comparison computable at
    all.
    """
    value = _quiet(KeywordToxicityScorer().score, "have a nice day")
    assert value == 0.0, (
        "the measured zero became a refusal; absence of toxic words IS evidence of "
        "absence for a presence detector, unlike a midpoint on a signed scale"
    )


def test_sentiment_refuses_where_toxicity_measures_and_that_is_on_purpose():
    """The two scorers answer the same blank input differently, by design. Pinned
    as a pair, because an inconsistency that is deliberate looks exactly like one
    that is not, and the next reader will otherwise 'fix' whichever they meet first."""
    assert _quiet(KeywordToxicityScorer().score, "have a nice day") == 0.0
    assert np.isnan(_quiet(KeywordSentimentScorer().score, "   "))


@pytest.mark.parametrize("cls", MIDPOINT_SCALES, ids=[c.__name__ for c in MIDPOINT_SCALES])
def test_a_refusing_batch_says_out_loud_that_it_refused(cls):
    """A NaN in an array a caller may never inspect element-wise is not enough on its
    own. Only the refusing family is asked this: a presence detector that measured
    everything has nothing to announce, and a warning there would be noise on every
    non-toxic response in the corpus."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls().score_batch(["", "  "])

    assert caught, f"{cls.__name__} refused two texts in silence"
