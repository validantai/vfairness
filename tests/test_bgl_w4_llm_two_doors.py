"""Audit wave 4, batch T1-llm: five overturned grades, each a SIBLING DOOR.

Every grade in this batch was published as PROVEN and overturned by an independent
auditor, by execution. The five defects are three shapes of one mistake:

1. A GUARD ON ONE DOOR AND NOT ITS SIBLING. ``response_not_recorded`` is complete
   as a predicate and was consulted in ``SidecarSentimentScorer.score`` and
   ``SidecarToxicityScorer.score`` but in NEITHER of their ``score_batch``
   methods, while the other nine scorers carry it in both. ``score_batch`` is the
   path every aggregate comparison consumes. Separately,
   ``OutputAnalyzer._filter_none`` held a SECOND, weaker definition of "missing"
   (``t is not None``) written in the same wave as the predicate, and it ran
   first.

2. A THRESHOLD THAT ANY LARGE ENOUGH CARRIER DEFEATS. ``lexicon_readable_share``
   is a letter-count share against 0.5, and a letter is not a unit of meaning
   across scripts, so a Japanese refusal plus " https://example.com/help" read
   0.514 and was declared readable. It then published HelpfulnessScorer 0.27,
   InformationQualityScorer 0.25 and RefusalScorer 0.0 with ZERO warnings: a model
   that refused every Japanese user, published as refusing nobody. The WRONG POLE,
   not a missing measurement.

3. THE SAME ERROR RUNNING BACKWARDS. ``_is_zero_power`` was written for the door
   where p=1.0 is FORCED by ``_compare``'s ``np.array_equal`` short circuit, and
   it closed as well on the sibling door where p=1.0 is COMPUTED by a real test
   whose two group means are equal. A genuine negative, equal refusal rates, which
   is the outcome a fair model is supposed to produce, was republished as "no
   comparison was performed" with a factually false sentence about what the scorer
   read, and removed from the correction family, which LOWERS every surviving
   row's adjusted p.

THE CONTROLS ARE AS LOAD-BEARING AS THE REFUSALS, and this batch has that on
record: the state-1 pin it replaces could not fail its own recorded sabotage,
because its fixture held ZERO ASCII letters and the sabotage made every score NaN,
which is what that pin asserted. Every fixture here that must be refused CARRIES AN
ASCII BLOCK, and every control asserts the real measured number rather than "it did
not raise".

Every pin here was sabotaged at its source line, confirmed RED, restored, and the
source diffed byte-identical against ``git show HEAD:<path>``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

import vfairness.llm.scorers as S
from vfairness.llm.output_analysis import (
    MAX_TOTAL_UNSCORED,
    ZERO_POWER_REASON,
    OutputAnalyzer,
    _is_zero_power,
)

# ===========================================================================
# Fixtures
# ===========================================================================

#: A total refusal in Japanese. Bare, 0.000 of its letters are readable to an
#: ASCII lexicon and 1.000 of them are in a script it cannot read.
_JA_REFUSAL = "申し訳ございません、お手伝いできません。"

#: The four ordinary ASCII blocks the auditor measured walking through the 0.5
#: share gate, with the share each one produces. NONE of them is prose the English
#: lexicons "genuinely read", which is the justification the threshold's own
#: docstring gives for not being 1.0.
_ASCII_CARRIERS = {
    "support_url": (" https://example.com/help", 0.514),
    "code_fragment": (" print(result) return value", 0.550),
    "english_footer": (" Powered by the assistant, see our terms and conditions.", 0.714),
    "email_address": (" contact support at helpdesk@example.com", 0.654),
}

_EN_ANSWER = (
    "Here is a detailed plan. First, check your eligibility for the stipend, "
    "because the programme pays monthly and a mentor is assigned to you."
)

_EN_REFUSALS = [
    "I cannot help with that request, sorry.",
    "I am unable to assist with this, I'm afraid.",
    "I cannot provide that information.",
    "I'm sorry, but I can't help with that.",
]

_EN_ANSWERS = [
    "Here is the monthly stipend schedule and the mentor assignment process.",
    "The eligibility rules are listed below, with the deadline for each stage.",
    "You can apply online; the form takes about ten minutes to complete.",
    "The programme runs for six months and includes a weekly workshop.",
]


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        return fn(*args, **kwargs), caught


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


class _AnsweringBridge:
    """A sidecar that ANSWERS, and records exactly what it was asked about.

    The hole in the two batch paths is live ONLY when the sidecar answers: a dead
    sidecar returns NaN for everything and masks it, which is why the grade read
    PROVEN on a machine with no sidecar.
    """

    def __init__(self, score_for_prose: float = 0.9):
        self.score_for_prose = score_for_prose
        self.sent: list = []

    def call(self, op, **payload):
        if op.endswith("_batch"):
            texts = payload["texts"]
            self.sent.append((op, list(texts)))
            return [
                self.score_for_prose if (isinstance(t, str) and t.strip()) else 0.0 for t in texts
            ]
        text = payload["text"]
        self.sent.append((op, text))
        return self.score_for_prose if (isinstance(text, str) and text.strip()) else 0.0


@pytest.fixture
def answering_bridge(monkeypatch):
    bridge = _AnsweringBridge()
    monkeypatch.setattr(S._SidecarBridge, "get", classmethod(lambda cls: bridge))
    return bridge


@pytest.fixture(autouse=True)
def _restore_the_once_per_process_warning_flags():
    """The sidecar warnings latch per process by design; put the flags back."""
    saved = {
        name: getattr(S, name)
        for name in ("_sidecar_down_warned", "_sidecar_bad_reply_warned", "_sidecar_probe_failed")
        if hasattr(S, name)
    }
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(S, name, value)


class _CallerJudge:
    """An ordinary caller-supplied judge, written the way a judge is written.

    It interpolates the response into a prompt string, so EVERY shape of missing
    response becomes a scorable string and the judge answers a number for text
    that was never produced. ``llm_judge_scorer`` has no default, so a
    caller-supplied judge is the only kind ``analyze_llm_judge`` ever has.
    """

    def score(self, text) -> float:
        prompt = "Rate this response on 0..10:\n%s" % text
        return 0.9 if "stipend" in prompt else 0.6

    def score_batch(self, texts) -> np.ndarray:
        return np.array([self.score(t) for t in texts], dtype=float)


# ===========================================================================
# 1. response_not_recorded: the batch sibling of a guarded single-text door
# ===========================================================================


@pytest.mark.parametrize(
    "scorer_cls, op",
    [
        (S.SidecarSentimentScorer, "score_sentiment_batch"),
        (S.SidecarToxicityScorer, "score_toxicity_batch"),
    ],
)
def test_the_batch_path_refuses_a_response_that_was_never_recorded(
    scorer_cls, op, answering_bridge
) -> None:
    """The sibling door, with the auditor's own inputs.

    Measured before, with this exact bridge:
    ``score_batch([None, "a lovely person", nan, None])`` -> ``[0., 0.9, 0., 0.]``,
    n_nan 0, warnings [], and ``[None, nan, None]`` were SENT to the sidecar as
    texts; ``score_batch([None] * 25)`` -> 25 measured zeros, n_nan 0, warnings [].
    On these scales 0.0 is the MEASURED claim "neutral sentiment" / "not toxic".
    """
    scorer = scorer_cls()
    texts = [None, "a lovely person", float("nan"), None]
    scores, caught = _caught(scorer.score_batch, texts)

    assert len(scores) == 4
    assert [math.isnan(v) for v in scores] == [True, False, True, True]
    # The one real response keeps its real score, at its own position.
    assert scores[1] == 0.9
    assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught), [
        str(w.message) for w in caught
    ]
    # And the items that hold no response are not SENT at all.
    assert answering_bridge.sent[-1] == (op, ["a lovely person"])

    # pd.NA and pd.NaT are the same absence arriving from a dataframe column.
    for shape in (pd.NA, pd.NaT):
        scores, caught = _caught(scorer.score_batch, ["a lovely person", shape])
        assert math.isnan(scores[1]), shape
        assert scores[0] == 0.9, shape
        assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught), shape

    scores, caught = _caught(scorer.score_batch, [None] * 25)
    assert int(np.count_nonzero(~np.isfinite(scores))) == 25
    assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught)


@pytest.mark.parametrize("scorer_cls", [S.SidecarSentimentScorer, S.SidecarToxicityScorer])
def test_control_the_batch_path_still_measures_what_it_is_given(
    scorer_cls, answering_bridge
) -> None:
    """A gate that refused everything would pass the pin above and delete the rung.

    The REAL numbers: this bridge answers 0.9 for prose and 0.0 for blank text, and
    both must arrive unchanged, with no warning at all.
    """
    scorer = scorer_cls()
    scores, caught = _caught(scorer.score_batch, ["a lovely person", "an ordinary answer"])
    assert list(scores) == [0.9, 0.9]
    assert not caught, [str(w.message) for w in caught]

    # '' is a RESPONSE, and keeps its measured 0.0 on both paths. This decision is
    # separately pinned in scorers.py and must never be collapsed into the absence.
    scores, caught = _caught(scorer.score_batch, ["", "a lovely person"])
    assert list(scores) == [0.0, 0.9]
    assert not caught, [str(w.message) for w in caught]
    assert scorer.score("") == 0.0


@pytest.mark.parametrize("lost", [None, float("nan"), pd.NA, pd.NaT], ids=str)
def test_the_two_definitions_of_missing_now_agree_at_the_analyzer(lost, answering_bridge) -> None:
    """One world, ONE description, whichever shape the loss arrives in.

    ``_filter_none`` defined missing as ``t is not None`` while
    ``response_not_recorded`` defines it as ``not isinstance(text, str)``, and the
    weaker one ran first. Measured before, 8 of group B's 10 responses lost, the
    other group intact, with an ANSWERING sidecar:

        as None  -> assessed=False, 'differential_unscored', every field None
        as nan   -> assessed=True, group_b 0.29999999999999993, delta 0.0,
                    p_value 1.0, n_supplied_b 10 read as full coverage
        as pd.NA -> identical to nan
        as pd.NaT-> identical to nan
    """
    analyzer = OutputAnalyzer(sentiment_scorer=S.SidecarSentimentScorer())
    texts_a = [_EN_ANSWER] * 10
    texts_b = [_EN_ANSWER] * 2 + [lost] * 8

    result, caught = _caught(analyzer.analyze_sentiment, texts_a, texts_b, "men", "women")

    assert result.assessed is False
    assert result.not_assessed_reason == "differential_unscored"
    assert result.group_a_value is None and result.group_b_value is None
    assert result.delta is None and result.p_value is None and result.is_significant is None
    assert any(w.category is RuntimeWarning and "UNEVEN" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_control_full_coverage_still_measures_a_real_sentiment_disparity(
    answering_bridge,
) -> None:
    """The over-correction control for the two fixes above, with real numbers.

    Nothing is missing, so nothing may be refused: the sidecar answers 0.9 for
    prose and 0.0 for blank text, and the comparison reports exactly that.
    """
    analyzer = OutputAnalyzer(sentiment_scorer=S.SidecarSentimentScorer())
    result = _quiet(analyzer.analyze_sentiment, [_EN_ANSWER] * 10, [""] * 10, "men", "women")
    assert result.assessed is True
    assert result.not_assessed_reason is None
    assert result.group_a_value == pytest.approx(0.9)
    assert result.group_b_value == pytest.approx(0.0)
    assert result.delta == pytest.approx(0.9)
    assert result.p_value == pytest.approx(1.593791168806624e-05, rel=1e-9)
    assert result.is_significant is True
    assert result.n_scored_a is None and result.n_scored_b is None


# ===========================================================================
# 2 + 3. analyze_helpfulness / analyze_information_quality: the carrier
# ===========================================================================


@pytest.mark.parametrize("carrier_name", sorted(_ASCII_CARRIERS))
def test_an_ascii_carrier_does_not_make_a_non_latin_text_readable(carrier_name) -> None:
    """THE FIXTURE CARRIES AN ASCII BLOCK, which is the point.

    The state-1 pin this replaces used a fixture with ZERO ASCII letters, so it
    read share 0.000 trivially and its recorded sabotage (every score reaches
    _compare as NaN) could not redden it: the pin asserted NaN and the sabotage
    produced NaN. Each carrier below crosses MIN_LEXICON_READABLE_SHARE at the
    share recorded beside it and was published as a measurement.
    """
    carrier, expected_share = _ASCII_CARRIERS[carrier_name]
    text = _JA_REFUSAL + carrier

    # The fixture really does carry an ASCII block, and really did cross the gate.
    assert sum(1 for ch in text if ch.isascii() and ch.isalpha()) >= 10
    assert S.lexicon_readable_share(text) == pytest.approx(expected_share, abs=5e-4)
    assert S.lexicon_readable_share(text) >= S.MIN_LEXICON_READABLE_SHARE
    assert S.word_tokens(text), "the tokeniser does see words in the carrier"

    # And it is still not a text these lexicons read.
    assert S.non_lexicon_script_share(text) > S.MAX_NON_LEXICON_SCRIPT_SHARE
    assert S.lexicon_can_read(text) is False

    for scorer in (
        S.HelpfulnessScorer(),
        S.InformationQualityScorer(),
        S.RefusalScorer(),
    ):
        value, caught = _caught(scorer.score, text)
        assert math.isnan(value), f"{type(scorer).__name__} scored {value}"
        assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught), (
            f"{type(scorer).__name__} refused SILENTLY: {[str(w.message) for w in caught]}"
        )


def test_a_refused_every_japanese_user_run_is_a_could_not_check_not_a_clean_bill() -> None:
    """The WRONG POLE, at the analyzer's own surface.

    Measured before, ten English answers against ten Japanese refusals each
    carrying a support URL:

        analyze_helpfulness         assessed=True a=0.445 b=0.27 delta=0.175
                                    p=1.593791168806624e-05 is_significant=True
        analyze_information_quality assessed=True a=0.45  b=0.25 delta=0.2
                                    p=1.593791168806624e-05 is_significant=True
        analyze_refusal_rate        a=0.0 b=0.0 delta=0.0 p=1.0 is_significant=False

    The last line is the defect: the one metric that would have NAMED the
    discrimination published "this model refused nobody" for a model that refused
    every Japanese user, and its magnitude was a keyword count over a URL.
    """
    analyzer = OutputAnalyzer()
    texts_a = [_EN_ANSWER] * 10
    texts_b = [_JA_REFUSAL + _ASCII_CARRIERS["support_url"][0]] * 10

    for name in (
        "analyze_helpfulness",
        "analyze_information_quality",
        "analyze_refusal_rate",
    ):
        result = _quiet(getattr(analyzer, name), texts_a, texts_b, "english", "japanese")
        assert result.assessed is False, f"{name} published {result.group_b_value}"
        assert result.not_assessed_reason in (
            "non_finite_scores",
            "differential_unscored",
        ), (name, result.not_assessed_reason)
        assert result.group_a_value is None and result.group_b_value is None, name
        assert result.p_value is None and result.is_significant is None, name


def test_control_the_text_these_lexicons_do_read_keeps_its_real_score() -> None:
    """A readability gate that refuses everything passes every refusal test above
    and destroys the scorers. These are the measured numbers after the fix."""
    helpfulness, quality, refusal = (
        S.HelpfulnessScorer(),
        S.InformationQualityScorer(),
        S.RefusalScorer(),
    )

    # English prose: read in full, no warning, real numbers.
    value, caught = _caught(helpfulness.score, _EN_ANSWER)
    assert value == pytest.approx(0.445, abs=5e-4)
    assert not caught, [str(w.message) for w in caught]
    value, caught = _caught(quality.score, _EN_ANSWER)
    assert value == pytest.approx(0.45, abs=5e-4)
    assert not caught, [str(w.message) for w in caught]
    assert _quiet(refusal.score, _EN_ANSWER) == 0.0
    assert _quiet(refusal.score, "I cannot help with that request, sorry.") == 1.0

    # ACCENTED LATIN PROSE is Latin script: the script test must not touch it, and
    # the language test that refuses it is a different question with its own pin.
    accented = "Désolé, je ne peux pas vous aider avec cette demande."
    assert S.non_lexicon_script_share(accented) == 0.0
    assert S.lexicon_can_read(accented) is True
    assert _quiet(helpfulness.score, accented) == pytest.approx(0.31, abs=5e-4)
    assert _quiet(quality.score, accented) == pytest.approx(0.25, abs=5e-4)

    vietnamese = "Tôi không thể giúp bạn với yêu cầu này."
    assert S.non_lexicon_script_share(vietnamese) == 0.0
    assert S.lexicon_can_read(vietnamese) is True

    # English prose QUOTING another script is English prose these lexicons read.
    quoting = "The applicant wrote 田中 on the form, and I cannot help with that request."
    assert S.non_lexicon_script_share(quoting) == pytest.approx(0.035, abs=5e-3)
    assert S.lexicon_can_read(quoting) is True
    assert _quiet(refusal.score, quoting) == 1.0

    # A GENUINELY MIXED string, most of whose letters ARE read, still scores.
    mixed = _JA_REFUSAL + " " + _EN_ANSWER + " Here is the deadline for each stage."
    assert S.non_lexicon_script_share(mixed) < S.MAX_NON_LEXICON_SCRIPT_SHARE
    assert S.lexicon_can_read(mixed) is True
    value, caught = _caught(helpfulness.score, mixed)
    assert value == pytest.approx(0.445, abs=5e-4)
    assert not caught, [str(w.message) for w in caught]

    # AND A MATCHED REFUSAL IS ITS OWN EVIDENCE. This one IS mostly another script
    # (0.381), and the English refusal inside it is plainly found: refusing it
    # would be a measurement republished as a could-not-check.
    detected = "そのリクエストには対応できません。 I cannot help with that request."
    assert S.non_lexicon_script_share(detected) > S.MAX_NON_LEXICON_SCRIPT_SHARE
    assert S.lexicon_readable_share(detected) >= S.MIN_LEXICON_READABLE_SHARE
    assert _quiet(refusal.score, detected) == 1.0

    # Blank text is a RESPONSE and keeps its pinned, measured 0.0.
    for blank in ("", "   "):
        assert _quiet(refusal.score, blank) == 0.0
        assert _quiet(helpfulness.score, blank) == 0.0
        assert _quiet(quality.score, blank) == 0.0


def test_control_a_real_english_disparity_is_still_found_and_significant() -> None:
    """The public surface of the control: the gate must not swallow a real finding."""
    analyzer = OutputAnalyzer()
    result = _quiet(
        analyzer.analyze_refusal_rate,
        _EN_REFUSALS * 3,
        _EN_ANSWERS * 3,
        "women",
        "men",
    )
    assert result.assessed is True
    assert result.group_a_value == 1.0
    assert result.group_b_value == 0.0
    assert result.p_value == pytest.approx(1.911834771154503e-06, rel=1e-9)
    assert result.is_significant is True


# ===========================================================================
# 4. analyze_llm_judge: the third shape, and "small" with code behind it
# ===========================================================================


@pytest.mark.parametrize("lost", [None, float("nan"), pd.NA], ids=str)
def test_a_caller_judge_cannot_score_a_response_that_was_never_produced(lost) -> None:
    """The filter's own docstring names the hazard it did not cover.

    Measured before, 8 of group B's 10 responses lost, with the caller judge
    above: as nan -> assessed=True, group_a 0.9, group_b 0.6599999999999999,
    delta 0.2400000000000001, p_value 0.00044051921287451386,
    is_significant True, n_supplied_b 10, n_scored_b None, ZERO RuntimeWarnings.
    pd.NA was identical.
    """
    analyzer = OutputAnalyzer(llm_judge_scorer=_CallerJudge())
    texts_a = ["The programme offers a monthly stipend and a mentor."] * 10
    texts_b = ["The programme offers a monthly stipend and a mentor."] * 2 + [lost] * 8

    result, caught = _caught(analyzer.analyze_llm_judge, texts_a, texts_b, "men", "women")

    assert result.assessed is False
    assert result.not_assessed_reason == "differential_unscored"
    assert result.p_value is None and result.is_significant is None
    assert result.group_b_value is None
    assert [w.category is RuntimeWarning for w in caught].count(True) >= 1


def test_an_even_but_large_loss_is_refused_because_small_now_means_something() -> None:
    """The promise with no code behind it.

    ``_compare`` promised "attrition is even and SMALL -> MEASURE, and disclose
    coverage" and the only size test was ``kept < 2``. Measured before, 98 of BOTH
    groups' 100 responses unscored: assessed=True, group_a 0.9, group_b 0.6,
    delta 0.30000000000000004, p_value 0.19393085228241058, n_scored 2 and 2
    against n_supplied 100 and 100. A 2-of-200 comparison published as an
    assessment of two groups of a hundred.
    """

    class _NanJudge:
        def score_batch(self, texts) -> np.ndarray:
            return np.array(
                [
                    (0.9 if "stipend" in t else 0.6) if isinstance(t, str) else float("nan")
                    for t in texts
                ],
                dtype=float,
            )

    analyzer = OutputAnalyzer(llm_judge_scorer=_NanJudge())
    texts_a = ["The programme offers a monthly stipend."] * 2 + [float("nan")] * 98
    texts_b = ["a plain short answer"] * 2 + [float("nan")] * 98

    result, caught = _caught(analyzer.analyze_llm_judge, texts_a, texts_b, "men", "women")

    assert result.assessed is False
    assert result.not_assessed_reason == "majority_unscored"
    assert result.group_a_value is None and result.group_b_value is None
    assert result.p_value is None and result.is_significant is None
    assert any(
        w.category is RuntimeWarning and f"{MAX_TOTAL_UNSCORED:.0%} bound" in str(w.message)
        for w in caught
    ), [str(w.message) for w in caught]


def test_control_an_even_and_genuinely_small_loss_is_measured_and_disclosed() -> None:
    """The over-correction control for the bound above, with the real numbers.

    A bound that refused every partial coverage would pass the pin above and throw
    away real measurements. Two of thirty-two per side is the case the third arm
    exists for: it MEASURES, and the coverage travels on the result.
    """

    class _NanJudge:
        def score_batch(self, texts) -> np.ndarray:
            return np.array(
                [
                    (0.9 if "stipend" in t else 0.6) if isinstance(t, str) else float("nan")
                    for t in texts
                ],
                dtype=float,
            )

    analyzer = OutputAnalyzer(llm_judge_scorer=_NanJudge())
    result = _quiet(
        analyzer.analyze_llm_judge,
        ["The programme offers a monthly stipend."] * 30 + [float("nan")] * 2,
        ["a plain short answer"] * 30 + [float("nan")] * 2,
        "men",
        "women",
    )
    assert result.assessed is True
    assert result.not_assessed_reason is None
    assert result.group_a_value == pytest.approx(0.9)
    assert result.group_b_value == pytest.approx(0.6)
    assert result.p_value == pytest.approx(1.6852981948926427e-14, rel=1e-6)
    assert result.is_significant is True
    assert (result.n_scored_a, result.n_scored_b) == (30, 30)
    assert (result.n_supplied_a, result.n_supplied_b) == (32, 32)

    # And exactly at the bound it still measures: the refusal is strictly above it.
    at_bound = _quiet(
        analyzer.analyze_llm_judge,
        ["The programme offers a monthly stipend."] * 10 + [float("nan")] * 10,
        ["a plain short answer"] * 10 + [float("nan")] * 10,
        "men",
        "women",
    )
    assert at_bound.assessed is True
    assert (at_bound.n_scored_a, at_bound.n_scored_b) == (10, 10)


# ===========================================================================
# 5. analyze_all: a real negative is not a non-test
# ===========================================================================


def _interleaved_equal_refusal_rates() -> tuple:
    """Four different texts per group, refusals INTERLEAVED.

    Both groups refuse the same NUMBER of prompts and not the same ones, so the
    refusal score arrays are [1, 0, 1, 0] against [0, 1, 0, 1]: np.array_equal is
    False, no short circuit runs, and a real Mann-Whitney test returns p exactly
    1.0 because the two group means are equal. That is the strongest negative
    result this analyzer can reach on the metric that matters most.
    """
    texts_a = [_EN_REFUSALS[0], _EN_ANSWERS[0], _EN_REFUSALS[1], _EN_ANSWERS[1]]
    texts_b = [_EN_ANSWERS[2], _EN_REFUSALS[2], _EN_ANSWERS[3], _EN_REFUSALS[3]]
    return texts_a, texts_b


def test_a_tested_negative_is_not_republished_as_no_comparison_performed() -> None:
    """The two doors, told apart.

    Measured before: analyze_refusal_rate called directly gave p_value 1.0, delta 0.0,
    effect_size 0.0, group_a_value 0.5, group_b_value 0.5, assessed True, and
    ``_is_zero_power`` on that REAL negative answered True. Through analyze_all the
    row came back p_value None, is_significant None, zero_power True,
    not_assessed_reason 'identical_scores_nothing_to_test', and the call warned
    "the scorer read the same value for every text in both groups", which is false
    of scores [1, 0, 1, 0] and [0, 1, 0, 1].
    """
    texts_a, texts_b = _interleaved_equal_refusal_rates()
    scorer = S.RefusalScorer()
    scores_a = _quiet(scorer.score_batch, texts_a)
    scores_b = _quiet(scorer.score_batch, texts_b)
    # The fixture really is the COMPUTED door, not the short circuit.
    assert list(scores_a) == [1.0, 0.0, 1.0, 0.0]
    assert list(scores_b) == [0.0, 1.0, 0.0, 1.0]
    assert np.array_equal(scores_a, scores_b) is False

    analyzer = OutputAnalyzer()
    direct = _quiet(analyzer.analyze_refusal_rate, texts_a, texts_b, "men", "women")
    # A real test, whose p happens to be exactly 1.0 because the means are equal.
    assert direct.p_value == 1.0
    assert direct.delta == 0.0 and direct.effect_size == 0.0
    assert direct.group_a_value == direct.group_b_value == 0.5
    assert direct.assessed is True
    assert direct.metadata.parameters["p_value_source"] == "mannwhitneyu"
    assert _is_zero_power(direct) is False

    rows = _quiet(analyzer.analyze_all, texts_a, texts_b, "men", "women")
    by_metric = {r.metric: r for r in rows}
    refusal = by_metric["refusal_rate"]
    assert refusal.zero_power is False
    assert refusal.not_assessed_reason is None
    assert refusal.p_value is not None and math.isfinite(refusal.p_value)
    assert refusal.is_significant is False
    assert refusal.group_a_value == refusal.group_b_value == 0.5

    # It is IN the family, and the family counts it.
    tested = [r for r in rows if r.p_value is not None]
    assert "refusal_rate" in {r.metric for r in tested}
    assert refusal.metadata.parameters["n_tests_in_family"] == len(tested)
    assert len(tested) >= 4, [r.metric for r in tested]


def test_control_a_real_non_test_is_still_excluded_and_says_something_true() -> None:
    """The under-correction control: the FORCED door must stay shut.

    A fix that simply stopped excluding anything would pass the pin above and undo
    the BGL5 work. On the same fixture, the metrics whose two score arrays ARE
    identical element for element still leave the family, and the sentence they
    carry no longer claims the scorer read one value for every text.
    """
    texts_a, texts_b = _interleaved_equal_refusal_rates()
    analyzer = OutputAnalyzer()
    rows, caught = _caught(analyzer.analyze_all, texts_a, texts_b, "men", "women")

    excluded = [r for r in rows if r.zero_power]
    assert {"toxicity", "stereotype", "representation"} <= {r.metric for r in excluded}
    for row in excluded:
        assert row.p_value is None and row.is_significant is None, row.metric
        assert row.not_assessed_reason == "identical_scores_nothing_to_test", row.metric
        assert row.metadata.parameters["zero_power_raw_p"] == 1.0, row.metric
        assert row.metadata.parameters["p_value_source"] == "identical_samples_short_circuit", (
            row.metric
        )
        assert "nothing to compare" in row.zero_power_reason, row.metric

    # THE SENTENCE A READER SEES IS TRUE OF THE DATA.
    assert "same value for every text" not in ZERO_POWER_REASON
    assert "value for value" in ZERO_POWER_REASON
    for row in excluded:
        assert "same value for every text" not in row.zero_power_reason, row.metric
    messages = [str(w.message) for w in caught if w.category is RuntimeWarning]
    zp_messages = [m for m in messages if "performed NO comparison" in m]
    assert zp_messages, messages
    assert all("same value for every text" not in m for m in zp_messages), zp_messages


def test_the_zero_power_predicate_still_reads_a_row_that_carries_no_provenance() -> None:
    """A row built by hand, or restored from a store written before the
    provenance key existed, keeps the answer the numeric signature has always
    given. The provenance decides it only when there IS provenance."""
    from vfairness.llm.output_analysis import OutputAnalysisResult

    row = OutputAnalysisResult(
        group_a="a",
        group_b="b",
        metric="toxicity",
        group_a_value=np.float32(0.0),
        group_b_value=np.float32(0.0),
        delta=np.float32(0.0),
        effect_size=np.int64(0),
        p_value=np.float32(1.0),
        is_significant=False,
        sample_size=4,
    )
    assert row.metadata.parameters.get("p_value_source") is None
    assert _is_zero_power(row) is True


def test_the_pulse_power_report_keeps_the_tested_negative_and_says_something_true() -> None:
    """THE READER SURFACE, which is where this defect was doing its damage.

    ``operations/pulse/orchestrator.py`` decides this twice: it RELAYS the label
    the analyzer writes, and it also carries its own nested copy of the numeric
    signature. The analyzer's fix alone therefore moved the over-correction one
    layer out rather than closing it: a row whose p of exactly 1.0 came from a real
    test is not marked by the analyzer any more, so the orchestrator's own
    signature test caught it instead, excluded it from the family and wrote the
    same false sentence into ``zero_power_reason`` and the metric's name into
    ``familyPower.excludedZeroPower``.

    Measured on six responses per group, three refusals and three answers in EACH
    group and never the same prompt refused: refusal_rate comes back p 1.0 with a
    family-adjusted p, it is NOT in excludedZeroPower, familySize counts it, and
    the three metrics whose score arrays really are identical are excluded with a
    sentence that describes what happened to them.
    """
    from vfairness.operations.pulse.orchestrator import _generative_pulse

    refusals = _EN_REFUSALS + ["I will not be able to help here.", "I refuse to answer this one."]
    answers = _EN_ANSWERS + [
        "Certainly, the documents you need are payslips, an ID and a bank statement.",
        "Of course, the process takes about four weeks from the application.",
    ]
    group_a = [refusals[0], answers[0], refusals[1], answers[1], refusals[2], answers[2]]
    group_b = [answers[3], refusals[3], answers[4], refusals[4], answers[5], refusals[5]]
    frame = pd.DataFrame({"group": ["A"] * 6 + ["B"] * 6, "output": group_a + group_b})
    data = _quiet(_generative_pulse, frame, {}, "output", ["group"], "hiring", "EU")["data"]
    generative = data["generative"]
    power = generative["familyPower"]
    metrics = [
        m
        for entry in generative["perAttribute"]
        for comp in entry["comparisons"]
        for m in comp["metrics"]
    ]
    by_metric = {m["metric"]: m for m in metrics}

    refusal = by_metric["refusal_rate"]
    assert refusal["p_value"] == 1.0
    assert refusal["zero_power"] is not True
    assert refusal["p_value_family_adjusted"] is not None
    assert "refusal_rate" not in power["excludedZeroPower"]

    in_family = {m["metric"] for m in metrics if m.get("p_value_family_adjusted") is not None}
    assert "refusal_rate" in in_family
    assert power["familySize"] == len(in_family)

    # The under-correction half: the FORCED door is still shut downstream too.
    assert power["excludedZeroPower"], "nothing was excluded; this fixture pins nothing"
    for name in power["excludedZeroPower"]:
        assert name not in in_family, name
        assert by_metric[name]["zero_power_reason"], name

    # And not one reader-visible sentence claims the scorer read one value.
    for m in metrics:
        reason = m.get("zero_power_reason")
        if reason:
            assert "same value for every text" not in reason, m["metric"]
            assert reason == ZERO_POWER_REASON, m["metric"]


def test_one_gate_not_two_on_the_input_that_discriminates() -> None:
    """The sibling MODULE, which is how this gate drifted the first time.

    ``llm/decodingtrust.py`` imported the SHARE and the THRESHOLD from
    ``llm/scorers.py`` and kept its own predicate over them, so the moment the
    scorers' gate gained its second half the two modules answered differently
    again. The existing pin of that invariant compares them on a fixture BOTH
    refuse (a Japanese refusal whose only ASCII letters are "AI", share 0.095), so
    it could not see the drift. Measured before this delegation, on the Japanese
    refusal plus a support URL: ``scorers.lexicon_can_read`` False,
    ``decodingtrust._lexicon_can_read`` True.
    """
    from vfairness.llm import decodingtrust as D

    for carrier, _share in _ASCII_CARRIERS.values():
        text = _JA_REFUSAL + carrier
        assert S.lexicon_can_read(text) is False, carrier
        assert D._lexicon_can_read(text) is S.lexicon_can_read(text), carrier
    # And the control, on both sides: English prose is read by both.
    assert D._lexicon_can_read(_EN_ANSWER) is True
    assert S.lexicon_can_read(_EN_ANSWER) is True
