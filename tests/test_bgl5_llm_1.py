"""BGL5 batch A-llm-1: the ten overturned grades, closed and pinned.

An independent audit attacked the twenty six grades of batch A-llm-1 and
overturned ten of them. Every one of those overturns was reproduced by
execution before anything was changed, and this file pins the corrected
behaviour with the measured value beside it.

TWO ROOT CAUSES ACCOUNT FOR NINE OF THE TEN:

1. A TYPE CHECK STANDING IN FOR A VALUE CHECK. ``isinstance(x, (int, float))``
   accepts ``nan``, ``inf`` and ``True``, and rejects ``np.float32``; a pair of
   comparisons written as ``x <= 0 or x >= 1`` lets ``nan`` through because both
   halves are False. So a guard whose message says "not a number on the 0 to 10
   scale" never looked at the scale, and a guard saying "min_runs must be at
   least 2" never looked at anything but the ordering. Fixed at the root in
   ``_reply_number_or_none`` / ``_judge_rating_or_none`` (scorers) and
   ``_require_open_unit_interval`` (nondeterminism), not per call site.

2. A READABILITY GATE THAT IS REALLY A SCRIPT GATE. ``RefusalScorer``'s
   readability test is ``[a-z][a-z\\-']*``, so a Chinese refusal correctly
   scored NaN and a French one scored a MEASURED 0.0, which on that scale is the
   categorical claim "this response did not refuse". The fix does not refuse more
   languages: it stops claiming a measurement it cannot take, and says which of
   the two unscorable cases the caller is in.

EVERY TEST HERE HAS BEEN SABOTAGED: the defect was put back, the named test was
confirmed RED, and the source was restored and diffed byte-identical. The
controls are as load-bearing as the refusals, because a fix that refuses
everything passes every refusal assertion and destroys the library, so each
control asserts the ACTUAL measured number and not merely "not NaN".

ONE OVERTURN IS NOT CLOSED HERE and is not pretended to be: the
``ContextualStereotypeScorer`` coverage disclosure reaches no consumer, because
nothing in ``OutputAnalysisResult`` carries it. That fix belongs in
``llm/output_analysis.py``, which this batch does not own; it stays a strict
xfail in tests/test_bgl4_llm_1.py with the file it needs named in its reason.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vfairness.llm.scorers as S
from vfairness.llm.nondeterminism import NonDeterminismAnalyzer, noise_floor_from_runs

_FRENCH_REFUSAL = "Je ne peux pas vous aider avec cette demande."
_GERMAN_REFUSAL = "Ich kann Ihnen dabei nicht helfen."
_CHINESE_REFUSAL = "我不能帮助你处理这个请求。"
_JUDGE_DIMENSIONS = ("helpfulness", "fairness", "specificity", "completeness")


@pytest.fixture(autouse=True)
def _restore_the_once_per_process_warning_flags():
    """Put the module level warn-once booleans back, as the batch path pins do.

    ``_warn_sidecar_down`` and ``_warn_sidecar_bad_reply`` latch for the whole
    process by design (a 10k-text run must not emit 10k identical lines), so a
    test that spends that budget has to return it or the next test observes no
    warning and fails for a reason that has nothing to do with it.
    """
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


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        return fn(*args, **kwargs), caught


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ==========================================================================
# OVERTURN 1. NonDeterminismAnalyzer.required_runs
# The 2026-09-27 guard refuses min_runs of 0, 1 and -5 and goes red when
# disabled. `nan < 2` is False, so a float walked past it.
# ==========================================================================


@pytest.mark.parametrize(
    "min_runs", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "minus_inf"]
)
def test_a_non_finite_minimum_is_refused_like_a_minimum_below_two(min_runs):
    """Measured before: ``NonDeterminismAnalyzer('llm', min_runs=float('nan'))``
    was ACCEPTED and ``required_runs()`` answered nan; inf was accepted and
    answered inf. After: both raise, beside the integer cases."""
    with pytest.raises(ValueError, match="min_runs must be at least 2"):
        NonDeterminismAnalyzer("llm", min_runs=min_runs)


def test_a_non_finite_minimum_cannot_silence_the_thin_floor_disclosure():
    """The whole defect at the public entry point.

    Measured before, on two runs per variant with metrics=['response_length']:
    min_runs=nan gave variant states ['measured', 'measured'], limitations 0 and
    zero warnings, where the default gives 'measured_with_limitation' twice, two
    limitations and four 'recommended minimum is 25' warnings. That is character
    for character the min_runs=0 outcome the guard was added to refuse.
    """
    runs = {"a": ["one two three four", "one two three"], "b": ["five six", "five six seven"]}
    for min_runs in (0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_runs must be at least 2"):
            noise_floor_from_runs(runs, metrics=["response_length"], min_runs=min_runs)


def test_control_the_recommended_run_count_and_the_thin_floor_warning_are_unchanged():
    """OVER-CORRECTION CONTROL, with the numbers.

    A constructor that refused every minimum would satisfy the two assertions
    above. Measured after the fix: 25 for 'llm', 50 for 'agent', 2 for
    min_runs=2, 30 for min_runs=30, and the DEFAULT path on two runs per variant
    still reports 'measured_with_limitation' for both variants with exactly two
    limitations and the 'recommended minimum is 25' warning.
    """
    assert NonDeterminismAnalyzer("llm").required_runs() == 25
    assert NonDeterminismAnalyzer("agent").required_runs() == 50
    assert NonDeterminismAnalyzer("llm", min_runs=2).required_runs() == 2
    assert NonDeterminismAnalyzer("llm", min_runs=30).required_runs() == 30

    runs = {"a": ["one two three four", "one two three"], "b": ["five six", "five six seven"]}
    out, caught = _caught(noise_floor_from_runs, runs, metrics=["response_length"])
    assert [v["state"] for v in out["variants"].values()] == [
        "measured_with_limitation",
        "measured_with_limitation",
    ]
    assert len(out["limitations"]) == 2
    assert any("recommended minimum is 25" in str(w.message) for w in caught)


# ==========================================================================
# OVERTURN 2. NonDeterminismAnalyzer.equivalence_test
# It validated sesoi and not alpha, while BOTH sibling methods in the same
# class refuse an alpha outside (0, 1).
# ==========================================================================


@pytest.mark.parametrize(
    "alpha",
    [0.0, 1.5, -1.0, float("nan"), float("inf")],
    ids=["zero", "above_one", "negative", "nan", "inf"],
)
def test_an_alpha_outside_the_unit_interval_cannot_confirm_fairness(alpha):
    """Measured before, on one fixed pair of N(0.5, 0.1) samples, n=20, seed 7:
    alpha=1.5 was ACCEPTED and answered 'fairness_confirmed' on a TOST p of
    0.487941, with the data unchanged and nothing warning, because
    ``p_tost < alpha`` is true for every p once alpha exceeds 1. alpha=0.0,
    alpha=-1.0 and alpha=nan were accepted too."""
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.5, 0.1, 20), rng.normal(0.5, 0.1, 20)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        NonDeterminismAnalyzer("llm").equivalence_test(a, b, alpha=alpha)


@pytest.mark.parametrize(
    "sesoi",
    [float("inf"), float("nan"), 0.0, -1.0],
    ids=["inf", "nan", "zero", "negative"],
)
def test_a_sesoi_that_is_not_a_finite_positive_bound_cannot_confirm_fairness(sesoi):
    """THE SAME SHAPE, IN THE PARAMETER THE ALPHA FIX SAT NEXT TO.

    Found by the B4 grade audit on 2026-09-29, one guard above the alpha guard
    the test before this one pins. It read ``if sesoi <= 0``, and `inf <= 0` and
    `nan <= 0` are both False, so both walked straight through.

    Measured before, on two groups drawn from N(0.20, 0.01) and N(0.80, 0.01),
    n=40, seed 0, which this same method calls 'bias_detected' at a real sesoi of
    0.05: sesoi=inf answered 'fairness_confirmed' with p_tost 0.0, no warning and
    no raise. An equivalence bound of infinity makes any observed difference
    equivalent by construction, so that is a clean bill of fairness on a
    difference of 0.6 between the groups.

    WHY THE ALPHA PIN COULD NOT CATCH IT: the two parameters share a
    precondition, and checking one of them is not checking the precondition. The
    control below is what stops a blanket refusal passing both.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.20, 0.01, 40)
    b = rng.normal(0.80, 0.01, 40)
    with pytest.raises(ValueError, match=r"SESOI must be a finite positive number"):
        NonDeterminismAnalyzer("llm").equivalence_test(a, b, sesoi=sesoi)


def test_control_a_real_sesoi_on_the_same_separated_groups_still_measures():
    """OVER-CORRECTION CONTROL for the guard above, with the number.

    The same groups the pin above uses, at a sesoi a caller would actually pass,
    must still reach a MEASURED verdict rather than being refused along with the
    infinities. Measured after the fix: 'bias_detected', p_tost 1.0.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.20, 0.01, 40)
    b = rng.normal(0.80, 0.01, 40)
    out = NonDeterminismAnalyzer("llm").equivalence_test(a, b, sesoi=0.05)
    assert out["verdict"] == "bias_detected"
    assert out["p_tost"] == pytest.approx(1.0, abs=1e-9)


def test_control_equivalence_test_still_grades_both_ends_of_its_scale():
    """OVER-CORRECTION CONTROL, with the numbers.

    Measured after the fix, seed 7: two N(0.5, 0.1) samples of n=20 answer
    'undetermined' with p_tost 0.487941, p_diff 0.550777 and d 0.1904; seed 11,
    N(0.9, 0.1) against N(0.2, 0.1) at n=30 answers 'bias_detected' with d
    7.6468 and p_diff 0.0. The alpha guard did not move either verdict.
    """
    analyzer = NonDeterminismAnalyzer("llm")
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.5, 0.1, 20), rng.normal(0.5, 0.1, 20)
    flat = analyzer.equivalence_test(a, b)
    assert flat["verdict"] == "undetermined"
    assert flat["p_tost"] == pytest.approx(0.487941, abs=1e-6)
    assert flat["p_diff"] == pytest.approx(0.550777, abs=1e-6)
    assert flat["effect_size"] == pytest.approx(0.1904, abs=1e-4)

    rng = np.random.default_rng(11)
    x, y = rng.normal(0.9, 0.1, 30), rng.normal(0.2, 0.1, 30)
    tilted = analyzer.equivalence_test(x, y)
    assert tilted["verdict"] == "bias_detected"
    assert tilted["effect_size"] == pytest.approx(7.6468, abs=1e-4)
    assert tilted["p_diff"] == pytest.approx(0.0, abs=1e-12)

    # The siblings still refuse their own out-of-range level, and still measure.
    assert analyzer.is_significant_after_offset(0.4, 0.1) is True
    assert analyzer.is_significant_after_offset(0.02, 0.5) is False
    offset = analyzer.compute_noise_offset(0.4, 0.1)
    assert offset["state"] == "measured"
    assert offset["systematic_offset"] == pytest.approx(0.302, abs=1e-3)


# ==========================================================================
# OVERTURN 3. LLMJudgeScorer.score and .score_batch
# The bool half of the guard was real. The guard tested the TYPE while its own
# message promised the RANGE, and min(max(v, 0), 10) clipped the rest.
# ==========================================================================


def _judge_answering(monkeypatch, reply):
    monkeypatch.setattr(S.LLMJudgeScorer, "_call_judge", lambda self, text: reply)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.LLMJudgeScorer("https://judge.example.com/v1/chat", model_name="judge-x")


def test_a_rating_outside_the_zero_to_ten_scale_is_not_a_rating(monkeypatch):
    """Measured before: a judge answering on a 0 to 100 rubric
    ({85, 92, 78, 88}) scored 1.0 with ZERO warnings, so every text scored 1.0
    and every group comparison read delta 0.0, p 1.0, is_significant False: a
    false clean bill. {-5, -5, -5, -5} scored 0.0, the other end, also silently.
    After: nan plus a PlaceholderScorerWarning naming all four dimensions."""
    for reply in (
        {"helpfulness": 85, "fairness": 92, "specificity": 78, "completeness": 88},
        dict.fromkeys(_JUDGE_DIMENSIONS, -5),
    ):
        scorer = _judge_answering(monkeypatch, reply)
        value, caught = _caught(scorer.score, "an ordinary generated reply about a loan")
        assert math.isnan(value), f"an out of range reply was scored {value}"
        placeholders = [w for w in caught if issubclass(w.category, S.PlaceholderScorerWarning)]
        assert placeholders, f"{reply} was refused in silence"
        assert "0 to 10 scale" in str(placeholders[0].message)


@pytest.mark.parametrize(
    "reply",
    [
        dict.fromkeys(_JUDGE_DIMENSIONS, float("inf")),
        {"helpfulness": float("nan"), "fairness": 5, "specificity": 5, "completeness": 5},
        dict.fromkeys(_JUDGE_DIMENSIONS, True),
    ],
    ids=["infinity", "nan_on_one", "booleans"],
)
def test_a_non_finite_rating_is_announced_rather_than_clipped(monkeypatch, reply):
    """``json.loads`` accepts the bare tokens NaN and Infinity by default, so a
    real reply body can carry them. Measured before: Infinity on all four
    dimensions scored 1.0 with zero warnings, and NaN on one dimension scored
    nan with zero warnings, a silent refusal where every other refusal in this
    method is announced. The bool row was already correct and is pinned here so
    the three cannot drift apart again."""
    scorer = _judge_answering(monkeypatch, reply)
    value, caught = _caught(scorer.score, "an ordinary generated reply")
    assert math.isnan(value), f"{reply} was scored {value}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


def test_the_judge_batch_path_does_not_carry_a_clipped_rating(monkeypatch):
    """``score_batch`` is ``np.array([self.score(t) for t in texts])``, so it
    inherits every gap ``score`` has, and it is the path every aggregate
    comparison reads. Measured before: array([1.0, 1.0]) from a judge answering
    0 to 100, no warning."""
    scorer = _judge_answering(
        monkeypatch,
        {"helpfulness": 85, "fairness": 92, "specificity": 78, "completeness": 88},
    )
    values, caught = _caught(scorer.score_batch, ["one reply", "another reply"])
    assert np.all(np.isnan(values)), f"the batch path scored {values}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


def test_control_the_judge_still_weights_a_complete_reply(monkeypatch):
    """OVER-CORRECTION CONTROL, with the numbers.

    Measured after the fix: {8, 4, 7, 6} is 0.58 exactly (weights 0.20 / 0.40 /
    0.20 / 0.20) with NO warning, both ends of the rubric are still accepted
    (all zeros -> 0.0, all tens -> 1.0, inclusive bounds), and the batch of two
    is array([0.58, 0.58]). A guard that refused an out of range rating by
    refusing every rating would fail every line here.
    """
    scorer = _judge_answering(
        monkeypatch, {"helpfulness": 8, "fairness": 4, "specificity": 7, "completeness": 6}
    )
    value, caught = _caught(scorer.score, "an ordinary generated reply about a loan")
    assert value == pytest.approx(0.58, abs=1e-9)
    assert not caught, [str(w.message) for w in caught]
    batch, batch_caught = _caught(scorer.score_batch, ["one reply", "another reply"])
    np.testing.assert_allclose(batch, [0.58, 0.58])
    assert not batch_caught

    floor = _judge_answering(monkeypatch, dict.fromkeys(_JUDGE_DIMENSIONS, 0))
    assert _quiet(floor.score, "a reply the judge rated zero throughout") == 0.0
    ceiling = _judge_answering(monkeypatch, dict.fromkeys(_JUDGE_DIMENSIONS, 10))
    assert _quiet(ceiling.score, "a reply the judge rated ten throughout") == 1.0


# ==========================================================================
# OVERTURN 4. RefusalScorer.score and .score_batch
# The could-not-check path was verified on CJK only. The gate is SCRIPT, not
# language, so a Latin-script refusal was "read" and scored a measured 0.0.
# ==========================================================================


@pytest.mark.parametrize("text", [_FRENCH_REFUSAL, _GERMAN_REFUSAL], ids=["french", "german"])
def test_a_refusal_the_english_lexicon_cannot_express_is_not_a_measured_zero(text):
    """Measured before: the French and German refusals both scored 0.0 with NO
    warning, while the identical refusal in Chinese scored nan with an
    UnscorableTextWarning. 0.0 on this scale is the categorical claim "this
    response did not refuse", about a response that refused."""
    value, caught = _caught(S.RefusalScorer().score, text)
    assert math.isnan(value), f"a real refusal scored {value}, the clean end of the scale"
    unscorable = [w for w in caught if issubclass(w.category, S.UnscorableTextWarning)]
    assert unscorable, "the refusal was silent, which is how the aggregate hid it"
    # It has to say WHICH case this is: words were read and they are not English.
    assert "no English function word" in str(unscorable[0].message)


def test_the_two_unscorable_cases_are_reported_as_different_cases():
    """A three-state answer is not enough if the two refusals are one message.

    ``no_readable_word`` (a non-Latin script, emoji, punctuation) and
    ``not_english`` (words read, not English) call for different remedies, so
    they are counted separately and warned separately. Measured: a batch of one
    Chinese and one French refusal emits both warnings, and the batch path
    reports both texts as NaN.
    """
    scorer = S.RefusalScorer()
    assert scorer._score_and_reason(_CHINESE_REFUSAL) == (None, "no_readable_word")
    assert scorer._score_and_reason(_FRENCH_REFUSAL) == (None, "not_english")

    values, caught = _caught(scorer.score_batch, [_CHINESE_REFUSAL, _FRENCH_REFUSAL])
    assert np.all(np.isnan(values))
    messages = [str(w.message) for w in caught]
    assert any("produced no readable word" in m for m in messages), messages
    assert any("no English function word" in m for m in messages), messages


def test_a_total_refusal_disparity_in_french_is_not_reported_as_no_disparity():
    """The defect at the public entry point, which is where it matters.

    Measured before, on 25 runs per variant where applicant_A is refused in
    French every single time and applicant_B is helped in French every single
    time: refusal_rate state 'measured', observed 0.0, exceeds_noise False,
    systematic_offset 0.0, n_excluded_reference 0, scorer_quality 'production',
    limitations []. That is the outcome ``_score_or_none``'s docstring says it
    removed. After: state 'could_not_check' and two limitations, each naming
    0 of 25 runs measured for its variant.
    """
    runs = {
        "applicant_A": [_FRENCH_REFUSAL] * 25,
        "applicant_B": ["Voici comment proceder pour votre demande de pret."] * 25,
    }
    out, _caught_warnings = _caught(noise_floor_from_runs, runs, metrics=["refusal_rate"])
    comparison = out["metrics"]["refusal_rate"]["comparisons"][0]
    assert comparison["state"] == "could_not_check", (
        f"a total refusal disparity was reported as observed={comparison['observed']} / "
        f"exceeds_noise={comparison['exceeds_noise']}"
    )
    assert comparison["observed"] is None and comparison["exceeds_noise"] is None
    assert len(out["limitations"]) == 2
    assert all("NOT measured" in limitation for limitation in out["limitations"])


def test_control_the_refusal_scorer_still_reads_the_english_it_is_written_for():
    """OVER-CORRECTION CONTROL, with the numbers.

    A gate that refused anything it was not certain about would pass every
    assertion above and delete the scorer. Measured after the fix: the hard
    refusals still score 1.0, readable English non-refusals still score a
    MEASURED 0.0 with no warning, a blank generation keeps its measured 0.0,
    invented words ("zzzz qqqq") keep theirs because nothing says they are
    another language, and a French sentence that DOES carry an English refusal
    phrase is still read, at 1.0.
    """
    scorer = S.RefusalScorer()
    for text in ("I cannot help with that request.", "As an AI, I must decline.", "I won't."):
        value, caught = _caught(scorer.score, text)
        assert value >= 0.5, f"{text!r} scored {value}"
        assert not caught, [str(w.message) for w in caught]

    for text in (
        "I can help with that.",
        "Here is how to do it.",
        "Cannot be determined from the context given.",
        "Yes, I am able to walk you through the whole process.",
        "zzzz qqqq",
        "",
        "   ",
    ):
        value, caught = _caught(scorer.score, text)
        assert value == 0.0, f"{text!r} scored {value} instead of a measured absence"
        assert not caught, [str(w.message) for w in caught]

    # The score is computed BEFORE the language test, so a detected refusal is
    # never discarded by it.
    assert _quiet(scorer.score, "Je ne peux pas. I cannot help.") == 1.0

    # And the same total disparity in ENGLISH is still measured and still fires.
    english = {
        "applicant_A": ["I cannot help with that request."] * 25,
        "applicant_B": ["Here is how to proceed with your loan application."] * 25,
    }
    out = _quiet(noise_floor_from_runs, english, metrics=["refusal_rate"])
    comparison = out["metrics"]["refusal_rate"]["comparisons"][0]
    assert comparison["state"] == "measured"
    assert comparison["observed"] == 1.0
    assert comparison["exceeds_noise"] is True


# ==========================================================================
# OVERTURN 5. SidecarSentimentScorer.score_batch / SidecarToxicityScorer.score_batch
# The three pins that named these methods executed the bridge-is-None outage
# only: coverage showed `np.array([float(x) for x in r])` never ran.
# ==========================================================================


class _AnsweringBridge:
    def __init__(self, reply):
        self.reply = reply

    def call(self, op, **payload):
        return self.reply


def _bridge_answering(monkeypatch, reply):
    monkeypatch.setattr(S._SidecarBridge, "get", classmethod(lambda cls: _AnsweringBridge(reply)))
    monkeypatch.setattr(S, "_sidecar_bad_reply_warned", False)


_SIDECAR_CLASSES = [S.SidecarSentimentScorer, S.SidecarToxicityScorer]
_SIDECAR_IDS = ["sentiment", "toxicity"]
_THREE_TEXTS = ["a lovely person", "a vile idiot", "an ordinary sentence"]


@pytest.mark.parametrize("cls", _SIDECAR_CLASSES, ids=_SIDECAR_IDS)
def test_a_boolean_reply_is_not_a_batch_of_scores(monkeypatch, cls):
    """``score()`` guards ``not isinstance(r, bool)`` and answers nan plus a
    PlaceholderScorerWarning. ``score_batch`` checked only ``isinstance(r,
    list)`` and then called float() on each element, so [True, False, True]
    measured array([1., 0., 1.]) with ZERO warnings: on toxicity that is
    "maximally toxic, not toxic, maximally toxic" invented from three bools."""
    _bridge_answering(monkeypatch, [True, False, True])
    values, caught = _caught(cls().score_batch, _THREE_TEXTS)
    assert np.all(np.isnan(values)), f"booleans became scores {values}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


@pytest.mark.parametrize("cls", _SIDECAR_CLASSES, ids=_SIDECAR_IDS)
@pytest.mark.parametrize(
    "reply", [[0.9], [], [0.1, 0.2, 0.3, 0.4, 0.5]], ids=["short", "empty", "long"]
)
def test_a_reply_of_the_wrong_length_is_not_a_batch_of_scores(monkeypatch, cls, reply):
    """The reply's LENGTH was never checked. Measured before, on three texts:
    [0.9] returned a one element array, [] an empty array and a five element
    reply five values, all with zero warnings. A caller that slices the result
    by group index then reads one text's score under another text's label."""
    _bridge_answering(monkeypatch, reply)
    values, caught = _caught(cls().score_batch, _THREE_TEXTS)
    assert len(values) == len(_THREE_TEXTS), (
        f"{len(_THREE_TEXTS)} texts produced {len(values)} values, silently"
    )
    assert np.all(np.isnan(values)), f"a reply of the wrong length scored {values}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


@pytest.mark.parametrize("cls", _SIDECAR_CLASSES, ids=_SIDECAR_IDS)
def test_one_unusable_value_refuses_that_text_and_keeps_the_others(monkeypatch, cls):
    """A reply of the RIGHT length is refused per ELEMENT, not wholesale.

    Element i belongs to text i, so one bad value invalidates one text and
    discarding the other two would be this library's own defect running
    backwards. Measured: [0.1, True, 0.3] -> array([0.1, nan, 0.3]) with the
    warning naming 1 of 3.
    """
    _bridge_answering(monkeypatch, [0.1, True, 0.3])
    values, caught = _caught(cls().score_batch, _THREE_TEXTS)
    assert values[0] == pytest.approx(0.1) and values[2] == pytest.approx(0.3)
    assert math.isnan(values[1])
    messages = [str(w.message) for w in caught]
    assert any("1 of 3 value(s) is not a finite number" in m for m in messages), messages


@pytest.mark.parametrize("cls", _SIDECAR_CLASSES, ids=_SIDECAR_IDS)
def test_control_a_healthy_sidecar_batch_is_still_measured(monkeypatch, cls):
    """OVER-CORRECTION CONTROL, with the numbers.

    Measured after the fix: a reply of [0.1, 0.2, 0.3] for three texts returns
    array([0.1, 0.2, 0.3]) with NO warning at all, and the single-text path
    still returns 0.42 for a reply of 0.42. A validator that refused every
    reply would pass every assertion above.
    """
    _bridge_answering(monkeypatch, [0.1, 0.2, 0.3])
    values, caught = _caught(cls().score_batch, _THREE_TEXTS)
    np.testing.assert_allclose(values, [0.1, 0.2, 0.3])
    assert not caught, [str(w.message) for w in caught]

    _bridge_answering(monkeypatch, 0.42)
    single, single_caught = _caught(cls().score, "a lovely person")
    assert single == pytest.approx(0.42)
    assert not single_caught, [str(w.message) for w in single_caught]

    # An empty batch is still an empty array and not a scalar zero.
    _bridge_answering(monkeypatch, [])
    assert np.asarray(_quiet(cls().score_batch, [])).shape == (0,)


# ==========================================================================
# OVERTURN 6. ContextualStereotypeScorer.score, the half this batch owns.
# An NLI reply outside [0, 1] was clipped to an end of the scale and POSITIVELY
# reported as a two-stage reading. (The other half of this overturn, the
# disclosure reaching a consumer, needs llm/output_analysis.py: see the module
# docstring.)
# ==========================================================================


def test_an_nli_reading_outside_zero_to_one_is_not_a_two_stage_measurement(monkeypatch):
    """Measured before, on "The candidate presented a clear plan for the
    quarter." with a stubbed bridge: an NLI reply of -1.0 scored 0.0 and an NLI
    reply of 5.0 scored 1.0, BOTH with stage_coverage {'wordlist': True, 'nli':
    True} and no warning. 0.0 there is "no stereotype" and 1.0 is "certainly a
    stereotype", from a reply that is not a probability at all."""
    text = "The candidate presented a clear plan for the quarter."
    for reply in (-1.0, 5.0, True):
        _bridge_answering(monkeypatch, reply)
        scorer = S.ContextualStereotypeScorer()
        _value, caught = _caught(scorer.score, text)
        coverage = _quiet(scorer.stage_coverage, text)
        assert coverage == {"wordlist": True, "nli": False}, (
            f"an NLI reply of {reply} was reported as a two-stage reading"
        )
        assert any(issubclass(w.category, S.PartialCoverageWarning) for w in caught), (
            "a single-stage value was returned without the coverage disclosure"
        )


def test_control_a_two_stage_fusion_still_answers_without_a_coverage_warning(monkeypatch):
    """OVER-CORRECTION CONTROL, with the numbers.

    Measured after the fix: an NLI reading of 0.77 on an ordinary sentence still
    fuses to 0.8 with stage_coverage {'wordlist': True, 'nli': True} and NO
    coverage warning, and a lexical stereotype still scores 1.0. A range check
    that refused every reading would leave stage 2 permanently dead.
    """
    text = "The candidate presented a clear plan for the quarter."
    _bridge_answering(monkeypatch, 0.77)
    scorer = S.ContextualStereotypeScorer()
    value, caught = _caught(scorer.score, text)
    assert value == pytest.approx(0.8, abs=1e-9)
    assert _quiet(scorer.stage_coverage, text) == {"wordlist": True, "nli": True}
    assert not caught, [str(w.message) for w in caught]
    assert _quiet(scorer.score, "women are naturally more nurturing") == 1.0


# ==========================================================================
# OVERTURN 7. TextScorer, regraded rather than fixed.
# Claimed SEMI-PROVEN on the stated evidence that it was "executed for the
# first time by any test". It cannot be executed.
# ==========================================================================


def test_the_text_scorer_protocol_has_no_behaviour_to_grade():
    """A typing Protocol with `...` bodies. There is no degenerate input for it.

    `TextScorer()` raises TypeError, and `runtime_checkable` makes isinstance a
    METHOD PRESENCE test only, so a scorer returning the string 'not a number'
    satisfies it. The grade's named pin asserted `TextScorer._is_protocol` and
    nothing else, so it could not have disagreed about any behaviour. The honest
    class is NOT A MEASUREMENT, which is what src/vfairness/_registry.py already
    records for it, and the class docstring now says so where a reader meets it.
    """
    with pytest.raises(TypeError, match="Protocols cannot be instantiated"):
        S.TextScorer()

    class _ReturnsNonsense:
        def score(self, text):
            return "not a number"

        def score_batch(self, texts):
            return None

    assert isinstance(_ReturnsNonsense(), S.TextScorer), (
        "runtime_checkable checks method presence only; this is the point, not a bug"
    )
    assert "NOT A MEASUREMENT" in (S.TextScorer.__doc__ or ""), (
        "the regrade has to be legible where a reader meets the class, not only in a report"
    )
