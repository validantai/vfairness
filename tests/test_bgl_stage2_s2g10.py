"""Beta Go-Live stage 2, group s2g10: the LLM text scorers stop inventing a score.

Six findings, one file (``src/vfairness/llm/scorers.py``), one shape: a text the
scorer could not read at all was given a neutral default, and that default was
then averaged, significance-tested, corrected for multiple comparisons and
reported as a measurement.

What was measured BEFORE the fix, at the public entry points:

* ``OutputAnalyzer().analyze_all`` on two groups of lexicon-free gibberish
  returned ten clean "no disparity" findings: framing 0.575/0.575, regard
  0.0/0.0, semantic_quality 0.142/0.146, each ``assessed=True``,
  ``not_assessed_reason=None``, from text nobody could read.
* ``FramingScorer().score`` returned the CONSTANT 0.575 for every text without
  a marker, including "!!!" and "El gato come pescado.".
* ``KeywordRegardScorer().score("a respected but lazy person")`` (a measured
  0.0 on a signed scale) was byte-identical to
  ``score("the weather today is mild")`` (nothing read).
* ``SemanticQualityScorer`` ranked never-read text ABOVE explicitly low-tier
  advice at equal word count, 0.216 against 0.091, and the analyzer graded that
  gap significant at p = 0.0013.
* ``noise_floor_from_runs`` reported ``state="measured"``, ``observed=0.0``,
  ``exceeds_noise=False`` for abusive text in Chinese, which the ASCII-only
  tokeniser produced zero tokens from.
* ``judge_is_subject`` answered ``False`` both for "two identities compared,
  they differ" and for "no subject supplied, nothing compared".

Every test below asserts the third state at the PUBLIC entry, and every test
carries a CONTROL on healthy data, because a scorer that refuses everything is a
worse defect than one that invents a number and passes any test that only
exercises the degenerate case.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness import OutputAnalyzer, noise_floor_from_runs
from vfairness.llm import judge_is_subject
from vfairness.llm.scorers import (
    FramingScorer,
    KeywordRegardScorer,
    KeywordToxicityScorer,
    LLMJudgeScorer,
    SemanticQualityScorer,
    UnscorableTextWarning,
    _lexicon_tokens,
)

# --------------------------------------------------------------------------
# Fixtures. Each one is asserted to REACH the branch it is supposed to test:
# in stage 1 the single green sabotage was a fixture that never got there.
# --------------------------------------------------------------------------

GIBBERISH_A = [
    "zq frmbl xktwn dplvgh ssrmt",
    "vv qlmzt brdhn kkfpw",
    "xnth qrrl vbbz mmkd tprl",
    "ggwn zzrt plmq vvkd",
    "hhrn qqlw zzmt bbfd",
    "mmkd tprl ssrmt xnth",
]
GIBBERISH_B = [t + " q" for t in GIBBERISH_A]

# Healthy data: a real, findable framing / regard / quality disparity, in the
# shape these scorers exist to read (advice about a person). All eleven metrics
# must be able to answer on it, which is what makes it a control.
HEALTHY_A = [
    "You should apply to a top-tier and highly selective programme. It is clear "
    "that this respected and accomplished candidate is outstanding, for example "
    "in her 3 published papers.",
    "I recommend you pursue the rigorous advanced track. This skilled and "
    "dedicated professional has demonstrated a remarkable record, such as 4 "
    "completed projects in Zurich.",
] * 8
HEALTHY_B = [
    "You should perhaps consider a basic local part-time course. It is possible "
    "that this unreliable and lazy applicant is questionable, for example in his "
    "3 unexplained gaps.",
    "I recommend you look into the simple introductory option. This incompetent "
    "and dishonest applicant has arguably a problematic record, such as 4 "
    "disputed claims in Zurich.",
] * 8

CJK_BENIGN = "这位候选人很优秀"  # "this candidate is excellent"
CJK_ABUSIVE = "这个候选人又蠢又恶心"  # abusive, same meaning as the English below
EN_ABUSIVE = "You are a stupid worthless idiot and a disgusting loser."
EN_BENIGN = "This candidate is excellent and the panel was pleased."


def _quiet_analyze_all(analyzer, texts_a, texts_b, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {r.metric: r for r in analyzer.analyze_all(texts_a, texts_b, **kw)}


def _lexicon_analyzer() -> OutputAnalyzer:
    """An analyzer whose regard scorer is the LEXICON one, which is this file's subject.

    Every regard fixture here is built from the ABSENCE of a lexicon term, and
    test_the_fixtures_reach_the_branches_these_tests_claim_to_test checks the
    gibberish against KeywordRegardScorer's own POSITIVE_REGARD and
    NEGATIVE_REGARD sets to prove it. "No regard term was read" is a state only a
    lexicon scorer can be in.

    OutputAnalyzer's DEFAULT regard scorer is TransformerRegardScorer whenever
    transformers is installed, and sasha/regardv3 reads gibberish and returns a
    real polarity, which IS a measurement of what the model saw rather than a
    neutral substitute for one. Measured 2026-09-28 on a machine that has
    transformers: analyze_regard(GIBBERISH_A, GIBBERISH_B) came back assessed
    True, so these three refusals were being asked of a scorer this file is not
    about, and they failed for that reason and no other. Naming the scorer makes
    the pin say what it means on either machine.
    """
    return OutputAnalyzer(regard_scorer=KeywordRegardScorer())


def test_the_fixtures_reach_the_branches_these_tests_claim_to_test():
    """A fixture that never reaches the guard makes every sabotage go green."""
    # Gibberish is ASCII, so the tokeniser DOES read words from it. That is
    # deliberate: it isolates "no lexicon term" from "no readable token", which
    # are two different refusals in this file.
    assert _lexicon_tokens(GIBBERISH_A[0]), "the gibberish must still tokenise"
    for text in GIBBERISH_A + GIBBERISH_B:
        lower = text.lower()
        assert not (set(_lexicon_tokens(lower)) & KeywordRegardScorer.POSITIVE_REGARD)
        assert not (set(_lexicon_tokens(lower)) & KeywordRegardScorer.NEGATIVE_REGARD)
        assert not any(m in lower for m in FramingScorer._HEDGING)
        assert not any(m in lower for m in FramingScorer._CERTAINTY)
        assert not any(m in lower for m in FramingScorer._POSITIVE_EVALUATIVE)
        assert not any(m in lower for m in FramingScorer._NEGATIVE_EVALUATIVE)
    # The CJK fixtures are the OTHER refusal: zero tokens at all.
    assert _lexicon_tokens(CJK_ABUSIVE) == [], "the CJK fixture must produce no token"
    assert _lexicon_tokens(EN_ABUSIVE), "the English control must produce tokens"
    # And the healthy corpus must carry the vocabulary the controls rely on.
    joined = " ".join(HEALTHY_A + HEALTHY_B).lower()
    assert any(m in joined for m in FramingScorer._POSITIVE_EVALUATIVE)
    assert any(m in joined for m in FramingScorer._HEDGING)


# ==========================================================================
# Finding 1: OutputAnalyzer.analyze_all
# ==========================================================================


def test_analyze_all_refuses_the_three_unreadable_metrics():
    """BEFORE: framing 0.575/0.575, regard 0.0/0.0, semantic_quality
    0.142/0.146, all ``assessed=True``, ``p_value=1.0``,
    ``not_assessed_reason=None``, and all three counted in the correction
    family. AFTER: three named could-not-checks."""
    rows = _quiet_analyze_all(_lexicon_analyzer(), GIBBERISH_A, GIBBERISH_B)

    for metric in ("framing", "regard", "semantic_quality"):
        row = rows[metric]
        assert row.assessed is False, f"{metric} still reports a measurement"
        assert row.not_assessed_reason == "non_finite_scores", metric
        assert row.group_a_value is None and row.group_b_value is None, metric
        assert row.delta is None and row.p_value is None, metric
        # The distinguishing property: a reader must be able to tell this from
        # a measured "no disparity" WITHOUT reading the source.
        assert row.is_significant is None, metric

    # A refusal must also leave the correction family, or it shrinks every
    # other metric's adjusted p-value on the strength of a test that never ran.
    family = {
        r.metadata.parameters.get("n_tests_in_family")
        for r in rows.values()
        if r.metadata.parameters.get("n_tests_in_family") is not None
    }
    # ONE, not "eleven minus the four refusals" (corrected 2026-09-28). That
    # subtraction assumed every row which does not refuse is a test, and on two
    # groups of gibberish it is not: six scorers read the SAME value for every
    # text on BOTH sides (toxicity 0.0, refusal_rate 0.0, stereotype 0.0,
    # representation 0.0, helpfulness 0.27, information_quality 0.25), so they
    # performed no comparison and leave the family with
    # not_assessed_reason='identical_scores_nothing_to_test' rather than spending
    # a test on a p of 1.0. Measured 2026-09-28: 4 refuse, 6 are zero-power, and
    # response_length is the only metric that compares anything (p 0.0183). A
    # family of 7 would divide that one real p-value by seven tests, six of which
    # could never have been discoveries, which is the same error this test exists
    # to catch pointed the other way.
    assert family == {1}, f"expected only the one metric that compared anything, got {family}"
    zero_power = {
        m for m, r in rows.items() if r.not_assessed_reason == "identical_scores_nothing_to_test"
    }
    assert len(zero_power) == 6, f"expected six constant-scored rows, got {sorted(zero_power)}"
    assert rows["response_length"].p_value is not None, (
        "the only row that compared anything has no p-value, so the family of 1 is "
        "empty and this assertion pins nothing"
    )


def test_analyze_all_still_measures_healthy_text():
    """CONTROL. The same eleven metrics on readable text with a real
    disparity: the three that now refuse must still answer here."""
    rows = _quiet_analyze_all(_lexicon_analyzer(), HEALTHY_A, HEALTHY_B)

    for metric in ("framing", "regard", "semantic_quality"):
        row = rows[metric]
        assert row.assessed is True, f"{metric} refuses text it can read"
        assert row.not_assessed_reason is None, metric
        assert row.group_a_value is not None and math.isfinite(row.group_a_value)
        assert row.group_b_value is not None and math.isfinite(row.group_b_value)
    # And the disparity is still FOUND, not merely reported as a number.
    assert rows["regard"].group_a_value > rows["regard"].group_b_value
    assert rows["framing"].group_a_value > rows["framing"].group_b_value


# ==========================================================================
# Finding 2: OutputAnalyzer.analyze_regard / KeywordRegardScorer
# ==========================================================================


def test_analyze_regard_refuses_two_lexicon_free_groups():
    """BEFORE: 0.0 / 0.0, delta 0.0, p 1.0, assessed True, reason None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _lexicon_analyzer().analyze_regard(GIBBERISH_A, GIBBERISH_B)

    assert result.assessed is False
    assert result.not_assessed_reason == "non_finite_scores"
    assert result.group_a_value is None and result.group_b_value is None
    assert result.delta is None and result.p_value is None


def test_analyze_regard_refuses_a_disparity_made_of_one_groups_vocabulary_absence():
    """BEFORE: delta 1.0 at p 0.0013, ``is_significant=True`` - a large
    significant finding manufactured because group B used no regard word."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _lexicon_analyzer().analyze_regard(
            ["a respected accomplished leader"] * 6,
            ["the weather today is mild"] * 6,
        )

    assert result.assessed is False
    assert result.is_significant is None
    assert result.delta is None


def test_keyword_regard_separates_a_measured_zero_from_an_unread_text():
    """The collision this closes was total: both returned 0.0."""
    scorer = KeywordRegardScorer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        balanced = scorer.score("a respected but lazy person")
    assert balanced == 0.0, "a measured neutral was swallowed by the refusal"
    assert not any(isinstance(w.message, UnscorableTextWarning) for w in caught), (
        "a text the lexicon read was reported as unscorable"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unread = scorer.score("the weather today is mild")
    assert math.isnan(unread), f"a text with no regard term scored {unread!r}"
    assert any(isinstance(w.message, UnscorableTextWarning) for w in caught), (
        "the refusal was silent, which is how the aggregate hid it before"
    )


def test_keyword_regard_control_the_poles_and_the_empty_string_are_unchanged():
    """CONTROL. The measurement itself must not have moved."""
    scorer = KeywordRegardScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert scorer.score("They are widely respected.") == 1.0
        assert scorer.score("a lazy dishonest criminal") == -1.0
        # The empty string REFUSES, and this line used to assert the opposite.
        # It was written against the sentiment sibling's carve-out and was left
        # behind when the same change closed that carve-out here: the scorer's
        # own docstring records the measurement that closed it, eight blank
        # generations against eight denigrating sentences coming back
        # delta=1.0, p=0.000138, is_significant=True, a significant regard
        # ADVANTAGE for the group that generated nothing at all. On a signed
        # scale 0.0 is measured NEUTRAL STANDING, not "no text".
        assert math.isnan(scorer.score(""))


# ==========================================================================
# Finding 3: noise_floor_from_runs / KeywordToxicityScorer
# ==========================================================================


def test_noise_floor_refuses_runs_the_tokeniser_cannot_read():
    """BEFORE: top level ``available=True, state='measured'`` and
    ``observed=0.0, disparity_noise_floor=0.0, exceeds_noise=False,
    systematic_offset=0.0`` for abusive text in Chinese, which produced zero
    tokens."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = noise_floor_from_runs(
            {"reference": [CJK_BENIGN] * 25, "other": [CJK_ABUSIVE] * 25},
            metrics=["toxicity"],
            random_state=0,
        )

    assert out["state"] != "measured"
    assert out["available"] is False
    metric = out["metrics"]["toxicity"]
    assert metric["state"] == "could_not_check"
    comparison = metric["comparisons"][0]
    assert comparison["state"] == "could_not_check"
    assert comparison["reason"]
    assert comparison["observed"] is None
    assert comparison["disparity_noise_floor"] is None
    assert comparison["exceeds_noise"] is None


def test_noise_floor_control_still_measures_text_the_scorer_can_read():
    """CONTROL. The identical comparison in English must still be measured,
    or the fix has traded a fabricated verdict for a refusal to answer."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = noise_floor_from_runs(
            {"reference": [EN_BENIGN] * 25, "other": [EN_ABUSIVE] * 25},
            metrics=["toxicity"],
            random_state=0,
        )

    assert out["state"] == "measured"
    metric = out["metrics"]["toxicity"]
    assert metric["state"] == "measured"
    comparison = metric["comparisons"][0]
    assert comparison["state"] == "measured"
    # Signed: the reference is the benign variant, so the gap is negative.
    assert comparison["observed"] is not None and abs(comparison["observed"]) > 0.0
    assert comparison["exceeds_noise"] is True


def test_keyword_toxicity_keeps_measured_absence_and_refuses_unreadable_text():
    """Two different facts that used to share one ``return 0.0``."""
    scorer = KeywordToxicityScorer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_and_clean = scorer.score("The committee reconvened on Tuesday.")
        abusive = scorer.score(EN_ABUSIVE)
        empty = scorer.score("")
    assert read_and_clean == 0.0, "absence of toxic words is a real reading"
    assert abusive == pytest.approx(0.5)
    assert empty == 0.0, "an empty generation really is non-toxic"
    assert not any(isinstance(w.message, UnscorableTextWarning) for w in caught)

    for unreadable in (CJK_ABUSIVE, "\U0001f600\U0001f600", "!!!"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = scorer.score(unreadable)
        assert math.isnan(value), f"{unreadable!r} scored {value!r}"
        assert any(isinstance(w.message, UnscorableTextWarning) for w in caught)


# ==========================================================================
# Finding 4: OutputAnalyzer.analyze_framing / FramingScorer
# ==========================================================================


def test_analyze_framing_refuses_marker_free_text():
    """BEFORE: 0.5750000000000001 on both sides, delta 0.0, p 1.0,
    assessed True."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = OutputAnalyzer().analyze_framing(["zzzz qqqq"] * 6, ["qqqq zzzz"] * 6)

    assert result.assessed is False
    assert result.not_assessed_reason == "non_finite_scores"
    assert result.group_a_value is None
    assert result.delta is None and result.p_value is None


def test_framing_scorer_no_longer_answers_a_constant_for_any_text_whatsoever():
    """The constant 0.575 was the whole score: 0.5 * 0.40 + 0.5 * 0.45 +
    1.0 * 0.15, i.e. both neutral defaults."""
    scorer = FramingScorer()
    for text in (
        "zzzz qqqq",
        "the doctor was competent and kind",
        "!!!",
        "a b c d e f g h",
        "She grew up in Lagos and now runs a bakery.",
        "1234 5678",
        "El gato come pescado.",
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = scorer.score(text)
        assert math.isnan(value), f"{text!r} scored {value!r}"
        assert any(isinstance(w.message, UnscorableTextWarning) for w in caught), text


def test_framing_control_a_text_with_markers_is_still_measured_and_ordered():
    """CONTROL. Framing must still be READ, and the ordering it exists to
    detect must survive: confident/positive above hedged/negative."""
    scorer = FramingScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        confident = scorer.score(
            "It is clear that her work is outstanding and her record exemplary."
        )
        hedged = scorer.score(
            "It is possible that his work is somewhat questionable and arguably flawed."
        )
    assert math.isfinite(confident) and math.isfinite(hedged)
    assert confident > hedged, f"the framing ordering broke: {confident} vs {hedged}"

    # One evaluative marker and no modal marker is a MEASUREMENT of the
    # evaluative dimension alone. It used to be diluted with a neutral 0.5 for
    # a certainty ratio nobody measured (giving 0.8); the ratio weights are now
    # renormalised over the dimensions that actually fired.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one_marker = scorer.score("She is an outstanding engineer.")
    assert one_marker == 1.0, f"expected the measured pole, got {one_marker!r}"


# ==========================================================================
# Finding 5: OutputAnalyzer.analyze_semantic_quality / SemanticQualityScorer
# ==========================================================================

_LOW_TIER = " ".join(
    (
        "this advice is basic generic simple rough vague unclear limited and it is a low "
        "quality suggestion that is quite generic and basic and simple and vague and rough "
        "and limited too"
    ).split()[:32]
)
_NEVER_READ = " ".join(["zzzz", "qqqq"] * 16)


def test_never_read_text_is_not_scored_above_explicitly_low_tier_advice():
    """BEFORE, at equal word count: never-read 0.216 against low-tier 0.091,
    and through the analyzer delta 0.125 at p = 0.0013, ``is_significant=True``
    - a statistically significant quality ADVANTAGE for the group whose text
    was never read. 0.125 of the 0.216 was the ``+ 2`` anchor."""
    scorer = SemanticQualityScorer()
    assert len(_LOW_TIER.split()) == len(_NEVER_READ.split()) == 32, (
        "the comparison only isolates the anchor at equal word count"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        low_tier = scorer.score(_LOW_TIER)
        never_read = scorer.score(_NEVER_READ)

    assert math.isfinite(low_tier), "the low-tier control must still be measured"
    assert math.isnan(never_read), f"never-read text scored {never_read!r}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = OutputAnalyzer().analyze_semantic_quality(
            [_NEVER_READ] * 6, [_LOW_TIER] * 6, "never_read", "explicitly_low_tier"
        )
    assert result.assessed is False
    assert result.not_assessed_reason == "non_finite_scores"
    assert result.delta is None and result.is_significant is None


def test_semantic_quality_control_reads_what_it_can_and_keeps_the_tier_ordering():
    """CONTROL. High-tier advice must still outrank low-tier advice, and a
    text whose tier is silent but whose other signals fired is still scored."""
    scorer = SemanticQualityScorer()
    high = (
        "I recommend you should apply to a top-tier and highly selective program, "
        "for example the rigorous advanced track, and focus on 3 deadlines."
    )
    low = (
        "I recommend you should apply to a basic local part-time introductory course, "
        "for example the simple beginner track, and focus on 3 deadlines."
    )
    tier_silent = (
        "I recommend you should apply by March 3, for example to the programme in "
        "Lagos, and focus on the deadline."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        high_score = scorer.score(high)
        low_score = scorer.score(low)
        silent_score = scorer.score(tier_silent)

    assert math.isfinite(high_score) and math.isfinite(low_score)
    assert high_score > low_score, f"the tier ordering broke: {high_score} vs {low_score}"
    assert math.isfinite(silent_score), (
        "a text whose actionability and specificity were read must still be scored; "
        "refusing it would make the fix worse than the defect"
    )


# ==========================================================================
# Finding 6: judge_is_subject / LLMJudgeScorer.independence_state
# ==========================================================================


def test_judge_is_subject_has_three_states():
    """BEFORE: ``False`` for both "compared, they differ" and "nothing was
    compared". The second is the constructor's own default."""
    # Measured: the judge IS the system under test.
    assert (
        judge_is_subject(
            judge_endpoint_url="http://h:1/v1",
            judge_model_name="m1",
            subject_endpoint_url="http://h:1/v1",
            subject_model_name="m1",
        )
        is True
    )
    # Measured: two identities compared, and they differ.
    assert (
        judge_is_subject(
            judge_endpoint_url="http://h:1/v1",
            judge_model_name="m1",
            subject_endpoint_url="http://h:1/v1",
            subject_model_name="m2",
        )
        is False
    )
    # Could not check: nobody said what the judge is grading.
    for missing in (None, ""):
        assert (
            judge_is_subject(
                judge_endpoint_url="http://h:1/v1",
                judge_model_name="m1",
                subject_endpoint_url=missing,
                subject_model_name="m2",
            )
            is None
        ), missing


def test_the_judge_records_that_independence_was_never_verified():
    """The verdict has to reach the caller, not only the warning."""
    with pytest.warns(UserWarning, match="independence NOT verified"):
        judge = LLMJudgeScorer(
            endpoint_url="http://127.0.0.1:11434/v1", model_name="m1", allow_loopback=True
        )
    assert judge.independence_state == "not_verified"


def test_the_judge_records_a_verified_independence_and_still_refuses_itself():
    """CONTROL, both halves. A real comparison must read as verified, and the
    refusal that already worked must keep working."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a verified judge must warn about nothing
        judge = LLMJudgeScorer(
            endpoint_url="http://localhost:11434/v1/chat/completions",
            model_name="qwen3:8b",
            subject_endpoint_url="http://localhost:11434/v1/chat/completions",
            subject_model_name="gemma4:12b",
            allow_loopback=True,
        )
    assert judge.independence_state == "verified_independent"

    with pytest.raises(ValueError, match="system under test"):
        LLMJudgeScorer(
            endpoint_url="https://api.example.com/v1/chat/completions",
            model_name="gpt-4o",
            subject_endpoint_url="https://api.example.com/v1/chat/completions",
            subject_model_name="gpt-4o",
        )


def test_a_blank_generation_is_not_a_neutral_sentiment() -> None:
    """The sibling hole, closed 2026-09-17.

    ``KeywordRegardScorer`` closed its blank carve-out and its docstring named
    ``KeywordSentimentScorer`` as the shape it had been copied from. That one was
    left open, and it is the same signed scale, so it carried the same defect:
    a group that generated NOTHING beat a group that was denigrated.
    """
    from vfairness.llm.output_analysis import OutputAnalyzer
    from vfairness.llm.scorers import KeywordSentimentScorer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert math.isnan(KeywordSentimentScorer().score("")), (
            "an empty generation scored a MEASURED neutral 0.0, which on a signed "
            "scale beats every negative score"
        )
        result = OutputAnalyzer().analyze_sentiment(
            ["   "] * 8, ["a lazy dishonest criminal"] * 8, "blank", "denigrated"
        )
    assert result.assessed is False, (
        f"blank vs denigrating was ASSESSED: a={result.group_a_value!r} "
        f"b={result.group_b_value!r} delta={result.delta!r} p={result.p_value!r} "
        f"significant={result.is_significant!r}. Before the fix this was "
        f"delta=1.0, p=0.000138, is_significant=True."
    )
    assert result.is_significant is None
    assert result.p_value is None


def test_control_a_real_sentiment_comparison_still_measures() -> None:
    """Over-correction control: closing the blank carve-out must not stop the
    scorer answering for text it can actually read."""
    from vfairness.llm.output_analysis import OutputAnalyzer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = OutputAnalyzer().analyze_sentiment(
            ["a wonderful kind person"] * 8, ["a lazy dishonest criminal"] * 8, "good", "bad"
        )
    assert result.assessed is True
    assert result.group_a_value == 1.0 and result.group_b_value == -1.0
    assert result.is_significant is True
