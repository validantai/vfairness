"""BGL4 audit of batch A-llm-1: the overturns, recorded as executable evidence.

Ten of the twenty six grades in batch A-llm-1 were overturned on audit. Each
test below asserts what an HONEST answer would be, and was marked
``xfail(strict=True)`` while the defect was OPEN, so that the XPASS on the day
somebody fixed it would turn the marker into a failure.

No test here was written to pass. Every one of them was first run without the
marker, and the observed value is quoted in its docstring verbatim.

AUDITOR NOTE: this file recorded defects. It did not fix them.

BGL5, 2026-09-27: THE FIXES LANDED, and every marker below has been removed
except one. Each test now asserts the corrected behaviour, keeps its subject and
carries the measured "before" that its xfail reason used to hold, so the evidence
that the defect was real survives the fix. The one marker left is
``test_the_single_stage_coverage_reaches_the_result_a_reader_consults``: the
producer is honest and the disclosure dies at
``llm/output_analysis.py::OutputAnalysisResult``, which batch A-llm-1 does not
own, so it is recorded as a deferral with the file it needs rather than quietly
converted. The fixes are pinned independently in tests/test_bgl5_llm_1.py, which
also carries the over-correction controls.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vfairness.llm.scorers as S
from vfairness.llm.nondeterminism import NonDeterminismAnalyzer, noise_floor_from_runs
from vfairness.llm.output_analysis import OutputAnalyzer


@pytest.fixture(autouse=True)
def _restore_the_once_per_process_warning_flags():
    """Put the module level warn once booleans back, as the batch path pins do."""
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


# ==========================================================================
# OVERTURN 1. vfairness.llm.nondeterminism.NonDeterminismAnalyzer.required_runs
# Claimed PROVEN (bgl3, 2026-09-27). The guard added that day refuses min_runs
# of 0, 1 and -5, and the named pin goes red when it is disabled. It is
# bypassed by a value the pin never tried, because `nan < 2` is False.
# ==========================================================================


def test_a_non_finite_minimum_is_refused_like_a_minimum_below_two():
    """CLOSED BGL5. Measured before: NonDeterminismAnalyzer(min_runs=float('nan'))
    was ACCEPTED and required_runs() answered nan, so every 'n < required_runs()'
    disclosure in the module was unreachable. That is the same off switch the
    2026-09-27 guard was added to close, reached by a float instead of an int,
    because `nan < 2` is False."""
    with pytest.raises(ValueError, match="min_runs must be at least 2"):
        NonDeterminismAnalyzer("llm", min_runs=float("nan"))


def test_a_non_finite_minimum_cannot_silence_the_thin_floor_disclosure():
    """CLOSED BGL5. min_runs=nan reproduced the whole defect at the public entry
    point. Measured before: variant state 'measured', reason None, limitations [],
    and both 'below the recommended 25' RuntimeWarnings gone, on the identical two
    runs per variant fixture on which the default answers
    'measured_with_limitation' with two limitations. A two run floor read exactly
    like a twenty five run floor.

    The MECHANISM of the assertion changed and its subject did not: a non-finite
    minimum is now refused outright, so there is no result left to read a silenced
    disclosure off. The disclosure it was silencing is asserted right after, on
    the default path, because a fix that refused the default too would satisfy the
    first half alone.
    """
    runs = {"a": ["one two three four", "one two three"], "b": ["five six", "five six seven"]}
    with pytest.raises(ValueError, match="min_runs must be at least 2"):
        noise_floor_from_runs(runs, metrics=["response_length"], min_runs=float("nan"))

    out, caught = _caught(noise_floor_from_runs, runs, metrics=["response_length"])
    assert all(v["state"] == "measured_with_limitation" for v in out["variants"].values())
    assert len(out["limitations"]) == 2
    assert any("recommended minimum is 25" in str(w.message) for w in caught)


# ==========================================================================
# OVERTURN 2. vfairness.llm.scorers.LLMJudgeScorer.score / .score_batch
# Claimed PROVEN (bgl3). The bool guard added that day is real and its pin goes
# red when reverted. The guard checks the TYPE only, while its own warning text
# says "not a number on the 0 to 10 scale", so an out of range rating is
# clipped into a computed score with no warning at all.
# ==========================================================================


def _judge_answering(monkeypatch, reply):
    monkeypatch.setattr(S.LLMJudgeScorer, "_call_judge", lambda self, text: reply)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.LLMJudgeScorer("https://judge.example.com/v1/chat", model_name="judge-x")


def test_a_rating_outside_the_zero_to_ten_scale_is_not_a_rating(monkeypatch):
    """CLOSED BGL5. Measured before: a judge answering on a 0 to 100 scale scored
    every text exactly 1.0 with ZERO warnings, because min(max(v, 0), 10) clips
    85, 92, 78 and 88 to 10. Every text at 1.0 makes every group comparison report
    delta 0.0, p 1.0, is_significant False: a false clean bill. The guard above it
    refused a bool and a string and never looked at the range its own message
    names."""
    scorer = _judge_answering(
        monkeypatch,
        {"helpfulness": 85, "fairness": 92, "specificity": 78, "completeness": 88},
    )
    value, caught = _caught(scorer.score, "an ordinary generated reply about a loan")
    assert math.isnan(value), f"an out of range reply was scored {value}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


def test_a_non_finite_rating_is_announced_rather_than_clipped(monkeypatch):
    """CLOSED BGL5. json.loads accepts the bare tokens NaN and Infinity by default,
    so a judge reply can carry them. Measured before: Infinity on all four
    dimensions scored 1.0 with zero warnings, and NaN on one dimension scored nan
    with zero warnings, where every other refusal in this method is announced."""
    inf_scorer = _judge_answering(
        monkeypatch,
        {k: float("inf") for k in ("helpfulness", "fairness", "specificity", "completeness")},
    )
    value, caught = _caught(inf_scorer.score, "an ordinary generated reply")
    assert math.isnan(value), f"an infinite rating was scored {value}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


def test_the_judge_batch_path_does_not_carry_a_clipped_rating(monkeypatch):
    """CLOSED BGL5. score_batch is np.array([self.score(t) for t in texts]), so it
    carried the clipping straight into the array every aggregate comparison reads.
    Measured before: array([1.0, 1.0]) from a judge answering 0 to 100, no
    warning."""
    scorer = _judge_answering(
        monkeypatch,
        {"helpfulness": 85, "fairness": 92, "specificity": 78, "completeness": 88},
    )
    values, _ = _caught(scorer.score_batch, ["one reply", "another reply"])
    assert np.all(np.isnan(values)), f"the batch path scored {values}"


# ==========================================================================
# OVERTURN 3. vfairness.llm.scorers.RefusalScorer.score / .score_batch
# Claimed PROVEN (wave3 / wave5). The could not check path was verified on CJK
# input only. The readability gate is `_lexicon_tokens`, which matches ASCII
# letters, so a refusal in any LATIN script language is "read", matches none of
# the 50+ English patterns, and is reported as a MEASURED 0.0.
# ==========================================================================

_FRENCH_REFUSAL = "Je ne peux pas vous aider avec cette demande."
_GERMAN_REFUSAL = "Ich kann Ihnen dabei nicht helfen."


@pytest.mark.parametrize("text", [_FRENCH_REFUSAL, _GERMAN_REFUSAL], ids=["french", "german"])
def test_a_refusal_the_english_lexicon_cannot_express_is_not_a_measured_zero(text):
    """CLOSED BGL5. Measured before: a refusal in a Latin script language scored a
    MEASURED 0.0 with no warning, while the identical refusal in Chinese scored nan
    with an UnscorableTextWarning. The scorer's own warning text says 'use a
    multilingual refusal detector for non-English output', and that advice reached
    only the callers who already got a NaN. 0.0 here is the categorical claim 'this
    response did not refuse', about a response that refused."""
    value, caught = _caught(S.RefusalScorer().score, text)
    assert math.isnan(value), f"a real refusal scored {value}, the clean end of the scale"
    assert any(issubclass(w.category, S.UnscorableTextWarning) for w in caught)


def test_a_total_refusal_disparity_in_french_is_not_reported_as_no_disparity():
    """CLOSED BGL5, at the public entry point. Measured before, on 25 runs per
    variant where applicant_A is refused in French every single time and
    applicant_B is helped in French every single time: refusal_rate state
    'measured', observed 0.0, exceeds_noise False, systematic_offset 0.0,
    n_excluded_reference 0, scorer_quality 'production', limitations []. That is
    byte for byte the outcome RefusalScorer._score_or_none's docstring says it
    fixed. After: state 'could_not_check'."""
    runs = {
        "applicant_A": [_FRENCH_REFUSAL] * 25,
        "applicant_B": ["Voici comment proceder pour votre demande de pret."] * 25,
    }
    out, _ = _caught(noise_floor_from_runs, runs, metrics=["refusal_rate"])
    comparison = out["metrics"]["refusal_rate"]["comparisons"][0]
    assert comparison["state"] == "could_not_check" or comparison["exceeds_noise"] is True, (
        f"a total refusal disparity was reported as observed={comparison['observed']} / "
        f"exceeds_noise={comparison['exceeds_noise']}"
    )


# ==========================================================================
# OVERTURN 4. SidecarSentimentScorer.score_batch / SidecarToxicityScorer.score_batch
# Claimed PROVEN (wave3). The three named pins cover the bridge is None outage
# only: coverage of them shows `return np.array([float(x) for x in r])` is
# never executed. That unexecuted line accepts a reply the single text sibling
# two lines up refuses.
# ==========================================================================


class _AnsweringBridge:
    def __init__(self, reply):
        self.reply = reply

    def call(self, op, **payload):
        return self.reply


def _bridge_answering(monkeypatch, reply):
    monkeypatch.setattr(S._SidecarBridge, "get", classmethod(lambda cls: _AnsweringBridge(reply)))
    monkeypatch.setattr(S, "_sidecar_bad_reply_warned", False)


@pytest.mark.parametrize(
    "cls", [S.SidecarSentimentScorer, S.SidecarToxicityScorer], ids=["sentiment", "toxicity"]
)
def test_a_boolean_reply_is_not_a_batch_of_scores(monkeypatch, cls):
    """CLOSED BGL5. score() guards `not isinstance(r, bool)` and answers nan plus a
    PlaceholderScorerWarning for a boolean reply. score_batch checked only
    `isinstance(r, list)` and then called float() on each element, so [True, False,
    True] measured array([1.0, 0.0, 1.0]) with ZERO warnings: on toxicity that is
    'maximally toxic' invented from a bool. It was the same defect the
    LLMJudgeScorer row fixed on 2026-09-27, live in the sibling batch path, and
    both now go through one shared check."""
    _bridge_answering(monkeypatch, [True, False, True])
    texts = ["a lovely person", "a vile idiot", "an ordinary sentence"]
    values, caught = _caught(cls().score_batch, texts)
    assert np.all(np.isnan(values)), f"booleans became scores {values}"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


@pytest.mark.parametrize(
    "cls", [S.SidecarSentimentScorer, S.SidecarToxicityScorer], ids=["sentiment", "toxicity"]
)
def test_a_reply_of_the_wrong_length_is_not_a_batch_of_scores(monkeypatch, cls):
    """CLOSED BGL5. The reply's LENGTH was never checked. Measured before, on three
    texts: a reply of [0.9] returned a one element array, [] returned an empty
    array and a five element reply returned five values, all with zero warnings. A
    caller that slices the result by group index then reads one text's score under
    another text's label, which is a disparity computed from a misalignment."""
    _bridge_answering(monkeypatch, [0.9])
    texts = ["a lovely person", "a vile idiot", "an ordinary sentence"]
    values, caught = _caught(cls().score_batch, texts)
    assert len(values) == len(texts), f"{len(texts)} texts produced {len(values)} values, silently"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


# ==========================================================================
# OVERTURN 5. vfairness.llm.scorers.ContextualStereotypeScorer.score
# Claimed PROVEN (bgl3). The per call PartialCoverageWarning is real and its pin
# goes red when removed. The disclosure does not survive to the only surface a
# reader consults: nothing in OutputAnalysisResult carries it, and
# stage_coverage() has no caller anywhere in the package.
# ==========================================================================


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STILL OPEN after BGL5, and DEFERRED rather than fixed: the change belongs in "
        "llm/output_analysis.py (OutputAnalysisResult needs to carry a coverage field, and "
        "_compare needs a coverage value for its existing not_assessed_reason channel), "
        "which batch A-llm-1 does not own. The PRODUCER half is closed: every call now "
        "emits the PartialCoverageWarning, stage_coverage() is machine readable, and an NLI "
        "reading outside [0, 1] no longer counts as a stage-2 reading at all "
        "(tests/test_bgl5_llm_1.py). What is missing is the consumer. "
        "BGL4 A-llm-1 overturn, the disclosure does not reach a consumer. Measured with a "
        "configured but dead sidecar over 8 contextual stereotypes against 8 neutral texts: "
        "the result dict reads group_a_value 0.0, group_b_value 0.0, delta 0.0, p_value 1.0, "
        "is_significant False, assessed True, not_assessed_reason None, scorer_quality "
        "'unknown', and carries NO coverage field of any kind. The only thing that says the "
        "stage which exists to catch a keyword free stereotype never looked is a Python "
        "warning, and stage_coverage() is called by tests and by nothing in src/."
    ),
)
def test_the_single_stage_coverage_reaches_the_result_a_reader_consults(monkeypatch):
    monkeypatch.setattr(S._SidecarBridge, "get", classmethod(lambda cls: None))
    monkeypatch.setattr(S, "_sidecar_down_warned", False)
    analyzer = OutputAnalyzer(stereotype_scorer=S.ContextualStereotypeScorer())
    group_a = ["Men are naturally better suited to leadership roles than women."] * 8
    group_b = ["The candidate presented a clear plan for the quarter."] * 8

    result, _ = _caught(analyzer.analyze_stereotype, group_a, group_b)
    payload = result.to_dict()

    serialised = str(payload).lower()
    assert ("coverage" in serialised) or payload.get("not_assessed_reason"), (
        "the result a reader consults says assessed=True, delta=0.0 and nothing at all "
        f"about single stage coverage: {payload}"
    )


# ==========================================================================
# OVERTURN 6. vfairness.llm.scorers.TextScorer
# Claimed SEMI-PROVEN (wave4), on the stated evidence that it was "executed for
# the first time by any test and judged against the three-state rule on the
# degenerate input for its own family". It cannot be executed at all. This test
# is the evidence for the regrade to NOT A MEASUREMENT and it PASSES today.
# ==========================================================================


def test_the_text_scorer_protocol_has_no_behaviour_to_grade():
    """A typing Protocol with `...` bodies. There is no degenerate input for it.

    `TextScorer()` raises TypeError, and `runtime_checkable` makes isinstance a
    METHOD PRESENCE test only, so a scorer returning the string 'not a number'
    satisfies it. The named pin asserts `TextScorer._is_protocol` and nothing
    else, so it could not have disagreed about any behaviour.
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
