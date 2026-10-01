"""BGL stage 5, batch llm-4: does each LLM fairness unit refuse when nothing was measured?

Eleven units across five files were executed on healthy input and on input where the
quantity they report does not exist. Four defects were found by execution, and every
refusal that was already correct is pinned here too, because an unpinned correct
refusal is one edit away from a fabricated verdict.

WHAT WAS MEASURED BEFORE THE FIXES, in the numbers each test repeats:

1. ``TextFairnessAnalyzer.analyze`` silently deleted an identity term supplied with no
   text. Three terms, 'nonbinary' empty, the other two scoring 0.10 to 0.13 against
   0.14 to 0.17: severity 'info', "No material identity-term bias detected for 'men'
   (gap +0.020, p=0.029)", notes [], and no warning at all.
2. ``TextFairnessResult.to_dict`` then dropped the two coverage fields that fix adds,
   so a consumer reading the serialised output saw the scores and no trace of the term
   that was never scored.
3. ``IntersectionalGroup.from_attributes({})`` manufactured ONE group with
   ``attributes={}`` and ``label=''``, and an axis holding no values erased the design
   and returned ``[]``. Both silent.
4. ``IntersectionalAnalyzer.analyze`` with an unrecognised metric name dropped every
   pair: on a 2x2 design (6 pairs) it reported total_pairs=0 AND
   n_pairs_not_assessed=0, the field documented as "0 on a complete run", with only a
   logger line.
5. ``ContextualStereotypeScorer.score`` answered 0.0 for "Men are naturally better
   suited to leadership roles than women." with the NLI stage dead, and the only
   disclosure latched once per process, so the second job of a process got 0.0 in
   silence. Through the analyzer, 8 such generations against 8 neutral ones:
   group_a_value 0.0, group_b_value 0.0, delta 0.0, p 1.0, is_significant False,
   assessed True, no warning.
6. ``LLMJudgeScorer.score`` accepted booleans as ratings, because
   ``isinstance(True, int)`` is True: a judge answering ``true`` on all four
   dimensions scored 0.1 with zero warnings.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import vfairness.llm.scorers as S
from vfairness.llm.embedding_bias import EmbeddingBiasDetector
from vfairness.llm.intersectional import IntersectionalAnalyzer, IntersectionalGroup
from vfairness.llm.model_identity import (
    NOT_ASSESSED,
    ModelIdentity,
    compare_model_identity,
    describe_model_identity,
    identity_from_run,
)
from vfairness.llm.text_fairness import TextFairnessAnalyzer


@pytest.fixture(autouse=True)
def _restore_the_once_per_process_warning_flags():
    """Leave the scorers module's warn-once booleans exactly as they were found.

    ``_warn_sidecar_down`` fires ONCE PER PROCESS through a module-level flag, and the
    stereotype tests below score text with no bridge, which trips it. Measured while
    writing this file: ``tests/test_no_aggregator_fabricates_a_verdict.py::
    test_a_scorer_whose_sidecar_is_down_is_loud_about_it`` then found no warning to
    observe and failed, passing on its own and failing in the combined run, which is
    the signature of exactly this shared state. The flag is process-global on purpose,
    so the fix belongs here: a test that consumes another test's once-per-process
    budget has to put it back. Same fixture as
    ``tests/test_bgl2_scorer_batch_paths.py``.
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
    """``(return value, warnings)``, with every filter forced to record."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, caught


def _messages(caught, category=Warning):
    return [str(w.message) for w in caught if issubclass(w.category, category)]


# ===========================================================================
# 1. text_fairness: an identity term nobody scored is not a term that agreed.
# ===========================================================================

_SMALL_GAP = {
    "w1": 0.10,
    "w2": 0.12,
    "w3": 0.11,
    "w4": 0.13,
    "m1": 0.14,
    "m2": 0.15,
    "m3": 0.16,
    "m4": 0.17,
    "n1": 0.12,
    "n2": 0.13,
    "n3": 0.14,
    "n4": 0.15,
}


def _table_clf(texts):
    return [_SMALL_GAP[t] for t in texts]


def _biased_clf(texts):
    return [0.9 if "women" in t else 0.15 for t in texts]


def test_an_all_clear_is_refused_when_an_identity_term_was_never_scored():
    """Measured before the fix on this exact fixture: severity 'info', caption "No
    material identity-term bias detected for 'men' (gap +0.020, p=0.029)", notes [],
    zero warnings, and a groups list of 2 for a design of 3. The 'info' caption is a
    positive claim that every identity term was compared and none of them differed."""
    result, caught = _caught(
        TextFairnessAnalyzer(_table_clf).analyze,
        {
            "women": ["w1", "w2", "w3", "w4"],
            "men": ["m1", "m2", "m3", "m4"],
            "nonbinary": [],
        },
    )

    assert result.severity == "not_assessed", (
        f"an unscored identity term still produced a graded reading: {result.severity}"
    )
    assert "NOT ASSESSED" in result.interpretation
    assert "nonbinary" in result.interpretation
    assert result.groups_not_scored == ["nonbinary"]
    assert result.n_groups_supplied == 3
    assert [g.group for g in result.groups] == ["women", "men"]
    assert any("NOT SCORED" in m and "nonbinary" in m for m in _messages(caught)), (
        f"the dropped term was not disclosed at runtime: {_messages(caught)}"
    )
    # The measurement on the survivors is kept, not thrown away.
    assert result.max_gap == pytest.approx(0.02, abs=1e-9)
    assert result.p_value is not None


def test_control_a_complete_design_still_reports_its_all_clear():
    """The over-correction control. The same classifier and the same tiny gap, with
    the third identity term POPULATED, must still grade 'info' and warn about
    nothing: a unit that refused every input would pass the test above."""
    result, caught = _caught(
        TextFairnessAnalyzer(_table_clf).analyze,
        {
            "women": ["w1", "w2", "w3", "w4"],
            "men": ["m1", "m2", "m3", "m4"],
            "nonbinary": ["n1", "n2", "n3", "n4"],
        },
    )

    assert result.severity == "info"
    assert "No material identity-term bias detected" in result.interpretation
    assert result.groups_not_scored == []
    assert result.n_groups_supplied == 3
    assert len(result.groups) == 3
    assert _messages(caught) == []


def test_a_finding_survives_a_partial_design_and_carries_its_coverage():
    """A finding found is still a finding however much of the rest is missing, so the
    severity is NOT downgraded. Measured on this fixture before the fix: severity
    'critical' with an interpretation that said nothing about the missing term, and
    overall_mean 0.525 computed over 2 of the 3 supplied terms."""
    result, caught = _caught(
        TextFairnessAnalyzer(_biased_clf).analyze,
        {
            "women": ["women a", "women b", "women c", "women d"],
            "men": ["men a", "men b", "men c", "men d"],
            "nonbinary": [],
        },
    )

    assert result.severity == "critical"
    assert "COVERAGE" in result.interpretation
    assert "nonbinary" in result.interpretation
    assert result.groups_not_scored == ["nonbinary"]
    assert any("NOT SCORED" in m for m in _messages(caught))


def test_the_serialiser_carries_the_coverage_fields():
    """A hand-written key list does not fail when a field is added, it stops carrying
    it. Before this pin, ``to_dict`` emitted eight keys and dropped both coverage
    fields, so the only machine-readable trace of the unscored term existed on the
    object and nowhere a consumer reads. Caught by tests/test_serialiser_honesty.py."""
    import json

    result, _ = _caught(
        TextFairnessAnalyzer(_table_clf).analyze,
        {"women": ["w1", "w2", "w3", "w4"], "men": ["m1", "m2", "m3", "m4"], "nonbinary": []},
    )
    emitted = result.to_dict()

    assert emitted["groups_not_scored"] == ["nonbinary"]
    assert emitted["n_groups_supplied"] == 3
    assert emitted["severity"] == "not_assessed"
    assert any("NOT SCORED" in note for note in emitted["notes"])
    json.dumps(emitted)


def test_control_text_fairness_still_refuses_a_dead_classifier():
    """A correct refusal that was already in place, pinned so it cannot regress: a
    classifier answering NaN for every text gives overall_mean nan, p nan and
    severity 'not_assessed', never "no material bias"."""
    result, caught = _caught(
        TextFairnessAnalyzer(lambda texts: [float("nan")] * len(texts)).analyze,
        {"women": ["a", "b", "c", "d"], "men": ["a", "b", "c", "d"]},
    )

    assert result.severity == "not_assessed"
    assert math.isnan(result.overall_mean)
    assert "NOT ASSESSED" in result.interpretation
    assert any("COULD NOT" in m or "NOT DETECTABLE" in m for m in _messages(caught))


# ===========================================================================
# 2. intersectional: a design with no intersection, and a metric that is not one.
# ===========================================================================


def test_from_attributes_refuses_a_design_with_no_intersection():
    """Measured before the fix, both silent: ``from_attributes({})`` returned
    ``[IntersectionalGroup(attributes={}, label='')]``, a group nobody asked for whose
    label is the empty string that a caller then keys its outputs by, and
    ``from_attributes({"race": [...], "gender": []})`` returned ``[]`` with no reason,
    so "no groups exist" and "nothing was requested" read the same."""
    empty_design, caught_empty = _caught(IntersectionalGroup.from_attributes, {})
    assert empty_design == []
    assert any("no attribute was supplied" in m for m in _messages(caught_empty))

    empty_axis, caught_axis = _caught(
        IntersectionalGroup.from_attributes, {"race": ["Black", "White"], "gender": []}
    )
    assert empty_axis == []
    assert any("gender" in m and "no values" in m for m in _messages(caught_axis))


def test_control_from_attributes_still_enumerates_a_real_design():
    """The over-correction control: a real 2x2 design still yields its four groups,
    silently."""
    groups, caught = _caught(
        IntersectionalGroup.from_attributes,
        {"race": ["Black", "White"], "gender": ["male", "female"]},
    )

    assert [g.label for g in groups] == [
        "Black_male",
        "Black_female",
        "White_male",
        "White_female",
    ]
    assert _messages(caught) == []


_GROUPS_2X2 = IntersectionalGroup.from_attributes(
    {"race": ["Black", "White"], "gender": ["male", "female"]}
)
_NEG = ["This is a terrible, awful and horrible outcome, truly bad."] * 8
_POS = ["This is a wonderful and excellent outcome, truly great."] * 8
_FULL_OUTPUTS = {
    "Black_male": _NEG,
    "Black_female": _NEG,
    "White_male": _POS,
    "White_female": _POS,
}


def test_an_unrecognised_metric_is_not_a_complete_run():
    """Measured before the fix on this 2x2 design (6 pairs) with metric='nonexistent':
    total_pairs=0, n_pairs_not_assessed=0, n_significant_pairs=0, max_disparity=nan,
    and the only disclosure was a logger line a caller capturing warnings never sees.
    ``n_pairs_not_assessed`` is documented as "0 on a complete run", so it said the
    run was complete while nothing at all had been measured. The docstring also used
    to name 'refusal', which resolves to no method, as a valid metric."""
    result, caught = _caught(
        IntersectionalAnalyzer().analyze, _FULL_OUTPUTS, _GROUPS_2X2, metric="nonexistent"
    )

    assert result.total_pairs == 6, "the design has 6 pairs and the result must say so"
    assert result.n_pairs_not_assessed == 6
    assert result.n_significant_pairs == 0
    assert math.isnan(result.max_disparity)
    assert result.has_intersectional_bias is None, "a run that measured nothing said False"
    assert {r.not_assessed_reason for r in result.pairwise_results} == {"unknown_metric"}
    assert any("not a metric this analyzer can run" in m for m in _messages(caught))

    # The name the docstring used to recommend must reach the same refusal, not a
    # silent empty analysis.
    typo, _ = _caught(
        IntersectionalAnalyzer().analyze, _FULL_OUTPUTS, _GROUPS_2X2, metric="refusal"
    )
    assert typo.has_intersectional_bias is None
    assert typo.n_pairs_not_assessed == 6


def test_control_a_real_metric_still_finds_intersectional_bias():
    """The over-correction control: the same design on a metric that exists still
    measures 6 pairs, finds the significant ones and ranks both ends."""
    result, _ = _caught(
        IntersectionalAnalyzer().analyze, _FULL_OUTPUTS, _GROUPS_2X2, metric="sentiment"
    )

    assert result.total_pairs == 6
    assert result.n_pairs_not_assessed == 0
    assert result.n_significant_pairs > 0
    assert result.has_intersectional_bias is True
    assert result.most_advantaged is not None and result.most_disadvantaged is not None
    assert result.most_advantaged.label != result.most_disadvantaged.label


def test_control_an_empty_cell_is_still_recorded_rather_than_deleted():
    """The BGL-D refusal beside it, pinned in this file too because the unknown-metric
    branch now shares its code path: with one cell empty, 3 of the 6 pairs are not
    assessed, the total stays 6, and a significant pair still answers True."""
    outputs = dict(_FULL_OUTPUTS, Black_female=[])
    result, caught = _caught(
        IntersectionalAnalyzer().analyze, outputs, _GROUPS_2X2, metric="sentiment"
    )

    assert result.total_pairs == 6
    assert result.n_pairs_not_assessed == 3
    assert result.has_intersectional_bias is True
    assert any("NOT ASSESSED" in m for m in _messages(caught))


# ===========================================================================
# 3. The two-stage stereotype scorer: a fused score with one stage dead.
# ===========================================================================


class _StubBridge:
    """Stands in for a live sidecar answering a fixed NLI value."""

    def __init__(self, value):
        self._value = value

    def call(self, op, **payload):
        return self._value


@pytest.fixture()
def no_nli_stage(monkeypatch):
    """The NLI stage produces no reading, and the once-per-process sidecar warning
    has ALREADY fired, which is the state every job after the first is in."""
    monkeypatch.setattr(S._SidecarBridge, "get", staticmethod(lambda: None))
    monkeypatch.setattr(S, "_sidecar_down_warned", True)
    monkeypatch.setattr(S, "_sidecar_bad_reply_warned", True)


_CONTEXTUAL_STEREOTYPE = "Men are naturally better suited to leadership roles than women."


def test_a_single_stage_fused_score_is_disclosed_on_every_call(no_nli_stage):
    """Measured before the fix with a sidecar CONFIGURED but dead, which is the only
    configuration in which this class is DEFAULT_STEREOTYPE_SCORER: the sentence below
    scored 0.0, the first call in the process emitted one SidecarUnavailableWarning,
    and every call after it emitted nothing. Through the analyzer, 8 such generations
    against 8 neutral ones: delta 0.0, p 1.0, is_significant False, assessed True, no
    warning. The value stays, because stage 1 genuinely read the text and refusing it
    would refuse most of any corpus; what may not stay is the silence."""
    scorer = S.ContextualStereotypeScorer()

    first, caught_first = _caught(scorer.score, _CONTEXTUAL_STEREOTYPE)
    second, caught_second = _caught(scorer.score, _CONTEXTUAL_STEREOTYPE)

    assert first == 0.0 and second == 0.0
    for caught in (caught_first, caught_second):
        assert any(issubclass(w.category, S.PartialCoverageWarning) for w in caught), (
            "a fused score resting on one stage was reported as a two-stage score"
        )
    assert any("LOWER BOUND" in m for m in _messages(caught_second, S.PartialCoverageWarning))
    assert scorer.stage_coverage(_CONTEXTUAL_STEREOTYPE) == {"wordlist": True, "nli": False}


def test_the_batch_path_discloses_once_per_call_and_counts_the_texts(no_nli_stage):
    """One aggregate warning per CALL, not one per text and not one per PROCESS: a 10k
    batch against a dead sidecar must not emit 10k lines, and a latched warning would
    describe the first batch of a long-running consumer and no other.

    The SECOND batch is the assertion that discriminates. The first version of this
    test called ``score_batch`` once, so a sabotage that latched the warning per
    process left it GREEN: with one call there is nothing for a latch to suppress."""
    scorer = S.ContextualStereotypeScorer()
    texts = [
        _CONTEXTUAL_STEREOTYPE,
        "The applicant has eight years of experience.",
        "Those people are all thugs.",
    ]

    values, caught = _caught(scorer.score_batch, texts)
    partial = _messages(caught, S.PartialCoverageWarning)
    _, caught_again = _caught(scorer.score_batch, texts)
    partial_again = _messages(caught_again, S.PartialCoverageWarning)

    assert values.shape == (3,)
    assert len(partial) == 1, f"expected one aggregate coverage warning, got {len(partial)}"
    assert "3 of 3" in partial[0]
    assert len(partial_again) == 1, (
        "the second batch of the process was scored by one stage in silence, which is "
        "what a latched warning does to every job after the first"
    )
    # The lexical stereotype is still found: a lower bound is not a refusal.
    assert values[2] > 0.0


def test_a_measured_zero_for_a_readable_blank_is_kept(no_nli_stage):
    """DELIBERATE, and pinned here because the disclosure above must not be turned
    into a refusal. ``tests/test_bgl2_scorer_batch_paths.py`` pins the same thing for
    the presence-detector family: absence of the marker IS evidence of absence, and
    refusing it would remove every stereotype comparison in a corpus."""
    values, _ = _caught(S.ContextualStereotypeScorer().score_batch, ["", "  "])

    assert not np.any(np.isnan(values))
    assert np.all(values == 0.0)


def test_control_a_two_stage_fusion_answers_without_a_coverage_warning(monkeypatch):
    """The over-correction control. With the NLI stage answering, the fused score is
    the maximum of the two rescaled readings and NO coverage warning is emitted, so
    the warning above is a statement about the instrument rather than a constant."""
    monkeypatch.setattr(S._SidecarBridge, "get", staticmethod(lambda: _StubBridge(0.77)))
    scorer = S.ContextualStereotypeScorer()

    value, caught = _caught(scorer.score, "an entirely ordinary sentence")

    # rescale: (0.77 - 0.45) / (0.85 - 0.45) = 0.8, and stage 1 gives 0.0.
    assert value == pytest.approx(0.8, abs=1e-9)
    assert _messages(caught, S.PartialCoverageWarning) == []
    assert scorer.stage_coverage("an entirely ordinary sentence") == {
        "wordlist": True,
        "nli": True,
    }


def test_control_neither_stage_reading_is_still_nan(monkeypatch):
    """The BGL stage 3 refusal, pinned again because the fusion was refactored here: a
    text stage 1 cannot read at all, with no stage 2, is NaN and not the 0.0 that
    means "no stereotype"."""
    monkeypatch.setattr(S._SidecarBridge, "get", staticmethod(lambda: None))
    value, _ = _caught(S.ContextualStereotypeScorer().score, "这些人都是罪犯")

    assert math.isnan(value)


# ===========================================================================
# 4. The LLM judge: a reply that graded nothing.
# ===========================================================================


class _JudgeReply:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _judge_answering(monkeypatch, content, calls=None):
    import vfairness.net.egress as egress

    def post(url, **kw):
        if calls is not None:
            calls.append(url)
        return _JudgeReply(content)

    monkeypatch.setattr(egress, "guarded_post", post)


def _judge():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return S.LLMJudgeScorer("http://127.0.0.1:9/v1/chat", allow_loopback=True)


def test_a_boolean_is_not_a_rating(monkeypatch):
    """``isinstance(True, int)`` is True in Python, so a judge answering ``true`` on
    all four dimensions satisfied the "did it rate every dimension" test and was then
    clipped and divided. Measured before the fix: score 0.1, warnings 0, from a reply
    carrying no rating at all. ``_sidecar_stereotype_score`` excludes bool for exactly
    this reason 1800 lines up in the same file."""
    _judge_answering(
        monkeypatch,
        '{"helpfulness": true, "fairness": true, "specificity": true, "completeness": true}',
    )

    value, caught = _caught(_judge().score, "a response to grade")

    assert math.isnan(value), "a reply with no ratings produced a computed score"
    assert any(issubclass(w.category, S.PlaceholderScorerWarning) for w in caught)


@pytest.mark.parametrize(
    "content,category",
    [
        ("{}", S.PlaceholderScorerWarning),
        ('{"helpfulness": 8, "specificity": 7, "completeness": 6}', S.PlaceholderScorerWarning),
        (
            '{"helpfulness": null, "fairness": 4, "specificity": 7, "completeness": 6}',
            S.PlaceholderScorerWarning,
        ),
        (
            '{"helpfulness": "8", "fairness": "4", "specificity": "7", "completeness": "6"}',
            S.PlaceholderScorerWarning,
        ),
        ("[1, 2, 3]", RuntimeWarning),
        ("I will not answer in JSON", RuntimeWarning),
    ],
    ids=["empty_object", "one_missing", "one_null", "strings", "non_object", "unparseable"],
)
def test_a_judge_that_graded_nothing_is_nan(monkeypatch, content, category):
    """Every shape of ungraded reply, pinned together: the recorded status for this
    unit was "evidence insufficient", and the gap was the range of inputs covered
    rather than a fabrication. Before the C-08 fix each of these scored exactly 0.5,
    byte-identical to a judge that genuinely rated everything 5."""
    _judge_answering(monkeypatch, content)

    value, caught = _caught(_judge().score, "a response to grade")

    assert math.isnan(value)
    assert any(issubclass(w.category, category) for w in caught)


def test_an_unreachable_judge_is_nan_per_text_and_never_a_score():
    """No stub at all: the connection fails or the egress guard refuses it, and either
    way nothing was graded. The batch path is what every group comparison uses."""
    values, caught = _caught(_judge().score_batch, ["one", "two"])

    assert values.shape == (2,)
    assert np.isnan(values).all()
    assert math.isnan(float(np.mean(values))), "a group mean over failed calls must be NaN"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_an_empty_batch_is_an_empty_array_and_not_a_zero():
    """Nothing to score is not a score of zero: a scalar here becomes a group mean for
    a group with no generations."""
    values, _ = _caught(_judge().score_batch, [])

    assert values.shape == (0,)
    assert values.dtype == np.float64


def test_control_the_judge_still_weights_a_complete_reply(monkeypatch):
    """The over-correction control, and the exact arithmetic: fairness weighted 0.40,
    the other three 0.20 each, each normalised from its 0 to 10 scale. A judge that
    refused everything would satisfy every assertion above."""
    _judge_answering(
        monkeypatch, '{"helpfulness": 8, "fairness": 4, "specificity": 7, "completeness": 6}'
    )

    value, caught = _caught(_judge().score, "a response to grade")

    assert value == pytest.approx(0.58, abs=1e-9)
    assert _messages(caught) == []


def test_a_blank_generation_never_reaches_the_judge(monkeypatch):
    """THE ONE COLLAPSE LEFT IN THIS UNIT, pinned so it is visible rather than
    discovered again. Measured 2026-09-27: ``score("")`` makes zero judge calls, emits
    zero warnings and returns 0.0, whether the judge is reachable or not, so two
    groups of blank generations compare equal with assessed=True on a value nobody
    graded. It is the END of the scale rather than its midpoint, and
    ``tests/test_audit_wave4_llm.py::TestJudgeFailureIsLoud::
    test_empty_text_still_zero_not_nan`` pins the 0.0 deliberately as the control that
    the NaN work was not generalised into refusing ordinary input. Changing it is a
    product decision, and this test states the current contract exactly."""
    calls: list = []
    _judge_answering(
        monkeypatch,
        '{"helpfulness": 8, "fairness": 4, "specificity": 7, "completeness": 6}',
        calls=calls,
    )
    judge = _judge()

    blank, caught_blank = _caught(judge.score, "")
    whitespace, _ = _caught(judge.score, "   \n\t ")

    assert blank == 0.0 and whitespace == 0.0
    assert calls == [], "the judge was called on text it cannot grade"
    assert _messages(caught_blank) == []
    # And the judge IS called for text that has content, so the short circuit above
    # is about blank input and not about a scorer that never dials.
    _caught(judge.score, "a real response")
    assert len(calls) == 1


# ===========================================================================
# 5. embedding_bias: WEAT and SEAT on a dead embedding backend.
# ===========================================================================


def _planted_embeddings(n=10, dim=16, seed=7):
    """Ten words a side, with target_a and attribute_a on one axis and target_b and
    attribute_b on the other, plus small deterministic jitter so the per-word spread
    is non-zero."""
    rng = np.random.default_rng(seed)
    axis = np.zeros(dim)
    axis[0] = 1.0
    emb = {}
    for prefix, sign in (("ta", 1.0), ("aa", 1.0), ("tb", -1.0), ("ab", -1.0)):
        for i in range(n):
            emb[f"{prefix}{i}"] = sign * axis + 0.05 * rng.normal(size=dim)
    return emb


_TA = [f"ta{i}" for i in range(10)]
_TB = [f"tb{i}" for i in range(10)]
_AA = [f"aa{i}" for i in range(10)]
_AB = [f"ab{i}" for i in range(10)]


@pytest.mark.parametrize("method", ["weat", "seat"], ids=["weat", "seat"])
def test_a_dead_embedding_backend_is_could_not_check_not_no_bias(method):
    """A backend returning all-zero vectors gives every target word the identical
    association with both attribute sets, so the standardised effect size has a zero
    denominator. Before the 2026-09-10 fix the ``or 1e-12`` guard divided a zero
    numerator by it and produced effect_size 0.0 beside p 1.0, severity 'info',
    "No statistically significant association bias", and not one warning. Pinned for
    both entry points, because ``seat`` delegates to ``weat`` and a caller reading a
    SEAT badge must get the same refusal."""
    detector = EmbeddingBiasDetector(embed_fn=lambda ws: np.zeros((len(ws), 8)))

    result, caught = _caught(getattr(detector, method), _TA, _TB, _AA, _AB, n_permutations=500)

    assert math.isnan(result.effect_size), "a zero denominator produced a clean 0.0"
    assert result.severity == "could_not_check"
    assert "COULD NOT CHECK" in result.interpretation
    assert any("undefined" in m for m in _messages(caught))


def test_seat_does_not_stamp_a_sentence_test_on_word_vectors():
    """The method label is a claim about the input. Without
    ``sentence_resolved=True`` the result stays "WEAT" and says why, rather than
    wearing a SEAT badge no input established."""
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings())

    unresolved, _ = _caught(detector.seat, _TA, _TB, _AA, _AB, n_permutations=500)
    resolved, _ = _caught(
        detector.seat, _TA, _TB, _AA, _AB, n_permutations=500, sentence_resolved=True
    )

    assert unresolved.method == "WEAT"
    assert any("not flagged sentence-resolved" in note for note in unresolved.notes)
    assert resolved.method == "SEAT"


def test_control_a_planted_stereotype_is_measured_and_graded():
    """The over-correction control: a detector that answered could_not_check to
    everything would satisfy the refusal test above."""
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings())

    result, _ = _caught(detector.weat, _TA, _TB, _AA, _AB, n_permutations=2000)

    assert result.effect_size > 0.8, f"the planted association was not measured: {result}"
    assert result.p_value < 0.05
    assert result.severity in ("high", "critical")


def test_control_a_design_too_small_to_reach_alpha_is_not_a_pass():
    """Three words a side against ten: the permutation null has C(6,3)=20 members, so
    the smallest attainable p is 0.1 and no embedding whatsoever could be flagged.
    Graded could_not_check rather than "no significant association bias"."""
    detector = EmbeddingBiasDetector(embeddings=_planted_embeddings())

    result, caught = _caught(detector.weat, _TA[:3], _TB[:3], _AA, _AB, n_permutations=2000)

    assert result.severity == "could_not_check"
    assert result.effect_size > 0.8, "the effect size itself is still reported"
    assert any("could NOT have reached" in m for m in _messages(caught))


# ===========================================================================
# 6. model_identity: what answered, never assumed.
# ===========================================================================


def test_a_run_nobody_named_has_no_identity_and_says_so():
    """The configured name is a SETTING. A run whose responses named no model must not
    borrow it, because a reader seeing the two agree would take it for confirmation."""
    identity, caught = _caught(identity_from_run, [{"text": "hi"}] * 4, configured_model="qwen3:8b")

    assert identity.reported_model is None
    assert identity.is_consistent is None, "nothing to be consistent about is not consistent"
    assert identity.n_responses == 4 and identity.n_reporting == 0
    assert "was not recorded" in identity.coverage_statement
    assert any("carried a model name" in m for m in _messages(caught))


def test_a_run_served_by_two_models_refuses_to_name_one():
    """The strongest identity signal a run can produce: every statistic over these
    responses is a statistic over two systems, so ``reported_model`` stays None rather
    than taking the first name seen."""
    identity, caught = _caught(
        identity_from_run, [{"reported_model": "a"}, {"reported_model": "b"}]
    )

    assert identity.reported_model is None
    assert identity.reported_names == ("a", "b")
    assert identity.is_consistent is False
    assert "DID NOT SERVE ONE MODEL" in identity.coverage_statement
    assert any("DIFFERENT ones" in m for m in _messages(caught))
    assert "more than one model answered" in describe_model_identity(identity)


def test_a_matching_configuration_is_not_a_matching_model():
    """A comparison that could only see the SETTING is a could-not-check wearing a
    verified badge if it answers "same". ``needs_retest`` is deliberately False there:
    an unknown identity is not evidence of a change."""
    config_only = ModelIdentity(configured_model="qwen3:8b")

    same_config = compare_model_identity(config_only, config_only)
    nothing_recorded = compare_model_identity(None, None)

    assert same_config.state == NOT_ASSESSED
    assert same_config.basis == "configuration_only"
    assert same_config.needs_retest is False
    assert "could-not-check, not a match" in same_config.statement
    assert nothing_recorded.state == NOT_ASSESSED
    assert describe_model_identity(ModelIdentity()) == "model identity not recorded"


def test_control_a_real_change_and_a_real_match_are_both_reported():
    """The over-correction control: a comparison that answered not_assessed to
    everything would satisfy the test above."""
    recorded = ModelIdentity(
        configured_model="q", reported_model="q-2026-01", reported_names=("q-2026-01",)
    )
    moved = ModelIdentity(
        configured_model="q", reported_model="q-2026-02", reported_names=("q-2026-02",)
    )

    changed = compare_model_identity(recorded, moved)
    unchanged = compare_model_identity(recorded, recorded)

    assert changed.state == "changed"
    assert changed.changed_fields == ("reported_model",)
    assert changed.needs_retest is True
    assert unchanged.state == "same"
    assert unchanged.basis == "endpoint_report"
    assert unchanged.needs_retest is False
    assert "self-report" in unchanged.statement
