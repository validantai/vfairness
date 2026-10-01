"""Audit wave 2, 2026-09-29: three more faces of one mechanism.

``str(x)`` applied to a value that is ABSENT mints content out of nothing, and the
content is then counted, cross-tabulated, compared and published. The fourth face,
``score(None) -> 0.0`` across every scorer in ``llm/scorers.py``, is pinned in
``tests/test_w2_missing_response_is_not_an_empty_one.py``.

FACE 2: ``MultiAgentRunHarness.record_routing`` did
``routing_demographics.append(str(demographic))``, and ``str(None)`` is the
five-character label ``'None'``. Reproduced, ten calls with ``demographic=None``
against ten with ``demographic="a"``, each group routed one way, through
``DelegationRoutingAuditor().analyze``:

    groups=['None', 'a']  cramers_v=1.0  odds_ratio=inf
    p_value=1.082508822446903e-05  is_significant=True  warnings=0

A maximal, statistically significant finding of routing discrimination against a
group that does not exist. ``str(route)`` on the same line above it had the same
problem and minted a route named 'None'.

FACE 3: ``llm.model_identity.clean_model_name`` gated on ``str.strip()``, which
does not remove the Unicode FORMAT characters, so a name made of U+200B, U+200C,
U+FEFF and U+00AD survived it: four characters, truthy, and invisible. Measured on
four responses reporting that name:

    n_reporting      0 of 4 -> 4 of 4
    is_consistent    None   -> True
    the "none of the N responses carried a model name" warning: fired -> silent
    compare_model_identity  state='could_not_check'
                            -> state='same', basis='endpoint_report'

FACE 4: ``OutputAnalyzer`` called ``_filter_none`` at the top of every
``analyze_*`` method AND at the top of ``analyze_all``, which deleted the evidence
of differential attrition before the guard that refuses it could count anything.
One world, two descriptions, 8 of group B's 10 responses lost, a stub judge rating
A at 0.80 and B at 0.76:

    lost as None TEXTS  -> assessed=True, delta=+0.040, p=0.0016,
                           is_significant=True, n_supplied_b=2 reported as full
                           coverage, 1 warning and it was about the sample size
    the SAME loss as NaN SCORES
                        -> assessed=False, not_assessed_reason='differential_unscored'

EVERY PIN HERE WAS SABOTAGED: the defect was put back, the named test was
confirmed RED, and the source was restored and diffed byte-identical.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.llm import model_identity as MI
from vfairness.llm.output_analysis import OutputAnalyzer
from vfairness.multi_agent.delegation import DelegationRoutingAuditor
from vfairness.multi_agent.harness import MultiAgentRunHarness

# ==========================================================================
# FACE 2. A demographic group named 'None'.
# ==========================================================================

_ABSENT_LABELS = [None, float("nan"), np.float64("nan"), "", "   "]


@pytest.mark.parametrize("absent", _ABSENT_LABELS)
def test_a_demographic_that_was_not_recorded_is_refused(absent):
    """Before: appended as the label 'None' (or ''), and analysed as a group."""
    harness = MultiAgentRunHarness()
    with pytest.raises(ValueError, match="not a recorded label"):
        harness.record_routing(route="junior", demographic=absent)
    assert harness.trace.routing_demographics == [], "nothing may be recorded"
    assert harness.trace.routing_decisions == [], (
        "and the route must not be recorded either: a pair half-recorded would "
        "misalign routing_decisions against routing_demographics for every later task"
    )


@pytest.mark.parametrize("absent", _ABSENT_LABELS)
def test_a_route_that_was_not_recorded_is_refused(absent):
    """THE SIBLING ON THE SAME LINE ABOVE. ``str(route)`` had the identical defect,
    and a fix to one of the two lines would have left a route named 'None'."""
    harness = MultiAgentRunHarness()
    with pytest.raises(ValueError, match="not a recorded label"):
        harness.record_routing(route=absent, demographic="a")
    assert harness.trace.routing_decisions == []


def test_the_maximal_false_finding_can_no_longer_be_built():
    """The whole reproduction, end to end, as the caller met it.

    Ten routing calls with no demographic against ten labelled ones gave
    cramers_v=1.0, odds_ratio=inf, p=1.08e-05, is_significant=True and zero
    warnings. The refusal is what makes that table unbuildable.
    """
    harness = MultiAgentRunHarness()
    with pytest.raises(ValueError):
        for _ in range(10):
            harness.record_routing(route="junior", demographic=None)
    for _ in range(10):
        harness.record_routing(route="senior", demographic="a")
    routes, demographics = harness.as_delegation_inputs()
    assert set(demographics) == {"a"}, "'None' is not a demographic group"
    with pytest.raises(ValueError):
        # One group is not a comparison, which is the honest state of this run.
        DelegationRoutingAuditor().analyze(routes, demographics)


# OVER-CORRECTION CONTROL. `str()` is kept for every label a caller MEANT.


def test_real_labels_are_still_recorded_and_still_stringified():
    """0 and 0.0 are FALSY and are perfectly good group labels, so the predicate
    is a type-and-value test rather than ``if not value``."""
    harness = MultiAgentRunHarness()
    for route, demographic in [("x", "A"), ("y", "B"), (0, 1), ("x", 0), (1.5, 2.5)]:
        harness.record_routing(route, demographic)
    assert harness.trace.routing_decisions == ["x", "y", "0", "x", "1.5"]
    assert harness.trace.routing_demographics == ["A", "B", "1", "0", "2.5"]


def test_numpy_labels_are_still_recorded():
    harness = MultiAgentRunHarness()
    harness.record_routing(np.str_("senior"), np.int64(1))
    assert harness.as_delegation_inputs() == (["senior"], ["1"])


def test_a_real_routing_disparity_is_still_found():
    """The measurement this module exists for, unchanged: ten tasks per group,
    one group routed to the junior agent every time."""
    harness = MultiAgentRunHarness()
    for _ in range(10):
        harness.record_routing(route="junior", demographic="b")
    for _ in range(10):
        harness.record_routing(route="senior", demographic="a")
    result = DelegationRoutingAuditor().analyze(*harness.as_delegation_inputs())
    assert result.cramers_v == 1.0
    assert result.is_significant is True
    assert result.p_value == pytest.approx(1.082508822446903e-05)


# ==========================================================================
# FACE 3. A model name that renders as nothing.
# ==========================================================================

#: ZERO WIDTH SPACE, ZERO WIDTH NON-JOINER, BYTE ORDER MARK, SOFT HYPHEN. Every
#: one is Unicode category Cf, which ``str.strip()`` does not remove.
_INVISIBLE_NAME = "".join(chr(cp) for cp in (0x200B, 0x200C, 0xFEFF, 0x00AD))


def test_a_blank_rendering_model_name_is_not_a_name():
    """Before: returned the four characters, so it counted as a reported name."""
    assert MI.clean_model_name(_INVISIBLE_NAME) is None
    assert MI.clean_model_name(chr(0x2060)) is None, "WORD JOINER alone"
    assert MI.clean_model_name(f" {chr(0x200B)} ") is None, "mixed with real whitespace"


def test_an_endpoint_that_named_nothing_is_not_recorded_as_having_named_something():
    """The full before-state: n_reporting 4 of 4, is_consistent True, no warning."""
    responses = [{"reported_model": _INVISIBLE_NAME} for _ in range(4)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        identity = MI.identity_from_run(responses, configured_model="qwen3:8b")
    assert identity.n_responses == 4
    assert identity.n_reporting == 0
    assert identity.reported_names == ()
    assert identity.reported_model is None
    assert identity.is_consistent is None, "not True, and not False either"
    assert "carried a model name" in identity.coverage_statement
    assert [w for w in caught if "carried a model name" in str(w.message)], (
        "the 'nobody reported a name' warning stopped firing, which is what made "
        "the invisible name look like an identity"
    )


def test_the_comparison_does_not_claim_the_same_model_answered():
    """Before: state='same', basis='endpoint_report', needs_retest=False, over a
    name nobody could see. That is the claim this module exists to refuse."""
    left = MI.identity_from_run(
        [{"reported_model": _INVISIBLE_NAME} for _ in range(4)], configured_model="qwen3:8b"
    )
    right = MI.identity_from_run(
        [{"reported_model": _INVISIBLE_NAME} for _ in range(4)], configured_model="qwen3:8b"
    )
    comparison = MI.compare_model_identity(left, right)
    assert comparison.state != "same"
    assert comparison.basis == "configuration_only"
    assert "reported_model" in comparison.not_compared


# OVER-CORRECTION CONTROL for face 3.


def test_a_real_model_name_is_untouched():
    identity = MI.identity_from_run(
        [{"reported_model": "qwen3:8b"} for _ in range(4)], configured_model="qwen3:8b"
    )
    assert identity.n_reporting == 4
    assert identity.reported_names == ("qwen3:8b",)
    assert identity.is_consistent is True
    assert MI.compare_model_identity(identity, identity).state == "same"


def test_a_name_that_merely_carries_an_invisible_character_keeps_every_codepoint():
    """NOT sanitised into a name it is not. Only an ENTIRELY invisible name is
    refused; rewriting a recorded identity to make two runs match would be this
    same defect in the mirror, and it would be silent."""
    stray = "qwen3:8b" + chr(0x200B)
    assert MI.clean_model_name(stray) == stray
    left = MI.identity_from_run([{"reported_model": "qwen3:8b"}])
    right = MI.identity_from_run([{"reported_model": stray}])
    assert MI.compare_model_identity(left, right).state != "same", (
        "two different recorded names are a flag, which is the loud direction"
    )


def test_a_non_latin_model_name_is_still_a_name():
    """The gate is visibility, not script."""
    assert MI.clean_model_name("模型-8b") == "模型-8b"
    assert MI.clean_model_name("  glm-4  ") == "glm-4"


# ==========================================================================
# FACE 4. The filter above the guard.
# ==========================================================================


class _StubJudge:
    """A judge that rates from a table. Deterministic, and calls no endpoint."""

    def __init__(self, table, missing_is_nan=False):
        self._table = table
        self._missing_is_nan = missing_is_nan

    def score(self, text):
        return self.score_batch([text])[0]

    def score_batch(self, texts):
        return np.array(
            [
                float("nan") if (text is None or text not in self._table) else self._table[text]
                for text in texts
            ],
            dtype=float,
        )


_A = [f"a{i}" for i in range(10)]
_B = [f"b{i}" for i in range(10)]
_RATINGS = {**{t: 0.80 for t in _A}, **{t: 0.76 for t in _B}}


def _analyze(texts_a, texts_b, table=_RATINGS):
    analyzer = OutputAnalyzer(llm_judge_scorer=_StubJudge(table))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = analyzer.analyze_llm_judge(texts_a, texts_b, "a", "b")
    return result, list(caught)


def test_responses_lost_as_none_texts_reach_the_attrition_guard():
    """Before: assessed=True, delta=+0.040, p=0.0016, is_significant=True, with
    n_supplied_b=2 standing in for a group of ten."""
    result, caught = _analyze(_A, _B[:2] + [None] * 8)
    assert result.assessed is False
    assert result.not_assessed_reason == "differential_unscored"
    assert result.delta is None and result.p_value is None
    assert result.effect_size is None
    assert result.n_supplied_b == 10, (
        "the denominator is what was SUPPLIED; before this it was the count of "
        "survivors, which is what made 2 of 10 read as full coverage"
    )
    assert [w for w in caught if "UNEVEN between the" in str(w.message)]


def test_the_two_descriptions_of_one_world_now_agree():
    """The same loss as NaN SCORES already gave the honest answer. Which one a
    caller got depended on whether the loss arrived as a missing text or a refused
    score, and that is the whole defect."""
    as_texts, _ = _analyze(_A, _B[:2] + [None] * 8)
    table_without_b = {t: v for t, v in _RATINGS.items() if t in _A or t in _B[:2]}
    as_scores, _ = _analyze(_A, _B, table=table_without_b)
    for field in ("assessed", "not_assessed_reason", "n_supplied_a", "n_supplied_b"):
        assert getattr(as_texts, field) == getattr(as_scores, field), field


def test_analyze_all_no_longer_filters_the_evidence_away_before_delegating():
    """THE SIBLING AT THE ENTRY POINT MOST CALLERS USE. ``analyze_all`` ran the
    same filter and assigned back, so each per-metric count of missing responses
    was zero however many had been lost."""
    analyzer = OutputAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = analyzer.analyze_all(
            ["I can help with that. Here are three concrete steps to follow."] * 10,
            ["I can help with that. Here are three concrete steps to follow."] * 2 + [None] * 8,
            "a",
            "b",
        )
    assert results, "analyze_all still returns a row per metric"
    reasons = {r.metric: r.not_assessed_reason for r in results}
    assert all(r.assessed is False for r in results), reasons
    # The metrics whose scorer READ the two surviving responses are the ones that
    # make this pin discriminating: before the fix they measured 10 against 2 and
    # returned assessed=True. The four keyword scorers that read nothing at all in
    # this text (semantic_quality, sentiment, regard, framing) refuse it earlier,
    # with 10 of 10 unscored in BOTH groups, so their honest reason is
    # 'non_finite_scores'. Asserting one reason for all eleven would have been an
    # assumption about what those lexicons match, not a fact about attrition.
    for metric in (
        "refusal_rate",
        "toxicity",
        "helpfulness",
        "stereotype",
        "information_quality",
        "representation",
        "response_length",
    ):
        assert reasons[metric] == "differential_unscored", reasons
    assert set(reasons.values()) <= {"differential_unscored", "non_finite_scores"}, reasons
    assert [w for w in caught if "UNEVEN between the" in str(w.message)]


def test_an_all_missing_group_is_refused_and_the_reason_says_so():
    """A group whose every response was None arrives with an empty score array.
    The verdict was already a refusal; what was missing was the sentence, and the
    supplied count, which read 0 for ten responses that were supplied."""
    result, caught = _analyze(_A, [None] * 10)
    assert result.assessed is False
    assert result.not_assessed_reason == "empty_input"
    assert result.n_supplied_b == 10 and result.n_scored_b == 0
    assert [w for w in caught if "never produced (None)" in str(w.message)]


# OVER-CORRECTION CONTROLS for face 4.


def test_an_even_and_small_loss_is_still_measured_and_disclosed():
    """The middle case, which is the one a guard like this gets wrong by refusing
    everything. One response missing from each group of ten: attrition is even, so
    the comparison RUNS and the coverage travels with it."""
    result, caught = _analyze(_A[:9] + [None], _B[:9] + [None])
    assert result.assessed is True
    assert result.group_a_value == pytest.approx(0.80)
    assert result.group_b_value == pytest.approx(0.76)
    assert result.n_supplied_a == 10 and result.n_scored_a == 9
    assert result.n_supplied_b == 10 and result.n_scored_b == 9
    assert [w for w in caught if "EXCLUDED" in str(w.message)], (
        "measured on a subset must never be readable as measured on the whole"
    )


def test_a_run_with_no_missing_responses_is_completely_unchanged():
    """The control that matters most: the ordinary path pays nothing for this."""
    result, caught = _analyze(_A, _B)
    assert result.assessed is True
    assert result.group_a_value == pytest.approx(0.80)
    assert result.group_b_value == pytest.approx(0.76)
    assert result.delta == pytest.approx(0.04)
    assert result.n_supplied_a == 10 and result.n_supplied_b == 10
    assert result.n_scored_a is None and result.n_scored_b is None, (
        "None means full coverage; a number here would read as a subset"
    )
    assert not [w for w in caught if "never produced" in str(w.message)]
    assert result.p_value is not None and not math.isnan(result.p_value)
