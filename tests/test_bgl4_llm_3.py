"""BGL4 audit of batch A-llm-3: the overturns, recorded as executable evidence.

Eight of the twenty three grades in batch A-llm-3 were overturned on audit. Each
test below asserts what an HONEST answer would be. They were first recorded as
``xfail(strict=True)`` because the defects were OPEN, and every one of them was
run without the marker first, so the observed value quoted in each docstring is
verbatim.

CLOSED IN BGL5 (2026-09-27). Every marker below has been removed and every test
now asserts the CORRECTED behaviour: the defects are fixed in
``src/vfairness/llm/api_proxy.py``, ``counterfactual.py``, ``embedding_bias.py``
and ``intersectional.py``, each fix carrying a comment with the measured before
and after. The measured-before values stay in the docstrings here, because they
are the evidence that these assertions could once fail. Two tests changed their
MECHANISM and not their subject: a design whose arms carry the same prompt is now
refused by ``ValueError`` before any request is sent, so they assert the refusal
rather than the record it used to produce. The wider pins, the sabotages and the
over-correction controls are in ``tests/test_bgl5_llm_3.py``.

The one test that was PASSING here from the start is the pin that batch A-llm-3
claimed to have and did not: the row for ``usable_binary_pairs`` cites a sabotage
("n_supplied from max to min") that leaves its named test GREEN, because the
named test never calls that helper with arms of different lengths. No code
changed for it; the grade was corrected to SEMI-PROVEN.
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np
import pytest

from vfairness.llm.api_proxy import LLMApiProxy
from vfairness.llm.counterfactual import CounterfactualTester
from vfairness.llm.embedding_bias import EmbeddingBiasDetector
from vfairness.llm.intersectional import IntersectionalAnalyzer, IntersectionalGroup
from vfairness.llm.paired import usable_binary_pairs

# ---------------------------------------------------------------------------
# Harnesses. Nothing leaves the process: the proxy session is replaced, and the
# counterfactual tester is handed a stub proxy.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: object) -> None:
        self._body = body
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._body


class _FakeSession:
    def __init__(self, body: object) -> None:
        self.body = body

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(self.body)


def _proxy(api_format: str, body: object) -> LLMApiProxy:
    proxy = LLMApiProxy(
        endpoint_url="http://127.0.0.1:9/v1/chat/completions",
        api_format=api_format,
        model_name="m",
        allow_loopback=True,
    )
    proxy._session = _FakeSession(body)  # type: ignore[assignment]
    return proxy


class _EchoProxy:
    """Deterministic and prompt sensitive, so distinct prompts make distinct arms."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def send_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        tail = "wonderful excellent great" if "Jamal" not in prompt else "terrible awful bad"
        return {"text": f"{prompt} {tail}", "latency_ms": 1.0, "token_count": 5}


# ==========================================================================
# OVERTURN 1. vfairness.llm.api_proxy.LLMApiProxy.test_connection
# Claimed PROVEN (wave4), evidence tests/test_bgl2_analysis_surfaces.py. Under
# coverage that test executes ONLY the except branch (lines 686 to 693); the
# success path 679 to 685 is never reached. On a 200 whose body carries no
# readable generation the record is
#   {'ok': False, 'latency_ms': 0.0, 'error': None, 'reported_model': None}
# for all four unreadable shapes AND for a genuinely empty generation, so the
# three state reason send_prompt computes one layer down is dropped here.
# ==========================================================================


def test_a_body_that_could_not_be_read_is_not_a_failure_with_no_reason() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    test_connection collapsed could-not-read into a failure with error=None.
    MEASURED before the fix on custom format body
    {'error': 'quota exceeded', 'code': 429}:
        {'ok': False, 'latency_ms': 0.0, 'error': None, 'reported_model': None}
    while send_prompt had text_unavailable_reason='no_text_in_response' in hand
    on the same call. After: error names the reason and the record carries
    text_unavailable_reason='no_text_in_response'.
    """
    result = _proxy("custom", {"error": "quota exceeded", "code": 429}).test_connection()

    assert result["ok"] is False
    assert result.get("error") or result.get("text_unavailable_reason"), (
        "the connection test reports a failure with no reason at all, and the named "
        "reason existed one layer down"
    )


def test_the_connection_log_does_not_claim_ok_for_a_record_that_says_not_ok(caplog) -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    api_proxy.py logged 'test_connection: ok=True, latency=0.0ms' before ok was
    computed, so the log claimed success on exactly the runs whose returned
    record said ok False. MEASURED on openai body {}: the INFO line read
    'test_connection: ok=True, latency=0.0ms' beside a returned record of
    {'ok': False, ..., 'error': None}. After: the line reports the ok that is
    returned, plus the reason.
    """
    with caplog.at_level(logging.INFO, logger="vfairness.llm.api_proxy"):
        result = _proxy("openai", {}).test_connection()

    assert result["ok"] is False
    assert not [r for r in caplog.records if "ok=True" in r.getMessage()], (
        "the log said ok=True for a run the record calls a failure"
    )


# ==========================================================================
# OVERTURN 2. vfairness.llm.api_proxy.LLMApiProxy.send_prompt
# Claimed PROVEN (bgl3). The custom format coerces a bare number under the text
# key with str(value), and json.loads accepts a bare NaN literal, so
# {"text": NaN} comes back as text='nan' with text_unavailable_reason None: a
# three character generation every presence scorer will read, from a value that
# is the absence of a number.
# ==========================================================================


def test_a_not_a_number_under_the_text_key_is_not_a_generation() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    MEASURED before the fix: send_prompt on custom body {'text': NaN} (json.loads
    accepts the bare literal) returned
        {'text': 'nan', 'latency_ms': 0.0, 'token_count': -1,
         'reported_model': None, 'text_unavailable_reason': None}
    A non finite float is not a generation, and the guard tested isinstance only.
    After: text '' with text_unavailable_reason 'no_text_in_response', while a
    finite number under the key is still coerced ({'text': 42} -> '42').
    """
    body = json.loads('{"text": NaN}')
    out = _proxy("custom", body).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] == "no_text_in_response"


# ==========================================================================
# OVERTURN 3. vfairness.llm.counterfactual.CounterfactualTester.generate_swaps
# Claimed PROVEN (bgl3). The guard added that day asks whether ALL the generated
# prompts are identical, which is a whole run question, and the property that has
# to hold is per arm. With two swap keys and a template holding one placeholder,
# attribute_inversion emits 4 variants and 3 distinct prompts: the two arms of
# the unsubstituted key carry the byte identical raw template.
# ==========================================================================


def test_no_two_demographic_arms_may_carry_the_same_prompt() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5. Same subject, new mechanism.

    MEASURED before the fix: generate_swaps('Rate the {race} candidate.',
    {'race': ['white','Black'], 'gender': ['man','woman']}, 'attribute_inversion')
    returned 4 variants and 3 distinct prompts, the 'man' and 'woman' arms both
    carrying 'Rate the {race} candidate.' verbatim, with no warning. The pin's own
    control asserts len(distinct) == len(variants) for the single key case only.

    After: the per-arm check refuses the design outright, so the property is now
    asserted through the refusal rather than over the returned variants. The
    subject is unchanged: no two demographic arms may carry the same prompt.
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)
    with pytest.raises(ValueError, match="BYTE IDENTICAL") as refusal:
        tester.generate_swaps(
            "Rate the {race} candidate.",
            {"race": ["white", "Black"], "gender": ["man", "woman"]},
            "attribute_inversion",
        )

    message = str(refusal.value)
    assert "man" in message and "woman" in message, (
        "the arms that were sent the identical prompt must be named: " + message
    )


# ==========================================================================
# OVERTURN 4. vfairness.llm.counterfactual.CounterfactualTester.run_test
# Claimed PROVEN (bgl3). Same root: the reference arm and one other arm carry the
# same prompt, so _test_metric's identical distributions branch reports
# tested True with p_value 1.0 for all four metrics, data_quality 'good',
# n_tests_not_run 0, and no warning. Measured at n_runs=6, so the discrete floor
# gate the same day added is cleared and cannot catch it.
# ==========================================================================


def test_a_self_comparison_is_never_a_test_that_ran() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5. Same subject, new mechanism.

    MEASURED before the fix: run_test('Rate the {race} candidate.',
    {'gender': ['man','woman'], 'race': ['white','Black']}, 'attribute_inversion')
    at n_runs=6 (so the discrete floor gate is cleared and cannot catch it)
    reported the 'woman' arm as 4 tests tested True, p_value 1.0, reason
    'identical distributions', data_quality 'good', n_tests_not_run 0, for an arm
    whose prompt is byte identical to the reference arm's. Those four entered the
    Benjamini-Hochberg family and raised the bar for the real comparison.

    After: run_test refuses the design before a single request is paid for, so
    there is no such record to credit. The subject is unchanged: a comparison of a
    cell against itself is never a test that ran.
    """
    proxy = _EchoProxy()
    tester = CounterfactualTester(proxy, n_runs=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="BYTE IDENTICAL"):
            tester.run_test(
                "Rate the {race} candidate.",
                {"gender": ["man", "woman"], "race": ["white", "Black"]},
                "attribute_inversion",
            )

    assert proxy.prompts == [], (
        "the refusal must land before any request is paid for, and before any "
        "per-metric record can be written for an arm compared against itself"
    )


# ==========================================================================
# OVERTURN 5. vfairness.llm.counterfactual.CounterfactualTester.compute_disparity
# Claimed PROVEN (bgl3). The deltas are honest. The provenance beside them is a
# template constant: the no_data branch returns empty_rate_a 1.0 AND
# empty_rate_b 1.0 whichever side is dead, so a one sided outage reads as a two
# sided one and the arm that failed cannot be identified. The adjacent
# 'insufficient' branch computes the same two rates correctly, which is what
# shows the pair is not a measurement.
# ==========================================================================


def test_the_empty_rate_of_the_healthy_arm_is_measured_not_stamped() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    MEASURED before the fix: compute_disparity([], ['a real answer'] * 25) returned
    empty_rate_a 1.0 and empty_rate_b 1.0, and the mirrored call with the arms
    swapped returned the byte-identical record, so the arm that failed could not be
    identified. Side B supplied 25 usable responses and its empty rate is 0.0. The
    named pin asserts the deltas in that branch and never looks at the rates.

    After: each rate is measured from its own arm, and an arm that supplied no
    response at all gets None (a rate over zero responses is neither 0.0 nor 1.0,
    the same choice nondeterminism.py makes with nan).
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)
    record = tester.compute_disparity([], ["a real answer"] * 25)

    assert record["data_quality"] == "no_data"
    assert record["empty_rate_a"] is None, (
        "arm A supplied no response at all, so it has no rate to report"
    )
    assert record["empty_rate_b"] == 0.0, (
        "the arm that produced 25 usable responses is reported as 100 percent empty"
    )
    mirrored = tester.compute_disparity(["a real answer"] * 25, [])
    assert (mirrored["empty_rate_a"], mirrored["empty_rate_b"]) == (0.0, None), (
        "the record must say WHICH arm produced nothing, so the two calls differ"
    )


# ==========================================================================
# OVERTURN 6. vfairness.llm.embedding_bias.EmbeddingBiasDetector.weat
# Claimed PROVEN (bgl3) on the all zero vector backend. severity holds on every
# dead backend I tried, including NaN and inf ones. What does not hold is the
# number beside it: with NaN vectors every comparison in the permutation loop is
# False, so count stays 0 and p_value is a finite 0.0, the most significant value
# there is, on a row whose severity says could_not_check. A permutation test that
# enumerates the observed split can never return 0.
# ==========================================================================


def test_a_nan_embedding_backend_does_not_publish_a_p_value_of_zero() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    MEASURED before the fix: weat over an embedding map of all NaN vectors returned
    effect_size nan, test_statistic nan, severity 'could_not_check' and p_value 0.0,
    the most significant value the scale has, which is arithmetically impossible for
    a permutation test that enumerates the observed split. An all-inf backend and a
    single NaN vector among healthy ones did the same. paired.py's own rule is that
    a test that could not run must not put a number on the record.

    After: p_value nan, and the all-zero backend still returns its real 1.0.
    """
    words = [f"w{i}" for i in range(32)]
    detector = EmbeddingBiasDetector(embeddings={w: np.full(16, np.nan) for w in words})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detector.weat(words[:8], words[8:16], words[16:24], words[24:])

    assert result.severity == "could_not_check"
    assert not np.isfinite(result.p_value), (
        f"p_value {result.p_value} was published for a run in which nothing was computed"
    )


# ==========================================================================
# OVERTURN 7. vfairness.llm.intersectional.IntersectionalAnalyzer.analyze
# Claimed PROVEN (bgl3). The unknown metric route was hoisted above the dispatch
# that day. Two other routes reach the same broken record.
#   a) Two supplied values that produce ONE label (a duplicate, or a collision in
#      the "_".join) make analyze compare a cell against ITSELF and answer
#      has_intersectional_bias False, max_disparity 0.0, n_pairs_not_assessed 0.
#   b) groups=[] or a single group, which is exactly what from_attributes now
#      returns for a design with an empty axis, gives total_pairs 0 with
#      n_pairs_not_assessed 0 and NO warning from analyze: the field documented
#      as "0 on a complete run" says the run was complete.
# ==========================================================================


def test_a_cell_compared_against_itself_is_not_a_finding_of_no_bias() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    MEASURED before the fix: from_attributes({'race': ['Black','Black'],
    'gender': ['female']}) returns two groups both labelled 'Black_female', and
    analyze then reported total_pairs 1, n_pairs_not_assessed 0, max_disparity 0.0
    and has_intersectional_bias False from one cell compared against itself, because
    outputs_by_group.get(label) returns the same list for both sides.

    After: the pair is recorded not assessed with reason 'duplicate_group_label',
    max_disparity is nan, has_intersectional_bias is None, and a UserWarning names
    the duplicated label.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups = IntersectionalGroup.from_attributes(
            {"race": ["Black", "Black"], "gender": ["female"]}
        )
        result = IntersectionalAnalyzer().analyze(
            {"Black_female": ["a good answer", "another good answer", "ok"]}, groups
        )

    assert result.has_intersectional_bias is not False, (
        "a positive claim of no intersectional bias was made from a group compared "
        f"against itself (labels {[g.label for g in groups]})"
    )


def test_an_analysis_of_no_pairs_says_so() -> None:
    """BGL4 A-llm-3 overturn, CLOSED in BGL5.

    MEASURED before the fix: analyze({}, []) returned total_pairs 0 with
    n_pairs_not_assessed 0 and emitted NO warning, which is the record the same day's
    unknown metric fix exists to prevent (the field documented as '0 on a complete
    run' said the run was complete). from_attributes hands analyze exactly this empty
    list for a design with an empty axis, and a single group did the same.

    After: a UserWarning naming the group count, and the docstring of
    n_pairs_not_assessed says to read it against total_pairs.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = IntersectionalAnalyzer().analyze({}, [])

    assert result.total_pairs == 0
    assert caught or result.n_pairs_not_assessed > 0, (
        "nothing was measured and nothing said so: total_pairs 0 beside "
        "n_pairs_not_assessed 0, which is documented as a complete run"
    )


# ==========================================================================
# THE PIN THE ROW CLAIMED AND DID NOT HAVE.
# vfairness.llm.paired.usable_binary_pairs, claimed PROVEN with the sabotage
# "n_supplied from max to min so a mismatch would vanish: 1 failed". Executed on
# 2026-09-27: that change inside usable_binary_pairs leaves
# tests/test_bgl3_llm_3.py::test_usable_pairs_count_what_they_dropped_rather_
# than_shortening_silently GREEN (1 passed), because the test calls the binary
# helper only with EQUAL length arms, where min and max agree. The sabotage that
# reddens lives in usable_numeric_pairs. This test is the missing pin: it passes
# today and goes red under that change.
# ==========================================================================


def test_usable_binary_pairs_counts_the_longer_arm_on_a_length_mismatch() -> None:
    ref, var, supplied = usable_binary_pairs([1, 0, 1], [0, 0])

    assert ref == [True, False] and var == [False, False]
    assert supplied == 3, "the dropped remainder must be visible as a count"
