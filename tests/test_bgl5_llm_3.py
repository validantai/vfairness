"""BGL5 batch A-llm-3: the pins for the eight overturned grades, plus controls.

An independent audit overturned eight of the twenty three grades in batch
A-llm-3. Seven were code defects and are fixed; the eighth
(``paired.usable_binary_pairs``) was a defect in the EVIDENCE, not in the code:
the row cited a sabotage ("n_supplied from max to min") that leaves its named
test green, because that test never calls the binary helper with arms of
different lengths. Its behaviour is correct and untouched; the missing pin is at
the bottom of this file and the grade was corrected to SEMI-PROVEN.

Every pin here was sabotage-checked: the fix was reverted in the source, the
named test was shown to go RED, and the source was restored and confirmed
byte-identical with ``diff``. Every refusal pin has an over-correction control
beside it asserting that healthy input still gets its REAL measured value, with
the number written out, because a fix that refuses everything passes every
refusal test and destroys the library.

Nothing here leaves the process: the proxy's session is replaced and the
counterfactual tester is handed a stub proxy.
"""

from __future__ import annotations

import json
import logging
import math
import warnings

import numpy as np
import pytest

from vfairness.llm.api_proxy import LLMApiProxy
from vfairness.llm.counterfactual import CounterfactualTester
from vfairness.llm.embedding_bias import EmbeddingBiasDetector
from vfairness.llm.intersectional import IntersectionalAnalyzer, IntersectionalGroup
from vfairness.llm.paired import usable_binary_pairs

# ===========================================================================
# Harnesses
# ===========================================================================


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
        self.calls = 0

    def post(self, *args: object, **kwargs: object) -> _FakeResponse:
        self.calls += 1
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


def _caught(fn, *args, **kwargs):
    """``(return value, warnings)``, with every filter forced to record."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, caught


def _messages(caught) -> list[str]:
    return [str(w.message) for w in caught]


# ===========================================================================
# 1. api_proxy.LLMApiProxy.test_connection
# A 200 carrying no readable generation returned
#   {'ok': False, 'latency_ms': 0.0, 'error': None, 'reported_model': None}
# for four unreadable shapes AND for a genuinely empty generation: five
# byte-identical records, a failure with no reason at all, while send_prompt had
# text_unavailable_reason in hand on the same call. The INFO line said ok=True
# for all five.
# ===========================================================================

_UNREADABLE = [
    ("custom", {"error": "quota exceeded", "code": 429}, "no_text_in_response"),
    ("openai", {}, "no_text_in_response"),
    ("openai", {"choices": [{"message": {"content": None}}]}, "no_text_in_response"),
    ("openai", ["nope"], "response_body_not_an_object"),
]


@pytest.mark.parametrize("api_format,body,expected_reason", _UNREADABLE)
def test_connection_names_why_a_body_could_not_be_read(api_format, body, expected_reason) -> None:
    record = _proxy(api_format, body).test_connection()

    assert record["ok"] is False
    assert record["text_unavailable_reason"] == expected_reason, (
        "the reason send_prompt computed one layer down was dropped at this boundary"
    )
    assert record["error"] and expected_reason in record["error"], (
        f"a failure with no reason is not a finding: {record}"
    )


def test_connection_tells_an_unreadable_body_from_an_empty_generation() -> None:
    """The three states, on the record a caller reads.

    Before: all five of these returned the identical record, so 'the endpoint
    answered and nothing could be read' could not be told from 'the model
    generated nothing'.
    """
    unreadable = _proxy("openai", {}).test_connection()
    empty_generation = _proxy("openai", {"choices": [{"message": {"content": ""}}]})
    empty = empty_generation.test_connection()

    assert unreadable["text_unavailable_reason"] == "no_text_in_response"
    assert empty["text_unavailable_reason"] is None, (
        "a genuinely empty generation was read, so it has no unavailable reason"
    )
    assert empty["error"] and "generation was empty" in empty["error"]
    assert unreadable != empty, "the two states must not be the same record"


def test_connection_log_reports_the_ok_it_returns(caplog) -> None:
    """api_proxy.py formatted 'test_connection: ok=True' before ok was computed."""
    with caplog.at_level(logging.INFO, logger="vfairness.llm.api_proxy"):
        record = _proxy("openai", {}).test_connection()

    assert record["ok"] is False
    lines = [r.getMessage() for r in caplog.records if "test_connection: ok=" in r.getMessage()]
    assert lines, "the connection test logged no outcome at all"
    assert not [line for line in lines if "ok=True" in line], (
        f"the log said ok=True for a run the record calls a failure: {lines}"
    )


def test_control_a_connection_that_answers_is_still_ok(caplog) -> None:
    """OVER-CORRECTION CONTROL, with the measured values written out."""
    body = {"choices": [{"message": {"content": "ok"}}], "model": "gpt-4o"}
    with caplog.at_level(logging.INFO, logger="vfairness.llm.api_proxy"):
        record = _proxy("openai", body).test_connection()

    assert record["ok"] is True
    assert record["error"] is None
    assert record["text_unavailable_reason"] is None
    assert record["reported_model"] == "gpt-4o"
    assert record["latency_ms"] >= 0.0
    assert [r for r in caplog.records if "test_connection: ok=True" in r.getMessage()]


def test_control_a_request_that_never_landed_still_reports_the_transport_error() -> None:
    """The pre-existing subject (BGL2): port 9 refuses at once, no network needed."""
    proxy = LLMApiProxy(
        endpoint_url="http://127.0.0.1:9/v1",
        allow_loopback=True,
        timeout=1,
        max_retries=1,
    )
    record = proxy.test_connection()

    assert record["ok"] is False
    assert record["error"], "a failed connection with no error to show"
    assert record["latency_ms"] < 0, (
        "a failed connection reported a non-negative latency, which reads as a measured round trip"
    )
    assert record["text_unavailable_reason"] == "request_failed"


# ===========================================================================
# 2. api_proxy.LLMApiProxy.send_prompt
# json.loads accepts the bare literals, and the custom extractor coerced any
# int or float with str(value), testing isinstance and never finiteness:
# {"text": NaN} came back as text='nan' with text_unavailable_reason None, three
# characters every presence scorer reads as a model answer.
# ===========================================================================


@pytest.mark.parametrize(
    "raw",
    ['{"text": NaN}', '{"text": Infinity}', '{"text": -Infinity}', '{"output": NaN}'],
)
def test_a_non_finite_number_under_a_text_key_is_not_a_generation(raw) -> None:
    out = _proxy("custom", json.loads(raw)).send_prompt("hi")

    assert out["text"] == ""
    assert out["text_unavailable_reason"] == "no_text_in_response"


@pytest.mark.parametrize("value,expected", [(42, "42"), (3.5, "3.5"), (0, "0")])
def test_control_a_finite_number_under_a_text_key_is_still_read(value, expected) -> None:
    """OVER-CORRECTION CONTROL: a bare number is odd but unambiguous, and stays."""
    out = _proxy("custom", {"text": value}).send_prompt("hi")

    assert out["text"] == expected
    assert out["text_unavailable_reason"] is None


def test_control_a_real_generation_beside_a_non_finite_key_is_still_read() -> None:
    """OVER-CORRECTION CONTROL: the non-finite key is skipped, not the whole body."""
    body = json.loads('{"text": NaN, "output": "Fairness is comparable treatment."}')
    out = _proxy("custom", body).send_prompt("hi")

    assert out["text"] == "Fairness is comparable treatment."
    assert out["text_unavailable_reason"] is None


# ===========================================================================
# 3. counterfactual.CounterfactualTester.compute_disparity
# THE FINDING THIS CAMPAIGN IS ABOUT: a PAIR of rates both stamped 1.0 when one
# arm is empty, so the failed arm cannot be identified, while the adjacent
# branch computes the same two rates correctly from the data.
#   compute_disparity([], ['a real answer'] * 25)
#     -> empty_rate_a 1.0, empty_rate_b 1.0   (side B supplied 25 usable answers)
#   compute_disparity(['a real answer'] * 25, [])
#     -> the byte-IDENTICAL record
# ===========================================================================


def test_each_arms_empty_rate_is_measured_from_that_arm_alone() -> None:
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    a_dead = tester.compute_disparity([], ["a real answer"] * 25)
    b_dead = tester.compute_disparity(["a real answer"] * 25, [])

    assert a_dead["data_quality"] == "no_data" and b_dead["data_quality"] == "no_data"
    assert (a_dead["empty_rate_a"], a_dead["empty_rate_b"]) == (None, 0.0)
    assert (b_dead["empty_rate_a"], b_dead["empty_rate_b"]) == (0.0, None)
    assert a_dead != b_dead, "which arm produced nothing must be readable off the record"
    # The deltas stay refused, which is the half the original row pinned.
    assert a_dead["sentiment_delta"] is None and b_dead["sentiment_delta"] is None


def test_a_blank_arm_keeps_its_measured_rate_beside_the_healthy_one() -> None:
    """The adjacent branch, unchanged, is the reason 1.0/1.0 was not a measurement."""
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    record = tester.compute_disparity(["", ""], ["hello there", "hi"])

    assert record["data_quality"] == "insufficient"
    assert record["empty_rate_a"] == 1.0
    assert record["empty_rate_b"] == 0.0
    assert record["sentiment_delta"] is None


def test_control_two_healthy_arms_report_their_real_rates_and_a_real_delta() -> None:
    """OVER-CORRECTION CONTROL with the measured numbers."""
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        healthy = tester.compute_disparity(
            ["wonderful excellent day", "great lovely"],
            ["terrible awful bad", "horrible worst"],
        )
        partial = tester.compute_disparity(
            ["wonderful excellent day", "", ""],
            ["terrible awful bad", "horrible worst", "bad awful"],
        )

    assert healthy["data_quality"] == "good"
    assert (healthy["empty_rate_a"], healthy["empty_rate_b"]) == (0.0, 0.0)
    assert healthy["sentiment_delta"] == pytest.approx(2.0)
    # Two of three blank on one arm is a measured 2/3, not a stamped 1.0.
    assert partial["empty_rate_a"] == pytest.approx(2.0 / 3.0)
    assert partial["empty_rate_b"] == 0.0


# ===========================================================================
# 4. counterfactual.CounterfactualTester.generate_swaps
# The guard asked a WHOLE RUN question (len(set(prompts)) < 2) where the
# property that has to hold is PER ARM, so a third distinct prompt bought
# silence for two arms that were byte identical to each other.
# ===========================================================================


def test_two_arms_carrying_the_same_prompt_are_refused_even_when_a_third_differs() -> None:
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    with pytest.raises(ValueError, match="BYTE IDENTICAL") as refusal:
        tester.generate_swaps(
            "Rate the {race} candidate.",
            {"race": ["white", "Black"], "gender": ["man", "woman"]},
            "attribute_inversion",
        )

    message = str(refusal.value)
    assert "man" in message and "woman" in message, message
    assert "Rate the {race} candidate." in message, "the colliding prompt must be shown"


def test_a_swap_key_holding_a_repeated_value_is_refused_too() -> None:
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    with pytest.raises(ValueError, match="BYTE IDENTICAL"):
        tester.generate_swaps("Rate {name}.", {"name": ["James", "James", "Jamal"]}, "name_swap")


@pytest.mark.parametrize(
    "strategy",
    [
        "name_swap",
        "pronoun_swap",
        "attribute_inversion",
        "contextual_framing",
        "paraphrase_invariance",
        "order_invariance",
        "irrelevant_attribute_addition",
        "negation_consistency",
        "persona_based",
    ],
)
def test_control_every_strategy_still_generates_one_prompt_per_arm(strategy) -> None:
    """OVER-CORRECTION CONTROL: the per-arm rule is what all nine already satisfy."""
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    variants = tester.generate_swaps(
        "Write a recommendation letter for {name}.", {"name": ["James", "Jamal"]}, strategy
    )

    prompts = [v["prompt"] for v in variants]
    assert len(variants) in (2, 4), strategy
    assert len(set(prompts)) == len(prompts), f"{strategy} emitted a duplicate prompt"


@pytest.mark.parametrize("seed", [None, 0, 1, 2, 3, 7, 99])
@pytest.mark.parametrize("n_values", [2, 3, 4, 5])
def test_control_order_invariance_emits_one_arm_per_distinct_order(seed, n_values) -> None:
    """OVER-CORRECTION CONTROL, and the reason the generator changed with the guard.

    order_invariance draws random permutations, and each draw used to be checked
    against the ORIGINAL order only, so two draws could land on the same
    permutation. Measured over these 7 seeds and 20 draws each at 3, 4 and 5
    values: 4 of the 21 combinations emitted two arms with the byte-identical
    prompt, which the new per-arm guard refuses, so a legitimate design would have
    been refused depending on the seed. The generator now emits one arm per
    DISTINCT order: 0 refusals over the same 21 combinations.
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2, random_seed=seed)
    values = [chr(65 + i) for i in range(n_values)]

    for _ in range(20):
        variants = tester.generate_swaps(
            "Rank these: {name}.", {"name": values}, "order_invariance"
        )
        prompts = [v["prompt"] for v in variants]
        assert len(prompts) >= 2
        assert len(set(prompts)) == len(prompts), (seed, n_values, prompts)


def test_control_a_two_axis_design_with_both_placeholders_is_accepted() -> None:
    """OVER-CORRECTION CONTROL: the multi-key design is fine once it really swaps."""
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    variants = tester.generate_swaps(
        "Rate the {race} {gender} candidate.",
        {"race": ["white", "Black"], "gender": ["man", "woman"]},
        "attribute_inversion",
    )

    assert len(variants) == 4
    assert len({v["prompt"] for v in variants}) == 4


# ===========================================================================
# 5. counterfactual.CounterfactualTester.run_test
# At n_runs=6 the discrete floor gate is cleared, so the arm carrying the
# reference arm's own prompt was credited with four per-metric tests that "ran
# and found nothing" (tested True, p_value 1.0, 'identical distributions'),
# data_quality 'good', n_tests_not_run 0, no warning, and those four entered the
# Benjamini-Hochberg family.
# ===========================================================================


def test_a_self_comparison_never_reaches_a_request() -> None:
    proxy = _EchoProxy()
    tester = CounterfactualTester(proxy, n_runs=6)

    with pytest.raises(ValueError, match="BYTE IDENTICAL"):
        tester.run_test(
            "Rate the {race} candidate.",
            {"gender": ["man", "woman"], "race": ["white", "Black"]},
            "attribute_inversion",
        )

    assert proxy.prompts == [], "the refusal must land before any request is paid for"


def test_run_test_refuses_colliding_arms_a_generator_handed_it() -> None:
    """The guard has to sit where the REQUESTS are paid for, not only in the builder.

    The generate_swaps refusal alone leaves run_test green, so removing the check
    at the call site would be invisible. This subclass hands run_test the shape
    generate_swaps now refuses, which is the only way to reach the second guard,
    and it is what makes that guard pinned rather than decorative.
    """

    class _ColludingTester(CounterfactualTester):
        def generate_swaps(self, template, swap_pairs, strategy="name_swap"):
            return [
                {"demographic": "James", "prompt": "Rate the candidate."},
                {"demographic": "Jamal", "prompt": "Rate the candidate."},
                {"demographic": "Lakisha", "prompt": "Rate this other candidate."},
            ]

    proxy = _EchoProxy()
    with pytest.raises(ValueError, match="BYTE IDENTICAL"):
        _ColludingTester(proxy, n_runs=2).run_test("Rate {name}.", {"name": ["a", "b"]})

    assert proxy.prompts == []


def test_control_a_design_that_really_swaps_still_runs_every_test() -> None:
    """OVER-CORRECTION CONTROL with the measured counts: 4 arms, 3 pairs, 12 tests."""
    proxy = _EchoProxy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CounterfactualTester(proxy, n_runs=6).run_test(
            "Rate the {race} {gender} candidate.",
            {"gender": ["man", "woman"], "race": ["white", "Black"]},
            "attribute_inversion",
        )

    params = result.metadata.parameters
    assert [v["demographic"] for v in result.variants] == ["man", "woman", "white", "Black"]
    assert len({v["prompt"] for v in result.variants}) == 4
    assert params["n_tests_run"] == 12 and params["n_tests_not_run"] == 0
    assert result.disparity_metrics["n_pairs_compared"] == 3
    assert result.is_significant is False, "a measured False, not a could-not-check"


def test_control_a_two_arm_run_still_measures_its_disparity() -> None:
    """OVER-CORRECTION CONTROL: the shape the pre-existing suite pins."""
    proxy = _EchoProxy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CounterfactualTester(proxy, n_runs=4).run_test(
            "Write a recommendation letter for {name}.", {"name": ["James", "Jamal"]}
        )

    assert len(set(proxy.prompts)) == 2
    assert result.disparity_metrics["sentiment_delta"] == pytest.approx(2.0)
    assert result.disparity_metrics["data_quality"] == "good"


# ===========================================================================
# 6. embedding_bias.EmbeddingBiasDetector.weat
# Every comparison in the permutation loop is `value >= abs_observed`, and with
# a NaN observed statistic each one is False, so count stayed 0 and the exact
# branch returned a finite p_value of 0.0: the most extreme value the scale has,
# beside a severity that says could_not_check, and to_dict() carried it out.
# ===========================================================================

_WORDS = [f"w{i}" for i in range(32)]


def _weat(embeddings):
    detector = EmbeddingBiasDetector(embeddings=embeddings)
    return _caught(detector.weat, _WORDS[:8], _WORDS[8:16], _WORDS[16:24], _WORDS[24:])


def _one_nan_among_healthy():
    rng = np.random.default_rng(0)
    emb = {w: rng.normal(size=16) for w in _WORDS}
    emb["w0"] = np.full(16, np.nan)
    return emb


@pytest.mark.parametrize(
    "label,embeddings",
    [
        ("all nan", {w: np.full(16, np.nan) for w in _WORDS}),
        ("all inf", {w: np.full(16, np.inf) for w in _WORDS}),
        ("one nan among healthy", _one_nan_among_healthy()),
    ],
)
def test_a_non_finite_embedding_backend_publishes_no_p_value(label, embeddings) -> None:
    result, caught = _weat(embeddings)

    assert result.severity == "could_not_check", label
    assert not np.isfinite(result.p_value), (
        f"{label}: p_value {result.p_value} was published for a run in which nothing was computed"
    )
    assert not np.isfinite(result.to_dict()["p_value"]), (
        "the consumer boundary must carry the refusal, not a number"
    )
    assert any("no p-value at all" in m for m in _messages(caught)), _messages(caught)


def test_control_a_degenerate_but_finite_backend_keeps_its_measured_p() -> None:
    """OVER-CORRECTION CONTROL, and the case that proves the guard discriminates.

    All-zero vectors are a dead backend too, but the statistic IS computable
    (0.0), so its p is a real 1.0 and must stay a real 1.0. A guard that refused
    on degeneracy rather than on non-finiteness would take this number away.
    """
    result, _ = _weat({w: np.zeros(16) for w in _WORDS})

    assert result.test_statistic == 0.0
    assert result.p_value == 1.0
    assert math.isnan(result.effect_size), "the undefined effect size is still nan"
    assert result.severity == "could_not_check"


def test_control_a_planted_stereotype_is_still_measured_and_graded() -> None:
    """OVER-CORRECTION CONTROL with the measured values."""
    rng = np.random.default_rng(0)
    male, female = rng.normal(0, 1, 16), rng.normal(0, 1, 16)
    emb = {w: male + rng.normal(0, 0.2, 16) for w in _WORDS[:8] + _WORDS[16:24]}
    emb.update({w: female + rng.normal(0, 0.2, 16) for w in _WORDS[8:16] + _WORDS[24:]})

    result, caught = _weat(emb)

    assert result.effect_size == pytest.approx(1.932, abs=1e-3)
    assert result.p_value == pytest.approx(0.000155, abs=1e-5)
    assert result.severity == "critical"
    assert _messages(caught) == []


# ===========================================================================
# 7. intersectional.IntersectionalAnalyzer.analyze
# Two routes reached the record the unknown-metric hoist exists to prevent:
#   a) groups=[] or one group -> total_pairs 0, n_pairs_not_assessed 0, and NO
#      warning, while n_pairs_not_assessed is documented as "0 on a complete
#      run".
#   b) two groups sharing one label -> has_intersectional_bias False,
#      max_disparity 0.0, n_pairs_not_assessed 0, from one cell compared against
#      itself.
# ===========================================================================

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


@pytest.mark.parametrize("n_groups", [0, 1])
def test_a_design_with_no_pair_says_nothing_was_measured(n_groups) -> None:
    groups = _GROUPS_2X2[:n_groups]
    outputs = {g.label: _NEG for g in groups}

    result, caught = _caught(IntersectionalAnalyzer().analyze, outputs, groups)

    assert result.total_pairs == 0
    assert result.has_intersectional_bias is None
    assert math.isnan(result.max_disparity)
    assert any("no pair to compare" in m for m in _messages(caught)), (
        "nothing was measured and nothing said so: total_pairs 0 beside "
        "n_pairs_not_assessed 0, which is documented as a complete run"
    )


def test_two_groups_sharing_a_label_are_not_a_finding_of_no_bias() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups = IntersectionalGroup.from_attributes(
            {"race": ["Black", "Black"], "gender": ["female"]}
        )
    assert [g.label for g in groups] == ["Black_female", "Black_female"]

    result, caught = _caught(IntersectionalAnalyzer().analyze, {"Black_female": _NEG}, groups)

    assert result.total_pairs == 1
    assert result.n_pairs_not_assessed == 1
    assert [r.not_assessed_reason for r in result.pairwise_results] == ["duplicate_group_label"]
    assert result.has_intersectional_bias is None, (
        "a positive claim of no intersectional bias was made from a group compared against itself"
    )
    assert math.isnan(result.max_disparity)
    assert any("carried by more than one" in m for m in _messages(caught)), _messages(caught)


def test_control_a_real_two_by_two_design_still_measures_and_ranks() -> None:
    """OVER-CORRECTION CONTROL with the measured counts."""
    result, caught = _caught(
        IntersectionalAnalyzer().analyze, _FULL_OUTPUTS, _GROUPS_2X2, metric="sentiment"
    )

    assert [g.label for g in _GROUPS_2X2] == [
        "Black_male",
        "Black_female",
        "White_male",
        "White_female",
    ]
    assert result.total_pairs == 6
    assert result.n_pairs_not_assessed == 0
    assert result.n_significant_pairs == 4
    assert result.has_intersectional_bias is True
    assert result.max_disparity == pytest.approx(2.0)
    assert result.most_advantaged is not None and result.most_disadvantaged is not None
    assert result.most_advantaged.label != result.most_disadvantaged.label
    assert [m for m in _messages(caught) if "IntersectionalAnalyzer" in m] == []


def test_control_an_empty_cell_is_still_recorded_rather_than_refused_wholesale() -> None:
    """OVER-CORRECTION CONTROL: the BGL-D behaviour beside the new guards."""
    outputs = dict(_FULL_OUTPUTS, Black_female=[])

    result, caught = _caught(
        IntersectionalAnalyzer().analyze, outputs, _GROUPS_2X2, metric="sentiment"
    )

    assert result.total_pairs == 6
    assert result.n_pairs_not_assessed == 3
    assert {r.not_assessed_reason for r in result.pairwise_results if not r.assessed} == {
        "empty_input"
    }
    assert result.has_intersectional_bias is True, "a finding found is still a finding"
    assert any("NOT ASSESSED" in m for m in _messages(caught))


# ===========================================================================
# 8. paired.usable_binary_pairs: THE PIN THE ROW CLAIMED AND DID NOT HAVE.
# The row cites the sabotage "n_supplied from max to min so a mismatch would
# vanish: 1 failed". Executed 2026-09-27 on the named node id, that change
# inside usable_binary_pairs leaves it GREEN (1 passed), because the named test
# calls this helper only with EQUAL length arms, where min and max agree; the
# red it records belongs to usable_numeric_pairs. The BEHAVIOUR is correct and
# nothing was changed for this row: the grade was corrected to SEMI-PROVEN and
# this is the missing pin.
# ===========================================================================


def test_usable_binary_pairs_counts_the_longer_arm_on_a_length_mismatch() -> None:
    ref, var, supplied = usable_binary_pairs([1, 0, 1], [0, 0])

    assert ref == [True, False] and var == [False, False]
    assert supplied == 3, "the dropped remainder must be visible as a count"

    # The same, with the short arm first: still 3 supplied, never 2.
    ref_r, var_r, supplied_r = usable_binary_pairs([0, 0], [1, 0, 1])
    assert ref_r == [False, False] and var_r == [True, False]
    assert supplied_r == 3


def test_control_usable_binary_pairs_keeps_every_pair_of_a_matched_design() -> None:
    """OVER-CORRECTION CONTROL: equal arms lose nothing and report their length."""
    ref, var, supplied = usable_binary_pairs([1, 0, 1], [0, 0, 1])

    assert ref == [True, False, True]
    assert var == [False, False, True]
    assert supplied == 3
