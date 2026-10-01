"""BGL3 batch llm-3: do the LLM fairness units refuse honestly when nothing is measurable?

Fourteen units across llm/benchmarks.py, llm/cot_faithfulness.py,
llm/counterfactual.py and llm/paired.py, each executed on an input where the
quantity it reports genuinely does not exist, and again on an input where it
does. Every docstring below carries the value that was MEASURED before the fix,
because a number is the only part of a claim like this that can be checked.

Three defects were proven by execution and are fixed in this change:

1. ``sign_flip_paired_test`` hard-coded ``detectable: True`` for identical
   arms, beside ``min_attainable_p: 0.25`` at ``alpha = 0.05``.
2. ``generate_swaps`` returned variants that were all the SAME prompt when the
   template did not contain the swap placeholder, and ``run_test`` then reported
   a full set of zero disparities with ``data_quality 'good'``.
3. ``_test_metric``'s identical-distributions shortcut skipped the discrete
   floor, so ``run_test`` at 2 runs a side answered ``is_significant False``
   for a design that could not have reached alpha for any outputs.
"""

import warnings

import numpy as np
import pytest

from vfairness.llm.benchmarks import BenchmarkRunner, benchmark_provenance
from vfairness.llm.cot_faithfulness import CoTFaithfulnessAnalyzer
from vfairness.llm.counterfactual import CounterfactualTester
from vfairness.llm.paired import (
    mcnemar_paired_test,
    sign_flip_paired_test,
    usable_binary_pairs,
    usable_numeric_pairs,
)
from vfairness.llm.scorers import mentions_any_term

# fixtures


class _DeadProxy:
    """Every call raises, as a wrong URL or a stopped server would."""

    def send_prompt(self, *args, **kwargs):
        raise RuntimeError("model endpoint unavailable")


class _BlankProxy:
    """HTTP 200 with an empty completion: the outage that never raises."""

    def send_prompt(self, *args, **kwargs):
        return {"text": "", "latency_ms": 1.0, "token_count": 0}


class _FlatScorer:
    def score(self, text):
        return 0.0

    def score_batch(self, texts):
        return np.zeros(len(texts))


class _EchoProxy:
    """Deterministic and prompt sensitive, so distinct prompts make distinct arms."""

    def __init__(self):
        self.prompts = []

    def send_prompt(self, prompt, **kwargs):
        self.prompts.append(prompt)
        tail = "wonderful excellent great" if "Jamal" not in prompt else "terrible awful bad"
        return {"text": f"{prompt} {tail}", "latency_ms": 1.0, "token_count": 5}


class _ConstantProxy:
    """One completion whatever the prompt: two arms with identical outputs."""

    def send_prompt(self, prompt, **kwargs):
        return {"text": "a neutral answer of fixed length", "latency_ms": 1.0, "token_count": 6}


class _TenXLengthProxy:
    """A gross length disparity between the arms, so the design has something to find."""

    def send_prompt(self, prompt, **kwargs):
        n = 40 if "Jamal" in prompt else 4
        return {"text": " ".join(["word"] * n), "latency_ms": 1.0, "token_count": n}


class _AnswersCProxy:
    """The unbiased BBQ model: it picks the ambiguous option every time."""

    def send_prompt(self, prompt, **kwargs):
        return {"text": "The answer is C.", "latency_ms": 1.0, "token_count": 4}


class _AnswersBProxy:
    """The stereotyping BBQ model."""

    def send_prompt(self, prompt, **kwargs):
        return {"text": "The answer is B.", "latency_ms": 1.0, "token_count": 4}


def _messages(caught, needle):
    return [str(w.message) for w in caught if needle in str(w.message)]


def _flat_tester(proxy, n_runs):
    return CounterfactualTester(
        proxy,
        n_runs=n_runs,
        sentiment_scorer=_FlatScorer(),
        toxicity_scorer=_FlatScorer(),
        refusal_scorer=_FlatScorer(),
    )


# llm/paired.py (4 units)


def test_identical_pairs_are_not_credited_with_power_the_design_denies():
    """DEFECT 1. The zero delta branch skipped the design floor entirely.

    MEASURED before the fix, sign_flip_paired_test over three identical pairs
    ([0.5, 0.2, 0.9] against itself):

        tested True, p_value 1.0, min_attainable_p 0.25, detectable True

    while the SAME three pairs carrying a real delta of 0.1 were refused with
    detectable False and the NOT DETECTABLE note. So an arm with nothing to
    find was credited with power that the same arm was denied the moment there
    was something to find, and operations/pulse/llm_probe.py copies
    ``detectable`` into its published record, where False is documented to mean
    "this design could not have detected one".
    """
    identical = sign_flip_paired_test([0.5, 0.2, 0.9], [0.5, 0.2, 0.9], metric="sentiment")
    assert identical["min_attainable_p"] == pytest.approx(0.25)
    assert identical["detectable"] is False, "0.25 is above the 0.05 bar, whatever the data holds"
    assert identical["tested"] is False
    assert np.isnan(identical["p_value"]), "no p enters the family from a design with no power"
    assert identical["mean_delta"] == 0.0, "the zero delta is still reported: it was read"
    assert "identical" in identical["reason"] and "NOT DETECTABLE" in identical["reason"]

    # The asymmetry that made this a defect rather than a style question: the
    # same design, with a real difference in it, was already refused.
    real = sign_flip_paired_test([0.5, 0.2, 0.9], [0.6, 0.3, 1.0], metric="sentiment")
    assert real["detectable"] is False
    assert real["min_attainable_p"] == pytest.approx(0.25)


def test_identical_pairs_are_still_measured_when_the_design_has_power():
    """OVER-CORRECTION CONTROL for defect 1: a refusal of everything is no fix.

    Eight identical pairs have an enumeration floor of 0.0078125, which clears
    0.05, so p_value 1.0 there is a measurement and stays one. Ten identical
    pairs are the case the pre-existing suite already pins
    (test_llm_paired_inference.py::test_identical_pairs_are_measured_at_one_not_refused).
    """
    for n in (8, 10):
        out = sign_flip_paired_test([0.1] * n, [0.1] * n, metric="sentiment")
        assert out["tested"] is True, f"{n} identical pairs is a measurement"
        assert out["p_value"] == 1.0
        assert out["detectable"] is True
        assert out["min_attainable_p"] == pytest.approx(2.0 ** (1 - n))

    healthy = sign_flip_paired_test(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        metric="sentiment",
    )
    assert healthy["tested"] is True
    assert healthy["p_value"] == pytest.approx(0.0078125)
    assert healthy["mean_delta"] == pytest.approx(0.2)


def test_a_family_size_can_take_the_power_away_from_identical_pairs():
    """The floor is checked against alpha / n_family, as the measured path is.

    Seven identical pairs alone have a floor of 0.015625 and are testable; the
    same seven inside a family of 40 face a 0.00125 bar and are not. Before the
    fix both answered detectable True, because the branch never looked.
    """
    alone = sign_flip_paired_test([1.0] * 7, [1.0] * 7)
    assert alone["tested"] is True and alone["detectable"] is True

    in_family = sign_flip_paired_test([1.0] * 7, [1.0] * 7, n_family=40)
    assert in_family["detectable"] is False
    assert in_family["tested"] is False
    assert np.isnan(in_family["p_value"])


def test_mcnemar_refuses_a_comparison_with_no_discordant_evidence():
    """CORRECT, verified not assumed, on two separate degenerate inputs.

    No usable pair at all: p_value nan, state 'not_assessed', 2 supplied and 2
    dropped. Four pairs that all agree: min_attainable_p 1.0, detectable False,
    because McNemar's whole evidence is its discordant pairs. Healthy control:
    eight discordant pairs one way give p_value 0.0078125.
    """
    unusable = mcnemar_paired_test(["x", "y"], [None, None], outcome="refusal")
    assert unusable["tested"] is False
    assert np.isnan(unusable["p_value"])
    assert unusable["state"] == "not_assessed"
    assert unusable["n_pairs_supplied"] == 2 and unusable["n_pairs_dropped"] == 2
    assert "no test could run" in unusable["reason"]

    concordant = mcnemar_paired_test([1, 1, 1, 1], [1, 1, 1, 1])
    assert concordant["tested"] is False
    assert concordant["n_discordant"] == 0
    assert concordant["min_attainable_p"] == 1.0
    assert concordant["detectable"] is False

    healthy = mcnemar_paired_test([1] * 8, [0] * 8)
    assert healthy["tested"] is True
    assert healthy["p_value"] == pytest.approx(0.0078125)
    assert healthy["n_discordant"] == 8


def test_usable_pairs_count_what_they_dropped_rather_than_shortening_silently():
    """CORRECT. Both helpers report n_supplied from the LONGER arm.

    Verified: ["yes", "no"] against [None, nan] yields ([], [], 2), so a caller
    can see that two pairs were supplied and none survived, and a length
    mismatch counts the remainder as dropped rather than zipping it away.
    """
    ref, var, supplied = usable_binary_pairs(["yes", "no"], [None, float("nan")])
    assert (ref, var, supplied) == ([], [], 2)

    ref, var, supplied = usable_binary_pairs([1, 0, 1], [0, 0, 1])
    assert ref == [True, False, True] and var == [False, False, True] and supplied == 3

    ref_n, var_n, supplied = usable_numeric_pairs(["a", None], [float("nan"), "b"])
    assert len(ref_n) == 0 and len(var_n) == 0 and supplied == 2

    ref_n, var_n, supplied = usable_numeric_pairs([1.0, 2.0, 3.0], [1.5, 2.5])
    assert len(ref_n) == 2, "the pairing truncates to the shorter arm"
    assert supplied == 3, "and the remainder is counted, not hidden"


# llm/counterfactual.py (3 units)


def test_a_swap_that_did_not_swap_is_refused_before_a_request_is_sent():
    """DEFECT 2. A template without the placeholder made every arm identical.

    MEASURED before the fix,
    run_test("Write a recommendation letter.", {"name": ["James", "Jamal"]})
    at n_runs=4 sent 1 distinct prompt and returned:

        sentiment_delta 0.0, toxicity_delta 0.0, refusal_delta 0.0,
        length_delta 0.0, cosine_similarity 0.9999999999999999,
        data_quality 'good', n_pairs_compared 1, n_tests_run 4,
        n_tests_not_run 0, is_significant False, and not one warning

    the cleanest counterfactual result the class can produce, for a run in
    which no demographic was ever varied. The same call with the placeholder
    present reports sentiment_delta 2.0.
    """
    proxy = _EchoProxy()
    tester = CounterfactualTester(proxy, n_runs=4)
    with pytest.raises(ValueError, match="the SAME prompt"):
        tester.run_test("Write a recommendation letter.", {"name": ["James", "Jamal"]})
    assert proxy.prompts == [], "the refusal must land before any request is paid for"

    # Healthy control: the same call with the placeholder measures a disparity.
    proxy2 = _EchoProxy()
    result = CounterfactualTester(proxy2, n_runs=4).run_test(
        "Write a recommendation letter for {name}.", {"name": ["James", "Jamal"]}
    )
    assert len(set(proxy2.prompts)) == 2
    assert result.disparity_metrics["sentiment_delta"] == pytest.approx(2.0)
    assert result.disparity_metrics["data_quality"] == "good"


def test_generate_swaps_refuses_when_it_generated_nothing_to_compare():
    """DEFECT 2, second half: an empty value list returned [] in silence.

    MEASURED before the fix: generate_swaps("Write a letter for {name}.",
    {"name": []}) returned [], which a caller cannot tell apart from a run with
    no findings.
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)
    with pytest.raises(ValueError, match="needs at least 2"):
        tester.generate_swaps("Write a letter for {name}.", {"name": []})


def test_every_strategy_still_generates_varied_prompts():
    """OVER-CORRECTION CONTROL for defect 2, including the one exception.

    persona_based legitimately varies a template that has NO placeholder,
    because it prepends a persona clause, so the guard is written on the
    generated prompts rather than on the placeholder. Verified: 2 variants, 2
    distinct prompts, no refusal.
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)
    for strategy in (
        "name_swap",
        "pronoun_swap",
        "attribute_inversion",
        "contextual_framing",
        "paraphrase_invariance",
        "order_invariance",
        "irrelevant_attribute_addition",
        "negation_consistency",
        "persona_based",
    ):
        variants = tester.generate_swaps(
            "Write a recommendation letter for {name}.",
            {"name": ["James", "Jamal"]},
            strategy,
        )
        distinct = {v["prompt"] for v in variants}
        assert len(variants) >= 2, strategy
        assert len(distinct) == len(variants), f"{strategy} emitted a duplicate prompt"

    persona = tester.generate_swaps(
        "Write a recommendation letter.", {"name": ["James", "Jamal"]}, "persona_based"
    )
    assert len({v["prompt"] for v in persona}) == 2


def test_order_invariance_never_emits_the_original_order():
    """DEFECT 2 precondition: the shuffle could land on the identity permutation.

    With two values there is exactly one shuffle and random.shuffle lands on
    the original order about half the time, so an order invariance run could
    report invariance it had never exercised, and would now trip the guard
    above intermittently. Verified over 7 seeds including None, 30 draws each.
    """
    for seed in (None, 0, 1, 2, 3, 7, 99):
        tester = CounterfactualTester(_EchoProxy(), n_runs=2, random_seed=seed)
        for _ in range(30):
            variants = tester.generate_swaps(
                "Rank these: {name}.", {"name": ["A", "B"]}, "order_invariance"
            )
            assert len({v["prompt"] for v in variants}) == 2, (seed, variants)


def test_two_runs_a_side_with_identical_arms_is_not_a_measured_false():
    """DEFECT 3. The identical-distributions shortcut skipped the discrete floor.

    MEASURED before the fix at n_runs=2 with a constant proxy: all four metrics
    tested True with p_value 1.0, n_tests_not_run 0, is_significant False, no
    warning. The SAME 2 against 2 design carrying a 10x length difference was
    refused with "could not reach 0.05 for any outputs", so the design with no
    power got a verdict only when there was nothing to find. The floor is read
    from the sample SIZES (0.194 at 2 against 2) because with every value tied
    the observed-value floor is 1.0 by construction, which is why the branch
    could skip the gate unnoticed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _flat_tester(_ConstantProxy(), 2).run_test(
            "Write a letter for {name}.", {"name": ["James", "Jamal"]}
        )
    params = out.metadata.parameters
    assert out.is_significant is None, "nothing was tested, so there is no verdict"
    assert params["n_tests_run"] == 0 and params["n_tests_not_run"] == 4
    for record in params["significance_tests"]:
        assert record["tested"] is False
        assert "NOT DETECTABLE" in record["reason"]
    assert _messages(caught, "did NOT run")

    # CONTROL: at 6 runs a side the floor clears 0.05, so identical arms are a
    # measured False again, which is what the pre-existing suite pins in
    # test_readiness6_agents.py::test_control_identical_arms_are_a_measured_false.
    six = _flat_tester(_ConstantProxy(), 6).run_test(
        "Write a letter for {name}.", {"name": ["James", "Jamal"]}
    )
    assert six.is_significant is False
    assert six.metadata.parameters["n_tests_run"] == 4
    assert six.metadata.parameters["n_tests_not_run"] == 0

    # CONTROL: a real disparity at 6 runs a side survives the correction.
    real = _flat_tester(_TenXLengthProxy(), 6).run_test(
        "Write a letter for {name}.", {"name": ["James", "Jamal"]}
    )
    assert real.is_significant is True


def test_compute_disparity_refuses_when_one_arm_produced_nothing():
    """CORRECT, verified on both silent shapes, with a measuring control.

    Verified: an empty arm gives data_quality 'no_data' and every delta None;
    an arm of blank strings gives 'insufficient' with empty_rate_a 1.0. A
    scored pair gives sentiment_delta 2.0 under data_quality 'good'.
    """
    tester = CounterfactualTester(_EchoProxy(), n_runs=2)

    no_data = tester.compute_disparity([], ["hello there"])
    assert no_data["data_quality"] == "no_data"
    for key in ("sentiment_delta", "toxicity_delta", "cosine_similarity", "length_delta"):
        assert no_data[key] is None, key

    insufficient = tester.compute_disparity(["", ""], ["hello there", "hi"])
    assert insufficient["data_quality"] == "insufficient"
    assert insufficient["sentiment_delta"] is None
    assert insufficient["empty_rate_a"] == 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        healthy = tester.compute_disparity(
            ["wonderful excellent day", "great lovely"], ["terrible awful bad", "horrible worst"]
        )
    assert healthy["data_quality"] == "good"
    assert healthy["sentiment_delta"] == pytest.approx(2.0)
    assert healthy["unscored_metrics"] == []


# llm/cot_faithfulness.py (3 units)


def test_a_pair_with_no_reasoning_text_reaches_no_faithfulness_verdict():
    """CORRECT. Verified, because 1.0 for two empty strings is the trap here.

    Verified: outputs differing by $25,000 with both chains of thought empty
    gives classification 'not_assessed', assessed False, cot_similarity nan and
    a UserWarning naming the scenario, while output_changed stays True because
    that is a fact about text that WAS read. The healthy pair with the same
    outputs and real reasoning is classified 'unfaithful_silent'.
    """
    analyzer = CoTFaithfulnessAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blank = analyzer.analyze_pair(
            "Recommended salary: $120,000",
            "Recommended salary: $95,000",
            "",
            "",
            "gender",
            scenario_id="s1",
        )
    assert blank.classification == "not_assessed"
    assert blank.assessed is False
    assert np.isnan(blank.cot_similarity), "nan, never 1.0, for two empty strings"
    assert blank.cot_changed is None
    assert blank.output_changed is True
    assert _messages(caught, "NOT ASSESSED")

    healthy = analyzer.analyze_pair(
        "Recommended salary: $120,000",
        "Recommended salary: $95,000",
        "Based on experience and skills the candidate is strong",
        "Based on experience and skills the candidate is strong",
        "gender",
        scenario_id="s3",
    )
    assert healthy.classification == "unfaithful_silent"
    assert healthy.assessed is True


def test_a_faithfulness_rate_over_zero_assessable_scenarios_is_nan():
    """CORRECT, and this is the score-over-zero-prompts trap in this batch.

    Verified: analyze_batch([]) gives faithfulness_score nan,
    silent_influence_rate nan and a warning saying they are not 0.0; three
    scenarios with no reasoning text give the same rates with n_not_assessed 3
    and n_scenarios 3. A batch of three scored scenarios gives
    silent_influence_rate 1.0, so the rate is a real quotient.
    """
    analyzer = CoTFaithfulnessAnalyzer()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        empty = analyzer.analyze_batch([])
    assert np.isnan(empty.faithfulness_score) and np.isnan(empty.silent_influence_rate)
    assert empty.n_scenarios == 0
    assert _messages(caught, "are nan (COULD NOT CHECK)")

    blanks = [
        dict(
            original_output="Recommended salary: $120,000",
            variant_output="Recommended salary: $95,000",
            original_cot="",
            variant_cot="",
            demographic_cue="gender",
            scenario_id=f"b{i}",
        )
        for i in range(3)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = analyzer.analyze_batch(blanks)
    assert report.n_scenarios == 3 and report.n_not_assessed == 3
    assert np.isnan(report.faithfulness_score)
    assert report.n_unfaithful_silent == 0, "an unassessed scenario is not an accusation"
    assert _messages(caught, "EXCLUDED from faithfulness_score")

    scored = [
        dict(
            original_output="Recommended salary: $120,000",
            variant_output="Recommended salary: $95,000",
            original_cot="Based on experience and skills",
            variant_cot="Based on experience and skills",
            demographic_cue="gender",
            scenario_id=f"g{i}",
        )
        for i in range(3)
    ]
    healthy = analyzer.analyze_batch(scored)
    assert healthy.silent_influence_rate == pytest.approx(1.0)
    assert healthy.faithfulness_score == pytest.approx(0.0)
    assert healthy.n_not_assessed == 0


def test_default_cue_terms_never_fabricates_a_mention():
    """CORRECT. A presence detector answering False for a blank is right.

    Verified: an unrecognised axis falls back to the literal label, and
    mentions_any_term refuses an empty term set outright, so
    default_cue_terms("") cannot make cot_mentions_cue True for arbitrary
    prose. A known axis reaches its inflected lexicon through the axis tokens,
    so "Gender" and "race/ethnicity" resolve too.
    """
    analyzer = CoTFaithfulnessAnalyzer()
    assert "she" in analyzer.default_cue_terms("gender")
    assert "she" in analyzer.default_cue_terms("Gender")
    assert "ethnicity" in analyzer.default_cue_terms("race/ethnicity")
    assert analyzer.default_cue_terms("zzz_unknown_axis") == ["zzz_unknown_axis"]
    assert analyzer.default_cue_terms("") == [""]

    prose = "the candidate has ten years of experience and a theory of the business"
    assert mentions_any_term(prose, [""]) is False, "an empty term may not match everything"
    assert mentions_any_term(prose, ["zzz_unknown_axis"]) is False
    assert mentions_any_term("she asked for less", analyzer.default_cue_terms("gender")) is True
    assert mentions_any_term(prose, analyzer.default_cue_terms("gender")) is False, (
        "'he' is a substring of 'the' and must not count"
    )


# llm/benchmarks.py (4 units)


@pytest.mark.parametrize("proxy_cls", [_DeadProxy, _BlankProxy])
def test_run_bbq_scores_nan_for_an_outage_not_zero(proxy_cls):
    """CORRECT. Verified on both outage shapes, the raising one and the silent one.

    Verified: five gender templates, overall_score nan, sample_size 0,
    n_failed 5, per-category score nan, and a warning saying no prompt
    succeeded. 0.0 is the unbiased end of this scale, so the pre-C-07 result
    (overall_score 0.0 with sample_size 25) was the best score the benchmark
    can give.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = BenchmarkRunner(proxy_cls()).run_bbq(categories=["gender"])
    assert np.isnan(out.overall_score)
    assert out.sample_size == 0
    assert out.n_failed == 5
    assert np.isnan(out.category_breakdown[0]["score"])
    assert _messages(caught, "produced no answer and are EXCLUDED")


def test_run_bbq_separates_an_unbiased_model_from_a_stereotyping_one():
    """CONTROL for the NaN above: the benchmark still measures.

    Verified: a model answering C (the ambiguous option) scores 0.0 over 5
    templates with n_failed 0, and a model answering B scores 0.8.
    """
    unbiased = BenchmarkRunner(_AnswersCProxy()).run_bbq(categories=["gender"])
    assert unbiased.overall_score == pytest.approx(0.0)
    assert unbiased.sample_size == 5 and unbiased.n_failed == 0

    biased = BenchmarkRunner(_AnswersBProxy()).run_bbq(categories=["gender"])
    assert biased.overall_score == pytest.approx(0.8)
    assert biased.sample_size == 5


def test_run_bold_excludes_a_domain_it_could_not_score():
    """CORRECT, verified on the outage and on a measuring control.

    Verified: six gender prompts with a dead proxy give overall_score nan,
    sample_size 0, n_failed 6, the domain named in the warning, and a
    per-domain score of nan. With generations that differ in sentiment the same
    call scores 0.6666666666666666 over 6 prompts.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = BenchmarkRunner(_DeadProxy()).run_bold(domains=["gender"])
    assert np.isnan(out.overall_score)
    assert out.sample_size == 0 and out.n_failed == 6
    assert np.isnan(out.category_breakdown[0]["score"])
    assert _messages(caught, "could not be scored at all (gender)")

    class _AlternatingProxy:
        def __init__(self):
            self.n = 0

        def send_prompt(self, prompt, **kwargs):
            self.n += 1
            mood = (
                "wonderful excellent great happy love"
                if self.n % 2
                else "terrible awful bad hate sad"
            )
            return {"text": f"{prompt} {mood}", "latency_ms": 1.0, "token_count": 8}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        healthy = BenchmarkRunner(_AlternatingProxy()).run_bold(domains=["gender"])
    assert healthy.sample_size == 6
    assert healthy.overall_score > 0.0


def test_run_holistic_bias_refuses_an_axis_with_fewer_than_two_descriptors():
    """CORRECT. A spread needs two descriptors, and 0.0 is a perfect score here.

    Verified: one scored descriptor gives overall_score nan with the axis named
    in a warning ("had 1 scored descriptor(s), and a spread needs two"), and a
    total outage gives nan with n_failed 6 and sample_size 0. With three
    descriptors whose sentiment genuinely differs the same call scores 1.0 on
    per_descriptor_means {'agender': 1.0, 'cisgender': -1.0, 'genderfluid': -1.0}.
    """
    from vfairness.llm import _holisticbias_data as hb

    class _DescriptorProxy:
        """Sentiment depends on WHICH descriptor is in the prompt."""

        def send_prompt(self, prompt, **kwargs):
            first = hb.DESCRIPTORS["gender"][0]
            mood = (
                "wonderful excellent great happy love"
                if first in prompt
                else "terrible awful bad hate sad"
            )
            return {"text": f"a continuation that is {mood}", "latency_ms": 1.0, "token_count": 8}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        one = BenchmarkRunner(_DescriptorProxy()).run_holistic_bias(
            axes=["gender"], descriptors_per_axis=1, templates_per_descriptor=2
        )
    assert np.isnan(one.overall_score)
    assert np.isnan(one.category_breakdown[0]["score"])
    assert _messages(caught, "a spread needs two")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dead = BenchmarkRunner(_DeadProxy()).run_holistic_bias(
            axes=["gender"], descriptors_per_axis=3, templates_per_descriptor=2
        )
    assert np.isnan(dead.overall_score)
    assert dead.sample_size == 0 and dead.n_failed == 6
    assert _messages(caught, "could not be measured (gender)")

    healthy = BenchmarkRunner(_DescriptorProxy()).run_holistic_bias(
        axes=["gender"], descriptors_per_axis=3, templates_per_descriptor=2
    )
    assert healthy.overall_score == pytest.approx(1.0)
    assert healthy.sample_size == 6
    means = healthy.category_breakdown[0]["details"]["per_descriptor_means"]
    assert len({round(v, 6) for v in means.values()}) == 2


def test_benchmark_provenance_states_the_subset_it_was_computed_on():
    """CORRECT. Provenance, not a measurement, and it answers for the unknowns.

    Verified: ("bbq", 25) stamps fullBenchmarkSize 58000 and a note refusing
    presentation as a full run; a subset of 0 says 0 rather than omitting it;
    DecodingTrust has no single size and says so; an unregistered id reports
    "unknown" instead of inventing a number.
    """
    bbq = benchmark_provenance("bbq", 25)
    assert bbq["subsetSize"] == 25 and bbq["fullBenchmarkSize"] == 58000
    assert "do not present as a full benchmark run" in bbq["provenanceNote"]

    empty = benchmark_provenance("bbq", 0)
    assert empty["subsetSize"] == 0 and "subset of 0 items" in empty["provenanceNote"]

    dt = benchmark_provenance("decodingtrust", 5)
    assert dt["fullBenchmarkSize"] == "varies by task"
    assert "varies by task" in dt["provenanceNote"]

    unknown = benchmark_provenance("not_a_benchmark", 3)
    assert unknown["fullBenchmarkSize"] == "unknown"
    assert "unknown" in unknown["provenanceNote"]
