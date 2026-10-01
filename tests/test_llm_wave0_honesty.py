"""Wave 0 of the LLM fairness honesty work (LF-01, LF-03, LF-04, LF-06).

Every test here pins a defect that was live on 2026-09-09 and reproduced by
execution before the fix:

* LF-06  ``OutputAnalyzer._compare`` returned ``delta=0.0, p_value=1.0,
         is_significant=False`` for EMPTY input and for groups with a single
         sample: a clean no-disparity finding for a comparison that never ran.
* LF-04  ``CounterfactualTester.run_test`` hard-coded ``temperature=0.0`` and
         no sampling parameter reached the request, so a run never reflected
         the deployment's sampling and never recorded what it used.
* LF-03  ``LLMJudgeScorer`` accepted the system under test as its own judge.
* LF-01  the noise floor was computed from the wrong series; the new
         ``noise_floor_from_runs`` consumes repeated identical-prompt responses.

Each honesty case is paired with a control on healthy input, so the fix cannot
be achieved by refusing everything.
"""

from __future__ import annotations

import math
import warnings

import pytest

from vfairness.llm import (
    CounterfactualTester,
    IntersectionalAnalyzer,
    IntersectionalGroup,
    LLMApiProxy,
    LLMJudgeScorer,
    OutputAnalyzer,
    judge_is_subject,
    noise_floor_from_runs,
)
from vfairness.llm.scorers import scorer_status

POSITIVE = "great excellent wonderful fantastic helpful clear"
NEGATIVE = "terrible awful horrible poor useless unclear"


# ---------------------------------------------------------------------------
# LF-06: OutputAnalyzer three states
# ---------------------------------------------------------------------------


class TestOutputAnalyzerThreeStates:
    def test_empty_input_is_not_assessed_not_zero(self):
        analyzer = OutputAnalyzer()
        with pytest.warns(RuntimeWarning):
            r = analyzer.analyze_sentiment([], [POSITIVE] * 5)
        assert r.assessed is False
        assert r.not_assessed_reason == "empty_input"
        assert r.delta is None
        assert r.p_value is None
        assert r.is_significant is None
        assert r.effect_size is None
        assert r.group_a_value is None and r.group_b_value is None
        assert r.effect_size_interpretation == "not_assessed"
        assert r.sample_size == 0

    def test_single_sample_measures_delta_but_not_the_test(self):
        analyzer = OutputAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.analyze_sentiment([POSITIVE], [NEGATIVE] * 5)
        assert r.assessed is True
        assert r.delta is not None and r.delta > 0  # the means ARE measured
        assert r.p_value is None  # the test is NOT
        assert r.is_significant is None
        assert r.effect_size is None
        assert r.not_assessed_reason == "fewer_than_2_samples_per_group"

    def test_control_healthy_input_still_tests(self):
        analyzer = OutputAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = analyzer.analyze_sentiment([POSITIVE] * 8, [NEGATIVE] * 8)
        assert r.assessed is True
        assert r.p_value is not None and 0.0 <= r.p_value <= 1.0
        assert r.is_significant is True
        assert r.delta > 0

    def test_analyze_all_on_empty_input_reports_nothing_as_a_finding(self):
        analyzer = OutputAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyzer.analyze_all([], [], correction_method="benjamini_hochberg")
        assert len(results) == 11
        assert all(r.assessed is False for r in results)
        assert all(r.p_value is None and r.is_significant is None for r in results)
        # Serialisable, and None survives the trip.
        d = results[0].to_dict()
        assert d["p_value"] is None and d["assessed"] is False

    def test_analyze_all_correction_excludes_unassessed_from_the_family(self):
        analyzer = OutputAnalyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyzer.analyze_all(
                [POSITIVE] * 8, [NEGATIVE] * 8, correction_method="bonferroni"
            )
        tested = [r for r in results if r.p_value is not None]
        assert tested, "a healthy comparison must still produce tests"
        n_family = tested[0].metadata.parameters["n_tests_in_family"]
        assert n_family == len(tested)


class TestIntersectionalThreeStates:
    def test_untestable_pairs_do_not_become_no_bias(self):
        groups = IntersectionalGroup.from_attributes({"g": ["a", "b"]})
        outputs = {"a": [POSITIVE], "b": [NEGATIVE]}  # one text each: no test possible
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = IntersectionalAnalyzer().analyze(outputs, groups, metric="sentiment")
        assert res.total_pairs == 1
        assert res.pairwise_results[0].p_value is None
        assert res.n_significant_pairs == 0
        assert res.has_intersectional_bias is None  # not False

    def test_control_real_gap_is_still_found(self):
        groups = IntersectionalGroup.from_attributes({"g": ["a", "b"]})
        outputs = {"a": [POSITIVE] * 8, "b": [NEGATIVE] * 8}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = IntersectionalAnalyzer().analyze(outputs, groups, metric="sentiment")
        assert res.has_intersectional_bias is True
        assert math.isfinite(res.max_disparity) and res.max_disparity > 0


# ---------------------------------------------------------------------------
# LF-04: sampling parameters reach the request and are recorded
# ---------------------------------------------------------------------------


class _RecordingProxy:
    """Stands in for LLMApiProxy; records every send_prompt call."""

    def __init__(self, reply=lambda prompt: f"answer about {prompt[:10]}"):
        self.calls = []
        self._reply = reply

    def send_prompt(self, prompt, system_prompt=None, temperature=0.0, top_p=None, seed=None):
        self.calls.append(
            {"prompt": prompt, "temperature": temperature, "top_p": top_p, "seed": seed}
        )
        return {"text": self._reply(prompt), "latency_ms": 1.0, "token_count": 3}


class TestSamplingReachesTheRequest:
    def test_constructor_sampling_is_used_and_recorded(self):
        proxy = _RecordingProxy()
        tester = CounterfactualTester(proxy=proxy, n_runs=2, temperature=0.8, top_p=0.9, seed=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = tester.run_test("Assess {name}.", {"name": ["James", "Jamal"]})
        assert proxy.calls, "no requests were made"
        assert all(c["temperature"] == 0.8 for c in proxy.calls)
        assert all(c["top_p"] == 0.9 and c["seed"] == 7 for c in proxy.calls)
        assert res.sampling == {"temperature": 0.8, "top_p": 0.9, "seed": 7}
        assert res.metadata.parameters["sampling"]["temperature"] == 0.8

    def test_per_call_override_wins(self):
        proxy = _RecordingProxy()
        tester = CounterfactualTester(proxy=proxy, n_runs=2, temperature=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = tester.run_test("Assess {name}.", {"name": ["A", "B"]}, temperature=1.2)
        assert all(c["temperature"] == 1.2 for c in proxy.calls)
        assert res.sampling["temperature"] == 1.2

    def test_default_is_still_deterministic_zero(self):
        proxy = _RecordingProxy()
        tester = CounterfactualTester(proxy=proxy, n_runs=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = tester.run_test("Assess {name}.", {"name": ["A", "B"]})
        assert all(c["temperature"] == 0.0 for c in proxy.calls)
        assert res.sampling == {"temperature": 0.0, "top_p": None, "seed": None}

    def test_invalid_sampling_is_refused(self):
        proxy = _RecordingProxy()
        with pytest.raises(ValueError):
            CounterfactualTester(proxy=proxy, n_runs=2, temperature=-0.1)
        with pytest.raises(ValueError):
            CounterfactualTester(proxy=proxy, n_runs=2, top_p=1.5)

    # The proxy vets its endpoint through the egress guard at construction (a
    # hostname must resolve), so the tests use a loopback literal that nothing
    # listens on. No request is ever sent: _build_payload is pure, and the
    # validation tests raise before the network.
    _LOOPBACK = "http://127.0.0.1:9/v1/chat/completions"

    def test_proxy_payload_carries_top_p_and_seed_per_format(self):
        openai = LLMApiProxy(endpoint_url=self._LOOPBACK, api_format="openai", allow_loopback=True)
        body = openai._build_payload("hi", None, 0.7, 64, top_p=0.9, seed=3)
        assert body["top_p"] == 0.9 and body["seed"] == 3 and body["temperature"] == 0.7

        anthropic = LLMApiProxy(
            endpoint_url=self._LOOPBACK, api_format="anthropic", allow_loopback=True
        )
        body = anthropic._build_payload("hi", None, 0.7, 64, top_p=0.9, seed=3)
        assert body["top_p"] == 0.9
        assert "seed" not in body  # the Anthropic API has no seed parameter

        custom = LLMApiProxy(endpoint_url=self._LOOPBACK, api_format="custom", allow_loopback=True)
        body = custom._build_payload("hi", None, 0.7, 64, top_p=0.9, seed=3)
        assert body["top_p"] == 0.9 and body["seed"] == 3

        body = openai._build_payload("hi", None, 0.0, 64)
        assert "top_p" not in body and "seed" not in body  # None is omitted, not sent

    def test_proxy_validates_top_p_and_seed(self):
        proxy = LLMApiProxy(endpoint_url=self._LOOPBACK, api_format="openai", allow_loopback=True)
        with pytest.raises(ValueError, match="top_p"):
            proxy.send_prompt("hi", top_p=0.0)
        with pytest.raises(ValueError, match="seed"):
            proxy.send_prompt("hi", seed="7")  # type: ignore[arg-type]

    def test_all_empty_responses_give_no_verdict_not_a_clean_one(self):
        proxy = _RecordingProxy(reply=lambda prompt: "")
        tester = CounterfactualTester(proxy=proxy, n_runs=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = tester.run_test("Assess {name}.", {"name": ["A", "B"]})
        assert res.is_significant is None
        assert res.disparity_metrics["sentiment_delta"] is None
        assert res.disparity_metrics["data_quality"] in ("no_data", "insufficient")

    def test_compute_disparity_no_data_is_none(self):
        tester = CounterfactualTester(proxy=_RecordingProxy(), n_runs=2)
        d = tester.compute_disparity([], [POSITIVE])
        assert d["data_quality"] == "no_data"
        assert d["sentiment_delta"] is None and d["cosine_similarity"] is None
        # control
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = tester.compute_disparity([POSITIVE] * 3, [NEGATIVE] * 3)
        assert d["data_quality"] == "good"
        assert d["sentiment_delta"] is not None and d["sentiment_delta"] > 0


# ---------------------------------------------------------------------------
# LF-03: the judge is not the defendant
# ---------------------------------------------------------------------------


class TestJudgeIsNotTheSubject:
    def test_same_endpoint_and_model_is_refused(self):
        assert judge_is_subject(
            "https://api.example.com/v1/chat/completions/",
            "gpt-4o",
            "http://API.example.com/v1/chat/completions",
            "GPT-4o",
        )
        with pytest.raises(ValueError, match="system under test"):
            LLMJudgeScorer(
                endpoint_url="https://api.example.com/v1/chat/completions",
                model_name="gpt-4o",
                subject_endpoint_url="https://api.example.com/v1/chat/completions",
                subject_model_name="gpt-4o",
            )

    def test_same_host_different_model_is_allowed(self):
        assert not judge_is_subject(
            "http://localhost:11434/v1/chat/completions",
            "qwen3:8b",
            "http://localhost:11434/v1/chat/completions",
            "gemma4:12b",
        )
        LLMJudgeScorer(
            endpoint_url="http://localhost:11434/v1/chat/completions",
            model_name="qwen3:8b",
            subject_endpoint_url="http://localhost:11434/v1/chat/completions",
            subject_model_name="gemma4:12b",
            allow_loopback=True,
        )

    def test_no_model_names_on_a_shared_endpoint_counts_as_the_same_model(self):
        assert judge_is_subject("https://x.test/v1", None, "https://x.test/v1", None)

    def test_no_subject_given_means_no_refusal(self):
        assert not judge_is_subject("https://x.test/v1", "m", None, None)

    def test_judge_tier_is_unvalidated_not_production(self):
        assert scorer_status()["llm_judge"]["quality"] == "unvalidated"


# ---------------------------------------------------------------------------
# LF-01: the noise floor from repeated identical prompts
# ---------------------------------------------------------------------------


class TestNoiseFloorFromRuns:
    def _runs(self, text, n):
        return [text] * n

    def test_constant_runs_give_zero_floor_and_a_systematic_gap(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"James": self._runs(POSITIVE, 6), "Jamal": self._runs(NEGATIVE, 6)},
                metrics=["sentiment", "response_length"],
                sampling={"temperature": 0.0, "top_p": None, "seed": None},
            )
        assert out["state"] == "measured" and out["available"] is True
        assert out["series_kind"] == "per_run_responses"
        assert out["sampling"]["temperature"] == 0.0
        v = out["variants"]["James"]
        assert v["n_valid"] == 6 and v["state"] == "measured_with_limitation"  # 6 < 25
        assert v["metrics"]["sentiment"]["profile"]["noise_floor"] == 0.0
        cmp = out["metrics"]["sentiment"]["comparisons"][0]
        assert cmp["variant"] == "Jamal" and cmp["state"] == "measured"
        assert cmp["disparity_noise_floor"] == 0.0
        assert cmp["observed"] > 0 and cmp["exceeds_noise"] is True
        assert "scorer_quality" in out["metrics"]["sentiment"]

    def test_noisy_runs_with_equal_means_do_not_exceed_noise(self):
        mixed = [POSITIVE, NEGATIVE, POSITIVE, NEGATIVE, POSITIVE, NEGATIVE, POSITIVE, NEGATIVE]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": mixed, "B": list(reversed(mixed))}, metrics=["sentiment"]
            )
        cmp = out["metrics"]["sentiment"]["comparisons"][0]
        assert cmp["disparity_noise_floor"] > 0
        assert abs(cmp["observed"]) < 1e-9
        assert cmp["exceeds_noise"] is False

    def test_variant_with_one_valid_response_is_could_not_check(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": self._runs(POSITIVE, 5), "B": [NEGATIVE, "", None, "  "]},
                metrics=["sentiment"],
            )
        assert out["variants"]["B"]["state"] == "could_not_check"
        assert out["variants"]["B"]["n_valid"] == 1
        assert out["variants"]["B"]["metrics"] == {}
        cmp = out["metrics"]["sentiment"]["comparisons"][0]
        assert cmp["state"] == "could_not_check"
        assert cmp["observed"] is None and cmp["exceeds_noise"] is None
        assert out["state"] == "could_not_check" and out["available"] is False

    def test_no_variants_is_could_not_check(self):
        out = noise_floor_from_runs({})
        assert out["available"] is False and out["state"] == "could_not_check"
        assert out["reason"] == "no_variants"

    def test_unknown_metric_is_refused(self):
        with pytest.raises(ValueError, match="Unknown metrics"):
            noise_floor_from_runs({"A": [POSITIVE] * 3}, metrics=["helpfulness"])

    def test_enough_runs_is_measured_without_limitation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": self._runs(POSITIVE, 25), "B": self._runs(POSITIVE, 25)},
                metrics=["response_length"],
            )
        assert out["variants"]["A"]["state"] == "measured"
        assert out["variants"]["A"]["reason"] is None
        assert out["required_runs"] == 25
        cmp = out["metrics"]["response_length"]["comparisons"][0]
        assert cmp["observed"] == 0.0 and cmp["exceeds_noise"] is False

    def test_result_is_json_serialisable(self):
        import json

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs({"A": [POSITIVE] * 3, "B": [NEGATIVE] * 3})
        json.dumps(out)  # must not raise
        assert set(out["metrics"]) == {"sentiment", "toxicity", "refusal_rate", "response_length"}


class TestNoiseFloorEquivalence:
    def test_sesoi_adds_a_tost_verdict_per_comparison(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": [POSITIVE] * 8, "B": [NEGATIVE] * 8}, metrics=["sentiment"], sesoi=0.2
            )
        cmp = out["metrics"]["sentiment"]["comparisons"][0]
        assert "equivalence" in cmp
        assert cmp["equivalence"]["verdict"] in {
            "bias_detected",
            "fairness_confirmed",
            "below_practical_significance",
            "undetermined",
            "could_not_check",
        }
        assert cmp["equivalence"]["sesoi"] == 0.2

    def test_without_sesoi_no_equivalence_block(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = noise_floor_from_runs(
                {"A": [POSITIVE] * 3, "B": [NEGATIVE] * 3}, metrics=["sentiment"]
            )
        assert "equivalence" not in out["metrics"]["sentiment"]["comparisons"][0]


class TestEmptyInputNeverReachesAScorer:
    """W0A (2026-09-09), found on the deployed consumer, not locally.

    ``analyze_all`` used to call every ``analyze_*`` method before any
    three-state logic ran, so each scorer received an empty list. The
    alt-profanity-check toxicity fallback, which is the ACTIVE toxicity scorer
    on the deployment host, raises ``ValueError: Found array with 0 sample(s)`` from
    scikit-learn's TfidfTransformer on empty input. Empty input therefore
    produced an exception on that host and an honest not-assessed result on a
    host with Detoxify installed: the same call, two different contracts.
    """

    def test_empty_input_returns_not_assessed_without_calling_any_scorer(self):
        class ExplodingScorer:
            """Stands in for any model-backed scorer. Must never be called."""

            def score(self, text):  # pragma: no cover - must not run
                raise AssertionError("a scorer was asked to score empty input")

            def score_batch(self, texts):  # pragma: no cover - must not run
                raise AssertionError("a scorer was asked to score empty input")

        analyzer = OutputAnalyzer(alpha=0.05)
        # Replace every scorer the analyzer holds, so any call at all fails.
        for name in vars(analyzer):
            if name.endswith("_scorer") and getattr(analyzer, name) is not None:
                setattr(analyzer, name, ExplodingScorer())

        for texts_a, texts_b in (([], ["x"]), (["x"], []), ([], [])):
            with pytest.warns(RuntimeWarning, match="not assessed"):
                results = analyzer.analyze_all(texts_a, texts_b, "A", "B")
            assert results, "an empty comparison must still name its metrics"
            for r in results:
                assert r.assessed is False
                assert r.not_assessed_reason == "empty_input"
                assert r.p_value is None
                assert r.is_significant is None
                assert r.delta is None
                assert r.effect_size is None

    def test_the_empty_metric_list_matches_a_real_run(self):
        """Over-correction control: the shortcut must name exactly the metrics a
        populated run produces, or the empty answer would quietly omit one."""
        analyzer = OutputAnalyzer(alpha=0.05)
        real = analyzer.analyze_all(["a good answer"] * 3, ["a poor answer"] * 3, "A", "B")
        with pytest.warns(RuntimeWarning):
            empty = analyzer.analyze_all([], [], "A", "B")
        assert [r.metric for r in empty] == [r.metric for r in real]
