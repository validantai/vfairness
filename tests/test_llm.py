"""Tests for vfairness.llm module."""

import numpy as np
import pytest

from vfairness.llm.nondeterminism import NoiseProfile, NonDeterminismAnalyzer
from vfairness.llm.output_analysis import OutputAnalysisResult, OutputAnalyzer


class TestOutputAnalyzer:
    """Test output analysis without API dependency."""

    def test_analyze_length(self):
        analyzer = OutputAnalyzer()
        texts_a = ["short reply", "another short one", "brief"]
        texts_b = [
            "this is a much longer response with more words",
            "another lengthy response here",
            "extended reply with detail",
        ]
        result = analyzer.analyze_length(texts_a, texts_b)
        assert isinstance(result, OutputAnalysisResult)
        assert result.group_a_value < result.group_b_value  # Group B has longer responses
        assert result.delta != 0

    def test_analyze_refusal_rate(self):
        analyzer = OutputAnalyzer()
        texts_a = ["Here is the answer", "I cannot help with that", "Sure, the answer is 42"]
        texts_b = ["I'm sorry, I can't assist", "I cannot do that", "I'm unable to help"]
        result = analyzer.analyze_refusal_rate(texts_a, texts_b)
        assert result.group_a_value < result.group_b_value  # Group B refuses more

    def test_analyze_sentiment(self):
        analyzer = OutputAnalyzer()
        texts_a = ["great excellent wonderful fantastic"] * 5
        texts_b = ["terrible awful horrible poor"] * 5
        result = analyzer.analyze_sentiment(texts_a, texts_b)
        assert result.group_a_value > result.group_b_value

    def test_analyze_toxicity(self):
        analyzer = OutputAnalyzer()
        texts_a = ["hello nice to meet you"] * 5
        texts_b = ["stupid idiot hate ugly pathetic"] * 5
        result = analyzer.analyze_toxicity(texts_a, texts_b)
        assert result.group_a_value < result.group_b_value

    def test_analyze_all(self):
        analyzer = OutputAnalyzer()
        texts_a = ["Good response"] * 10
        texts_b = ["Bad response"] * 10
        results = analyzer.analyze_all(texts_a, texts_b, "group_a", "group_b")
        # 11 metrics: semantic_quality, sentiment, toxicity, refusal_rate,
        # helpfulness, stereotype, regard, information_quality,
        # representation, framing, response_length. (LLMJudge is opt-in
        # via an explicit endpoint and is not included here.)
        assert len(results) == 11
        metric_names = {r.metric for r in results}
        assert metric_names == {
            "semantic_quality",
            "sentiment",
            "toxicity",
            "refusal_rate",
            "helpfulness",
            "stereotype",
            "regard",
            "information_quality",
            "representation",
            "framing",
            "response_length",
        }

    def test_empty_inputs(self):
        analyzer = OutputAnalyzer()
        with pytest.warns(RuntimeWarning):
            result = analyzer.analyze_length([], [])
        assert result.sample_size == 0

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            OutputAnalyzer(alpha=0.0)
        with pytest.raises(ValueError):
            OutputAnalyzer(alpha=1.0)


class TestNonDeterminismAnalyzer:
    """Test non-determinism analysis."""

    def test_required_runs_llm(self):
        assert NonDeterminismAnalyzer("llm").required_runs() == 25

    def test_required_runs_agent(self):
        assert NonDeterminismAnalyzer("agent").required_runs() == 50

    def test_invalid_system_type(self):
        with pytest.raises(ValueError, match="Unsupported system_type"):
            NonDeterminismAnalyzer("predictive")

    def test_characterize_noise(self):
        rng = np.random.RandomState(42)
        values = rng.normal(0.5, 0.1, 100)
        analyzer = NonDeterminismAnalyzer("llm")
        profile = analyzer.characterize_noise(values)
        assert isinstance(profile, NoiseProfile)
        assert profile.sample_size == 100
        assert profile.noise_floor > 0
        assert abs(profile.mean - 0.5) < 0.05

    def test_is_significant_after_offset(self):
        analyzer = NonDeterminismAnalyzer("llm")
        assert analyzer.is_significant_after_offset(0.3, 0.05) is True
        assert analyzer.is_significant_after_offset(0.01, 0.05) is False

    def test_bootstrap_ci(self):
        rng = np.random.RandomState(42)
        values = rng.normal(1.0, 0.5, 50)
        analyzer = NonDeterminismAnalyzer("llm")
        # Pass random_state so resampling is deterministic; bootstrap_ci uses its
        # own default_rng, so a global np.random.seed does NOT reach it.
        lo, hi = analyzer.bootstrap_ci(values, n_bootstrap=500, random_state=42)
        assert lo < 1.0 < hi

    def test_characterize_noise_too_few(self):
        analyzer = NonDeterminismAnalyzer("llm")
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.characterize_noise(np.array([]))
        with pytest.raises(ValueError, match="at least 2"):
            analyzer.characterize_noise(np.array([1.0]))

    def test_bootstrap_ci_empty(self):
        analyzer = NonDeterminismAnalyzer("llm")
        with pytest.raises(ValueError, match="empty"):
            analyzer.bootstrap_ci(np.array([]))

    def test_compute_noise_offset(self):
        analyzer = NonDeterminismAnalyzer("llm")
        result = analyzer.compute_noise_offset(0.3, 0.05)
        assert result["observed"] == 0.3
        assert result["noise_floor"] == 0.05
        assert result["exceeds_noise"] is True
        assert result["systematic_offset"] > 0
