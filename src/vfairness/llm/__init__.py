"""
vfairness.llm: LLM Fairness Testing Module

Provides tools for measuring bias in Large Language Models through:
- Counterfactual prompt testing (demographic swapping)
- Benchmark evaluation (BBQ, BOLD, DecodingTrust)
- Output analysis (sentiment, toxicity, refusal rate)
- Non-determinism management (noise offset calculations)
- Model identity: what the endpoint reported actually answered (LF-14)
"""

from ._base import RunMetadata, SerializableMixin
from .api_proxy import LLMApiProxy
from .benchmarks import BenchmarkResult, BenchmarkRunner
from .cot_faithfulness import CoTFaithfulnessAnalyzer, CoTFaithfulnessResult, FaithfulnessReport
from .counterfactual import CounterfactualResult, CounterfactualTester
from .decodingtrust import DecodingTrustRunner
from .intersectional import IntersectionalAnalyzer, IntersectionalGroup, IntersectionalResult
from .model_identity import (
    IdentityComparison,
    ModelIdentity,
    compare_model_identity,
    describe_model_identity,
    identity_from_run,
)
from .nondeterminism import RUN_METRICS, NoiseProfile, NonDeterminismAnalyzer, noise_floor_from_runs
from .output_analysis import OutputAnalysisResult, OutputAnalyzer
from .scorers import (
    _ALT_PROFANITY_AVAILABLE,
    _HF_PIPELINE_AVAILABLE,
    _VADER_AVAILABLE,
    DEFAULT_FRAMING_SCORER,
    DEFAULT_HELPFULNESS_SCORER,
    DEFAULT_INFORMATION_QUALITY_SCORER,
    DEFAULT_REFUSAL_SCORER,
    DEFAULT_REGARD_SCORER,
    DEFAULT_REPRESENTATION_SCORER,
    DEFAULT_SENTIMENT_SCORER,
    DEFAULT_STEREOTYPE_SCORER,
    DEFAULT_TOXICITY_SCORER,
    FramingScorer,
    HelpfulnessScorer,
    InformationQualityScorer,
    KeywordRegardScorer,
    KeywordSentimentScorer,
    KeywordToxicityScorer,
    LLMJudgeScorer,
    RefusalScorer,
    RepresentationScorer,
    StereotypeScorer,
    TextScorer,
    judge_is_subject,
    scorer_status,
)

# Conditionally import VADERSentimentScorer if vaderSentiment is installed
if _VADER_AVAILABLE:
    from .scorers import VADERSentimentScorer

# Conditionally import AltProfanityCheckScorer if alt-profanity-check is installed
if _ALT_PROFANITY_AVAILABLE:
    from .scorers import AltProfanityCheckScorer

# Conditionally import TransformerRegardScorer if transformers is installed
if _HF_PIPELINE_AVAILABLE:
    from .scorers import TransformerRegardScorer

__all__ = [
    "RunMetadata",
    "SerializableMixin",
    "CounterfactualTester",
    "CounterfactualResult",
    "BenchmarkRunner",
    "BenchmarkResult",
    "DecodingTrustRunner",
    "OutputAnalyzer",
    "OutputAnalysisResult",
    "NonDeterminismAnalyzer",
    "NoiseProfile",
    "noise_floor_from_runs",
    "RUN_METRICS",
    "judge_is_subject",
    "LLMApiProxy",
    # LF-14: what model actually answered, recorded rather than assumed.
    "ModelIdentity",
    "IdentityComparison",
    "identity_from_run",
    "compare_model_identity",
    "describe_model_identity",
    "IntersectionalAnalyzer",
    "IntersectionalResult",
    "IntersectionalGroup",
    "CoTFaithfulnessAnalyzer",
    "CoTFaithfulnessResult",
    "FaithfulnessReport",
    "TextScorer",
    "KeywordSentimentScorer",
    "KeywordToxicityScorer",
    "RefusalScorer",
    "HelpfulnessScorer",
    "StereotypeScorer",
    "KeywordRegardScorer",
    "LLMJudgeScorer",
    "InformationQualityScorer",
    "RepresentationScorer",
    "FramingScorer",
    "DEFAULT_SENTIMENT_SCORER",
    "DEFAULT_TOXICITY_SCORER",
    "DEFAULT_REFUSAL_SCORER",
    "DEFAULT_HELPFULNESS_SCORER",
    "DEFAULT_STEREOTYPE_SCORER",
    "DEFAULT_REGARD_SCORER",
    "DEFAULT_INFORMATION_QUALITY_SCORER",
    "DEFAULT_REPRESENTATION_SCORER",
    "DEFAULT_FRAMING_SCORER",
    "scorer_status",
]
