"""An LLM fairness number must name the instrument that produced it.

Two findings from the 2026-08-28 pre-release audit, both about a measurement
whose quality was invisible at the point a reader would look.

F-A: the serialized audit metadata did not record WHICH scorer ran. The
getting-started page presents ``result.metadata`` / ``result.to_dict()`` as the
record you persist, and it was byte-identical whether the 30-word keyword
placeholder or VADER produced the values: the same texts give 1.0/1.0 from one
and 0.6249/0.6249 from the other. In ``analyze_all`` four of the eleven metrics
can come back as exactly 0.0/0.0 with p=1.0 -- a clean no-disparity finding --
and the only live signal was a ``PlaceholderScorerWarning`` that Python's
default filter shows once per process, so results 2..n were silent.

F-B: ``scorer_status()`` graded the keyword regard scorer ``"good"`` while the
same class raises ``PlaceholderScorerWarning``, and while sentiment and toxicity
reported ``"placeholder"`` for the equivalent rung. docs/API_REFERENCE.md
defines "placeholder" as exactly that fallback. The rule is pinned below for
EVERY rung rather than for regard alone.
"""

import warnings

import numpy as np
import pytest

from vfairness.llm import OutputAnalyzer, scorer_status
from vfairness.llm.scorers import (
    METRIC_STATUS_KEYS,
    LLMJudgeScorer,
    PlaceholderScorerWarning,
    _default_scorers,
    scorer_provenance,
)

TEXTS_A = [
    "She is a hardworking professional and a widely respected leader.",
    "The candidate is accomplished, dedicated and highly skilled.",
] * 15
TEXTS_B = [
    "He is lazy and unreliable, and frankly quite dangerous.",
    "The applicant is uneducated and irresponsible.",
] * 15


def _analyze_all_quietly(analyzer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analyzer.analyze_all(TEXTS_A, TEXTS_B)


class TestTheStoredRecordNamesTheInstrument:
    def test_every_metric_records_a_scorer_and_a_quality(self):
        results = _analyze_all_quietly(OutputAnalyzer())

        assert results, "analyze_all returned nothing to inspect"
        for result in results:
            params = result.metadata.parameters
            assert params["scorer"], result.metric
            assert params["scorer_quality"], result.metric

    def test_the_recorded_scorer_is_the_class_that_actually_ran(self):
        analyzer = OutputAnalyzer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyzer.analyze_sentiment(TEXTS_A, TEXTS_B)

        expected = type(analyzer.sentiment_scorer)
        assert result.metadata.parameters["scorer"] == (
            f"{expected.__module__}.{expected.__qualname__}"
        )

    def test_the_recorded_quality_agrees_with_scorer_status(self):
        """Derived from ``scorer_status()`` rather than hardcoded, so the pin
        holds on a core-only install and on one with the extras."""
        status = scorer_status()

        for result in _analyze_all_quietly(OutputAnalyzer()):
            key = METRIC_STATUS_KEYS.get(result.metric)
            if key is None:  # response_length is counted, not scored
                assert result.metadata.parameters["scorer_quality"] == "not_applicable"
                continue
            assert result.metadata.parameters["scorer_quality"] == status[key]["quality"], (
                result.metric
            )

    def test_swapping_the_scorer_changes_the_stored_record(self):
        """The load-bearing property: two installs, or two configurations, must
        not produce the same audit record for different instruments."""

        class StubSentiment:
            def score(self, text):
                return 0.5

            def score_batch(self, texts):
                return np.full(len(texts), 0.5)

        default_result = OutputAnalyzer().analyze_sentiment
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline = default_result(TEXTS_A, TEXTS_B).metadata.parameters
            swapped = (
                OutputAnalyzer(sentiment_scorer=StubSentiment())
                .analyze_sentiment(TEXTS_A, TEXTS_B)
                .metadata.parameters
            )

        assert baseline["scorer"] != swapped["scorer"]
        assert swapped["scorer"].endswith("StubSentiment")

    def test_a_scorer_the_library_did_not_build_is_not_granted_a_tier(self):
        """Three states. An unknown instrument is "unknown", never the tier of
        the default it replaced."""

        class StubSentiment:
            def score_batch(self, texts):
                return np.full(len(texts), 0.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = (
                OutputAnalyzer(sentiment_scorer=StubSentiment())
                .analyze_sentiment(TEXTS_A, TEXTS_B)
                .metadata.parameters
            )

        assert params["scorer_quality"] == "unknown"

    def test_the_record_survives_serialization(self):
        """``to_dict()`` is what reaches a database; the identity has to be in
        there, not only on the live object."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = OutputAnalyzer().analyze_regard(TEXTS_A, TEXTS_B)

        stored = result.to_dict()["metadata"]["parameters"]

        assert stored["scorer"].endswith(type(OutputAnalyzer().regard_scorer).__name__)
        assert stored["scorer_quality"] == scorer_status()["regard"]["quality"]

    def test_multiple_testing_correction_does_not_drop_the_provenance(self):
        """analyze_all rebuilds every result to carry the adjusted p-value, and
        that rebuild constructs a fresh RunMetadata."""
        results = _analyze_all_quietly(OutputAnalyzer())

        for result in results:
            assert "correction_method" in result.metadata.parameters
            assert "scorer" in result.metadata.parameters

    def test_an_empty_run_is_recorded_too(self):
        """The zero result is the record most in need of provenance: it reads as
        'no disparity' while nothing was measured at all."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = OutputAnalyzer().analyze_sentiment([], [])

        params = result.metadata.parameters
        assert params["sample_size_a"] == 0
        assert params["scorer"], "an empty comparison still names the instrument it did not use"

    def test_the_settings_that_were_already_recorded_are_still_recorded(self):
        """Over-correction control: the four original keys are untouched."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = (
                OutputAnalyzer(alpha=0.01).analyze_toxicity(TEXTS_A, TEXTS_B).metadata.parameters
            )

        assert params["alpha"] == 0.01
        assert params["metric"] == "toxicity"
        assert params["sample_size_a"] == len(TEXTS_A)
        assert params["sample_size_b"] == len(TEXTS_B)


class TestProvenanceHelperEdgeCases:
    def test_a_counted_metric_claims_no_tier(self):
        assert scorer_provenance("response_length") == {
            "scorer": "word count (len(text.split()))",
            "scorer_quality": "not_applicable",
        }

    def test_an_unpassed_scorer_is_unrecorded_not_assumed(self):
        assert scorer_provenance("sentiment") == {
            "scorer": "unrecorded",
            "scorer_quality": "unknown",
        }

    def test_an_unrecognised_metric_gets_no_tier(self):
        assert scorer_provenance("not_a_metric", object())["scorer_quality"] == "unknown"

    def test_the_judge_rung_has_no_default_instance_and_is_matched_by_class(self):
        """``llm_judge`` is the one rung with no module default (it needs an
        endpoint), so its tier is resolved from the class instead. A judge the
        library did not define is still "unknown"."""
        judge = LLMJudgeScorer(endpoint_url="https://judge.example.invalid/v1/chat/completions")

        assert scorer_provenance("llm_judge", judge) == {
            "scorer": "vfairness.llm.scorers.LLMJudgeScorer",
            "scorer_quality": scorer_status()["llm_judge"]["quality"],
        }
        assert scorer_provenance("llm_judge", object())["scorer_quality"] == "unknown"


class TestAQualityGradeMatchesTheWarningTheScorerRaises:
    """docs/API_REFERENCE.md: "placeholder" means a keyword fallback is in use.
    The code says the same thing with ``PlaceholderScorerWarning``. Where the
    two disagree, the grade is the one that is wrong: it is what a user reads
    when deciding whether their audit is defensible."""

    @pytest.mark.parametrize("key", sorted(_default_scorers()))
    def test_a_scorer_that_warns_it_is_a_placeholder_is_graded_placeholder(self, key):
        default = _default_scorers()[key]
        try:
            fresh = type(default)()  # the warning latches after the first score
        except TypeError:  # pragma: no cover - a rung needing constructor args
            pytest.skip(f"{type(default).__name__} cannot be constructed without arguments")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fresh.score("A short, entirely unremarkable sentence about a person.")

        warned = any(issubclass(w.category, PlaceholderScorerWarning) for w in caught)
        if warned:
            assert scorer_status()[key]["quality"] == "placeholder", (
                f"{key} raises PlaceholderScorerWarning but scorer_status() grades it "
                f"{scorer_status()[key]['quality']!r}"
            )

    def test_the_regard_rung_is_covered_by_that_rule_here_and_now(self):
        """The specific case the audit found, asserted directly so the rule
        above cannot pass by covering nothing on this install."""
        status = scorer_status()["regard"]

        if "Keyword" in status["scorer"]:
            assert status["quality"] == "placeholder"
        else:  # pragma: no cover - transformers installed
            assert status["quality"] == "production"

    def test_the_tier_vocabulary_is_the_documented_one(self):
        for key, entry in scorer_status().items():
            # "unvalidated" was added for the LLM judge (LF-03, 2026-09-09): an
            # instrument never checked against human ratings is neither
            # production nor a placeholder, and calling it "good" would grade it.
            assert entry["quality"] in {"production", "good", "placeholder", "unvalidated"}, key
