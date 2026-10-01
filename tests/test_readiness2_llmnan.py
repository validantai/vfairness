"""Readiness 2, lane ``llmnan``: a NaN p-value walked past the three-state guard.

Reproduced by execution on 2026-09-10, before any fix, with one unscored judge
score in a group of six against six scored ones:

    OutputAnalyzer._compare -> p_value=nan, is_significant=False,
                               assessed=True, not_assessed_reason=None,
                               effect_size=nan, "Cohen's d: large"

``OutputAnalysisResult`` already carried ``is_significant: Optional[bool]`` and
``not_assessed_reason`` for exactly this case. The guard was ``is None`` and
never ``isfinite``, so the NaN that this library's own scorers emit on purpose
(``scorers.py`` returns ``float("nan")`` for a failed judge call, an unusable
reply and an unavailable model) routed around both: a comparison whose test
produced no answer was reported as assessed and not significant.

Through ``analyze_all`` it was worse. The unassessed metric also joined the
multiple-comparison family (``n_tests_in_family`` 11 rather than 10), so a test
that produced no answer shrank every other metric's adjusted p-value.

Every refusal pin here is paired with a control that asserts a MEASURED number,
so the fix cannot be reached by refusing everything.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.llm import output_analysis
from vfairness.llm.output_analysis import (
    OutputAnalysisResult,
    OutputAnalyzer,
    _magnitude_label,
)

# Six against six. Fully separated, so the exact Mann-Whitney U two-sided
# p-value is 2/924 = 0.0021645..., well below alpha=0.05.
SIG_A = ["0.90", "0.91", "0.92", "0.93", "0.94", "0.95"]
SIG_B = ["0.10", "0.11", "0.12", "0.13", "0.14", "0.15"]
SIG_P = 0.0021645021645021645
SIG_MEAN_A = 0.925
SIG_MEAN_B = 0.125
SIG_DELTA = 0.8

# Overlapping, so a real medium effect that is genuinely NOT significant.
NS_A = ["0.50", "0.58", "0.44", "0.61", "0.47", "0.55"]
NS_B = ["0.46", "0.52", "0.41", "0.57", "0.43", "0.50"]
NS_P = 0.26149617619114607
NS_MEAN_A = 0.525
NS_MEAN_B = 0.4816666666666667
NS_DELTA = 0.043333333333333335
NS_COHENS_D = 0.688322233038896

# Three distinct values across both groups, so _select_effect_size takes the
# ordinal (Cliff's Delta) branch. That is the branch that raised TypeError for
# a scorer whose score_batch returned a plain Python list.
ORD_A = ["1", "2", "3", "2", "3", "1"]
ORD_B = ["1", "1", "2", "1", "2", "1"]
ORD_P = 0.18985402693947395
ORD_CLIFF = 0.4444444444444444

UNSCORED = "UNSCORED"


class NumericScorer:
    """Reads the score out of the text; refuses to score ``UNSCORED``.

    ``float("nan")`` is what ``LLMJudgeScorer.score`` returns for a judge call
    that did not come back (scorers.py), chosen so the failure cannot be read
    as a clean 0.0.
    """

    def score(self, text: str) -> float:
        return float("nan") if text == UNSCORED else float(text)

    def score_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts], dtype=np.float64)


class ListNumericScorer(NumericScorer):
    """Same scores, returned as a plain Python list rather than an ndarray."""

    def score_batch(self, texts: list[str]):  # type: ignore[override]
        return [self.score(t) for t in texts]


def _analyzer(scorer=None) -> OutputAnalyzer:
    return OutputAnalyzer(alpha=0.05, sentiment_scorer=scorer or NumericScorer())


def _quiet(fn, *args, **kwargs):
    """Run without the below-minimum-sample-size UserWarning in the way."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _run_recording(fn, *args, **kwargs):
    """Return ``(result, [warning messages])``.

    Deliberately NOT ``pytest.warns``. A pin that asserts the warning inside a
    ``pytest.warns`` block fails on the missing WARNING before it ever reaches
    the VERDICT, so it would pass sabotage review while proving nothing about
    ``is_significant``. The verdict is asserted first here; the warning is
    checked afterwards, as a second, separate claim.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ===========================================================================
# Refusal pins
# ===========================================================================


class TestUnscoredValuesAreNotANegativeVerdict:
    def test_a_nan_score_is_not_assessed_rather_than_not_significant(self):
        analyzer = _analyzer()
        r, said = _run_recording(
            analyzer.analyze_sentiment, SIG_A[:-1] + [UNSCORED], SIG_B, "women", "men"
        )

        # The VERDICT first. This is the assertion the defect breaks.
        assert r.is_significant is None, (
            f"is_significant={r.is_significant!r} for a test that produced no answer"
        )
        assert r.assessed is False
        # REASON SHARPENED 2026-09-10 (READINESS-6), verdict unchanged. This
        # fixture drops one of six on side A and none on side B, which is 17
        # percent against 0: DIFFERENTIAL attrition, and that is now its own
        # named reason. The refusal is the same refusal and this test's subject,
        # that an unscored value is not a negative verdict, is untouched.
        #
        # Why the distinction earns its keep: the analyzer used to refuse the
        # whole comparison on a SINGLE non-finite score, which was right while
        # NaN was rare and became wrong once the keyword scorers started
        # honestly answering NaN for text they cannot read. One unreadable
        # response in twenty-five discarded twenty-four real measurements. It
        # now measures the survivors when attrition is small AND EVEN, and
        # refuses when it is uneven, because a response is unreadable for
        # reasons of content and content is the thing under test.
        assert r.not_assessed_reason == "differential_unscored"
        assert r.p_value is None, f"p_value={r.p_value!r} reported as a measurement"
        assert r.group_a_value is None and r.group_b_value is None
        assert r.delta is None
        assert r.effect_size is None
        assert r.effect_size_interpretation == "not_assessed"
        # Then, separately, that the reader was told.
        assert any("Non-finite scores" in m for m in said), said

    def test_the_effect_size_label_is_refused_with_the_number(self):
        """A "large" label attached to a NaN is its own small lie."""
        analyzer = _analyzer()
        r, _ = _run_recording(
            analyzer.analyze_sentiment, SIG_A[:-1] + [UNSCORED], SIG_B, "women", "men"
        )
        assert r.effect_size_interpretation not in ("large", "medium", "small", "negligible"), (
            f"effect_size={r.effect_size!r} was labelled {r.effect_size_interpretation!r}"
        )
        assert r.effect_size_interpretation == "not_assessed"
        assert r.effect_size is None

    def test_an_unassessed_metric_does_not_join_the_correction_family(self):
        analyzer = _analyzer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = analyzer.analyze_all(
                SIG_A[:-1] + [UNSCORED],
                SIG_B,
                "women",
                "men",
                correction_method="benjamini_hochberg",
            )

        by_metric = {r.metric: r for r in results}
        sentiment = by_metric["sentiment"]
        assert sentiment.assessed is False
        assert sentiment.p_value is None and sentiment.is_significant is None

        tested = [r for r in results if r.p_value is not None]
        assert sentiment not in tested

        # Assert the PROPERTY, not a count. This used to read
        # `len(tested) == len(results) - 1`, which silently assumed that exactly
        # ONE metric refuses on this fixture. That held only while the other
        # scorers still answered for text they could not read; once they began
        # refusing honestly, five of eleven ran and the count broke while the
        # property it stood for was never in danger. The subject here is that an
        # unassessed metric must not join the correction family, so say that.
        # AN EMPTY FAMILY IS A STATE, NOT A BROKEN FIXTURE (2026-09-28). This used to
        # read `assert tested, "no metric ran at all; the fixture pins nothing"`, and it
        # was true of the fixture only on an interpreter that HAS transformers: there
        # the regard model reads a numeric string and measures, so the family had one
        # member. Under .venv/bin/python, which has the declared extras and not
        # transformers, every lexicon scorer refuses a numeric string, the injected
        # NumericScorer refuses the UNSCORED text, response_length reads one token on
        # both sides, and NOTHING tests. Measured there: ten refusals plus
        # response_length identical_scores_nothing_to_test.
        #
        # Nothing else can be made to test on this fixture either: the numeric strings
        # ARE the scores, so any second metric given the same texts sees the same
        # unscorable one and refuses for the same reason. Making the texts multi-token
        # only breaks float() in NumericScorer.
        #
        # So assert the SUBJECT, which an empty family pins perfectly well: the family
        # is exactly the rows that carry a p-value, and every row SAYS so. The code
        # reports n_tests_in_family=0 on every row rather than omitting the key, which
        # was itself a fix, so an empty family is distinguishable from an unreported
        # one. The control that a measured number still comes back lives in the other
        # tests of this file, on the same SIG_A/SIG_B without the unscorable text.
        families = {
            r.metadata.parameters.get("n_tests_in_family")
            for r in results
            if "n_tests_in_family" in r.metadata.parameters
        }
        assert families == {len(tested)}, (
            f"{len(tested)} row(s) carry a p-value and the rows report family sizes "
            f"{families}; an unassessed row has joined the family or the size is unsaid"
        )
        unassessed = [r for r in results if not r.assessed]
        assert unassessed, "no metric refused; this fixture cannot test the property"
        for r in unassessed:
            assert r not in tested, f"{r.metric} was unassessed but joined the family"
            assert r.p_value is None, f"{r.metric} is unassessed with a p-value"
        # THE THIRD STATE, which this line collapsed twice. `tested` and
        # `unassessed` do not partition the rows, so `len(results) - len(unassessed)`
        # is not the size of the family. A row can measure both group values and
        # still be unable to run a test, and the dataclass documents that case in
        # so many words: "A result with assessed=True and p_value=None measured the
        # group values but could not run the test". On this fixture response_length
        # is exactly that, reading 1.0 on both sides with not_assessed_reason
        # 'identical_scores_nothing_to_test'. Measured 2026-09-28: 11 rows, 9
        # unassessed, 1 tested, 1 measured-but-untestable.
        #
        # So assert the exhaustive partition and name the third group, which has
        # more teeth than the old subtraction: it fails if a row falls into none
        # of the three, and it fails if a measured-but-untestable row goes silent
        # about why, which is the could-not-check state quietly becoming a pass.
        measured_not_tested = [r for r in results if r.assessed and r.p_value is None]
        for r in measured_not_tested:
            assert r.not_assessed_reason is not None, (
                f"{r.metric} measured both groups, ran no test, and says nothing about "
                "why; a reader cannot tell that from a test that came back clean"
            )
            assert r.is_significant is None, (
                f"{r.metric} carries a significance verdict with no p-value"
            )
        assert len(tested) + len(unassessed) + len(measured_not_tested) == len(results), (
            f"{len(results)} rows do not split into {len(tested)} tested, "
            f"{len(unassessed)} unassessed and {len(measured_not_tested)} "
            "measured-but-untestable, so a row is in a state nobody named"
        )
        for r in tested:
            assert r.metadata.parameters["n_tests_in_family"] == len(tested), (
                f"{r.metric}: family={r.metadata.parameters['n_tests_in_family']!r}, "
                f"tested={len(tested)}"
            )

    def test_the_correction_boundary_refuses_a_nan_p_value_handed_to_it(self, monkeypatch):
        """The DOWNSTREAM boundary, pinned independently of ``_compare``.

        ``analyze_all`` decided family membership with ``p_value is not None``.
        A NaN is not None, so a metric with no answer joined the family and
        shrank every other metric's adjusted p-value. Pinned here by handing
        the boundary the result a regressed ``_compare`` would produce, so the
        boundary is proved on its own rather than on the fix upstream.
        """
        regressed = OutputAnalysisResult(
            group_a="women",
            group_b="men",
            metric="sentiment",
            group_a_value=float("nan"),
            group_b_value=0.125,
            delta=float("nan"),
            effect_size=float("nan"),
            p_value=float("nan"),
            is_significant=False,
            sample_size=6,
            effect_size_interpretation="Cohen's d: large",
        )
        monkeypatch.setattr(
            OutputAnalyzer, "analyze_sentiment", lambda *a, **k: regressed, raising=True
        )
        analyzer = _analyzer()
        results = _quiet(
            analyzer.analyze_all, SIG_A, SIG_B, "women", "men", correction_method="bonferroni"
        )

        sentiment = next(r for r in results if r.metric == "sentiment")
        assert sentiment.is_significant is None, (
            f"is_significant={sentiment.is_significant!r} survived the boundary"
        )
        assert sentiment.p_value is None
        # Same correction as the test above: assert the PROPERTY. The family
        # must be exactly the metrics that produced a p-value, and every member
        # must agree on its size. `len(results) - 1` assumed the NaN-p sentiment
        # was the only refusal in the run, which stopped being true once the
        # other scorers began refusing text they cannot read.
        tested = [r for r in results if r.p_value is not None]
        assert sentiment not in tested, "the NaN p-value joined the family"
        family = {
            r.metadata.parameters["n_tests_in_family"]
            for r in results
            if "n_tests_in_family" in r.metadata.parameters
        }
        assert family == {len(tested)}, (
            f"family size {family} disagrees with the {len(tested)} metric(s) "
            f"that actually produced a p-value"
        )

    def test_a_non_finite_p_value_from_the_test_is_not_a_negative_verdict(self, monkeypatch):
        """The boundary itself, with finite scores: the test comes back with no answer."""
        monkeypatch.setattr(
            output_analysis.stats,
            "mannwhitneyu",
            lambda *a, **k: (float("nan"), float("nan")),
        )
        analyzer = _analyzer()
        r, said = _run_recording(analyzer.analyze_sentiment, SIG_A, SIG_B, "women", "men")

        assert r.is_significant is None, (
            f"is_significant={r.is_significant!r} from p_value={r.p_value!r}"
        )
        assert r.p_value is None
        assert r.not_assessed_reason == "test_returned_no_p_value"
        assert r.effect_size is None
        assert r.effect_size_interpretation == "not_assessed"
        # The group means WERE measured, so they are still reported as numbers.
        assert r.assessed is True
        assert r.group_a_value == pytest.approx(SIG_MEAN_A)
        assert r.group_b_value == pytest.approx(SIG_MEAN_B)
        assert r.delta == pytest.approx(SIG_DELTA)
        assert any("non-finite p-value" in m for m in said), said


# ===========================================================================
# Over-correction controls: measured values, asserted as numbers
# ===========================================================================


class TestMeasuredResultsStillReportExactly:
    def test_a_significant_result_is_still_significant(self):
        analyzer = _analyzer()
        r = _quiet(analyzer.analyze_sentiment, SIG_A, SIG_B, "women", "men")

        assert r.assessed is True
        assert r.not_assessed_reason is None
        assert r.p_value == pytest.approx(SIG_P, rel=1e-9)
        assert r.is_significant is True
        assert r.group_a_value == pytest.approx(SIG_MEAN_A)
        assert r.group_b_value == pytest.approx(SIG_MEAN_B)
        assert r.delta == pytest.approx(SIG_DELTA)
        assert r.effect_size is not None and r.effect_size > 1.0
        assert r.effect_size_interpretation == "Cohen's d: large"
        assert r.sample_size == 6

    def test_a_non_significant_result_is_still_a_measured_negative(self):
        analyzer = _analyzer()
        r = _quiet(analyzer.analyze_sentiment, NS_A, NS_B, "women", "men")

        assert r.assessed is True
        assert r.not_assessed_reason is None
        assert r.p_value == pytest.approx(NS_P, rel=1e-9)
        assert r.is_significant is False
        assert r.group_a_value == pytest.approx(NS_MEAN_A)
        assert r.group_b_value == pytest.approx(NS_MEAN_B)
        assert r.delta == pytest.approx(NS_DELTA)
        assert r.effect_size == pytest.approx(NS_COHENS_D, rel=1e-9)
        assert r.effect_size_interpretation == "Cohen's d: medium"
        assert r.sample_size == 6

    def test_measured_magnitudes_keep_their_labels(self):
        # Cohen's d and Cohen's h thresholds.
        assert _magnitude_label(0.19, 0.2, 0.5, 0.8) == "negligible"
        assert _magnitude_label(0.2, 0.2, 0.5, 0.8) == "small"
        assert _magnitude_label(-0.6, 0.2, 0.5, 0.8) == "medium"
        assert _magnitude_label(0.8, 0.2, 0.5, 0.8) == "large"
        # Cliff's Delta thresholds.
        assert _magnitude_label(0.146, 0.147, 0.33, 0.474) == "negligible"
        assert _magnitude_label(0.4444444444444444, 0.147, 0.33, 0.474) == "medium"
        assert _magnitude_label(0.5, 0.147, 0.33, 0.474) == "large"

    def test_an_unmeasurable_magnitude_gets_no_label(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            assert _magnitude_label(bad, 0.2, 0.5, 0.8) == "not_assessed", (
                f"{bad!r} was labelled {_magnitude_label(bad, 0.2, 0.5, 0.8)!r}"
            )


# ===========================================================================
# Same family, reported as non-blocking: a scorer that returns a list
# ===========================================================================


class TestAScorerThatReturnsAList:
    def test_the_ordinal_branch_no_longer_raises_and_agrees_with_the_array_path(self):
        """Used to raise TypeError: '<' not supported between list and float.

        Only the ordinal branch raised. The binary and continuous branches
        accepted a list by accident, so the crash depended on the data.
        """
        from_list = _quiet(
            _analyzer(ListNumericScorer()).analyze_sentiment, ORD_A, ORD_B, "women", "men"
        )
        from_array = _quiet(
            _analyzer(NumericScorer()).analyze_sentiment, ORD_A, ORD_B, "women", "men"
        )

        assert from_list.p_value == pytest.approx(ORD_P, rel=1e-9)
        assert from_list.effect_size == pytest.approx(ORD_CLIFF, rel=1e-9)
        assert from_list.effect_size_interpretation == "Cliff's Delta: medium"
        assert from_list.is_significant is False
        assert from_list.p_value == pytest.approx(from_array.p_value, rel=1e-12)
        assert from_list.effect_size == pytest.approx(from_array.effect_size, rel=1e-12)

    def test_a_list_scorer_that_could_not_score_is_refused_too(self):
        analyzer = _analyzer(ListNumericScorer())
        r, said = _run_recording(
            analyzer.analyze_sentiment, ORD_A[:-1] + [UNSCORED], ORD_B, "women", "men"
        )
        assert r.is_significant is None, f"is_significant={r.is_significant!r}"
        assert r.assessed is False
        # Same sharpening as the sibling above: one of six on one side only is
        # differential attrition. Still a refusal, now with the reason named.
        assert r.not_assessed_reason == "differential_unscored"
        assert any("Non-finite scores" in m for m in said), said


class TestEvenAttritionIsMeasuredAndDisclosed:
    """The other half of the READINESS-6 change, and the reason it is not just
    a loosening.

    Refusing a whole comparison because ONE response was unreadable throws away
    established signal, which is this audit's own defect mirrored: an assessment
    reads empty when it is not. But comparing only the readable responses is not
    automatically safe either, because a response is unreadable for reasons of
    content, and content is the thing under test.

    So the rule has three cases, and both boundaries are pinned here.
    """

    def test_small_even_attrition_measures_the_survivors_and_says_so(self):
        analyzer = _analyzer()
        # One dropped on EACH side: even, and well inside the bound.
        r, said = _run_recording(
            analyzer.analyze_sentiment,
            SIG_A[:-1] + [UNSCORED],
            SIG_B[:-1] + [UNSCORED],
            "women",
            "men",
        )
        assert r.assessed is True, (
            "an even, tiny loss discarded a comparison that could still be made"
        )
        assert r.not_assessed_reason is None
        assert r.delta is not None and r.p_value is not None
        # The coverage travels with the verdict, so a SUBSET can never be read
        # as a full comparison.
        assert (r.n_scored_a, r.n_scored_b) == (5, 5)
        assert any("EXCLUDED" in m for m in said), said
        assert any("SUBSET" in m for m in said), said

    def test_full_coverage_reports_no_coverage_caveat(self):
        """OVER-CORRECTION CONTROL. A complete run must not acquire a caveat."""
        analyzer = _analyzer()
        r, said = _run_recording(analyzer.analyze_sentiment, SIG_A, SIG_B, "women", "men")
        assert r.assessed is True
        assert (r.n_scored_a, r.n_scored_b) == (None, None)
        assert not any("EXCLUDED" in m for m in said), said

    def test_uneven_attrition_is_refused_and_the_reason_names_why(self):
        analyzer = _analyzer()
        r, said = _run_recording(
            analyzer.analyze_sentiment,
            SIG_A,
            SIG_B[:-3] + [UNSCORED] * 3,
            "women",
            "men",
        )
        assert r.assessed is False
        assert r.not_assessed_reason == "differential_unscored"
        assert any("UNEVEN" in m for m in said), said
        assert any("cannot be used to compare them" in m for m in said), said

    def test_too_few_survivors_is_still_the_old_refusal(self):
        analyzer = _analyzer()
        r, _ = _run_recording(
            analyzer.analyze_sentiment,
            SIG_A[:1] + [UNSCORED] * 5,
            SIG_B[:1] + [UNSCORED] * 5,
            "women",
            "men",
        )
        assert r.assessed is False
        assert r.not_assessed_reason == "non_finite_scores"
