"""Agent verdicts must not be fabricated from measurements that never happened.

Readiness wave 4, agents lane. Three verdicts in ``vfairness.agents`` were
returned from unmeasured inputs, each one indistinguishable from the same
verdict reached on real evidence:

1. ``CorrespondenceTester.four_fifths_rule(0.0, 0.0)`` returned
   ``{"ratio": 1.0, "adverse_impact": False}``: the EEOC four-fifths finding of
   NO ADVERSE IMPACT, byte for byte what a measured parity of 0.5 against 0.5
   returns, for an audit in which nobody at all was selected. A non-finite rate
   went the same way, because ``nan < 0.8`` is False.
2. ``RAGBiasAnalyzer.full_analysis`` reported ``is_retrieval_biased=False`` and
   ``is_output_biased=False`` for a run that retrieved nothing and generated
   nothing, and ``analyze_output([], [])`` returned a hard 0.0, which on that
   scale means "identical outputs". Nothing retrieved is not agreement.
3. ``ActionBiasAnalyzer.analyze_outcomes`` reported ``is_significant=False``
   for a Mann-Whitney test that explicitly refused to run and said so in a
   warning, and the result carried no ``p_value`` at all, so the fabricated
   flag was the only significance channel a caller had.

Every test below pins the refusal AND asserts the measured answers still come
through unchanged, because a fix that answers "could not check" to everything
is the same defect pointing the other way.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from vfairness.agents.action_bias import ActionBiasAnalyzer, ActionBiasResult
from vfairness.agents.correspondence import CorrespondenceTester
from vfairness.agents.rag_bias import RAGBiasAnalyzer


def test_an_eeoc_verdict_needs_a_measured_selection_rate():
    """R4-A. adverse_impact False is the EEOC finding of no adverse impact."""
    tester = CorrespondenceTester()

    # REFUSAL. 0/0: nobody selected in either group, no ratio to form.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        nothing_selected = tester.four_fifths_rule(0.0, 0.0)
    assert math.isnan(nothing_selected["ratio"]), nothing_selected
    assert nothing_selected["adverse_impact"] is None, nothing_selected
    # The rates ARE equal, so "neither group was favoured" stays measured.
    assert nothing_selected["favored_group"] == "neither"
    assert caught

    # REFUSAL. A rate that was never measured must not reach `< 0.8`.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unmeasured = tester.four_fifths_rule(float("nan"), 0.5)
    assert math.isnan(unmeasured["ratio"]), unmeasured
    assert unmeasured["adverse_impact"] is None, unmeasured
    assert unmeasured["favored_group"] == "undetermined"
    assert caught

    # CONTROLS. Every measured verdict survives, at both ends and on the line.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adverse = tester.four_fifths_rule(0.90, 0.50)
        clean = tester.four_fifths_rule(0.85, 0.80)
        parity = tester.four_fifths_rule(0.5, 0.5)
        on_the_line = tester.four_fifths_rule(0.5, 0.4)
    assert adverse["ratio"] == pytest.approx(0.5 / 0.9)
    assert adverse["adverse_impact"] is True
    assert adverse["favored_group"] == "group_a"
    assert clean["ratio"] == pytest.approx(0.80 / 0.85)
    assert clean["adverse_impact"] is False
    # Measured parity keeps the exact ratio 1.0 / adverse False pair that the
    # 0/0 case used to borrow. That is the whole point of separating them.
    assert parity["ratio"] == 1.0
    assert parity["adverse_impact"] is False
    # The rule is "less than four fifths", so exactly 0.8 passes.
    assert on_the_line["ratio"] == pytest.approx(0.8)
    assert on_the_line["adverse_impact"] is False
    assert not caught


def test_a_rag_run_that_produced_nothing_is_not_an_unbiased_one():
    """R4-B. `nan > threshold` is False, so an unmeasured stage read as clean."""
    analyzer = RAGBiasAnalyzer()

    # REFUSAL. 0.0 output disparity means "identical outputs".
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        no_outputs = analyzer.analyze_output([], [])
    assert math.isnan(no_outputs), f"empty outputs reported as {no_outputs!r}"
    assert caught

    # REFUSAL. A whole run that produced nothing: both flags withheld.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        empty_run = analyzer.full_analysis([], [], [], [], [], [])
    assert math.isnan(empty_run.retrieval_disparity)
    assert math.isnan(empty_run.output_disparity)
    assert empty_run.is_retrieval_biased is None, empty_run
    assert empty_run.is_output_biased is None, empty_run
    assert caught

    # CONTROLS. Measured bias is still flagged and measured agreement is still
    # cleared, both with real numbers rather than None.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        biased = analyzer.full_analysis(
            ["q"],
            ["q"],
            [{"id": "doc_a"}],
            [{"id": "doc_b"}],
            ["the first group answer"],
            ["a completely unrelated reply"],
        )
        agreeing = analyzer.full_analysis(
            ["q"],
            ["q"],
            [{"id": "doc_a"}],
            [{"id": "doc_a"}],
            ["the same answer for both"],
            ["the same answer for both"],
        )
    assert biased.retrieval_disparity == 1.0
    assert biased.is_retrieval_biased is True
    assert biased.output_disparity > 0.5
    assert biased.is_output_biased is True
    assert agreeing.retrieval_disparity == 0.0
    assert agreeing.is_retrieval_biased is False
    assert agreeing.output_disparity == 0.0
    assert agreeing.is_output_biased is False
    assert not caught


def test_a_significance_test_that_refused_to_run_is_not_a_null_result():
    """R4-C. is_significant False means tested and not significant."""
    # The verdict needs somewhere else to live: without a p_value field the
    # fabricated flag was the only significance channel on the result.
    field_names = {f.name for f in dataclasses.fields(ActionBiasResult)}
    assert "p_value" in field_names, sorted(field_names)

    analyzer = ActionBiasAnalyzer()

    # REFUSAL. Mann-Whitney needs 2 values per group; this run has 1.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        untested = analyzer.analyze_outcomes(
            [{"action": "approve", "score": 0.9}],
            [{"action": "review", "score": 0.1}],
            "score",
        )
    assert math.isnan(untested.p_value), untested
    assert untested.is_significant is None, untested
    assert caught

    # CONTROLS. A measured significant gap and a measured NON-significant one.
    # The second control is the one that catches an over-correction: "not
    # significant" must still be False, never None.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        separated = analyzer.analyze_outcomes(
            [{"action": "approve", "score": 0.9 + i * 0.001} for i in range(30)],
            [{"action": "review", "score": 0.1 + i * 0.001} for i in range(30)],
            "score",
        )
        identical = analyzer.analyze_outcomes(
            [{"action": "approve", "score": 0.5} for _ in range(30)],
            [{"action": "approve", "score": 0.5} for _ in range(30)],
            "score",
        )
    assert separated.p_value < 0.05
    assert separated.is_significant is True
    assert separated.disparity == pytest.approx(0.8)
    assert identical.p_value == 1.0
    assert identical.is_significant is False, identical
    assert identical.disparity == 0.0
    assert not caught


class TestOneUnrecordedOutcomeDoesNotEraseAHiringDisparity:
    """READINESS-6, 2026-09-10. `analyze_outcomes` filtered missing outcomes
    with `isinstance(v, float) and np.isnan(v)`, which is true of a Python float
    NaN and of nothing else.

    `None` is not a float and a numpy scalar is not a Python float, so both
    survived the filter, became NaN inside `np.asarray(..., dtype=float)`, and
    carried NaN through the mean, the disparity and the test. The verdict was
    `p_value < self.alpha`, and `nan < 0.05` is False.

    Measured that day on a maximal callback disparity, group A called back 40 of
    40 and group B 0 of 40::

        all measured        disparity=1.0 p=1.86e-23 significant=True  n=40
        one None in B       disparity=nan p=nan      significant=False n=40
        one np.float32 nan  disparity=nan p=nan      significant=False n=40

    One unusable row turned a total hiring disparity into "not significant", and
    the sample size still claimed all 40. This is a correspondence study, the
    resume-audit method, so that verdict is the entire output.

    `four_fifths_rule` in the same file already answers None for a ratio it
    could not compute and explains why; this path was missed.
    """

    @staticmethod
    def _run(outcomes_b):
        import warnings as _w

        from vfairness.agents.correspondence import CorrespondenceTester

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            result = CorrespondenceTester().analyze_outcomes([1.0] * 40, outcomes_b)
        return result, [str(x.message) for x in caught]

    @pytest.mark.parametrize("missing", [None, np.float32("nan"), np.float64("nan"), float("nan")])
    def test_the_disparity_survives_one_unrecorded_outcome(self, missing):
        result, texts = self._run([0.0] * 39 + [missing])

        assert result.is_significant is True, (
            f"a 100%-vs-0% callback gap was reported as {result.is_significant!r} "
            f"because one outcome was {missing!r}"
        )
        assert result.disparity_metric == pytest.approx(1.0)
        assert math.isfinite(result.p_value) and result.p_value < 0.05

        # The row was dropped, counted, and the sample size reports what
        # REMAINED rather than what was supplied.
        assert result.n_unmeasurable == 1
        assert result.sample_size == 39, (
            f"sample_size {result.sample_size} still counts a row that carried no value"
        )
        assert any("EXCLUDED" in t for t in texts), "the exclusion was silent"

    def test_a_fully_measured_run_is_untouched(self):
        """OVER-CORRECTION CONTROL. Nothing is dropped and nothing is warned
        about when every outcome is real."""
        result, texts = self._run([0.0] * 40)
        assert result.is_significant is True
        assert result.n_unmeasurable == 0
        assert result.sample_size == 40
        assert not any("EXCLUDED" in t for t in texts)

    def test_a_genuinely_equal_pair_still_reports_not_significant(self):
        """The other control: `None` must not become the answer to everything.
        Two groups treated identically really are not significantly different,
        and that is a measurement."""
        result, _ = self._run([1.0] * 40)
        assert result.is_significant is False, (
            "a measured absence of disparity was reported as could-not-check"
        )
        assert result.p_value == pytest.approx(1.0)
        assert result.n_unmeasurable == 0

    def test_a_test_that_yields_no_p_value_is_not_reported_as_not_significant(self, monkeypatch):
        """The third state, pinned by forcing it.

        Stated honestly: with the filter fixed I could not reach a non-finite
        p-value through the public API, so this branch is a fail-safe rather
        than a live path today. It is pinned anyway, because the whole defect
        above was `nan < alpha` quietly answering False, and the next change to
        the test selection could reintroduce exactly that.
        """
        from scipy import stats

        from vfairness.agents.correspondence import CorrespondenceTester

        monkeypatch.setattr(stats, "fisher_exact", lambda *a, **k: (float("nan"), float("nan")))
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            result = CorrespondenceTester().analyze_outcomes([1.0] * 20, [0.0] * 20)

        assert result.is_significant is None, (
            f"a test that produced no p-value reported is_significant="
            f"{result.is_significant!r}, which reads as 'the groups were treated alike'"
        )
        assert math.isnan(result.p_value)
        assert any("could not check" in t.lower() for t in [str(x.message) for x in caught])
