"""BGL5 batch A-evaluation-1: the pins for the seven grades an audit overturned.

Written by the FIXER. Every number quoted in a docstring here was produced by
running the unit on this repo on 2026-09-27, before and after the change it
holds, and each test below exists because an independent auditor showed that the
evidence it replaces could not see the defect it named.

THE COMMON ROOT of four of the seven: infinity is the OTHER HALF of the NaN hole.
``_triage.is_measured`` refuses ``inf`` by name, ``check_threshold`` was fixed for
it at READINESS-6, and ``FairExplAIner._finite_or_none`` says in its own docstring
that "NaN/inf mean nothing was measured", but the could-not-check predicate the
explanation surfaces consult tests ``np.isnan(value)`` alone. So an infinite ratio
was graded "Excellent! The ratio of inf indicates near-perfect parity between
groups." at severity 'info', and a frame whose every metric was infinite got a
CRITICAL verdict on metrics the same report says were never compared. inf reaches
a ratio from a denominator of zero, which is a group with no selections at all:
the most extreme unfairness the data can express arriving as the most reassuring
verdict the surface can write.

THREE of the seven were downgrades about the EVIDENCE, not the behaviour:

  compute_all_metrics   its refusal rule was ``assert msgs or not measured``, and
                        every degenerate frame emits at least one warning, so the
                        left side was always true. The defect could be
                        reinstated at the entry point with the whole file green.
                        Pinned here per metric and per frame instead.
  plot_fairness_report  both halves of its honesty (the withheld score and the
                        per-value marks) could be broken ONE AT A TIME with its
                        named tests green, because the assertion was "NOT
                        ASSESSABLE in text OR NOT MEASURED in text" and the two
                        markers come from two independent producers.
  save_fairness_plots   its three named tests never executed it: coverage of the
                        function body was 1 line, the ``def``. Pinned here
                        through the file it actually writes.
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pytest

from vfairness._triage import is_measured
from vfairness.evaluation.vfairness_metrics._metric_direction import improvement_amount
from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer, MetricResult

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vfairness.evaluation.vfairness_metrics import visualization as VZ  # noqa: E402
from vfairness.evaluation.vfairness_metrics.report import (  # noqa: E402
    classification_fairness_report,
)

# The input classes ``is_measured`` refuses that are NOT NaN. Every one of them
# used to come back from at least one of the surfaces below as a number or a
# grade. A string and a bool are here because the canonical rule names them: a
# bool coerces to 1.0 and a numeric string parses, so both reach arithmetic.
UNMEASURABLE = [
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="minus_inf"),
    pytest.param(True, id="bool"),
    pytest.param("0.9", id="numeric_string"),
    pytest.param(None, id="none"),
]

# The same classes MINUS None, for surfaces where None is not a value at all.
# ``FairnessAnalyzer.explain_metric(name, value=None)`` means "compute it for me",
# and it does: measured 2026-09-27, explain_metric("demographic_parity_ratio",
# None) on the two-group fixture resolves 0.6957 and grades it 'critical', which
# is correct. Feeding None in as an unmeasurable VALUE would have been a fixture
# error dressed as a finding.
UNMEASURABLE_VALUES = [p for p in UNMEASURABLE if p.values[0] is not None]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _caught(fn, *args, **kwargs):
    """``(value, [warning messages])``, with every filter forced to record."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, [str(w.message) for w in rec]


# ===========================================================================
# 1. improvement_amount: the one function in _metric_direction with no
#    is_measured gate at all.  Overturned from PROVEN to DEFECT OPEN.
# ===========================================================================


class TestImprovementAmountRefusesWhatItCannotMeasure:
    """Measured 2026-09-27 before the guard, all with a KNOWN direction so the
    ``return None`` branch is not involved:

        improvement_amount("disparate_impact_ratio", inf, 0.5)        -> inf
        improvement_amount("demographic_parity_difference", 0.5, inf) -> inf
        improvement_amount("disparate_impact_ratio", True, 0.5)       -> 0.5
        improvement_amount("disparate_impact_ratio", "0.9", 0.5)      -> 0.4

    After the guard each of those returns NaN. ``check_threshold`` in the same
    module already answered COULD_NOT_CHECK for every one of them.
    """

    @pytest.mark.parametrize("value", UNMEASURABLE)
    @pytest.mark.parametrize("side", ["value", "baseline"])
    def test_an_unmeasurable_side_yields_no_improvement_number(self, value, side):
        assert not is_measured(value), "fixture error: this input must be unmeasurable"
        if side == "value":
            got = improvement_amount("disparate_impact_ratio", value, 0.5)
        else:
            got = improvement_amount("disparate_impact_ratio", 0.9, value)
        assert got is None or (isinstance(got, float) and math.isnan(got)), (
            f"an unmeasurable {side} produced the improvement {got!r}, which a caller "
            f"enforcing 'improvement >= margin' cannot tell from a measured one"
        )

    def test_the_two_non_numbers_stay_distinguishable(self):
        """NaN and None mean different things and a caller acts differently on them.

        NaN: there was nothing to compare. None: we do not know which way better
        points. Collapsing them would hide an unknown direction, which is what
        inverted the deployment gate for the whole ratio family.
        """
        assert improvement_amount("shiny_new_metric_2027", 0.10, 0.30) is None
        assert math.isnan(improvement_amount("demographic_parity_difference", float("nan"), 0.30))

    def test_control_a_real_change_is_still_measured_to_its_real_size(self):
        """The over-correction half: a function that refuses everything passes
        every refusal test above."""
        assert improvement_amount("demographic_parity_difference", 0.10, 0.30) == pytest.approx(
            0.20
        )
        assert improvement_amount("disparate_impact_ratio", 0.90, 0.60) == pytest.approx(0.30)
        # A degradation is negative, not absorbed.
        assert improvement_amount("disparate_impact_ratio", 0.60, 0.90) == pytest.approx(-0.30)
        # A numpy scalar is a real measurement (READINESS-6) and must survive.
        assert improvement_amount(
            "disparate_impact_ratio", np.float32(0.9), np.float32(0.6)
        ) == pytest.approx(0.30, abs=1e-6)


# ===========================================================================
# 2. FairnessAnalyzer.explain_metric: an infinite ratio graded "Excellent",
#    severity info, with the recommendation written for a PASS.
#    Overturned from PROVEN to DEFECT OPEN.
# ===========================================================================


def _two_group_analyzer() -> FairnessAnalyzer:
    rng = np.random.default_rng(0)
    y = (rng.random(80) < 0.5).astype(int)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    return _quiet(FairnessAnalyzer, y, y, groups)


class TestExplainMetricRefusesAnInfiniteRatio:
    @pytest.mark.parametrize("value", UNMEASURABLE_VALUES)
    def test_no_unmeasurable_value_is_graded(self, value):
        explanation = _quiet(
            _two_group_analyzer().explain_metric, "demographic_parity_ratio", value
        )
        assert explanation.severity == "could_not_check", (
            f"{value!r} was graded {explanation.severity!r}: {explanation.evaluation}"
        )
        assert "COULD NOT CHECK" in explanation.evaluation

    def test_control_a_missing_value_is_computed_rather_than_refused(self):
        """``value=None`` is this method's "compute it for me" sentinel, not an
        unmeasurable input, and a guard that refused it would silence the
        one-argument call the docstring shows. Measured 2026-09-27:
        explain_metric("demographic_parity_ratio") resolves 0.6957 and grades it
        'critical'."""
        explanation = _quiet(_two_group_analyzer().explain_metric, "demographic_parity_ratio")
        assert explanation.value == pytest.approx(0.6957, abs=5e-5)
        assert explanation.severity == "critical"

    def test_the_refused_card_keeps_the_value_and_names_the_reason(self):
        """Measured 2026-09-27 at the public entry point, before the guard:

            analyzer.explain_metric("demographic_parity_ratio", float("inf"))
              severity    'info'
              evaluation  'Excellent! The ratio of inf indicates near-perfect
                           parity between groups.'
              recommend.  'Continue monitoring this metric as part of regular
                           fairness audits. Document your fairness practices ...'

        After the guard:

              severity    'could_not_check'
              evaluation  'COULD NOT CHECK: demographic_parity_ratio could not be
                           measured on this data (infinite: no comparison can
                           grade it), so no threshold was applied to it. ...'
              value       inf, still, because the report's own assessment block
                          names that same infinity and the two halves of one
                          artifact may not disagree about what was computed.
        """
        explanation = _quiet(
            _two_group_analyzer().explain_metric, "demographic_parity_ratio", float("inf")
        )
        assert math.isinf(explanation.value), (
            f"the value the reader is shown was replaced with {explanation.value!r}"
        )
        assert "infinite" in explanation.evaluation, (
            f"the reason is not named where a reader sees it: {explanation.evaluation}"
        )
        assert "Excellent" not in explanation.evaluation
        assert "Continue monitoring" not in explanation.recommendation, (
            "the recommendation written for a PASSING metric survived the refusal"
        )

    def test_control_a_measured_value_is_still_graded_to_its_real_severity(self):
        """The over-correction half, with the actual grades.

        Measured 2026-09-27, unchanged by the guard: 0.0 -> 'info' with
        "Excellent! The difference of 0.0000 indicates near-perfect fairness
        across groups.", and 0.42 -> 'critical'.
        """
        analyzer = _two_group_analyzer()
        benign = _quiet(analyzer.explain_metric, "demographic_parity_difference", 0.0)
        assert benign.severity == "info"
        assert "Excellent" in benign.evaluation
        assert benign.value == 0.0
        severe = _quiet(analyzer.explain_metric, "demographic_parity_difference", 0.42)
        assert severe.severity == "critical"
        assert "COULD NOT CHECK" not in severe.evaluation

    def test_control_a_structured_result_is_unwrapped_rather_than_refused(self):
        """The MIRROR defect, a measurement reported as a could-not-check.

        Measured 2026-09-27 before this fix: explain_metric with the
        MetricResult that compute_all_metrics(include_ci=True) returns came back
        severity 'could_not_check' saying "this metric was never computed", for a
        metric measured at 0.2000, and its to_dict() then carried a MetricResult
        object in the "value" field, which json.dumps cannot encode. After:
        severity 'high', value 0.2.
        """
        wrapped = MetricResult(metric_name="demographic_parity_difference", value=0.2)
        explanation = _quiet(
            _two_group_analyzer().explain_metric, "demographic_parity_difference", wrapped
        )
        assert explanation.value == pytest.approx(0.2)
        assert explanation.severity == "high", (
            f"a measured 0.2 was reported as {explanation.severity!r}: {explanation.evaluation}"
        )
        # And a structured result that was never measured is still refused.
        empty = MetricResult(metric_name="demographic_parity_difference", value=float("nan"))
        assert (
            _quiet(
                _two_group_analyzer().explain_metric, "demographic_parity_difference", empty
            ).severity
            == "could_not_check"
        )


# ===========================================================================
# 3. FairnessAnalyzer.get_explanations: a CRITICAL verdict on metrics the same
#    run says were never compared. Overturned from PROVEN to DEFECT OPEN.
# ===========================================================================


def _infinite_regression_analyzer() -> FairnessAnalyzer:
    """Two groups, one infinite prediction, which validate_inputs admits in
    silence, so every regression metric comes back inf."""
    rng = np.random.default_rng(2)
    groups = np.array(["A"] * 40 + ["B"] * 40)
    y = rng.normal(10, 2, 80)
    pred = y + rng.normal(0, 0.5, 80)
    pred[0] = np.inf
    return _quiet(FairnessAnalyzer, y, pred, groups)


def _gap_analyzer() -> FairnessAnalyzer:
    """A real, total disparity: group a is always selected, group b never."""
    y = np.array([1] * 40 + [0] * 40)
    groups = np.array(["a"] * 40 + ["b"] * 40)
    return _quiet(FairnessAnalyzer, y, y.copy(), groups)


class TestGetExplanationsDoesNotGradeWhatItCouldNotCompare:
    def test_the_summary_neither_asserts_a_disparity_nor_drops_the_disclosure(self):
        """Measured 2026-09-27 on the infinite-prediction frame, before the guard:

            'Overall Fairness Score: NOT AVAILABLE (could not check) (0 passed,
             0 failed, 4 not assessable) | CRITICAL: 4 metric(s) show critical
             disparities requiring immediate attention. | COULD NOT CHECK: ...'
            'NOT GRADED' in summary: False

        Two findings in one string: a CRITICAL verdict on four metrics the same
        report's assessment block calls NOT_ASSESSABLE "(infinite: no comparison
        can grade it)", and the NOT GRADED clause missing because the pin that
        exists for it had only ever been run with NaN. After the guard the
        CRITICAL clause is gone and the summary carries "NOT GRADED: 4 metric(s)
        were never compared to a threshold, so they are neither passing nor
        failing."
        """
        summary = _quiet(_infinite_regression_analyzer().get_explanations)["summary"]
        assert "CRITICAL" not in summary.upper(), (
            f"a critical disparity was asserted for metrics nobody compared: {summary}"
        )
        assert "NOT GRADED" in summary, f"the not-graded disclosure is missing: {summary}"
        assert "NOT AVAILABLE (could not check)" in summary

    def test_each_refused_card_is_three_state_and_keeps_its_value(self):
        """The card dict is the surface a dashboard reads, so the disclosure has
        to survive to_dict, not only to the prose."""
        analyzer = _infinite_regression_analyzer()
        explanations = _quiet(analyzer.get_explanations)
        cards = explanations["metrics"]
        assert cards, "no per-metric cards at all, so this would pass vacuously"
        for name, card in cards.items():
            assert card["severity"] == "could_not_check", (
                f"{name} was graded {card['severity']!r} on an infinite value: {card['evaluation']}"
            )
            assert "COULD NOT CHECK" in card["evaluation"]
            assert "infinite" in card["evaluation"], (
                f"{name} does not say WHY it could not be checked: {card['evaluation']}"
            )
            assert math.isinf(card["value"]), (
                f"{name} hides the value the assessment block names: {card['value']!r}"
            )

    def test_the_report_itself_still_carries_the_infinity_it_refused(self):
        """The substitution exists for the grading pass only. Overwriting the
        report's metric value would destroy the evidence for the sentence its own
        assessment block writes about it."""
        report = _quiet(_infinite_regression_analyzer().get_report, include_explanations=True)
        assert all(math.isinf(v) for v in report["metrics"].values()), (
            f"the guard altered the measured metrics: {report['metrics']}"
        )
        reasons = " ".join(
            str(m.get("reason", "")) for m in report["assessment"]["not_assessable_metrics"]
        )
        assert "infinite" in reasons

    def test_the_run_says_out_loud_that_nothing_was_graded(self):
        """A gate polling the report dict reads neither the cards nor the prose."""
        _report, messages = _caught(
            _infinite_regression_analyzer().get_report, include_explanations=True
        )
        joined = " ".join(messages)
        assert "cannot be graded" in joined, f"get_report graded nothing and said nothing: {joined}"
        assert "mae_parity_difference" in joined, (
            f"the disclosure does not name the metrics it refused: {joined}"
        )

    def test_control_a_real_disparity_still_reaches_the_summary(self):
        """The over-correction half. A total gap must still be reported as one,
        with its real numbers: measured 2026-09-27, fairness_score 0.0, four
        failed metrics, and a CRITICAL clause that is now earned."""
        analyzer = _gap_analyzer()
        explanations = _quiet(analyzer.get_explanations)
        summary = explanations["summary"]
        assert "CRITICAL" in summary.upper(), f"a total disparity was not reported: {summary}"
        assert "NOT AVAILABLE" not in summary
        assert "Overall Fairness Score: 0.0%" in summary, summary


# ===========================================================================
# 4. FairnessAnalyzer.compute_all_metrics: the behaviour was right and the pin
#    could not see the defect. Overturned from PROVEN to SEMI-PROVEN.
# ===========================================================================

_N = 60


def _frame(kind: str):
    rng = np.random.default_rng(3)
    y = (rng.random(_N) < 0.5).astype(int)
    pred = y.copy()
    groups = np.array(["a"] * (_N // 2) + ["b"] * (_N - _N // 2))
    if kind == "single_group":
        groups = np.array(["a"] * _N)
    elif kind == "one_label":
        y = np.zeros(_N, dtype=int)
        pred = np.zeros(_N, dtype=int)
    elif kind == "n_equals_2":
        return y[:2], pred[:2], groups[:2]
    elif kind == "empty":
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=object)
    elif kind == "real_gap":
        y = np.array([1] * (_N // 2) + [0] * (_N - _N // 2))
        pred = y.copy()
    return y, pred, groups


def _frame_analyzer(kind: str) -> FairnessAnalyzer:
    return _quiet(FairnessAnalyzer, *_frame(kind))


class TestComputeAllMetricsRefusesEachMetricByName:
    """WHY THIS EXISTS. The evidence for this method was
    ``assert msgs or not measured``: ANY warning satisfied it. Measured
    2026-09-27, every degenerate frame emits at least one warning (single_group
    1, one_label 3, n_equals_2 8, empty 3) and the single_group one is about
    PRECISION, a different metric, so the refusal half of that pin could never
    fail. An in-memory sabotage of ``classification.demographic_parity_difference``
    returning 0.0 instead of NaN for fewer than two groups put
    ``{'demographic_parity_difference': 0.0, ...}`` on a single-group frame, the
    canvas for perfect parity, and the whole file stayed green at 12 passed.

    The rule here is per metric and per frame instead, and the disclosure is read
    for the NAMES it carries rather than for its existence.
    """

    @pytest.mark.parametrize("kind", ["single_group", "n_equals_2", "empty"])
    def test_no_metric_is_measured_where_no_pair_exists(self, kind):
        metrics, messages = _caught(_frame_analyzer(kind).compute_all_metrics)
        assert metrics, "no metric keys at all, so this test would pass vacuously"
        graded = {name: value for name, value in metrics.items() if is_measured(value)}
        assert not graded, (
            f"{kind} has fewer than two comparable groups and returned measured "
            f"metrics anyway: {graded}"
        )
        disclosure = [m for m in messages if "could not be measured" in m]
        assert disclosure, f"{kind} refused every metric silently: {messages}"
        for name in metrics:
            assert any(name in m for m in disclosure), (
                f"{kind}: {name} came back unmeasurable and no warning names it: {disclosure}"
            )

    def test_a_single_label_frame_keeps_its_one_legitimate_measurement(self):
        """The partition, exactly, because this frame is where a blanket refusal
        would destroy a real number.

        Measured 2026-09-27: every prediction is 0, so both selection rates are
        an observed 0.0 and their difference is a measured 0.0. The other four
        metrics need a positive label that does not exist, and they are NaN.
        """
        metrics, messages = _caught(_frame_analyzer("one_label").compute_all_metrics)
        assert metrics["demographic_parity_difference"] == 0.0, (
            "an observed, equal selection rate is a measurement and must survive"
        )
        refused = sorted(name for name, value in metrics.items() if not is_measured(value))
        assert refused == [
            "demographic_parity_ratio",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "predictive_parity_difference",
        ], refused
        disclosure = " ".join(m for m in messages if "could not be measured" in m)
        for name in refused:
            assert name in disclosure, f"{name} was refused silently: {disclosure}"
        assert "demographic_parity_difference=" not in disclosure, (
            f"the one measured metric was reported as unmeasurable: {disclosure}"
        )

    def test_control_a_healthy_frame_measures_everything_and_says_nothing(self):
        """The over-correction half, with the actual numbers.

        Measured 2026-09-27 on the two-group frame: five metrics, all measured,
        ZERO warnings, demographic_parity_difference 0.06666666666666665, which
        this test recomputes from the arrays rather than copying.
        """
        y, pred, groups = _frame("healthy")
        metrics, messages = _caught(_quiet(FairnessAnalyzer, y, pred, groups).compute_all_metrics)
        unmeasured = sorted(name for name, value in metrics.items() if not is_measured(value))
        assert not unmeasured, f"a comparable frame refused {unmeasured}"
        assert messages == [], f"a fully measured run warned about something: {messages}"
        rate_a = pred[groups == "a"].mean()
        rate_b = pred[groups == "b"].mean()
        assert metrics["demographic_parity_difference"] == pytest.approx(abs(rate_a - rate_b))

    def test_control_a_total_gap_still_reads_one(self):
        metrics = _quiet(_frame_analyzer("real_gap").compute_all_metrics)
        assert metrics["demographic_parity_difference"] == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 5 and 6. plot_fairness_report and save_fairness_plots.
#     Overturned from PROVEN to SEMI-PROVEN: neither half of the honesty was
#     load-bearing, and the second function was never executed at all.
# ===========================================================================


def _chart_report(kind: str):
    rng = np.random.default_rng(5)
    n = 200
    y = (rng.random(n) < 0.5).astype(int)
    pred = y.copy()
    if kind == "single_group":
        groups = np.array(["a"] * n)
    else:
        groups = np.array(["a"] * 100 + ["b"] * 100)
    if kind == "gap":
        # A measured total failure: group a always selected, group b never.
        pred = np.concatenate([np.ones(100, dtype=int), np.zeros(100, dtype=int)])
    return _quiet(classification_fairness_report, y, pred, groups)


def _texts(fig) -> list:
    """Every string a reader of this matplotlib figure can see."""
    out = []
    for ax in getattr(fig, "axes", []) or []:
        out += [t.get_text() for t in ax.texts]
        out.append(ax.get_title())
        out += [label.get_text() for label in ax.get_xticklabels()]
        out += [label.get_text() for label in ax.get_yticklabels()]
    out += [t.get_text() for t in getattr(fig, "texts", []) or []]
    return [t for t in out if t and str(t).strip()]


def _unmeasured_metric_count(report) -> int:
    return sum(1 for value in report["metrics"].values() if not is_measured(value))


class TestPlotFairnessReportIsHonestOnBothHalves:
    """The two markers come from two INDEPENDENT producers, so each masked the
    other under an "A or B" assertion. Measured 2026-09-27 with each half
    sabotaged on its own, the named evidence fully green both times:

      _is_measured forced True     the canvas became "NOT ASSESSABLE: no metric
                                   could be measured | nan | nan | nan | nan |
                                   nan | ...": the per-value marks gone, replaced
                                   by five printed NaNs, and 24 passed for this
                                   chart.
      _score_display forced to
      ("100%", success, True)      the canvas became "Overall Score: 100% | n/a
                                   not measured | ..." with the suptitle
                                   "Classification Fairness Audit Report (Score:
                                   100%)", on a report whose assessment
                                   fairness_score is None, and the whole file was
                                   24 passed.

    So the two halves are asserted separately below, and each one alone must be
    able to fail.
    """

    def test_the_withheld_score_is_not_borrowed_by_the_title(self):
        report = _chart_report("single_group")
        assert report["assessment"]["fairness_score"] is None, "fixture error"
        fig = _quiet(VZ.plot_fairness_report, report)
        suptitle = fig._suptitle.get_text()
        assert VZ.NOT_ASSESSABLE_TEXT in suptitle, (
            f"the title of a run with no score reads: {suptitle!r}"
        )
        assert not re.search(r"\d+(?:\.\d+)?\s*%", suptitle), (
            f"a percentage was printed for a score that was never measured: {suptitle!r}"
        )
        drawn = " | ".join(_texts(fig))
        assert "Overall Score:" not in drawn, (
            f"a score line was drawn for a withheld score: {drawn[:200]}"
        )
        assert VZ.NOT_ASSESSABLE_REASON in drawn, "the figure does not say why"

    def test_every_unmeasured_metric_is_marked_and_no_nan_is_drawn(self):
        report = _chart_report("single_group")
        fig = _quiet(VZ.plot_fairness_report, report)
        texts = _texts(fig)
        marks = [t for t in texts if "not measured" in t]
        assert len(marks) == _unmeasured_metric_count(report) == 5, (
            f"{len(marks)} could-not-measure marks for "
            f"{_unmeasured_metric_count(report)} unmeasured metrics: {texts}"
        )
        for text in texts:
            assert not re.search(r"(?<![a-z])nan(?![a-z])", text.lower()), (
                f"a NaN was printed as a value on the canvas: {text!r}"
            )

    def test_control_a_measured_score_still_reaches_the_title(self):
        """The over-correction half, with the actual number. Measured
        2026-09-27: the total-gap report scores 0.0 with four failed metrics, so
        the title must read "(Score: 0%)". A chart that refused everything, or
        one that borrowed the 100% of the sabotage above, fails here."""
        report = _chart_report("gap")
        assert report["assessment"]["fairness_score"] == 0.0, "fixture error"
        fig = _quiet(VZ.plot_fairness_report, report)
        suptitle = fig._suptitle.get_text()
        assert "Score: 0%" in suptitle, suptitle
        assert VZ.NOT_ASSESSABLE_TEXT not in suptitle
        # The one genuinely unmeasurable metric in that report is still marked.
        marks = [t for t in _texts(fig) if "not measured" in t]
        assert len(marks) == _unmeasured_metric_count(report) == 1, marks


class TestSaveFairnessPlotsWritesWhatItDrew:
    """Its three named tests never reached this function: run alone with coverage
    on the module, the body of ``save_fairness_plots`` (lines 2608 to 2691)
    executed ONE line, 2608, which is the ``def`` and runs at import. The six
    charts that ARE parametrised in that file reached 29, 51, 21, 91, 42 and 33
    body lines. The only test that does execute it asserts
    ``path.stat().st_size > 0``, so it stayed green while the files it weighed
    carried "Overall Score: 100%" for a withheld score.

    These pins read the FILE. ``svg.fonttype = "none"`` makes matplotlib write
    the strings as text rather than as glyph outlines, which is the only way the
    content of a saved chart can be read back at all; the figure is otherwise
    exactly the one the function writes for a caller.
    """

    @staticmethod
    def _saved(report, tmp_path, prefix):
        plt.rcParams["svg.fonttype"] = "none"
        paths = _quiet(
            VZ.save_fairness_plots,
            report,
            output_dir=str(tmp_path),
            prefix=prefix,
            format="svg",
        )
        assert paths, "save_fairness_plots wrote no files at all"
        full = [p for p in paths if "full_report" in p]
        assert full, f"the full report was not written: {paths}"
        import pathlib

        return pathlib.Path(full[0]).read_text()

    def test_the_saved_artifact_carries_the_could_not_check_state(self, tmp_path):
        report = _chart_report("single_group")
        svg = self._saved(report, tmp_path, "refusal")
        assert "<text" in svg, "the text did not survive into the file, so nothing was checked"
        assert VZ.NOT_ASSESSABLE_TEXT in svg, "the saved chart does not say it assessed nothing"
        assert not re.search(r"Score:\s*\d", svg), (
            "the saved chart printed a score for a run that has none"
        )
        assert svg.count("not measured") == _unmeasured_metric_count(report) == 5
        assert not re.search(r">\s*nan\s*<", svg), "a NaN was printed as a value in the saved file"

    def test_control_the_saved_artifact_carries_a_measured_score(self, tmp_path):
        """A save path that wrote a refusal onto every chart would pass the test
        above. The measured score has to reach the file too: 0%, four failures."""
        report = _chart_report("gap")
        svg = self._saved(report, tmp_path, "measured")
        assert "Score: 0%" in svg, "the measured score did not reach the saved file"
        assert VZ.NOT_ASSESSABLE_TEXT not in svg


# ===========================================================================
# 7. preview_palette: the fix was real and it was in a channel the artifact does
#    not carry. Overturned from PROVEN to DEFECT OPEN.
# ===========================================================================


UNKNOWN_STYLE = "colorblind_safe_v2"


def _figure_blob(fig) -> str:
    """Everything the figure serialises, whichever backend drew it."""
    if hasattr(fig, "to_dict"):
        return str(fig.to_dict())
    return " | ".join([fig._suptitle.get_text() if fig._suptitle else ""] + _texts(fig))


class TestPreviewPaletteLabelsThePaletteItDrew:
    """Measured 2026-09-27 AFTER the warning was added and BEFORE this fix:

        preview_palette("colorblind_safe_v2")
          warning  "_get_palette: 'colorblind_safe_v2' is not a known
                    visualization style ... Falling back to 'modern'"
          title    '<b>Colorblind_Safe_V2 Color Palette</b>'
          swatches the MODERN palette (_get_palette(unknown) is PALETTES["modern"])
          the figure contained none of 'not a known' (False), 'falling back'
          (False), 'modern palette' (False) or 'does not exist' (False)

    A ``warnings.warn`` does not travel with a saved PNG, an exported HTML or a
    notebook with filters set, so the artifact still reported the colours of a
    style that does not exist. After the fix the title is
    '<b>Modern Color Palette</b>' followed by "requested 'colorblind_safe_v2',
    which is not a known style: these are the MODERN colours, so nothing here
    shows 'colorblind_safe_v2'".
    """

    def test_the_figure_names_the_palette_it_actually_drew(self):
        assert UNKNOWN_STYLE not in VZ.PALETTES, "fixture error"
        assert _quiet(VZ._get_palette, UNKNOWN_STYLE) == VZ.PALETTES["modern"], (
            "fixture error: these swatches must be another palette's"
        )
        figure, messages = _caught(VZ.preview_palette, UNKNOWN_STYLE)
        assert any("is not a known visualization style" in m for m in messages), (
            f"the warning that was already there must stay: {messages}"
        )
        blob = _figure_blob(figure).lower()
        assert "modern color palette" in blob, (
            f"the figure does not name the palette it drew: {blob[:300]}"
        )
        assert "not a known style" in blob, (
            "the figure carries no disclosure that the colours belong to another style"
        )
        assert UNKNOWN_STYLE in blob, (
            "the requested name vanished, so a reader cannot tell what was asked for"
        )

    def test_the_matplotlib_branch_carries_it_too(self, monkeypatch):
        """Two branches draw this figure and only one was ever looked at."""
        monkeypatch.setattr(VZ, "_check_plotly", lambda: False)
        fig = _quiet(VZ.preview_palette, UNKNOWN_STYLE)
        assert fig._suptitle.get_text() == "Modern Color Palette", fig._suptitle.get_text()
        drawn = " | ".join(_texts(fig))
        assert "not a known style" in drawn, drawn[:300]

    @pytest.mark.parametrize("style", sorted(VZ.PALETTES))
    def test_control_a_documented_style_previews_silently_under_its_own_name(self, style):
        """The over-correction half: a disclosure on every figure would be a
        false alarm on the six styles that do exist."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig = VZ.preview_palette(style)
        blob = _figure_blob(fig).lower()
        assert f"{style.lower()} color palette" in blob
        assert "not a known style" not in blob
