"""The renderer must CONSUME the producer's disclosure, never re-derive it.

THE DEFECT CLASS. A producer measures something correctly, three states and all,
and a renderer then re-derives the verdict from the value's SHAPE instead of
reading the field the producer wrote. Every producer-side check passes, every
unit test on the producer stays green, and the reader sees a fabrication. Two
shapes of it, both found live in this file's subject on 2026-09-30:

* ``accuracy_measured`` re-derived as ``accuracy is not None``. A NaN is not
  None, and ``accuracy`` on ``MethodComparison`` is typed as a plain ``float``
  and so CANNOT carry None, which is exactly why both producer arms carry
  ``parameters['accuracy_measured'] = False`` beside a NaN. Measured directly:
  ``parameters['accuracy_measured']`` was ``False`` while
  ``adapters_training._method_state`` published ``True``.
* a field-by-field rebuild that lists eight names and not the two added later,
  so ``parameters['degenerate_constant_predictions']`` never reached the canvas
  and a model that collapsed to a constant prediction drew a green PASS chip.

WHY EVERY ASSERTION HERE IS ON A RENDERED STRING. A pin that asserted the
producer's own field would be green today AND green for this whole defect class,
which makes it worthless: the producer was always right. So each test below reads
the markup a person is served, or the accessible ``<desc>`` a screen-reader user
is served, or the severity a report pipeline consumes instead of the sentence.

THE CONTROLS are at the bottom and they assert REAL numbers, not merely that
nothing raised. A guard that withholds everything passes every refusal test.
"""

from __future__ import annotations

import re
import warnings

import pytest

from vfairness.in_processing.analyzer import FairnessTrainingReport, MethodComparison

jinja2 = pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.rendering.adapters_training import (  # noqa: E402
    _method_state,
    method_comparison_to_svg,
    training_analysis_report_to_svg,
    training_report_to_svg,
)
from vfairness.rendering.engine import _get_jinja_env, render_svg  # noqa: E402
from vfairness.rendering.explain import build_explanation  # noqa: E402

NAN = float("nan")
INF = float("inf")


def _desc(svg: str) -> str:
    """The accessible one-liner, which is all a screen-reader user receives."""
    match = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    assert match, "the rendered SVG carries no <desc>"
    return match.group(1)


def _report(baseline, methods=(), fairness_analysis=None):
    return FairnessTrainingReport(
        timestamp="2026-09-30T09:00:00",
        data_info={"n_samples": 120, "n_groups": 2, "n_features": 5},
        task_type="classification",
        baseline_metrics=baseline,
        fairness_analysis=fairness_analysis or {},
        method_comparisons=list(methods),
        recommendation=None,
        tradeoff_analysis={},
        critical_issues=[],
        action_items=[],
    )


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ───────────────────────────────────────────────────────────────────────────
# 1. The producer's own accuracy_measured flag, on the canvas.
# ───────────────────────────────────────────────────────────────────────────


class TestTheProducersAccuracyFlagIsObeyed:
    #: Exactly what ``compare_methods`` builds for the threshold arm when 10 of
    #: 120 rows carry a non-finite target: accuracy carried as NaN with the flag
    #: and the reason beside it.
    DISOWNED = dict(
        method_name="Threshold",
        accuracy=NAN,
        fairness_violation=0.06,
        constraint_satisfied=True,
        training_time=1.4,
        parameters={
            "accuracy_measured": False,
            "accuracy_not_measured_reason": (
                "10 of 120 row(s) carry a non-finite target or prediction"
            ),
        },
    )

    def test_the_adapter_no_longer_contradicts_the_producer(self):
        m = MethodComparison(**self.DISOWNED)
        assert m.parameters["accuracy_measured"] is False
        assert _method_state(m)["accuracy_measured"] is False, (
            "the renderer re-derived accuracy_measured from `accuracy is not None`, and a "
            "NaN is not None, so every refusal came back out as a claimed measurement"
        )

    @pytest.mark.parametrize(
        "render",
        [method_comparison_to_svg, None],
        ids=["method_comparison", "training_report"],
    )
    def test_no_reader_is_shown_the_disowned_number(self, render):
        m = MethodComparison(**self.DISOWNED)
        if render is None:
            svg = _quiet(
                training_report_to_svg,
                _report(
                    {"accuracy": 0.8, "fairness_violation": 0.02, "constraint_satisfied": True}, [m]
                ),
            )
        else:
            svg = _quiet(render, [m])
        assert ">nan<" not in svg, "the accuracy the producer disowned is drawn as a number"
        assert "nan" not in _desc(svg).lower(), (
            "the accessible description prints the disowned accuracy, which is the same "
            "fabrication moved where a sighted reader cannot see it"
        )
        assert "not measured" in svg, "the third state is nowhere on the canvas"

    def test_the_producers_reason_reaches_the_markup(self):
        """The WHY was produced and consumed nowhere, so no reader ever got it."""
        svg = _quiet(method_comparison_to_svg, [MethodComparison(**self.DISOWNED)])
        assert "10 of 120 row(s) carry a non-finite target" in svg

    def test_a_nan_with_no_disclosure_at_all_is_withheld_too(self):
        """THE SIBLING DOOR. A producer that carries a NaN and writes no flag is
        not covered by reading the flag, and ``_coord`` in this same module has
        refused NaN for other cells since 2026-08-28 while these cells did not.
        """
        m = MethodComparison("Reduction", NAN, 0.06, True, 1.0, {})
        assert _method_state(m)["accuracy_measured"] is False
        svg = _quiet(method_comparison_to_svg, [m])
        assert ">nan<" not in svg
        assert "not measured" in svg

    def test_a_finite_number_the_producer_disowned_is_withheld_too(self):
        """A CHECK COUNTS ONLY IF IT COULD HAVE DISAGREED. Refusing the NaN is a
        test of the VALUE; this is the only test of the FLAG, because it is the
        one case where the two answers differ. It is not hypothetical: the number
        is the 0.4583 ``compare_methods`` used to publish for the threshold arm,
        a mean over 120 rows that counted 10 unscorable ones as rows the model
        got wrong. It is part measurement and part fabrication, and the producer
        says so in the flag; nothing about its shape says so.
        """
        m = MethodComparison(
            "Threshold",
            0.4583333333333333,
            0.06,
            True,
            1.0,
            {
                "accuracy_measured": False,
                "accuracy_not_measured_reason": (
                    "10 of 120 row(s) carry a non-finite target or prediction, so the "
                    "accuracy counted them as rows it got wrong"
                ),
            },
        )
        assert _method_state(m)["accuracy_measured"] is False
        svg = _quiet(method_comparison_to_svg, [m])
        assert ">0.458<" not in svg, "the partly fabricated number is drawn as a measurement"
        assert "not measured" in svg
        assert "counted them as rows it got wrong" in svg

    @pytest.mark.parametrize("sentinel", [NAN, INF, -INF, True])
    def test_no_sentinel_is_drawn_as_a_violation_magnitude(self, sentinel):
        """The SIBLING CELL. ``fairness_violation`` was tested the same way, and a
        NaN there drew "nan" in the alert colour under a red FAIL chip: an
        invented breach, which sends a reader after a violation that does not
        exist. ``True`` is an ``int`` in Python and would print as 1.000.
        """
        m = MethodComparison("Reduction", 0.83, sentinel, False, 1.0, {})
        assert _method_state(m)["violation_measured"] is False
        svg = _quiet(method_comparison_to_svg, [m])
        assert ">nan<" not in svg and ">inf<" not in svg and ">-inf<" not in svg
        assert ">1.000<" not in svg


# ───────────────────────────────────────────────────────────────────────────
# 2. The disclosure the rebuild dropped: a vacuous verdict.
# ───────────────────────────────────────────────────────────────────────────


class TestAVacuousVerdictIsNotAMeasuredPass:
    HEALTHY = MethodComparison("Reweight", 0.8102, 0.0301, True, 1.0, {"accuracy_measured": True})
    #: What ``compare_methods`` builds when a method collapses to one class: it
    #: satisfies every rate-based constraint by construction, the producer warns,
    #: and ``generate_recommendation`` excludes it from both rankings.
    COLLAPSED = MethodComparison(
        "Reduction",
        0.6200,
        0.0,
        True,
        1.0,
        {
            "accuracy_measured": True,
            "degenerate_constant_predictions": True,
            "n_prediction_classes": 1,
        },
    )

    def test_the_row_does_not_wear_a_pass_chip(self):
        svg = _quiet(method_comparison_to_svg, [self.HEALTHY, self.COLLAPSED])
        assert svg.count(">PASS<") == 1, (
            "the collapsed method is drawn with a green PASS chip: it satisfies the "
            "constraint by construction and measures nothing"
        )
        assert ">VACUOUS<" in svg, "the vacuous state is nowhere a reader looks"

    def test_the_headline_count_names_it(self):
        """Headline rule (c): the count goes on the CANVAS beside the headline. A
        reader who takes the subtitle and stops must not be left with "2 methods
        evaluated" over a table one of whose verdicts measures nothing.
        """
        svg = _quiet(method_comparison_to_svg, [self.HEALTHY, self.COLLAPSED])
        assert "2 methods evaluated (1 verdict vacuous)" in svg

    def test_it_is_out_of_the_satisfied_numerator(self):
        """``explain._count_true`` counts the ``satisfied`` field's truthiness
        into "N of M method(s) satisfy the fairness constraint"."""
        desc = _desc(_quiet(method_comparison_to_svg, [self.HEALTHY, self.COLLAPSED]))
        assert "1 of 1 method(s) satisfy" in desc, desc
        assert "2 of 2" not in desc and "2 of 1" not in desc
        assert "collapsed to a constant prediction" in desc

    def test_a_page_of_nothing_but_collapsed_methods_says_so(self):
        """PUT THE GUARD ABOVE THE DISPATCH: the whole-page could-not-check
        message must not claim the methods reported no constraint result. They
        each reported one."""
        svg = _quiet(method_comparison_to_svg, [self.COLLAPSED])
        desc = _desc(svg)
        assert "COULD NOT CHECK" in desc
        assert "collapsed to a constant prediction" in desc
        assert "reported no constraint result" not in desc, desc

    def test_the_analysis_report_agrees_with_the_comparison_chart(self):
        """SIBLING RENDERER. Three templates draw this row and all three take
        their state from ``_method_state``."""
        for render in (training_report_to_svg, training_analysis_report_to_svg):
            svg = _quiet(
                render,
                _report(
                    {"accuracy": 0.8, "fairness_violation": 0.02, "constraint_satisfied": True},
                    [self.HEALTHY, self.COLLAPSED],
                ),
            )
            assert ">VACUOUS<" in svg, render.__name__
            assert svg.count(">PASS<") == 1, render.__name__


# ───────────────────────────────────────────────────────────────────────────
# 3. The baseline panel, which reads the same disclosure from a dict.
# ───────────────────────────────────────────────────────────────────────────


class TestTheBaselinePanelObeysItsProducer:
    DISOWNED = {
        "accuracy": NAN,
        "fairness_violation": 0.02,
        "constraint_satisfied": True,
        "accuracy_measured": False,
        "accuracy_not_measured_reason": "no row carried a scorable target",
    }

    @pytest.mark.parametrize("render", [training_report_to_svg, training_analysis_report_to_svg])
    def test_a_disowned_baseline_accuracy_is_not_drawn(self, render):
        svg = _quiet(render, _report(dict(self.DISOWNED)))
        assert ">nan<" not in svg
        assert "nan" not in _desc(svg).lower()
        assert "not measured" in svg
        assert "no row carried a scorable target" in svg, "the producer's reason is dropped"

    @pytest.mark.parametrize("render", [training_report_to_svg, training_analysis_report_to_svg])
    def test_a_finite_baseline_accuracy_the_producer_disowned_is_withheld(self, render):
        """The FLAG half, which the NaN cases above cannot prove: only here do the
        value's shape and the producer's disclosure give different answers."""
        svg = _quiet(
            render,
            _report(
                {
                    "accuracy": 0.4583333333333333,
                    "fairness_violation": 0.02,
                    "constraint_satisfied": True,
                    "accuracy_measured": False,
                    "accuracy_not_measured_reason": "10 of 120 row(s) were not scorable",
                }
            ),
        )
        assert ">0.4583<" not in svg, "the accuracy the producer disowned is drawn in bold"
        assert "0.4583" not in _desc(svg)
        assert "not measured" in svg
        assert "10 of 120 row(s) were not scorable" in svg

    @pytest.mark.parametrize("render", [training_report_to_svg, training_analysis_report_to_svg])
    def test_one_unmeasured_cell_costs_that_cell_and_not_the_page(self, render):
        """``baseline_measured`` was an OR over two INDEPENDENT cells gating a
        block that prints both, so a measured accuracy with no violation beside
        it raised ``TypeError: '>' not supported between instances of 'NoneType'
        and 'float'`` inside the colour branch. The adapter swallows a render
        failure as a warning and returns an 1179-byte fallback card, so the
        reader lost the entire report over one absent number.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svg = render(
                _report(
                    {"accuracy": 0.8200, "fairness_violation": None, "constraint_satisfied": True}
                )
            )
        assert not [w for w in caught if "rendering failed" in str(w.message)], (
            "the render still fails and is swallowed into a fallback card"
        )
        assert len(svg) > 5000, f"this is the fallback card, not the report ({len(svg)} bytes)"
        assert ">0.8200<" in svg, "the cell that WAS measured must still print its number"
        assert "not measured" in svg

    def test_the_accessible_layer_is_never_more_confident_than_the_badge(self):
        """The canvas said "Accuracy and fairness violation were not measured on
        this run" while the ``<desc>`` said "Baseline accuracy ?, violation ?" at
        severity LOW. A question mark is neither a measurement nor a statement
        that there is none, and LOW is the machine-readable all-clear that a
        report pipeline consumes INSTEAD of the sentence.
        """
        svg = _quiet(
            training_report_to_svg,
            _report(
                {
                    "accuracy": None,
                    "fairness_violation": None,
                    "constraint_satisfied": True,
                    "accuracy_measured": False,
                }
            ),
        )
        desc = _desc(svg)
        assert "?" not in desc, desc
        assert "COULD NOT CHECK" in desc
        assert re.search(r"severity:\s*MEDIUM", desc), desc

    def test_a_nan_base_rate_disparity_is_not_drawn_in_the_subtitle(self):
        """The sibling cell one panel down: the per-group positive rate goes
        through ``_coord`` and this one went through nothing, so a NaN rendered
        "Base rate disparity: nan"."""
        svg = _quiet(
            training_analysis_report_to_svg,
            _report(
                {"accuracy": 0.8, "fairness_violation": 0.02, "constraint_satisfied": True},
                fairness_analysis={
                    "group_statistics": {
                        "A": {"size": 60, "proportion": 0.5, "positive_rate": 0.6},
                        "B": {"size": 60, "proportion": 0.5, "positive_rate": 0.4},
                    },
                    "base_rate_disparity": NAN,
                    "base_rate_not_measured_reason": (
                        "at least one group's base rate is not a finite number"
                    ),
                },
            ),
        )
        assert "disparity: nan" not in svg.lower()
        assert "Base rate disparity was not computed" in svg
        assert "not a finite number" in svg, "the analyser's reason is dropped"


# ───────────────────────────────────────────────────────────────────────────
# 4. The engine filters: the LAST door before the markup, for every template.
# ───────────────────────────────────────────────────────────────────────────


class TestNoNumericFilterPrintsASentinel:
    @pytest.mark.parametrize("name", ["f1", "f2", "f3", "f4", "pct", "pct1"])
    @pytest.mark.parametrize("sentinel", [NAN, INF, -INF, True, False])
    def test_the_filter_answers_with_the_third_state(self, name, sentinel):
        """``f"{float('nan'):.4f}"`` is the string "nan" and ``pct`` renders
        "nan%", which reads as a percentage. "N/A" is the spelling these filters
        already use for None, so the third state has one spelling library-wide.
        """
        env = _get_jinja_env()
        assert env is not None
        assert env.filters[name](sentinel) == "N/A"

    def test_it_holds_through_a_real_render_of_a_hand_built_dict(self):
        """Not the filter in isolation: the markup. This data dict claims the
        accuracy IS measured and carries a NaN, which is what a hand-built or
        third-party dict looks like when it bypasses the adapter entirely, so the
        engine filter is the only thing left between it and the reader.
        """
        svg = render_svg(
            "method_comparison",
            {
                "title": "Fairness Training Method Comparison",
                "timestamp": "2026-09-30 09:00",
                "methods": [
                    {
                        "name": "Handmade",
                        "accuracy": NAN,
                        "violation": NAN,
                        "satisfied": True,
                        "time": 1.0,
                        "graded": True,
                        "accuracy_measured": True,
                        "violation_measured": True,
                        "accuracy_reason": "",
                        "vacuous": False,
                        "vacuous_reason": "",
                    }
                ],
                "n_ungraded": 0,
                "n_vacuous": 0,
            },
        )
        assert ">nan<" not in svg, "a NaN reached the markup through a numeric filter"
        assert ">N/A<" in svg


# ───────────────────────────────────────────────────────────────────────────
# 5. THE CONTROLS. A guard that refuses everything passes every refusal test.
#    Each of these asserts the REAL number or the REAL verdict, not absence of
#    an exception.
# ───────────────────────────────────────────────────────────────────────────


class TestHealthyInputRendersExactlyWhatItMeasured:
    def test_the_method_table_prints_its_real_numbers_and_verdicts(self):
        svg = method_comparison_to_svg(
            [
                MethodComparison(
                    "Reweight", 0.8102, 0.0301, True, 1.0, {"accuracy_measured": True}
                ),
                MethodComparison(
                    "Adversarial", 0.7734, 0.1201, False, 2.0, {"accuracy_measured": True}
                ),
            ]
        )
        assert ">0.810<" in svg and ">0.773<" in svg
        assert ">0.030<" in svg and ">0.120<" in svg
        assert svg.count(">PASS<") == 1 and svg.count(">FAIL<") == 1
        assert ">VACUOUS<" not in svg and ">NOT CHECKED<" not in svg
        assert "not measured" not in svg
        desc = _desc(svg)
        assert "1 of 2 method(s) satisfy the fairness constraint." in desc
        assert "not covered by this finding" not in desc
        assert re.search(r"severity:\s*LOW", desc), desc

    @pytest.mark.parametrize("render", [training_report_to_svg, training_analysis_report_to_svg])
    def test_the_baseline_panel_prints_its_real_numbers(self, render):
        svg = render(
            _report(
                {
                    "accuracy": 0.8200,
                    "fairness_violation": 0.0300,
                    "constraint_satisfied": True,
                    "accuracy_measured": True,
                    "accuracy_not_measured_reason": "",
                },
                [
                    MethodComparison(
                        "Reweight", 0.8102, 0.0301, True, 1.0, {"accuracy_measured": True}
                    )
                ],
            )
        )
        assert ">0.8200<" in svg and ">0.0300<" in svg
        assert ">SATISFIED<" in svg
        assert ">0.8102<" in svg and ">0.0301<" in svg
        assert svg.count(">PASS<") == 1
        assert "not measured" not in svg
        desc = _desc(svg)
        assert "Baseline accuracy 0.820, violation 0.030 (constraint satisfied)" in desc
        assert "COULD NOT CHECK" not in desc
        assert re.search(r"severity:\s*LOW", desc), desc

    def test_a_measured_zero_is_a_measurement_and_still_prints(self):
        """The over-correction that would be easiest to ship: withholding on
        truthiness. An accuracy that really came out 0.0, and a violation that
        really came out 0.0, are results."""
        svg = method_comparison_to_svg(
            [MethodComparison("Collapse", 0.0, 0.0, True, 1.0, {"accuracy_measured": True})]
        )
        assert ">0.000<" in svg
        assert "not measured" not in svg
        assert ">PASS<" in svg

    def test_a_false_degeneracy_flag_does_not_withhold_the_verdict(self):
        """``degenerate_constant_predictions=False`` is the producer saying it
        LOOKED and the model did not collapse. That must read as a measured
        verdict, not as a vacuous one."""
        m = MethodComparison(
            "Reweight",
            0.8102,
            0.0301,
            True,
            1.0,
            {"accuracy_measured": True, "degenerate_constant_predictions": False},
        )
        state = _method_state(m)
        assert state["vacuous"] is False and state["graded"] is True
        svg = method_comparison_to_svg([m])
        assert ">PASS<" in svg and ">VACUOUS<" not in svg

    def test_the_finder_still_reads_a_legacy_dict_with_no_disclosure_keys(self):
        """An absent flag must not withhold a number the dict does carry: these
        keys did not exist before 2026-09-30 and third-party dicts still omit
        them."""
        explanation = build_explanation(
            "training_report",
            {
                "baseline_accuracy": 0.82,
                "baseline_violation": 0.03,
                "baseline_satisfied": True,
                "recommendation": {"method": "Reweight", "priority": "low"},
                "issues": [],
            },
        )
        assert "Baseline accuracy 0.820, violation 0.030" in explanation.finding
        assert explanation.severity == "low"


# ───────────────────────────────────────────────────────────────────────────
# 6. The THIRD way a group leaves the comparison, which the renderer read two
#    of. Driven end to end through the real engine, not a hand-built dict.
# ───────────────────────────────────────────────────────────────────────────


class TestAGroupThatLostEveryRowIsNamed:
    """``data_info['groups_dropped']`` reaches the reader.

    The producer publishes three independent reasons a group is absent from
    every disparity metric, and ``_excluded_groups`` read two of them. A group
    that lost EVERY row to missing-data cleaning is not too small and is not
    short of evidence: it is gone, so it appears in neither of the two keys the
    renderer read, and no canvas named it.
    """

    MIN_SIZE = 30

    @staticmethod
    def _report(*, drop_every_row_of_c: bool):
        import numpy as np

        from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

        rng = np.random.default_rng(0)
        if drop_every_row_of_c:
            y_true = np.concatenate([rng.integers(0, 2, 40) for _ in range(3)]).astype(float)
            y_pred = y_true.copy()
            y_pred[80:] = NAN  # every one of c's predictions is missing
            sens = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40)
        else:
            # THE CONTROL PATH the renderer already read: c is present and below
            # min_group_size. Its wording must not move.
            y_true = np.concatenate(
                [rng.integers(0, 2, 40), rng.integers(0, 2, 40), rng.integers(0, 2, 10)]
            ).astype(float)
            y_pred = y_true.copy()
            sens = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 10)
        return _quiet(
            lambda: FairnessAnalyzer(
                y_true, y_pred, sens, min_group_size=TestAGroupThatLostEveryRowIsNamed.MIN_SIZE
            ).get_report()
        )

    def test_the_engine_really_does_report_the_drop(self):
        """Non-vacuity: if this stops holding, every assertion below is checking
        a fixture in which nothing was dropped."""
        info = self._report(drop_every_row_of_c=True)["data_info"]
        assert info["groups_dropped"] == ["c"]
        assert (info["n_groups_before"], info["n_groups_after"]) == (3, 2)
        assert not info.get("invalid_groups"), (
            "this fixture drops c through the missing-data path, so it must NOT also be "
            "below min_group_size, or the old code would have named it anyway"
        )

    @pytest.mark.parametrize("chart", ["radar_chart_to_svg", "disparity_heatmap_to_svg"])
    def test_the_canvas_and_the_desc_both_name_it(self, chart):
        import vfairness.rendering.adapters_fairness as AF

        svg = _quiet(getattr(AF, chart), self._report(drop_every_row_of_c=True))
        desc = _desc(svg)
        assert "Excluded: c (0 rows left)" in desc, desc
        assert "Excluded" in svg.split("<desc>")[0] + svg.split("</desc>")[-1] or "Excluded" in svg
        assert re.search(r"severity:\s*MEDIUM", desc), (
            "an all-clear produced by losing a whole group is published at the benign "
            f"severity: {desc}"
        )

    def test_the_min_group_size_path_is_worded_exactly_as_before(self):
        """THE CONTROL. The arm that already worked must not move, and it must
        keep its own wording: "(n=10)" is a MEASURED size and "0 rows left" is
        not a smaller version of it."""
        import vfairness.rendering.adapters_fairness as AF

        desc = _desc(_quiet(AF.radar_chart_to_svg, self._report(drop_every_row_of_c=False)))
        assert "Excluded: c (n=10)" in desc, desc
        assert "0 rows left" not in desc

    def test_a_run_that_dropped_nothing_carries_no_clause(self):
        """The over-correction control: a clean run's sentence must be what it
        always was, with no caveat and at its own severity."""
        import numpy as np

        import vfairness.rendering.adapters_fairness as AF
        from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

        rng = np.random.default_rng(0)
        y_true = np.concatenate([rng.integers(0, 2, 40), rng.integers(0, 2, 40)]).astype(float)
        report = _quiet(
            lambda: FairnessAnalyzer(
                y_true, y_true.copy(), np.array(["a"] * 40 + ["b"] * 40), min_group_size=30
            ).get_report()
        )
        assert report["data_info"]["groups_dropped"] == []
        desc = _desc(_quiet(AF.radar_chart_to_svg, report))
        assert "Excluded" not in desc, desc
        assert "0 rows left" not in desc
        assert re.search(r"severity:\s*INFO", desc), desc

    def test_two_different_mechanisms_are_both_named(self):
        """PINS THE UNION, which the two engine fixtures above cannot: neither of
        them puts a group in both populations, so neither would notice
        ``groups_dropped`` being folded in under the ``if not names`` fallback and
        silenced whenever another group was already named. One group below
        min_group_size and a DIFFERENT one that lost every row is an ordinary
        outcome of one run, and a reader must be told about both.
        """
        from vfairness.rendering.adapters_fairness import _excluded_groups

        names = _excluded_groups(
            {
                "assessment": {"assessable": True, "fairness_score": 92.0},
                "data_info": {
                    "invalid_groups": ["small"],
                    "group_sizes": {"small": 7},
                    "groups_dropped": ["gone"],
                    "valid_groups": ["a", "b"],
                },
            }
        )
        assert names == ["small (n=7)", "gone (0 rows left)"], names

    def test_a_group_in_both_populations_is_named_once(self):
        """The de-duplication, so the subtitle it lands in is not doubled."""
        from vfairness.rendering.adapters_fairness import _excluded_groups

        names = _excluded_groups(
            {
                "assessment": {
                    "assessable": True,
                    "insufficient_evidence_groups": [{"group": "c", "n": 4}],
                },
                "data_info": {"groups_dropped": ["c"], "valid_groups": ["a", "b"]},
            }
        )
        assert names == ["c (n=4)"], names


# ───────────────────────────────────────────────────────────────────────────
# 7. The fallback card: the one canvas a reader gets when the chart did not draw.
# ───────────────────────────────────────────────────────────────────────────


class TestTheFallbackCardPrintsNoSentinelEither:
    @pytest.mark.parametrize("sentinel", [NAN, INF, -INF, True, False, None, "x"])
    def test_a_sentinel_is_a_phrase_and_not_a_number(self, sentinel):
        from vfairness.rendering.adapters_training import _fmt3, _fmt4

        assert _fmt4(sentinel) == "not measured"
        assert _fmt3(sentinel) == "not measured"

    def test_the_fallback_card_itself_prints_no_nan(self):
        from vfairness.rendering.adapters_training import _generate_fallback_svg

        svg = _generate_fallback_svg(
            {
                "title": "Fairness Training Report",
                "baseline_accuracy": NAN,
                "baseline_violation": NAN,
                "methods": [
                    {"name": "Threshold", "accuracy": NAN, "violation": NAN, "graded": False}
                ],
                "recommendation": {"method": "N/A", "rationale": "none"},
            }
        )
        assert "nan" not in svg.lower()
        assert svg.count("not measured") >= 4

    def test_a_measured_number_still_prints_on_the_fallback_card(self):
        """CONTROL. The fallback card's job is to carry what WAS measured."""
        from vfairness.rendering.adapters_training import _fmt3, _fmt4, _generate_fallback_svg

        assert _fmt4(0.0) == "0.0000" and _fmt3(0.0) == "0.000"
        assert _fmt4(0.8123456) == "0.8123" and _fmt3(0.8123456) == "0.812"
        svg = _generate_fallback_svg(
            {
                "title": "Fairness Training Report",
                "baseline_accuracy": 0.8200,
                "baseline_violation": 0.0300,
                "methods": [
                    {
                        "name": "Reweight",
                        "accuracy": 0.8102,
                        "violation": 0.0301,
                        "graded": True,
                        "satisfied": True,
                    }
                ],
                "recommendation": {"method": "Reweight", "rationale": "best trade-off"},
            }
        )
        assert "0.8200" in svg and "0.0300" in svg
        assert "0.810" in svg and "0.030" in svg
        assert "not measured" not in svg
