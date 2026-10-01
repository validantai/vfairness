"""G13 grading pins for the seven SVG adapters in this batch.

Every assertion here reads the RENDERED MARKUP, never the data dict the adapter
hands the template: a pin on the producer's own field would have been green for
five of the six defects below, because the field was right and the boundary
after it was wrong.

Measured and fixed on 2026-09-30:

1. ``adapters_workflow.report_card_to_svg`` copied ``ev.passed`` straight from
   the GateDecision. The gate sets ``passed`` False for a COULD-NOT-CHECK as
   well as for a measured breach, and both ``report_card.svg`` and
   ``explain._fr_report_card`` key their third state on ``passed is none``, so
   that whole third state was unreachable from the only producer that feeds it.
2. ``adapters_reporting._row_outcome`` reinvented the vacuous-bound rule and
   covered only the higher-is-better half, so every vacuous MAXIMUM was graded:
   0.01 against a maximum of 2.0 on a metric whose range is [0, 1] rendered a
   green PASS.
3. ``adapters_ranking`` read the metric value as ``float(raw) if raw is not
   None``, so ``value=True`` became 1.0 and drew a green PASS against an
   exposure-parity floor, and ``value="N/A"`` raised ValueError and took the
   whole export down.
4. ``adapters_robustness`` did the same for the robustness score, the observed
   statistic, the max deviation and both subgroup counts: ``nan`` raised out of
   ``int(score * 200)``, ``np.True_`` rendered ROBUST at 1.00, and a NaN
   ``worst_disparity`` passed the template's ``is not none`` gate to print
   "worst: <name> (N/A)" instead of "worst disparity not reported".
5. ``adapters_validation._fmt_val`` printed the string "nan" for a NaN metric,
   because it formats in Python and never passes the engine's ``_to_float`` net.
6. The same module read ``iss.get("severity", "INFO")``: a key present holding
   None skipped the default and badged the issue "NONE", and an issue with no
   severity at all was badged INFO in informational blue instead of the UNKNOWN
   its own classifier would have given it.

Each class carries an over-correction control on a healthy input, because an
adapter that refuses everything passes every refusal test.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from vfairness.operations.cicd.gate import (
    GateDecision,
    GateStatus,
    MetricEvaluation,
    ModelFairnessGate,
    _metric_row_state,
)
from vfairness.rendering.adapters_discovery import auto_discovery_to_svg
from vfairness.rendering.adapters_ranking import ranking_fairness_to_svg
from vfairness.rendering.adapters_regression import regression_fairness_to_svg
from vfairness.rendering.adapters_reporting import reporting_dashboard_to_svg
from vfairness.rendering.adapters_robustness import robustness_testing_to_svg
from vfairness.rendering.adapters_validation import data_validation_to_svg
from vfairness.rendering.adapters_workflow import report_card_to_svg

NAN = float("nan")


def _texts(svg: str):
    return [t.strip() for t in re.findall(r">([^<>]*)<", svg) if t.strip()]


def _badges(svg: str):
    return re.findall(r">(PASS|FAIL|WARN|NOT GRADED|NOT MEASURED|NOT CHECKED)<", svg)


def _desc(svg: str) -> str:
    m = re.search(r"<desc>(.*?)</desc>", svg, re.S)
    return m.group(1) if m else ""


class _Dict:
    """A result object whose only interface is to_dict, like the producers'."""

    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


# ── 1. report_card_to_svg ───────────────────────────────────────────────────


def _one_group_decision():
    """A gate decision whose metric CANNOT be computed: one protected group."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    gate = ModelFairnessGate(thresholds={"demographic_parity_difference": 0.1})
    return gate.evaluate(y_true=y, y_pred=y, protected_attr=np.array(["a"] * 200))


class TestTheReportCardCarriesTheGatesOwnRowVerdict:
    def test_a_metric_the_gate_could_not_compute_is_not_badged_a_breach(self):
        """Before: RESULT cells ['FAIL'] and desc "0/1 metrics pass", for a
        value the same object reports as not measured. The markdown report of
        the SAME decision said "Could not check"."""
        decision = _one_group_decision()
        states = [_metric_row_state(ev) for ev in decision.metric_evaluations]
        assert "unmeasured" in states, "fixture must carry an ungradable row"

        svg = report_card_to_svg(decision=decision, model_name="one_group")

        assert "FAIL" not in _badges(svg)
        assert "NOT GRADED" in _badges(svg)
        assert "not measured" in _texts(svg)
        assert "COULD NOT CHECK" in _desc(svg)
        # Fail-closed is preserved: the page verdict is still blocked.
        assert "blocked" in _desc(svg)

    def test_the_pass_rate_denominator_counts_only_the_graded_rows(self):
        evs = [
            MetricEvaluation("demographic_parity_difference", 0.02, 0.1, True),
            MetricEvaluation("equal_opportunity_difference", 0.40, 0.1, False),
            MetricEvaluation("disparate_impact_ratio", NAN, 0.8, False),
            # A vacuous maximum: 2.0 on a metric whose range is [0, 1].
            MetricEvaluation("statistical_parity_difference", 0.01, 2.0, False),
        ]
        decision = GateDecision(
            approved=False,
            status=GateStatus.BLOCKED,
            metric_evaluations=evs,
            blocking_reasons=["x"],
            warnings=[],
        )

        svg = report_card_to_svg(decision=decision, model_name="mixed")

        assert _badges(svg) == ["PASS", "FAIL", "NOT GRADED", "NOT GRADED"]
        n_graded = sum(1 for ev in evs if _metric_row_state(ev) != "unmeasured")
        n_pass = sum(1 for ev in evs if _metric_row_state(ev) == "pass")
        assert f"{n_pass}/{n_graded}" in svg
        assert "not graded, not in this rate" in svg

    def test_control_rows_that_really_were_measured_keep_their_verdicts(self):
        """If this reddens, the fix is refusing rows it should grade."""
        evs = [
            MetricEvaluation("demographic_parity_difference", 0.02, 0.1, True),
            MetricEvaluation("equal_opportunity_difference", 0.40, 0.1, False),
        ]
        decision = GateDecision(
            approved=False,
            status=GateStatus.BLOCKED,
            metric_evaluations=evs,
            blocking_reasons=["x"],
            warnings=[],
        )

        svg = report_card_to_svg(decision=decision, model_name="real")

        assert _badges(svg) == ["PASS", "FAIL"]
        assert "NOT GRADED" not in svg
        assert "1/2" in svg

    def test_control_no_decision_at_all_still_refuses(self):
        svg = report_card_to_svg()
        assert "PASS" not in _badges(svg)
        assert "no metric was evaluated" in svg.lower()


# ── 2. reporting_dashboard_to_svg ───────────────────────────────────────────


def _dashboard(metrics):
    return reporting_dashboard_to_svg(
        {
            "health_score": {"score": 88, "status": "green"},
            "tier": "OPERATIONAL",
            "metrics": metrics,
        }
    )


class TestTheDashboardRefusesABoundNoDataCanBreach:
    @pytest.mark.parametrize(
        "name, value, threshold",
        [
            ("demographic_parity_difference", 0.01, 2.0),
            ("equal_opportunity_difference", 0.02, 5.0),
            # the half that was already covered, kept so the guard above the
            # dispatch is shown to carry both
            ("disparate_impact_ratio", 0.45, 0.0),
        ],
    )
    def test_a_vacuous_bound_grades_nothing(self, name, value, threshold):
        """Before, for the two maxima: a green PASS swatch for a requirement no
        value in the metric's range can breach."""
        svg = _dashboard([{"name": name, "value": value, "threshold": threshold}])

        assert "[UNCHECKED]" in svg
        assert "NOT CHECKED" in svg

    @pytest.mark.parametrize(
        "name, value, threshold",
        [
            ("demographic_parity_difference", 0.01, 0.1),
            ("demographic_parity_difference", 0.45, 0.1),
            # a REAL zero-tolerance maximum, which abs(value) can exceed
            ("demographic_parity_difference", 0.01, 0.0),
            ("disparate_impact_ratio", 0.62, 0.8),
            ("disparate_impact_ratio", 0.95, 0.8),
        ],
    )
    def test_control_a_bound_that_can_be_breached_is_still_graded(self, name, value, threshold):
        svg = _dashboard([{"name": name, "value": value, "threshold": threshold}])

        assert "[UNCHECKED]" not in svg
        assert "NOT CHECKED" not in svg


# ── 3. ranking_fairness_to_svg ──────────────────────────────────────────────


class TestTheRankingBadgeRefusesANonMeasurement:
    def test_a_boolean_value_is_not_a_passing_exposure_ratio(self):
        """Before: float(True) == 1.0, the best value on a four-fifths floor,
        rendered as a GREEN PASS badge for a row carrying no measurement."""
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity_ratio", "value": True, "threshold": 0.8}]
        )

        assert "PASS" not in _badges(svg)
        assert "NOT MEASURED" in _badges(svg)

    def test_an_unparseable_cell_is_one_ungraded_row_not_a_dead_page(self):
        """Before: ValueError out of float("N/A"), so one bad cell destroyed
        every measured row on the page with it."""
        svg = ranking_fairness_to_svg(
            [
                {"metric_name": "exposure_parity_ratio", "value": "N/A", "threshold": 0.8},
                {"metric_name": "ndkl", "value": 0.05, "threshold": 0.1},
            ]
        )

        assert "NOT MEASURED" in _badges(svg)
        assert "PASS" in _badges(svg), "the measured row must survive"

    def test_control_a_serialised_numeric_string_is_still_a_measurement(self):
        """JSON and CSV rows arrive with numbers as text; refusing them would
        discard evidence. 0.62 against a 0.80 FLOOR is a FAIL."""
        svg = ranking_fairness_to_svg(
            [{"metric_name": "exposure_parity_ratio", "value": "0.62", "threshold": 0.8}]
        )

        assert _badges(svg) == ["FAIL"]

    def test_control_an_unmeasurable_number_still_fails_closed(self):
        """NaN and inf must NOT be relaxed into the neutral state: this page has
        always failed closed on a metric that was requested and not produced."""
        for bad in (NAN, float("inf")):
            svg = ranking_fairness_to_svg(
                [{"metric_name": "exposure_parity_ratio", "value": bad, "threshold": 0.8}]
            )
            assert _badges(svg) == ["FAIL"], bad

    def test_a_row_whose_name_is_none_is_not_labelled_none(self):
        svg = ranking_fairness_to_svg([{"metric_name": None, "value": 0.62, "threshold": 0.8}])
        assert "NONE" not in _texts(svg)


# ── 4. robustness_testing_to_svg ────────────────────────────────────────────


class TestTheRobustnessScoreIsNotDerivedFromANonMeasurement:
    def test_a_refused_score_does_not_take_the_export_down(self):
        """Before: ValueError('cannot convert float NaN to integer') out of the
        bar-width arithmetic, for the value this library uses to say a check
        REFUSED to run."""
        svg = robustness_testing_to_svg(
            None, [{"perturbation_type": "dropout", "robustness_score": NAN}]
        )

        assert "ROBUST" not in _texts(svg)
        assert "NOT CHECKED" in _badges(svg)

    @pytest.mark.parametrize("flag", [True, np.True_])
    def test_a_boolean_is_not_a_perfect_robustness_score(self, flag):
        """float(True) is 1.0, the best score on this scale, and np.bool_ is the
        half a plain isinstance check misses."""
        svg = robustness_testing_to_svg(
            None, [{"perturbation_type": "dropout", "robustness_score": flag}]
        )

        assert "ROBUST" not in _texts(svg)
        assert "1.00" not in _texts(svg)

    def test_a_nan_worst_disparity_is_reported_as_not_reported(self):
        """Before: the template's `is not none` gate let NaN through and printed
        "worst: age_under_25 (N/A)", which reads as a measured disparity whose digits
        were lost."""
        svg = robustness_testing_to_svg(
            None,
            None,
            {"n_subgroups_analyzed": 6, "worst_subgroup": "age_under_25", "worst_disparity": NAN},
        )

        assert "worst disparity not reported" in svg
        assert "worst: age_under_25" not in svg

    def test_control_a_real_score_is_still_graded_and_drawn(self):
        svg = robustness_testing_to_svg(
            None, [{"perturbation_type": "dropout", "robustness_score": 0.85}]
        )

        assert "ROBUST" in _texts(svg)
        assert "0.85" in svg

    def test_control_a_real_worst_disparity_is_still_printed(self):
        svg = robustness_testing_to_svg(
            None,
            None,
            {"n_subgroups_analyzed": 6, "worst_subgroup": "age_under_25", "worst_disparity": 0.31},
        )

        assert "worst: age_under_25" in svg
        assert "0.310" in svg


# ── 5 and 6. data_validation_to_svg ─────────────────────────────────────────


class TestTheValidationCanvasNeverPrintsAMintedValue:
    def test_a_nan_metric_is_not_printed_as_the_word_nan(self):
        """`f"{float('nan'):.3f}"` is the string "nan", and this formatter runs
        in Python so the engine's `_to_float` net never sees it."""
        svg = data_validation_to_svg(
            _Dict({"passed": True, "checks_run": 3, "metrics": {"missing_rate": NAN}})
        )

        assert "nan" not in [t.lower() for t in _texts(svg)]
        assert "N/A" in _texts(svg)

    @pytest.mark.parametrize("bad", [None, NAN, "bogus"], ids=["None", "nan", "bogus"])
    def test_an_ungradable_severity_is_badged_unknown_not_stringified(self, bad):
        """Before: "NONE", "NAN" and "BOGUS" were printed as severity badges,
        content MINTED by `str()` out of a severity the page cannot grade, while
        the colour and the counter both said UNKNOWN."""
        svg = data_validation_to_svg(
            _Dict(
                {
                    "passed": True,
                    "checks_run": 3,
                    "issues": [{"issue_type": "t", "severity": bad, "message": "m"}],
                }
            )
        )

        assert "UNKNOWN" in _texts(svg)
        for minted in ("NONE", "NAN", "BOGUS"):
            assert minted not in _texts(svg)

    def test_an_issue_with_no_severity_is_not_filed_as_advisory(self):
        """Before: `iss.get("severity", "INFO")` defaulted the VERDICT above the
        classifier that exists to answer it, so an issue nobody graded was
        badged INFO in informational blue and counted in the info bucket."""
        svg = data_validation_to_svg(
            _Dict(
                {
                    "passed": True,
                    "checks_run": 3,
                    "issues": [{"issue_type": "t", "message": "m"}],
                }
            )
        )

        assert "UNKNOWN" in _texts(svg)
        assert "INFO" not in _texts(svg)

    def test_control_a_real_severity_is_badged_as_reported(self):
        svg = data_validation_to_svg(
            _Dict(
                {
                    "passed": False,
                    "checks_run": 3,
                    "issues": [
                        {"issue_type": "t", "severity": "CRITICAL", "message": "m"},
                        {"issue_type": "u", "severity": "warning", "message": "m"},
                    ],
                }
            )
        )

        assert "CRITICAL" in _texts(svg)
        assert "WARNING" in _texts(svg)
        assert "UNKNOWN" not in _texts(svg)

    def test_control_a_real_metric_is_still_printed(self):
        svg = data_validation_to_svg(
            _Dict({"passed": True, "checks_run": 3, "metrics": {"missing_rate": 0.125}})
        )
        assert "0.125" in _texts(svg)


# ── the two adapters in this batch where no defect was found ────────────────


class TestRegressionAndDiscoveryRefuseAndMeasure:
    """No defect was measured in these two beyond the minted-name line; both
    were executed on a refusal shape, a healthy shape and every degenerate
    field, and both already withhold per row through `_finite` /
    `_number_or_none`. These pins hold that."""

    def test_regression_withholds_every_cell_of_a_group_that_reported_nothing(self):
        svg = regression_fairness_to_svg(
            {"Female": {}, "Male": {"mae": 0.28, "rmse": 0.4, "r2": 0.8, "n": 100}}
        )

        assert "reported no mean residual" in svg
        assert "0.280" in svg, "the measured group must keep its real number"

    @pytest.mark.parametrize("bad", [NAN, True, "x", None])
    def test_regression_never_plots_a_pair_it_could_not_compare(self, bad):
        svg = regression_fairness_to_svg(
            {"a": {"mae": 0.2, "n": 10}}, None, [{"pair": "a-b", "cohens_d": bad}]
        )
        assert "reported no effect size" in svg

    def test_regression_control_a_real_effect_size_is_drawn(self):
        svg = regression_fairness_to_svg(
            {"a": {"mae": 0.2, "n": 10}}, None, [{"pair": "a-b", "cohens_d": 0.62}]
        )
        assert "reported no effect size" not in svg
        assert "Large effect size detected" in svg

    def test_discovery_refuses_a_scan_that_never_ran(self):
        svg = auto_discovery_to_svg()
        assert "No significant fairness issues auto-discovered." not in svg

    def test_discovery_withholds_a_row_that_reported_no_number(self):
        svg = auto_discovery_to_svg(
            [{"name": "zip", "confidence": None}],
            [{"attribute": "zip", "metric": "spd", "severity": None}],
            None,
        )
        assert "not reported" in svg
        assert "NOT GRADED" in svg
        assert "None" not in _texts(svg)

    def test_discovery_does_not_list_an_attribute_called_none(self):
        """A candidate whose `name` field is PRESENT holding None skipped the
        `.get` default and reached `str(None)`, so the panel listed an attribute
        literally called "None". The row above does not reach this door: its
        name is populated, which is why this case is separate."""
        svg = auto_discovery_to_svg([{"name": None, "attribute": None, "confidence": 0.8}])

        assert "None" not in _texts(svg)
        assert "attr_0" in _texts(svg)

    def test_discovery_control_a_real_scan_reports_its_real_finding(self):
        svg = auto_discovery_to_svg(
            [{"name": "zipcode", "confidence": 0.82, "category": "geo"}],
            [
                {
                    "attribute": "zipcode",
                    "metric": "spd",
                    "value": 0.31,
                    "threshold": 0.1,
                    "severity": "high",
                }
            ],
            [],
        )
        assert "0.310" in svg
        assert "HIGH" in _texts(svg)
        assert "not reported" not in svg
