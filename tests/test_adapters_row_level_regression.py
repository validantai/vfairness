"""Row-level could-not-check tests for the SVG adapters.

The campaign that closed the CHART-level fabricated all-clear left the same
defect alive one level down. An adapter whose headline correctly reads NOT
ASSESSABLE was still filling the rows underneath it from numeric defaults, and
an honest headline over invented rows is worse than a wrong headline: the badge
lends the rows its credibility.

Every test here is written from a row, a cell or a count that reported NOTHING.
The rule under test is the same one the chart-level waves established, applied
per row: a row that reported nothing gets no number, no badge, no colour, no
point on a plot, and no place in any count or headline that implies it was
measured. A default is not a measurement, a sentinel is not a measurement, and
absent and zero are different.

What each class pins, and what it looked like before 2026-08-27:

* ``TestRegressionGroupRows`` - ``regression_fairness_to_svg({"Female": {}}, ...)``
  tabulated the group as N 0, MAE 0.000, RMSE 0.000, R-squared 0.000, MEAN RES.
  +0.0000, STD RES. 0.000, and drew a residual bar labelled "over-predict". Six
  invented measurements and a plotted point, under a correct NOT ASSESSABLE
  badge.
* ``TestHierarchicalGateLevelBadges`` - a decision carrying no metric evaluation
  rendered OVERALL FAIL 0/0 metrics, SINGLE ATTRIBUTE FAIL 0/0 checks and
  INTERSECTIONAL FAIL 0/0 combos. Three fabricated failures for three levels
  nobody ran.
* ``TestCorrelationHeatmapUncomputedCells`` - a correlation that could not be
  computed was rewritten to 0.0, drawn as "0.00", counted into "0 HIGH
  CORRELATIONS" and cleared by "No high correlations detected". A pair nobody
  tested was reported as a pair tested and found low.
* ``TestBiasAuditCoverageCount`` - the partial-coverage sentence counted the
  names the report SUPPLIED rather than the ones it RECOGNISED, so an unknown
  module name inflated how much of the audit was credited as having run.
"""

import math
import re

import pytest

pytest.importorskip("jinja2", reason="SVG rendering needs the [rendering] extra")

from vfairness.preprocessing.bias_detection.detector import (  # noqa: E402
    BiasAuditReport,
)
from vfairness.rendering.adapters import bias_audit_to_svg  # noqa: E402
from vfairness.rendering.adapters_feature_engineering import (  # noqa: E402
    correlation_heatmap_to_svg,
)
from vfairness.rendering.adapters_regression import (  # noqa: E402
    regression_fairness_to_svg,
)
from vfairness.rendering.adapters_workflow import (  # noqa: E402
    hierarchical_gate_to_svg,
)
from vfairness.rendering.skins import BLANCO_MAP  # noqa: E402


def _skinned(hex_color: str) -> str:
    """The colour as it survives the Blanco skin, which rewrites the palette.

    Asserting on the raw template colour passes vacuously: the skin has already
    rewritten it by the time the markup exists.
    """
    return BLANCO_MAP.get(hex_color, hex_color)


# The measurement colour of the per-group table. A figure painted in it is being
# presented to the reader as something the run actually measured.
_MEASURED_FILL = _skinned("#475569")
_FAIL_FILL = _skinned("#dc2626")
_FAIL_BG = _skinned("#fef2f2")
_UNCHECKED_FILL = _skinned("#64748b")


def _canvas_only(svg: str) -> str:
    """The drawn canvas, with the accessibility layer stripped.

    ``<desc>``/``<metadata>`` are built by ``rendering.explain`` from the same
    data dict but by a separate finder, so a claim there is a different defect
    from a claim on the canvas and must not silently satisfy a canvas test.
    """
    svg = re.sub(r"<desc\b.*?</desc>", "", svg, flags=re.S)
    svg = re.sub(r"<metadata\b.*?</metadata>", "", svg, flags=re.S)
    return re.sub(r"<title\b.*?</title>", "", svg, flags=re.S)


HEALTHY_GROUPS = {
    "Female": {
        "size": 980,
        "mae": 0.168,
        "rmse": 0.212,
        "r2": 0.791,
        "mean_residual": -0.045,
        "std_residual": 0.201,
    },
    "Male": {
        "size": 1200,
        "mae": 0.142,
        "rmse": 0.185,
        "r2": 0.823,
        "mean_residual": 0.032,
        "std_residual": 0.178,
    },
}

ALL_PASSING = {
    "mae_parity": 0.02,
    "rmse_parity": 0.03,
    "mean_pred_diff": 0.01,
    "r2_parity": 0.04,
}


def _texts(svg: str):
    """Every rendered text run, in document order."""
    return [t for t in re.findall(r">([^<>]+)<", svg) if t.strip()]


def _measured_numbers(svg: str):
    """Numbers painted in the table's measurement colour."""
    return [
        m.group(1)
        for m in re.finditer(rf'fill="{_MEASURED_FILL}"[^>]*>([^<>]+)<', svg)
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", m.group(1).strip())
    ]


class TestRegressionGroupRows:
    """A group row may only print what the caller measured for that group."""

    def test_a_group_carrying_only_its_name_gets_no_number_at_all(self):
        svg = regression_fairness_to_svg({"Female": {}}, ALL_PASSING)

        assert ">Female<" in svg, "the group itself is still listed"
        assert not _measured_numbers(svg), (
            "a group that reported nothing was given figures in the measurement "
            f"colour: {_measured_numbers(svg)}"
        )
        # Six columns, six withheld cells: N, MAE, RMSE, R-squared, mean and std
        # residual. Before the fix every one of them read as a measured 0.
        assert svg.count(">N/A<") >= 6

    def test_a_group_that_reported_nothing_gets_no_zero(self):
        svg = regression_fairness_to_svg({"Female": {}}, ALL_PASSING)

        assert ">0.000<" not in svg
        assert ">+0.0000<" not in svg
        # A bare ">0<" would also match the two axis origin labels, which are
        # part of the chart furniture and not a claim about this group.
        assert f'fill="{_MEASURED_FILL}">0<' not in svg

    def test_a_group_with_no_residual_is_not_plotted_on_the_residual_axis(self):
        svg = regression_fairness_to_svg({"Female": {}}, ALL_PASSING)

        # The bar is the only element carrying opacity 0.6 in this panel. A bar of
        # width zero sitting on the axis still reads as "measured, and neutral".
        assert 'opacity="0.6"' not in svg
        assert "over-predict" not in svg
        assert "under-predict" not in svg
        assert "no residual reported" in svg

    def test_the_direction_word_is_not_invented_from_a_defaulted_zero(self):
        """0.0 satisfies `>= 0`, so the missing residual used to read over-predict."""
        svg = regression_fairness_to_svg({"A": {"mae": 0.1}}, ALL_PASSING)

        assert "over-predict" not in svg

    def test_a_partly_reported_group_keeps_the_cells_it_did_measure(self):
        svg = regression_fairness_to_svg({"Female": {"mae": 0.168, "size": 980}}, ALL_PASSING)

        assert ">0.168<" in svg and ">980<" in svg
        # rmse, r2, mean_residual and std_residual were not supplied.
        assert svg.count(">N/A<") >= 4

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "n/a"])
    def test_a_value_that_is_not_a_finite_number_is_withheld_not_printed(self, bad):
        svg = regression_fairness_to_svg({"Female": {"mae": bad, "rmse": 0.2}}, ALL_PASSING)

        assert ">0.200<" in svg, "the sibling cell that WAS measured still prints"
        assert "nan" not in svg.lower().replace("financial", "")
        assert "inf" not in [t.strip().lower() for t in _texts(svg)]
        assert ">N/A<" in svg

    def test_a_group_with_no_residual_is_named_in_the_recommendations(self):
        svg = regression_fairness_to_svg({"Female": {}, "Male": {}}, ALL_PASSING)

        assert "2 group(s) reported no mean residual" in svg
        assert "Female" in svg and "Male" in svg
        assert "residual bias was not checked for them" in svg

    def test_rows_that_measured_nothing_do_not_earn_the_clean_sheet_line(self):
        """The fall-through recommendation is an all-clear across every check."""
        svg = regression_fairness_to_svg({"Female": {}}, ALL_PASSING)

        assert "All regression fairness checks pass" not in svg

    def test_a_fully_reported_healthy_report_is_unchanged(self):
        svg = regression_fairness_to_svg(HEALTHY_GROUPS, ALL_PASSING)

        assert ">N/A<" not in svg
        assert ">0.168<" in svg and ">0.212<" in svg and ">0.791<" in svg
        assert ">980<" in svg and ">1200<" in svg
        assert ">-0.0450<" in svg and ">+0.0320<" in svg
        assert "over-predict" in svg and "under-predict" in svg
        assert 'opacity="0.6"' in svg
        assert "All regression fairness checks pass" in svg
        assert "reported no mean residual" not in svg
        assert "EQUITABLE" in svg


# Minimal stand-ins for IntersectionalGateDecision and its parts. The adapter
# reads only these attributes.


class _Eval:
    def __init__(self, name, value, threshold, passed):
        self.metric_name = name
        self.value = value
        self.threshold = threshold
        self.passed = passed


class _LevelDecision:
    def __init__(self, approved, evaluations):
        self.approved = approved
        self.metric_evaluations = evaluations


class _Decision:
    def __init__(self, approved, level_results):
        self.approved = approved
        self.level_results = level_results
        self.small_sample_warnings = []
        self.hierarchical_config = None


class TestHierarchicalGateLevelBadges:
    """A level with no check is NOT CHECKED, never a red FAIL."""

    def test_a_decision_with_no_metric_evaluated_shows_no_failure(self):
        canvas = _canvas_only(
            hierarchical_gate_to_svg(_Decision(True, {"overall": _LevelDecision(True, [])}))
        )
        # The summary panel: the three level badges, up to the first detail panel.
        summary = canvas.split("Overall (all groups)", 1)[0]

        assert ">FAIL<" not in summary, "three levels nobody ran were reported as failures"
        assert summary.count(">NOT CHECKED<") == 3
        assert "0/0 metrics" not in summary
        assert "0/0 checks" not in summary
        assert "0/0 combos" not in summary

    # CLOSED 2026-08-28. This carried an xfail recording that
    # rendering.explain._fr_hierarchical still wrote "Hierarchical gate APPROVED
    # - overall 0/0, single-attr 0/0, intersectional 0/0 metrics passed" into the
    # explanation panel and the accessible <desc>, at severity INFO, for a gate
    # that evaluated nothing, while the badges above already said NOT CHECKED.
    # The finder learned the third state, so the marker is gone rather than left
    # to xpass: an xfail that passes records a defect that no longer exists, and
    # the next reader takes it as a live one.
    def test_the_explanation_panel_does_not_claim_an_approved_gate_either(self):
        svg = hierarchical_gate_to_svg(_Decision(True, {"overall": _LevelDecision(True, [])}))

        assert "0/0 metrics passed" not in svg
        assert "gate APPROVED" not in svg

    def test_the_unchecked_badges_are_slate_and_not_the_failure_red(self):
        canvas = _canvas_only(
            hierarchical_gate_to_svg(_Decision(True, {"overall": _LevelDecision(True, [])}))
        )

        # The summary panel, up to where the per-level detail panels begin.
        head = canvas.split("Overall (all groups)", 1)[0]
        assert _FAIL_FILL not in head
        assert _FAIL_BG not in head
        assert head.count(_UNCHECKED_FILL) >= 3

    def test_a_level_that_was_evaluated_and_failed_is_still_reported_as_failing(self):
        svg = hierarchical_gate_to_svg(
            _Decision(
                False,
                {
                    "overall": _LevelDecision(
                        False, [_Eval("demographic_parity_difference", 0.31, 0.10, False)]
                    )
                },
            )
        )

        assert ">FAIL<" in svg
        assert "0/1 metrics" in svg
        assert "BLOCKED" in svg

    def test_a_level_that_was_evaluated_and_passed_is_unchanged(self):
        svg = hierarchical_gate_to_svg(
            _Decision(
                True,
                {
                    "overall": _LevelDecision(
                        True, [_Eval("demographic_parity_difference", 0.04, 0.10, True)]
                    ),
                    "attr:gender": _LevelDecision(
                        True, [_Eval("equalized_odds_difference", 0.06, 0.10, True)]
                    ),
                },
            )
        )

        assert "1/1 metrics" in svg and "1/1 checks" in svg
        assert ">PASS<" in svg
        # The intersectional level genuinely was not run in this decision.
        assert svg.count(">NOT CHECKED<") == 1
        assert "APPROVED" in svg

    def test_an_evaluated_level_is_never_downgraded_to_not_checked(self):
        svg = hierarchical_gate_to_svg(
            _Decision(
                True,
                {
                    "intersection:gender_x_race": _LevelDecision(
                        True, [_Eval("demographic_parity_difference", 0.09, 0.12, True)]
                    )
                },
            )
        )

        assert "1/1 combos" in svg


class TestCorrelationHeatmapUncomputedCells:
    """A correlation that could not be computed is not a correlation of zero."""

    def test_a_nan_cell_is_shown_as_not_available_and_never_as_a_number(self):
        svg = correlation_heatmap_to_svg(
            {
                "income": {"gender": 0.12, "race": float("nan")},
                "zip": {"gender": 0.55, "race": 0.05},
            }
        )

        assert ">N/A<" in svg
        assert ">0.00<" not in svg, "an uncomputed correlation was drawn as a measured 0.00"
        assert "nan" not in svg.lower()

    def test_an_uncomputed_cell_does_not_enter_the_high_correlation_count(self):
        base = correlation_heatmap_to_svg({"income": {"gender": 0.55}}, threshold=0.3)
        with_nan = correlation_heatmap_to_svg(
            {"income": {"gender": 0.55, "race": float("nan")}}, threshold=0.3
        )

        assert "1 potential proxy variable detected" in base
        assert "1 potential proxy variable detected" in with_nan

    def test_the_canvas_says_how_many_pairs_could_not_be_computed(self):
        svg = correlation_heatmap_to_svg(
            {
                "income": {"gender": 0.12, "race": float("nan")},
                "zip": {"gender": 0.55, "race": 0.05},
            }
        )

        assert "1 of 4 pair(s) not computed, so not tested." in svg

    def test_a_missing_pair_is_withheld_rather_than_defaulted_to_zero(self):
        """The dict-of-dicts .get default used to manufacture a 0.0 here."""
        svg = correlation_heatmap_to_svg({"income": {"gender": 0.55}, "zip": {}}, threshold=0.3)

        assert ">N/A<" in svg
        assert ">0.00<" not in svg

    def test_a_grid_where_nothing_computed_is_could_not_check_not_an_all_clear(self):
        svg = correlation_heatmap_to_svg(
            {"income": {"gender": float("nan")}, "zip": {"gender": None}}
        )

        assert "No high correlations detected" not in svg
        assert "NOT CHECKED" in svg
        assert "no pair was computed" in svg

    def test_a_fully_computed_matrix_is_unchanged(self):
        svg = correlation_heatmap_to_svg(
            {"income": {"gender": 0.12, "race": 0.55}, "zip": {"gender": 0.05, "race": 0.71}},
            threshold=0.3,
        )

        assert ">N/A<" not in svg
        assert "not computed" not in svg
        assert ">0.12<" in svg and ">0.55<" in svg and ">0.71<" in svg
        assert "2 potential proxy variables detected" in svg

    def test_a_clean_matrix_still_earns_its_all_clear(self):
        svg = correlation_heatmap_to_svg({"income": {"gender": 0.01}}, threshold=0.3)

        assert "No high correlations detected" in svg
        assert "not computed" not in svg


def _audit_report(modules_run):
    """A hand-built audit report for the coverage-count tests.

    ``attribute_assessed`` added 2026-09-27. This helper predates the field, so a
    report naming every module in ``modules_run`` had an unrecorded ASSESSMENT
    half, and ``execution_coverage()`` answered "complete" over it. It answers
    "unrecorded" now, which is correct. These tests are about a complete audit
    earning its all-clear, so the fixture has to be a complete audit; the
    alternative was weakening what they assert.
    """
    return BiasAuditReport(
        timestamp="2026-01-01T12:00:00",
        dataset_info={"n_rows": 10, "n_columns": 3},
        protected_attributes=["gender"],
        attribute_assessed={"gender": True},
        historical_findings=[],
        representation_findings=[],
        disparity_findings=[],
        proxy_findings=[],
        overall_risk_score=0.0,
        critical_issues=[],
        recommendations=[],
        modules_run=modules_run,
    )


def _audit_data(monkeypatch, modules_run):
    """The dict ``bias_audit_to_svg`` hands the template, plus the rendered SVG.

    The coverage SENTENCE reaches the canvas only through ``rendering.explain``'s
    explanation panel, which is a separate surface under active change, so
    asserting on the rendered string alone would make these tests fail for
    reasons that have nothing to do with the count. The count itself is decided
    in the adapter, so that is where it is read.
    """
    import vfairness.rendering.adapters as adapters_mod

    captured = {}
    real = adapters_mod.render_svg

    def _spy(template_name, data, *args, **kwargs):
        captured.update(data)
        return real(template_name, data, *args, **kwargs)

    monkeypatch.setattr(adapters_mod, "render_svg", _spy)
    svg = bias_audit_to_svg(_audit_report(modules_run))
    return captured, svg


class TestBiasAuditCoverageCount:
    """How much of the audit ran is counted from the four modules, not the input."""

    def test_an_unrecognised_module_name_does_not_inflate_the_count(self, monkeypatch):
        data, _ = _audit_data(monkeypatch, ["historical", "bogus_a", "bogus_b"])

        assert "only 1 of the 4 audit modules executed" in data["not_assessable_reason"]
        assert "only 3 of the 4" not in data["not_assessable_reason"]
        assert data["n_modules_run"] == 1

    def test_the_sentence_and_the_subtitle_report_the_same_number(self, monkeypatch):
        data, svg = _audit_data(monkeypatch, ["historical", "bogus_a", "bogus_b"])

        assert "Only 1 of 4 audit modules ran" in svg
        assert "Only 3 of 4 audit modules ran" not in svg
        assert f"only {data['n_modules_run']} of the 4" in data["not_assessable_reason"]

    def test_the_two_halves_of_the_sentence_can_no_longer_contradict_each_other(self, monkeypatch):
        data, _ = _audit_data(monkeypatch, ["historical", "bogus_a", "bogus_b"])
        reason = data["not_assessable_reason"]

        n_ran = int(re.search(r"only (\d+) of the 4 audit modules executed", reason).group(1))
        named = re.search(r"executed \(([^)]*) did not run\)", reason).group(1)
        n_not = len([m for m in named.split(",") if m.strip()])
        assert n_ran + n_not == 4, f"{n_ran} ran plus {n_not} did not run is not 4"

    def test_a_report_naming_only_unknown_modules_is_credited_with_nothing(self, monkeypatch):
        data, svg = _audit_data(monkeypatch, ["renamed_historical"])

        assert "only 0 of the 4 audit modules executed" in data["not_assessable_reason"]
        assert "Only 0 of 4 audit modules ran" in svg

    def test_a_genuinely_partial_audit_is_unchanged(self, monkeypatch):
        data, svg = _audit_data(monkeypatch, ["historical", "representation"])

        assert "only 2 of the 4 audit modules executed" in data["not_assessable_reason"]
        assert "disparities, proxies did not run" in data["not_assessable_reason"]
        assert "Only 2 of 4 audit modules ran" in svg
        assert data["n_modules_run"] == 2

    def test_a_complete_audit_still_earns_its_measured_all_clear(self, monkeypatch):
        data, svg = _audit_data(
            monkeypatch, ["historical", "representation", "disparities", "proxies"]
        )

        assert data["not_assessable"] is False
        assert data["n_modules_run"] == 4
        assert "audit modules ran" not in svg

    def test_an_extra_name_beside_all_four_does_not_disturb_a_complete_audit(self, monkeypatch):
        data, svg = _audit_data(
            monkeypatch, ["historical", "representation", "disparities", "proxies", "extra"]
        )

        assert data["not_assessable"] is False
        assert data["n_modules_run"] == 4, "a fifth, unknown name must not read as 5 of 4"
        assert "audit modules ran" not in svg


def test_the_row_level_rule_holds_for_every_shape_of_absence():
    """One table, every way a caller can report nothing for a group."""
    for gm in ({}, {"mae": None}, {"mae": float("nan")}, {"size": None}):
        svg = regression_fairness_to_svg({"G": gm}, ALL_PASSING)
        for value in _measured_numbers(svg):
            assert not math.isclose(float(value), 0.0), (
                f"a zero was painted as a measurement for {gm!r}"
            )
