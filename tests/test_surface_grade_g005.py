"""Batch g005: CalibrationAnalyzer / CalibrationReport, a value nobody measured.

Four defects were found by execution against
``src/vfairness/post_processing/calibration/analyzer.py`` and fixed there.

G005-SUM  ``CalibrationReport.summary()`` printed every headline metric through
          ``self.overall_metrics.get('ece', 0)``. An absent metric therefore
          rendered as ``0.0000``, which on the ECE / MCE / Brier scale is the
          BEST value there is. Measured on a report with ``overall_metrics={}``
          the text read three perfect scores beside "Well Calibrated: Not
          assessable (not measured)". ``rendering.adapters`` carries a standing
          "DO NOT reinstate the overall.get('ece', 0) seeds" comment about this
          exact shape on the SVG surface; the text surface, the one the class
          docstring tells readers to print, still had it. The per-group table
          had the same default one level down.

G005-CI   ``_identify_critical_issues`` tests every quantity with ``>``, and
          ``nan > 0.1`` is False, so quantities that produced no value produced
          no entry. Measured on a one-row analyzer (overall ECE NaN, MCE NaN,
          disparity NaN) ``full_analysis().critical_issues`` came back ``[]``,
          and an empty critical-issues list reads as a clean bill of health in
          ``summary()`` (the section is dropped), in ``to_dict()`` and in the
          SVG issues panel.

G005-INS  ``evaluate_calibration_improvement`` scores the calibrator against
          the same rows it was fitted on and said so nowhere. With the default
          ``method='isotonic'`` the after ECE is the fit residual, ~1e-17 on
          every seed and bin count, so the dict reports that calibration
          perfectly eliminated both miscalibration and its between-group
          disparity, on any data at all. The API reference prints ``ECE After``
          straight from it. The numbers are kept (they are measured, and a zero
          residual is informative about the fit); the reading is labelled.

G005-JSON ``to_json`` wrote a bare ``NaN`` token, which is not JSON. Measured
          on a single-group report it carried ``"ece_disparity": NaN`` and five
          more, Python's own ``json.loads`` accepted them as an extension, and
          ``JSON.parse`` in node rejected the WHOLE document with "Unexpected
          token 'N'". The one report that had something important to say, that
          its disparity could not be measured, was the one a browser consumer
          could not read at all, and the cheapest local repair for whoever hit
          it is to put a 0.0 back. Non-finite floats now serialise as JSON
          ``null``, the encoding the repo already uses for this third state.

Every pin below is paired with a healthy-data control, because the reverse
defect, refusing a value that IS measurable, is worse than the original: a
genuine ECE of exactly 0.0 must still render as ``0.0000`` and must serialise
as ``0.0``, never as null.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.post_processing.calibration.analyzer import (
    CalibrationAnalyzer,
    CalibrationReport,
)

# ---------------------------------------------------------------------------
# Fixtures. Both thresholds in the constructor signature matter here:
# min_group_size defaults to 30 and n_bins to 10, and a fixture too small for
# min_group_size supplies a refusal through a parameter nobody was looking at.
# Each group below holds 80 rows, well clear of 30, and n_bins is pinned to 2
# so the expected ECE can be computed by hand rather than copied from the code.
# ---------------------------------------------------------------------------


def _handmade():
    """A fixture whose ECE is arithmetic, not an observation of the code.

    Two uniform bins, edges [0, 0.5, 1].

    Group A: 40 rows at p=0.25 with 10 positives -> accuracy 0.25 == confidence
             0.25, contribution 0. 40 rows at p=0.75 with 30 positives ->
             accuracy 0.75 == confidence 0.75, contribution 0. ECE(A) = 0.0,
             a genuinely perfectly calibrated group.
    Group B: 40 rows at p=0.25 with 0 positives -> 0.5 * 0.25 = 0.125.
             40 rows at p=0.75 with 0 positives -> 0.5 * 0.75 = 0.375.
             ECE(B) = 0.5.
    Disparity = 0.5 - 0.0 = 0.5.
    Overall:  bin 0 holds 80 rows with 10 positives -> 0.5 * |0.125 - 0.25|
              = 0.0625; bin 1 holds 80 rows with 30 positives ->
              0.5 * |0.375 - 0.75| = 0.1875. Overall ECE = 0.25.
    """
    y = np.concatenate(
        [
            np.array([1] * 10 + [0] * 30),  # A, p=0.25
            np.array([1] * 30 + [0] * 10),  # A, p=0.75
            np.zeros(40, dtype=int),  # B, p=0.25
            np.zeros(40, dtype=int),  # B, p=0.75
        ]
    )
    p = np.concatenate([np.full(40, 0.25), np.full(40, 0.75)] * 2)
    g = np.array(["A"] * 80 + ["B"] * 80)
    return y, p, g


EXPECTED_OVERALL_ECE = 0.25
EXPECTED_GROUP_ECE = {"A": 0.0, "B": 0.5}
EXPECTED_DISPARITY = 0.5


@pytest.fixture
def healthy():
    y, p, g = _handmade()
    return CalibrationAnalyzer(y, p, g, attribute_name="grp", n_bins=2)


@pytest.fixture
def unmeasurable():
    """One row. ``compute_ece_single`` refuses below 2 rows, so the overall ECE
    and MCE are NaN, and a single group of 1 clears no disparity comparison."""
    return CalibrationAnalyzer(np.array([1]), np.array([0.7]), np.array(["A"]))


def _real_disparity():
    """A genuine CalibrationDisparityResult, so summary() is exercised on the
    shape it is typed for rather than on a None the field never allows."""
    y, p, g = _handmade()
    return CalibrationAnalyzer(y, p, g, n_bins=2).analyze_disparity()


def _report_with(**over):
    base = dict(
        timestamp="2026-09-17T00:00:00",
        data_info={"n_samples": 160},
        protected_attribute="grp",
        n_groups=2,
        overall_metrics={},
        group_metrics={},
        disparity_analysis=_real_disparity(),
        brier_decomposition=None,
        tradeoff_analysis=None,
        recommendation=None,
        is_well_calibrated=None,
        has_significant_disparity=None,
        critical_issues=[],
        recommendations=[],
    )
    base.update(over)
    return CalibrationReport(**base)


def _line(text, prefix):
    hits = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


# ===========================================================================
# The measurement control. Everything else is a refusal pin, and a refusal pin
# alone is satisfied by a function that refuses everything.
# ===========================================================================


class TestItStillMeasures:
    def test_the_numbers_match_an_independent_computation(self, healthy):
        ece = healthy.evaluate_ece()
        assert ece.overall_value == pytest.approx(EXPECTED_OVERALL_ECE, abs=1e-12)
        assert set(ece.group_values) == {"A", "B"}
        for group, expected in EXPECTED_GROUP_ECE.items():
            assert ece.group_values[group] == pytest.approx(expected, abs=1e-12)
        assert healthy.analyze_disparity().ece_disparity == pytest.approx(
            EXPECTED_DISPARITY, abs=1e-12
        )
        assert healthy.analyze_disparity().most_miscalibrated_group == "B"

    def test_the_summary_renders_those_numbers_including_a_real_zero(self, healthy):
        # A measured 0.0 is a finding: group A is perfectly calibrated here and
        # must print as 0.0000. Refusing it would be the reverse defect.
        text = healthy.full_analysis().summary()
        assert _line(text, "Expected Calibration Error (ECE):").endswith("0.2500")
        assert _line(text, "  A:").endswith("0.0000")
        assert _line(text, "  B:").endswith("0.5000")
        assert "not measured" not in _line(text, "  A:")

    def test_the_healthy_report_carries_measured_verdicts_and_real_issues(self, healthy):
        report = healthy.full_analysis()
        assert report.is_well_calibrated is False
        assert report.has_significant_disparity is True
        types = [i["type"] for i in report.critical_issues]
        assert "Poor Calibration" in types
        assert "Calibration Disparity" in types
        # G005-CI must not fire on data that WAS measured.
        assert "Not Assessed" not in types


# ===========================================================================
# G005-SUM: an absent metric is not a perfect score
# ===========================================================================


class TestSummaryDoesNotDefaultAMetricToZero:
    @pytest.mark.parametrize(
        "prefix",
        [
            "Expected Calibration Error (ECE):",
            "Maximum Calibration Error (MCE):",
            "Brier Score:",
        ],
    )
    def test_absent_metric_is_named_not_printed_as_zero(self, prefix):
        text = _report_with(overall_metrics={}).summary()
        line = _line(text, prefix)
        assert "0.0000" not in line, line
        assert "not measured" in line, line

    @pytest.mark.parametrize(
        "prefix,key",
        [
            ("Expected Calibration Error (ECE):", "ece"),
            ("Maximum Calibration Error (MCE):", "mce"),
            ("Brier Score:", "brier"),
        ],
    )
    def test_a_nan_metric_is_named_not_printed_as_the_token_nan(self, prefix, key):
        text = _report_with(overall_metrics={key: float("nan")}).summary()
        line = _line(text, prefix)
        assert "not measured" in line, line
        assert "nan" not in line.lower().replace("not measured", ""), line

    def test_a_present_key_holding_none_is_not_zero_either(self):
        # .get(key, default) does NOT fire when the key is present holding None.
        line = _line(_report_with(overall_metrics={"ece": None}).summary(), "Expected")
        assert "0.0000" not in line, line
        assert "not measured" in line, line

    def test_the_per_group_table_has_the_same_three_states(self):
        text = _report_with(
            group_metrics={
                "measured": {"ece": 0.0},
                "absent": {},
                "undefined": {"ece": float("nan")},
            }
        ).summary()
        assert _line(text, "  measured:").endswith("0.0000")
        assert "not measured" in _line(text, "  absent:")
        assert "not measured" in _line(text, "  undefined:")

    def test_an_unmeasurable_analyzer_prints_no_number_at_all(self, unmeasurable):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = unmeasurable.full_analysis().summary()
        assert "not measured" in _line(text, "Expected Calibration Error (ECE):")
        assert "not measured" in _line(text, "Maximum Calibration Error (MCE):")
        assert "Well Calibrated: Not assessable (not measured)" in text


# ===========================================================================
# G005-CI: an empty critical-issues list is not a clean bill of health
# ===========================================================================


class TestCriticalIssuesDiscloseWhatWasNeverChecked:
    def test_nothing_measurable_produces_a_named_could_not_check(self, unmeasurable):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = unmeasurable.full_analysis()
        assert not np.isfinite(report.overall_metrics["ece"])
        assert report.critical_issues, "an empty list here reads as 'no issues found'"
        first = report.critical_issues[0]
        assert first["type"] == "Not Assessed"
        assert "COULD NOT CHECK" in first["description"]
        # It must name which quantities went unmeasured, not just that some did.
        for quantity in ("expected calibration error", "disparity"):
            assert quantity in first["description"].lower()
        # The refusal is also carried out of band, so a caller reading only
        # warnings still sees it.
        assert [w for w in caught if "NaN" in str(w.message) or "not 0.0" in str(w.message)]

    def test_the_coverage_note_survives_the_truncated_renderings(self, unmeasurable):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = unmeasurable.full_analysis()
        # summary() shows [:5], the SVG panel [:4]. Naming it first is what
        # keeps it visible when real findings fill the list.
        assert (
            report.critical_issues.index(
                next(i for i in report.critical_issues if i["type"] == "Not Assessed")
            )
            < 4
        )
        assert "CRITICAL ISSUES" in report.summary()
        assert "COULD NOT CHECK" in report.summary()

    def test_a_measured_ece_beside_an_unmeasured_disparity_names_only_the_disparity(self):
        # One adequate group and one too small: the overall ECE IS measured, so
        # the note must not claim otherwise, and the real findings below it
        # must survive. A guard that swallowed them would be the worse bug.
        # Keep all 80 of the badly calibrated group B (ECE 0.5, above the 0.1
        # critical threshold) and only 6 rows of A, which is below
        # min_group_size=30 and so leaves the disparity unmeasurable.
        y, p, g = _handmade()
        keep = np.concatenate([np.arange(80, 160), np.arange(6)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = CalibrationAnalyzer(y[keep], p[keep], g[keep], n_bins=2).full_analysis()
        note = next(i for i in report.critical_issues if i["type"] == "Not Assessed")
        assert "disparity" in note["description"].lower()
        assert "expected calibration error" not in note["description"].lower()
        assert np.isfinite(report.overall_metrics["ece"])
        assert "Poor Calibration" in [i["type"] for i in report.critical_issues]


# ===========================================================================
# G005-INS: the improvement figures are a fit residual
# ===========================================================================


class TestCalibrationImprovementIsLabelledInSample:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("n_bins", [10, 20])
    def test_the_after_figure_is_a_training_residual(self, seed, n_bins):
        """The observation that makes the disclosure load bearing: on continuous
        scores the default isotonic method drives the in-sample after ECE to
        numerical zero for every seed and bin count, so 'improvement' equals
        'before' by construction and the after disparity is a fabricated
        perfect parity."""
        rng = np.random.default_rng(seed)
        n = 400
        g = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        p = rng.uniform(0.02, 0.98, n)
        y = (rng.uniform(size=n) < p * 0.3).astype(int)  # severely overconfident
        analyzer = CalibrationAnalyzer(y, p, g, n_bins=n_bins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer.fit_calibrator(method="isotonic")
            result = analyzer.evaluate_calibration_improvement()
        assert result["before"]["overall_ece"] > 0.2, "the fixture must carry a real defect"
        assert result["before"]["ece_disparity"] > 0.01
        assert abs(result["after"]["overall_ece"]) < 1e-9
        assert abs(result["after"]["ece_disparity"]) < 1e-9
        assert result["improvement"]["overall_ece"] == pytest.approx(
            result["before"]["overall_ece"], abs=1e-9
        )
        assert result["coverage"]["in_sample"] is True

    def test_the_before_figure_is_a_real_measurement(self, healthy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            healthy.fit_calibrator(method="isotonic")
            result = healthy.evaluate_calibration_improvement()
        assert result["before"]["overall_ece"] == pytest.approx(EXPECTED_OVERALL_ECE, abs=1e-12)
        assert result["before"]["ece_disparity"] == pytest.approx(EXPECTED_DISPARITY, abs=1e-12)

    def test_it_says_in_sample_in_the_result(self, healthy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            healthy.fit_calibrator(method="isotonic")
            result = healthy.evaluate_calibration_improvement()
        assert result["coverage"]["in_sample"] is True
        assert "IN SAMPLE" in result["coverage"]["note"]
        assert result["coverage"]["n_rows_scored"] == 160

    def test_it_warns_so_a_caller_reading_only_the_numbers_is_told(self, healthy):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            healthy.fit_calibrator(method="isotonic")
            healthy.evaluate_calibration_improvement()
        named = [str(w.message) for w in caught if "IN SAMPLE" in str(w.message)]
        assert named, [str(w.message) for w in caught]
        assert "held-out" in named[0]

    def test_supplying_calibrated_probabilities_is_still_in_sample(self, healthy):
        # They are scored against this analyzer's own y_true either way.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = healthy.evaluate_calibration_improvement(y_prob_calibrated=np.full(160, 0.25))
        assert result["coverage"]["in_sample"] is True
        assert [w for w in caught if "IN SAMPLE" in str(w.message)]

    def test_control_it_still_refuses_with_no_calibrator_and_no_probabilities(self, healthy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(RuntimeError, match="No calibrator fitted"):
                healthy.evaluate_calibration_improvement()


# ===========================================================================
# The rest of the batch: behaviour confirmed by execution and held here so a
# regression is visible. These are not renderers of a fabricated value.
# ===========================================================================


class TestTheDelegatingMetricsKeepThreeStates:
    """Every entry point below forwards to metrics.py / tradeoffs.py. The
    forwarding is what is pinned here: a healthy value computed independently
    in the docstring of ``_handmade``, and the refusal that must survive the
    forward rather than being flattened into a number on the way out.
    """

    def test_mce_and_brier_match_the_hand_computation(self, healthy):
        # MCE = worst bin gap = max(|0.125-0.25|, |0.375-0.75|) = 0.375.
        # Brier = mean((p-y)^2) = 40.0 / 160 = 0.25.
        assert healthy.evaluate_mce().overall_value == pytest.approx(0.375, abs=1e-12)
        assert healthy.evaluate_brier().overall_value == pytest.approx(0.25, abs=1e-12)
        both = healthy.evaluate_calibration()
        assert set(both) == {"ece", "mce", "brier"}
        assert both["mce"].overall_value == pytest.approx(0.375, abs=1e-12)

    def test_ece_and_mce_refuse_below_two_rows(self, unmeasurable):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ece = unmeasurable.evaluate_ece()
            mce = unmeasurable.evaluate_mce()
        assert not np.isfinite(ece.overall_value)
        assert not np.isfinite(mce.overall_value)
        # The exclusion is disclosed, not left as an absent dict key.
        assert ece.group_values == {}
        assert [w for w in caught if "NOT assessed" in str(w.message)]

    def test_brier_decomposition_refuses_the_skill_score_on_one_class(self):
        y, p, g = _handmade()
        single_class = CalibrationAnalyzer(np.zeros(160, dtype=int), p, g, n_bins=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decomposition = single_class.decompose_brier()
        assert not np.isfinite(decomposition.skill_score)
        assert [w for w in caught if "skill_score is NaN, not 0.0" in str(w.message)]

    def test_brier_decomposition_control_on_two_classes(self, healthy):
        decomposition = healthy.decompose_brier()
        assert np.isfinite(decomposition.skill_score)
        # uncertainty = base_rate * (1 - base_rate), base rate 40/160 = 0.25.
        assert decomposition.uncertainty == pytest.approx(0.1875, abs=1e-12)

    def test_tradeoffs_refuse_when_no_two_groups_were_compared(self, unmeasurable):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tradeoffs = unmeasurable.analyze_tradeoffs()
        assert not np.isfinite(tradeoffs.base_rate_disparity)
        assert tradeoffs.tradeoff_severity == "not assessed"
        assert [w for w in caught if "no base-rate comparison was made" in str(w.message)]

    def test_tradeoffs_control_measure_a_real_base_rate_gap(self, healthy):
        tradeoffs = healthy.analyze_tradeoffs()
        # A has 40 positives of 80, B has none: the gap is exactly 0.5.
        assert tradeoffs.base_rate_disparity == pytest.approx(0.5, abs=1e-12)
        assert tradeoffs.tradeoff_severity != "not assessed"

    def test_impossibility_diagnosis_is_three_state(self, healthy, unmeasurable):
        measured = healthy.get_impossibility_diagnosis()
        assert measured["base_rates_differ"] is True
        assert measured["base_rate_disparity"] == pytest.approx(0.5, abs=1e-12)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused = unmeasurable.get_impossibility_diagnosis()
        # None, never False: False here would be "the theorem does not bind".
        assert refused["base_rates_differ"] is None

    def test_fit_calibrator_refuses_the_unimplemented_strategy(self, healthy):
        with pytest.raises(NotImplementedError, match="borrow"):
            healthy.fit_calibrator(fallback_strategy="borrow")
        assert healthy._group_calibrator is None

    def test_fit_calibrator_says_when_a_group_got_no_usable_calibrator(self, healthy):
        # Group B is single class here, so it cannot be fitted. A calibrator
        # that quietly passed those rows through would report a better parity
        # number for free.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            healthy.fit_calibrator(method="platt")
        assert [w for w in caught if "did NOT get a usable ca" in str(w.message)]

    def test_get_recommendation_is_not_shared_between_contexts(self, healthy):
        general = healthy.get_recommendation(context="general")
        healthcare = healthy.get_recommendation(context="healthcare")
        assert healthy.get_recommendation(context="general") is general
        assert healthy.get_recommendation(context="healthcare") is healthcare

    def test_get_explanation_carries_the_measured_ece_into_its_prose(self, healthy):
        report = healthy.full_analysis()
        explanation = healthy.get_explanation(report)
        ece_cards = [
            e for e in explanation.explanations if e.metric_name.startswith("Expected Calibration")
        ]
        assert len(ece_cards) == 1
        assert ece_cards[0].value == pytest.approx(EXPECTED_OVERALL_ECE, abs=1e-12)
        assert "0.2500" in ece_cards[0].evaluation

    def test_to_svg_renders_and_marks_the_unmeasurable_report_not_assessed(
        self, healthy, unmeasurable
    ):
        measured = healthy.full_analysis().to_svg()
        assert "<svg" in measured
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            refused = unmeasurable.full_analysis().to_svg()
        assert "NOT ASSESSED" in refused
        assert "NOT ASSESSED" not in measured


class TestTheOtherGradedItems:
    def test_transform_before_fit_refuses_rather_than_returning_the_input(self, healthy):
        with pytest.raises(RuntimeError, match="fit_calibrator"):
            healthy.transform(np.full(4, 0.3), np.array(["A"] * 4))
        with pytest.raises(RuntimeError, match="fit_calibrator"):
            healthy.get_calibration_result()

    def test_transform_on_an_unseen_group_warns_that_it_was_not_group_calibrated(self, healthy):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            healthy.fit_calibrator(method="platt")
            out = healthy.transform(np.full(4, 0.3), np.array(["Z"] * 4))
        assert out.shape == (4,)
        assert [w for w in caught if "not seen during fit" in str(w.message)]

    def test_calibrate_actually_changes_the_probabilities(self, healthy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = healthy.calibrate()
        # A mitigation that returns its input unchanged reports a better parity
        # number for free. Group B is all negatives, so a fitted calibrator has
        # to move its scores down.
        assert out.shape == (160,)
        assert np.mean(out[80:]) < np.mean(healthy.y_prob[80:]) - 0.1

    def test_get_calibration_curve_refuses_an_unknown_group_by_name(self, healthy):
        with pytest.raises(KeyError, match="NOPE"):
            healthy.get_calibration_curve(group="NOPE")

    def test_get_data_summary_describes_the_input_it_was_given(self, healthy):
        summary = healthy.get_data_summary()
        assert summary["n_samples"] == 160
        assert summary["group_sizes"] == {"A": 80, "B": 80}
        # Independently: A holds 40 positives of 80, B holds none.
        assert summary["group_base_rates"]["A"] == pytest.approx(0.5)
        assert summary["group_base_rates"]["B"] == pytest.approx(0.0)
        assert summary["base_rate"] == pytest.approx(0.25)

    def test_clear_cache_returns_nothing_and_forces_recomputation(self, healthy):
        first = healthy.evaluate_ece()
        assert healthy.clear_cache() is None
        assert healthy.evaluate_ece() is not first

    def test_to_dict_and_to_json_carry_the_three_state_verdicts(self, unmeasurable):
        import json

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = unmeasurable.full_analysis()
        as_dict = report.to_dict()
        assert as_dict["is_well_calibrated"] is None
        assert as_dict["has_significant_disparity"] is None
        parsed = json.loads(report.to_json())
        assert parsed["is_well_calibrated"] is None
        assert parsed["has_significant_disparity"] is None

    def test_to_json_emits_no_bare_nan_token(self, unmeasurable):
        """G005-JSON. A bare NaN is not JSON: JSON.parse rejects the whole
        document, so the one report whose disparity could not be measured was
        the one a browser could not read at all."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            as_json = unmeasurable.full_analysis().to_json()
        assert "NaN" not in as_json, [ln for ln in as_json.splitlines() if "NaN" in ln][:5]
        assert "Infinity" not in as_json

    def test_to_json_uses_null_for_could_not_check_and_keeps_the_key(self, unmeasurable):
        import json

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = unmeasurable.full_analysis()
        disparity = json.loads(report.to_json())["disparity_analysis"]
        assert "ece_disparity" in disparity, "an absent key is a different thing from null"
        assert disparity["ece_disparity"] is None
        # to_dict() is the in-process surface and must keep the float NaN that
        # its consumers test with np.isfinite.
        assert not np.isfinite(report.to_dict()["disparity_analysis"]["ece_disparity"])

    def test_control_to_json_keeps_every_measured_number_including_a_real_zero(self, healthy):
        import json

        parsed = json.loads(healthy.full_analysis().to_json())
        assert parsed["overall_metrics"]["ece"] == pytest.approx(EXPECTED_OVERALL_ECE, abs=1e-12)
        assert parsed["disparity_analysis"]["ece_disparity"] == pytest.approx(
            EXPECTED_DISPARITY, abs=1e-12
        )
        # A measured 0.0 must stay 0.0 and must never be flattened to null.
        assert parsed["group_metrics"]["A"]["ece"] == pytest.approx(0.0, abs=1e-12)
        assert parsed["group_metrics"]["A"]["ece"] is not None
