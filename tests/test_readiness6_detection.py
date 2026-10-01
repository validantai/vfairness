"""R-11, 2026-09-10. Seven could-not-checks that were reported as clean readings.

Every defect in this file has the same shape and it is the shape this repo keeps
finding: a measurement that did not happen, replaced by the value that reads as
its calm answer, and then published with nothing to say it never ran. ``0.0``
for a risk score, ``"complete"`` for a coverage record, ``False`` for a drift
verdict, ``"stable"`` for a classification, ``jira`` for a routing decision, and
``MaxSkew 0.0``, the best attainable score on its scale, for an image set
with no demographic labels in it at all.

Each block below carries BOTH halves, and the second half is the one that costs
something to get wrong:

  * a REFUSAL test, which fails if the could-not-check is ever published as a
    finding again;
  * an OVER-CORRECTION CONTROL, which fails if the fix started refusing things
    it should still measure. A clean audit must still read clean, a flat series
    must still read stable, a real baseline must still catch real drift, and a
    fully labelled image set must still score its real skew.

Numpy note: every label array here is ``dtype=object``. A ``'<U5'`` array
silently TRUNCATES longer strings, which voided fixtures earlier in this audit
("black_female" became "black").
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.alerts import (
    AdaptiveThresholdManager,
    FairnessAlertPrioritizer,
)
from vfairness.operations.monitoring.drift import FairnessDriftDetector
from vfairness.operations.monitoring.sequential import (
    cusum_drift,
    page_hinkley,
    sequential_fairness_drift,
)
from vfairness.operations.pulse.vision_probe import vision_probe_pulse
from vfairness.preprocessing.bias_detection.detector import (
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
    COVERAGE_UNASSESSED,
    BiasDetector,
)
from vfairness.preprocessing.bias_detection.representation import _analyze_intersectional

N_ROWS = 500


def _frame(gender_values):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "gender": pd.Series(gender_values, dtype=object),
            "income": rng.normal(50_000, 10_000, N_ROWS),
            "approved": rng.integers(0, 2, N_ROWS),
        }
    )


def _audit(df, attributes):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = BiasDetector(
            df, protected_attributes=attributes, outcome_column="approved"
        ).full_audit()
    return report, [str(w.message) for w in caught]


# ---------------------------------------------------------------- defect 1
# A bias audit over an all-null protected column reported complete coverage,
# 0.0 risk and "no critical bias issues detected".


class TestAnAuditThatAssessedNothing:
    """MEASURED before the fix, 500 rows with gender entirely NULL::

        execution_coverage : complete
        overall_risk_score : 0.0
        recommendations    : ['No critical bias issues detected. Continue
                              monitoring and perform periodic audits.']

    while three sub-modules each warned UNASSESSED and every one of those
    statements died in ``warnings``.
    """

    def test_an_all_null_protected_column_is_not_complete_coverage(self):
        report, _ = _audit(_frame([None] * N_ROWS), ["gender"])
        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert report.assessment_coverage() == COVERAGE_NONE
        assert report.attribute_observations == {"gender": 0}
        assert report.unassessable_attributes() == ["gender"]

    def test_the_modules_did_run_and_the_record_still_says_so(self):
        """The invocation record is not falsified to make the coverage honest."""
        report, _ = _audit(_frame([None] * N_ROWS), ["gender"])
        assert set(report.modules_run) == {
            "historical",
            "representation",
            "disparities",
            "proxies",
        }

    def test_the_zero_risk_score_is_no_longer_produced_in_silence(self):
        """The ~609 guard was DISARMED by an INSUFFICIENT_DATA representation
        result being appended as a measured risk of 0.0."""
        _, messages = _audit(_frame([None] * N_ROWS), ["gender"])
        assert any("NOT a measurement of low risk" in m for m in messages)

    def test_the_recommendation_is_not_an_all_clear(self):
        report, _ = _audit(_frame([None] * N_ROWS), ["gender"])
        joined = " ".join(report.recommendations)
        assert "No critical bias issues detected" not in joined
        assert "none of them assessed anything" in joined
        assert "clears the data" in joined

    def test_the_text_summary_says_it(self):
        report, _ = _audit(_frame([None] * N_ROWS), ["gender"])
        assert "none of them assessed anything" in report.summary()

    def test_the_serialised_report_carries_both_records(self):
        report, _ = _audit(_frame([None] * N_ROWS), ["gender"])
        exported = report.to_dict()
        assert exported["execution_coverage"] == COVERAGE_UNASSESSED
        assert exported["assessment_coverage"] == COVERAGE_NONE
        assert exported["attribute_observations"] == {"gender": 0}

    def test_a_misspelled_attribute_name_lands_in_the_same_place(self):
        """It is filtered out of protected_attributes and used to vanish."""
        report, _ = _audit(_frame([None] * N_ROWS), ["gendre"])
        assert report.execution_coverage() == COVERAGE_UNASSESSED
        assert report.attribute_observations == {"gendre": 0}

    # ---- over-correction controls ----

    def test_a_genuinely_clean_populated_audit_still_reads_complete(self):
        report, messages = _audit(_frame(["male", "female"] * (N_ROWS // 2)), ["gender"])
        assert report.execution_coverage() == COVERAGE_COMPLETE
        assert report.assessment_coverage() == COVERAGE_COMPLETE
        assert report.unassessable_attributes() == []
        assert not any("NOT a measurement of low risk" in m for m in messages)

    def test_a_genuinely_clean_audit_still_gets_the_all_clear_wording(self):
        report, _ = _audit(_frame(["male", "female"] * (N_ROWS // 2)), ["gender"])
        assert any("No critical bias issues detected" in r for r in report.recommendations)

    def test_one_populated_and_one_empty_attribute_is_partial_not_none(self):
        df = _frame(["male", "female"] * (N_ROWS // 2))
        df["region"] = pd.Series([None] * N_ROWS, dtype=object)
        report, _ = _audit(df, ["gender", "region"])
        # Every module ran and had SOMETHING, so execution is complete; the
        # attribute that had nothing is named rather than swept in either way.
        assert report.execution_coverage() == COVERAGE_COMPLETE
        assert report.assessment_coverage() == COVERAGE_PARTIAL
        assert report.unassessable_attributes() == ["region"]
        assert any("region" in r and "clears them of nothing" in r for r in report.recommendations)

    def test_a_hand_built_report_still_says_unrecorded_rather_than_guessing(self):
        """attribute_observations defaults to None, which is unknown."""
        from vfairness.preprocessing.bias_detection.detector import (
            AUDIT_MODULES,
            COVERAGE_UNRECORDED,
            BiasAuditReport,
        )

        report = BiasAuditReport(
            timestamp="2026-09-10T00:00:00",
            dataset_info={},
            protected_attributes=["gender"],
            historical_findings=[],
            representation_findings=[],
            disparity_findings=[],
            proxy_findings=[],
            overall_risk_score=0.0,
            critical_issues=[],
            recommendations=[],
            modules_run=list(AUDIT_MODULES),
        )
        assert report.attribute_observations is None
        assert report.assessment_coverage() == COVERAGE_UNRECORDED
        assert report.unassessable_attributes() is None
        # This read COVERAGE_COMPLETE until 2026-09-27, under the comment "an
        # unrecorded second half must not downgrade the historical answer". That
        # reasoning is superseded and is kept here rather than deleted, because it
        # is the reasoning somebody will reach for again.
        #
        # Why it does not hold: execution_coverage's own docstring defines
        # "complete" as every module having run AND at least one protected
        # attribute having been assessed. Answering "complete" for a report with
        # no assessment half is not leaving that half alone, it is asserting
        # something about it. And the consequence was not abstract: the renderer
        # keys its all-clear on that word, so the canvas published "OVERALL RISK
        # 0% MINIMAL" over an audit of an entirely NULL column.
        assert report.execution_coverage() == COVERAGE_UNRECORDED


# ---------------------------------------------------------------- defect 2
# In an INTERSECTIONAL analysis, smallness IS the signal, and min_group_size
# was dropping exactly the groups that carry it.


def _intersectional_frame():
    """white_male 332, black_male 332, white_female 331, black_female 5."""
    race = ["white"] * 663 + ["black"] * 337
    sex = ["male"] * 332 + ["female"] * 331 + ["male"] * 332 + ["female"] * 5
    return pd.DataFrame({"race": np.array(race, dtype=object), "sex": np.array(sex, dtype=object)})


class TestIntersectionalSmallnessIsTheSignal:
    """MEASURED before the fix, on the counts above::

        min_group_size=30 (DEFAULT) -> [{'intersection': 'white_male',
                                         'ratio': 0.754, 'severity': 'high'}]
        min_group_size=5            -> [{'intersection': 'black_female',
                                         'ratio': 0.044, 'severity': 'critical'}, ...]
        warnings: []

    The default returned the MAJORITY group as the only finding and said
    nothing had been skipped.
    """

    def _run(self, min_group_size=30):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            findings = _analyze_intersectional(
                _intersectional_frame(),
                ["race", "sex"],
                min_group_size=min_group_size,
                underrepresentation_threshold=0.8,
            )
        return findings, [str(w.message) for w in caught]

    def test_the_worst_represented_intersection_is_reported_at_the_default(self):
        findings, _ = self._run()
        names = [f["intersection"] for f in findings]
        assert "black_female" in names
        worst = next(f for f in findings if f["intersection"] == "black_female")
        assert worst["severity"] == "critical"
        assert worst["ratio"] == pytest.approx(0.044, abs=0.005)
        assert worst["count"] == 5

    def test_it_sorts_ahead_of_the_majority_group(self):
        findings, _ = self._run()
        assert findings[0]["intersection"] == "black_female"

    def test_the_share_comparison_is_reported_and_the_inference_is_withheld(self):
        findings, _ = self._run()
        small = next(f for f in findings if f["intersection"] == "black_female")
        big = next(f for f in findings if f["intersection"] == "white_male")
        # Deterministic arithmetic on counts: reported for both.
        assert isinstance(small["ratio"], float)
        # The claim that the share generalises: withheld for the small one only.
        assert small["sampling_support"] == "not_assessed"
        assert big["sampling_support"] == "adequate"
        assert small["min_group_size"] == 30

    def test_the_exclusion_is_disclosed(self):
        _, messages = self._run()
        assert any("only 3 of 4 intersectional group(s)" in m for m in messages)
        assert any("not_assessed" in m for m in messages)

    # ---- over-correction controls ----

    def test_an_independent_balanced_set_still_reports_nothing(self):
        rng = np.random.default_rng(7)
        balanced = pd.DataFrame(
            {
                "race": rng.choice(np.array(["white", "black"], dtype=object), 1000),
                "sex": rng.choice(np.array(["male", "female"], dtype=object), 1000),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            findings = _analyze_intersectional(
                balanced, ["race", "sex"], min_group_size=30, underrepresentation_threshold=0.8
            )
        assert findings == []
        assert [str(w.message) for w in caught] == []

    def test_a_fully_sized_set_claims_adequate_support_everywhere(self):
        findings, messages = self._run(min_group_size=5)
        assert all(f["sampling_support"] == "adequate" for f in findings)
        assert messages == []


# ---------------------------------------------------------------- defect 3
# `0 < split_index < len(values)` fell through to a midpoint split, so an empty
# baseline made check_drift compare the current window against ITSELF.


def _drift_series():
    rng = np.random.default_rng(1)
    index = pd.date_range("2025-01-01", periods=120, freq="D")
    return pd.Series(
        np.concatenate([rng.normal(0.10, 0.02, 60), rng.normal(0.45, 0.02, 60)]), index=index
    )


class TestDriftAgainstAnEmptyWindow:
    """MEASURED before the fix::

        real baseline  (n=60): drift_detected=True  overall=0.9950
        empty baseline (n=0) : drift_detected=False overall=0.0720  warnings=[]
        worst_scale ref_mean=0.4478 cur_mean=0.4503

    against the current window's own mean of 0.4489: the reference and current
    "distributions" were the two halves of one window. The mirror case, an
    all-NaN CURRENT window, gave mmd_score 0.0.
    """

    def _check(self, baseline, current):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detector = FairnessDriftDetector()
            detector.set_baseline(baseline)
            result = detector.check_drift(current, metric="dpd")
        return result, [str(w.message) for w in caught]

    def test_an_empty_baseline_is_could_not_check_not_stability(self):
        series = _drift_series()
        empty = pd.Series([np.nan] * 60, index=series.index[:60])
        result, messages = self._check(empty, series.iloc[60:])
        assert result.drift_detected is None
        assert np.isnan(result.overall_drift_score)
        assert result.mmd_score is None
        assert any("holds no usable reading" in m for m in messages)

    def test_set_baseline_says_so_where_the_caller_can_still_see_the_series(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FairnessDriftDetector().set_baseline(pd.Series([np.nan] * 60))
        assert any("stored baseline is EMPTY" in str(w.message) for w in caught)

    def test_an_all_nan_current_window_is_not_a_perfect_mmd_match(self):
        series = _drift_series()
        current = pd.Series([np.nan] * 60, index=series.index[60:])
        result, messages = self._check(series.iloc[:60], current)
        assert result.mmd_score is None, "mmd 0.0 is the strongest identical-distribution claim"
        assert result.drift_detected is None
        assert np.isnan(result.overall_drift_score)
        assert any("current window holds no usable reading" in m for m in messages)

    def test_a_boundary_that_empties_an_arm_is_refused_at_the_scale_level(self):
        """Reached directly, without going through check_drift."""
        detector = FairnessDriftDetector()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = detector.detect_drift_ks(_drift_series(), metric="dpd", split_index=0)
        assert np.isnan(result.drift_score)
        assert result.reference_n == 0 and result.current_n == 0
        assert any("leaves the reference window EMPTY" in str(w.message) for w in caught)

    # ---- over-correction controls ----

    def test_a_real_baseline_still_detects_real_drift(self):
        series = _drift_series()
        result, messages = self._check(series.iloc[:60], series.iloc[60:])
        assert result.drift_detected is True
        assert result.overall_drift_score > 0.9
        assert result.mmd_score is not None and result.mmd_score > 0.0
        assert messages == []

    def test_a_real_baseline_against_a_stable_window_still_reads_stable(self):
        rng = np.random.default_rng(11)
        index = pd.date_range("2025-01-01", periods=120, freq="D")
        flat = pd.Series(rng.normal(0.10, 0.02, 120), index=index)
        result, _ = self._check(flat.iloc[:60], flat.iloc[60:])
        assert result.drift_detected is False
        assert not np.isnan(result.overall_drift_score)

    def test_the_documented_midpoint_default_still_works(self):
        """split_index=None is not a caller-supplied boundary and is untouched."""
        detector = FairnessDriftDetector()
        result = detector.detect_drift_ks(_drift_series(), metric="dpd")
        assert result.reference_n == 60 and result.current_n == 60
        assert not np.isnan(result.drift_score)


# ---------------------------------------------------------------- defect 4
# `sigma = np.std(x)` is NaN when any value is NaN, and every downstream test
# then answered the calm way by accident.


def _stepped_series():
    """100 windows with a 17-sigma step at t=50."""
    rng = np.random.default_rng(3)
    return np.concatenate([rng.normal(0.10, 0.01, 50), rng.normal(0.27, 0.01, 50)])


class TestUncomputableWindowsDoNotEraseAStep:
    """MEASURED before the fix::

        A) 100 windows containing a 17-sigma step        -> abrupt_drift, 24.61
        B) the SAME 100 windows plus 10 uncomputable     -> stable, 0.00, warnings []
        C) an empty series                               -> stable
        D) a genuinely flat measured series              -> stable

    B was indistinguishable from D.
    """

    def _run(self, x):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = sequential_fairness_drift(x)
        return result, [str(w.message) for w in caught]

    def test_ten_uncomputable_windows_do_not_erase_a_seventeen_sigma_step(self):
        clean, _ = self._run(_stepped_series())
        poisoned, messages = self._run(np.concatenate([_stepped_series(), np.full(10, np.nan)]))
        assert clean["classification"] == "abrupt_drift"
        assert poisoned["classification"] == "abrupt_drift"
        assert poisoned["cusum"]["max_cusum"] == pytest.approx(clean["cusum"]["max_cusum"])
        assert any("10 of 110 window(s)" in m for m in messages)

    def test_the_per_step_arrays_stay_aligned_with_the_callers_series(self):
        """drift_index must index the series that was passed in, not a filtered one."""
        x = np.concatenate([_stepped_series(), np.full(10, np.nan)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cusum_drift(x)
        assert len(result["cusum_pos"]) == len(x)
        assert len(result["cusum_neg"]) == len(x)
        assert all(np.isnan(v) for v in result["cusum_pos"][-10:])
        assert result["drift_index"] is not None
        assert np.isfinite(x[result["drift_index"]])

    def test_an_empty_series_is_not_stable(self):
        result, messages = self._run(np.array([]))
        assert result["classification"] == "not_assessed"
        assert result["cusum"]["has_drift"] is None
        assert result["page_hinkley"]["has_drift"] is None
        assert np.isnan(result["cusum"]["max_cusum"])
        assert any("only 0 of 0" in m for m in messages)

    def test_an_all_nan_series_is_not_stable(self):
        result, _ = self._run(np.full(20, np.nan))
        assert result["classification"] == "not_assessed"
        assert result["n_measured"] == 0 and result["n_windows"] == 20

    def test_max_cusum_is_nan_not_zero_when_nothing_was_charted(self):
        """0.0 is the calmest reading the scale has."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.isnan(cusum_drift(np.array([]))["max_cusum"])
            assert np.isnan(page_hinkley(np.array([]))["min_ph"])

    # ---- over-correction controls ----

    def test_a_genuinely_flat_measured_series_is_still_stable(self):
        result, messages = self._run(np.full(100, 0.10))
        assert result["classification"] == "stable"
        assert result["cusum"]["has_drift"] is False
        assert result["page_hinkley"]["has_drift"] is False
        assert result["cusum"]["max_cusum"] == 0.0
        assert messages == []

    def test_the_flat_page_hinkley_contract_is_unchanged(self):
        """Pinned by tests/test_audit_wave6_drift.py; restated here because the
        NaN handling runs through the same branch."""
        result = page_hinkley(np.full(50, 0.25))
        assert result["has_drift"] is False
        assert result["ph_statistic"] == [0.0] * 50

    def test_a_clean_step_is_still_abrupt_drift(self):
        result, messages = self._run(_stepped_series())
        assert result["classification"] == "abrupt_drift"
        assert messages == []

    def test_stationary_noise_is_still_mostly_stable(self):
        rng = np.random.default_rng(5)
        stable = sum(
            sequential_fairness_drift(rng.normal(0.1, 0.02, 100))["classification"] == "stable"
            for _ in range(50)
        )
        assert stable >= 40


# ---------------------------------------------------------------- defect 5
# The H-12 fix changed the LABEL and not the ROUTING.


def _alert_event(**overrides):
    event = {
        "regulatory_risk": 1.0,
        "population_impact": 0.8,
        "drift_velocity": 0.9,
        "historical_discrimination": 1.0,
        "metric_name": "equalized_odds",
        "affected_groups": ["Black", "Female"],
        "mean_shift": 0.07,
        "drift_score": 0.95,
    }
    event.update(overrides)
    return event


class TestAnUnscorableAlertIsNotATicket:
    """MEASURED before the fix, on one event whose only difference was an
    unmeasurable drift score::

        drift_score=0.95 -> score 11.30, CRITICAL, {'channel': 'pagerduty'}
        drift_score=NaN  -> score nan,   UNSCORED, {'channel': 'jira'}

    which is verbatim the downgrade H-12's own docstring says it closed.
    """

    def test_unscored_has_its_own_route(self):
        routing = FairnessAlertPrioritizer().route_alert("UNSCORED")
        assert routing != FairnessAlertPrioritizer._DEFAULT_ROUTING["LOW"]
        assert routing["channel"] != "jira"
        assert "COULD NOT" in routing["triage"]

    def test_an_unscorable_alert_is_not_routed_where_a_low_one_goes(self):
        prioritizer = FairnessAlertPrioritizer()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unscorable = prioritizer.create_alert(_alert_event(drift_score=float("nan")))
        assert unscorable.severity == "UNSCORED"
        assert unscorable.routing["channel"] != "jira"
        assert "COULD NOT be computed" in unscorable.message

    def test_an_unknown_severity_does_not_fall_back_to_low(self):
        prioritizer = FairnessAlertPrioritizer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            routing = prioritizer.route_alert("SOMETHING_ELSE")
        assert routing["channel"] != "jira"
        assert any("NOT to the LOW one" in str(w.message) for w in caught)

    def test_a_custom_routing_map_without_the_entry_refuses_rather_than_guesses(self):
        prioritizer = FairnessAlertPrioritizer(
            routing_map={
                "CRITICAL": {"channel": "pagerduty"},
                "HIGH": {"channel": "slack"},
                "LOW": {"channel": "jira"},
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            routing = prioritizer.route_alert("UNSCORED")
        assert routing["channel"] == "unrouted"
        assert routing["channel"] != "jira"
        assert any("has NO route" in str(w.message) for w in caught)

    # ---- over-correction controls ----

    def test_a_measured_critical_alert_still_pages_someone(self):
        prioritizer = FairnessAlertPrioritizer()
        alert = prioritizer.create_alert(_alert_event())
        assert alert.severity == "CRITICAL"
        assert alert.routing == {"channel": "pagerduty", "team": "@on-call-ml-eng"}

    def test_a_measured_low_alert_still_gets_a_ticket(self):
        prioritizer = FairnessAlertPrioritizer()
        alert = prioritizer.create_alert(
            _alert_event(
                regulatory_risk=0.0,
                population_impact=0.05,
                drift_velocity=0.05,
                historical_discrimination=0.0,
                drift_score=0.01,
            )
        )
        assert alert.severity == "LOW"
        assert alert.routing == {"channel": "jira", "project": "FAIR"}


# ---------------------------------------------------------------- defect 6
# `not e["valid"]` counted an UNREVIEWED alert as a confirmed false positive.


class TestAnUntriagedAlertIsNotAFalsePositive:
    """MEASURED before the fix: 30 untriaged alerts pushed the threshold from
    0.70 to 0.9435, and a real 0.75 drift stopped firing."""

    def _feed(self, verdicts):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager = AdaptiveThresholdManager(initial_threshold=0.70)
            for verdict in verdicts:
                manager.update_from_feedback("dpd_gender", alert_was_valid=verdict)
        return manager, [str(w.message) for w in caught]

    def test_a_backlog_does_not_move_the_threshold(self):
        manager, messages = self._feed([None] * 30)
        assert manager.get_threshold("dpd_gender") == 0.70
        assert any("is not a" in m and "false positive" in m for m in messages)

    def test_a_real_drift_still_fires_after_a_backlog(self):
        manager, _ = self._feed([None] * 30)
        assert manager.is_alert_warranted("dpd_gender", 0.75) is True

    def test_the_backlog_is_visible_rather_than_hidden(self):
        manager, _ = self._feed([None] * 30)
        stats = manager.get_feedback_stats("dpd_gender")
        assert stats["n_alerts"] == 30
        assert stats["n_untriaged"] == 30
        assert stats["n_false_positive"] == 0
        assert stats["false_positive_rate"] is None, "0.0 reads as a perfect record"

    def test_untriaged_entries_are_excluded_from_both_sides_of_the_rate(self):
        manager, _ = self._feed([None] * 20 + [False] * 10)
        stats = manager.get_feedback_stats("dpd_gender")
        assert stats["n_untriaged"] == 20
        assert stats["n_false_positive"] == 10
        assert stats["false_positive_rate"] == 1.0

    # ---- over-correction controls ----

    def test_confirmed_false_positives_still_raise_the_threshold(self):
        manager, _ = self._feed([False] * 30)
        assert manager.get_threshold("dpd_gender") > 0.70
        assert manager.get_feedback_stats("dpd_gender")["false_positive_rate"] == 1.0

    def test_confirmed_valid_alerts_still_lower_the_threshold(self):
        manager, _ = self._feed([True] * 30)
        assert manager.get_threshold("dpd_gender") < 0.70
        assert manager.get_feedback_stats("dpd_gender")["false_positive_rate"] == 0.0


# ---------------------------------------------------------------- defect 7
# `df[demo_col].astype(str)` turned a missing demographic into the literal
# string "nan", which skew() then read as one legitimate group.


def _image_frame(labels):
    return pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(len(labels))],
            "detected_race": pd.Series(labels, dtype=object),
        }
    )


def _vision(labels):
    out = vision_probe_pulse(_image_frame(labels), {}, "detected_race", "generic", "EU")
    return out["data"]["vision"], out["data"]


class TestAnImageSetNobodyCouldLabel:
    """MEASURED before the fix, 100 images the classifier labelled NONE::

        vision.skew: {"available": true, "perGroup": {"nan": 0.0},
                      "maxSkew": 0.0, "mostOverrepresented": "nan"}
        vision.ndkl: {'available': True, 'value': 0.0}
        vision.summary: 'Within representation tolerance on the perceived labels...'

    MaxSkew 0.0 and NDKL 0.0 are the BEST attainable score on both scales.
    """

    @pytest.mark.parametrize(
        "labels",
        [
            pytest.param([None] * 100, id="python-none"),
            pytest.param([np.nan] * 100, id="numpy-nan"),
            pytest.param(["nan"] * 100, id="literal-string-nan"),
            pytest.param(["None"] * 100, id="literal-string-None"),
            pytest.param([""] * 100, id="empty-string"),
        ],
    )
    def test_no_label_is_never_a_perfect_score(self, labels):
        vision, _ = _vision(labels)
        assert vision["available"] is False
        assert vision["skew"].get("maxSkew") is None
        assert vision["ndkl"]["available"] is False
        assert vision["labelCoverage"] == {
            "images": 100,
            "labelled": 0,
            "unlabelled": 100,
            "complete": False,
        }

    def test_the_summary_is_not_a_tolerance_statement(self):
        vision, _ = _vision([None] * 100)
        assert "Within representation tolerance" not in vision["summary"]
        assert vision["summary"].startswith("NOT ASSESSED")

    def test_no_representation_finding_is_manufactured_from_the_fake_group(self):
        _, data = _vision([None] * 100)
        assert [f for f in data["bias"] if f["type"] == "image_representation_skew"] == []

    def test_partly_labelled_measures_what_it_has_and_says_what_it_dropped(self):
        labels = ["white"] * 45 + ["black"] * 5 + [None] * 50
        vision, data = _vision(labels)
        assert vision["available"] is True
        assert vision["labelCoverage"]["labelled"] == 50
        assert vision["labelCoverage"]["unlabelled"] == 50
        assert "nan" not in vision["skew"]["perGroup"]
        assert set(vision["skew"]["perGroup"]) == {"white", "black"}
        assert "50 of 100 image(s) carry no demographic label" in vision["summary"]
        assert any("carrying no demographic label" in item for item in data["scope"]["notCovered"])

    # ---- over-correction controls ----

    def test_a_fully_labelled_skewed_set_still_scores_its_real_skew(self):
        labels = ["white"] * 90 + ["black"] * 5 + ["asian"] * 5
        vision, data = _vision(labels)
        assert vision["available"] is True
        assert vision["labelCoverage"]["complete"] is True
        assert vision["skew"]["maxSkew"] > 0.9
        assert vision["skew"]["mostOverrepresented"] == "white"
        assert vision["ndkl"]["available"] is True and vision["ndkl"]["value"] > 1.0
        assert [f for f in data["bias"] if f["type"] == "image_representation_skew"]

    def test_a_fully_labelled_balanced_set_still_reads_within_tolerance(self):
        vision, data = _vision(["white", "black", "asian", "latino"] * 25)
        assert vision["available"] is True
        assert vision["skew"]["maxSkew"] == 0.0
        assert "Within representation tolerance" in vision["summary"]
        assert data["bias"] == []
