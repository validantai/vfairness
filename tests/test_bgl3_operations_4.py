"""BGL3 batch operations-4: does an operations unit refuse when nothing can be measured?

Eight units across monitoring, the CI/CD gates, and the reporting store, each
executed on an input where the quantity it reports genuinely does not exist, and
on a healthy input to show it is not refusing everything.

Every docstring below records what the unit ACTUALLY returned before the fix in
this wave, measured on 2026-09-27, not what it looked like it would return.
"""

import math
import warnings
from datetime import datetime, timedelta
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.cicd.quality_report import build_quality_report
from vfairness.operations.cicd.task_handlers import handle_data_validation
from vfairness.operations.cicd.validator import (
    VALIDATION_CHECKS,
    DataBiasValidator,
    DataValidationResult,
)
from vfairness.operations.monitoring.alerts import FairnessAlertPrioritizer
from vfairness.operations.monitoring.drift import FairnessDriftDetector
from vfairness.operations.monitoring.tracker import WindowMetrics
from vfairness.operations.reporting.store import MetricsStore, MetricsStoreConfig


def _catch(fn, *args, **kwargs) -> Tuple[Any, List[str]]:
    """Run *fn* and return its result plus every warning message it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


# Fixtures


def _two_group_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "gender": ["M" if i % 2 else "F" for i in range(n)],
            "score": rng.normal(size=n).round(6),
            "approved": rng.integers(0, 2, n),
        }
    )


def _gap_frame(n: int = 400) -> pd.DataFrame:
    """400 unique rows in which M is approved about 80 percent of the time and
    F about 30 percent: a real raw outcome gap of roughly 2.6x."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "applicant_id": [f"a{i}" for i in range(n)],
            "gender": ["M" if i % 2 else "F" for i in range(n)],
            "score": rng.normal(size=n).round(6),
            "approved": [int(rng.random() < (0.8 if i % 2 else 0.3)) for i in range(n)],
        }
    )


# Unit 1 and 2: the representation check over an attribute with no groups
#
# vfairness.operations.cicd.validator.DataBiasValidator.validate
# vfairness.operations.cicd.quality_report.build_quality_report


class TestRepresentationOverNoGroupsAtAll:
    def test_an_attribute_no_row_carries_a_value_is_not_a_representation_pass(self):
        """Measured before the fix, 300 rows whose only protected attribute was
        blank in every row:

            passed=True, coverage='partial', issues=[('constant_columns',
            'warning')], metrics['representation'] == {'gender': {'counts': {},
            'fractions': {}}}

        value_counts was empty, so both threshold tests iterated over nothing and
        the check returned no issues while 'representation' was recorded as run.
        """
        frame = _two_group_frame()
        frame["gender"] = np.nan

        result = DataBiasValidator(protected_attributes=["gender"]).validate(frame)

        types = {i.issue_type for i in result.issues}
        assert "representation_not_measurable" in types, types
        assert not result.passed, "a CI gate passed a dimension nobody measured"
        assert result.metrics["representation"]["gender"]["representation_measured"] is False
        issue = next(i for i in result.issues if i.issue_type == "representation_not_measurable")
        assert "could-not-check" in issue.message
        assert issue.details["n_groups"] == 0
        assert issue.details["n_rows_without_value"] == 300

    def test_the_claim_it_used_to_make_is_demonstrably_false(self):
        """The evidence half: the same frame with real values in that column has
        groups to size, so "every group is large enough" is a statement that CAN
        be true or false. Over zero groups it is neither."""
        frame = _two_group_frame()
        frame.loc[frame.index[:20], "gender"] = "X"  # 20 rows, under min 30

        result = DataBiasValidator(protected_attributes=["gender"]).validate(frame)

        types = {i.issue_type for i in result.issues}
        assert "insufficient_group_samples" in types, types
        assert "representation_not_measurable" not in types

    def test_control_two_real_groups_still_measure_and_still_pass(self):
        """OVER-CORRECTION CONTROL. 300 rows, two groups of 150."""
        result = DataBiasValidator(protected_attributes=["gender"]).validate(
            _two_group_frame(), outcome_column="approved"
        )
        assert result.passed, [i.issue_type for i in result.issues]
        assert result.metrics["representation"]["gender"]["representation_measured"] is True
        assert result.execution_coverage() == "complete"

    def test_control_one_group_is_still_measurable(self):
        """OVER-CORRECTION CONTROL. One group is not a disparity, but its count
        and its share of the frame ARE real numbers, so the representation check
        must still answer. The refusal is scoped to zero groups."""
        frame = _two_group_frame()
        frame["gender"] = "M"

        result = DataBiasValidator(protected_attributes=["gender"]).validate(frame)

        types = {i.issue_type for i in result.issues}
        assert "representation_not_measurable" not in types, types
        assert result.metrics["representation"]["gender"]["counts"] == {"M": 300}


class TestTheQualityReportNeverPassesAnUnmeasuredCheck:
    def test_no_green_representation_tile_when_no_group_exists(self):
        """Measured before this wave, on the same all-blank frame:

            tone 'warn', headline "The data is usable, but 1 thing(s) widen the
            uncertainty", and the tile
              [pass] Group representation: Every group of "gender" is large
                     enough for a reliable read.

        rendered over zero groups, with no warning of any kind.

        The assertion is deliberately about the tile a reader sees and not about
        which layer refuses: both were measured on 2026-09-27. The validator now
        answers `representation_not_measurable` (see the class above), and
        ``prepare_protected_attributes`` separately learned to exclude a column
        with no values at all, which routes this frame to the could-not-run
        tiles. Either way the green claim must be gone.
        """
        frame = _two_group_frame()
        frame["gender"] = np.nan

        report = build_quality_report(frame, ["gender"])

        rep = [c for c in report["checks"] if c["label"] == "Group representation"]
        assert len(rep) == 1, rep
        assert rep[0]["status"] != "pass", rep[0]
        assert report["tone"] != "pass", report["tone"]
        assert not any(
            c["status"] == "pass" and "large enough" in c["detail"] for c in report["checks"]
        ), report["checks"]

    def test_a_representation_refusal_from_the_validator_suppresses_the_pass_tile(self):
        """The renderer's own half, executed on a REAL validator result.

        The result is produced by the real DataBiasValidator on the all-blank
        frame, then handed to the renderer for a frame it would otherwise call
        clean. Before the gating fix the renderer emitted the validator's
        could-not-check row AND, off the same result, "[pass] Group
        representation: Every group of "gender" is large enough for a reliable
        read", because the pass tile was gated on two issue types only.
        """
        blank = _two_group_frame()
        blank["gender"] = np.nan
        refusal = DataBiasValidator(protected_attributes=["gender"]).validate(blank)
        assert any(i.issue_type == "representation_not_measurable" for i in refusal.issues)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(DataBiasValidator, "validate", lambda self, *a, **k: refusal)
            report = build_quality_report(_two_group_frame(), ["gender"])

        rep = [c for c in report["checks"] if c["label"] == "Group representation"]
        assert len(rep) == 1, rep
        assert rep[0]["status"] != "pass"
        assert "could NOT be checked" in rep[0]["detail"]
        assert not any(
            c["status"] == "pass" and "large enough" in c["detail"] for c in report["checks"]
        ), report["checks"]

    def test_control_a_clean_frame_still_shows_its_three_pass_tiles(self):
        """OVER-CORRECTION CONTROL."""
        report = build_quality_report(_two_group_frame(), ["gender"], outcome_column="approved")
        by_key = {c["key"]: c["status"] for c in report["checks"]}
        assert by_key["vf_representation"] == "pass"
        assert by_key["vf_missing"] == "pass"
        assert by_key["vf_hygiene"] == "pass"
        assert report["tone"] == "pass"


# Unit 3: vfairness.operations.cicd.validator.DataValidationResult.execution_coverage


class TestExecutionCoverageKeepsFourStates:
    def _result(self, checks_run):
        return DataValidationResult(
            passed=True, issues=[], metrics={}, summary="s", checks_run=checks_run
        )

    def test_an_unrecorded_coverage_is_not_a_complete_one(self):
        """CORRECT before this wave, pinned here because nothing had judged it.
        The distinction that matters: None (this result does not say) must not
        read as either "" nor as complete."""
        assert self._result(None).execution_coverage() == "unrecorded"
        assert self._result([]).execution_coverage() == "none"
        assert self._result(["representation"]).execution_coverage() == "partial"
        assert self._result(list(VALIDATION_CHECKS)).execution_coverage() == "complete"

    def test_a_zero_issue_result_that_checked_nothing_says_could_not_check(self):
        empty = self._result([])
        assert empty.issues == []
        assert "COULD NOT CHECK" in repr(empty)
        assert 'failures="1"' in empty.to_junit_xml()
        assert empty.to_dict()["execution_coverage"] == "none"


# Unit 4: vfairness.operations.cicd.task_handlers.handle_data_validation


class TestTheHandlerNeverNarrowsTheRequestInSilence:
    def test_an_outcome_column_that_is_not_there_is_not_a_pass(self):
        """The worst finding in this batch. Measured before the fix, on 400 rows
        with a real 2.6x raw outcome gap between M and F:

            outcome_column='approved' -> tone 'critical', "[critical] Raw
                outcome gap ... Outcome rates differ sharply between groups"
            outcome_column='aproved'  -> tone 'pass', headline "The data is fit
                for a reliable fairness read", three green tiles, 0 warnings

        on the SAME rows. The name was silently set to None, so the gap was
        never compared and the report said the data was fine.
        """
        csv = _gap_frame().to_csv(index=False)

        out = handle_data_validation(
            {"csv_data": csv, "protected_attributes": ["gender"], "outcome_column": "aproved"}
        )

        assert out["success"] is True
        report = out["data"]
        assert report["tone"] != "pass", report["headline"]
        outcome_rows = [c for c in report["checks"] if c["label"] == "Outcome column"]
        assert len(outcome_rows) == 1, report["checks"]
        assert "aproved" in outcome_rows[0]["detail"]
        assert "could NOT run" in outcome_rows[0]["detail"]
        # And the checks that DID run still report their result. The tile for
        # blank values is gated on a substring match over the issue types, which
        # `missing_outcome_column` matched, so this row went missing altogether:
        # the check ran, found nothing, and the report said neither pass nor
        # could-not-check about it.
        by_key = {c["key"]: c["status"] for c in report["checks"]}
        assert by_key.get("vf_missing") == "pass", report["checks"]
        assert by_key.get("vf_hygiene") == "pass", report["checks"]

    def test_control_the_real_outcome_column_still_finds_the_gap(self):
        """OVER-CORRECTION CONTROL, and the evidence that the gap the misspelled
        run hid is really there."""
        csv = _gap_frame().to_csv(index=False)

        out = handle_data_validation(
            {"csv_data": csv, "protected_attributes": ["gender"], "outcome_column": "approved"}
        )

        report = out["data"]
        assert report["tone"] == "critical"
        assert any(c["label"] == "Raw outcome gap" for c in report["checks"]), report["checks"]

    def test_control_no_outcome_requested_is_still_a_clean_pass(self):
        """OVER-CORRECTION CONTROL. A caller with no outcome column (Pulse) asked
        for nothing about outcomes, so nothing is owed about them."""
        csv = _gap_frame().to_csv(index=False)

        out = handle_data_validation({"csv_data": csv, "protected_attributes": ["gender"]})

        assert out["data"]["tone"] == "pass", out["data"]["checks"]
        assert not any(c["label"] == "Outcome column" for c in out["data"]["checks"])

    def test_a_requested_attribute_that_is_not_in_the_data_is_named(self):
        """Measured before the fix: protected_attributes ['gender','disability']
        on a frame with no disability column produced a report byte-identical to
        the ['gender'] request, with no row, note or warning saying that half the
        requested scope was never assessed."""
        csv = _gap_frame().to_csv(index=False)

        out = handle_data_validation(
            {
                "csv_data": csv,
                "protected_attributes": ["gender", "disability"],
                "outcome_column": "approved",
            }
        )

        rows = [c for c in out["data"]["checks"] if c["label"] == "Attribute not in the data"]
        assert len(rows) == 1, out["data"]["checks"]
        assert "disability" in rows[0]["detail"]
        assert rows[0]["status"] == "warn"

    def test_the_disclosure_lifts_the_tone_off_pass(self):
        """A frame with nothing else wrong: the dropped attribute alone must stop
        the headline claiming the data is fit for a reliable read."""
        csv = _gap_frame().to_csv(index=False)

        out = handle_data_validation(
            {"csv_data": csv, "protected_attributes": ["gender", "disability"]}
        )

        assert out["data"]["tone"] == "warn", out["data"]["checks"]


# Unit 5: vfairness.operations.monitoring.alerts.FairnessAlertPrioritizer.create_alert


def _full_event() -> dict:
    return {
        "metric_name": "demographic_parity",
        "affected_groups": ["Roma"],
        "regulatory_risk": 1.0,
        "population_impact": 0.6,
        "drift_velocity": 0.8,
        "historical_discrimination": 1.0,
        "drift_score": 0.9,
        "mean_shift": -0.12,
    }


class TestTheAlertRefusesToGradeWhatNobodyMeasured:
    def test_an_event_with_nothing_measured_is_unscored_and_says_so_in_the_message(self):
        """The evidence the earlier grading wave lacked: executed on the empty
        drift event, create_alert returns severity 'UNSCORED', priority NaN,
        drift NaN and mean shift NaN, routes to hand triage rather than to the
        destination for alerts graded harmless, and says all of it in the one
        sentence a person reads off the channel."""
        prioritizer = FairnessAlertPrioritizer()

        payload, warned = _catch(
            prioritizer.create_alert, {"metric_name": "dp", "affected_groups": ["Roma"]}
        )

        assert payload.severity == "UNSCORED"
        assert math.isnan(payload.priority_score)
        assert math.isnan(payload.drift_score)
        assert math.isnan(payload.mean_shift)
        assert payload.routing["channel"] == "slack"
        assert "triage" in payload.routing
        assert "Drift score: NOT MEASURED" in payload.message
        assert "COULD NOT be computed" in payload.message
        assert any("could not be measured" in w for w in warned), warned
        assert round(payload.to_dict()["drift_score"], 6) != 0.0 or math.isnan(
            payload.to_dict()["drift_score"]
        )

    def test_a_measured_drift_score_is_not_thrown_away_for_its_dtype(self):
        """The refusal defect running backwards: np.float32 is not a float
        subclass. Pinned so a future tightening cannot de-escalate a real
        CRITICAL again."""
        event = _full_event()
        event["drift_score"] = np.float32(0.9)

        payload, warned = _catch(FairnessAlertPrioritizer().create_alert, event)

        assert payload.severity == "CRITICAL"
        assert payload.routing["channel"] == "pagerduty"
        assert warned == []

    def test_the_supplied_override_decides_both_the_payload_and_the_score(self):
        """Measured before this wave, with the four context factors all present
        (score 9.70 from context alone):

            event {'drift_score': None}, drift_score=0.9 -> severity UNSCORED,
              priority nan, payload drift_score 0.9, message "Drift score: 0.900
              ... severity COULD NOT be computed", and a warning saying the
              drift "was never measured" about a value just supplied
            event {'drift_score': 0.7}, drift_score=0.0 -> priority 10.60
              CRITICAL pagerduty computed from the 0.7, printed as
              "Drift score: 0.000"

        The argument is documented as an override, so it is now the one number
        both halves use.
        """
        unreadable = _full_event()
        unreadable["drift_score"] = None
        payload, warned = _catch(
            FairnessAlertPrioritizer().create_alert, unreadable, drift_score=0.9
        )
        assert payload.severity == "CRITICAL"
        assert payload.drift_score == pytest.approx(0.9)
        assert payload.priority_score == pytest.approx(10.6)
        assert not any("never measured" in w for w in warned), warned

        measured = _full_event()
        measured["drift_score"] = 0.7
        payload2, _ = _catch(FairnessAlertPrioritizer().create_alert, measured, drift_score=0.0)
        assert payload2.drift_score == pytest.approx(0.0)
        assert payload2.priority_score == pytest.approx(9.7), (
            "the score was computed from a drift value the payload denies"
        )

    def test_control_a_fully_measured_event_still_pages(self):
        """OVER-CORRECTION CONTROL."""
        payload, warned = _catch(FairnessAlertPrioritizer().create_alert, _full_event())

        assert payload.severity == "CRITICAL"
        assert payload.priority_score == pytest.approx(10.6)
        assert payload.routing["channel"] == "pagerduty"
        assert "Drift score: 0.900" in payload.message
        assert warned == []


# Unit 6: vfairness.operations.monitoring.drift.FairnessDriftDetector.run_sprt


def _stable_stream() -> List[float]:
    rng = np.random.default_rng(3)
    return list(np.round(0.10 + rng.normal(0, 0.01, 40), 6))


def _drifted_stream() -> List[float]:
    rng = np.random.default_rng(3)
    return list(np.round(0.30 + rng.normal(0, 0.01, 40), 6))


class TestSprtRefusesErrorRatesThatSpecifyNoTest:
    @pytest.mark.parametrize(
        "kwargs,before",
        [
            ({"alpha": 0.5, "beta": 0.8}, "('stable', 1, -118.23) with 0 warnings"),
            ({"alpha": -0.05}, "('stable', 1, -118.23), numpy RuntimeWarning only"),
            ({"beta": 1.0}, "('drift', 1, -118.23), a NEGATIVE llr reported as drift"),
            ({"beta": 0.0}, "('continue', 40, -5966.17)"),
            ({"alpha": float("nan")}, "('continue', 40, -5966.17)"),
            ({"alpha": 0.0}, "ZeroDivisionError"),
            ({"alpha": 1.0}, "ZeroDivisionError"),
        ],
    )
    def test_a_degenerate_error_rate_yields_no_verdict(self, kwargs, before):
        """G-20 validated the two hypotheses and the stream and left the two
        error rates unchecked, although they are the only inputs to both Wald
        boundaries. The *before* column of each case is what this exact call
        returned on 2026-09-27; the stream is the same one the defaults decide as
        ('stable', 1, -118.227)."""
        decision, _, llr = FairnessDriftDetector().run_sprt(
            _stable_stream(), null_value=0.10, alternative_value=0.30, **kwargs
        )

        assert decision == "could_not_check", f"{decision!r}, was {before}"
        assert math.isnan(llr), "a likelihood ratio was reported for a test that cannot run"

    def test_the_refusal_names_what_is_wrong(self):
        (_, _, _), warned = _catch(
            FairnessDriftDetector().run_sprt,
            _stable_stream(),
            null_value=0.10,
            alternative_value=0.30,
            alpha=0.5,
            beta=0.8,
        )
        assert any("no acceptance region" in w for w in warned), warned
        assert any("could_not_check" in w for w in warned), warned

    def test_control_the_default_error_rates_still_decide_both_ways(self):
        """OVER-CORRECTION CONTROL. The stable stream's log-likelihood ratio is
        pinned to the value measured before the guard was added, so a change to
        the arithmetic is visible; the drifted one is pinned by sign and
        magnitude, which is what a manufactured verdict would break (the repo's
        own control uses an abs bound of 1e4 for exactly that)."""
        stable, n_stable, llr_stable = FairnessDriftDetector().run_sprt(
            _stable_stream(), null_value=0.10, alternative_value=0.30
        )
        assert (stable, n_stable) == ("stable", 1)
        assert llr_stable == pytest.approx(-118.22721539073946)

        drift, n_drift, llr_drift = FairnessDriftDetector().run_sprt(
            _drifted_stream(), null_value=0.10, alternative_value=0.30
        )
        assert (drift, n_drift) == ("drift", 1)
        assert llr_drift > 0 and abs(llr_drift) < 1e4, llr_drift

    def test_control_a_valid_non_default_error_rate_still_runs(self):
        """OVER-CORRECTION CONTROL. alpha 0.01 with beta 0.10 is a legitimate,
        stricter test and must still produce a verdict."""
        decision, _, llr = FairnessDriftDetector().run_sprt(
            _drifted_stream(), null_value=0.10, alternative_value=0.30, alpha=0.01, beta=0.10
        )
        assert decision == "drift"
        assert math.isfinite(llr)


# Unit 7: vfairness.operations.reporting.store.MetricsStore.ingest_dataframe


def _tidy_frame(determined) -> pd.DataFrame:
    """Six readings of demographic_parity = 0.95, about as unfair as that metric
    gets, none of which anybody compared to a threshold."""
    now = datetime.now()
    n = len(determined)
    return pd.DataFrame(
        {
            "timestamp": [now - timedelta(hours=i) for i in range(n)],
            "metric": ["demographic_parity"] * n,
            "value": [0.95] * n,
            "group_size": [500] * n,
            "alert": [False] * n,
            "alert_determined": determined,
        }
    )


class TestIngestDataframeReadsTheDeterminationFlagHonestly:
    def _ingest(self, frame):
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
        _, warned = _catch(
            store.ingest_dataframe, frame, alert_col="alert", group_size_col="group_size"
        )
        health, health_warned = _catch(store.compute_health_score)
        return store, health, warned + health_warned

    def test_a_nan_determination_flag_is_not_a_clean_comparison(self):
        """Measured before the fix, on the six 0.95 readings above:

            alert_determined=False -> alert None, health None / 'not_assessed',
                n_not_assessable=6, 2 warnings
            alert_determined=NaN   -> alert False, health 100.0 / 'green',
                metric_compliance 100.0, n_not_assessable=0, 0 warnings

        byte-identical to the control where the comparison really ran and came
        back clean, because bool(float("nan")) is True. It is verbatim the
        READINESS-6 scenario this method's docstring says was fixed, reached
        through an unrecorded flag instead of a missing column.
        """
        store, health, warned = self._ingest(_tidy_frame([np.nan] * 6))

        assert [r.alert for r in store._records] == [None] * 6
        assert health.score is None
        assert health.status == "not_assessed"
        assert health.n_not_assessable == 6
        assert any("no readable value" in w for w in warned), warned

    def test_a_recorded_absence_still_reads_as_an_absence(self):
        store, health, _ = self._ingest(_tidy_frame([False] * 6))
        assert [r.alert for r in store._records] == [None] * 6
        assert health.score is None

    def test_control_a_recorded_clean_comparison_still_scores(self):
        """OVER-CORRECTION CONTROL. A determination that really was made must
        still reach the compliance mean, or the store refuses everything."""
        store, health, warned = self._ingest(_tidy_frame([True] * 6))

        assert [r.alert for r in store._records] == [False] * 6
        assert health.score == pytest.approx(100.0)
        assert health.n_not_assessable == 0
        # The only warning left is the drift-coverage disclosure added lower down
        # in this file: this store holds no drift result, and that component is
        # now excluded instead of defaulting to 100.
        assert not any("determination" in w for w in warned), warned

    def test_control_a_recorded_breach_survives_the_round_trip(self):
        """OVER-CORRECTION CONTROL: True must stay True."""
        frame = _tidy_frame([True] * 6)
        frame["alert"] = [True] * 6
        store, _, _ = self._ingest(frame)
        assert [r.alert for r in store._records] == [True] * 6


# Unit 8: vfairness.operations.reporting.store.MetricsStore.ingest_window_metrics


def _snapshot(**kw) -> WindowMetrics:
    base = dict(
        batch_id="b1",
        timestamp=datetime.now(),
        sample_count=500,
        metrics={"demographic_parity_gender": 0.04},
        group_rates={"gender": {"F": 0.31, "M": 0.33}},
        alerts={"demographic_parity_gender": False},
        mmd_scores={"gender": 0.02},
        excluded_groups={},
    )
    base.update(kw)
    return WindowMetrics(**base)


class TestIngestWindowMetricsDisclosesWhatItCouldNotMeasure:
    def test_a_per_group_rate_says_whose_size_it_carries(self):
        """Measured before this wave, on a 500-row window with group_rates
        {"gender": {"F": 0.31, "M": 0.33}} and privacy enabled:

            positive_rate gender_F  0.31  group_size 500  privacy_level exact
            positive_rate gender_M  0.33  group_size 500  privacy_level exact

        with no warning. 'exact' asserts the GROUP cleared noisy_threshold, and
        the snapshot reports no per-group count at all, so the k-anonymity gate
        was answered with the size of the whole window. The value cannot move
        from inside this method (see its Notes and the report for this batch);
        the claim it implies is disclosed instead.
        """
        store = MetricsStore(MetricsStoreConfig(enable_privacy=True))

        n, warned = _catch(store.ingest_window_metrics, _snapshot())

        assert n == 4
        assert any("per-group rate record" in w for w in warned), warned
        assert any("NOT a verified k-anonymity result" in w for w in warned), warned

    def test_a_group_the_floor_excluded_is_named(self):
        """Measured before this wave, on a window that kept gender=F (4 rows) out
        of every aggregate: the store held a NaN demographic_parity with no
        determination, one positive_rate row for gender_M, and nothing at all
        recording that a group had been left out."""
        snap = _snapshot(
            metrics={"demographic_parity_gender": float("nan")},
            alerts={},
            group_rates={"gender": {"M": 0.33}},
            excluded_groups={"gender": {"F": 4}},
        )
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))

        n, warned = _catch(store.ingest_window_metrics, snap)

        assert n == 3
        assert any("kept group(s) OUT" in w for w in warned), warned
        assert any("gender" in w and "F" in w for w in warned), warned
        by_name = {r.metric_name: r for r in store._records}
        assert math.isnan(by_name["demographic_parity"].value)
        assert by_name["demographic_parity"].alert is None

    def test_the_third_state_still_travels_with_every_record(self):
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
        _catch(store.ingest_window_metrics, _snapshot())
        by_name = {r.metric_name: r for r in store._records}
        assert by_name["demographic_parity"].alert is False, "a real comparison, clean"
        assert by_name["positive_rate"].alert is None, "no threshold exists for a rate"
        assert by_name["mmd_score"].alert is None

    def test_control_a_window_with_nothing_to_disclose_is_silent(self):
        """OVER-CORRECTION CONTROL. With privacy off there is no k-anonymity
        claim to qualify, and with no excluded group there is no absence, so a
        clean ingest must warn about nothing at all."""
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))

        n, warned = _catch(store.ingest_window_metrics, _snapshot())

        assert n == 4
        assert warned == []


# Units 9 and 10, handed over by the agent on reporting/dashboard.py and
# reports.py, which proved both and could not fix them from outside this file:
#
# vfairness.operations.reporting.store.MetricsStore.compute_health_score
# vfairness.operations.reporting.store._drift_stability


def _clean_store(n: int = 4, previous: int = 0) -> MetricsStore:
    """A store holding *n* clean, fully determined metric records in the current
    window and *previous* of them one window back. No drift result at all, which
    is the default state of every MetricsStore."""
    from vfairness.operations.reporting.store import StoredMetricRecord

    now = datetime.now()
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    for i in range(n):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(hours=i),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.02,
                group="gender",
                group_size=500,
                alert=False,
            )
        )
    for i in range(previous):
        store._records.append(
            StoredMetricRecord(
                timestamp=now - timedelta(days=8, hours=i),
                source="FairnessMonitor",
                metric_name="demographic_parity",
                value=0.50,
                group="gender",
                group_size=500,
                alert=True,
            )
        )
    return store


class TestTheHealthScoreCarriesNoConstantForAnUnmeasuredComponent:
    def test_an_empty_drift_table_is_not_perfect_stability(self):
        """Measured before the fix:

            _drift_stability(pd.DataFrame()) -> (100.0, 0, [])

        and on four clean records with zero drift rows:

            score=100.0 green, components={'metric_compliance': 100.0,
            'alert_frequency': 100.0, 'drift_stability': 100.0}, 0 warnings

        so a fifth of a colour-banded composite was a constant standing in for a
        component nothing had measured, and the word drift appeared nowhere a
        reader would see it.
        """
        from vfairness.operations.reporting.store import _drift_stability

        assert _drift_stability(pd.DataFrame()) == (None, 0, [])

        health, warned = _catch(_clean_store().compute_health_score)

        assert "drift_stability" not in health.components, health.components
        assert any("could not be measured" in w for w in warned), warned
        assert "NOT measured" in health.explanation
        assert health.score == pytest.approx(100.0), (
            "two components at 100 renormalise to 100; what must not survive is the "
            "unmeasured third one"
        )

    def test_the_unmeasured_component_no_longer_lifts_an_imperfect_score(self):
        """The half that shows the constant was load bearing: one breaching
        record out of two, no drift row.

            before: 0.50 * 0.0 + 0.30 * 100 + 0.20 * 100 = 50.0
            then:   (0.50 * 0.0 + 0.30 * 100) / 0.80     = 37.5
            now:    (0.50 * 0.0) / 0.50                  = 0.0

        BGL5 A-operations-3, 2026-09-27: the 37.5 still contained 0.30 * 100.0
        for the ALERT component, and `_clean_store` never feeds the alert channel
        either, so that term was the same no-evidence constant this test exists
        to refuse, one component over. An audit overturned the grade on it. The
        subject is unchanged and is stronger now: an imperfect score is lifted by
        nothing that nobody measured.
        """
        store = _clean_store(n=2)
        store._records[0].alert = True

        health, _ = _catch(store.compute_health_score)

        assert health.components["metric_compliance"] == pytest.approx(0.0)
        assert "alert_frequency" not in health.components, health.components
        assert health.score == pytest.approx(0.0), health.components

    def test_control_a_measured_drift_result_is_still_graded_and_weighted(self):
        """OVER-CORRECTION CONTROL. A store that DOES carry drift verdicts keeps
        grading and weighting them.

        BGL5 A-operations-3, 2026-09-27: the expected composite moved from 80.0
        to 71.4 because the alert component left it (see the test above), so the
        weights over the components that HAVE evidence are 0.50 and 0.20:
        (0.50 * 100 + 0.20 * 0) / 0.70 = 71.4. The subject, that a measured drift
        verdict of 0.0 is graded and pulls the score down, is unchanged. A store
        that feeds all three channels is asserted at the full documented 50/30/20
        in tests/test_bgl5_operations_3.py::
        test_control_a_fed_alert_channel_keeps_the_documented_composite (87.0).
        """
        store = _clean_store()
        store._drift_records.append(
            {
                "timestamp": datetime.now(),
                "metric": "demographic_parity",
                "overall_drift_score": 0.9,
                "drift_detected": True,
                "mmd_score": None,
            }
        )

        health, _ = _catch(store.compute_health_score)

        assert health.components["drift_stability"] == pytest.approx(0.0)
        assert health.components["metric_compliance"] == pytest.approx(100.0)
        assert health.score == pytest.approx(71.4), health.components

    def test_a_first_window_has_no_trend_rather_than_a_stable_one(self):
        """Measured before the fix, four clean records whose previous window held
        NO record at all:

            trend='stable', trend_slope=0.0, explanation "Fairness health score
            is 100/100 (healthy) with a stable trend."

        "there was no previous window" and "the score did not move" reached the
        caller as the same string, and 'stable' is the reassuring one.
        """
        health, _ = _catch(_clean_store(previous=0).compute_health_score)

        assert health.trend == "unknown", health.trend
        assert health.trend_slope == 0.0
        assert "not a stable trend" in health.explanation

    def test_control_a_real_previous_window_still_produces_a_trend(self):
        """OVER-CORRECTION CONTROL. Four breaching records one window back and
        four clean ones now is a comparison that CAN be made, so it must be."""
        health, _ = _catch(_clean_store(previous=4).compute_health_score)

        assert health.trend == "improving", (health.trend, health.trend_slope)
        assert health.trend_slope > 0.5


class TestSubDayWindowStillHasATrend:
    """A window shorter than a day must not report a stable trend by truncation.

    Found by the fabricated-verdict ledger gate rather than by a probe, which is worth
    recording: the gate flagged the surviving ternary at this site after the rest of
    compute_health_score had been fixed, and reading it turned up a live defect the
    batch had not reached.

    The slope was `(score - prev_score) / time_window.days`, and timedelta.days
    TRUNCATES: timedelta(hours=12).days and timedelta(hours=23).days are both 0. So for
    any window shorter than a day the guarded divisor sent the slope to exactly 0.0,
    which lands in the `else` arm of the trend mapping, and "stable" was asserted for
    two windows whose scores may differ by anything at all.

    It now divides by total_seconds()/86400, so a twelve-hour window is measured as
    half a day, and a window of no duration is routed to 'unknown' rather than to 0.0,
    because two readings at the same instant have not been shown to agree.
    """

    @staticmethod
    def _falling_store():
        """Two windows whose health scores differ, so a real slope exists."""
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
        now = datetime.now()
        rows = []
        # Older half: clean. Newer half: every metric in breach, so the score falls.
        for i, (ts, value, alert) in enumerate(
            [(now - timedelta(hours=20), 0.95, False)] * 6
            + [(now - timedelta(hours=2), 0.20, True)] * 6
        ):
            rows.append(
                {
                    "timestamp": ts,
                    "metric": "demographic_parity_ratio",
                    "group": f"g{i % 2}",
                    "value": value,
                    "group_size": 500,
                    "alert": alert,
                    "alert_determined": True,
                }
            )
        store.ingest_dataframe(pd.DataFrame(rows), alert_col="alert", group_size_col="group_size")
        return store

    def test_a_twelve_hour_window_is_not_truncated_to_a_stable_trend(self):
        store = self._falling_store()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            health = store.compute_health_score(time_window=timedelta(hours=12))
        # `trend_slope`, not `slope`. The first version of this assertion read
        # `health.trend != "stable" or health.slope != 0.0`, which SHORT-CIRCUITS: it
        # passed only because the trend was not stable, and would have raised
        # AttributeError in exactly the case it exists to catch. Asserted positively
        # now, so there is nothing for a short circuit to skip.
        assert health.trend == "degrading", (
            f"a 12 hour window over a falling score reported trend={health.trend!r}. "
            f"timedelta(hours=12).days is 0, so dividing by .days forced the slope to "
            f"exactly 0.0 and the mapping read that as stable."
        )
        assert health.trend_slope < 0.0, health.trend_slope

    def test_a_window_of_no_duration_refuses_before_any_slope_is_computed(self):
        """Asserts the mechanism that ACTUALLY produces this, not the one I added.

        The first version of this test asserted trend == 'unknown' and credited a
        zero-duration branch in the slope arithmetic. Sabotaging that branch left this
        test GREEN, which showed the branch was unreachable: a window of no duration
        contains no records, so compute_health_score returns earlier with score=None.
        The branch was removed rather than left as a guard nobody reaches, and this
        test now names the real refusal, including the sentence a reader sees.
        """
        store = self._falling_store()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            health = store.compute_health_score(time_window=timedelta(0))
        assert health.score is None, (
            f"score={health.score!r} over a window containing no records at all"
        )
        assert health.trend == "unknown"
        assert health.n_metrics == 0
        assert "not a score of 100" in health.explanation, health.explanation

    @staticmethod
    def _two_week_store():
        """Rows across fourteen days, so a SEVEN day window has a populated previous
        window to compare against.

        _falling_store puts everything inside twenty hours, which is right for the
        sub-day cases above and wrong here: a seven day window's previous window would
        hold no record, and 'unknown' would then be the correct answer for a reason
        that has nothing to do with the divisor. The first version of this control used
        it and failed for exactly that reason, which is the control working.
        """
        store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
        now = datetime.now()
        rows = []
        for i in range(24):
            day = 13 - i // 2
            rows.append(
                {
                    "timestamp": now - timedelta(days=day),
                    "metric": "demographic_parity_ratio",
                    "group": f"g{i % 2}",
                    # Clean in the older week, in breach in the newer one.
                    "value": 0.95 if day > 7 else 0.20,
                    "group_size": 500,
                    "alert": day <= 7,
                    "alert_determined": True,
                }
            )
        store.ingest_dataframe(pd.DataFrame(rows), alert_col="alert", group_size_col="group_size")
        return store

    def test_a_multi_day_window_still_measures_a_trend(self):
        """OVER-CORRECTION CONTROL. The change must not make every trend unknown."""
        store = self._two_week_store()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            health = store.compute_health_score(time_window=timedelta(days=7))
        assert health.trend in ("improving", "degrading", "stable"), (
            f"trend={health.trend!r}: a seven day window with a populated previous "
            f"week must produce a measured trend, not a refusal"
        )
        assert isinstance(health.trend_slope, float)
