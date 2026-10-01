"""Readiness lane: an all-clear printed over a could-not-check.

The library computes the honest three-state answer and the report generator
threw it away at the narrative boundary, so the only surface a stakeholder
actually reads asserted compliance for things nobody measured. Both findings
below were reproduced by execution on 2026-09-09 before any fix:

- F1 ``reports._narrate_metrics`` / ``_generate_recommendations``. The store
     returns ``score=None, status="not_assessed", n_not_assessable=3`` and
     warns twice naming the metrics; ``get_metrics`` ships an
     ``alert_determined`` column beside ``alert`` for exactly this purpose.
     The narrative read only ``alert`` (``bool(None)`` is False) and the
     recommendations tested only ``red``/``yellow``, so an executive report
     printed "All 2 tracked fairness metrics are within acceptable limits."
     and "All systems nominal." on the same page whose header said
     "Fairness health could NOT be assessed".
- F2 ``reports.generate_drift_report``. ``MultiscaleDriftResult.drift_detected``
     is ``None`` when no scale could be computed, so no drift test ran.
     ``if drift_result.drift_detected:`` is a two-state test and ``None`` is
     falsy, so the report recommended "No significant drift detected. Continue
     routine monitoring." for a check that never happened, while
     ``drift_report_to_svg`` rendered COULD NOT CHECK for the same object.

Every finding carries a refusal pin AND an over-correction control asserting
the MEASURED strings verbatim: a genuinely clean report must still say it is
clean, in its existing words, or the fix has merely broken the product.
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.drift import FairnessDriftDetector
from vfairness.operations.monitoring.tracker import WindowMetrics
from vfairness.operations.reporting.reports import (
    OutputFormat,
    ReportGenerator,
    ReportTier,
    _generate_recommendations,
    _narrate_metrics,
)
from vfairness.operations.reporting.store import MetricsStore

# The two sentences this lane exists to keep off an unassessed page.
ALL_CLEAR = "All 2 tracked fairness metrics are within acceptable limits."
NOMINAL = (
    "All systems nominal. Continue routine monitoring and maintain "
    "the current alert threshold configuration."
)
NO_DRIFT = "No significant drift detected. Continue routine monitoring."

BUILTIN_A = "demographic_parity_difference_group_sex"
BUILTIN_B = "equal_opportunity_difference_group_sex"


def _store(metrics, alerts) -> MetricsStore:
    """A MetricsStore holding one real FairnessMonitor window snapshot.

    ``alerts`` is the record of which guardrails were APPLIED: a metric present
    in ``metrics`` and absent from ``alerts`` was never compared to anything.
    ``sample_count`` is 500 so no row is suppressed or noised by the privacy
    layer, which would confound the assessed/unassessed split under test.
    """
    store = MetricsStore()
    store.ingest_window_metrics(
        WindowMetrics(
            batch_id="b1",
            timestamp=datetime.now(),
            sample_count=500,
            metrics=metrics,
            group_rates={},
            alerts=alerts,
        )
    )
    return store


def _health(store: MetricsStore):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return store.compute_health_score(time_window=timedelta(days=30))


def _report(store: MetricsStore) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return (
            ReportGenerator(store)
            .generate_executive_report(output_format=OutputFormat.MARKDOWN)
            .content
        )


# ---------------------------------------------------------------------------
# F1: an executive report must not assert compliance for an ungraded metric
# ---------------------------------------------------------------------------


class TestExecutiveAllClear:
    def test_refuses_all_clear_when_nothing_was_compared(self):
        """REFUSAL PIN. Three custom metrics, maximally unfair, none of them
        ever compared to a threshold. The producer withholds the score; the
        narrative used to certify them anyway."""
        store = _store(
            {
                "custom_gap_group_male": 0.97,
                "custom_gap_group_female": 0.02,
                "custom_ratio_group_male": 0.99,
            },
            {},
        )
        hs = _health(store)
        assert hs.score is None and hs.status == "not_assessed"
        assert hs.n_not_assessable == 3

        content = _report(store)
        assert "within acceptable limits" not in content, (
            f"the executive report certified metrics nobody compared to a threshold:\n{content}"
        )
        assert "All systems nominal" not in content, (
            f"an all-clear was recommended for a window nobody graded:\n{content}"
        )
        assert "COULD NOT CHECK" in content
        # The header and the body must not contradict each other on one page.
        assert "could NOT be assessed" in content
        # The withheld sub-scores are not a measured zero either.
        assert "| Metric Compliance | not assessed |" in content
        assert "| Metric Compliance | 0.0 |" not in content

    def test_refuses_all_clear_when_only_some_metrics_were_compared(self):
        """REFUSAL PIN, the partial case: the health score is a real 100/100
        GREEN computed from the two graded records, and the all-clear used to
        be stretched over all four ("All 4 tracked fairness metrics ...")."""
        store = _store(
            {
                BUILTIN_A: 0.01,
                BUILTIN_B: 0.02,
                "custom_gap_group_sex": 0.93,
                "custom_ratio_group_sex": 0.88,
            },
            {BUILTIN_A: False, BUILTIN_B: False},
        )
        hs = _health(store)
        assert hs.score == pytest.approx(100.0) and hs.status == "green"
        assert hs.n_not_assessable == 2

        df = store.get_metrics()
        exec_text = _narrate_metrics(df, ReportTier.EXECUTIVE)
        assert "All 4 tracked fairness metrics are within acceptable limits." not in exec_text
        assert exec_text.startswith("All 2 assessed fairness metrics are within acceptable limits.")
        assert "COULD NOT CHECK: 2 of 4 measurement(s)" in exec_text
        assert "custom_gap" in exec_text and "custom_ratio" in exec_text

        recs = _generate_recommendations(hs, df, [])
        assert NOMINAL not in recs, (
            "a GREEN score over two graded records recommended an all-clear "
            f"covering two nobody graded: {recs}"
        )
        assert any("COULD NOT CHECK" in r for r in recs)

    def test_refuses_all_clear_for_a_metric_that_could_not_be_computed(self):
        """REFUSAL PIN, H-10 at the narrative boundary: the metric WAS compared
        (a determination exists) but its value is NaN, so nothing was measured.
        ``store._compliance_rows`` excludes it; the narrative counted it clean."""
        store = _store({BUILTIN_A: float("nan")}, {BUILTIN_A: False})
        df = store.get_metrics()
        assert bool(df.loc[0, "alert_determined"]) is True
        assert pd.isna(df.loc[0, "value"])

        text = _narrate_metrics(df, ReportTier.EXECUTIVE)
        assert "within acceptable limits" not in text, text
        assert "COULD NOT CHECK" in text

    def test_operational_and_technical_tiers_name_the_ungraded_records(self):
        """The other two tiers read the same frame and must not report a rate
        whose denominator includes records nobody graded."""
        store = _store(
            {BUILTIN_A: 0.01, "custom_gap_group_sex": 0.93},
            {BUILTIN_A: False},
        )
        df = store.get_metrics()
        ops = _narrate_metrics(df, ReportTier.OPERATIONAL)
        tech = _narrate_metrics(df, ReportTier.TECHNICAL)
        assert "COULD NOT CHECK" in ops and "custom_gap" in ops
        assert "COULD NOT CHECK" in tech and "custom_gap" in tech
        assert "the alert rate is over the 1 assessed record(s)" in tech

    # OVER-CORRECTION CONTROLS

    def test_a_genuinely_clean_report_still_says_so_in_its_existing_words(self):
        """OVER-CORRECTION CONTROL. Two built-in metrics, both compared, both
        clean. The exact strings, because a fix that hedges everything has
        broken the product just as surely as one that certified everything."""
        store = _store({BUILTIN_A: 0.01, BUILTIN_B: 0.02}, {BUILTIN_A: False, BUILTIN_B: False})
        hs = _health(store)
        assert hs.score == pytest.approx(100.0) and hs.status == "green"

        df = store.get_metrics()
        assert _narrate_metrics(df, ReportTier.EXECUTIVE) == ALL_CLEAR
        assert _generate_recommendations(hs, df, []) == [NOMINAL]

        content = _report(store)
        assert ALL_CLEAR in content
        assert NOMINAL in content
        assert "COULD NOT CHECK" not in content
        # The measured sub-scores are still printed as numbers.
        assert "| Metric Compliance | 100.0 |" in content

    def test_a_genuinely_alerting_report_still_reports_its_measured_rate(self):
        """OVER-CORRECTION CONTROL. One of two graded records alerted: the
        measured sentence, verbatim, with the measured percentage."""
        store = _store({BUILTIN_A: 0.9, BUILTIN_B: 0.02}, {BUILTIN_A: True, BUILTIN_B: False})
        df = store.get_metrics()
        assert (
            _narrate_metrics(df, ReportTier.EXECUTIVE)
            == "1 of 2 measurements (50%) triggered alerts."
        )
        assert (
            _narrate_metrics(df, ReportTier.TECHNICAL)
            == "Total records: 2 | Unique metrics: 2 | Groups: 1 | Alerts: 1 (50.00%)"
        )

    def test_a_privacy_suppressed_value_is_still_a_graded_record(self):
        """OVER-CORRECTION CONTROL, the one that decides whether this fix is
        usable. A k-anonymity SUPPRESSED row has value NaN in the frame, but
        the threshold comparison behind it ran on the true value. Calling every
        suppressed row could-not-check would raise a false alarm on every
        privacy-protected report ever generated."""
        store = _store({BUILTIN_A: 0.01, BUILTIN_B: 0.02}, {BUILTIN_A: False, BUILTIN_B: False})
        store._records[0].group_size = 4  # below k = 10, so the value is withheld
        store._records[1].group_size = 4
        df = store.get_metrics()
        assert list(df["privacy_level"]) == ["suppressed", "suppressed"]
        assert df["value"].isna().all()
        assert bool(df["alert_determined"].all()) is True

        assert _narrate_metrics(df, ReportTier.EXECUTIVE) == ALL_CLEAR


# ---------------------------------------------------------------------------
# F2: a drift report must not call an un-run test stability
# ---------------------------------------------------------------------------


def _drift_result(baseline, current, metric="demographic_parity_difference"):
    det = FairnessDriftDetector()
    det.set_baseline(pd.Series(baseline, index=pd.date_range("2026-01-01", periods=len(baseline))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return det.check_drift(
            pd.Series(current, index=pd.date_range("2026-06-01", periods=len(current))),
            metric=metric,
        )


def _drift_report(res):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ReportGenerator(MetricsStore()).generate_drift_report(
            res, output_format=OutputFormat.MARKDOWN
        )


class TestDriftAllClear:
    def test_refuses_stability_when_no_scale_could_be_computed(self):
        """REFUSAL PIN. A two-point series: no scale is computable, so no drift
        test ran. The producer says so (score NaN, drift_detected None, a
        warning); this consumer used to answer routine monitoring."""
        with pytest.warns(UserWarning, match="COULD NOT CHECK"):
            det = FairnessDriftDetector()
            det.set_baseline(pd.Series([0.20], index=pd.date_range("2025-12-31", periods=1)))
            res = det.check_drift(
                pd.Series([0.21, 0.23], index=pd.date_range("2026-01-01", periods=2)),
                metric="demographic_parity_difference",
            )
        assert res.drift_detected is None
        assert np.isnan(res.overall_drift_score)
        assert res.scales == {}

        rep = _drift_report(res)
        assert NO_DRIFT not in rep.recommendations, (
            f"an un-run drift test was reported as stability: {rep.recommendations}"
        )
        assert any("COULD NOT CHECK" in r for r in rep.recommendations)
        assert "COULD NOT CHECK" not in NO_DRIFT  # the pin above is not vacuous

        summary = rep.sections[0]["content"]
        assert "Drift Detected: None" not in summary, summary
        assert "Overall Drift Score: nan" not in summary, summary
        assert "COULD NOT CHECK (no scale was analysed, so no drift test ran)" in summary
        assert "Overall Drift Score: not measured" in summary
        assert "COULD NOT CHECK" in rep.content

    def test_a_measured_stable_result_still_says_no_significant_drift(self):
        """OVER-CORRECTION CONTROL. 120 reference points against 120 more from
        the same distribution: a real drift test ran and found nothing."""
        rng = np.random.default_rng(7)
        res = _drift_result(rng.normal(0.2, 0.01, 120), rng.normal(0.2, 0.01, 120))
        assert res.drift_detected is False
        assert not np.isnan(res.overall_drift_score)

        rep = _drift_report(res)
        assert rep.recommendations == [NO_DRIFT]
        summary = rep.sections[0]["content"]
        assert "Drift Detected: False" in summary
        assert f"Overall Drift Score: {res.overall_drift_score:.4f}" in summary
        assert "not measured" not in summary
        assert "COULD NOT CHECK" not in rep.content

    def test_a_measured_drifted_result_still_confirms_drift(self):
        """OVER-CORRECTION CONTROL. A 0.4 mean shift: drift is found, and the
        two existing recommendations are unchanged."""
        rng = np.random.default_rng(7)
        res = _drift_result(rng.normal(0.2, 0.01, 120), rng.normal(0.6, 0.01, 120))
        assert res.drift_detected is True

        rep = _drift_report(res)
        assert rep.recommendations == [
            "Drift confirmed. Investigate data distribution changes.",
            "Consider retraining the model with recent data.",
        ]
        assert "Drift Detected: True" in rep.sections[0]["content"]


# ---------------------------------------------------------------------------
# Siblings of the same shape, in the same file
# ---------------------------------------------------------------------------


class TestSiblingNeutralDefaults:
    def test_metric_table_counts_the_records_nobody_graded(self):
        """The Tier-2 table showed n_alerts 0 beside n_records 1 for a metric
        that was never compared: arithmetic, read as evidence."""
        store = _store({BUILTIN_A: 0.01, "custom_gap_group_sex": 0.93}, {BUILTIN_A: False})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rep = ReportGenerator(store).generate_operational_report(
                output_format=OutputFormat.HTML
            )
        table = [s for s in rep.sections if s["title"] == "Metric Statistics"][0]["content"]
        assert "n_not_assessed" in table, table
        # Parsed off the rendered HTML rather than the DataFrame, because the
        # rendered table is what a reader sees. `n_not_assessed` is the last
        # column, so it is the last cell of each row.
        rows = {
            re.findall(r"<td[^>]*>([^<]*)</td>", tr)[0]: re.findall(r"<td[^>]*>([^<]*)</td>", tr)[
                -1
            ]
            for tr in re.findall(r"<tr>(.*?)</tr>", table, flags=re.S)
            if "<td>" in tr
        }
        assert rows["custom_gap"] == "1", rows
        assert rows["demographic_parity_difference"] == "0", rows

    def test_an_alert_with_no_drift_score_does_not_report_a_drift_score_of_zero(self):
        """``d.get('drift_score', 0):.4f`` printed 0.0000, the calmest number
        on the page, for a payload that carries no drift score at all."""
        gen = ReportGenerator(MetricsStore())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            partial = gen.generate_alert_report(
                {"severity": "HIGH", "metric_name": "demographic_parity_difference"},
                output_format=OutputFormat.MARKDOWN,
            )
            measured = gen.generate_alert_report(
                {
                    "severity": "HIGH",
                    "metric_name": "demographic_parity_difference",
                    "priority_score": 7.25,
                    "drift_score": 0.4321,
                },
                output_format=OutputFormat.MARKDOWN,
            )
        detail = partial.sections[0]["content"]
        assert "Drift Score: 0.0000" not in detail, detail
        assert "Drift Score: not measured" in detail
        assert "Priority Score: not measured" in detail
        # OVER-CORRECTION CONTROL: a payload that HAS the numbers prints them.
        assert "Drift Score: 0.4321" in measured.sections[0]["content"]
        assert "Priority Score: 7.25" in measured.sections[0]["content"]

    def test_a_zero_threshold_has_no_percentage_beyond_it(self):
        """``if threshold else 0`` printed "0.0% beyond threshold" for a breach
        of any size on a page whose whole subject is that breach."""
        gen = ReportGenerator(MetricsStore())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            undefined = gen.generate_threshold_breach_report(
                "demographic_parity_difference", 0.35, 0.0, OutputFormat.MARKDOWN
            )
            measured = gen.generate_threshold_breach_report(
                "demographic_parity_difference", 0.12, 0.10, OutputFormat.MARKDOWN
            )
        text = undefined.sections[0]["content"]
        assert "0.0% beyond threshold" not in text, text
        assert "not measurable" in text
        # OVER-CORRECTION CONTROL: a real threshold still reports its magnitude.
        assert "Breach Magnitude: 20.0% beyond threshold" in measured.sections[0]["content"]
