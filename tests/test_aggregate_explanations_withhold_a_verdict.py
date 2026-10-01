"""The last three observed fabrications on the public surface.

"Observed" is the distinction that matters. Of the defects the grading waves
recorded, most are evidence gaps: nobody ever saw the unit lie, and the audit
overturned a PROVEN claim because the pins covered less input than the function
accepts. These three were different. Each was SEEN handing back a clean aggregate
over an analysis that assessed nothing, which is the one shape that reaches a user
as a false clean bill.

Examined 2026-09-25, one at a time, and they did not come out the same way:

  FeatureEngineeringAnalyzer.get_explanation   STILL REPRODUCED, and is fixed here.
      On a 2-row single-group frame the report was honest: `screens_not_run` listed
      three screens with the reason "only 2 non-null row(s), and the proxy screen
      needs min_sample_size=100". The explanation built from it read "Analysed 1
      features ... 0 proxy variable(s) and 0 high-risk feature(s)" at severity
      'info'. Nothing ran; the reader was handed a clean result. Fixed with the
      pattern the bias audit already uses (`detector._qualify_explanation`, BGL
      g021): the aggregate verdict is withheld and the severity raised off 'info',
      while findings, values and recommendations are untouched.

  CalibrationAnalyzer.get_explanation          DID NOT REPRODUCE. It already says
      "Calibration disparity COULD NOT CHECK: no significance verdict was measured
      for this report" and grades that item 'medium' rather than 'info'. The record
      was stale; nothing is changed.

  MetricsStore.compute_health_score            DID NOT REPRODUCE. An empty store
      returns score=None with a warning naming the state, rather than 100.

A stale defect record is not the same as a fixed defect, and the difference is a
test. All three are pinned here, each with a control on data where the aggregate IS
measurable, because withholding a verdict from a real clean result would be the
same defect running backwards.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting.store import MetricsStore
from vfairness.post_processing.calibration.analyzer import CalibrationAnalyzer
from vfairness.preprocessing.feature_engineering.analyzer import FeatureEngineeringAnalyzer


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        return fn(*a, **k), [str(w.message) for w in rec]


def _analyzer(df: pd.DataFrame) -> FeatureEngineeringAnalyzer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FeatureEngineeringAnalyzer(
            df, protected_attributes=["gender"], target_column="hired"
        )


TINY = pd.DataFrame({"gender": ["F", "F"], "income": [50000, 52000], "hired": [1, 0]})


def _populated(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "gender": np.where(rng.random(n) < 0.5, "F", "M"),
            "income": rng.normal(50000, 9000, n),
            "tenure": rng.normal(5, 2, n),
            "hired": (rng.random(n) < 0.5).astype(int),
        }
    )


# ── FeatureEngineeringAnalyzer.get_explanation: the one that reproduced ────────


def test_an_analysis_whose_screens_did_not_run_withholds_its_verdict():
    analyzer = _analyzer(TINY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)
    assert report.screens_not_run, "the premise is gone: this frame used to skip three screens"
    assert "COULD NOT CHECK" in explanation.summary, (
        f"an analysis that ran no screen summarised as {explanation.summary[:80]!r}"
    )
    assert explanation.severity != "info", (
        "an unmeasured analysis was graded 'info', the band a genuinely clean one gets"
    )


def test_control_a_populated_analysis_keeps_its_measured_verdict():
    """The over-correction control. A real clean result must not be caveated."""
    analyzer = _analyzer(_populated())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = analyzer.full_analysis()
        explanation = analyzer.get_explanation(report)
    assert not report.screens_not_run and not report.ungraded_proxy_screens
    assert "COULD NOT CHECK" not in explanation.summary, (
        "a fully screened analysis was told it could not check"
    )
    assert explanation.severity == "info"


# ── CalibrationAnalyzer.get_explanation: stale record, pinned ─────────────────


def _calibration(n: int, groups: np.ndarray) -> CalibrationAnalyzer:
    rng = np.random.default_rng(6)
    y = (rng.random(n) < 0.5).astype(int)
    prob = np.clip(y * 0.6 + rng.random(n) * 0.4, 0.01, 0.99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CalibrationAnalyzer(y, prob, groups)


def test_a_calibration_disparity_nobody_could_test_is_not_reported_as_absent():
    explanation = _calibration(2, np.array(["a", "a"])).get_explanation()
    assert "COULD NOT CHECK" in explanation.summary, (
        "one group and two rows produced a disparity verdict"
    )
    assert "No significant calibration disparity" not in explanation.summary


def test_control_a_real_two_group_calibration_states_its_disparity_verdict():
    explanation = _calibration(300, np.array(["a"] * 150 + ["b"] * 150)).get_explanation()
    assert "COULD NOT CHECK" not in explanation.summary, (
        "a 300-row two-group comparison was withheld"
    )
    assert "calibration disparity" in explanation.summary.lower()


# ── MetricsStore.compute_health_score: stale record, pinned ───────────────────


def test_a_health_score_over_nothing_is_withheld_not_perfect():
    score, msgs = _caught(MetricsStore().compute_health_score)
    assert getattr(score, "score", "missing") is None, (
        f"an empty store scored {getattr(score, 'score', None)!r} for fairness health"
    )
    assert getattr(score, "components", None) in ({}, None)
    assert any("no health score was computed" in m for m in msgs), (
        "an empty store scored in silence"
    )


def _store_with(alerts: int, n: int = 12) -> MetricsStore:
    """A store fed through the PUBLIC ingest api, with a real threshold comparison.

    Two things had to be learned the hard way here, and both are findings about
    the store rather than about the test. Hand-appending to `_records` writes rows
    that compute_health_score does not read, so the first version of this control
    was measuring my fixture. And a row carrying a value but no ALERT DETERMINATION
    is still "never compared to a threshold": the store refuses those, which is
    narrower and more honest than the defect record suggested.
    """
    base = pd.Timestamp.now()  # naive: a tz-aware column makes the window filter raise
    frame = pd.DataFrame(
        {
            "timestamp": [base - pd.Timedelta(days=i % 5) for i in range(n)],
            "metric": ["demographic_parity"] * n,
            "value": [0.02] * n,
            "alert": [False] * (n - alerts) + [True] * alerts,
            "determined": [True] * n,
        }
    )
    store = MetricsStore()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(frame, alert_col="alert", alert_determined_col="determined")
    return store


def test_control_a_store_with_records_computes_a_health_score():
    """The over-correction control, and it asserts the score RESPONDS.

    A refusal that also refused measurable data would pass the test above, and a
    health score frozen at a constant would pass a bare "is not None". So this
    asserts both that a fully compared store scores, and that alerting records
    push the score DOWN.
    """
    clean, _msgs = _caught(_store_with(alerts=0).compute_health_score)
    assert clean.score is not None, (
        "a store whose records were all compared to a threshold still refused to "
        "score, which is the refusal over-correcting onto measurable data"
    )
    assert clean.score == pytest.approx(100.0)

    alerting, _msgs2 = _caught(_store_with(alerts=4).compute_health_score)
    assert alerting.score is not None
    assert alerting.score < clean.score, (
        f"four alerting records out of twelve scored {alerting.score}, no worse than "
        f"the clean {clean.score}: the score is not reading its input"
    )
