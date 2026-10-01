"""An empty collection from something that never ran is not a finding of "clean".

THE SHAPE, named in this library's own doctrine: "An empty list that means 'no
proxies found' when the scan never ran." Three units returned one in silence, and
two of them had documented the ambiguity at length in their DOCSTRINGS since
2026-09-09 and 2026-09-17:

    FairnessMonitor.get_alert_summary()        -> {}   on a monitor with no windows
    TemporalFairnessAnalyzer.get_metric_summary() -> {} for an untracked metric
    MetricsStore.get_alerts()                  -> []   on a store holding nothing

A caveat only a reader of the source can see is not a disclosure. `if not
monitor.get_alert_summary(): print("no fairness alerts")` reads as a clean bill
whether the monitor compared a thousand windows or none.

WHAT MADE THIS FIXABLE rather than merely documentable: the two cases really are
distinguishable. A monitor that never ran holds no history, an analyzer that never
tracked a metric has no column for it, and a store that was never fed holds no
records. The RETURN TYPE is unchanged in every case, so no caller breaks; what was
added is the disclosure at the call.

Every test below is paired with a control proving the honest case stays SILENT,
because a warning on every empty result would be the over-correction: "monitored
a thousand windows and nothing breached" is a real finding and must not be
drowned in a caveat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.monitoring.tracker import FairnessMonitor, TemporalFairnessAnalyzer
from vfairness.operations.reporting.store import MetricsStore


def _caught(fn, *a, **k):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = fn(*a, **k)
    return value, [str(w.message) for w in rec]


def _clean_batch(n: int = 400) -> pd.DataFrame:
    """A batch with no disparity at all: the honest "monitored and clean" case.

    FIXTURE CORRECTED, BGL5 A-operations-2, 2026-09-27, subject unchanged. This
    built its columns as ``gender`` / ``y_true`` / ``y_pred`` while the configured
    ``prediction_col`` is ``"prediction"`` and ``label_col`` is ``"label"``.
    Measured on the old fixture: ``update_and_check`` warned "the configured
    prediction column 'prediction' is NOT among the 3 column(s) ... so every
    fairness metric for this window is NaN and NO threshold comparison was made.
    This is could-not-check, not a clean window", metrics
    ``{'disparate_impact_gender': nan, 'demographic_parity_gender': nan}``,
    ``alerts {}``, ``any_alert None``. It was a compared-NOTHING window, so the
    control below was pinning the very silence its own grade calls a defect, and
    it FAILED any correct fix with "a real monitored-and-clean result was drowned
    in a could-not-check caveat".

    Now it is what its name says: both groups selected at exactly 0.5, so
    disparate impact is 1.0 and the parity gap 0.0, every metric is compared to
    its threshold and every comparison comes back clean. Measured after:
    metrics ``{'disparate_impact_group_gender': 1.0,
    'demographic_parity_group_gender': 0.0}``, alerts
    ``{'disparate_impact_group_gender': False,
    'demographic_parity_group_gender': False}``, ``any_alert False``, warnings
    ``[]``. Deterministic rather than seeded, because "no disparity at all" is an
    exact claim and a draw from an RNG is only approximately one.
    """
    half = n // 2
    per_group = [1] * (half // 2) + [0] * (half - half // 2)
    return pd.DataFrame(
        {
            "group_gender": ["F"] * half + ["M"] * (n - half),
            "label": (per_group + per_group)[:n],
            "prediction": (per_group + per_group)[:n],
        }
    )


# ── FairnessMonitor.get_alert_summary ─────────────────────────────────────────


def test_a_monitor_that_never_ran_says_so():
    summary, msgs = _caught(FairnessMonitor().get_alert_summary)
    assert summary == {}
    assert any("nothing was compared to a threshold" in m for m in msgs), (
        "an unmonitored summary read as a clean bill"
    )


def test_control_a_monitor_that_did_run_and_found_nothing_stays_silent():
    monitor = FairnessMonitor()
    snap, ingest_msgs = _caught(monitor.update_and_check, _clean_batch())
    # The fixture is asserted to BE a compared-and-clean window, because the
    # version of it this control shipped until 2026-09-27 was not one and nothing
    # here would have said so: see _clean_batch. A window that compared nothing
    # would satisfy the silence assertion below for the wrong reason.
    assert snap.metrics["disparate_impact_group_gender"] == 1.0, snap.metrics
    assert snap.metrics["demographic_parity_group_gender"] == 0.0, snap.metrics
    assert snap.alerts == {
        "disparate_impact_group_gender": False,
        "demographic_parity_group_gender": False,
    }, snap.alerts
    assert snap.any_alert is False
    assert ingest_msgs == [], ingest_msgs

    summary, msgs = _caught(monitor.get_alert_summary)
    assert not any("nothing was compared" in m for m in msgs), (
        "a real monitored-and-clean result was drowned in a could-not-check caveat"
    )
    assert isinstance(summary, dict)
    assert summary == {}


# ── TemporalFairnessAnalyzer.get_metric_summary ───────────────────────────────


def test_an_untracked_metric_is_could_not_check_not_zeros():
    summary, msgs = _caught(TemporalFairnessAnalyzer().get_metric_summary, "demographic_parity")
    assert summary == {}
    assert any("could not check" in m for m in msgs), (
        "an untracked metric answered in silence, and R-2 already shows a caller "
        "turning that into 'Mean = 0.0000, std = 0.0000'"
    )


def test_control_a_tracked_metric_still_reports_its_statistics():
    analyzer = TemporalFairnessAnalyzer()
    rng = np.random.default_rng(7)
    values = rng.normal(0.1, 0.02, 30)
    # Fed through the PUBLIC api. Hand-building `_daily_metrics` is how the first
    # version of this control failed: the frame it wrote had no `date` column and
    # detect_trend raised, so the test was measuring my fixture rather than the
    # analyzer.
    for i, v in enumerate(values):
        analyzer.update_daily_metrics(
            pd.Timestamp("2026-08-01") + pd.Timedelta(days=i), {"demographic_parity": float(v)}
        )
    summary, msgs = _caught(analyzer.get_metric_summary, "demographic_parity")
    assert summary and summary["n_days"] == 30
    assert summary["mean"] == pytest.approx(float(np.mean(values)))
    assert not any("could not check" in m for m in msgs)


# ── MetricsStore.get_alerts ───────────────────────────────────────────────────


def test_a_store_that_holds_nothing_says_so():
    alerts, msgs = _caught(MetricsStore().get_alerts)
    assert alerts == []
    assert any("nothing has been ingested" in m for m in msgs), (
        "an empty store reported 'no alert fired'"
    )


def test_control_a_store_with_records_does_not_caveat_its_empty_alert_list():
    store = MetricsStore()
    store._records.append({"timestamp": pd.Timestamp("2026-09-01"), "metric": "dp", "value": 0.02})
    alerts, msgs = _caught(store.get_alerts)
    assert alerts == []
    assert not any("nothing has been ingested" in m for m in msgs), (
        "a store that WAS fed and simply had no alerts was reported as unexamined"
    )
