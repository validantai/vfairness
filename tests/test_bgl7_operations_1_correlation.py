"""BGL7 PIN, batch A-operations-1-w2 (2026-09-29): reporting/dashboard.py.

``FairnessDashboard.create_technical_view`` had its alert-evidence and
drift-coverage halves proved, and its (3,2) Metric Correlation panel published a
Pearson r of exactly -1.0000 computed from TWO observations with no sample-size
statement anywhere on the figure.

THE ROW COUNT IS THE SIBLING OF THE COLUMN COUNT, and only the column count was
guarded: ``if pivot.shape[1] > 1`` asks how many METRICS there are, never how many
OBSERVATIONS the coefficient rests on. Any two points are perfectly collinear, so
|r| is exactly 1 for every 2-row overlap whatever the numbers are.

MEASURED BEFORE THIS CHANGE (canonical interpreter, OMP_NUM_THREADS=1) on a store
holding two metrics at TWO timestamps, ingested with ``group_size_col`` so the
privacy layer releases the values, ``create_technical_view()``::

    heatmaps = 1
    z        = [[1.0, -1.0], [-1.0, 1.0]]
    labels   = ['dp_diff', 'eo_diff']
    annotations mentioning correlation / points / observations:
        ['Metric Correlation']      <- the subplot title, and nothing else

so a perfect anticorrelation was painted at the extreme of a colorscale pinned
zmin=-1 / zmax=1, from two points. n=3 gave the same +-1.0000.

Read with tests/test_bgl5_operations_1.py and tests/test_bgl4_operations_1.py,
which pinned the other two panels of this same figure.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from vfairness.operations.reporting.dashboard import FairnessDashboard
from vfairness.operations.reporting.store import MetricsStore, MetricsStoreConfig

pytest.importorskip("plotly")

NOW = datetime.now()


def _store(n_observations: int, *, noisy: bool = False, thin_third: int = 0) -> MetricsStore:
    """Two metrics observed at *n_observations* distinct timestamps.

    ``group_size_col`` is passed so the privacy layer releases the values, which is
    how the auditor reached the panel.
    """
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_observations):
        ts = NOW - timedelta(hours=n_observations - i)
        rows.append({"timestamp": ts, "metric": "dp_diff", "value": 0.10 + 0.01 * i, "n": 500})
        rows.append(
            {
                "timestamp": ts,
                "metric": "eo_diff",
                "value": 0.50 - 0.01 * i + (float(rng.normal(0, 0.02)) if noisy else 0.0),
                "n": 500,
            }
        )
        if i < thin_third:
            rows.append({"timestamp": ts, "metric": "spd", "value": 0.30 + 0.02 * i, "n": 500})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(pd.DataFrame(rows), group_size_col="n")
    return store


def _view(store: MetricsStore):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessDashboard(store).create_technical_view()


def _heatmaps(fig):
    return [trace for trace in fig.data if trace.type == "heatmap"]


def _notes(fig):
    return [a.text for a in fig.layout.annotations]


@pytest.mark.parametrize("n_observations", [2, 3, 9], ids=["two", "three", "nine"])
def test_a_correlation_over_too_few_observations_is_not_painted(n_observations):
    """BEFORE: one heatmap holding +-1.0000, and the only annotation mentioning
    correlation was the subplot title. A matrix is the strongest claim this figure
    can make, so below the floor it is withheld entirely rather than painted at the
    extreme of a pinned colorscale."""
    fig = _view(_store(n_observations))

    assert _heatmaps(fig) == [], "a correlation matrix was painted over too few points"
    withheld = [t for t in _notes(fig) if "NOT COMPUTED" in t]
    assert len(withheld) == 1, _notes(fig)
    assert f"Only {n_observations} observation(s)" in withheld[0], withheld[0]
    assert "could-not-check" in withheld[0]
    # The reason, not just the refusal: a reader has to know why.
    assert "perfectly collinear" in withheld[0]


def test_the_withheld_panel_is_distinguishable_from_an_empty_window():
    """A withheld panel and a panel nobody had data for look identical on a grid of
    six subplots, which is why the note is an annotation and not only a warning."""
    fig = _view(_store(2))
    assert any("Metric Correlation: NOT COMPUTED" in t for t in _notes(fig)), _notes(fig)


def test_control_a_correlation_over_enough_observations_is_still_painted():
    """OVER-CORRECTION CONTROL, asserting the healthy case's REAL number.

    Thirty observations of a genuinely noisy relation: the matrix is painted and
    its off-diagonal is -0.9848, the measured value, not a constructed -1. A guard
    that withheld the panel whenever any coefficient was extreme would pass the
    assertions above and delete the panel.
    """
    fig = _view(_store(30, noisy=True))
    heat = _heatmaps(fig)

    assert len(heat) == 1, heat
    z = np.asarray(heat[0].z, dtype=float)
    assert list(heat[0].x) == ["dp_diff", "eo_diff"]
    assert z[0][0] == pytest.approx(1.0)
    assert round(float(z[0][1]), 4) == -0.9848, z
    # The sample size travels WITH the matrix: colour alone cannot tell 30 from 2.
    sized = [t for t in _notes(fig) if "observation(s)" in t]
    assert sized == ["Metric Correlation over 30 observation(s)."], sized


def test_control_exactly_at_the_floor_is_measured():
    """The floor is the one this package already uses for a Pearson coefficient
    (``feature_engineering.correlation.MIN_SAMPLE_SIZE``), and at exactly that many
    observations the panel IS painted. An off-by-one here would silently withhold a
    measurement the rest of the library grades."""
    from vfairness.preprocessing.feature_engineering.correlation import MIN_SAMPLE_SIZE

    at_floor = _view(_store(MIN_SAMPLE_SIZE))
    assert len(_heatmaps(at_floor)) == 1
    assert any(f"over {MIN_SAMPLE_SIZE} observation(s)" in t for t in _notes(at_floor))

    below = _view(_store(MIN_SAMPLE_SIZE - 1))
    assert _heatmaps(below) == []


def test_a_pair_with_too_little_overlap_is_blank_and_counted():
    """The per-PAIR sibling of the per-FRAME row count. ``pivot.corr()`` computes
    pairwise, so a third metric observed at only 4 of the 20 timestamps would
    otherwise borrow the frame's row count. ``min_periods`` leaves it NaN, which
    plotly renders blank, and blank is counted and named: blank is
    could-not-check, not zero correlation."""
    fig = _view(_store(20, noisy=True, thin_third=4))
    heat = _heatmaps(fig)

    assert len(heat) == 1
    assert list(heat[0].x) == ["dp_diff", "eo_diff", "spd"]
    z = np.asarray(heat[0].z, dtype=float)
    assert np.isnan(z[0][2]) and np.isnan(z[1][2]), z
    assert not np.isnan(z[0][1]), z

    sized = [t for t in _notes(fig) if "observation(s)" in t]
    assert len(sized) == 1, sized
    assert "2 of 3 metric pair(s) are BLANK" in sized[0], sized[0]
    assert "not zero correlation" in sized[0]


def test_the_blank_pair_count_is_one_cell_per_pair_not_half_the_nans():
    """A metric that does NOT VARY has a NaN on the DIAGONAL too, and that cell is
    not a pair. Counting every NaN and halving it happens to be right for one thin
    metric and wrong as soon as a constant one appears, so the count reads the upper
    triangle only. Four metrics = 6 pairs; a constant metric blanks 3 of them and
    the thin one blanks 3 more, sharing one: 5."""
    store = _store(20, noisy=True, thin_third=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(
            pd.DataFrame(
                [
                    {
                        "timestamp": NOW - timedelta(hours=20 - i),
                        "metric": "flat",
                        "value": 0.25,
                        "n": 500,
                    }
                    for i in range(20)
                ]
            ),
            group_size_col="n",
        )
    fig = _view(store)
    sized = [t for t in _notes(fig) if "observation(s)" in t]
    assert "5 of 6 metric pair(s) are BLANK" in sized[0], sized[0]


def test_control_a_single_metric_still_paints_nothing_and_says_nothing():
    """The column guard is untouched: with one metric there is no pair to correlate
    and the panel has nothing to disclose, so it must NOT start emitting a
    could-not-check note for a question nobody asked."""
    store = MetricsStore(MetricsStoreConfig(enable_privacy=False))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        store.ingest_dataframe(
            pd.DataFrame(
                [
                    {
                        "timestamp": NOW - timedelta(hours=20 - i),
                        "metric": "dp_diff",
                        "value": 0.10 + 0.01 * i,
                        "n": 500,
                    }
                    for i in range(20)
                ]
            ),
            group_size_col="n",
        )
    fig = _view(store)
    assert _heatmaps(fig) == []
    assert [t for t in _notes(fig) if "NOT COMPUTED" in t or "observation(s)" in t] == []
