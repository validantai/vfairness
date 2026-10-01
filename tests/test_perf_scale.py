"""Performance / scale stress tests (VB-PERF-1).

These exercise the core metric functions and the ``FairnessAnalyzer`` on large
synthetic datasets to confirm they complete without error and stay within a
generous wall-clock bound. They are NOT precision benchmarks: the time bounds
are deliberately loose so the suite does not flake on a busy CI runner. The
point is to catch accidental quadratic blowups or memory explosions, not to
pin a millisecond budget.

Every test is marked ``slow`` so a plain ``pytest -m "not slow"`` skips them
and fast local runs stay quick. The marker is registered in
``pyproject.toml`` under ``[tool.pytest.ini_options] markers``.

Peak memory is captured with the stdlib ``tracemalloc`` (no third-party
dependency) and only asserted against a very loose ceiling, so the number is
informative in the captured output without making the test brittle.
"""

import time
import tracemalloc

import numpy as np
import pytest

import vfairness as vf
from vfairness import FairnessAnalyzer

# Generous ceilings. On a modern laptop the tall case runs in well under a
# second; CI runners are slower and shared, so we allow a large margin.
TALL_ROWS = 100_000
WIDE_ROWS = 5_000
WIDE_COLS = 15
TIME_BUDGET_SECONDS = 90.0
# Peak allocation ceiling for the tall case. Three int/float arrays of 100k
# rows plus intermediate group masks are a few MB; 1 GiB is a very loose guard
# that still catches a runaway allocation.
MEMORY_CEILING_BYTES = 1_024 * 1_024 * 1_024


def _make_tall_dataset(n_rows: int, n_groups: int = 4, seed: int = 12345):
    """Binary-classification arrays with a multi-value protected attribute."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n_rows) < 0.4).astype(int)
    y_pred = (rng.random(n_rows) < 0.4).astype(int)
    groups = rng.integers(0, n_groups, size=n_rows).astype(str)
    return y_true, y_pred, groups


def _make_wide_dataset(n_rows: int, n_cols: int, seed: int = 999):
    """A wide feature frame plus label/prediction/protected columns.

    The metrics only consume one protected column at a time, so ``n_cols``
    extra feature columns mainly stress construction and slicing rather than
    the metric math. This mirrors a realistic wide-table audit input.
    """
    rng = np.random.default_rng(seed)
    features = rng.random((n_rows, n_cols))
    y_true = (rng.random(n_rows) < 0.5).astype(int)
    y_pred = (rng.random(n_rows) < 0.5).astype(int)
    groups = rng.choice(["A", "B", "C"], size=n_rows)
    return features, y_true, y_pred, groups


@pytest.mark.slow
def test_core_metrics_scale_tall():
    """Core classification metrics finish on 100k rows within budget."""
    y_true, y_pred, groups = _make_tall_dataset(TALL_ROWS)

    tracemalloc.start()
    start = time.perf_counter()

    dp = vf.demographic_parity_difference(y_true, y_pred, groups)
    eo = vf.equalized_odds_difference(y_true, y_pred, groups)
    eop = vf.equal_opportunity_difference(y_true, y_pred, groups)
    dpr = vf.demographic_parity_ratio(y_true, y_pred, groups)

    elapsed = time.perf_counter() - start
    _peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # All metrics returned real numbers in a sane range.
    for value in (dp, eo, eop):
        assert isinstance(value, float)
        assert -1.0 <= value <= 1.0
        assert not np.isnan(value)
    assert isinstance(dpr, float)
    assert dpr >= 0.0

    assert elapsed < TIME_BUDGET_SECONDS, (
        f"core metrics on {TALL_ROWS} rows took {elapsed:.2f}s (budget {TIME_BUDGET_SECONDS}s)"
    )
    assert _peak < MEMORY_CEILING_BYTES, f"peak memory {_peak / 1e6:.1f} MB exceeded ceiling"


@pytest.mark.slow
def test_analyzer_report_scale_tall():
    """FairnessAnalyzer.get_report finishes on 100k rows within budget."""
    y_true, y_pred, groups = _make_tall_dataset(TALL_ROWS)

    tracemalloc.start()
    start = time.perf_counter()

    analyzer = FairnessAnalyzer(y_true, y_pred, groups)
    # include_ci=False keeps this a scale test rather than a bootstrap test:
    # bootstrap resampling has its own (much heavier) cost profile.
    report = analyzer.get_report(include_ci=False)

    elapsed = time.perf_counter() - start
    _peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert isinstance(report, dict)
    assert report["task_type"] == "classification"
    assert "metrics" in report and report["metrics"]
    assert report["data_info"]["n_samples"] == TALL_ROWS

    assert elapsed < TIME_BUDGET_SECONDS, (
        f"analyzer report on {TALL_ROWS} rows took {elapsed:.2f}s (budget {TIME_BUDGET_SECONDS}s)"
    )
    assert _peak < MEMORY_CEILING_BYTES, f"peak memory {_peak / 1e6:.1f} MB exceeded ceiling"


@pytest.mark.slow
def test_metrics_scale_wide():
    """Metrics run against one protected column of a wide (15-col) frame."""
    features, y_true, y_pred, groups = _make_wide_dataset(WIDE_ROWS, WIDE_COLS)
    # Sanity: the wide frame really is wide.
    assert features.shape == (WIDE_ROWS, WIDE_COLS)

    start = time.perf_counter()
    analyzer = FairnessAnalyzer(y_true, y_pred, groups)
    results = analyzer.compute_all_metrics(include_ci=False)
    elapsed = time.perf_counter() - start

    assert isinstance(results, dict)
    assert "demographic_parity_difference" in results
    assert elapsed < TIME_BUDGET_SECONDS, (
        f"wide-frame metrics took {elapsed:.2f}s (budget {TIME_BUDGET_SECONDS}s)"
    )


@pytest.mark.slow
def test_regression_metrics_scale_tall():
    """Regression metrics finish on 100k rows within budget."""
    rng = np.random.default_rng(7)
    y_true = rng.normal(100.0, 20.0, TALL_ROWS)
    y_pred = y_true + rng.normal(0.0, 10.0, TALL_ROWS)
    groups = rng.choice(["X", "Y"], size=TALL_ROWS)

    start = time.perf_counter()
    mae = vf.mae_parity_difference(y_true, y_pred, groups)
    rmse = vf.rmse_parity_difference(y_true, y_pred, groups)
    mean_diff = vf.mean_prediction_difference(y_true, y_pred, groups)
    elapsed = time.perf_counter() - start

    for value in (mae, rmse, mean_diff):
        assert isinstance(value, float)
        assert not np.isnan(value)
    assert mae >= 0.0
    assert rmse >= 0.0

    assert elapsed < TIME_BUDGET_SECONDS, (
        f"regression metrics on {TALL_ROWS} rows took {elapsed:.2f}s "
        f"(budget {TIME_BUDGET_SECONDS}s)"
    )
