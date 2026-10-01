"""Opt-in result cache tests (VB-PERF-3).

The cache must be off by default, never change a computed result, and isolate
callers from the stored objects.
"""

import numpy as np

from vfairness import FairnessAnalyzer


def _fixture():
    rng = np.random.RandomState(0)
    n = 200
    half = n // 2
    sensitive = np.array(["A"] * half + ["B"] * half)
    y_true = rng.binomial(1, 0.5, n)
    y_pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.4, half)])
    return y_true, y_pred, sensitive


def test_cache_off_by_default():
    yt, yp, s = _fixture()
    an = FairnessAnalyzer(yt, yp, s, task_type="classification")
    an.compute_all_metrics()
    assert an._cache == {}  # nothing cached when disabled


def test_cache_stores_and_returns_equal_but_isolated():
    yt, yp, s = _fixture()
    an = FairnessAnalyzer(yt, yp, s, task_type="classification", cache=True)
    r1 = an.compute_all_metrics()
    assert len(an._cache) == 1  # stored on first call
    r2 = an.compute_all_metrics()
    assert r1 == r2
    assert r1 is not r2  # deep-copied, not the same object
    # Mutating a returned result must not corrupt the cache.
    r1["demographic_parity_difference"] = 999.0
    r3 = an.compute_all_metrics()
    assert r3["demographic_parity_difference"] != 999.0


def test_cached_result_matches_uncached():
    yt, yp, s = _fixture()
    cached = FairnessAnalyzer(
        yt, yp, s, task_type="classification", cache=True
    ).compute_all_metrics()
    fresh = FairnessAnalyzer(yt, yp, s, task_type="classification").compute_all_metrics()
    assert cached == fresh  # caching never changes the numbers


def test_clear_cache():
    yt, yp, s = _fixture()
    an = FairnessAnalyzer(yt, yp, s, task_type="classification", cache=True)
    an.compute_all_metrics()
    assert an._cache
    an.clear_cache()
    assert an._cache == {}


def test_get_report_is_cached():
    yt, yp, s = _fixture()
    an = FairnessAnalyzer(yt, yp, s, task_type="classification", cache=True)
    rep1 = an.get_report(random_state=0)
    assert any(k[0] == "get_report" for k in an._cache)
    rep2 = an.get_report(random_state=0)
    # Same structure and assessment, isolated objects.
    assert rep1["task_type"] == rep2["task_type"]
    assert list(rep1["metrics"]) == list(rep2["metrics"])
    assert rep1["assessment"] == rep2["assessment"]
    assert rep1 is not rep2
