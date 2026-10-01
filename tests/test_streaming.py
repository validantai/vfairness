"""Chunked/streaming computation tests (VB-PERF-2).

The streamed result must be identical regardless of chunk size, and must match
computing over the whole array at once and the standard demographic-parity API.
"""

import numpy as np
import pytest

import vfairness
from vfairness.streaming import (
    stream_group_counts,
    streaming_demographic_parity,
    streaming_selection_rates,
)


def _data(n=5000, seed=0):
    rng = np.random.RandomState(seed)
    half = n // 2
    sensitive = np.array(["A"] * half + ["B"] * half)
    y_pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.35, half)])
    return y_pred, sensitive


def _whole_counts(y_pred, sensitive):
    out = {}
    for g in np.unique(sensitive):
        mask = sensitive == g
        out[g] = (int((y_pred[mask] == 1).sum()), int(mask.sum()))
    return out


@pytest.mark.parametrize("chunk_size", [1, 7, 100, 999, 5000, 10_000])
def test_counts_identical_across_chunk_sizes(chunk_size):
    y_pred, sensitive = _data()
    streamed = stream_group_counts(y_pred, sensitive, chunk_size=chunk_size)
    assert streamed == _whole_counts(y_pred, sensitive)


def test_selection_rates_match_direct():
    y_pred, sensitive = _data()
    rates = streaming_selection_rates(y_pred, sensitive, chunk_size=333)
    whole = _whole_counts(y_pred, sensitive)
    for g, (pos, tot) in whole.items():
        assert rates[g] == pytest.approx(pos / tot)


def test_streaming_dp_matches_standard_api():
    y_pred, sensitive = _data()
    y_true = np.zeros_like(y_pred)  # DP does not use y_true
    standard = vfairness.demographic_parity_difference(y_true, y_pred, sensitive)
    streamed = streaming_demographic_parity(y_pred, sensitive, chunk_size=512)
    assert streamed == pytest.approx(standard, abs=1e-12)


def test_streaming_dp_stable_across_chunk_sizes():
    y_pred, sensitive = _data(seed=3)
    vals = {
        streaming_demographic_parity(y_pred, sensitive, chunk_size=cs)
        for cs in (1, 50, 1000, 100_000)
    }
    assert len(vals) == 1  # identical regardless of chunking


def test_nan_predictions_excluded_matching_standard_api():
    # NaN preds must be dropped (not counted as negatives), matching the
    # whole-array missing_strategy='exclude' default, across chunk sizes.
    rng = np.random.RandomState(5)
    n = 600
    half = n // 2
    sensitive = np.array(["A"] * half + ["B"] * half)
    y_pred = np.concatenate([rng.binomial(1, 0.6, half), rng.binomial(1, 0.4, half)]).astype(float)
    y_pred[::7] = np.nan  # scatter missing predictions
    y_true = np.zeros(n)
    standard = vfairness.demographic_parity_difference(y_true, y_pred, sensitive)
    for cs in (1, 64, 100_000):
        assert streaming_demographic_parity(y_pred, sensitive, chunk_size=cs) == pytest.approx(
            standard, abs=1e-12
        )


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_missing_sensitive_values_excluded(chunk_size):
    # Rows with a missing sensitive value are dropped from both numerator and
    # denominator (never grouped under a None/NaN key), and np.unique over a mix
    # of None/NaN and strings must not crash -- regardless of where a missing
    # value falls relative to the chunk boundary.
    y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    sensitive = np.array(["A", "A", None, "B", "B", np.nan, "A", "B"], dtype=object)
    counts = stream_group_counts(y_pred, sensitive, chunk_size=chunk_size)
    assert set(counts) == {"A", "B"}
    assert counts["A"] == (1, 3)  # rows 0,1,6 -> preds 1,0,0 (None row 2 dropped)
    assert counts["B"] == (2, 3)  # rows 3,4,7 -> preds 1,0,1 (nan row 5 dropped)


def test_non_binary_predictions_raise():
    y_pred = np.array([0, 1, 2, 1, 0, 1])
    sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    with pytest.raises(vfairness.InvalidDataError):
        stream_group_counts(y_pred, sensitive)


def test_small_groups_excluded_and_bad_chunk_size_raises():
    """This line USED TO READ `== 0.0`, and that assertion was the bug.

    With B below the size gate only one group survives, so no between-group
    comparison happens at all. Returning 0.0 there is the forbidden sentinel:
    indistinguishable downstream from a measured perfect parity, so it certifies
    FAIR exactly where nothing was measured. The CONVENTION block in
    evaluation/vfairness_metrics/classification.py spells this out and ends "do
    not restore a 0.0 / 1.0 return at any of them" - streaming.py was written
    before it and this test held the old behaviour in place.
    """
    y_pred = np.array([1, 0, 1])
    sensitive = np.array(["A", "A", "B"])  # B has 1 sample, below default gate
    with pytest.warns(UserWarning, match="no between-group comparison"):
        assert np.isnan(streaming_demographic_parity(y_pred, sensitive))
    with pytest.raises(vfairness.InvalidDataError):
        stream_group_counts(y_pred, sensitive, chunk_size=0)


def test_two_eligible_groups_still_measure_a_real_gap():
    """OVER-CORRECTION CONTROL. The fix must not make every streamed answer NaN:
    where the comparison genuinely happens, the number must still come back, and
    still equal the whole-array result."""
    y_pred = np.r_[np.ones(50, int), np.zeros(50, int)]
    sensitive = np.r_[np.array(["A"] * 50), np.array(["B"] * 50)]
    assert streaming_demographic_parity(y_pred, sensitive) == 1.0
