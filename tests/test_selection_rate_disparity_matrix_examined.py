"""selection_rate_disparity_matrix, examined in its own right (2026-10-01).

It was the last public unit graded UNPROVEN: on 2026-09-30 the fix credited to it
had been made in a harness script, not in this function, so its own behaviour
had never been judged. Run here on healthy data checked against hand computation
and on every input where the headline cannot exist. Two sabotages were run: the
headline computed over every group (a 2-row group that selected nobody then
defines min_ratio 0.0) and the excluded-groups field removed; each turned this
file red.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    selection_rate_disparity_matrix as srdm,
)

rng = np.random.default_rng(0)
S = np.array(["a"] * 300 + ["b"] * 200 + ["c"] * 100)
P = np.r_[rng.random(300) < 0.6, rng.random(200) < 0.4, rng.random(100) < 0.3].astype(int)


def _run(p, s):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return srdm(p, s)


def test_healthy_rates_and_headline_match_hand_computation():
    r = _run(P, S)
    for g in "abc":
        assert r["rates"][g]["rate"] == pytest.approx(P[S == g].mean(), abs=1e-12)
    assert r["min_ratio"] == pytest.approx(P[S == "c"].mean() / P[S == "a"].mean(), abs=1e-12)
    assert r["min_ratio_pair"] == ("c", "a")
    assert r["max_difference"] == pytest.approx(P[S == "a"].mean() - P[S == "c"].mean(), abs=1e-12)
    assert r["headline_basis"] == "interpretable_groups"
    assert r["headline_excluded_groups"] == []


@pytest.mark.parametrize(
    "p,s",
    [
        (P, np.array(["a"] * 600)),
        (np.array([], dtype=int), np.array([], dtype=object)),
        (np.full(600, np.nan), S),
        (P, np.array([np.nan] * 600, dtype=object)),
    ],
    ids=["one group", "no rows", "every prediction missing", "every group label missing"],
)
def test_no_headline_where_no_pair_exists(p, s):
    r = _run(p, s)
    assert math.isnan(r["min_ratio"])
    assert r["min_ratio_pair"] is None
    assert r["headline_basis"] == "no_pair"


def test_nobody_selected_is_zero_over_zero_not_parity():
    r = _run(np.zeros(600, dtype=int), S)
    assert math.isnan(r["min_ratio"])  # never 1.0 ("no adverse impact")


def test_a_group_shut_out_entirely_is_the_real_worst_reading():
    r = _run(np.where(S == "c", 0, P), S)
    assert r["min_ratio"] == 0.0 and r["min_ratio_pair"] == ("c", "a")


def test_a_tiny_group_does_not_define_the_headline_and_is_named():
    p = np.r_[P[:598], [0, 0]]
    s = np.r_[S[:598], ["z", "z"]]
    r = _run(p, s)
    assert r["rates"]["z"]["interpretable"] is False
    assert r["min_ratio_pair"] == ("c", "a")
    assert r["min_ratio"] > 0.5
    assert r["headline_excluded_groups"] == ["z"]
