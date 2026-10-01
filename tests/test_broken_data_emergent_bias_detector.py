"""EmergentBiasDetector: a one-row group must not produce a significance verdict.

Found 2026-10-01 by the broken-data check: 399 rows against 1 came back
is_significant=True at p=0.0064. A single row has no spread to resample, so the
bootstrap understated the variability of that group's mean to zero.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.multi_agent.emergent import EmergentBiasDetector


def _run(groups, system, component):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return EmergentBiasDetector().analyze({"c": component}, system, groups)


def test_a_one_row_group_gives_no_significance_verdict():
    rng = np.random.default_rng(0)
    g = np.array(["a"] * 399 + ["b"])
    r = _run(g, np.r_[rng.random(399) * 0.2, 1.0], np.r_[rng.random(399) * 0.2, 0.2])
    assert r.is_significant is None
    assert math.isnan(r.p_value)


def test_two_real_groups_are_still_tested():
    rng = np.random.default_rng(0)
    g = np.array(["a"] * 200 + ["b"] * 200)
    system = np.r_[rng.random(200) * 0.2 + 0.6, rng.random(200) * 0.2]
    component = np.r_[rng.random(200) * 0.2 + 0.1, rng.random(200) * 0.2]
    r = _run(g, system, component)
    assert r.is_significant is True
    assert r.p_value < 0.05
    gap = abs(system[g == "a"].mean() - system[g == "b"].mean())
    assert r.system_bias == pytest.approx(gap, abs=1e-12)
