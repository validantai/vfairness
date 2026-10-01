"""counterfactual_fairness() metric test suite, Workstream F (CF-001)."""

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
    CounterfactualFairnessResult,
    counterfactual_fairness,
)


def test_perfectly_fair_no_change():
    f = np.array([0.1, 0.4, 0.6, 0.9])
    res = counterfactual_fairness(f, f.copy())
    assert isinstance(res, CounterfactualFairnessResult)
    assert res.flip_rate == 0.0
    assert res.mean_abs_diff == 0.0
    assert res.severity == "info"


def test_violation_flips_decisions():
    # Counterfactual pushes several scores across the 0.5 threshold.
    f = np.array([0.45, 0.48, 0.49, 0.52, 0.55])
    c = np.array([0.55, 0.58, 0.59, 0.42, 0.45])  # all 5 flip
    res = counterfactual_fairness(f, c, threshold=0.5)
    assert res.flip_rate == 1.0
    assert res.severity == "critical"
    assert res.max_abs_diff > 0.0


def test_label_inputs_threshold_none():
    f = np.array([1, 1, 0, 0])
    c = np.array([1, 0, 0, 1])  # 2 of 4 flip
    res = counterfactual_fairness(f, c, threshold=None)
    assert res.flip_rate == 0.5
    assert res.severity == "critical"


def test_small_change_below_threshold_is_low_or_info():
    f = np.array([0.10, 0.20, 0.30, 0.40, 0.80, 0.90, 0.10, 0.20, 0.30, 0.40])
    c = f + 0.01  # tiny, crosses nothing
    res = counterfactual_fairness(f, c, threshold=0.5)
    assert res.flip_rate == 0.0
    assert res.severity == "info"


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        counterfactual_fairness([0.1, 0.2], [0.1])


def test_empty_raises():
    with pytest.raises(ValueError):
        counterfactual_fairness([], [])


def test_to_dict_json_friendly():
    import json

    res = counterfactual_fairness([0.4, 0.6], [0.6, 0.4])
    d = res.to_dict()
    expected = {
        "n",
        "flip_rate",
        "mean_abs_diff",
        "max_abs_diff",
        "threshold",
        "severity",
        "interpretation",
        "notes",
    }
    assert set(d) == expected
    json.dumps(d)
