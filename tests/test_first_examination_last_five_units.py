"""First examination of the last five public units no test had ever run.

Beta condition B1 ("every public code unit has been executed and its result
recorded") stood at 1,575 of 1,580 on 2026-10-01. All five were written during
the 2026-09-30 fixes and were reached only through their callers. Each is run
here directly, on the input it exists for and on a healthy control, and each
test was sabotaged (the fix reverted) and seen to fail before the grade was
recorded in docs/surface-grading-2026-10-01.json.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from vfairness.operations.reporting.store import MetricsStore
from vfairness.post_processing.threshold_optimization.constraints import FeasibleThresholds
from vfairness.preprocessing.feature_engineering.transformers import finite_or_nan
from vfairness.xai.explainers.base import prediction_fn_available


# MetricsStore.window_now: the window bound must share the records' tz-awareness,
# or every reader above compute_health_score raises a pandas TypeError.
def test_window_now_is_aware_for_an_aware_store_and_naive_for_a_naive_one():
    store = MetricsStore()
    store._records = [SimpleNamespace(timestamp=datetime(2026, 9, 30, tzinfo=timezone.utc))]
    aware = store.window_now()
    assert aware.tzinfo is not None
    assert abs((aware - datetime.now(timezone.utc)).total_seconds()) < 5

    store._records = [SimpleNamespace(timestamp=datetime(2026, 9, 30))]
    naive = store.window_now()
    assert naive.tzinfo is None
    assert abs((naive - datetime.now()).total_seconds()) < 5


# FeasibleThresholds.get / setdefault: a group that was NOT assessed must not come
# back as [], which reads as "measured, and no threshold works".
def _ft():
    return FeasibleThresholds({"a": [0.1, 0.2]}, not_assessable={"b": "b has no negatives"})


def test_get_refuses_a_not_assessed_group_and_returns_a_measured_one():
    ft = _ft()
    assert ft.get("a") == [0.1, 0.2]
    assert ft.get("zz", "fallback") == "fallback"  # an unknown key is an ordinary miss
    with pytest.raises(KeyError, match="NOT assessed"):
        ft.get("b", [])


def test_setdefault_will_not_insert_a_fabrication():
    ft = _ft()
    with pytest.raises(KeyError, match="NOT assessed"):
        ft.setdefault("b", [])
    assert "b" not in ft
    assert ft.setdefault("a", []) == [0.1, 0.2]
    assert ft.setdefault("c", [0.5]) == [0.5]


# finite_or_nan: a real finite number as float, everything else NaN.
@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), -np.inf, True, np.bool_(False), "abc", [1]]
)
def test_finite_or_nan_refuses_what_is_not_a_finite_number(value):
    assert math.isnan(finite_or_nan(value))


@pytest.mark.parametrize(
    "value,want", [(0.3, 0.3), (0, 0.0), (np.float32(0.25), 0.25), ("0.5", 0.5), (np.int64(7), 7.0)]
)
def test_finite_or_nan_keeps_real_numbers(value, want):
    assert finite_or_nan(value) == pytest.approx(want)


# prediction_fn_available: an unfitted estimator HAS .predict and cannot predict.
def test_prediction_fn_available_refuses_an_unfitted_model_and_accepts_a_fitted_one():
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    assert prediction_fn_available(LogisticRegression(), where="test") is False
    assert prediction_fn_available(LogisticRegression().fit(X, y), where="test") is True
    assert prediction_fn_available(lambda x: x, where="test") is True
