"""Shared fixtures.

status_with_every_state: the published status data with every state populated.

Added 2026-10-01, when the last NOT CHECKED unit was examined and FIX PENDING had
already been empty since 2026-09-27. Several tests proved that the status API and
the rendered badges keep the three states apart by borrowing a live example of
each state from the ledger. Once a state empties there is nothing to borrow, and
the right response is not to keep a unit unexamined so a test has an example: it
is to give those tests their own. This copies the shipped status payload, sets
real units to NOT CHECKED and FIX PENDING, recounts, and installs the copy as the
status module's cache for the duration of one test.
"""

from __future__ import annotations

import collections
import copy
import importlib

import pytest

INJECTED_STATES = {
    "vfairness.preprocessing.feature_engineering.transformers.finite_or_nan": "NOT CHECKED",
    "vfairness.xai.explainers.base.prediction_fn_available": "NOT CHECKED",
    "vfairness.operations.reporting.store.MetricsStore.window_now": "FIX PENDING",
}


@pytest.fixture
def status_with_every_state(monkeypatch):
    status_module = importlib.import_module("vfairness.status")
    status_module._cache = None
    data = copy.deepcopy(status_module._load())
    for unit, state in INJECTED_STATES.items():
        assert unit in data["states_by_unit"], f"the injected unit {unit} no longer exists"
        data["states_by_unit"][unit] = state
    data["counts"] = dict(collections.Counter(data["states_by_unit"].values()))
    monkeypatch.setattr(status_module, "_cache", data)
    yield data
    status_module._cache = None
