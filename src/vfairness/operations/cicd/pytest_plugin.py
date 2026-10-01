"""
pytest plugin for vfairness fairness testing.

Provides markers, fixtures, and terminal reporting for fairness tests
in CI/CD pipelines.  Registered as a pytest11 entry point so that
``pip install vfairness`` automatically enables the plugin.

Markers:
    @pytest.mark.fairness: tag a test as a fairness test
    @pytest.mark.fairness_gate: tag a test as a gate-level check

Fixtures:
    fairness_gate: provides a pre-configured ModelFairnessGate
    assert_fairness_fn: provides the assert_fairness helper

Usage::

    @pytest.mark.fairness
    def test_demographic_parity(assert_fairness_fn):
        assert_fairness_fn(
            y_true, y_pred, gender,
            metrics=['demographic_parity_difference'],
            thresholds={'demographic_parity_difference': 0.1},
        )
"""

# The implementation moved to the top-level module `_vfairness_pytest_plugin`
# (src/_vfairness_pytest_plugin.py) on 2026-08-27. CRITICAL, DO NOT MOVE IT BACK.
# The pytest11 entry point in pyproject.toml points there, not here, because
# reaching this module makes Python execute vfairness/__init__.py first, which
# imports numpy/pandas/scipy/sklearn and ~1500 modules into EVERY pytest session
# of EVERY project that has vfairness installed (measured 3.2s vs 0.35s on a
# one-assert throwaway project). See that module's docstring for the numbers.
#
# This module stays as a re-export so the documented import path and
# `-p vfairness.operations.cicd.pytest_plugin` keep working unchanged. Anything
# imported here is already paying the package-import cost, so the re-export at
# module scope costs nothing extra.
from _vfairness_pytest_plugin import (  # noqa: F401
    assert_fairness_fn,
    fairness_gate,
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_terminal_summary,
)

__all__ = [
    "pytest_configure",
    "pytest_collection_modifyitems",
    "pytest_terminal_summary",
    "fairness_gate",
    "assert_fairness_fn",
]
