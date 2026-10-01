"""The data-provenance clause must disclose EVERY exclusion, not just one kind.

docs/API_REFERENCE.md tells readers to trust this clause as the disclosure of
how rows were handled. It counted only rows dropped for MISSING VALUES. A group
removed by ``min_group_size`` keeps its rows inside ``final_size`` while
contributing to no metric at all, so the clause read

    "2302 of 2302 rows assessed, 0 excluded"

on the run where 1102 rows across 50 small groups had left every measurement.
That is the sentence a careful reader consults to decide whether a verdict
covers their population, and it stated the opposite of what happened.

Measured 2026-09-10 on a model that denied one protected class outright, with
that class spread over 50 levels of about 22 rows each, every one below the
default floor of 30.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness import FairnessAnalyzer
from vfairness.evaluation.vfairness_metrics.report import _missing_data_clause


def _denied_class_run():
    rng = np.random.default_rng(11)
    base = ["white"] * 1200 + ["black"] * 1102
    y_true = rng.integers(0, 2, len(base))
    y_pred = np.array([1 if (g == "white" and rng.random() < 0.5) else 0 for g in base])
    # object dtype on purpose: a numpy "<U5" array silently TRUNCATES
    # "black_0" back to "black", which quietly undoes the whole fixture.
    sub = np.array(
        [g if g == "white" else f"black_{i % 50}" for i, g in enumerate(base)],
        dtype=object,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(y_true=y_true, y_pred=y_pred, sensitive_attr=sub).get_report()


def _fully_sampled_run():
    rng = np.random.default_rng(11)
    base = ["white"] * 1200 + ["black"] * 1102
    y_true = rng.integers(0, 2, len(base))
    y_pred = np.array([1 if (g == "white" and rng.random() < 0.5) else 0 for g in base])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FairnessAnalyzer(
            y_true=y_true, y_pred=y_pred, sensitive_attr=np.array(base, dtype=object)
        ).get_report()


class TestTheClauseDisclosesTheSizeGate:
    def test_the_withheld_rows_are_stated(self):
        clause = _missing_data_clause(_denied_class_run()["data_info"])
        assert "1102 in 50 group(s) withheld by the group-size floor" in clause, clause

    def test_it_no_longer_reads_as_a_complete_assessment(self):
        """The exact sentence that made the defect invisible."""
        clause = _missing_data_clause(_denied_class_run()["data_info"])
        assert "2302 of 2302 rows assessed, 0 excluded," not in clause, clause

    def test_the_two_exclusion_kinds_are_named_apart(self):
        """A reader must be able to tell missing values from the size gate:
        they have different causes and different remedies."""
        clause = _missing_data_clause(_denied_class_run()["data_info"])
        assert "excluded for missing values" in clause, clause
        assert "withheld by the group-size floor" in clause, clause

    def test_the_fixture_actually_gates_groups(self):
        """ANTI-VACUITY. If the dtype truncation returned, or the floor moved,
        this fixture would gate nothing and every assertion above would pass for
        the wrong reason."""
        info = _denied_class_run()["data_info"]
        assert len(info["invalid_groups"]) == 50, info["invalid_groups"]
        assert info["valid_groups"] == ["white"], info["valid_groups"]

    def test_control_a_fully_sampled_run_reports_nothing_withheld(self):
        """Over-correction control: the clause must not imply an exclusion that
        did not happen, and the measured counts stay exactly as before."""
        clause = _missing_data_clause(_fully_sampled_run()["data_info"])
        assert "0 in 0 group(s) withheld by the group-size floor" in clause, clause
        assert "2302 of 2302 rows assessed" in clause, clause

    @pytest.mark.parametrize(
        "info",
        [
            {"final_size": 10, "original_size": 10, "n_excluded": 0, "missing_strategy": "exclude"},
            {
                "final_size": 10,
                "original_size": 10,
                "n_excluded": 0,
                "missing_strategy": "exclude",
                "invalid_groups": ["a"],
            },
        ],
        ids=["no-group-fields", "sizes-missing"],
    )
    def test_an_unrecorded_count_prints_unknown_not_zero(self, info):
        """A number nobody recorded is not a zero. The clause already held this
        rule for its other counts and the new one must too."""
        clause = _missing_data_clause(info)
        assert "unknown withheld by the group-size floor" in clause, clause
