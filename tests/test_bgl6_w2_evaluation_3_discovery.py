"""BGL5 audit wave 2, batch A-evaluation-3: units OVERTURNED to DEFECT OPEN.

Each unit in this batch had been graded PROVEN with a pin and a sabotage behind
it. An auditor reproduced every one of them anyway, because in each case the pin
proved the branch it was written for while the SIBLING of that branch fabricated:
the other argument of the same signature, the other reason a group can leave a
comparison, the other metric in the menu, the array the unit COMPUTED rather than
the one it was handed.

WHAT WAS MEASURED, 2026-09-29, on HEAD before the fixes in this wave
-------------------------------------------------------------------
4. ``classify_column_roles``. An absent ``declared_protected`` is named on three
   channels and an absent ``declared_target`` sets ``refuse=True``; an absent
   ``declared_prediction`` got NOTHING: ``model_output []``,
   ``not_assessable []``, ``mismatches []``, ``refuse False``, no warning, while
   three Pulse steps read ``schema["model_output"]`` as a finding about the data.
   Writing the control for it found the other half: a PRESENT declared_prediction
   named 'y_hat' was classified as the ORACLE, because the oracle name heuristic
   ran first and the token 'y' matched, so an explicit declaration lost to a
   one-character pattern and model_output was empty again.
5. ``rank_fairness_issues``. With ``y_true=None`` the scan narrows from four
   registered metrics to one and said so nowhere. On 2000 rows whose selection
   rates are identical at 0.500 while the model is perfect for one group and
   inverted for the other (a real error-rate disparity of 1.0) it published
   ``max_disparity 0.0, n_violations 0`` and "No significant fairness violations
   detected. Continue monitoring."; the SAME data with the real labels gives
   ``n_violations 3, max_disparity 1.0``.

The statistics half of the same batch is in
tests/test_bgl6_w2_evaluation_3.py.

Every pin below is paired with an OVER-CORRECTION CONTROL that asserts the
healthy case's REAL number, so "refuse everything" cannot pass this file.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics.discovery import (
    classify_column_roles,
    rank_fairness_issues,
    scan_fairness_violations,
)

ALL_CLEAR = "No significant fairness violations detected. Continue monitoring."


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(c.message) for c in caught]


# ---------------------------------------------------------------------------
# 4. classify_column_roles
# ---------------------------------------------------------------------------


def _small_frame(extra=None):
    rng = np.random.default_rng(0)
    data = {"gender": rng.choice(["M", "F"], 400), "salary": rng.normal(50000, 9000, 400)}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def test_classify_column_roles_names_an_absent_declared_prediction():
    """model_output [] for a declaration that names nothing, on every channel.

    BEFORE (measured on HEAD, 400 rows holding only `gender` and `salary`,
    classify_column_roles(df, declared_protected=['gender'],
    declared_prediction='y_hat')):
        model_output [], n_not_assessable 0, not_assessable [], mismatches [],
        refuse False, warnings []
    A misspelt prediction column was indistinguishable from a frame that
    genuinely holds no model output, and that empty list is read at
    operations/pulse/orchestrator.py:4640 (the leakage-exclusion set),
    operations/pulse/regulatory.py:491 (the governance narrative) and
    operations/pulse/causal_skeleton.py:352 (the causal decision node). The
    sibling arguments of the same signature already answered: an absent
    declared_protected on three channels, an absent declared_target with
    refuse=True.
    """
    df = _small_frame()

    schema, caught = _caught(
        classify_column_roles, df, declared_protected=["gender"], declared_prediction="y_hat"
    )

    assert schema["model_output"] == []
    assert schema["n_not_assessable"] == 1
    assert "y_hat" in schema["not_assessable"][0]
    assert "declared as the model output" in schema["not_assessable"][0]
    assert [m for m in schema["mismatches"] if m["column"] == "y_hat"] == [
        {
            "column": "y_hat",
            "declared": "model_output",
            "inferred": "absent",
            "detail": schema["mismatches"][0]["detail"],
        }
    ]
    assert [c for c in caught if "declared model output 'y_hat' is NOT a column" in c], caught
    # NOT a refuse: one misspelt output column must not abort an analysis whose
    # data checks are entirely measurable. `declared_target` is the one that
    # refuses, because without the answer key nothing downstream is measurable.
    assert schema["refuse"] is False


def test_classify_column_roles_still_classifies_a_present_prediction():
    """OVER-CORRECTION CONTROL, and a PIN: the declaration outranks the name.

    A declaration that names a real column is honoured, silently and completely:
    the column IS the model output, nothing is unassessable about it and nothing
    warns.

    Writing this control found the other half of the same defect. BEFORE
    (measured on HEAD, 400 rows holding gender, salary and y_hat, with
    declared_prediction='y_hat'):
        y_hat -> role 'oracle', reason "name matches a ground-truth / oracle
        pattern"; model_output [], oracle_columns ['y_hat']
    The oracle test ran before the model-output test and `_name_hit` matches on
    tokens, so the one-character pattern 'y' in {'y', 'hat'} beat the explicit
    declaration that this function's contract says overrides name heuristics. The
    empty model_output it produced is the same neutral value as the absent-column
    defect above, by a different route.
    """
    rng = np.random.default_rng(1)
    df = _small_frame({"y_hat": rng.integers(0, 2, 400)})

    schema, caught = _caught(
        classify_column_roles, df, declared_protected=["gender"], declared_prediction="y_hat"
    )

    assert schema["model_output"] == ["y_hat"]
    assert schema["oracle_columns"] == []
    assert [r for r in schema["roles"] if r["column"] == "y_hat"][0]["reason"] == (
        "user-declared model output"
    )
    assert [m for m in schema["mismatches"] if m["column"] == "y_hat"] == []
    assert [c for c in caught if "y_hat" in c] == []
    assert schema["refuse"] is False


# ---------------------------------------------------------------------------
# 5. rank_fairness_issues / scan_fairness_violations
# ---------------------------------------------------------------------------


def _inverted_model(n=1000):
    """Selection rates identical at 0.500 in both groups, while the model is
    perfect for M and fully inverted for F: a 1.0 error-rate disparity that no
    label-free metric can see."""
    gender = np.array(["M"] * n + ["F"] * n)
    half = n // 2
    y_true = np.r_[np.ones(half, int), np.zeros(half, int), np.ones(half, int), np.zeros(half, int)]
    y_pred = np.r_[np.ones(half, int), np.zeros(half, int), np.zeros(half, int), np.ones(half, int)]
    df = pd.DataFrame({"gender": gender, "salary": np.linspace(1.0, 2.0, 2 * n)})
    return df, y_pred, y_true


def test_rank_fairness_issues_names_the_metrics_it_never_attempted():
    """An all-clear for a run structurally incapable of seeing a 1.0 disparity.

    BEFORE (measured on HEAD, 2000 rows, selection rates identical at 0.500, the
    model perfect for M and inverted for F, y_true=None):
        {'n_attributes_checked': 1, 'n_violations': 0, 'max_disparity': 0.0,
         'n_disparities_measured': 1, 'n_not_assessable': 0}
        not_assessable [], warnings [],
        recommendations ['No significant fairness violations detected.
                          Continue monitoring.']
    Three of the four registered metrics were never attempted, and the summary
    said so nowhere: the metric menu narrows at the default
    `_DEFAULT_METRICS_NO_LABELS` without a field, a count or a warning.
    """
    df, y_pred, _y_true = _inverted_model()
    assert y_pred[:1000].mean() == 0.5 and y_pred[1000:].mean() == 0.5

    out, caught = _caught(
        rank_fairness_issues, df, y_pred, None, protected_columns=["gender"], auto_detect=False
    )
    summary = out["summary"]

    assert summary["metrics_not_attempted"] == [
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "predictive_parity_difference",
    ]
    assert summary["n_metrics_not_attempted"] == 3
    assert ALL_CLEAR not in " ".join(out["recommendations"])
    assert [r for r in out["recommendations"] if r.startswith("NOT MEASURED:")], out[
        "recommendations"
    ]
    assert [c for c in caught if "ran WITHOUT labels" in c], caught
    # The demographic-parity gap this run DID measure keeps its number: 0.0 here
    # is a measurement of equal selection rates, taken over one metric, and the
    # count beside it says so. It is not turned into a could-not-check.
    assert summary["max_disparity"] == 0.0
    assert summary["n_disparities_measured"] == 1
    assert summary["n_attributes_checked"] == 1

    scan, _ = _caught(
        scan_fairness_violations, df, y_pred, None, protected_columns=["gender"], auto_detect=False
    )
    assert scan.metrics_not_attempted == summary["metrics_not_attempted"]
    # The metric menu is NOT an unscanned attribute, and the two channels stay
    # apart: every attribute was scanned.
    assert scan.not_assessable == []
    assert scan.skipped_columns == []


def test_rank_fairness_issues_finds_the_disparity_once_it_has_labels():
    """OVER-CORRECTION CONTROL, and the proof that the fixture carries a real
    finding: the SAME data with its real y_true attempts all four metrics, finds
    three violations and a max_disparity of 1.0, and nothing is disclosed as
    unattempted."""
    df, y_pred, y_true = _inverted_model()

    out, caught = _caught(
        rank_fairness_issues, df, y_pred, y_true, protected_columns=["gender"], auto_detect=False
    )
    summary = out["summary"]

    assert summary["metrics_not_attempted"] == []
    assert summary["n_metrics_not_attempted"] == 0
    assert summary["n_violations"] == 3
    assert summary["max_disparity"] == pytest.approx(1.0, abs=1e-12)
    assert summary["n_disparities_measured"] == 4
    assert [r for r in out["recommendations"] if r.startswith("NOT MEASURED:")] == []
    assert [c for c in caught if "ran WITHOUT labels" in c] == []
