"""BGL5 audit wave 2, batch A-evaluation-3: units OVERTURNED to DEFECT OPEN.

Each unit in this batch had been graded PROVEN with a pin and a sabotage behind
it. An auditor reproduced every one of them anyway, because in each case the pin
proved the branch it was written for while the SIBLING of that branch fabricated:
the other argument of the same signature, the other reason a group can leave a
comparison, the other metric in the menu, the array the unit COMPUTED rather than
the one it was handed.

WHAT WAS MEASURED, 2026-09-29, on HEAD before the fixes in this wave
-------------------------------------------------------------------
1. ``compute_regression_effect_sizes``. ``cohens_d``'s degeneracy guard was
   ``float(np.ptp(g1)) == 0.0 and float(np.ptp(g2)) == 0.0``, exact for an array
   a caller made constant. The residual is formed INSIDE the unit as
   ``y_true[mask] - y_pred[mask]``, so a model whose error is constant within a
   group yields an array constant only up to rounding: measured ptp
   2.842170943040401e-14, and ``cohens_d`` returned -6948356750493589.0 with
   ZERO warnings, published as ``cohens_d_residuals``. Second gap in the same
   unit: ``interpret()`` was applied to ``d_pred`` only, so the residual effect
   size carried no three-state label at all.
2. ``permutation_test_equal_opportunity``. ``groups_omitted`` is derived as the
   levels of the attribute minus the levels holding a positive label, so it can
   only see a group missing an OUTCOME label. A group whose own GROUP LABEL is
   missing is in both sets (``np.unique`` lists NaN, and ``str(nan)`` matches on
   both sides) while ``attr == nan`` is False at every row, so it contributes no
   TPR. A real 0.50 TPR gap was published as 0.04999999999999999 with
   ``significant_at_05=False``, ``groups_omitted=()`` and no warning.
3. ``robust_fairness_comparison``. Same root, one layer down:
   ``compute_robust_metrics`` dropped such a level with ``if n == 0: continue``
   before the caller could see it. An 8.0 error disparity was published as
   0.20000000000000018 with ``conclusion_changed=False`` and
   ``groups_not_measured=[]``, the field a previous wave added for exactly this
   question.

The discovery half of the same batch is in
tests/test_bgl6_w2_evaluation_3_discovery.py.

Every pin below is paired with an OVER-CORRECTION CONTROL that asserts the
healthy case's REAL number, so "refuse everything" cannot pass this file.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics._statistics import cohens_d
from vfairness.evaluation.vfairness_metrics.regression import compute_regression_effect_sizes
from vfairness.evaluation.vfairness_metrics.robustness import (
    compute_robust_metrics,
    permutation_test_equal_opportunity,
    robust_fairness_comparison,
)

NOT_MEASURED = "not interpretable (effect size is nan, not a measured value)"


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(c.message) for c in caught]


# ---------------------------------------------------------------------------
# 1. compute_regression_effect_sizes / cohens_d
# ---------------------------------------------------------------------------


def _constant_error_frame(seed=7, n=40, err=50.0):
    """Two groups of `n`, predictions with real variance, the model's error
    exactly 0.0 in one group and exactly `err` in the other."""
    rng = np.random.default_rng(seed)
    attr = np.array(["a"] * n + ["b"] * n)
    pred = np.concatenate([rng.normal(100.0, 9.0, n), rng.normal(108.0, 9.0, n)])
    y_true = pred + np.concatenate([np.zeros(n), np.full(n, err)])
    return y_true, pred, attr


def test_regression_effect_sizes_refuses_a_residual_constant_up_to_rounding():
    """An error-bias effect size of 6.9e15, published silently as a measurement.

    BEFORE (measured on HEAD, 80 rows in two groups of 40, error exactly 0.0 and
    exactly 50.0, predictions with real variance):
        group b residual: 3 distinct values, ptp 2.842170943040401e-14,
                          std(ddof=1) 1.0176604434369563e-14
        cohens_d(residuals) = -6948356750493589.0, warnings []
        a_vs_b {'cohens_d_predictions': -1.5275, 'cohens_d_residuals':
                -6948356750493589.0, 'interpretation': 'large',
                'mean_residual_diff': -50.0}, warnings []
    and no label on the residual d at all, so a reader had a bare finite float
    and nothing saying whether it was measured. -6.9e15 is the same magnitude
    class (-4.3e15 at n=20) that the block comment inside `cohens_d` records as
    the defect its guard was written to remove; the guard was an EXACT equality
    with zero, which a computed array reaches only by luck.
    """
    y_true, pred, attr = _constant_error_frame()
    residual_b = y_true[40:] - pred[40:]
    # The premise: the array really is constant only up to rounding, so the
    # exact guard really was false here.
    assert len(np.unique(residual_b)) == 3
    assert float(np.ptp(residual_b)) == pytest.approx(2.842170943040401e-14, rel=1e-6)
    assert float(np.ptp(residual_b)) != 0.0

    out, caught = _caught(compute_regression_effect_sizes, y_true, pred, attr)
    pair = out["a_vs_b"]

    assert math.isnan(pair["cohens_d_residuals"]), (
        f"cohens_d_residuals was {pair['cohens_d_residuals']!r}; an undefined "
        "standardised effect must not be published as a finite number"
    )
    assert pair["interpretation_residuals"] == NOT_MEASURED
    assert [c for c in caught if "standardising denominator is zero" in c], caught
    # The SEPARATION is real and is still reported: only its size in standard
    # deviations is undefined.
    assert pair["mean_residual_diff"] == pytest.approx(-50.0, abs=1e-9)


def test_regression_effect_sizes_refuses_a_residual_formed_from_large_operands():
    """THE BOUNDARY ONE STEP OUT: the tolerance has to scale with the OPERANDS.

    `cohens_d`'s own default judges constancy relative to the magnitude of the
    values handed IN. A residual of 0.1 formed by subtracting two arrays at 1e6
    carries about 1.16e-10 of rounding, which is 1.16e-9 of its OWN magnitude:
    seven orders above the default floor, so the default alone still answers
    "this array has spread". Measured here, with group b's predictions straddling
    the 2**20 binade boundary so one constant error rounds two ways:
        cohens_d(r1, r2) with the default  -> -2402042605.1685834, warnings []
        through the unit, which knows the operands -> nan, refused, 1 warning
    That is why `compute_regression_effect_sizes` passes `constant_atol` derived
    from y_true/y_pred rather than relying on the default.
    """
    n = 40
    attr = np.array(["a"] * n + ["b"] * n)
    rng = np.random.default_rng(13)
    pred = np.concatenate(
        [rng.normal(1_000_000.0, 300.0, n), np.linspace(1_048_000.0, 1_049_200.0, n)]
    )
    y_true = pred + np.concatenate([np.zeros(n), np.full(n, 0.1)])
    r1, r2 = y_true[:n] - pred[:n], y_true[n:] - pred[n:]
    assert float(np.ptp(r2)) == pytest.approx(1.1641532182693481e-10, rel=1e-6)

    bare, bare_caught = _caught(cohens_d, r1, r2)
    assert np.isfinite(bare) and abs(bare) > 1e9, bare
    assert bare_caught == []

    out, caught = _caught(compute_regression_effect_sizes, y_true, pred, attr)
    assert math.isnan(out["a_vs_b"]["cohens_d_residuals"])
    assert out["a_vs_b"]["interpretation_residuals"] == NOT_MEASURED
    assert [c for c in caught if "standardising denominator is zero" in c], caught


def test_regression_effect_sizes_still_measures_a_real_error_bias():
    """OVER-CORRECTION CONTROL. A residual with real variance keeps its number.

    Two groups of 40 whose errors are drawn N(0, 5) and N(8, 5): a real error
    bias, standardisable, and the measured d is -2.1169650655495382 with the
    band 'large'. Nothing is refused and nothing warns. If the tolerance above
    were widened into "refuse whenever the arms look similar", this fails.
    """
    rng = np.random.default_rng(5)
    n = 40
    attr = np.array(["a"] * n + ["b"] * n)
    pred = np.concatenate([rng.normal(100.0, 9.0, n), rng.normal(100.0, 9.0, n)])
    y_true = pred + np.concatenate([rng.normal(0.0, 5.0, n), rng.normal(8.0, 5.0, n)])

    out, caught = _caught(compute_regression_effect_sizes, y_true, pred, attr)
    pair = out["a_vs_b"]
    assert pair["cohens_d_residuals"] == pytest.approx(-2.1169650655495382, abs=1e-12)
    assert pair["interpretation_residuals"] == "large"
    assert pair["cohens_d_predictions"] == pytest.approx(-0.34016305367215116, abs=1e-12)
    assert pair["interpretation"] == "small"
    assert caught == []


def test_regression_effect_sizes_keeps_a_measured_zero_error_bias():
    """OVER-CORRECTION CONTROL, the other side of the same guard. A model with NO
    error in either group has a residual that is constant AND identical, which is
    a genuine zero effect: 0.0, graded 'negligible', not a refusal."""
    rng = np.random.default_rng(5)
    n = 40
    attr = np.array(["a"] * n + ["b"] * n)
    pred = np.concatenate([rng.normal(100.0, 9.0, n), rng.normal(108.0, 9.0, n)])

    out, caught = _caught(compute_regression_effect_sizes, pred.copy(), pred, attr)
    pair = out["a_vs_b"]
    assert pair["cohens_d_residuals"] == 0.0
    assert pair["interpretation_residuals"] == "negligible"
    assert caught == []


# ---------------------------------------------------------------------------
# 2. permutation_test_equal_opportunity
# ---------------------------------------------------------------------------


def _three_blocks(third_label):
    """300 rows in three blocks of 100, every row a positive label, TPRs
    0.50 / 0.45 / 0.95, the extreme rate in the third block."""
    n = 100
    attr = np.concatenate([np.full(n, 0.0), np.full(n, 1.0), np.full(n, third_label)])
    y_true = np.ones(3 * n, dtype=int)
    y_pred = np.concatenate(
        [
            (np.arange(n) < 50).astype(int),
            (np.arange(n) < 45).astype(int),
            (np.arange(n) < 95).astype(int),
        ]
    )
    return y_true, y_pred, attr


def test_equal_opportunity_permutation_names_a_level_no_row_can_match():
    """A 0.50 TPR gap published as 0.05 with an empty completeness field.

    BEFORE (measured on HEAD, the attribute [0.0]*100 + [1.0]*100 + [nan]*100,
    TPRs 0.50 / 0.45 / 0.95):
        groups_omitted (), observed_statistic 0.04999999999999999,
        p_value 0.4992503748125937, significant_at_05 False, warnings []
    The real max-minus-min TPR gap over the attribute is 0.95 - 0.45 = 0.50. One
    third of the audit was outside the comparison, and the field a previous wave
    added for "which group did this not cover" held the neutral empty tuple,
    because it is derived from a set difference in which `str(nan)` appears on
    both sides.
    """
    y_true, y_pred, attr = _three_blocks(np.nan)
    assert int(np.sum(np.isnan(attr))) == 100

    result, caught = _caught(
        permutation_test_equal_opportunity,
        y_true,
        y_pred,
        attr,
        n_permutations=2000,
        random_state=1,
    )

    assert result.groups_omitted == ("nan",), result.groups_omitted
    assert "nan" in result.to_dict()["groups_omitted"]
    assert [c for c in caught if "match NO row" in c and "100 of 300" in c], caught
    # The verdict is NOT withdrawn by its caveat: the statistic is real for the
    # two levels that do match rows.
    assert result.observed_statistic == pytest.approx(0.05, abs=1e-12)


def test_equal_opportunity_permutation_still_measures_the_whole_gap():
    """OVER-CORRECTION CONTROL. The same three blocks with the third level
    LABELLED: every row belongs to a group, groups_omitted is empty, and the
    statistic is the real 0.50 max-minus-min TPR gap, significant at 0.05."""
    y_true, y_pred, attr = _three_blocks(2.0)

    result, caught = _caught(
        permutation_test_equal_opportunity,
        y_true,
        y_pred,
        attr,
        n_permutations=2000,
        random_state=1,
    )

    assert result.groups_omitted == ()
    assert result.observed_statistic == pytest.approx(0.5, abs=1e-12)
    assert result.significant_at_05 is True
    assert [c for c in caught if "match NO row" in c] == []


# ---------------------------------------------------------------------------
# 3. robust_fairness_comparison / compute_robust_metrics
# ---------------------------------------------------------------------------


def _error_blocks(third_label):
    """90 rows in three blocks of 30 with constant absolute errors 1.0 / 1.2 /
    9.0, the extreme one in the third block."""
    n = 30
    attr = np.concatenate([np.full(n, 0.0), np.full(n, 1.0), np.full(n, third_label)])
    err = np.concatenate([np.full(n, 1.0), np.full(n, 1.2), np.full(n, 9.0)])
    return np.zeros(3 * n), -err, attr, err


def test_robust_comparison_names_a_level_no_row_can_match():
    """An 8.0 error disparity published as 0.2, with groups_not_measured empty.

    BEFORE (measured on HEAD, the attribute [0.0]*30 + [1.0]*30 + [nan]*30 and
    constant absolute errors 1.0 / 1.2 / 9.0):
        {'standard_disparity': 0.20000000000000018,
         'robust_disparity': 0.19999999999999996,
         'disparity_change_ratio': 1.1102230246251556e-15,
         'outlier_influence_detected': False, 'conclusion_changed': False,
         'n_groups_compared': 2, 'groups_not_measured': []}
        group_metrics keys ['0.0', '1.0'], warnings []
    The real gap across the attribute is 9.0 - 1.0 = 8.0. `compute_robust_metrics`
    dropped the level with `if n == 0: continue`, so the caller could not name a
    group it was never told about and its own completeness field held [].
    """
    y_true, y_pred, attr, err = _error_blocks(np.nan)

    out, caught = _caught(robust_fairness_comparison, y_true, y_pred, attr, metric="mae")

    assert out["groups_not_measured"] == ["nan"], out["groups_not_measured"]
    assert out["n_groups_compared"] == 2
    assert "nan" in out["group_metrics"]
    nan_row = out["group_metrics"]["nan"]
    assert nan_row.n_samples == 0
    assert math.isnan(nan_row.standard_value)
    assert nan_row.outlier_influence_detected is None
    assert [c for c in caught if "match NO row" in c and "30 of 90" in c], caught

    # The layer below reports it too, rather than the caller having to infer it.
    metrics, metric_caught = _caught(compute_robust_metrics, err, attr)
    assert sorted(metrics) == ["0.0", "1.0", "nan"]
    assert [c for c in metric_caught if "match NO row" in c], metric_caught


def test_robust_comparison_still_measures_the_whole_disparity():
    """OVER-CORRECTION CONTROL. The same three blocks with the third level
    LABELLED: nothing is unmeasured and the published disparity is the real
    9.0 - 1.0 = 8.0 over three groups."""
    y_true, y_pred, attr, _err = _error_blocks(2.0)

    out, caught = _caught(robust_fairness_comparison, y_true, y_pred, attr, metric="mae")

    assert out["groups_not_measured"] == []
    assert out["n_groups_compared"] == 3
    assert out["standard_disparity"] == pytest.approx(8.0, abs=1e-9)
    assert out["robust_disparity"] == pytest.approx(8.0, abs=1e-9)
    assert sorted(out["group_metrics"]) == ["0.0", "1.0", "2.0"]
    assert [c for c in caught if "match NO row" in c] == []
