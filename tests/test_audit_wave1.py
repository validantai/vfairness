"""Regression tests pinning the Wave-1 fixes from the 2026-08 deep audit.

Each test targets one confirmed defect; the test names carry the finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── threshold_optimization/optimizer.py: operator-precedence crash ──────────


def test_threshold_optimizer_fit_no_longer_crashes():
    from vfairness.post_processing.threshold_optimization.optimizer import (
        ThresholdOptimizer,
        _compute_performance_metrics,
    )

    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.5, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    g = np.where(np.arange(300) % 2 == 0, "A", "B")

    pred = ThresholdOptimizer(constraint="demographic_parity", tolerance=0.05).fit_predict(y, p, g)
    assert pred.shape == (300,)

    # weighted path: bool mask must be parenthesized before * weights
    m = _compute_performance_metrics(y, (p > 0.5).astype(int), rng.random(300))
    assert 0.0 <= m["f1_score"] <= 1.0


def test_confusion_matrix_weights_are_actually_applied():
    from vfairness.post_processing.threshold_optimization.optimizer import (
        _compute_performance_metrics,
    )

    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    # Weight the TP row 10x: precision/recall must move accordingly
    w = np.array([10.0, 1.0, 1.0, 1.0])
    m = _compute_performance_metrics(y_true, y_pred, w)
    assert m["precision"] == pytest.approx(10.0 / 11.0)
    assert m["recall"] == pytest.approx(10.0 / 11.0)


# ── threshold_optimization/constraints.py: equalized_odds always feasible ───


def test_equalized_odds_feasibility_is_selective():
    from vfairness.post_processing.threshold_optimization.constraints import (
        find_feasible_thresholds,
    )

    rng = np.random.default_rng(1)
    n = 800
    g = np.where(np.arange(n) % 2 == 0, "A", "B")
    y = rng.binomial(1, 0.5, n)
    # Wildly separated score distributions: most thresholds cannot equalise
    p = np.where(g == "A", 0.9 + 0.1 * rng.random(n), 0.1 * rng.random(n))
    feasible = find_feasible_thresholds(
        y,
        p,
        g,
        constraint="equalized_odds",
        tolerance=0.02,
        n_thresholds=50,
    )
    total = sum(len(v) for v in feasible.values())
    assert total < 100, "equalized_odds must not declare every threshold feasible"


# ── in_processing FairRegressor: silent unconstrained fit ────────────────────


def test_fair_regressor_refuses_unimplemented_constraints():
    sklearn = pytest.importorskip("sklearn.linear_model")
    from vfairness.in_processing.wrappers.sklearn_wrappers import FairRegressor

    X = np.random.default_rng(2).random((50, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    g = np.array(["A", "B"] * 25)

    reg = FairRegressor(sklearn.Ridge(), fairness_constraint="error_parity")
    with pytest.raises(NotImplementedError):
        reg.fit(X, y, sensitive_attr=g)

    with pytest.raises(ValueError):
        FairRegressor(sklearn.Ridge(), fairness_constraint="nonsense").fit(X, y, sensitive_attr=g)

    # mean_parity still works
    FairRegressor(sklearn.Ridge(), fairness_constraint="mean_parity").fit(
        X, y, sensitive_attr=g
    ).predict(X)


# ── _grouping.compute_max_difference: value-ordered pair ────────────────────


def test_compute_max_difference_orders_pair_by_value():
    from vfairness.evaluation.vfairness_metrics._grouping import (
        compute_max_difference,
    )

    # Alphabetically first group has the LOWER rate: previously reported
    # as the "high" member of the pair.
    diff, hi, lo = compute_max_difference({"A": 0.30, "B": 0.80})
    assert diff == pytest.approx(0.5)
    assert hi == "B" and lo == "A"


# ── preprocessing representation: visualize KeyError ────────────────────────


def test_compare_to_benchmark_visualize_works():
    from vfairness.preprocessing.bias_detection.representation import (
        compare_to_benchmark,
    )

    df = pd.DataFrame({"gender": ["M"] * 60 + ["F"] * 40})
    out = compare_to_benchmark(
        df,
        "gender",
        {"M": 0.5, "F": 0.5},
        visualize=True,
    )
    viz = out["visualization_data"]
    # group keys are normalised to lowercase by the function
    assert set(viz["groups"]) == {"m", "f"}
    assert len(viz["actual"]) == len(viz["groups"])
    assert len(viz["expected"]) == len(viz["groups"])


# ── cicd gate/monitor/testing: groups[:2] truncation ─────────────────────────


def _three_group_data():
    # A and B identical; C never selected: the disparity IS the A/C gap.
    protected = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    y_pred = np.array([1] * 20 + [0] * 20 + [1] * 20 + [0] * 20 + [0] * 40)
    y_true = np.array(([1, 0] * 20) + ([1, 0] * 20) + ([1, 0] * 20))
    return y_true, y_pred, protected


def test_gate_default_metrics_cover_all_groups():
    from vfairness.operations.cicd.gate import ModelFairnessGate

    y_true, y_pred, prot = _three_group_data()
    gate = ModelFairnessGate.__new__(ModelFairnessGate)  # method needs no state
    metrics = gate._compute_default_metrics(y_true, y_pred, prot)
    assert metrics["demographic_parity_difference"] == pytest.approx(0.5)


def test_monitor_default_metrics_cover_all_groups():
    from vfairness.operations.cicd.monitor import BiasMonitor

    y_true, y_pred, prot = _three_group_data()
    mon = BiasMonitor.__new__(BiasMonitor)  # avoid ctor deps
    metrics = mon._compute_default_metrics(y_true, y_pred, prot)
    assert metrics["demographic_parity_difference"] == pytest.approx(0.5)


def test_testing_default_metrics_cover_all_groups():
    import vfairness.operations.cicd.testing as tmod

    cls = None
    for name in dir(tmod):
        obj = getattr(tmod, name)
        if isinstance(obj, type) and hasattr(obj, "_compute_default_metrics"):
            cls = obj
            break
    assert cls is not None
    y_true, y_pred, prot = _three_group_data()
    inst = cls.__new__(cls)
    metrics = inst._compute_default_metrics(y_true, y_pred, prot)
    assert metrics["demographic_parity_difference"] == pytest.approx(0.5)


# ── ranking: [0,1] scores floored to position 0 ─────────────────────────────


def test_ranking_scores_in_unit_interval_not_flattened():
    from vfairness.evaluation.vfairness_metrics.ranking import (
        exposure_parity_difference,
    )

    scores = np.array([0.9, 0.8, 0.2, 0.1])
    groups = np.array(["A", "A", "B", "B"])
    d_scores = exposure_parity_difference(scores, groups, min_group_size=1)
    d_positions = exposure_parity_difference(np.array([0, 1, 2, 3]), groups, min_group_size=1)
    assert d_scores > 0.2, "unit-interval scores must not read as perfect parity"
    assert d_scores == pytest.approx(d_positions)


# ── report.py: pseudo p-values were inverted ────────────────────────────────


def test_multiple_testing_uses_real_p_values_unfair_is_significant():
    from vfairness.evaluation.vfairness_metrics.report import (
        classification_fairness_report,
    )

    rng = np.random.default_rng(3)
    n = 2000
    g = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_true = rng.binomial(1, 0.5, n)
    # Blatant demographic disparity: A selected 80%, B 20%
    y_pred = np.where(g == "A", rng.binomial(1, 0.8, n), rng.binomial(1, 0.2, n))
    rep = classification_fairness_report(
        y_true,
        y_pred,
        g,
        include_ci=True,
        n_bootstrap=50,
        multiple_testing_correction="bonferroni",
    )
    corr = rep["multiple_testing_correction"]
    names = corr["tested_metrics"]
    mask = corr.get("rejection_mask") or corr.get("reject") or corr.get("rejected")
    assert mask is not None, f"correction dict keys: {list(corr)}"
    dp_idx = names.index("demographic_parity_difference")
    assert bool(np.asarray(mask)[dp_idx]), (
        "a 60-point selection gap must be statistically significant"
    )


def test_regression_report_refuses_to_fabricate_p_values():
    from vfairness.evaluation.vfairness_metrics.report import (
        regression_fairness_report,
    )

    rng = np.random.default_rng(4)
    n = 400
    g = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    y_true = rng.normal(0, 1, n)
    y_pred = y_true + np.where(g == "A", 0.0, 1.0) + rng.normal(0, 0.1, n)
    rep = regression_fairness_report(
        y_true,
        y_pred,
        g,
        include_ci=True,
        n_bootstrap=50,
        multiple_testing_correction="fdr",
    )
    corr = rep["multiple_testing_correction"]
    assert corr["available"] is False
    assert "pseudo" in corr["reason"] or "fabricate" in corr["reason"]


# ── xai shap_adapter: class-consistent base value ───────────────────────────


def test_shap_class_consistent_helper():
    from vfairness.xai.explainers.shap_adapter import _class_consistent

    class0 = np.array([[0.1, -0.2]])
    class1 = np.array([[-0.1, 0.2]])
    expected = np.array([0.7, 0.3])
    values, base = _class_consistent([class0, class1], expected)
    assert np.allclose(values, class1)
    assert base == pytest.approx(0.3)  # class-1 base, not class-0
    # scalar expected_value passthrough
    v2, b2 = _class_consistent(class0, 0.5)
    assert np.allclose(v2, class0) and b2 == 0.5


# ── mcp server: import-time SystemExit escaped except Exception ─────────────


def test_mcp_server_import_error_is_catchable(monkeypatch):
    import builtins
    import importlib
    import sys

    for mod in list(sys.modules):
        if mod == "mcp" or mod.startswith("mcp."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delitem(sys.modules, "vfairness.mcp.server", raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        # Must be an ImportError (catchable by `except Exception`),
        # NOT SystemExit which killed introspecting host processes.
        importlib.import_module("vfairness.mcp.server")


# ── experimentation: heterogeneity no longer ANOVA-on-replicates ────────────


def test_heterogeneity_not_spuriously_significant_for_equal_effects():
    # NOTE: this pin used to scan the module for a class with an `analyze` method,
    # which no class has, so it silently SKIPPED and asserted nothing (a load-bearing
    # skip found in the 2026-08-23 release pass). It now drives the real
    # FairnessExperiment API directly so the assertion actually runs.
    from vfairness.operations.experimentation.experiment import FairnessExperiment

    rng = np.random.default_rng(5)
    n = 1200
    gender = rng.choice(["M", "F"], n)
    base = rng.normal(0, 1, n)
    # SAME true effect (+0.5) in every intersection -> no heterogeneity.
    control = pd.DataFrame({"gender": gender[: n // 2], "outcome": base[: n // 2]})
    treatment = pd.DataFrame({"gender": gender[n // 2 :], "outcome": base[n // 2 :] + 0.5})
    exp = FairnessExperiment(
        control_data=control,
        treatment_data=treatment,
        protected_attributes=["gender"],
        outcome_column="outcome",
    )
    result = exp.detect_heterogeneous_effects()
    assert result.heterogeneity_detected is False, (
        "identical effects across groups must not be flagged heterogeneous"
    )
