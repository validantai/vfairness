#!/usr/bin/env python3
"""
Regenerate every SVG gallery example from LIVE computation.

This is the sample-generation pipeline the gallery page renders from. For each
template the gallery shows, it drives the real vfairness path (analysers on
synthetic datasets where one exists, otherwise the documented adapter input) and
writes two files:

    docs/site/img/svg-gallery/<name>.svg                 with explanation
    docs/site/img/svg-gallery/<name>_no_explanation.svg  no explanation

Renders are Blanco: `render_svg` (and therefore every adapter and `.to_svg()`)
applies the Blanco design language by default, so the files written here are
already Blanco — there is no separate scheme.

Data mirrors the published demo notebook
(notebooks/vfairness_0_svg_rendering_demo.ipynb). Run it from the repository
root with a full env (numpy/pandas/scipy/sklearn/jinja2), e.g. a virtualenv
at .venv:

    PYTHONPATH=src .venv/bin/python scripts/regenerate_gallery_examples.py
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from vfairness.rendering import (  # noqa: E402
    alert_timeline_to_svg,
    auto_discovery_to_svg,
    bias_audit_to_svg,
    calibration_disparity_to_svg,
    calibration_report_to_svg,
    causal_decomposition_to_svg,
    cicd_pipeline_to_svg,
    confidence_intervals_to_svg,
    correlation_heatmap_to_svg,
    correlation_matrix_to_svg,
    data_validation_to_svg,
    disparity_heatmap_to_svg,
    drift_report_to_svg,
    effect_sizes_to_svg,
    experiment_recommendation_to_svg,
    experiment_results_to_svg,
    fairness_detailed_report_to_svg,
    fairness_report_to_svg,
    group_calibration_to_svg,
    group_comparison_to_svg,
    hierarchical_gate_to_svg,
    intersectional_analysis_to_svg,
    intersectional_disparity_to_svg,
    method_comparison_to_svg,
    metrics_bar_chart_to_svg,
    monitoring_dashboard_to_svg,
    pareto_frontier_to_svg,
    power_analysis_to_svg,
    proxy_risk_to_svg,
    radar_chart_to_svg,
    ranking_fairness_to_svg,
    regression_fairness_to_svg,
    reliability_diagram_to_svg,
    report_card_to_svg,
    reporting_dashboard_to_svg,
    reweighting_comparison_to_svg,
    robustness_testing_to_svg,
    temporal_analysis_to_svg,
    threshold_optimization_to_svg,
    tradeoff_analysis_to_svg,
    training_analysis_report_to_svg,
    training_report_to_svg,
    transformation_comparison_to_svg,
    workflow_overview_to_svg,
)

OUT = REPO / "docs" / "site" / "img" / "svg-gallery"

# Vivid original tones that must never survive in a Blanco render.
_LEAK = [
    "#dc2626",
    "#ef4444",
    "#f59e0b",
    "#f97316",
    "#059669",
    "#10b981",
    "#3b82f6",
    "#8b5cf6",
    "#ec4899",
    "#0aafe3",
    "#3fb818",
    "#0f172a",
    "#94a3b8",
]

np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Shared synthetic datasets (mirrors notebook cells 2 / 44 / 61)
# ─────────────────────────────────────────────────────────────────────────────


def build_data():
    d = {}
    n = 2000
    race = np.random.choice(["White", "Black", "Hispanic", "Asian"], n, p=[0.60, 0.15, 0.18, 0.07])
    gender = np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.72, 0.22, 0.06])
    income = np.random.lognormal(10.8, 0.5, n).astype(int)
    credit_score = np.clip(np.random.normal(700, 50, n), 300, 850).astype(int)
    bias = np.where(race == "White", 0.12, -0.08) + np.where(gender == "Male", 0.07, -0.03)
    base_prob = 0.5 + (credit_score - 650) / 600 + bias
    loan_approved = (np.random.rand(n) < np.clip(base_prob, 0.05, 0.95)).astype(int)
    d["df"] = pd.DataFrame(
        {
            "race": race,
            "gender": gender,
            "income": income,
            "credit_score": credit_score,
            "loan_approved": loan_approved,
        }
    )

    n_fair = 1000
    fair_gender = np.random.choice(["Male", "Female"], n_fair, p=[0.55, 0.45])
    y_true = np.random.binomial(1, 0.6, n_fair)
    fair_bias = np.where(fair_gender == "Male", 0.15, -0.05)
    prob = np.clip(0.5 + fair_bias + 0.2 * y_true + np.random.normal(0, 0.1, n_fair), 0, 1)
    y_pred = (prob > 0.5).astype(int)
    d.update(fair_gender=fair_gender, y_true=y_true, y_pred=y_pred)

    n_cal = 1000
    d["cal_y_true"] = np.random.randint(0, 2, n_cal)
    d["cal_y_prob"] = np.clip(d["cal_y_true"] * 0.6 + np.random.normal(0.3, 0.15, n_cal), 0, 1)
    d["cal_protected"] = np.random.choice(["Group A", "Group B", "Group C"], n_cal)

    now = datetime(2026, 8, 1, 12, 0)
    d["now"] = now
    return d


def build_specs(d):
    """Return {name: render_fn(explanation) -> svg}."""
    from vfairness import FairnessAnalyzer
    from vfairness.post_processing.calibration import CalibrationAnalyzer
    from vfairness.preprocessing.bias_detection import BiasDetector

    # ── Bias audit (BiasDetector.full_audit) ──
    benchmarks = {
        "gender": {"Male": 0.49, "Female": 0.51, "Non-binary": 0.03},
        "race": {"White": 0.58, "Black": 0.13, "Hispanic": 0.19, "Asian": 0.06},
    }
    detector = BiasDetector(
        d["df"],
        protected_attributes=["race", "gender"],
        outcome_column="loan_approved",
        benchmarks=benchmarks,
    )
    bias_report = detector.full_audit(
        include_historical=True,
        include_representation=True,
        include_disparities=True,
        include_proxies=True,
    )

    # ── Calibration (CalibrationAnalyzer.full_analysis) ──
    n_c = 2000
    pa = np.random.choice(["Group_A", "Group_B"], n_c, p=[0.6, 0.4])
    yt = np.random.binomial(1, 0.4, n_c)
    yp = np.zeros(n_c)
    ma = pa == "Group_A"
    yp[ma] = np.clip(yt[ma] * 0.8 + np.random.normal(0.2, 0.15, ma.sum()), 0.01, 0.99)
    yp[~ma] = np.clip(yt[~ma] * 0.5 + np.random.normal(0.35, 0.2, (~ma).sum()), 0.01, 0.99)
    cal_report = CalibrationAnalyzer(
        y_true=yt,
        y_prob=yp,
        protected_attr=pa,
        attribute_name="group",
        n_bins=10,
        min_group_size=50,
    ).full_analysis(
        include_tradeoffs=True, include_recommendation=True, include_brier_decomposition=True
    )

    # ── Fairness metrics (FairnessAnalyzer.get_report) ──
    fair_report = FairnessAnalyzer(d["y_true"], d["y_pred"], d["fair_gender"]).get_report()

    # ── static / mock inputs (mirror the notebook) ──
    disparity_result = {
        "group_metrics": {
            "Group A": {"ece": 0.035, "mce": 0.12, "brier": 0.18},
            "Group B": {"ece": 0.072, "mce": 0.21, "brier": 0.24},
            "Group C": {"ece": 0.045, "mce": 0.15, "brier": 0.20},
        },
        "reference_group": "Group A",
        "max_ece_disparity": 0.037,
        "max_mce_disparity": 0.09,
    }
    cal_errors = [0.02, 0.05, 0.08, 0.12, 0.04, 0.07, 0.15, 0.03]
    fair_violations = [0.25, 0.15, 0.08, 0.03, 0.20, 0.10, 0.02, 0.22]
    model_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

    corr_matrix = SimpleNamespace(
        features=["income", "zip_code", "credit_score", "education", "occupation"],
        protected_attributes=["gender", "race", "age"],
        correlations={
            "income": {"gender": 0.35, "race": 0.12, "age": 0.45},
            "zip_code": {"gender": 0.05, "race": 0.72, "age": 0.08},
            "credit_score": {"gender": 0.18, "race": 0.22, "age": 0.55},
            "education": {"gender": 0.28, "race": 0.15, "age": 0.32},
            "occupation": {"gender": 0.42, "race": 0.08, "age": 0.19},
        },
    )
    corr_data = {
        "income": {
            "income": 1.0,
            "credit_score": 0.72,
            "age": 0.35,
            "education": 0.48,
            "zip_code": -0.15,
        },
        "credit_score": {
            "income": 0.72,
            "credit_score": 1.0,
            "age": 0.28,
            "education": 0.55,
            "zip_code": -0.08,
        },
        "age": {
            "income": 0.35,
            "credit_score": 0.28,
            "age": 1.0,
            "education": 0.12,
            "zip_code": 0.05,
        },
        "education": {
            "income": 0.48,
            "credit_score": 0.55,
            "age": 0.12,
            "education": 1.0,
            "zip_code": -0.22,
        },
        "zip_code": {
            "income": -0.15,
            "credit_score": -0.08,
            "age": 0.05,
            "education": -0.22,
            "zip_code": 1.0,
        },
    }
    corr_methods = {
        "income": {
            "income": "pearson",
            "credit_score": "pearson",
            "age": "pearson",
            "education": "point_biserial",
            "zip_code": "cramers_v",
        },
        "credit_score": {
            "income": "pearson",
            "credit_score": "pearson",
            "age": "pearson",
            "education": "point_biserial",
            "zip_code": "cramers_v",
        },
        "age": {
            "income": "pearson",
            "credit_score": "pearson",
            "age": "pearson",
            "education": "spearman",
            "zip_code": "cramers_v",
        },
        "education": {
            "income": "point_biserial",
            "credit_score": "point_biserial",
            "age": "spearman",
            "education": "cramers_v",
            "zip_code": "cramers_v",
        },
        "zip_code": {
            "income": "cramers_v",
            "credit_score": "cramers_v",
            "age": "cramers_v",
            "education": "cramers_v",
            "zip_code": "cramers_v",
        },
    }
    proxy_vars = [
        SimpleNamespace(
            feature="zip_code",
            protected_attribute="race",
            correlation=0.72,
            risk_level=SimpleNamespace(value="high"),
            confidence_score=0.91,
        ),
        SimpleNamespace(
            feature="income",
            protected_attribute="age",
            correlation=0.45,
            risk_level=SimpleNamespace(value="medium"),
            confidence_score=0.78,
        ),
        SimpleNamespace(
            feature="occupation",
            protected_attribute="gender",
            correlation=0.42,
            risk_level=SimpleNamespace(value="medium"),
            confidence_score=0.65,
        ),
        SimpleNamespace(
            feature="education",
            protected_attribute="age",
            correlation=0.32,
            risk_level=SimpleNamespace(value="low"),
            confidence_score=0.55,
        ),
        SimpleNamespace(
            feature="credit_score",
            protected_attribute="race",
            correlation=0.22,
            risk_level=SimpleNamespace(value="low"),
            confidence_score=0.42,
        ),
    ]
    before = {
        "zip_code": 0.72,
        "income": 0.45,
        "occupation": 0.42,
        "education": 0.32,
        "credit_score": 0.22,
    }
    after = {
        "zip_code": 0.28,
        "income": 0.15,
        "occupation": 0.20,
        "education": 0.18,
        "credit_score": 0.10,
    }
    intersect_data = {
        "x_attr": "Gender",
        "y_attr": "Race",
        "matrix": {
            "White": {"Male": 0.65, "Female": 0.72, "Non-Binary": 0.60},
            "Black": {"Male": 0.42, "Female": 0.38, "Non-Binary": 0.45},
            "Hispanic": {"Male": 0.55, "Female": 0.50, "Non-Binary": 0.52},
            "Asian": {"Male": 0.68, "Female": 0.70, "Non-Binary": 0.62},
        },
        "counts": {
            "White": {"Male": 320, "Female": 310, "Non-Binary": 45},
            "Black": {"Male": 180, "Female": 195, "Non-Binary": 28},
            "Hispanic": {"Male": 210, "Female": 200, "Non-Binary": 32},
            "Asian": {"Male": 150, "Female": 160, "Non-Binary": 22},
        },
    }
    intersect_disparity_data = {
        "privilegedGroup": {
            "group": "Asian · Male",
            "positiveRate": 0.71,
            "groundTruthRate": 0.63,
            "size": 150,
        },
        "disadvantagedGroup": {
            "group": "Black · Female",
            "positiveRate": 0.34,
            "groundTruthRate": 0.55,
            "size": 195,
        },
        "allGroups": [
            {
                "group": "Asian · Male",
                "positiveRate": 0.71,
                "groundTruthRate": 0.63,
                "predictionDelta": 0.08,
                "size": 150,
                "severity": "low",
            },
            {
                "group": "White · Male",
                "positiveRate": 0.66,
                "groundTruthRate": 0.60,
                "predictionDelta": 0.06,
                "size": 320,
                "severity": "low",
            },
            {
                "group": "Hispanic · Male",
                "positiveRate": 0.55,
                "groundTruthRate": 0.54,
                "predictionDelta": 0.01,
                "size": 210,
                "severity": "low",
            },
            {
                "group": "White · Female",
                "positiveRate": 0.52,
                "groundTruthRate": 0.58,
                "predictionDelta": -0.06,
                "size": 310,
                "severity": "medium",
            },
            {
                "group": "Hispanic · Female",
                "positiveRate": 0.44,
                "groundTruthRate": 0.52,
                "predictionDelta": -0.08,
                "size": 200,
                "severity": "medium",
            },
            {
                "group": "Black · Female",
                "positiveRate": 0.34,
                "groundTruthRate": 0.55,
                "predictionDelta": -0.21,
                "size": 195,
                "severity": "high",
            },
        ],
        "maxDisparity": 0.37,
        "maxGroundTruthDisparity": 0.11,
        "disparitySeverity": "high",
        "insights": [
            "Black · Female subgroup shows a 37pp approval gap versus the privileged subgroup",
            "Compound disadvantage exceeds either single-attribute gap (race 18pp, gender 14pp)",
            "Prediction rate for Black · Female is 21pp below their ground-truth base rate",
        ],
        "findings": [{"severity": "high"}, {"severity": "medium"}, {"severity": "info"}],
    }

    # ── training mocks (notebook cell 44) ──
    method_comparisons = [
        SimpleNamespace(
            method_name="Baseline",
            accuracy=0.852,
            fairness_violation=0.185,
            constraint_satisfied=False,
            training_time=2.1,
        ),
        SimpleNamespace(
            method_name="Reweighting",
            accuracy=0.834,
            fairness_violation=0.082,
            constraint_satisfied=True,
            training_time=3.4,
        ),
        SimpleNamespace(
            method_name="Adversarial",
            accuracy=0.841,
            fairness_violation=0.065,
            constraint_satisfied=True,
            training_time=8.7,
        ),
        SimpleNamespace(
            method_name="Constrained",
            accuracy=0.828,
            fairness_violation=0.041,
            constraint_satisfied=True,
            training_time=5.2,
        ),
    ]
    tradeoff = {
        "all_results": [
            {"accuracy": 0.852, "violation": 0.185, "lambda": 0.0, "satisfied": False},
            {"accuracy": 0.845, "violation": 0.142, "lambda": 0.1, "satisfied": False},
            {"accuracy": 0.834, "violation": 0.082, "lambda": 0.3, "satisfied": True},
            {"accuracy": 0.828, "violation": 0.041, "lambda": 0.5, "satisfied": True},
            {"accuracy": 0.810, "violation": 0.025, "lambda": 0.8, "satisfied": True},
            {"accuracy": 0.785, "violation": 0.012, "lambda": 1.0, "satisfied": True},
        ],
        "pareto_frontier": [
            {"accuracy": 0.852, "violation": 0.185},
            {"accuracy": 0.834, "violation": 0.082},
            {"accuracy": 0.828, "violation": 0.041},
            {"accuracy": 0.785, "violation": 0.012},
        ],
        "best_fair": {"accuracy": 0.828, "violation": 0.041},
        "best_accurate": {"accuracy": 0.852, "violation": 0.185},
    }
    training_report = SimpleNamespace(
        timestamp=d["now"].strftime("%Y-%m-%d %H:%M"),
        task_type="binary_classification",
        data_info={"n_samples": 5000, "n_groups": 3, "n_features": 12, "attribute_name": "gender"},
        baseline_metrics={
            "accuracy": 0.852,
            "fairness_violation": 0.185,
            "constraint_satisfied": False,
        },
        method_comparisons=method_comparisons,
        recommendation=SimpleNamespace(
            recommended_method="Constrained",
            priority="high",
            rationale="Best fairness-accuracy trade-off with constraint satisfaction",
            alternative_methods=["Adversarial", "Reweighting"],
        ),
        critical_issues=[
            {
                "type": "fairness",
                "description": "Baseline violates demographic parity",
                "severity": "high",
            },
            {
                "type": "data",
                "description": "Underrepresented group < 5% of data",
                "severity": "medium",
            },
        ],
        action_items=[
            "Apply constrained training method",
            "Collect more data for underrepresented group",
            "Monitor fairness metrics in production",
        ],
        tradeoff_analysis=tradeoff,
        fairness_analysis={
            "constraint_type": "demographic_parity",
            "base_rate_disparity": 0.12,
            "group_statistics": {
                "Male": {"size": 2750, "proportion": 0.55, "positive_rate": 0.72},
                "Female": {"size": 2000, "proportion": 0.40, "positive_rate": 0.58},
                "Non-binary": {"size": 250, "proportion": 0.05, "positive_rate": 0.61},
            },
        },
    )

    threshold_result = {
        "title": "Threshold Optimization Analysis",
        "constraint_type": "demographic_parity",
        "tolerance": 0.05,
        "original_threshold": 0.5,
        "groups": [
            {
                "name": "Male",
                "threshold": 0.55,
                "original_rate": 0.72,
                "optimized_rate": 0.63,
                "size": 2750,
            },
            {
                "name": "Female",
                "threshold": 0.42,
                "original_rate": 0.58,
                "optimized_rate": 0.64,
                "size": 2000,
            },
            {
                "name": "Non-binary",
                "threshold": 0.44,
                "original_rate": 0.61,
                "optimized_rate": 0.63,
                "size": 250,
            },
        ],
        "original_disparity": 0.14,
        "optimized_disparity": 0.01,
        "original_accuracy": 0.852,
        "optimized_accuracy": 0.841,
        "is_feasible": True,
        "recommendation": "Use optimised group-specific thresholds.",
    }
    reweighting_report = {
        "metadata": {"n_samples": 5000, "n_groups": 3},
        "method_results": [
            {
                "method": "uniform_reweighting",
                "original_fairness": {"demographic_parity_diff": 0.18},
                "adjusted_fairness": {"demographic_parity_diff": 0.09},
                "original_performance": {"accuracy": 0.85},
                "adjusted_performance": {"accuracy": 0.83},
                "calibration_metrics": {"ece_change": 0.01},
                "trade_off_score": 0.72,
            },
            {
                "method": "group_specific",
                "original_fairness": {"demographic_parity_diff": 0.18},
                "adjusted_fairness": {"demographic_parity_diff": 0.04},
                "original_performance": {"accuracy": 0.85},
                "adjusted_performance": {"accuracy": 0.82},
                "calibration_metrics": {"ece_change": 0.02},
                "trade_off_score": 0.85,
            },
            {
                "method": "calibrated_reweighting",
                "original_fairness": {"demographic_parity_diff": 0.18},
                "adjusted_fairness": {"demographic_parity_diff": 0.06},
                "original_performance": {"accuracy": 0.85},
                "adjusted_performance": {"accuracy": 0.84},
                "calibration_metrics": {"ece_change": -0.005},
                "trade_off_score": 0.88,
            },
        ],
        "best_method": "calibrated_reweighting",
        "recommendations": [
            "Use calibrated_reweighting for best accuracy-fairness trade-off",
            "Monitor calibration after deployment",
        ],
    }
    detailed_report = {
        "title": "Fairness Analysis Report",
        "task_type": "binary_classification",
        "assessment": {"fairness_score": 0.45},
        "metrics": {
            "demographic_parity": {
                "value": 0.14,
                "threshold": 0.10,
                "interpretation": "Significant gap",
            },
            "equal_opportunity": {
                "value": 0.08,
                "threshold": 0.10,
                "interpretation": "Within bounds",
            },
            "equalized_odds": {
                "value": 0.18,
                "threshold": 0.15,
                "interpretation": "Exceeds threshold",
            },
            "predictive_parity": {"value": 0.05, "threshold": 0.10, "interpretation": "Acceptable"},
        },
        "group_statistics": {
            "Male": {"size": 550, "positive_rate": 0.72, "tpr": 0.85, "fpr": 0.22},
            "Female": {"size": 450, "positive_rate": 0.58, "tpr": 0.77, "fpr": 0.18},
        },
        "pairwise_comparisons": [{"group_a": "Male", "group_b": "Female", "disparity": 0.14}],
        "key_findings": [
            "Demographic parity gap of 14% exceeds the 10% threshold",
            "Equal opportunity is within acceptable bounds",
        ],
        "recommendations": [
            "Apply threshold adjustment or reweighting",
            "Collect additional samples for minority group",
        ],
    }

    # ── CI/CD gate mocks (notebook cell 11) ──
    from enum import Enum

    class Status(Enum):
        passed = "passed"
        failed = "failed"

    class GateStatus(Enum):
        approved = "approved"
        rejected = "rejected"

    validation = SimpleNamespace(
        passed=True,
        errors=[],
        warnings=[
            SimpleNamespace(
                message="Minor class imbalance detected", severity=SimpleNamespace(value="warning")
            )
        ],
    )
    tests = [
        SimpleNamespace(
            test_name="test_demographic_parity[gender]",
            status=Status.passed,
            metric_name="DP",
            metric_value=0.08,
            threshold=0.10,
        ),
        SimpleNamespace(
            test_name="test_equal_opportunity[gender]",
            status=Status.passed,
            metric_name="EO",
            metric_value=0.06,
            threshold=0.10,
        ),
        SimpleNamespace(
            test_name="test_equalized_odds[gender]",
            status=Status.failed,
            metric_name="EOD",
            metric_value=0.22,
            threshold=0.15,
        ),
        SimpleNamespace(
            test_name="test_data_representation[gender]",
            status=Status.passed,
            metric_name="min_frac",
            metric_value=0.12,
            threshold=0.05,
        ),
    ]
    gate = SimpleNamespace(
        status=GateStatus.rejected,
        approved=False,
        blocking_reasons=["Equalized odds exceeds threshold (0.22 > 0.15)"],
        warnings=["Consider model retraining with fairness constraints"],
    )

    # ── monitoring mocks (notebook cell 61) ──
    now = d["now"]

    def make_window(ts, has_alert=False):
        dp = np.random.uniform(0.04, 0.18)
        eo = np.random.uniform(0.03, 0.12)
        metrics = {
            "demographic_parity_diff": dp,
            "equal_opportunity_diff": eo,
            "predictive_parity_diff": np.random.uniform(0.02, 0.08),
        }
        alerts = {
            "demographic_parity_diff": dp > 0.12 or has_alert,
            "equal_opportunity_diff": eo > 0.10,
            "predictive_parity_diff": False,
        }
        return SimpleNamespace(
            timestamp=ts,
            metrics=metrics,
            alerts=alerts,
            any_alert=any(alerts.values()),
            sample_count=int(np.random.randint(800, 1200)),
            group_rates={
                "gender": {
                    "Male": np.random.uniform(0.60, 0.75),
                    "Female": np.random.uniform(0.50, 0.65),
                }
            },
            mmd_scores={
                "gender": np.random.uniform(0.001, 0.08),
                "age": np.random.uniform(0.005, 0.15),
            },
        )

    monitoring_history = [
        make_window(now - timedelta(hours=10 - i), has_alert=(i in [3, 7])) for i in range(10)
    ]
    latest_window = monitoring_history[-1]

    mock_drift = SimpleNamespace(
        drift_detected=True,
        overall_drift_score=0.45,
        metric="demographic_parity_diff",
        scales={
            "short_term": SimpleNamespace(
                drift_score=0.15,
                ks_statistic=0.08,
                p_value=0.32,
                mean_shift=0.01,
                reference_mean=0.08,
                current_mean=0.09,
                drift_detected=False,
            ),
            "medium_term": SimpleNamespace(
                drift_score=0.45,
                ks_statistic=0.22,
                p_value=0.04,
                mean_shift=0.05,
                reference_mean=0.08,
                current_mean=0.13,
                drift_detected=True,
            ),
            "long_term": SimpleNamespace(
                drift_score=0.72,
                ks_statistic=0.35,
                p_value=0.001,
                mean_shift=0.09,
                reference_mean=0.06,
                current_mean=0.15,
                drift_detected=True,
            ),
        },
    )

    class MockTemporal:
        def to_dataframe(self):
            return pd.DataFrame(
                {
                    "metric_name": ["dp_diff"] * 14 + ["eo_diff"] * 14,
                    "value": np.random.uniform(0.05, 0.18, 28),
                    "timestamp": [now - timedelta(days=i) for i in range(14)] * 2,
                }
            )

        def get_metric_summary(self, name):
            return {
                "mean": 0.11,
                "std": 0.03,
                "min": 0.05,
                "max": 0.18,
                "trend_direction": "increasing",
                "trend_slope": 0.008,
                "n_days": 14,
            }

        def detect_seasonal_pattern(self, name, period=7):
            return {f"day_{i}": 0.08 + 0.02 * np.sin(i * 0.9) for i in range(period)}

        def detect_trend(self, name):
            return ("increasing", 0.008)

        def detect_weekly_degradation(self, name):
            return (True, "Friday")

    mock_temporal = MockTemporal()

    # ── experimentation mocks (notebook cells 72 / 74 / 76) ──
    experiment_result_data = {
        "overall_effect": 0.0342,
        "overall_ci": (-0.0021, 0.0705),
        "overall_p_value": 0.065,
        "heterogeneity_detected": True,
        "heterogeneity_p_value": 0.023,
        "n_intersections": 4,
        "n_excluded": 0,
        "design_type": "independent",
        "metadata": {"protected_attributes": ["gender", "age_group"]},
        "intersection_effects": [
            {
                "intersection": ("Female", "Young"),
                "effect": 0.052,
                "effect_size_d": 0.35,
                "ci_lower": 0.01,
                "ci_upper": 0.094,
                "p_value": 0.015,
                "n_control": 200,
                "n_treatment": 210,
                "significant": True,
                "powered": True,
            },
            {
                "intersection": ("Female", "Old"),
                "effect": 0.041,
                "effect_size_d": 0.28,
                "ci_lower": -0.005,
                "ci_upper": 0.087,
                "p_value": 0.081,
                "n_control": 180,
                "n_treatment": 175,
                "significant": False,
                "powered": True,
            },
            {
                "intersection": ("Male", "Young"),
                "effect": 0.018,
                "effect_size_d": 0.12,
                "ci_lower": -0.03,
                "ci_upper": 0.066,
                "p_value": 0.462,
                "n_control": 250,
                "n_treatment": 240,
                "significant": False,
                "powered": True,
            },
            {
                "intersection": ("Male", "Old"),
                "effect": 0.028,
                "effect_size_d": 0.19,
                "ci_lower": -0.02,
                "ci_upper": 0.076,
                "p_value": 0.253,
                "n_control": 220,
                "n_treatment": 215,
                "significant": False,
                "powered": False,
            },
        ],
    }
    recommendation_data = {
        "decision": "EXTEND_EXPERIMENT",
        "confidence": 0.62,
        "reasoning": [
            "Treatment shows positive effect in Female-Young intersection (p=0.015)",
            "Overall effect not yet statistically significant (p=0.065)",
            "Heterogeneous effects detected across intersections (p=0.023)",
            "One intersection (Male-Old) is underpowered: need more data",
        ],
        "trade_offs": {
            "Fairness": "Treatment improves demographic parity by 15%",
            "Accuracy": "Minimal accuracy impact (-0.2%)",
            "Sample size": "Male-Old needs ~340 more observations",
        },
        "caveats": [
            "Small sample in Male-Old intersection may bias estimates",
            "Results may not generalise to other demographics",
            "Seasonal effects not yet controlled for",
        ],
    }
    power_data = [
        {
            "intersection": ("Female", "Young"),
            "power": 0.92,
            "required_n": 350,
            "is_powered": True,
            "n_control": 200,
            "n_treatment": 210,
            "effect_size": 0.35,
        },
        {
            "intersection": ("Female", "Old"),
            "power": 0.78,
            "required_n": 420,
            "is_powered": False,
            "n_control": 180,
            "n_treatment": 175,
            "effect_size": 0.28,
        },
        {
            "intersection": ("Male", "Young"),
            "power": 0.45,
            "required_n": 800,
            "is_powered": False,
            "n_control": 250,
            "n_treatment": 240,
            "effect_size": 0.12,
        },
        {
            "intersection": ("Male", "Old"),
            "power": 0.61,
            "required_n": 560,
            "is_powered": False,
            "n_control": 220,
            "n_treatment": 215,
            "effect_size": 0.19,
        },
    ]

    # ── the render map: name -> fn(explanation) -> svg ──
    return {
        "bias_audit": lambda e: bias_audit_to_svg(bias_report, explanation=e),
        "calibration_report": lambda e: calibration_report_to_svg(cal_report, explanation=e),
        "fairness_report_dashboard": lambda e: fairness_report_to_svg(fair_report, explanation=e),
        "cicd_pipeline": lambda e: cicd_pipeline_to_svg(
            validation_result=validation, test_results=tests, gate_decision=gate, explanation=e
        ),
        "radar_chart": lambda e: radar_chart_to_svg(fair_report, explanation=e),
        "disparity_heatmap": lambda e: disparity_heatmap_to_svg(fair_report, explanation=e),
        "metrics_bar_chart": lambda e: metrics_bar_chart_to_svg(fair_report, explanation=e),
        "group_comparison": lambda e: group_comparison_to_svg(
            fair_report, metric="positive_rate", explanation=e
        ),
        "effect_sizes": lambda e: effect_sizes_to_svg(fair_report, explanation=e),
        "confidence_intervals": lambda e: confidence_intervals_to_svg(fair_report, explanation=e),
        "reliability_diagram": lambda e: reliability_diagram_to_svg(
            d["cal_y_true"], d["cal_y_prob"], n_bins=10, show_histogram=True, explanation=e
        ),
        "group_calibration": lambda e: group_calibration_to_svg(
            d["cal_y_true"], d["cal_y_prob"], d["cal_protected"], n_bins=10, explanation=e
        ),
        "calibration_disparity": lambda e: calibration_disparity_to_svg(
            disparity_result, explanation=e
        ),
        "pareto_frontier": lambda e: pareto_frontier_to_svg(
            cal_errors,
            fair_violations,
            labels=model_labels,
            x_label="Calibration Error (ECE)",
            y_label="Fairness Violation (DP Diff)",
            explanation=e,
        ),
        "correlation_heatmap": lambda e: correlation_heatmap_to_svg(
            corr_matrix, threshold=0.3, explanation=e
        ),
        "correlation_matrix": lambda e: correlation_matrix_to_svg(
            corr_data, methods=corr_methods, threshold=0.5, explanation=e
        ),
        "proxy_risk": lambda e: proxy_risk_to_svg(proxy_vars, explanation=e),
        "transformation_comparison": lambda e: transformation_comparison_to_svg(
            before, after, threshold=0.3, explanation=e
        ),
        "intersectional_analysis": lambda e: intersectional_analysis_to_svg(
            intersect_data, feature="loan_approval_rate", explanation=e
        ),
        "intersectional_disparity": lambda e: intersectional_disparity_to_svg(
            intersect_disparity_data, explanation=e
        ),
        "training_report": lambda e: training_report_to_svg(training_report, explanation=e),
        "training_analysis_report": lambda e: training_analysis_report_to_svg(
            training_report, explanation=e
        ),
        "method_comparison": lambda e: method_comparison_to_svg(method_comparisons, explanation=e),
        "tradeoff_analysis": lambda e: tradeoff_analysis_to_svg(tradeoff, explanation=e),
        "threshold_optimization_report": lambda e: threshold_optimization_to_svg(
            threshold_result, explanation=e
        ),
        "reweighting_comparison_report": lambda e: reweighting_comparison_to_svg(
            reweighting_report, explanation=e
        ),
        "fairness_detailed_report": lambda e: fairness_detailed_report_to_svg(
            detailed_report, explanation=e
        ),
        "monitoring_dashboard": lambda e: monitoring_dashboard_to_svg(latest_window, explanation=e),
        "drift_report": lambda e: drift_report_to_svg(mock_drift, explanation=e),
        "alert_timeline": lambda e: alert_timeline_to_svg(monitoring_history, explanation=e),
        "temporal_analysis": lambda e: temporal_analysis_to_svg(
            mock_temporal, metric_names=["dp_diff", "eo_diff"], explanation=e
        ),
        "experiment_results": lambda e: experiment_results_to_svg(
            experiment_result_data, explanation=e
        ),
        "experiment_recommendation": lambda e: experiment_recommendation_to_svg(
            recommendation_data, experiment_result_data, explanation=e
        ),
        "power_analysis": lambda e: power_analysis_to_svg(power_data, mde=0.15, explanation=e),
        "causal_decomposition": lambda e: causal_decomposition_to_svg(example=True, explanation=e),
        "workflow_overview": lambda e: workflow_overview_to_svg(example=True, explanation=e),
        "hierarchical_gate": lambda e: hierarchical_gate_to_svg(example=True, explanation=e),
        "report_card": lambda e: report_card_to_svg(
            model_name="loan_approval_xgb_v2", example=True, explanation=e
        ),
        "robustness_testing": lambda e: robustness_testing_to_svg(example=True, explanation=e),
        "ranking_fairness": lambda e: ranking_fairness_to_svg(example=True, explanation=e),
        "data_validation": lambda e: data_validation_to_svg(example=True, explanation=e),
        "auto_discovery": lambda e: auto_discovery_to_svg(example=True, explanation=e),
        "regression_fairness": lambda e: regression_fairness_to_svg(example=True, explanation=e),
        "reporting_dashboard_executive": lambda e: reporting_dashboard_to_svg(
            tier="executive", example=True, explanation=e
        ),
        "reporting_dashboard_operational": lambda e: reporting_dashboard_to_svg(
            tier="operational", example=True, explanation=e
        ),
        "reporting_dashboard_technical": lambda e: reporting_dashboard_to_svg(
            tier="technical", example=True, explanation=e
        ),
    }


def main() -> int:
    d = build_data()
    specs = build_specs(d)
    OUT.mkdir(parents=True, exist_ok=True)

    written, failed, leaks = 0, [], []
    for name, fn in specs.items():
        try:
            with_expl = fn(None)  # None -> render_svg auto-generates the explanation
            no_expl = fn("")  # "" -> explanation panel suppressed
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}\n{traceback.format_exc()}")
            continue
        variants = {
            f"{name}.svg": with_expl,
            f"{name}_no_explanation.svg": no_expl,
        }
        for fname, svg in variants.items():
            (OUT / fname).write_text(svg, encoding="utf-8")
            low = svg.lower()
            present = [h for h in _LEAK if h in low]
            if present:
                leaks.append(f"{fname}: {', '.join(present)}")
            written += 1

    print(f"\nWrote {written} files ({len(specs)} templates x 2 variants, all Blanco).")
    if failed:
        print(f"FAILED templates ({len(failed)}): {', '.join(failed)}")
    if leaks:
        print("Blanco leak check FAILED:")
        print("\n".join("  " + x for x in leaks))
    if not failed and not leaks:
        print("All templates rendered; Blanco leak check clean.")
    return 1 if (failed or leaks) else 0


if __name__ == "__main__":
    sys.exit(main())
