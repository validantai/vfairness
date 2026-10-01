"""Every public enum in the library, pinned: its members, its values, and in
particular its ability to SAY "could not check".

Why this file exists. 36 public enums had never been touched by an assertion, and
line coverage structurally cannot notice: an enum body executes once at import,
before pytest-cov starts measuring, so these classes read as "never executed" in
docs/suite-coverage.json no matter how much of the library uses them. The only way
to put a recorded result behind them is to assert on them directly.

Two properties are pinned here.

1. THE MEMBER SET AND THE WIRE VALUES. These enums cross process boundaries: values
   land in report JSON, in gate payloads and in the platform's database. Renaming a
   member or editing a ``.value`` string is an API break for every consumer that
   compares against the string, and nothing else in the suite would have caught it.

2. THE THIRD STATE. The library's rule is three states, never two: measured,
   failed, could-not-check. An outcome enum that offers only PASS and FAIL forces
   every caller to collapse the third into one of the other two, which is how a
   could-not-measure gets published as a clean pass. The enums that carry an
   explicit third state carry it ON PURPOSE, so removing one is a regression in the
   library's core contract and is pinned by name below.
"""

from __future__ import annotations

import enum
import importlib

import pytest

# (module, class name, [(member name, value), ...]) as of 2026-09-25.
PINNED_ENUMS = [
    (
        "vfairness.evaluation.vfairness_metrics._metric_direction",
        "MetricDirection",
        [
            ("LOWER_IS_BETTER", "lower_is_better"),
            ("HIGHER_IS_BETTER", "higher_is_better"),
            ("UNKNOWN", "unknown"),
        ],
    ),
    (
        "vfairness.evaluation.vfairness_metrics._metric_direction",
        "ThresholdOutcome",
        [("PASS", "pass"), ("FAIL", "fail"), ("COULD_NOT_CHECK", "could_not_check")],
    ),
    (
        "vfairness.evaluation.vfairness_metrics._statistics",
        "IntervalType",
        [("CONFIDENCE", "confidence"), ("CREDIBLE", "credible")],
    ),
    (
        "vfairness.in_processing.calibrators.group_calibrators",
        "CalibrationMethodType",
        [
            ("TEMPERATURE", "temperature"),
            ("PLATT", "platt"),
            ("BETA", "beta"),
            ("FOCAL", "focal"),
            ("HISTOGRAM", "histogram"),
        ],
    ),
    (
        "vfairness.in_processing.constraints.base",
        "FairnessConstraintType",
        [
            ("DEMOGRAPHIC_PARITY", "demographic_parity"),
            ("EQUALIZED_ODDS", "equalized_odds"),
            ("EQUAL_OPPORTUNITY", "equal_opportunity"),
            ("FALSE_POSITIVE_RATE_PARITY", "fpr_parity"),
            ("TRUE_POSITIVE_RATE_PARITY", "tpr_parity"),
            ("PREDICTIVE_PARITY", "predictive_parity"),
            ("BOUNDED_GROUP_LOSS", "bounded_group_loss"),
            ("ERROR_RATE_PARITY", "error_rate_parity"),
        ],
    ),
    (
        "vfairness.in_processing.loss_functions.base",
        "BaseLossType",
        [
            ("BINARY_CROSS_ENTROPY", "bce"),
            ("CROSS_ENTROPY", "ce"),
            ("MEAN_SQUARED_ERROR", "mse"),
            ("MEAN_ABSOLUTE_ERROR", "mae"),
            ("HINGE", "hinge"),
            ("FOCAL", "focal"),
        ],
    ),
    (
        "vfairness.in_processing.loss_functions.base",
        "FairnessMetricType",
        [
            ("DEMOGRAPHIC_PARITY", "demographic_parity"),
            ("EQUALIZED_ODDS", "equalized_odds"),
            ("EQUAL_OPPORTUNITY", "equal_opportunity"),
            ("PREDICTIVE_PARITY", "predictive_parity"),
            ("CALIBRATION", "calibration"),
            ("INDIVIDUAL_FAIRNESS", "individual_fairness"),
            ("COUNTERFACTUAL_FAIRNESS", "counterfactual_fairness"),
        ],
    ),
    (
        "vfairness.in_processing.regularizers.fairness_regularizers",
        "RegularizerType",
        [
            ("STATISTICAL_PARITY", "statistical_parity"),
            ("CONDITIONAL_INDEPENDENCE", "conditional_independence"),
            ("GROUP_FAIRNESS", "group_fairness"),
            ("HSIC", "hsic"),
            ("MUTUAL_INFORMATION", "mutual_information"),
            ("CORRELATION", "correlation"),
        ],
    ),
    (
        "vfairness.operations.cicd.gate",
        "GateStatus",
        [("APPROVED", "approved"), ("BLOCKED", "blocked"), ("CONDITIONAL", "conditional")],
    ),
    (
        "vfairness.operations.cicd.monitor",
        "AlertSeverity",
        [("INFO", "info"), ("WARNING", "warning"), ("CRITICAL", "critical")],
    ),
    (
        "vfairness.operations.cicd.monitor",
        "DriftType",
        [
            ("METRIC_DRIFT", "metric_drift"),
            ("DISTRIBUTION_DRIFT", "distribution_drift"),
            ("PERFORMANCE_DRIFT", "performance_drift"),
            ("NOT_MEASURABLE", "not_measurable"),
        ],
    ),
    (
        "vfairness.operations.cicd.testing",
        "TestStatus",
        [("PASSED", "passed"), ("FAILED", "failed"), ("SKIPPED", "skipped"), ("ERROR", "error")],
    ),
    (
        "vfairness.operations.cicd.validator",
        "ValidationSeverity",
        [("INFO", "info"), ("WARNING", "warning"), ("ERROR", "error"), ("CRITICAL", "critical")],
    ),
    (
        "vfairness.operations.experimentation.analysis",
        "RecommendationDecision",
        [
            ("DEPLOY_TREATMENT", "deploy_treatment"),
            ("KEEP_CONTROL", "keep_control"),
            ("EXTEND_EXPERIMENT", "extend_experiment"),
            ("INVESTIGATE_FURTHER", "investigate_further"),
        ],
    ),
    (
        "vfairness.operations.experimentation.experiment",
        "DesignType",
        [
            ("SIMPLE_AB", "simple_ab"),
            ("STRATIFIED", "stratified"),
            ("CLUSTER", "cluster"),
            ("FACTORIAL", "factorial"),
        ],
    ),
    (
        "vfairness.operations.experimentation.power",
        "SPRTDecision",
        [
            ("ACCEPT_NULL", "accept_null"),
            ("REJECT_NULL", "reject_null"),
            ("CONTINUE", "continue"),
            ("COULD_NOT_CHECK", "could_not_check"),
        ],
    ),
    (
        "vfairness.operations.reporting.dashboard",
        "TimeWindow",
        [
            ("LAST_24H", "24h"),
            ("LAST_7D", "7d"),
            ("LAST_30D", "30d"),
            ("LAST_90D", "90d"),
            ("CUSTOM", "custom"),
        ],
    ),
    (
        "vfairness.operations.reporting.reports",
        "OutputFormat",
        [("HTML", "html"), ("MARKDOWN", "markdown"), ("JSON", "json")],
    ),
    (
        "vfairness.operations.reporting.reports",
        "ReportTier",
        [("EXECUTIVE", 1), ("OPERATIONAL", 2), ("TECHNICAL", 3)],
    ),
    (
        "vfairness.operations.reporting.store",
        "PrivacyLevel",
        [
            ("SUPPRESSED", "suppressed"),
            ("NOISY", "noisy"),
            ("EXACT", "exact"),
            ("UNKNOWN_SIZE", "unknown_size"),
        ],
    ),
    (
        "vfairness.post_processing.calibration.methods",
        "CalibrationMethod",
        [
            ("PLATT", "platt"),
            ("ISOTONIC", "isotonic"),
            ("BETA", "beta"),
            ("TEMPERATURE", "temperature"),
            ("HISTOGRAM", "histogram"),
        ],
    ),
    (
        "vfairness.post_processing.reweighting.reweighter",
        "ReweightingMethod",
        [
            ("MULTIPLICATIVE", "multiplicative"),
            ("ADDITIVE", "additive"),
            ("REJECTION_OPTION", "rejection_option"),
            ("DISTRIBUTION_MATCHING", "distribution_matching"),
            ("CALIBRATED_EQUALIZATION", "calibrated_equalization"),
        ],
    ),
    (
        "vfairness.post_processing.threshold_optimization.constraints",
        "FairnessConstraintType",
        [
            ("DEMOGRAPHIC_PARITY", "demographic_parity"),
            ("EQUALIZED_ODDS", "equalized_odds"),
            ("EQUAL_OPPORTUNITY", "equal_opportunity"),
            ("PREDICTIVE_PARITY", "predictive_parity"),
            ("FALSE_POSITIVE_PARITY", "false_positive_parity"),
            ("CALIBRATION", "calibration"),
            ("CUSTOM", "custom"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.geographic_data",
        "HOLCGrade",
        [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("UNKNOWN", "Unknown")],
    ),
    (
        "vfairness.preprocessing.bias_detection.historical",
        "HistoricalRiskLevel",
        [
            ("CRITICAL", "critical"),
            ("HIGH", "high"),
            ("MEDIUM", "medium"),
            ("LOW", "low"),
            ("NONE", "none"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.proxy",
        "ProxyRiskLevel",
        [
            ("CRITICAL", "critical"),
            ("HIGH", "high"),
            ("MEDIUM", "medium"),
            ("LOW", "low"),
            ("NEGLIGIBLE", "negligible"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.proxy",
        "ProxyType",
        [
            ("DIRECT", "direct"),
            ("INDIRECT", "indirect"),
            ("INTERSECTIONAL", "intersectional"),
            ("HISTORICAL", "historical"),
            ("UNCLASSIFIED", "unclassified"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.representation",
        "RepresentationSeverity",
        [
            ("CRITICAL", "critical"),
            ("HIGH", "high"),
            ("MEDIUM", "medium"),
            ("LOW", "low"),
            ("ADEQUATE", "adequate"),
            ("OVERREPRESENTED", "overrepresented"),
            ("INSUFFICIENT_DATA", "insufficient_data"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.statistical",
        "DisparityType",
        [
            ("OUTCOME", "outcome"),
            ("DISTRIBUTION", "distribution"),
            ("QUALITY", "quality"),
            ("INTERSECTIONAL", "intersectional"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.statistical",
        "EffectSizeInterpretation",
        [
            ("NEGLIGIBLE", "negligible"),
            ("SMALL", "small"),
            ("MEDIUM", "medium"),
            ("LARGE", "large"),
            ("NOT_MEASURABLE", "not_measurable"),
        ],
    ),
    (
        "vfairness.preprocessing.bias_detection.statistical",
        "SignificanceLevel",
        [
            ("HIGHLY_SIGNIFICANT", "highly_significant"),
            ("SIGNIFICANT", "significant"),
            ("MARGINALLY_SIGNIFICANT", "marginally_significant"),
            ("NOT_SIGNIFICANT", "not_significant"),
            ("NOT_TESTABLE", "not_testable"),
        ],
    ),
    (
        "vfairness.preprocessing.feature_engineering.correlation",
        "CorrelationType",
        [
            ("PEARSON", "pearson"),
            ("SPEARMAN", "spearman"),
            ("CRAMERS_V", "cramers_v"),
            ("MUTUAL_INFORMATION", "mutual_information"),
            ("POINT_BISERIAL", "point_biserial"),
        ],
    ),
    (
        "vfairness.preprocessing.feature_engineering.correlation",
        "ProxyRiskLevel",
        [
            ("CRITICAL", "critical"),
            ("HIGH", "high"),
            ("MEDIUM", "medium"),
            ("LOW", "low"),
            ("NEGLIGIBLE", "negligible"),
        ],
    ),
    (
        "vfairness.preprocessing.feature_engineering.correlation",
        "ProxyType",
        [
            ("DIRECT", "direct"),
            ("INDIRECT", "indirect"),
            ("INTERSECTIONAL", "intersectional"),
            ("HISTORICAL", "historical"),
            ("UNCLASSIFIED", "unclassified"),
        ],
    ),
    (
        "vfairness.preprocessing.feature_engineering.transformers",
        "FairnessObjective",
        [
            ("DEMOGRAPHIC_PARITY", "demographic_parity"),
            ("EQUALIZED_ODDS", "equalized_odds"),
            ("EQUAL_OPPORTUNITY", "equal_opportunity"),
            ("INDIVIDUAL_FAIRNESS", "individual_fairness"),
            ("COUNTERFACTUAL_FAIRNESS", "counterfactual_fairness"),
        ],
    ),
    (
        "vfairness.preprocessing.feature_engineering.transformers",
        "TransformationMethod",
        [
            ("CORRELATION_REDUCTION", "correlation_reduction"),
            ("FAIR_REPRESENTATION", "fair_representation"),
            ("FEATURE_SUPPRESSION", "feature_suppression"),
            ("RESIDUALIZATION", "residualization"),
            ("REWEIGHTING", "reweighting"),
            ("INTERSECTIONAL", "intersectional"),
        ],
    ),
]


# The members that exist so a surface can say "I could not check this". Each one is
# load-bearing at a named call site, so deleting it must break a test here rather
# than quietly narrow what the library is able to report.
THIRD_STATE_MEMBERS = [
    (
        "vfairness.evaluation.vfairness_metrics._metric_direction",
        "ThresholdOutcome",
        "COULD_NOT_CHECK",
    ),
    ("vfairness.evaluation.vfairness_metrics._metric_direction", "MetricDirection", "UNKNOWN"),
    ("vfairness.operations.experimentation.power", "SPRTDecision", "COULD_NOT_CHECK"),
    ("vfairness.operations.cicd.testing", "TestStatus", "SKIPPED"),
    ("vfairness.operations.cicd.testing", "TestStatus", "ERROR"),
    ("vfairness.operations.reporting.store", "PrivacyLevel", "UNKNOWN_SIZE"),
    (
        "vfairness.preprocessing.bias_detection.representation",
        "RepresentationSeverity",
        "INSUFFICIENT_DATA",
    ),
    ("vfairness.preprocessing.bias_detection.statistical", "SignificanceLevel", "NOT_TESTABLE"),
    ("vfairness.preprocessing.bias_detection.geographic_data", "HOLCGrade", "UNKNOWN"),
]


def _load(module: str, name: str):
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("module,name,members", PINNED_ENUMS, ids=[r[1] for r in PINNED_ENUMS])
def test_enum_member_set_and_values_are_the_published_contract(module, name, members):
    """The member names and their wire values are exactly as published."""
    cls = _load(module, name)
    assert issubclass(cls, enum.Enum)
    assert [(m.name, m.value) for m in cls] == [tuple(x) for x in members], (
        f"{name} changed. Its values travel in report JSON and gate payloads, so a "
        f"rename or reordering is an API break for consumers comparing the string."
    )


@pytest.mark.parametrize("module,name,members", PINNED_ENUMS, ids=[r[1] for r in PINNED_ENUMS])
def test_enum_values_are_unique_and_round_trip(module, name, members):
    """Lookup by value returns the member it came from, for every member."""
    cls = _load(module, name)
    values = [m.value for m in cls]
    assert len(values) == len(set(values)), f"{name} has duplicate values: {values}"
    for member in cls:
        assert cls(member.value) is member
        assert cls[member.name] is member


@pytest.mark.parametrize(
    "module,name,member", THIRD_STATE_MEMBERS, ids=[f"{r[1]}.{r[2]}" for r in THIRD_STATE_MEMBERS]
)
def test_the_could_not_check_state_still_exists(module, name, member):
    """Removing a third state would force callers to collapse it into pass or fail."""
    cls = _load(module, name)
    assert member in cls.__members__, (
        f"{name}.{member} is gone. It exists so a surface can report that it could "
        f"not check something; without it the caller has only success and failure, "
        f"and a could-not-measure gets published as one of them."
    )


def test_every_pinned_enum_is_reachable_from_its_module():
    """A guard on this file itself: the table cannot drift into naming nothing."""
    for module, name, _members in PINNED_ENUMS:
        assert isinstance(_load(module, name), type)
    assert len(PINNED_ENUMS) >= 36, "enums disappeared from the pinned table"
