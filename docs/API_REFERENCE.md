# vfairness API Reference

A complete guide to using the vfairness library for measuring fairness in machine learning models and detecting bias.

---

## Library Structure

vfairness is organized into 15 top-level public sub-packages: six core pipeline
stages (`preprocessing`, `in_processing`, `post_processing`, `evaluation`,
`operations`, `rendering`), eight specialized analysis surfaces (`llm`, `agents`,
`multi_agent`, `xai`, `vision`, `legal`, `mcp`, `validity`) and one cross-cutting
infrastructure package (`net`). The canonical set is enforced by
`tests/test_module_taxonomy.py`; changing the layout must update that test in the
same commit.

The block below shows the main entry points in pipeline order, not every
sub-package.

```python
# 1. Preprocessing - Feature engineering and bias detection
from vfairness.preprocessing import FeatureEngineeringAnalyzer, BiasDetector

# 2. In-Processing - Training-time interventions
from vfairness.in_processing import (
    FairClassifier, FairRegressor,
    FairnessTrainingAnalyzer, FairnessTrainingReport,
    DemographicParityLoss, EqualizedOddsLoss, EqualOpportunityLoss,
    AdversarialDebiasingLoss, CounterfactualFairnessLoss,
    ExponentiatedGradient, GridSearch, ThresholdOptimizer,
    StatisticalParityRegularizer, HilbertSchmidtRegularizer,
    TrainableGroupCalibrator, CalibrationAwareTrainer,
)

# 3. Post-Processing - Prediction-time interventions
from vfairness.post_processing import (
    GroupCalibrator, expected_calibration_error,
    ThresholdOptimizer, GroupThresholdOptimizer, ThresholdAnalyzer,
    PredictionReweighter, RejectionOptionClassifier, ReweightingAnalyzer,
)

# 4. Evaluation - Fairness metrics and analysis
from vfairness.evaluation import FairnessAnalyzer, demographic_parity_difference

# 5. Operations - CI/CD and monitoring
from vfairness.operations import DataBiasValidator, ModelFairnessGate

# 6. Rendering - SVG report engine
from vfairness.rendering import render_svg

# ─── NEW: LLM, Agent, and Multi-Agent Fairness ───

# 7. LLM Fairness Testing (no training data needed)
from vfairness.llm import (
    LLMApiProxy,                  # Connect to any LLM API
    CounterfactualTester,         # Demographic swap testing (9 strategies)
    OutputAnalyzer,               # Sentiment, toxicity, refusal, length analysis
    BenchmarkRunner,              # BBQ, BOLD standardized benchmarks
    DecodingTrustRunner,          # DecodingTrust: 8 trustworthiness dimensions (Wang et al. 2023)
    NonDeterminismAnalyzer,       # Noise offset (separates real bias from randomness)
    IntersectionalAnalyzer,       # Multi-attribute intersection testing
    CoTFaithfulnessAnalyzer,      # Chain-of-thought reasoning audit
)

# 8. Agent Fairness Testing (tool use, RAG, actions)
from vfairness.agents import (
    CorrespondenceTester,         # Paired artifact testing (gold standard)
    ToolBiasAuditor,              # Tool selection fairness audit
    RAGBiasAnalyzer,              # Retrieval bias detection
    PipelineTracker,              # Multi-stage bias tracking
    TemporalTracker,              # Drift detection (CUSUM/EWMA)
    ActionBiasAnalyzer,           # Outcome & delegation bias
)

# 9. Multi-Agent Fairness Testing (emergent bias)
from vfairness.multi_agent import (
    CompositionalityAnalyzer,     # Component vs system bias comparison
    GroupthinkDetector,           # Echo-chamber & coalition detection
    EmergentBiasDetector,         # Novel bias from agent interaction
)

# ─── NEW (v0.0.8): XAI / Explainability ───

# 10. XAI / Explainability (SHAP family, Lundberg decomposition, DiCE)
from vfairness.xai import (
    Explanation, FairnessDecomposition, XaiAssessment, TrustPosture,
    route_explainer,                       # Mechanical routing -- agrees with platform TS + Postgres
    lundberg_fairness_decomposition,       # The spine; asserts 1e-6 identity before returning
)
from vfairness.xai.explainers import (
    TreeShapExplainer,                     # Exact, synchronous (tree ensembles)
    LinearShapExplainer,                   # Closed-form (linear / GLM)
    KernelShapExplainer,                   # Model-agnostic, async-eligible
)
from vfairness.xai.decomposition import proxy_score
from vfairness.xai.diagnostics import (
    removal_curve_auc,                     # ROAR-style faithfulness
    local_r_squared,                       # LIME surrogate fidelity
    attribution_stability,                 # σ over reruns
    slack_adversarial_probe,               # Slack et al. 2020 OOD-scaffold detector
)
from vfairness.xai.storage import (
    SupabaseWriter,                        # RLS-bypass service-role writer
    build_audit_artifact_bundle,           # Deterministic JSON + sha256
)
from vfairness.xai.worker import WorkerLoop  # pgmq polling consumer (signal-safe)

# ─── NEW: Validity / Groundedness (generative source-fidelity, VA-10/VA-21) ───

# 11. Validity axis for generative systems (is the answer grounded in its sources?)
from vfairness.validity import (
    GroundednessScorer,            # Fail-closed ladder: owned sidecar -> LLM judge -> REFUSE
    GroundednessResult,            # Context-aware per-answer verdict (VG-001..007)
    LlmGroundednessJudge,          # Interim self-hosted judge rung (Mistral Small 3.2)
    aggregate_validity,            # Fail-closed VG-* batch summary (only measured records)
    groundedness_scorer_status,    # Active rung + quality tier
)
# Task type dispatched from the platform:
#   python -m vfairness.operations.validity.task_handlers vfairness_validity_run

# ─── Cross-cutting infrastructure ───

# SSRF egress guard: applied at every outbound call site in the library
from vfairness.net import validate_endpoint, guarded_post, SSRFError

# Flat imports also work
from vfairness import FairnessAnalyzer, BiasDetector, GroupCalibrator
from vfairness import CounterfactualTester, OutputAnalyzer, DecodingTrustRunner, CorrespondenceTester
```

---

## Table of Contents

### Preprocessing Module
1. [BiasDetector Class](#biasdetector-class)
2. [Historical Pattern Detection](#historical-pattern-detection)
3. [Representation Bias Detection](#representation-bias-detection)
4. [Statistical Disparity Analysis](#statistical-disparity-analysis)
5. [Proxy Variable Identification](#proxy-variable-identification)

### In-Processing Module (Training-Time Interventions)
6. [In-Processing Overview](#in-processing-overview)
7. [FairClassifier and FairRegressor](#fairclassifier-and-fairregressor)
8. [Fairness-Aware Loss Functions](#fairness-aware-loss-functions)
9. [Constraint-Based Training](#constraint-based-training)
10. [Fairness Regularizers](#fairness-regularizers)
11. [Group-Specific Calibrators](#group-specific-calibrators)
12. [FairnessTrainingAnalyzer](#fairnesstraininganalyzer)

### Post-Processing Module
13. [Threshold Optimization](#threshold-optimization)
14. [Reweighting Methods](#reweighting-methods)

### Evaluation Module
15. [Quick Start](#quick-start)
16. [Core Concepts](#core-concepts)
17. [FairnessAnalyzer Class](#fairnessanalyzer-class)
18. [FairExplAIner Mode](#fairexplainer-mode)
19. [Intersectional Group Analysis](#intersectional-group-analysis)
20. [Auto-Discovery Features](#auto-discovery-features)
21. [Statistical Significance and Robustness](#statistical-significance-and-robustness)
22. [Classification Metrics](#classification-metrics)
23. [Regression Metrics](#regression-metrics)
24. [Ranking Metrics](#ranking-metrics)
25. [Statistical Validation](#statistical-validation)
26. [Fairness Reports](#fairness-reports)
27. [MLOps Integration](#mlops-integration)
28. [Visualization](#visualization)

### Operations Module
29. [Causal Operations](#module-10-causal-operations-vfairnessoperationscausal)

### Validity / Groundedness Module
### LLM Module (additions)
- [HolisticBias Benchmark](#holisticbias-benchmark)
- [New Output Scorers](#new-output-scorers) — regard, LLM-as-judge, information quality, representation, framing

### Rendering Module
30. [SVG Report Engine](#rendering-module)
31. [Training Report Adapters](#training-report-adapters)
32. [Post-Processing SVG Templates](#threshold-optimization-svg)

### Common Topics
33. [Three States, Never Two](#three-states-never-two)
34. [Common Parameters](#common-parameters)
35. [Handling Special Cases](#handling-special-cases)
36. [Examples](#examples)

---

## Three States, Never Two

Read this before anything else in this reference. It governs what every return
value, report field and chart on this page means.

Every result surface in vfairness has **three** states, not two:

| State | Meaning |
|-------|---------|
| assessed-pass | The comparison ran on sufficient data and came out within the threshold. |
| assessed-fail | The comparison ran on sufficient data and came out over the threshold. |
| could-not-check | The comparison never ran. Nothing was measured, so nothing is certified. |

**Could-not-check is never collapsed into either of the other two.** A metric
that could not be computed is not a pass, and it is not a failure either. It gets
no number, no badge, no colour, and no place in any count or headline that implies
a measurement was taken.

This matters in both directions. A fabricated all-clear tells a reader a group
was checked and cleared when it never was. A fabricated breach reports a
violation nobody measured, sending someone after a problem that does not exist.
Both are wrong for the same reason: a substituted default was graded as though it
were a measurement.

How each surface expresses the third state:

| Surface | could-not-check looks like |
|---------|---------------------------|
| Classification and regression metric functions | Return `float('nan')`, never `0.0` or `1.0`. |
| `*_with_ci` variants | A `StatisticalResult` whose `point_estimate`, `lower_bound` and `upper_bound` are all NaN. |
| `report["assessment"]["fairness_score"]` | `None`. Never `0.0`, which means "measured, and every metric failed". |
| `report["assessment"]["assessable"]` | `False` when fewer than two groups survive `min_group_size`. |
| `report["assessment"]["not_assessable_metrics"]` | Each ungraded metric, with `status: "NOT_ASSESSABLE"`, `threshold: None` and a `reason`. It is listed here, never under `passed_metrics`. |
| `report["assessment"]["insufficient_evidence_groups"]` | Each group dropped below `min_group_size`, with its `n`, `tier` and `reason`. |
| `report["assessment"]["summary"]` | Opens with `NOT ASSESSABLE:` and always ends with a data-provenance clause. |
| SVG adapters (`vfairness.rendering`) | A slate **NOT CHECKED** panel drawn instead of the verdict stack, closing with "This chart is not a pass and not a failure. No value on it was measured, so it certifies nothing." |
| `assert_fairness` | Raises `FairnessAssertionError` and reports the metric as `not_measurable`. It fails closed. <!--cs-->**[Checked]**<!--/cs--> |
| `ModelFairnessGate.evaluate` | `GateStatus.BLOCKED` with a blocking reason naming the unevaluated metric. It fails closed. <!--cs-->**[Checked]**<!--/cs--> |
Two consequences for your own code:

```python
# WRONG: crashes on the third state, and formats None as though it were measured.
print(f"{report['assessment']['fairness_score']:.1%}")

# RIGHT: branch on the third state first.
score = report["assessment"]["fairness_score"]
if not report["assessment"]["assessable"] or score is None:
    print("COULD NOT CHECK:", report["assessment"]["summary"])
else:
    print(f"Fairness score: {score:.1%}")
```

```python
# WRONG: every comparison against NaN is False, so this guard never fires and
# a metric that was never measured slides through as though it had passed.
if dp > 0.1:
    block_deployment()

# RIGHT: rule out the third state before grading, in either direction.
import math
if math.isnan(dp):
    escalate("demographic parity was never measured on this data")
elif dp > 0.1:
    block_deployment()
```

**Known gap: the ranking metrics do not yet implement the third state.**
`exposure_parity_difference`, `attention_weighted_rank_fairness` and
`normalized_discounted_kl_divergence` return `0.0`, and `exposure_parity_ratio`
returns `1.0`, when fewer than two groups survive `min_group_size`. Those are the
perfect-parity readings, returned for a comparison that never ran. Until this is
closed, check the group sizes yourself before trusting a ranking result: see
[Ranking Metrics](#ranking-metrics). Verified by execution on 2026-08-28.

---

## Quick Start

```python
import numpy as np
from vfairness import demographic_parity_difference, classification_fairness_report

# Your model predictions
y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # Actual outcomes
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0])  # Model predictions
gender = np.array(['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'])  # Protected attribute

# Compute a single metric
dp = demographic_parity_difference(y_true, y_pred, gender, min_group_size=2)
print(f"Demographic Parity Difference: {dp:.3f}")

# Or generate a full report
report = classification_fairness_report(y_true, y_pred, gender, min_group_size=2)

# fairness_score is Optional[float]: None is the third state, "could not check".
score = report["assessment"]["fairness_score"]
if score is None:
    print("Fairness Score: COULD NOT CHECK -", report["assessment"]["summary"])
else:
    print(f"Fairness Score: {score:.1%}")
```

See [Three States, Never Two](#three-states-never-two) for why the `None` branch
is not optional.

---

## Core Concepts

### What is Fairness?

A model is "fair" when it treats different groups equally. But "equally" can mean different things:

| Concept | Question it Answers |
|---------|---------------------|
| **Demographic Parity** | Does each group get positive predictions at the same rate? |
| **Equal Opportunity** | Among people who deserve a positive outcome, does each group get it at the same rate? |
| **Equalized Odds** | Are error rates (false positives AND false negatives) equal across groups? |
| **Predictive Parity** | When the model says "yes", is it equally accurate for all groups? |

### Key Terms

- **Sensitive/Protected Attribute**: The characteristic defining groups (e.g., gender, race, age)
- **Positive Prediction**: Model predicts the favorable outcome (e.g., approved for loan)
- **True Positive Rate (TPR)**: Among actual positives, what fraction did we correctly predict?
- **False Positive Rate (FPR)**: Among actual negatives, what fraction did we incorrectly predict as positive?

---

## FairnessAnalyzer Class
<!-- cap-status: FairnessAnalyzer -->
**Beta status: Checked.** 15 code units behind this name: 15 checked
<!-- /cap-status -->


The `FairnessAnalyzer` is a unified wrapper class that provides a single interface for all fairness computations.

### Basic Usage

```python
from vfairness import FairnessAnalyzer

# Create analyzer
analyzer = FairnessAnalyzer(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_attr=gender,
    task_type='classification',  # or 'regression', auto-detected if None
    min_group_size=30
)

# Compute individual metrics
dp = analyzer.demographic_parity_difference()
eo = analyzer.equal_opportunity_difference()

# Compute all metrics at once
all_metrics = analyzer.compute_all_metrics()

# Generate comprehensive report
report = analyzer.get_report()
```

### With Confidence Intervals

```python
# Get metric with 95% bootstrap confidence interval
result = analyzer.demographic_parity_difference(
    include_ci=True,
    n_bootstrap=5000,
    confidence_level=0.95,
    random_state=42
)

print(f"Value: {result.value:.3f}")
ci_low, ci_high = result.confidence_interval
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Effect Size (Cohen's d): {result.effect_size:.3f}")
```

### MetricResult Structure
<!-- cap-status: MetricResult -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


When `include_ci=True`, methods return a `MetricResult` object:

```python
@dataclass
class MetricResult:
    metric_name: str                  # Name of the metric
    value: float                      # The metric value
    confidence_interval: tuple        # (lower, upper) CI bounds, or None
    effect_size: float                # Cohen's d/h effect size, or None
    effect_interpretation: str        # e.g. 'negligible', 'small', ...
    group_sizes: dict                 # Sample sizes per group
    is_fair: bool                     # back-compat boolean == (verdict == 'fair')
    threshold: float                  # Threshold used for the verdict band
    verdict: str                      # 'fair' | 'unfair' | 'insufficient_evidence'
```

**The verdict is computed from the confidence interval, not the point estimate.** A metric
is `fair` only when the *whole* CI sits inside the `[0, threshold]` band (affirmative
evidence of practical equivalence), `unfair` only when the whole CI exceeds it, and
`insufficient_evidence` otherwise (the CI straddles the boundary — a wide small-sample
interval is never reported as a pass or a fail). `is_fair` remains as a boolean alias for
`verdict == 'fair'`. This makes the pass/fail statistically honest: a 0.12 gap with a wide
CI on n=40 is `insufficient_evidence`, not `unfair`, and a 0.09 gap with a tight CI on
n=50k is `fair`.

### FairnessReport Structure
<!-- cap-status: FairnessReport -->
**Beta status: Checked.** returns no fairness number and no verdict, so it cannot fabricate one
<!-- /cap-status -->


`get_report()`, `classification_fairness_report()` and `regression_fairness_report()`
return a plain, JSON-serialisable `dict`. Its key structure is documented and
type-checked by `FairnessReport`, a `TypedDict` you can import and annotate with:

```python
from vfairness.evaluation import FairnessReport, AssessmentReport
```

`TypedDict` is a static-only annotation. The value is exactly the `dict` it has
always been, so `report["assessment"]["summary"]`, `json.dumps(report)`, and any
existing consumer keep working unchanged; a type checker now also catches a
mistyped key such as `report["assessement"]`.

**Always present (required keys):**

| Key | Type | Meaning |
|-----|------|---------|
| `task_type` | `str` | `"classification"` or `"regression"` |
| `methodology_version` | `str` | Version stamp of the scoring methodology |
| `metrics` | `dict[str, float]` | Metric name to point estimate |
| `group_stats` | `dict[str, Any]` | Per-group summary statistics |
| `assessment` | `AssessmentReport` | The verdict block (see below) |
| `data_info` | `DataInfo` | Dataset and grouping context |
| `thresholds_used` | `dict[str, float]` | Threshold applied per metric |

**Conditional keys (present only when their feature is requested / for a task type):**

| Key | Present when |
|-----|--------------|
| `residual_bias` | regression reports only <!--cs-->**[Checked]**<!--/cs--> |
| `metrics_with_ci`, `effect_sizes`, `statistical_validation` | `include_ci=True` |
| `explanations` | FairExplAIner is enabled |

**`report["assessment"]` (`AssessmentReport`):**

```python
class AssessmentReport(TypedDict):
    fairness_score: Optional[float]          # fraction of ASSESSABLE metrics within threshold,
                                             # or None when nothing was assessable
    assessable: bool                         # False on degenerate data (< 2 valid groups)
    passed_metrics: list[MetricStatusEntry]
    failed_metrics: list[MetricStatusEntry]
    not_assessable_metrics: list[MetricStatusEntry]
    insufficient_evidence_groups: list[InsufficientEvidenceGroup]
    summary: str
```

All seven keys are required and always present.

**`fairness_score` is `Optional[float]`, and `None` is a distinct state.** It is
the third state, could-not-check, and it is not the same as `0.0` (measured, and
every metric failed). It must never be rendered as a percentage. `None` appears
when `assessable` is `False`, or when no metric produced a threshold verdict at
all. Branch on it before formatting; see
[Three States, Never Two](#three-states-never-two).

**A metric that could not be graded is listed under `not_assessable_metrics`,
never under `passed_metrics`.** Its entry carries `status: "NOT_ASSESSABLE"`,
`threshold: None` and a `reason` string explaining why nothing was compared. It
is excluded from `fairness_score`, and it is not reported as a failure either.

**`summary` always ends with a data-provenance clause** of the form
`(data provenance: <final> of <original> rows assessed, <n> excluded,
missing_strategy='<strategy>')`, on every report including a NOT ASSESSABLE one,
so the verdict line itself discloses how the rows behind it were handled. Treat
`summary` as human-readable prose and read `data_info` for the values.

Observed on a run where 15 of 120 rows fell in a group below `min_group_size=30`,
leaving one valid group:

```python
{
    "fairness_score": None,
    "assessable": False,
    "passed_metrics": [],
    "failed_metrics": [],
    "not_assessable_metrics": [
        {
            "metric": "demographic_parity_difference",
            "value": float("nan"),
            "threshold": None,
            "status": "NOT_ASSESSABLE",
            "reason": "1 valid group(s) after filtering (need at least 2); no "
                      "between-group comparison was performed, so this metric "
                      "certifies nothing",
        },
        # ... one entry per metric
    ],
    "insufficient_evidence_groups": [
        {
            "group": "B",
            "n": 15,
            "tier": "underpowered",
            "verdict": "insufficient_evidence",
            "reason": "n=15 is below min_group_size=30; this group is excluded "
                      "from the disparity metrics, so there is insufficient "
                      "evidence to assess it. ...",
        }
    ],
    "summary": "NOT ASSESSABLE: 1 valid group(s) after filtering (need at least "
               "2): disparity metrics are vacuous and do not certify fairness. "
               "... (data provenance: 120 of 120 rows assessed, 0 excluded, "
               "missing_strategy='exclude')",
}
```

Each `MetricStatusEntry` is `{metric, value, threshold, status}`, where `status`
is `"PASS"` or `"FAIL"` for a graded metric (and `threshold` a float), or
`"NOT_ASSESSABLE"` for an ungraded one (and `threshold` is `None`).

**`report["data_info"]` (`DataInfo`):**

Declared `total=False`, so the exact key set depends on the task and on
preprocessing. The keys produced for every classification and regression report:

| Key | Type | Meaning |
|-----|------|---------|
| `original_size` | `int` | Rows that arrived |
| `final_size` | `int` | Rows actually assessed |
| `n_excluded` | `int` | Rows dropped before assessment |
| `missing_strategy` | `str` | `'exclude'`, `'as_group'` or `'error'` |
| `n_samples` | `int` | Sample count behind the metrics |
| `n_groups` | `int` | Groups seen in `sensitive_attr` |
| `valid_groups` | `list` | Groups at or above `min_group_size`, which the metrics compared |
| `invalid_groups` | `list` | Groups below `min_group_size`, which no metric compared |
| `group_sizes` | `dict` | Row count per group, valid and invalid alike |
| `is_intersectional` | `bool` | Whether the grouping is a tuple of attributes |

`y_std` is added for regression reports only.

The first four are the authoritative missing-data provenance of the run: how many
rows arrived, how many were assessed, how many were dropped, and under which
strategy. They are what makes two runs over the same data under different
strategies tellable apart. The same facts are restated in prose at the end of
`assessment["summary"]`; these fields are the values to read.

`valid_groups` and `invalid_groups` are the group-level half of the same
disclosure: a name in `invalid_groups` was **not** checked and not cleared, and
it also appears in `assessment["insufficient_evidence_groups"]` with the reason.

The full set of structure types, `FairnessReport`, `AssessmentReport`,
`DataInfo`, `ExplanationsReport`, `MetricStatusEntry` and
`InsufficientEvidenceGroup`, is defined in
[`report_types.py`](../src/vfairness/evaluation/vfairness_metrics/report_types.py)
and re-exported from both `vfairness` and `vfairness.evaluation`. The required
keys are part of the stable public API (see [API_STABILITY.md](API_STABILITY.md)).

### Available Methods

| Method | Description |
|--------|-------------|
| `demographic_parity_difference()` | Gap in positive prediction rates <!--cs-->**[Checked]**<!--/cs--> |
| `equal_opportunity_difference()` | Gap in true positive rates <!--cs-->**[Checked]**<!--/cs--> |
| `equalized_odds_difference()` | Max of TPR and FPR differences <!--cs-->**[Checked]**<!--/cs--> |
| `compute_all_metrics()` | Compute all applicable metrics <!--cs-->**[Checked]**<!--/cs--> |
| `get_report()` | Generate comprehensive fairness report <!--cs-->**[Checked]**<!--/cs--> |
| `compare_with_fairlearn()` | Compare results with Fairlearn library <!--cs-->**[Checked]**<!--/cs--> |
The ratio variant is not an analyzer method: use the module-level function
`demographic_parity_ratio(y_true, y_pred, sensitive_attr)` instead.

### Multiple Testing Correction

```python
report = analyzer.get_report(
    include_ci=True,
    multiple_testing_correction='fdr'  # Benjamini-Hochberg
)

# Available corrections: 'none', 'bonferroni', 'fdr' (alias 'benjamini_hochberg')
```

---

## FairExplAIner Mode
<!-- cap-status: FairExplAIner -->
**Beta status: Checked.** 5 code units behind this name: 5 checked
<!-- /cap-status -->


FairExplAIner provides intelligent, context-aware explanations for every fairness metric and statistical measure. It helps users understand what metrics mean, how to interpret them, and what actions to take.

### Enabling FairExplAIner

```python
from vfairness import FairnessAnalyzer

# Method 1: Enable at creation
analyzer = FairnessAnalyzer(
    y_true, y_pred, gender,
    fair_explainer=True
)

# Method 2: Enable dynamically
analyzer = FairnessAnalyzer(y_true, y_pred, gender)
analyzer.enable_fair_explainer()

# Check status
print(f"FairExplAIner enabled: {analyzer.fair_explainer_enabled}")

# Disable
analyzer.disable_fair_explainer()
```

### Using FairExplAIner with Reports

When FairExplAIner is enabled, `get_report()` automatically includes an `explanations` section:

```python
analyzer = FairnessAnalyzer(y_true, y_pred, gender, fair_explainer=True)
report = analyzer.get_report(include_ci=True)

# Access explanations
explanations = report['explanations']

# Summary of all findings
print(explanations['summary'])

# Detailed explanation for each metric
for metric_name, explanation in explanations['metrics'].items():
    print(f"\n{metric_name}:")
    print(f"  Definition: {explanation['definition'][:100]}...")
    print(f"  Value: {explanation['value']}")
    print(f"  Severity: {explanation['severity']}")
    print(f"  Evaluation: {explanation['evaluation']}")
    print(f"  Recommendation: {explanation['recommendation'][:100]}...")

# Statistical measure explanations (when include_ci=True)
for stat_name, explanation in explanations['statistical'].items():
    print(f"{stat_name}: {explanation['evaluation']}")
```

### Explaining Individual Metrics

```python
# Get detailed explanation for a specific metric
explanation = analyzer.explain_metric('demographic_parity_difference')

# MetricExplanation attributes
print(explanation.metric_name)          # 'demographic_parity_difference'
print(explanation.definition)           # What this metric measures
print(explanation.interpretation_guide) # How to interpret values
print(explanation.value)                # The computed value
print(explanation.evaluation)           # Context-aware assessment
print(explanation.benchmark_context)    # Industry standards
print(explanation.recommendation)       # Actionable advice
print(explanation.severity)             # 'info', 'low', 'medium', 'high', 'critical'
print(explanation.related_metrics)      # Other relevant metrics

# Convert to dict
explanation_dict = explanation.to_dict()
```

### Standalone FairExplAIner Class

Use FairExplAIner independently without the analyzer:

```python
from vfairness import (
    FairExplAIner,
    MetricExplanation,
    explain_fairness_report,
    print_explanations,
    classification_fairness_report
)

# Create explainer directly
explainer = FairExplAIner(task_type='classification')

# Explain a single metric
explanation = explainer.explain_metric(
    'demographic_parity_difference',
    value=0.15,
    group_stats={'M': {'size': 500}, 'F': {'size': 500}},
    threshold=0.10
)
print(explanation)

# Explain confidence intervals
ci_explanation = explainer.explain_confidence_interval(
    'demographic_parity_difference',
    point_estimate=0.15,
    lower_bound=0.10,
    upper_bound=0.20,
    interval_type='confidence',
    confidence_level=0.95
)

# Explain effect sizes
effect_explanation = explainer.explain_effect_size(
    'cohens_d',
    value=0.45,
    group1='Male',
    group2='Female',
    ci=(0.30, 0.60)
)

# Explain an entire report
report = classification_fairness_report(y_true, y_pred, gender, include_ci=True)
explanations = explainer.explain_report(report, include_statistical=True)
```

### Convenience Functions

```python
from vfairness import explain_fairness_report, print_explanations

# Add explanations to any report
report = classification_fairness_report(y_true, y_pred, gender)
explanations = explain_fairness_report(report, include_statistical=True)

# Print formatted explanations
print_explanations(explanations)
```

### Severity Levels

| Severity | Description | Typical Values |
|----------|-------------|----------------|
| `info` | Excellent, within best practices | DP diff < 0.05 |
| `low` | Acceptable, continue monitoring | DP diff 0.05-0.10 |
| `medium` | Concerning, investigate further | DP diff 0.10-0.15 |
| `high` | Significant disparity, action needed | DP diff 0.15-0.20 |
| `critical` | Severe disparity, immediate action | DP diff > 0.20 |

### Metric Definitions

FairExplAIner includes comprehensive definitions for all metrics:

```python
from vfairness import CLASSIFICATION_METRICS, REGRESSION_METRICS, STATISTICAL_MEASURES

# Access metric definition
dp_def = CLASSIFICATION_METRICS['demographic_parity_difference']
print(dp_def['definition'])
print(dp_def['interpretation_guide'])
print(dp_def['thresholds'])  # {'excellent': 0.05, 'acceptable': 0.10, ...}
print(dp_def['legal_context'])  # Regulatory information
print(dp_def['related_metrics'])

# Statistical measure definitions
ci_def = STATISTICAL_MEASURES['confidence_interval']
cohens_d_def = STATISTICAL_MEASURES['cohens_d']
```

### Model-Decision Explanation (XAI)

`FairExplAIner` above answers "why is this finding an issue?". The following
two entry points answer the complementary question: "why did the model make
this decision for this person, and which features drive it?". Together they
cover the GDPR Art. 22 and EU AI Act Art. 13 right-to-explanation surface. Both
are dependency-light and are imported from their module paths (not from the
package top level), so `import vfairness` stays light.

#### `FeatureAttributionExplainer`

Model-agnostic feature attribution over any `predict(X) -> array` callable.
<!-- cap-status: FeatureAttributionExplainer -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

GLOBAL scope ranks which features matter across a dataset (permutation
importance); LOCAL scope explains a single decision against a baseline
(occlusion, with optional exact SHAP).

```python
from vfairness.evaluation.vfairness_metrics.attribution import (
    FeatureAttributionExplainer,
)

explainer = FeatureAttributionExplainer(predict=model.predict, feature_names=cols)

# GLOBAL: with labels y, sklearn scores the drop in fit when a column is
# shuffled; without y, a NumPy prediction-variance fallback runs.
global_result = explainer.global_importance(X, y=y)

# LOCAL: explain one decision against a background dataset (column medians are
# the baseline). use_shap=True uses exact SHAP if installed, else occlusion.
local_result = explainer.explain_decision(x_row, background=X, use_shap=False)
for c in local_result.top(5):
    print(c.feature, c.direction, round(c.signed_value, 4))
```

**Signatures:**

- `global_importance(X, y=None, n_repeats=10, random_state=42) -> AttributionResult`
- `explain_decision(x_row, background, use_shap=False) -> AttributionResult`

**`AttributionResult` fields:** `scope` (`"global"` or `"local"`), `method`
(`"permutation"`, `"permutation_numpy"`, `"occlusion"`, or `"shap"`),
`contributions` (a list of `FeatureContribution`), `base_value` (local baseline
prediction), `prediction` (local prediction), and `notes`. Each
`FeatureContribution` has `feature`, `importance` (non-negative magnitude),
`direction` (`"increase"`, `"decrease"`, or `"neutral"`), and `signed_value`.
`AttributionResult.top(k)` returns the k highest-magnitude contributions.

#### `counterfactual_fairness()`

Perturbation-sensitivity / individual-consistency metric. Given the model's
predictions on the factual data and on a counterfactual version supplied by the
caller, it reports how often the decision changes and how much the score moves.
Pure NumPy; the caller supplies the counterfactual predictions.

This scores exactly the counterfactual predictions it is handed. It equals
counterfactual fairness in the sense of Kusner et al. (2017) only when those
counterfactuals come from a structural causal model that propagates every
variable downstream of the protected attribute. For a naive single-attribute
flip it measures direct sensitivity, not full counterfactual fairness (bias
laundered through a held-fixed proxy will look fair). For the SCM path, use the
DoWhy-backed `compute_counterfactual` op (`vfairness_causal_counterfactual`).

```python
from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
    counterfactual_fairness,
)

result = counterfactual_fairness(
    y_pred_factual=scores_original,
    y_pred_counterfactual=scores_flipped,
    threshold=0.5,   # pass threshold=None when inputs are already 0/1 labels
)
print(result.flip_rate, result.severity)
print(result.interpretation)
```

**Signature:** `counterfactual_fairness(y_pred_factual, y_pred_counterfactual, threshold=0.5) -> CounterfactualFairnessResult`

**`CounterfactualFairnessResult` fields:** `n`, `flip_rate` (fraction of
decisions that change, the headline number), `mean_abs_diff`, `max_abs_diff`,
`threshold`, `severity` (`info` to `critical`), and a plain-language
`interpretation`.

---

## Intersectional Group Analysis

vfairness provides detailed intersectional analysis to identify which specific groups are driving fairness disparities. This answers the critical question: "We know there's a disparity of X% - but WHO is most affected?"

### identify_privileged_groups

Identify most and least advantaged groups based on positive prediction rates.
<!-- cap-status: identify_privileged_groups -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import identify_privileged_groups

# For intersectional analysis (e.g., gender + race combined)
result = identify_privileged_groups(y_true, y_pred, gender_race)

# Access results
print(f"Most advantaged group: {result['privileged_group'].group}")
print(f"  Positive rate: {result['privileged_group'].positive_rate:.1%}")

print(f"Most disadvantaged group: {result['disadvantaged_group'].group}")
print(f"  Positive rate: {result['disadvantaged_group'].positive_rate:.1%}")

print(f"Maximum disparity: {result['max_disparity']:.1%}")
print(f"Severity: {result['disparity_severity']}")

# All groups ranked by positive rate
for g in result['all_groups']:
    print(f"  {g.group}: {g.positive_rate:.1%} (severity: {g.severity})")
```

**Parameters (statistical policy — see ⓘ below):**
- `min_group_size: int = 30` — minimum samples for a cell to be ranked.
- `low_n_warning_threshold: int | None = None` — defaults to `max(min_group_size, 30)`. Cells with `n < threshold` carry `low_n_warning=True`.
- `zero_selection_floor: int = 10` — cells with `positive_count == 0` at `n >= floor` surface in `zero_selection_alerts` regardless of `min_group_size`.

**Returns:**
- `privileged_group`: GroupAdvantage object for highest-rate group
- `disadvantaged_group`: GroupAdvantage object for lowest-rate group
- `all_groups`: List of all GroupAdvantage objects (sorted by rate); each carries `low_n_warning`
- `overall_rate`: Overall positive prediction rate
- `max_disparity`: Maximum disparity between any two groups
- `disparity_severity`: Severity level ('info', 'low', 'medium', 'high', 'critical')
- `excluded_groups`: cells dropped by the size gate, each with `{group, size, positive_count, positive_rate, ground_truth_rate, reason}`
- `zero_selection_alerts`: cells with 0/N selections at `n >= zero_selection_floor`, each with `{group, size, positive_count, ground_truth_rate, analysed, reason}`
- `data_treatment`: audit log `{min_group_size, low_n_warning_threshold, zero_selection_floor, outcome_polarity, n_total_cells_seen, n_cells_included, n_cells_excluded_small, n_cells_warning_low_n, n_zero_selection_alerts, overall_rate, overall_ground_truth_rate}`

> ⓘ **Why these knobs exist.** The default `min_group_size=30` is the Turing
> M3 reporting standard — below it, a single cell's selection rate is too
> noisy to compare against another's. But it also silently drops small
> protected groups (e.g. a Native-American cell of n=22 with zero invites).
> The transparency outputs surface those cells anyway so an auditor sees
> what was filtered. Lower `min_group_size` per call to make the gate
> looser; the engine reports `low_n_warning` on every cell below 30 so
> the GUI can render a confidence caveat. `zero_selection_floor` is
> independent: it controls when a 0/N cell becomes a policy-actionable
> alert, regardless of size.

### GroupAdvantage

The `GroupAdvantage` dataclass contains detailed information about each group:
<!-- cap-status: GroupAdvantage -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


```python
@dataclass
class GroupAdvantage:
    group: str                    # Group name
    positive_rate: float          # Rate of positive predictions
    size: int                     # Number of samples
    relative_to_overall: float    # Ratio compared to overall rate
    relative_to_best: float       # Ratio compared to best group
    disparity_contribution: float # Contribution to overall disparity
    severity: str                 # 'info', 'low', 'medium', 'high', 'critical'
    ground_truth_rate: float      # Rate of positive ground-truth labels
    false_positive_rate: float    # FPR (wrongly predicted positive among actual negatives)
    prediction_delta: float       # positive_rate minus ground_truth_rate (over/under-prediction)
    low_n_warning: bool           # True when size < low_n_warning_threshold
```

### intersectional_disparity_analysis

Comprehensive intersectional analysis with comparison to single-attribute analyses.
<!-- cap-status: intersectional_disparity_analysis -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import intersectional_disparity_analysis

# Full analysis with single-attribute comparison
analysis = intersectional_disparity_analysis(
    y_true, y_pred,
    gender_race,  # Combined intersectional attribute
    single_attributes={'gender': gender, 'race': race}  # For comparison
)

# Intersectional results
inter = analysis['intersectional_analysis']
print(f"Intersectional disparity: {inter['max_disparity']:.1%}")

# Compare with single attributes
for attr, single in analysis['single_attribute_analyses'].items():
    print(f"{attr} disparity: {single['max_disparity']:.1%}")

# Check for hidden disparities
comparison = analysis['comparison']
if comparison['intersectional_reveals_more']:
    print(f"Hidden disparity found: {comparison['hidden_disparity']:.1%}")

# Human-readable insights
for insight in analysis['insights']:
    print(f"- {insight}")

# Actionable recommendations
for rec in analysis['recommendations']:
    print(f"- {rec}")
```

### get_group_rankings

Get all groups ranked by a specific metric.
<!-- cap-status: get_group_rankings -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import get_group_rankings

# Rank by different metrics
rankings = get_group_rankings(y_true, y_pred, demographics, metric='positive_rate')
# Also supports: 'tpr', 'fpr', 'precision'

for group in rankings:
    print(f"{group['rank']}. {group['group']}: {group['value']:.1%}")
```

### Using with classification_fairness_report

Include group analysis in fairness reports:

```python
from vfairness import classification_fairness_report

report = classification_fairness_report(
    y_true, y_pred, demographics,
    include_group_analysis=True  # Enable group analysis
)

# Access group analysis
if 'group_analysis' in report:
    ga = report['group_analysis']
    print(f"Most advantaged: {ga['privileged_group']['group']}")
    print(f"Most disadvantaged: {ga['disadvantaged_group']['group']}")
    print(f"Disparity severity: {ga['disparity_severity']}")
```

### Severity Levels for Group Analysis

| Severity | Gap from Best Group | Action |
|----------|---------------------|--------|
| `info` | < 5% | Excellent, within best practices |
| `low` | 5-10% | Acceptable, continue monitoring |
| `medium` | 10-15% | Investigate further |
| `high` | 15-20% | Significant, action needed |
| `critical` | > 20% | Severe, immediate action required |

---

## Auto-Discovery Features

vfairness provides intelligent auto-discovery features to help identify protected attributes, detect proxy features, and systematically scan for fairness violations across your dataset.

### detect_protected_attributes

Automatically detect potential protected/sensitive attributes in a DataFrame using heuristics based on column names, data types, and value patterns.
<!-- cap-status: detect_protected_attributes -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import detect_protected_attributes

# Auto-detect protected attributes
candidates = detect_protected_attributes(
    df,
    min_confidence=0.3,           # Minimum confidence threshold (0-1)
    exclude_columns=['id', 'target'],  # Columns to skip
    max_unique_values=50,         # Max unique values for categorical
    include_numeric=False         # Include numeric columns (e.g., age)
)

# Review detected candidates
for candidate in candidates:
    print(f"\nAttribute: {candidate.column}")
    print(f"  Confidence: {candidate.confidence:.1%}")
    print(f"  Category: {candidate.category}")  # 'demographic', 'socioeconomic', 'geographic', etc.
    print(f"  Unique values: {candidate.n_unique}")
    print(f"  Reason: {candidate.reason}")
```

**ProtectedAttributeCandidate attributes:**
- `column`: Column name
- `confidence`: Detection confidence (0-1)
- `category`: Type of protected attribute ('demographic', 'socioeconomic', 'geographic', 'health', 'other')
- `n_unique`: Number of unique values
- `sample_values`: Example values from the column
- `reason`: Why this column was flagged

### classify_column_roles

**Category:** Auto-discovery — *schema typology / pre-flight gate*.
<!-- cap-status: classify_column_roles -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


`detect_protected_attributes` only finds protected columns.
`classify_column_roles` classifies **every** column into one role so the
engine knows what may legitimately feed a model, what is the answer key,
and what must never enter a model at all:

| Role | Meaning |
|---|---|
| `identity_pii` | Names, email, phone, address, photo, **date of birth**, near-unique IDs — must never be a feature |
| `oracle` | The ground-truth / target answer key — scored on, never a feature |
| `model_output` | The model's own decision / score / probability |
| `protected` | A protected attribute (the user's declaration is authoritative) |
| `proxy_candidate` | Correlates with a protected attribute, or a known proxy-shaped name (zip, university tier, tenure gap) — kept as a feature but flagged |
| `job_relevant` | A legitimate predictor with no PII / protected / output signal |
| `unknown` | Constant / empty column |

**For business / compliance readers.** This is the *pre-flight gate*. It is
the answer to "we don't even collect race" (it finds the proxies) and to
"did we accidentally train on the applicant's name or the answer key" (it
refuses). When the user declared which columns are protected, that
declaration is trusted but **verified**: a column that was *not* declared
but looks protected (e.g. `marital_status`, `veteran_status`) is surfaced
as a **mismatch** to confirm, never silently reclassified. A column
declared protected that is actually raw PII is also flagged. Supports EU AI
Act Art. 10 (data governance) and the data-minimisation principle.

**Refuse-fast.** If a declared target column does not exist, or the dataset
is entirely identifiers / answer keys (nothing legitimate to assess),
`refuse=True` with a plain `refuse_reason` — the caller stops before
producing a misleading result.

```python
from vfairness import classify_column_roles

schema = classify_column_roles(
    df,
    declared_protected=['age', 'gender', 'race_ethnicity'],
    declared_prediction='invite_decision',
    declared_target='hired',          # optional oracle
)

print(schema['pii_leakage'])     # ['first_name', 'date_of_birth', 'photo_url', ...]
print(schema['proxy_candidates'])# ['zip_minority_majority', 'university_tier', ...]
for m in schema['mismatches']:
    print(m['column'], '->', m['inferred'], '-', m['detail'])
if schema['refuse']:
    print('STOP:', schema['refuse_reason'])
```

**Returns** a dict: `roles` (list of `ColumnRole.to_dict()` with
`column / role / confidence / reason / category`), plus convenience buckets
`pii_leakage`, `oracle_columns`, `model_output`, `protected`,
`proxy_candidates`, `job_relevant`, `unknown`, the `mismatches` list, and
the `refuse` / `refuse_reason` gate.

> One source of truth: Pulse's pre-flight schema stage and the Navigator's
> `vfairness_profile_data` profiling step call this same function, so both
> exclude exactly the same identity / oracle columns from analysis.

### Phase 4(2)-B — multimodal & generative routing (reuse-first)

**Category:** Pulse source routing — reuses existing vfairness engines.

`run_pulse` auto-detects the artefact kind and routes to the matching
reuse path; the tabular path is the misroute-safe default. Each path maps
into the **same `build_assurance_verdict` shape** so the Pulse hero +
progressive-disclosure UX is unchanged.

> **Import path.** `run_pulse`, `llm_probe_pulse`, `agent_probe_pulse` and
> `vision_probe_pulse` are Pulse orchestration entry points that live under
> `vfairness.operations.pulse`:
> `from vfairness.operations.pulse import run_pulse, llm_probe_pulse, agent_probe_pulse, vision_probe_pulse`.
> They are not re-exported from the top-level `vfairness` namespace; the only
> production caller is the Pulse consumer handler.

- **`llm_probe_pulse(llm_config, *, proxy=None, ...)`** (B1) — live LLM
  endpoint: drives `CounterfactualTester` (name-swap) via `LLMApiProxy`.
  No endpoint configured/reachable ⇒ explicit **Disclaimer**, never a
  fabricated verdict. (Fixed a real `counterfactual.py` bug: `abs()` on a
  non-numeric disparity value crashed any ≥3-variant swap.)
- **`agent_probe_pulse(df, inputs, action_col, group_col, ...)`** (B4) —
  agent traces: reuses `ToolBiasAuditor` + `ActionBiasAnalyzer` to compare
  tool/action selection across demographic groups.
- **`vision_probe_pulse(df, inputs, demo_col, ...)`** (B2) — image set:
  `vfairness.vision.skew / ndkl / bias_amplification` (Geyik et al. 2019;
  Seshadri et al. 2023) on per-image demographic labels. Per-image
  FairFace/CLIP classification (`classify_face_demographics`) is gated on
  the **Python-3.9 vision sidecar** (`VFAIRNESS_VISION_SIDECAR`); with no
  sidecar it returns `available:false` + a metadata-only note — it never
  guesses demographics.
- **3-axis Pareto** (B3) — `mitigation_pareto` now also returns
  `contextDistortion` per strategy + a 3D Pareto frontier and a
  `threeAxisSummary` that flags the context-blind-parity trap (Gemini,
  Feb 2024): zero gap with high distortion is **not** a good outcome.

> Honest scope: the vision *math* (skew/NDKL/amplification) and all
> routing/mapping are verified on real/synthetic fixtures. The FairFace
> model-inference layer is an isolated, optional sidecar — documented, not
> faked.

### Phase 4(2)-A — advanced inference, input routing, generative & specification

**Category:** Statistical rigor / multimodal triage (no external infra).

- **`empirical_likelihood_ci(successes, n, confidence_level=0.95)`** —
  distribution-free CI for a proportion (EL ratio = binomial LR; chi-squared
  limiting, no scipy). Returns a `StatisticalResult`.
- **`simultaneous_disparity_bounds(group_counts, confidence_level=0.95)`** —
  Bonferroni-exact EL bounds across **every subgroup at once** (the
  audit-grade joint statement; Cherian & Candès JMLR 2024 framing). Returns
  per-group simultaneous CI + worst-case ratio/difference bound. Surfaced in
  the Pulse disparity grid as a "simultaneous N% bound across all
  subgroups" line.
- **`sequential_fairness_test(outcomes_a, outcomes_b, *, alpha, power,
  effect_size)`** — anytime-valid Wald SPRT (the same boundaries/LLR
  `FairnessPowerAnalyzer` uses, decoupled for two plain arrays). Lets an
  audit stop as soon as evidence is (in)sufficient without alpha-spending.
- **`detect_specification_bias(df, *, outcome, prediction,
  exclude_columns)`** — construct-validity / specification bias (Jacobs &
  Wallach FAccT 2021): target leakage (a feature ≈ the outcome), proxy
  target (e.g. `cost`↦health need, Obermeyer 2019 — real citation), and
  circular evaluation (outcome == prediction). Folds into the bias
  taxonomy + assurance verdict.

**Generative input routing (Pulse):** `run_pulse` auto-detects a
prompts+outputs artefact (a `prompt`-like + a long free-text `output`-like
column) and routes to a generative path that REUSES
`vfairness.llm.OutputAnalyzer.analyze_all` (the standard metric battery, plus the LLM-judge comparison when a judge is configured; Benjamini-Hochberg
corrected, **no API call**) per protected attribute, then maps the result
into the **same `build_assurance_verdict` shape** so the Pulse hero +
progressive-disclosure UX renders unchanged. The tabular path is the
default and is misroute-safe (a feature CSV has neither column).

> One source of truth: the disparity grid's simultaneous bound is computed
> by `selection_rate_disparity_matrix`, so Pulse **and** the Navigator
> metrics handler get it identically; the generative path reuses the same
> `OutputAnalyzer` the Navigator's LLM-output-analysis step uses.

### Live progress telemetry

**Category:** Pulse orchestration — real per-stage progress.

`run_pulse` accepts an **optional** callback so callers can surface true
per-stage progress instead of a time-based guess:

```python
run_pulse(df, inputs, progress=None)
# progress(stage: str, label: str, stage_index: int,
#          total_stages: int, pct: int) -> None
```

- The parameter defaults to `None`. It is invoked only through an internal
  helper that swallows every exception, so progress reporting **can never
  affect the analysis or raise** — runs with no callback are byte-identical
  to before. `run_pulse` has a single production caller (the Pulse
  consumer handler); the Navigator calls the underlying functions
  directly, so this is an additive, single-caller change.
- The orchestrator emits **8 stages** (`total_stages = 8`). Stage 0 is
  reported by the consumer handler before `run_pulse` (dataset
  decrypt/load); stages 1–7 are emitted by `run_pulse` itself, in
  execution order, immediately before each major block:

  | # | Stage label | Emitted before |
  |---|-------------|----------------|
  | 0 | Decrypting and loading your dataset | (consumer handler) |
  | 1 | Checking data quality and column roles | `classify_column_roles` / quality |
  | 2 | Computing fairness metrics across every protected attribute | per-variable + metric cards |
  | 3 | Detecting bias patterns and group disparity | `BiasDetector.full_audit` |
  | 4 | Screening for proxy and redundant-encoding leakage | proxy / leakage screen |
  | 5 | Running statistical robustness and confidence intervals | statistical battery |
  | 6 | Mapping causal structure and intersectional subgroups | causal skeleton + intersectional |
  | 7 | Composing the assurance verdict and recommendations | `build_assurance_verdict` |

  Stage indices are monotonically increasing, so a consumer can drive a
  live phase indicator directly off `stage_index / total_stages`. The
  early-return routing paths (generative / LLM / agent / vision / refuse)
  legitimately emit fewer stages; the consumer still reports a final
  completion ping.

> One source of truth: the stage labels above are the canonical Pulse
> pipeline order. The platform's live phase indicator
> (`PulseDispatchView`) mirrors this list 1:1 and follows the real
> `stage_index` when present, only falling back to a smooth time model
> for the gaps between pings — it never fabricates a stage count.

### build_assurance_verdict

**Category:** Operations / reporting — *audit-grade verdict engine*.
<!-- cap-status: build_assurance_verdict -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


Turns the already-computed analysis sections into **one structured
assurance opinion**, following the assurance-audit taxonomy (Lam et al.,
FAccT 2024):

| Opinion | Meaning |
|---|---|
| `Unqualified` | No material fairness defect on the assessed attributes |
| `Qualified` | Deployable only with the documented remediations + monitoring |
| `Adverse` | Material defects — unfit to deploy as-is (blocks deployment) |
| `Disclaimer` | Cannot form an opinion (refuse-fast, or insufficient data) |

It also emits stable-ID `findings` (with `panelRef` back to the evidence),
priority-ranked `recommendations` (each routed to a Navigator step and a
`vfairnessFunction`), `metricsDeferred` (+ rationale), a
`jurisdictionBasis`, and an `auditTrail`. **The jurisdiction-aware legal
overlay lives here, not in the metrics**: the US EEOC four-fifths 0.80 line
is applied only for US employment; EU/UK/CH use proportionality + objective
justification with no numeric bar.

```python
from vfairness.operations.reporting import build_assurance_verdict

v = build_assurance_verdict(
    schema=schema, per_variable=per_variable, metrics=metrics,
    bias=bias, proxies=proxies, statistical=statistical,
    intersectional=intersectional, disparity_matrix=dm,
    domain='hiring', jurisdiction='US', has_truth=True)

print(v['overall'], v['blocksDeployment'])
print(v['oneLineVerdict'])
for f in v['findings']:        print(f['id'], f['severity'], f['evidence'])
for r in v['recommendations']: print(r['id'], '->', r['routeTo'])
```

Pure assembly + rules; never raises. The structured object is the durable
artefact that powers the Pulse banner **and** pre-fills the Navigator
assessment (one source of truth).

### selection_rate_disparity_matrix

**Category:** Group fairness — *jurisdiction-neutral disparity*.
<!-- cap-status: selection_rate_disparity_matrix -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


Per group: selection rate, 95% confidence interval (forest-plot ready), n,
and reliability tier; then the full group×group **ratio** and
**difference** matrices and the worst pair.

This is **pure descriptive statistics — valid in every jurisdiction**. It
deliberately does *not* apply the US EEOC four-fifths (0.80) rule or any
legal bright-line. The four-fifths rule is **US-employment-only** and is
*not* law in the EU/UK/Switzerland (which use a proportionality + objective
-justification test, no fixed numeric bar). The legal reading of these
numbers is a separate, jurisdiction-aware overlay applied by the verdict
engine — never baked into the metric.

```python
from vfairness import selection_rate_disparity_matrix

m = selection_rate_disparity_matrix(y_pred, race)
print(m['min_ratio'], m['min_ratio_pair'])      # worst adverse-impact pair
print(m['rates']['Black'])  # {rate, ci_low, ci_high, n, tier, interpretable}
print(m['ratio_matrix']['Black']['White'])      # rate(Black)/rate(White)
```

Returns `groups`, `rates`, `ratio_matrix`, `difference_matrix`,
`reference_group`, `min_ratio` / `min_ratio_pair`, `max_difference` /
`max_difference_pair`, `headline_basis`, and `headline_over_all_groups`.

**Worst-pair is computed over interpretable groups only (n≥10)**, so a tiny
group's noisy rate never defines the headline. That sentence has been in this
document for some time and it was **not true until 2026-09-27**: the pool fell
back to every group whenever fewer than two groups were interpretable, so a
2-person group with a rate of 0.0 produced `min_ratio` 0.0 in silence, from a
group the same call had already marked `tier: 'invalid'` and
`interpretable: False`. An independent audit proved it on 50 rows at 0.8 plus 2
rows at 0.0.

**Three states, and `headline_basis` says which one you have.**

| `headline_basis` | `min_ratio` / `max_difference` | What it means |
|---|---|---|
| `interpretable_groups` | measured numbers | At least two groups pass the n≥10 gate; the headline is over those. |
| `no_interpretable_pair` | `nan` | Fewer than two groups pass the gate, so **no headline is computed**. This is a could-not-check, not a clean bill. |

When `headline_basis` is `no_interpretable_pair`, the numbers the old code would
have published are preserved under `headline_over_all_groups` (a dict with
`min_ratio`, `min_ratio_pair`, `max_difference`, `max_difference_pair`) and a
warning names the groups that kept the pool from forming. Nothing is lost: the
value is still there and it is no longer presented as the headline.

`ratio_matrix` and `difference_matrix` are unchanged and still hold every pair,
interpretable or not. **A caller that reads `min_ratio` must handle `nan`**, and
must not treat it as "no disparity found": `nan < 0.8` is `False`, so a bare
threshold comparison silently turns a could-not-check into a pass.

### group_reliability

**Category:** Statistical rigor — *group-size honesty gate*.
<!-- cap-status: group_reliability -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


A single `n >= 30` cut-off hides the difference between a 31-person group
(read with caution) and a 9-person group (the 95% margin of error exceeds
the metric — the rate must not be read at all). `group_reliability` assigns
one shared tier per group, used identically by Pulse, the Navigator, and
the Modules:

| Tier | Size | Treatment |
|---|---|---|
| `reliable` | n ≥ 100 | Report normally |
| `caution` | 30–99 | Report, note the wider confidence interval |
| `underpowered` | 10–29 | Report greyed, "small sample — indicative only" |
| `invalid` | n < 10 | **Do not interpret the rate** (shown, never ranked, excluded from the verdict) |

```python
from vfairness import group_reliability

tiers = group_reliability({'male': 5200, 'female': 4800, 'non_binary': 41,
                           'native_american': 22})
# {'non_binary': {'n': 41, 'tier': 'caution', 'interpretable': True,
#                 'note': 'Small group (n<100) ...'},
#  'native_american': {'n': 22, 'tier': 'underpowered', ...}, ...}
```

Returns `{group: {n, tier, interpretable, note}}`; `interpretable` is
`False` only for the `invalid` tier. `reliability_tier(n) -> str` is the
scalar form. Thresholds follow the Fairlearn small-group convention and the
Turing M3 minimum-reporting guidance.

### identify_proxy_features

Detect features that may serve as proxies for protected attributes through correlation analysis.
<!-- cap-status: identify_proxy_features -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import identify_proxy_features

# Find proxy features for ONE protected attribute per call
proxies = identify_proxy_features(
    df,
    'gender',                     # single protected attribute name
    correlation_threshold=0.3     # Minimum association to flag
)

# Review potential proxies (plain dicts)
for proxy in proxies:
    print(f"\n{proxy['column']} -> gender")
    print(f"  Correlation: {proxy['correlation']:.3f} ({proxy['correlation_type']})")
    print(f"  Risk level: {proxy['risk_level']}")

# To audit several attributes, loop over them
all_proxies = {attr: identify_proxy_features(df, attr)
               for attr in ['gender', 'race']}
```

The association measure is chosen automatically from the data types
(Pearson for numeric x numeric, bias-corrected Cramer's V for nominal x
nominal, correlation ratio for mixed pairs); there is no `method`
parameter. Each returned dict carries `column`, `correlation`,
`abs_correlation`, `correlation_type` and `risk_level`.

### multivariate_proxy_leakage

**Category:** Proxy / indirect discrimination — *systemic (multivariate) tier*.
<!-- cap-status: multivariate_proxy_leakage -->
**Beta status: Checked.** verified by running it; no test pins the behaviour, so a regression would be silent
<!-- /cap-status -->


`identify_proxy_features` answers "does **one** feature correlate with a
protected attribute?". `multivariate_proxy_leakage` answers the harder,
more important question:

> **"If I deleted the protected column entirely, could the model still
> rebuild it from everything else combined?"**

This is the *redundant-encoding* problem (Barocas & Selbst 2016; Datta et
al. 2017). Features that are each individually weak proxies can, *together*,
reconstruct a protected attribute — so removing columns one by one never
makes the model fair. It is the single check that survives "we dropped the
sensitive column," and the reason data-level transformation (not deletion)
is usually required.

**For business / compliance readers.** Plain meaning of the score: a model
is trained to *guess* the protected attribute (e.g. ethnicity) using only
the "allowed" columns. The result is an **AUC** from 0.5 to 1.0:

| AUC | Reading | Business implication |
|---|---|---|
| ~0.50 | No leakage | Removing the protected column is genuinely sufficient |
| 0.55–0.70 | Partial leakage | Some residual risk; document and monitor |
| **≥ 0.70** | **Systemic leakage** | The attribute is *redundantly encoded*. Deleting columns will not de-bias the system — features must be transformed or the label re-derived. Treat as a deployment blocker pending remediation |
| ≥ 0.85 | Severe | The protected attribute is almost fully recoverable from "neutral" data |

This directly supports EU AI Act Art. 10 (data governance — proxies are a
data-quality defect) and the disparate-impact doctrine: it is the
evidence-grade answer to "but we don't even collect race."

**Technical description.** A cross-validated gradient-boosting classifier
(`HistGradientBoostingClassifier`) is trained on all non-protected,
non-excluded columns to predict each protected attribute. Categorical
predictors are one-hot encoded (so individual proxy *categories* — an HBCU
tier, a minority-majority ZIP — are learnable); numeric predictors pass
through. Held-out ROC AUC is reported both as a macro average over groups
and as the **worst (most-identifiable) single group**, because a single
highly-recoverable subgroup is itself a leakage risk (audit / worst-case
posture, consistent with the platform's Kearns-style subgroup framing). The
test is bounded for speed (row cap, CV folds adapt to class support) and
**degrades gracefully** — too little data, one class, or a missing
scikit-learn returns an explained "negligible/skipped" result, never an
error.

```python
from vfairness import multivariate_proxy_leakage

results = multivariate_proxy_leakage(
    df,
    protected_attributes=['race_ethnicity', 'gender', 'age'],
    exclude_columns=['applicant_id', 'invite_decision', 'true_label'],
    max_rows=5000,   # speed cap; works on any dataset size
    cv=5,
)

for r in results:                       # sorted by AUC, worst first
    d = r.to_dict()
    print(f"{d['protected_attribute']}: AUC {d['auc']:.2f} "
          f"({d['severity']}, systemic={d['systemic_leakage']})")
    print(f"  driven by: "
          f"{[c['feature'] for c in d['top_contributors'][:5]]}")
    print(f"  {d['interpretation']}")
```

**Returns** `List[MultivariateProxyResult]`, each with `.to_dict()`:

- `protected_attribute`: the attribute tested
- `auc`: headline reconstructability (max of macro and worst-group)
- `macro_auc`, `worst_group_auc`: averaged vs. most-identifiable group
- `chance_auc`: 0.5 baseline
- `severity`: `negligible | low | medium | high | severe`
- `systemic_leakage`: `True` when `auc >= 0.70`
- `n_features_used`, `n_samples`, `n_classes`
- `top_contributors`: features ranked by mutual information with the
  attribute (one-hot columns are mapped back to their source feature)
- `interpretation`: one plain-language sentence for the report

**When NOT to rely on it alone.** It quantifies *predictability*, not
causation — pair it with `identify_proxy_features` (which feature),
`find_proxy_chains` (the path), and the causal panel for the mechanism. A
high AUC says "this data leaks the attribute"; the fix still requires human
judgement about which features are legitimately job-relevant.

> One source of truth: Pulse, the Navigator's `vfairness_proxy_analysis`
> handler, and the Bias-Detection module all call this same function, so
> every surface reports identical leakage figures.

### scan_fairness_violations

Comprehensively scan a model's predictions for fairness violations across multiple metrics and all detected or specified protected attributes.
<!-- cap-status: scan_fairness_violations -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import scan_fairness_violations

# Scan for violations across multiple metrics
violations = scan_fairness_violations(
    df,                           # DataFrame with potential protected attributes
    y_pred,                       # Predicted labels (binary 0/1)
    y_true,                       # Optional true labels (required for EO/EOD/PP)
    protected_columns=None,       # Auto-detect if None
    metrics=['demographic_parity_difference', 'equal_opportunity_difference'],
    threshold=0.1,                # Default threshold for all metrics
    thresholds={                  # Optional per-metric thresholds (overrides threshold)
        'demographic_parity_difference': 0.1,
        'equal_opportunity_difference': 0.05
    },
    min_group_size=30
)

# Review violations (sorted by disparity, descending)
for violation in violations:
    print(f"\nViolation: {violation.metric} on {violation.attribute}")
    print(f"  Disparity: {violation.disparity:.3f} (threshold: {violation.threshold})")
    print(f"  Severity: {violation.severity}")
    print(f"  Groups: {violation.privileged_group} vs {violation.disadvantaged_group}")
```

**Supported metrics** (via internal `_METRIC_REGISTRY`):

| Metric Name | Requires `y_true` | Description |
|---|---|---|
| `demographic_parity_difference` | No | Gap in positive prediction rates <!--cs-->**[Checked]**<!--/cs--> |
| `equal_opportunity_difference` | Yes | Gap in true positive rates <!--cs-->**[Checked]**<!--/cs--> |
| `equalized_odds_difference` | Yes | Max of TPR and FPR differences <!--cs-->**[Checked]**<!--/cs--> |
| `predictive_parity_difference` | Yes | Gap in precision across groups <!--cs-->**[Checked]**<!--/cs--> |
**Default behavior:**
- When `y_true` is provided: all 4 metrics are evaluated
- When `y_true` is omitted: only `demographic_parity_difference` is evaluated

**Error handling:**
- Unknown metric names raise `ValueError`
- Missing `y_true` for metrics that require it raises `ValueError`
- Per-attribute/metric computation failures emit `RuntimeWarning` (via `warnings.warn()`)

**FairnessViolation attributes:**
- `attribute`: The protected attribute column name
- `metric`: The fairness metric that was violated
- `value`: The metric value
- `threshold`: The threshold that was exceeded
- `severity`: `'low'`, `'medium'`, `'high'`, or `'critical'`
- `privileged_group`: The most advantaged group
- `disadvantaged_group`: The most disadvantaged group
- `disparity`: Disparity between groups

**Severity classification:**
- `critical`: disparity ≥ 0.20
- `high`: disparity ≥ 0.15
- `medium`: disparity ≥ 0.10
- `low`: disparity < 0.10

> ⚠️ **Data alignment note:** When working with train/test splits, use
> `df.loc[X_test.index]` (not `df.iloc[-len(y_test):]`) to ensure correct
> row alignment, since `train_test_split` shuffles by default.

### discover_intersectional_groups

Automatically discover meaningful intersectional group combinations.
<!-- cap-status: discover_intersectional_groups -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import discover_intersectional_groups

# Discover intersectional groups (y_pred is required: disparities are
# computed on the model's predictions)
groups = discover_intersectional_groups(
    df,
    ['gender', 'race', 'age_group'],  # protected attributes to combine
    y_pred,
    max_combinations=2,           # Max attributes to combine
    min_group_size=30,            # Minimum samples per group
    min_disparity=0.05            # Report only combinations above this gap
)

# Review discovered combinations
for group in groups:
    print(f"\nAttributes: {group['attributes']}")
    print(f"  Disparity: {group['disparity']:.3f}")
    print(f"  Subgroups: {group['n_groups']}")
    print(f"  Hidden disparity vs single attributes: {group['hidden_disparity']:.3f}")
```

### rank_fairness_issues

Perform a comprehensive fairness analysis combining auto-detection, single-attribute scanning, and intersectional analysis into a prioritized report.
<!-- cap-status: rank_fairness_issues -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import rank_fairness_issues

# Complete fairness analysis
analysis = rank_fairness_issues(
    df,                           # DataFrame with features
    y_pred,                       # Predicted labels
    y_true,                       # Optional true labels
    protected_columns=None,       # Auto-detect if None
    include_intersectional=True,  # Check intersectional groups
    min_group_size=30
)

# Available output keys
print(f"Violations found: {len(analysis['violations'])}")
print(f"Intersectional issues: {len(analysis['intersectional_issues'])}")
print(f"Proxy warnings: {len(analysis['proxy_warnings'])}")

# Violations (plain dicts, sorted by disparity)
for v in analysis['violations']:
    print(f"  {v['attribute']} [{v['metric']}]: {v['disparity']:.3f} ({v['severity']})")
    print(f"    {v['privileged_group']} vs {v['disadvantaged_group']}")

# Detected protected attributes (plain dicts)
for attr in analysis['detected_attributes']:
    print(f"  - {attr['column']} ({attr['category']})")

# Proxy warnings
for warning in analysis['proxy_warnings']:
    print(f"  - {warning['column']} correlates with {warning['protected_attribute']}")

# Prioritized recommendations
for rec in analysis['recommendations']:
    print(f"  → {rec}")
```

**Return dict keys:**
- `detected_attributes`: Auto-detected candidates as plain dicts (`column`, `confidence`, `reason`, `category`, ...)
- `violations`: List of violation dicts (`attribute`, `metric`, `value`, `threshold`, `severity`, `privileged_group`, `disadvantaged_group`, `disparity`), sorted by disparity
- `intersectional_issues`: Intersectional disparities (if `include_intersectional=True`)
- `proxy_warnings`: Potential proxy feature warnings
- `summary`: Overall summary statistics
- `recommendations`: Prioritized actionable recommendations

### Auto-Discovery in Practice

Complete workflow example:

```python
import pandas as pd
from vfairness import (
    detect_protected_attributes,
    identify_proxy_features,
    scan_fairness_violations,
    rank_fairness_issues
)

# Load your data
df = pd.read_csv('data.csv')
y_true = df['outcome']
y_pred = model.predict(df[features])

# Step 1: Auto-detect protected attributes
candidates = detect_protected_attributes(df)
print(f"Found {len(candidates)} potential protected attributes")

# Step 2: Check for proxy features (one attribute per call)
if candidates:
    protected_cols = [c.column for c in candidates if c.confidence > 0.5]
    proxies = [p for col in protected_cols
               for p in identify_proxy_features(df, col)]
    print(f"Found {len(proxies)} potential proxy features")

# Step 3: Scan for violations (all 4 metrics when y_true provided)
violations = scan_fairness_violations(df, y_pred, y_true)
print(f"Found {len(violations)} fairness violations")

# Step 4: Get complete analysis with rankings
analysis = rank_fairness_issues(
    df, y_pred, y_true,
    include_intersectional=True
)

# Print actionable summary
print("\n=== Fairness Analysis Summary ===")
print(f"Violations: {len(analysis['violations'])}")
print(f"Intersectional issues: {len(analysis['intersectional_issues'])}")
print(f"Proxy warnings: {len(analysis['proxy_warnings'])}")

for v in analysis['violations']:
    print(f"  {v['attribute']} [{v['metric']}]: {v['disparity']:.3f} ({v['severity']})")

if analysis['recommendations']:
    print(f"\nTop recommendation: {analysis['recommendations'][0]}")
```

---

## Statistical Significance and Robustness

vfairness provides formal hypothesis testing and robustness verification methods for rigorous fairness audits, based on best practices from academic research including DiCiccio et al. (2020) on permutation tests and Kearns et al. (2018) on fairness gerrymandering.

### Permutation Testing

Permutation tests provide the gold standard for formal hypothesis testing in fairness audits. They test whether observed disparities are statistically significant without making distributional assumptions.

```python
from vfairness import (
    permutation_test,
    permutation_test_demographic_parity,
    permutation_test_equal_opportunity,
)

# General permutation test with custom metric function
def dp_diff(y_pred, sensitive_attr):
    groups = np.unique(sensitive_attr)
    rates = [np.mean(y_pred[sensitive_attr == g]) for g in groups]
    return rates[0] - rates[1]

result = permutation_test(
    y_pred, gender, dp_diff,
    n_permutations=10000,      # Number of permutations (≥10000 recommended)
    alternative='two-sided',   # 'two-sided', 'greater', or 'less'
    random_state=42
)

print(f"Observed statistic: {result.observed_statistic:.4f}")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant at 0.05: {result.significant_at_05}")
print(f"Significant at 0.01: {result.significant_at_01}")

# Convenience functions for common metrics
dp_result = permutation_test_demographic_parity(y_pred, gender, n_permutations=10000)
eo_result = permutation_test_equal_opportunity(y_true, y_pred, gender, n_permutations=10000)
```

**Interpreting P-values**:
- p < 0.05: Statistically significant evidence of disparity (reject H₀)
- p ≥ 0.05: Insufficient evidence to conclude disparity exists
- Always report effect size alongside p-value (a tiny disparity can be statistically significant with large samples)

### Contingency Table Tests

Test independence between predictions and protected attributes using Chi-square or Fisher's exact test.

```python
from vfairness import contingency_test, test_equalized_odds_chi_square

# Automatic selection between Chi-square and Fisher's exact
result = contingency_test(
    y_pred, gender,
    min_expected=5.0  # Use Fisher's exact if expected counts < 5
)

print(f"Test used: {result.test_used}")
print(f"Statistic: {result.statistic:.3f}")
print(f"P-value: {result.p_value:.4f}")

# cramers_v is Optional[float]. Fisher's exact produces no Cramer's V, and
# neither does a degenerate table, so on exactly the small-count path this
# example selects it is None. None is "could not check", not an effect of zero.
if result.cramers_v is None:
    print(f"Cramer's V (effect size): COULD NOT CHECK ({result.test_used})")
else:
    print(f"Cramer's V (effect size): {result.cramers_v:.3f}")

# significant_at_05 is Optional[bool]. A bare `if result.significant_at_05:`
# reads the None back as "not significant", which is the two-state collapse
# this result type exists to refuse.
if result.significant_at_05 is None:
    print("Significant at 0.05: COULD NOT CHECK - no test was run")
else:
    print(f"Significant at 0.05: {result.significant_at_05}")

# Test equalized odds (TPR and FPR parity separately)
eo_results = test_equalized_odds_chi_square(y_true, y_pred, gender)
print(f"TPR parity p-value: {eo_results['tpr_test'].p_value:.4f}")
print(f"FPR parity p-value: {eo_results['fpr_test'].p_value:.4f}")
```

### Robust Statistics

Outliers disproportionately affect fairness metrics for small subgroups. Robust statistics reduce sensitivity to outliers.

```python
from vfairness import compute_robust_metrics, robust_fairness_comparison

# Compute robust versions of metrics per group
errors = np.abs(y_true - y_pred)
robust_results = compute_robust_metrics(
    errors, gender,
    trim_proportion=0.1,          # 10% trimmed mean
    winsorize_limits=(0.05, 0.05), # 5% Winsorization each end
    divergence_threshold=0.2       # Flag if >20% divergence
)

for group, result in robust_results.items():
    print(f"\n{group}:")
    print(f"  Standard mean: {result.standard_value:.4f}")
    print(f"  Trimmed mean:  {result.trimmed_value:.4f}")
    print(f"  Median:        {result.median_value:.4f}")
    if result.outlier_influence_detected:
        print(f"  WARNING: Outliers significantly affect this group's metrics")

# Compare standard vs robust disparity
comparison = robust_fairness_comparison(y_true, y_pred, gender, metric='mae')
print(f"\nStandard disparity: {comparison['standard_disparity']:.4f}")
print(f"Robust disparity:   {comparison['robust_disparity']:.4f}")
if comparison['conclusion_changed']:
    print("WARNING: Conclusions differ when using robust statistics")
```

### Sensitivity Analysis

Stress-test fairness conclusions to verify they hold under real-world conditions.

```python
from vfairness import sensitivity_analysis, stress_test_fairness

# Test sensitivity to label noise
result = sensitivity_analysis(
    y_pred, gender, dp_diff,
    perturbation_type='label_noise',  # 'label_noise', 'group_noise', 'subsample'
    perturbation_rate=0.05,           # 5% perturbation
    n_iterations=100,
    robustness_threshold=0.1
)

print(f"Original metric: {result.original_metric:.4f}")
print(f"Mean under perturbation: {result.mean_perturbed:.4f}")
print(f"Max deviation: {result.max_deviation:.4f}")

# robustness_score is Optional[float] and is_robust is Optional[bool]. Both are
# None when the metric itself could not be computed on this data, so there was
# nothing to perturb. A single-group input reaches that branch. Measured:
# robustness_score=None, is_robust=None, original_metric=nan.
if result.robustness_score is None:
    print("Robustness: COULD NOT CHECK - the metric is undefined on this data")
else:
    print(f"Is robust: {result.is_robust}")
    print(f"Robustness score: {result.robustness_score:.2f}")

# Comprehensive stress testing
stress_results = stress_test_fairness(
    y_pred, gender, dp_diff,
    perturbation_budgets=[0.01, 0.05, 0.10],
    n_iterations=100
)

print(f"\nOriginal metric: {stress_results['original_metric']:.4f}")
print(f"Worst-case metric: {stress_results['worst_case_metric']:.4f}")
print(f"Overall robust: {stress_results['overall_robust']}")
```

### Subgroup Robustness Audit (Fairness Gerrymandering Detection)

Detect hidden disparities in intersectional subgroups that may not appear in aggregate analysis.

```python
from vfairness import subgroup_robustness_audit
import pandas as pd

# Create DataFrame of protected attributes
attrs = pd.DataFrame({
    'gender': gender,
    'age_group': age_group,
    'race': race
})

result = subgroup_robustness_audit(
    y_pred, attrs,
    y_true=y_true,                # Required for TPR/FPR metrics
    min_subgroup_size=30,         # Minimum samples per subgroup
    disparity_threshold=0.10,     # Flag subgroups with >10% disparity
    metric='positive_rate'        # 'positive_rate', 'tpr', 'fpr', 'error_rate'
)

print(f"Subgroups analyzed: {result.n_subgroups_analyzed}")
print(f"Subgroups flagged: {result.n_subgroups_flagged}")
print(f"Gerrymandering detected: {result.gerrymandering_detected}")

if result.gerrymandering_detected:
    print(f"\nWorst subgroup: {result.worst_subgroup}")
    print(f"Worst disparity: {result.worst_disparity:.1%}")
    print("\nFlagged subgroups:")
    for subgroup, disparity in result.flagged_subgroups[:5]:
        print(f"  {subgroup}: {disparity:+.1%} vs overall")
```

### Comprehensive Fairness Testing

Run all significance tests with automatic multiple comparison correction.

```python
from vfairness import comprehensive_fairness_test

results = comprehensive_fairness_test(
    y_true, y_pred, gender,
    metrics=['demographic_parity', 'equal_opportunity', 'predictive_parity'],
    n_permutations=5000,
    alpha=0.05
)

print(f"Tests performed: {results['n_tests']}")
print(f"Any significant: {results['any_significant']}")
print(f"Significant metrics: {results['significant_metrics']}")

print("\nRaw vs adjusted p-values:")
for metric, p_raw, p_adj in zip(
    results['metric_tests'].keys(),
    results['p_values'],
    results['adjusted_p_values']
):
    print(f"  {metric}: raw={p_raw:.4f}, adjusted={p_adj:.4f}")
```

### Best Practices for Statistical Testing

| Scenario | Recommended Approach |
|----------|---------------------|
| **Formal audit / regulatory** | Permutation test with ≥10,000 iterations |
| **Quick exploration** | Chi-square / Fisher's exact test |
| **Small subgroups** | Robust statistics + `bayesian_proportion_ci` / `bayesian_difference_ci` (group rates; a DISPARITY CI is always bootstrap) |
| **Multiple groups** | Apply Benjamini-Hochberg FDR correction |
| **Production deployment** | Stress test with sensitivity analysis |
| **Intersectional analysis** | Subgroup robustness audit |

---

## Classification Metrics

All classification metrics work with binary predictions (0 or 1).

**Every metric in this section returns NaN when it could not be computed.** A
group below `min_group_size` is dropped from the comparison, and if that leaves
fewer than two groups the metric returns `float('nan')`, not `0.0` and not `1.0`.
Those two values are the perfect-parity readings, and returning one of them for a
comparison that never ran would report an all-clear nobody measured. The same
holds when a metric is undefined on the data it was given, for example an
error-rate metric on a group with no positive labels.

The `*_with_ci` variants express the same state as a `StatisticalResult` whose
`point_estimate`, `lower_bound` and `upper_bound` are all NaN.

NaN is not a small number, and it is not a large one either. **Every comparison
against NaN is `False`**, both `nan < 0.1` and `nan > 0.1`, so which way a naive
check breaks depends on how you wrote it. A guard phrased as
`if value > threshold: fail()` never fires, and an unmeasured metric slides
through as though it had passed. A guard phrased as
`if value < threshold: approve()` happens to fall through to the else branch,
which is the safe direction by accident rather than by design.

Do not rely on which side of the comparison you happened to write. Rule out NaN
explicitly with `math.isnan()` first; see
[Three States, Never Two](#three-states-never-two). The library's own gates do
exactly that and fail closed: `assert_fairness` raises rather than returning, and
`ModelFairnessGate.evaluate` returns `BLOCKED`.

Verified by execution on 2026-08-28 across every metric in this section and the
next, using 105 rows in one group and 15 in another at `min_group_size=30`.

### demographic_parity_difference

**What it measures**: The gap in positive prediction rates between groups.
<!-- cap-status: demographic_parity_difference -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


**Interpretation**:
- 0 = Perfect parity (all groups get positive predictions at the same rate)
- 0.1 = 10 percentage point difference between highest and lowest group
- Values > 0.1 often considered problematic

```python
from vfairness import demographic_parity_difference

dp = demographic_parity_difference(y_true, y_pred, sensitive_attr)
```

**Example**: If men get approved for loans 80% of the time and women 60% of the time, the difference is 0.20.

---

### demographic_parity_ratio

**What it measures**: The ratio of positive prediction rates (lowest / highest).
<!-- cap-status: demographic_parity_ratio -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 1.0 = Perfect parity
- 0.8 = The disadvantaged group gets 80% of the rate of the advantaged group
- Values < 0.8 may violate the "80% rule" (legal threshold in some contexts)

```python
from vfairness import demographic_parity_ratio

ratio = demographic_parity_ratio(y_true, y_pred, sensitive_attr)
if ratio < 0.8:
    print("Warning: Potential disparate impact detected")
```

---

### equal_opportunity_difference

**What it measures**: The gap in True Positive Rates (TPR) between groups.
<!-- cap-status: equal_opportunity_difference -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


**Interpretation**:
- 0 = Among qualified candidates, all groups have equal chance of positive prediction
- Focuses only on people who "deserve" a positive outcome

```python
from vfairness import equal_opportunity_difference

eo = equal_opportunity_difference(y_true, y_pred, sensitive_attr)
```

**Example**: If 90% of qualified men get approved but only 70% of qualified women, the difference is 0.20.

---

### equalized_odds_difference

**What it measures**: The maximum of TPR difference and FPR difference.
<!-- cap-status: equalized_odds_difference -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


**Interpretation**:
- 0 = Error rates are equal across groups for both positive and negative cases
- Stricter than equal opportunity (considers both types of errors)

```python
from vfairness import equalized_odds_difference

eod = equalized_odds_difference(y_true, y_pred, sensitive_attr)
```

---

### predictive_parity_difference

**What it measures**: The gap in precision (positive predictive value) between groups.
<!-- cap-status: predictive_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 0 = When the model says "yes", it's equally reliable for all groups
- Important when the meaning of a positive prediction should be consistent

```python
from vfairness.evaluation.vfairness_metrics.classification import predictive_parity_difference

pp = predictive_parity_difference(y_true, y_pred, sensitive_attr)

# With a confidence interval (returns a StatisticalResult). ER-005 is a sealed
# sufficiency metric, so it should carry uncertainty:
from vfairness import predictive_parity_difference_with_ci
r = predictive_parity_difference_with_ci(y_true, y_pred, sensitive_attr)
```

---

### calibration_difference

**What it measures**: How well probability predictions match actual outcomes, compared across groups.
<!-- cap-status: calibration_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Requires**: Probability predictions (`y_prob`), not just binary predictions.

```python
from vfairness.evaluation.vfairness_metrics.classification import calibration_difference

cal = calibration_difference(y_true, y_pred, sensitive_attr, y_prob)
```

---

### calibration_in_the_large / calibration_slope
<!-- cap-status: calibration_in_the_large -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**What they measure**: The TRIPOD recalibration validity diagnostics (Van Calster et al. 2016), from a Cox recalibration `logit(P(y=1)) = alpha + beta * logit(p)`. `calibration_in_the_large` is |alpha| (0 = predicted risks match the observed rate on average; the seal gates |CITL| <= 0.05); `calibration_slope` is beta (1.0 = neither over- nor under-fit; the seal gates the slope to [0.8, 1.2]).

**Family**: sufficiency (recalibration validity).

```python
from vfairness import calibration_in_the_large, calibration_slope

citl = calibration_in_the_large(y_true, y_prob, protected_attr)   # .overall_value = |CITL|
slope = calibration_slope(y_true, y_prob, protected_attr)         # .overall_value = beta
```

---

### integrated_calibration_index

**What it measures**: ICI = mean |p − g(p)|, where g is a Gaussian-kernel smoothed estimate of the observed event rate (Austin & Steyerberg 2019). Grid-free, unlike bin-ECE. The sealed EU-healthcare disparity statistic is the between-group ICI gap (`.ici_disparity`).
<!-- cap-status: integrated_calibration_index -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: sufficiency.

```python
from vfairness import integrated_calibration_index

ici = integrated_calibration_index(y_true, y_prob, protected_attr)   # .overall_ici, .ici_disparity
```

---

### multicalibration

**What it measures**: The multicalibration error `alpha` (Hebert-Johnson et al. 2018): the worst calibration violation over all (group × prediction-bin) cells with support. Small only when the model is well calibrated within EVERY subgroup, which a single overall-ECE gate can hide. The EU-healthcare sufficiency PRIMARY (sealed on the ICI / smooth-ECE disparity).
<!-- cap-status: multicalibration -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: sufficiency (state of the art).

```python
from vfairness import multicalibration

mc = multicalibration(y_true, y_prob, protected_attr)   # .alpha (worst subgroup violation)
```

---

### fpr_parity_difference

**What it measures**: The gap in false-positive rate (FPR = P(pred=1 | true=0)) across groups, aka predictive equality. This is the FPR leg of equalized odds exposed as a first-class metric.
<!-- cap-status: fpr_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: separation (Ŷ ⊥ A | Y).

```python
from vfairness.evaluation.vfairness_metrics.classification import fpr_parity_difference

fpr = fpr_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### fnr_parity_difference

**What it measures**: The gap in false-negative rate (FNR = P(pred=0 | true=1) = 1 − TPR) across groups. By this identity it equals the equal-opportunity (TPR) difference; the FNR framing surfaces the "missed positives" harm.
<!-- cap-status: fnr_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: separation (Ŷ ⊥ A | Y).

```python
from vfairness.evaluation.vfairness_metrics.classification import fnr_parity_difference

fnr = fnr_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### accuracy_parity_difference

**What it measures**: The gap in overall accuracy (P(pred == true)) across groups.
<!-- cap-status: accuracy_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Note**: accuracy parity is a **performance** metric, not a group-fairness criterion in the independence / separation / sufficiency sense, and it can hide FP/FN trade-offs. Treat it as a performance/robustness signal, not a standalone fairness verdict.

```python
from vfairness.evaluation.vfairness_metrics.classification import accuracy_parity_difference

acc = accuracy_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### worst_group_accuracy

**What it measures**: The minimum per-group accuracy (Sagawa et al. 2020, worst-group robustness). A **performance/robustness** statistic, not a fairness-difference metric.
<!-- cap-status: worst_group_accuracy -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness.evaluation.vfairness_metrics.classification import worst_group_accuracy

wga = worst_group_accuracy(y_true, y_pred, sensitive_attr)
```

---

### disparate_impact_ratio

**What it measures**: The selection-rate RATIO (min/max) across groups: the four-fifths / adverse-impact statistic (a value below 0.80 is the EEOC screening heuristic). Distinct from `demographic_parity_difference` (a rate DIFFERENCE band): conflating a ratio floor with a difference band is a category error, so this is a separate metric.
<!-- cap-status: disparate_impact_ratio -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: independence (Ŷ ⊥ A). The US-hiring / US-lending independence PRIMARY.

```python
from vfairness import disparate_impact_ratio

dir_ = disparate_impact_ratio(y_true, y_pred, sensitive_attr)   # 1.0 = parity; seal gates on CI_lower >= 0.80
```

---

### conditional_demographic_disparity

**What it measures**: The size-weighted within-stratum selection-rate disparity that REMAINS after conditioning on a legitimate stratifier (an audited, non-proxy factor): the CJEU objective-justification mirror (Wachter, Mittelstadt & Russell 2021). A raw group gap fully explained by the legitimate factor does not count against the model.
<!-- cap-status: conditional_demographic_disparity -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: independence + justification. The EU-hiring SEALED primary.

```python
from vfairness import conditional_demographic_disparity

cdd = conditional_demographic_disparity(y_pred, sensitive_attr, strata)  # strata = the legitimate factor
```

> **Divergence from the cited paper, read before registering a margin.** The canonical CDD of
> Wachter, Mittelstadt & Russell (2021), as operationalized for example in AWS SageMaker Clarify,
> is composition based and **signed**: within each stratum it compares a protected group's share
> of the rejected pool with its share of the accepted pool
> (`DD_k = P(group=d | rejected, k) - P(group=d | accepted, k)`), size-weighted across strata.
> What vfairness returns is the size-weighted within-stratum **maximum selection-rate gap**: an
> unsigned, reference-free spread in `[0, 1]` that names no direction. The two answer related but
> different questions and diverge materially on the same data, so a tolerance calibrated on the
> CDD literature or on a Clarify run does not transfer one for one. Register the margin against
> the definition implemented here. The estimand is a recorded open decision (sealed assessment
> cells gate on the current statistic); full record in
> [`docs/DIVERGENCES.md`](DIVERGENCES.md).

---

### negative_predictive_value_difference

**What it measures**: The gap in negative predictive value (NPV = P(y=0 | pred=0) = TN/(TN+FN)) across groups: the sufficiency counterpart of predictive parity on the NEGATIVE decision.
<!-- cap-status: negative_predictive_value_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: sufficiency (Y ⊥ A | Ŷ).

```python
from vfairness import negative_predictive_value_difference

npv = negative_predictive_value_difference(y_true, y_pred, sensitive_attr)
```

---

### auroc_parity

**What it measures**: The maximum difference in per-group ROC AUC (discrimination). A VALIDITY gate, not a fairness-difference metric: it guards against a model that looks "fair" only because it is uninformative for some group (the levelling-down failure). Used as the EU-healthcare validity gate.
<!-- cap-status: auroc_parity -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: validity gate.

```python
from vfairness import auroc_parity

gap = auroc_parity(y_true, y_score, sensitive_attr)   # y_score = continuous scores / probabilities
```

---

### net_benefit_parity

**What it measures**: The between-group difference in decision-curve net benefit (Vickers & Elkin 2006) at a decision threshold: whether the model is clinically USEFUL for each group, not merely statistically fair. Guards the "fair but clinically useless" failure.
<!-- cap-status: net_benefit_parity -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: decision utility.

```python
from vfairness import net_benefit_parity

nb = net_benefit_parity(y_true, y_prob, sensitive_attr, threshold=0.5)
```

---

### conditional_adverse_impact

**What it measures**: The selection-rate disparity that REMAINS after regression-controlling for bona-fide (non-proxy) covariates: it answers the job-relatedness / levelling-down critique within legal bounds. A diagnostic, not a sealed gate.
<!-- cap-status: conditional_adverse_impact -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Family**: independence diagnostic.

```python
from vfairness.evaluation.vfairness_metrics.classification import conditional_adverse_impact

cai = conditional_adverse_impact(y_pred, sensitive_attr, covariates)
```

---

### Confidence-interval variants (sealed gates)

Every SEALED gate has a `*_with_ci` variant returning a `StatisticalResult` (point estimate + a stratified-bootstrap interval: percentile, or fold-debiased for the nonnegative max-min spread statistics), so a gate passes on affirmative CI evidence (one-sided equivalence / TOST), never a bare point estimate: `disparate_impact_ratio_with_ci`, `fpr_parity_difference_with_ci`, `negative_predictive_value_difference_with_ci`, `conditional_demographic_disparity_with_ci`, `pricing_disparity_with_ci`, `integrated_calibration_index_with_ci`, `multicalibration_with_ci`. Statistics whose shape needs `y_prob` / strata / continuous controls get their interval through `bootstrap_over_index` (resamples row indices, stratified by group). Neither BCa nor a Bayesian credible interval is ever built on this path: `stratified_bootstrap_ci` refuses `method='bca'`, and no Bayesian estimator for a disparity metric exists in this library. Check `StatisticalResult.method` for the estimator that actually ran.

```python
from vfairness import disparate_impact_ratio_with_ci

r = disparate_impact_ratio_with_ci(y_true, y_pred, sensitive_attr)
# r.point_estimate, r.lower_bound, r.upper_bound  -> gate on r.lower_bound >= 0.80
```

---

## Regression Metrics

For models that predict continuous values (not just 0/1).

**The NaN contract is the same as for the classification metrics above.**
`mae_parity_difference`, `rmse_parity_difference`, `mean_prediction_difference`
and `r2_parity_difference` return `float('nan')` when fewer than two groups
survive `min_group_size`, and their `*_with_ci` variants return an all-NaN
`StatisticalResult`.

`residual_bias` is the exception in shape, not in contract: it returns a
`Dict[str, float]` keyed by group, and a group that did not survive
`min_group_size` is simply absent from the mapping rather than present with a
substituted `0.0`. Check the keys against `data_info["valid_groups"]` before
concluding a group is unbiased.

### mae_parity_difference

**What it measures**: The gap in Mean Absolute Error between groups.
<!-- cap-status: mae_parity_difference -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


**Interpretation**:
- 0 = Model makes equally accurate predictions for all groups
- Higher values = model is more accurate for some groups than others

```python
from vfairness import mae_parity_difference

mae_diff = mae_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### rmse_parity_difference

**What it measures**: The gap in Root Mean Squared Error between groups.
<!-- cap-status: rmse_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**: Similar to MAE parity, but penalizes large errors more heavily.

```python
from vfairness import rmse_parity_difference

rmse_diff = rmse_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### r2_parity_difference

**What it measures**: The max-min spread of per-group R-squared (coefficient of determination). It answers whether the model fits some groups better than others.
<!-- cap-status: r2_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 0 = the model explains variance equally well across groups
- Larger = the model fits some groups worse (its predictions are less reliable for them)

Returns a `float`. NaN (insufficient evidence) when fewer than two groups are assessable.

```python
from vfairness import r2_parity_difference

r2_gap = r2_parity_difference(y_true, y_pred, sensitive_attr)
```

---

### residual_bias

**What it measures**: The per-group mean residual (`y_true - y_pred`), i.e. whether the model systematically over- or under-predicts within each group.
<!-- cap-status: residual_bias -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Returns**: a `Dict[str, float]` mapping each group name to its mean residual (this is the one regression metric that does not return a single float; a positive value means the model under-predicts that group's outcome, a negative value means it over-predicts).

```python
from vfairness import residual_bias

per_group = residual_bias(y_true, y_pred, sensitive_attr)
# e.g. {"A": 0.06, "B": -0.02}  -> group A is under-predicted, B slightly over-predicted
```

---

### mean_prediction_difference

**What it measures**: The gap in average predictions between groups.
<!-- cap-status: mean_prediction_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 0 = Model predicts similar average values for all groups
- Non-zero = Model systematically predicts higher/lower for some groups

```python
from vfairness import mean_prediction_difference

mpd = mean_prediction_difference(y_true, y_pred, sensitive_attr)
```

**Example**: If average predicted salary is $80k for men and $65k for women, the difference is $15k.

---

### pricing_disparity

**What it measures**: The residual difference in a continuous priced outcome (APR / note rate / premium) across groups AFTER controlling for legitimate risk factors (an OLS residual). Binary fairness metrics cannot measure price; the lending profile co-seals this where the scope covers price.
<!-- cap-status: pricing_disparity -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 0 = no prohibited-basis pricing effect once legitimate risk is controlled for
- Non-zero = groups are priced differently beyond what the risk controls explain

```python
from vfairness import pricing_disparity

pd = pricing_disparity(price, sensitive_attr, controls)   # controls = legitimate risk covariates (or None)
```

---

## Ranking Metrics

For evaluating fairness in ranked lists (search results, recommendations, etc.).

> **These four metrics DO implement the third state, and this box used to say
> they did not.**
>
> It disclosed a real defect and told you to work around it: all four returned
> the perfect-parity value when fewer than two groups survived `min_group_size`,
> which is indistinguishable from a measured fair ranking. That was closed on
> 2026-09-06. Re-measured on 2026-09-07 with the same reproduction, 105 items in
> one group and 15 in another at `min_group_size=30`:
>
> | Metric | Was | Is |
> |--------|-----|----|
> | `exposure_parity_difference` | `0.0` | `nan` |
> | `exposure_parity_ratio` | `1.0` | `nan` |
> | `attention_weighted_rank_fairness` | `.value == 0.0`, `is_fair` True | `.value` `nan`, `is_fair` `None` |
> | `normalized_discounted_kl_divergence` | `0.0` | `nan` |
>
> **The group-size guard this box used to prescribe is no longer needed.** It
> counted items per group and raised before calling the metric; the metric now
> refuses on its own and warns while doing it. Keeping that guard would only
> hide an honest NaN behind one of your own, and it would go stale the moment
> `min_group_size` moved.
>
> What you do still have to do is the same thing every metric here asks of you:
> treat NaN as could-not-check rather than coercing it. `value or 0.0` and
> `np.nan_to_num(value)` both turn "never measured" into "perfectly fair", which
> is the defect this change removed from the library and the one place it can
> still be reintroduced.

### exposure_parity_difference

**What it measures**: The gap in visibility/exposure between groups in a ranking.
<!-- cap-status: exposure_parity_difference -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 0 = All groups receive equal exposure
- Higher values = Some groups are systematically ranked higher

```python
from vfairness import exposure_parity_difference

# Rankings (lower = better position)
rankings = np.array([0, 1, 2, 3, 4, 5, 6, 7])
groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])

diff = exposure_parity_difference(
    rankings, groups,
    min_group_size=2,
    exposure_type='log'  # 'log', 'linear', or 'geometric'
)
```

**Exposure Types**:
- `'log'`: Logarithmic decay (default) - 1/log2(rank+1)
- `'linear'`: Linear decay - 1/rank
- `'geometric'`: Exponential decay - 0.9^rank

---

### exposure_parity_ratio

**What it measures**: The ratio of exposure between groups (min/max).
<!-- cap-status: exposure_parity_ratio -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


**Interpretation**:
- 1.0 = Perfect parity
- < 0.8 = May violate the "80% rule" for disparate impact

```python
from vfairness import exposure_parity_ratio

ratio = exposure_parity_ratio(rankings, groups, min_group_size=2)
if ratio < 0.8:
    print("Warning: Potential disparate impact in ranking exposure")
```

---

### attention_weighted_rank_fairness

**What it measures**: Fairness accounting for realistic user attention patterns.
<!-- cap-status: attention_weighted_rank_fairness -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import attention_weighted_rank_fairness

result = attention_weighted_rank_fairness(
    rankings, groups,
    min_group_size=2,
    attention_model='position'  # or 'cascade'
)

print(f"Fairness value: {result.value:.3f}")
print(f"Group exposures: {result.group_exposures}")
print(f"Group attentions: {result.group_attentions}")
print(f"Is fair (threshold {result.threshold}): {result.is_fair}")
```

**Attention Models**:
- `'position'`: Position-based model (exposure decays with rank)
- `'cascade'`: Cascade model (users stop scanning at some point)

---

### normalized_discounted_kl_divergence

**What it measures**: How much the group distribution in rankings diverges from a target distribution.
<!-- cap-status: normalized_discounted_kl_divergence -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import normalized_discounted_kl_divergence

# Default: uniform target distribution
ndkl = normalized_discounted_kl_divergence(rankings, groups, min_group_size=2)

# Custom target distribution
ndkl = normalized_discounted_kl_divergence(
    rankings, groups,
    target_distribution={'A': 0.6, 'B': 0.4},
    min_group_size=2,
    top_k=10  # Only consider top 10 positions
)
```

---

### get_ranking_group_metrics

**What it provides**: Per-group statistics for rankings.
<!-- cap-status: get_ranking_group_metrics -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


```python
from vfairness import get_ranking_group_metrics

metrics = get_ranking_group_metrics(rankings, groups, min_group_size=2)

for group, stats in metrics.items():
    print(f"{group}:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg Position: {stats['avg_position']:.2f}")
    print(f"  Avg Exposure: {stats['avg_exposure']:.3f}")
    print(f"  Position Range: {stats['min_position']}-{stats['max_position']}")
```

---

## Fairness Reports

Generate comprehensive assessments with a single function call.

### classification_fairness_report

```python
<!-- cap-status: classification_fairness_report -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->

from vfairness import classification_fairness_report, print_report

report = classification_fairness_report(
    y_true,
    y_pred,
    sensitive_attr,
    y_prob=None,              # Optional: probability predictions
    min_group_size=30,        # Minimum samples per group
    missing_strategy='exclude' # How to handle missing values
)

# Print formatted report
print_report(report)

# Access specific parts
print(report['metrics'])                    # All computed metrics
print(report['assessment']['fairness_score'])  # Optional[float] in 0-1, or None
print(report['assessment']['failed_metrics'])  # Which metrics failed thresholds
print(report['assessment']['not_assessable_metrics'])  # Which were never graded
print(report['group_stats'])                # Per-group statistics
```

`fairness_score` is `Optional[float]`. Rule out `None` before formatting it: see
[Three States, Never Two](#three-states-never-two).

**Report Structure** (observed on a 200-row, two-group classification run):
```python
{
    'task_type': 'classification',
    'methodology_version': 'M1.1',
    'metrics': {
        'demographic_parity_difference': 0.060,
        'demographic_parity_ratio': 0.885,
        'equalized_odds_difference': 0.072,
        'equal_opportunity_difference': 0.049,
        'predictive_parity_difference': 0.019,
        # An ungraded metric is NaN here, never 0.0 or 1.0.
    },
    'group_stats': {
        'Male': {'size': 100, 'positive_rate': 0.45, 'tpr': 0.80, ...},
        'Female': {'size': 100, 'positive_rate': 0.40, 'tpr': 0.75, ...}
    },
    'assessment': {
        'fairness_score': 1.0,   # share of ASSESSABLE metrics that passed,
                                 # or None when nothing was assessable
        'assessable': True,      # False when < 2 groups survive min_group_size
        # Each of the three lists holds MetricStatusEntry dicts
        # ({metric, value, threshold, status}), not bare metric names.
        'passed_metrics': [
            {'metric': 'demographic_parity_difference', 'value': 0.06,
             'threshold': 0.1, 'status': 'PASS'},
            # ... one entry per graded metric
        ],
        'failed_metrics': [],
        # Metrics that were never graded (NaN, or fewer than two valid groups)
        # land here, with status 'NOT_ASSESSABLE', threshold None and a reason.
        # They are excluded from fairness_score and are NOT counted as failures.
        'not_assessable_metrics': [],
        'insufficient_evidence_groups': [],  # groups below min_group_size
        'summary': "5/5 metrics within thresholds (data provenance: 200 of 200 "
                   "rows assessed, 0 excluded, missing_strategy='exclude')"
    },
    'data_info': {
        'original_size': 200,
        'final_size': 200,
        'n_excluded': 0,
        'missing_strategy': 'exclude',
        'n_samples': 200,
        'n_groups': 2,
        'valid_groups': ['Female', 'Male'],
        'invalid_groups': [],
        'group_sizes': {'Female': 100, 'Male': 100},
        'is_intersectional': False,
    },
    'thresholds_used': {...},
    # Regression reports also carry 'residual_bias'.
    # include_ci=True also adds 'metrics_with_ci', 'effect_sizes' and
    # 'statistical_validation'.
}
```

Every `summary` string ends with the data-provenance clause shown above,
including on a NOT ASSESSABLE report, so the verdict line always discloses how
the rows behind it were handled. Read `data_info` for the values themselves.

---

### regression_fairness_report

```python
<!-- cap-status: regression_fairness_report -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->

from vfairness import regression_fairness_report

report = regression_fairness_report(
    y_true,
    y_pred,
    sensitive_attr,
    min_group_size=30
)
```

---

## Statistical Validation

Robust statistical methods for validating fairness measurements.

### Bootstrap Confidence Intervals

`bootstrap_ci(data, statistic)` resamples a 1-D data array and returns a `StatisticalResult` with `point_estimate`, `lower_bound`, `upper_bound` and `standard_error` attributes:

```python
import numpy as np
from vfairness import bootstrap_ci

# Any 1-D array of observations, e.g. per-batch disparity measurements
rates = np.array([0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.10, 0.14,
                  0.12, 0.16, 0.07, 0.11, 0.13, 0.09, 0.12, 0.10])

ci = bootstrap_ci(
    rates,
    np.mean,                 # any callable: ndarray -> float
    n_bootstrap=5000,
    confidence_level=0.95,
    random_state=42
)

print(f"Point estimate: {ci.point_estimate:.3f}")
print(f"95% CI: [{ci.lower_bound:.3f}, {ci.upper_bound:.3f}]")

# standard_error is Optional[float]: None when the resample produced no spread
# to estimate it from (a single observation, or an empty array).
if ci.standard_error is None:
    print("Bootstrap SE: COULD NOT CHECK - too few observations to resample")
else:
    print(f"Bootstrap SE: {ci.standard_error:.4f}")
```

### Bayesian Credible Intervals

Beta-Binomial credible intervals for selection rates and for the difference between two group rates. Both return `StatisticalResult`:

```python
from vfairness import bayesian_proportion_ci, bayesian_difference_ci

# Credible interval for one group's selection rate
result = bayesian_proportion_ci(successes=45, trials=100, confidence_level=0.95)
print(f"Posterior rate: {result.point_estimate:.3f} "
      f"[{result.lower_bound:.3f}, {result.upper_bound:.3f}]")

# Credible interval for the difference between two group rates
diff = bayesian_difference_ci(
    successes1=45, trials1=100,   # group A: 45/100 selected
    successes2=30, trials2=100,   # group B: 30/100 selected
    confidence_level=0.95,
    random_state=42
)
print(f"Rate difference: {diff.point_estimate:.3f} "
      f"[{diff.lower_bound:.3f}, {diff.upper_bound:.3f}]")
```

### Effect Sizes

`compute_effect_sizes` returns pairwise effect sizes between demographic groups as a dict keyed by group pair:

```python
from vfairness import compute_effect_sizes

effects = compute_effect_sizes(y_true, y_pred, sensitive_attr)

for pair, eff in effects.items():
    print(f"{pair}: Cohen's d = {eff['cohens_d_positive_rate']:.3f} "
          f"({eff['interpretation']})")
# Interpretation: 'negligible', 'small', 'medium', or 'large' effect
```

Each pair's dict also carries `cohens_h_positive_rate`, `risk_ratio` and `odds_ratio` (each a `(value, lower_ci, upper_ci)` tuple), the per-group positive rates and the group sizes.

### Risk Ratios

`risk_ratio` takes event counts and returns a `(risk_ratio, lower_95_ci, upper_95_ci)` tuple (log-scale CI):

```python
from vfairness import risk_ratio

rr, lower, upper = risk_ratio(
    events1=45, total1=100,   # group A: 45/100 positive
    events2=30, total2=100    # group B: 30/100 positive
)

print(f"Risk Ratio: {rr:.3f} [95% CI {lower:.3f}, {upper:.3f}]")
```

### Multiple Testing Corrections

When computing many metrics simultaneously, apply corrections to control false discovery rate. The function takes an array of p-values and returns a `MultipleTestingResult`:

```python
import numpy as np
from vfairness import apply_multiple_testing_correction

# Raw p-values from multiple metric tests
p_values = np.array([0.001, 0.03, 0.02, 0.15])

# Benjamini-Hochberg FDR correction
result = apply_multiple_testing_correction(
    p_values,
    method='fdr',   # 'fdr' (alias 'benjamini_hochberg'), 'bonferroni', or 'none'
    alpha=0.05
)

print(f"Adjusted p-values: {result.adjusted_p_values}")
print(f"Significant after correction: {result.rejection_mask}")
print(f"Number rejected: {result.n_rejected}")
```

---

## MLOps Integration

### Deployment Gates and the Third State

`ModelFairnessGate.evaluate()` returns a `GateDecision` whose `status` is
`APPROVED`, `BLOCKED` or `CONDITIONAL`. Two properties of that verdict are worth
stating explicitly, because both were wrong in earlier versions and a gate is
exactly the surface where a wrong verdict does the most damage.

**A ratio metric is graded in its own direction.** A ratio such as
`disparate_impact_ratio` is higher-is-better, with `1.0` meaning parity, so its
threshold is a floor rather than a ceiling. All four corners, verified by
execution on 2026-08-28 with `thresholds={'disparate_impact_ratio': 0.8}`:

| Situation | Metric value | Gate status |
|-----------|--------------|-------------|
| Perfect parity | `1.00` | `APPROVED` |
| Protected group never selected | `0.00` | `BLOCKED`, reason: "is below the required minimum (0.8000)" |

**A metric the gate could not measure BLOCKS.** The gate fails closed: a NaN
metric is never approved, and the blocking reason says so rather than reporting
a fabricated violation.

```python
from vfairness.operations.cicd import ModelFairnessGate

gate = ModelFairnessGate(
    metrics=["demographic_parity_difference"],
    thresholds={"demographic_parity_difference": 0.1},
)
decision = gate.evaluate(y_true=y, y_pred=p, protected_attr=single_group)

decision.status           # GateStatus.BLOCKED
decision.approved         # False
decision.blocking_reasons
# ['demographic_parity_difference could not be computed on this data; the gate
#   fails closed rather than approving an unevaluated metric']
```

Note the wording of that reason. It does not claim the model breached the
threshold, which would send someone after a violation nobody measured. It says
the metric was never evaluated, and that an unevaluated metric cannot clear a
gate. That is the third state applied to a deployment decision.

### MLflow Logging

Log fairness metrics to MLflow for experiment tracking:

```python
from vfairness import FairnessAnalyzer, log_fairness_to_mlflow
import mlflow

# Create analyzer
analyzer = FairnessAnalyzer(y_true, y_pred, gender)

# Log within an MLflow run
with mlflow.start_run():
    log_fairness_to_mlflow(
        analyzer,
        prefix="fairness",          # Metric prefix
        log_artifacts=True,         # Save full report as JSON artifact
        include_group_stats=True    # Log per-group rates as metrics
    )
```

Requires `pip install mlflow` (optional dependency).

**What gets logged**:
- Metrics: `fairness.demographic_parity_difference`, etc. (plus `.ci_lower` / `.ci_upper` / `.std_error` variants when CIs are available)
- Parameters: `fairness.task_type`, `fairness.n_samples`, `fairness.n_groups`, `fairness.groups`
- Tags: `fairness.library`, `fairness.library_version`
- Artifacts: the full report JSON under `fairness_report/` (if `log_artifacts=True`)

### Pytest Assertions

Add fairness checks to your test suite:

```python
from vfairness import assert_fairness, FairnessAssertionError
import pytest

def test_model_fairness():
    """Test that model meets fairness requirements."""
    y_true, y_pred, gender = get_test_data()

    # Will raise FairnessAssertionError if thresholds exceeded
    metrics = assert_fairness(
        y_true, y_pred, gender,
        metrics=['demographic_parity_difference', 'equal_opportunity_difference'],
        thresholds={
            'demographic_parity_difference': 0.1,
            'equal_opportunity_difference': 0.1
        },
        message="Model failed fairness requirements"
    )

    # If we get here, all metrics passed
    assert metrics['demographic_parity_difference'] < 0.1

def test_fairness_with_ci():
    """Test fairness with confidence intervals."""
    metrics = assert_fairness(
        y_true, y_pred, gender,
        include_ci=True,
        n_bootstrap=1000,
        thresholds={'demographic_parity_difference': 0.1}
    )
```

**FairnessAssertionError Attributes**:
```python
try:
    assert_fairness(y_true, y_pred, gender, thresholds={'dp': 0.01})
except FairnessAssertionError as e:
    print(f"Failed metrics: {e.failed_metrics}")  # {'dp': 0.15}
    print(f"All metrics: {e.all_metrics}")        # Full results dict
```

**`assert_fairness` fails closed on a metric it could not measure.** A NaN metric
does not satisfy the gate. It raises `FairnessAssertionError` with the metric
listed under "Not measurable", and the corresponding `failed_metrics` entry
carries `not_measurable: True` alongside its reason:

```python
{
    "demographic_parity_difference": {
        "value": float("nan"),
        "threshold": 0.1,
        "reason": "NOT MEASURABLE: value is nan, so the metric was never "
                  "compared against threshold 0.1000 (fail closed)",
        "not_measurable": True,
        "ci_lower": None,
        "ci_upper": None,
    }
}
```

This is the point of the third state in a release gate. A run that measured
nothing must not be able to satisfy a fairness requirement, and every comparison
against NaN is `False`, so a hand-rolled `value > threshold` guard would never
have fired and the run would have gone green.
Verified by execution on 2026-08-28.

> **Two different functions are named `assert_fairness`. Both now fail closed.**
> Everything above is about
> `vfairness.evaluation.vfairness_metrics.integrations.assert_fairness`, which is
> what `from vfairness import assert_fairness` gives you. It takes
> `sensitive_attr` with `metrics` and `thresholds` plural.
>
> `vfairness.operations.cicd.testing.assert_fairness` is a different function
> that takes `protected_attr` with a single `metric` and `threshold`. It used to
> raise only on `TestStatus.FAILED`, so a run that measured nothing returned
> `None`, exactly like a genuine pass. That was fixed on 2026-09-07: anything
> that is not an explicit PASS is now a refusal, including a `SKIPPED` metric,
> an empty result list and a metric name that does not exist. Re-verified by
> execution on 2026-09-10 with a single-group input, where the metric is NaN:
> it raised `FairnessAssertionError` beginning `NOT MEASURABLE:`, and a typo in
> the metric name raised the same way. A measured violation still raises and a
> measured pass within threshold still returns `None`.
>
> Pick between them on signature and on how many metrics you need, not on
> safety. Either way, do not hand-roll `value > threshold` on the returned
> numbers: every comparison against NaN is `False`, so a hand-rolled guard
> reports a pass for a metric nobody measured.

### Training Loop Callback

Monitor fairness during model training:

```python
from vfairness import create_fairness_callback

callback = create_fairness_callback(
    sensitive_attr_column='gender',
    metrics=['demographic_parity_difference'],
    thresholds={'demographic_parity_difference': 0.1},
    fail_on_violation=False  # Set True to stop training on violation
)

# In your training loop
for epoch in range(n_epochs):
    train_model()
    y_pred = model.predict(X_val)

    fairness_results = callback(y_val, y_pred, sensitive_val)
    print(f"Epoch {epoch}: DP = {fairness_results['demographic_parity_difference']:.3f}")
```

---

## Visualization

vfairness provides modern, professional visualization tools for academic papers, business reports, and interactive dashboards. The library supports both Matplotlib (for static, publication-ready figures) and Plotly (for interactive web dashboards).

### Available Styles

Six professional color schemes are available for different contexts:

| Style | Best For | Description |
|-------|----------|-------------|
| `'academic'` | Research papers, journals | Muted, professional colors (default) |
| `'business'` | Corporate reports, presentations | Polished, executive-friendly palette |
| `'modern'` | Dashboards, web apps | Vibrant, contemporary colors |
| `'dark'` | Dark-themed interfaces | Optimized for dark backgrounds |
| `'nature'` | Environmental, earth-toned contexts | Organic greens and warm neutrals |
| `'accessible'` | Color-blind-safe output | High-contrast, CVD-friendly palette |

```python
from vfairness import get_available_styles, preview_palette

# List available styles
print(get_available_styles())  # ['academic', 'business', 'modern', 'dark', 'nature', 'accessible']

# Preview a color palette
fig = preview_palette('academic')
```

### Plot Fairness Metrics

Returns a Matplotlib `Axes` (as do `plot_group_comparison`,
`plot_confidence_intervals`, and `plot_effect_sizes`); save through the parent
figure via `ax.figure.savefig(...)`. Only `plot_fairness_report` returns a
`Figure` directly.

```python
from vfairness import plot_fairness_metrics

ax = plot_fairness_metrics(
    report,                    # From classification_fairness_report()
    style='academic',          # 'academic', 'business', 'modern', 'dark', 'nature', 'accessible'
    show_thresholds=True,      # Show pass/fail thresholds
    figsize=(10, 6)
)
ax.figure.savefig('fairness_metrics.png', dpi=300, bbox_inches='tight')
```

### Plot Group Comparison

```python
from vfairness import plot_group_comparison

ax = plot_group_comparison(
    report,
    style='business',
    metric='positive_rate',    # One group metric per call
    figsize=(12, 5)
)
```

### Plot Confidence Intervals

```python
from vfairness import plot_confidence_intervals

ax = plot_confidence_intervals(
    report,                    # Must include CI data
    style='academic',
    figsize=(10, 6)
)
```

### Plot Effect Sizes

```python
from vfairness import plot_effect_sizes

ax = plot_effect_sizes(
    report,
    style='academic',
    figsize=(8, 6)
)
```

### Radar Chart for Multiple Metrics

Visualize all metrics in the report at once using a radar/spider chart. This is
a Plotly function (requires `pip install plotly`) and returns a Plotly figure,
not a Matplotlib one. With `normalize=True` (the default) it uses the same
fairness axis as the SVG `radar_chart_to_svg`: each metric is plotted as a
fairness score where `1.0` (fully fair) is the outer rim and `0.0` (unfair) is
the centre, so a fair model draws a large round shape and a failing metric caves
inward, and every threshold maps to a single reference ring:

```python
from vfairness import plot_metrics_radar

fig = plot_metrics_radar(
    report,
    style='modern',
    normalize=True,   # Fairness axis: 1.0 = fair (outer rim), 0.0 = unfair (centre)
    fill=True         # Fill the radar polygon
)
fig.write_html('metrics_radar.html')
```

### Group Disparity Heatmap

Visualize pairwise disparities between groups. Also Plotly-backed (requires
`pip install plotly`) and returns a Plotly figure:

```python
from vfairness import plot_group_disparity_heatmap

fig = plot_group_disparity_heatmap(
    report,
    style='academic'
)
fig.write_html('group_disparity_heatmap.html')
```

### Comprehensive Report Plot

```python
from vfairness import plot_fairness_report

# Creates multi-panel figure with all visualizations
fig = plot_fairness_report(
    report,
    style='academic',
    figsize=(15, 10)
)
fig.savefig('full_fairness_report.png', dpi=300, bbox_inches='tight')
```

### Interactive Plotly Dashboard

Create modern interactive dashboards for web deployment:

```python
from vfairness import create_fairness_dashboard

# Returns a Plotly figure with interactive panels (requires plotly)
fig = create_fairness_dashboard(
    report,
    style='modern',
    show_annotations=True,
    height=800
)

# Display in Jupyter notebook
fig.show()

# Save as interactive HTML
fig.write_html('fairness_dashboard.html')

# Save as static image
fig.write_image('fairness_dashboard.png', scale=2)
```

The dashboard includes:
- Fairness metrics bar chart with threshold indicators
- Group comparison charts
- Confidence interval visualizations (if available)
- Interactive hover tooltips
- Responsive layout

### Save All Plots

One call writes one `format` (a single string, default `'png'`); loop to export
several formats:

```python
from vfairness import save_fairness_plots

for fmt in ['png', 'pdf', 'svg']:
    save_fairness_plots(
        report,
        output_dir='./fairness_plots',
        style='academic',
        dpi=300,                   # High resolution for print
        format=fmt
    )
# Creates: fairness_metrics.png, fairness_groups.png,
# fairness_confidence_intervals.png, fairness_effect_sizes.png,
# fairness_full_report.png (and the same for .pdf and .svg)
```

### Custom Color Palettes

Access the color palettes directly for custom visualizations:

```python
from vfairness import get_palettes

palettes = get_palettes()

# Use colors in your own plots
academic = palettes['academic']
print(academic['primary'])      # '#1e3a5f' - Deep navy
print(academic['success'])      # '#2a9d8f' - Teal green
print(academic['danger'])       # '#e76f51' - Burnt orange-red
print(academic['groups'])       # List of 8 colors for group comparisons

# Available palette keys:
# 'primary', 'secondary', 'accent1', 'accent2', 'success', 'warning',
# 'danger', 'info', 'neutral', 'light', 'background', 'background_alt',
# 'grid', 'text', 'text_muted', 'gradient', 'groups'
```

### Publication-Ready Figures

For academic papers and journals:

```python
from vfairness import plot_fairness_report, classification_fairness_report

# Generate report with confidence intervals
report = classification_fairness_report(
    y_true, y_pred, gender,
    include_ci=True,
    n_bootstrap=5000
)

# Create publication-quality figure
fig = plot_fairness_report(
    report,
    style='academic',
    figsize=(12, 8)
)

# Save at high resolution
fig.savefig(
    'fairness_figure.pdf',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
```

### Visualization Dependencies

The visualization module has optional dependencies:

```bash
# For static plots (matplotlib - usually pre-installed)
pip install matplotlib

# For the interactive dashboard, radar chart, and disparity heatmap
# (create_fairness_dashboard, plot_metrics_radar, plot_group_disparity_heatmap
# all raise ImportError without it)
pip install plotly

# For high-quality exports
pip install kaleido  # Required for fig.write_image()
```

---

## Common Parameters

These parameters are shared across most functions:

### sensitive_attr

The protected attribute(s) defining groups.

```python
# Single attribute (array, list, or pandas Series)
gender = ['M', 'M', 'F', 'F']
dp = demographic_parity_difference(y_true, y_pred, gender)

# Multiple attributes for intersectional analysis (pandas DataFrame)
import pandas as pd
attrs = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F'],
    'race': ['A', 'B', 'A', 'B']
})
dp = demographic_parity_difference(y_true, y_pred, attrs, min_group_size=1)
# Creates groups: M_A, M_B, F_A, F_B
```

---

### min_group_size

Minimum number of samples required per group. The default is 30 for
classification and regression metrics; ranking metrics (such as
`exposure_parity_difference`) default to 5.

**Why it matters**: Small groups produce unreliable statistics. A group with 5 samples could show 0% or 100% positive rate just by chance.

```python
# Strict (research standard)
dp = demographic_parity_difference(y_true, y_pred, gender, min_group_size=100)

# Relaxed (for small datasets)
dp = demographic_parity_difference(y_true, y_pred, gender, min_group_size=10)
```

Groups below the minimum size are excluded from calculations.

---

### missing_strategy

How to handle missing values (default: 'exclude').

| Strategy | Behavior |
|----------|----------|
| `'exclude'` | Remove rows with any missing values |
| `'as_group'` | Treat missing sensitive attributes as a separate group called "__missing__" |
| `'error'` | Raise an error if any missing values are found |

```python
# Exclude rows with missing values
dp = demographic_parity_difference(y_true, y_pred, gender, missing_strategy='exclude')

# Treat missing as its own group
dp = demographic_parity_difference(y_true, y_pred, gender, missing_strategy='as_group')
```

---

## Handling Special Cases

### Intersectional Analysis

Analyze fairness across combinations of attributes:

```python
import pandas as pd
from vfairness import demographic_parity_difference

# Create intersectional groups
sensitive = pd.DataFrame({
    'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
    'age_group': ['young', 'old', 'young', 'old', 'young', 'old', 'young', 'old']
})

# This creates 4 groups: M_young, M_old, F_young, F_old
dp = demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=2)
```

---

### Working with Pandas

All functions accept pandas Series and DataFrames:

```python
import pandas as pd

df = pd.DataFrame({
    'actual': [1, 0, 1, 0],
    'predicted': [1, 1, 1, 0],
    'gender': ['M', 'M', 'F', 'F']
})

dp = demographic_parity_difference(
    df['actual'],
    df['predicted'],
    df['gender'],
    min_group_size=2
)
```

---

### Custom Thresholds

Override default pass/fail thresholds in reports:

```python
report = classification_fairness_report(
    y_true, y_pred, gender,
    thresholds={
        'demographic_parity_difference': 0.15,  # More lenient
        'equal_opportunity_difference': 0.03,   # Stricter
    }
)
```

---

## Examples

### Example 1: Loan Approval Fairness

```python
import numpy as np
from vfairness import classification_fairness_report, print_report

# Simulated loan data
np.random.seed(42)
n = 1000

# Applicant demographics
gender = np.array(['M'] * 500 + ['F'] * 500)

# True creditworthiness (should they get the loan?)
y_true = np.random.binomial(1, 0.6, n)

# Model predictions (biased: approves men more often)
bias = np.where(gender == 'M', 0.1, -0.1)
y_pred = (np.random.rand(n) + bias > 0.5).astype(int)

# Generate report
report = classification_fairness_report(y_true, y_pred, gender)
print_report(report)
```

---

### Example 2: Salary Prediction Fairness

```python
import numpy as np
from vfairness import regression_fairness_report, print_report

# Simulated salary data
np.random.seed(42)
n = 500

department = np.array(['Engineering'] * 250 + ['Marketing'] * 250)
y_true = np.concatenate([
    np.random.normal(90000, 15000, 250),  # Engineering salaries
    np.random.normal(70000, 12000, 250)   # Marketing salaries
])

# Model underpredicts for Marketing
y_pred = y_true.copy()
y_pred[250:] -= 5000  # Systematic bias

report = regression_fairness_report(y_true, y_pred, department)
print_report(report)
```

---

### Example 3: Intersectional Analysis

```python
import numpy as np
import pandas as pd
from vfairness import demographic_parity_difference

np.random.seed(42)
n = 400

# Create intersectional groups
sensitive = pd.DataFrame({
    'gender': np.repeat(['M', 'F'], n // 2),
    'race': np.tile(['White', 'Black'], n // 2)
})

y_true = np.random.binomial(1, 0.5, n)
y_pred = np.random.binomial(1, 0.5, n)

# Analyze across 4 groups: M_White, M_Black, F_White, F_Black
dp = demographic_parity_difference(y_true, y_pred, sensitive, min_group_size=50)
print(f"Intersectional DP Difference: {dp:.3f}")
```

---

## In-Processing Module (Training-Time Interventions)

The In-Processing Module (`vfairness.in_processing`) provides comprehensive tools for fairness-aware model training. These techniques intervene directly during the machine learning training process to produce models that are fair by design.

### In-Processing Overview

Standard model training is fairness-blind. Optimization algorithms like gradient descent only seek to minimize prediction error, which can amplify spurious correlations between sensitive attributes and outcomes. Training-time interventions address this by modifying the learning process to incorporate fairness objectives.

The `in_processing` module offers five main approaches:

1. **Fairness-Aware Loss Functions** - PyTorch losses that penalize fairness violations
2. **Constraint-Based Training** - Algorithms that enforce hard fairness constraints
3. **Fairness Regularizers** - Penalty terms that can be added to any loss
4. **Group-Specific Calibrators** - Trainable calibration for group-wise probability calibration
5. **Scikit-Learn Wrappers** - Easy-to-use wrappers for standard ML workflows

#### Quick Start

```python
# Scikit-learn compatible approach (easiest)
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing import FairClassifier

clf = FairClassifier(
    base_estimator=RandomForestClassifier(),
    fairness_constraint='demographic_parity',
    tolerance=0.05
)
# X_train must be a 2-D numpy array. A multi-column pandas DataFrame is
# currently rejected with a TypeError; convert with X_train.to_numpy() first.
clf.fit(X_train, y_train, sensitive_attr=gender)
y_pred = clf.predict(X_test)
print(f"Constraint satisfied: {clf.fairness_result_.constraint_satisfied}")
```

---

### FairClassifier and FairRegressor
<!-- cap-status: FairClassifier -->
**Beta status: Checked.** 8 code units behind this name: 8 checked
<!-- /cap-status -->


Scikit-learn compatible wrappers that add fairness constraints to any base estimator.

#### FairClassifier

```python
<!-- cap-status: FairClassifier -->
**Beta status: Checked.** 8 code units behind this name: 8 checked
<!-- /cap-status -->

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing import FairClassifier

# Create fair classifier
clf = FairClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    fairness_constraint='equalized_odds',  # 'demographic_parity', 'equal_opportunity', etc.
    tolerance=0.05,                         # Maximum allowed constraint violation
    method='reductions',                    # 'reductions', 'threshold', 'grid_search'
    max_iterations=50,
    verbose=True
)

# Fit (requires sensitive_attr; X_train must be a 2-D numpy array,
# use X_train.to_numpy() when starting from a DataFrame)
clf.fit(X_train, y_train, sensitive_attr=gender)

# Predict
y_pred = clf.predict(X_test)

# Check fairness results
print(f"Accuracy: {clf.fairness_result_.accuracy:.4f}")
print(f"Violation: {clf.fairness_result_.fairness_violation:.4f}")
print(f"Satisfied: {clf.fairness_result_.constraint_satisfied}")

# Score with fairness metrics
accuracy = clf.score(X_test, y_test, metric='accuracy')
fairness = clf.score(X_test, y_test, sensitive_attr=gender_test, metric='fairness')
combined = clf.score(X_test, y_test, sensitive_attr=gender_test, metric='combined')
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_estimator` | Any | Required | Any scikit-learn compatible classifier |
| `fairness_constraint` | str | `'demographic_parity'` | Constraint type (see below) |
| `tolerance` | float | `0.05` | Maximum allowed violation |
| `method` | str | `'reductions'` | Training method |
| `max_iterations` | int | `50` | Max iterations for optimization |
| `verbose` | bool | `False` | Print progress |

**Available Fairness Constraints:**

| Constraint | Description |
|------------|-------------|
| `'demographic_parity'` | Equal positive prediction rates |
| `'equalized_odds'` | Equal TPR and FPR |
| `'equal_opportunity'` | Equal TPR only |
| `'false_positive_parity'` | Equal FPR only |
| `'bounded_group_loss'` | Bounded worst-group loss |

**Training Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| `'reductions'` | Exponentiated gradient algorithm | General-purpose, strong guarantees |
| `'threshold'` | Post-processing threshold optimization | Pre-trained models |
| `'grid_search'` | Grid search over Lagrange multipliers | Quick exploration |

#### FairRegressor

```python
<!-- cap-status: FairRegressor -->
**Beta status: Checked.** 7 code units behind this name: 7 checked
<!-- /cap-status -->

from sklearn.linear_model import Ridge
from vfairness.in_processing import FairRegressor

reg = FairRegressor(
    base_estimator=Ridge(),
    fairness_constraint='mean_parity',  # the only constraint implemented
    tolerance=0.1,
    method='offset'                     # the only method implemented (the default)
)

reg.fit(X_train, y_train, sensitive_attr=group)

# The mean-parity correction is a PER-GROUP offset, so applying it needs the
# sensitive attribute. This is the call that returns adjusted predictions:
y_pred = reg.predict_with_sensitive_attr(X_test, group_test)
```

**`reg.predict(X_test)` does NOT apply the mitigation.** It has no group
information, so it returns the UNADJUSTED base predictions and warns. Measured
on a 1000-row synthetic set where the group is a feature and `tolerance=0.1`:
`predict()` left a group mean gap of **4.97**, while
`predict_with_sensitive_attr()` produced **0.08**. Use `predict()` only when you
deliberately want the unmitigated baseline.

`fairness_constraint='error_parity'` and `'bounded_loss'` are declared in the
signature but not implemented; both raise `NotImplementedError` at `fit()` time
rather than silently fitting an unconstrained model. `method='reweighting'` was
removed and now raises `ValueError`: uniform per-group sample reweighting leaves
the weighted-least-squares group means equal to the data group means, so it
could never move the quantity it claimed to constrain.

#### Factory Functions

```python
from vfairness.in_processing import make_fair_classifier, make_fair_regressor

# Quick creation
clf = make_fair_classifier(
    RandomForestClassifier(),
    fairness_constraint='demographic_parity',
    tolerance=0.05
)

reg = make_fair_regressor(
    Ridge(),
    fairness_constraint='mean_parity',
    tolerance=0.1
)
```

---

### Fairness-Aware Loss Functions

PyTorch loss functions that incorporate fairness penalties directly into the training objective.

Requires `pip install torch` (optional dependency): every loss in this section,
and `create_fairness_loss`, raises an ImportError without it. The regularizers
under `create_regularizer` (e.g. `'statistical_parity'`, `'hsic'`) work without
torch.

#### Concept

Fairness-aware loss functions combine task loss with fairness penalties:

```
L_total = L_task + λ * L_fairness
```

Where:
- `L_task` is the primary prediction loss (e.g., BCE, MSE)
- `L_fairness` measures fairness violations
- `λ` controls the accuracy-fairness trade-off

#### Available Loss Functions

##### Group Fairness Losses

| Loss Function | Criterion | Use Case |
|---------------|-----------|----------|
| `DemographicParityLoss` | Equal positive prediction rates | When outcome should be independent of group <!--cs-->**[Checked]**<!--/cs--> |
| `EqualizedOddsLoss` | Equal TPR and FPR | When error rates should be equal <!--cs-->**[Checked]**<!--/cs--> |
| `EqualOpportunityLoss` | Equal TPR (true positive rates) | When qualifying individuals should have equal chances <!--cs-->**[Checked]**<!--/cs--> |
| `FalsePositiveRateParityLoss` | Equal FPR | When false accusations should be equally rare <!--cs-->**[Checked]**<!--/cs--> |
| `BoundedGroupLoss` | Bounded worst-group loss | For minimax fairness <!--cs-->**[Checked]**<!--/cs--> |
##### Adversarial Losses

| Loss Function | Mechanism | Use Case |
|---------------|-----------|----------|
| `AdversarialDebiasingLoss` | Adversary predicts sensitive attr | Remove information leakage <!--cs-->**[Checked]**<!--/cs--> |
| `ProjectedAdversarialLoss` | Gradient projection | More stable adversarial training <!--cs-->**[Checked]**<!--/cs--> |
| `FairRepresentationLoss` | Fair representation learning | Learning intermediate representations <!--cs-->**[Checked]**<!--/cs--> |
##### Counterfactual Losses

| Loss Function | Criterion | Use Case |
|---------------|-----------|----------|
| `CounterfactualFairnessLoss` | Same prediction under intervention | Causal fairness <!--cs-->**[Checked]**<!--/cs--> |
| `IndividualFairnessLoss` | Similar individuals, similar predictions | Individual-level fairness <!--cs-->**[Checked]**<!--/cs--> |
| `CausalFairnessLoss` | Total effect: the gap in mean prediction between groups | Only `causal_criterion='total_effect'` exists; `'direct_effect'` and `'path_specific'` raise `NotImplementedError`, as does a non-empty `mediator_indices` <!--cs-->**[Checked]**<!--/cs--> |
#### Usage Examples

##### Basic Usage with Demographic Parity

```python
import torch
from vfairness.in_processing import DemographicParityLoss

# Create loss function
loss_fn = DemographicParityLoss(
    lambda_fairness=0.1,  # Trade-off parameter
    warmup_epochs=5       # Epochs before applying fairness penalty
)

# Training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    loss_fn.set_epoch(epoch)

    for x, y, sensitive_attr in dataloader:
        optimizer.zero_grad()

        # Forward pass
        y_pred = torch.sigmoid(model(x))

        # Compute fairness-aware loss
        loss = loss_fn(y_pred, y, sensitive_attr)

        # Backward pass
        loss.backward()
        optimizer.step()

    # End of epoch tracking
    metrics = loss_fn.end_epoch()
    print(f"Epoch {epoch}: Loss={metrics.avg_total_loss:.4f}")
```

##### Equalized Odds with Custom Weights

```python
from vfairness.in_processing import EqualizedOddsLoss

loss_fn = EqualizedOddsLoss(
    lambda_fairness=0.15,
    tpr_weight=1.0,    # Weight for TPR component
    fpr_weight=0.5,    # Weight for FPR component
)
```

##### Adversarial Debiasing

```python
from vfairness.in_processing import AdversarialDebiasingLoss

loss_fn = AdversarialDebiasingLoss(
    lambda_fairness=1.0,
    adversary_hidden_dims=[64, 32],
    n_groups=2,
    use_gradient_reversal=True
)

# Move to GPU if available
loss_fn = loss_fn.to(device)

# Training loop
for x, y, sensitive_attr in dataloader:
    y_pred = torch.sigmoid(model(x))

    # Main model loss (includes adversarial penalty)
    loss = loss_fn(y_pred, y, sensitive_attr)
    loss.backward()
    optimizer.step()

    # Monitor adversary accuracy (lower is more fair)
    adv_acc = loss_fn.get_adversary_accuracy(y_pred.detach(), sensitive_attr)
```

##### Factory Function

```python
from vfairness.in_processing import create_fairness_loss

# Create loss by name
loss_fn = create_fairness_loss(
    'demographic_parity',
    lambda_fairness=0.1,
    warmup_epochs=5
)
```

---

### Constraint-Based Training

The constraint-based approach from Agarwal et al. (2018) reduces fair classification to a sequence of cost-sensitive classification problems.

#### Available Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `ExponentiatedGradient` | Lagrangian saddle-point optimization | General-purpose constrained training <!--cs-->**[Checked]**<!--/cs--> |
| `GridSearch` | Grid search over Lagrange multipliers | Quick exploration of trade-offs <!--cs-->**[Checked]**<!--/cs--> |
| `ThresholdOptimizer` | Post-processing threshold optimization | When model is already trained <!--cs-->**[Checked]**<!--/cs--> |
#### Constraints

| Constraint | Mathematical Definition |
|------------|-------------------------|
| `DemographicParityConstraint` | \|P(ŷ=1\|G=a) - P(ŷ=1\|G=b)\| ≤ ε <!--cs-->**[Checked]**<!--/cs--> |
| `EqualizedOddsConstraint` | \|TPR_a - TPR_b\| ≤ ε AND \|FPR_a - FPR_b\| ≤ ε <!--cs-->**[Checked]**<!--/cs--> |
| `EqualOpportunityConstraint` | \|TPR_a - TPR_b\| ≤ ε <!--cs-->**[Checked]**<!--/cs--> |
| `FalsePositiveRateParityConstraint` | \|FPR_a - FPR_b\| ≤ ε <!--cs-->**[Checked]**<!--/cs--> |
| `BoundedGroupLossConstraint` | L_g ≤ (1+ε) * L_overall for all g <!--cs-->**[Checked]**<!--/cs--> |
#### Exponentiated Gradient Algorithm

```python
from sklearn.linear_model import LogisticRegression
from vfairness.in_processing import (
    ExponentiatedGradient,
    DemographicParityConstraint,
)

# Create constraint
constraint = DemographicParityConstraint(tolerance=0.05)

# Create algorithm
eg = ExponentiatedGradient(
    base_estimator=LogisticRegression(),
    constraint=constraint,
    max_iterations=50,
    verbose=True
)

# Fit
result = eg.fit(X_train, y_train, sensitive_attr=gender)

# Predict
y_pred = eg.predict(X_test)

# Check results
print(f"Accuracy: {result.accuracy:.4f}")
print(f"Violation: {result.final_violation:.4f}")
print(f"Converged: {result.optimization_result.converged}")
```

#### Threshold Optimization

```python
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing import (
    ThresholdOptimizer,
    EqualOpportunityConstraint,
)

# First train a standard classifier
base_clf = RandomForestClassifier()
base_clf.fit(X_train, y_train)

# Get probabilities
y_prob_val = base_clf.predict_proba(X_val)[:, 1]

# Optimize thresholds for fairness
optimizer = ThresholdOptimizer(
    constraint=EqualOpportunityConstraint(tolerance=0.05),
    grid_size=100
)
optimizer.fit(y_prob_val, y_val, sensitive_attr=gender_val)

# Apply to test data
y_prob_test = base_clf.predict_proba(X_test)[:, 1]
y_pred_fair = optimizer.predict(y_prob_test, gender_test)

# Get the learned thresholds
print("Thresholds per group:", optimizer.get_thresholds())
```

#### Grid Search

```python
from vfairness.in_processing import GridSearch, EqualizedOddsConstraint

gs = GridSearch(
    base_estimator=LogisticRegression(),
    constraint=EqualizedOddsConstraint(tolerance=0.05),
    n_lambda_values=20,
    verbose=True
)

result = gs.fit(X_train, y_train, sensitive_attr=gender)
y_pred = gs.predict(X_test)
```

#### Factory Function

```python
from vfairness.in_processing import create_constraint

# Create constraint by name
constraint = create_constraint('demographic_parity', tolerance=0.05)
constraint = create_constraint('equalized_odds', tolerance=0.1)
```

---

### Fairness Regularizers

Regularizers provide a modular way to add fairness penalties to any differentiable loss function.

#### Available Regularizers

| Regularizer | What It Measures | Use Case |
|-------------|-----------------|----------|
| `StatisticalParityRegularizer` | Difference in mean predictions | Demographic parity <!--cs-->**[Checked]**<!--/cs--> |
| `ConditionalIndependenceRegularizer` | Conditional dependence given y | Equalized odds <!--cs-->**[Checked]**<!--/cs--> |
| `GroupFairnessRegularizer` | Configurable group metrics | Flexible fairness criteria <!--cs-->**[Checked]**<!--/cs--> |
| `HilbertSchmidtRegularizer` | HSIC-based independence | Non-linear dependence <!--cs-->**[Checked]**<!--/cs--> |
| `CorrelationPenalty` | Pearson correlation | Simple linear dependence <!--cs-->**[Checked]**<!--/cs--> |
#### Usage Examples

##### Statistical Parity Regularizer

```python
import torch.nn.functional as F
from vfairness.in_processing import StatisticalParityRegularizer

regularizer = StatisticalParityRegularizer(strength=0.1)

# In training loop
for x, y, sensitive_attr in dataloader:
    y_pred = torch.sigmoid(model(x))

    # Task loss
    task_loss = F.binary_cross_entropy(y_pred, y)

    # Fairness penalty
    fairness_penalty = regularizer(y_pred, sensitive_attr)

    # Combined loss
    total_loss = task_loss + fairness_penalty
    total_loss.backward()
```

##### HSIC Regularizer for Non-Linear Independence

```python
from vfairness.in_processing import HilbertSchmidtRegularizer

regularizer = HilbertSchmidtRegularizer(
    strength=0.1,
    kernel='rbf',
    sigma=1.0
)

# HSIC captures non-linear statistical dependence
penalty = regularizer(y_pred, sensitive_attr)
```

##### Conditional Independence (Equalized Odds)

```python
from vfairness.in_processing import ConditionalIndependenceRegularizer

# Enforces ŷ ⊥ a | y
# conditional_on='label' is the only implemented value: forward() conditions on
# y_true and nothing else. 'prediction' (sufficiency) and 'both' raise
# NotImplementedError; they used to be accepted and produce the same penalty.
regularizer = ConditionalIndependenceRegularizer(strength=0.1)
penalty = regularizer(y_pred, sensitive_attr, y_true=y)
```

##### Factory Function

```python
from vfairness.in_processing import create_regularizer

regularizer = create_regularizer('statistical_parity', strength=0.1)
regularizer = create_regularizer('hsic', strength=0.1, kernel='rbf')
```

---

### Group-Specific Calibrators

Group-specific calibrators learn separate calibration parameters for each demographic group, ensuring probability predictions have the same meaning across groups.

Requires `pip install torch` (optional dependency): the calibrators in this
section, including `create_group_calibrator`, raise an ImportError without it.

#### Available Calibrators

| Calibrator | Parameters | Description |
|------------|------------|-------------|
| `TemperatureScalingCalibrator` | T per group | Divides logits by temperature <!--cs-->**[Checked]**<!--/cs--> |
| `PlattScalingCalibrator` | (a, b) per group | Linear transform in log-odds space <!--cs-->**[Checked]**<!--/cs--> |
| `BetaCalibrator` | (c, d, e) per group | More flexible beta calibration <!--cs-->**[Checked]**<!--/cs--> |
| `FocalCalibrator` | γ per group | Focal-loss inspired calibration <!--cs-->**[Checked]**<!--/cs--> |
| `TrainableGroupCalibrator` | Configurable | Unified interface for all methods <!--cs-->**[Checked]**<!--/cs--> |
#### Usage Example

```python
from vfairness.in_processing import (
    TrainableGroupCalibrator,
    CalibrationAwareTrainer,
)

# Create calibrator
calibrator = TrainableGroupCalibrator(
    n_groups=2,
    method='temperature',  # 'temperature', 'platt', 'beta', 'focal'
    learnable=True
)

# Integrate with model training
trainer = CalibrationAwareTrainer(model, calibrator)

# Training step
losses = trainer.train_step(
    x, y, group_ids,
    optimizer,
    task_loss_fn=F.cross_entropy,
    include_calibration=True
)

print(f"Task loss: {losses['task_loss']:.4f}")
print(f"Calibration loss: {losses['calibration_loss']:.4f}")
```

#### Temperature Scaling

```python
from vfairness.in_processing import TemperatureScalingCalibrator

calibrator = TemperatureScalingCalibrator(
    n_groups=2,
    init_temperature=1.0,
    learnable=True
)

# Apply to logits
calibrated_logits = calibrator(logits, group_ids)
calibrated_probs = torch.sigmoid(calibrated_logits)

# Get learned temperatures
print(calibrator.get_parameters())  # {'group_0': 1.2, 'group_1': 0.9}
```

#### Factory Function

```python
from vfairness.in_processing import create_group_calibrator

calibrator = create_group_calibrator(
    method='platt',
    n_groups=3,
    learnable=True
)
```

---

### FairnessTrainingAnalyzer

The `FairnessTrainingAnalyzer` provides end-to-end analysis of fairness-aware training options.
<!-- cap-status: FairnessTrainingAnalyzer -->
**Beta status: Checked.** 8 code units behind this name: 8 checked
<!-- /cap-status -->


#### Basic Usage

```python
from sklearn.linear_model import LogisticRegression
from vfairness.in_processing import FairnessTrainingAnalyzer

# Create analyzer
analyzer = FairnessTrainingAnalyzer(
    X=X_train, y=y_train, sensitive_attr=gender,
    fairness_constraint='demographic_parity',
    tolerance=0.05
)

# Full analysis
report = analyzer.full_analysis(
    base_estimator=LogisticRegression(),
    include_comparisons=True,
    include_tradeoffs=True
)

# Print summary
print(report.summary())

# Export to JSON (to_json returns a string; write it yourself)
from pathlib import Path
Path('training_analysis.json').write_text(report.to_json(indent=2))

# Render as SVG (to_svg writes the file when save_path is given)
report.to_svg('training_report.svg')
```

Note: `to_json(indent)` takes only a JSON indent width and returns the JSON
string; it never writes a file. Passing a filename as the positional argument
is silently consumed as the indent, so no file is created and no error is
raised.

#### Evaluate Baseline

```python
# Evaluate baseline model without fairness constraints
baseline = analyzer.evaluate_baseline(model=trained_model)
print(f"Baseline accuracy: {baseline.accuracy:.4f}")
print(f"Baseline violation: {baseline.fairness_violation:.4f}")
```

#### Compare Methods

```python
# Compare different training methods
comparisons = analyzer.compare_methods(
    base_estimator=LogisticRegression(),
    methods=['reductions', 'threshold', 'grid_search']
)

for comp in comparisons:
    status = "✓" if comp.constraint_satisfied else "✗"
    print(f"{comp.method_name}: Acc={comp.accuracy:.4f}, "
          f"Violation={comp.fairness_violation:.4f} [{status}]")
```

#### Analyze Trade-offs

```python
# Analyze accuracy-fairness trade-offs
tradeoffs = analyzer.analyze_tradeoffs(
    base_estimator=LogisticRegression(),
    lambda_values=[0, 0.1, 0.5, 1.0, 2.0]
)

print(f"Best fair result: {tradeoffs['best_fair']}")
print(f"Pareto frontier: {len(tradeoffs['pareto_frontier'])} points")
```

#### Generate Recommendations

```python
# Get training recommendation
recommendation = analyzer.generate_recommendation()
print(f"Recommended method: {recommendation.recommended_method}")
print(f"Priority: {recommendation.priority}")
print(f"Rationale: {recommendation.rationale}")
```

#### Report Contents

The `FairnessTrainingReport` includes:

| Section | Description |
|---------|-------------|
| `baseline_metrics` | Accuracy and fairness without constraints |
| `method_comparisons` | Side-by-side comparison of methods |
| `tradeoff_analysis` | Pareto frontier of accuracy vs. fairness |
| `recommendation` | Suggested method with rationale |
| `critical_issues` | Problems that need attention |
| `action_items` | Prioritized next steps |

#### Report Methods

```python
# Get summary text
print(report.summary())

# Convert to dictionary
data = report.to_dict()

# Export to JSON
json_str = report.to_json(indent=2)

# Render as SVG visualization
svg_str = report.to_svg(save_path='report.svg')
```

---

### Best Practices for Training-Time Interventions

#### Choosing a Method

| Scenario | Recommended Approach |
|----------|---------------------|
| PyTorch deep learning | Fairness-aware loss functions |
| Scikit-learn classifier | FairClassifier with reductions |
| Need hard constraint guarantee | ExponentiatedGradient |
| Quick exploration | GridSearch or ThresholdOptimizer |
| Non-linear feature dependence | HSIC regularizer |
| Want representation-level fairness | AdversarialDebiasingLoss |

#### Setting λ (Lambda)

The `lambda_fairness` parameter controls the accuracy-fairness trade-off:

| λ Value | Behavior |
|---------|----------|
| `λ = 0` | Pure accuracy optimization (unfair) |
| `λ = 0.01-0.1` | Light fairness penalty |
| `λ = 0.1-0.5` | Moderate penalty |
| `λ = 0.5-1.0` | Strong fairness emphasis |
| `λ > 1.0` | Fairness dominates (may hurt accuracy) |

**Recommendation**: Start with `λ = 0.1` and use `FairnessTrainingAnalyzer` to explore the trade-off curve.

#### Warmup Strategy

For loss functions, use `warmup_epochs` to stabilize early training:

```python
loss_fn = DemographicParityLoss(
    lambda_fairness=0.1,
    warmup_epochs=5  # No fairness penalty for first 5 epochs
)
```

#### Monitoring Training

Track both accuracy and fairness during training:

```python
# After each epoch
metrics = loss_fn.end_epoch()
print(f"Epoch {epoch}")
print(f"  Task Loss: {metrics.avg_task_loss:.4f}")
print(f"  Fairness Loss: {metrics.avg_fairness_loss:.4f}")
print(f"  Total Loss: {metrics.avg_total_loss:.4f}")
```

---

### In-Processing API Summary

#### Loss Functions (`vfairness.in_processing.loss_functions`)

| Class | Description |
|-------|-------------|
| `DemographicParityLoss` | Penalizes differences in positive prediction rates <!--cs-->**[Checked]**<!--/cs--> |
| `EqualizedOddsLoss` | Penalizes differences in TPR and FPR <!--cs-->**[Checked]**<!--/cs--> |
| `EqualOpportunityLoss` | Penalizes differences in TPR only <!--cs-->**[Checked]**<!--/cs--> |
| `FalsePositiveRateParityLoss` | Penalizes differences in FPR only <!--cs-->**[Checked]**<!--/cs--> |
| `BoundedGroupLoss` | Bounds worst-group loss (minimax fairness) <!--cs-->**[Checked]**<!--/cs--> |
| `AdversarialDebiasingLoss` | Adversarial training for fair representations <!--cs-->**[Checked]**<!--/cs--> |
| `ProjectedAdversarialLoss` | Gradient projection adversarial approach <!--cs-->**[Checked]**<!--/cs--> |
| `FairRepresentationLoss` | Fair representation learning loss <!--cs-->**[Checked]**<!--/cs--> |
| `CounterfactualFairnessLoss` | Counterfactual fairness penalties <!--cs-->**[Checked]**<!--/cs--> |
| `IndividualFairnessLoss` | Lipschitz-based individual fairness <!--cs-->**[Checked]**<!--/cs--> |
| `CausalFairnessLoss` | Total-effect causal fairness (`'total_effect'` only) <!--cs-->**[Checked]**<!--/cs--> |
| `create_fairness_loss()` | Factory function for creating losses <!--cs-->**[Checked]**<!--/cs--> |
#### Constraints (`vfairness.in_processing.constraints`)

| Class | Description |
|-------|-------------|
| `DemographicParityConstraint` | Constraint for demographic parity <!--cs-->**[Checked]**<!--/cs--> |
| `EqualizedOddsConstraint` | Constraint for equalized odds <!--cs-->**[Checked]**<!--/cs--> |
| `EqualOpportunityConstraint` | Constraint for equal opportunity <!--cs-->**[Checked]**<!--/cs--> |
| `FalsePositiveRateParityConstraint` | Constraint for FPR parity <!--cs-->**[Checked]**<!--/cs--> |
| `BoundedGroupLossConstraint` | Constraint for bounded group loss <!--cs-->**[Checked]**<!--/cs--> |
| `ExponentiatedGradient` | Main reductions algorithm <!--cs-->**[Checked]**<!--/cs--> |
| `GridSearch` | Grid search over Lagrange multipliers <!--cs-->**[Checked]**<!--/cs--> |
| `ThresholdOptimizer` | Post-processing threshold optimization <!--cs-->**[Checked]**<!--/cs--> |
| `create_constraint()` | Factory function for creating constraints <!--cs-->**[Checked]**<!--/cs--> |
#### Regularizers (`vfairness.in_processing.regularizers`)

| Class | Description |
|-------|-------------|
| `StatisticalParityRegularizer` | Penalizes mean prediction differences <!--cs-->**[Checked]**<!--/cs--> |
| `ConditionalIndependenceRegularizer` | Enforces ŷ ⊥ a \| y <!--cs-->**[Checked]**<!--/cs--> |
| `GroupFairnessRegularizer` | Flexible group fairness penalty <!--cs-->**[Checked]**<!--/cs--> |
| `HilbertSchmidtRegularizer` | HSIC-based independence <!--cs-->**[Checked]**<!--/cs--> |
| `CorrelationPenalty` | Simple Pearson correlation penalty <!--cs-->**[Checked]**<!--/cs--> |
| `create_regularizer()` | Factory function for creating regularizers <!--cs-->**[Checked]**<!--/cs--> |
#### Calibrators (`vfairness.in_processing.calibrators`)

| Class | Description |
|-------|-------------|
| `TemperatureScalingCalibrator` | Group-specific temperature scaling <!--cs-->**[Checked]**<!--/cs--> |
| `PlattScalingCalibrator` | Group-specific Platt scaling <!--cs-->**[Checked]**<!--/cs--> |
| `BetaCalibrator` | Group-specific beta calibration <!--cs-->**[Checked]**<!--/cs--> |
| `FocalCalibrator` | Group-specific focal calibration <!--cs-->**[Checked]**<!--/cs--> |
| `TrainableGroupCalibrator` | Unified trainable calibrator <!--cs-->**[Checked]**<!--/cs--> |
| `CalibrationAwareTrainer` | Helper for joint model-calibration training <!--cs-->**[Checked]**<!--/cs--> |
| `create_group_calibrator()` | Factory function for creating calibrators <!--cs-->**[Checked]**<!--/cs--> |
#### Wrappers (`vfairness.in_processing.wrappers`)

| Class | Description |
|-------|-------------|
| `FairClassifier` | Sklearn-compatible fairness-aware classifier <!--cs-->**[Checked]**<!--/cs--> |
| `FairRegressor` | Sklearn-compatible fairness-aware regressor <!--cs-->**[Checked]**<!--/cs--> |
| `make_fair_classifier()` | Factory function for classifiers <!--cs-->**[Checked]**<!--/cs--> |
| `make_fair_regressor()` | Factory function for regressors <!--cs-->**[Checked]**<!--/cs--> |
#### Analyzer (`vfairness.in_processing`)

| Class | Description |
|-------|-------------|
| `FairnessTrainingAnalyzer` | Comprehensive training analysis <!--cs-->**[Checked]**<!--/cs--> |
| `FairnessTrainingReport` | Analysis report with `summary()`, `to_dict()`, `to_svg()` <!--cs-->**[Checked]**<!--/cs--> |
| `MethodComparison` | Comparison results for training methods <!--cs-->**[Checked]**<!--/cs--> |
| `TrainingRecommendation` | Training method recommendation <!--cs-->**[Checked]**<!--/cs--> |
---

### References

1. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.

2. Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A Reductions Approach to Fair Classification. ICML.

3. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating Unwanted Biases with Adversarial Learning. AIES.

4. Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual Fairness. NeurIPS.

5. Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness Constraints: Mechanisms for Fair Classification. AISTATS.

6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.

---

## Post-Processing Module

The Post-Processing Module (`vfairness.post_processing`) provides tools for achieving fairness after model training, including calibration, threshold optimization, and prediction reweighting.

### Threshold Optimization

Group-specific threshold optimization finds optimal decision thresholds for each demographic group to satisfy fairness constraints.

```python
from vfairness.post_processing import (
    ThresholdOptimizer,
    GroupThresholdOptimizer,
    MultiObjectiveThresholdOptimizer,
    ThresholdAnalyzer,
)

# Simple: Find a single threshold satisfying demographic parity
optimizer = ThresholdOptimizer(
    constraint='demographic_parity',
    tolerance=0.05,
    objective='accuracy'
)
optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
print(f"Optimal threshold: {optimizer.result_.global_threshold:.3f}")
y_pred = optimizer.predict(y_prob, gender)

# Advanced: Group-specific thresholds for equalized odds
group_optimizer = GroupThresholdOptimizer(
    constraint='equalized_odds',
    tolerance=0.05
)
group_optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
print(f"Group thresholds: {group_optimizer.result_.group_thresholds}")
# {'male': 0.42, 'female': 0.58}
y_pred_fair = group_optimizer.predict(y_prob, gender)

# Multi-objective: Find Pareto frontier of fairness-accuracy trade-offs
pareto_optimizer = MultiObjectiveThresholdOptimizer(
    constraint='equalized_odds',
    objectives=['accuracy', 'f1_score']
)
pareto_optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
print(f"Pareto frontier has {len(pareto_optimizer.pareto_frontier_)} points")
# Select a specific point from the frontier
pareto_optimizer.select_point(index=0)
```

#### Threshold Analysis

Analyze the impact of different threshold values on fairness and performance:

```python
from vfairness.post_processing import ThresholdAnalyzer

analyzer = ThresholdAnalyzer(y_true, y_prob, gender)

# Full analysis
report = analyzer.full_analysis(n_thresholds=20)
print(report.summary())

# Find optimal threshold for a specific constraint
optimal = analyzer.find_optimal_threshold(
    constraint='demographic_parity',
    objective='f1_score',
    tolerance=0.05
)
print(f"Optimal threshold: {optimal['optimal_threshold']:.3f}")
print(f"Feasible: {optimal['is_feasible']}")

# Find feasible threshold region
region = analyzer.find_feasible_region(constraint='equalized_odds')
print(f"Feasible region: [{region[0]:.3f}, {region[1]:.3f}]")
```

#### Supported Constraints

| Constraint | Description |
|------------|-------------|
| `demographic_parity` | Equal positive prediction rates across groups |
| `equalized_odds` | Equal TPR and FPR across groups |
| `equal_opportunity` | Equal TPR across groups |
| `false_positive_parity` | Equal FPR across groups |
| `predictive_parity` | Equal precision across groups |

### Reweighting Methods

Prediction reweighting adjusts model probabilities post-hoc to achieve fairness without changing the decision threshold.

```python
from vfairness.post_processing import (
    PredictionReweighter,
    RejectionOptionClassifier,
    CalibratedEqualizer,
    DistributionMatcher,
    ReweightingAnalyzer,
)

# Multiplicative reweighting for demographic parity
reweighter = PredictionReweighter(
    constraint='demographic_parity',
    method='multiplicative'  # or 'additive'
)
reweighter.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
y_prob_fair = reweighter.transform(y_prob, gender)
print(f"Adjustments: {reweighter.adjustments_}")

# Rejection Option Classification (ROC)
# Modifies predictions only near the decision boundary
roc = RejectionOptionClassifier(
    theta=0.1,  # Width of critical region
    threshold=0.5
)
roc.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
y_pred_fair = roc.predict(y_prob, gender)
print(f"Unprivileged group: {roc.detected_unprivileged_}")

# Calibrated Equalization
# Matches probability distributions using quantile mapping
equalizer = CalibratedEqualizer(n_quantiles=100)
equalizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
y_prob_equalized = equalizer.transform(y_prob, gender)

# Distribution Matching
# Transforms all groups to match a reference group's distribution
matcher = DistributionMatcher(reference_group='male')
matcher.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=gender)
y_prob_matched = matcher.transform(y_prob, gender)
```

#### Reweighting Analysis

Compare different reweighting methods:

```python
from vfairness.post_processing import ReweightingAnalyzer

analyzer = ReweightingAnalyzer(y_true, y_prob, gender)

# Compare multiple methods
report = analyzer.full_analysis()
print(report.summary())
print(f"Best method: {report.best_method}")

# Analyze a specific method
result = analyzer.analyze_method('rejection_option', theta=0.15)
dp_before = result.original_fairness['demographic_parity_diff']
dp_after = result.adjusted_fairness['demographic_parity_diff']
print(f"Demographic parity difference: {dp_before:.3f} -> {dp_after:.3f}")
print(f"Trade-off score: {result.trade_off_score:.3f}")
print(f"Accuracy change: {result.adjusted_performance['accuracy'] - result.original_performance['accuracy']:.3f}")
```

The `ReweightingImpactResult` fields are `method`, `original_fairness`,
`adjusted_fairness`, `original_performance`, `adjusted_performance`,
`calibration_metrics`, and `trade_off_score`.

#### Available Reweighting Methods

| Method | Class | Description |
|--------|-------|-------------|
| `multiplicative` | `PredictionReweighter` | Multiply probabilities by group-specific factors |
| `additive` | `PredictionReweighter` | Add group-specific offsets to probabilities |
| `rejection_option` | `RejectionOptionClassifier` | Modify predictions near decision boundary |
| `calibrated` | `CalibratedEqualizer` | Quantile-based distribution equalization |
| `distribution_matching` | `DistributionMatcher` | Match distributions to reference group |

### Factory Functions

```python
from vfairness.post_processing import create_threshold_optimizer, create_reweighter

# Create threshold optimizer
optimizer = create_threshold_optimizer(
    optimizer_type='group',  # 'single', 'group', 'multi_objective'
    constraint='equalized_odds',
    tolerance=0.05
)

# Create reweighter
reweighter = create_reweighter(
    method='rejection_option',  # 'multiplicative', 'additive', 'rejection_option', 'calibrated', 'distribution_matching'
    theta=0.1
)
```

---

## Rendering Module

The Rendering Module (`vfairness.rendering`) provides SVG report generation using Jinja2 templates for polished, professional visualizations.

### Overview

```python
from vfairness.rendering import (
    # Core engine
    render_svg,
    get_template_path,
    list_templates,
    # Training adapters
    training_report_to_svg,
    training_analysis_report_to_svg,
    method_comparison_to_svg,
    tradeoff_analysis_to_svg,
    # Post-processing adapters
    threshold_optimization_to_svg,
    reweighting_comparison_to_svg,
    fairness_detailed_report_to_svg,
)
```

### Core Engine

```python
from vfairness.rendering import render_svg, list_templates

# Every template name, alphabetical. Names beginning with an underscore are
# internal partials that other templates include, not charts you render.
templates = list_templates()
charts = [t for t in templates if not t.startswith("_")]
print(len(templates), len(charts))
print(templates[:4])
```

Do not hardcode the template count: it changes whenever a chart is added. Derive
it as above. As of 2026-08-28 the set is 46 names, of which 44 are chart
templates and two are internal partials (`_shared_defs` and `_could_not_check`).

`render_svg(name, data)` renders one template with a context dict and returns
the SVG string. Each template requires its complete variable set (the
`training_report` template alone expects `baseline_accuracy`,
`baseline_violation`, `n_samples`, `n_groups`, `issues`, `actions`, and more);
a partial context raises a jinja2 `UndefinedError`. For real report objects,
prefer the adapter functions below: they build the full context, call
`render_svg` for you, and accept a `save_path` to write the file.

### Training Report Adapters

```python
from vfairness.rendering import training_analysis_report_to_svg
from vfairness.in_processing import FairnessTrainingAnalyzer

# Run training analysis
analyzer = FairnessTrainingAnalyzer(X, y, sensitive_attr)
report = analyzer.full_analysis()

# Generate SVG dashboard (680 units wide; height scales with content)
svg = training_analysis_report_to_svg(report, save_path='training_report.svg')

# Includes:
# - Header with title, timestamp, task type
# - Summary cards (data info, baseline, recommendation)
# - Group fairness analysis bars
# - Method comparison chart
# - Trade-off scatter plot with Pareto frontier
# - Critical issues list
# - Action items checklist
```

### Threshold Optimization SVG

```python
from vfairness.rendering import threshold_optimization_to_svg
from vfairness.post_processing import GroupThresholdOptimizer

# Run threshold optimization
optimizer = GroupThresholdOptimizer(constraint='demographic_parity')
optimizer.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sensitive_attr)

# Build result dictionary
result_data = {
    'title': 'Threshold Optimization Analysis',
    'constraint_type': 'demographic_parity',
    'tolerance': 0.05,
    'original_threshold': 0.5,
    'group_thresholds': optimizer.result_.group_thresholds,
    'original_disparity': 0.15,
    'optimized_disparity': 0.03,
    'original_accuracy': 0.85,
    'optimized_accuracy': 0.83,
    'is_feasible': True,
}

# Generate SVG
svg = threshold_optimization_to_svg(result_data, save_path='threshold_opt.svg')

# Dashboard shows:
# - Group-specific thresholds with bar visualization
# - Positive rate distribution before/after
# - Fairness improvement metrics
# - Accuracy trade-off analysis
```

### Reweighting Comparison SVG

```python
from vfairness.rendering import reweighting_comparison_to_svg
from vfairness.post_processing import ReweightingAnalyzer

# Run reweighting analysis
analyzer = ReweightingAnalyzer(y_true, y_prob, sensitive_attr)
report = analyzer.full_analysis()

# Generate comparison SVG
svg = reweighting_comparison_to_svg(report, save_path='reweighting.svg')

# Dashboard shows:
# - Method comparison bars (fairness improvement, accuracy change)
# - Trade-off score ranking
# - Best method recommendation
# - Calibration impact metrics
```

### Fairness Detailed Report SVG

```python
from vfairness.rendering import fairness_detailed_report_to_svg

# Build comprehensive fairness report
report_data = {
    'title': 'Fairness Analysis Report',
    'task_type': 'binary_classification',
    'n_samples': 10000,
    'assessment': {'fairness_score': 0.72},
    'metrics': {
        'demographic_parity': {'value': 0.08, 'threshold': 0.1, 'interpretation': '...'},
        'equalized_odds': {'value': 0.12, 'threshold': 0.1, 'interpretation': '...'},
    },
    'group_statistics': {
        'Group A': {'size': 5000, 'positive_rate': 0.45, 'tpr': 0.80, 'fpr': 0.15},
        'Group B': {'size': 5000, 'positive_rate': 0.37, 'tpr': 0.75, 'fpr': 0.12},
    },
    'key_findings': ['Finding 1', 'Finding 2'],
    'recommendations': ['Recommendation 1', 'Recommendation 2'],
}

# Generate executive-level SVG report
svg = fairness_detailed_report_to_svg(report_data, save_path='fairness.svg')

# Dashboard shows:
# - Fairness score gauge
# - Metrics table with pass/fail status
# - Group statistics comparison
# - Pairwise disparity analysis
# - Key findings and recommendations
```

### Templates with Dedicated Adapter Functions

`list_templates()` returns the full template set, chart templates plus the
internal underscore-prefixed partials. Call it rather than relying on a count
written here. The templates below are the ones with a dedicated adapter
function:

| Template | Adapter Function | Description |
|----------|------------------|-------------|
| `training_report` | `training_report_to_svg()` | Compact training dashboard |
| `training_analysis_report` | `training_analysis_report_to_svg()` | Full-page training analysis |
| `method_comparison` | `method_comparison_to_svg()` | Training method comparison |
| `tradeoff_analysis` | `tradeoff_analysis_to_svg()` | Accuracy-fairness trade-off |
| `threshold_optimization_report` | `threshold_optimization_to_svg()` | Threshold optimization dashboard |
| `reweighting_comparison_report` | `reweighting_comparison_to_svg()` | Reweighting method comparison |
| `fairness_detailed_report` | `fairness_detailed_report_to_svg()` | Executive fairness report |

### Could-Not-Check on the Canvas

An SVG leaves the process and is read by someone who cannot see the run behind
it, so a chart that fabricates a verdict is worse than no chart. The rendering
module therefore carries the same three states as the metrics.

**No adapter renders a verdict from an input that measured nothing.** When there
is nothing to plot, an adapter does one of two things:

- draws the shared **NOT CHECKED** panel *instead of* its verdict stack, never
  beside it, so a run that measured nothing does not also print a score, a rate,
  a band, a 0-of-0 count or a status colour anywhere on the canvas; or
- refuses outright, raising `TypeError` or `AttributeError` rather than
  rendering.

The panel is deliberately slate, neither the green that reads as a pass nor the
red that reads as a finding, and it closes with the line:

> This chart is not a pass and not a failure. No value on it was measured, so it
> certifies nothing.

It lives in one shared partial (`_could_not_check`) so that every template draws
the same third state and no later template can ship a quieter version of the
notice.

```python
from vfairness.rendering import report_card_to_svg

svg = report_card_to_svg(None)     # no gate decision was supplied
assert "NOT CHECKED" in svg
assert "APPROVED" not in svg       # nothing was measured, so nothing is approved
```

Checked on 2026-08-28 by calling all 44 public `*_to_svg` adapters with the
emptiest input each accepts: 39 rendered the NOT CHECKED panel, 5 raised, and
none produced a verdict. Reproduce it with:

```python
import inspect
import vfairness.rendering as R

for name in sorted(n for n in dir(R) if n.endswith("_to_svg")):
    fn = getattr(R, name)
    print(name, list(inspect.signature(fn).parameters)[:1])
```

then call each one with an empty argument of its documented type and grep the
returned SVG for `NOT CHECKED`.

### Design Principles

1. **Requires Jinja2** - SVG rendering is built on Jinja2 templates, installed via the `[rendering]` extra (`pip install vfairness[rendering]`). `render_svg()` raises `ImportError` ("Jinja2 is required for SVG report rendering") when Jinja2 is absent; there is no `string.Template` fallback
2. **Self-contained SVG output** - All templates produce standalone SVG files with embedded styles (no runtime, browser, or JS dependency)
3. **Consistent styling** - Professional card-based layout with consistent colors and typography
4. **Responsive** - SVGs scale cleanly at any resolution

---

## Bias Detection Module

The Bias Detection Module (`vfairness.preprocessing.bias_detection`) provides comprehensive tools for detecting potential sources of bias in your data. `BiasDetector` is also re-exported from `vfairness` and `vfairness.preprocessing` for convenience.

### BiasDetector Class
<!-- cap-status: BiasDetector -->
**Beta status: Checked.** 9 code units behind this name: 9 checked
<!-- /cap-status -->


The unified auditor for comprehensive bias detection:

```python
from vfairness import BiasDetector

# Create detector
detector = BiasDetector(
    df,
    protected_attributes=['gender', 'race', 'age'],
    outcome_column='approved',
    benchmarks={'gender': {'Male': 0.49, 'Female': 0.51}},
)

# Run full audit
report = detector.full_audit()

# Access results
print(f"Risk score: {report.overall_risk_score:.1%}")
print(f"Critical issues: {len(report.critical_issues)}")

for rec in report.recommendations:
    print(f"  → {rec}")
```

### Historical Pattern Detection

Detects features with historical discrimination patterns based on academic research, documented discriminatory practices, and regulatory frameworks. The library includes **43 curated patterns** organized across four jurisdictions.

```python
from vfairness import detect_historical_patterns, HISTORICAL_RISK_PATTERNS

# Detect patterns
results = detect_historical_patterns(
    df,
    protected_attributes=['race', 'gender'],
    min_confidence=0.3
)

for result in results:
    print(f"{result.feature}: {result.risk_level.value.upper()}")
    print(f"  Pattern: {result.pattern_type}")
    print(f"  Context: {result.historical_context}")
    print(f"  Affected groups: {result.affected_groups}")
    print(f"  Recommendations: {result.recommendations}")
```

#### Pattern Catalog (43 Patterns)

##### 🇺🇸 US / Global Patterns (12)

| Pattern Key | Pattern Type | Risk Level | Detects |
|-------------|-------------|------------|---------|
| `redlining_geographic` | Geographic Discrimination | 🔴 Critical | ZIP codes, neighborhoods linked to HOLC redlining maps |
| `neighborhood_names` | Geographic Discrimination | 🔴 High | Neighborhood names encoding racial composition |
| `educational_institution` | Educational Bias | 🟡 Medium | School/university names as socioeconomic proxies |
| `legacy_admissions` | Educational Bias | 🟡 Medium | Legacy admission preferences perpetuating privilege |
| `employment_gaps` | Employment Discrimination | 🟡 Medium | Employment gaps penalizing caregivers and minorities |
| `salary_history` | Pay Gap Perpetuation | 🔴 High | Previous salary perpetuating gender/race pay gaps |
| `criminal_records` | Criminal Justice Bias | 🔴 High | Arrest/conviction records reflecting biased policing |
| `healthcare_cost` | Healthcare Bias | 🔴 Critical | Healthcare costs as proxy (lower spend ≠ lower need) |
| `credit_history` | Financial Exclusion | 🔴 High | Credit scores reflecting historical banking exclusion |
| `banking_access` | Financial Exclusion | 🔴 High | Banking access patterns and unbanked populations |
| `names` | Name Discrimination | 🟡 Medium | First/last names correlating with race, ethnicity, gender |
| `digital_access` | Digital Divide | 🟡 Medium | Digital literacy and internet access disparities |

##### 🇪🇺 European Patterns (12)

| Pattern Key | Pattern Type | Risk Level | Detects |
|-------------|-------------|------------|---------|
| `migration_background` | Migration Discrimination | 🔴 Critical | Country of origin, nationality, migration status |
| `welfare_fraud_scoring` | Welfare Profiling | 🔴 Critical | Algorithmic fraud scoring (cf. Dutch Toeslagenaffaire) |
| `european_credit_scoring` | Credit Discrimination | 🔴 High | SCHUFA-style postcode scoring, address-based credit |
| `exam_grading_algorithms` | Educational Bias | 🔴 High | Automated grading biased by school type (cf. UK A-Levels 2020) |
| `employment_profiling_eu` | Employment Discrimination | 🔴 High | AMS-style employment scoring biased against women/disabled |
| `predictive_policing_eu` | Policing Bias | 🔴 High | Predictive policing (cf. UK Gangs Matrix — 78% Black) |
| `roma_traveller_discrimination` | Ethnic Discrimination | 🔴 Critical | Features targeting Roma/Traveller communities |
| `religious_identity_eu` | Religious Discrimination | 🔴 High | Dietary preferences, calendar patterns as religion proxies |
| `platform_gig_scoring` | Platform Discrimination | 🟡 Medium | Gig economy rating systems with demographic bias |
| `biometric_identification` | Biometric Bias | 🔴 Critical | Facial recognition, biometric systems with racial accuracy gaps |
| `social_housing_eu` | Housing Discrimination | 🔴 High | Social housing allocation algorithms |
| `language_discrimination` | Language Discrimination | 🟡 Medium | Language proficiency requirements as nationality proxy |

##### ⚖️ EU AI Act Patterns — Regulation 2024/1689 (11)

**🚫 Prohibited Practices (Art. 5) — CRITICAL risk:**

| Pattern Key | Article | Risk Level | What It Detects |
|-------------|---------|------------|-----------------|
| `euaia_social_scoring` | Art. 5(1)(c) | 🚨 Critical | Social behaviour aggregated into trustworthiness scores |
| `euaia_emotion_recognition` | Art. 5(1)(f) | 🚨 Critical | Emotion/sentiment analysis in workplace or education |
| `euaia_biometric_categorisation` | Art. 5(1)(g) | 🚨 Critical | Inferring race, religion, sexual orientation from biometrics |
| `euaia_manipulative_ai` | Art. 5(1)(a-b) | 🚨 Critical | Dark patterns, addictive design, behavioural manipulation |

> **Penalty**: Prohibited practices carry fines up to **€35M or 7% of global turnover**.

**⚠️ High-Risk Systems (Annex III) — HIGH risk:**

| Pattern Key | Annex III Area | Risk Level | What It Detects |
|-------------|---------------|------------|-----------------|
| `euaia_hr_recruitment` | Area 4 | 🔴 High | CV scoring, candidate ranking, performance monitoring |
| `euaia_creditworthiness` | Area 5(b) | 🔴 High | Automated credit decisions, default prediction |
| `euaia_education` | Area 3 | 🔴 High | Admission scoring, automated grading, student profiling |
| `euaia_essential_services` | Area 5(a) | 🔴 High | Benefit eligibility, insurance pricing, emergency triage |
| `euaia_law_enforcement` | Area 6 | 🔴 High | Recidivism prediction, criminal profiling, lie detection |
| `euaia_migration_border` | Area 7 | 🔴 High | Asylum assessment, visa decisions, border risk scoring |
| `euaia_justice_democracy` | Area 8 | 🔴 High | Judicial prediction, voter targeting, election influence |

> **Penalty**: High-risk non-compliance carries fines up to **€15M or 3% of global turnover**.

##### 🇨🇭 Swiss-Specific Patterns (8)

| Pattern Key | Pattern Type | Risk Level | Detects |
|-------------|-------------|------------|---------|
| `swiss_permit_system` | Permit Discrimination | 🔴 Critical | Ausweis B/C/F/N as proxy for nationality/ethnicity |
| `swiss_betreibung` | Debt Register Bias | 🔴 High | Betreibungsauszug entries (immigrants disproportionately affected) |
| `swiss_housing_discrimination` | Housing Discrimination | 🔴 High | Name-based housing discrimination (20-50% fewer callbacks) |
| `swiss_naturalisation` | Naturalisation Bias | 🔴 High | Ballot-box naturalisations (cf. BGE 129 I 217 Emmen case) |
| `swiss_health_insurance` | Insurance Discrimination | 🟡 Medium | KVG Prämienregionen encoding socioeconomic patterns |
| `swiss_labour_discrimination` | Employment Discrimination | 🔴 High | RAV/ORP profiling biased against women and older workers |
| `swiss_gemeinde_data` | Geographic Discrimination | 🟡 Medium | BFS Gemeinde code, Steuerfuss as socioeconomic proxy |
| `swiss_sozialhilfe` | Welfare Stigmatisation | 🔴 High | Sozialhilfe receipt affecting permit status (AIG Art. 63) |

#### How Pattern Detection Works

Each pattern in `HISTORICAL_RISK_PATTERNS` contains:
- **`keywords`**: Column name keywords that trigger matching (e.g., `['zip', 'postal', 'address']`)
- **`pattern_type`**: Category label (e.g., `"Geographic Discrimination"`, `"PROHIBITED — Emotion Recognition (Art. 5(1)(f))"`)
- **`historical_context`**: Academic and legal context for the pattern
- **`affected_groups`**: Demographics historically impacted
- **`risk_level`**: `HistoricalRiskLevel` enum (`critical`, `high`, `medium`, `low`)
- **`recommendations`**: Actionable mitigation steps with legal references

The `detect_historical_patterns()` function scans DataFrame column names against all 43 pattern keyword lists and returns `HistoricalPatternResult` objects for each match, ranked by risk level and confidence.

#### Cross-walk: `attribute_historical_pattern(attribute, domain)`

Column-name keyword matching catches a feature when its *name* signals a historical pattern (`zip_code`, `last_name`, `religion_code`). It does NOT catch a finding that the engine produced about a *protected attribute* in a regulated domain when the engine phrased its evidence statistically ("selection-rate gap on age", "four-fifths ratio on gender"). Those findings are also documented historical-discrimination patterns -- they just don't say the word "historical".

`attribute_historical_pattern(attribute, domain, jurisdiction='')` closes that gap. It is a structured lookup over the canonical (attribute class, domain) precedents already encoded in `_ATTR_DOMAIN_PATTERNS` and returns a citation-backed envelope (or `None` when the combination has no documented precedent; silence beats invention).

```python
from vfairness import attribute_historical_pattern

attribute_historical_pattern("race", "lending")
# {
#   "pattern_id": "lending_race_redlining",
#   "label": "Race in lending (redlining)",
#   "summary": "Credit-allocation models inherit decades of redlining; ...",
#   "citations": ["Rothstein (2017), 'The Color of Law'",
#                 "Bartlett et al. (2022), JFE"],
#   "attribute_class": "race",
#   "domain": "lending",
#   "jurisdiction": "",
# }

attribute_historical_pattern("age", "hiring")
# -> hiring_age (ADEA / EU Directive 2000/78)
attribute_historical_pattern("gender", "insurance")
# -> insurance_gender (CJEU Test-Achats 2011)
attribute_historical_pattern("religion", "education")
# -> None  (no documented precedent encoded for this pair)
```

**Coverage** (24 documented pairs):

| Attribute class | Documented domains | Named precedent |
|---|---|---|
| `race` | hiring, lending, healthcare, justice, insurance, education | Bertrand-Mullainathan, redlining, Obermeyer, COMPAS, NAIC, Gender Shades |
| `gender` | hiring, lending, healthcare, insurance, education | Amazon recruiter, ECOA, FDA clinical-trials, Test-Achats |
| `age` | hiring, lending, insurance | ADEA, ECOA, EU AI Act Annex III(5)(c) |
| `national_origin` | hiring, lending | Title VII, ECOA |
| `religion` | hiring, lending | Title VII, ECOA |
| `disability` | hiring, healthcare, insurance | ADA, Section 1557 ACA |
| `geographic` (zip, postcode, neighbourhood, census tract) | lending, insurance, healthcare | Redlining (Rothstein, Bartlett), NAIC, Section 1557 |

**Input normalisation**. The attribute argument is lower-cased and aliased: `Race / Ethnicity`, `race_ethnicity`, `skin_colour`, `sex`, `gender_identity`, `age_group`, `age_bracket`, `zip`, `postcode`, `neighbourhood`, `census_tract` all resolve to their canonical class. (`dob` is not an alias: date-of-birth columns are `identity_pii` in the role typology and return `None` here.) Domain aliases (`recruitment` → `hiring`, `credit` / `loan` → `lending`, `medical` → `healthcare`, `recidivism` → `justice`) follow the same map as `domain_historical_context()`. The two helpers read separate registries, though: `attribute_historical_pattern` looks up the attribute-by-domain precedents in `_ATTR_DOMAIN_PATTERNS`, while `domain_historical_context()` reads the coarser domain-only `_DOMAIN_HISTORICAL_PRECEDENT`.

**Consumer**. Pulse calls this in `operations/pulse/orchestrator.py` for every bias finding and attaches the result as a `historicalPattern` envelope. The Validant frontend then routes findings into the "Historical pattern" channel by checking that structured flag instead of regex-matching evidence text -- so race / age / gender / national-origin disparities in regulated domains land in the channel even when their evidence reads "selection-rate gap" or "four-fifths ratio".

**No fabrication contract**. Citations are pulled by reference from `_ATTR_DOMAIN_PATTERNS`. Adding a new (attribute, domain) entry requires that entry to carry curated citations; otherwise the helper returns `None` rather than emit an unattributed pattern.

#### Academic References

| Region | Pattern | Key Research |
|--------|---------|--------------|
| 🇺🇸 | **Redlining** | Rothstein (2017). *The Color of Law*. [Mapping Inequality Project](https://dsl.richmond.edu/panorama/redlining/) |
| 🇺🇸 | **Healthcare Bias** | [Obermeyer et al. (2019)](https://doi.org/10.1126/science.aax2342). Dissecting racial bias in healthcare algorithms. *Science*. |
| 🇺🇸 | **Name Discrimination** | [Bertrand & Mullainathan (2004)](https://doi.org/10.1257/0002828042002561). Are Emily and Greg More Employable? *AER*. |
| 🇪🇺 | **Dutch Childcare Scandal** | Toeslagenaffaire (2020). Dutch Parliamentary Inquiry — algorithm targeted dual-nationality families. |
| 🇪🇺 | **SyRI Welfare System** | [The Hague District Court (2020)](https://uitspraken.rechtspraak.nl/details?id=ECLI:NL:RBDHA:2020:1878). NJCM v. Netherlands — ECHR Art. 8 violation. |
| 🇪🇺 | **SCHUFA Credit Scoring** | BGH ruling (2024); [CJEU C-634/21 (2023)](https://curia.europa.eu/). GDPR Art. 22 and automated scoring. |
| 🇪🇺 | **Facial Recognition** | [Buolamwini & Gebru (2018)](https://doi.org/10.1145/3287560.3287596). Gender Shades. *FAT Conference*. |
| 🇪🇺 | **Emotion Recognition** | [Barrett et al. (2019)](https://doi.org/10.1177/1529100619832930). Emotional expressions reconsidered. *Psychological Science in the Public Interest*. |
| ⚖️ | **EU AI Act** | [Regulation 2024/1689](https://artificialintelligenceact.eu/). Full text of the EU AI Act. |
| 🇨🇭 | **Permit Discrimination** | SFM/Uni Neuchâtel. Permit type as proxy for nationality. [EKR/CFR reports](https://www.ekr.admin.ch/). |
| 🇨🇭 | **Housing Discrimination** | [Uni Zürich correspondence studies](https://www.uzh.ch/): 20-50% fewer callbacks for Balkan/Turkish/African names. |
| 🇨🇭 | **Naturalisation Bias** | [BGE 129 I 217 (2003)](https://www.bger.ch/). Emmen ballot-box naturalisations ruled discriminatory. |

#### Official Data Sources & Legal Frameworks

| Source | Description | URL |
|--------|-------------|-----|
| **HOLC Redlining Maps** | US historical maps (1930s-40s) | [dsl.richmond.edu/panorama/redlining](https://dsl.richmond.edu/panorama/redlining/) |
| **HMDA Data** | US home mortgage lending data | [ffiec.cfpb.gov](https://ffiec.cfpb.gov/) |
| **EEOC Guidelines** | Criminal records in employment | [eeoc.gov](https://www.eeoc.gov/laws/guidance/enforcement-guidance-consideration-arrest-and-conviction-records) |
| **CDC Social Vulnerability Index** | Area-level social vulnerability scores | [atsdr.cdc.gov/svi](https://www.atsdr.cdc.gov/placeandhealth/svi/) |
| ⚖️ **EU AI Act (2024/1689)** | Full text — prohibited & high-risk AI | [artificialintelligenceact.eu](https://artificialintelligenceact.eu/) |
| ⚖️ **EU AI Act Art. 5** | Prohibited AI practices | [Art. 5](https://artificialintelligenceact.eu/article/5/) |
| ⚖️ **EU AI Act Annex III** | High-risk AI use cases | [Annex III](https://artificialintelligenceact.eu/annex/3/) |
| **GDPR Art. 22** | Right against purely automated decisions | [gdpr-info.eu/art-22](https://gdpr-info.eu/art-22-gdpr/) |
| **EU Race Equality Directive** | 2000/43/EC anti-discrimination | [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32000L0043) |
| **FRA (Fundamental Rights Agency)** | EU discrimination surveys & data | [fra.europa.eu](https://fra.europa.eu/) |
| **EBA AI Guidelines** | AI in credit risk assessment | [eba.europa.eu](https://www.eba.europa.eu/) |
| 🇨🇭 **EKR/CFR** | Swiss Federal Commission against Racism | [ekr.admin.ch](https://www.ekr.admin.ch/) |
| 🇨🇭 **nDSG (revDSG)** | Swiss Data Protection Act (2023) | [fedlex.admin.ch](https://www.fedlex.admin.ch/eli/cc/2022/491/de) |
| 🇨🇭 **BFS / OFS** | Swiss Federal Statistical Office | [bfs.admin.ch](https://www.bfs.admin.ch/) |
| 🇨🇭 **SKOS / CSIAS** | Swiss social welfare guidelines | [skos.ch](https://skos.ch/) |

### Representation Bias Detection

Compares dataset demographics to population benchmarks:

```python
from vfairness import detect_representation_bias, compare_to_benchmark

# Compare to benchmark (one result per protected attribute)
results = detect_representation_bias(
    df,
    ['gender'],
    benchmarks={'gender': {'Male': 0.49, 'Female': 0.51}},
)

for r in results:
    print(f"{r.attribute}: severity {r.severity.value}")
    # chi_squared_pvalue is Optional[float]. It is None with no benchmark to
    # test against, and None when severity is 'insufficient_data', where the
    # sample was too small to draw any conclusion from.
    if r.chi_squared_pvalue is None:
        print("  Chi-squared p-value: COULD NOT CHECK - no test was run")
    else:
        print(f"  Chi-squared p-value: {r.chi_squared_pvalue:.4f}")
    for group, share in r.group_distributions.items():
        print(f"  {group}: {share:.1%} of dataset")
```

### Statistical Disparity Analysis

Comprehensive hypothesis testing with effect sizes:

```python
from vfairness import analyze_statistical_disparities, run_disparity_tests

# Full analysis (one result per detected disparity;
# an empty list means nothing was flagged, not that nothing ran)
results = analyze_statistical_disparities(
    df,
    ['gender'],
    outcome_columns=['approved'],
)

for r in results:
    print(f"{r.feature} by {r.protected_attribute}: {r.test_name}")
    print(f"  p-value: {r.pvalue:.4f}")
    print(f"  Effect size: {r.effect_size:.3f} ({r.effect_interpretation.value})")
```

### Proxy Variable Identification

Detects features correlated with protected attributes:

```python
from vfairness import identify_proxy_variables, compute_proxy_correlations

# Find proxies
proxies = identify_proxy_variables(
    df,
    protected_attributes=['gender', 'race'],
    correlation_threshold=0.3
)

for proxy in proxies:
    print(f"{proxy.feature} → {proxy.protected_attribute}")
    print(f"  Correlation: {proxy.correlation:.3f}")
    print(f"  Risk level: {proxy.risk_level.value}")
```

### Geographic Discrimination Data Integration

The library provides integration with official geographic discrimination databases, including historical HOLC redlining maps.

#### HOLC Redlining Maps

Access historical Home Owners' Loan Corporation (HOLC) redlining data:

```python
from vfairness.preprocessing.bias_detection import (
    HOLCGrade,
    get_available_holc_cities,
    fetch_holc_data,
    lookup_holc_grade_by_zip,
    assess_geographic_feature_risk,
)

# List cities with available HOLC data
cities = get_available_holc_cities()
print(f"HOLC data available for {len(cities)} cities")
print(cities[:5])  # ['Atlanta_GA', 'Baltimore_MD', ...]

# Fetch HOLC data for a specific city (requires internet)
holc_data = fetch_holc_data('Detroit', 'MI')
if holc_data:
    areas = holc_data.get('features', [])
    print(f"Found {len(areas)} HOLC areas in Detroit")

# Look up approximate HOLC grade for a ZIP code
grade = lookup_holc_grade_by_zip('48201')  # Detroit downtown
print(f"HOLC Grade: {grade}")  # HOLCGrade.D (redlined)

# Assess geographic feature risk in your dataset
zip_codes = df['zip_code'].tolist()
assessment = assess_geographic_feature_risk(zip_codes, feature_type='zip_code')

print(f"Risk score: {assessment.risk_score:.2f}")
print(f"HOLC coverage: {assessment.holc_coverage:.1%}")
print(f"Disparate impact risk: {assessment.disparate_impact_risk}")
print(f"Grade distribution: {assessment.grade_distribution}")

for rec in assessment.recommendations:
    print(f"  → {rec}")
```

**HOLC Grades:**

| Grade | Name | Historical Meaning | Risk Level |
|-------|------|-------------------|------------|
| A | "Best" | White, affluent neighborhoods | Low |
| B | "Still Desirable" | Middle-class, generally white | Medium |
| C | "Definitely Declining" | Immigrant/mixed areas | High |
| D | "Hazardous" | Minority neighborhoods (redlined) | Critical |

#### CDC Social Vulnerability Index (SVI)

Access contemporary area-level vulnerability data:

```python
from vfairness.preprocessing.bias_detection import SVI_THEMES, get_svi_data_url

# View SVI themes
for theme, description in SVI_THEMES.items():
    print(f"{theme}: {description}")

# Get data download URL
url = get_svi_data_url(year=2022)
print(f"Download SVI data: {url}")
```

#### Data Quality Notes

```python
from vfairness.preprocessing.bias_detection import print_data_quality_notes

# Print data quality and limitations documentation
print_data_quality_notes()
```

**Important Considerations:**
- HOLC maps are from the 1930s-1940s; neighborhood demographics have changed
- ZIP codes are an approximation; actual HOLC boundaries require spatial joins
- Use geographic risk as one signal among many in bias detection
- Always validate with local knowledge and contemporary data

---

## Comparison with Other Libraries

| Feature | vfairness | AIF360 | Fairlearn |
|---------|-----------|--------|-----------|
| **Classification Metrics** | | | |
| Demographic Parity | `demographic_parity_difference()` | `statistical_parity_difference()` | `demographic_parity_difference()` |
| Equal Opportunity | `equal_opportunity_difference()` | `equal_opportunity_difference()` | Use `MetricFrame` with TPR |
| Equalized Odds | `equalized_odds_difference()` | `average_odds_difference()` | `equalized_odds_difference()` |
| Predictive Parity | `predictive_parity_difference()` | Manual computation | Use `MetricFrame` with precision |
| **Regression Metrics** | | | |
| MAE Parity | `mae_parity_difference()` | Not available | Manual |
| RMSE Parity | `rmse_parity_difference()` | Not available | Manual |
| **Ranking Metrics** | | | |
| Exposure Parity | `exposure_parity_difference()` | Not available | Not available |
| Attention-Weighted | `attention_weighted_rank_fairness()` | Not available | Not available |
| **Statistical Validation** | | | |
| Bootstrap CI | Built-in | Separate module | Not built-in |
| Bayesian CI (group rates, not disparities) | Built-in | Not available | Not available |
| Effect Sizes | Built-in | Not available | Not available |
| Multiple Testing | Built-in (Bonferroni, FDR) | Not available | Not available |
| **Explainability** | | | |
| FairExplAIner | `FairExplAIner`, `explain_fairness_report()` | Not available | Not available <!--cs-->**[Checked]**<!--/cs--> |
| Metric Explanations | Built-in (definitions, benchmarks, recommendations) | Not available | Not available |
| Severity Assessment | Built-in (info, low, medium, high, critical) | Not available | Not available |
| **MLOps Integration** | | | |
| MLflow Logging | `log_fairness_to_mlflow()` | Manual | Manual |
| pytest Assertions | `assert_fairness()` | Not available | Not available |
| **Visualization** | | | |
| Built-in Plots | Yes | Yes | Yes |
| Report Generation | `plot_fairness_report()` | Manual | Manual |

---

## Production Features (All Modules)

> All LLM, Agent, and Multi-Agent result types include these enterprise features.

### RunMetadata

Every result object includes a `metadata` attribute with audit trail information:
<!-- cap-status: RunMetadata -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->


```python
result = analyzer.analyze_sentiment(texts_a, texts_b)

result.metadata.timestamp        # ISO 8601 UTC (e.g. "2026-03-27T14:30:00Z")
result.metadata.library_version  # the installed vfairness.__version__
result.metadata.parameters       # {"alpha": 0.05, "metric": "sentiment", ...}
```

### Serialization (to_dict / to_json)

All result objects support serialization for database storage:

```python
data = result.to_dict()    # Plain Python dict (numpy arrays → lists, datetime → ISO string)
json_str = result.to_json()  # JSON string with indent=2

# Round-trip safe
import json
assert json.loads(json_str) == data
```

### Structured Logging

All modules use Python's `logging` module:

```python
import logging
logging.basicConfig(level=logging.INFO)

# All vfairness operations now emit structured logs:
# INFO:vfairness.llm.output_analysis:analyze_all: group_a=group_a (50 samples), group_b=group_b (50 samples), correction=benjamini_hochberg
# WARNING:vfairness.llm.output_analysis:Sample size (18) below recommended minimum of 25
# INFO:vfairness.agents.correspondence:Starting correspondence test with 50 samples per group
```

### Progress Callbacks

Batch operations accept an optional `progress_callback`:

```python
from tqdm import tqdm

pbar = tqdm(total=100)
proxy.send_batch(
    prompts=my_prompts,
    n_runs=25,
    progress_callback=lambda current, total: pbar.update(1)
)
pbar.close()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `progress_callback` | `Callable[[int, int], None]` | Called with `(current_step, total_steps)` |

---

## Module 7: LLM Fairness Testing (`vfairness.llm`)

> Test LLMs for bias without training data access. Send prompts, collect responses, measure disparities.

### LLMApiProxy

```python
<!-- cap-status: LLMApiProxy -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

from vfairness.llm import LLMApiProxy

proxy = LLMApiProxy(
    endpoint_url="https://api.openai.com/v1/chat/completions",
    api_format="openai",       # "openai" | "anthropic" | "custom"
    auth_token="sk-...",
    model_name="gpt-4o-mini"
)

# Test connection
result = proxy.test_connection()  # → {success, latency_ms, model_name, ...}

# Single prompt
response = proxy.send_prompt("Hello", system_prompt="Be helpful")  # → {text, latency_ms}

# Batch (25 runs per prompt for non-determinism)
responses = proxy.send_batch(["prompt1", "prompt2"], n_runs=25)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint_url` | `str` | Base URL of the LLM API endpoint. Validated by the egress guard (see below) |
| `api_format` | `str` | `"openai"` \| `"anthropic"` \| `"custom"` |
| `auth_token` | `str, optional` | Bearer token or API key |
| `model_name` | `str, optional` | Model identifier included in requests |
| `timeout` | `int` | Request timeout in seconds (default `_DEFAULT_TIMEOUT`) |
| `max_retries` | `int` | Retry attempts on transient failures (default `_DEFAULT_MAX_RETRIES`) |
| `allow_loopback` | `bool` | Opt-in for a model server on this machine (default `False`) |

**Egress guard.** The endpoint URL is validated by the `vfairness.net` SSRF guard
at construction time. A loopback, RFC1918, link-local or cloud-metadata host
raises `ValueError: endpoint_url refused by the egress guard: ...`. Set
`allow_loopback=True` only for a trusted model server on this machine (Ollama,
vLLM, a test stub): it unlocks 127.0.0.0/8 and ::1 and nothing else. RFC1918,
link-local and the cloud-metadata addresses stay refused whatever the caller
asks for. Never set it from an untrusted URL.

```python
# Local model server on this machine
proxy = LLMApiProxy(
    endpoint_url="http://127.0.0.1:11434/v1/chat/completions",
    api_format="openai",
    model_name="mistral-small",
    allow_loopback=True,
)
```

### CounterfactualTester

```python
<!-- cap-status: CounterfactualTester -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

from vfairness.llm import CounterfactualTester

tester = CounterfactualTester(proxy, n_runs=25, alpha=0.05, random_seed=42)

# 9 swap strategies (all 6 Salimian et al. metamorphic relations + 3 more):
#   name_swap, pronoun_swap, attribute_inversion, contextual_framing,
#   paraphrase_invariance, order_invariance, irrelevant_attribute_addition,
#   negation_consistency, persona_based

result = tester.run_test(
    template="Write a recommendation for {name}.",
    swap_pairs={"name": ["James", "Jamal"]},
    strategy="name_swap"
)
# result.disparity_metrics → {sentiment_delta, toxicity_delta, cosine_similarity, ...}
# result.is_significant → bool

# Persona-based testing (Cheng et al. 2023 "Marked Personas"):
# Prepends full demographic persona descriptions to the prompt.
# Mode 1: Auto-enrich name pairs with known demographic associations
result = tester.run_test(
    template="Evaluate the creditworthiness of {name}.",
    swap_pairs={"name": ["James", "Jamal"]},
    strategy="persona_based"
)
# Generates: "The following question concerns James, a white man, who works
#  as a mid-level professional, with a bachelor's degree, living in a
#  mid-sized city.\n\nEvaluate the creditworthiness of James."
# vs: "...concerns Jamal, a Black man, ..."

# Mode 2: Explicit attribute pairs for intersectional analysis
result = tester.run_test(
    template="Write a recommendation letter for {name}.",
    swap_pairs={"race": ["white", "Black"], "gender": ["male", "female"]},
    strategy="persona_based"
)
# Generates 4 factorial variants: white man, white woman, Black man, Black woman
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `proxy` | `LLMApiProxy` | Connected API proxy |
| `n_runs` | `int` | Runs per variant (default 25, per LangFair) |
| `alpha` | `float` | Significance level (default 0.05) |
| `random_seed` | `int, optional` | Seed for reproducible randomization |
| `template` | `str` | Prompt with `{placeholder}` |
| `swap_pairs` | `dict` | `{"placeholder": ["value_a", "value_b"]}`. For `persona_based`, also accepts explicit attribute keys like `{"race": [...], "gender": [...]}` |
| `strategy` | `str` | One of the 9 swap strategies |

### OutputAnalyzer

```python
<!-- cap-status: OutputAnalyzer -->
**Beta status: Checked.** 14 code units behind this name: 14 checked
<!-- /cap-status -->

from vfairness.llm import OutputAnalyzer

analyzer = OutputAnalyzer(alpha=0.05)

# Individual analysis: one call per metric
result = analyzer.analyze_sentiment(texts_a, texts_b)     # VADER (production) — Mann-Whitney U + Cohen's d
result = analyzer.analyze_toxicity(texts_a, texts_b)       # alt-profanity-check SVM (production)
result = analyzer.analyze_refusal_rate(texts_a, texts_b)   # 50+ patterns, 5 categories (production)
result = analyzer.analyze_helpfulness(texts_a, texts_b)    # Multi-signal heuristic (6 quality signals)
result = analyzer.analyze_stereotype(texts_a, texts_b)     # 61 terms + 14 phrase patterns
result = analyzer.analyze_semantic_quality(texts_a, texts_b)      # Claim extraction + quality scoring
result = analyzer.analyze_regard(texts_a, texts_b)                # sasha/regardv3 (Sheng et al. 2019)
result = analyzer.analyze_information_quality(texts_a, texts_b)   # Entities, statistics, reasoning depth
result = analyzer.analyze_representation(texts_a, texts_b)        # Demographic reference coverage
result = analyzer.analyze_framing(texts_a, texts_b)               # Hedging, certainty, evaluative language
result = analyzer.analyze_length(texts_a, texts_b)         # Word count comparison
result = analyzer.analyze_llm_judge(texts_a, texts_b)      # Requires a judge endpoint; None if unconfigured

# All at once (with Bonferroni correction)
results = analyzer.analyze_all(texts_a, texts_b, "group_a", "group_b",
                                correction_method="bonferroni")  # or "benjamini_hochberg"
```

`analyze_all()` runs every standard metric listed above except the LLM judge,
which is appended only when the analyzer is configured with a judge endpoint.

**Pluggable scorers** — the library ships with multiple scorer backends:

**VADER sentiment (recommended):** Installed automatically with `pip install vfairness[llm]`.
Uses the VADER lexicon (7,500+ words, valence-aware) and returns a compound score in [-1, +1].
No configuration needed — if `vaderSentiment` is installed, it is used by default.

**alt-profanity-check toxicity (recommended):** Installed automatically with `pip install vfairness[llm]`.
Uses an SVM model trained on 200K samples from Wikipedia and Reddit to predict offensiveness
probability in [0, 1]. No configuration needed — if `alt-profanity-check` is installed, it is
used by default. Approximately 95% accuracy.

**Keyword fallback scorers:** If VADER / alt-profanity-check are not installed, the library
falls back to simple keyword-based scorers (60 words for sentiment: 30 positive + 30 negative, and 25 for toxicity). These
emit `UserWarning` at instantiation to remind you they are low-accuracy placeholders.

**Check active scorers at runtime:**
```python
from vfairness.llm import scorer_status
print(scorer_status())
# Values depend on which optional scorers are installed. With VADER and
# alt-profanity-check available in-process, and no ML sidecar configured:
# {
#   "sentiment": {"scorer": "VADER (lexicon-based, 2014). Limitations: no context/sarcasm detection.",
#                  "quality": "good",
#                  "upgrade": "Enable ML sidecar (set VFAIRNESS_SIDECAR_PYTHON + VFAIRNESS_SIDECAR_SCRIPT) ..."},
#   "toxicity":  {"scorer": "alt-profanity-check (SVM, bag-of-words). Cannot capture contextual toxicity.",
#                  "quality": "good",
#                  "upgrade": "Enable ML sidecar ..."},
#   "refusal":   {"scorer": "Pattern-based (50+ refusal patterns across 5 categories)",
#                  "quality": "production", "upgrade": None},
# }
# "quality" is "production" only for the top rung of a scorer hierarchy (for
# example Detoxify for toxicity); "good" carries a non-None upgrade hint,
# "placeholder" means a keyword fallback is in use, and "unvalidated" means the
# instrument has never been checked against human ratings on this platform
# (the LLM judge, until LF-24): its scores are supplementary evidence.
```

(Excerpt. `scorer_status()` returns one entry per scorer: sentiment, toxicity,
refusal, helpfulness, stereotype, semantic_quality, regard, llm_judge,
information_quality, representation and framing. The sentiment, toxicity,
stereotype and regard entries vary with which optional backends are installed.)

**Custom scorers** — replace any default with your own classifier:
```python
from vfairness.llm.scorers import TextScorer
import numpy as np

class MyTransformerScorer:
    def score(self, text: str) -> float: ...
    def score_batch(self, texts: list[str]) -> np.ndarray: ...

analyzer = OutputAnalyzer(sentiment_scorer=MyTransformerScorer())
```

### NonDeterminismAnalyzer

```python
<!-- cap-status: NonDeterminismAnalyzer -->
**Beta status: Checked.** 7 code units behind this name: 7 checked
<!-- /cap-status -->

from vfairness.llm import NonDeterminismAnalyzer

nda = NonDeterminismAnalyzer(system_type="llm", min_runs=30)  # Override default (25 for LLM, 50 for agent)
print(nda.required_runs())  # 30 (overridden)

# Characterize noise from repeated identical queries
profile = nda.characterize_noise(values)  # → NoiseProfile(mean, variance, noise_floor, ...)

# Separate real bias from randomness
offset = nda.compute_noise_offset(observed_disparity=0.15, noise_floor=0.03)
# offset → {observed, noise_floor, systematic_offset, noise_contribution,
#           exceeds_noise, z_threshold}

# Bootstrap confidence interval (reproducible)
lo, hi = nda.bootstrap_ci(values, n_bootstrap=1000, random_state=42)
```

### noise_floor_from_runs (LF-01, 2026-09-09)
<!-- cap-status: noise_floor_from_runs -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses
<!-- /cap-status -->


`characterize_noise` has always required "metric values from repeated identical
runs". `noise_floor_from_runs` builds that series for you from the raw
responses a counterfactual run already stores, per metric and per variant, and
answers in three states at every level.

```python
from vfairness.llm import noise_floor_from_runs, RUN_METRICS

# result.variants[i]["responses"] from CounterfactualTester.run_test, keyed by variant
out = noise_floor_from_runs(
    {"James": responses_james, "Jamal": responses_jamal},   # first key = reference
    metrics=["sentiment", "refusal_rate", "response_length"],  # subset of RUN_METRICS
    sampling={"temperature": 0.7, "top_p": 1.0, "seed": None},  # recorded verbatim
)

out["state"]                     # "measured" | "could_not_check"
out["variants"]["James"]         # n_runs, n_valid, empty_rate, state
                                 #   ("measured" | "measured_with_limitation" below the
                                 #   recommended run count | "could_not_check" under 2 valid),
                                 #   metrics -> NoiseProfile dict + bootstrap mean_ci
out["metrics"]["sentiment"]      # reference_variant, scorer provenance, and per other variant:
                                 #   observed (mean_ref - mean_v),
                                 #   disparity_noise_floor = 2*sqrt(var_ref/n_ref + var_v/n_v),
                                 #   exceeds_noise, systematic_offset, z_threshold
out["limitations"]               # every reason a number is missing or weaker, in words
```

Nothing in the result is `0.0` for something that was not measured; an
unmeasured value is `None` with a `reason`. A floor measured with a keyword
placeholder scorer carries that fact in `scorer_quality`.

### IntersectionalAnalyzer

```python
<!-- cap-status: IntersectionalAnalyzer -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->

from vfairness.llm import IntersectionalAnalyzer, IntersectionalGroup

# Generate all intersectional groups
groups = IntersectionalGroup.from_attributes({
    "race": ["Black", "White"],
    "gender": ["male", "female"]
})  # → 4 groups: Black_male, Black_female, White_male, White_female

analyzer = IntersectionalAnalyzer(alpha=0.05)
result = analyzer.analyze(
    outputs_by_group={"Black_male": [...], "White_female": [...]},
    groups=groups, metric="sentiment"
)
# result.has_intersectional_bias → bool
# result.most_disadvantaged → IntersectionalGroup | None (None when no significant bias)
# result.n_significant_pairs → int (after Bonferroni)
```

### CoTFaithfulnessAnalyzer

```python
<!-- cap-status: CoTFaithfulnessAnalyzer -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

from vfairness.llm import CoTFaithfulnessAnalyzer

analyzer = CoTFaithfulnessAnalyzer()
result = analyzer.analyze_pair(
    original_output="Salary: $120,000",
    variant_output="Salary: $95,000",
    original_cot="Based on experience and skills...",
    variant_cot="Based on experience and skills...",
    demographic_cue="gender",
    cue_terms=["male", "female", "he", "she"]
)
# result.classification → "unfaithful_silent" (output changed, CoT didn't mention why)
```

### DecodingTrustRunner

Comprehensive trustworthiness benchmark based on Wang et al. (2023, NeurIPS).
<!-- cap-status: DecodingTrustRunner -->
**Beta status: Checked.** 10 code units behind this name: 10 checked
<!-- /cap-status -->

Evaluates LLMs across 8 dimensions and 24 demographic groups using 308 built-in prompts.

```python
from vfairness.llm import DecodingTrustRunner, LLMApiProxy

proxy = LLMApiProxy(endpoint_url="https://api.openai.com/v1/chat/completions",
                    api_format="openai", model_name="gpt-4o")
dt = DecodingTrustRunner(proxy=proxy, random_seed=42)

# Run all 8 dimensions
results = dt.run_all(sample_size=50, progress_callback=lambda cur, tot: print(f"{cur}/{tot}"))
# results → dict of dimension_name → BenchmarkResult

# Or run individual dimensions
stereotype = dt.run_stereotype_bias(categories=["gender", "race"], sample_size=30)
fairness   = dt.run_fairness(sample_size=50)
toxicity   = dt.run_toxicity(sample_size=40)
privacy    = dt.run_privacy(sample_size=30)
ethics     = dt.run_machine_ethics(sample_size=40)
adv_robust = dt.run_adversarial_robustness(sample_size=30)
ood_robust = dt.run_ood_robustness(sample_size=30)
adv_demos  = dt.run_adversarial_demonstrations(sample_size=30)
```

**Dimensions:**

| Dimension | What it measures | Scoring |
|-----------|-----------------|---------|
| Stereotype Bias | Agreement with stereotypical statements | % stereotypical completions (lower = better) |
| Fairness | Outcome disparity across 24 demographic groups | Max demographic parity gap (lower = better) |
| Toxicity | Differential toxicity with adversarial system prompts | Mean toxicity score by condition |
| Privacy | Resistance to PII extraction attempts | % of probes where PII was refused (higher = better) |
| Machine Ethics | Consistency of moral reasoning across demographics | % of symmetric decisions (higher = better) |
| Adversarial Robustness | Bias detection under adversarial perturbations | % of perturbed prompts with consistent responses |
| OOD Robustness | Fairness under distribution shift (style/domain) | Consistency across registers and domains |
| Adversarial Demonstrations | Resistance to biased few-shot examples | % of cases where biased demos did not affect output |

---

### HolisticBias Benchmark

Screening based on Smith et al. (2022, EMNLP). Fills 5 canonical templates with 10 descriptors across 6 identity axes (race/ethnicity, gender, sexual orientation, age, disability, religion) and reports the maximum sentiment disparity within each axis.

```python
from vfairness.llm import BenchmarkRunner, LLMApiProxy

proxy = LLMApiProxy(endpoint_url="https://api.openai.com/v1/chat/completions",
                    api_format="openai", model_name="gpt-4o")
runner = BenchmarkRunner(proxy=proxy)   # proxy is the only constructor argument

# Run all 6 axes (default: 60 descriptors x 5 templates = 300 prompts)
result = runner.run_holistic_bias(progress_callback=lambda cur, tot: print(f"{cur}/{tot}"))

# Subset axes and trim for a quick screening pass
result = runner.run_holistic_bias(
    axes=["gender", "race_ethnicity"],
    descriptors_per_axis=5,
    templates_per_descriptor=3,
)

# result.overall_score            → mean max-min sentiment disparity across axes, in [0, 1]
# result.category_breakdown       → dict keyed by axis with per-descriptor sentiment means
#                                    and lowest_sentiment_descriptor / highest_sentiment_descriptor
# result.category_breakdown["gender"]["per_descriptor_means"]  → {descriptor: mean_sentiment}
```

**Signature**

```python
BenchmarkRunner.run_holistic_bias(
    axes: list[str] | None = None,                    # default: all 6 axes
    descriptors_per_axis: int | None = None,          # default: 10 (all)
    templates_per_descriptor: int | None = None,      # default: 5 (all)
    progress_callback: Callable[[int, int], None] | None = None,
) -> BenchmarkResult
```

**Scope**

The bundled subset (`vfairness.llm._holisticbias_data`) is a curated 60-descriptor / 5-template slice — roughly 1/10 of the full Smith et al. dataset — sized for fast per-assessment screening. Use `DecodingTrustRunner` for a heavier-weight evaluation.

---

### New Output Scorers

These plug into `OutputAnalyzer` (and any code that takes a scorer with `.score(text)` / `.score_batch(texts)`). All return floats; ranges noted per scorer.

#### KeywordRegardScorer / TransformerRegardScorer
<!-- cap-status: KeywordRegardScorer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->


Measures **regard** — how language frames a demographic group's social standing (Sheng et al. 2019). Returns `[-1, 1]` (negative = lower regard, positive = higher regard).

```python
from vfairness.llm import KeywordRegardScorer, DEFAULT_REGARD_SCORER

# Auto-selected: TransformerRegardScorer if `transformers` is installed,
# otherwise KeywordRegardScorer
scorer = DEFAULT_REGARD_SCORER
scorer.score("She is a respected surgeon.")          # → positive value
scorer.score_batch(["...", "..."])                    # → np.ndarray
```

`TransformerRegardScorer(model="sasha/regardv3")` wraps the HuggingFace `sasha/regardv3` NLI pipeline. Falls back to keyword lists when transformers are unavailable.

#### LLMJudgeScorer

LLM-as-judge scoring with a structured rubric (helpfulness, fairness, specificity, completeness). Returns `[0, 1]`.
<!-- cap-status: LLMJudgeScorer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->


```python
from vfairness.llm import LLMJudgeScorer

judge = LLMJudgeScorer(
    endpoint_url="https://api.openai.com/v1/chat/completions",
    api_format="openai",          # or "anthropic"
    auth_token="sk-...",
    model_name="gpt-4o",
    timeout=60,
)
judge.score("The applicant should be reviewed against the published criteria...")
```

Single-response design (no pairwise comparison) to mitigate position bias.

#### InformationQualityScorer

Information density via named-entity count, statistics, conditional reasoning, evidence markers, and filler detection. Returns `[0, 1]`.
<!-- cap-status: InformationQualityScorer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->


```python
from vfairness.llm import InformationQualityScorer, DEFAULT_INFORMATION_QUALITY_SCORER

iq = DEFAULT_INFORMATION_QUALITY_SCORER
iq.score("In 2022, 78% of applicants in cohort A...")     # → high
iq.score("It depends on many factors.")                    # → low
```

Used to detect "information withholding" — a subtle disparity where one group gets shallow answers while another gets richly sourced ones.

#### RepresentationScorer

Counts explicit references to 5 demographic categories (gender, race/ethnicity, age, disability, religion). Returns `[0, 1]` where 1 = broad proportional coverage.
<!-- cap-status: RepresentationScorer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->


```python
from vfairness.llm import RepresentationScorer, DEFAULT_REPRESENTATION_SCORER

rep = DEFAULT_REPRESENTATION_SCORER
rep.score("The example mentions a young Black woman with a disability...")  # high coverage
```

Detects **erasure bias** — outputs that mention only the majority group.

#### FramingScorer

Positivity of linguistic framing. Combines hedging density, certainty markers, and evaluative-language balance. Returns `[0, 1]` (0 = heavily hedged/negative, 0.5 = neutral, 1 = confident/positive).
<!-- cap-status: FramingScorer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->


```python
from vfairness.llm import FramingScorer, DEFAULT_FRAMING_SCORER

fr = DEFAULT_FRAMING_SCORER
fr.score("Maybe consider, possibly, that...")        # low
fr.score("This is the recommended approach.")        # high
```

Useful for catching disparities where one group receives confident guidance and another receives hedged, qualified guidance.

#### Default Scorer Constants

```python
from vfairness.llm import (
    DEFAULT_REGARD_SCORER,                # TransformerRegardScorer or KeywordRegardScorer
    DEFAULT_INFORMATION_QUALITY_SCORER,
    DEFAULT_REPRESENTATION_SCORER,
    DEFAULT_FRAMING_SCORER,
)
```

Module-level singletons that auto-select the best available implementation. Pass them straight into `OutputAnalyzer` or call `.score()` / `.score_batch()` directly.


### `EmbeddingBiasDetector`

Measures bias baked into word or sentence embeddings with the Word Embedding
<!-- cap-status: EmbeddingBiasDetector -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->

Association Test (WEAT, Caliskan et al., 2017) and its sentence-level extension
SEAT (May et al., 2019). It tests whether two target concept sets (for example
career terms vs family terms) are differentially associated with two attribute
sets (for example male terms vs female terms). Dependency-light: NumPy plus
SciPy, with any `{word: vector}` map or an injected `embed_fn`, so no model
download happens at import time.

```python
from vfairness.llm.embedding_bias import EmbeddingBiasDetector

detector = EmbeddingBiasDetector(embeddings=word_vectors)   # or embed_fn=...
result = detector.weat(
    target_a=career_terms, target_b=family_terms,
    attribute_a=male_terms, attribute_b=female_terms,
)
print(result.effect_size, result.p_value, result.severity)

# SEAT is the same test applied to sentence embeddings. Pass sentence-resolved
# inputs and set sentence_resolved=True so the result is honestly tagged SEAT;
# on plain word vectors the result stays method="WEAT" with a clarifying note.
seat_result = detector.seat(
    target_a=career_sentences, target_b=family_sentences,
    attribute_a=male_sentences, attribute_b=female_sentences,
    sentence_resolved=True,
)
```

**Signatures:**

- `weat(target_a, target_b, attribute_a, attribute_b, n_permutations=10000, random_state=42) -> WEATResult`
- `seat(..., sentence_resolved=False)` (same arguments as `weat`; tags the result `method="SEAT"` only when `sentence_resolved=True`, otherwise `method="WEAT"` plus a note)

**`WEATResult` fields:** `effect_size` (Cohen's d style), `p_value`
(**two-sided** permutation-test p on `|statistic|`), `test_statistic`, the four
set sizes, `severity`, `interpretation`, `notes`, and `method` (`"WEAT"` or
`"SEAT"`). The test is two-sided, so a stereotype in either direction is
flagged: a large positive effect size means the embedding associates the first
target set with the first attribute set, a large negative effect size means it
associates the first target set with the second attribute set; both are bias.

### `TextFairnessAnalyzer`

Measures identity-term bias of a text classifier (toxicity, sentiment,
<!-- cap-status: TextFairnessAnalyzer -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->

moderation) by comparing per-group mean scores. Works on any
`score_fn(texts) -> scores` callable, so it wraps any model or API.

```python
from vfairness.llm.text_fairness import TextFairnessAnalyzer

analyzer = TextFairnessAnalyzer(score_fn=toxicity_model.predict)
result = analyzer.analyze(
    texts_by_group={
        "muslim": muslim_texts,
        "christian": christian_texts,
        "jewish": jewish_texts,
    },
    higher_is_worse=True,   # toxicity: high score = harm
)
print(result.worst_group, result.max_gap, result.p_value, result.severity)
```

**Signature:** `analyze(texts_by_group, higher_is_worse=True) -> TextFairnessResult`

**`TextFairnessResult` fields:** `overall_mean`, `groups` (per-group
`GroupScore` records), `worst_group`, `max_gap` (signed gap of the worst group
vs the overall mean), `p_value` (Mann-Whitney U, worst group vs the rest),
`severity`, and `interpretation`.

---

## Module 8: Agent Fairness Testing (`vfairness.agents`)

> Test AI agents that take actions, use tools, and make decisions.

### CorrespondenceTester

```python
<!-- cap-status: CorrespondenceTester -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

from vfairness.agents import CorrespondenceTester

tester = CorrespondenceTester(alpha=0.05)

# Test paired outcomes
result = tester.analyze_outcomes(
    outcomes_a=[85, 92, 78, 88],  # Scores for Group A
    outcomes_b=[72, 65, 70, 68],  # Scores for Group B
    artifact_type="resume"
)
# result.disparity_metric, result.p_value, result.confidence_interval, result.is_significant

# Four-fifths rule (EEOC adverse impact)
info = tester.four_fifths_rule(rate_a=0.85, rate_b=0.65)
# info → {ratio, adverse_impact (bool)}
```

### ToolBiasAuditor

```python
<!-- cap-status: ToolBiasAuditor -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->

from vfairness.agents import ToolBiasAuditor

auditor = ToolBiasAuditor(alpha=0.05)
results = auditor.analyze_tool_calls(
    traces_a=[{"tool": "search"}, {"tool": "calculator"}],  # Group A: one dict per tool invocation
    traces_b=[{"tool": "search"}, {"tool": "search"}]       # Group B: one dict per tool invocation
)
# Each trace dict must carry a "tool" key (singular) naming one invoked tool.
# Per-tool result: tool_name, invocation_rate_a, invocation_rate_b,
# disparity_ratio, p_value, confidence_interval, is_significant
```

### PipelineTracker

```python
<!-- cap-status: PipelineTracker -->
**Beta status: Checked.** 4 code units behind this name: 4 checked
<!-- /cap-status -->

from vfairness.agents import PipelineTracker

tracker = PipelineTracker(["retrieval", "reasoning", "tool_selection", "action"])
tracker.record_stage("retrieval", outcomes_a, outcomes_b)
tracker.record_stage("reasoning", outcomes_a, outcomes_b)
tracker.record_stage("tool_selection", outcomes_a, outcomes_b)
tracker.record_stage("action", outcomes_a, outcomes_b)

results = tracker.compute_cumulative()  # Per-stage + cumulative bias
print(tracker.identify_bias_source())   # → "retrieval" (biggest contributor)
```

### TemporalTracker

```python
<!-- cap-status: TemporalTracker -->
**Beta status: Checked.** 7 code units behind this name: 7 checked
<!-- /cap-status -->

from vfairness.agents import TemporalTracker

tracker = TemporalTracker()
for turn in range(20):
    tracker.record_turn(turn, group_a_outcomes, group_b_outcomes)

tracker.detect_drift(threshold=0.1)              # Simple threshold
tracker.detect_drift_cusum(drift_limit=5.0)      # CUSUM change-point
tracker.detect_drift_ewma(span=5, sigma_limit=3) # EWMA control chart
tracker.detect_feedback_loop()                    # Kendall's tau monotonicity
```

Three states, never two. All four need at least 3 turns with a finite
disparity; a turn whose metric could not be computed is excluded and counted,
not fed to the test. When fewer than 3 remain, the verdict is `None` (could
not check, NOT "no drift") and the statistics are `nan`, with a warning naming
how many turns qualified:

| Key | Measured | Not assessed |
|-----|----------|--------------|
| `detect_drift(...)` | `True` / `False` | `None` |
| `has_feedback_loop`, `has_drift` | `True` / `False` | `None` |
| `trend_direction` | `'increasing'` / `'decreasing'` / `'stable'` | `'not_assessed'` |
| `trend_strength` (tau), `p_value`, `max_cusum`, `center_line` | a number | `nan` |

Every result also carries `n_turns_measured` and `n_turns_unmeasurable`, so a
verdict says how much of the interaction it rests on.

### ActionBiasAnalyzer

```python
<!-- cap-status: ActionBiasAnalyzer -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->

from vfairness.agents import ActionBiasAnalyzer

analyzer = ActionBiasAnalyzer(alpha=0.05)

# Measure outcome disparities (e.g., salary recommendations, approval rates)
result = analyzer.analyze_outcomes(
    actions_a=[{"outcome": 120000}, {"outcome": 115000}, {"outcome": 125000}],
    actions_b=[{"outcome": 95000}, {"outcome": 92000}, {"outcome": 98000}],
    outcome_field="outcome"
)
# result.disparity → float
# result.effect_size → Cohen's d
# result.confidence_interval → bootstrap CI
# result.is_significant → bool (two-sided Mann-Whitney U at alpha;
#   the raw p-value itself is not exposed on the result)

# Analyze delegation routing patterns
delegation = analyzer.analyze_delegation(
    routing_a=["tier1", "tier1", "tier2"],  # Where Group A gets routed
    routing_b=["tier2", "tier3", "tier3"]   # Where Group B gets routed
)
# delegation → {overall_chi2, overall_p_value, is_significant, per_target: {...}}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | `float` | Significance level (default 0.05) |
| `actions_a` | `list[dict]` | Action outcomes for Group A |
| `actions_b` | `list[dict]` | Action outcomes for Group B |
| `outcome_field` | `str` | Key in action dict containing the outcome value |

---

## Module 9: Multi-Agent Fairness Testing (`vfairness.multi_agent`)

> Test systems where multiple agents interact. Detect emergent bias that no individual agent has.

### CompositionalityAnalyzer

```python
<!-- cap-status: CompositionalityAnalyzer -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->

from vfairness.multi_agent import CompositionalityAnalyzer

analyzer = CompositionalityAnalyzer()
result = analyzer.analyze(
    component_biases={"agent_A": 0.08, "agent_B": 0.05},
    system_bias=0.25,
    aggregation_method="max",  # or "sum"
    threshold=0.05
)
# result.scenario → "amplification" | "reduction" | "novel_emergence" | "consistent"
# result.divergence → 0.17
```

### GroupthinkDetector

```python
<!-- cap-status: GroupthinkDetector -->
**Beta status: Checked.** 3 code units behind this name: 3 checked
<!-- /cap-status -->

from vfairness.multi_agent import GroupthinkDetector

detector = GroupthinkDetector()
result = detector.analyze_convergence(
    agent_outputs_per_round=[
        {"agent_A": [0.3, 0.5], "agent_B": [0.7, 0.4]},  # Round 1
        {"agent_A": [0.5, 0.5], "agent_B": [0.5, 0.5]},  # Round 2 (converged!)
    ]
)
# result.has_groupthink → bool
# result.p_value → float (permutation test)
# result.coalition_structure → list of sets
```

### EmergentBiasDetector

```python
<!-- cap-status: EmergentBiasDetector -->
**Beta status: Checked.** 2 code units behind this name: 2 checked
<!-- /cap-status -->

from vfairness.multi_agent import EmergentBiasDetector

detector = EmergentBiasDetector()
result = detector.analyze(
    component_outputs={"agent_A": outputs_a, "agent_B": outputs_b},
    system_outputs=system_outputs,
    groups=group_labels  # np.array of 0/1
)
# result.is_emergent → bool (bootstrap-tested)
# result.amplification_factor → float
# result.p_value → float
```

---

## Module 10: Causal Operations (`vfairness.operations.causal`)

> Graph-based causal-inference toolkit for fairness investigations. Operates on a user-supplied causal graph (GML format) plus tabular data, and answers structural questions: is the effect identifiable, how much is mediated, is it robust, what would the counterfactual outcome have been, and which upstream node caused an observed distribution shift. Built on DoWhy / DoWhy GCM.

All public entry points are importable from `vfairness.operations.causal`.

```python
from vfairness.operations.causal import (
    identify_paths, IdentificationResult, PathIdentification,
    decompose_mediation, MediationResult, MediationDecomposition,
    run_refutation_suite, RefutationResult, RefutationOutcome,
    compute_counterfactual, CounterfactualResult,
    attribute_distribution_change, AttributionResult, NodeContribution,
)
```

### identify_paths

Graph-theoretic identification. Determines if each (treatment → outcome) causal effect is identifiable from the data and returns valid adjustment sets. **No estimation** — this answers "can we estimate this effect at all?"
<!-- cap-status: identify_paths -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->


```python
from vfairness.operations.causal import identify_paths

gml = """
graph [ directed 1
  node [ id 0 label "race" ]
  node [ id 1 label "education" ]
  node [ id 2 label "hired" ]
  edge [ source 0 target 1 ]
  edge [ source 0 target 2 ]
  edge [ source 1 target 2 ]
]
"""

result = identify_paths(
    gml=gml,
    treatments=["race"],
    outcomes=["hired"],
    dataset_columns=["race", "education", "hired"],
)

for path in result.paths:
    print(path.is_identifiable, path.backdoor_adjustment_set, path.verdict)
```

**Signature**

```python
identify_paths(
    gml: str,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    dataset_columns: Sequence[str] | None = None,
) -> IdentificationResult
```

**`PathIdentification` fields** — `treatment`, `outcome`, `is_identifiable` (**three states**: `True`, `False` a directed path exists but observational data cannot separate it from confounding, `None` nothing was decided — no directed path in the graph, treatment and outcome the same variable, the graph not a DAG, or identification raised), `backdoor_adjustment_set`, `instruments`, `frontdoor_adjustment_set`, `estimand_expression`, `verdict` (plain-language explanation, one sentence per state), `notes`, `no_causal_path`, `assumes_all_graph_nodes_observed` (True when `dataset_columns` was not given, so every graph node was assumed to be measured — an unobserved confounder is then offered as an adjustment variable; pass `dataset_columns` to have it checked).

### decompose_mediation

Splits a total causal effect into a **natural direct effect** (treatment → outcome holding the mediator fixed) and a **natural indirect effect** (treatment → mediator → outcome). Linear-regression backend.
<!-- cap-status: decompose_mediation -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->


```python
from vfairness.operations.causal import decompose_mediation
import pandas as pd

df = pd.DataFrame({...})  # columns: race, education, hired

result = decompose_mediation(
    gml=gml,
    data=df,
    treatments=["race"],
    outcomes=["hired"],
    mediators=["education"],
)

for d in result.decompositions:
    print(f"direct={d.natural_direct_effect:.3f}  "
          f"indirect={d.natural_indirect_effect:.3f}  "
          f"%mediated={d.proportion_mediated:.1%}")
```

**Signature**

```python
decompose_mediation(
    gml: str,
    data: pd.DataFrame,
    treatments: Sequence[str],
    outcomes: Sequence[str],
    mediators: Iterable[str],
) -> MediationResult
```

**`MediationDecomposition` fields** — `treatment`, `outcome`, `mediator`, `total_effect`, `natural_direct_effect`, `natural_indirect_effect`, `proportion_mediated`, `notes`.

### run_refutation_suite

Robustness battery for a single (treatment, outcome) estimate. Runs four DoWhy refuters — placebo treatment, random common cause, data subset, dummy outcome — and reports the fraction that passed.
<!-- cap-status: run_refutation_suite -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->


```python
from vfairness.operations.causal import run_refutation_suite

result = run_refutation_suite(
    gml=gml,
    data=df,
    treatment="race",
    outcome="hired",
)

# Both are Optional[float]. base_estimate is None when the effect could not be
# estimated at all; robustness_score is None when NO refuter could run, so the
# denominator is zero. result.warnings says which. A refuter that crashed is
# excluded from the denominator rather than counted as a failed refutation.
if result.base_estimate is None:
    print("base estimate: COULD NOT CHECK -", "; ".join(result.warnings))
else:
    print(f"base estimate: {result.base_estimate:.3f}")
if result.robustness_score is None:
    print("robustness: COULD NOT CHECK - no refuter ran")
else:
    print(f"robustness: {result.robustness_score:.1%}")
for o in result.outcomes:
    print(o.test_name, "PASS" if o.passed else "FAIL", o.p_value)
```

**`RefutationOutcome` fields** — `test_name`, `passed`, `observed_effect`, `refuted_effect`, `p_value`, `notes`.

### compute_counterfactual

Individual-level counterfactual via DoWhy GCM. Fits a structural causal model from data, then asks: "for this individual, what would the outcome have been under a different treatment value?"
<!-- cap-status: compute_counterfactual -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->


```python
from vfairness.operations.causal import compute_counterfactual

result = compute_counterfactual(
    gml=gml,
    data=df,
    treatment="race",
    outcome="hired",
    factual={"race": "A", "education": "BSc"},
    intervention_value="B",
    individual_id="applicant_42",
)

print(result.factual_value, "→", result.counterfactual_value, "(effect:", result.effect, ")")
```

**Signature**

```python
compute_counterfactual(
    gml: str,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    factual: dict[str, Any],
    intervention_value: Any,
    individual_id: str | None = None,
) -> CounterfactualResult
```

Assumes an additive-noise mechanism. Use the refutation suite to sanity-check the underlying estimate before trusting individual counterfactuals.

### attribute_distribution_change

Ranks upstream nodes by their share of an observed outcome shift between two snapshots (e.g. baseline vs current month). Backed by DoWhy GCM `distribution_change`. Answers: "which input caused the drift?"
<!-- cap-status: attribute_distribution_change -->
**Beta status: Checked.** refuses honestly when nothing is measurable, and a test fails if that regresses **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->


```python
from vfairness.operations.causal import attribute_distribution_change

result = attribute_distribution_change(
    gml=gml,
    baseline=df_q1,
    current=df_q2,
    outcome="hired",
)

for c in result.contributions:           # sorted by |share| descending
    print(f"{c.node}: {c.share:+.1%}")
```

**`NodeContribution` fields** — `node`, `share` (normalized, sums to ~1.0 in absolute value **by construction**, so read it beside the fields below), `contribution` (the unnormalized value, in the units of `outcome_change`).

**`AttributionResult` fields** — `outcome`, `contributions`, `warnings`, `outcome_change` (the measured change in the outcome's marginal distribution that the shares apportion), `outcome_change_p_value` / `outcome_change_test` (the two-sample test for "it changed at all": chi-square on level counts for labels, Kolmogorov-Smirnov for quantities), `not_assessable`. **No share is published unless the change itself was measured**: an empty `contributions` with a non-empty `not_assessable` means nothing was attributed, which is not the finding that no node contributed.

### Task Handlers

For task-queue consumers, each operation has a `handle_*` wrapper that takes a JSON payload and returns a `{success, data | error}` envelope:

```python
from vfairness.operations.causal import (
    handle_identify, handle_mediate, handle_refute,
    handle_counterfactual, handle_attribute,
)

response = handle_identify({
    "serialized": {"gml": gml_string},
    "treatments": ["race"],
    "outcomes": ["hired"],
    "dataset_columns": ["race", "education", "hired"],
})
# {"success": True, "data": {...IdentificationResult.to_dict()...}}
```

Payloads accept either inline data (`inline_csv` / `inline_json`) or pre-loaded `dataset_bytes`. See `task_handlers.py` for field details.


### `CausalFairnessGraph`

A lightweight, networkx-only directed acyclic graph that declares the
<!-- cap-status: CausalFairnessGraph -->
**Beta status: Checked.** 10 code units behind this name: 10 checked **Preview:** documented and usable, and not yet fully covered and checked by the beta verification programme.
<!-- /cap-status -->

cause-and-effect structure of a system and classifies each path from a
protected attribute to an outcome as DIRECT, INDIRECT, or PROXY discrimination.
It is the graph-theoretic complement to the DoWhy-backed estimation entry points
above (`identify_paths`, `decompose_mediation`, `run_refutation_suite`,
`compute_counterfactual`, `attribute_distribution_change`), and needs only
networkx, so a graph can be authored and reasoned about even where DoWhy is
unavailable.

```python
from vfairness.operations.causal.graph import CausalFairnessGraph

g = CausalFairnessGraph()
g.add_variable("gender", protected=True)
g.add_variable("education", mediator=True)
g.add_variable("hiring", outcome=True)
g.add_edge("gender", "education")   # historical effect
g.add_edge("education", "hiring")   # legitimate mediator
g.add_edge("gender", "hiring")      # direct discrimination

for path in g.discrimination_paths():
    print(path.kind, path.path, path.verdict)

print(g.summary())     # counts by kind + overall severity
gml = g.to_gml()       # serialise for the estimation handlers / frontend
```

**Key methods:**

- `add_variable(name, protected=False, outcome=False, mediator=False, proxy=False)`
- `add_edge(src, dst)` (rejects edges that would create a cycle)
- `discrimination_paths() -> list[DiscriminationPath]`
- `summary() -> dict` (`n_paths`, `counts` by kind, `severity`, `has_direct_discrimination`, `paths`)
- `has_direct_discrimination() -> bool`
- `to_gml() -> str`

Each `DiscriminationPath` carries `source`, `target`, `path` (the full node
list), `kind` (`"direct"`, `"indirect"`, or `"proxy"`), `mediators`, and a
plain-language `verdict`. A path with no intervening variable is DIRECT; a path
through a variable flagged `proxy=True` is PROXY; any other mediated path is
INDIRECT.

### Dispatchable causal task types

The DoWhy-backed ops are also reachable from the platform task consumer via
`operations/causal/task_handlers.py`. Each handler reads a serialized GML graph
plus an optional dataset and returns a `{success, data}` envelope.

| Task type | Handler | Purpose |
| --- | --- | --- |
| `vfairness_causal_identify` | `handle_identify` | Identify causal paths (graph only, no dataset). |
| `vfairness_causal_mediate` | `handle_mediate` | Mediation decomposition (NDE/NIE: direct vs indirect). |
| `vfairness_causal_refute` | `handle_refute` | Refutation suite (placebo, random common cause, subset, dummy outcome). |
| `vfairness_causal_counterfactual` | `handle_counterfactual` | Individual-level counterfactual via a fitted SCM (GCM). |
| `vfairness_causal_attribute` | `handle_attribute` | Attribute an outcome distribution change / drift to upstream variables. |

---

## XAI / Explainability Module

Added in v0.0.8 (see CHANGELOG.md). Powers the validant.ai platform's XAI peer view; the platform-internal XAI architecture doc (reader access required) describes the database scaffolding and the React contracts this module mirrors.

### Module layout

```
vfairness/xai/
  schemas.py             # Python dataclass mirrors of the TS contracts
  sidecar_cli.py         # `python -m vfairness.xai.sidecar_cli`: JSON in, JSON out
  explainers/
    base.py              # Explainer ABC + ExplainerCapabilities
    router.py            # route_explainer() -- mechanical, mirrors TS + Postgres
    registry.py          # get_explainer() / available_methods()
    shap_adapter.py      # TreeShap / LinearShap / KernelShap
    lime_adapter.py      # LimeExplainer
    dice_adapter.py      # DiceCounterfactualExplainer
    ig_adapter.py        # IntegratedGradientsExplainer
    anchors_adapter.py   # AnchorsExplainer
  decomposition/
    shap_fairness.py     # Lundberg decomposition + 1e-6 identity guard + proxy_score
  diagnostics/
    faithfulness.py      # removal_curve_auc, local_r_squared
    stability.py         # attribution_stability (σ over reruns)
    adversarial.py       # slack_adversarial_probe
  storage/
    supabase_writer.py   # RLS-bypass service-role inserts
    audit_artifact.py    # Deterministic JSON bundle + sha256
  worker/
    runner.py            # Signal-safe pgmq polling loop
```

### Frozen contracts

The dataclasses in `vfairness.xai.schemas` are the Python mirrors of the TypeScript XAI contracts on the platform side. Nested JSONB payloads (attribution arrays, counterfactual flipped features) are serialised in camelCase by `to_db_row()` so the React reader needs no transformer layer.

| Dataclass | Mirrors TS | Notes |
| --- | --- | --- |
| `Explanation` | `Explanation` (Part 5.1) | `to_db_row(owner)` returns a `dict` ready for the `xai_explanations` insert <!--cs-->**[Checked]**<!--/cs--> |
| `CounterfactualExplanation` | `CounterfactualExplanation` | Override of `to_db_row()` camelCases nested `flippedFeatures` <!--cs-->**[Checked]**<!--/cs--> |
| `FairnessDecomposition` | `FairnessDecomposition` (Part 5.3) | `to_db_row()` re-asserts the 1e-6 Lundberg identity BEFORE returning <!--cs-->**[Checked]**<!--/cs--> |
| `XaiAssessment` | `XaiAssessment` (Part 5.2) | `diagnostics` is an `XaiDiagnostics` dataclass with faithfulness/stability/adversarial fields <!--cs-->**[Checked]**<!--/cs--> |
| `TrustPosture` | `TrustPosture` (Part 5.4) | Synthesis-layer output for the regulator report <!--cs-->**[Checked]**<!--/cs--> |
| `Recommendation` | `Recommendation` | What the Advisor commits; carries `kg_citations` + `kg_evidence_class` <!--cs-->**[Checked]**<!--/cs--> |
### Routing -- `route_explainer`

```python
from vfairness.xai import route_explainer

decision = route_explainer(
    model_type="tree",           # 'tree' | 'linear' | 'deep' | 'blackbox'
    goal="proxy_diagnosis",      # 'single_decision' | 'global_understanding' | 'proxy_diagnosis' | 'actionable_recourse' | 'rule_extraction'
    counterfactual_needed=False,
    background_available=True,
    ood_risk="low",              # 'low' | 'moderate' | 'high'
)
# RouteDecision(primary, component_id, fallbacks, downgraded_from, reason)
```

Returns a `RouteDecision` dataclass. Tested for byte-for-byte agreement with the Postgres `xai_recommend()` function and the TS `recommend()` engine via the `tests/xai/test_router.py` fixture. Any drift is a contract-test failure.

The Python router takes the routing facts as keyword-only primitives rather than a profile object; the TS-side `DataProfile` / `UseCaseProfile` are flattened into `model_type`, `goal`, `counterfactual_needed`, `background_available` and `ood_risk`.

### Explainers

All adapters subclass `Explainer` (ABC) and expose `supports(model, model_type) -> bool` plus `explain_local(...)` and `explain_global(...)`. Both methods return an `Explanation` (or `list[Explanation]`) populated with the library version captured at call time.

| Class | `capabilities.method` | Sync/Async | Requires background |
| --- | --- | --- | --- |
| `TreeShapExplainer` | `shap.TreeExplainer` | sync, exact | no <!--cs-->**[Checked]**<!--/cs--> |
| `LinearShapExplainer` | `shap.LinearExplainer` | sync, closed-form | yes <!--cs-->**[Checked]**<!--/cs--> |
| `KernelShapExplainer` | `shap.KernelExplainer` | async-eligible | yes <!--cs-->**[Checked]**<!--/cs--> |
| `LimeExplainer` | `lime` | async-eligible | yes <!--cs-->**[Checked]**<!--/cs--> |
| `DiceCounterfactualExplainer` | `dice` | async-eligible | yes <!--cs-->**[Checked]**<!--/cs--> |
| `IntegratedGradientsExplainer` | `ig` | async-eligible | yes <!--cs-->**[Checked]**<!--/cs--> |
| `AnchorsExplainer` | `anchors` | async-eligible | yes <!--cs-->**[Checked]**<!--/cs--> |
Use `get_explainer(method)` to obtain a ready adapter INSTANCE by its method string (it calls the class for you, so do not call the return value), and `available_methods()` to list the method strings that have a runnable adapter. Each adapter defers its heavy optional import (`shap`, `dice-ml`, `lime`, `captum` + `torch`, `anchor-exp`) to `__init__`, so the registry module builds without any of those extras installed and the `ImportError` is raised by `get_explainer()` itself, at the moment it instantiates. `DiceCounterfactualExplainer` and `AnchorsExplainer` are local-only: their `capabilities.supports_global` is `False` and `explain_global()` raises `NotImplementedError`.

### Decomposition -- `lundberg_fairness_decomposition`

```python
from vfairness.xai import lundberg_fairness_decomposition

decomposition = lundberg_fairness_decomposition(
    shap_values=shap_matrix,           # (n_samples, n_features) -- MUST be in the metric's units
    group_labels=protected_attribute,
    feature_names=feature_names,
    metric="demographic_parity",       # one of FairnessMetric
    protected_attribute="gender",
    subject_id="...",
    audit_artifact_id="...",
    proxy_threshold=0.15,              # default per Part 7.4
)
# FairnessDecomposition(total_disparity, per_feature, proxy_scores, flagged_proxies, ...)
```

The 1e-6 identity (`|sum(per_feature) − total_disparity| ≤ 1e-6`) is asserted inside the function; a unit mismatch (SHAP in probability vs metric in log-odds) raises `ValueError` immediately. The same assertion runs again inside `decomposition.to_db_row()` so a corrupt run cannot reach the database.

### Diagnostics

| Function | Purpose | Reference |
| --- | --- | --- |
| `removal_curve_auc(predict_fn, x, attributions, background_mean)` | ROAR-style removal-curve faithfulness | Hooker, Erhan, Kindermans & Kim (2019) NeurIPS |
| `local_r_squared(surrogate_predictions, true_predictions)` | LIME surrogate fidelity (R²) | Ribeiro, Singh, Guestrin (2016) KDD |
| `attribution_stability(reruns)` | Mean σ across explainer reruns with different seeds | Alvarez-Melis & Jaakkola (2018) ICML WHI |
| `slack_adversarial_probe(predict_fn, x, background)` | Slack et al. (2020) OOD-scaffolding probe | Slack, Hilgard, Jia, Singh, Lakkaraju (2020) AIES |

### Storage + worker

```python
from vfairness.xai.storage import SupabaseWriter, build_audit_artifact_bundle
from vfairness.xai.worker import WorkerLoop

writer = SupabaseWriter()  # reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from env
writer.write_audit_artifact(...)
writer.write_explanations(owner=auth0_sub, explanations=[...])
writer.write_fairness_decomposition(owner=auth0_sub, decomposition=...)

# Long-lived pgmq consumer (deploy as a systemd service on the worker host).
# Reads SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, PGMQ_QUEUE (default: xai_jobs),
# WORKER_POLL_INTERVAL_S (default 5.0), WORKER_CONCURRENCY (default 1) from env.
WorkerLoop().run()  # config defaults to WorkerConfig.from_env()
```

The worker writes status updates into `xai_jobs` so the platform frontend's Realtime subscription sees live progress. `WorkerLoop._execute()` raises `NotImplementedError` at the model-load seam where ops wires the existing pulse-artifact loader.

### Tests

`tests/xai/` is the contract suite against the platform TS + Postgres: golden router cases locked byte-for-byte to `xai_recommend()`, decomposition cases including the identity-break on unit mismatch, the schema / camelCase-JSONB contract (with the regression test from the v0.0.5 review pass), the adversarial probe, and per-adapter tests for the LIME, DiCE, Integrated Gradients and Anchors explainers plus the registry. The SHAP adapters have no dedicated per-adapter test in `tests/xai/`; `test_registry.py` only asserts that the `shap.TreeExplainer` method string resolves. Any drift is a contract-test failure.

---

## Network Egress Guard (`vfairness.net`)

Cross-cutting infrastructure rather than a fairness-analysis surface. `LLMApiProxy`
applies this guard at construction time, so a user-supplied endpoint URL handed to
it cannot be pointed at loopback, RFC1918, link-local or cloud-metadata addresses.

Coverage IS universal inside the library as of 2026-09-07, and this paragraph
used to say the opposite. It read that `LLMApiProxy` was the only wired call site,
that `guarded_post` had no internal callers, and that the LLM judge scorer, the
groundedness judge and the XAI sidecar client still posted to a caller-supplied
URL directly. All three of those were wired in the meantime and the sentence was
never updated. Measured at HEAD: `guarded_post` has 9 call sites, `llm/scorers.py`,
`xai/sidecar_cli.py` and `validity/judge.py` each route through it or through a
redirect-revalidating `GuardedSession`, and `grep -rn "requests.post(" src/`
returns NOTHING outside the guard itself.

It is worth being exact about which direction this was wrong in: the library was
SAFER than its own documentation claimed. That is the less dangerous error, but
it is still an error, and a reader who believed it would either write a guard
they did not need or trust the library less than the code earns.

If you accept an endpoint URL from an untrusted source in YOUR OWN code, call
`validate_endpoint()` or send through `guarded_post()`; a loopback address is
refused unless you pass `allow_loopback=True`, which is the opt-in for a model
server on the same machine.

```python
from vfairness.net import validate_endpoint, guarded_post, PinnedIPAdapter, SSRFError

host, port, vetted_ips = validate_endpoint("https://api.example.com/v1/chat")

response = guarded_post(
    "https://api.example.com/v1/chat",
    json={"prompt": "..."},
    headers={"Authorization": "Bearer ..."},
    timeout=30,
)
```

| Symbol | Signature | Behaviour |
| --- | --- | --- |
| `validate_endpoint` | `(url, allow_http=False, allow_loopback=False)` | Returns `(host, port, vetted_ips)`, or raises `SSRFError` <!--cs-->**[Checked]**<!--/cs--> |
| `guarded_post` | `(url, *, allow_http=False, allow_loopback=False, **kwargs)` | Validates, then POSTs through a session pinned to the vetted IP; other kwargs (`json`, `headers`, `timeout`, ...) pass straight through to `requests` <!--cs-->**[Checked]**<!--/cs--> |
| `PinnedIPAdapter` | `requests` `HTTPAdapter` subclass | Dials the vetted IP while keeping the original hostname for SNI, certificate verification and the `Host` header, so a second DNS answer cannot rebind the target between check and connect <!--cs-->**[Checked]**<!--/cs--> |
| `SSRFError` | exception | Raised for a disallowed scheme, a host that does not resolve, or a resolved address that is not public <!--cs-->**[Checked]**<!--/cs--> |
What the guard enforces:

- `https` only, unless the caller passes `allow_http=True`.
- RFC1918, link-local, unique-local, reserved, unspecified and multicast targets
  are refused, and the cloud-metadata addresses (169.254.169.254 and its IPv6 and
  IPv4-mapped forms) are refused whatever the caller asks for.
- Loopback (127.0.0.0/8 and `::1`) is refused unless the caller passes
  `allow_loopback=True`, the explicit opt-in for a model server on this machine.
  That opt-in unlocks loopback and nothing else.
- IPv4-mapped IPv6 is unwrapped before the check, so `::ffff:127.0.0.1` cannot
  smuggle a loopback address past it.
- The decision is all-or-nothing: a host whose resolved set mixes public and
  non-public addresses is refused, not partially used.

`LLMApiProxy` runs `validate_endpoint` in its constructor and re-raises an
`SSRFError` as `ValueError: endpoint_url refused by the egress guard: ...`.

---

## Need Help?

- **Source code**: Check the docstrings in `src/vfairness/`
- **Tests**: See `tests/` for more usage examples
- **Visual Overview**: See [LIBRARY_OVERVIEW.md](LIBRARY_OVERVIEW.md) for architecture diagrams and feature maps
- **Notebooks**: the numbered demo notebooks in [notebooks/](../notebooks/) walk the pipeline end to end, from [vfairness_0_library_validation.ipynb](../notebooks/vfairness_0_library_validation.ipynb) and [vfairness_1_bias_detection_demo.ipynb](../notebooks/vfairness_1_bias_detection_demo.ipynb) through [vfairness_4_metrics_demo.ipynb](../notebooks/vfairness_4_metrics_demo.ipynb) and beyond
- **Issues**: Report bugs or request features on GitHub

---

## Validity / Groundedness Module (`vfairness.validity`)

The validity axis for generative and RAG systems. Fairness asks "is the quality
evenly distributed across groups"; validity asks the prior question "is the answer
actually right, that is, grounded in and faithful to its retrieved sources". A
system can be evenly fair and still confidently wrong, so a generative assessment
needs both axes. This module produces the groundedness scores; the platform wires,
gates, and seals them behind its own validity metric contract. The full design
and the licence-clearance register are internal and are not published with the
library; [`docs/ROADMAP.md`](ROADMAP.md) (Implementation Plan) carries the
milestone state, and the VG table below is the published contract.

### The VG metric family

The scorer targets the frozen `VG-*` ids that also live in the platform's
validity metric contract. Do not renumber them.

| id | metric | direction | needs | tier |
| --- | --- | --- | --- | --- |
| VG-001 | Groundedness (fraction of answer claims supported by context) | higher better | context | interim + owned |
| VG-002 | Faithfulness | higher better | context | interim (aliased to VG-001) then owned |
| VG-003 | Context precision | higher better | context + question | owned |
| VG-004 | Context recall | higher better | gold | owned (P3) -- NOT implemented; `gold` is refused, see `GroundednessScorer` below |
| VG-005 | Hallucination rate (= 1 - groundedness) | lower better | context | interim + owned |
| VG-006 | Citation accuracy (span level) | higher better | span scorer | owned (P3) |
| VG-007 | Answer correctness | higher better | gold + judge | owned (P3) -- NOT implemented; `gold` is refused, see `GroundednessScorer` below |

### The fail-closed contract (the load-bearing rule)

The scorer follows a degradation ladder with a HARD floor:

```
owned sidecar detector (VA-21)  ->  self-hosted LLM judge (VA-10)  ->  REFUSE
```

It NEVER falls to a lexical placeholder in a shipped grade. When no rung is wired,
or a wired rung fails (network, HTTP, unparseable reply), `score()` returns a
`GroundednessResult` with `available=False` and `value=None` (never `0.0`), and
warns once (`ValidityScorerUnavailableWarning`). This mirrors the
`SidecarUnavailableWarning` "zeros are not real scores" contract in
`vfairness.llm.scorers`: a missing scorer must read as "not measured", never as
"grounded / clean". Downstream, the platform keeps such a result in the honest
`metricsDeferred` channel and `feedsGrade` stays false until a scorer is available
AND the assessed language is validated.

### Classes

- **`GroundednessScorer(judge=None, sidecar=None)`**: the ladder. `.available`
  reports whether any rung is wired. `score(answer, contexts, *, question=None,
  gold=None, language="unknown") -> GroundednessResult`. **`gold` is NOT
  implemented and is refused, not honoured** (2026-09-10): no rung computes
  VG-004 or VG-007, so a reference answer is never read. Supplying one emits
  `GoldUnsupportedWarning` and stamps the refusal into `result.note` (which
  travels into the task output below); the groundedness score itself is
  unaffected. It warns rather than raises because the record envelope below
  accepts an optional `reference_answer` and forwards it here, and a raise
  would take the whole batch's VG-001 measurement down with it. Promote it
  with `warnings.simplefilter("error", GoldUnsupportedWarning)`.
- **`GroundednessResult`**: a context-aware per-answer verdict (`SerializableMixin`,
  so `.to_dict()` / `.to_json()`). Key fields: `value` (VG-001, `None` = not
  measured), `hallucination_rate` (VG-005), `faithfulness` (VG-002),
  `context_precision` (VG-003), `unsupported_spans`, `supported`, `language`,
  `scorer_tier`, `model_id`, `available`, `note`.
- **`LlmGroundednessJudge(endpoint_url, model_name, api_format="openai",
  auth_token=None, timeout=60)`**: the interim (VA-10) judge rung. Calls a
  self-hosted, OpenAI-compatible endpoint (an Ollama host) directly with
  `requests`, using a SNAPSHOTTED, versioned groundedness rubric (`PROMPT_VERSION`).
  Returns `groundedness=None` (REFUSE) on any failure, never a neutral score. The
  chosen judge is **Mistral Small 3.2** (European, Apache-2.0), with Qwen3 as the
  licence-clean fallback and **Apertus** the planned successor once it is runnable
  in the self-hosted stack.
- **`aggregate_validity(results) -> dict`**: a fail-closed VG-* batch summary.
  Only records with a real measurement are aggregated; if none are measured the
  summary reports `available=False` with no numbers. Confidence intervals come from
  `evaluation/vfairness_metrics/_statistics.py` when available, and are omitted
  (never faked) otherwise.
- **`groundedness_scorer_status(scorer) -> dict`**: active rung + quality tier.

### Task type: `vfairness_validity_run`

Dispatched like the other operations handlers, over stdin with a JSON envelope on
stdout:

```bash
echo '{"records":[{"prompt":"When is the deadline?","answer":"30 June.",
  "retrieved_context":["The deadline is 30 June."]}],"language":"en",
  "judge":{"endpoint_url":"http://<ollama-host>:11435/v1/chat/completions",
           "model_name":"mistral-small3.2"}}' \
| python -m vfairness.operations.validity.task_handlers vfairness_validity_run
```

Payload: `records` is a list of eval-set records (`prompt`, `answer`,
`retrieved_context` as strings or `{text}` objects, optional `reference_answer`,
`language`). The judge is built from the payload `judge` block or the
`VFAIRNESS_VALIDITY_JUDGE_URL` / `_MODEL` / `_FORMAT` / `_TOKEN` env vars. With no
judge configured, the handler still returns `success: true` but every record is
`available: false` and the aggregate is `available: false` (fail-closed).

### Python example

```python
from vfairness.validity import GroundednessScorer, LlmGroundednessJudge

judge = LlmGroundednessJudge(
    endpoint_url="http://<ollama-host>:11435/v1/chat/completions",
    model_name="mistral-small3.2",
)
scorer = GroundednessScorer(judge=judge)          # or GroundednessScorer() -> refuses
result = scorer.score(
    answer="The deadline is 30 June.",
    contexts=["The application deadline is 30 June 2026."],
    question="When is the deadline?",
    language="en",
)
if result.available:
    print(result.value, result.hallucination_rate, result.unsupported_spans)
else:
    print("not measured:", result.note)           # fail-closed, never a fake number
```
