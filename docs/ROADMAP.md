# vfairness Development Roadmap

## Closing the Gaps: From Comprehensive to Complete

This roadmap identifies every gap in vfairness coverage based on the Turing AI Fairness course mapping and outlines the optimal approach to close each one. The goal is to create the most compelling, holistic AI fairness library available.

---

## Current Architecture (Capability Inventory)

The library has grown well beyond the handful of areas this roadmap originally tracked. The authoritative inventory (source-file, class, function and registered-capability counts) is regenerated into `vfairness-manifest.json` on every build; read it there rather than from this page. It is organized into fifteen top-level subpackages under `src/vfairness/` (`validity/` and `net/` are the newest; `validity/` is intentionally not yet counted in the registered-capability total, see below):

1. `evaluation/vfairness_metrics/` (SHIPPED): the core fairness metrics suite. Classification (demographic parity, equalized odds, equal opportunity, calibration, disparate-impact ratio, AUROC parity), regression (error parity), ranking (exposure / relevance / NDCG parity), intersectional, robustness / sensitivity, attribution, a counterfactual metric, automatic bias discovery, statistics (confidence and Bayesian intervals, Benjamini-Hochberg multiple-comparison correction), explanation diagnostics, and structured reports. A unified `FairnessAnalyzer` is the entry point.
2. `preprocessing/` (SHIPPED): `bias_detection` (BiasDetector, GeographicRiskAssessment for HOLC redlining, proxy detection via Cramer's V, historical-pattern / representation / statistical-disparity results) and `feature_engineering` (CorrelationReducer, FeatureSuppressor, ResidualTransformer, IntersectionalTransformer, ReweightingTransformer, DisparateImpactRemover (Feldman et al.), LabelMassager, Resampler with SMOTE, FairRepresentationTransformer).
3. `in_processing/` (SHIPPED): training-time interventions. Loss functions (DemographicParityLoss, EqualizedOddsLoss, EqualOpportunityLoss, AdversarialDebiasingLoss), constraints (ExponentiatedGradient, GridSearch, ThresholdOptimizer), regularizers (StatisticalParityRegularizer, HilbertSchmidtRegularizer, GroupFairnessRegularizer, ConditionalIndependenceRegularizer), and sklearn-compatible wrappers (FairClassifier, FairRegressor, with make_fair_classifier / make_fair_regressor factories).
4. `post_processing/` (SHIPPED): GroupCalibrator (Platt / Isotonic / Beta / Temperature, with ECE / MCE / Brier), threshold optimization (Single and Group-Specific optimizers, Pareto boundary), and output reweighting.
5. `llm/` (SHIPPED): CounterfactualTester, OutputAnalyzer, NonDeterminismAnalyzer, LLMApiProxy, BenchmarkRunner (BBQ / BOLD / HolisticBias, curated subsets), DecodingTrustRunner, IntersectionalAnalyzer, CoTFaithfulnessAnalyzer, EmbeddingBiasDetector (WEAT / SEAT), TextFairnessAnalyzer.
6. `agents/` (SHIPPED): CorrespondenceTester, ToolBiasAuditor, RAGBiasAnalyzer, PipelineTracker, TemporalTracker, ActionBiasAnalyzer.
7. `multi_agent/` (SHIPPED): CompositionalityAnalyzer, GroupthinkDetector, EmergentBiasDetector, DelegationRoutingAuditor, AdversarialCollusionDetector, NegotiationFairnessTracker, MultiAgentRunHarness.
8. `xai/` (SHIPPED, a major new subsystem absent from the earlier roadmap): explainer adapters (TreeSHAP, LinearSHAP, KernelSHAP, LIME, DiCE counterfactuals, Integrated Gradients, Anchors) behind a `route_explainer()` router; diagnostics (faithfulness, stability, adversarial probing); `lundberg_fairness_decomposition()` (per-feature attribution of group-fairness metrics, which asserts a 1e-6 additive identity before persisting); a Supabase storage writer; and a long-lived pgmq worker loop.
9. `operations/` (SHIPPED): `causal` (CausalFairnessGraph plus identify / mediate (NDE, NIE) / refute / counterfactual_fairness / attribute via DoWhy), `cicd` (DataBiasValidator, ModelFairnessGate, pre-commit hook, pytest plugin), `monitoring` (FairnessMonitor, FairnessDriftDetector with CUSUM / EWMA, TemporalFairnessAnalyzer), `experimentation` (FairnessExperiment for A/B testing, power analysis), `pulse` (orchestrator, pipeline runner, agent / LLM / vision probes, Pareto mitigation recommender), and `reporting` (reports, dashboards, compliance docs).
10. `rendering/` (SHIPPED): data-driven SVG report and dashboard generation via Jinja2 templates.
11. `legal/` (SHIPPED): hand-curated admissibility rule packs mapping (use case x jurisdiction) to per-column admissibility verdicts (lending, employment, housing, public benefits). No LLM runs at runtime, and it reports "uncovered" rather than fabricating a verdict when no pack exists.
12. `vision/` (PARTIAL): image-fairness math in pure numpy (MaxSkew, MinSkew, NDKL (Geyik et al. 2019), bias amplification (Seshadri et al. 2023)). Optional FairFace / CLIP demographic classification runs in a separate Python 3.9 vision sidecar.
13. `validity/` (LIVE, fail-closed): the second assurance axis. Where fairness asks whether a system treats groups equitably, validity asks whether a generative or retrieval-augmented answer is grounded in its sources. `GroundednessScorer` runs a degrading ladder (owned detector -> self-hosted LLM judge -> REFUSE) that never fabricates a score, `aggregate_validity()` summarises the frozen VG-* metric family, and `operations/validity/task_handlers.py` dispatches the `vfairness_validity_run` task type. The interim judge rung (`LlmGroundednessJudge`, Mistral Small 3.2, Apertus the planned successor) is DEPLOYED and scores real groundedness end to end; the owned detector (VA-21) is planned. It is deliberately NOT registered in `CAPABILITY_REGISTRY` or the synced manifest (so the registered-capability total in the manifest excludes it) until it is VALIDATED against a per-language gold set (it currently feeds the grade only PROVISIONALLY, via the interim judge). See Gap 7.
14. `mcp/` (SHIPPED): an MCP server exposing the fairness surface as tools. The pure tool logic (`mcp/tools.py`) carries no `mcp` dependency and is importable on its own; `mcp/server.py` wraps it with FastMCP behind the optional `mcp` extra, run as `vfairness-mcp` or `python -m vfairness.mcp`.
15. `net/` (SHIPPED): the cross-cutting SSRF egress guard (`validate_endpoint`, `guarded_post`, `PinnedIPAdapter`, `SSRFError`). Every caller-supplied endpoint URL that carries a credential is vetted before any outbound call and the connection is pinned to the vetted IP, with every redirect hop re-vetted; `allow_loopback=True` is the only opt-in, and it unlocks loopback alone (RFC1918, link-local and cloud-metadata addresses stay refused). Wired at: `LLMApiProxy`, `validity.judge`, `llm.scorers.LLMJudgeScorer` and `xai.sidecar_cli` (the last two since 2026-08-27, VF-2). **Not universal, and this sentence used to claim it was:** `operations.pulse.task_handlers._download_bytes` fetches a first-party signed artifact URL with its own defences (https pinned, certifi CA, size cap) but no SSRF guard, and `preprocessing.bias_detection.geographic_data` fetches a fixed public API whose URL is built from a module constant and a whitelisted city id, not from caller input.

---

## Executive Summary

| Category | Current Status | Gaps Identified | Priority |
|----------|---------------|-----------------|----------|
| **Causal Fairness** | Substantially implemented | Pathway classification and DoWhy estimation done; standalone direct/indirect effect helpers and the SVG template remain | 🟡 Medium |
| **LLM Fairness** | **Complete** | `vfairness.llm` (EmbeddingBiasDetector and TextFairnessAnalyzer now shipped) | ✅ Done |
| **Real-time Monitoring** | **Complete** | Monitoring, drift, alerting | ✅ Done |
| **A/B Testing** | **Shipped** | FairnessExperiment plus `operations/experimentation` (power analysis) | ✅ Done |
| **Explainability / XAI** | **Shipped** | `vfairness.xai`: SHAP/LIME/DiCE/IG/Anchors, diagnostics, fairness decomposition | ✅ Done |
| **Causal estimation (DoWhy)** | **Shipped** | identify / mediate / refute / counterfactual / attribute end to end | ✅ Done |
| **In-processing interventions** | **Shipped** | `vfairness.in_processing`: fair losses, constraints, regularizers, wrappers | ✅ Done |
| **Post-processing interventions** | **Shipped** | `vfairness.post_processing`: calibration, threshold optimization, reweighting | ✅ Done |
| **Preprocessing / feature engineering** | **Shipped** | `vfairness.preprocessing`: bias detection plus transformers | ✅ Done |
| **CI/CD fairness gates** | **Shipped** | `operations/cicd`: validators, model gate, pre-commit hook, pytest plugin | ✅ Done |
| **Pulse orchestrator** | **Shipped** | `operations/pulse`: end-to-end auditing orchestrator and probes | ✅ Done |
| **Legal admissibility** | **Shipped** | `vfairness.legal`: curated (use case x jurisdiction) rule packs | ✅ Done |
| **Validity / Groundedness** | Interim (fail-closed) | `vfairness.validity`: VG-* metrics, judge rung landed (VA-10); owned detector (VA-21) and grade-feed (M1 + gold) pending | 🔴 Active |
| **Agile Integration** | Not started | SAFE / FAIR framework templates | 🟡 Medium |
| **Exposure Parity** | Partial | Full ranking fairness framework | 🟡 Medium |
| **Multi-modal Systems** | Partial | Image fairness shipped; audio and multimodal fusion remain | 🟠 Future |

---

## Gap 1: Causal Fairness Framework

**Status:** Substantially implemented (Phase 1 shipped).

### Current State
- ✅ Proxy variable detection via correlation analysis
- ✅ Feature correlation heatmaps
- ✅ Causal graph specification: `CausalFairnessGraph` (`operations/causal/graph.py`) with direct / indirect / proxy pathway classification
- ✅ Counterfactual fairness metric: `counterfactual_fairness()` (`evaluation/vfairness_metrics/counterfactual_metric.py`)
- ✅ Formal causal estimation: DoWhy-backed handlers (identify, mediate, refute, counterfactual, attribute) reachable end to end via the task consumer

### What's Missing
The course covers **Causal Approaches to Fairness** (Module 2, Sprint 1) including:
- Directed Acyclic Graphs (DAGs) for bias mapping
- Counterfactual fairness definitions
- Path-specific effects (direct vs. indirect discrimination)
- Intervention vs. observation distinction

### Implementation Plan

#### Phase 1: Causal Graph Infrastructure

**Status:** Shipped as `CausalFairnessGraph` in `vfairness.operations.causal.graph`
(not `vfairness.preprocessing` as originally sketched). The shipped API:

```python
from vfairness.operations.causal.graph import CausalFairnessGraph

graph = CausalFairnessGraph()
graph.add_variable('gender', protected=True)
graph.add_variable('education', mediator=True)
graph.add_variable('hiring_decision', outcome=True)

graph.add_edge('gender', 'education')           # historical effect
graph.add_edge('education', 'hiring_decision')  # legitimate path
graph.add_edge('gender', 'hiring_decision')     # direct discrimination

for path in graph.discrimination_paths():
    print(path.kind, path.path)
# direct ['gender', 'hiring_decision']
# indirect ['gender', 'education', 'hiring_decision']
```

#### Phase 2: Counterfactual Fairness Metrics

**Status:** Shipped as `counterfactual_fairness()` in
`vfairness.evaluation.vfairness_metrics.counterfactual_metric` (it is not
re-exported from `vfairness.evaluation`). It compares factual against
caller-supplied counterfactual predictions; for the full structural-causal-model
path, generate the counterfactuals with the DoWhy-backed `compute_counterfactual`
op first. The shipped API:

```python
from vfairness.evaluation.vfairness_metrics.counterfactual_metric import (
    counterfactual_fairness,
)

# Factual scores vs. scores after flipping the protected attribute
result = counterfactual_fairness(
    y_pred_factual=[0.81, 0.34, 0.65, 0.12],
    y_pred_counterfactual=[0.42, 0.31, 0.66, 0.10],
    threshold=0.5,
)
print(result.flip_rate, round(result.mean_abs_diff, 4))
# 0.25 0.1125
```

#### Phase 3: Path-Specific Effect Analysis

**Status:** Not shipped as sketched. The shipped path-specific decomposition is
the mediation route (NDE / NIE) through the DoWhy-backed causal estimation
handlers (see the deliverables below); a standalone `PathAnalyzer` class does not
exist, and the standalone direct / indirect effect helpers remain open. The
sketch below is the original target API, kept for reference:

```python
# Target API sketch (not implemented)
from vfairness.preprocessing import PathAnalyzer

analyzer = PathAnalyzer(graph)

# Decompose total effect
effects = analyzer.decompose_effect(
    treatment='gender',
    outcome='hiring_decision'
)
# Returns: {
#   'direct_effect': 0.15,
#   'indirect_effect_via_education': 0.08,
#   'total_effect': 0.23
# }

# Identify problematic paths
problematic = analyzer.identify_unfair_paths(threshold=0.1)
```

#### Technical Approach
1. **Lightweight DAG library**: Use `networkx` for graph operations (already common dependency)
2. **Integration with DoWhy**: Optional integration with Microsoft's DoWhy for advanced causal inference
3. **Visualization**: SVG templates for causal graph visualization

#### Deliverables
- [x] `CausalFairnessGraph` class with DAG operations (direct / indirect / proxy pathway classification)
- [x] `counterfactual_fairness()` metric function
- [x] Path-specific effects: mediation decomposition (NDE/NIE) via the causal estimation handlers
- [ ] Standalone `direct_discrimination_effect()` and `indirect_discrimination_effect()` metric helpers
- [ ] SVG template: `causal_graph.svg` for visualization
- [ ] Integration with `BiasDetector` for causal-aware auditing
- [x] Documentation with examples (API_REFERENCE.md, LIBRARY_OVERVIEW.md)

#### Dependencies
- `networkx>=3.0` (graph operations)
- Optional: `dowhy>=0.13` (advanced causal inference; the `causal` extra's floor,
  because dowhy 0.11 and 0.12 call a networkx function networkx has removed)

---

## Gap 2: LLM Fairness Support (COMPLETE)

### Current State
- ✅ Classification, regression, ranking metrics
- ✅ **Complete**: `vfairness.llm` module implemented with 10 classes. Eight import directly from `vfairness.llm` (CounterfactualTester, OutputAnalyzer, NonDeterminismAnalyzer, LLMApiProxy, BenchmarkRunner, DecodingTrustRunner, IntersectionalAnalyzer, CoTFaithfulnessAnalyzer); the remaining two live in their own modules, `vfairness.llm.embedding_bias` (EmbeddingBiasDetector) and `vfairness.llm.text_fairness` (TextFairnessAnalyzer)
- 2026-09-09, wave 0 of the LLM honesty work (`d610f51`): the module's contract was corrected where it answered without measuring. `OutputAnalyzer` and `compute_disparity` now report `None` / `not_assessed` on empty or single-sample input instead of a clean result; `CounterfactualTester` takes the deployment sampling (temperature, top-p, seed) and records it on every result; `LLMJudgeScorer` refuses to judge its own subject and is graded `unvalidated` until checked against human ratings; `noise_floor_from_runs()` computes the run-to-run floor from repeated identical prompts. The open LLM items (paired tests, judge validation, full benchmark loaders) are tracked as LF-11 onwards in the platform's own, unpublished LLM fairness assessment concept.

### Course coverage this gap closes
The course covers **Advanced Architecture Cookbook** (Module 3, Sprint 3) including:
- LLM fairness approaches
- Embedding bias detection
- Prompt fairness analysis
- Text generation bias metrics

### Implementation Plan

#### Phase 1: Embedding Bias Detection

**Status:** Implemented as `EmbeddingBiasDetector` (`llm/embedding_bias.py`): WEAT (Caliskan et al., 2017) and SEAT (May et al., 2019) on any `{word: vector}` map or injected `embed_fn`, with a permutation-test p-value and severity grade. The shipped `weat()` / `seat()` interface (see API_REFERENCE.md) supersedes the sketch below.

```python
from vfairness.preprocessing import EmbeddingBiasDetector

detector = EmbeddingBiasDetector()

# Analyze word embeddings for bias
bias_report = detector.analyze_embeddings(
    embeddings=word_vectors,  # numpy array or dict
    word_pairs=[
        ('man', 'woman'),
        ('doctor', 'nurse'),
        ('engineer', 'teacher')
    ],
    protected_concepts=['male', 'female']
)

# WEAT (Word Embedding Association Test) score
weat_score = detector.weat_score(
    target_words=['programmer', 'engineer', 'scientist'],
    attribute_words_a=['male', 'man', 'boy'],
    attribute_words_b=['female', 'woman', 'girl']
)
```

#### Phase 2: Text Classification Fairness

**Status:** Implemented as `TextFairnessAnalyzer` (`llm/text_fairness.py`): per-group mean-score disparity over any `score_fn(texts)`, with a Mann-Whitney U significance test and severity grade. The shipped `analyze()` interface supersedes the sketch below.

```python
from vfairness.evaluation import TextFairnessAnalyzer

analyzer = TextFairnessAnalyzer()

# Analyze sentiment/toxicity classifier
report = analyzer.analyze(
    model=toxicity_classifier,
    texts=test_texts,
    sensitive_attributes={
        'identity_mentions': ['black', 'white', 'asian', 'hispanic'],
        'gender_mentions': ['he', 'she', 'they']
    }
)

# Identity term bias
identity_bias = analyzer.identity_term_bias(
    texts=test_texts,
    predictions=model_predictions,
    identity_terms=['muslim', 'christian', 'jewish', 'hindu']
)
```

#### Phase 3: Prompt Fairness Analysis

**Status:** Delivered, but as part of `CounterfactualTester` (`llm/counterfactual.py`) rather than a standalone `PromptFairnessAnalyzer`, which does not exist under any module. The tester generates demographic-swapped variants of a prompt template (`generate_swaps()`), runs each through `LLMApiProxy` n times (`run_test()`) and reports the disparity with a significance test (`compute_disparity()`). The shipped interface supersedes the sketch below.

```python
from vfairness.preprocessing import PromptFairnessAnalyzer

analyzer = PromptFairnessAnalyzer()

# Test prompt templates for bias
template = "The {profession} walked into the room. {pronoun} was carrying a briefcase."

bias_results = analyzer.analyze_template(
    template=template,
    variables={
        'profession': ['doctor', 'nurse', 'engineer', 'teacher'],
        'pronoun': ['He', 'She', 'They']
    },
    llm_model=openai_client  # or any LLM interface
)

# Stereotype reinforcement score
stereotype_score = analyzer.stereotype_reinforcement(
    prompts=prompt_list,
    completions=llm_completions,
    stereotype_pairs=[
        ('nurse', 'female'),
        ('engineer', 'male'),
        ('CEO', 'male')
    ]
)
```

#### Phase 4: Generation Fairness Metrics

**Status:** Delivered as `OutputAnalyzer` (`llm/output_analysis.py`) plus the pluggable `TextScorer` family, not as a `GenerationFairnessMetrics` class, which does not exist under any module. `OutputAnalyzer` compares generated text across demographic groups with Mann-Whitney U and an effect size, over sentiment, toxicity, regard, stereotype, refusal rate, helpfulness, information quality, representation, framing, semantic quality and length, or all of them via `analyze_all()`. The shipped interface supersedes the sketch below.

```python
from vfairness.evaluation import GenerationFairnessMetrics

metrics = GenerationFairnessMetrics()

# Demographic parity in generated content
dp_score = metrics.generation_demographic_parity(
    generated_texts=llm_outputs,
    demographic_classifier=gender_classifier
)

# Representation in generated content
representation = metrics.content_representation(
    generated_texts=llm_outputs,
    identity_terms={
        'gender': ['man', 'woman', 'non-binary'],
        'race': ['Black', 'White', 'Asian', 'Hispanic']
    }
)

# Toxicity disparity
toxicity_disparity = metrics.conditional_toxicity(
    prompts=prompts_with_identity,
    completions=llm_completions,
    toxicity_model=perspective_api
)
```

#### Technical Approach
1. **Framework agnostic**: Work with any embedding format (numpy arrays, torch tensors)
2. **LLM agnostic**: Abstract interface for OpenAI, Anthropic, HuggingFace, local models
3. **Lightweight core**: Core metrics don't require heavy NLP dependencies
4. **Optional deep integration**: HuggingFace transformers for advanced features

#### Deliverables
- [x] `CounterfactualTester`: counterfactual prompt testing with 9 swap strategies (see `_SWAP_STRATEGIES`)
- [x] `OutputAnalyzer`: sentiment, toxicity, refusal, length analysis
- [x] `NonDeterminismAnalyzer`: noise offset (separates bias from randomness)
- [x] `LLMApiProxy`: unified API abstraction (OpenAI / Anthropic / Custom)
- [x] `BenchmarkRunner`: BBQ, BOLD standardized benchmarks (HolisticBias added later)
- [x] `DecodingTrustRunner`: DecodingTrust (Wang et al. 2023, NeurIPS), with 8 trustworthiness dimensions, 24 demographic groups, 308 built-in prompts
- [x] `IntersectionalAnalyzer`: multi-attribute intersection testing
- [x] `CoTFaithfulnessAnalyzer`: chain-of-thought reasoning audit
- [x] `EmbeddingBiasDetector`: WEAT (Caliskan et al. 2017) and SEAT (May et al. 2019)
- [x] `TextFairnessAnalyzer`: per-group score-disparity analysis with significance testing
- [ ] `PromptFairnessAnalyzer`: prompt-template bias analysis (not yet implemented)
- [ ] Generation-fairness metrics (not yet implemented)
- [ ] SVG templates: `embedding_bias.svg`, `llm_fairness_report.svg`
- [x] Integration examples with OpenAI, Anthropic, HuggingFace
- [ ] Documentation with LLM fairness best practices

#### Dependencies
- Core: No additional dependencies (works with numpy arrays)
- Optional: `transformers>=4.0` (HuggingFace integration)
- Optional: `openai>=1.0` (OpenAI integration)
- Optional: `anthropic>=0.5` (Anthropic integration)

---

## Gap 3: Real-time Monitoring and Drift Detection (COMPLETE)

### Current State
- ✅ Static fairness analysis
- ✅ CI/CD deployment gates
- ✅ **Complete**: `vfairness.operations.monitoring` implemented with FairnessMonitor, FairnessDriftDetector (CUSUM/EWMA), AdaptiveThresholdManager, FairnessAlertPrioritizer

### Course coverage this gap closes
The course covers **Fairness in Monitoring** (Module 4, Part 4) including:
- Real-time metric tracking
- Fairness drift detection
- A/B testing frameworks for fairness
- Alerting and escalation

### Implementation Plan

#### Phase 1: Streaming Fairness Monitor

**Status:** Implemented as `FairnessMonitor` (`operations/monitoring/`), but with a DataFrame-batch interface, not the keyword constructor sketched below. The shipped form is `FairnessMonitor(window_size=..., alert_threshold=..., config=..., custom_metrics=...)`, then `set_reference(df)` once and `update_and_check(batch_df) -> WindowMetrics` per batch, reading back through `get_current_metrics()`, `get_metric_history()`, `get_alert_summary()` and `get_window_df()`. There is no `update()` and no `get_historical()`. The shipped interface supersedes the sketch below.

```python
from vfairness.operations import FairnessMonitor

monitor = FairnessMonitor(
    metrics=['demographic_parity', 'equalized_odds'],
    protected_attrs=['gender', 'race'],
    window_size=1000,  # Rolling window
    alert_thresholds={
        'demographic_parity_difference': 0.1,
        'equalized_odds_difference': 0.15
    }
)

# Process predictions in real-time
for prediction_batch in production_stream:
    alerts = monitor.update(
        y_true=prediction_batch['labels'],
        y_pred=prediction_batch['predictions'],
        protected=prediction_batch['demographics']
    )

    if alerts:
        for alert in alerts:
            logging.warning(f"Fairness alert: {alert}")

# Get current state
current_metrics = monitor.get_current_metrics()
historical_metrics = monitor.get_historical(hours=24)
```

#### Phase 2: Drift Detection

**Status:** Implemented as `FairnessDriftDetector` (`operations/monitoring/`), and it detects drift in a metric SERIES rather than by diffing two datasets. The shipped form is `FairnessDriftDetector(window_sizes=..., wavelet=..., significance_level=..., min_drift_score=..., compute_mmd=..., n_permutations=..., random_state=...)`, then `set_baseline(series)` and `check_drift(current_series, metric=...) -> MultiscaleDriftResult`, with `detect_drift_ks()`, `detect_drift_mmd()`, `detect_drift_multiscale()`, `run_sprt()` and `decompose_temporal_patterns()` alongside. There is no `detect()`, no `detect_concept_drift()` and no `method='psi'` selector. The shipped interface supersedes the sketch below.

```python
from vfairness.operations import FairnessDriftDetector

drift_detector = FairnessDriftDetector(
    reference_data=training_data,
    method='psi',  # Population Stability Index
    # or 'ks' (Kolmogorov-Smirnov), 'js' (Jensen-Shannon)
)

# Detect drift in fairness metrics
drift_report = drift_detector.detect(
    current_data=production_data,
    metrics=['demographic_parity', 'equal_opportunity']
)

# Returns drift scores and alerts
# {
#   'demographic_parity': {'drift_score': 0.12, 'alert': True},
#   'equal_opportunity': {'drift_score': 0.05, 'alert': False},
#   'group_distribution_shift': {'gender': 0.08, 'race': 0.15}
# }

# Concept drift in protected group performance
concept_drift = drift_detector.detect_concept_drift(
    reference_model_predictions=training_predictions,
    current_model_predictions=production_predictions
)
```

#### Phase 3: Alert & Escalation Framework

**Status:** Shipped, under different names. There is no `FairnessAlertManager`: alerting is `FairnessAlertPrioritizer` plus `AdaptiveThresholdManager` (both `vfairness.operations`), and neither takes a `tiers=` / `notification_channels=` mapping. Notification transport (Slack, email, PagerDuty) is deliberately NOT in the library. The shipped interface supersedes the sketch below.

```python
from vfairness.operations import FairnessAlertManager

alert_manager = FairnessAlertManager(
    tiers={
        'info': {'threshold': 0.05, 'action': 'log'},
        'warning': {'threshold': 0.10, 'action': 'notify'},
        'critical': {'threshold': 0.15, 'action': 'escalate'},
        'emergency': {'threshold': 0.25, 'action': 'halt_predictions'}
    },
    notification_channels={
        'slack': slack_webhook_url,
        'email': alert_email_list,
        'pagerduty': pagerduty_api_key
    }
)

# Integrate with monitor
monitor = FairnessMonitor(
    alert_manager=alert_manager,
    ...
)

# Or use standalone
alert_manager.evaluate_and_alert(
    metric='demographic_parity_difference',
    value=0.18,
    context={'model': 'loan_approval_v2', 'group': 'race'}
)
```

#### Phase 4: A/B Testing Framework

**Status:** Shipped as `FairnessExperiment` (`operations/experimentation/`), not `FairnessABTest`, which does not exist under any module. Power analysis ships with it. The shipped interface supersedes the sketch below.

```python
from vfairness.operations import FairnessABTest

ab_test = FairnessABTest(
    control_model=model_v1,
    treatment_model=model_v2,
    metrics=['accuracy', 'demographic_parity', 'equalized_odds'],
    protected_attrs=['gender', 'race'],
    significance_level=0.05
)

# Run test
results = ab_test.run(
    X=test_features,
    y_true=test_labels,
    protected=test_demographics,
    sample_ratio=0.5
)

# Statistical comparison
comparison = ab_test.compare()
# {
#   'accuracy': {'control': 0.85, 'treatment': 0.87, 'significant': True},
#   'demographic_parity': {'control': 0.12, 'treatment': 0.08, 'significant': True},
#   'recommendation': 'DEPLOY_TREATMENT'  # Based on Pareto improvement
# }
```

#### Technical Approach
1. **Lightweight streaming**: No external message queue required for basic use
2. **Integration ready**: Easy integration with Kafka, Redis, cloud services
3. **Stateless option**: Can run without persistent state for simple deployments
4. **Dashboard ready**: Metrics exposed in Prometheus/StatsD format

#### Deliverables
- [x] `FairnessMonitor` class for real-time tracking
- [x] `FairnessDriftDetector` with CUSUM, EWMA, multiscale methods
- [x] `AdaptiveThresholdManager` with adaptive alerting
- [x] `FairnessAlertPrioritizer` with tiered alerting
- [x] `TemporalFairnessAnalyzer` for temporal pattern analysis
- [x] `FairnessExperiment` for A/B testing comparison
- [x] SVG templates: `monitoring_dashboard.svg`, `drift_report.svg` (plus `alert_timeline.svg`, `temporal_analysis.svg`)
- [ ] Integration examples: Prometheus, Grafana, DataDog
- [ ] Documentation with production monitoring best practices

#### Dependencies
- Core: No additional dependencies
- Optional: `prometheus-client>=0.14` (Prometheus export)
- Optional: `redis>=4.0` (distributed state)
- Optional: `kafka-python>=2.0` (Kafka streaming)

---

## Gap 4: Agile Integration (SAFE Framework)

### Current State
- ✅ pytest assertions for testing
- ✅ FairnessTestSuite for CI/CD
- ❌ **Not started**: No SAFE User Story or FAIR Acceptance Criteria templates. This is the main remaining product gap.

### What's Missing
The course covers **Fair AI Scrum Toolkit** (Module 3, Sprint 1) including:
- SAFE (Socially-Aware Feature Engineering) User Story Framework
- FAIR (Fairness-Aware Implementation Requirements) Acceptance Criteria
- Sprint ceremony fairness checkpoints
- Documentation templates

### Implementation Plan

#### Phase 1: SAFE User Story Generator
```python
from vfairness.operations import SAFEUserStoryGenerator

generator = SAFEUserStoryGenerator()

# Generate fairness-aware user story
story = generator.create_story(
    feature="loan approval automation",
    user_groups=['loan applicants', 'loan officers'],
    protected_attributes=['race', 'gender', 'age'],
    business_context="retail banking"
)

# Output structured user story
print(story.to_markdown())
# As a [loan applicant], I want [automated loan decisions]
# so that [I receive fair and timely responses].
#
# FAIRNESS CONSIDERATIONS:
# - Protected attributes: race, gender, age
# - Historical patterns to avoid: redlining, gender discrimination in credit
# - Relevant regulations: ECOA, Fair Housing Act
#
# SAFE CHECKLIST:
# [ ] Data audit for historical bias completed
# [ ] Proxy variable analysis performed
# [ ] Fairness metrics defined (DP, EO thresholds)
# [ ] Stakeholder review scheduled
```

#### Phase 2: FAIR Acceptance Criteria
```python
from vfairness.operations import FAIRAcceptanceCriteria

criteria = FAIRAcceptanceCriteria(
    feature="loan approval model",
    fairness_requirements={
        'demographic_parity_difference': {'max': 0.1},
        'equalized_odds_difference': {'max': 0.15},
        'disparate_impact_ratio': {'min': 0.8}
    },
    protected_attrs=['race', 'gender'],
    regulatory_requirements=['ECOA', 'EU_AI_ACT']
)

# Generate acceptance criteria
print(criteria.to_gherkin())
# Given the loan approval model is deployed
# When predictions are made on the test population
# Then demographic_parity_difference < 0.1 for all groups
# And equalized_odds_difference < 0.15 for all groups
# And disparate_impact_ratio >= 0.8 (80% rule) for all groups
# And EU AI Act Article 10 data governance is documented

# Auto-generate pytest tests
criteria.generate_pytest_tests(output_path='tests/test_fairness.py')
```

#### Phase 3: Sprint Ceremony Checklists
```python
from vfairness.operations import SprintFairnessChecklist

checklist = SprintFairnessChecklist(
    sprint_type='planning',  # or 'review', 'retrospective'
    team_context={
        'project': 'credit_scoring',
        'current_sprint': 'S23',
        'fairness_maturity': 'intermediate'
    }
)

# Get ceremony-specific checklist
items = checklist.get_items()
# Sprint Planning:
# [ ] Review fairness metrics from previous sprint
# [ ] Identify stories with fairness implications
# [ ] Assign fairness champion for high-risk stories
# [ ] Schedule bias review meetings
# [ ] Update fairness debt backlog

# Export to various formats
checklist.to_confluence()
checklist.to_notion()
checklist.to_jira()
```

#### Phase 4: Documentation Templates
```python
from vfairness.operations import FairnessDocumentation

docs = FairnessDocumentation()

# Generate model card with fairness section
model_card = docs.generate_model_card(
    model_name="LoanApprovalV2",
    model_type="XGBoost Classifier",
    fairness_evaluation=fairness_analyzer.full_report(),
    intended_use="Automated loan pre-approval",
    limitations="Not validated for commercial loans > $500k",
    ethical_considerations="See fairness analysis below"
)

# Generate FMEA (Failure Mode and Effects Analysis) for fairness
fmea = docs.generate_fairness_fmea(
    system="loan_approval_pipeline",
    components=['data_ingestion', 'feature_engineering', 'model', 'deployment'],
    protected_attrs=['race', 'gender', 'age']
)

# Generate regulatory compliance documentation
compliance_doc = docs.generate_compliance_doc(
    regulations=['EU_AI_ACT', 'ECOA'],
    fairness_evidence=fairness_analyzer.full_report()
)
```

#### Deliverables
- [ ] `SAFEUserStoryGenerator` class
- [ ] `FAIRAcceptanceCriteria` class with Gherkin output
- [ ] `SprintFairnessChecklist` for all ceremony types
- [ ] `FairnessDocumentation` for model cards, FMEA, compliance
- [ ] Templates: Confluence, Notion, Jira integration
- [ ] Documentation with agile fairness best practices

#### Dependencies
- Core: No additional dependencies
- Optional: `jira>=3.0` (Jira integration)
- Optional: `notion-client>=2.0` (Notion integration)

---

## Gap 5: Ranking & Exposure Fairness

### Current State
- ✅ Basic ranking metrics (NDCG parity)
- ⚠️ **Partial**: Limited exposure fairness, no comprehensive ranking framework

### What's Missing
The course covers ranking fairness for recommendation systems including:
- Exposure fairness metrics
- Position bias correction
- Multi-sided marketplace fairness
- Supplier/producer fairness

### Implementation Plan

#### Phase 1: Comprehensive Exposure Metrics
```python
from vfairness.evaluation import ExposureFairnessMetrics

metrics = ExposureFairnessMetrics()

# Exposure disparity
exposure_disparity = metrics.exposure_disparity(
    rankings=recommendation_rankings,  # List of ranked item lists
    item_groups=item_demographics,     # Item -> group mapping
    position_weights='logarithmic'     # or 'linear', 'inverse'
)

# Attention-weighted fairness
attention_fairness = metrics.attention_weighted_fairness(
    rankings=rankings,
    click_through_rates=ctrs_by_position,
    item_groups=item_demographics
)

# Equity of Attention
eoa = metrics.equity_of_attention(
    rankings=rankings,
    relevance_scores=true_relevance,
    item_groups=item_demographics
)
```

#### Phase 2: Position Bias Correction
```python
from vfairness.post_processing import PositionBiasCorrector

corrector = PositionBiasCorrector(
    method='inverse_propensity',  # or 'click_model', 'intervention'
)

# Estimate position bias
position_bias = corrector.estimate_bias(
    click_data=historical_clicks,
    position_data=historical_positions
)

# Apply correction to rankings
corrected_rankings = corrector.correct(
    rankings=original_rankings,
    item_groups=item_demographics,
    fairness_target='demographic_parity'
)

# Re-rank for fairness
fair_rankings = corrector.rerank(
    scores=relevance_scores,
    item_groups=item_demographics,
    fairness_constraint='exposure_parity',
    lambda_fairness=0.3
)
```

#### Phase 3: Multi-Stakeholder Fairness
```python
from vfairness.evaluation import MarketplaceFairnessAnalyzer

analyzer = MarketplaceFairnessAnalyzer()

# Analyze two-sided marketplace
report = analyzer.analyze(
    recommendations=platform_recommendations,
    consumers=consumer_data,
    producers=producer_data,
    consumer_protected_attrs=['location', 'income_bracket'],
    producer_protected_attrs=['seller_size', 'seller_tenure']
)

# Consumer-side fairness
consumer_fairness = report['consumer_side']
# - Relevance parity across consumer groups
# - Price fairness across consumer groups

# Producer-side fairness
producer_fairness = report['producer_side']
# - Exposure parity across producer groups
# - Revenue opportunity parity

# Platform balance score
balance = report['platform_balance']
# Measures trade-offs between stakeholder groups
```

#### Deliverables
- [ ] `ExposureFairnessMetrics` class with comprehensive metrics
- [ ] `PositionBiasCorrector` for ranking corrections
- [ ] `MarketplaceFairnessAnalyzer` for multi-stakeholder analysis
- [ ] `FairRanker` for fairness-constrained re-ranking
- [ ] SVG templates: `exposure_report.svg`, `marketplace_fairness.svg`
- [ ] Documentation with recommendation system examples

#### Dependencies
- Core: `numpy`, `scipy` (already required)
- Optional: `cvxpy>=1.0` (constrained optimization for re-ranking)

---

## Gap 6: Multi-Modal System Fairness

### Current State
- ✅ Tabular data focus
- ✅ **Image fairness shipped**: the `vfairness.vision` subpackage provides pure-numpy image-fairness metrics (MaxSkew, MinSkew, NDKL, bias amplification), with optional FairFace / CLIP demographic classification in a Python 3.9 vision sidecar
- ⚠️ **Partial**: audio / speech fairness and multimodal fusion remain open

### What's Missing
The course mentions multi-modal systems in the Advanced Architecture Cookbook:
- Image classification fairness
- Audio/speech recognition fairness
- Multi-modal fusion fairness

### Implementation Plan (Future Phase)

#### Phase 1: Image Classification Fairness
```python
from vfairness.evaluation import ImageFairnessAnalyzer

analyzer = ImageFairnessAnalyzer()

# Analyze face recognition system
report = analyzer.analyze(
    model=face_recognition_model,
    images=test_images,
    true_labels=true_identities,
    demographic_labels=demographics  # skin_tone, gender, age
)

# Skin tone analysis (Fitzpatrick scale)
skin_tone_report = analyzer.skin_tone_analysis(
    predictions=model_predictions,
    skin_tone_labels=fitzpatrick_labels
)

# Object detection demographic parity
object_report = analyzer.object_detection_fairness(
    detections=detector_outputs,
    ground_truth=gt_boxes,
    context_demographics=scene_demographics
)
```

#### Phase 2: Audio/Speech Fairness
```python
from vfairness.evaluation import AudioFairnessAnalyzer

analyzer = AudioFairnessAnalyzer()

# Analyze speech recognition
report = analyzer.analyze(
    model=asr_model,
    audio_samples=test_audio,
    transcriptions=true_transcriptions,
    speaker_demographics=demographics  # accent, gender, age
)

# Word Error Rate by demographic
wer_by_group = report['wer_disparity']

# Accent fairness
accent_report = analyzer.accent_analysis(
    predictions=asr_outputs,
    true_transcriptions=transcriptions,
    accent_labels=accents
)
```

#### Deliverables (Future)
- [x] Image fairness metrics shipped: `vfairness.vision` (MaxSkew, MinSkew, NDKL, bias amplification) plus optional FairFace / CLIP demographic classification in a Python 3.9 sidecar
- [ ] `AudioFairnessAnalyzer` class
- [ ] `MultiModalFairnessAnalyzer` for fusion systems
- [ ] Integration with common CV / audio libraries

#### Dependencies (Future)
- Optional: `pillow>=9.0` (image processing)
- Optional: `librosa>=0.9` (audio processing)
- Optional: `torchvision>=0.12` (CV models)

---

## Gap 7: Validity / Groundedness Axis (second assurance axis)

**Status:** LIVE end-to-end (2026-08-16), fail-closed. The interim judge rung (VA-10, Mistral Small 3.2) is deployed on the platform consumer and scores real groundedness; the `vfairness_validity_run` task, run-record persistence, and the platform grade + DoD-seal wire are all live. Validity now feeds the assurance grade PROVISIONALLY for supported languages (interim judge, disclosed). STILL PENDING: the owned detector (VA-21) that replaces the LLM judge in the hot path, Apertus as the transparent judge successor, and a per-language gold study to move from provisional to VALIDATED. Do NOT register in CAPABILITY_REGISTRY/manifest until VALIDATED.

The whole library so far answers one question: is the system **fair**? Generative
and retrieval-augmented systems raise a second, orthogonal question the fairness
metrics cannot touch: is the answer **grounded**, that is, actually supported by its
sources rather than confabulated? Validity is that second axis. It is not a
separate product or a separate segment: it reuses the same task-dispatch, the same
`TaskResult` envelope, and feeds the same grade and seal as fairness.

### Current State
- ✅ Frozen VG-* metric family (`vfairness.validity`), matching the platform's validity metric contract
- ✅ `GroundednessScorer`: a degrading ladder that is fail-closed by construction (an absent scorer reads as "not measured", never "grounded"; `value=None`, never `0.0`-as-clean)
- ✅ `aggregate_validity()`: batch VG-001 / VG-005 (and optional VG-002/003) over measured records only; CIs from `_statistics.py`, omitted never faked
- ✅ Interim judge rung `LlmGroundednessJudge` (VA-10): self-hosted, OpenAI-compatible, snapshotted+versioned rubric (`PROMPT_VERSION`), REFUSE on any failure. Judge is Mistral Small 3.2 (European, Apache-2.0); Qwen3 the licence-clean fallback; Apertus the planned successor
- ✅ `operations/validity/task_handlers.py`: the `vfairness_validity_run` task type, JSON over stdin, envelope on stdout
- ✅ A dedicated suite (`tests/test_validity_scorer.py`) asserting the fail-closed property end to end

### What's Missing
- **VA-21 owned detector:** a self-hosted claim-support model trained on owned, adjudicated gold, replacing the LLM judge in the hot path with no third-party licence dependency. This is the target primary rung.
- **M1 context capture:** the platform run record must persist the retrieved contexts per answer, without them the judge has nothing defensible to score against.
- **Owned gold set:** an adjudicated reference set for calibration and for VG-004 / VG-007. Only owned, licence-clean gold may train or tune a rung (no CC-BY-NC/ND or research-only data, including as training data).
- **VA-64 group-cut aggregation (the thesis in code, opened 2026-08-21):** there is NO group cut anywhere in `vfairness.validity` today. `aggregate_validity()` collapses a batch to one number, and grep for group/persona/cohort/parity across the validity path returns zero hits. So "groundedness cut by protected group", which is the stated market differentiator and the central claim that fairness IS the equal distribution of validity, is not computable by this library. `aggregate_validity_by_group()` closes it: per-cell mean with `wilson_score_interval`, `proportion_z_test` routed to `fisher_exact_test` on small cells, Benjamini-Hochberg across cells, and `insufficient_evidence` as a first-class fail-closed verdict for a thin cell. Closes audit requirement F1, which was PARTIAL only because no Axis-1 metric existed to parity on; VG-001 now exists. Composition of existing `_statistics.py` primitives, not new statistics.
- **VA-65 declared detection limit:** every run returns `minimum_detectable_effect` at the actual per-cell n, plus `power_warning` on underpowered cells. Rule: a run reporting "no gap found" MUST also report the smallest gap it could have found, so an absence of evidence is never read as evidence of absence.
- **VA-66 noise floor:** feed the existing per-query repeats to `NonDeterminismAnalyzer.characterize_noise` so a run states its own wiggle; any gap or drift smaller than the floor is reported as within-noise, never as a finding. Also flags identical-across-repeats (temperature 0, caching, or a copied run).
- **VA-67 run-to-run drift comparison:** delta per metric and per cell against the VA-66 floor, and a REFUSAL to compare unless gold-set id and version, judge model, rubric `PROMPT_VERSION`, persona set and repeat count all match, so instrument drift is never published as system drift. Library half of platform Pulse VA-38. Worth noting for sequencing: a same-instrument delta cancels systematic judge error, so drift detection is defensible with the INTERIM judge whereas certification is not.
- **Manifest registration:** the scorer joins `CAPABILITY_REGISTRY` and the synced platform manifest ONLY once it can feed a grade end to end. Registering earlier would advertise a capability the engine cannot yet deliver, which the house rule forbids. Until then the platform keeps validity results in the `metricsDeferred` channel.

### Architectural requirement: two tracks, one library (VA-68..VA-73)

**This library is not a platform client.** vfairness serves researchers and practitioners who
will never touch validant.ai, so every validity capability must be usable standalone: importable,
documented, exercisable on a plain list of records, with no dependency on task dispatch, on a
Supabase schema, or on any platform payload shape. `operations/validity/task_handlers.py` is ONE
consumer of the axis, not its interface. A capability reachable only through the dispatch envelope
is not finished.

**Validity and fairness are separate tracks that compose at exactly one named point.** They answer
different questions, they degrade differently, and one must never silently become the other. The
single legitimate join is the group cut (VA-64), where a validity measurement is read as a fairness
property. Every other coupling is a defect.

This is a standing requirement on ALL future VA work, not a one-off task, and it is written down
because convention already failed twice on the platform side. Seeding VG-001..VG-007 into the shared
metric catalogue made "Groundedness" selectable in the CI/CD gate metric picker (toggle to 'block',
straight into an exported gate YAML) and as the A/B primary outcome metric feeding treatment-effect
math; a completeness sweep then found a second leak in the validation-metrics hook. In both cases an
enforceable fairness decision was being fabricated on an axis with no validated scorer. Convention
did not prevent it. A predicate plus a filter plus a test did. **The library currently has no such
predicate and no such test**; the ids stay inside `vfairness.validity` today by habit alone.

The rule for every future validity capability: it lands inside `vfairness.validity`, it exports no
metric id into any fairness registry or enumeration, it carries a test asserting that it cannot
appear in a fairness enumeration, and it fails closed when unmeasured. If a capability genuinely
needs a new metric id, VG-* is frozen at 001..007 and a change requires a deliberate contract
revision synced with the platform's validity metric contract, never a drive-by addition.

| ID | Item | VG metrics | Effort | Depends |
| --- | --- | --- | --- | --- |
| VA-68 | Track-separation invariant, enforced by test | all VG-* | 1 to 2 d | none |
| VA-69 | Library-first research API for the validity axis | all VG-* | 2 to 4 d | none |
| VA-70 | Hallucination taxonomy as first-class output | VG-001, VG-002, VG-005 | 3 to 5 d | none |
| VA-71 | Method-family plurality behind the scorer protocol | all VG-* | 5 to 8 d | VA-68 |
| VA-72 | Retrieval preconditions gate the groundedness claim | VG-003, VG-004 | 3 to 5 d | none |
| VA-73 | Abstention and refusal appropriateness as an outcome | none (no new id) | 2 to 4 d | none |

**VA-68 track-separation invariant.** Add a public `is_validity_metric(metric_id)` predicate, the
library twin of the platform's `isValidityMetricId()`, and a test that enumerates every
fairness-facing metric registry and asserts no VG-* id appears in any of them. The test is the
deliverable; the predicate is what makes it expressible. Also document the one legitimate join so
that VA-64's group cut reads as intentional composition rather than as a leak.

**VA-69 library-first research API.** A researcher must be able to score an eval set from a Python
session in a handful of lines, with no platform in the picture. Concretely: a documented entry point
taking records of (question, contexts, answer), a worked example in `docs/examples.md`, a notebook,
and an API-reference section that treats the validity axis as a first-class subsystem rather than a
dispatch target. Verify by writing the example against the installed package, not against the repo.

**VA-70 hallucination taxonomy as first-class output.** The literature's distinctions are load-bearing
and the library currently collapses them. Two specific defects: the judge returns `faithfulness` as
literally the SAME value as `groundedness` (an acknowledged interim conflation in `judge.py`), so
VG-002 carries no independent information; and an unsupported claim is counted but not CLASSIFIED.
Intrinsic hallucination (the answer CONTRADICTS the source) and extrinsic hallucination (the source
can neither confirm nor deny) are different failures with different remediations: intrinsic points at
generation with retrieval working, extrinsic often points at retrieval. Extend the per-claim result to
carry that classification, de-conflate VG-002 from VG-001, and keep the aggregate backward compatible
so an existing caller sees no behaviour change. Reporting a single "unsupported" count where the field
distinguishes two failure modes is a loss of information a research user will immediately notice.

**VA-71 method-family plurality.** The `GroundednessJudge` protocol is already the extension point,
but it is framed as "rungs of our ladder" rather than as "the method families a practitioner may
choose". The field has four: NLI and entailment based, QA based, LLM-as-judge, and uncertainty or
self-consistency based. Publish the protocol as a supported public interface, document what each
family buys and costs, and ship at least one non-LLM reference implementation so the library is not
implicitly an LLM-judge library. This also serves us directly: the owned detector (VA-21) is simply
another implementation of the same protocol, and validating it against a second family is stronger
evidence than validating it against itself.

**VA-72 retrieval preconditions gate the claim.** Groundedness is undefined when retrieval failed: if
the right passage was never fetched, the answer had nothing to be faithful to, and a low score
misattributes a retrieval defect to generation. Context precision and recall (VG-003, VG-004) are
therefore preconditions rather than optional extras. Recall needs gold and stays blocked; precision
does not. Report the precondition alongside the score and fail closed to "undefined" rather than
"poorly grounded" where retrieval is the actual failure.

**VA-73 abstention and refusal appropriateness.** A correct refusal must never score as a failure,
and this is not hypothetical: 50 out of 50 genuine German refusals scored zero in the audit run. The
judge already returns 1.0 for a bare refusal, which prevents the false penalty, but appropriateness
(SHOULD it have refused?) is unmeasured and needs the gold set's should-refuse label. Report
abstention as a distinct outcome class rather than folding it into the groundedness mean, since a
system that refuses everything would otherwise score perfectly. No new VG id: this is a qualifier on
the run, not an eighth metric.

**Sequencing.** VA-68 first and cheaply, because it protects everything after it. VA-70 is the item a
research user will value most and the one that most visibly reflects the field's own vocabulary.
VA-71 is the largest and unlocks the strongest validation story for VA-21.

### Implementation Plan
Milestones M0 through M8 cover the data plan, the licence-clearance register and
the interim versus owned build. They are planned in an internal document that is
not published with the library, so this is the published statement of where they
stand: M0 (judge choice and licence sign-off) is complete; the interim rung is
built. Next is M1 (context capture) so the interim judge can emit a defensible
number, then the owned detector.

---

## Completed Beyond Original Roadmap

The following modules were implemented ahead of schedule, extending vfairness beyond the original roadmap scope:

### Agent Fairness Testing (`vfairness.agents`): 6 classes

Full-lifecycle fairness auditing for autonomous AI agents, based on correspondence testing methodology from employment discrimination research.

| Class | Purpose |
|-------|---------|
| `CorrespondenceTester` | Paired-artifact testing (gold standard) |
| `ToolBiasAuditor` | Tool selection fairness audit |
| `RAGBiasAnalyzer` | RAG retrieval bias detection |
| `PipelineTracker` | Multi-stage bias tracking |
| `TemporalTracker` | Drift detection (CUSUM / EWMA) |
| `ActionBiasAnalyzer` | Outcome and delegation bias |

### Multi-Agent Fairness Testing (`vfairness.multi_agent`)

Detection of emergent bias in multi-agent systems, meaning bias that appears at the system level even when individual agents pass fairness checks.

| Class | Purpose |
|-------|---------|
| `CompositionalityAnalyzer` | Component vs system bias (non-compositionality) |
| `GroupthinkDetector` | Echo-chamber and coalition detection |
| `EmergentBiasDetector` | Novel bias from agent interaction |
| `DelegationRoutingAuditor` | Delegation and routing fairness across agents |
| `AdversarialCollusionDetector` | Collusion and adversarial coordination detection |
| `NegotiationFairnessTracker` | Negotiation and bargaining fairness across agents |
| `MultiAgentRunHarness` | Run harness producing a `HarnessTrace` for the analysers |

### Explainability / XAI (`vfairness.xai`): a major new subsystem

Explainer adapters (TreeSHAP, LinearSHAP, KernelSHAP, LIME, DiCE counterfactuals, Integrated Gradients, Anchors) behind a `route_explainer()` router; trustworthiness diagnostics (faithfulness, stability, adversarial probing); `lundberg_fairness_decomposition()` for per-feature attribution of group-fairness metrics (asserts a 1e-6 additive identity before persisting); a Supabase storage writer; and a long-lived pgmq worker loop.

### Operations, Rendering, and Legal subpackages

Also shipped beyond the original roadmap: the `operations` umbrella (causal estimation, CI/CD gates, monitoring, experimentation, the Pulse orchestrator, reporting), the `rendering` subpackage (data-driven Jinja2 SVG reports and dashboards), and the `legal` subpackage (curated admissibility rule packs by use case and jurisdiction, with no runtime LLM).

---

## Implementation Timeline

### Phase 1: Q1 2026 (COMPLETE)
| Feature | Status | Notes |
|---------|--------|-------|
| Causal Graph Basic | **Complete** | networkx dependency; `CausalFairnessGraph` shipped |
| LLM Fairness (`vfairness.llm`) | **Complete** | 10 classes implemented |
| Real-time Monitor (`vfairness.operations.monitoring`) | **Complete** | FairnessMonitor, TemporalFairnessAnalyzer |
| Drift Detection | **Complete** | FairnessDriftDetector (CUSUM / EWMA) |
| Agent Fairness (`vfairness.agents`) | **Complete** | 6 classes (beyond roadmap) |
| Multi-Agent Fairness (`vfairness.multi_agent`) | **Complete** | 7 classes (beyond roadmap) |

### Phase 2: Q2 2026 (Current, June 2026)

**Recently shipped** (no longer pending): Causal counterfactual estimation via DoWhy (identify / mediate / refute / counterfactual / attribute, end to end), A/B testing (`FairnessExperiment` plus `operations/experimentation` with power analysis), the `vfairness.xai` explainability subsystem, and the `vfairness.legal` admissibility packs.

Remaining Phase 2 work:

| Feature | Effort | Impact | Dependencies |
|---------|--------|--------|--------------|
| SAFE User Stories | 2 weeks | Medium | None |
| FAIR Criteria | 2 weeks | Medium | None |
| Sprint Checklists | 1 week | Medium | None |
| Standalone causal direct / indirect effect helpers | 1 week | Medium | DoWhy bridge (shipped) |
| `PromptFairnessAnalyzer` and generation-fairness metrics | 2 weeks | Medium | None |

### Phase 3: Q3 2026 (Medium Priority)
| Feature | Effort | Impact | Dependencies |
|---------|--------|--------|--------------|
| Full exposure / ranking fairness framework | 3 weeks | Medium | None |
| Multi-stakeholder marketplace fairness | 3 weeks | Medium | None |
| Remaining per-module SVG templates (LLM, causal, exposure) | 2 weeks | Medium | None |

### Phase 4: Q4 2026 (Future)
| Feature | Effort | Impact | Dependencies |
|---------|--------|--------|--------------|
| Image Fairness | 4 weeks | Medium | Shipped (`vfairness.vision`) |
| Audio Fairness | 4 weeks | Medium | External |
| Multi-modal Fusion | 4 weeks | Medium | External |

---

## Beta Readiness (Engineering Hardening)

Feature completeness is only half of "beta-ready." Before the first public
release the library also went through a structured engineering-hardening pass.
The full inventory (with issue ids) is on the
[Quality and Hardening](QUALITY_AND_HARDENING.md) page and in the changelog; the
beta exit criteria are in [BETA.md](BETA.md). In brief, all of the following now
run in CI on every change:

- **Deep audit closed.** A two-workflow, adversarially-verified audit produced
  242 findings; all fixed, zero open. The suite grew from 831 to over 1,280
  tests during the campaign and has kept growing since (more than 1,600 as of
  2026-08-22).
- **Lint + format gates (VB-LINT-1, VB-LINT-2).** `ruff check` and
  `ruff format --check` are blocking; the whole tree is lint-clean and formatted.
- **Type gates (VB-TYPE-1, VB-TYPE-2).** A full burndown took `mypy src` from 500
  errors to zero and made it blocking, plus a `mypy --strict` island on the
  foundational core. The package ships `py.typed`.
- **Coverage gates (VB-TEST).** Branch coverage is on with a ratcheting floor,
  and a pull-request diff-cover gate requires new code to be tested.
- **Security + supply chain.** Bandit, pip-audit (`--strict`, over a
  non-editable pip freeze), Dependabot, a CycloneDX SBOM, and PyPI Trusted
  Publishing with provenance. CodeQL and OpenSSF Scorecard are wired but gated
  on the repository being public (no GitHub Advanced Security on a private
  repo); both re-arm automatically if the repo goes public.
- **Release integrity (VB-REL-3).** No publish happens unless the built wheel
  installs into a clean environment and its public API runs.
- **API stability + methodology.** A frozen public surface with a deprecation
  policy, and a `methodology_version` stamped into every report.

The type-checking pass also fixed two latent correctness bugs it surfaced: a
subgroup-robustness audit and the correlation-based proxy detector (Gap 1 above)
had each been silently disabled by a wrong-argument bug.

---

## Technical Principles

### 1. Minimal Dependencies
- Core functionality works with the five hard runtime dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, and `requests`
- Heavy dependencies (torch, transformers) should be optional
- Use lazy imports to avoid loading unused modules

### 2. Consistent API Design
```python
# All analyzers follow the same pattern
class NewAnalyzer:
    def __init__(self, **config):
        """Initialize with configuration."""
        pass

    def fit(self, data, protected_attrs):
        """Learn from reference data if needed."""
        return self

    def analyze(self, data, **kwargs):
        """Run analysis, return structured report."""
        return AnalysisReport(...)

    def to_dict(self):
        """Export results as dictionary."""
        pass

    def to_dataframe(self):
        """Export results as pandas DataFrame."""
        pass
```

### 3. SVG-First Visualization
- Every new feature should have corresponding SVG template
- Templates should be data-driven (Jinja2)
- Support both light and dark themes
- Export to PNG/PDF when needed

### 4. Testing Requirements
- Unit tests for all new functions
- Integration tests with real datasets
- Performance benchmarks for production use
- Fairness testing on standard datasets (Adult, COMPAS, German Credit)

### 5. Documentation Standards
- Docstrings with examples for all public APIs
- Jupyter notebook examples for each major feature
- API reference auto-generated from docstrings
- Concepts page updated with theoretical background

---

## Success Metrics

### Coverage Completeness
- [x] Almost all course topics have a corresponding library feature (fifteen subpackages; the registered-capability total is in `vfairness-manifest.json`); only SAFE / FAIR agile templates, the full exposure / ranking framework, and audio / multimodal fusion remain
- [x] LLM and Monitoring "partial" items upgraded to "complete"
- [x] Causal estimation, A/B testing, explainability (XAI), in-processing, post-processing, preprocessing, CI/CD gates, Pulse, and legal admissibility shipped
- [ ] No "planned" items remaining in core functionality (SAFE / FAIR templates and full ranking / audio framework still open)

### API Quality
- [ ] Consistent API across all modules
- [ ] <5 lines of code for common tasks
- [ ] Clear error messages with remediation suggestions

### Performance
- [ ] Real-time monitor handles 10k predictions/second
- [ ] Drift detection runs in <1 second for 100k samples
- [ ] Full analysis completes in <10 seconds for typical datasets

### Adoption
- [ ] PyPI downloads increase 50% after major releases
- [ ] GitHub stars reach 500+
- [ ] At least 3 case studies from production deployments

---

## Conclusion

This roadmap transforms vfairness from a comprehensive fairness library to the definitive, complete solution for AI fairness. By systematically closing each identified gap, we create a library that:

1. **Covers the full ML lifecycle**: From data auditing through production monitoring
2. **Supports all model types**: Tabular, text, LLMs, ranking systems
3. **Integrates with development workflows**: Agile, CI/CD, documentation
4. **Provides theoretical foundation**: Causal fairness, counterfactuals
5. **Scales to production**: Real-time monitoring, drift detection, alerting

The result is not just another fairness toolkit, but a complete platform for building, deploying, and maintaining fair AI systems.

---

*vfairness Roadmap v1.3 (August 2026). Package version 0.1.0, the first public beta, released 2026-08-23.*
