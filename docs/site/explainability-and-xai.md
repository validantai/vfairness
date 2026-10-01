# Explainability and XAI

How vfairness explains both fairness findings and individual model decisions

---

## Two kinds of explanation

Explainability in vfairness answers two distinct questions.

1. **Why is this finding an issue?** This is fairness-result explanation: it
   turns a raw metric value or a whole results dictionary into a clear,
   regulator-facing narrative. There are two entry points with separate
   knowledge bases: `FairnessExplainer` is the facade that turns supported
   result objects (`BiasAuditReport`, `CalibrationReport`, and friends; it
   refuses raw dicts) into an `ExplanationReport`, while `FairExplAIner`
   explains individual metric values (`explain_metric()` returns a
   `MetricExplanation`) and whole metric dictionaries (`explain_report()`
   returns a plain dict). They share the `MetricExplanation` container and
   severity vocabulary, not one knowledge base.

2. **Why this decision for this person, and which features drive the model?**
   This is model-decision explanation. It is the surface that supports the GDPR
   Art. 22 and EU AI Act Art. 86 right to explanation, and it is built from
   feature attribution, counterfactual fairness, embedding and text bias tests,
   and causal pathway classification.

## The building blocks

| Capability | Class or function | What it tells you |
| --- | --- | --- |
| Fairness-result narrative | `FairExplAIner`, `FairnessExplainer` | Plain-language meaning, severity, and recommendations for a metric or result set |
| Feature attribution | `FeatureAttributionExplainer` | Which features drive predictions, globally across a dataset or locally for one decision |
| Counterfactual fairness | `counterfactual_fairness()` | How often a decision changes when only the protected attribute is flipped |
| Embedding bias | `EmbeddingBiasDetector` | Stereotypical associations in word or sentence embeddings (WEAT/SEAT) |
| Text-classifier bias | `TextFairnessAnalyzer` | Whether a classifier scores identity groups differently |
| Causal pathways | `CausalFairnessGraph` | Whether each protected-to-outcome path is direct, indirect, or proxy discrimination |

## How the pieces fit together

The XAI layer takes model predictions and supporting inputs, runs the relevant
explainers, and produces a single `ExplanationReport` that the platform renders
for auditors and reviewers.

```
   INPUTS                       XAI LAYER                       OUTPUT

   model / predictions  >|
   factual vs CF preds  >|    [ attribution      ]
   embeddings           >| => [ counterfactual   ] => ExplanationReport
   texts by group       >|    [ embedding / text ]    (summary, severity,
   causal DAG           >|    [ causal graph     ]     recommendations,
   fairness metrics     >|    [ fairness result  ]     per-item findings)
```

## Feature attribution

`FeatureAttributionExplainer` wraps any `predict(X)` callable, so it works with a
fitted estimator, a wrapped API client, or a plain function.

- **Global**: permutation importance across a dataset. With labels it scores the
  drop in fit when a column is shuffled (`method="permutation"`). Without labels
  it still permutes, but scores the mean absolute shift in the predictions
  instead of a drop in fit (`method="permutation_numpy"`), which gives magnitude
  only: there is no fit to lose, so no direction can be inferred.
- **Local**: `explain_decision()` gives per-decision attribution against a
  baseline (the column medians of a background set), using occlusion by default.
  Exact SHAP is opt-in with `use_shap=True` and additionally requires the `shap`
  package; installing `shap` alone does not change the default.

Each result lists per-feature contributions with a magnitude and a direction
(increase, decrease, or neutral). The label-bearing paths also carry a signed
value; the label-free global path reports direction `neutral` and a null
`signed_value`, and says so in the result's notes rather than implying a sign it
did not measure.

## Counterfactual fairness

`counterfactual_fairness()` compares the model's predictions on the original
data with its predictions on caller-supplied counterfactual predictions. It
reports the flip rate (the headline number), the mean and maximum score change,
a severity grade, and a plain-language interpretation. It is pure NumPy and
needs no causal library. This is a perturbation-sensitivity measure; it matches
Kusner et al. (2017) counterfactual fairness only when the counterfactuals come
from a structural causal model (use the DoWhy-backed causal counterfactual op
for that). A naive single-attribute flip measures direct sensitivity and will
not catch bias laundered through a held-fixed proxy.

## LLM and text bias

- `EmbeddingBiasDetector` runs WEAT (Caliskan et al., 2017) and its
  sentence-level extension SEAT (May et al., 2019) on a word-to-vector map or an
  injected embedding function, reporting an effect size, a two-sided
  permutation-test p-value (so a stereotype in either direction is flagged), and
  a severity grade. The SEAT label is applied only when the inputs are
  sentence-resolved; on plain word vectors the result is reported as WEAT.
- `TextFairnessAnalyzer` measures identity-term bias of a text classifier
  (toxicity, sentiment, moderation) by comparing per-group mean scores, with a
  Mann-Whitney U significance test and a severity grade. It accepts any
  `score_fn(texts)` callable.

## Causal pathways

`CausalFairnessGraph` is a networkx directed acyclic graph of variables tagged
with fairness roles (protected, outcome, mediator, proxy). It enumerates every
path from a protected attribute to an outcome and labels each one:

- **Direct**: the outcome depends on the protected attribute with no intervening
  variable.
- **Proxy**: the effect flows through a variable that stands in for the protected
  attribute.
- **Indirect**: the effect flows through other mediators, which may be
  legitimate.

For numerical estimation, the DoWhy-backed causal task handlers (identify,
mediate, refute, counterfactual, attribute) are reachable from the platform task
consumer; DoWhy is an optional dependency, while the graph itself needs only
networkx.
