# Training-Time Fairness Interventions

This documentation covers the `vfairness.in_processing` module, which provides comprehensive tools for fairness-aware model training. These in-processing techniques intervene directly during the machine learning training process to produce models that are fair by design.

## Overview

Standard model training is fairness-blind. Optimization algorithms like gradient descent only seek to minimize prediction error, which can amplify spurious correlations between sensitive attributes and outcomes. Training-time interventions address this by modifying the learning process to incorporate fairness objectives.

### Quick Start

```python
# Scikit-learn compatible approach
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing import FairClassifier

clf = FairClassifier(
    base_estimator=RandomForestClassifier(),
    fairness_constraint='demographic_parity',
    tolerance=0.05
)
clf.fit(X_train, y_train, sensitive_attr=gender)
y_pred = clf.predict(X_test)
print(f"Constraint satisfied: {clf.fairness_result_.constraint_satisfied}")
```

The `in_processing` module offers five main approaches:

1. **Fairness-Aware Loss Functions** - PyTorch losses that penalize fairness violations
2. **Constraint-Based Training** - Algorithms that enforce hard fairness constraints
3. **Fairness Regularizers** - Penalty terms that can be added to any loss
4. **Group-Specific Calibrators** - Trainable calibration for group-wise probability calibration
5. **Scikit-Learn Wrappers** - Easy-to-use wrappers for standard ML workflows

---

## 1. Fairness-Aware Loss Functions

> **Requires PyTorch.** The loss classes in this section need `torch`, which is
> not part of the base install: `pip install "vfairness[training]"` (or
> `pip install torch`). Constructing any of them without torch raises an
> `ImportError` that points you to `pip install torch`.

### Concept

Fairness-aware loss functions incorporate fairness penalties directly into the training objective:

```
L_total = L_task + λ * L_fairness
```

Where:
- `L_task` is the primary prediction loss (e.g., BCE, MSE)
- `L_fairness` measures fairness violations
- `λ` controls the accuracy-fairness trade-off

### Available Loss Functions

#### Group Fairness Losses

| Loss Function | Criterion | Use Case |
|---------------|-----------|----------|
| `DemographicParityLoss` | Equal positive prediction rates | When outcome should be independent of group |
| `EqualizedOddsLoss` | Equal TPR and FPR | When error rates should be equal |
| `EqualOpportunityLoss` | Equal TPR (true positive rates) | When qualifying individuals should have equal chances |
| `FalsePositiveRateParityLoss` | Equal FPR | When false accusations should be equally rare |
| `BoundedGroupLoss` | Bounded worst-group loss | For minimax fairness |

#### Adversarial Losses

| Loss Function | Mechanism | Use Case |
|---------------|-----------|----------|
| `AdversarialDebiasingLoss` | Adversary predicts sensitive attr | When you want to remove information leakage |
| `ProjectedAdversarialLoss` | Gradient projection | For more stable adversarial training |
| `FairRepresentationLoss` | Fair representation learning | When learning intermediate representations |

#### Counterfactual Losses

| Loss Function | Criterion | Use Case |
|---------------|-----------|----------|
| `CounterfactualFairnessLoss` | Same prediction under intervention | When causal fairness is required |
| `IndividualFairnessLoss` | Similar individuals, similar predictions | For individual-level fairness |
| `CausalFairnessLoss` | Block unfair causal pathways | When causal structure is known |

### Usage Examples

#### Basic Usage with Demographic Parity

```python
import torch
from vfairness.in_processing.loss_functions import DemographicParityLoss

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

#### Equalized Odds with Custom Weights

```python
from vfairness.in_processing.loss_functions import EqualizedOddsLoss

loss_fn = EqualizedOddsLoss(
    lambda_fairness=0.15,
    tpr_weight=1.0,    # Weight for TPR component
    fpr_weight=0.5,    # Weight for FPR component
)
```

#### Adversarial Debiasing

```python
from vfairness.in_processing.loss_functions import AdversarialDebiasingLoss

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

---

## 2. Constraint-Based Training

### The Reductions Approach

The constraint-based approach from Agarwal et al. (2018) reduces fair classification to a sequence of cost-sensitive classification problems. This enables using any standard classifier while achieving fairness guarantees.

### Available Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `ExponentiatedGradient` | Lagrangian saddle-point optimization | General-purpose constrained training |
| `GridSearch` | Grid search over Lagrange multipliers | Quick exploration of trade-offs |
| `ThresholdOptimizer` | Post-processing threshold optimization | When model is already trained |

### Constraints

| Constraint | Mathematical Definition |
|------------|-------------------------|
| `DemographicParityConstraint` | \|P(ŷ=1\|G=a) - P(ŷ=1\|G=b)\| ≤ ε |
| `EqualizedOddsConstraint` | \|TPR_a - TPR_b\| ≤ ε AND \|FPR_a - FPR_b\| ≤ ε |
| `EqualOpportunityConstraint` | \|TPR_a - TPR_b\| ≤ ε |
| `BoundedGroupLossConstraint` | L_g ≤ (1+ε) * L_overall for all g |

### Usage Examples

#### Exponentiated Gradient Algorithm

```python
from sklearn.linear_model import LogisticRegression
from vfairness.in_processing.constraints import (
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
print(f"Satisfied: {result.optimization_result.converged}")
```

#### Threshold Optimization

```python
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing.constraints import (
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

---

## 3. Fairness Regularizers

### Concept

Regularizers provide a modular way to add fairness penalties to any differentiable loss function. The regularizer classes construct without PyTorch installed, but calling them (as in the training loops below) operates on torch tensors, so the loops need `torch` as well:

```python
total_loss = task_loss + regularizer(y_pred, sensitive_attr)
```

### Available Regularizers

| Regularizer | What It Measures | Use Case |
|-------------|-----------------|----------|
| `StatisticalParityRegularizer` | Difference in mean predictions | Demographic parity |
| `ConditionalIndependenceRegularizer` | Conditional dependence given y | Equalized odds |
| `GroupFairnessRegularizer` | Configurable group metrics | Flexible fairness criteria |
| `HilbertSchmidtRegularizer` | HSIC-based independence | Non-linear dependence |
| `CorrelationPenalty` | Pearson correlation | Simple linear dependence |

### Usage Examples

#### Statistical Parity Regularizer

```python
from vfairness.in_processing.regularizers import StatisticalParityRegularizer

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

#### HSIC Regularizer for Non-Linear Independence

```python
from vfairness.in_processing.regularizers import HilbertSchmidtRegularizer

regularizer = HilbertSchmidtRegularizer(
    strength=0.1,
    kernel='rbf',
    sigma=1.0
)

# HSIC captures non-linear statistical dependence
penalty = regularizer(y_pred, sensitive_attr)
```

---

## 4. Group-Specific Calibrators

> **Requires PyTorch.** `TrainableGroupCalibrator` and
> `CalibrationAwareTrainer` need `torch`, which is not part of the base
> install: `pip install "vfairness[training]"` (or `pip install torch`).

### Concept

Group-specific calibrators learn separate calibration parameters for each demographic group, ensuring that probability predictions have the same meaning across groups.

### Available Calibrators

| Calibrator | Parameters | Description |
|------------|------------|-------------|
| `TemperatureScalingCalibrator` | T per group | Divides logits by temperature |
| `PlattScalingCalibrator` | (a, b) per group | Linear transform in log-odds space |
| `BetaCalibrator` | (c, d, e) per group | More flexible beta calibration |
| `FocalCalibrator` | γ per group | Focal-loss inspired calibration |

### Usage Example

```python
from vfairness.in_processing.calibrators import (
    TrainableGroupCalibrator,
    CalibrationAwareTrainer,
)

# Create calibrator
calibrator = TrainableGroupCalibrator(
    n_groups=2,
    method='temperature',
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

---

## 5. Scikit-Learn Wrappers

### FairClassifier

The `FairClassifier` wrapper makes it easy to add fairness constraints to any scikit-learn classifier.

```python
from sklearn.ensemble import RandomForestClassifier
from vfairness.in_processing.wrappers import FairClassifier

# Create fair classifier
clf = FairClassifier(
    base_estimator=RandomForestClassifier(n_estimators=100),
    fairness_constraint='equalized_odds',
    tolerance=0.05,
    method='reductions'  # or 'threshold', 'grid_search'
)

# Fit (requires sensitive_attr)
clf.fit(X_train, y_train, sensitive_attr=gender)

# Predict
y_pred = clf.predict(X_test)

# Check fairness results
print(f"Accuracy: {clf.fairness_result_.accuracy:.4f}")
print(f"Violation: {clf.fairness_result_.fairness_violation:.4f}")
print(f"Satisfied: {clf.fairness_result_.constraint_satisfied}")
```

### FairRegressor

```python
from sklearn.linear_model import Ridge
from vfairness.in_processing.wrappers import FairRegressor

reg = FairRegressor(
    base_estimator=Ridge(),
    fairness_constraint='mean_parity',   # the only constraint implemented
    tolerance=0.1
)

reg.fit(X_train, y_train, sensitive_attr=group)

# mean_parity is enforced by a PER-GROUP offset, so applying it needs the
# sensitive attribute. reg.predict(X_test) returns the UNADJUSTED base
# predictions and warns: it has no group information to apply the offset with.
y_pred = reg.predict_with_sensitive_attr(X_test, group_test)
```

Measured on a 1000-row synthetic set where the group is a feature, with
`tolerance=0.1`: `predict()` left a group mean gap of **4.97**, while
`predict_with_sensitive_attr()` produced **0.08**. Reach for `predict()` only
when you deliberately want the unmitigated baseline to compare against.

---

## 6. FairnessTrainingAnalyzer

### Comprehensive Analysis

The `FairnessTrainingAnalyzer` provides end-to-end analysis of fairness-aware training options.

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

# Export to JSON
report.to_json('training_analysis.json')

# Render as SVG
report.to_svg('training_report.svg')
```

### Report Contents

The analysis report includes:

- **Baseline Performance**: Accuracy and fairness metrics without constraints
- **Method Comparisons**: Side-by-side comparison of different approaches
- **Trade-off Analysis**: Pareto frontier of accuracy vs. fairness
- **Recommendation**: Suggested method with rationale
- **Critical Issues**: Problems that need attention
- **Action Items**: Prioritized next steps

---

## Best Practices

### Choosing a Method

| Scenario | Recommended Approach |
|----------|---------------------|
| PyTorch deep learning | Fairness-aware loss functions |
| Scikit-learn classifier | FairClassifier with reductions |
| Need hard constraint guarantee | ExponentiatedGradient |
| Quick exploration | GridSearch or ThresholdOptimizer |
| Non-linear feature dependence | HSIC regularizer |
| Want representation-level fairness | AdversarialDebiasingLoss |

### Setting λ (Lambda)

The `lambda_fairness` parameter controls the accuracy-fairness trade-off:

- **λ = 0**: Pure accuracy optimization (unfair)
- **λ = 0.01-0.1**: Light fairness penalty
- **λ = 0.1-0.5**: Moderate penalty
- **λ = 0.5-1.0**: Strong fairness emphasis
- **λ > 1.0**: Fairness dominates (may hurt accuracy)

**Recommendation**: Start with λ = 0.1 and use `FairnessTrainingAnalyzer` to explore the trade-off curve.

### Warmup Strategy

For loss functions, use `warmup_epochs` to stabilize early training:

```python
loss_fn = DemographicParityLoss(
    lambda_fairness=0.1,
    warmup_epochs=5  # No fairness penalty for first 5 epochs
)
```

### Monitoring Training

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

## Mathematical Background

### Lagrangian Formulation

Constrained fair classification can be written as:

```
min_θ max_λ≥0 L(θ) + Σᵢ λᵢ * gᵢ(θ)
```

Where:
- `L(θ)` is the prediction loss
- `gᵢ(θ)` are constraint violations
- `λᵢ` are Lagrange multipliers

The ExponentiatedGradient algorithm solves this using:
1. **Primal update**: Train classifier with cost-sensitive weights
2. **Dual update**: Update λ using multiplicative weights

### Differentiable Fairness Metrics

Since hard metrics like TPR are non-differentiable, we use soft approximations:

**Soft TPR**:
```
TPR_soft = E[ŷ | y=1] ≈ mean(ŷ[y==1])
```

**Soft FPR**:
```
FPR_soft = E[ŷ | y=0] ≈ mean(ŷ[y==0])
```

These soft metrics are differentiable and can be optimized with gradient descent.

---

## References

1. Hardt, M., Price, E., & Srebro, N. (2016). Equality of Opportunity in Supervised Learning. NeurIPS.

2. Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., & Wallach, H. (2018). A Reductions Approach to Fair Classification. ICML.

3. Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating Unwanted Biases with Adversarial Learning. AIES.

4. Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2017). Counterfactual Fairness. NeurIPS.

5. Zafar, M. B., Valera, I., Gomez Rodriguez, M., & Gummadi, K. P. (2017). Fairness Constraints: Mechanisms for Fair Classification. AISTATS.

6. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. ICML.

---

## API Reference Summary

### Loss Functions (`vfairness.in_processing.loss_functions`)

| Class | Description |
|-------|-------------|
| `DemographicParityLoss` | Penalizes differences in positive prediction rates |
| `EqualizedOddsLoss` | Penalizes differences in TPR and FPR |
| `EqualOpportunityLoss` | Penalizes differences in TPR only |
| `FalsePositiveRateParityLoss` | Penalizes differences in FPR only |
| `BoundedGroupLoss` | Bounds worst-group loss (minimax fairness) |
| `AdversarialDebiasingLoss` | Adversarial training for fair representations |
| `ProjectedAdversarialLoss` | Gradient projection adversarial approach |
| `FairRepresentationLoss` | Fair representation learning loss |
| `CounterfactualFairnessLoss` | Counterfactual fairness penalties |
| `IndividualFairnessLoss` | Lipschitz-based individual fairness |
| `CausalFairnessLoss` | Causal pathway fairness |
| `create_fairness_loss()` | Factory function for creating losses |

### Constraints (`vfairness.in_processing.constraints`)

| Class | Description |
|-------|-------------|
| `DemographicParityConstraint` | Constraint for demographic parity |
| `EqualizedOddsConstraint` | Constraint for equalized odds |
| `EqualOpportunityConstraint` | Constraint for equal opportunity |
| `FalsePositiveRateParityConstraint` | Constraint for FPR parity |
| `BoundedGroupLossConstraint` | Constraint for bounded group loss |
| `ExponentiatedGradient` | Main reductions algorithm |
| `GridSearch` | Grid search over Lagrange multipliers |
| `ThresholdOptimizer` | Post-processing threshold optimization |
| `create_constraint()` | Factory function for creating constraints |

### Regularizers (`vfairness.in_processing.regularizers`)

| Class | Description |
|-------|-------------|
| `StatisticalParityRegularizer` | Penalizes mean prediction differences |
| `ConditionalIndependenceRegularizer` | Enforces ŷ ⊥ a \| y |
| `GroupFairnessRegularizer` | Flexible group fairness penalty |
| `HilbertSchmidtRegularizer` | HSIC-based independence |
| `CorrelationPenalty` | Simple Pearson correlation penalty |
| `create_regularizer()` | Factory function for creating regularizers |

### Calibrators (`vfairness.in_processing.calibrators`)

| Class | Description |
|-------|-------------|
| `TemperatureScalingCalibrator` | Group-specific temperature scaling |
| `PlattScalingCalibrator` | Group-specific Platt scaling |
| `BetaCalibrator` | Group-specific beta calibration |
| `FocalCalibrator` | Group-specific focal calibration |
| `TrainableGroupCalibrator` | Unified trainable calibrator |
| `CalibrationAwareTrainer` | Helper for joint model-calibration training |
| `create_group_calibrator()` | Factory function for creating calibrators |

### Wrappers (`vfairness.in_processing.wrappers`)

| Class | Description |
|-------|-------------|
| `FairClassifier` | Sklearn-compatible fairness-aware classifier |
| `FairRegressor` | Sklearn-compatible fairness-aware regressor |
| `make_fair_classifier()` | Factory function for classifiers |
| `make_fair_regressor()` | Factory function for regressors |

### Analyzer (`vfairness.in_processing`)

| Class | Description |
|-------|-------------|
| `FairnessTrainingAnalyzer` | Comprehensive training analysis |
| `FairnessTrainingReport` | Analysis report with `summary()`, `to_dict()`, `to_json()`, `to_svg()` |
| `MethodComparison` | Comparison results for training methods |
| `TrainingRecommendation` | Training method recommendation |

---

## Integration with vfairness Pipeline

The `in_processing` module integrates seamlessly with the broader vfairness fairness pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vfairness Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│  1. PRE-PROCESSING         │  Bias Detection & Feature Engineering │
│  ─────────────────────────────────────────────────────────────────  │
│  2. IN-PROCESSING  ←──────── This module (training interventions)  │
│  ─────────────────────────────────────────────────────────────────  │
│  3. POST-PROCESSING        │  Calibration & Threshold Adjustment   │
│  ─────────────────────────────────────────────────────────────────  │
│  4. EVALUATION             │  Fairness Metrics & Auditing          │
│  ─────────────────────────────────────────────────────────────────  │
│  5. CI/CD INTEGRATION      │  Automated Fairness Testing           │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: Full Pipeline

```python
from vfairness.preprocessing.bias_detection import BiasDetector
from vfairness.in_processing import FairClassifier, FairnessTrainingAnalyzer
from vfairness.evaluation import FairnessAnalyzer

# 1. Pre-processing: Detect bias
detector = BiasDetector(df, protected_attributes=["gender"], outcome_column="approved")
audit_report = detector.full_audit()

# 2. In-processing: Fair training
clf = FairClassifier(
    base_estimator=LogisticRegression(),
    fairness_constraint='demographic_parity'
)
clf.fit(X_train, y_train, sensitive_attr=gender_train)

# 3. Evaluation: Assess fairness
y_pred = clf.predict(X_test)
evaluator = FairnessAnalyzer(y_test, y_pred, sensitive_attr=gender_test)
eval_report = evaluator.get_report(include_ci=True)
print(eval_report["metrics"])
```
