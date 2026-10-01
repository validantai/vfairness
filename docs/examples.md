# Executable examples

These examples are verified in the test suite: `tests/test_docs_examples.py`
runs every `>>>` block below with the standard `doctest` runner, so if the
library API changes in a way that breaks a snippet, the suite fails.

The examples deliberately assert on stable, structural facts (types, bounds,
key sets) rather than printing raw metric values, so they stay deterministic
across platforms and numpy versions.

## Core metric: demographic parity difference

Compute a single fairness metric from label, prediction and protected-attribute
arrays. The result is a non-negative float in the range `[0, 1]`: it is the
max-minus-min spread of the per-group selection rates, so 0 means perfect
demographic parity.

```python
>>> import numpy as np
>>> import vfairness as vf
>>> rng = np.random.default_rng(0)
>>> n = 200
>>> y_true = rng.integers(0, 2, n)
>>> y_pred = rng.integers(0, 2, n)
>>> groups = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
>>> dp = vf.demographic_parity_difference(y_true, y_pred, groups)
>>> 0.0 <= float(dp) <= 1.0
True

```

## Unified analyzer report

`FairnessAnalyzer` auto-detects the task type and produces a structured report.

```python
>>> from vfairness import FairnessAnalyzer
>>> analyzer = FairnessAnalyzer(y_true, y_pred, groups)
>>> analyzer.task_type
'classification'
>>> report = analyzer.get_report(include_ci=False)
>>> sorted(report.keys())
['assessment', 'data_info', 'group_stats', 'methodology_version', 'metrics', 'task_type', 'thresholds_used']
>>> report["task_type"]
'classification'
>>> "demographic_parity_difference" in report["metrics"]
True
>>> report["data_info"]["n_samples"]
200

```

## Group inspection

The analyzer exposes the protected groups it found and their sizes.

```python
>>> sorted(analyzer.groups)
['A', 'B']
>>> sizes = analyzer.group_sizes
>>> sum(sizes.values()) == analyzer.n_samples
True

```

## Regression metrics

The same interface handles regression targets; parity-of-error metrics are
non-negative.

```python
>>> rng2 = np.random.default_rng(7)
>>> y_reg = rng2.normal(100.0, 20.0, n)
>>> pred_reg = y_reg + rng2.normal(0.0, 10.0, n)
>>> mae = vf.mae_parity_difference(y_reg, pred_reg, groups)
>>> float(mae) >= 0.0
True

```
