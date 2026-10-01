"""READINESS-6: the threshold optimizer graded on quantities that had no value.

`_compute_performance_metrics` returned ``0.0`` for every undefined metric, and
0.0 is not neutral for these five: it is the WORST attainable score. Measured on
this repo before the fix:

===============================  ==========================================
data                             what was reported
===============================  ==========================================
``y_pred`` all zeros             ``precision 0.0``, though no positive was
                                 ever predicted, so precision has no value
``y_true`` all ones, perfect     ``balanced_accuracy 0.5``, "no better than
classifier                       chance", because ``tnr`` was undefined and
                                 contributed 0
``y_true`` all zeros, perfect    ``recall 0.0`` and ``f1_score 0.0``, so
classifier                       optimising for f1 ranked every candidate at
                                 0.0 and the frontier was decided by a metric
                                 nobody could compute
===============================  ==========================================

The third matters most in this package, because these metrics are also computed
PER GROUP: a small group with no positive labels got a fabricated recall of 0.0
and read as catastrophically underserved when nothing about it was measured.

Downstream, three consumers turned that into a decision:

* Pareto dominance compared ``.get(obj, 0)``, so an objective nobody computed
  made its own candidate look maximally bad and dropped it off the frontier;
* the selection summed ``.get(obj, 0)``, and that sum picks ``result_``, the
  thresholds a caller deploys;
* ``find_optimal_threshold`` compared ``obj_value > best_objective``, which is
  False for NaN, so the FIRST feasible threshold set the bar to NaN, nothing
  could ever beat it, and whatever came first in the sweep was returned as the
  optimum.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from vfairness.post_processing.threshold_optimization.analyzer import ThresholdAnalyzer
from vfairness.post_processing.threshold_optimization.optimizer import (
    _compute_performance_metrics,
)


def _metrics(y_true, y_pred, weight=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _compute_performance_metrics(np.asarray(y_true), np.asarray(y_pred), weight)


# ------------------------------------------------------- the metrics themselves


def test_precision_has_no_value_when_no_positive_was_predicted():
    m = _metrics([1, 1, 0, 0], [0, 0, 0, 0])
    assert math.isnan(m["precision"]), (
        "0.0 says every positive call was wrong; no positive call was MADE"
    )


def test_a_perfect_classifier_is_not_reported_as_chance_level():
    """The sharpest one. No negative label exists, so `tnr` is undefined and
    contributed 0 to `(tpr + tnr) / 2`, halving a perfect score."""
    m = _metrics([1, 1, 1, 1], [1, 1, 1, 1])
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert math.isnan(m["balanced_accuracy"]), (
        f"balanced_accuracy is {m['balanced_accuracy']!r} for a PERFECT classifier; "
        f"0.5 reads as no better than chance"
    )


def test_recall_and_f1_have_no_value_when_no_positive_label_exists():
    m = _metrics([0, 0, 0, 0], [0, 0, 0, 0])
    assert m["accuracy"] == 1.0
    assert math.isnan(m["recall"])
    assert math.isnan(m["f1_score"])


def test_a_measured_zero_recall_still_produces_a_measured_zero_f1():
    """OVER-CORRECTION CONTROL, and the one that matters most here.

    F1 is 2PR/(P+R). The numerator is zero whenever EITHER component is a
    measured zero, so F1 is measured too. A model that predicts no positives has
    recall 0, which is measured and means it found NONE of the real positives.
    Turning F1 into a could-not-check there would hide a complete failure on the
    positive class behind a shrug.
    """
    m = _metrics([1, 1, 0, 0], [0, 0, 0, 0])
    assert m["recall"] == 0.0, "recall is measured here: positives exist and none were found"
    assert m["f1_score"] == 0.0, "the finding was converted into a could-not-check"


def test_ordinary_data_is_untouched():
    """OVER-CORRECTION CONTROL."""
    m = _metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert m == {
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5,
        "balanced_accuracy": 0.5,
    }


def test_the_undefined_metrics_are_named_out_loud():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _compute_performance_metrics(np.zeros(4, dtype=int), np.zeros(4, dtype=int))
    messages = [str(w.message) for w in caught]
    assert messages, "silently returning NaN is only half a disclosure"
    joined = " ".join(messages)
    assert "NOT as 0.0" in joined
    assert "recall (the data holds no positive label)" in joined


def test_per_group_metrics_carry_the_same_honesty():
    """The consequence that reaches a fairness report: a small group with no
    positive labels must not read as catastrophically underserved."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.2, 0.1])
    groups = np.array(["A", "A", "A", "A", "B", "B"])  # group B has no positives

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = ThresholdAnalyzer(y_true, y_prob, groups).analyze_threshold(0.5)

    assert math.isnan(result.group_metrics["B"]["recall"]), (
        "group B has no positive label, so its recall has no value; 0.0 would "
        "read as 'this group is never served' and it was never measured"
    )
    assert result.group_metrics["A"]["recall"] == 1.0, "group A is measured and must stay so"


# ------------------------------------------------- find_optimal_threshold


def _no_negative_labels(n=400):
    """Every label positive, so `tnr` and therefore `balanced_accuracy` have no
    value at ANY threshold, while accuracy still does.

    An earlier version of this fixture used no POSITIVE labels and asked for
    f1_score, and that premise was wrong: with only negative labels, a low
    threshold predicts positives that are all false, so precision is a MEASURED
    0.0 and F1 is measured with it. The code was right and the test was wrong.
    Choosing a fixture where the objective is genuinely undefined at every
    threshold is what makes this test about the thing it names.
    """
    rng = np.random.default_rng(0)
    return (
        np.ones(n, dtype=int),
        rng.uniform(0, 1, n),
        np.array(["a"] * (n // 2) + ["b"] * (n // 2)),
    )


def test_an_uncomputable_objective_yields_no_optimum():
    """`obj_value > best_objective` is False for NaN, so the FIRST feasible
    threshold set the bar and nothing could ever beat it. Whatever came first in
    the sweep was returned as `optimal_threshold`."""
    analyzer = ThresholdAnalyzer(*_no_negative_labels())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = analyzer.find_optimal_threshold(
            objective="balanced_accuracy", constraint="demographic_parity"
        )

    assert out["optimal_threshold"] is None, (
        f"returned {out['optimal_threshold']!r} as the optimum on an objective that has "
        f"no value at any threshold"
    )
    assert out["n_thresholds_objective_unmeasurable"] > 0
    assert any("none could be ranked" in str(w.message) for w in caught), [
        str(w.message) for w in caught
    ]


def test_feasibility_is_still_reported_as_measured_true():
    """The over-correction I made myself, caught by running the fix.

    The first version returned `is_feasible=False` here. Ninety of the hundred
    searched thresholds SATISFIED the constraint; what could not be established
    is which of them is best. Reporting infeasible is a fabricated negative built
    out of the objective's silence.
    """
    analyzer = ThresholdAnalyzer(*_no_negative_labels())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = analyzer.find_optimal_threshold(
            objective="balanced_accuracy", constraint="demographic_parity"
        )
    assert out["is_feasible"] is True, (
        "feasible thresholds exist and were measured; only the ranking failed"
    )


def test_a_computable_objective_still_finds_its_optimum():
    """OVER-CORRECTION CONTROL. accuracy IS defined on the same data."""
    analyzer = ThresholdAnalyzer(*_no_negative_labels())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = analyzer.find_optimal_threshold(objective="accuracy", constraint="demographic_parity")
    assert out["optimal_threshold"] is not None
    assert out["is_feasible"] is True
    assert out["metrics"]["accuracy"] > 0.9
    assert out["n_thresholds_objective_unmeasurable"] == 0


def test_a_normal_sweep_is_unchanged():
    """OVER-CORRECTION CONTROL on ordinary data."""
    rng = np.random.default_rng(1)
    n = 400
    y_prob = rng.uniform(0, 1, n)
    y_true = (y_prob + rng.normal(0, 0.2, n) > 0.5).astype(int)
    groups = np.array(["a"] * (n // 2) + ["b"] * (n // 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = ThresholdAnalyzer(y_true, y_prob, groups).find_optimal_threshold(
            objective="balanced_accuracy", constraint="demographic_parity"
        )
    assert out["optimal_threshold"] is not None
    assert out["n_thresholds_objective_unmeasurable"] == 0
    assert 0.0 < out["metrics"]["f1_score"] <= 1.0
