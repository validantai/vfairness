"""Task-type auto-detection in FairnessAnalyzer (VF-10).

The defect these tests pin: a single missing prediction turned a binary
CLASSIFIER into a REGRESSION analysis. ``np.unique`` reports NaN as a third
"value", the subset test against {0, 1, True, False} then fails, and the
analyzer silently reported mae_parity_difference / mean_prediction_difference
for a plainly 0/1 classifier, under a fairness heading, with no warning
anywhere that a different analysis had been chosen.

Two things are pinned here and both matter equally:

* the fix: detection reads the DEFINED values, so missing predictions no
  longer flip the task type, and an undetectable or ambiguous prediction
  column is disclosed instead of guessed at silently;
* the controls: unambiguous binary data and genuinely continuous data must
  come out NUMERICALLY unchanged and, for ordinary runs, silently. A library
  that warns on every run trains its users to ignore warnings, and one that
  answers "cannot tell" for everything is useless.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

# --------------------------------------------------------------------------
# Deterministic fixtures. The metric values pinned below were measured on the
# code BEFORE the detection fix and must survive it unchanged.
# --------------------------------------------------------------------------

ATTR = np.array(["group_a"] * 60 + ["group_b"] * 60)


def _binary_data():
    """A plain binary classifier: y_pred is exactly 0/1, 120 rows."""
    rng = np.random.default_rng(20260827)
    n = 120
    y_true = (rng.random(n) < 0.5).astype(float)
    y_pred = (rng.random(n) < 0.45).astype(float)
    return y_true, y_pred


def _continuous_data():
    """A plain regression: predictions on a wide continuous scale."""
    rng = np.random.default_rng(19990101)
    n = 120
    y_true = rng.normal(50.0, 10.0, n)
    y_pred = y_true + rng.normal(0.0, 3.0, n)
    return y_true, y_pred


def _with_missing(y_pred, k=15):
    holed = y_pred.copy()
    holed[:k] = np.nan
    return holed


CLASSIFICATION_KEYS = {
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equal_opportunity_difference",
    "equalized_odds_difference",
    "predictive_parity_difference",
}
REGRESSION_KEYS = {
    "mae_parity_difference",
    "mean_prediction_difference",
    "rmse_parity_difference",
}

# Measured on the pre-fix code with the fixtures above.
CLEAN_BINARY_METRICS = {
    "demographic_parity_difference": 0.15,
    "demographic_parity_ratio": 0.7,
    "equal_opportunity_difference": 0.206896551724,
    "equalized_odds_difference": 0.206896551724,
    "predictive_parity_difference": 0.057142857143,
}
CLEAN_CONTINUOUS_METRICS = {
    "mae_parity_difference": 0.094964078141,
    "mean_prediction_difference": 0.008324347843,
    "rmse_parity_difference": 0.284704003862,
}
# What the binary-with-NaN column is worth when the caller states the task
# type by hand. Auto-detection must land on exactly these numbers.
BINARY_WITH_NAN_METRICS = {
    "demographic_parity_difference": 0.116666666667,
    "demographic_parity_ratio": 0.75,
    "equal_opportunity_difference": 0.200626959248,
    "equalized_odds_difference": 0.200626959248,
    "predictive_parity_difference": 0.095238095238,
}


def _assert_metrics_match(actual, expected):
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert float(actual[name]) == pytest.approx(value, abs=1e-9), name


# --------------------------------------------------------------------------
# THE DEFECT: a missing prediction must not change the analysis
# --------------------------------------------------------------------------


def test_binary_with_missing_predictions_is_still_classification():
    y_true, y_pred = _binary_data()
    analyzer = FairnessAnalyzer(y_true, _with_missing(y_pred), ATTR)
    assert analyzer.task_type == "classification"


def test_binary_with_missing_predictions_reports_classification_metrics():
    """The visible harm: regression metrics under a fairness heading."""
    y_true, y_pred = _binary_data()
    metrics = FairnessAnalyzer(y_true, _with_missing(y_pred), ATTR).compute_all_metrics()
    assert set(metrics) == CLASSIFICATION_KEYS
    assert not (set(metrics) & REGRESSION_KEYS)


def test_auto_detection_equals_explicit_classification_on_the_same_rows():
    """Auto-detect must produce the explicit answer, on identical rows."""
    y_true, y_pred = _binary_data()
    holed = _with_missing(y_pred)
    auto = FairnessAnalyzer(y_true, holed, ATTR)
    explicit = FairnessAnalyzer(y_true, holed, ATTR, task_type="classification")

    # Same rows analysed: the fix touches detection only, never row selection.
    assert auto.data_info["final_size"] == explicit.data_info["final_size"] == 105
    assert auto.data_info["n_excluded"] == 15
    _assert_metrics_match(auto.compute_all_metrics(), BINARY_WITH_NAN_METRICS)
    _assert_metrics_match(explicit.compute_all_metrics(), BINARY_WITH_NAN_METRICS)


def test_binary_with_a_single_missing_prediction_is_still_classification():
    """One hole out of 120 was enough to flip the whole analysis."""
    y_true, y_pred = _binary_data()
    analyzer = FairnessAnalyzer(y_true, _with_missing(y_pred, k=1), ATTR)
    assert analyzer.task_type == "classification"


def test_boolean_predictions_with_missing_are_classification():
    y_true, y_pred = _binary_data()
    holed = _with_missing(y_pred)
    analyzer = FairnessAnalyzer(y_true.astype(bool), holed, ATTR)
    assert analyzer.task_type == "classification"


# --------------------------------------------------------------------------
# HONESTY: undetectable and ambiguous columns are disclosed, not guessed at
# --------------------------------------------------------------------------


def test_all_missing_predictions_warns_that_detection_was_impossible():
    y_true, _ = _binary_data()
    with pytest.warns(UserWarning) as record:
        FairnessAnalyzer(y_true, np.full(120, np.nan), ATTR)
    messages = [str(w.message) for w in record]
    assert any("could not be auto-detected" in m for m in messages), messages
    assert any("task_type=" in m for m in messages), messages


def test_ambiguous_small_integer_labels_are_disclosed():
    """{0, 1, 2} is neither cleanly binary nor cleanly continuous."""
    rng = np.random.default_rng(5)
    y_true = rng.integers(0, 3, 120).astype(float)
    y_pred = rng.integers(0, 3, 120).astype(float)
    with pytest.warns(UserWarning) as record:
        analyzer = FairnessAnalyzer(y_true, y_pred, ATTR)
    assert analyzer.task_type == "regression"
    messages = [str(w.message) for w in record]
    assert any("ambiguous" in m for m in messages), messages
    assert any("task_type=" in m for m in messages), messages


def test_ambiguous_integer_labels_with_missing_values_are_still_disclosed():
    rng = np.random.default_rng(6)
    y_true = rng.integers(0, 3, 120).astype(float)
    y_pred = rng.integers(0, 3, 120).astype(float)
    with pytest.warns(UserWarning, match="ambiguous"):
        FairnessAnalyzer(y_true, _with_missing(y_pred), ATTR)


# --------------------------------------------------------------------------
# OVER-CORRECTION CONTROLS: ordinary runs are unchanged, and silent
# --------------------------------------------------------------------------


def test_control_clean_binary_still_classification_and_silent():
    y_true, y_pred = _binary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FairnessAnalyzer(y_true, y_pred, ATTR)
    assert analyzer.task_type == "classification"
    assert [str(w.message) for w in caught] == []


def test_control_clean_binary_metrics_are_numerically_unchanged():
    y_true, y_pred = _binary_data()
    analyzer = FairnessAnalyzer(y_true, y_pred, ATTR)
    _assert_metrics_match(analyzer.compute_all_metrics(), CLEAN_BINARY_METRICS)
    assert analyzer.data_info["final_size"] == 120
    assert analyzer.data_info["n_excluded"] == 0


def test_control_continuous_still_regression_and_silent():
    y_true, y_pred = _continuous_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FairnessAnalyzer(y_true, y_pred, ATTR)
    assert analyzer.task_type == "regression"
    assert [str(w.message) for w in caught] == []


def test_control_continuous_metrics_are_numerically_unchanged():
    y_true, y_pred = _continuous_data()
    analyzer = FairnessAnalyzer(y_true, y_pred, ATTR)
    _assert_metrics_match(analyzer.compute_all_metrics(), CLEAN_CONTINUOUS_METRICS)
    assert analyzer.data_info["final_size"] == 120


def test_control_continuous_with_missing_is_still_regression_and_silent():
    y_true, y_pred = _continuous_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FairnessAnalyzer(y_true, _with_missing(y_pred), ATTR)
    assert analyzer.task_type == "regression"
    # ASSERTS NO *DETECTION* WARNING, not global silence. This read
    # `[str(w.message) for w in caught] == []` until 2026-09-27. validate_inputs now
    # discloses how much of the input survived missing_strategy='exclude', which is a
    # different concern from task-type detection, and a blanket silence assertion made
    # this test fail on a disclosure it was never about.
    detection_warnings = [
        str(w.message) for w in caught if "validate_inputs:" not in str(w.message)
    ]
    assert detection_warnings == [], detection_warnings
    assert analyzer.data_info["final_size"] == 105


def test_control_single_valued_binary_column_is_classification():
    """All-zero predictions are a degenerate classifier, not a regression."""
    y_true, _ = _binary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FairnessAnalyzer(y_true, np.zeros(120), ATTR)
    assert analyzer.task_type == "classification"
    assert [str(w.message) for w in caught] == []


def test_control_single_valued_non_binary_column_is_regression():
    y_true, _ = _continuous_data()
    analyzer = FairnessAnalyzer(y_true, np.full(120, 7.0), ATTR)
    assert analyzer.task_type == "regression"


def test_control_explicit_task_type_is_never_second_guessed():
    """A stated task type skips detection entirely, DETECTION warnings included.

    This asserted total silence until 2026-09-27, and that was the wrong assertion for
    this particular input. Every one of the 120 predictions here is NaN, so
    missing_strategy='exclude' drops all of them, and validate_inputs now says so:
    "120 of 120 row(s) were excluded ... 0 row(s) (0.0 percent of the input) carry
    every number computed from this call". That is not noise beside the test's subject,
    it is the single most important thing that can be said about this call, and the old
    assertion required it to be absent.

    So the silence assertion is narrowed to what this test is actually about, and the
    exclusion disclosure is asserted POSITIVELY rather than merely tolerated.
    """
    y_true, y_pred = _binary_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer = FairnessAnalyzer(y_true, np.full(120, np.nan), ATTR, task_type="regression")
    assert analyzer.task_type == "regression"
    messages = [str(w.message) for w in caught]
    assert [m for m in messages if "validate_inputs:" not in m] == [], messages
    assert any("120 of 120" in m and "excluded" in m for m in messages), (
        f"every row was dropped and nothing said so: {messages}"
    )
    assert analyzer.data_info["final_size"] == 0


def test_control_probability_predictions_still_warn_about_probabilities():
    """The pre-existing y_prob disclosure must survive the detection change."""
    rng = np.random.default_rng(31)
    y_true = (rng.random(120) < 0.5).astype(float)
    with pytest.warns(UserWarning, match="PROBABILITIES"):
        analyzer = FairnessAnalyzer(y_true, rng.random(120), ATTR)
    assert analyzer.task_type == "regression"


def test_probability_predictions_with_missing_values_also_warn():
    """NaN used to defeat the isfinite() check and suppress that disclosure."""
    rng = np.random.default_rng(32)
    y_true = (rng.random(120) < 0.5).astype(float)
    with pytest.warns(UserWarning, match="PROBABILITIES"):
        FairnessAnalyzer(y_true, _with_missing(rng.random(120)), ATTR)
