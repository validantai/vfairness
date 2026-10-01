"""Fail-closed pins for the frozen validation options (audit finding #6).

An unrecognised ``missing_strategy`` used to fall through to the 'exclude'
branch, so ``missing_strategy='raise'`` (the obvious typo for 'error') ran a
fail-OPEN audit that silently dropped every row with a missing protected
attribute. docs/API_STABILITY.md names ``missing_strategy`` as part of the
frozen ``FairnessAnalyzer`` constructor surface and promises
``ConfigurationError`` for "an unknown option", so the documented behaviour and
the real behaviour disagreed in the dangerous direction.

The same silent fallback existed for ``task_type``: anything that was not
exactly 'classification' skipped the binary-label validation entirely.

These tests pin BOTH directions: a typo raises, and every legitimate value
still behaves exactly as it did before (the over-correction control).
"""

import numpy as np
import pandas as pd
import pytest

from vfairness.evaluation.vfairness_metrics._validation import (
    handle_missing_values,
    validate_inputs,
)
from vfairness.exceptions import ConfigurationError, InvalidDataError

# 300 rows, 60 of them with a missing protected attribute: the shape the audit
# used to show that a typo'd strategy silently dropped 20% of the data.
_N = 300
_N_MISSING = 60


def _data_with_missing_attr():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, _N).astype(float)
    y_pred = rng.integers(0, 2, _N).astype(float)
    sensitive = np.array(["a" if i % 2 else "b" for i in range(_N)], dtype=object)
    sensitive[:_N_MISSING] = np.nan
    return y_true, y_pred, sensitive


# --------------------------------------------------------------------------
# missing_strategy: unknown values must FAIL CLOSED
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bogus", ["raise", "nonsense", "EXCLUDE", "Error", "", "drop"])
def test_handle_missing_values_rejects_unknown_strategy(bogus):
    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(ConfigurationError) as exc:
        handle_missing_values(y_true, y_pred, sensitive, bogus)
    message = str(exc.value)
    assert "missing_strategy" in message
    # the message must name the accepted values, or the user cannot fix the typo
    for accepted in ("exclude", "as_group", "error"):
        assert accepted in message


@pytest.mark.parametrize("bogus", ["raise", "nonsense", "EXCLUDE"])
def test_validate_inputs_rejects_unknown_strategy(bogus):
    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(ConfigurationError):
        validate_inputs(y_true, y_pred, sensitive, missing_strategy=bogus)


def test_validate_inputs_intersectional_rejects_unknown_strategy():
    """The intersectional branch had its own copy of the same fall-through."""
    frame = pd.DataFrame(
        {
            "g": ["m", "f", np.nan, "m", "f", "m"],
            "r": ["x", "y", "x", np.nan, "y", "x"],
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    with pytest.raises(ConfigurationError):
        validate_inputs(y, y, frame, task_type="classification", missing_strategy="raise")


def test_metric_function_rejects_unknown_strategy():
    """The public metric entry points inherit the guard through validate_inputs."""
    from vfairness.evaluation.vfairness_metrics.classification import (
        demographic_parity_difference,
    )

    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(ConfigurationError):
        demographic_parity_difference(y_true, y_pred, sensitive, missing_strategy="raise")


def test_analyzer_constructor_rejects_unknown_strategy():
    """missing_strategy is frozen constructor surface: the typo must not construct."""
    from vfairness.evaluation.vfairness_metrics.analyzer import FairnessAnalyzer

    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(ConfigurationError):
        FairnessAnalyzer(y_true, y_pred, sensitive, missing_strategy="nonsense")


def test_configuration_error_stays_a_value_error():
    """API_STABILITY.md promises the subclasses keep working with except ValueError."""
    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(ValueError):
        handle_missing_values(y_true, y_pred, sensitive, "raise")


# --------------------------------------------------------------------------
# task_type: unknown values must FAIL CLOSED
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bogus", ["classifcation", "Classification", "ranking", ""])
def test_validate_inputs_rejects_unknown_task_type(bogus):
    """A typo'd task_type used to skip the binary-label validation silently."""
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.0, 1.0, 1.0, 0.0])
    sensitive = np.array(["a", "b", "a", "b"], dtype=object)
    with pytest.raises(ConfigurationError) as exc:
        validate_inputs(y_true, y_pred, sensitive, task_type=bogus)
    message = str(exc.value)
    assert "task_type" in message
    assert "classification" in message and "regression" in message


def test_typo_task_type_no_longer_skips_binary_validation():
    """The concrete harm: non-binary labels passed validation under a typo."""
    y_true = np.array([1.0, 2.0, 1.0, 2.0])  # not 0/1
    y_pred = np.array([1.0, 2.0, 2.0, 1.0])
    sensitive = np.array(["a", "b", "a", "b"], dtype=object)
    # 'classification' correctly refuses these labels
    with pytest.raises(InvalidDataError):
        validate_inputs(y_true, y_pred, sensitive, task_type="classification")
    # and the typo must not be a way around that refusal
    with pytest.raises(ConfigurationError):
        validate_inputs(y_true, y_pred, sensitive, task_type="classifcation")


# --------------------------------------------------------------------------
# Over-correction control: every legitimate value still works, unchanged
# --------------------------------------------------------------------------


def test_exclude_still_drops_the_missing_rows():
    y_true, y_pred, sensitive = _data_with_missing_attr()
    yt, yp, sa, _, n_excluded = handle_missing_values(y_true, y_pred, sensitive, "exclude")
    assert n_excluded == _N_MISSING
    assert len(yt) == len(yp) == len(sa) == _N - _N_MISSING
    assert set(np.unique(sa)) == {"a", "b"}


def test_as_group_still_keeps_every_row_in_a_third_group():
    y_true, y_pred, sensitive = _data_with_missing_attr()
    yt, _, sa, _, n_excluded = handle_missing_values(y_true, y_pred, sensitive, "as_group")
    assert n_excluded == 0
    assert len(yt) == _N
    assert set(np.unique(sa)) == {"a", "b", "__missing__"}


def test_error_still_raises_invalid_data_error():
    y_true, y_pred, sensitive = _data_with_missing_attr()
    with pytest.raises(InvalidDataError):
        handle_missing_values(y_true, y_pred, sensitive, "error")


def test_default_strategy_is_unchanged():
    """Called with no strategy at all, the documented default still applies."""
    y_true, y_pred, sensitive = _data_with_missing_attr()
    yt, _, _, _, n_excluded = handle_missing_values(y_true, y_pred, sensitive)
    assert n_excluded == _N_MISSING
    assert len(yt) == _N - _N_MISSING


@pytest.mark.parametrize("strategy", ["exclude", "as_group"])
def test_validate_inputs_records_the_strategy_it_honoured(strategy):
    y_true, y_pred, sensitive = _data_with_missing_attr()
    *_, info = validate_inputs(y_true, y_pred, sensitive, missing_strategy=strategy)
    assert info["missing_strategy"] == strategy


def test_both_legitimate_task_types_still_validate():
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    y_pred = np.array([0.0, 1.0, 1.0, 0.0])
    sensitive = np.array(["a", "b", "a", "b"], dtype=object)
    *_, info = validate_inputs(y_true, y_pred, sensitive, task_type="classification")
    assert info["final_size"] == 4

    y_cont = np.array([1.5, 2.5, 3.5, 4.5])
    *_, info = validate_inputs(y_cont, y_cont, sensitive, task_type="regression")
    assert info["final_size"] == 4


def test_intersectional_legitimate_strategies_still_work():
    frame = pd.DataFrame(
        {
            "g": ["m", "f", np.nan, "m", "f", "m"],
            "r": ["x", "y", "x", np.nan, "y", "x"],
        }
    )
    y = np.array([1, 0, 1, 0, 1, 0])
    yt, _, sens, _, info = validate_inputs(
        y, y, frame, task_type="classification", missing_strategy="exclude"
    )
    assert info["n_excluded"] == 2
    assert len(yt) == 4
    assert not sens.isna().any().any()

    with pytest.raises(InvalidDataError):
        validate_inputs(y, y, frame, task_type="classification", missing_strategy="error")


class TestMultipleTestingCorrectionFailsClosed:
    """get_report must refuse an unrecognised correction method.

    This was the last option on ``get_report`` that still fell through to a
    default. ``missing_strategy`` and ``task_type`` were already fail-closed, so
    a caller could reasonably assume this one was too, and a typo silently got
    them a different correction than the one they asked for.

    The guard sits OUTSIDE the cache branch deliberately. Caching is off by
    default, so a check placed inside it runs for almost nobody: the first
    attempt at this fix did exactly that and the negative case below caught it.
    """

    @staticmethod
    def _analyzer(cache=False):
        import numpy as np

        from vfairness import FairnessAnalyzer

        return FairnessAnalyzer(
            np.array([0, 1] * 60),
            np.array([0, 1] * 60),
            np.array(["A"] * 60 + ["B"] * 60),
            cache=cache,
        )

    @pytest.mark.parametrize("cache", [False, True])
    def test_an_unknown_correction_method_is_refused(self, cache):
        from vfairness.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            self._analyzer(cache).get_report(multiple_testing_correction="not_a_method")

    @pytest.mark.parametrize("cache", [False, True])
    @pytest.mark.parametrize("method", ["none", "fdr", "bonferroni", "benjamini_hochberg"])
    def test_every_documented_method_is_still_accepted(self, cache, method):
        """The over-correction control: the guard must not reject real methods."""
        report = self._analyzer(cache).get_report(multiple_testing_correction=method)
        assert report["assessment"] is not None

    def test_compute_all_metrics_still_runs(self):
        """compute_all_metrics does NOT take this option.

        Pinned because the first attempt at the guard was inserted into
        compute_all_metrics by a mis-anchored edit, referencing a name that does
        not exist there, and every test then in the suite still passed because
        none of them called it. A NameError in a public method survived a green
        run, so it gets its own pin.
        """
        assert len(self._analyzer().compute_all_metrics()) > 0
