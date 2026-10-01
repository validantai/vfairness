"""Readiness pins for classification.selection_rate_disparity_matrix.

THE DEFECT these pin. ``missing_strategy`` was accepted, defaulted to
'exclude', documented nowhere and never read, while the body did
``fillna("missing")`` on the protected attribute, which is the OPPOSITE of
'exclude'. Records with NO protected attribute became a group literally named
"missing"; that bucket could take the ``reference_group`` slot and one half of
the worst adverse-impact pair, so the headline disparity was reported about
records that carry no protected attribute at all. Measured 2026-09-09 on
40 A + 40 B + 40 attribute-less records: reference_group 'missing',
min_ratio 0.211 over the pair ('B', 'missing'), while the only real comparison,
A against B, is 0.667. A caller passing missing_strategy='error' precisely to
be refused on incomplete data got that number instead of a refusal.

Same shape for a missing PREDICTION: the coercion branches turned it into
not-selected and left it in its group's denominator, so 20 unmeasured rows of
40 read as rate 0.50 against a measured 1.00, and the function published
min_ratio 1.0 / max_difference 0.0, a perfect-parity certificate for a
comparison that never happened.

Each fix carries BOTH a refusal pin and an over-correction control asserting
MEASURED values, so a change that simply drops or NaNs everything fails too.
"""

import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics.classification import (
    selection_rate_disparity_matrix,
    worst_group_accuracy,
)
from vfairness.exceptions import ConfigurationError, InvalidDataError


def _frame_with_missing_attribute():
    """40 A at 0.30, 40 B at 0.20, 40 records with NO protected attribute at 0.95.

    The attribute-less block is deliberately the highest-rate block, so if it
    is admitted it wins the reference_group slot and both worst pairs.
    """
    y_pred = np.concatenate(
        [
            (np.arange(40) < 12).astype(int),  # A: 12/40 = 0.30
            (np.arange(40) < 8).astype(int),  # B:  8/40 = 0.20
            (np.arange(40) < 38).astype(int),  # no attribute: 38/40 = 0.95
        ]
    )
    sens = np.array(["A"] * 40 + ["B"] * 40 + [None] * 40, dtype=object)
    return y_pred, sens


def _clean_frame():
    """The same two real groups, with no missing values anywhere."""
    y_pred = np.concatenate(
        [
            (np.arange(40) < 12).astype(int),
            (np.arange(40) < 8).astype(int),
        ]
    )
    sens = np.array(["A"] * 40 + ["B"] * 40)
    return y_pred, sens


class TestMissingAttributeIsNotAGroup:
    """'exclude', the declared default, must actually exclude."""

    def test_default_excludes_records_with_no_protected_attribute(self):
        y_pred, sens = _frame_with_missing_attribute()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens)

        # REFUSAL PIN: no bucket made out of records that have no attribute.
        assert res["groups"] == ["A", "B"]
        assert "missing" not in res["rates"]
        assert "__missing__" not in res["rates"]
        assert res["reference_group"] == "A"
        assert res["min_ratio_pair"] == ("B", "A")
        assert res["max_difference_pair"] == ("A", "B")

        # OVER-CORRECTION CONTROL: the two real groups are still MEASURED, at
        # their exact rates and their exact n. A fix that dropped everything,
        # or NaN'd the verdict, fails here.
        assert res["rates"]["A"]["rate"] == pytest.approx(0.30)
        assert res["rates"]["B"]["rate"] == pytest.approx(0.20)
        assert res["rates"]["A"]["n"] == 40
        assert res["rates"]["B"]["n"] == 40
        assert res["min_ratio"] == pytest.approx(0.20 / 0.30)
        assert res["max_difference"] == pytest.approx(0.10)
        assert not np.isnan(res["min_ratio"])

    def test_exclusion_is_named_in_a_warning(self):
        y_pred, sens = _frame_with_missing_attribute()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selection_rate_disparity_matrix(y_pred, sens)
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        assert any("excluded 40 of 120" in m for m in messages), messages
        assert any("no protected attribute" in m for m in messages), messages

    def test_as_group_keeps_them_under_the_library_sentinel(self):
        y_pred, sens = _frame_with_missing_attribute()
        res = selection_rate_disparity_matrix(y_pred, sens, missing_strategy="as_group")

        # The sentinel is the one _validation.handle_missing_values uses, not
        # the old plain "missing" (which a real category could also be called).
        assert "__missing__" in res["rates"]
        assert "missing" not in res["rates"]
        assert res["rates"]["__missing__"]["n"] == 40
        assert res["rates"]["__missing__"]["rate"] == pytest.approx(0.95)
        # Asked for explicitly, so it IS allowed to drive the headline here.
        assert res["reference_group"] == "__missing__"
        # And the real groups keep their measured rates.
        assert res["rates"]["A"]["rate"] == pytest.approx(0.30)
        assert res["rates"]["B"]["rate"] == pytest.approx(0.20)

    def test_a_real_group_named_missing_is_still_a_real_group(self):
        """The sentinel must not swallow a genuine category spelled 'missing'."""
        y_pred = np.concatenate([np.ones(30), np.zeros(30), np.ones(20), np.zeros(40)]).astype(int)
        sens = np.array(["A"] * 60 + ["missing"] * 60)
        res = selection_rate_disparity_matrix(y_pred, sens)
        assert res["groups"] == ["A", "missing"]
        assert res["rates"]["missing"]["n"] == 60
        assert res["rates"]["missing"]["rate"] == pytest.approx(20 / 60)
        assert res["rates"]["A"]["rate"] == pytest.approx(0.5)


class TestErrorStrategyRefuses:
    """A caller who asks to be refused on incomplete data IS refused."""

    def test_error_refuses_missing_protected_attribute(self):
        y_pred, sens = _frame_with_missing_attribute()
        with pytest.raises(InvalidDataError, match="40 rows with missing values"):
            selection_rate_disparity_matrix(y_pred, sens, missing_strategy="error")

    def test_error_refuses_missing_prediction(self):
        y_pred = np.array([1.0] * 20 + [np.nan] * 20 + [1.0] * 20 + [0.0] * 20)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        with pytest.raises(InvalidDataError, match="with no prediction"):
            selection_rate_disparity_matrix(y_pred, sens, missing_strategy="error")

    def test_error_and_exclude_no_longer_agree(self):
        """The tell that the parameter was inert: identical output either way."""
        y_pred, sens = _frame_with_missing_attribute()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            excluded = selection_rate_disparity_matrix(y_pred, sens, missing_strategy="exclude")
        with pytest.raises(InvalidDataError):
            selection_rate_disparity_matrix(y_pred, sens, missing_strategy="error")
        # OVER-CORRECTION CONTROL: 'exclude' still returns a real measurement.
        assert excluded["min_ratio"] == pytest.approx(0.20 / 0.30)

    def test_error_computes_normally_when_nothing_is_missing(self):
        """OVER-CORRECTION CONTROL: 'error' is not a blanket refusal."""
        y_pred, sens = _clean_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens, missing_strategy="error")
        assert res["groups"] == ["A", "B"]
        assert res["rates"]["A"]["rate"] == pytest.approx(0.30)
        assert res["rates"]["B"]["rate"] == pytest.approx(0.20)
        assert res["min_ratio"] == pytest.approx(0.20 / 0.30)

    def test_unknown_strategy_is_refused_not_defaulted(self):
        """'raise' is the obvious typo for 'error'; it must not fail open."""
        y_pred, sens = _clean_frame()
        with pytest.raises(ConfigurationError, match="Unknown missing_strategy"):
            selection_rate_disparity_matrix(y_pred, sens, missing_strategy="raise")


class TestMissingPredictionLeavesEveryAggregate:
    """An unmeasured prediction is not a not-selected one."""

    def test_nan_predictions_do_not_dilute_the_measured_rate(self):
        # A: 20 selected + 20 UNMEASURED  -> measured rate 20/20 = 1.00
        # B: 20 selected + 20 not selected -> measured rate 20/40 = 0.50
        y_pred = np.array([1.0] * 20 + [np.nan] * 20 + [1.0] * 20 + [0.0] * 20)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens)

        # REFUSAL PIN: the unmeasured rows are in no numerator and no
        # denominator, so A's n is 20, not 40, and the fake parity is gone.
        assert res["rates"]["A"]["n"] == 20
        assert res["rates"]["B"]["n"] == 40
        assert res["min_ratio"] != pytest.approx(1.0)
        assert res["max_difference"] != pytest.approx(0.0)

        # OVER-CORRECTION CONTROL: the surviving measurement is exact.
        assert res["rates"]["A"]["rate"] == pytest.approx(1.0)
        assert res["rates"]["B"]["rate"] == pytest.approx(0.5)
        assert res["min_ratio"] == pytest.approx(0.5)
        assert res["min_ratio_pair"] == ("B", "A")
        assert res["max_difference"] == pytest.approx(0.5)

    def test_missing_string_labels_are_excluded_not_read_as_not_selected(self):
        y_pred = np.array(["yes"] * 20 + [None] * 20 + ["yes"] * 20 + ["no"] * 20, dtype=object)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens)
        assert res["rates"]["A"]["n"] == 20
        assert res["rates"]["A"]["rate"] == pytest.approx(1.0)
        assert res["rates"]["B"]["rate"] == pytest.approx(0.5)
        assert res["max_difference"] == pytest.approx(0.5)

    def test_as_group_still_drops_the_unmeasured_predictions(self):
        """'as_group' keeps the missing ATTRIBUTE, never the missing prediction."""
        y_pred = np.array([1.0] * 20 + [np.nan] * 20 + [1.0] * 20 + [0.0] * 20)
        sens = np.array(["A"] * 40 + [None] * 40, dtype=object)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens, missing_strategy="as_group")
        assert res["rates"]["A"]["n"] == 20
        assert res["rates"]["A"]["rate"] == pytest.approx(1.0)
        assert res["rates"]["__missing__"]["n"] == 40
        assert res["rates"]["__missing__"]["rate"] == pytest.approx(0.5)

    def test_all_predictions_unmeasured_is_not_assessable(self):
        """Nothing measured must read as NaN, never as perfect parity."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = selection_rate_disparity_matrix(
                np.array([np.nan] * 20), np.array(["A"] * 10 + ["B"] * 10)
            )
        assert res["groups"] == []
        assert res["reference_group"] is None
        assert np.isnan(res["min_ratio"])
        assert np.isnan(res["max_difference"])
        assert res["min_ratio_pair"] is None
        assert res["max_difference_pair"] is None


class TestCleanDataIsUntouched:
    """OVER-CORRECTION CONTROL: nothing changes when nothing is missing."""

    def test_clean_frame_is_identical_under_every_strategy(self):
        y_pred, sens = _clean_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            results = [
                selection_rate_disparity_matrix(y_pred, sens, missing_strategy=strategy)
                for strategy in ("exclude", "as_group", "error")
            ]
        assert results[0] == results[1] == results[2]
        assert results[0]["groups"] == ["A", "B"]
        assert results[0]["rates"]["A"]["rate"] == pytest.approx(0.30)
        assert results[0]["rates"]["B"]["rate"] == pytest.approx(0.20)
        assert results[0]["rates"]["A"]["n"] == 40
        assert results[0]["min_ratio"] == pytest.approx(0.20 / 0.30)
        assert results[0]["max_difference"] == pytest.approx(0.10)

    def test_clean_frame_emits_no_exclusion_warning(self):
        y_pred, sens = _clean_frame()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            selection_rate_disparity_matrix(y_pred, sens)
        assert not [w for w in caught if "excluded" in str(w.message)]

    def test_continuous_scores_still_thresholded_at_the_median(self):
        """The coercion branches keep working; only unmeasured rows changed."""
        y_pred = np.arange(80, dtype=float)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            res = selection_rate_disparity_matrix(y_pred, sens)
        # median of 0..79 is 39.5, so nobody in A (0..39) clears it and
        # everybody in B (40..79) does.
        assert res["rates"]["A"]["rate"] == pytest.approx(0.0)
        assert res["rates"]["B"]["rate"] == pytest.approx(1.0)
        assert res["rates"]["A"]["n"] == 40
        assert res["min_ratio"] == pytest.approx(0.0)
        assert res["max_difference"] == pytest.approx(1.0)


class TestWorstGroupAccuracyHonoursMissingStrategy:
    """SIBLING, same root cause: missing_strategy applied only AFTER the gate.

    ``worst_group_accuracy`` built its GroupManager on the RAW protected
    attribute, so records with no attribute became a phantom group that failed
    min_group_size and made the whole statistic 'not assessable'. Measured
    2026-09-09: 40 records at accuracy 1.000 and 40 at 0.500, plus 5 records
    with no attribute, returned NaN where the answer is 0.500.
    """

    @staticmethod
    def _frame(missing_dtype):
        # group 0/A: 40 rows, accuracy 1.0 ; group 1/B: 40 rows, accuracy 0.5
        y_true = np.array([1, 0] * 20 + [1, 0] * 20 + [1] * 5)
        y_pred = np.array([1, 0] * 20 + [1, 1] * 20 + [1] * 5)
        if missing_dtype == "numeric":
            sens = np.array([0.0] * 40 + [1.0] * 40 + [np.nan] * 5)
        else:
            sens = np.array(["A"] * 40 + ["B"] * 40 + [np.nan] * 5, dtype=object)
        return y_true, y_pred, sens

    @pytest.mark.parametrize("dtype", ["numeric", "object"])
    def test_attribute_less_records_do_not_make_it_unassessable(self, dtype):
        y_true, y_pred, sens = self._frame(dtype)
        # RECORDS warnings rather than raising on them. This used
        # simplefilter("error", UserWarning) until 2026-09-27, which was a mechanism for
        # asserting "no PHANTOM GROUP warning", not a claim that this call must be
        # silent. validate_inputs now discloses "5 of 85 row(s) were excluded by
        # missing_strategy='exclude'", which is true, is the caller's business, and is
        # exactly what the five missing-attribute rows here are. Erroring on every
        # UserWarning made an honest disclosure indistinguishable from the phantom group
        # this test exists to rule out, so the phantom group is now named directly.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = worst_group_accuracy(y_true, y_pred, sens)
        phantom = [
            str(w.message)
            for w in caught
            if "nan" in str(w.message).lower() and "validate_inputs:" not in str(w.message)
        ]
        assert phantom == [], f"a phantom group was created from the missing rows: {phantom}"

        # REFUSAL PIN: no phantom group, so no spurious NaN and no TypeError.
        assert not np.isnan(value)

        # OVER-CORRECTION CONTROL: the MEASURED worst group, to the digit, and
        # the same number as the frame with those rows removed by hand.
        assert value == pytest.approx(0.5)
        assert value == pytest.approx(worst_group_accuracy(y_true[:80], y_pred[:80], sens[:80]))

    @pytest.mark.parametrize("dtype", ["numeric", "object"])
    def test_error_strategy_refuses_instead_of_raising_a_typeerror(self, dtype):
        y_true, y_pred, sens = self._frame(dtype)
        with pytest.raises(InvalidDataError, match="5 rows with missing values"):
            worst_group_accuracy(y_true, y_pred, sens, missing_strategy="error")

    def test_as_group_keeps_them_and_the_size_gate_then_applies(self):
        """'as_group' asks for the bucket, and 5 rows fail min_group_size=30.

        NaN is the documented answer when ANY group is dropped, because the
        dropped group could be the worst one. Pinned so the fix cannot be
        'silently exclude under every strategy'.
        """
        y_true, y_pred, sens = self._frame("object")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = worst_group_accuracy(y_true, y_pred, sens, missing_strategy="as_group")
        assert np.isnan(value)
        assert any("__missing__" in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]
        # ... and with a gate the bucket clears, the bucket really is IN: give
        # it accuracy 0.0 and it becomes the worst group, at 0.0, not 0.5.
        y_true_w = np.array([1, 0] * 20 + [1, 0] * 20 + [1] * 5)
        y_pred_w = np.array([1, 0] * 20 + [1, 1] * 20 + [0] * 5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            lowered = worst_group_accuracy(
                y_true_w, y_pred_w, sens, missing_strategy="as_group", min_group_size=5
            )
            excluded = worst_group_accuracy(y_true_w, y_pred_w, sens, min_group_size=5)
        assert lowered == pytest.approx(0.0)
        assert excluded == pytest.approx(0.5)

    def test_clean_frame_is_unchanged(self):
        """OVER-CORRECTION CONTROL: no missing values, same answer as always."""
        y_true = np.array([1, 0] * 20 + [1, 0] * 20)
        y_pred = np.array([1, 0] * 20 + [1, 1] * 20)
        sens = np.array(["A"] * 40 + ["B"] * 40)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            assert worst_group_accuracy(y_true, y_pred, sens) == pytest.approx(0.5)
        # a group below the gate still returns NaN, as documented
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            small = worst_group_accuracy(
                np.concatenate([y_true, [1] * 4]),
                np.concatenate([y_pred, [0] * 4]),
                np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 4),
            )
        assert np.isnan(small)
