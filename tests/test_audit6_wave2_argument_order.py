"""Audit 6, wave 2: a reversed ``fit`` call must be refused, not computed.

Every post-processing ``fit`` is ``fit(y_true, y_prob, sensitive_attr)`` and
each one immediately runs ``coerce_to_array(y_true).astype(int)``. That cast is
what made a reversed call SILENT: a probability column becomes all zeros, every
threshold collapses to the floor, and nearly every row is accepted. The
resulting fairness gap reads HEALTHY, because accepting everybody is trivially
equal, so the mitigation reported success while doing nothing.

Found in the production consumer on 2026-09-09 at eight call sites. Measured
there on 600 rows with divergent base rates: ``ThresholdOptimizer`` correct
threshold 0.8613 with 15 acceptances against reversed 0.0100 with 600, 97.5
percent of decisions differing; ``GroupThresholdOptimizer`` 0.6561/0.4173 with
246 against 0.05/0.05 with 598, 58.7 percent differing.

Each refusal pin here has a matching over-correction control, because a guard
that refuses legitimate label arrays would be worse than the defect it
replaces: labels arrive as ints, as floats holding 0.0 and 1.0, as Python
lists, and sometimes with NaN in them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vfairness.post_processing.reweighting.reweighter import (
    CalibratedEqualizer,
    DistributionMatcher,
    PredictionReweighter,
    RejectionOptionClassifier,
)
from vfairness.post_processing.threshold_optimization.optimizer import (
    GroupThresholdOptimizer,
    ThresholdOptimizer,
)

ALL_FITTERS = [
    PredictionReweighter,
    RejectionOptionClassifier,
    CalibratedEqualizer,
    DistributionMatcher,
    ThresholdOptimizer,
    GroupThresholdOptimizer,
]


def _data(n: int = 400):
    rng = np.random.default_rng(20260909)
    sens = rng.choice(["a", "b"], n)
    y_prob = np.clip(rng.normal(np.where(sens == "a", 0.62, 0.38), 0.15), 0, 1)
    y_true = (rng.random(n) < np.where(sens == "a", 0.30, 0.25)).astype(int)
    return y_true, y_prob, sens


class TestReversedFitIsRefused:
    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_a_positional_call_is_now_impossible(self, cls):
        """The production defect, fit(y_prob, y_true, sens), cannot be
        WRITTEN any more: the parameters are keyword-only, so the reversed
        call is a TypeError at the call site rather than a wrong answer.
        This is the primary defence and it closes the case the value-based
        guard below cannot see, a hard 0/1 score column."""
        y_true, y_prob, sens = _data()
        with pytest.raises(TypeError, match="positional"):
            cls().fit(y_prob, y_true, sens)

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_scores_passed_where_labels_belong_are_refused(self, cls):
        """The second line of defence: keywords used, but MISLABELLED. This
        is still reachable (nothing stops y_true=<scores>) and still wrong."""
        y_true, y_prob, sens = _data()
        with pytest.raises(ValueError, match="y_true must be labels"):
            cls().fit(y_true=y_prob, y_prob=y_true, sensitive_attr=sens)

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_the_refusal_names_the_swap_and_the_way_out(self, cls):
        y_true, y_prob, sens = _data()
        with pytest.raises(ValueError) as excinfo:
            cls().fit(y_true=y_prob, y_prob=y_true, sensitive_attr=sens)
        msg = str(excinfo.value)
        assert cls.__name__ in msg, msg
        assert "reversed call" in msg, msg
        assert "keywords" in msg, msg
        # It must say what it saw, not merely that something was wrong.
        assert "non-integral" in msg, msg


class TestControlLegitimateLabelsStillFit:
    """A guard that refuses everything is as useless as one that passed
    everything. Every shape a real label array arrives in must still fit."""

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_integer_labels(self, cls):
        y_true, y_prob, sens = _data()
        cls().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_float_labels_holding_zero_and_one(self, cls):
        y_true, y_prob, sens = _data()
        cls().fit(y_true=y_true.astype(float), y_prob=y_prob, sensitive_attr=sens)

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_python_list_labels(self, cls):
        y_true, y_prob, sens = _data()
        cls().fit(y_true=list(y_true), y_prob=y_prob, sensitive_attr=sens)

    @pytest.mark.parametrize("cls", ALL_FITTERS, ids=lambda c: c.__name__)
    def test_keyword_call(self, cls):
        y_true, y_prob, sens = _data()
        cls().fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)

    def test_nan_in_labels_is_not_read_as_a_swap(self):
        """NaN is missing, not continuous. Non-finite values are skipped, so a
        label array with a hole is not mistaken for a score column."""
        y_true, y_prob, sens = _data()
        holed = y_true.astype(float)
        holed[3] = np.nan
        GroupThresholdOptimizer().fit(y_true=holed, y_prob=y_prob, sensitive_attr=sens)

    def test_the_measured_answer_is_unchanged_by_the_guard(self):
        """The guard must not move a single number on the correct path."""
        y_true, y_prob, sens = _data()
        opt = GroupThresholdOptimizer()
        opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        first = {k: float(v) for k, v in opt.result_.group_thresholds.items()}
        again = GroupThresholdOptimizer()
        again.fit(y_true=y_true.astype(float), y_prob=y_prob, sensitive_attr=sens)
        assert {k: float(v) for k, v in again.result_.group_thresholds.items()} == first


class TestGuardScopeIsStated:
    def test_a_hard_zero_one_score_column_is_admittedly_not_caught(self):
        """Recorded, not hidden. Hard 0/1 predictions are genuinely
        indistinguishable from labels, so this swap cannot be detected by any
        value-based check. This is NOT hypothetical: it is the production
        platform's most common shape, because a dataset with no probability
        column falls back to the prediction column. Only keyword-only
        parameters would close it. What IS caught in that shape is the shape
        itself, pinned in TestScoreResolutionIsReported below."""
        y_true, _, sens = _data()
        hard = (np.arange(y_true.size) % 2).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GroupThresholdOptimizer().fit(y_true=hard, y_prob=y_true, sensitive_attr=sens)


class TestScoreResolutionIsReported:
    """A threshold sweep on a hard 0/1 column is a no-op that used to present
    itself as a mitigation: both group thresholds landed on the search floor,
    the returned predictions were IDENTICAL to the input, the acceptance gap
    was unchanged, and optimization_details still claimed 50 thresholds per
    group. Measured 2026-09-09, and it is the platform's default shape."""

    def test_a_two_valued_score_column_is_named_and_recorded(self):
        y_true, _, sens = _data()
        hard = (np.arange(y_true.size) % 2).astype(float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = GroupThresholdOptimizer()
            opt.fit(y_true=y_true, y_prob=hard, sensitive_attr=sens)
        named = [str(w.message) for w in caught if "distinct value" in str(w.message)]
        assert named, [str(w.message) for w in caught]
        assert "2 distinct value" in named[0], named[0]
        assert "hard 0/1 prediction column" in named[0], named[0]
        assert opt.result_.optimization_details["n_distinct_scores"] == 2
        assert opt.result_.optimization_details["score_resolution_sufficient"] is False

    def test_the_no_op_is_visible_in_the_result(self):
        """The output equalling the input is the fact a reader needs. It is not
        asserted as desirable, it is asserted as REPORTED: the details say the
        resolution was insufficient, so nobody has to infer it."""
        y_true, _, sens = _data()
        hard = (np.arange(y_true.size) % 2).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            opt = GroupThresholdOptimizer()
            opt.fit(y_true=y_true, y_prob=hard, sensitive_attr=sens)
            out = opt.predict(hard, sens)
        assert (out == hard.astype(int)).all(), "the sweep did move something after all"
        assert opt.result_.optimization_details["score_resolution_sufficient"] is False

    @pytest.mark.parametrize(
        "cls", [ThresholdOptimizer, GroupThresholdOptimizer], ids=lambda c: c.__name__
    )
    def test_control_a_real_score_column_is_silent_and_marked_sufficient(self, cls):
        """Over-correction control: the warning must not fire on the shape the
        technique is actually for."""
        y_true, y_prob, sens = _data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            opt = cls()
            opt.fit(y_true=y_true, y_prob=y_prob, sensitive_attr=sens)
        assert not [w for w in caught if "distinct value" in str(w.message)]
        assert opt.result_.optimization_details["score_resolution_sufficient"] is True
        assert opt.result_.optimization_details["n_distinct_scores"] > 100
