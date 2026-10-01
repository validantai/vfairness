"""Wave-6 audit pins (second-iteration audit, 2026-08-22).

Finding: predictive_parity_difference, fpr_parity_difference, fnr_parity_difference
and equalized_odds_difference collapsed an UNDEFINED per-group rate (NaN from a zero
denominator) into a 0.0 "perfect parity" result, which reads as PASS with a
deceptively tight [0.0, 0.0] bootstrap CI. The same class was fixed for NPV, AUROC
parity, CDD and multicalibration in 0b8bf54; these four siblings were missed.
Record: docs/audits/second-iteration-audit-2026-08-22.md (unit spec-v2-metrics).
"""

import math
import warnings

import numpy as np
import pytest

from vfairness.evaluation.vfairness_metrics import classification as C


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


class TestUndefinedGroupRateIsNaNNotZero:
    """Two qualifying groups, one with an undefined rate: NaN, never 0.0 parity."""

    def test_ppv_undefined_group_returns_nan(self):
        # Group B never receives a positive prediction, so its precision is
        # undefined; sufficiency is unmeasurable, not perfect.
        n = 100
        y_true = np.array([1, 0] * (n // 2) + [1, 0] * (n // 2))
        y_pred = np.concatenate([np.array([1, 0] * (n // 2)), np.zeros(n, dtype=int)])
        sens = np.array(["A"] * n + ["B"] * n)
        assert math.isnan(_quiet(C.predictive_parity_difference, y_true, y_pred, sens))

    def test_fpr_undefined_group_returns_nan(self):
        # Group B has no negatives (y_true all 1), so its FPR is undefined.
        n = 100
        y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.ones(n, dtype=int)])
        y_pred = np.concatenate([np.array([1] * 90 + [0] * 10), np.ones(n, dtype=int)])
        sens = np.array(["A"] * n + ["B"] * n)
        assert math.isnan(_quiet(C.fpr_parity_difference, y_true, y_pred, sens))

    def test_fnr_undefined_group_returns_nan(self):
        # Group B has no positives (y_true all 0), so its FNR is undefined.
        n = 100
        y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.zeros(n, dtype=int)])
        y_pred = np.concatenate([np.array([0] * 80 + [1] * 20), np.zeros(n, dtype=int)])
        sens = np.array(["A"] * n + ["B"] * n)
        assert math.isnan(_quiet(C.fnr_parity_difference, y_true, y_pred, sens))

    def test_equalized_odds_undefined_leg_returns_nan(self):
        # Group B has no negatives, so the FPR leg of equalized odds is
        # undefined for it; the metric must not report parity on the TPR leg
        # alone while silently dropping the unmeasurable leg.
        n = 100
        y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.ones(n, dtype=int)])
        y_pred = np.concatenate([np.array([1, 0] * (n // 2)), np.ones(n, dtype=int)])
        sens = np.array(["A"] * n + ["B"] * n)
        assert math.isnan(_quiet(C.equalized_odds_difference, y_true, y_pred, sens))

    def test_equal_opportunity_agrees_with_fnr_on_same_data(self):
        # FNR parity is 1 - TPR parity; on data where one is NaN the other must
        # be NaN too, otherwise the report contradicts itself.
        n = 100
        y_true = np.concatenate([np.array([1, 0] * (n // 2)), np.zeros(n, dtype=int)])
        y_pred = np.concatenate([np.array([0] * 80 + [1] * 20), np.zeros(n, dtype=int)])
        sens = np.array(["A"] * n + ["B"] * n)
        eo = _quiet(C.equal_opportunity_difference, y_true, y_pred, sens)
        fnr = _quiet(C.fnr_parity_difference, y_true, y_pred, sens)
        assert math.isnan(eo) == math.isnan(fnr)


class TestDefinedCasesStillMeasure:
    """The guard must not weaken real measurements or the pinned edge behavior."""

    def test_three_groups_one_undefined_is_nan_not_partial_parity(self):
        n = 60
        y_true = np.concatenate(
            [np.array([1, 0] * (n // 2)), np.array([1, 0] * (n // 2)), np.ones(n, dtype=int)]
        )
        y_pred = np.concatenate(
            [np.array([1] * 30 + [0] * 30), np.array([1] * 50 + [0] * 10), np.ones(n, dtype=int)]
        )
        sens = np.array(["A"] * n + ["B"] * n + ["C"] * n)
        # C has no negative labels, so it has NO false-positive rate to compare.
        #
        # This assertion used to read `not math.isnan(...)`, contradicting the
        # name of its own test: it demanded the spread over the two DEFINED
        # groups. Corrected 2026-09-10, because that spread is a lower bound
        # presented as the disparity and it is monotone the wrong way, the more
        # groups fall out the better the number reads. The decisive evidence is
        # the library's own arithmetic: FNR = 1 - TPR, so fnr_parity_difference
        # and equal_opportunity_difference are the SAME comparison, and on this
        # shape FNR answered a confident 0.5000 while equal opportunity answered
        # NaN over identical rates. One report called one measurement both
        # measured and not assessable.
        #
        # equal_opportunity_difference has always used the rule now applied
        # here: fewer than two qualifying groups is NaN, and ANY qualifying
        # group with an undefined rate is NaN plus a warning naming the group.
        assert math.isnan(_quiet(C.fpr_parity_difference, y_true, y_pred, sens))
        with pytest.warns(UserWarning, match=r"FPR is undefined for group\(s\).*'C'"):
            C.fpr_parity_difference(y_true, y_pred, sens)

    def test_single_qualifying_group_is_not_assessable(self):
        # This echoed the wave-5 T1 pin, which encoded a release-blocking defect:
        # with one qualifying group nothing is compared, and 0.0 is the value of
        # PERFECT parity, so an unmeasurable run certified as fair. Both metrics
        # now report NaN (insufficient evidence) instead. Kept here so the
        # wave-6 undefined-leg guard cannot regress the group-count case either
        # way. See test_audit_wave5_hardening T1 and tests/test_assessability_chain.py.
        y = np.array([1, 0] * 40)
        sens = np.array(["A"] * 70 + ["B"] * 10)
        assert math.isnan(_quiet(C.equalized_odds_difference, y, y, sens))
        assert math.isnan(_quiet(C.predictive_parity_difference, y, y, sens))

    def test_real_gap_still_measured(self):
        n = 100
        y_true = np.array([1, 0] * (n // 2) + [1, 0] * (n // 2))
        y_pred = np.concatenate([np.array([1] * 60 + [0] * 40), np.array([1] * 20 + [0] * 80)])
        sens = np.array(["A"] * n + ["B"] * n)
        assert _quiet(C.fpr_parity_difference, y_true, y_pred, sens) > 0.1
