"""BGL final, group d00: the two capabilities still recorded as carrying an open,
reproduced could-not-check defect.

    brier_score                 src/vfairness/post_processing/calibration/metrics.py
    create_threshold_optimizer  src/vfairness/post_processing/threshold_optimization/

Both halves of the standing rule are pinned here, because this group's findings
run in BOTH directions:

  * A neutral default standing in for a value nobody measured. That is the
    familiar one: max_group_disparity 0.0 for a comparison that never happened,
    a reference-group violation of 0.0 while another group's rate is NaN.
  * A REFUSAL standing in for a value that WAS measured. That is the same defect
    running backwards, and it is the more expensive one, because the evidence it
    throws away is a finding. `_CAL_SEPARATION_BOUND = 20.0` refused every
    converged calibration slope past 20, and the fixture below produces a
    converged, closed-form-verifiable slope of 736 that the bound reported as
    "the logistic fit diverged (separation)".

Every assertion that names a refusal is paired with a CONTROL on healthy data
that still measures, so a fix that simply stops answering cannot pass this file.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from vfairness.post_processing.calibration.metrics import (
    brier_score,
    calibration_in_the_large,
    calibration_slope,
    expected_calibration_error,
    group_calibration_metrics,
)
from vfairness.post_processing.threshold_optimization.constraints import (
    compute_constraint_violation,
)
from vfairness.post_processing.threshold_optimization.optimizer import (
    create_threshold_optimizer,
)


def _silently(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, List[str]]:
    """Call ``fn`` capturing its warnings, and return (result, warning texts)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _narrow_band_extreme_slope(half: int = 2000):
    """A REAL, extreme, converged calibration slope.

    Scores take only two values 0.002 apart while the outcome rate swings from
    5 percent to 95 percent between them. The MLE slope is therefore enormous
    and it EXISTS: both outcome classes appear at both score levels, so the
    likelihood has an interior maximum. Its closed form is

        (logit(0.95) - logit(0.05)) / (logit(0.501) - logit(0.499)) = 736.109

    which :func:`test_an_extreme_but_converged_slope_is_measured` checks the
    returned value against, so the pin is anchored to arithmetic done outside
    the library rather than to whatever the library happens to print.
    """
    p = np.concatenate([np.full(half, 0.499), np.full(half, 0.501)])
    lo = np.array([1] * (half // 20) + [0] * (half - half // 20))
    hi = np.array([1] * (half - half // 20) + [0] * (half // 20))
    return np.concatenate([lo, hi]), p


def _closed_form_narrow_band_slope() -> float:
    return (math.log(0.95 / 0.05) - math.log(0.05 / 0.95)) / (
        math.log(0.501 / 0.499) - math.log(0.499 / 0.501)
    )


def _healthy_calibrated(n: int = 4000, seed: int = 7):
    """Well calibrated scores: the slope is near 1 and every arm is measurable."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def _one_group_unmeasurable(seed: int = 3):
    """Group A healthy (2000 rows), group B a single outcome class (60 rows).

    B is large enough to clear ``min_group_size`` so it is NOT excluded: it is
    kept, attempted and found unmeasurable, which is the state that used to be
    counted as a comparison.
    """
    rng = np.random.default_rng(seed)
    n_a = 2000
    p_a = rng.uniform(0.05, 0.95, n_a)
    y_a = (rng.uniform(size=n_a) < p_a).astype(int)
    n_b = 60
    p_b = np.linspace(0.05, 0.95, n_b)
    y_b = np.ones(n_b, dtype=int)
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * n_a + ["B"] * n_b),
    )


def _two_measurable_groups(seed: int = 11):
    """CONTROL: two groups that both fit, with a real calibration gap between
    them (B's scores are shifted, so its slope genuinely differs from A's)."""
    rng = np.random.default_rng(seed)
    n = 2000
    p_a = rng.uniform(0.05, 0.95, n)
    y_a = (rng.uniform(size=n) < p_a).astype(int)
    p_b = rng.uniform(0.05, 0.95, n)
    y_b = (rng.uniform(size=n) < np.clip(0.5 + 0.35 * (p_b - 0.5), 0.01, 0.99)).astype(int)
    return (
        np.concatenate([y_a, y_b]),
        np.concatenate([p_a, p_b]),
        np.array(["A"] * n + ["B"] * n),
    )


# ===========================================================================
# brier_score, item 1: the bound that deleted a real finding
# ===========================================================================


class TestConvergedFitIsAMeasurement:
    def test_an_extreme_but_converged_slope_is_measured(self) -> None:
        """PIN, the POSITIVE case the auditor asked for explicitly.

        BEFORE (``_CAL_SEPARATION_BOUND = 20.0`` live, measured 2026-09-17 at
        the public entry on 4000 rows)::

            calibration_slope(y, p).overall_value      -> nan
            .metadata['measured']                      -> False
            .metadata['unmeasured_reason']             -> 'the logistic fit
                                                          diverged (separation)'
            .is_well_calibrated                        -> None
            warnings                                   -> 1 could-not-check

        AFTER::

            overall_value 736.1088, measured True, is_well_calibrated False,
            no warning.

        A slope of 736 against a target band of [0.8, 1.2] is one of the
        strongest miscalibration findings this metric can produce. Refusing it
        is the fabricated-verdict defect running backwards.
        """
        y, p = _narrow_band_extreme_slope()
        result, texts = _silently(calibration_slope, y, p)

        assert np.isfinite(result.overall_value), (
            "a converged MLE was reported as could-not-check; the separation bound is back"
        )
        # Anchored to arithmetic done outside the library, not to its own output.
        assert result.overall_value == pytest.approx(_closed_form_narrow_band_slope(), rel=1e-9)
        assert result.overall_value > 700
        assert result.metadata["measured"] is True
        assert result.metadata["unmeasured_reason"] is None
        # The finding itself, not merely the number: this is NOT well calibrated.
        assert result.is_well_calibrated is False
        assert not [t for t in texts if "could not be measured" in t]

    def test_the_slope_of_a_narrow_band_survives_into_the_group_breakdown(self) -> None:
        """PIN. The same extreme fit reached through the per-group path, which
        is the one a fairness reader consumes."""
        y, p = _narrow_band_extreme_slope()
        attr = np.array(["A"] * len(y))
        result, _ = _silently(calibration_slope, y, p, attr)
        assert result.group_values is not None
        assert result.group_values["A"] == pytest.approx(_closed_form_narrow_band_slope(), rel=1e-9)
        assert result.metadata["unmeasured_groups"] == {}

    def test_calibration_in_the_large_measures_a_large_converged_intercept(self) -> None:
        """PIN. The sibling helper carried the identical bound on ``coef[0]``.

        A constant score of sigmoid(-2) = 0.1192 against an observed rate of
        99.9 percent needs an intercept of

            logit(0.999) - logit(0.1192) = 6.9068 + 2 = 8.9068

        to explain the data: a large, genuinely converged CITL, checked here
        against that closed form rather than against the library's own output.
        """
        n = 4000
        p = np.full(n, 1.0 / (1.0 + math.exp(2.0)))
        k = int(round(0.999 * n))
        y = np.array([1] * k + [0] * (n - k))
        result, texts = _silently(calibration_in_the_large, y, p)
        assert np.isfinite(result.overall_value)
        assert result.metadata["measured"] is True
        assert result.metadata["signed"] == pytest.approx(math.log(0.999 / 0.001) + 2.0, rel=1e-6)
        assert result.overall_value > 8.0
        assert result.is_well_calibrated is False  # |CITL| >> 0.05
        assert not [t for t in texts if "could not be measured" in t]

    def test_a_single_outcome_class_citl_refuses_structurally_not_by_magnitude(self) -> None:
        """PIN, and the reason this file exists in the shape it does.

        Removing the magnitude bound EXPOSED what the bound had been hiding:
        ``_citl_value`` had no single-outcome-class test, and for an
        intercept-only fit a single class IS complete separation. With every
        label 1 and a CONSTANT score, ``mu`` saturates, ``mu * (1 - mu)`` hits
        the ``1e-9`` clip that floors the Hessian while the gradient decays
        exponentially, the Newton step falls under ``tol``, and the loop reports
        ``converged=True`` at a point that is not the maximum. Measured on 4000
        rows, every label 1, with the bound removed and before the structural
        guard::

            p = sigmoid(-4)     -> 55.598     measured=True
            p = sigmoid(-13.8)  -> 984607.69  measured=True

        The gradient at both points is EXACTLY 0.0, because sigmoid underflows
        to 1.0 in float64, so no gradient test could have caught it either. The
        refusal has to be structural, and it has to stay structural: a magnitude
        bound put back in its place would refuse the measured 736 slope above
        and the measured 8.9 intercept in the test before it.
        """
        n = 4000
        for log_odds in (-4.0, -13.8):
            p = np.full(n, 1.0 / (1.0 + math.exp(-log_odds)))
            y = np.ones(n, dtype=int)
            result, texts = _silently(calibration_in_the_large, y, p)
            assert np.isnan(result.overall_value), (
                f"a separable intercept-only fit reported {result.overall_value}"
            )
            assert result.metadata["measured"] is False
            assert result.is_well_calibrated is None
            reason = result.metadata["unmeasured_reason"]
            assert "single outcome class" in reason and "separation" in reason
            assert any("could not be measured" in t for t in texts)

    # -- the refusals the bound was NOT what was holding up ------------------

    @pytest.mark.parametrize(
        "fn", [calibration_slope, calibration_in_the_large], ids=["slope", "citl"]
    )
    def test_a_separable_fit_still_refuses(self, fn) -> None:
        """PIN. Removing the magnitude bound must not resurrect the finding the
        convergence flag catches: 60 rows, every label 1, no contrast to fit."""
        y = np.ones(60, dtype=int)
        p = np.linspace(0.05, 0.95, 60)
        result, texts = _silently(fn, y, p)
        assert np.isnan(result.overall_value)
        assert result.metadata["measured"] is False
        assert result.is_well_calibrated is None
        assert any("could not be measured" in t for t in texts)

    def test_a_constant_score_still_refuses(self) -> None:
        """PIN. logit(p) collinear with the intercept: the slope is not
        identifiable, and the ridge makes the singular system answerable."""
        y = np.array([1, 0] * 30)
        p = np.full(60, 0.7)
        result, texts = _silently(calibration_slope, y, p)
        assert np.isnan(result.overall_value)
        assert "not identifiable" in result.metadata["unmeasured_reason"]
        assert any("could not be measured" in t for t in texts)

    @pytest.mark.parametrize(
        "fn", [calibration_slope, calibration_in_the_large], ids=["slope", "citl"]
    )
    def test_zero_rows_still_refuse(self, fn) -> None:
        """PIN. The ``beta = np.zeros(k)`` initialiser passes the convergence
        test on iteration 1 for an empty gradient; the explicit length test is
        what refuses, and it is above the fit."""
        result, texts = _silently(fn, np.array([], dtype=int), np.array([], dtype=float))
        assert np.isnan(result.overall_value)
        assert result.metadata["unmeasured_reason"] == "no rows"
        assert result.is_well_calibrated is None
        assert any("could not be measured" in t for t in texts)

    def test_control_healthy_data_is_measured_and_graded_well(self) -> None:
        """CONTROL. A refusal-only implementation cannot pass this."""
        y, p = _healthy_calibrated()
        slope = calibration_slope(y, p)
        assert np.isfinite(slope.overall_value)
        assert 0.8 <= slope.overall_value <= 1.2
        assert slope.is_well_calibrated is True
        assert slope.metadata["measured"] is True

        citl = calibration_in_the_large(y, p)
        assert np.isfinite(citl.overall_value)
        assert citl.overall_value < 0.05
        assert citl.is_well_calibrated is True


# ===========================================================================
# brier_score, item 2: max_group_disparity / n_groups_compared over NaN members
# ===========================================================================


@pytest.mark.parametrize(
    "fn",
    [
        brier_score,
        expected_calibration_error,
        calibration_slope,
        calibration_in_the_large,
    ],
    ids=["brier_score", "ece", "calibration_slope", "calibration_in_the_large"],
)
class TestDisparityOverPartiallyMeasuredGroups:
    """The auditor's item 2. The existing pin file exercises max_group_disparity
    only through brier / ECE / MCE, whose per-group values are always finite, so
    the NaN member case lived entirely outside it. calibration_slope and
    calibration_in_the_large DO put NaN in ``group_values``, and they are
    parametrised in here for that reason."""

    def test_an_unmeasured_group_is_not_counted_as_compared(self, fn) -> None:
        """PIN. BEFORE (measured 2026-09-17)::

            calibration_slope(y, p, g).to_dict()
            -> {'group_values': {'A': 0.961, 'B': nan},
                'max_group_disparity': nan,
                'n_groups_compared': 2}

        a refusal in one field and a claim of a two-group comparison in the
        next. ``n_groups_compared`` is the field a JSON consumer reads to tell
        the third state apart from measured parity, so it counted the very
        groups whose absence made the disparity NaN.
        """
        y, p, attr = _one_group_unmeasurable()
        result, _ = _silently(fn, y, p, attr)
        assert result.group_values is not None
        unmeasured = [g for g, v in result.group_values.items() if not np.isfinite(v)]
        if not unmeasured:
            pytest.skip(f"{fn.__name__} produces no NaN group value on this fixture")

        assert "B" in unmeasured  # the fixture really did go unmeasurable
        assert np.isnan(result.to_dict()["max_group_disparity"])
        assert result.n_groups_compared == 1, (
            "an unmeasured group was counted as compared: n_groups_compared "
            f"{result.n_groups_compared} over group_values {result.group_values}"
        )
        assert result.to_dict()["n_groups_compared"] == 1
        # The disclosure a reader needs: fewer counted than present.
        assert result.n_groups_compared < len(result.group_values)

    def test_a_single_group_is_never_perfect_parity(self, fn) -> None:
        """PIN. One group: min == max == the same number, which is exactly 0.0
        and reads as PERFECT PARITY ON EVERY SCALE."""
        y, p = _healthy_calibrated(n=600, seed=5)
        attr = np.array(["only"] * len(y))
        result, _ = _silently(fn, y, p, attr)
        assert np.isnan(result.to_dict()["max_group_disparity"])
        assert result.n_groups_compared <= 1

    def test_control_two_measurable_groups_keep_a_measured_disparity(self, fn) -> None:
        """CONTROL, the other polarity. A real gap between two measurable groups
        is still a number, and it still equals the spread of the values."""
        y, p, attr = _two_measurable_groups()
        result, _ = _silently(fn, y, p, attr)
        assert result.group_values is not None
        values = [v for v in result.group_values.values() if np.isfinite(v)]
        assert len(values) == 2, f"the control fixture went unmeasurable: {result.group_values}"
        assert result.n_groups_compared == 2
        disparity = result.to_dict()["max_group_disparity"]
        assert np.isfinite(disparity)
        assert disparity == pytest.approx(max(values) - min(values))


def test_group_calibration_metrics_counts_only_measured_groups() -> None:
    """PIN at the aggregate entry, which is where a report reads these."""
    y, p, attr = _two_measurable_groups()
    metrics, _ = _silently(group_calibration_metrics, y, p, attr)
    for key in ("ece", "mce", "brier"):
        assert metrics[key].n_groups_compared == 2, key
        assert np.isfinite(metrics[key].max_group_disparity), key


def _three_groups_one_unmeasurable(seed: int = 21):
    """A and B measurable and FAR APART, C a single outcome class.

    This is the shape where refusing outright would delete a real finding: the
    A-to-B gap is determinate however C turns out.
    """
    rng = np.random.default_rng(seed)
    n = 2000
    p_a = rng.uniform(0.05, 0.95, n)
    y_a = (rng.uniform(size=n) < p_a).astype(int)
    p_b = rng.uniform(0.05, 0.95, n)
    y_b = (rng.uniform(size=n) < np.clip(0.5 + 0.1 * (p_b - 0.5), 0.01, 0.99)).astype(int)
    n_c = 60
    p_c = np.linspace(0.05, 0.95, n_c)
    y_c = np.ones(n_c, dtype=int)
    return (
        np.concatenate([y_a, y_b, y_c]),
        np.concatenate([p_a, p_b, p_c]),
        np.array(["A"] * n + ["B"] * n + ["C"] * n_c),
    )


class TestTheLowerBoundIsNotThrownAway:
    """Refusing the overall disparity must not also delete the evidence there
    IS. ``max_group_disparity`` stays NaN when any group is unmeasured, because
    a spread over a subset is not the disparity; ``measured_subset_disparity``
    carries that subset spread, labelled as the floor it is."""

    def test_a_determinate_gap_between_two_measured_groups_survives(self) -> None:
        """PIN. A and B differ by a large, determinate amount and C could not be
        measured. NaN for the overall disparity is right, and losing the A-to-B
        gap with it would not be."""
        y, p, attr = _three_groups_one_unmeasurable()
        result, _ = _silently(calibration_slope, y, p, attr)
        assert result.group_values is not None
        assert not np.isfinite(result.group_values["C"])  # the fixture bit
        measured = [result.group_values[g] for g in ("A", "B")]
        assert all(np.isfinite(v) for v in measured)

        assert np.isnan(result.max_group_disparity)  # unknowable overall
        bound = result.measured_subset_disparity
        assert np.isfinite(bound)
        assert bound == pytest.approx(max(measured) - min(measured))
        assert bound > 0.1  # the finding is real, not a rounding artefact
        assert result.n_groups_compared == 2 < len(result.group_values)
        assert np.isfinite(result.to_dict()["measured_subset_disparity"])

    def test_with_every_group_measured_the_two_agree(self) -> None:
        """CONTROL. When nothing is missing the bound IS the disparity, so a
        consumer cannot be misled into reading a complete measurement as
        partial."""
        y, p, attr = _two_measurable_groups()
        result, _ = _silently(calibration_slope, y, p, attr)
        assert np.isfinite(result.max_group_disparity)
        assert result.measured_subset_disparity == pytest.approx(result.max_group_disparity)
        assert result.n_groups_compared == len(result.group_values)

    def test_one_measured_group_gives_no_bound_either(self) -> None:
        """PIN. With a single measured group there is no pair, so there is no
        floor to report: a subset of one is not a comparison."""
        y, p, attr = _one_group_unmeasurable()
        result, _ = _silently(calibration_slope, y, p, attr)
        assert result.n_groups_compared == 1
        assert np.isnan(result.max_group_disparity)
        assert np.isnan(result.measured_subset_disparity)


# ===========================================================================
# brier_score, item 4: the three-state disclosure has to reach the CHART
# ===========================================================================


@pytest.mark.parametrize("with_calibrated", [False, True], ids=["single_series", "before_after"])
class TestDashboardShowsTheThirdState:
    """The dashboard draws the metrics panel through TWO different code paths,
    one bar series or two, and the second one is the one a recalibration report
    actually renders. Both are parametrised here: pinning only the single-series
    branch left the grouped branch free to drop the disclosure with every
    assertion in this file still green (measured, sabotage S8, 2026-09-17)."""

    @staticmethod
    def _dashboard_texts(y, p, attr, with_calibrated):
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from vfairness.post_processing.calibration.visualization import (
            create_calibration_dashboard,
        )

        calibrated = np.clip(np.asarray(p, dtype=float) * 0.9 + 0.05, 0.01, 0.99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = create_calibration_dashboard(
                y, p, attr, calibrated_probs=calibrated if with_calibrated else None
            )
        try:
            return [t.get_text() for ax in fig.axes for t in ax.texts]
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_an_unmeasurable_disparity_says_so_on_the_bar_chart(self, with_calibrated) -> None:
        """PIN. A NaN bar and a measured 0.0 bar are both zero pixels tall and
        both sit comfortably under the red 0.05 target line, so the reader of
        the dashboard could not tell "perfect parity" from "nobody could compare
        the groups". The value and ``to_dict()`` had the third state; the
        surface a person actually looks at did not."""
        y, p = _healthy_calibrated(n=600, seed=9)
        attr = np.array(["only"] * len(y))  # one group -> disparity is NaN
        texts = self._dashboard_texts(y, p, attr, with_calibrated)
        marks = [t for t in texts if "not measured" in t]
        assert marks, "the ECE Disparity bar is NaN and the chart says nothing about it"
        if with_calibrated:
            # Both series are NaN here, and a reader has to be able to tell
            # which bar each label belongs to.
            assert any("before" in t for t in marks)
            assert any("after" in t for t in marks)

    def test_control_a_measured_disparity_is_not_labelled_unmeasured(self, with_calibrated) -> None:
        """CONTROL. Labelling every bar would be the over-correction."""
        y, p, attr = _two_measurable_groups()
        texts = self._dashboard_texts(y, p, attr, with_calibrated)
        assert not any("not measured" in t for t in texts)


# ===========================================================================
# create_threshold_optimizer: the constraint layer it dispatches into
# ===========================================================================


def _fpr_arm_dead(tpr_gap: float = 0.6, n: int = 100):
    """Every label 1, so the FPR arm of equalized odds has an empty denominator
    in every group. The TPR arm is measurable and carries ``tpr_gap``."""
    y = np.ones(2 * n, dtype=int)
    a_pos = int(round(0.9 * n))
    b_pos = int(round((0.9 - tpr_gap) * n))
    pred = np.concatenate(
        [
            np.array([1] * a_pos + [0] * (n - a_pos)),
            np.array([1] * b_pos + [0] * (n - b_pos)),
        ]
    )
    groups = np.array(["A"] * n + ["B"] * n)
    return y, pred, groups


def _tpr_arm_dead(fpr_gap: float = 0.6, n: int = 100):
    """THE MIRROR of the fixture above: every label 0, so the TPR arm is the
    dead one and the FPR arm carries the gap. The auditor's sabotage S-D swapped
    the two operands and stayed green because only one shape was pinned."""
    y = np.zeros(2 * n, dtype=int)
    a_pos = int(round(0.9 * n))
    b_pos = int(round((0.9 - fpr_gap) * n))
    pred = np.concatenate(
        [
            np.array([1] * a_pos + [0] * (n - a_pos)),
            np.array([1] * b_pos + [0] * (n - b_pos)),
        ]
    )
    groups = np.array(["A"] * n + ["B"] * n)
    return y, pred, groups


def _healthy_two_groups(gap: float = 0.4, n: int = 200, seed: int = 0):
    """CONTROL: both arms measurable in both groups, with a real disparity."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 2 * n)
    groups = np.array(["A"] * n + ["B"] * n)
    rate = np.where(groups == "A", 0.5 + gap / 2, 0.5 - gap / 2)
    pred = (rng.uniform(size=2 * n) < rate).astype(int)
    return y, pred, groups


class TestReferenceGroupPath:
    """The auditor's item 1. The reference-group branch of ``_metric_disparity``
    returned ``max(group_violations.values())`` from ABOVE the NaN guard, so the
    guard was reachable only with ``reference_group=None``. It is exercised by
    no other test in the pin file."""

    @pytest.mark.parametrize(
        "constraint",
        [
            "false_positive_parity",
            "equalized_odds",
            "demographic_parity",
            "equal_opportunity",
            "predictive_parity",
        ],
    )
    def test_a_reference_group_cannot_walk_past_the_nan_guard(self, constraint) -> None:
        """PIN. BEFORE, group B holding no negative rows::

            reference_group=None -> violation nan, is_satisfied None, 1 warning
            reference_group='A'  -> violation 0.0, is_satisfied True, 0 warnings

        the exact fabrication the guard was written to remove, reachable through
        a keyword argument. Both callers must now reach the same verdict: which
        group you name as the yardstick cannot decide whether the data was
        measurable.
        """
        rng = np.random.default_rng(0)
        n = 100
        y = np.concatenate([rng.integers(0, 2, n), np.ones(n, dtype=int)])
        pred = (rng.uniform(size=2 * n) < 0.5).astype(int)
        groups = np.array(["A"] * n + ["B"] * n)

        default, default_texts = _silently(
            compute_constraint_violation, y, pred, groups, constraint=constraint
        )
        referenced, referenced_texts = _silently(
            compute_constraint_violation,
            y,
            pred,
            groups,
            constraint=constraint,
            reference_group="A",
        )
        assert referenced.is_satisfied is default.is_satisfied, (
            f"{constraint}: reference_group='A' reached {referenced.is_satisfied} "
            f"where reference_group=None reached {default.is_satisfied}"
        )
        assert bool(referenced_texts) is bool(default_texts)
        if default.is_satisfied is None:
            # This is the shape the defect lived in: an unmeasurable cell.
            assert np.isnan(referenced.violation), (
                f"{constraint}: reference_group='A' reported a measured "
                f"violation of {referenced.violation} over an unmeasured rate"
            )
            assert referenced.unmeasured, "the unmeasured cells were not disclosed"
            assert any("COULD NOT CHECK" in t for t in referenced_texts)

    def test_control_a_reference_group_still_measures_healthy_data(self) -> None:
        """CONTROL, and it also pins the 2-group equivalence: for two groups the
        reference form and the max-min spread are the same number, so a guard
        that simply refused every reference call would fail here."""
        y, pred, groups = _healthy_two_groups()
        referenced = compute_constraint_violation(
            y, pred, groups, constraint="demographic_parity", reference_group="A"
        )
        default = compute_constraint_violation(y, pred, groups, constraint="demographic_parity")
        assert np.isfinite(referenced.violation)
        assert referenced.violation == pytest.approx(default.violation)
        assert referenced.is_satisfied is False  # a 0.4 gap against tolerance 0.05
        assert referenced.unmeasured == ()
        assert referenced.violation_is_lower_bound is False

    def test_control_a_reference_group_can_still_report_satisfied(self) -> None:
        """CONTROL, the other polarity: refusing everything would pass a
        violated-only check."""
        y, pred, groups = _healthy_two_groups(gap=0.0, n=400, seed=2)
        result, texts = _silently(
            compute_constraint_violation,
            y,
            pred,
            groups,
            constraint="demographic_parity",
            reference_group="B",
        )
        assert result.is_satisfied is True
        assert np.isfinite(result.violation)
        assert result.violation <= result.tolerance
        assert not texts


class TestEqualizedOddsWithOneDeadArm:
    """The auditor's items 2 and 3. Equalized odds is a CONJUNCTION, and the
    two directions it can fail in are opposite defects:

      * taking the measured arm as the whole verdict fabricates a clean pass
      * refusing outright deletes a determinate breach
    """

    def test_a_determinate_breach_survives_a_dead_arm_as_a_lower_bound(self) -> None:
        """PIN. TPR gap 0.60 against tolerance 0.05 with the FPR arm dead: no
        value of the unknown arm can bring the conjunction inside tolerance, so
        this is a VIOLATION, reported with the caveat rather than instead of
        it."""
        y, pred, groups = _fpr_arm_dead(tpr_gap=0.6)
        result, texts = _silently(
            compute_constraint_violation, y, pred, groups, constraint="equalized_odds"
        )
        assert result.is_satisfied is False
        assert result.violation == pytest.approx(0.6, abs=1e-9)
        assert result.violation_is_lower_bound is True
        assert set(result.unmeasured) == {"A.fpr", "B.fpr"}
        assert any("LOWER BOUND" in t for t in texts)

    def test_the_mirrored_shape_reaches_the_same_verdict(self) -> None:
        """PIN, the auditor's S-D hole. Same data with the ARMS EXCHANGED: every
        label 0, so the TPR arm is the dead one. An operand swap in the
        conjunction passes the fixture above and re-arms the original defect for
        this shape, so both shapes are pinned."""
        y, pred, groups = _tpr_arm_dead(fpr_gap=0.6)
        result, texts = _silently(
            compute_constraint_violation, y, pred, groups, constraint="equalized_odds"
        )
        assert result.is_satisfied is False
        assert result.violation == pytest.approx(0.6, abs=1e-9)
        assert result.violation_is_lower_bound is True
        assert set(result.unmeasured) == {"A.tpr", "B.tpr"}
        assert any("LOWER BOUND" in t for t in texts)

    @pytest.mark.parametrize(
        "fixture", [_fpr_arm_dead, _tpr_arm_dead], ids=["fpr_dead", "tpr_dead"]
    )
    def test_the_per_group_cells_are_nan_not_the_surviving_arm(self, fixture) -> None:
        """PIN, the auditor's S-C hole. ``group_violations`` is the per-GROUP
        equalized-odds violation, and that cell really is unknown as soon as one
        of its two rates is: ``max(0.02, nan)`` is 0.02 and ``max(nan, 0.02)``
        is nan, so the answer used to depend on argument order.

        Reverting ``_max_or_nan`` to a builtin ``max`` here leaves every other
        assertion in this file green while corrupting the warning text to
        "group(s) []", which is why this assertion is written against the cells
        and not against the warning."""
        y, pred, groups = fixture(0.6)
        result, _ = _silently(
            compute_constraint_violation, y, pred, groups, constraint="equalized_odds"
        )
        assert set(result.group_violations) == {"A", "B"}
        for group, value in result.group_violations.items():
            assert np.isnan(value), (
                f"group {group} carries {value} for a conjunction with a dead arm"
            )

    def test_a_measured_arm_inside_tolerance_is_could_not_check(self) -> None:
        """PIN, the other half of the rule. Here the unknown arm really could
        decide it, so None is the honest answer and the violation is NaN."""
        y, pred, groups = _fpr_arm_dead(tpr_gap=0.02)
        result, texts = _silently(
            compute_constraint_violation, y, pred, groups, constraint="equalized_odds"
        )
        assert result.is_satisfied is None
        assert np.isnan(result.violation)
        assert result.violation_is_lower_bound is False
        assert any("COULD NOT CHECK" in t for t in texts)
        assert any("could still decide it either way" in t for t in texts)

    def test_control_both_arms_measurable_gives_a_plain_verdict(self) -> None:
        """CONTROL. Nothing is a lower bound and nothing is unmeasured."""
        y, pred, groups = _healthy_two_groups()
        result, texts = _silently(
            compute_constraint_violation, y, pred, groups, constraint="equalized_odds"
        )
        assert result.is_satisfied is False
        assert np.isfinite(result.violation)
        assert result.violation > result.tolerance
        assert result.unmeasured == ()
        assert result.violation_is_lower_bound is False
        assert all(np.isfinite(v) for v in result.group_violations.values())
        assert not texts


class TestCreateThresholdOptimizerSurface:
    """The registered capability itself. ``ThresholdResult.is_feasible`` and the
    line ``summary()`` prints first are what a reader consumes, so the third
    state is pinned there and not only in the constraint object."""

    @staticmethod
    def _fit_with_a_dead_arm():
        rng = np.random.default_rng(0)
        n = 120
        y = np.ones(2 * n, dtype=int)  # no negatives anywhere -> FPR undefined
        prob = np.concatenate([rng.uniform(0.55, 0.99, n), rng.uniform(0.01, 0.45, n)])
        groups = np.array(["A"] * n + ["B"] * n)
        optimizer = create_threshold_optimizer(
            optimizer_type="group", constraint="equalized_odds", tolerance=0.05
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(y_true=y, y_prob=prob, sensitive_attr=groups)
        return optimizer.result_

    def test_a_partly_measured_breach_is_printed_as_a_lower_bound(self) -> None:
        """PIN. The number in ``summary()`` is a lower bound and says so, rather
        than reading as the whole disparity."""
        result = self._fit_with_a_dead_arm()
        assert result.is_feasible is False
        text = result.summary()
        assert "at least" in text
        assert "lower bound" in text
        assert "A.fpr" in text and "B.fpr" in text
        assert "COULD NOT CHECK" in text
        violation = result.constraint_violations[0]
        assert violation.violation_is_lower_bound is True
        assert violation.is_satisfied is False

    def test_control_a_healthy_fit_prints_a_plain_number(self) -> None:
        """CONTROL. No caveat where nothing was missing."""
        rng = np.random.default_rng(4)
        n = 200
        y = rng.integers(0, 2, 2 * n)
        prob = np.clip(rng.uniform(0, 1, 2 * n) * 0.5 + 0.5 * y, 0.01, 0.99)
        groups = np.array(["A"] * n + ["B"] * n)
        optimizer = create_threshold_optimizer(
            optimizer_type="group", constraint="demographic_parity", tolerance=0.05
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimizer.fit(y_true=y, y_prob=prob, sensitive_attr=groups)
        result = optimizer.result_
        assert result.is_feasible in (True, False)
        violation = result.constraint_violations[0]
        assert np.isfinite(violation.violation)
        assert violation.unmeasured == ()
        assert violation.violation_is_lower_bound is False
        assert "lower bound" not in result.summary()
