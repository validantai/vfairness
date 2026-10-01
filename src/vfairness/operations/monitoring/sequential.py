"""
Sequential drift detection for fairness metric time series.

CS-T-23. Generalizes the CUSUM change-point logic in
``vfairness.agents.temporal.TemporalTracker.detect_drift_cusum`` from a
two-group multi-turn tracker to a plain 1-D series of fairness-metric values
(one value per time window), and adds the Page-Hinkley test.

Two complementary online change detectors:

    - CUSUM (Page, 1954): cumulative sum of standardized deviations from a
      reference level. Signals a sustained shift once the running sum leaves a
      tolerance band. Good at detecting small persistent drifts.
    - Page-Hinkley: cumulative deviation from the running mean with a magnitude
      allowance ``delta``. Signals when the statistic rises ``lambda_`` above its
      running minimum. Good at detecting an increase quickly.

On the scale of a fairness metric (a disparity in roughly [0, 1]) the classic
tabular-CUSUM defaults slack k = 0.5 and threshold h = 5.0 are meant in units
of the series standard deviation (sigma), not raw units. An absolute slack of
0.5 would swamp the entire signal range. So CUSUM here standardizes the
deviation by the series std (z = (x - target) / sigma); k and h are then the
canonical sigma-unit values. A flat series (sigma ~ 0) has no variation and so
reports no drift, EXCEPT when an explicit target was supplied and the flat
level sits away from it: the excursion is then real and cannot be expressed in
sigma units at all, so the detector refuses (``has_drift`` None) rather than
answering the calm way.

Page-Hinkley standardizes the same way: the deviation from the running mean is
divided by the series std, so ``delta`` and ``lambda_`` are in sigma units and
the detector behaves identically at any metric scale. Raw-unit PH parameters
(the pre-audit defaults delta = 0.005, lambda_ = 0.05) fired on essentially
every stationary series at realistic fairness-metric noise (sigma >= 0.02),
because the PH excursion grows with the series std (Gama et al. 2014 require
the threshold to be tuned to the signal scale).

References:
    - Page, E. S. (1954). Continuous Inspection Schemes. Biometrika 41(1/2).
    - Hinkley, D. V. (1971). Inference about the change-point from cumulative
      sum tests. Biometrika 58(3).
    - Montgomery, D. C. (2013). Introduction to Statistical Quality Control
      (tabular CUSUM: k = 0.5 sigma, h = 4 or 5 sigma).
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..._not_assessed import NOT_ASSESSED, warn_not_assessed
from ..._triage import is_measured

# Numerical tolerance for the threshold crossing test. A clean step of exactly
# one sigma over exactly h/k windows lands the CUSUM statistic on the threshold
# to the last bit; comparing with a small tolerance keeps that deterministic
# rather than knife-edge on floating point.
_EPS = 1e-9

# Below this standard deviation a series is treated as constant (no variation to
# accumulate), so CUSUM reports no drift instead of dividing by ~0.
_SIGMA_FLOOR = 1e-12

# Fewest MEASURED windows either detector will standardize against. Same number
# and same reason as ``agents.temporal._MIN_TURNS_FOR_TREND``, whose CUSUM this
# module generalizes: below three points there is no mean and no sigma to
# standardize by, and a verdict computed from fewer is a verdict about nothing.
_MIN_POINTS_FOR_SIGMA = 3


def _warn_vacuous_parameters(func: str, reasons: List[str], reporting: str) -> None:
    """Disclose that a DECISION PARAMETER, not the data, stopped the test running.

    BGL5 AUDIT, 2026-09-27. Both detectors validated the series and neither
    validated the parameters that turn the accumulated statistic into a verdict,
    so a bound no data can cross was reported as the calmest reading on the
    scale. See the measured before/after in :func:`cusum_drift` and
    :func:`page_hinkley`. ``run_sprt`` in ../monitoring/drift.py was given
    exactly this guard for its alpha and beta in the previous wave; its siblings
    here were left without one.
    """
    warnings.warn(
        f"{func}: NOT ASSESSED. "
        + " ".join(reasons)
        + f" No test ran, so there is no verdict about the series: reporting {reporting}, "
        f"NOT a measured has_drift=False.",
        UserWarning,
        stacklevel=3,
    )


def _measurable(series) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Split a raw series into its values, its finite mask and the two counts.

    R-11, 2026-09-10. Both detectors opened with ``sigma = np.std(x)``, which is
    NaN as soon as ONE value is NaN, and every downstream test then answered the
    calm way by accident: ``if sigma <= _SIGMA_FLOOR`` is False for NaN so the
    constant-series shortcut did not fire, ``max(0.0, nan)`` is 0.0 so the CUSUM
    arms never left the floor, and ``nan > lambda_`` is False so Page-Hinkley
    never crossed. Measured::

        A) 100 windows containing a 17-sigma step        -> abrupt_drift, max_cusum 24.61
        B) the SAME 100 windows plus 10 uncomputable     -> stable,       max_cusum 0.00, warnings []
        C) an empty series                               -> stable
        D) a genuinely flat measured series              -> stable

    B was indistinguishable from D: ten windows nobody could compute erased a
    seventeen-sigma step and reported the calmest verdict on the scale.

    The unmeasurable windows are EXCLUDED and disclosed rather than poisoning
    the statistic, which is the policy the sibling
    ``TemporalTracker.detect_drift_cusum`` already follows. Positional
    alignment is preserved by the callers (a NaN goes into the per-step arrays
    at every excluded index and the accumulator carries across the gap), so
    ``drift_index`` still indexes the series the caller passed in.
    """
    x = np.asarray(series, dtype=float).ravel()
    finite = np.isfinite(x)
    return x, finite, int(finite.sum()), int(len(x))


def cusum_drift(
    series,
    target: Optional[float] = None,
    slack: float = 0.5,
    threshold: float = 5.0,
) -> Dict:
    """Two-sided CUSUM change-point detection on a 1-D series.

    Standardizes each value by the series standard deviation and accumulates the
    positive and negative one-sided CUSUM statistics::

        z_i   = (x_i - target) / sigma
        S_pos = max(0, S_pos + (z_i - slack))
        S_neg = max(0, S_neg - z_i - slack)

    Drift is flagged the first time either arm reaches ``threshold``. ``slack``
    (k) and ``threshold`` (h) are in units of sigma (the canonical tabular-CUSUM
    convention); the defaults k = 0.5, h = 5.0 are the textbook values.

    Args:
        series: 1-D sequence of metric values (for example one fairness metric
            per time window).
        target: Reference level. Defaults to the series mean.
        slack: Allowed slack / reference value k, in sigma units.
        threshold: Decision interval h, in sigma units.

    Returns:
        Dict with keys:
            has_drift: Whether either CUSUM arm crossed the threshold. THREE
                STATES: ``True``, ``False``, or ``None`` when the chart could
                not be drawn at all. Five ways to reach ``None``: fewer than
                ``_MIN_POINTS_FOR_SIGMA`` measured windows, so there is no
                sigma to standardize by; a constant measured series sitting
                away from an explicit ``target``, where sigma is 0 and the
                excursion from that target therefore has no sigma-unit size;
                a ``slack`` or ``threshold`` that specifies no test (non-finite,
                a negative slack, or a threshold at or below 0); a finite
                ``slack`` so large that no window in the series could add
                anything to either arm; and a finite ``threshold`` above the
                largest value either arm could reach on a series of this length
                and scale. That last route is BGL6, measured 2026-09-29: on
                ``[0.10] * 25 + [0.40] * 25`` the defaults flag drift at index 9
                with max_cusum 12.500000000000002, while ``threshold=1e9`` and
                ``threshold=1e300`` each answered a measured ``False`` carrying
                that same max_cusum 12.500000000000002 and ZERO warnings.
                ``None`` is never a synonym for ``False``, and each route warns.
            drift_index: Index of the first crossing, or None. Indexes the
                series as PASSED IN, including any unmeasurable windows.
            cusum_pos: Positive-arm statistic per step (sigma units), ``nan``
                at any window that could not be measured.
            cusum_neg: Negative-arm statistic per step (sigma units), ``nan``
                at any window that could not be measured.
            max_cusum: Largest value reached by either arm, ``nan`` when not
                assessed (never 0.0, which is the calmest reading on the scale).
            mean: Mean of the measured windows (the default reference level),
                ``nan`` when not assessed.
            n_measured / n_windows: How many of the supplied windows carried a
                finite value, and how many were supplied.
    """
    x, finite, n_measured, n = _measurable(series)

    # BGL5 AUDIT, 2026-09-27. The two DECISION parameters were never checked,
    # although they are the only inputs that turn the accumulated statistic into a
    # verdict. Measured on a 100-window series carrying a genuine 80-point step,
    # which the defaults call drift at index 9 with max_cusum 24.99:
    #   threshold=nan -> has_drift False, drift_index None, max_cusum 24.99,
    #                    warnings []   (the statistic reached 25 and the verdict
    #                    said no drift: a contradiction inside one dict)
    #   threshold=inf -> the same
    #   threshold=0.0 and threshold=-5.0 -> has_drift True at index 0, for ANY
    #                    series, because both arms start at 0.0
    #   slack=nan     -> has_drift False, max_cusum 0.0, warnings []
    #   slack=1e9     -> has_drift False, max_cusum 0.0, warnings []  (FINITE, so
    #                    this is not a NaN-hygiene point)
    # After this guard every one of those returns has_drift None, max_cusum nan
    # and a UserWarning naming the parameter. The defaults are untouched: the same
    # stepped series still returns has_drift True, drift_index 9, max_cusum
    # 24.98554351979067.
    # ``is_measured`` rather than ``np.isfinite``: it is the library's canonical
    # rule and also refuses None, a bool and a string, none of which is a decision
    # boundary either, and np.isfinite raises TypeError on all three.
    param_reasons: List[str] = []
    if not is_measured(slack):
        param_reasons.append(
            f"slack is {slack!r}, not a finite number, so every comparison of a "
            f"standardized deviation against it is undefined and both arms stay pinned "
            f"at 0.0."
        )
    elif slack < 0.0:
        param_reasons.append(
            f"slack is {slack:g}: a negative allowance ADDS to both arms at every window, "
            f"so a series that does not move still crosses eventually."
        )
    if not is_measured(threshold):
        param_reasons.append(
            f"threshold is {threshold!r}, which no accumulated statistic can reach, so the "
            f"chart cannot fire whatever the data does."
        )
    elif threshold <= 0.0:
        param_reasons.append(
            f"threshold is {threshold:g}: both CUSUM arms are non-negative by construction, "
            f"so the first window crosses it whatever the data does."
        )
    if param_reasons:
        _warn_vacuous_parameters("cusum_drift", param_reasons, "has_drift=None and a nan max_cusum")
        return {
            "has_drift": None,
            "drift_index": None,
            "cusum_pos": [float("nan")] * n,
            "cusum_neg": [float("nan")] * n,
            "max_cusum": float("nan"),
            "mean": float("nan"),
            "n_measured": n_measured,
            "n_windows": n,
        }

    if n_measured < _MIN_POINTS_FOR_SIGMA:
        warn_not_assessed(
            "cusum_drift",
            measured=n_measured,
            total=n,
            unit="window(s) carried a finite metric value",
            requirement=(
                f"a CUSUM needs at least {_MIN_POINTS_FOR_SIGMA} to have a mean and a "
                f"sigma to standardize against"
            ),
            reporting="has_drift=None and a nan max_cusum",
            instead_of="False with max_cusum 0.0",
            stacklevel=3,
        )
        return {
            "has_drift": None,
            "drift_index": None,
            "cusum_pos": [float("nan")] * n,
            "cusum_neg": [float("nan")] * n,
            "max_cusum": float("nan"),
            "mean": float("nan"),
            "n_measured": n_measured,
            "n_windows": n,
        }

    if n_measured < n:
        warnings.warn(
            f"cusum_drift: {n - n_measured} of {n} window(s) carried no finite metric "
            f"value and were EXCLUDED; the chart rests on the {n_measured} that remain, "
            f"and their per-step entries are nan. Before this they made sigma NaN, which "
            f"pinned both arms at 0.0 and reported no drift.",
            UserWarning,
            stacklevel=2,
        )

    measured = x[finite]
    mean_val = float(np.mean(measured))
    tgt = mean_val if target is None else float(target)
    sigma = float(np.std(measured))

    if sigma <= _SIGMA_FLOOR:
        # BGL-3, measured 2026-09-27. A constant series sitting AWAY from the
        # reference level the caller declared is the one case where "no
        # variation to accumulate" is not the whole question. z = (x - target)
        # / sigma cannot be formed at all when sigma is 0, and the excursion
        # the caller asked about is real:
        #
        #   cusum_drift([0.9] * 50, target=0.1)
        #     -> has_drift False, max_cusum 0.00, warnings []
        #   sequential_fairness_drift([0.9] * 50, {"target": 0.1})
        #     -> classification "stable"
        #
        # A process pinned 0.8 above its own reference for fifty consecutive
        # windows, reported with the calmest reading the scale has, beside a
        # genuinely on-target flat series that returns the identical dict. This
        # is a tabular CUSUM (Page 1954: continuous inspection against a
        # target), so being off target IS the thing it exists to detect, and the
        # threshold is stated in sigma units that do not exist here. Refuse:
        # has_drift None and a nan max_cusum, never False with 0.0.
        off_target = abs(mean_val - tgt) > _EPS
        if target is not None and off_target:
            warn_not_assessed(
                "cusum_drift",
                measured=n_measured,
                total=n,
                unit=(
                    f"window(s) carried a finite metric value, all of them at "
                    f"{mean_val:.6g} against a reference target of {tgt:.6g}"
                ),
                requirement=(
                    "a CUSUM standardizes by the series sigma, which is 0 for a "
                    "constant series, so an excursion from the target cannot be "
                    "expressed in the sigma units slack and threshold are given in"
                ),
                reporting="has_drift=None and a nan max_cusum",
                instead_of="False with max_cusum 0.0, the calmest reading on the scale",
                stacklevel=3,
            )
            return {
                "has_drift": None,
                "drift_index": None,
                "cusum_pos": [float("nan")] * n,
                "cusum_neg": [float("nan")] * n,
                "max_cusum": float("nan"),
                "mean": mean_val,
                "n_measured": n_measured,
                "n_windows": n,
            }

        # Constant MEASURED series ON its reference level (or with no reference
        # level but its own mean): no variation to accumulate. This is a
        # verdict, not a refusal: the windows were read and they do not move.
        return {
            "has_drift": False,
            "drift_index": None,
            "cusum_pos": [0.0 if ok else float("nan") for ok in finite],
            "cusum_neg": [0.0 if ok else float("nan") for ok in finite],
            "max_cusum": 0.0,
            "mean": mean_val,
            "n_measured": n_measured,
            "n_windows": n,
        }

    cusum_pos: List[float] = []
    cusum_neg: List[float] = []
    s_pos = 0.0
    s_neg = 0.0
    max_cusum = 0.0
    max_abs_z = 0.0
    drift_index: Optional[int] = None

    for i in range(n):
        if not finite[i]:
            # The window contributes nothing and the accumulator carries across
            # it. A nan here keeps cusum_pos/cusum_neg the same length as the
            # input, so drift_index still indexes the caller's own series.
            cusum_pos.append(float("nan"))
            cusum_neg.append(float("nan"))
            continue
        z = (x[i] - tgt) / sigma
        max_abs_z = max(max_abs_z, abs(z))
        s_pos = max(0.0, s_pos + (z - slack))
        s_neg = max(0.0, s_neg - z - slack)
        cusum_pos.append(float(s_pos))
        cusum_neg.append(float(s_neg))
        max_cusum = max(max_cusum, s_pos, s_neg)
        if drift_index is None and (s_pos >= threshold - _EPS or s_neg >= threshold - _EPS):
            drift_index = i

    # A FINITE slack can be just as vacuous as a NaN one, and that is the half of
    # this defect a finiteness check cannot see. Measured 2026-09-27, same stepped
    # series: slack=1e9 -> has_drift False, max_cusum 0.0, warnings []. No window
    # contributed anything at all, so "no drift" was a statement about the slack,
    # not about the data. Now has_drift None with the largest standardized
    # deviation in the series named.
    #
    # The test is a TOLERANCE, not `max_cusum == 0.0`: an accumulated statistic
    # that barely left the floor (a slack a hair under the largest deviation)
    # would slip past exact equality, which is how run_sprt's own constancy guard
    # was defeated by one bit of jitter. For a series standardized by its own
    # sigma the largest absolute deviation is always at least 1.0, so a real
    # tabular-CUSUM slack (k = 0.5) always accumulates and never lands here.
    #
    # BGL6 AUDIT, 2026-09-29. The paragraph above was written for `slack` ALONE
    # and the sibling decision parameter walked straight through it, so the half
    # of the defect it describes stayed live one parameter over. Measured on
    # [0.10] * 25 + [0.40] * 25, a real abrupt step the defaults flag at index 9
    # with max_cusum 12.500000000000002:
    #   slack=1e9       -> has_drift None, max_cusum nan, 1 warning   (this block)
    #   threshold=1e9   -> has_drift False, max_cusum 12.500000000000002, 0 warnings
    #   threshold=1e300 -> identical
    # Both arms are bounded by n_measured * max|z| (about 50 for that series), so
    # 1e9 is out of reach BY CONSTRUCTION, and "no drift" is then a statement
    # about the threshold rather than about the data. That is verbatim the
    # argument the paragraph above makes for slack, and the finiteness guard at
    # the top of this function already says as much in words ("which no
    # accumulated statistic can reach") while only testing it for a non-finite
    # value. WHEN TWO PARAMETERS SHARE A PRECONDITION, CHECKING ONE OF THEM IS
    # NOT CHECKING THE PRECONDITION: both are judged here, in ONE block, above
    # the `has_drift = drift_index is not None` dispatch below.
    reach_reasons: List[str] = []
    if drift_index is None and max_cusum <= _EPS and max_abs_z > _EPS:
        reach_reasons.append(
            f"slack is {slack:g} and the largest standardized deviation in the series is "
            f"{max_abs_z:.4g}, so no window could add anything to either arm and the "
            f"statistic never left 0.0."
        )
    # A CEILING, deliberately not the observed max_cusum. Falling short of a
    # reachable threshold is an ordinary MEASURED no-drift and must stay one; a
    # threshold above the ceiling could not have fired for any arrangement of
    # these windows, so False would be arithmetic on the parameter. The bound is
    # provable: slack >= 0 here, so each window adds at most max(0, |z| - slack)
    # <= max|z| to one arm, over n_measured windows. It is loose on purpose,
    # because erring wide refuses fewer honest measurements.
    reachable = n_measured * max_abs_z
    if drift_index is None and threshold > reachable:
        reach_reasons.append(
            f"threshold is {threshold:g}, and the largest value either CUSUM arm could "
            f"reach on {n_measured} window(s) whose largest standardized deviation is "
            f"{max_abs_z:.4g} is {reachable:.4g}, so the chart could not have fired for any "
            f"arrangement of these windows."
        )
    if reach_reasons:
        _warn_vacuous_parameters(
            "cusum_drift",
            reach_reasons,
            "has_drift=None and a nan max_cusum",
        )
        return {
            "has_drift": None,
            "drift_index": None,
            "cusum_pos": [float("nan")] * n,
            "cusum_neg": [float("nan")] * n,
            "max_cusum": float("nan"),
            "mean": mean_val,
            "n_measured": n_measured,
            "n_windows": n,
        }

    return {
        "has_drift": drift_index is not None,
        "drift_index": drift_index,
        "cusum_pos": cusum_pos,
        "cusum_neg": cusum_neg,
        # Tracked in the loop rather than by max() over the lists: those now
        # hold nan at excluded windows, and Python's max() with a nan in the
        # sequence returns whichever value the comparison order happens to land
        # on, which is not the maximum.
        "max_cusum": float(max_cusum),
        "mean": mean_val,
        "n_measured": n_measured,
        "n_windows": n,
    }


def page_hinkley(
    series,
    delta: float = 0.05,
    lambda_: float = 20.0,
    alpha: float = 0.9999,
) -> Dict:
    """Page-Hinkley test for an increase in a 1-D series.

    Tracks the cumulative standardized deviation of each value from the running
    mean, with a magnitude allowance ``delta`` and a forgetting factor
    ``alpha``::

        mean_i = running mean of x_0 .. x_i
        z_i    = (x_i - mean_i) / sigma        (sigma = series std)
        m_i    = alpha * m_{i-1} + (z_i - delta)
        PH_i   = m_i - min(m_0 .. m_i)

    Drift is flagged the first time ``PH_i`` exceeds ``lambda_``. ``alpha``
    defaults to ~1, so ``m`` is essentially the plain cumulative sum specified
    for the increase variant while still honoring the standard forgetting factor.

    ``delta`` and ``lambda_`` are in units of the series standard deviation
    (sigma), matching :func:`cusum_drift`; the detector is therefore invariant
    to the metric's scale. The defaults (delta = 0.05 sigma, lambda_ = 20
    sigma) were calibrated by simulation: on stationary iid noise of any scale
    the false-alarm rate is ~0.3% at 50 points, ~1% at 100 points and ~9% at
    200 points, while a sustained 2-sigma mean step, a 4-sigma step (median
    detection lag ~12 points) and a 5-sigma linear ramp over 100 points are all
    detected in 100% of simulations. The pre-audit raw-unit defaults
    (delta = 0.005, lambda_ = 0.05) flagged ~100% of stationary series at
    fairness-metric noise levels (sigma >= 0.02). A flat series (sigma ~ 0)
    has no variation to accumulate and reports no drift.

    Args:
        series: 1-D sequence of metric values.
        delta: Magnitude allowance (tolerated drift per step), in sigma units.
        lambda_: Detection threshold on the PH statistic, in sigma units.
        alpha: Forgetting factor for the cumulative sum (~1 = no forgetting).

    Returns:
        Dict with keys:
            has_drift: Whether the PH statistic exceeded lambda_. THREE STATES,
                exactly as :func:`cusum_drift`: ``None`` when fewer than
                ``_MIN_POINTS_FOR_SIGMA`` windows carried a finite value, so
                there was no sigma to standardize by; when ``delta``, ``lambda_``
                or ``alpha`` specifies no test (non-finite, a negative delta, a
                lambda_ at or below 0, an alpha outside (0, 1]); when a finite
                ``delta`` exceeds every standardized deviation in the series, so
                the statistic could not rise; or when a finite ``lambda_`` sits
                above the largest value the PH statistic could reach on a series
                of this length and scale. That last route is BGL6, measured
                2026-09-29: on ``[0.10] * 25 + [0.40] * 25`` the defaults flag
                drift at index 38 with the statistic peaking at
                32.87119981717022, while ``lambda_=1e9`` and ``lambda_=1e300``
                each answered a measured ``False`` with ZERO warnings. Never a
                synonym for ``False``, and each route warns.
            drift_index: Index where it first exceeded lambda_, or None.
                Indexes the series as PASSED IN.
            ph_statistic: PH_i per step (m_i minus its running minimum, >= 0),
                in sigma units, ``nan`` at any unmeasurable window.
            cumulative_mean: Running mean of the measured windows per step (raw
                metric units), ``nan`` at any unmeasurable window.
            min_ph: Running minimum of the cumulative sum m (the M_T term),
                in sigma units, ``nan`` when not assessed.
            n_measured / n_windows: How many of the supplied windows carried a
                finite value, and how many were supplied.
    """
    x, finite, n_measured, n = _measurable(series)

    # BGL5 AUDIT, 2026-09-27, the same gap as in cusum_drift above. Measured on a
    # 100-window series carrying a genuine 80-point step, which the defaults flag
    # at index 61 with min_ph -2.2727160989587887:
    #   lambda_=nan -> has_drift False, drift_index None, warnings []
    #   lambda_=inf -> the same
    #   delta=nan   -> has_drift False, drift_index None, min_ph inf, warnings []
    #                  (min_ph inf is itself the tell that nothing accumulated)
    #   alpha=0.0   -> has_drift False, drift_index None, warnings []
    #   delta=1e9   -> has_drift False, drift_index None, warnings []
    # After this guard all five return has_drift None with a nan min_ph and a
    # UserWarning naming the parameter; the defaults still flag index 61.
    param_reasons: List[str] = []
    if not is_measured(delta):
        param_reasons.append(
            f"delta is {delta!r}, so the cumulative sum is undefined from the first window on "
            f"and min_ph comes back as inf."
        )
    elif delta < 0.0:
        param_reasons.append(
            f"delta is {delta:g}: a negative allowance ADDS to the statistic at every window, "
            f"so it rises on a series that does not move."
        )
    if not is_measured(lambda_):
        param_reasons.append(
            f"lambda_ is {lambda_!r}, which the statistic cannot exceed, so the test cannot fire "
            f"whatever the data does."
        )
    elif lambda_ <= 0.0:
        param_reasons.append(
            f"lambda_ is {lambda_:g}: the PH statistic is non-negative by construction, so the "
            f"first window that moves at all crosses it whatever the data does."
        )
    if not is_measured(alpha) or not (0.0 < alpha <= 1.0):
        param_reasons.append(
            f"alpha is {alpha!r}, outside the (0, 1] range a forgetting factor is defined on; at "
            f"or below 0 nothing is carried between windows, so there is no cumulative sum to "
            f"test."
        )
    if param_reasons:
        _warn_vacuous_parameters(
            "page_hinkley", param_reasons, "has_drift=None and a nan ph_statistic"
        )
        return {
            "has_drift": None,
            "drift_index": None,
            "ph_statistic": [float("nan")] * n,
            "cumulative_mean": [float("nan")] * n,
            "min_ph": float("nan"),
            "n_measured": n_measured,
            "n_windows": n,
        }

    if n_measured < _MIN_POINTS_FOR_SIGMA:
        warn_not_assessed(
            "page_hinkley",
            measured=n_measured,
            total=n,
            unit="window(s) carried a finite metric value",
            requirement=(
                f"the Page-Hinkley statistic needs at least {_MIN_POINTS_FOR_SIGMA} to "
                f"have a sigma to standardize against"
            ),
            reporting="has_drift=None and a nan ph_statistic",
            instead_of="False with a flat 0.0 ph_statistic",
            stacklevel=3,
        )
        return {
            "has_drift": None,
            "drift_index": None,
            "ph_statistic": [float("nan")] * n,
            "cumulative_mean": [float("nan")] * n,
            "min_ph": float("nan"),
            "n_measured": n_measured,
            "n_windows": n,
        }

    if n_measured < n:
        warnings.warn(
            f"page_hinkley: {n - n_measured} of {n} window(s) carried no finite metric "
            f"value and were EXCLUDED; the statistic rests on the {n_measured} that "
            f"remain, and their per-step entries are nan. Before this they made sigma "
            f"NaN, so `ph > lambda_` was False at every step and no drift could fire.",
            UserWarning,
            stacklevel=2,
        )

    measured = x[finite]
    sigma = float(np.std(measured))

    if sigma <= _SIGMA_FLOOR:
        # Constant MEASURED series: no variation to accumulate (mirrors the
        # cusum_drift guard; without it the standardization divides by ~0).
        # A verdict, not a refusal.
        cm: List[float] = []
        running = 0.0
        seen = 0
        for i in range(n):
            if not finite[i]:
                cm.append(float("nan"))
                continue
            running += x[i]
            seen += 1
            cm.append(float(running / seen))
        return {
            "has_drift": False,
            "drift_index": None,
            "ph_statistic": [0.0 if ok else float("nan") for ok in finite],
            "cumulative_mean": cm,
            "min_ph": 0.0,
            "n_measured": n_measured,
            "n_windows": n,
        }

    cumulative_mean: List[float] = []
    ph_statistic: List[float] = []
    m = 0.0
    min_m = float("inf")
    running_sum = 0.0
    seen = 0
    max_abs_dev = 0.0
    drift_index: Optional[int] = None

    for i in range(n):
        if not finite[i]:
            # Excluded, and the running mean and the cumulative sum m carry
            # across it unchanged. The nan keeps both per-step arrays aligned
            # with the caller's series.
            cumulative_mean.append(float("nan"))
            ph_statistic.append(float("nan"))
            continue
        running_sum += x[i]
        seen += 1
        mean_i = running_sum / seen
        cumulative_mean.append(float(mean_i))
        deviation = (x[i] - mean_i) / sigma
        max_abs_dev = max(max_abs_dev, abs(deviation))
        m = alpha * m + (deviation - delta)
        if m < min_m:
            min_m = m
        ph = m - min_m
        ph_statistic.append(float(ph))
        if drift_index is None and ph > lambda_:
            drift_index = i

    # The FINITE half of the same defect, as in cusum_drift. Measured 2026-09-27:
    # delta=1e9 on the stepped series -> has_drift False, drift_index None, no
    # warning, because every window subtracted a billion sigma from the cumulative
    # sum, so the statistic could not rise at all. A delta larger than the biggest
    # standardized deviation the series contains is an allowance no observation can
    # exceed, which is not a test of this series. Narrow on purpose: a legitimate
    # delta (0.05 sigma) is far below the largest deviation of any series
    # standardized by its own sigma, which is always at least 1.0, and a monotone
    # series whose PH statistic genuinely never rises keeps its measured False
    # because its deviations are larger than the allowance.
    #
    # BGL6 AUDIT, 2026-09-29, symmetric with the cusum_drift ceiling. This check
    # was written for `delta` ALONE, so the same hole stayed live one parameter
    # over. Measured on [0.10] * 25 + [0.40] * 25, a real step the defaults flag
    # at index 38 with the PH statistic peaking at 32.87119981717022:
    #   delta=1e9      -> has_drift None, min_ph nan, 1 warning   (this block)
    #   lambda_=1e9    -> has_drift False, drift_index None, 0 warnings
    #   lambda_=1e300  -> identical
    # The finiteness guard at the top already states the reason in words ("which
    # the statistic cannot exceed, so the test cannot fire whatever the data
    # does") while only testing it for a NON-FINITE value, and a finite
    # unreachable lambda_ satisfies that same sentence. Both parameters are
    # judged here now, in one block, above the has_drift dispatch below.
    reach_reasons: List[str] = []
    if drift_index is None and delta > max_abs_dev:
        reach_reasons.append(
            f"delta is {delta:g} and the largest standardized deviation from the running "
            f"mean is {max_abs_dev:.4g}, so every window subtracted more than the data "
            f"could supply and the statistic could not rise."
        )
    # A provable CEILING on the PH statistic, not the value it happened to reach.
    # alpha lies in (0, 1] so |alpha * m| <= |m| and the sum never amplifies,
    # giving |m_i| <= sum |z_k - delta| <= n_measured * (max|dev| + delta); PH_i
    # is m_i minus a running minimum of that same sequence, hence at most twice
    # it. Wide on purpose, for the same reason as the cusum ceiling: an
    # over-generous bound refuses fewer honest measurements, and the default
    # lambda_ = 20 sigma sits far below it on the 50 to 200 window series it was
    # calibrated for.
    reachable = 2.0 * n_measured * (max_abs_dev + delta)
    if drift_index is None and lambda_ > reachable:
        reach_reasons.append(
            f"lambda_ is {lambda_:g}, and the largest value the PH statistic could reach on "
            f"{n_measured} window(s) whose largest standardized deviation from the running "
            f"mean is {max_abs_dev:.4g} is {reachable:.4g}, so the test could not have fired "
            f"for any arrangement of these windows."
        )
    if reach_reasons:
        _warn_vacuous_parameters(
            "page_hinkley",
            reach_reasons,
            "has_drift=None and a nan ph_statistic",
        )
        return {
            "has_drift": None,
            "drift_index": None,
            "ph_statistic": [float("nan")] * n,
            "cumulative_mean": cumulative_mean,
            "min_ph": float("nan"),
            "n_measured": n_measured,
            "n_windows": n,
        }

    return {
        "has_drift": drift_index is not None,
        "drift_index": drift_index,
        "ph_statistic": ph_statistic,
        "cumulative_mean": cumulative_mean,
        "min_ph": float(min_m),
        "n_measured": n_measured,
        "n_windows": n,
    }


def sequential_fairness_drift(
    series,
    cusum_kwargs: Optional[Dict] = None,
    ph_kwargs: Optional[Dict] = None,
) -> Dict:
    """Run CUSUM and Page-Hinkley on a fairness-metric series and classify it.

    Classification:
        stable: neither detector flags drift. This is a MEASURED verdict and it
            requires both detectors to have run.
        abrupt_drift: a change point is found and the shift in the series mean
            across that point is large relative to the series std (a step).
        gradual_drift: a detector flags drift but the shift at the change point
            is small relative to the series std (a slow trend).
        not_assessed: at least one detector could not run, so "neither
            detector flags drift" is not a statement anyone can make. Reached
            when fewer than ``_MIN_POINTS_FOR_SIGMA`` windows carried a finite
            value (both detectors refuse); when ``cusum_kwargs`` supplies a
            ``target`` and the measured series is constant away from it (CUSUM
            refuses; see its sigma-floor branch); or when ``cusum_kwargs`` /
            ``ph_kwargs`` carry a DECISION PARAMETER that specifies no test, for
            example ``{"threshold": nan}`` or ``{"lambda_": nan}``. That last
            route is BGL5, measured 2026-09-27: on a series carrying an 80-point
            step this wrapper answered "stable" with zero warnings for both of
            those, because each detector returned a measured False rather than
            None. A finite but UNREACHABLE decision parameter specifies no test
            either, and that half stayed live until BGL6, measured 2026-09-29: on
            ``[0.10] * 25 + [0.40] * 25`` (the defaults: "abrupt_drift")
            ``{"threshold": 1e300}`` with ``{"lambda_": 1e300}`` answered
            "stable" with ZERO warnings, and 1e9 the same. This wrapper's own
            three-state logic was correct throughout; it published the calmest
            classification because both detectors handed it a measured False.
            Fixed entirely in the two detectors. It is NOT "stable". See the
            measurement in :func:`_measurable`, where an empty series and a
            series ten of whose windows were uncomputable both reported
            "stable" beside a genuinely flat one.

    Args:
        series: 1-D sequence of metric values (one per time window).
        cusum_kwargs: Optional overrides passed to ``cusum_drift``.
        ph_kwargs: Optional overrides passed to ``page_hinkley``.

    Returns:
        Dict with keys: series (list), cusum (dict), page_hinkley (dict),
        classification (one of "stable", "gradual_drift", "abrupt_drift",
        "not_assessed"), n_measured, n_windows.
    """
    x, finite, n_measured, n = _measurable(series)
    cusum = cusum_drift(x, **(cusum_kwargs or {}))
    ph = page_hinkley(x, **(ph_kwargs or {}))

    common = {
        "series": x.tolist(),
        "cusum": cusum,
        "page_hinkley": ph,
        "n_measured": n_measured,
        "n_windows": n,
    }

    # Three states before two. has_drift is None from BOTH detectors exactly
    # when neither could run, and `not None` is True, so the original
    # `if not cusum[...] and not ph[...]` branch answered "stable" for it.
    if cusum["has_drift"] is None or ph["has_drift"] is None:
        return {**common, "classification": NOT_ASSESSED}

    # Statistics over the MEASURED windows only. np.std / np.mean over a series
    # holding one NaN are NaN, and `jump >= std` is False for NaN, which sent a
    # real step down the gradual_drift branch.
    std = float(np.std(x[finite]))

    # Prefer the Page-Hinkley change point: it tends to sit at the true change,
    # while CUSUM lags by its decision interval. Fall back to the CUSUM index.
    change_idx = ph["drift_index"] if ph["has_drift"] else cusum["drift_index"]

    if not cusum["has_drift"] and not ph["has_drift"]:
        classification = "stable"
    else:
        jump = 0.0
        if change_idx is not None and 0 < change_idx < n:
            before = x[:change_idx][finite[:change_idx]]
            after = x[change_idx:][finite[change_idx:]]
            if len(before) and len(after):
                jump = abs(float(np.mean(after)) - float(np.mean(before)))
        elif change_idx is not None and change_idx > 0:
            prev = x[change_idx - 1]
            if finite[change_idx] and finite[change_idx - 1]:
                jump = abs(float(x[change_idx]) - float(prev))
        is_abrupt = change_idx is not None and std > 0 and jump >= std
        classification = "abrupt_drift" if is_abrupt else "gradual_drift"

    return {**common, "classification": classification}
