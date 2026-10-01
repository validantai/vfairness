"""
Unit 2: Drift Detection: FairnessDriftDetector
================================================

This module implements the statistical drift-detection machinery described in
Part 4, Unit 2 of the Fairness Pipeline Development Toolkit.

The central class, :class:`FairnessDriftDetector`, detects fairness drift at
multiple temporal scales using:

1. **Kolmogorov-Smirnov (KS) two-sample test**: a non-parametric test that
   compares the empirical CDFs of a reference window and a current window.
   The KS statistic is the maximum absolute difference between the two CDFs;
   the p-value expresses the probability of observing this difference by chance.

2. **Wavelet decomposition** (via ``pywt``, optional): decomposes the metric
   time series into frequency components (approximation = long-term trend;
   detail levels = progressively shorter cycles).  Running the KS test on each
   component separately allows detection of drift at multiple time scales
   simultaneously: a technique that catches both sudden shocks *and* slow,
   insidious biases that are invisible in daily or weekly reports.  Because
   the reconstructed components are strongly autocorrelated, the per-scale
   p-values are calibrated by max-T permutation resampling rather than taken
   from the analytic KS distribution (which assumes independent samples and
   collapses to ~0 on smoothed noise).

3. **Sequential Probability Ratio Test (SPRT)**: an online, sample-efficient
   test for detecting whether a metric has crossed a pre-specified threshold,
   without requiring a fixed sample size.

4. **MMD-based drift**: distribution-level detection via the unbiased MMD²
   statistic with a median-heuristic bandwidth and a permutation-calibrated
   threshold (Gretton et al. 2012); :meth:`check_drift` additionally reports
   the raw :func:`mmd_gaussian` score from :mod:`tracker` as a diagnostic.

When ``pywt`` is not installed, :meth:`detect_drift_multiscale` falls back to
a single-scale KS test on the raw series.

References
----------
Rabanser et al. (2019). Failing loudly: An empirical study of methods for
  detecting dataset shift. NeurIPS.
Garg et al. (2024). Fairness in the face of distribution shift: A survey.
  Nature Machine Intelligence.
Kumar et al. (2023). Fine-tuning can distort pretrained features. ICLR.
Page, E.S. (1954). Continuous inspection schemes. Biometrika 41(1/2), 100-115.
Gretton et al. (2012). A kernel two-sample test. JMLR 13, 723-773.
Westfall, P.H. & Young, S.S. (1993). Resampling-Based Multiple Testing. Wiley.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .tracker import mmd_gaussian

# Try to import pywt for wavelet decomposition (optional dependency).
try:
    import pywt  # type: ignore

    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


#: Smallest number of GENUINELY OBSERVED readings a window must hold for a
#: two-sample comparison to be attempted. It is not a new number: it is the same
#: per-arm floor ``_run_ks`` and ``_permutation_calibrated_scales`` already
#: enforce on sample counts. The difference is what it is applied to. Gaps are
#: forward-filled before any test runs, so a window can reach those checks
#: holding sixty "samples" of which one was ever observed (G-20, below).
_MIN_ARM_OBSERVATIONS = 5

#: Relative span below which a metric stream carries no variation to standardize
#: by. Used by :meth:`FairnessDriftDetector.run_sprt`, whose constancy guard
#: compared ``np.ptp(stream) == 0.0`` exactly until the BGL5 audit: a stream of
#: 40 readings spanning 1e-15 (one bit of float jitter on an averaged metric) then
#: reported ('stable', 1, -9.8e+30) where the bit-exact stream correctly reported
#: ('could_not_check', 40, nan). 1e-12 of the largest magnitude in play is about
#: four thousand float64 ulps near 1.0, i.e. rounding noise, and three orders of
#: magnitude below any movement a fairness metric reports.
_CONSTANT_STREAM_REL_TOL = 1e-12


def _resolve_split(
    split_index: Optional[int],
    n: int,
    *,
    metric: str,
    where: str,
) -> Optional[int]:
    """The reference/current boundary, or ``None`` when there is no comparison.

    ``None`` for *split_index* keeps the documented midpoint default. A boundary
    the caller DID supply is honoured only when it leaves points on both sides;
    otherwise this refuses, because one of the two arms is empty and there is
    nothing to compare against.

    R-11, 2026-09-10. Every call site wrote the guard as
    ``if split_index is not None and 0 < split_index < len(values)`` with the
    midpoint as its ``else``, so a boundary of 0 (an EMPTY reference arm) and a
    boundary of ``n`` (an empty CURRENT arm) both fell through to splitting the
    ONE window that did have points down the middle and comparing it against
    ITSELF. ``set_baseline`` calls ``series.dropna()``, so a baseline whose
    metric was never computable becomes empty in silence and produces exactly
    that boundary of 0. Measured on a 60-point window against a 60-point
    all-NaN baseline::

        real baseline  (n=60): drift_detected=True  overall=0.9950
        empty baseline (n=0) : drift_detected=False overall=0.0720  warnings=[]
        worst_scale ref_mean=0.4478 cur_mean=0.4503

    against a current-window mean of 0.4489: the two "distributions" were the
    first and second halves of one window. The mirror case, an all-NaN CURRENT
    window, drove ``mmd_score`` to 0.0, the strongest available statement that
    two distributions are identical, over a window holding no observation.

    Returning None instead of a midpoint routes every caller into the NaN /
    could-not-check path this module already has (see the ``valid_scores``
    branch in :meth:`FairnessDriftDetector.detect_drift_multiscale`), which
    reports ``overall_drift_score`` NaN and ``drift_detected`` None.
    """
    if split_index is None:
        return n // 2
    idx = int(split_index)
    if 0 < idx < n:
        return idx
    empty_arm = "reference" if idx <= 0 else "current"
    warnings.warn(
        f"{where}: the reference/current boundary for metric {metric!r} is "
        f"{idx} over {n} usable point(s), which leaves the {empty_arm} window "
        f"EMPTY, so no two-sample comparison exists. Reporting NaN / no verdict "
        f"(could not check), NOT a midpoint split of the one window that does "
        f"have points. That compares a window against itself and reads as "
        f"stability.",
        UserWarning,
        stacklevel=3,
    )
    return None


# Data classes


@dataclass
class DriftResult:
    """Drift analysis for a single temporal scale.

    Attributes
    ----------
    metric : str
        Name of the monitored fairness metric.
    scale : str
        Temporal scale label: ``"full_signal"`` for no decomposition,
        ``"approximation"`` for the low-frequency trend, or
        ``"detail_{level}"`` for a specific detail band.
    ks_statistic : float
        KS test statistic (max |CDF_ref - CDF_cur|) in [0, 1].
    p_value : float
        KS test p-value.  Small values (< ``significance_level``) indicate
        statistically significant drift.  In the wavelet path this is a
        family-wise adjusted permutation p-value (the analytic KS p-value is
        invalid on autocorrelated wavelet reconstructions); in the raw path
        it is the analytic ``ks_2samp`` p-value.
    drift_score : float
        Composite score = ks_statistic × (1 - p_value) ∈ [0, 1].
        Higher = more confident drift.
    drift_detected : bool
        ``True`` if drift is statistically significant at the configured
        significance level AND drift_score exceeds the threshold.
    reference_mean : float
        Mean of the reference window component.
    current_mean : float
        Mean of the current window component.
    reference_n : int
        Number of samples in the reference component.
    current_n : int
        Number of samples in the current component.
    """

    metric: str
    scale: str
    ks_statistic: float
    p_value: float
    drift_score: float
    drift_detected: bool
    reference_mean: float
    current_mean: float
    reference_n: int
    current_n: int

    @property
    def mean_shift(self) -> float:
        """Signed difference: current_mean - reference_mean."""
        return self.current_mean - self.reference_mean

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary.

        ``comparison_ran`` is False when no KS test was run for this scale (too
        few points on one side of the boundary, or none at all). ``drift_score``
        and ``p_value`` are then NaN and ``drift_detected`` is False only
        because the dataclass field cannot hold a third state.

        G-20, 2026-09-17: this dict is what ``generate_drift_report`` serialises
        into the JSON report, and it published ``"drift_detected": false`` for
        every refused scale with nothing beside it but a NaN a JSON reader has
        to notice. Three states, never two.
        """
        return {
            "metric": self.metric,
            "scale": self.scale,
            "ks_statistic": round(self.ks_statistic, 6),
            "p_value": round(self.p_value, 6),
            "drift_score": round(self.drift_score, 6),
            "drift_detected": self.drift_detected,
            "comparison_ran": bool(self.drift_score is not None and not np.isnan(self.drift_score)),
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "mean_shift": round(self.mean_shift, 6),
            "reference_n": self.reference_n,
            "current_n": self.current_n,
        }


@dataclass
class MultiscaleDriftResult:
    """Drift analysis across all temporal scales for a single metric.

    Attributes
    ----------
    metric : str
        Name of the monitored fairness metric.
    timestamp : datetime
        When this analysis was run.
    scales : dict[str, DriftResult]
        Per-scale results keyed by scale name.
    overall_drift_score : float
        Maximum drift_score across all scales.
    drift_detected : bool
        ``True`` if drift is detected at any scale.
    mmd_score : float, optional
        MMD score between reference and current distributions (if computed).
    decomposition_available : bool
        Whether wavelet decomposition was used.
    """

    metric: str
    timestamp: datetime
    scales: Dict[str, DriftResult]
    overall_drift_score: float
    #: None when NO scale could be computed: drift was not looked for, which is
    #: neither "drift found" nor "no drift". H-11.
    drift_detected: Optional[bool]
    mmd_score: Optional[float] = None
    decomposition_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "metric": self.metric,
            "timestamp": self.timestamp.isoformat(),
            "overall_drift_score": round(self.overall_drift_score, 6),
            "drift_detected": self.drift_detected,
            "mmd_score": round(self.mmd_score, 6) if self.mmd_score is not None else None,
            "decomposition_available": self.decomposition_available,
            "scales": {k: v.to_dict() for k, v in self.scales.items()},
        }

    @property
    def worst_scale(self) -> Optional[DriftResult]:
        """Return the scale with the highest drift score (NaN scores last)."""
        if not self.scales:
            return None
        valid = [
            r
            for r in self.scales.values()
            if r.drift_score is not None and not np.isnan(r.drift_score)
        ]
        pool = valid if valid else list(self.scales.values())
        return max(pool, key=lambda r: r.drift_score)


class FairnessDriftDetector:
    """Multi-scale statistical drift detector for fairness metrics.

    This class implements the core analytical layer described in Part 4,
    Unit 2.  It takes a time series of fairness metric values (e.g., daily
    demographic parity readings) and applies statistical tests to determine
    whether the distribution has shifted from a reference baseline.

    Wavelet-based multi-scale analysis (when ``pywt`` is installed) allows
    detection across three time horizons simultaneously:

    - **Approximation (low-frequency)**: Captures long-term secular trends.
    - **Detail (high-frequency)**: Captures sudden regime shifts or spikes.

    This mirrors the three-tier tiered monitoring strategy recommended in
    Unit 1 and is directly inspired by the MegaShop e-commerce case study.

    Wavelet reconstructions are strongly autocorrelated (the approximation is
    a low-pass smoothed curve), so the analytic ``ks_2samp`` p-value, which
    assumes independent samples, collapses toward 0 on them even for
    stationary iid noise. In the wavelet path the per-scale p-values are
    therefore calibrated by permutation (max-T resampling, Westfall & Young
    1993): the series is repeatedly shuffled, each shuffle is decomposed the
    same way, and each scale's observed KS statistic is compared against the
    null distribution of the studentized maximum across scales. This controls
    the family-wise false-alarm rate at ``significance_level`` under the null
    of an exchangeable (no-drift) series. The raw series participates as the
    ``full_signal`` scale of the same family.

    Parameters
    ----------
    window_sizes : list[int], default [24, 168, 720]
        Reference window sizes (in time-series points) for hourly, weekly,
        and monthly analysis.  Currently used as documentation; the actual
        split is always 50/50 in :meth:`detect_drift_multiscale`.
    wavelet : str, default ``"db4"``
        PyWavelets wavelet name.  ``"db4"`` (Daubechies-4) is a good
        general-purpose choice for non-stationary time series.
    significance_level : float, default 0.05
        p-value threshold for the KS test; drift is "significant" when
        p < significance_level.
    min_drift_score : float, default 0.3
        Minimum composite drift score (KS × (1-p)) needed to set
        ``drift_detected = True``.  Raising this reduces false positives.
    compute_mmd : bool, default True
        Whether to compute MMD scores when a baseline is set via
        :meth:`set_baseline`.
    n_permutations : int, default 200
        Number of permutations used to calibrate per-scale p-values in the
        wavelet path (ignored when ``pywt`` is absent; the raw-signal KS
        p-value is analytic and needs no calibration).
    random_state : int, default 0
        Seed for the permutation calibration, making verdicts reproducible.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from vfairness.operations.monitoring import FairnessDriftDetector
    >>> rng = np.random.default_rng(1)
    >>> # Stable reference, then a sudden shift
    >>> series = pd.Series(
    ...     np.concatenate([rng.normal(0.1, 0.02, 60),
    ...                     rng.normal(0.18, 0.02, 60)]),
    ...     index=pd.date_range("2025-01-01", periods=120, freq="D"),
    ... )
    >>> detector = FairnessDriftDetector()
    >>> detector.set_baseline(series.iloc[:60])
    >>> result = detector.check_drift(series.iloc[60:])
    >>> result.drift_detected
    True

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: drift_detection. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        wavelet: str = "db4",
        significance_level: float = 0.05,
        min_drift_score: float = 0.3,
        compute_mmd: bool = True,
        n_permutations: int = 200,
        random_state: int = 0,
    ) -> None:
        self.window_sizes = window_sizes or [24, 168, 720]
        self.wavelet = wavelet
        self.significance_level = significance_level
        self.min_drift_score = min_drift_score
        self.compute_mmd = compute_mmd
        self.n_permutations = n_permutations
        self.random_state = random_state

        self._baseline: Optional[pd.Series] = None
        self._results_history: List[MultiscaleDriftResult] = []

    # Baseline management

    def set_baseline(self, series: pd.Series) -> None:
        """Store a reference time series as the baseline distribution.

        Parameters
        ----------
        series : pd.Series
            Historical metric readings that represent a "known-good" state.
            Values should be float; index should be a DatetimeIndex.
        """
        cleaned = series.dropna()
        # The dropna() is the point where a baseline whose metric was never
        # computable turns into an EMPTY reference in silence, and an empty
        # reference is what made check_drift() split the current window against
        # itself. Say it here, where the caller can still see which series it
        # handed over, rather than only at the comparison two calls later.
        if len(cleaned) == 0 and len(series) > 0:
            warnings.warn(
                f"set_baseline() was given {len(series)} reading(s) and NONE of them is a "
                "number, so the stored baseline is EMPTY. Every check_drift() against it "
                "is a could-not-check, not a stable reading.",
                UserWarning,
                stacklevel=2,
            )
        elif len(cleaned) < len(series):
            # G-20. The partial case is quieter and was silent: 60 readings of
            # which 55 are gaps become a FIVE point reference, and every later
            # check_drift() compares its window against those five while the
            # caller believes the baseline is the series they handed over.
            # Nothing in the result says how wide the reference really is.
            warnings.warn(
                f"set_baseline() was given {len(series)} reading(s) of which "
                f"{len(series) - len(cleaned)} are not numbers and were dropped, so the "
                f"stored baseline is {len(cleaned)} point(s) wide. Every later "
                f"check_drift() compares against those {len(cleaned)}, not against the "
                f"{len(series)} you passed.",
                UserWarning,
                stacklevel=2,
            )
        self._baseline = cleaned.copy()

    def check_drift(
        self,
        current_series: pd.Series,
        metric: str = "metric",
    ) -> MultiscaleDriftResult:
        """Detect drift between the stored baseline and *current_series*.

        Parameters
        ----------
        current_series : pd.Series
            Recent metric readings to compare against the baseline.
        metric : str, default ``"metric"``
            Label used in the returned :class:`MultiscaleDriftResult`.

        Returns
        -------
        MultiscaleDriftResult
        """
        if self._baseline is None:
            raise RuntimeError("No baseline set.  Call set_baseline() before check_drift().")

        # BOTH arms are checked here, before anything is concatenated, because
        # this is the only place that still knows which points are the baseline
        # and which are the current window. Downstream the two are one array.
        #
        # The current arm needs its own guard on top of _resolve_split:
        # decompose_temporal_patterns() FORWARD-FILLS, so an all-NaN current
        # window is not short, it is 60 manufactured copies of the last baseline
        # reading. Measured: mmd_score 0.0 (fixed by _resolve_split) but also
        # drift_detected=True at overall 0.9751, a confident verdict computed
        # entirely from points nobody observed. Neither True nor False is
        # honest about a window holding no observation.
        n_baseline = int(self._baseline.notna().sum())
        n_current = int(pd.Series(current_series).notna().sum())
        if n_baseline == 0 or n_current == 0:
            empty = "baseline" if n_baseline == 0 else "current window"
            warnings.warn(
                f"check_drift() for metric {metric!r}: the {empty} holds no usable "
                f"reading (baseline n={n_baseline}, current n={n_current}), so the two "
                "windows cannot be compared. overall_drift_score is NaN and "
                "drift_detected is None: this is COULD NOT CHECK, not stability and not "
                "drift.",
                UserWarning,
                stacklevel=2,
            )
            return self._record(
                MultiscaleDriftResult(
                    metric=metric,
                    timestamp=datetime.now(),
                    scales={},
                    overall_drift_score=float("nan"),
                    drift_detected=None,
                    decomposition_available=False,
                )
            )

        combined = pd.concat([self._baseline, current_series])
        # Split at the baseline boundary, not the midpoint: when the two
        # arms differ in length a midpoint split would mix baseline points
        # into "current" (or vice versa) and dilute real shifts.
        return self.detect_drift_multiscale(
            combined,
            metric=metric,
            split_index=len(self._baseline),
        )

    # Core: multi-scale drift detection

    def decompose_temporal_patterns(
        self,
        time_series: pd.Series,
        wavelet: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Decompose a metric time series into frequency components.

        Uses Discrete Wavelet Transform (DWT) to separate the signal into:
        - ``"approximation"``: Low-frequency, long-term trend component.
        - ``"detail_{level}"``: High-frequency detail at each decomposition
          level (from finest = level 1 to coarsest = level N).

        Falls back to ``{"full_signal": series.values}`` when:
        - ``pywt`` is not installed.
        - The series is too short to decompose.

        Parameters
        ----------
        time_series : pd.Series
            Metric values (float).  NaN are forward-filled then dropped, so a
            gap becomes a COPY of the last observed reading and is
            indistinguishable from an observation in the returned arrays.  The
            count of genuine readings is checked by
            :meth:`detect_drift_multiscale` before any test runs; a caller using
            this method directly gets no such protection and must count its own
            observations (``series.notna().sum()``).
        wavelet : str, optional
            Override the instance wavelet for this call.

        Returns
        -------
        dict[str, np.ndarray]
            Component name → reconstructed signal array.
        """
        wvt = wavelet or self.wavelet
        series_clean = time_series.ffill().dropna()
        values = series_clean.values.astype(float)
        n = len(values)

        if not _HAS_PYWT or n < 20:
            return {"full_signal": values}

        try:
            max_level = pywt.dwt_max_level(n, pywt.Wavelet(wvt))
            level = min(max_level, 5)
            if level <= 0:
                return {"full_signal": values}

            coeffs = pywt.wavedec(values, wvt, level=level)
            components: Dict[str, np.ndarray] = {}

            # Approximation (trend)
            approx_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            recon = pywt.waverec(approx_coeffs, wvt)
            components["approximation"] = recon[:n]

            # Detail levels (high-frequency components)
            for i, detail_coeff in enumerate(coeffs[1:], start=1):
                detail_cs = [np.zeros_like(c) for c in coeffs]
                detail_cs[i] = detail_coeff
                recon = pywt.waverec(detail_cs, wvt)
                components[f"detail_{level + 1 - i}"] = recon[:n]

            return components

        except Exception as exc:
            warnings.warn(
                f"Wavelet decomposition failed ({exc}); using full signal.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"full_signal": values}

    def detect_drift_ks(
        self,
        time_series: pd.Series,
        metric: str = "metric",
        scale: str = "full_signal",
        split_index: Optional[int] = None,
    ) -> DriftResult:
        """Apply a two-sample KS test by splitting *time_series* in two.

        By default the split is at the midpoint: the first half is treated
        as the "reference" distribution, the second half as "current."
        When *split_index* is given (e.g. the baseline length from
        :meth:`check_drift`), the split honours the true reference/current
        boundary instead.

        Parameters
        ----------
        time_series : pd.Series
            Full metric time series.
        metric : str
        scale : str
            Scale label used in the returned :class:`DriftResult`.
        split_index : int, optional
            Index separating reference (before) from current (after).
            Falls back to the midpoint when omitted or out of range.

        Returns
        -------
        DriftResult
        """
        values = time_series.dropna().values.astype(float)
        if split_index is not None and len(values) != len(time_series):
            # G-20. The boundary is applied AFTER the missing readings are
            # dropped, so an index the caller measured on the series they hold
            # lands somewhere else, and the two windows reported are not the two
            # they asked to compare.
            warnings.warn(
                f"detect_drift_ks: '{metric}' [{scale}] was given a reference/current "
                f"boundary of {int(split_index)}, but {len(time_series) - len(values)} of "
                f"the {len(time_series)} reading(s) are missing and are dropped before the "
                f"split, so the boundary is applied to the {len(values)} remaining points "
                f"and does not fall where you counted it. Drop the gaps yourself, or pass "
                f"an index into the cleaned series.",
                UserWarning,
                stacklevel=2,
            )
        if len(values) < 20:
            # The verdict field stays a plain ``False`` on purpose (it is
            # declared bool, tests/test_readiness2_explainer.py pins the
            # annotation, and every aggregator keys on the NaN drift_score),
            # so the WARNING is what tells a caller reading only the boolean
            # that no test ran. Without it this was the last silent refusal in
            # this file.
            warnings.warn(
                f"detect_drift_ks: '{metric}' [{scale}] has {len(values)} usable "
                f"point(s), below the 20 a two-sample KS test needs here, so no test "
                f"was run. drift_detected is False only because the field cannot hold "
                f"a third state: the NaN drift_score and p_value are the refusal, and "
                f"this is COULD NOT CHECK, not 'no drift'.",
                UserWarning,
                stacklevel=2,
            )
            return DriftResult(
                metric=metric,
                scale=scale,
                ks_statistic=np.nan,
                p_value=np.nan,
                drift_score=np.nan,
                drift_detected=False,
                reference_mean=np.nan,
                current_mean=np.nan,
                reference_n=0,
                current_n=0,
            )

        split = _resolve_split(
            split_index, len(values), metric=metric, where=f"detect_drift_ks[{scale}]"
        )
        if split is None:
            # NaN across the board so the aggregate in detect_drift_multiscale
            # lands in its could-not-check branch. drift_detected stays a plain
            # False here and is NOT widened to None: the per-scale field is
            # declared bool on purpose (tests/test_readiness2_explainer.py pins
            # the annotation, and explainer.py reads it two-state under that
            # pin). The NaN drift_score is what every aggregator and the SVG
            # adapter actually key on.
            return DriftResult(
                metric=metric,
                scale=scale,
                ks_statistic=np.nan,
                p_value=np.nan,
                drift_score=np.nan,
                drift_detected=False,
                reference_mean=np.nan,
                current_mean=np.nan,
                reference_n=0,
                current_n=0,
            )
        reference = values[:split]
        current = values[split:]
        return self._run_ks(reference, current, metric=metric, scale=scale)

    def detect_drift_multiscale(
        self,
        time_series: pd.Series,
        metric: str = "metric",
        split_index: Optional[int] = None,
    ) -> MultiscaleDriftResult:
        """Detect drift across all temporal scales via wavelet decomposition.

        For each frequency component produced by :meth:`decompose_temporal_patterns`,
        a KS two-sample test is applied (reference = first part, current =
        second part of the component; the boundary is *split_index* when
        given, otherwise the midpoint).  The overall drift signal is the
        maximum drift score across scales.

        In the wavelet path the raw series is included as the ``full_signal``
        scale and every per-scale p-value is calibrated by max-T permutation
        (see the class docstring): the analytic KS p-value is invalid on the
        autocorrelated wavelet reconstructions and would flag ~75% of
        stationary noise as drift.  In the raw path (no ``pywt`` or short
        series) the analytic p-value is valid and is used directly.

        Parameters
        ----------
        time_series : pd.Series
            Full metric time series (ideally at least 60+ points for
            meaningful wavelet decomposition).
        metric : str
            Metric name for labelling.
        split_index : int, optional
            Index separating the reference window from the current window
            (set by :meth:`check_drift` to the baseline length).

        Returns
        -------
        MultiscaleDriftResult
            ``overall_drift_score`` NaN and ``drift_detected`` None is COULD NOT
            CHECK and must not be read as stability. It is returned when no
            scale could be computed, when no scale produced a usable score, and
            when either window holds fewer than
            ``_MIN_ARM_OBSERVATIONS`` genuinely observed readings (gaps are
            forward-filled before any test runs, so a window of copies would
            otherwise be graded as though every copy were an observation).
            A ``UserWarning`` names the reason in each case.
        """
        # G-20, 2026-09-17. Everything below runs on the FORWARD-FILLED grid
        # (see decompose_temporal_patterns), so a window arrives here holding
        # one observation and reaches the KS test holding sixty samples. The
        # count of genuine readings on each side of the boundary is knowable
        # only here, before the fill. Measured before this guard::
        #
        #   1 reading (0.10) + 59 NaN, midpoint split
        #       -> overall_drift_score 0.0, drift_detected False,
        #          worst scale reference_n 30, current_n 30
        #   2 readings (0.10 then 0.90) + 58 NaN
        #       -> overall_drift_score 0.9950, drift_detected True,
        #          reference_n 30, current_n 30, means 0.1000 vs 0.9000
        #   baseline 60 real vs current [0.90, NaN x 59] via check_drift
        #       -> overall_drift_score 0.9950, drift_detected True, current_n 60
        #
        # The first is the neutral default this audit hunts: "no drift" over a
        # series with one observation, and a report generator turns that into
        # "continue routine monitoring". The second and third are the same
        # fabrication pointing the other way: a maximally confident drift
        # verdict, and a sample count of sixty, from a single reading.
        raw_series = pd.Series(time_series)
        kept = raw_series.ffill().notna().to_numpy()
        observed = raw_series.notna().to_numpy()[kept]
        n_grid = int(observed.size)
        n_lead_dropped = int(len(kept) - int(kept.sum()))
        boundary: Optional[int]
        if split_index is None:
            boundary = n_grid // 2
        elif 0 < int(split_index) < n_grid:
            boundary = int(split_index)
        else:
            # Out of range. _resolve_split owns that case further down: it
            # warns and routes it into the same refusal, so do not say it twice.
            boundary = None

        if boundary is not None:
            if n_lead_dropped and split_index is not None:
                warnings.warn(
                    f"detect_drift_multiscale: metric {metric!r} was given a "
                    f"reference/current boundary of {int(split_index)}, but the "
                    f"{n_lead_dropped} leading reading(s) of the series are missing and "
                    f"are dropped before any test runs, so the boundary lands "
                    f"{n_lead_dropped} point(s) further into the series than the "
                    f"positions you passed. The two windows reported below are shifted "
                    f"by that much. Pass an index into the series AFTER the leading gap, "
                    f"or drop the gap yourself.",
                    UserWarning,
                    stacklevel=2,
                )
            ref_obs = int(observed[:boundary].sum())
            cur_obs = int(observed[boundary:].sum())
            if min(ref_obs, cur_obs) < _MIN_ARM_OBSERVATIONS:
                warnings.warn(
                    f"detect_drift_multiscale: metric {metric!r} has {ref_obs} observed "
                    f"reading(s) before the boundary and {cur_obs} after it, over a grid "
                    f"of {boundary} and {n_grid - boundary} point(s); the rest are "
                    f"forward-filled copies, not observations. A two-sample comparison "
                    f"needs at least {_MIN_ARM_OBSERVATIONS} real readings on each side, "
                    f"so no drift test was run and no verdict is derived from the filled "
                    f"copies. overall_drift_score is NaN and drift_detected is None: "
                    f"this is COULD NOT CHECK, not stability and not drift.",
                    UserWarning,
                    stacklevel=2,
                )
                return self._record(
                    MultiscaleDriftResult(
                        metric=metric,
                        timestamp=datetime.now(),
                        scales={},
                        overall_drift_score=float("nan"),
                        drift_detected=None,
                        decomposition_available=False,
                    )
                )
            if ref_obs < boundary or cur_obs < n_grid - boundary:
                # Enough real readings to measure, so measure: refusing here
                # would throw away the evidence that IS present. Say out loud
                # that the sample counts reported below are grid points rather
                # than observations, because nothing in the result distinguishes
                # them.
                warnings.warn(
                    f"detect_drift_multiscale: metric {metric!r} is missing "
                    f"{boundary - ref_obs} of {boundary} reading(s) before the boundary "
                    f"and {(n_grid - boundary) - cur_obs} of {n_grid - boundary} after "
                    f"it. Those gaps are forward-filled, so the reference_n and "
                    f"current_n reported below count grid points, NOT observations, and "
                    f"the KS statistic treats each filled copy as an independent sample.",
                    UserWarning,
                    stacklevel=2,
                )

        components = self.decompose_temporal_patterns(time_series)
        decomposed = "full_signal" not in components
        if decomposed:
            # Include the raw series alongside the wavelet components: its
            # KS statistic anchors the permutation family and is the most
            # powerful member against plain distribution shifts.
            values = time_series.ffill().dropna().values.astype(float)
            components = {"full_signal": values, **components}

        scale_results: Dict[str, DriftResult] = {}

        if decomposed:
            scale_results = self._permutation_calibrated_scales(
                components,
                metric=metric,
                split_index=split_index,
            )
        else:
            for scale_name, component in components.items():
                if len(component) < 20:
                    continue
                series_component = pd.Series(component)
                scale_result = self.detect_drift_ks(
                    series_component,
                    metric=metric,
                    scale=scale_name,
                    split_index=split_index,
                )
                scale_results[scale_name] = scale_result

        if not scale_results:
            # H-11. This returned overall_drift_score=0.0 and
            # drift_detected=False when NOT ONE scale could be computed, which is
            # the same answer as "we looked at every scale and the metric is
            # stable". Reproduced 2026-09-07 with a two-point series: scales={},
            # score 0.0, drift_detected False, and the report generator turns
            # that into "No significant drift detected. Continue routine
            # monitoring." `drift_report_to_svg` already handles this correctly
            # and renders COULD NOT CHECK; every non-SVG consumer did not.
            warnings.warn(
                f"No scale could be computed for metric {metric!r} (the series is "
                "too short to decompose), so no drift test was run. "
                "overall_drift_score is NaN and drift_detected is None: this is "
                "COULD NOT CHECK, not stability.",
                UserWarning,
                stacklevel=2,
            )
            return self._record(
                MultiscaleDriftResult(
                    metric=metric,
                    timestamp=datetime.now(),
                    scales={},
                    overall_drift_score=float("nan"),
                    drift_detected=None,
                    decomposition_available=_HAS_PYWT and len(components) > 1,
                )
            )

        valid_scores = [
            r.drift_score
            for r in scale_results.values()
            if r.drift_score is not None and not np.isnan(r.drift_score)
        ]
        # Scales exist but none produced a usable score: same reasoning as above.
        # `max(...) if valid_scores else 0.0` put a computed-looking 0.0 on a
        # comparison that never ran.
        if valid_scores:
            overall_score = max(valid_scores)
            any_drift: Optional[bool] = any(r.drift_detected for r in scale_results.values())
        else:
            warnings.warn(
                f"No scale produced a usable drift score for metric {metric!r}; "
                "overall_drift_score is NaN and drift_detected is None.",
                UserWarning,
                stacklevel=2,
            )
            overall_score = float("nan")
            any_drift = None

        # Optionally compute MMD if a baseline is stored
        mmd_score: Optional[float] = None
        if self.compute_mmd and self._baseline is not None:
            try:
                ref_vals = self._baseline.dropna().values.astype(float)
                cur_vals = time_series.dropna().values.astype(float)
                # When check_drift() passed the concatenated series, the
                # current window is everything AFTER the baseline boundary.
                # Comparing baseline vs baseline+current dilutes the MMD
                # toward zero because most "current" points ARE baseline.
                #
                # The slice is now conditional on the boundary being REAL. It
                # used to be skipped for a boundary of 0 or of len(cur_vals),
                # leaving cur_vals holding the baseline itself: measured on an
                # all-NaN current window, mmd_gaussian(baseline, baseline) gave
                # mmd_score 0.0, the strongest available statement that two
                # distributions are identical, over a window with zero
                # observations. mmd_score stays None (not computed) instead.
                sliceable = _resolve_split(
                    split_index, len(cur_vals), metric=metric, where="detect_drift_multiscale[mmd]"
                )
                if split_index is not None and sliceable is None:
                    cur_vals = np.empty(0, dtype=float)
                elif split_index is not None:
                    cur_vals = cur_vals[sliceable:]
                if len(ref_vals) >= 5 and len(cur_vals) >= 5:
                    mmd_score = mmd_gaussian(ref_vals, cur_vals)
            except Exception:
                # mmd_score stays None, which is the right could-not-check
                # answer, but discarding the exception leaves a reader with no
                # way to find out WHY it is None. The house pattern is to
                # return the honest value AND log the cause.
                logging.getLogger(__name__).debug(
                    "MMD diagnostic failed for metric %r; mmd_score stays None "
                    "(could not check), and no drift verdict is derived from it",
                    metric,
                    exc_info=True,
                )

        result = MultiscaleDriftResult(
            metric=metric,
            timestamp=datetime.now(),
            scales=scale_results,
            overall_drift_score=overall_score,
            drift_detected=any_drift,
            mmd_score=mmd_score,
            decomposition_available=_HAS_PYWT and len(components) > 1,
        )
        return self._record(result)

    # MMD drift detection

    def detect_drift_mmd(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        sigma: Optional[float] = None,
        threshold: Optional[float] = None,
        n_permutations: int = 200,
        alpha: float = 0.05,
        random_state: int = 0,
    ) -> Tuple[Optional[bool], float]:
        """MMD-based drift detection between two raw arrays.

        Computes the UNBIASED MMD² estimate (Gretton et al. 2012, Lemma 6,
        which excludes the k(x_i, x_i) diagonal terms) with a Gaussian kernel,
        and calibrates the verdict by a permutation test on the pooled sample
        (Gretton et al. 2012, sec. 8).  Both choices are scale-aware fixes for
        the pre-audit defaults, which were wrong in both directions:

        - The biased V-statistic includes the diagonal, flooring the score at
          ~2/n for well-separated data, so two samples from the SAME wide
          distribution (e.g. U(0, 100), n = 20) exceeded the fixed 0.05
          threshold and declared drift.
        - The fixed bandwidth sigma = 1.0 saturates on fairness-metric data
          (values ~0.1, distances << sigma), so a genuine 0.2 mean shift
          stayed under the fixed threshold and was missed.

        Parameters
        ----------
        reference : np.ndarray
            Samples from the reference distribution (needs >= 2 samples).
        current : np.ndarray
            Samples from the current distribution (needs >= 2 samples).
        sigma : float, optional
            Gaussian kernel bandwidth.  Default ``None`` selects the median
            heuristic (median pairwise distance of the pooled sample), which
            adapts the kernel to the data scale.  The pre-audit default was a
            fixed 1.0.  Must be finite and strictly positive when given: a
            non-positive or non-finite bandwidth raises ``ValueError``.  It
            used to reach the kernel and raise a bare ``ZeroDivisionError``
            from inside it (sigma=0.0), or silently invert the exponent so the
            kernel GREW with distance (sigma<0).
        threshold : float, optional
            Fixed decision threshold on the unbiased MMD² (legacy escape
            hatch).  Default ``None`` calibrates the threshold by permutation
            instead: drift is declared when the permutation p-value of the
            observed statistic falls below ``alpha``.  The pre-audit default
            was a fixed 0.05, which is exceeded by the diagonal bias alone at
            small n and never reached under a saturated kernel.
        n_permutations : int, default 200
            Number of pooled-sample permutations for the calibrated threshold.
            The test's smallest possible p-value is ``1/(n_permutations+1)``,
            so it must satisfy ``1/(n_permutations+1) < alpha`` for drift to be
            reportable at all; below that the call REFUSES (see Returns)
            instead of answering "no drift" it cannot contradict.  At the
            default alpha=0.05 that means at least 20 permutations.
        alpha : float, default 0.05
            Significance level for the permutation test.
        random_state : int, default 0
            Seed for the permutation test, making the verdict reproducible.

        Returns
        -------
        (drift_detected, mmd_score) : tuple[bool or None, float]
            ``drift_detected`` is ``True`` when the permutation test rejects
            the null, ``False`` when it does not, and ``None`` when NO test
            could be run at all.  ``mmd_score`` is the observed unbiased MMD²
            (can be slightly negative under the null), and ``nan`` whenever
            ``drift_detected`` is ``None``.

            ``None`` is COULD NOT CHECK and must not be read as "no drift".
            It is returned when either side holds fewer than 2 finite samples,
            when the statistic itself comes back non-finite, or when
            ``n_permutations`` is too small for the permutation test to reach
            ``alpha`` at all (its floor ``1/(n_permutations+1)`` is not below
            ``alpha``).  A ``UserWarning`` naming the reason accompanies each.
            The ``threshold`` escape hatch runs no permutations and is not
            subject to the last of those.

        Notes
        -----
        BGL S2, 2026-09-16. The refusal was a bare ``False``, i.e. "no drift",
        for a comparison that never happened. Measured::

            empty vs empty      -> (False, nan)   0 warnings
            1 vs 1              -> (False, nan)   0 warnings
            0 vs 60             -> (False, nan)   0 warnings
            1 vs 60, huge shift -> (False, nan)   0 warnings
            30 vs 30 shifted    -> (True, 0.9556) 0 warnings   <- healthy

        and the companion ``nan`` did NOT mark the refusal: a NaN-contaminated
        30-vs-30 call DID run the permutation test and returned ``(True, nan)``,
        because every ``nan >= nan`` comparison is False, so ``exceed`` stayed
        0 and ``p_value`` came out at 1/(n_permutations+1). A confident drift
        verdict derived from a statistic that is not a number. Both shapes are
        now ``None``, matching :meth:`detect_drift_multiscale` in this file and
        ``TemporalFairnessAnalyzer.detect_weekly_degradation`` in ``tracker``.
        """
        # Bandwidth first, because it is a CALLER ERROR rather than an
        # unmeasurable batch and must stay loud. sigma=0.0 used to reach
        # ``gamma = 1.0 / (2.0 * sigma * sigma)`` and raise a bare
        # ZeroDivisionError from the middle of the kernel with nothing said
        # about which argument was wrong; a negative sigma is worse, because
        # it flips the exponent's sign and the kernel grows with distance,
        # producing a finite statistic nobody can interpret.
        if sigma is not None and not (np.isfinite(sigma) and sigma > 0):
            raise ValueError(
                f"sigma must be a finite positive kernel bandwidth, got {sigma!r}. "
                f"Pass sigma=None to select it by the median heuristic."
            )
        # Same treatment and same reason as sigma. G-20, measured: alpha=0.0
        # reached the n_permutations refusal below and raised ZeroDivisionError
        # from inside the warning message that was explaining the refusal, so a
        # could-not-check came back as a crash from the middle of a formatter.
        # alpha >= 1 is the other end: `p_value < alpha` is then true for every
        # possible permutation outcome, so the call reports drift whatever the
        # data shows.
        if not (np.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError(
                f"alpha must be a significance level strictly between 0 and 1, got "
                f"{alpha!r}. At alpha <= 0 no permutation test can ever reject, and at "
                f"alpha >= 1 it rejects whatever the data shows."
            )

        ref_raw = np.asarray(reference, dtype=float).ravel()
        cur_raw = np.asarray(current, dtype=float).ravel()
        # Non-finite readings are absent observations, not values. Keeping them
        # poisons the kernel and every permutation comparison alike, which is
        # how a contaminated sample produced a confident True above.
        ref = ref_raw[np.isfinite(ref_raw)]
        cur = cur_raw[np.isfinite(cur_raw)]
        n_dropped = (len(ref_raw) - len(ref)) + (len(cur_raw) - len(cur))
        n, m = len(ref), len(cur)
        if n < 2 or m < 2:
            # Cannot compute the unbiased statistic, so make NO drift claim.
            warnings.warn(
                f"detect_drift_mmd: the unbiased MMD estimate needs at least 2 finite "
                f"samples on each side and it has reference n={n}, current n={m} "
                f"(of {len(ref_raw)} and {len(cur_raw)} supplied). No permutation test "
                f"was run, so drift_detected is None: COULD NOT CHECK, which is not "
                f"the same answer as 'no drift'.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")
        if n_dropped:
            warnings.warn(
                f"detect_drift_mmd: {n_dropped} non-finite reading(s) were excluded, so "
                f"the test compares reference n={n} against current n={m} rather than "
                f"the {len(ref_raw)} and {len(cur_raw)} supplied.",
                UserWarning,
                stacklevel=2,
            )

        pooled = np.concatenate([ref, cur])
        sq_dist = (pooled[:, None] - pooled[None, :]) ** 2

        if sigma is None:
            # Median heuristic on the pooled pairwise distances.
            iu = np.triu_indices(n + m, k=1)
            pair_dist = np.sqrt(sq_dist[iu])
            med = float(np.median(pair_dist))
            if med <= 0:
                pos = pair_dist[pair_dist > 0]
                med = float(np.median(pos)) if len(pos) else 1.0
            sigma = med

        gamma = 1.0 / (2.0 * sigma * sigma)
        kernel = np.exp(-gamma * sq_dist)

        def _mmd2_unbiased(idx: np.ndarray) -> float:
            idx_x, idx_y = idx[:n], idx[n:]
            k_xx = kernel[np.ix_(idx_x, idx_x)]
            k_yy = kernel[np.ix_(idx_y, idx_y)]
            k_xy = kernel[np.ix_(idx_x, idx_y)]
            term_x = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
            term_y = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
            return float(term_x + term_y - 2.0 * k_xy.mean())

        score = _mmd2_unbiased(np.arange(n + m))

        # ABOVE the branch selection, not inside one of them. Both the fixed
        # threshold arm and the permutation arm compare with operators that are
        # False for NaN, so a non-finite statistic silently became a verdict in
        # either: `nan > threshold` is a confident False, and the permutation
        # loop scores `nan >= nan` as False every time, which is a confident
        # True. A statistic that is not a number supports neither answer.
        if not np.isfinite(score):
            warnings.warn(
                f"detect_drift_mmd: the unbiased MMD statistic came back {score!r} on "
                f"reference n={n} vs current n={m}, so no verdict can be derived from "
                f"it. drift_detected is None: COULD NOT CHECK, not 'no drift' and not "
                f"drift.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")

        if threshold is not None:
            return bool(score > threshold), score

        # BGL-S2b, 2026-09-17. The permutation test's SMALLEST achievable
        # p-value is 1/(n_permutations+1), reached when not one permutation
        # matches or beats the observed statistic. When that floor is not
        # strictly below alpha, "drift" is unreachable BY CONSTRUCTION: the
        # False below is then a property of n_permutations and says nothing
        # about the data. Measured on a 50-sigma shift, 40 vs 40 samples,
        # MMD2 = 0.8619: n_permutations = 0 (no permutation is run at all), 5,
        # 18 and 19 each returned (False, 0.8619) with ZERO warnings, while
        # n_permutations=200 returned (True, 0.8619) on the same data.
        #
        # The comparison is >=, not >, because the verdict uses a STRICT
        # p < alpha: at the default alpha=0.05, n_permutations=19 puts the
        # floor at exactly 0.05 and True is still unreachable.
        n_perm = int(n_permutations)
        min_p = 1.0 / (n_perm + 1.0) if n_perm >= 0 else float("inf")
        if n_perm < 1 or min_p >= alpha:
            warnings.warn(
                f"detect_drift_mmd: with n_permutations={n_perm} the smallest "
                f"p-value the permutation test can produce is {min_p:.4g}, which is "
                f"not below alpha={alpha}, so this test cannot report drift whatever "
                f"the data shows. No verdict is being derived from it: "
                f"drift_detected is None (COULD NOT CHECK), not 'no drift'. Raise "
                f"n_permutations above {int(np.ceil(1.0 / alpha)) - 1} for this alpha, "
                f"or pass an explicit threshold.",
                UserWarning,
                stacklevel=2,
            )
            return None, float("nan")

        rng = np.random.default_rng(random_state)
        exceed = 0
        for _ in range(n_perm):
            if _mmd2_unbiased(rng.permutation(n + m)) >= score:
                exceed += 1
        p_value = (1.0 + exceed) / (n_perm + 1.0)
        return bool(p_value < alpha), score

    # Sequential Probability Ratio Test (SPRT)

    def run_sprt(
        self,
        metric_stream: List[float],
        null_value: float,
        alternative_value: float,
        alpha: float = 0.05,
        beta: float = 0.20,
    ) -> Tuple[str, int, float]:
        """Sequential Probability Ratio Test for online drift detection.

        SPRT (Wald, 1945; Page, 1954) allows drift detection without a fixed
        sample size.  It accumulates a log-likelihood ratio and stops as soon
        as the evidence is strong enough to accept or reject the null hypothesis.

        Parameters
        ----------
        metric_stream : list[float]
            Ordered stream of metric values (e.g., daily demographic parity).
        null_value : float
            Expected value under the null (stable) hypothesis.
        alternative_value : float
            Expected value under the alternative (drift) hypothesis.
        alpha : float, default 0.05
            Type I error rate (false positive).
        beta : float, default 0.20
            Type II error rate (false negative).

        Returns
        -------
        (decision, stopping_n, log_lambda) : tuple[str, int, float]
            *decision* is ``"drift"`` (reject H₀), ``"stable"`` (accept H₀),
            ``"continue"`` (the test ran and the evidence has not yet reached
            either boundary), or ``"could_not_check"`` (NO test ran).
            *stopping_n* is the index at which the decision was made, or the
            number of USABLE observations if no decision.
            *log_lambda* is the final log-likelihood ratio, and ``nan``
            whenever the decision is ``"could_not_check"``.

            ``"could_not_check"`` is not ``"stable"`` and not ``"continue"``:
            no amount of further data changes it. It is returned when the two
            hypotheses are not distinguishable numbers (identical, or not
            finite), when fewer than three usable observations were supplied,
            and when every observation is the identical value. Each carries a
            ``UserWarning`` naming the reason.

        Notes
        -----
        G-20, 2026-09-17. Two shapes reached the caller as ``"continue"``, the
        answer that means "keep collecting, the evidence is still accumulating",
        for streams on which no evidence can ever accumulate::

            null == alternative        -> ('continue', 10, 0.0)   0 warnings
            one NaN in a 40-point stream -> ('continue', 40, nan) 0 warnings

        The first invented a neutral log-likelihood ratio of 0.0 for a test that
        cannot be specified: two identical hypotheses are not a comparison, and
        a caller looping "while decision == 'continue'" waits forever for a
        boundary crossing that no data can produce. The second is the NaN
        verdict this module fixed in :meth:`detect_drift_mmd`, running one layer
        up: ``np.std`` of a stream holding a NaN is NaN, every increment is then
        NaN, and ``nan >= upper`` is False at every step, so the loop ran to the
        end and reported "no decision yet". Non-finite readings are now excluded
        (and named) so the remaining observations are still measured, which is
        what ``detect_drift_mmd`` does with the same input.
        """
        if not (np.isfinite(null_value) and np.isfinite(alternative_value)):
            warnings.warn(
                f"run_sprt: the hypotheses must be two finite numbers and they are "
                f"null_value={null_value!r}, alternative_value={alternative_value!r}, so "
                f"no likelihood ratio is defined and no test was run. Returning "
                f"'could_not_check', NOT 'stable' and NOT 'continue'.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(metric_stream), float("nan")
        if abs(alternative_value - null_value) < 1e-10:
            warnings.warn(
                f"run_sprt: the null ({null_value:.6g}) and the alternative "
                f"({alternative_value:.6g}) are the same value, so there are not two "
                f"hypotheses to weigh and no evidence can accumulate for either. "
                f"Returning 'could_not_check', NOT 'continue': more data will not "
                f"resolve this, only a different alternative_value will.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(metric_stream), float("nan")

        # BGL3 operations-4, 2026-09-27. G-20 (above) validated the two
        # HYPOTHESES and the stream, and left the two ERROR RATES unchecked,
        # although they are the other half of the test's specification: they are
        # the only inputs to both Wald boundaries. Measured on HEAD, one stream
        # of 40 readings centred on the null (which the defaults decide as
        # 'stable' at observation 1 with log_lambda -118.2):
        #
        #   alpha=0.5, beta=0.8  -> ('stable',   1, -118.23)  0 warnings
        #   alpha=-0.05          -> ('stable',   1, -118.23)  numpy RuntimeWarning only
        #   beta=1.0             -> ('drift',    1, -118.23)  numpy RuntimeWarning only
        #   beta=0.0             -> ('continue', 40, -5966.17)
        #   alpha=nan            -> ('continue', 40, -5966.17)
        #   alpha=0.0 / alpha=1.0-> ZeroDivisionError
        #
        # Every one of those is a verdict from a test that cannot be specified.
        # beta=1.0 is the clearest: upper becomes log(0) = -inf, so the FIRST
        # observation "crosses" it and the method returns 'drift' carrying a
        # NEGATIVE log-likelihood ratio, which is evidence for the null. And
        # alpha=nan makes both comparisons False at every step and reports
        # 'continue', the answer that means "keep collecting", for a test no
        # amount of data can resolve: verbatim the shape G-20 fixed one
        # parameter over.
        #
        # The guard is the maths, not a taste: a Wald test needs an acceptance
        # region, i.e. upper > 0 > lower. That holds exactly when both rates are
        # finite, both lie strictly inside (0, 1), and alpha + beta < 1. The
        # range test comes first because 1-alpha == 0 and alpha == 0 raise
        # ZeroDivisionError before any boundary can be computed.
        if not (
            np.isfinite(alpha) and np.isfinite(beta) and 0.0 < alpha < 1.0 and 0.0 < beta < 1.0
        ):
            warnings.warn(
                f"run_sprt: alpha and beta are error RATES and must each be a finite "
                f"number strictly between 0 and 1; they are alpha={alpha!r}, "
                f"beta={beta!r}, so neither Wald boundary is defined and no test was "
                f"run. Returning 'could_not_check', NOT 'stable', NOT 'drift' and NOT "
                f"'continue'.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(metric_stream), float("nan")

        # SPRT boundaries (Wald approximation)
        upper = np.log((1 - beta) / alpha)  # Reject H0 (drift)
        lower = np.log(beta / (1 - alpha))  # Accept H0 (stable)

        if not (upper > 0.0 > lower):
            # alpha + beta >= 1: (1-beta)/alpha <= 1 and beta/(1-alpha) >= 1, so
            # the two boundaries meet or cross and there is no region in which
            # the evidence is still accumulating. The first observation lands
            # outside whichever one it is compared against, whatever the data
            # says, which is a decision taken by the parameters alone.
            warnings.warn(
                f"run_sprt: alpha={alpha:.6g} and beta={beta:.6g} sum to "
                f"{alpha + beta:.6g}, which leaves the Wald boundaries "
                f"(upper={float(upper):.6g}, lower={float(lower):.6g}) with no "
                f"acceptance region between them, so the first observation decides "
                f"whatever the stream contains. No test was run. Returning "
                f"'could_not_check': choose error rates with alpha + beta < 1.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(metric_stream), float("nan")

        # Assume Gaussian observations with known std estimated from data.
        #
        # READINESS-6, 2026-09-10. Two fabrications sat on these two lines, and
        # both MANUFACTURE VERDICTS rather than suppress them.
        #
        # `else 0.1` invented a standard deviation for a stream too short to
        # estimate one from. 0.1 is not a neutral placeholder: it sets the scale
        # every log-likelihood increment below is measured in, so the decision
        # this method returns was calibrated against a number nobody observed.
        #
        # `max(sigma_est, 1e-6)` reads as a divide-by-zero guard and is one, but
        # what it guards is the crash. A CONSTANT metric stream has no variance,
        # and clamping to 1e-6 turns each observation's z-score into ~1e6, so the
        # Wald boundary is crossed on the FIRST point. Measured before this
        # change, on a constant stream of 40 identical values:
        #
        #   offset 0.2    -> ('drift',  1 obs, log_lambda = +2.0e10)
        #   offset 0.01   -> ('stable', 1 obs, log_lambda = -1.8e10)
        #   offset 0.001  -> ('stable', 1 obs, log_lambda = -2.0e10)
        #
        # Both directions are manufactured, and the 'stable' one is the worse of
        # the two: an all-clear on a monitored metric, declared from a single
        # observation, with a likelihood ratio of twenty billion behind it.
        #
        # A constant stream is the ordinary shape for a metric that has not
        # moved, which is exactly when someone is relying on this to say so.
        #
        # `np.ptp` rather than `std(...) == 0`: see the note in
        # evaluation.vfairness_metrics._statistics.sequential_fairness_test.
        stream_array = np.asarray(metric_stream, dtype=float)
        finite = np.isfinite(stream_array)
        n_excluded = int(stream_array.size - int(finite.sum()))
        if n_excluded:
            # Excluded rather than refused: 39 readable observations out of 40
            # are still a measurable stream, and this is what detect_drift_mmd
            # does with the same contamination. Keeping them made sigma_est NaN
            # and every boundary comparison False, which is how a destroyed test
            # reported 'continue'.
            warnings.warn(
                f"run_sprt: {n_excluded} of the {stream_array.size} reading(s) supplied "
                f"are not finite numbers and were excluded, so the test runs over "
                f"{int(finite.sum())} observation(s) and the stopping index counts those, "
                f"not the positions in the stream you passed.",
                UserWarning,
                stacklevel=2,
            )
            stream_array = stream_array[finite]
        if len(stream_array) <= 2:
            warnings.warn(
                f"run_sprt: {len(stream_array)} observation(s) is too few to estimate the "
                f"standard deviation the test measures evidence in, so no decision was "
                f"taken. Returning 'could_not_check', NOT 'stable': nothing was tested.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(stream_array), float("nan")
        # BGL5 AUDIT, 2026-09-27. This test was `float(np.ptp(...)) == 0.0`, and
        # ONE BIT of float jitter, which is what a real averaged metric stream
        # carries, walked straight past it into the fabrication the block comment
        # above says it removed. Measured on HEAD:
        #
        #   [0.9]*40                  null=0.9 alt=0.2 -> ('could_not_check', 40, nan)
        #   [0.9]*39 + [0.9 + 1e-15]  null=0.9 alt=0.2 -> ('stable', 1, -9.8e+30)
        #   [0.9]*39 + [0.9 + 1e-15]  null=0.1 alt=0.2 -> ('drift',  1, +3.0e+30)
        #
        # A range of 1e-15 makes sigma_est ~1.6e-16, so every z-score is ~1e15 and
        # the FIRST observation crosses a Wald boundary with a likelihood ratio of
        # 1e31 behind it: "an all-clear on a monitored metric, declared from a
        # single observation", in this method's own words, and the drift direction
        # is equally manufactured. After this change both jittered streams return
        # ('could_not_check', 40, nan) with the range named, and the measurable
        # controls are unchanged to the last digit: a genuinely noisy stable
        # stream still returns ('stable', 1, -118.22721539073946) and a drifted
        # one ('drift', 1, +201.96...).
        #
        # The test is RELATIVE, because "no variation" only means anything against
        # the scale of the numbers: 1e-12 of the largest magnitude in play is
        # about four thousand float64 ulps near 1.0, i.e. rounding and
        # accumulation noise, while a fairness metric that has actually moved
        # moves in the third or fourth decimal. Exact equality on an accumulated
        # or averaged quantity is the trap; see
        # tests/test_bgl5_operations_4.py for the jitter case pinned both ways.
        stream_span = float(np.ptp(stream_array))
        scale = max(
            abs(float(np.max(stream_array))),
            abs(float(np.min(stream_array))),
            abs(float(null_value)),
            abs(float(alternative_value)),
        )
        if stream_span <= _CONSTANT_STREAM_REL_TOL * max(scale, 1.0):
            identical = stream_span == 0.0
            warnings.warn(
                "run_sprt: "
                + (
                    f"every one of the {len(stream_array)} observations is the identical "
                    f"value {float(stream_array[0]):.6g}"
                    if identical
                    else f"the {len(stream_array)} observations span only "
                    f"{stream_span:.3g} around {float(stream_array[0]):.6g}, which is "
                    f"float rounding noise at this scale and not variation"
                )
                + f", so there is no variation to measure evidence against and the "
                f"standardised log-likelihood ratio has no value. Returning "
                f"'could_not_check', NOT 'stable' and NOT 'drift'. The metric genuinely "
                f"has not moved WITHIN this window; whether it sits at the null or the "
                f"alternative is a question about the value "
                f"({float(stream_array[0]):.6g} against a null of {null_value:.6g}), "
                f"not about drift.",
                UserWarning,
                stacklevel=2,
            )
            return "could_not_check", len(stream_array), float("nan")
        sigma_est = float(np.std(stream_array))

        log_lambda = 0.0
        for n, x in enumerate(stream_array):
            # Log-likelihood ratio increment
            ll_null = -0.5 * ((x - null_value) / sigma_est) ** 2
            ll_alt = -0.5 * ((x - alternative_value) / sigma_est) ** 2
            log_lambda += float(ll_alt - ll_null)

            if log_lambda >= upper:
                return "drift", n + 1, log_lambda
            if log_lambda <= lower:
                return "stable", n + 1, log_lambda

        return "continue", len(stream_array), log_lambda

    # History access

    def _record(self, result: MultiscaleDriftResult) -> MultiscaleDriftResult:
        """Store *result* in the history and return it.

        G-20, 2026-09-17. Only the runs that PRODUCED a verdict used to reach
        the history: every refusal returned before the append, so
        :meth:`get_history_df` showed a monitoring record in which every check
        had succeeded, and :meth:`~vfairness.operations.reporting.store.MetricsStore.ingest_from_detector`
        carried that same survivorship into the store. A check that could not be
        made is a result about the monitoring, not a non-event, and the store
        already ingests the NaN score and the None verdict correctly.
        """
        self._results_history.append(result)
        return result

    def get_results_history(self) -> List[MultiscaleDriftResult]:
        """Return all stored multi-scale drift results, including the runs that
        COULD NOT be checked (``overall_drift_score`` NaN, ``drift_detected``
        None). Filtering those out is the caller's decision to make, not this
        method's."""
        return list(self._results_history)

    def get_history_df(self) -> pd.DataFrame:
        """Return drift history as a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``metric``, ``overall_drift_score``,
            ``drift_detected``, ``mmd_score``.  ``drift_detected`` is
            object-typed and holds ``None`` for every run that COULD NOT be
            checked, so ``df["drift_detected"].astype(bool)`` counts those as
            "no drift"; filter on ``.notna()`` first.

            The columns are present even when the history is empty. G-20: an
            empty history returned a DataFrame with NO columns at all, so a
            caller summarising "how many runs found drift" got a KeyError from
            the promised column rather than an empty answer.
        """
        columns = [
            "timestamp",
            "metric",
            "overall_drift_score",
            "drift_detected",
            "mmd_score",
        ]
        rows = []
        for r in self._results_history:
            rows.append(
                {
                    "timestamp": r.timestamp,
                    "metric": r.metric,
                    "overall_drift_score": r.overall_drift_score,
                    "drift_detected": r.drift_detected,
                    "mmd_score": r.mmd_score,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def clear_history(self) -> None:
        """Clear stored results history."""
        self._results_history = []

    # Explainability

    def get_explanation(self, result):
        """Generate educational explanations for a drift detection result.

        Parameters
        ----------
        result : MultiscaleDriftResult
            The result from :meth:`check_drift` or :meth:`detect_drift_multiscale`.

        Returns
        -------
        ExplanationReport
        """
        from vfairness.explainer import FairnessExplainer

        return FairnessExplainer.explain(result)

    # Private helpers

    def _permutation_calibrated_scales(
        self,
        components: Dict[str, np.ndarray],
        metric: str,
        split_index: Optional[int],
    ) -> Dict[str, DriftResult]:
        """Per-scale KS results with permutation-calibrated p-values.

        The analytic ``ks_2samp`` p-value assumes independent samples.  The
        wavelet reconstructions violate that badly (a smoothed approximation
        of iid noise has nearly disjoint halves), so instead of trusting the
        analytic p, the null distribution of each scale's KS statistic is
        built by shuffling the raw series (exchangeable under "no drift"),
        re-decomposing each shuffle identically, and recomputing the per-scale
        statistics.  Family-wise calibration uses single-step max-T
        (Westfall & Young 1993): each scale's studentized statistic is
        compared against the permutation distribution of the maximum
        studentized statistic across all scales, so the probability of ANY
        scale firing on a stationary series is ~``significance_level``.

        Returns the same ``{scale: DriftResult}`` mapping as the analytic
        path, with ``p_value`` holding the family-wise adjusted permutation
        p-value (minimum attainable: ``1 / (n_permutations + 1)``).
        """
        values = np.asarray(components["full_signal"], dtype=float)
        n = len(values)
        resolved = _resolve_split(
            split_index, n, metric=metric, where="_permutation_calibrated_scales"
        )
        if resolved is None:
            # One arm is empty, so there is no two-sample statistic to calibrate
            # at ANY scale. Same NaN shape as _nan_result, built here because
            # there is no split to slice the components with.
            return {
                scale_name: DriftResult(
                    metric=metric,
                    scale=scale_name,
                    ks_statistic=np.nan,
                    p_value=np.nan,
                    drift_score=np.nan,
                    drift_detected=False,
                    reference_mean=np.nan,
                    current_mean=np.nan,
                    reference_n=0,
                    current_n=0,
                )
                for scale_name in components
            }
        split = resolved

        def _nan_result(scale_name: str, comp: np.ndarray) -> DriftResult:
            ref, cur = comp[:split], comp[split:]
            return DriftResult(
                metric=metric,
                scale=scale_name,
                ks_statistic=np.nan,
                p_value=np.nan,
                drift_score=np.nan,
                drift_detected=False,
                reference_mean=float(np.mean(ref)) if len(ref) else np.nan,
                current_mean=float(np.mean(cur)) if len(cur) else np.nan,
                reference_n=len(ref),
                current_n=len(cur),
            )

        results: Dict[str, DriftResult] = {}
        names: List[str] = []
        obs_stats: List[float] = []
        for scale_name, comp in components.items():
            comp = np.asarray(comp, dtype=float)
            if len(comp[:split]) < 5 or len(comp[split:]) < 5:
                results[scale_name] = _nan_result(scale_name, comp)
                continue
            names.append(scale_name)
            obs_stats.append(float(stats.ks_2samp(comp[:split], comp[split:]).statistic))

        if not names:
            return results

        # Null distribution: shuffle, re-decompose, recompute per-scale KS.
        n_perm = max(int(self.n_permutations), 19)
        rng = np.random.default_rng(self.random_state)
        null_stats = np.full((n_perm, len(names)), np.nan)
        for b in range(n_perm):
            perm = rng.permutation(values)
            comps_b = self.decompose_temporal_patterns(pd.Series(perm))
            comps_b["full_signal"] = perm
            for j, name in enumerate(names):
                comp_b = comps_b.get(name)
                if comp_b is None or len(comp_b) != n:
                    # Decomposition of this shuffle failed (rare); leave NaN.
                    continue
                null_stats[b, j] = stats.ks_2samp(comp_b[:split], comp_b[split:]).statistic

        valid_rows = ~np.all(np.isnan(null_stats), axis=1)
        null_valid = null_stats[valid_rows]
        if len(null_valid) == 0:
            # No usable permutations: cannot calibrate, so make no
            # significance claim at any scale (fail closed, never analytic-p).
            for j, name in enumerate(names):
                comp = np.asarray(components[name], dtype=float)
                r = _nan_result(name, comp)
                r.ks_statistic = obs_stats[j]
                results[name] = r
            return results

        col_mean = np.nanmean(null_valid, axis=0)
        col_std = np.maximum(np.nanstd(null_valid, axis=0), 1e-12)
        z_obs = (np.asarray(obs_stats) - col_mean) / col_std
        z_null = (null_valid - col_mean) / col_std
        null_max = np.nanmax(z_null, axis=1)
        b_eff = len(null_max)

        for j, name in enumerate(names):
            comp = np.asarray(components[name], dtype=float)
            ref, cur = comp[:split], comp[split:]
            adj_p = float((1.0 + np.sum(null_max >= z_obs[j])) / (b_eff + 1.0))
            ks_stat = obs_stats[j]
            drift_score = float(ks_stat * (1.0 - adj_p))
            detected = adj_p < self.significance_level and drift_score >= self.min_drift_score
            results[name] = DriftResult(
                metric=metric,
                scale=name,
                ks_statistic=ks_stat,
                p_value=adj_p,
                drift_score=drift_score,
                drift_detected=detected,
                reference_mean=float(np.mean(ref)),
                current_mean=float(np.mean(cur)),
                reference_n=len(ref),
                current_n=len(cur),
            )
        return results

    def _run_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        metric: str,
        scale: str,
    ) -> DriftResult:
        """Run a two-sample KS test and wrap results in :class:`DriftResult`."""
        if len(reference) < 5 or len(current) < 5:
            # G-20. This was the last SILENT refusal on the KS path. A lopsided
            # boundary (split_index=2 over 60 points) reaches here with 20+
            # usable points, so the loud guard in detect_drift_ks does not fire,
            # and the caller got drift_detected False with real-looking
            # reference_mean, current_mean and mean_shift beside a NaN score and
            # not one warning. Measured: split_index=2 over a 0.10 -> 0.30
            # series returned mean_shift +0.108496 with drift_score NaN.
            warnings.warn(
                f"detect_drift_ks: '{metric}' [{scale}] splits into a reference window of "
                f"{len(reference)} point(s) and a current window of {len(current)}, and a "
                f"two-sample KS test needs at least {_MIN_ARM_OBSERVATIONS} on each side, "
                f"so no test was run. The NaN ks_statistic, p_value and drift_score are "
                f"the refusal; drift_detected is False only because the field cannot hold "
                f"a third state. This is COULD NOT CHECK, not 'no drift'. The means and "
                f"mean_shift reported alongside are real, and are the difference between "
                f"the two windows, not a tested one.",
                UserWarning,
                stacklevel=3,
            )
            return DriftResult(
                metric=metric,
                scale=scale,
                ks_statistic=np.nan,
                p_value=np.nan,
                drift_score=np.nan,
                drift_detected=False,
                reference_mean=float(np.mean(reference)) if len(reference) else np.nan,
                current_mean=float(np.mean(current)) if len(current) else np.nan,
                reference_n=len(reference),
                current_n=len(current),
            )

        ks_stat, p_value = stats.ks_2samp(reference, current)
        drift_score = float(ks_stat * (1.0 - p_value))
        drift_detected = p_value < self.significance_level and drift_score >= self.min_drift_score

        return DriftResult(
            metric=metric,
            scale=scale,
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            drift_score=drift_score,
            drift_detected=drift_detected,
            reference_mean=float(np.mean(reference)),
            current_mean=float(np.mean(current)),
            reference_n=len(reference),
            current_n=len(current),
        )
