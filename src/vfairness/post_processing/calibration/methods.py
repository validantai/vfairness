"""
Calibration Methods for vfairness.

This module provides a comprehensive suite of calibration techniques for
transforming raw model outputs into well-calibrated probabilities. Each
method can be applied globally or per-group for fairness-aware calibration.

Methods Implemented:
    1. Platt Scaling: Logistic regression-based calibration (Platt, 1999)
    2. Isotonic Regression: Non-parametric monotonic calibration (Zadrozny & Elkan, 2002)
    3. Beta Calibration: Parametric calibration with beta distributions (Kull et al., 2017)
    4. Temperature Scaling: Simple scalar calibration for neural networks (Guo et al., 2017)
    5. Histogram Binning: Discretization-based calibration (Zadrozny & Elkan, 2001)

Library Comparisons:
    scikit-learn: CalibratedClassifierCV (Platt, Isotonic)
    netcal: Comprehensive calibration library
    uncertainty-calibration: Beta calibration, temperature scaling

References:
    - Platt, J. (1999). Probabilistic Outputs for SVMs. Advances in Large Margin Classifiers.
    - Zadrozny, B. & Elkan, C. (2002). Transforming Classifier Scores. KDD.
    - Kull, M., et al. (2017). Beta Calibration. AISTATS.
    - Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML.
    - Naeini, M. P., et al. (2015). Obtaining Well Calibrated Probabilities. AAAI.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
    validate_probabilities,
)


class CalibrationMethod(Enum):
    """Enumeration of available calibration methods."""

    PLATT = "platt"
    ISOTONIC = "isotonic"
    BETA = "beta"
    TEMPERATURE = "temperature"
    HISTOGRAM = "histogram"


@dataclass
class CalibrationFitResult:
    """
    Container for calibration fitting results.

    Attributes:
        method: Calibration method used
        n_samples: Number of rows SUPPLIED to fit(). Read it as the size of the
            input, never as the evidence behind the parameters. It used to be
            documented as "samples used for fitting", and every calibrator here
            drops rows carrying ``sample_weight`` 0, so on 60 rows of which 3
            carried weight this field said 60 while 57 contributed nothing
            (BGL5, 2026-09-27). ``parameters["n_weighted_rows"]`` is the count
            that actually entered the fit, and the fit notes name the gap.
        parameters: Fitted parameters (method-specific). Every calibrator that
            screens zero-weight rows publishes ``n_weighted_rows`` here.
        fit_metrics: Metrics computed during fitting
        warnings: Any warnings generated during fitting
    """

    method: str
    n_samples: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    fit_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "parameters": self.parameters,
            "fit_metrics": self.fit_metrics,
            "warnings": self.warnings,
        }


class BaseCalibrator(ABC):
    """
    Abstract base class for calibration methods.

    All calibration methods should inherit from this class and implement
    the fit() and transform() methods.

    Attributes:
        is_fitted: Whether the calibrator has been fitted
        fit_result: Results from the fitting process
    """

    def __init__(self):
        self.is_fitted = False
        self.fit_result: Optional[CalibrationFitResult] = None

    @abstractmethod
    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "BaseCalibrator":
        """
        Fit the calibrator to training data.

        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities for the positive class
            sample_weight: Optional sample weights

        Returns:
            self: The fitted calibrator
        """
        pass

    @abstractmethod
    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Transform probabilities using the fitted calibrator.

        Args:
            y_prob: Predicted probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        pass

    def fit_transform(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> np.ndarray:
        """
        Fit the calibrator and transform in one step.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            Calibrated probabilities
        """
        self.fit(y_true, y_prob, sample_weight)
        return self.transform(y_prob)

    def _check_is_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() before transform()."
            )

    def _validate_fit_inputs(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], list]:
        """Validate and convert inputs for fitting.

        Returns ``(y_true, y_prob, sample_weight, fit_notes)``. ``fit_notes`` is
        the DURABLE half of a caution this gate used to state only on stderr:
        every caller must pass it into ``CalibrationFitResult(warnings=...)`` so
        the serialised record cannot read as a clean fit while the same call
        emitted a UserWarning saying the opposite. Measured 2026-09-16, before
        this: a single-class fit of any of the five calibrators recorded
        ``warnings: []`` and, for Platt, ``log_loss 0.0025``, the best-looking
        number in a per-group run, while the one stderr line was emitted once
        and gone by the time anything read the results.
        """
        y_true = coerce_to_array(y_true, "y_true")
        y_prob = coerce_to_array(y_prob, "y_prob")

        check_consistent_length(y_true, y_prob)
        validate_probabilities(y_prob, "y_prob")

        # A calibrator handed NO data is not a calibrator. Measured 2026-09-08,
        # before this guard: four of the five concrete classes accepted
        # `fit([], [])`, set is_fitted=True and emitted no warning, then
        # transformed [0.1, 0.5, 0.9] to [0.5, 0.5, 0.5] (Platt, Beta,
        # Temperature) or straight back to the bin centres [0.15, 0.55, 0.95]
        # (HistogramBinning), which reads to a caller as "calibration confirmed
        # these probabilities". Isotonic did refuse, but with a bare
        # "IndexError: index 0 is out of bounds for axis 0 with size 0" out of
        # numpy. This is the single gate all five already pass through, so the
        # refusal is stated once and stated the same way.
        if len(y_true) == 0:
            raise ValueError(
                f"{self.__class__.__name__}.fit() was given 0 samples. A calibrator "
                f"cannot be fitted on no data; it would report is_fitted=True and map "
                f"every probability to a constant. Pass training data, or handle the "
                f"empty case in the caller."
            )

        # Validate binary labels
        unique_labels = np.unique(y_true)
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"y_true must contain only 0 and 1, got: {unique_labels}")

        # A SINGLE-CLASS fit is unlearnable for the same reason, and it is
        # quieter: with all-zero labels the same five classes fit without one
        # warning and then answered 0.0192 or a flat 0.0 for every input. That
        # is not calibration, it is the base rate wearing a calibrated label.
        # It does not raise, because a legitimate per-group run can hit a group
        # with one observed class and should degrade rather than die, but it
        # must never pass in silence.
        fit_notes: list = []
        if len(unique_labels) < 2:
            single_class_note = (
                f"{self.__class__.__name__}.fit() was given {len(y_true)} samples of a "
                f"SINGLE class ({unique_labels.tolist()}). No probability-to-outcome "
                f"mapping can be learned from one class; the calibrator will return "
                f"essentially the base rate for every input. Treat its output as "
                f"unvalidated, not as calibrated."
            )
            fit_notes.append(single_class_note)
            warnings.warn(single_class_note, UserWarning, stacklevel=3)

        if sample_weight is not None:
            sample_weight = coerce_to_array(sample_weight, "sample_weight")
            check_consistent_length(y_true, sample_weight)
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative")
            # A NaN or inf weight is not a weight, and NaN < 0 is False, so the
            # check above waved it through. Each of the five then did something
            # different with it, measured 2026-09-17 with one NaN among 60
            # weights on informative scores: Platt refused (NaN, one Python
            # warning), Beta returned NaN with a note about max_iter that did
            # not mention the weight, Isotonic published a normal-looking record
            # (n_knots 60, warnings []) with one silently poisoned block,
            # Histogram blanked the affected bin, and TemperatureScaling
            # published a FINITE temperature of 38.2027 with nll 0.6805 off an
            # objective that is NaN everywhere: the bracket midpoint of a
            # bounded search that never found anything. Refuse it once, here,
            # the same way the negative check and the empty check are stated
            # once here.
            if not np.all(np.isfinite(sample_weight)):
                n_bad = int(np.count_nonzero(~np.isfinite(sample_weight)))
                raise ValueError(
                    f"{self.__class__.__name__}.fit() was given {n_bad} non-finite "
                    f"sample_weight value(s) (NaN or inf). A non-finite weight cannot "
                    f"weight an observation: it makes every weighted sum it touches "
                    f"non-finite, and the calibrators answer that differently, one of "
                    f"them with a finite-looking parameter fitted to nothing. Drop or "
                    f"impute those rows before calling fit()."
                )

        return y_true, y_prob, sample_weight, fit_notes


class PlattScaling(BaseCalibrator):
    """
    Platt Scaling calibration method.

    Fits a logistic regression model to map raw scores to calibrated
    probabilities. Originally developed for SVMs but widely applicable.

    The transformation is: P(y=1|f) = 1 / (1 + exp(A*f + B))
    where f is the raw score and A, B are learned parameters.

    Attributes:
        regularization: L2 regularization strength (default: 0.0)
        max_iter: Maximum iterations for optimization
        tol: Convergence tolerance

    Example:
        >>> calibrator = PlattScaling()
        >>> calibrator.fit(y_true, y_prob)
        >>> calibrated = calibrator.transform(y_prob_test)

    References:
        Platt, J. (1999). Probabilistic Outputs for Support Vector Machines.
        Advances in Large Margin Classifiers, 61-74.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: platt_scaling. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, regularization: float = 0.0, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None

    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "PlattScaling":
        """
        Fit Platt scaling using logistic regression.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities (or raw scores)
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator
        """
        y_true, y_prob, sample_weight, fit_notes = self._validate_fit_inputs(
            y_true, y_prob, sample_weight
        )

        n = len(y_true)
        if sample_weight is None:
            sample_weight = np.ones(n)

        # Transform probabilities to log-odds (logit) for fitting
        # Clip to avoid numerical issues
        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Initialize parameters using Platt's method
        # Target probabilities with regularization
        n_pos = np.sum(sample_weight[y_true == 1])
        n_neg = np.sum(sample_weight[y_true == 0])
        t_pos = (n_pos + 1) / (n_pos + 2)
        t_neg = 1 / (n_neg + 2)

        targets = np.where(y_true == 1, t_pos, t_neg)

        # Platt's smoothing targets are built from the WEIGHTED class counts, so
        # they collapse toward 0.5 for both classes when the weights are small in
        # absolute terms (not merely unnormalised). Measured: sample_weight=1e-8
        # over 60 informative scores gives n_pos=n_neg=3e-07, hence
        # t_pos=0.5000001 and t_neg=0.4999999, and the optimum of that objective
        # really is the flat map (a=9.6e-08, b=-1.6e-07). The value is right for
        # the objective as specified, but a reader seeing a flat 0.5 needs to
        # know it came from the prior and not from the scores, so say it here
        # rather than let the near-zero coefficients pass as a finding.
        if min(float(n_pos), float(n_neg)) < 1.0:
            fit_notes.append(
                f"the weighted class counts (n_pos={float(n_pos):.4g}, "
                f"n_neg={float(n_neg):.4g}) are below 1, so Platt's smoothing targets "
                f"collapse toward 0.5 (t_pos={t_pos:.6g}, t_neg={t_neg:.6g}) and the "
                f"fitted map is dominated by that prior rather than by the scores. "
                f"Pass weights on the scale of observation counts to recover it."
            )

        # Newton-Raphson optimization for logistic regression
        # Minimize: -sum(w * [t*log(p) + (1-t)*log(1-p)]) + reg * (A^2 + B^2)
        a, b = 0.0, 0.0

        # Three states, not two: a Newton loop that never took a step has
        # MEASURED NOTHING, and a and b are still their 0.0 initialisation.
        # `n_steps` counts steps actually taken, so the singular-Hessian break
        # below can be told apart from a converged fit that happens to sit near
        # a = b = 0. `converged` is only set by the tolerance test.
        n_steps = 0
        converged = False
        singular_hessian = False

        # BGL-D sibling (2026-09-17). IDENTIFIABILITY is a property of the
        # DESIGN, so it is tested here, once, BEFORE the loop, on a
        # SCALE-INVARIANT quantity. The determinant test inside the loop was
        # neither, and `abs(det) < 1e-12` therefore got the answer wrong in both
        # directions. Measured at HEAD of this working tree, 60 rows:
        #   * A CONSTANT score column carries no information at all, but at
        #     0.01 the determinant only fell under the absolute floor after TWO
        #     Newton steps had already moved the parameters, so fit() published
        #     a=0.33125, b=0.425 and transform([0.01, 0.5, 0.99]) ->
        #     [0.2503, 0.6047, 0.8751]: a finite, differentiated calibration map
        #     read off a column with one distinct value, warnings carrying only
        #     "became singular after 2 step(s)" and ZERO Python warnings. At the
        #     constant 1.0 it published a=-0.0613, an ANTI-MONOTONE map
        #     ([0.5525, 0.4823, 0.4128]: higher score, LOWER calibrated risk).
        #     The old fixtures (0.5, 0.7) are exactly the two values where the
        #     absolute floor happens to fire at iteration 0.
        #   * sample_weight=1e-8 over 60 INFORMATIVE scores scales the whole
        #     Hessian by 1e-8, so det ~ 1e-16 sits under the same absolute
        #     floor at the initialisation, and the fit was REFUSED with a note
        #     claiming "the 60 scores carry no usable variation". That is false:
        #     the identical scores at weight 1.0 fit normally. A refusal that
        #     fires on the UNIT of the weights refuses what it can measure.
        # Distinct logits are a monotone image of distinct scores, so this
        # counts the resolution the design actually has; rows with zero weight
        # contribute nothing to the Hessian and are excluded from the count.
        positive_weight = sample_weight > 0
        n_informative = int(np.count_nonzero(positive_weight))
        distinct_scores = int(np.unique(logits[positive_weight]).size) if n_informative else 0
        design_identified = distinct_scores >= 2
        # BGL5 (2026-09-27), the sibling of the TemperatureScaling.fit finding in
        # this file. A zero-weight row contributes nothing to the gradient or the
        # Hessian below, so it is dropped in all but name, and `n_samples`
        # counted it anyway. MEASURED on 60 informative scores with 3 of them
        # carrying weight 1.0: n_samples 60, parameters {'a': 0.2511688296545174,
        # 'b': 0.5329310202950369}, fit_metrics {'log_loss': 0.6409180253636495,
        # 'iterations': 5}, warnings [] and zero UserWarnings, with nothing
        # saying the map came from 3 observations. IsotonicCalibrator.fit and
        # BetaCalibrator.fit already publish n_weighted_rows; this publishes it
        # too and names the dropped rows. A fit at uniform weight is unchanged,
        # warnings included.
        if n_informative < n:
            fit_notes.append(
                f"{n - n_informative} of {n} rows carry sample_weight 0 and contribute "
                f"nothing to the gradient or the Hessian; the map below is fitted from "
                f"{n_informative} observation(s), not from {n}"
            )

        for _iteration in range(self.max_iter if design_identified else 0):
            scores = a * logits + b
            probs = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

            # Gradient
            diff = probs - targets
            grad_a = np.sum(sample_weight * diff * logits) + 2 * self.regularization * a
            grad_b = np.sum(sample_weight * diff) + 2 * self.regularization * b

            # Hessian
            hess_diag = sample_weight * probs * (1 - probs)
            hess_aa = np.sum(hess_diag * logits**2) + 2 * self.regularization
            hess_bb = np.sum(hess_diag) + 2 * self.regularization
            hess_ab = np.sum(hess_diag * logits)

            # Newton step (solve 2x2 system). The determinant carries the units
            # of (weight * logit^2) squared, so it is compared with the product
            # of the diagonal it came from rather than with an absolute floor:
            # a relative test says "this 2x2 system cannot be solved" for any
            # scaling of the weights, where `< 1e-12` said it for any weight
            # small enough. A zero det_scale (every weight zero) still refuses,
            # because 0 <= 0.
            det = hess_aa * hess_bb - hess_ab**2
            det_scale = abs(hess_aa * hess_bb)
            if not np.isfinite(det) or abs(det) <= 1e-12 * det_scale:
                singular_hessian = True
                break

            delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det
            delta_b = (hess_aa * grad_b - hess_ab * grad_a) / det

            a -= delta_a
            b -= delta_b
            n_steps += 1

            if abs(delta_a) < self.tol and abs(delta_b) < self.tol:
                converged = True
                break

        # BGL-D (2026-09-16). With a constant score column every logit is
        # identical, so hess_aa*hess_bb == hess_ab**2 exactly, the loop breaks on
        # iteration 0, and fit() used to publish its own INITIALISATION as the
        # estimate: parameters {'a': 0.0, 'b': 0.0}, fit_metrics
        # {'log_loss': 0.6931 (= ln 2, the coin-flip loss), 'iterations': 1},
        # warnings [], is_fitted True and not one Python warning, while
        # transform([0.01, 0.5, 0.99]) answered a flat [0.5, 0.5, 0.5]. The
        # identical output came back for a constant 0.8 column with a base rate
        # of 0.10, i.e. the value did not depend on the data at all. A design
        # that identifies no map must say so, not hand back a neutral number.
        #
        # Widened 2026-09-17 from `singular_hessian and n_steps == 0` to
        # `n_steps == 0`, which is the real precondition: NO STEP WAS TAKEN, so
        # a and b are still the 0.0 initialisation whatever stopped the loop.
        # Measured before this widening, PlattScaling(max_iter=0) on HEALTHY
        # data (200 informative scores) published parameters {'a': 0.0,
        # 'b': 0.0}, fit_metrics {'log_loss': 0.6931471805599452 (ln 2),
        # 'iterations': 1}, warnings carrying only "hit max_iter (0) ... the
        # reported a and b are the last iterate" (untrue: there is no iterate),
        # zero Python warnings, and transform([0.1, 0.5, 0.9]) -> [0.5, 0.5,
        # 0.5]. The reason is chosen per cause so the note cannot claim the
        # scores carry no variation when they do.
        if n_steps == 0:
            if not design_identified:
                unmeasurable_cause = (
                    f"the design is rank-deficient: across the {n_informative} of {n} rows "
                    f"with non-zero weight the scores take {distinct_scores} distinct "
                    f"value(s), so the slope is not identified and any value of a gives "
                    f"the same likelihood"
                )
            elif self.max_iter < 1:
                unmeasurable_cause = (
                    f"max_iter={self.max_iter} allowed no Newton step, so a and b are "
                    f"still their 0.0 initialisation and nothing was estimated"
                )
            else:
                unmeasurable_cause = (
                    "the Newton system was singular at the initialisation, so no step "
                    "was taken and a and b are still their 0.0 initialisation"
                )
            unmeasurable_note = (
                f"{self.__class__.__name__}.fit() identified NO Platt map: "
                f"{unmeasurable_cause}. Parameters are reported as NaN rather than as "
                f"their 0.0 starting values, and transform() returns NaN. This is a "
                f"could-not-check, not a calibration."
            )
            fit_notes.append(unmeasurable_note)
            warnings.warn(unmeasurable_note, UserWarning, stacklevel=2)
            a = float("nan")
            b = float("nan")
        elif singular_hessian:
            fit_notes.append(
                f"the Newton system became singular after {n_steps} step(s); the "
                f"reported a and b are the last iterate, not a converged optimum"
            )
        elif not converged:
            fit_notes.append(
                f"the Newton loop hit max_iter ({self.max_iter}) without reaching "
                f"tol={self.tol:g}; the reported a and b are the last iterate, not a "
                f"converged optimum"
            )

        self.a_ = a
        self.b_ = b
        self.is_fitted = True

        calibrated = self.transform(y_prob)
        with np.errstate(invalid="ignore"):
            log_loss = -np.mean(
                y_true * np.log(np.clip(calibrated, eps, 1))
                + (1 - y_true) * np.log(np.clip(1 - calibrated, eps, 1))
            )

        self.fit_result = CalibrationFitResult(
            method="platt_scaling",
            n_samples=n,
            # How many of those n_samples rows carried weight, i.e. how many
            # actually entered the Newton fit. See the note above the loop.
            parameters={"a": a, "b": b, "n_weighted_rows": n_informative},
            # `iterations` is the number of Newton steps ACTUALLY TAKEN. It used
            # to be `iteration + 1`, the loop counter plus one, which reported
            # "iterations: 1" for a loop body that ran zero times: both the
            # could-not-check path and PlattScaling(max_iter=0) published a 1
            # next to parameters nothing had touched.
            fit_metrics={"log_loss": log_loss, "iterations": n_steps},
            warnings=fit_notes,
        )

        return self

    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Apply Platt scaling transformation.

        Args:
            y_prob: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        self._check_is_fitted()
        y_prob = coerce_to_array(y_prob, "y_prob")
        validate_probabilities(y_prob, "y_prob")

        # Transform to logits, apply scaling, transform back
        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        scaled_logits = self.a_ * logits + self.b_
        calibrated = 1 / (1 + np.exp(-np.clip(scaled_logits, -500, 500)))

        return np.clip(calibrated, 0, 1)


class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic Regression calibration method.

    Non-parametric calibration that fits a monotonically increasing
    function mapping predicted probabilities to calibrated values:
    constant within each PAVA block, linearly interpolated between
    blocks (as in sklearn's IsotonicRegression). Preserves rank
    ordering of predictions.

    Advantages:
        - Flexible, no parametric assumptions
        - Preserves monotonicity (rank order)
        - Can capture complex calibration curves

    Disadvantages:
        - May overfit with small samples
        - Piecewise constant output

    Example:
        >>> calibrator = IsotonicCalibrator(out_of_bounds='clip')
        >>> calibrator.fit(y_true, y_prob)
        >>> calibrated = calibrator.transform(y_prob_test)

    References:
        Zadrozny, B. & Elkan, C. (2002). Transforming Classifier Scores into
        Accurate Multiclass Probability Estimates. KDD.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: isotonic_calibration. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        out_of_bounds: Literal["nan", "clip", "error"] = "clip",
        y_min: float = 0.0,
        y_max: float = 1.0,
    ):
        super().__init__()
        self.out_of_bounds = out_of_bounds
        self.y_min = y_min
        self.y_max = y_max
        self.x_thresholds_: Optional[np.ndarray] = None
        self.y_values_: Optional[np.ndarray] = None

    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression using pool adjacent violators algorithm.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator
        """
        y_true, y_prob, sample_weight, fit_notes = self._validate_fit_inputs(
            y_true, y_prob, sample_weight
        )

        n_given = len(y_true)
        if sample_weight is None:
            sample_weight = np.ones(n_given)

        # BGL-D sibling (2026-09-17). A row with ZERO sample_weight is an
        # observation that does not count, and weighted PAVA cannot carry it:
        # merging two zero-weight blocks divides 0 by 0. Measured at HEAD of
        # this working tree, 60 informative scores with EVERY weight 0, PAVA
        # produced 11 numpy "invalid value encountered in scalar divide"
        # RuntimeWarnings, 53 of the 60 knot values came back NaN, and fit()
        # published parameters {'n_blocks': 49, 'n_knots': 60,
        # 'n_distinct_values': 2, 'x_range': (0.0226, 0.9773)} with
        # warnings: [] and NOT ONE UserWarning. The two "distinct values" were
        # 0.0 and NaN, so a field whose whole job is to say "this map is flat"
        # reported variation that does not exist, in a record that serialises
        # byte-identically to a healthy one. transform() did return NaN, so the
        # ARRAY was honest while the RECORD a reader grades was not.
        # PlattScaling.fit and TemperatureScaling.fit already screen the same
        # quantity the same way; drop the weightless rows and refuse when none
        # are left.
        positive_weight = sample_weight > 0
        n = int(np.count_nonzero(positive_weight))
        if n == 0:
            self.x_thresholds_ = np.unique(y_prob)
            self.y_values_ = np.full(self.x_thresholds_.size, np.nan)
            self.is_fitted = True
            unmeasurable_note = (
                f"{self.__class__.__name__}.fit() identified NO isotonic map: all "
                f"{n_given} rows carry sample_weight 0, so not one observation enters "
                f"the weighted fit. The knot values are reported as NaN rather than as "
                f"the 0/0 blocks PAVA produces, and transform() returns NaN. This is a "
                f"could-not-check, not a calibration."
            )
            fit_notes.append(unmeasurable_note)
            warnings.warn(unmeasurable_note, UserWarning, stacklevel=2)
            self.fit_result = CalibrationFitResult(
                method="isotonic_regression",
                n_samples=n_given,
                parameters={
                    "n_blocks": 0,
                    "n_knots": int(self.x_thresholds_.size),
                    "n_distinct_values": 0,
                    "n_weighted_rows": 0,
                    "x_range": (
                        float(self.x_thresholds_[0]),
                        float(self.x_thresholds_[-1]),
                    ),
                },
                warnings=fit_notes,
            )
            return self
        if n < n_given:
            fit_notes.append(
                f"{n_given - n} of {n_given} rows carry sample_weight 0 and were "
                f"excluded from the isotonic fit; the map below is fitted from "
                f"{n} observation(s), not from {n_given}"
            )
        y_true = y_true[positive_weight]
        y_prob = y_prob[positive_weight]
        sample_weight = sample_weight[positive_weight]

        order = np.argsort(y_prob)
        y_true_sorted = y_true[order]
        y_prob_sorted = y_prob[order]
        weights_sorted = sample_weight[order]

        # Pool Adjacent Violators Algorithm (PAVA)
        # Merge adjacent blocks that violate monotonicity
        n_blocks = n
        block_means = y_true_sorted.astype(float).copy()
        block_weights = weights_sorted.copy()
        block_starts = np.arange(n)

        changed = True
        while changed:
            changed = False
            i = 0
            while i < n_blocks - 1:
                if block_means[i] > block_means[i + 1]:
                    # Merge blocks
                    total_weight = block_weights[i] + block_weights[i + 1]
                    block_means[i] = (
                        block_means[i] * block_weights[i]
                        + block_means[i + 1] * block_weights[i + 1]
                    ) / total_weight
                    block_weights[i] = total_weight

                    # Shift remaining blocks
                    block_means[i + 1 : n_blocks - 1] = block_means[i + 2 : n_blocks]
                    block_weights[i + 1 : n_blocks - 1] = block_weights[i + 2 : n_blocks]
                    block_starts[i + 1 : n_blocks - 1] = block_starts[i + 2 : n_blocks]
                    n_blocks -= 1
                    changed = True
                else:
                    i += 1

        # The isotonic regression solution at every training point is its PAVA
        # block mean, constant across the WHOLE block. Keeping only one knot per
        # block (the block's first x) made transform() ramp interior points
        # toward the neighbouring block's mean, so the output was no longer the
        # least-squares-optimal monotone fit (residual training ECE ~3% where
        # the correct fit leaves ~0).
        fitted = np.empty(n, dtype=float)
        for i in range(n_blocks):
            start = block_starts[i]
            end = block_starts[i + 1] if i < n_blocks - 1 else n
            fitted[start:end] = block_means[i]

        # Collapse duplicate x values (ties in y_prob) to their weight-averaged
        # fitted value so the interpolation knots are strictly increasing in x.
        ux, inv = np.unique(y_prob_sorted, return_inverse=True)
        w_per_x = np.bincount(inv, weights=weights_sorted)
        # BGL3-PP3 (2026-09-27). This denominator was np.clip(w_per_x, 1e-12,
        # None), a MAGNITUDE floor written when a weightless row could still
        # reach here. It cannot any more (the positive_weight screen above drops
        # them, so every unique x holds at least one strictly positive weight and
        # a sum of positive doubles is positive), and the floor's only remaining
        # effect was to divide by the wrong number whenever the weights are
        # smaller than it. Isotonic regression is invariant to a uniform positive
        # rescale of the weights; measured on 8 rows, y = [0,0,1,1,0,1,0,1] at
        # scores 0.1 ... 0.9:
        #
        #   every weight 1.0    -> transform(0.1, 0.25, 0.4, 0.9) = [0, .3, .6, 1]
        #   every weight 1e-11  -> the same
        #   every weight 1e-13  -> [0, .03, .06, .1], every calibrated
        #                          probability 10x too small
        #
        # and the published record was byte-identical in all three: n_blocks 4,
        # n_knots 8, n_distinct_values 3, warnings []. A positivity guard keeps
        # the original no-divide-by-zero intent without rescaling anything: a
        # denominator that somehow is not positive yields NaN, which transform()
        # carries, rather than a quietly wrong map.
        safe_w_per_x = np.where(w_per_x > 0.0, w_per_x, np.nan)
        uy = np.bincount(inv, weights=fitted * weights_sorted) / safe_w_per_x
        # Tie-collapsing across a block boundary averages two block means, which
        # stays within their range; enforce monotonicity defensively anyway.
        uy = np.maximum.accumulate(uy)

        self.x_thresholds_ = ux
        self.y_values_ = np.clip(uy, self.y_min, self.y_max)
        self.is_fitted = True

        # BGL-D (2026-09-16). n_blocks is the PRE-tie-collapse PAVA count, so it
        # describes a resolution the fitted map does not have: on a constant
        # score column it read "n_blocks: 3" (and "n_blocks: 4" at other n)
        # while the map had exactly ONE knot and transform([0.1, 0.5, 0.9])
        # answered a flat [0.45, 0.45, 0.45]. That record serialised
        # byte-identically to a healthy one, warnings [] included, and on the
        # single-class path it read "warnings": [] while the same call had just
        # emitted a UserWarning saying nothing could be learned. Report the knot
        # count beside n_blocks and state the degeneracy in the durable record.
        n_knots = int(len(self.x_thresholds_))
        # The knot count only tests the X axis. A map can have 40 knots and
        # still be FLAT: PAVA pools every block when the scores carry no signal,
        # and the y values then come back identical. Measured 2026-09-17 on
        # plain no-signal data (n=40, seed 2, random scores, labels from a fair
        # coin): parameters {'n_blocks': 1, 'n_knots': 40}, warnings [], and
        # transform([0.1, 0.5, 0.9]) -> [0.425, 0.425, 0.425]. That record is
        # indistinguishable from a healthy fit on every published field, and it
        # happens in roughly 5% of no-signal fits at that n, so the y axis is
        # tested beside the x axis and the distinct-value count is published.
        n_distinct_values = int(np.unique(self.y_values_).size)
        if n_knots < 2:
            degenerate_note = (
                f"the fitted isotonic map has a single knot at "
                f"x={float(self.x_thresholds_[0]):.6g}: the {n} training scores carry "
                f"no variation, so transform() returns "
                f"{float(self.y_values_[0]):.6g} for every input and preserves no "
                f"ranking. This is the training base rate, not a calibration."
            )
            fit_notes.append(degenerate_note)
            # Stated on BOTH channels, as the single-class path in
            # _validate_fit_inputs already is: a note that lives only in
            # fit_result.warnings is invisible to a caller who reads the
            # transformed array and nothing else.
            warnings.warn(degenerate_note, UserWarning, stacklevel=2)
        elif n_distinct_values < 2:
            flat_note = (
                f"the fitted isotonic map is FLAT: its {n_knots} knots span "
                f"x=[{float(self.x_thresholds_[0]):.6g}, "
                f"{float(self.x_thresholds_[-1]):.6g}] but PAVA pooled every block, so "
                f"all of them carry the single value {float(self.y_values_[0]):.6g}. "
                f"transform() returns that number for every input and preserves no "
                f"ranking: the {n} training scores carry no monotone signal about the "
                f"outcome. This is the training base rate, not a calibration."
            )
            fit_notes.append(flat_note)
            warnings.warn(flat_note, UserWarning, stacklevel=2)

        self.fit_result = CalibrationFitResult(
            method="isotonic_regression",
            n_samples=n_given,
            parameters={
                "n_blocks": n_blocks,
                "n_knots": n_knots,
                # How many of those n_samples rows actually entered the fit.
                "n_weighted_rows": n,
                # n_knots is the resolution of the X axis; this is the
                # resolution of the OUTPUT, and 1 means the map is constant.
                "n_distinct_values": n_distinct_values,
                "x_range": (float(self.x_thresholds_[0]), float(self.x_thresholds_[-1])),
            },
            warnings=fit_notes,
        )

        return self

    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Apply isotonic calibration transformation.

        Args:
            y_prob: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        self._check_is_fitted()
        # fit() always assigns both arrays before setting is_fitted=True, so
        # _check_is_fitted() guarantees they are non-None here.
        assert self.x_thresholds_ is not None and self.y_values_ is not None
        y_prob = coerce_to_array(y_prob, "y_prob")
        # BGL-D sibling (2026-09-17). The same refusal HistogramBinning.transform
        # and PlattScaling.transform already state. Without it, out_of_bounds
        # "clip" clamped a score that is not a probability at all onto the end
        # knots and published an absolute certainty: measured at HEAD of this
        # working tree, fitted on 400 informative rows,
        # transform([1.7, 2.0, 42.0]) -> [1.0, 1.0, 1.0] and
        # transform([-3.0, -0.2, -1.0]) -> [0.0, 0.0, 0.0], with zero warnings.
        # out_of_bounds still governs scores INSIDE [0, 1] that fall outside the
        # training x range, which is what it was added for.
        validate_probabilities(y_prob, "y_prob")

        # Use linear interpolation between thresholds
        calibrated = np.interp(
            y_prob,
            self.x_thresholds_,
            self.y_values_,
            left=self.y_values_[0] if self.out_of_bounds == "clip" else np.nan,
            right=self.y_values_[-1] if self.out_of_bounds == "clip" else np.nan,
        )

        if self.out_of_bounds == "error":
            mask = (y_prob < self.x_thresholds_[0]) | (y_prob > self.x_thresholds_[-1])
            if np.any(mask):
                raise ValueError(
                    f"Probabilities outside training range: "
                    f"[{self.x_thresholds_[0]:.4f}, {self.x_thresholds_[-1]:.4f}]"
                )

        return np.clip(calibrated, 0, 1)


class BetaCalibrator(BaseCalibrator):
    """
    Beta Calibration method.

    Uses a beta distribution family to model the relationship between
    predicted probabilities and true labels. Well-suited for naturally
    bounded probability estimates.

    The transformation uses: g(p) = 1 / (1 + 1/(exp(c) * p^a / (1-p)^b))
    where a, b, c are learned parameters. Per Kull et al. 2017 the family
    requires a >= 0 and b >= 0 (that is what makes the map monotone); when
    an unconstrained coefficient comes out negative the appropriate reduced
    model is refit instead, and the clamp is recorded in
    ``fit_result.warnings`` (``fit_result.parameters['fitted_model']`` names
    the member actually fitted).

    Advantages:
        - Handles bounded probability outputs naturally
        - More flexible than Platt scaling
        - Theoretically motivated for classifier outputs

    Example:
        >>> calibrator = BetaCalibrator()
        >>> calibrator.fit(y_true, y_prob)
        >>> calibrated = calibrator.transform(y_prob_test)

    References:
        Kull, M., Silva Filho, T. M., & Flach, P. (2017). Beta Calibration:
        A Well-Founded and Easily Implemented Improvement on Logistic Calibration.
        AISTATS.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: beta_calibration. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self, parameters: Literal["abm", "am", "bm"] = "abm", max_iter: int = 100, tol: float = 1e-6
    ):
        super().__init__()
        self.parameters_type = parameters
        self.max_iter = max_iter
        self.tol = tol
        self.a_: Optional[float] = None
        self.b_: Optional[float] = None
        self.m_: Optional[float] = None

    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "BetaCalibrator":
        """
        Fit beta calibration parameters.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator
        """
        y_true, y_prob, sample_weight, fit_notes = self._validate_fit_inputs(
            y_true, y_prob, sample_weight
        )

        n = len(y_true)
        if sample_weight is None:
            sample_weight = np.ones(n)

        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

        # Log-odds transformation for beta calibration
        # log(p) and log(1-p) as features
        log_p = np.log(y_prob_clipped)
        log_1_minus_p = np.log(1 - y_prob_clipped)

        def _design(model_type: str) -> np.ndarray:
            """Design matrix for one member of the beta-calibration family."""
            if model_type == "abm":
                # Full model: a*log(p) - b*log(1-p) + m
                return np.column_stack([log_p, -log_1_minus_p, np.ones(n)])
            if model_type == "am":
                # Constrained: a*log(p) + m (b=0)
                return np.column_stack([log_p, np.ones(n)])
            if model_type == "bm":
                # Constrained: -b*log(1-p) + m (a=0)
                return np.column_stack([-log_1_minus_p, np.ones(n)])
            # 'm': intercept-only fallback (a=b=0), a constant map
            return np.ones((n, 1))

        # A design whose rank is below its column count does not IDENTIFY its
        # coefficients: infinitely many (a, b, m) give the same likelihood, and
        # whichever one the optimiser stops at is an artefact of where it
        # started. Measured 2026-09-16 on a constant score column (all 0.7,
        # both labels, n=200): every design column is constant, the gradient at
        # params=0 is exactly zero, the loop exits on its first iteration, and
        # fit() published its own INITIALISATION as the estimate: parameters
        # {'a': 0.0, 'b': 0.0, 'm': 0.0}, fitted_model 'abm', warnings [],
        # is_fitted True, no Python warning, and transform([0.1, 0.5, 0.9]) ->
        # [0.5, 0.5, 0.5]. The n=2 case is the same defect wearing the opposite
        # face: 3 parameters on 2 rows fitted a=36.2, b=35.1, m=-4.4 and
        # transformed to [2.96e-37, 5.40e-03, 1.0], near-absolute certainties
        # extrapolated from two observations, also with warnings [].
        # np.linalg.matrix_rank covers BOTH (constant column -> rank 1 of 3;
        # n=2 -> rank 2 of 3) with one test, above the model selection below,
        # so fixing one face cannot leave the other live.
        # BGL-D sibling (2026-09-17). Rows with ZERO sample_weight contribute
        # nothing to the gradient and nothing to the Hessian (both carry
        # `sample_weight` as a factor), so identifiability is a property of the
        # WEIGHTED rows, not of every row handed in. Ranking the full design
        # matrix got that wrong in the direction this audit exists to find.
        # Measured at HEAD of this working tree, 60 informative scores:
        #   * every weight 0 -> the Hessian is 1e-6*I, the gradient is exactly
        #     zero, the loop converges on its first iteration without moving,
        #     and fit() published its own INITIALISATION as the estimate:
        #     parameters {'a': 0.0, 'b': 0.0, 'm': 0.0, 'fitted_model': 'abm'},
        #     warnings [], no Python warning, is_fitted True, and
        #     transform([0.1, 0.5, 0.9]) -> [0.5, 0.5, 0.5].
        #   * exactly ONE row with weight 1.0 -> parameters {'a': 5.711097,
        #     'b': 0.0, 'm': -1.716100, 'fitted_model': 'am'} and
        #     transform([0.1, 0.5, 0.9]) -> [0.0, 0.003394, 0.089675]: a full
        #     differentiated calibration map read off ONE observation, with
        #     n_samples published as 60 and no note saying so.
        # PlattScaling.fit and TemperatureScaling.fit already screen the same
        # way on the same quantity; this is that test in the shape beta
        # calibration needs (a rank, because the design has three columns).
        positive_weight = sample_weight > 0

        def _design_rank(model_type: str) -> int:
            """Rank of the design over the rows that actually carry weight."""
            return int(np.linalg.matrix_rank(_design(model_type)[positive_weight]))

        def _is_identified(model_type: str) -> bool:
            return _design_rank(model_type) >= _design(model_type).shape[1]

        def _newton_fit(X: np.ndarray) -> Tuple[np.ndarray, bool]:
            """Newton-Raphson for logistic regression on design X.

            Returns ``(params, converged)``. ``converged`` is False when the
            loop exhausted ``max_iter`` or the Hessian solve failed, so the
            caller can record that the reported coefficients are the last
            iterate rather than an optimum.
            """
            n_params = X.shape[1]
            params = np.zeros(n_params)
            converged = False
            for _iteration in range(self.max_iter):
                scores = X @ params
                probs = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

                # Gradient and Hessian
                diff = probs - y_true
                grad = X.T @ (sample_weight[:, np.newaxis] * diff[:, np.newaxis]).flatten()

                hess_diag = sample_weight * probs * (1 - probs)
                hess = X.T @ (X * hess_diag[:, np.newaxis])

                # Add regularization for numerical stability
                hess += 1e-6 * np.eye(n_params)

                # Newton step
                try:
                    delta = np.linalg.solve(hess, grad)
                except np.linalg.LinAlgError:
                    break

                params -= delta

                if np.max(np.abs(delta)) < self.tol:
                    converged = True
                    break
            return params, converged

        # Kull et al. 2017 (Sec. 3) define the family with a >= 0 and b >= 0:
        # that constraint is exactly what makes sigmoid(a*ln p - b*ln(1-p) + m)
        # monotone non-decreasing (d/dp has the sign of a/p + b/(1-p)). The
        # unconstrained fit can return negative a or b, and then the "calibrator"
        # silently REVERSES the ranking of scores over part or all of [0, 1],
        # flipping downstream threshold decisions. Per the paper and the
        # reference betacal implementation, refit the reduced model with the
        # offending feature dropped; if the surviving coefficient is still
        # negative, fall back to the intercept-only map.
        fit_warnings: list = list(fit_notes)
        # Widened to str: the reduced-model fallbacks below assign "bm"/"am"/"m",
        # which are narrower design strings than the configured parameters_type.
        fitted_model: str = self.parameters_type

        # Guard ABOVE the model selection: every branch below shares this
        # precondition, and the reduced-model refits inherit it.
        if not _is_identified(fitted_model):
            n_informative = int(np.count_nonzero(positive_weight))
            unmeasurable_note = (
                f"{self.__class__.__name__}.fit() identified NO beta-calibration map: "
                f"the '{fitted_model}' design built from the {n_informative} of {n} "
                f"rows with non-zero weight is rank-deficient "
                f"(rank {_design_rank(fitted_model)} of "
                f"{_design(fitted_model).shape[1]} columns), so a, b and m are not "
                f"determined by the data. Parameters are reported as NaN rather than "
                f"as the optimiser's starting values, and transform() returns NaN. "
                f"This is a could-not-check, not a calibration."
            )
            fit_warnings.append(unmeasurable_note)
            warnings.warn(unmeasurable_note, UserWarning, stacklevel=2)
            self.a_ = float("nan")
            self.b_ = float("nan")
            self.m_ = float("nan")
            self.is_fitted = True
            self.fit_result = CalibrationFitResult(
                method=f"beta_calibration_{self.parameters_type}",
                n_samples=n,
                parameters={
                    "a": self.a_,
                    "b": self.b_,
                    "m": self.m_,
                    "fitted_model": fitted_model,
                    "design_rank": _design_rank(fitted_model),
                    "n_design_columns": int(_design(fitted_model).shape[1]),
                    # The rank above is taken over the weighted rows, so publish
                    # how many there were: a rank of 1 of 3 means something very
                    # different at n_weighted 60 than at n_weighted 1.
                    "n_weighted_rows": int(np.count_nonzero(positive_weight)),
                },
                warnings=fit_warnings,
            )
            return self

        params, converged = _newton_fit(_design(fitted_model))

        if fitted_model == "abm":
            a, b, m = (float(params[0]), float(params[1]), float(params[2]))
            if a < 0:
                fit_warnings.append(
                    f"unconstrained a={a:.4f} < 0 violates the beta-calibration "
                    f"family (Kull et al. 2017); refit reduced model 'bm' (a=0)"
                )
                fitted_model = "bm"
                params, converged = _newton_fit(_design("bm"))
                a, b, m = 0.0, float(params[0]), float(params[1])
            elif b < 0:
                fit_warnings.append(
                    f"unconstrained b={b:.4f} < 0 violates the beta-calibration "
                    f"family (Kull et al. 2017); refit reduced model 'am' (b=0)"
                )
                fitted_model = "am"
                params, converged = _newton_fit(_design("am"))
                a, b, m = float(params[0]), 0.0, float(params[1])
        elif fitted_model == "am":
            a, b, m = float(params[0]), 0.0, float(params[1])
        else:  # 'bm'
            a, b, m = 0.0, float(params[0]), float(params[1])

        if a < 0 or b < 0:
            # The reduced model is still outside the family (e.g. fully
            # anti-calibrated data). The only family member left is the
            # constant map: intercept-only, a=b=0.
            fit_warnings.append(
                f"reduced model '{fitted_model}' still has a negative "
                f"coefficient (a={a:.4f}, b={b:.4f}); falling back to the "
                f"intercept-only constant map (a=b=0)"
            )
            fitted_model = "m"
            params, converged = _newton_fit(_design("m"))
            a, b, m = 0.0, 0.0, float(params[0])

        if not converged:
            fit_warnings.append(
                f"the Newton loop for design '{fitted_model}' hit max_iter "
                f"({self.max_iter}) without reaching tol={self.tol:g}; the reported "
                f"a, b and m are the last iterate, not a converged optimum"
            )

        self.a_, self.b_, self.m_ = a, b, m
        self.is_fitted = True

        self.fit_result = CalibrationFitResult(
            method=f"beta_calibration_{self.parameters_type}",
            n_samples=n,
            parameters={"a": self.a_, "b": self.b_, "m": self.m_, "fitted_model": fitted_model},
            warnings=fit_warnings,
        )

        return self

    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Apply beta calibration transformation.

        Args:
            y_prob: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        self._check_is_fitted()
        y_prob = coerce_to_array(y_prob, "y_prob")
        # BGL-D sibling (2026-09-17). Same refusal as PlattScaling.transform and
        # HistogramBinning.transform. Without it the np.clip below silently
        # turned a non-probability into an end-of-range probability and the map
        # published an absolute certainty for it: measured at HEAD of this
        # working tree, fitted on 400 informative rows,
        # transform([1.7, 2.0, 42.0]) -> [1.0, 1.0, 1.0] and
        # transform([-3.0, -0.2, -1.0]) -> [0.0, 0.0, 0.0], with zero warnings.
        validate_probabilities(y_prob, "y_prob")

        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

        # Beta calibration transformation
        log_odds = self.a_ * np.log(y_prob_clipped) - self.b_ * np.log(1 - y_prob_clipped) + self.m_

        calibrated = 1 / (1 + np.exp(-np.clip(log_odds, -500, 500)))
        return np.clip(calibrated, 0, 1)


class TemperatureScaling(BaseCalibrator):
    """
    Temperature Scaling calibration method.

    A simple but effective calibration technique that divides logits by
    a single learned temperature parameter. Originally developed for
    neural networks but applicable to any probability output.

    The transformation is: P_cal = softmax(logits / T) for multiclass
    For binary: P_cal = sigmoid(logit / T)

    Advantages:
        - Single parameter, very simple
        - Preserves accuracy (only rescales)
        - Effective for modern neural networks

    Disadvantages:
        - May be too simple for complex miscalibration
        - Assumes symmetric miscalibration

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.fit(y_true, y_prob)
        >>> calibrated = calibrator.transform(y_prob_test)

    References:
        Guo, C., et al. (2017). On Calibration of Modern Neural Networks.
        ICML.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: temperature_scaling. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        """
        Args:
            max_iter: Iteration cap for the bounded scalar minimiser
            tol: Absolute tolerance on the temperature (xatol)

        An `init_temperature` argument used to be accepted here (F17,
        removed 2026-09-09). fit() minimises the NLL with
        scipy.optimize.minimize_scalar(method='bounded'), which takes a
        bracket and no starting point, so the value was stored and never
        read. Measured: init_temperature 0.1, 1.0 and 50.0 all fitted
        T=0.7168 on the same data. A starting point the optimiser cannot
        use is not offered.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.temperature_: Optional[float] = None

    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "TemperatureScaling":
        """
        Fit temperature parameter to minimize NLL.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator
        """
        y_true, y_prob, sample_weight, fit_notes = self._validate_fit_inputs(
            y_true, y_prob, sample_weight
        )

        n = len(y_true)
        if sample_weight is None:
            sample_weight = np.ones(n)

        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Optimize temperature by minimizing the weighted NLL of
        # sigmoid(logit / T) over T > 0 (Guo et al. 2017). This is a
        # smooth 1-D problem, so a bounded scalar minimizer is exact and
        # robust; the previous fixed-step gradient descent (a) used an
        # inverted NLL gradient, so it climbed NLL instead of descending,
        # and (b) stepped on the raw *summed* gradient with no line
        # search, so even with the correct sign it overshot the optimum
        # and left calibration worse than the uncalibrated input. Both are
        # avoided here by optimizing the objective directly. T = 1 (the
        # identity / uncalibrated map) lies inside the bounds, so the fit
        # can never do worse than leaving the probabilities untouched.
        from scipy.optimize import minimize_scalar

        def _weighted_nll(temperature: float) -> float:
            scaled = logits / temperature
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -500, 500)))
            probs = np.clip(probs, eps, 1 - eps)
            return -float(
                np.sum(sample_weight * (y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs)))
            )

        # BGL-D sibling (2026-09-17). Temperature scaling has ONE parameter and
        # it only RESCALES the logits, so a score column with a single distinct
        # value identifies nothing: every candidate T maps that one logit to one
        # probability, the likelihood is flat in the direction that matters, and
        # the minimiser simply runs to whichever bound the base rate points at.
        # Measured at HEAD of this working tree, 200 rows all scored 0.7:
        # base rate 0.50 -> temperature 99.9999975, nll 0.69316, warnings [],
        # transform([0.1, 0.5, 0.9]) -> [0.4945, 0.5, 0.5055]; base rate 0.10 ->
        # temperature 99.9999979 and the SAME [0.4945, 0.5, 0.5055], i.e. the
        # published calibration did not depend on the outcomes at all, and the
        # only thing separating it from a real fit was a T pinned at its own
        # search boundary, which nothing reported. Same test as PlattScaling
        # uses, on the same quantity, above the optimise.
        positive_weight = sample_weight > 0
        n_informative = int(np.count_nonzero(positive_weight))
        distinct_scores = int(np.unique(logits[positive_weight]).size) if n_informative else 0
        # BGL5 (2026-09-27). The guard below only fires on TOTAL weightlessness:
        # `distinct_scores` counts distinct logits over the WEIGHTED rows, so it
        # is under 2 only when no row carries weight (or they all share a score).
        # A PARTIALLY weightless fit sailed past it publishing the full row
        # count, while `CalibrationFitResult.n_samples` is documented as "Number
        # of samples used for fitting" (methods.py:61). MEASURED on 60
        # informative scores with 3 of them carrying weight 1.0 and 57 carrying
        # 0: n_samples 60, parameters {'temperature': 0.8645148917436684},
        # fit_metrics {'nll': 0.386125, 'iterations': 19}, record warnings [] and
        # ZERO UserWarnings, i.e. 57 rows that contributed exactly nothing to the
        # weighted objective were counted as having been fitted from. Both
        # siblings in this file disclose it on the identical input:
        # IsotonicCalibrator.fit publishes 'n_weighted_rows': 3 plus "57 of 60
        # rows carry sample_weight 0 and were excluded from the isotonic fit",
        # and BetaCalibrator.fit publishes 'n_weighted_rows' too. This mirrors
        # them: n_weighted_rows is published on every record, and the note is
        # written only when rows were actually dropped, so a fit at uniform
        # weight keeps warnings == [] (the control the existing pin asserts).
        if n_informative < n:
            fit_notes.append(
                f"{n - n_informative} of {n} rows carry sample_weight 0 and contribute "
                f"nothing to the weighted NLL; the temperature below is fitted from "
                f"{n_informative} observation(s), not from {n}"
            )
        if distinct_scores < 2:
            unmeasurable_note = (
                f"{self.__class__.__name__}.fit() identified NO temperature: the design is "
                f"rank-deficient (across the {n_informative} of {n} rows with non-zero "
                f"weight the scores take {distinct_scores} distinct value(s), and a "
                f"temperature only rescales logits, so no T is identified by the data). "
                f"temperature_ is reported as NaN rather than as the search bound the "
                f"minimiser runs to, and transform() returns NaN. This is a "
                f"could-not-check, not a calibration."
            )
            fit_notes.append(unmeasurable_note)
            warnings.warn(unmeasurable_note, UserWarning, stacklevel=2)
            self.temperature_ = float("nan")
            self.is_fitted = True
            self.fit_result = CalibrationFitResult(
                method="temperature_scaling",
                n_samples=n,
                # How many of those n_samples rows actually entered the fit, the
                # same key IsotonicCalibrator.fit and BetaCalibrator.fit publish.
                parameters={"temperature": float("nan"), "n_weighted_rows": n_informative},
                fit_metrics={"nll": float("nan"), "iterations": 0},
                warnings=fit_notes,
            )
            return self

        # Lower bound keeps the historical positive-temperature floor
        # (0.01); the upper bound is generous for real miscalibration.
        lower_bound, upper_bound = 0.01, 100.0
        result = minimize_scalar(
            _weighted_nll,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"maxiter": self.max_iter, "xatol": self.tol},
        )
        T = float(result.x)
        iteration = int(getattr(result, "nit", self.max_iter)) - 1

        # A bounded minimiser always returns a number inside its bracket, so the
        # only way to tell "the optimum is HERE" from "the optimum is outside
        # and I stopped at the edge" is to evaluate the objective at the bounds.
        # `result.x` alone does not say it: on 2 rows with distinct scores the
        # data is perfectly separable, the NLL falls monotonically as T -> 0, and
        # fit() published temperature 0.014951 with nll 8.3e-13 and
        # transform([0.1, 0.5, 0.9]) -> [1.5e-64, 0.5, 1.0], near-absolute
        # certainties extrapolated from two observations, with warnings [].
        # 0.014951 is not close enough to the 0.01 floor for a proximity test to
        # catch it, which is why this compares objective values instead.
        nll_at_t = _weighted_nll(T)
        nll_at_lower = _weighted_nll(lower_bound)
        nll_at_upper = _weighted_nll(upper_bound)
        # A bounded search over a non-finite objective still returns a number
        # inside the bracket, and every comparison against NaN below is False,
        # so a non-finite objective would slip past the bound test in silence.
        # Weights are screened for that at the gate now; this covers the routes
        # that survive finite inputs, such as a weighted sum that overflows.
        if not np.isfinite(nll_at_t):
            unmeasurable_note = (
                f"{self.__class__.__name__}.fit() identified NO temperature: the "
                f"weighted NLL is not finite at the temperature the search returned "
                f"(T={T:g}), so the search minimised nothing and T is the bracket it "
                f"started from. temperature_ is reported as NaN and transform() returns "
                f"NaN. This is a could-not-check, not a calibration."
            )
            fit_notes.append(unmeasurable_note)
            warnings.warn(unmeasurable_note, UserWarning, stacklevel=2)
            self.temperature_ = float("nan")
            self.is_fitted = True
            self.fit_result = CalibrationFitResult(
                method="temperature_scaling",
                n_samples=n,
                parameters={"temperature": float("nan"), "n_weighted_rows": n_informative},
                fit_metrics={"nll": float("nan"), "iterations": 0},
                warnings=fit_notes,
            )
            return self
        if nll_at_lower <= nll_at_t or nll_at_upper <= nll_at_t:
            edge = lower_bound if nll_at_lower <= nll_at_upper else upper_bound
            fit_notes.append(
                f"the temperature optimum is NOT interior: the weighted NLL at the search "
                f"bound T={edge:g} ({min(nll_at_lower, nll_at_upper):.6g}) is no worse than "
                f"at the reported T={T:.6g} ({nll_at_t:.6g}), so T is pinned by the bound "
                f"and not determined by the data. Treat the output as unvalidated: it "
                f"follows from where the search stopped, not from a fitted optimum."
            )
        if not bool(getattr(result, "success", True)):
            fit_notes.append(
                f"the bounded minimiser did not report success after maxiter="
                f"{self.max_iter} ({getattr(result, 'message', 'no message')}); the "
                f"reported temperature is the last iterate, not a converged optimum"
            )

        self.temperature_ = T
        self.is_fitted = True

        calibrated = self.transform(y_prob)
        # `nll` is published under the name of the objective this fit minimised,
        # and that objective is WEIGHTED (see _weighted_nll above). This was an
        # unweighted `-np.mean(...)` over every row, so with 3 of 60 rows
        # carrying weight it reported 0.386125: a mean over all 60, including the
        # 57 the fit excluded, under the name of the number that was minimised.
        # The weighted mean is BIT-IDENTICAL whenever the weights are uniform
        # (sum(w*ll)/sum(w) == mean(ll) at w == 1), so a healthy fit is unchanged:
        # measured on the same 60 rows at weight 1.0, 0.309467169222508 both
        # before and after, compared with ==. On the 3-of-60 input it now reports
        # 0.27708874982683834, the weighted mean over the 3 rows that entered the
        # fit, in place of 0.38612468225821545 over all 60.
        row_nll = y_true * np.log(np.clip(calibrated, eps, 1)) + (1 - y_true) * np.log(
            np.clip(1 - calibrated, eps, 1)
        )
        total_weight = float(np.sum(sample_weight))
        final_nll = (
            -float(np.sum(sample_weight * row_nll)) / total_weight
            if total_weight > 0
            else float("nan")
        )

        self.fit_result = CalibrationFitResult(
            method="temperature_scaling",
            n_samples=n,
            parameters={"temperature": T, "n_weighted_rows": n_informative},
            fit_metrics={"nll": final_nll, "iterations": iteration + 1},
            warnings=fit_notes,
        )

        return self

    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Apply temperature scaling transformation.

        Args:
            y_prob: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        self._check_is_fitted()
        y_prob = coerce_to_array(y_prob, "y_prob")
        # BGL-D sibling (2026-09-17). Same refusal as PlattScaling.transform and
        # HistogramBinning.transform. Without it the np.clip below silently
        # turned a non-probability into an end-of-range probability and the
        # temperature map published an absolute certainty for it: measured at
        # HEAD of this working tree, fitted on 400 informative rows,
        # transform([1.7, 2.0, 42.0]) -> [1.0, 1.0, 1.0] and
        # transform([-3.0, -0.2, -1.0]) -> [0.0, 0.0, 0.0], with zero warnings.
        validate_probabilities(y_prob, "y_prob")

        eps = 1e-10
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))

        scaled_logits = logits / self.temperature_
        calibrated = 1 / (1 + np.exp(-np.clip(scaled_logits, -500, 500)))

        return np.clip(calibrated, 0, 1)


class HistogramBinning(BaseCalibrator):
    """
    Histogram Binning calibration method.

    Discretizes the probability space into bins and assigns each bin
    a calibrated probability based on the empirical frequency of
    positive labels within that bin.

    Advantages:
        - Simple and interpretable
        - Naturally handles any calibration curve shape
        - Fast to train and apply

    Disadvantages:
        - Requires choice of number of bins
        - May not preserve probability ranking
        - Discontinuous outputs

    Example:
        >>> calibrator = HistogramBinning(n_bins=15)
        >>> calibrator.fit(y_true, y_prob)
        >>> calibrated = calibrator.transform(y_prob_test)

    References:
        Zadrozny, B. & Elkan, C. (2001). Obtaining Calibrated Probability
        Estimates from Decision Trees and Naive Bayesian Classifiers.
        ICML.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: histogram_binning. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    #: Bins fitted from fewer than this many observations are still MEASURED
    #: (the empirical rate is real), but a rate from one or two rows is not a
    #: reliable calibration, and nothing used to record how many rows it came
    #: from. Bins below this count are named in ``fit_result.warnings``; they
    #: are not blanked unless ``min_bin_count`` says so.
    _LOW_COUNT_DISCLOSURE = 10

    def __init__(
        self,
        n_bins: int = 10,
        strategy: Literal["uniform", "quantile"] = "uniform",
        min_bin_count: int = 1,
    ):
        """
        Args:
            n_bins: Number of bins
            strategy: 'uniform' (equal-width) or 'quantile' (equal-count) edges
            min_bin_count: Minimum training observations a bin needs before its
                empirical rate is published. Bins below it hold NaN in
                ``bin_values_`` and transform() returns NaN for scores landing
                there. The default 1 blanks only genuinely EMPTY bins; raise it
                (e.g. 10) to refuse single-observation 0.0/1.0 certainties too.
        """
        super().__init__()
        if not isinstance(min_bin_count, (int, np.integer)) or isinstance(min_bin_count, bool):
            raise TypeError(f"min_bin_count must be an int, got {type(min_bin_count).__name__}")
        if min_bin_count < 1:
            raise ValueError(f"min_bin_count must be >= 1, got {min_bin_count}")
        # n_bins was never validated, so a bad value surfaced as whatever numpy
        # said several frames later and named neither the argument nor this
        # class: n_bins=-1 raised "negative dimensions are not allowed" out of
        # np.linspace, and n_bins='ten' raised "can only concatenate str (not
        # int) to str". Same two checks min_bin_count already had.
        if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool):
            raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}")
        self.n_bins = n_bins
        self.strategy = strategy
        self.min_bin_count = int(min_bin_count)
        self.bin_edges_: Optional[np.ndarray] = None
        self.bin_values_: Optional[np.ndarray] = None
        self.bin_counts_: Optional[np.ndarray] = None

    def fit(
        self, y_true: ArrayLike, y_prob: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> "HistogramBinning":
        """
        Fit histogram binning calibrator.

        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            sample_weight: Optional sample weights

        Returns:
            self: Fitted calibrator
        """
        y_true, y_prob, sample_weight, fit_notes = self._validate_fit_inputs(
            y_true, y_prob, sample_weight
        )

        n = len(y_true)
        if sample_weight is None:
            sample_weight = np.ones(n)

        if self.strategy == "uniform":
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        else:  # quantile
            self.bin_edges_ = np.percentile(y_prob, np.linspace(0, 100, self.n_bins + 1))
            # Ensure edges are unique and sorted
            self.bin_edges_ = np.unique(self.bin_edges_)
            self.bin_edges_[0] = 0
            self.bin_edges_[-1] = 1

        bin_indices = np.digitize(y_prob, self.bin_edges_[1:-1])

        n_bins_actual = len(self.bin_edges_) - 1
        self.bin_values_ = np.full(max(n_bins_actual, 0), np.nan)
        self.bin_counts_ = np.zeros(max(n_bins_actual, 0), dtype=int)
        bin_weight_sums = np.full(max(n_bins_actual, 0), np.nan)

        # BGL-D (2026-09-16). A bin holding ZERO training samples used to be
        # given its own MIDPOINT, which is the identity map: the fitted object
        # handed a caller's own raw score straight back as a "calibrated
        # probability", and no field anywhere said which bins had been observed.
        # Measured on a constant score column (all 0.7, n=200): bin_values_ =
        # [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.365, 0.75, 0.85, 0.95] against
        # true per-bin counts [0,0,0,0,0,0,200,0,0,0], warnings [], and
        # transform([0.05, 0.25, 0.95, 0.72]) -> [0.05, 0.25, 0.95, 0.75], i.e.
        # the identity for three of four inputs. This is not fixture-only: it
        # fires on healthy data whenever a model's scores never reach the top
        # bins. An unobserved bin is a could-not-check, so it holds NaN, and the
        # counts are published so a measured 0.0 (no positives among 40 rows)
        # can be told apart from an unmeasured one.
        for i in range(n_bins_actual):
            mask = bin_indices == i
            count = int(np.count_nonzero(mask))
            self.bin_counts_[i] = count
            if count >= self.min_bin_count:
                total_weight = np.sum(sample_weight[mask])
                bin_weight_sums[i] = total_weight
                if total_weight > 0:
                    # Weighted mean of true labels in this bin
                    self.bin_values_[i] = np.sum(sample_weight[mask] * y_true[mask]) / total_weight

        unobserved = np.flatnonzero(self.bin_counts_ == 0).tolist()
        below_min = np.flatnonzero(
            (self.bin_counts_ > 0) & (self.bin_counts_ < self.min_bin_count)
        ).tolist()
        blank_despite_samples = (self.bin_counts_ >= self.min_bin_count) & np.isnan(
            self.bin_values_
        )
        # Split by CAUSE. The single note used to say the weights "summed to 0"
        # for both, and that is false for the second: a NaN in sample_weight
        # makes the sum NaN, not 0. The NaN calibrated value was right; only the
        # explanation of it was wrong, and a reader chasing "summed to 0" looks
        # for zero weights that are not there.
        zero_weight = np.flatnonzero(blank_despite_samples & (bin_weight_sums == 0)).tolist()
        unusable_weight = np.flatnonzero(
            blank_despite_samples & ~np.isfinite(bin_weight_sums)
        ).tolist()
        low_count = np.flatnonzero(
            ~np.isnan(self.bin_values_) & (self.bin_counts_ < self._LOW_COUNT_DISCLOSURE)
        ).tolist()

        if unobserved:
            fit_notes.append(
                f"{len(unobserved)} of {n_bins_actual} bins held NO training sample "
                f"(bin indices {unobserved}); their calibrated value is NaN, not the "
                f"bin midpoint, and transform() returns NaN for scores landing there"
            )
        if below_min:
            fit_notes.append(
                f"bins {below_min} held fewer than min_bin_count={self.min_bin_count} "
                f"samples (counts "
                f"{[int(self.bin_counts_[i]) for i in below_min]}); their calibrated "
                f"value is NaN"
            )
        if zero_weight:
            fit_notes.append(
                f"bins {zero_weight} held samples whose sample_weight summed to 0; "
                f"no rate is defined there and their calibrated value is NaN"
            )
        if unusable_weight:
            fit_notes.append(
                f"bins {unusable_weight} held samples whose sample_weight sum is not "
                f"finite ("
                f"{[float(bin_weight_sums[i]) for i in unusable_weight]}); no rate is "
                f"defined there and their calibrated value is NaN"
            )
        if low_count:
            fit_notes.append(
                f"bins {low_count} were fitted from fewer than "
                f"{self._LOW_COUNT_DISCLOSURE} observations (counts "
                f"{[int(self.bin_counts_[i]) for i in low_count]}); their rates are "
                f"empirical but unreliable, and a single-observation bin publishes an "
                f"absolute 0.0 or 1.0. Raise min_bin_count to refuse them"
            )
        unmeasured = len(unobserved) + len(below_min) + len(zero_weight) + len(unusable_weight)
        if unmeasured:
            warnings.warn(
                f"{self.__class__.__name__}.fit(): {unmeasured} of "
                f"{n_bins_actual} bins carry no usable rate and hold NaN; "
                f"transform() returns NaN for scores landing in them. See "
                f"fit_result.warnings and fit_result.parameters['bin_counts'].",
                UserWarning,
                stacklevel=2,
            )
        if low_count:
            # Same caution, second channel. With the default min_bin_count=1 a
            # bin fitted from ONE row publishes an absolute 0.0 or 1.0, and that
            # caution lived only in fit_result.warnings: a caller who reads the
            # transformed array and nothing else saw a certainty with no tell.
            warnings.warn(
                f"{self.__class__.__name__}.fit(): bins {low_count} of "
                f"{n_bins_actual} were fitted from fewer than "
                f"{self._LOW_COUNT_DISCLOSURE} observations (counts "
                f"{[int(self.bin_counts_[i]) for i in low_count]}); their rates are "
                f"measured but unreliable, and a single-observation bin publishes an "
                f"absolute 0.0 or 1.0. See fit_result.parameters['bin_counts'].",
                UserWarning,
                stacklevel=2,
            )

        # BGL-D sibling (2026-09-17). The per-bin disclosures above all assume
        # there IS a bin grid. When the GRID ITSELF collapses, every one of them
        # is empty, so the record read as a complete measurement of nothing:
        #   * strategy="quantile" on a constant score column: np.percentile
        #     returns one repeated edge, np.unique collapses it to a single
        #     element, and `bin_edges_[0] = 0; bin_edges_[-1] = 1` then write
        #     BOTH to that same element. Measured: parameters
        #     {'n_bins': 0, 'bin_counts': [], 'n_bins_unmeasured': 0},
        #     warnings [], no Python warning, is_fitted True, and transform()
        #     died with "IndexError: index -1 is out of bounds for axis 0 with
        #     size 0" at use time instead of refusing at fit time.
        #   * strategy="quantile" on a two-value column (950 rows at 0.2 with
        #     true rate 0.101, 50 at 0.8 with true rate 0.86): the 10 requested
        #     bins collapse to ONE, bin_values_ [0.139], warnings [], and
        #     transform is flat at 0.139 for every input, so the 0.86 group and
        #     the 0.101 group leave with the same calibrated probability and no
        #     field said the resolution was gone.
        # n_bins_requested is published beside the actual so a 10-bin request
        # that produced 1 is visible in the durable record, not only in prose.
        if n_bins_actual < 1:
            grid_note = (
                f"{self.__class__.__name__}.fit() built NO bin: the '{self.strategy}' "
                f"grid over {n} training scores collapsed to "
                f"{len(self.bin_edges_)} edge(s), so not one rate was measured. "
                f"transform() returns NaN for every input. This is a could-not-check, "
                f"not a calibration."
            )
            fit_notes.append(grid_note)
            warnings.warn(grid_note, UserWarning, stacklevel=2)
        elif n_bins_actual == 1:
            only_value = float(self.bin_values_[0])
            grid_note = (
                f"the '{self.strategy}' grid requested {self.n_bins} bins and produced 1: "
                f"every one of the {n} training scores fell in "
                f"[{float(self.bin_edges_[0]):.6g}, {float(self.bin_edges_[1]):.6g}], so "
                f"transform() returns {only_value:.6g} for every input and preserves no "
                f"ranking. This is the training base rate, not a calibration."
            )
            fit_notes.append(grid_note)
            warnings.warn(grid_note, UserWarning, stacklevel=2)

        self.is_fitted = True

        self.fit_result = CalibrationFitResult(
            method=f"histogram_binning_{self.strategy}",
            n_samples=n,
            parameters={
                "n_bins": n_bins_actual,
                "n_bins_requested": int(self.n_bins),
                "bin_edges": self.bin_edges_.tolist(),
                "bin_counts": self.bin_counts_.tolist(),
                "min_bin_count": self.min_bin_count,
                "n_bins_unmeasured": unmeasured,
            },
            warnings=fit_notes,
        )

        return self

    def transform(self, y_prob: ArrayLike) -> np.ndarray:
        """
        Apply histogram binning calibration.

        Args:
            y_prob: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        self._check_is_fitted()
        bin_edges = self.bin_edges_
        bin_values = self.bin_values_
        # fit() always assigns both arrays before setting is_fitted=True, so
        # _check_is_fitted() guarantees they are non-None here. Bound as locals
        # rather than narrowed with `assert`, which `python -O` strips.
        if bin_edges is None or bin_values is None:  # pragma: no cover
            raise RuntimeError(f"{self.__class__.__name__} has no fitted bins.")
        y_prob = coerce_to_array(y_prob, "y_prob")
        # BGL-D (2026-09-16). np.digitize puts a NaN in the LAST bin, so an
        # unscoreable row came back as that bin's empirical rate: measured,
        # transform(np.full(5, np.nan)) -> [0.96666667] x5, byte-identical to
        # transform([0.95]), i.e. the highest-risk calibrated probability the
        # fitted object can produce, handed to a row that has no score at all.
        # Out-of-range scores (1.7, -3, 42) were clamped to the end bins the
        # same way. The library already refuses exactly this input one method
        # earlier (HistogramBinning.fit and PlattScaling.transform both call
        # validate_probabilities), so the inconsistency was internal to this
        # file; state the refusal the same way here.
        validate_probabilities(y_prob, "y_prob")

        # A collapsed grid (see fit) leaves zero bins. np.clip(..., 0, -1) then
        # produces index -1 and numpy raised "index -1 is out of bounds for axis
        # 0 with size 0" at use time. fit() states the refusal now; this returns
        # it as a value, so a caller that never reads fit_result still gets NaN
        # rather than a crash or, worse, a number.
        if len(bin_values) == 0:
            return np.full(y_prob.shape, np.nan, dtype=float)

        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
        bin_indices = np.clip(bin_indices, 0, len(bin_values) - 1)

        return bin_values[bin_indices]


def create_calibrator(method: Union[str, CalibrationMethod], **kwargs) -> BaseCalibrator:
    """
    Factory function to create calibrators by name.

    Args:
        method: Calibration method name or enum
        **kwargs: Method-specific parameters

    Returns:
        Instantiated calibrator

    Example:
        >>> calibrator = create_calibrator('isotonic')
        >>> calibrator = create_calibrator(CalibrationMethod.PLATT, regularization=0.1)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: create_calibrator. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    if isinstance(method, CalibrationMethod):
        method = method.value

    method_lower = method.lower()

    calibrators = {
        "platt": PlattScaling,
        "isotonic": IsotonicCalibrator,
        "beta": BetaCalibrator,
        "temperature": TemperatureScaling,
        "histogram": HistogramBinning,
    }

    if method_lower not in calibrators:
        available = list(calibrators.keys())
        raise ValueError(f"Unknown calibration method: {method}. Available methods: {available}")

    return calibrators[method_lower](**kwargs)
