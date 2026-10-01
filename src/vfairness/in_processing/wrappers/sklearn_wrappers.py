"""
Scikit-Learn Compatible Fairness Wrappers.

This module provides scikit-learn compatible wrapper classes that integrate
fairness-aware training into standard ML workflows. These wrappers work
with any scikit-learn estimator and add fairness constraints or regularization.

Wrappers Implemented:
    1. FairClassifier: Fairness-aware classifier wrapper
    2. FairRegressor: Fairness-aware regressor wrapper
    3. FairPipeline: Pipeline with fairness integration
    4. FairGridSearchCV: Grid search with fairness metrics

Key Features:
    - Compatible with scikit-learn API (fit, predict, score)
    - Works with any base estimator
    - Multiple fairness constraint types
    - Hyperparameter search with fairness objectives

Example:
    >>> from sklearn.linear_model import LogisticRegression
    >>> from vfairness.in_processing.wrappers import FairClassifier
    >>>
    >>> clf = FairClassifier(
    ...     base_estimator=LogisticRegression(),
    ...     fairness_constraint='demographic_parity',
    ...     constraint_weight=0.1,
    ... )
    >>> clf.fit(X_train, y_train, sensitive_attr=gender)
    >>> y_pred = clf.predict(X_test)

References:
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
    - Scikit-learn API: https://scikit-learn.org/stable/developers/develop.html
"""

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from vfairness.evaluation.vfairness_metrics._grouping import GroupManager
from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
    check_consistent_length,
    coerce_to_array,
)

from ..constraints import (
    BaseFairnessConstraint,
    ExponentiatedGradient,
    FairnessConstraintType,
    GridSearch,
    ThresholdOptimizer,
    create_constraint,
)
from ..constraints.base import _reported_constraint
from ..constraints.reductions import _refuse_unscored_rows


@dataclass
class FairClassifierResult:
    """
    Result from FairClassifier fitting.

    THREE STATES, NEVER TWO (R4C, 2026-09-10). All three fit paths copied
    ``ConstraintViolation.is_satisfied`` straight into ``constraint_satisfied``,
    and that flag is deliberately False for a constraint that could not be
    EVALUATED (fail-closed, because the same object steers the reduction). Every
    reporting consumer read the copy as a measured verdict. Measured with
    ``method='reductions'`` on 240 rows whose third group had no positive
    labels: ``accuracy=0.8375, fairness_violation=nan,
    constraint_satisfied=False``, carried into ``MethodComparison`` and drawn as
    a red FAIL row with a violation bar labelled "nan".

        fairness_violation = 0.04, constraint_satisfied = True    met
        fairness_violation = 0.31, constraint_satisfied = False   measured breach
        fairness_violation = None, constraint_satisfied = None    NOT EVALUATED

    Attributes:
        accuracy: Classification accuracy
        fairness_violation: Final fairness constraint violation magnitude, or
            None when the constraint could not be evaluated. Never NaN, never a
            substituted 0.0.
        constraint_satisfied: True when the constraint was met, False for a
            MEASURED breach, None when it was never evaluated. Pair it with
            ``fairness_metrics['insufficient_data']``, which names the state
            the same way ``ExponentiatedGradient`` already reports it.
        fairness_metrics: Dictionary of fairness metrics
        training_history: Training history if available
    """

    accuracy: float
    fairness_violation: Optional[float]
    constraint_satisfied: Optional[bool]
    # Heterogeneous per-fit metrics: e.g. 'constraint_type' (str),
    # 'tolerance' (float), per-group violations (float), 'thresholds' (dict).
    fairness_metrics: Dict[str, Any] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "fairness_violation": self.fairness_violation,
            "constraint_satisfied": self.constraint_satisfied,
            "fairness_metrics": self.fairness_metrics,
            "training_history": self.training_history,
        }


class FairClassifier:
    """
    Fairness-Aware Classifier Wrapper.

    A scikit-learn compatible wrapper that adds fairness constraints to
    any base classifier. Uses the exponentiated gradient algorithm to
    train an ensemble of classifiers that satisfies fairness constraints.

    Args:
        base_estimator: Base scikit-learn classifier
        fairness_constraint: Type of fairness constraint to enforce
        tolerance: Maximum allowed constraint violation
        constraint_weight: Weight for constraint in optimization
        method: Fairness method ('reductions', 'threshold', 'grid_search')
        max_iterations: Maximum iterations for optimization
        random_state: Random seed for reproducibility
        verbose: Whether to print progress

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> clf = FairClassifier(
        ...     base_estimator=LogisticRegression(),
        ...     fairness_constraint='demographic_parity',
        ...     tolerance=0.05
        ... )
        >>> clf.fit(X_train, y_train, sensitive_attr=gender)
        >>> y_pred = clf.predict(X_test)
        >>> print(f"Fairness satisfied: {clf.fairness_result_.constraint_satisfied}")

    Comparison with Fairlearn:
        Similar to Fairlearn's ExponentiatedGradient and ThresholdOptimizer,
        but integrated with vfairness metrics and reporting.

    Attributes:
        fairness_result_: FairClassifierResult from last fit
        constraint_: The constraint object used
        is_fitted_: Whether the classifier is fitted

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: fair_classifier. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        base_estimator: Any,
        fairness_constraint: Union[
            str, FairnessConstraintType, BaseFairnessConstraint
        ] = "demographic_parity",
        tolerance: float = 0.05,
        constraint_weight: float = 0.5,
        method: Literal["reductions", "threshold", "grid_search"] = "reductions",
        max_iterations: int = 50,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.fairness_constraint = fairness_constraint
        self.tolerance = tolerance
        self.constraint_weight = constraint_weight
        self.method = method
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.is_fitted_ = False
        self.fairness_result_: Optional[FairClassifierResult] = None
        self.constraint_: Optional[BaseFairnessConstraint] = None
        # Holds a heterogeneous fitted estimator (ExponentiatedGradient,
        # GridSearch, or a bare cloned base estimator) depending on `method`.
        self._inner_model: Any = None
        self._threshold_optimizer: Optional[ThresholdOptimizer] = None

    def _setup_constraint(self) -> BaseFairnessConstraint:
        """Setup the constraint object."""
        if isinstance(self.fairness_constraint, BaseFairnessConstraint):
            return self.fairness_constraint
        return create_constraint(self.fairness_constraint, tolerance=self.tolerance)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "FairClassifier":
        """
        Fit the fair classifier.

        Args:
            X: Feature matrix
            y: Target labels
            sensitive_attr: Sensitive attribute for fairness constraint
            sample_weight: Optional sample weights

        Returns:
            self
        """
        X = coerce_to_array(X, "X")
        y = coerce_to_array(y, "y")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")
        check_consistent_length(X, y, sensitive_attr)

        self.constraint_ = self._setup_constraint()
        self._sensitive_attr_train = sensitive_attr  # Store for threshold optimizer

        if self.method == "reductions":
            result = self._fit_reductions(X, y, sensitive_attr, sample_weight)
        elif self.method == "threshold":
            result = self._fit_threshold(X, y, sensitive_attr)
        elif self.method == "grid_search":
            result = self._fit_grid_search(X, y, sensitive_attr)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.fairness_result_ = result
        self.is_fitted_ = True

        return self

    def _fit_reductions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attr: np.ndarray,
        sample_weight: Optional[ArrayLike],
    ) -> FairClassifierResult:
        """Fit using exponentiated gradient (reductions approach)."""
        # fit() sets self.constraint_ via _setup_constraint() before dispatching
        # to this helper, so it is never None here.
        assert self.constraint_ is not None
        eg = ExponentiatedGradient(
            base_estimator=deepcopy(self.base_estimator),
            constraint=self.constraint_,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )

        reduction_result = eg.fit(
            X,
            y,
            sensitive_attr=sensitive_attr,
            sample_weight=sample_weight,
        )

        self._inner_model = eg

        y_pred = eg.predict(X)
        accuracy = float(np.mean(y_pred == y))
        violation = self.constraint_.compute_violation(y_pred, y, sensitive_attr)

        # R4C. The pair is translated once, here: a constraint that could not be
        # evaluated leaves this method as None/None rather than wearing the
        # fail-closed False that steers the reduction.
        reported_violation, reported_satisfied = _reported_constraint(violation)

        return FairClassifierResult(
            accuracy=accuracy,
            fairness_violation=reported_violation,
            constraint_satisfied=reported_satisfied,
            fairness_metrics={
                "constraint_type": violation.constraint_type,
                "tolerance": violation.tolerance,
                **violation.group_violations,
                # After the per-group spread deliberately: this flag is the
                # third state and must not be shadowed by a group that happens
                # to share its name. ExponentiatedGradient reports the same key.
                "insufficient_data": violation.could_not_evaluate,
            },
            training_history={
                "loss": reduction_result.optimization_result.loss_history,
                "violation": reduction_result.optimization_result.violation_history,
            },
        )

    def _fit_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> FairClassifierResult:
        """Fit using post-processing threshold optimization."""
        # fit() sets self.constraint_ before dispatching here; never None.
        assert self.constraint_ is not None
        base_clf = deepcopy(self.base_estimator)
        base_clf.fit(X, y)

        if hasattr(base_clf, "predict_proba"):
            y_prob = base_clf.predict_proba(X)[:, 1]
        else:
            # Fall back to decision function
            y_prob = base_clf.decision_function(X)
            y_prob = 1 / (1 + np.exp(-y_prob))  # Sigmoid

        self._threshold_optimizer = ThresholdOptimizer(
            constraint=self.constraint_,
            grid_size=100,
        )
        self._threshold_optimizer.fit(y_prob, y, sensitive_attr=sensitive_attr)

        self._inner_model = base_clf

        y_pred = self._threshold_optimizer.predict(y_prob, sensitive_attr)
        accuracy = float(np.mean(y_pred == y))
        violation = self.constraint_.compute_violation(y_pred, y, sensitive_attr)

        # R4C, as in _fit_reductions: three states, never two.
        reported_violation, reported_satisfied = _reported_constraint(violation)

        return FairClassifierResult(
            accuracy=accuracy,
            fairness_violation=reported_violation,
            constraint_satisfied=reported_satisfied,
            fairness_metrics={
                "constraint_type": violation.constraint_type,
                "thresholds": self._threshold_optimizer.get_thresholds(),
                "insufficient_data": violation.could_not_evaluate,
            },
        )

    def _fit_grid_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attr: np.ndarray,
    ) -> FairClassifierResult:
        """Fit using grid search over constraint weights."""
        # fit() sets self.constraint_ before dispatching here; never None.
        assert self.constraint_ is not None
        gs = GridSearch(
            base_estimator=deepcopy(self.base_estimator),
            constraint=self.constraint_,
            n_lambda_values=20,
            verbose=self.verbose,
        )

        gs.fit(X, y, sensitive_attr=sensitive_attr)

        self._inner_model = gs

        y_pred = gs.predict(X)
        accuracy = float(np.mean(y_pred == y))
        violation = self.constraint_.compute_violation(y_pred, y, sensitive_attr)

        # R4C, as in _fit_reductions: three states, never two.
        reported_violation, reported_satisfied = _reported_constraint(violation)

        return FairClassifierResult(
            accuracy=accuracy,
            fairness_violation=reported_violation,
            constraint_satisfied=reported_satisfied,
            fairness_metrics={
                "constraint_type": violation.constraint_type,
                "insufficient_data": violation.could_not_evaluate,
            },
        )

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Make predictions.

        With ``method='threshold'`` the fitted group thresholds cannot be
        applied without the sensitive attribute, so this returns the uniform
        0.5 answer and warns; use :meth:`predict_with_sensitive_attr` for the
        mitigated one.

        A row the base estimator gave NO score to is REFUSED rather than
        decided (BGL3, 2026-09-27). ``y_prob >= 0.5`` is False for NaN under
        IEEE 754, so such a row was written 0, a confident REJECTION nobody
        made, and the returned int array has no value that could have said so.
        Measured on 120 rows, ``method='threshold'``, a base estimator whose
        ``predict_proba`` returned NaN for rows 0, 1 and 61::

            clf.predict(X)                        -> rows 0, 1, 61 == 0, 0, 0
                                                     one warning, about
                                                     sensitive_attr, nothing
                                                     about the missing scores
            clf.predict_with_sensitive_attr(X, g) -> ValueError naming
                                                     "3 of 120 row(s) have no
                                                     usable score ... first at
                                                     index [0, 1, 61]"

        One class, two predict methods, opposite answers to the same row. The
        refusal is :func:`..constraints.reductions._refuse_unscored_rows`, the
        same function the sibling reaches through ``ThresholdOptimizer.predict``,
        imported rather than restated so the two cannot drift apart.

        Args:
            X: Feature matrix

        Returns:
            Predicted labels

        Raises:
            ValueError: With ``method='threshold'``, when the base estimator
                produced a non-finite score for any row.
        """
        self._check_is_fitted()
        X = coerce_to_array(X, "X")

        if self.method == "threshold":
            if hasattr(self._inner_model, "predict_proba"):
                y_prob = self._inner_model.predict_proba(X)[:, 1]
            else:
                y_prob = self._inner_model.decision_function(X)
                y_prob = 1 / (1 + np.exp(-y_prob))
            # Above the threshold comparison, and above the warning: a caller
            # who has to fix the input should not first be told about a
            # different, survivable gap.
            _refuse_unscored_rows(y_prob, caller=f"FairClassifier(method='{self.method}')")
            # For threshold method, we need sensitive_attr - use uniform if not provided
            warnings.warn(
                "Threshold method requires sensitive_attr for predict. "
                "Using default threshold of 0.5 for all groups."
            )
            return (y_prob >= 0.5).astype(int)
        else:
            return self._inner_model.predict(X)

    def predict_with_sensitive_attr(
        self,
        X: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Make predictions with sensitive attribute (for threshold method).

        Args:
            X: Feature matrix
            sensitive_attr: Sensitive attribute

        Returns:
            Predicted labels
        """
        self._check_is_fitted()
        X = coerce_to_array(X, "X")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        if self.method == "threshold" and self._threshold_optimizer is not None:
            if hasattr(self._inner_model, "predict_proba"):
                y_prob = self._inner_model.predict_proba(X)[:, 1]
            else:
                y_prob = self._inner_model.decision_function(X)
                y_prob = 1 / (1 + np.exp(-y_prob))
            return self._threshold_optimizer.predict(y_prob, sensitive_attr)
        else:
            return self._inner_model.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities, always shaped ``(n_samples, 2)``.

        THREE STATES, NEVER TWO (BGL g009, 2026-09-17). A base estimator that
        cannot produce a probability has no probability to report, and this
        method used to manufacture one: ``np.column_stack([1 - predict(X),
        predict(X)])`` turned each HARD LABEL into a certainty of exactly 1.0.
        Measured on 60 rows, ``method='grid_search'`` with LogisticRegression
        (the inner ``GridSearch`` ensemble exposes no ``predict_proba``):
        ``np.unique(clf.predict_proba(X))`` was ``[0, 1]``, dtype int64, two
        distinct scores over sixty rows, and NOT ONE WARNING. Fed straight to
        ``roc_auc_score(y, proba[:, 1])`` that reads 0.7833, which is the
        accuracy of the hard labels wearing the name of an AUC, and any
        calibration or score-based parity metric over it sees a model that is
        never uncertain. ``method='threshold'`` with a ``LinearSVC`` base (a
        decision function, no probability) produced the same ``[0, 1]``.

            a real probability      0.0 <= p <= 1.0, from the base estimator
            no probability at all   nan, with a UserWarning naming the reason

        There is no third shape either. ``ExponentiatedGradient`` returns the
        weighted-ensemble probability of the POSITIVE class as a 1-D array, so
        ``method='reductions'`` returned ``(n,)`` while the other two methods
        returned ``(n, 2)``, and ``clf.predict_proba(X)[:, 1]`` raised
        ``IndexError`` on the default method alone. The 1-D case is widened
        here into the two columns ``[1 - p, p]``.

        Args:
            X: Feature matrix

        Returns:
            ``(n_samples, 2)`` of float; column 1 is the positive class. Every
            entry is ``nan`` when the base estimator produces no probability:
            that is the absence of a score, not a score of zero. Use
            :meth:`predict` for labels, since ``argmax`` over a row of NaN
            silently answers 0.
        """
        self._check_is_fitted()
        X = coerce_to_array(X, "X")
        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

        if hasattr(self._inner_model, "predict_proba"):
            proba = np.asarray(self._inner_model.predict_proba(X), dtype=float)
            if proba.ndim == 1:
                return np.column_stack([1.0 - proba, proba])
            return proba

        warnings.warn(
            f"FairClassifier(method='{self.method}'): the fitted "
            f"{type(self._inner_model).__name__} exposes no predict_proba, so NO "
            "probability was estimated for these rows and every entry is nan. This "
            "used to return the hard labels as a column_stack, i.e. a confidence of "
            "exactly 1.0 for whichever class was predicted, which is a certainty "
            "nobody measured and which reads as a perfectly calibrated model to any "
            "AUC, calibration or score-based parity metric. Use predict() for labels, "
            "or give the wrapper a base estimator with predict_proba.",
            stacklevel=2,
        )
        return np.full((n_samples, 2), np.nan, dtype=float)

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sensitive_attr: Optional[ArrayLike] = None,
        metric: Literal["accuracy", "fairness", "combined"] = "accuracy",
    ) -> float:
        """
        Score the classifier.

        Args:
            X: Feature matrix
            y: True labels
            sensitive_attr: Sensitive attribute (for fairness metrics)
            metric: Which metric to return

        Returns:
            Score value. For 'fairness' and 'combined' this is NaN when the
            constraint could not be evaluated, which is NOT a score of 0 and not
            a bad score: it is the absence of one, and it is warned about rather
            than returned in silence. It is NaN for the same reason when
            ``sensitive_attr`` is None, since a disparity has nothing to be
            measured across; that call used to come back as the accuracy.
            Callers that rank models (a grid search, a leaderboard) must drop
            NaN rather than order against it, since every comparison with NaN is
            False and an unmeasured model then wins or loses by accident.
        """
        self._check_is_fitted()

        X = coerce_to_array(X, "X")
        y = coerce_to_array(y, "y")

        if sensitive_attr is not None:
            y_pred = self.predict_with_sensitive_attr(X, sensitive_attr)
        else:
            y_pred = self.predict(X)

        accuracy = float(np.mean(y_pred == y))

        if metric == "accuracy":
            return accuracy

        if sensitive_attr is None:
            # BGL g009 (2026-09-17). This returned the ACCURACY, under the name
            # the caller asked for. Measured on 60 rows with a real 0.10
            # demographic-parity gap: score(X, y, metric='fairness') answered
            # -0.1000 with sensitive_attr and 0.7833 without it, so the
            # unmeasured call came back POSITIVE where every measured fairness
            # score is <= 0, and outranked every honestly measured model on the
            # same leaderboard. metric='combined' is worse because it is
            # arithmetically indistinguishable from a measurement: accuracy
            # minus a violation of 0.0 that nobody computed.
            warnings.warn(
                f"FairClassifier.score(metric='{metric}') needs sensitive_attr to measure a "
                "disparity and none was given, so nothing was measured and the result is "
                "NaN: not a poor score, no score. It used to return the accuracy, which "
                "for metric='fairness' is a positive number on a scale whose measured "
                "values are all <= 0, and for metric='combined' is accuracy minus a "
                "violation of 0.0 nobody computed. Pass sensitive_attr, or ask for "
                "metric='accuracy' if accuracy is what you want.",
                stacklevel=2,
            )
            return float("nan")

        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")
        # _check_is_fitted() above guarantees fit() ran, which always assigns
        # self.constraint_ via _setup_constraint(); never None here.
        assert self.constraint_ is not None
        violation = self.constraint_.compute_violation(y_pred, y, sensitive_attr)

        # R4C. A score is a number a caller will rank models by, and there is no
        # honest number for a constraint nobody could evaluate: 0.0 is the BEST
        # possible fairness score here, and -0.0 combined with accuracy would
        # hand an unmeasured model the top of the leaderboard. NaN stays, since
        # it cannot be mistaken for a good score, but it no longer travels
        # silently: `score()` returns a bare float, so the warning is the only
        # place this method can say which of the three states it is in.
        if violation.could_not_evaluate:
            warnings.warn(
                f"The {violation.constraint_type} constraint could not be evaluated on "
                "this data (a group rate with an empty denominator makes the disparity "
                f"unmeasurable), so score(metric='{metric}') is NaN: not a poor score, "
                "no score. Drop it rather than ranking against it.",
                stacklevel=2,
            )

        if metric == "fairness":
            # Return negative violation (higher is better)
            return -violation.overall_violation

        # Combined: accuracy - violation
        return accuracy - violation.overall_violation

    def _check_is_fitted(self):
        """Check if the classifier is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("FairClassifier is not fitted. Call fit() first.")

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator (sklearn compatibility)."""
        params = {
            "base_estimator": self.base_estimator,
            "fairness_constraint": self.fairness_constraint,
            "tolerance": self.tolerance,
            "constraint_weight": self.constraint_weight,
            "method": self.method,
            "max_iterations": self.max_iterations,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f"base_estimator__{k}": v for k, v in base_params.items()})
        return params

    def set_params(self, **params) -> "FairClassifier":
        """Set parameters for this estimator (sklearn compatibility).

        An unknown name RAISES (BGL g009, 2026-09-17). The ``elif hasattr(self,
        key)`` test silently dropped every misspelling and every parameter this
        wrapper does not have: measured, ``clf.set_params(tolerence=99)``
        returned self with ``tolerance`` still 0.05 and no warning anywhere. A
        hyperparameter sweep over such a name fits, scores, compares and ranks
        N IDENTICAL models and then reports a winner. The same test also let a
        caller assign fitted state (``is_fitted_``, ``fairness_result_``),
        which is not a parameter, so the whitelist is the constructor's own
        parameter list as ``get_params`` reports it.
        """
        valid = set(self.get_params(deep=False))
        unknown = [k for k in params if not k.startswith("base_estimator__") and k not in valid]
        if unknown:
            raise ValueError(
                f"FairClassifier.set_params: unknown parameter(s) {sorted(unknown)!r}. "
                f"Valid parameters are {sorted(valid)!r}, plus 'base_estimator__<name>' "
                f"for the base estimator's own parameters."
            )

        base_params = {}
        for key, value in params.items():
            if key.startswith("base_estimator__"):
                base_params[key[len("base_estimator__") :]] = value
            else:
                setattr(self, key, value)

        if base_params:
            if not hasattr(self.base_estimator, "set_params"):
                raise ValueError(
                    f"FairClassifier.set_params: base_estimator "
                    f"{type(self.base_estimator).__name__} has no set_params, so "
                    f"{sorted(base_params)!r} cannot be applied."
                )
            self.base_estimator.set_params(**base_params)

        return self


class FairRegressor:
    """
    Fairness-Aware Regressor Wrapper.

    A scikit-learn compatible wrapper that adds fairness constraints to
    regression tasks. Currently implements **mean parity** via post-fit
    per-group prediction offsets: after fitting the base regressor, each
    group's mean prediction (on the training data) is shifted into a band
    of half-width ``tolerance / 2`` around the pooled mean prediction, so
    the gap between any two groups' mean predictions is at most
    ``tolerance``. Applying the offsets at predict time requires the
    sensitive attribute: use :meth:`predict_with_sensitive_attr`. Plain
    :meth:`predict` returns the UNADJUSTED base predictions and warns when
    offsets were fitted, because it cannot know the group memberships.

    'error_parity' and 'bounded_loss' are NOT accepted. They are not
    implemented and raise ``NotImplementedError`` at ``fit()`` time rather
    than silently fitting an unconstrained model, so they are no longer
    named in the ``fairness_constraint`` annotation either: the ``Literal``
    lists what ``fit()`` honours and nothing else, because an annotation is
    a claim that an editor completes and a type checker blesses.

    The former ``method='reweighting'`` (uniform per-group sample
    reweighting) was removed: for any positive group-uniform weights the
    weighted-least-squares group means equal the data group means, so it
    provably could not move the quantity it claimed to constrain, and its
    weight factor went negative (sklearn ValueError) for tolerance > 1.
    Requesting it now raises ``ValueError``. Default method changed from
    'reweighting' to 'offset'. ``method='constrained'`` is likewise not
    implemented and raises ``NotImplementedError``. Neither value is in the
    ``method`` annotation any more, for the same reason.

    Args:
        base_estimator: Base scikit-learn regressor
        fairness_constraint: Type of fairness constraint ('mean_parity' is
            the only one implemented; anything else raises at fit() time)
        tolerance: Maximum allowed gap between any two groups' mean
            predictions (in target units)
        method: Fairness method ('offset')
        verbose: Whether to print progress
        on_unseen_group: What :meth:`predict_with_sensitive_attr` does with a
            row whose group ``fit`` never saw, and for which therefore NO
            offset was measured. ``'raise'`` (default) refuses, naming the
            groups and their row counts; ``'nan'`` returns ``nan`` for those
            rows, warns, and lists them in ``unseen_groups_``. There is no
            mode that returns them as though they had been adjusted.

    Example:
        >>> from sklearn.linear_model import Ridge
        >>> reg = FairRegressor(
        ...     base_estimator=Ridge(),
        ...     fairness_constraint='mean_parity',
        ...     tolerance=0.1
        ... )
        >>> reg.fit(X_train, y_train, sensitive_attr=group)
        >>> y_pred = reg.predict_with_sensitive_attr(X_test, group_test)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. This does NOT establish that
    the pin has been sabotage-checked, so it is not known whether the pin can fail
    at all. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: fair_regressor. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        base_estimator: Any,
        fairness_constraint: Literal["mean_parity"] = "mean_parity",
        tolerance: float = 0.1,
        method: Literal["offset"] = "offset",
        verbose: bool = False,
        on_unseen_group: Literal["raise", "nan"] = "raise",
    ):
        self.base_estimator = base_estimator
        self.fairness_constraint = fairness_constraint
        self.tolerance = tolerance
        self.method = method
        self.verbose = verbose
        if on_unseen_group not in ("raise", "nan"):
            raise ValueError(
                f"on_unseen_group must be 'raise' or 'nan', got {on_unseen_group!r}. "
                f"There is no mode that reports an unadjusted row as adjusted."
            )
        self.on_unseen_group = on_unseen_group

        self.is_fitted_ = False
        # Holds the fitted (deep-copied) base regressor once fit() has run.
        self._inner_model: Any = None
        # Per-group additive prediction offsets fitted for mean_parity.
        self.group_offsets_: Dict[str, float] = {}
        # BGL-S2 (2026-09-16): the THIRD state, distinct from "seen, and its
        # measured offset happened to be 0.0". {group: n_rows} for the groups
        # the LAST predict_with_sensitive_attr call could not adjust.
        self.unseen_groups_: Dict[str, int] = {}

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sensitive_attr: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> "FairRegressor":
        """
        Fit the fair regressor.

        Args:
            X: Feature matrix
            y: Target values
            sensitive_attr: Sensitive attribute
            sample_weight: Optional sample weights. Passed through to the base
                regressor's ``fit``. A base estimator that takes no
                ``sample_weight`` is fitted UNWEIGHTED and says so in a warning:
                the mean_parity offsets are measured off whatever model was
                actually fitted, so silently dropping the weights would report
                offsets for a model the caller did not ask for.

        Returns:
            self
        """
        X = coerce_to_array(X, "X")
        y = coerce_to_array(y, "y")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")
        check_consistent_length(X, y, sensitive_attr)

        # Fail loudly on constraints that are not implemented. The previous
        # behaviour silently fitted an UNCONSTRAINED model for any constraint
        # other than 'mean_parity', letting callers believe fairness was
        # applied when nothing happened.
        if self.fairness_constraint != "mean_parity":
            if self.fairness_constraint in ("error_parity", "bounded_loss"):
                raise NotImplementedError(
                    f"fairness_constraint='{self.fairness_constraint}' is not "
                    f"implemented in FairRegressor yet; only 'mean_parity' is. "
                    f"Refusing to silently fit an unconstrained model."
                )
            raise ValueError(
                f"Unknown fairness_constraint: {self.fairness_constraint!r}. "
                f"Supported: 'mean_parity'."
            )

        if self.method == "offset":
            # Coerce any array-like weights to ndarray so a list/Series input
            # doesn't crash in the base fit. ndarray input is unchanged.
            sample_weight_arr = (
                coerce_to_array(sample_weight, "sample_weight")
                if sample_weight is not None
                else None
            )
            self._fit_offset(X, y, sensitive_attr, sample_weight_arr)
        elif self.method == "reweighting":
            # The old reweighting intervention multiplied every sample of a
            # group by the SAME factor. For any positive group-uniform
            # weights the weighted-least-squares normal equation forces the
            # fitted group means to equal the data group means, so the
            # intervention provably could not move the group prediction
            # means it claimed to constrain, and its factor
            # 1 - sign(diff) * min(|diff|, tolerance) went NEGATIVE for
            # |diff| > 1 and tolerance > 1, crashing sklearn's sample_weight
            # validation. It was removed rather than fixed; refuse loudly.
            raise ValueError(
                "method='reweighting' was removed from FairRegressor: uniform "
                "per-group sample reweighting cannot move group prediction "
                "means, so it never enforced mean_parity. Use the default "
                "method='offset'."
            )
        elif self.method == "constrained":
            raise NotImplementedError(
                "method='constrained' is not implemented in FairRegressor yet; use method='offset'."
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True
        return self

    def _fit_offset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_attr: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ):
        """Fit the base regressor, then fit per-group mean-parity offsets.

        Each group's training mean prediction is shifted into the band
        [pooled_mean - tolerance/2, pooled_mean + tolerance/2], so the gap
        between any two groups' adjusted mean predictions is at most
        `tolerance`. Groups already inside the band get a zero offset (no
        overcorrection).
        """
        gm = GroupManager(sensitive_attr)

        self._inner_model = deepcopy(self.base_estimator)
        try:
            self._inner_model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            # A WEIGHTED FIT THAT NEVER HAPPENED IS NOT AN UNWEIGHTED FIT
            # (BGL3, 2026-09-27). This retry exists because many estimators take
            # no `sample_weight` keyword at all, and with sample_weight=None
            # dropping it costs nothing. With weights it silently threw the
            # caller's input away and then measured the mean-parity offsets off
            # the wrong model, and `fit` returned self as though nothing had
            # happened. Measured on 120 rows, two groups of 60, weights 5.0 on
            # group a and 1.0 on b, tolerance=0.1:
            #
            #   LinearRegression (accepts sample_weight)
            #       no weights   -> {'a':  0.26505074, 'b': -0.26505074}
            #       with weights -> {'a':  0.28920397, 'b': -0.28920397}
            #   KNeighborsRegressor (fit signature is (self, X, y))
            #       no weights   -> {'a': -0.11278072, 'b':  0.11278072}
            #       with weights -> {'a': -0.11278072, 'b':  0.11278072}
            #                       BYTE-IDENTICAL, and zero warnings
            #
            # So the weights demonstrably move the offsets where they are
            # honoured, and where they are not the caller was told nothing.
            if sample_weight is not None:
                warnings.warn(
                    f"FairRegressor.fit: {type(self._inner_model).__name__}.fit accepts no "
                    f"sample_weight, so the {len(sample_weight)} weight(s) passed were NOT "
                    f"applied and the base regressor was fitted UNWEIGHTED. The mean_parity "
                    f"offsets in group_offsets_ were then measured off that unweighted "
                    f"model, so they are not the offsets the weighting asked for. Use a "
                    f"base estimator whose fit takes sample_weight, or resample instead of "
                    f"weighting.",
                    UserWarning,
                    stacklevel=3,
                )
            self._inner_model.fit(X, y)

        y_hat = np.asarray(self._inner_model.predict(X), dtype=float)
        pooled_mean = float(np.mean(y_hat))
        half_band = float(self.tolerance) / 2.0

        offsets: Dict[str, float] = {}
        for group in gm.groups:
            mask = gm.get_mask(group)
            group_mean = float(np.mean(y_hat[mask]))
            diff = group_mean - pooled_mean
            target = pooled_mean + float(np.clip(diff, -half_band, half_band))
            offsets[str(group)] = target - group_mean
            if self.verbose:
                print(
                    f"FairRegressor mean_parity: group {group!r} mean "
                    f"{group_mean:.4f} -> {target:.4f} (offset {target - group_mean:+.4f})"
                )

        self.group_offsets_ = offsets

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Make predictions WITHOUT the fairness adjustment.

        The mean-parity offsets are per group and this method has no group
        information, so it returns the unadjusted base predictions and warns
        when nonzero offsets were fitted. Use
        :meth:`predict_with_sensitive_attr` to get the fairness-adjusted
        predictions.

        Args:
            X: Feature matrix

        Returns:
            Unadjusted predicted values
        """
        if not self.is_fitted_:
            raise RuntimeError("FairRegressor is not fitted. Call fit() first.")

        X = coerce_to_array(X, "X")
        if any(abs(v) > 1e-12 for v in self.group_offsets_.values()):
            warnings.warn(
                "FairRegressor.predict() cannot apply the fitted mean_parity "
                "group offsets without the sensitive attribute; returning "
                "UNADJUSTED base predictions. Use "
                "predict_with_sensitive_attr(X, sensitive_attr) for the "
                "fairness-adjusted predictions."
            )
        return self._inner_model.predict(X)

    def predict_with_sensitive_attr(
        self,
        X: ArrayLike,
        sensitive_attr: ArrayLike,
    ) -> np.ndarray:
        """
        Make predictions WITH the fitted mean-parity group offsets applied.

        A group that ``fit`` never saw has NO fitted offset, which is a third
        state: not "seen, and its measured offset was 0.0". ``on_unseen_group``
        decides what happens to those rows; neither setting returns them as
        though they had been adjusted. Either way the groups and their row
        counts are recorded on ``self.unseen_groups_``.

        Args:
            X: Feature matrix
            sensitive_attr: Sensitive attribute (group membership per row)

        Returns:
            Fairness-adjusted predicted values. With ``on_unseen_group='nan'``
            the rows of an unfitted group are ``nan``: no mean-parity offset
            was measured for them, so no adjusted prediction exists. Call
            :meth:`predict` if the UNADJUSTED base prediction is what you want.

        Raises:
            ValueError: With the default ``on_unseen_group='raise'``, when any
                row belongs to a group ``fit`` never saw.
        """
        if not self.is_fitted_:
            raise RuntimeError("FairRegressor is not fitted. Call fit() first.")

        X = coerce_to_array(X, "X")
        sensitive_attr = coerce_to_array(sensitive_attr, "sensitive_attr")

        y_hat = np.asarray(self._inner_model.predict(X), dtype=float)
        gm = GroupManager(sensitive_attr)

        # BGL-S2 (2026-09-16): this read `self.group_offsets_.get(str(group),
        # 0.0)`. 0.0 is exactly the offset that means "this group's mean
        # prediction is already inside the parity band and needs no
        # correction", so a group fit had NEVER SEEN was returned in the same
        # array as the corrected rows, documented as "Fairness-adjusted
        # predicted values", with no warning. Measured 2026-09-16: fitted on
        # a and b (offsets +0.1259 / -0.1259), predicted on ['a','b','c','c',
        # 'b','a'] -> per-row deltas [0.1259, -0.1259, 0.0, 0.0, -0.1259,
        # 0.1259] and ZERO warnings; against a control where a and b had
        # GENUINELY measured zero offsets the two outputs were byte-identical,
        # so no caller could separate "measured, no correction needed" from
        # "never treated". Downstream the untreated group's mean gap then
        # reads as "the mitigation failed for c" rather than "c was never
        # mitigated". ThresholdOptimizer._apply_group_thresholds and
        # reductions.py already refuse this exact input; this is the same
        # refusal, on the surface that returns floats and so can also carry it
        # as nan.
        unseen: Dict[str, int] = {}
        for group in gm.groups:
            key = str(group)
            mask = gm.get_mask(group)
            if key not in self.group_offsets_:
                unseen[key] = int(np.count_nonzero(mask))
                continue
            offset = self.group_offsets_[key]
            if offset != 0.0:
                y_hat[mask] += offset
        self.unseen_groups_ = unseen

        if unseen:
            listing = ", ".join(f"{g!r} ({n} row(s))" for g, n in sorted(unseen.items()))
            fitted = sorted(self.group_offsets_)
            if self.on_unseen_group == "raise":
                raise ValueError(
                    f"FairRegressor.predict_with_sensitive_attr: no mean-parity offset was "
                    f"fitted for group(s) {listing}. Fitted groups: {fitted!r}. fit() never "
                    f"saw them, so nothing was measured about their mean prediction; "
                    f"returning them unadjusted would report an untreated group as a treated "
                    f"one. Refit including these groups, drop their rows, or pass "
                    f"on_unseen_group='nan' to get nan for those rows."
                )
            warnings.warn(
                f"FairRegressor.predict_with_sensitive_attr: no mean-parity offset was fitted "
                f"for group(s) {listing}; fitted groups are {fitted!r}. Those rows are "
                f"returned as nan, NOT as an adjusted prediction, and are listed in "
                f"unseen_groups_. Use predict() for the unadjusted base prediction.",
                RuntimeWarning,
                stacklevel=2,
            )
            for group in gm.groups:
                if str(group) in unseen:
                    y_hat[gm.get_mask(group)] = np.nan
        return y_hat

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sensitive_attr: Optional[ArrayLike] = None,
    ) -> float:
        """R² of this estimator's predictions.

        WHICH MODEL IS BEING SCORED (BGL g009, 2026-09-17). This delegated
        straight to ``self._inner_model.score(X, y)``, which is the R² of the
        UNADJUSTED base regressor: the mean-parity offsets are never applied,
        so the number described a model this class does not deploy, and said
        nothing about it. Measured on 60 rows, two groups, a genuine group gap
        in the target (offsets +/-0.0968): ``reg.score(X, y)`` returned
        0.683537, exactly ``r2_score(y, base_predictions)``, while the R² of
        what :meth:`predict_with_sensitive_attr` actually returns was 0.642071.
        A reader comparing a FairRegressor against an unconstrained baseline by
        this number therefore saw a mitigation that cost nothing, because the
        mitigation was not in the number. :meth:`predict` has warned about
        exactly this since the offsets were added; ``score`` was silent.

        Pass ``sensitive_attr`` and the ADJUSTED predictions are scored, which
        is the model that gets deployed. Without it the base model's R² is
        still returned, because it is a real measurement of what :meth:`predict`
        returns, but it now carries the same warning :meth:`predict` does.

        Args:
            X: Feature matrix
            y: True target values
            sensitive_attr: Group membership per row. Given, the score is of
                the fairness-adjusted predictions; omitted, it is of the
                unadjusted base predictions.

        Returns:
            R², or NaN when y is constant (the total sum of squares is zero, so
            the ratio R² is built from is undefined; it is not a fit of 0.0).
            NaN also propagates from rows an unseen group made unadjustable
            under ``on_unseen_group='nan'``.
        """
        if not self.is_fitted_:
            raise RuntimeError("FairRegressor is not fitted. Call fit() first.")

        X = coerce_to_array(X, "X")
        y_true = np.asarray(coerce_to_array(y, "y"), dtype=float)

        if sensitive_attr is None:
            if any(abs(v) > 1e-12 for v in self.group_offsets_.values()):
                warnings.warn(
                    "FairRegressor.score() without sensitive_attr scores the UNADJUSTED "
                    "base predictions, the same ones predict() returns: the fitted "
                    "mean_parity group offsets cannot be applied without group "
                    "membership, so this R^2 does NOT describe the fairness-adjusted "
                    "model. Pass sensitive_attr to score the model that gets deployed.",
                    stacklevel=2,
                )
            # Computed here rather than delegated to the base estimator's own
            # score(), so that BOTH branches of this method answer the constant-y
            # case the same way. sklearn's r2_score returns 1.0 there for an exact
            # fit, which is a perfect score invented out of a 0/0; delegating left
            # score(X, y) == 1.0 beside score(X, y, sensitive_attr=g) == nan on one
            # dataset. The formula is sklearn's, so the measured value is unchanged
            # (verified equal to 15 significant figures on the fixture above).
            y_pred = np.asarray(self._inner_model.predict(X), dtype=float)
        else:
            y_pred = self.predict_with_sensitive_attr(X, sensitive_attr)
        check_consistent_length(y_true, y_pred)

        # np.ptp, not a variance compared against 0.0: an accumulated statistic
        # is exactly zero only for some constant arrays, so a constant target
        # would slip past ``np.var(y) == 0`` and divide by an almost-zero.
        if float(np.ptp(y_true)) == 0.0:
            warnings.warn(
                "FairRegressor.score: y is constant, so the total sum of squares is "
                "zero and R^2 is undefined (it is a ratio against that spread). "
                "Returning NaN, which is the absence of a fit, not a fit of 0.0.",
                stacklevel=2,
            )
            return float("nan")

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        ``on_unseen_group`` is in here (BGL g009, 2026-09-17). It was not, and
        every field-by-field rebuild from these params therefore dropped it:
        measured, ``type(reg)(**reg.get_params()).on_unseen_group`` came back
        ``'raise'`` for a regressor built with ``on_unseen_group='nan'``, so a
        clone silently swapped a caller's chosen policy for a different one.
        ``deep`` was accepted and ignored as well, so no ``base_estimator__*``
        key was ever produced and ``set_params(base_estimator__...)`` had
        nothing to round-trip with.
        """
        params: Dict[str, Any] = {
            "base_estimator": self.base_estimator,
            "fairness_constraint": self.fairness_constraint,
            "tolerance": self.tolerance,
            "method": self.method,
            "verbose": self.verbose,
            "on_unseen_group": self.on_unseen_group,
        }
        if deep and hasattr(self.base_estimator, "get_params"):
            base_params = self.base_estimator.get_params(deep=True)
            params.update({f"base_estimator__{k}": v for k, v in base_params.items()})
        return params

    def set_params(self, **params) -> "FairRegressor":
        """Set parameters for this estimator.

        An unknown name RAISES (BGL g009, 2026-09-17). ``hasattr(self, key)``
        let every misspelling through in silence, and it accepted no
        ``base_estimator__*`` key at all: measured,
        ``reg.set_params(tolerence=99, base_estimator__fit_intercept=False)``
        returned self, changed nothing, and said nothing. A sweep over a name
        this object does not have is a sweep of identical models, every one of
        them fitted, scored, compared and ranked as though the parameter had
        moved.
        """
        valid = set(self.get_params(deep=False))
        unknown = [k for k in params if not k.startswith("base_estimator__") and k not in valid]
        if unknown:
            raise ValueError(
                f"FairRegressor.set_params: unknown parameter(s) {sorted(unknown)!r}. "
                f"Valid parameters are {sorted(valid)!r}, plus 'base_estimator__<name>' "
                f"for the base estimator's own parameters."
            )

        base_params = {}
        for key, value in params.items():
            if key.startswith("base_estimator__"):
                base_params[key[len("base_estimator__") :]] = value
            else:
                if key == "on_unseen_group" and value not in ("raise", "nan"):
                    raise ValueError(
                        f"on_unseen_group must be 'raise' or 'nan', got {value!r}. "
                        f"There is no mode that reports an unadjusted row as adjusted."
                    )
                setattr(self, key, value)

        if base_params:
            if not hasattr(self.base_estimator, "set_params"):
                raise ValueError(
                    f"FairRegressor.set_params: base_estimator "
                    f"{type(self.base_estimator).__name__} has no set_params, so "
                    f"{sorted(base_params)!r} cannot be applied."
                )
            self.base_estimator.set_params(**base_params)

        return self


def make_fair_classifier(
    base_estimator: Any,
    fairness_constraint: str = "demographic_parity",
    tolerance: float = 0.05,
    **kwargs,
) -> FairClassifier:
    """
    Convenience function to create a fair classifier.

    Args:
        base_estimator: Base sklearn classifier
        fairness_constraint: Fairness constraint type
        tolerance: Constraint tolerance
        **kwargs: Additional arguments

    Returns:
        Configured FairClassifier
    """
    return FairClassifier(
        base_estimator=base_estimator,
        fairness_constraint=fairness_constraint,
        tolerance=tolerance,
        **kwargs,
    )


def make_fair_regressor(
    base_estimator: Any,
    fairness_constraint: Literal["mean_parity"] = "mean_parity",
    tolerance: float = 0.1,
    **kwargs,
) -> FairRegressor:
    """
    Convenience function to create a fair regressor.

    Args:
        base_estimator: Base sklearn regressor
        fairness_constraint: Fairness constraint type ('mean_parity' is the
            only one implemented, which is why the annotation lists no other)
        tolerance: Constraint tolerance
        **kwargs: Additional arguments

    Returns:
        Configured FairRegressor
    """
    return FairRegressor(
        base_estimator=base_estimator,
        fairness_constraint=fairness_constraint,
        tolerance=tolerance,
        **kwargs,
    )
