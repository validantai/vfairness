"""
vfairness.in_processing.wrappers - Scikit-Learn Compatible Wrappers

This module provides scikit-learn compatible wrapper classes that integrate
fairness-aware training into standard ML workflows.

Wrappers Implemented:
    1. FairClassifier: Fairness-aware classifier wrapper
    2. FairRegressor: Fairness-aware regressor wrapper

Quick Start:
    >>> from sklearn.linear_model import LogisticRegression
    >>> from vfairness.in_processing.wrappers import FairClassifier
    >>>
    >>> clf = FairClassifier(
    ...     base_estimator=LogisticRegression(),
    ...     fairness_constraint='demographic_parity',
    ...     tolerance=0.05
    ... )
    >>> clf.fit(X_train, y_train, sensitive_attr=gender)
    >>> y_pred = clf.predict(X_test)

Factory Functions:
    >>> from vfairness.in_processing.wrappers import make_fair_classifier
    >>> clf = make_fair_classifier(LogisticRegression(), 'equalized_odds')

References:
    - Agarwal et al. (2018): A Reductions Approach to Fair Classification
"""

from .sklearn_wrappers import (
    # Wrappers
    FairClassifier,
    # Result classes
    FairClassifierResult,
    FairRegressor,
    # Factory functions
    make_fair_classifier,
    make_fair_regressor,
)

__all__ = [
    # Result classes
    "FairClassifierResult",
    # Wrappers
    "FairClassifier",
    "FairRegressor",
    # Factory functions
    "make_fair_classifier",
    "make_fair_regressor",
]
