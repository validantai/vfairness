"""
Feature Transformers for Fairness-Aware Feature Engineering.

This module provides a comprehensive suite of transformers for creating fair
feature representations. Each transformer follows the scikit-learn API pattern
with fit(), transform(), and fit_transform() methods.

Transformers Implemented:
    1. BaseFeatureTransformer: Abstract base class for all fairness transformers
    2. CorrelationReducer: Reduces correlation between features and protected attributes
    3. FairRepresentationLearner: Learns fair representations via adversarial techniques
    4. FeatureSuppressor: Suppresses or removes discriminatory features
    5. IntersectionalTransformer: Handles intersectional fairness concerns
    6. ResidualTransformer: Creates residualized features controlling for protected attributes

References:
    - Zemel et al. (2013): Learning Fair Representations
    - Louizos et al. (2016): Variational Fair Autoencoder
    - Madras et al. (2018): Learning Adversarially Fair and Transferable Representations
    - Feldman et al. (2015): Certifying and Removing Disparate Impact
    - Barocas & Selbst (2016): Big Data's Disparate Impact
"""

import hashlib
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from vfairness.evaluation.vfairness_metrics._validation import (
    ArrayLike,
)

from ..._triage import is_flag


def finite_or_nan(value: Any) -> float:
    """A real, finite number as float; anything else is NaN, "not measured".

    THE ONE definition of that question for this package.
    ``preprocessing.feature_engineering.visualization`` imports it under its
    historical private name ``_finite_or_nan``, which is where it was written
    under READINESS-6 (2026-09-10) and where it was tested; it moved here
    because ``TransformationResult.correlation_reduction``, in this module,
    needs the same rule and ``visualization`` imports ``transformers``, so the
    dependency only runs one way. A second copy was the alternative, and two
    copies of this rule disagreeing is the defect it was written to close.

    A BOOL is refused, via the shared ``_triage.is_flag``: ``float(True)`` is
    1.0, a finite number that passes every other check here, and ``np.bool_``
    is the half that gets missed because it is not a Python ``bool`` and comes
    straight out of a DataFrame column. A numeric STRING is accepted,
    deliberately: these dicts arrive from JSON and CSV where "0.5" is a real
    measurement that was serialised, and refusing it would discard evidence.
    """
    if is_flag(value):
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _attribute_contrast_reason(attr_series: pd.Series) -> Optional[str]:
    """Why a protected attribute admits NO association measurement, or None if it does.

    BGL stage 2 (2026-09-16): an association between a feature and a protected
    attribute is only defined when the ATTRIBUTE varies. With a single observed
    level, ss_between = sum_g n_g * (mean_g - grand)^2 has one term whose group
    mean IS the grand mean, so eta is 0.0 BY CONSTRUCTION; with a constant
    numeric attribute numpy's own answer is nan (a divide by zero). Reporting
    either as 0.0 publishes the clean end of the scale for a comparison that was
    never made. This names the reason so the caller reads a refusal, not a
    measurement.

    Level counting uses np.unique on the RAW codes/values, never a variance
    tested for equality with zero: np.var of a constant array is exactly 0.0
    only at some n.
    """
    if pd.api.types.is_numeric_dtype(attr_series):
        av = pd.to_numeric(attr_series, errors="coerce").to_numpy(dtype=float)
        finite = av[np.isfinite(av)]
        if finite.size == 0:
            return "no non-missing values"
        if np.unique(finite).size < 2:
            return "constant numeric value, so there is no contrast to measure against"
        return None
    codes = pd.Categorical(attr_series).codes  # -1 marks missing values
    observed = np.unique(codes[codes >= 0])
    if observed.size == 0:
        return "no non-missing values"
    if observed.size < 2:
        return "only 1 observed level, fewer than the 2 groups a comparison needs"
    return None


def _feature_attribute_association_with_reason(
    feature_vals: ArrayLike,
    attr_series: pd.Series,
    min_samples: int = 10,
) -> Tuple[Optional[float], Optional[str], int]:
    """``(association, refusal_reason, n_non_finite_excluded)``.

    Exactly one of the first two is None: a measured association carries no
    reason, and a refusal carries no number. The third element counts feature
    cells that were dropped because they were not finite, so a caller can say
    how much of the column the measurement actually rests on.

    BGL stage 2b (2026-09-17): the refusal REASON used to be thrown away at this
    boundary, so ``_measure_associations`` had to guess it and fell back to "no
    numeric feature with at least 10 valid paired rows" for an attribute that in
    fact had 150 valid rows and had lost its contrast on the feature's valid
    subset. The reason is computed where it is known and carried out.
    """
    x = np.asarray(feature_vals, dtype=float)
    # BGL stage 2 (2026-09-16): guard ABOVE the categorical/numeric branch
    # selection, because both branches share this precondition. Fixing only one
    # of them would move the defect to its sibling.
    contrast_reason = _attribute_contrast_reason(attr_series)
    if contrast_reason is not None:
        return None, contrast_reason, 0
    # A protected attribute is "categorical" for association purposes whenever it is
    # not numeric. Testing dtype == "object" misses the pandas-3 string dtype (a str
    # column is not object), which routed a string attribute into the numeric branch
    # and crashed on float conversion; is_numeric_dtype is robust across pandas 2/3.
    is_categorical = not pd.api.types.is_numeric_dtype(attr_series)
    # BGL stage 2b (2026-09-17): np.isfinite, NOT ~np.isnan. An inf cell passed
    # the old mask and then poisoned every downstream statistic: the numeric
    # branch published `corr = nan -> return 0.0`, the clean end of the scale for
    # a comparison numpy itself refused (measured: a proxy of true strength
    # 0.9983 read 0.0, attributes_measured ['age'], features_removed [],
    # warnings []), and the categorical branch published a bare nan the same way.
    # An inf is not a measurement of that cell, so the cell is excluded exactly
    # like a NaN and the exclusion is COUNTED rather than absorbed.
    finite_x = np.isfinite(x)
    n_non_finite = int(np.sum(~finite_x & ~np.isnan(x)))

    if is_categorical:
        codes = pd.Categorical(attr_series).codes  # -1 marks missing values
        valid = finite_x & (codes >= 0)
        n_valid = int(valid.sum())
        if n_valid < min_samples:
            return None, _too_few_rows_reason(n_valid, min_samples, n_non_finite), n_non_finite
        xv = x[valid]
        cv = codes[valid]
        # The attribute can lose its contrast on the valid subset alone (every
        # row of the second level carries a NaN feature). Same refusal.
        if np.unique(cv).size < 2:
            return (
                None,
                "the attribute has only 1 observed level on the "
                f"{n_valid} row(s) where the feature is present, so the contrast it "
                "would be measured against is gone",
                n_non_finite,
            )
        # audit-6 lane 2 (2026-09-09): a CONSTANT column must read 0.0, and the
        # ss_total guard below alone does not guarantee it. np.mean of 900 copies
        # of 13.279640262203964 is one ulp off the value under pairwise
        # summation, so ss_total came out ~1e-26 instead of 0, the per-group
        # means carried the same ulp noise, and eta read 0.5 to 1.0 for a column
        # with no variation at all (a masked feature reported as a perfect proxy).
        if xv.max() == xv.min():
            return 0.0, None, n_non_finite
        grand = xv.mean()
        ss_total = float(np.sum((xv - grand) ** 2))
        if ss_total <= 0.0:
            return 0.0, None, n_non_finite
        ss_between = 0.0
        for g in np.unique(cv):
            xg = xv[cv == g]
            ss_between += len(xg) * (xg.mean() - grand) ** 2
        eta_squared = min(max(ss_between / ss_total, 0.0), 1.0)
        eta = float(np.sqrt(eta_squared))
        if not np.isfinite(eta):
            return None, "the correlation ratio came out non-finite", n_non_finite
        return eta, None, n_non_finite

    av = np.asarray(attr_series.to_numpy(), dtype=float)
    valid = finite_x & np.isfinite(av)
    n_valid = int(valid.sum())
    if n_valid < min_samples:
        return None, _too_few_rows_reason(n_valid, min_samples, n_non_finite), n_non_finite
    # BGL stage 2 (2026-09-16): these two used to share one `return 0.0` and
    # they are opposite states. A constant ATTRIBUTE on the valid subset is a
    # could-not-check (numpy itself answers nan here); a constant FEATURE
    # genuinely has no association with anything and must keep reading 0.0.
    # np.unique on the raw values, not a variance compared against zero.
    if np.unique(av[valid]).size < 2:
        return (
            None,
            f"the attribute is constant on the {n_valid} row(s) where the feature is "
            "present, so there is no contrast left to measure",
            n_non_finite,
        )
    if np.unique(x[valid]).size < 2:
        return 0.0, None, n_non_finite
    corr = np.corrcoef(x[valid], av[valid])[0, 1]
    # numpy answering nan is numpy REFUSING. Publishing 0.0 here converted that
    # refusal into "no association", the most reassuring value on the scale.
    if not np.isfinite(corr):
        return None, "numpy returned a non-finite correlation for these rows", n_non_finite
    return float(abs(corr)), None, n_non_finite


def _too_few_rows_reason(n_valid: int, min_samples: int, n_non_finite: int) -> str:
    """Refusal text for the too-few-paired-rows case, naming the inf cells that
    were dropped when that is why the rows ran out."""
    base = f"only {n_valid} valid paired row(s), fewer than the {min_samples} a correlation needs"
    if n_non_finite:
        base += f" ({n_non_finite} non-finite feature value(s) were excluded)"
    return base


def _feature_attribute_association(
    feature_vals: ArrayLike,
    attr_series: pd.Series,
    min_samples: int = 10,
) -> Optional[float]:
    """Association strength between a numeric feature and a protected attribute.

    For a CATEGORICAL (nominal) attribute this returns the correlation ratio
    eta = sqrt(SS_between / SS_total) in [0, 1], the canonical measure of
    numeric-vs-categorical association. Pearson correlation on integer-coded
    categories is WRONG for a nominal attribute: it imposes an arbitrary linear
    order and reads ~0 for any non-monotonic group-mean pattern, so a feature
    that perfectly separates a middle group would falsely look decorrelated
    (audit-3 fix; Feldman et al. 2015).

    For a NUMERIC attribute this returns |Pearson r|, unchanged.

    Returns None when there are fewer than ``min_samples`` valid paired rows,
    when the protected ATTRIBUTE has no contrast (one observed level, or a
    constant numeric value), and when the statistic itself comes out non-finite:
    with nothing to compare across, the statistic is 0.0 by construction rather
    than by measurement. A constant FEATURE is a different case and genuinely
    reads 0.0. Use ``_feature_attribute_association_with_reason`` when the caller
    has to publish WHY it refused.
    """
    return _feature_attribute_association_with_reason(feature_vals, attr_series, min_samples)[0]


def _row_content_keys(frame: pd.DataFrame) -> np.ndarray:
    """A per-row digest of the row's own values, used ONLY to order rows that a
    ranker scores identically.

    hashlib, never the builtin ``hash()``: PYTHONHASHSEED salts hashing per
    process, so a hash()-based order would silently differ between two runs of
    the same program on the same data - exactly the non-determinism this is here
    to remove.
    """
    if frame.shape[1] == 0:
        return np.zeros(len(frame), dtype=np.uint64)
    vals = np.ascontiguousarray(frame.to_numpy(dtype=float))
    return np.array(
        [int.from_bytes(hashlib.sha256(row.tobytes()).digest()[:8], "big") for row in vals],
        dtype=np.uint64,
    )


def _select_boundary_rows(
    candidates: np.ndarray,
    scores: np.ndarray,
    keys: np.ndarray,
    n_select: int,
    take_highest: bool,
) -> Tuple[np.ndarray, int, int]:
    """``(chosen, n_tie_broken, n_tied_at_boundary)`` for the ``n_select``
    candidates closest to the decision boundary.

    BGL stage 2b (2026-09-17): ``np.argsort`` alone returns ROW ORDER inside a
    tied block, so when the flip boundary falls inside one the selection is an
    artefact of how the frame happened to be sorted. Measured on an ordinary
    single binary feature: 2 distinct scores, 24 labels rewritten, and only 14 of
    the 24 selected rows survived a row permutation, with warnings == []. The
    order inside a tie is taken from a row-content digest, so a permutation
    re-selects the same ROWS whenever the tied rows differ at all; rows that are
    byte-identical in every feature the ranker sees are indistinguishable and the
    choice between them stays arbitrary. Either way the count that had to be
    tie-broken is returned, so the caller publishes the arbitrariness instead of
    implying proximity.
    """
    if candidates.size == 0 or n_select <= 0:
        return candidates[:0], 0, 0
    primary = -scores[candidates] if take_highest else scores[candidates]
    order = np.lexsort((keys[candidates], primary))
    chosen = candidates[order[:n_select]]
    if chosen.size == 0:
        return chosen, 0, 0
    boundary = scores[chosen[-1]]
    n_tied_total = int(np.sum(scores[candidates] == boundary))
    n_tied_chosen = int(np.sum(scores[chosen] == boundary))
    n_tie_broken = n_tied_chosen if n_tied_total > n_tied_chosen else 0
    return chosen, n_tie_broken, n_tied_total


def _measure_associations(
    owner: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    protected_attrs: List[str],
    warn: bool = True,
) -> Tuple[Dict[str, float], List[str]]:
    """Average feature/attribute association per protected attribute, plus the
    REASONS any attribute could not be measured.

    Shared by the three transformers that publish a correlation_before /
    correlation_after report. Returns ``(correlations, reasons)``; an attribute
    that could not be measured is absent from ``correlations`` (the plotter and
    ``TransformationResult.correlation_reduction`` treat absence as "not
    measured") and named in ``reasons``, which the caller puts on
    ``fit_result.warnings`` so the refusal survives into the serialised report
    and not only into a Python warning.

    BGL stage 2b (2026-09-17): a PARTIAL measurement is now disclosed too. The
    average used to be taken over whatever features happened to be scorable and
    the rest vanished, so "x alone" and "x plus an all-NaN column y" both
    published before = 0.9692627277016622 with warnings == []. The sibling
    FeatureSuppressor already discloses that as features_not_assessed; this puts
    the same fact on the two transformers that do not.
    """
    correlations: Dict[str, float] = {}
    reasons: List[str] = []

    for attr in protected_attrs:
        assocs = []
        unscored: Dict[str, str] = {}
        n_non_finite_total = 0
        for col in feature_cols:
            try:
                feature_vals = df[col].to_numpy(dtype=float)
            except (ValueError, TypeError) as exc:
                unscored[col] = f"not convertible to float ({exc.__class__.__name__})"
                continue
            assoc, why, n_non_finite = _feature_attribute_association_with_reason(
                feature_vals, df[attr]
            )
            n_non_finite_total += n_non_finite
            if assoc is None:
                unscored[col] = why or "no reason recorded"
                continue
            assocs.append(assoc)

        if assocs:
            correlations[attr] = float(np.mean(assocs))
            if unscored:
                detail = "; ".join(f"{c}: {r}" for c, r in sorted(unscored.items())[:4])
                partial = (
                    f"{owner}: the association reported for {attr!r} is the mean over "
                    f"{len(assocs)} of {len(feature_cols)} feature(s); "
                    f"{len(unscored)} could not be scored against it and are NOT in that "
                    f"average ({detail}" + (" ..." if len(unscored) > 4 else "") + ")."
                )
                reasons.append(partial)
                if warn:
                    warnings.warn(partial)
            if n_non_finite_total:
                note = (
                    f"{owner}: {n_non_finite_total} non-finite feature value(s) were "
                    f"excluded before measuring against {attr!r}; the association rests "
                    "on the finite rows only."
                )
                reasons.append(note)
                if warn:
                    warnings.warn(note)
            continue

        # audit-6 lane 2 (2026-09-09): nothing measurable is not 0.0. A 0.0 here
        # read as "no association", which the before/after report then graded as
        # success. BGL stage 2 (2026-09-16): say WHICH of the reasons it was, and
        # carry it on the result object, not only in a Python warning. BGL stage
        # 2b (2026-09-17): the reason comes from the measurement that refused,
        # not from a re-derivation that could not see the feature's valid subset.
        if attr not in df.columns:
            reason = "the attribute column is absent from the frame"
        elif unscored:
            distinct = sorted(set(unscored.values()))
            reason = "; ".join(distinct[:3]) + (" ..." if len(distinct) > 3 else "")
        else:
            reason = "there was no feature column to measure it against"
        message = (
            f"{owner}: no feature/attribute association could be measured for "
            f"{attr!r} ({reason}); it is left out of the correlation report "
            "rather than reported as 0.0."
        )
        reasons.append(message)
        if warn:
            warnings.warn(message)

    return correlations, reasons


class TransformationMethod(Enum):
    """Enumeration of available transformation methods."""

    CORRELATION_REDUCTION = "correlation_reduction"
    FAIR_REPRESENTATION = "fair_representation"
    FEATURE_SUPPRESSION = "feature_suppression"
    RESIDUALIZATION = "residualization"
    REWEIGHTING = "reweighting"
    INTERSECTIONAL = "intersectional"


class FairnessObjective(Enum):
    """Fairness objectives for transformation."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


@dataclass
class TransformationResult:
    """
    Container for transformation results.

    Attributes:
        method: Transformation method used
        n_features_original: Original number of features
        n_features_transformed: Number of features after transformation
        n_samples: Number of samples processed
        correlation_before: Average correlation with protected attrs before
        correlation_after: Average correlation with protected attrs after
        features_removed: List of features removed (if any)
        features_modified: List of features modified
        fit_metrics: Metrics computed during fitting
        warnings: Any warnings generated during transformation
    """

    method: str
    n_features_original: int
    n_features_transformed: int
    n_samples: int
    correlation_before: Dict[str, float] = field(default_factory=dict)
    correlation_after: Dict[str, float] = field(default_factory=dict)
    features_removed: List[str] = field(default_factory=list)
    features_modified: List[str] = field(default_factory=list)
    fit_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "method": self.method,
            "n_features_original": self.n_features_original,
            "n_features_transformed": self.n_features_transformed,
            "n_samples": self.n_samples,
            "correlation_before": self.correlation_before,
            "correlation_after": self.correlation_after,
            "features_removed": self.features_removed,
            "features_modified": self.features_modified,
            "fit_metrics": self.fit_metrics,
            "warnings": self.warnings,
        }

    @property
    def correlation_reduction(self) -> Dict[str, float]:
        """Compute correlation reduction per protected attribute.

        Only attributes measured on BOTH sides appear. An attribute that is
        absent from this dict is a could-not-check, never a zero reduction, and
        ``plot_feature_transformation_effect`` prints these as percentages, so a
        fabricated entry here is read as a headline result.

        G07 2026-09-30: the finiteness gate was ``np.isfinite(before) and
        np.isfinite(after)``, and it disagreed with ``finite_or_nan`` above (the
        rule the chart built from these same two dicts uses) in both directions.
        ``np.isfinite(True)`` is True and ``True > 0`` is True, so
        ``correlation_before={'race': True}`` with ``correlation_after={'race':
        False}`` computed ``(True - False) / True`` and published
        ``{'race': 1.0}``, which that panel renders as "race: 100.0%": the
        complete elimination of a proxy correlation, manufactured out of two
        flags. In the other direction ``np.isfinite("0.8")`` RAISES, so a pair
        of correlations that had been through JSON or CSV crashed this property
        with a TypeError while the chart drew them as real numbers.
        """
        reduction = {}
        for attr in self.correlation_before:
            if attr in self.correlation_after:
                # audit-6 lane 2 (2026-09-09): a NaN on either side is "not
                # measured"; it is excluded, never reported as a 0.0 reduction.
                # finite_or_nan collapses every other absence (None, a bool, a
                # non-numeric string, an infinity) into that same NaN.
                before = finite_or_nan(self.correlation_before[attr])
                after = finite_or_nan(self.correlation_after[attr])
                if math.isnan(before) or math.isnan(after):
                    continue
                if before > 0:
                    reduction[attr] = (before - after) / before
                elif after > 0:
                    # G07 2026-09-30: this was 0.0, and 0.0 here reads as "the
                    # transformation changed nothing" while the correlation went
                    # UP from nothing to something. A RELATIVE reduction has no
                    # value when the denominator is zero, and the one case that
                    # is not a fabrication is the one below, where there was no
                    # correlation before and there is none now.
                    warnings.warn(
                        f"TransformationResult.correlation_reduction: {attr!r} had no "
                        f"correlation before the transformation ({before:g}) and has "
                        f"{after:g} after it, so a relative reduction does not exist and "
                        f"{attr!r} is absent from this dict. The transformation INTRODUCED "
                        f"this correlation; reporting a 0.0 reduction would read as no "
                        f"change.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                else:
                    # Nothing before and nothing after: 0.0 is a true statement
                    # about a real pair of measurements, not a filled-in default.
                    reduction[attr] = 0.0
        return reduction


@dataclass
class FeatureImportanceResult:
    """
    Result of feature importance analysis for fairness.

    Attributes:
        feature: Feature name
        predictive_importance: Importance for prediction task
        fairness_impact: Impact on fairness metrics
        correlation_with_protected: Correlations with protected attributes
        recommendation: Suggested action (keep, modify, remove)
        rationale: Explanation for the recommendation
    """

    feature: str
    predictive_importance: float
    fairness_impact: float
    correlation_with_protected: Dict[str, float]
    recommendation: Literal["keep", "modify", "remove", "monitor"]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "feature": self.feature,
            "predictive_importance": self.predictive_importance,
            "fairness_impact": self.fairness_impact,
            "correlation_with_protected": self.correlation_with_protected,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


class BaseFeatureTransformer(ABC):
    """
    Abstract base class for fairness-aware feature transformers.

    All feature transformers should inherit from this class and implement
    the fit() and transform() methods. This follows the scikit-learn API
    pattern for compatibility with pipelines.

    Attributes:
        is_fitted: Whether the transformer has been fitted
        fit_result: Results from the fitting process
        protected_attributes: Names of protected attribute columns
        feature_names_in_: Names of input features (set during fit)
        feature_names_out_: Names of output features (set during fit)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
    ):
        """
        Initialize the transformer.

        Args:
            protected_attributes: Names of protected attribute columns.
                If None, must be provided during fit().
        """
        self.protected_attributes = protected_attributes
        self.is_fitted = False
        self.fit_result: Optional[TransformationResult] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "BaseFeatureTransformer":
        """
        Fit the transformer to training data.

        Args:
            X: Feature matrix (DataFrame or array)
            y: Optional target variable (for supervised transformers)
            protected_attributes: Names of protected attribute columns
                (overrides constructor value if provided)

        Returns:
            self: The fitted transformer
        """
        pass

    @abstractmethod
    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform features using the fitted transformer.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        pass

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit the transformer and transform in one step.

        Args:
            X: Feature matrix
            y: Optional target variable
            protected_attributes: Names of protected attribute columns

        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, protected_attributes)
        return self.transform(X)

    def _check_is_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() before transform()."
            )

    def _fitted_protected_attributes(self) -> List[str]:
        """The protected attributes settled by fit(), narrowed to a real list.

        fit() always routes through _validate_protected_attributes, which either
        returns a non-empty list or raises, so this is never None on a fitted
        transformer. The attribute itself stays Optional because it is also a
        constructor argument, so callers that need a List[str] (the resamplers,
        reweighters and augmenters) come through here instead of assuming the
        invariant silently.
        """
        self._check_is_fitted()
        attrs = self.protected_attributes
        if attrs is None:
            raise RuntimeError(
                f"{self.__class__.__name__} reports fitted but has no protected "
                "attributes; fit() must set them."
            )
        return attrs

    def _validate_protected_attributes(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        protected_attributes: Optional[List[str]] = None,
    ) -> List[str]:
        """Validate and return protected attributes list."""
        attrs = protected_attributes or self.protected_attributes

        if attrs is None:
            raise ValueError(
                "protected_attributes must be provided either in the constructor "
                "or in the fit() method."
            )

        if isinstance(X, pd.DataFrame):
            missing = [a for a in attrs if a not in X.columns]
            if missing:
                raise ValueError(f"Protected attributes not found in DataFrame: {missing}")

        return attrs

    def _to_dataframe(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convert input to DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X.copy()
        else:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=feature_names)

    def _extract_feature_columns(
        self,
        df: pd.DataFrame,
        protected_attributes: List[str],
    ) -> List[str]:
        """Extract feature columns (excluding protected attributes)."""
        return [c for c in df.columns if c not in protected_attributes]

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names after transformation."""
        self._check_is_fitted()
        if self.feature_names_out_ is None:
            raise RuntimeError("Feature names not available.")
        return self.feature_names_out_


class CorrelationReducer(BaseFeatureTransformer):
    """
    Reduces correlation between features and protected attributes.

    Uses orthogonalization techniques to project features onto a subspace
    that is less correlated with protected attributes while preserving
    predictive information.

    Methods:
        - 'residualize': Regress out protected attributes
        - 'decorrelate': Orthogonal projection

        'partial' is refused at construction (NotImplementedError): the
        "partial correlation adjustment" scaled every residual by exactly 1.0
        and was byte-identical to 'residualize' (measured 2026-09-09).

    Attributes:
        method: Correlation reduction method
        target_correlation: Maximum average association per protected
            attribute that the fit VERIFIES against the measured
            ``correlation_after``. It does not steer the reduction: both
            methods are full projections whose output is the same for every
            target (measured identical for 0.01 / 0.5 / 0.99, 2026-09-09).
            The verdict lives in ``fit_result.fit_metrics['target_met']``
            (per attribute: True / False / None when nothing could be
            measured) and an unmet target raises a warning.
        preserve_variance: Whether to preserve feature variance

    Example:
        >>> reducer = CorrelationReducer(
        ...     protected_attributes=['gender', 'race'],
        ...     method='residualize',
        ...     target_correlation=0.1
        ... )
        >>> X_fair = reducer.fit_transform(X)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. An independent
    check has argued with that judgement and upheld it. The pin was sabotage-
    checked: it was shown to go red when the defect is reintroduced, so it can fail.
    This does NOT establish that its statistics are accurate, nor that the pin
    covers every scenario.

    Ledger row: correlation_reduction. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        method: Literal["residualize", "decorrelate"] = "residualize",
        target_correlation: float = 0.1,
        preserve_variance: bool = True,
    ):
        """
        Initialize the correlation reducer.

        Args:
            protected_attributes: Names of protected attribute columns
            method: Correlation reduction method ('residualize' or
                'decorrelate'; 'partial' is refused, see the class docstring)
            target_correlation: Maximum average association per protected
                attribute, in [0, 1], verified against the measured
                ``correlation_after`` and reported in ``fit_result``. It does
                not change the transformed output.
            preserve_variance: Whether to preserve original feature variance
        """
        super().__init__(protected_attributes)
        # audit-6 lane 2 (2026-09-09): a Literal hint is not a runtime check.
        # 'partial' was documented as "partial correlation adjustment" but its
        # only distinct step set every scale factor to 1.0 ("Simplified"), so it
        # was byte-identical to 'residualize' while the result reported
        # 'correlation_reduction_partial'. Refuse it rather than keep the alias.
        if method == "partial":
            raise NotImplementedError(
                "CorrelationReducer(method='partial') is not implemented: the "
                "'partial correlation adjustment' scaled every residual by exactly "
                "1.0 and produced the same output as method='residualize'. Use "
                "'residualize' or 'decorrelate'."
            )
        if method not in ("residualize", "decorrelate"):
            raise ValueError(
                f"CorrelationReducer method must be 'residualize' or 'decorrelate', got {method!r}"
            )
        target = float(target_correlation)
        if not 0.0 <= target <= 1.0:
            raise ValueError(
                "target_correlation must be in [0, 1] (an association strength), "
                f"got {target_correlation!r}"
            )
        self.method = method
        self.target_correlation = target
        self.preserve_variance = preserve_variance
        self._projection_matrices: Dict[str, np.ndarray] = {}
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._protected_encoders: Dict[str, Any] = {}
        # feature -> least-squares beta (with intercept), or None when the
        # feature could not be fit (too few valid rows / singular design).
        self._residual_coefficients: Dict[str, Optional[np.ndarray]] = {}
        # (Z'Z)^+ for method='decorrelate', or None when THIS fit could not
        # estimate one. Surface grade g002 (2026-09-17): it has to be an
        # ordinary attribute initialised here, not one created only on success.
        # _transform_decorrelate tested `hasattr(self, "_ZtZ_inv")`, which stays
        # True forever once any fit succeeded, so a REFIT on a frame too small
        # to estimate a projection silently kept applying the PREVIOUS fit's
        # projection (measured: fit on 200 rows, refit on 8, transform != raw,
        # fit_result.warnings carried nothing about it). A projection estimated
        # from data that is not this fit's data is not a measurement of it.
        self._ZtZ_inv: Optional[np.ndarray] = None
        # Set by transform(): which rows it had no fitted group for. transform()
        # returns a bare frame, so without this the only channel is a warning.
        self.transform_result: Dict[str, Any] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "CorrelationReducer":
        """
        Fit the correlation reducer.

        Args:
            X: Feature matrix (must be DataFrame for named columns)
            y: Optional target variable (not used)
            protected_attributes: Names of protected attribute columns

        Returns:
            self: The fitted reducer
        """
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        feature_cols = self._extract_feature_columns(df, attrs)

        self.feature_names_in_ = list(df.columns)
        self.feature_names_out_ = feature_cols.copy()

        # BGL stage 2b (2026-09-17): the before-side reasons used to be dropped
        # here (only _compute_correlations was called), so a refused measurement
        # reached the serialised report as an absent dict key and nothing else.
        correlation_before, before_reasons = self._measure_with_reasons(
            df, feature_cols, attrs, stage="before"
        )

        feature_data = df[feature_cols].values.astype(float)
        self._feature_means = np.nanmean(feature_data, axis=0)
        self._feature_stds = np.nanstd(feature_data, axis=0)

        for attr in attrs:
            # Non-numeric (object / category / pandas-3 str) -> one-hot; numeric ->
            # used as-is. is_numeric_dtype is robust to the pandas-3 string dtype that
            # dtype == "object" misses.
            if not pd.api.types.is_numeric_dtype(df[attr]):
                # Store the (sorted, stable) category list so _encode_protected
                # can build a one-hot (drop-first) design matrix. A single
                # ordinal code would impose a false linear order and leave a
                # nominal attribute's group-mean structure largely intact
                # after residualization (audit-3 fix).
                self._protected_encoders[attr] = list(pd.Categorical(df[attr]).categories)
            else:
                self._protected_encoders[attr] = None

        if self.method == "residualize":
            self._fit_residualize(df, feature_cols, attrs)
        elif self.method == "decorrelate":
            self._fit_decorrelate(df, feature_cols, attrs)
        else:
            raise ValueError(f"Unsupported correlation reduction method {self.method!r}")

        # Surface grade g002 (2026-09-17): ``features_modified=feature_cols`` was
        # unconditional, so a column this fit could build NO reduction for was
        # reported as modified while transform() returned it byte-identical.
        # Measured on a column with 9 observed rows (below the 10-row least
        # squares guard) beside three healthy ones: features_modified named it,
        # fit_metrics['target_met'] read True, and the only warning was about the
        # ASSOCIATION being unscorable, not about the reduction never happening.
        # The sibling FeatureSuppressor already publishes features_not_suppressed
        # for exactly this shape; this is the same disclosure.
        if self.method == "residualize":
            features_not_reduced = [
                c for c in feature_cols if self._residual_coefficients.get(c) is None
            ]
        else:
            features_not_reduced = [] if self._ZtZ_inv is not None else list(feature_cols)

        # Mark as fitted before calling transform() for post-fit correlation check
        self.is_fitted = True

        X_transformed = self.transform(X)
        if isinstance(X_transformed, pd.DataFrame):
            df_transformed = X_transformed.copy()
        else:
            df_transformed = pd.DataFrame(X_transformed, columns=feature_cols)
        # Ensure protected attributes are present for correlation computation
        for attr in attrs:
            if attr not in df_transformed.columns:
                df_transformed[attr] = df[attr].values

        # ``_compute_correlations`` stays the VALUE channel here: it is the seam a
        # caller (and tests/test_audit6_lane2_feature_engineering.py) substitutes
        # to drive the verdict ladder, and moving fit() off it would silently
        # disconnect that override. BGL stage 2b (2026-09-17): the after-side
        # REASONS were dropped entirely, so an attribute the after-measurement
        # refused reached the report as an absent dict key with nothing saying
        # why. They are collected below, once, and only when something refused,
        # so the healthy path still measures exactly once.
        correlation_after = self._compute_correlations(df_transformed, feature_cols, attrs)
        after_reasons: List[str] = []
        if any(a not in correlation_after for a in attrs):
            _, after_reasons = _measure_associations(
                f"{self.__class__.__name__} (after)",
                df_transformed,
                feature_cols,
                attrs,
                warn=False,
            )

        # audit-6 lane 2 (2026-09-09): target_correlation was stored and never
        # read (outputs measured identical for 0.01 / 0.5 / 0.99). Both methods
        # are full projections, so the target cannot steer them; what it CAN do
        # honestly is be checked against the measurement. Three states per
        # attribute: True (measured, met), False (measured, not met, warned),
        # None (no association could be measured, so nothing to compare).
        # BGL stage 2b (2026-09-17): `bool(nan <= target)` is False, so a NaN that
        # reached this ladder was graded "measured, target not met" and warned
        # about as if a comparison had happened. A non-finite measurement is the
        # SAME could-not-check as an absent one and takes the None arm.
        target_met: Dict[str, Optional[bool]] = {}
        for attr in attrs:
            after = correlation_after.get(attr)
            if after is None or not np.isfinite(after):
                target_met[attr] = None
            else:
                target_met[attr] = bool(after <= self.target_correlation)
        unmet = {a: correlation_after[a] for a, ok in target_met.items() if ok is False}
        if unmet:
            warnings.warn(
                "CorrelationReducer: measured association after reduction exceeds "
                f"target_correlation={self.target_correlation} for {unmet}; the target "
                "is verified, not enforced (the projection has no strength knob)."
            )

        # BGL stage 2 (2026-09-16): a None verdict must be legible on the RESULT,
        # not only as an absent dict key. Before this, a single-level attribute
        # produced target_met={'race': True} with warnings == [], a PASS for a
        # target nothing was ever compared against.
        result_warnings: List[str] = list(before_reasons) + list(after_reasons)
        if features_not_reduced:
            not_reduced_message = (
                f"no reduction was fitted for {len(features_not_reduced)} of "
                f"{len(feature_cols)} feature(s) {features_not_reduced[:8]}"
                + (" ..." if len(features_not_reduced) > 8 else "")
                + ": transform() returns them UNCHANGED, so they keep whatever "
                "association with the protected attribute(s) they had. They are left "
                "out of features_modified rather than reported as reduced, and any of "
                "them that WAS scorable is still inside the correlation_after average, "
                "with its association unreduced. Treat them as could-not-check."
            )
            result_warnings.append(not_reduced_message)
            warnings.warn(f"CorrelationReducer: {not_reduced_message}")
        for attr, verdict in target_met.items():
            if verdict is not None:
                continue
            # BGL stage 2b (2026-09-17): the reason came from re-deriving the
            # attribute's contrast on the WHOLE frame, which cannot see that the
            # attribute lost its contrast on the feature's valid subset; it read
            # "no numeric feature with at least 10 valid paired rows" for a frame
            # with 150 valid rows. The measurement's own refusal is used first.
            measured_reasons = [m for m in after_reasons if f"{attr!r}" in m]
            if measured_reasons:
                reason = "; ".join(measured_reasons)
            else:
                reason = (
                    _attribute_contrast_reason(df[attr])
                    if attr in df.columns
                    else "the attribute column is absent from the frame"
                ) or "the after-measurement recorded no reason"
            result_warnings.append(
                f"target_met is None for {attr!r}: no association was measured, so "
                f"nothing was compared against "
                f"target_correlation={self.target_correlation}. {reason}"
            )

        self.fit_result = TransformationResult(
            method=f"correlation_reduction_{self.method}",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            correlation_before=correlation_before,
            correlation_after=correlation_after,
            features_modified=[c for c in feature_cols if c not in features_not_reduced],
            fit_metrics={
                "target_correlation": self.target_correlation,
                "target_met": target_met,
                "attributes_not_measured": [a for a, ok in target_met.items() if ok is None],
                "features_not_reduced": features_not_reduced,
                "n_features_not_reduced": len(features_not_reduced),
            },
            warnings=result_warnings,
        )

        return self

    def _fit_residualize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
    ) -> None:
        """Fit residualization model (regress out protected attributes)."""
        Z = self._encode_protected(df, protected_attrs)

        self._residual_coefficients = {}

        for col in feature_cols:
            y = df[col].values.astype(float)
            mask = ~np.isnan(y) & ~np.any(np.isnan(Z), axis=1)

            if mask.sum() < 10:
                self._residual_coefficients[col] = None
                continue

            # Add intercept
            Z_with_intercept = np.column_stack([np.ones(mask.sum()), Z[mask]])

            try:
                # Solve least squares: y = Z @ beta
                beta, _, _, _ = np.linalg.lstsq(Z_with_intercept, y[mask], rcond=None)
                self._residual_coefficients[col] = beta
            except np.linalg.LinAlgError:
                self._residual_coefficients[col] = None

    def _fit_decorrelate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
    ) -> None:
        """Fit decorrelation via orthogonal projection."""
        # Cleared FIRST, so a fit that cannot estimate a projection leaves no
        # projection behind rather than inheriting the previous fit's one.
        self._ZtZ_inv = None
        Z = self._encode_protected(df, protected_attrs)
        # Prepend an intercept so the projection removes each group's mean. The
        # drop-first one-hot spans only the non-reference group differences;
        # without the constant the reference-group mean survives and a nominal
        # attribute would stay associated with the feature (audit-3 fix).
        Z = np.column_stack([np.ones(len(Z)), Z])
        X = df[feature_cols].values.astype(float)

        mask = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isnan(Z), axis=1)
        X_clean = X[mask]
        Z_clean = Z[mask]

        if X_clean.shape[0] < 10:
            self._projection_matrices["decorrelate"] = np.eye(X.shape[1])
            return

        # Compute projection matrix: P = I - Z(Z'Z)^{-1}Z'
        try:
            ZtZ_inv = np.linalg.pinv(Z_clean.T @ Z_clean)
            self._Z_projection = Z_clean.T @ Z_clean
            self._ZtZ_inv = ZtZ_inv
        except np.linalg.LinAlgError:
            self._projection_matrices["decorrelate"] = np.eye(X.shape[1])

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform features to reduce correlation with protected attributes.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix with reduced correlations
        """
        self._check_is_fitted()

        assert self.protected_attributes is not None  # set in fit()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        feature_cols = self._extract_feature_columns(df, self.protected_attributes)
        self._record_unfitted_groups(df)

        if self.method == "residualize":
            result = self._transform_residualize(df, feature_cols)
        elif self.method == "decorrelate":
            result = self._transform_decorrelate(df, feature_cols)
        else:
            # audit-6 lane 2 (2026-09-09): this used to return the features
            # unchanged for an unrecognised method, a silent no-op.
            raise ValueError(f"Unsupported correlation reduction method {self.method!r}")

        if self.preserve_variance and self._feature_stds is not None:
            result_std = np.nanstd(result, axis=0)
            scale = np.where(result_std > 0, self._feature_stds / result_std, 1.0)
            result = result * scale

        if was_dataframe:
            return pd.DataFrame(result, columns=feature_cols, index=df.index)
        return result

    def _transform_residualize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Apply residualization transformation."""
        assert self.protected_attributes is not None  # set in fit()
        Z = self._encode_protected(df, self.protected_attributes)
        result = np.zeros((len(df), len(feature_cols)))

        for i, col in enumerate(feature_cols):
            y = df[col].values.astype(float)
            coeffs = self._residual_coefficients.get(col)

            if coeffs is not None:
                # Add intercept
                Z_with_intercept = np.column_stack([np.ones(len(df)), Z])
                predicted = Z_with_intercept @ coeffs
                result[:, i] = y - predicted
            else:
                result[:, i] = y

        return result

    def _transform_decorrelate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Apply decorrelation transformation."""
        assert self.protected_attributes is not None  # set in fit()
        X = df[feature_cols].values.astype(float)
        Z = self._encode_protected(df, self.protected_attributes)
        # Match the intercept-augmented design used in _fit_decorrelate so the
        # projection columns line up with the stored (Z'Z)^+ (audit-3 fix).
        Z = np.column_stack([np.ones(len(Z)), Z])

        if self._ZtZ_inv is not None:
            # Project out Z: X_fair = X - Z @ (Z'Z)^{-1} @ Z' @ X
            Z_proj = Z @ self._ZtZ_inv @ Z.T
            result = X - Z_proj @ X
        else:
            result = X

        return result

    def _record_unfitted_groups(self, df: pd.DataFrame) -> None:
        """Disclose rows whose protected level the fit never saw.

        Surface grade BGL stage 3 (2026-09-27): ``_encode_protected`` builds the
        one-hot from the categories observed at FIT and drops the first as the
        reference, so a level it never saw becomes the all-zero row, which IS the
        reference category. A row whose group was never fitted therefore received
        the REFERENCE group's correction and nothing said so. Measured with
        method='residualize' fitted on gender {a, b}: 20 rows of gender 'c' with
        proxy=40.0 came back 39.922704, byte-identical to the same rows labelled
        'a', where group 'b' would have been moved to 33.914867 - a 6.0 wide
        difference decided by a group the fit never observed, with warnings == [].
        A missing protected value encodes to the same all-zero row and is counted
        with it. The two siblings that condition on a group,
        ResidualTransformer.transform and DisparateImpactRemover.transform, both
        disclose this already; the rows are left as they are (mapping them onto a
        fitted group would fabricate a different thing) and named here.
        """
        assert self.protected_attributes is not None  # set in fit()
        unfitted_levels: Dict[str, List[str]] = {}
        n_rows_unfitted = 0
        n_rows_missing = 0
        for attr in self.protected_attributes:
            categories = self._protected_encoders.get(attr)
            # None marks a NUMERIC attribute: it enters the design as itself, so
            # there is no unseen LEVEL to report and a missing value propagates
            # visibly as NaN through the regression instead of silently.
            if categories is None or attr not in df.columns:
                continue
            col = df[attr]
            missing = col.isna()
            unfitted = ~col.isin(list(categories)) & ~missing
            if unfitted.any():
                unfitted_levels[attr] = sorted({str(v) for v in col[unfitted].unique()})
                n_rows_unfitted = max(n_rows_unfitted, int(unfitted.sum()))
            n_rows_missing = max(n_rows_missing, int(missing.sum()))
        self.transform_result = {
            "n_rows": int(len(df)),
            "unfitted_levels": unfitted_levels,
            "n_rows_with_an_unfitted_level": n_rows_unfitted,
            "n_rows_without_a_recorded_group": n_rows_missing,
        }
        if unfitted_levels or n_rows_missing:
            warnings.warn(
                f"CorrelationReducer: {n_rows_unfitted} row(s) carry a protected level the "
                f"fit never saw {unfitted_levels} and {n_rows_missing} row(s) carry no "
                "protected value at all. Both encode to the all-zero design row, which is "
                "the REFERENCE category, so those rows receive the reference group's "
                "correction rather than one estimated for them: treat them as "
                "could-not-check, not as reduced. See .transform_result."
            )

    def _encode_protected(
        self,
        df: pd.DataFrame,
        protected_attrs: List[str],
    ) -> np.ndarray:
        """Encode protected attributes into a regression design matrix.

        Categorical attributes are one-hot encoded with the first (reference)
        category dropped, so residualizing/decorrelating against Z removes the
        FULL (k-1 degrees of freedom) group-mean structure of a nominal
        attribute. A single label-encoded ordinal column only removes the
        spurious linear trend of an arbitrary integer order and leaves most of
        the protected signal intact (audit-3 fix). Numeric attributes are used
        as-is.
        """
        encoded = []

        for attr in protected_attrs:
            categories = self._protected_encoders.get(attr)
            if categories is not None:
                col = df[attr]
                # drop-first: the first category is the reference (an all-zero
                # row), which avoids collinearity with the intercept added by
                # the residualize/decorrelate design.
                for cat in list(categories)[1:]:
                    encoded.append((col == cat).to_numpy(dtype=float))
            else:
                encoded.append(df[attr].to_numpy(dtype=float))

        if not encoded:
            # Every protected attribute is constant (a single category), so
            # there is no group structure to regress out; a zero column leaves
            # the features unchanged after residualization.
            return np.zeros((len(df), 1))

        return np.column_stack(encoded)

    def _compute_correlations(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
    ) -> Dict[str, float]:
        """Average feature/attribute association per protected attribute.

        Uses the correlation ratio eta for categorical attributes and
        |Pearson r| for numeric ones (see _feature_attribute_association), so a
        nominal association is measured honestly instead of via Pearson on
        arbitrary integer codes. Without this, correlation_after read ~0 for an
        untouched nominal association, falsely reporting success (audit-3 fix).
        """
        return self._measure_with_reasons(df, feature_cols, protected_attrs)[0]

    def _measure_with_reasons(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
        stage: str = "",
    ) -> Tuple[Dict[str, float], List[str]]:
        """``_compute_correlations`` plus the reasons any attribute was omitted.

        ``stage`` labels which side of the before/after report is being measured,
        so the two refusals are distinguishable in ``warnings`` instead of being
        byte-identical strings (which also made the Python warning look like one
        duplicated emission rather than two separate measurements).
        """
        owner = f"{self.__class__.__name__} ({stage})" if stage else self.__class__.__name__
        return _measure_associations(owner, df, feature_cols, protected_attrs)


class FeatureSuppressor(BaseFeatureTransformer):
    """
    Suppresses or removes features based on fairness criteria.

    Provides multiple strategies for handling discriminatory features:
    - Remove: Completely remove the feature
    - Mask: Replace with a constant value
    - Noise: Add noise to reduce discriminatory signal
    - Bin: Discretize to reduce granularity

    Attributes:
        strategy: How to handle identified features
        correlation_threshold: Threshold for automatic identification
        features_to_suppress: Manually specified features to suppress

    Example:
        >>> suppressor = FeatureSuppressor(
        ...     protected_attributes=['race'],
        ...     strategy='remove',
        ...     correlation_threshold=0.3
        ... )
        >>> X_fair = suppressor.fit_transform(X)

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: feature_suppression. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        strategy: Literal["remove", "mask", "noise", "bin"] = "remove",
        correlation_threshold: float = 0.3,
        features_to_suppress: Optional[List[str]] = None,
        noise_scale: float = 0.1,
        n_bins: int = 5,
    ):
        """
        Initialize the feature suppressor.

        Args:
            protected_attributes: Names of protected attribute columns
            strategy: How to handle discriminatory features
            correlation_threshold: Threshold for automatic identification
            features_to_suppress: Manually specified features to suppress
            noise_scale: Scale of noise to add (for 'noise' strategy)
            n_bins: Number of bins (for 'bin' strategy)
        """
        super().__init__(protected_attributes)
        self.strategy = strategy
        self.correlation_threshold = correlation_threshold
        self.features_to_suppress = features_to_suppress or []
        self.noise_scale = noise_scale
        self.n_bins = n_bins
        self._identified_features: List[str] = []
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._mask_values: Dict[str, float] = {}
        # Fitted noise scale per column for strategy='noise'. Absent means THIS
        # fit could not estimate one (see fit()), so transform() adds nothing and
        # the column is not claimed as suppressed.
        self._noise_scales: Dict[str, float] = {}

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "FeatureSuppressor":
        """
        Fit the suppressor by identifying discriminatory features.

        Args:
            X: Feature matrix
            y: Optional target variable (not used)
            protected_attributes: Names of protected attribute columns

        Returns:
            self: The fitted suppressor
        """
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        feature_cols = self._extract_feature_columns(df, attrs)

        self.feature_names_in_ = list(df.columns)

        # Identify discriminatory features (only those actually present in the input)
        self._identified_features = [f for f in self.features_to_suppress if f in feature_cols]

        # BGL stage 2 (2026-09-16): `max_corr = 0.0` was BOTH the starting value
        # and the answer for a column nothing could be measured on, so a
        # single-level protected attribute produced the verdict "nothing to
        # suppress" without any between-group comparison, indistinguishable from
        # a genuine clean result. A column no attribute could be scored against
        # is now a could-not-check: it is neither suppressed nor certified.
        features_not_assessed: List[str] = []
        not_assessed_reasons: Dict[str, str] = {}
        for col in feature_cols:
            if col in self._identified_features:
                continue

            measured: List[float] = []
            why_not: List[str] = []
            for attr in attrs:
                try:
                    feature_vals = df[col].to_numpy(dtype=float)
                except (ValueError, TypeError) as exc:
                    why_not.append(f"{attr}: not convertible to float ({exc.__class__.__name__})")
                    continue
                # Correlation ratio eta for categorical attributes, |Pearson r|
                # for numeric ones. Pearson on integer codes reads ~0 for a
                # non-monotonic nominal association, so a strongly associated
                # feature was never identified for suppression (audit-3 fix).
                assoc, why, _ = _feature_attribute_association_with_reason(feature_vals, df[attr])
                # BGL stage 2b (2026-09-17): a non-finite association used to be
                # counted as an assessment here: `max([nan]) >= threshold` is
                # False, so the column was silently CERTIFIED rather than listed
                # as not assessed. np.isfinite decides, and the reason is kept.
                if assoc is None or not np.isfinite(assoc):
                    why_not.append(f"{attr}: {why or 'the statistic came out non-finite'}")
                    continue
                measured.append(assoc)

            if not measured:
                features_not_assessed.append(col)
                not_assessed_reasons[col] = "; ".join(why_not) or "no protected attribute"
                continue

            if max(measured) >= self.correlation_threshold:
                self._identified_features.append(col)

        if self.strategy == "remove":
            self.feature_names_out_ = [
                c for c in feature_cols if c not in self._identified_features
            ]
        else:
            self.feature_names_out_ = feature_cols.copy()

        # BGL stage 2b (2026-09-17): this `except: pass` left the column with no
        # bin edges, so transform() returned it BYTE-IDENTICAL while
        # features_modified still named it and warnings was empty (measured on an
        # all-NaN column reached through features_to_suppress). A suppression
        # that did not happen is not a suppression.
        suppression_failures: Dict[str, str] = {}
        # BGL stage 3 (2026-09-27): cleared FIRST, so a fit that cannot estimate a
        # suppression parameter leaves none behind rather than inheriting the
        # previous fit's. Measured: fit on a healthy frame stored bin edges
        # [-1.0671, 0.963, 2.985, 5.007, 7.0291]; a REFIT on the same frame with
        # 'proxy' blanked recorded the honest refusal ("could not be binned, so
        # transform() returns them UNCHANGED", features_modified == [],
        # features_not_suppressed == ['proxy']) and then transform() returned the
        # constant 3.0 for all 60 rows by digitizing against THOSE stale edges. A
        # binning estimated from data that is not this fit's data is not a
        # suppression of it, and the disclosure said the opposite of what happened.
        # This is the same shape as CorrelationReducer's _ZtZ_inv (surface grade
        # g002-2), which was the hasattr() version of the same inheritance.
        # BGL stage 4 (2026-09-27): the same reset for the noise scale added below,
        # for the same reason.
        self._bin_edges = {}
        self._mask_values = {}
        self._noise_scales = {}
        for col in self._identified_features:
            if col in df.columns:
                mask_value = float(df[col].mean())
                # A mean over no observation is nan, and writing nan over a column
                # is not a mask: transform() returned the all-NaN column it was
                # given while features_modified named it as suppressed (measured on
                # a 60-row all-NaN column reached through features_to_suppress).
                # Left unset so transform() skips it and it joins the same
                # not-suppressed disclosure the bin failures use.
                if np.isfinite(mask_value):
                    self._mask_values[col] = mask_value
                elif self.strategy == "mask":
                    suppression_failures[col] = (
                        "the column has no finite value, so its mean is nan and there is "
                        "no mask value to write"
                    )
                if self.strategy == "noise":
                    # BGL stage 4 (2026-09-27): the refusal above is written
                    # `elif self.strategy == "mask"`, so it could not fire for
                    # 'noise', the one strategy with no degeneracy check at all.
                    # The perturbation is noise_scale * nanstd(column), which is
                    # exactly 0.0 for a constant column and nan for a column with
                    # no observation, so the "suppressed" column came back
                    # byte-identical. Measured with features_to_suppress=['proxy']
                    # on 60 rows: a constant column of 7.0 gave
                    # features_modified ['proxy'], features_not_suppressed [] and
                    # output identical to input True; an all-NaN column gave the
                    # same record with numpy's "Degrees of freedom <= 0" as the
                    # only signal. Now both record
                    # features_not_suppressed ['proxy'], features_modified [] and
                    # the "could not be suppressed" disclosure, and a measurable
                    # column (sd 2.99 on the two-group frame) is unaffected: it
                    # still reports features_modified ['proxy'] and its output
                    # moves. The scale is FITTED here rather than recomputed per
                    # transform() call, so the parameter the disclosure describes
                    # is the parameter transform() applies.
                    col_values: Optional[np.ndarray] = None
                    try:
                        col_values = df[col].to_numpy(dtype=float)
                        col_sd = float(np.nanstd(col_values))
                    except (ValueError, TypeError) as exc:
                        col_values = None
                        col_sd = float("nan")
                        suppression_failures[col] = (
                            f"the column is not convertible to float "
                            f"({exc.__class__.__name__}), so no noise scale could be "
                            "estimated"
                        )
                    scale = self.noise_scale * col_sd
                    # BGL6 F02 (2026-09-29): `scale > 0.0` sits ONE ULP above the
                    # hole it was written to close. A positive scale below the
                    # float64 spacing of the values it is added to is lost in the
                    # rounding, so the "suppressed" column still came back
                    # byte-identical with features_modified naming it and
                    # features_not_suppressed empty. Measured on 60 rows through
                    # features_to_suppress=['proxy'], strategy='noise':
                    #   two distinct values at a magnitude of 1e16 (nanstd 1.41,
                    #   fitted scale 0.141, an entirely ordinary looking parameter;
                    #   epoch nanosecond timestamps sit at that magnitude)
                    #     before -> output identical to input True,
                    #               features_modified ['proxy'],
                    #               features_not_suppressed [], no warning
                    #   a near-constant column one ulp wide (scale 2.8e-18)
                    #     before -> the same record
                    # The comparison is against np.spacing of each row's own
                    # magnitude, not a fixed epsilon, because the resolution IS a
                    # function of the magnitude. Refused only when NO finite row
                    # can move: a column of mixed magnitudes where the noise is
                    # representable somewhere is genuinely perturbed there, and
                    # refusing it would throw a real suppression away.
                    n_below_resolution = 0
                    n_finite_rows = 0
                    if col_values is not None and np.isfinite(scale) and scale > 0.0:
                        finite_vals = col_values[np.isfinite(col_values)]
                        n_finite_rows = int(finite_vals.size)
                        if n_finite_rows:
                            spacing = np.spacing(np.abs(finite_vals))
                            n_below_resolution = int(np.count_nonzero(scale < spacing))
                    if not (np.isfinite(scale) and scale > 0.0):
                        if col not in suppression_failures:
                            suppression_failures[col] = (
                                f"the noise scale is {scale!r}: noise_scale={self.noise_scale} "
                                f"times a within-column standard deviation of {col_sd!r} "
                                "perturbs nothing, so the column would come back unchanged"
                            )
                    elif n_finite_rows and n_below_resolution == n_finite_rows:
                        widest = float(np.nanmax(np.abs(col_values)))
                        suppression_failures[col] = (
                            f"the noise scale is {scale!r}, which is below the float64 "
                            f"spacing of every one of the {n_finite_rows} finite value(s) "
                            f"in this column (its largest magnitude is {widest:.6g}, whose "
                            f"spacing is {float(np.spacing(widest))!r}): the perturbation is "
                            "lost in the rounding, so the column would come back unchanged"
                        )
                    else:
                        self._noise_scales[col] = scale
                if self.strategy == "bin":
                    try:
                        _, edges = pd.cut(df[col].dropna(), bins=self.n_bins, retbins=True)
                        self._bin_edges[col] = edges
                    except (ValueError, TypeError) as exc:
                        suppression_failures[col] = f"{exc.__class__.__name__}: {exc}"

        correlation_before, before_reasons = self._measure_with_reasons(
            df, feature_cols, attrs, stage="before"
        )

        # Fitted before the after-measurement below, which goes through transform().
        self.is_fitted = True

        # audit-6 lane 2 (2026-09-09): correlation_after was never populated by
        # this fit, and the before/after chart then drew a 0.000 "After" bar for
        # a value nobody computed. It is measured now on the fit-time output for
        # the deterministic strategies ('remove', 'mask', 'bin'). 'noise' draws
        # fresh, unseeded noise on every transform() call, so a fit-time draw
        # would not describe the output a caller receives: it stays unmeasured
        # and the result says so.
        correlation_after: Dict[str, float] = {}
        result_warnings: List[str] = list(before_reasons)
        if features_not_assessed:
            detail = "; ".join(
                f"{c} ({not_assessed_reasons.get(c, 'no reason recorded')})"
                for c in features_not_assessed[:4]
            )
            result_warnings.append(
                f"suppression not assessed for {len(features_not_assessed)} feature(s) "
                f"{features_not_assessed[:8]}: no protected attribute could be scored "
                "against them, so they were neither suppressed nor certified as clean. "
                f"Reasons: {detail}" + (" ..." if len(features_not_assessed) > 4 else "")
            )
        if suppression_failures:
            result_warnings.append(
                f"{len(suppression_failures)} feature(s) "
                f"{sorted(suppression_failures)[:8]} could not be suppressed with "
                f"strategy={self.strategy!r}, so transform() returns them UNCHANGED; they "
                "are excluded from features_modified rather than reported as suppressed. "
                "Reasons: "
                + "; ".join(f"{c}: {r}" for c, r in sorted(suppression_failures.items())[:4])
            )
            warnings.warn(f"FeatureSuppressor: {result_warnings[-1]}")
        if self.strategy == "noise":
            result_warnings.append(
                "correlation_after not measured: strategy='noise' draws fresh noise on "
                "every transform() call, so no single measurement describes the output."
            )
        elif not self.feature_names_out_:
            result_warnings.append(
                "correlation_after not measured: no feature columns remain after suppression."
            )
        else:
            try:
                df_after = self._to_dataframe(self.transform(df), self.feature_names_out_)
                for attr in attrs:
                    df_after[attr] = df[attr].values
                # BGL stage 2b (2026-09-17): these reasons were discarded, so an
                # after-measurement that REFUSED (every numeric feature removed,
                # only non-numeric columns left) published correlation_after {}
                # with attributes_not_measured [] and warnings silent about it,
                # while the identical refusal on the before side was carried.
                correlation_after, after_reasons = self._measure_with_reasons(
                    df_after, self.feature_names_out_, attrs, stage="after"
                )
                result_warnings.extend(after_reasons)
            except (ValueError, TypeError) as exc:
                result_warnings.append(f"correlation_after not measured: {exc}")

        self.fit_result = TransformationResult(
            method=f"feature_suppression_{self.strategy}",
            n_features_original=len(feature_cols),
            n_features_transformed=len(self.feature_names_out_),
            n_samples=len(df),
            correlation_before=correlation_before,
            correlation_after=correlation_after,
            features_removed=self._identified_features if self.strategy == "remove" else [],
            features_modified=(
                [c for c in self._identified_features if c not in suppression_failures]
                if self.strategy != "remove"
                else []
            ),
            fit_metrics={
                "correlation_threshold": self.correlation_threshold,
                "attributes_measured": sorted(correlation_before),
                "attributes_not_measured": [a for a in attrs if a not in correlation_before],
                "attributes_measured_after": sorted(correlation_after),
                "attributes_not_measured_after": [a for a in attrs if a not in correlation_after],
                "features_not_assessed": features_not_assessed,
                "n_features_not_assessed": len(features_not_assessed),
                "features_not_assessed_reasons": not_assessed_reasons,
                "features_not_suppressed": sorted(suppression_failures),
            },
            warnings=result_warnings,
        )

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform features by suppressing discriminatory features.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()

        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        feature_cols = self._extract_feature_columns(df, self.protected_attributes)

        if self.strategy == "remove":
            result_cols = [c for c in feature_cols if c not in self._identified_features]
            result = df[result_cols].values
        else:
            result = df[feature_cols].values.astype(float).copy()

            for i, col in enumerate(feature_cols):
                if col not in self._identified_features:
                    continue

                if self.strategy == "mask":
                    mask_value = self._mask_values.get(col)
                    # No mask value means THIS fit could not compute one (the mean of
                    # a column with no observation is nan) and published the column in
                    # features_not_suppressed. `.get(col, 0.0)` would write an invented
                    # 0.0 into it, a fabricated mask rather than a refused one.
                    if mask_value is None:
                        continue
                    result[:, i] = mask_value

                elif self.strategy == "noise":
                    # BGL stage 4 (2026-09-27): the scale used to be recomputed
                    # here as `self.noise_scale * np.nanstd(result[:, i])`, which is
                    # 0.0 for a constant column and nan for an all-NaN one, so this
                    # line added exactly nothing and returned the column
                    # byte-identical inside a column the fit record named as
                    # suppressed (measured: 60 rows of 7.0 in, the same 60 values
                    # out, features_modified ['proxy']). The fitted scale is used
                    # instead, and its absence means this fit refused the column and
                    # published it in features_not_suppressed, so there is nothing
                    # to apply. `.get(col, <anything>)` would invent a perturbation
                    # the fit never estimated, the same shape as the mask branch's
                    # old `.get(col, 0.0)`.
                    scale = self._noise_scales.get(col)
                    if scale is None:
                        continue
                    noise = np.random.normal(0, scale, size=len(result))
                    result[:, i] += noise

                elif self.strategy == "bin":
                    edges = self._bin_edges.get(col)
                    if edges is not None:
                        result[:, i] = np.digitize(result[:, i], edges[1:-1])

        if was_dataframe:
            return pd.DataFrame(result, columns=self.feature_names_out_, index=df.index)
        return result

    def _compute_correlations(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
    ) -> Dict[str, float]:
        """Average feature/attribute association per protected attribute.

        Uses the correlation ratio eta for categorical attributes and
        |Pearson r| for numeric ones (see _feature_attribute_association), the
        same honest measure used for identification (audit-3 fix).
        """
        return self._measure_with_reasons(df, feature_cols, protected_attrs)[0]

    def _measure_with_reasons(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
        stage: str = "",
    ) -> Tuple[Dict[str, float], List[str]]:
        """``_compute_correlations`` plus the reasons any attribute was omitted.

        ``stage`` labels which side of the before/after report is being measured,
        so the two refusals are distinguishable in ``warnings`` instead of being
        byte-identical strings (which also made the Python warning look like one
        duplicated emission rather than two separate measurements).
        """
        owner = f"{self.__class__.__name__} ({stage})" if stage else self.__class__.__name__
        return _measure_associations(owner, df, feature_cols, protected_attrs)


class ResidualTransformer(BaseFeatureTransformer):
    """
    Creates residualized features controlling for protected attributes.

    Implements the approach from Feldman et al. (2015) for removing
    disparate impact by conditioning features on protected attributes.

    The transformation computes: X_fair = X - E[X | A]

    where A is the protected attribute and E[X | A] is the conditional
    expectation of X given A.

    Attributes:
        method: Only 'group_mean' is implemented: E[X | A] is the per-group
            mean of the FIRST protected attribute (further attributes are not
            conditioned on; the fit records which one was used and warns).
            'regression' and 'quantile' are refused at construction: they were
            accepted, silently ran group_mean, and labelled the result
            'residualization_regression' / 'residualization_quantile'
            (measured 2026-09-09).
        smoothing: Smoothing parameter for conditional means

    Example:
        >>> transformer = ResidualTransformer(
        ...     protected_attributes=['gender'],
        ...     method='group_mean'
        ... )
        >>> X_fair = transformer.fit_transform(X)

    References:
        - Feldman et al. (2015): Certifying and Removing Disparate Impact

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: residual_transform. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        method: Literal["group_mean"] = "group_mean",
        smoothing: float = 0.0,
    ):
        """
        Initialize the residual transformer.

        Args:
            protected_attributes: Names of protected attribute columns
            method: Method for computing conditional expectation; only
                'group_mean' exists (anything else raises NotImplementedError)
            smoothing: Smoothing parameter for conditional means
        """
        super().__init__(protected_attributes)
        # audit-6 lane 2 (2026-09-09): fit/transform never branched on method, so
        # every value ran group_mean while fit_result.method reported
        # f"residualization_{method}" (even 'not_a_method'). Refuse what does
        # not exist instead of relabelling it.
        if method != "group_mean":
            raise NotImplementedError(
                f"ResidualTransformer(method={method!r}) is not implemented; only "
                "'group_mean' (X - E[X | A] + E[X], per-group means of the first "
                "protected attribute) exists."
            )
        self.method = method
        self.smoothing = smoothing
        self._group_means: Dict[str, Dict[Any, float]] = {}
        self._global_means: Dict[str, float] = {}
        # Group values of the conditioning attribute seen at fit, so transform()
        # can name the ones it has no conditional mean for.
        self._fitted_groups: List[Any] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "ResidualTransformer":
        """
        Fit the residual transformer.

        Args:
            X: Feature matrix
            y: Optional target variable (not used)
            protected_attributes: Names of protected attribute columns

        Returns:
            self: The fitted transformer
        """
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        feature_cols = self._extract_feature_columns(df, attrs)

        self.feature_names_in_ = list(df.columns)
        self.feature_names_out_ = feature_cols.copy()

        # For simplicity, use the first protected attribute
        # (could extend to handle intersections)
        attr = attrs[0]

        groups_without_a_mean: Dict[str, List[Any]] = {}
        for col in feature_cols:
            self._global_means[col] = float(df[col].mean())
            self._group_means[col] = {}

            for group in df[attr].unique():
                group_data = df[df[attr] == group][col]
                if len(group_data) > 0:
                    group_mean = float(group_data.mean())
                    if self.smoothing > 0:
                        global_mean = self._global_means[col]
                        group_mean = (
                            self.smoothing * global_mean + (1 - self.smoothing) * group_mean
                        )
                    # A LENGTH IS NOT A MEAN (BGL6 F02, 2026-09-28). The guard
                    # above is `len(group_data) > 0`, and a column holding twenty
                    # NaNs has length twenty and no mean, so a group whose FEATURE
                    # column carries no observation was recorded with a conditional
                    # mean of nan. It then counted toward the "fewer than 2 groups
                    # have a conditional mean" disclosure, which therefore did not
                    # fire, it stayed in _fitted_groups so transform's own refusal
                    # could not fire either, the column stayed in
                    # features_modified as residualized, and its rows came back nan
                    # with ZERO warnings on either channel. Measured on 60 rows,
                    # groups a and b with real values and c with twenty NaNs:
                    # _group_means['f']['c'] nan, _fitted_groups ['a','b','c'],
                    # features_modified ['f'], fit_result.warnings [].
                    #
                    # A group with no conditional mean is not fitted for that
                    # column, so it is not recorded, and it is named below.
                    if math.isfinite(group_mean):
                        self._group_means[col][group] = group_mean
                    else:
                        groups_without_a_mean.setdefault(col, []).append(group)

        # BGL stage 4 (2026-09-27): this was ``list(pd.unique(df[attr]))``, which
        # keeps nan although the loop above stores NO conditional mean for it
        # (``df[df[attr] == nan]`` is empty because nan is not equal to itself).
        # transform()'s refusal is ``[g for g in pd.unique(df[attr]) if g not in
        # self._fitted_groups]``, so with nan recorded as a fitted group that guard
        # could not fire for the rows that have no conditional mean at all.
        # Measured on 60 rows whose 'gender' held two real levels plus 10 missing:
        # _group_means['f'] keys ['a', 'b'], _fitted_groups [nan, 'a', 'b'],
        # features_modified ['f'], fit_result.warnings [], zero Python warnings,
        # and max|out - in| over the 10 groupless rows exactly 0.0 while the other
        # 50 moved by up to 9.727115410387928. Now _fitted_groups is ['a', 'b'],
        # both fit() and transform() count those rows and name them, and the
        # measurable frame is untouched (two real levels, no missing value: the
        # same features_modified and no extra warning).
        attr_missing = df[attr].isna().to_numpy()
        n_rows_without_group = int(attr_missing.sum())
        self._fitted_groups = list(pd.unique(df[attr][~attr_missing]))
        # A group with no conditional mean for ANY column cannot be residualized at
        # all, so it leaves _fitted_groups and transform's refusal reaches its rows.
        # One that has a mean for some columns and not others stays, and the columns
        # it is missing from are named in the warning below.
        self.groups_without_a_conditional_mean_ = {
            col: list(groups) for col, groups in groups_without_a_mean.items()
        }
        _no_mean_anywhere = {
            g
            for col in feature_cols
            for g in groups_without_a_mean.get(col, [])
            if not any(g in self._group_means.get(c, {}) for c in feature_cols)
        }
        if _no_mean_anywhere:
            self._fitted_groups = [g for g in self._fitted_groups if g not in _no_mean_anywhere]

        # BGL stage 3 (2026-09-27): ``features_modified=feature_cols`` was
        # unconditional. E[X | A] is estimated per group of ONE attribute, so with
        # fewer than 2 groups carrying data for a column the conditional mean IS
        # the global mean and ``X - E[X|A] + E[X]`` returns the column it was given.
        # Measured on a 60-row frame whose 'gender' held a single level: the largest
        # |out - in| across both feature columns was 1.11e-16 (float round-trip
        # noise), so nothing was residualized, while features_modified read
        # ['f', 'g'] and no channel said the conditioning had no contrast. The
        # sibling CorrelationReducer.fit already excludes a column it could build no
        # reduction for (surface grade g002-1); this is the same disclosure for the
        # residual case, and it looks at the intervention's OWN parameters (how many
        # conditional means it has) rather than at the fairness number it produces.
        # With a MISSING group value in the mix the one group mean is NOT the global
        # mean and the column does move (measured 0.0391 on the same frame with 10 of
        # the 60 'gender' values blanked), so the disclosure says no between-group
        # difference was removed rather than claiming the column came back unchanged.
        features_not_residualized = [
            c for c in feature_cols if len(self._group_means.get(c, {})) < 2
        ]

        # BGL stage 2 (2026-09-16): the reasons travel with the numbers now. A
        # single-level attribute used to publish correlation_before={'race': 0.0},
        # correlation_after={'race': 0.0} and a 0.0 reduction, graded and plotted,
        # with warnings == [] - a statistic that is 0.0 by construction, not by
        # measurement (one group's mean IS the grand mean).
        correlation_before, before_reasons = self._measure_with_reasons(
            df, feature_cols, attrs, stage="before"
        )

        # Fitted before the after-measurement below, which goes through transform().
        self.is_fitted = True

        # audit-6 lane 2 (2026-09-09): correlation_after was never populated by
        # this fit, and the before/after chart then drew a 0.000 "After" bar for
        # a value nobody computed. group_mean is deterministic, so it is measured
        # here on the fit-time output. Only the first protected attribute is
        # conditioned on; the others keep their association and the result says so.
        result_warnings: List[str] = list(before_reasons)
        if len(attrs) > 1:
            result_warnings.append(
                f"Residualized against {attr!r} only; {attrs[1:]} are not conditioned on "
                "and their association is expected to remain."
            )
        # NAME THE GROUPS THAT HAVE NO CONDITIONAL MEAN, on the same channel as the
        # rest. See the note at the `math.isfinite(group_mean)` guard: before this,
        # such a group was recorded with a nan mean and nothing said so.
        if self.groups_without_a_conditional_mean_:
            _detail = "; ".join(
                f"{col!r}: {sorted(map(str, groups))}"
                for col, groups in sorted(self.groups_without_a_conditional_mean_.items())
            )
            result_warnings.append(
                f"group(s) of {attr!r} carry no observation of a feature column, so "
                f"they have NO conditional mean and their rows are not residualized "
                f"for it ({_detail}). A nan mean is not a mean: those rows are a "
                f"could-not-check rather than a group that needed no adjustment."
            )

        if features_not_residualized:
            result_warnings.append(
                f"fewer than 2 groups of {attr!r} have a conditional mean on "
                f"{len(features_not_residualized)} of {len(feature_cols)} feature(s) "
                f"{features_not_residualized[:8]}"
                + (" ..." if len(features_not_residualized) > 8 else "")
                + ": with no second group mean there is no between-group difference to "
                "remove, so nothing was residualized against the protected attribute "
                "there. Whatever shift transform() applies to those columns rests on no "
                "group comparison, and their association with the protected attribute(s) "
                "is unchanged. They are left out of features_modified rather than "
                "reported as residualized. Treat them as could-not-check."
            )
            warnings.warn(f"ResidualTransformer: {result_warnings[-1]}")
        if n_rows_without_group:
            # The count itself, on the fit record. The sibling
            # DisparateImpactRemover.fit already publishes
            # n_rows_without_a_recorded_group and says those rows come back
            # UNREPAIRED; this is the same disclosure for the residual case. The
            # columns stay in features_modified because the 50 rows that DO carry a
            # group were genuinely residualized (they moved by up to 9.7275 on the
            # measured frame); withdrawing the column would delete that real
            # measurement. fit_metrics stays {"conditioned_on": attr} because an
            # existing pin asserts that dict whole.
            result_warnings.append(
                f"{n_rows_without_group} of {len(df)} row(s) have a missing value in "
                f"{attr!r}, so they belong to no known group and E[X | A] does not "
                "exist for them: transform() returns them UN-RESIDUALIZED in the same "
                "columns as the residualized rows, and their association with the "
                "protected attribute is unchanged. Treat those rows as "
                "could-not-check, not as residuals."
            )
            warnings.warn(f"ResidualTransformer: {result_warnings[-1]}")
        correlation_after: Dict[str, float] = {}
        try:
            df_after = self._to_dataframe(self.transform(df), feature_cols)
            for a in attrs:
                df_after[a] = df[a].values
            # BGL stage 2b (2026-09-17): the AFTER-side reasons were discarded
            # here. Measured with one np.inf cell: correlation_after == {} (the
            # after measurement refused), the reason reached a Python warning
            # only, and fit_result.warnings == [] - exactly the channel gap
            # _measure_associations' own docstring says must not happen.
            correlation_after, after_reasons = self._measure_with_reasons(
                df_after, feature_cols, attrs, stage="after"
            )
            result_warnings.extend(after_reasons)
        except (ValueError, TypeError) as exc:
            result_warnings.append(f"correlation_after not measured: {exc}")

        self.fit_result = TransformationResult(
            method=f"residualization_{self.method}",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            correlation_before=correlation_before,
            correlation_after=correlation_after,
            features_modified=[c for c in feature_cols if c not in features_not_residualized],
            # fit_metrics stays exactly {"conditioned_on": attr}: an existing pin
            # (test_audit6_lane2_feature_engineering.py) asserts that dict whole,
            # and the refusal has its own channel anyway. An attribute that could
            # not be measured is ABSENT from correlation_before/after and named,
            # with the reason, in warnings below - the same three-state shape the
            # sibling transformers in this file use.
            fit_metrics={"conditioned_on": attr},
            warnings=result_warnings,
        )

        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform features by computing residuals.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()

        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        feature_cols = self._extract_feature_columns(df, self.protected_attributes)
        attr = self.protected_attributes[0]

        # audit-6 lane 2 (2026-09-09): a group value never seen at fit has no
        # conditional mean; `.get(group, global_mean)` below leaves its rows
        # UN-residualized (value - global + global). That is not a measurement
        # of anything, so it is named here rather than passed off silently.
        # BGL stage 4 (2026-09-27): a MISSING protected value is not an unseen
        # LEVEL, and it used to reach neither disclosure: nan was recorded in
        # _fitted_groups by fit(), so `g not in self._fitted_groups` was False for
        # it and this warning could not fire for the rows that have no conditional
        # mean at all. Measured on 60 rows with 10 missing 'gender' values: those 10
        # came back byte-identical (|out - in| exactly 0.0) inside the residual
        # column with zero warnings on either channel. The two states are counted
        # separately now, because "a level the fit never saw" and "no value at all"
        # are different could-not-checks.
        attr_missing = df[attr].isna().to_numpy()
        n_rows_without_group = int(attr_missing.sum())
        unseen = [g for g in pd.unique(df[attr][~attr_missing]) if g not in self._fitted_groups]
        if unseen:
            # TWO REASONS, AND THE MESSAGE MUST NOT PICK THE WRONG ONE (BGL6 F02,
            # 2026-09-28). A value can be absent from _fitted_groups because it was
            # never at fit, or because fit SAW it and could compute no conditional
            # mean for it (its feature column carried no observation). Saying "not
            # seen at fit" for the second case misdescribes what a reader is being
            # told about their own data, which is the same defect as a disclosure
            # that names the wrong cause.
            no_mean = {
                g
                for groups in getattr(self, "groups_without_a_conditional_mean_", {}).values()
                for g in groups
            }
            seen_but_unmeasurable = [g for g in unseen if g in no_mean]
            never_seen = [g for g in unseen if g not in no_mean]
            if never_seen:
                warnings.warn(
                    f"ResidualTransformer: {len(never_seen)} value(s) of {attr!r} were not "
                    f"seen at fit ({never_seen[:8]}); their rows have no conditional mean "
                    "and pass through un-residualized."
                )
            if seen_but_unmeasurable:
                warnings.warn(
                    f"ResidualTransformer: {len(seen_but_unmeasurable)} value(s) of "
                    f"{attr!r} were seen at fit but carry no observation of a feature "
                    f"column ({seen_but_unmeasurable[:8]}), so no conditional mean could "
                    "be computed for them and their rows pass through un-residualized. "
                    "That is a could-not-check for those rows, not a group that needed "
                    "no adjustment."
                )
        if n_rows_without_group:
            warnings.warn(
                f"ResidualTransformer: {n_rows_without_group} of {len(df)} row(s) have no "
                f"value for {attr!r}, so they belong to no known group and have no "
                "conditional mean: they pass through un-residualized in the same columns "
                "as the residualized rows. That is a could-not-check for those rows, not "
                "a residual."
            )

        result = np.zeros((len(df), len(feature_cols)))

        for i, col in enumerate(feature_cols):
            for j, (idx, row) in enumerate(df.iterrows()):
                group = row[attr]
                group_mean = self._group_means[col].get(group, self._global_means[col])
                result[j, i] = row[col] - group_mean + self._global_means[col]

        if was_dataframe:
            return pd.DataFrame(result, columns=feature_cols, index=df.index)
        return result

    def _compute_correlations(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
    ) -> Dict[str, float]:
        """Average feature/attribute association per protected attribute.

        Uses the correlation ratio eta for categorical attributes and
        |Pearson r| for numeric ones (see _feature_attribute_association), the
        same honest measure as the sibling transformers. audit-6 lane 2
        (2026-09-09): this class still used Pearson on integer category codes,
        which reads ~0 for an untouched non-monotonic nominal association; now
        that correlation_after is populated here, that measure would have
        reported a false success for any protected attribute this transformer
        does not condition on.
        """
        return self._measure_with_reasons(df, feature_cols, protected_attrs)[0]

    def _measure_with_reasons(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        protected_attrs: List[str],
        stage: str = "",
    ) -> Tuple[Dict[str, float], List[str]]:
        """``_compute_correlations`` plus the reasons any attribute was omitted.

        ``stage`` labels which side of the before/after report is being measured,
        so the two refusals are distinguishable in ``warnings`` instead of being
        byte-identical strings (which also made the Python warning look like one
        duplicated emission rather than two separate measurements).
        """
        owner = f"{self.__class__.__name__} ({stage})" if stage else self.__class__.__name__
        return _measure_associations(owner, df, feature_cols, protected_attrs)


class IntersectionalTransformer(BaseFeatureTransformer):
    """
    Handles intersectional fairness in feature engineering.

    Addresses the interaction effects between multiple protected attributes
    (e.g., race AND gender together) by creating fair representations
    that account for intersectional subgroups.

    This transformer:
    1. Creates intersectional group indicators
    2. Ensures minimum representation across subgroups
    3. Applies group-specific transformations

    Attributes:
        min_group_size: Minimum samples per intersectional group
        handle_small_groups: Strategy for small groups

    Example:
        >>> transformer = IntersectionalTransformer(
        ...     protected_attributes=['gender', 'race'],
        ...     min_group_size=30
        ... )
        >>> X_fair = transformer.fit_transform(X)

    References:
        - Crenshaw (1989): Demarginalizing the Intersection of Race and Sex
        - Buolamwini & Gebru (2018): Gender Shades

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: intersectional_transform. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        min_group_size: int = 30,
        handle_small_groups: Literal["merge", "keep", "exclude"] = "merge",
    ):
        """
        Initialize the intersectional transformer.

        Args:
            protected_attributes: Names of protected attribute columns
            min_group_size: Minimum samples per intersectional group
            handle_small_groups: Strategy for handling small groups
        """
        super().__init__(protected_attributes)
        self.min_group_size = min_group_size
        self.handle_small_groups = handle_small_groups
        # group value-tuple -> merged label, or None when the group is excluded.
        self._intersectional_groups: Dict[Any, Optional[str]] = {}
        # feature col -> {label -> {'mean'/'std'/'count'/'transformable' -> value}}.
        self._group_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Populated by transform(): which rows came through UNTRANSFORMED.
        self.rows_not_transformed_: np.ndarray = np.array([], dtype=int)
        self.groups_not_transformed_: List[str] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "IntersectionalTransformer":
        """
        Fit the intersectional transformer.

        Args:
            X: Feature matrix
            y: Optional target variable (not used)
            protected_attributes: Names of protected attribute columns

        Returns:
            self: The fitted transformer
        """
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        feature_cols = self._extract_feature_columns(df, attrs)

        self.feature_names_in_ = list(df.columns)
        self.feature_names_out_ = feature_cols.copy()

        df["_intersect"] = df[attrs].apply(lambda row: tuple(row.values), axis=1)

        group_counts = df["_intersect"].value_counts()

        group_mapping: Dict[Any, Optional[str]] = {}
        for group, count in group_counts.items():
            if count >= self.min_group_size:
                group_mapping[group] = "_".join(str(v) for v in group)
            elif self.handle_small_groups == "merge":
                group_mapping[group] = "_OTHER_"
            elif self.handle_small_groups == "keep":
                group_mapping[group] = "_".join(str(v) for v in group)
            else:  # exclude
                group_mapping[group] = None

        self._intersectional_groups = group_mapping

        # BGL stage 2b (2026-09-17): the stats were keyed by LABEL and written
        # only `if label not in self._group_stats[col]`, so with two or more small
        # groups merged under '_OTHER_' the FIRST one's mean/std stood in for all
        # of them. Measured on two 5-row groups centred at 0 and at 500: the
        # second came out with mean 74446.97 against 16.61 for every other group,
        # with transformable=True, rows_not_transformed_=[], fit_result.warnings
        # ==[] and zero Python warnings. Rows that SHARE a label are pooled, which
        # is what a merged label means; the pooling itself is disclosed, because a
        # merged group's statistics describe none of its members on their own.
        members: Dict[str, List[Any]] = {}
        for group, label in group_mapping.items():
            if label is not None:
                members.setdefault(label, []).append(group)
        label_series = df["_intersect"].map(group_mapping)

        for col in feature_cols:
            self._group_stats[col] = {}
            for label, group_list in members.items():
                group_data = df.loc[(label_series == label).to_numpy(), col]
                if len(group_data) > 0:
                    # BGL stage 2 (2026-09-16): a 1-row group has std NaN
                    # (ddof=1) and `nan > 0` is False, so transform() dropped
                    # the row through RAW while features_modified still named
                    # the column. Whether a group can be normalised is decided
                    # HERE, once, and recorded, instead of being re-derived by
                    # a bare `> 0` test that silently answers False for NaN.
                    std = float(group_data.std())
                    self._group_stats[col][label] = {
                        "mean": float(group_data.mean()),
                        "std": std,
                        "count": len(group_data),
                        "n_groups_pooled": len(group_list),
                        "transformable": bool(
                            len(group_data) >= 2 and np.isfinite(std) and std > 0.0
                        ),
                    }

        df.drop("_intersect", axis=1, inplace=True)

        fit_warnings: List[str] = []
        # Surface grade g002 (2026-09-17): with ONE intersectional label there is
        # no second subgroup to normalise against, so the within-group
        # standardisation below is a whole-sample rescaling and no
        # intersectional adjustment happens at all. Measured on a 50-row frame
        # with one (gender, race) cell: every row came back rescaled by
        # sqrt((n-1)/n), features_modified named the column, fit_result.warnings
        # was [] and no Python warning fired, byte-identical to a run that had
        # four subgroups to level. Every sibling in this file
        # (DisparateImpactRemover, LabelMassager, Resampler,
        # FairRepresentationTransformer) already discloses its own fewer-than-2
        # groups case; this one did not.
        n_labels_used = len(set(group_mapping.values()) - {None})
        if n_labels_used < 2:
            fit_warnings.append(
                f"protected attribute(s) {list(attrs)} yield {n_labels_used} usable "
                "intersectional group(s) in these rows, fewer than the 2 an "
                "intersectional comparison needs: there is no other subgroup to bring "
                "onto a common scale, so NO intersectional adjustment was made and the "
                "output is a plain rescaling of the input. This is a could-not-check, "
                "not a frame whose subgroups were already aligned."
            )
            warnings.warn(f"IntersectionalTransformer: {fit_warnings[-1]}")
        untransformable = sorted(
            {
                label
                for col in feature_cols
                for label, stats in self._group_stats.get(col, {}).items()
                if not stats["transformable"]
            }
        )
        if untransformable:
            fit_warnings.append(
                f"{len(untransformable)} intersectional group(s) {untransformable[:8]}"
                + (" ..." if len(untransformable) > 8 else "")
                + " have no within-group spread to normalise by (fewer than 2 rows, or a "
                "constant value); their rows pass through UNTRANSFORMED on the affected "
                "features and sit on a different scale from the rest of the column."
            )
            warnings.warn(fit_warnings[-1])
        pooled = sorted(
            label
            for label, group_list in members.items()
            if len(group_list) > 1
            and any(label in self._group_stats.get(c, {}) for c in feature_cols)
        )
        if pooled:
            fit_warnings.append(
                f"{len(pooled)} label(s) {pooled[:8]} pool rows from more than one "
                "intersectional group ("
                + "; ".join(f"{lbl}: {len(members[lbl])} groups" for lbl in pooled[:4])
                + "). Their mean and std describe the POOLED rows, not any member "
                "group, so a member whose own centre is far from the pooled centre "
                "does not land on the common scale. Treat those rows as a "
                "could-not-check rather than as normalised."
            )
            warnings.warn(fit_warnings[-1])
        excluded = sorted(
            "_".join(str(v) for v in g) for g, label in group_mapping.items() if label is None
        )
        if excluded:
            fit_warnings.append(
                f"handle_small_groups='exclude' dropped {len(excluded)} group(s) "
                f"{excluded[:8]}"
                + (" ..." if len(excluded) > 8 else "")
                + " from the fitted statistics, but transform() preserves the row count: "
                "their rows are RETURNED UNTRANSFORMED rather than removed."
            )
            warnings.warn(fit_warnings[-1])

        self.fit_result = TransformationResult(
            method="intersectional_transformation",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            features_modified=feature_cols,
            fit_metrics={
                "n_intersectional_groups": n_labels_used,
                "n_original_groups": len(group_counts),
                "intersectional_contrast_measured": bool(n_labels_used >= 2),
                "groups_not_transformable": untransformable,
                "groups_excluded_but_returned": excluded,
                "labels_pooling_several_groups": pooled,
            },
            warnings=fit_warnings,
        )

        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform features accounting for intersectionality.

        Currently implements normalization within intersectional groups.

        Args:
            X: Feature matrix to transform

        Returns:
            Transformed feature matrix
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()

        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        feature_cols = self._extract_feature_columns(df, self.protected_attributes)

        df["_intersect"] = df[self.protected_attributes].apply(
            lambda row: tuple(row.values), axis=1
        )

        result = df[feature_cols].values.astype(float).copy()

        # BGL stage 2 (2026-09-16): both fall-throughs below used to be silent,
        # so one column carried two different scales and the caller was handed it
        # as one transformed matrix. Measured: a 1-row intersection went in at
        # 999.0 and came out at 999.0 while every other group mean was rescaled
        # to 11.131, with fit_result.warnings == [] and the column still named in
        # features_modified. The misses are counted now.
        untouched = np.zeros(len(df), dtype=bool)
        missed_groups: set = set()
        n_non_finite = 0

        for i, col in enumerate(feature_cols):
            global_mean = np.nanmean(result[:, i])
            global_std = np.nanstd(result[:, i])

            for j, (idx, row) in enumerate(df.iterrows()):
                group = row["_intersect"]
                label = self._intersectional_groups.get(group)
                stats = self._group_stats.get(col, {}).get(label) if label else None

                # `stats["std"] > 0` alone answered False for a NaN std without
                # distinguishing it from a genuine zero-spread group; the decision
                # is read from the flag fit() recorded.
                if stats is not None and stats["transformable"]:
                    # BGL stage 2b (2026-09-17): a NaN/inf INPUT inside a
                    # transformable group came out NaN and was counted as
                    # transformed (out row0 = nan, rows_not_transformed_ = [], no
                    # warning). An arithmetic result on a value that was never
                    # observed is not a normalisation of that row.
                    if not np.isfinite(result[j, i]):
                        untouched[j] = True
                        n_non_finite += 1
                        continue
                    z = (result[j, i] - stats["mean"]) / stats["std"]
                    result[j, i] = z * global_std + global_mean
                else:
                    untouched[j] = True
                    missed_groups.add("_".join(str(v) for v in group))

        df.drop("_intersect", axis=1, inplace=True)

        self.rows_not_transformed_ = np.flatnonzero(untouched)
        self.groups_not_transformed_ = sorted(missed_groups)
        # BGL stage 2b (2026-09-17): these two writes used to sit INSIDE the
        # `if untouched.any()` branch, so after transform(degenerate) then
        # transform(clean) the attributes said 0 rows missed while fit_metrics
        # still reported the previous call's n_rows_not_transformed=1 and
        # groups_not_transformed=['X_A']. The two disclosures contradicted each
        # other; both now describe the LAST call.
        if self.fit_result is not None:
            self.fit_result.fit_metrics["n_rows_not_transformed"] = int(untouched.sum())
            self.fit_result.fit_metrics["groups_not_transformed"] = self.groups_not_transformed_
            self.fit_result.fit_metrics["n_values_not_finite"] = n_non_finite
        if untouched.any():
            warnings.warn(
                f"IntersectionalTransformer: {int(untouched.sum())} of {len(df)} row(s) "
                f"were returned UNTRANSFORMED for at least one feature "
                f"(group(s) {self.groups_not_transformed_[:8]}"
                + (" ..." if len(self.groups_not_transformed_) > 8 else "")
                + f"; {n_non_finite} value(s) were not finite on input); they sit on "
                "their raw scale in the same columns as the normalised rows, AND they "
                "are included in the np.nanmean/np.nanstd that SETS the target scale "
                "for every row that was transformed, so they move those values too. "
                "See .rows_not_transformed_ for the row positions."
            )

        if was_dataframe:
            return pd.DataFrame(result, columns=feature_cols, index=df.index)
        return result


class ReweightingTransformer(BaseFeatureTransformer):
    """
    Applies reweighting to address representation imbalances.

    Computes sample weights to balance representation across groups
    and optionally creates weighted feature representations.

    Methods:
        - 'inverse_frequency': Weight inversely proportional to group size.
          Balances group representation only; being constant within each
          group it cannot change any within-group label rate, so it does
          not target demographic parity.
        - 'target_parity': Kamiran & Calders (2012) reweighing. Weights are
          conditioned on (group, label): W(g, c) = P(g) * P(c) / P(g, c),
          computed from the observed joint distribution over the
          intersectional group key, so the weighted per-group label rates
          equalize (demographic parity of the training distribution).
          Requires y in both fit() and get_sample_weights().
        - 'custom': Use provided weights

    Attributes:
        method: Reweighting method
        target_distribution: Target distribution for groups

    Example:
        >>> transformer = ReweightingTransformer(
        ...     protected_attributes=['race'],
        ...     method='inverse_frequency'
        ... )
        >>> transformer.fit(X)
        >>> weights = transformer.get_sample_weights(X)

    References:
        - Kamiran & Calders (2012): Data Preprocessing for Discrimination-Aware Mining

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. The pin was
    sabotage-checked: it was shown to go red when the defect is reintroduced, so it
    can fail. This does NOT establish that its statistics are accurate, nor that the
    pin covers every scenario.

    Ledger row: reweighting. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        method: Literal["inverse_frequency", "target_parity", "custom"] = "inverse_frequency",
        target_distribution: Optional[Dict[Any, float]] = None,
    ):
        """
        Initialize the reweighting transformer.

        Args:
            protected_attributes: Names of protected attribute columns
            method: Reweighting method
            target_distribution: Target distribution for groups (for 'custom' method)
        """
        super().__init__(protected_attributes)
        # audit-6 lane 2 (2026-09-09): a Literal hint is not a runtime check.
        # fit() had no else branch, so any other string fitted "successfully"
        # with group_weights {} and every sample weight came back 1.0 (measured:
        # method='not_a_method' -> is_fitted True, fit_result.method
        # 'reweighting_not_a_method', weights all 1.0). Refuse it here.
        if method not in ("inverse_frequency", "target_parity", "custom"):
            raise ValueError(
                "ReweightingTransformer method must be one of 'inverse_frequency', "
                f"'target_parity' or 'custom', got {method!r}"
            )
        self.method = method
        self.target_distribution = target_distribution
        # Keyed by group value for the group-marginal methods, and by the
        # (group_key, label) tuple for 'target_parity' (Kamiran & Calders).
        self._group_weights: Dict[Any, float] = {}
        # Observed at the LAST fit, so get_sample_weights can say whether the
        # weights it hands back rest on a between-group comparison at all.
        # None means no fit has run yet.
        self._n_groups_observed: Optional[int] = None
        self._n_labels_observed: Optional[int] = None

    def _group_key_series(self, df: pd.DataFrame) -> pd.Series:
        """Intersectional group key joined across ALL protected attributes, so
        'target_parity' equalizes label rates of the *joint* group (e.g.
        race x gender), matching the sibling transformers in this module.
        With one attribute this is just that column (as strings).

        BGL stage 3 (2026-09-27): ``astype(str)`` turned a MISSING protected
        value into the literal level ``"nan"``, so rows whose group is UNKNOWN
        were coined into a group of their own and handed Kamiran & Calders
        weights like any other. Measured on 60 rows of which 10 carried a missing
        'race': the weight keys included ``('nan', 0)`` and ``('nan', 1)`` with
        values 1.291667 and 0.805556, ``n_groups_observed`` read 2 while THREE
        groups carried weights, no weight came back NaN, and
        ``fit_result.warnings == []``. The two siblings in this file that key on a
        group, DisparateImpactRemover and LabelMassager, already mask the key for
        exactly this reason. The key stays missing here, so those rows match no
        fitted weight and come back NaN through ``_lookup_weights`` with its
        disclosure, which is the could-not-check they are.
        """
        attrs = self.protected_attributes
        assert attrs is not None  # set in fit()
        sub = df[list(attrs)]
        missing = sub.isna().any(axis=1)
        if len(attrs) == 1:
            keys = sub[attrs[0]].astype(str)
        else:
            keys = sub.astype(str).agg("|".join, axis=1)
        return keys.mask(missing.to_numpy())

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "ReweightingTransformer":
        """
        Fit the reweighting transformer.

        Args:
            X: Feature matrix
            y: Target labels. Required for 'target_parity' (the Kamiran &
               Calders weights are conditioned on group AND label); unused
               by the other methods.
            protected_attributes: Names of protected attribute columns

        Returns:
            self: The fitted transformer
        """
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        feature_cols = self._extract_feature_columns(df, attrs)

        self.feature_names_in_ = list(df.columns)
        self.feature_names_out_ = feature_cols.copy()

        # Use first protected attribute for the group-marginal methods
        # ('inverse_frequency', 'custom'); 'target_parity' conditions on the
        # intersectional group key AND the label instead.
        attr = attrs[0]
        group_counts = df[attr].value_counts()
        n_samples = len(df)
        n_groups = len(group_counts)
        # What the observed-group count is counted ON. The group-marginal methods
        # key their weights on this first attribute; 'target_parity' keys them on
        # the intersectional group and re-counts below, because a refusal counted
        # on the wrong key reports a could-not-check for a comparison that was
        # made.
        group_key_desc = repr(attr)

        # Reset on refit so the normalization below never mixes stale keys
        # from a previous fit with the new counts.
        self._group_weights = {}
        self._n_groups_observed = n_groups
        self._n_labels_observed = None
        # Observed count per weight key, feeding the sum-to-n normalization.
        key_counts: Dict[Any, int] = {}

        if self.method == "inverse_frequency":
            # Weight inversely proportional to group frequency
            for group, count in group_counts.items():
                self._group_weights[group] = n_samples / (n_groups * count)
                key_counts[group] = int(count)

        elif self.method == "target_parity":
            # Kamiran & Calders (2012) reweighing: label-conditional weights
            # W(g, c) = P(group=g) * P(y=c) / P(group=g, y=c) from the observed
            # joint distribution. A group-constant weight cancels in every
            # group-conditional mean, so conditioning on the label is what lets
            # the weighted per-group label rates equalize (demographic parity
            # of the training distribution). Wave-6 audit fix: the previous
            # group-constant formula could not move demographic parity at all.
            if y is None:
                raise ValueError(
                    "method='target_parity' implements Kamiran & Calders (2012) "
                    "reweighing, which conditions on the label: pass y to fit()."
                )
            y_arr = np.asarray(y)
            if len(y_arr) != n_samples:
                raise ValueError(f"y has length {len(y_arr)}, expected {n_samples} to match X")
            group_series = self._group_key_series(df)
            group_keys = group_series.to_numpy()
            label_values, label_counts = np.unique(y_arr, return_counts=True)
            # BGL6 F02 (2026-09-29): a MISSING label is not a class. np.unique
            # returns nan as a level of its own, so a frame of two real classes
            # plus 10 unlabelled rows published n_labels_observed 3, and a frame
            # with ONE real class plus unlabelled rows published 2, which is at or
            # above the 2 the "no label imbalance was measured" refusal below tests
            # for: the could-not-check was switched off by the very rows that
            # cannot be read. The weights are unaffected, because groupby never
            # forms a cell for a missing label, so no weight was ever keyed on one.
            label_readable = ~pd.isna(label_values)
            label_values = label_values[label_readable]
            label_counts = label_counts[label_readable]
            self._n_labels_observed = int(label_values.size)
            label_count = dict(zip(label_values.tolist(), label_counts.tolist()))
            # value_counts, not np.unique: the key is missing (not the string
            # "nan") for a row with no known group, and np.unique would have to
            # order a float against the group strings.
            group_count = group_series.value_counts().to_dict()
            # BGL stage 3 (2026-09-27): n_groups came from value_counts() on the
            # FIRST protected attribute while these weights are keyed on the
            # INTERSECTIONAL group. A frame whose first attribute has one level
            # and whose joint key has two therefore published
            # n_groups_observed=1, between_group_balance_measured=False and the
            # refusal "the weights below are 1.0 by construction" over weights of
            # 0.961538 / 1.041667 / 1.029412 / 0.972222 that had genuinely
            # equalized the joint-group positive rates (0.433333 and 0.400000 both
            # to 0.416667). A measurement reported as a could-not-check is the
            # same defect running backwards, and get_sample_weights repeated the
            # same false claim over the same non-1.0 array.
            n_groups = len(group_count)
            self._n_groups_observed = n_groups
            if len(attrs) > 1:
                group_key_desc = f"the intersection of {list(attrs)}"
            joint_counts = (
                pd.DataFrame({"g": group_keys, "c": y_arr}).groupby(["g", "c"], sort=False).size()
            )
            for (g, c), n_gc in joint_counts.items():
                self._group_weights[(g, c)] = (group_count[g] * label_count[c]) / (
                    n_samples * int(n_gc)
                )
                key_counts[(g, c)] = int(n_gc)

        elif self.method == "custom":
            if self.target_distribution is None:
                raise ValueError("target_distribution required for 'custom' method")
            for group, target_prob in self.target_distribution.items():
                if group in group_counts:
                    current_prob = group_counts[group] / n_samples
                    self._group_weights[group] = target_prob / current_prob
                    key_counts[group] = int(group_counts[group])
            # audit-6 lane 2 (2026-09-09): a target_distribution that names none
            # of the observed groups used to fit "successfully" with {} weights;
            # one that misses some groups leaves those rows without a weight
            # (NaN in get_sample_weights, see there). Say so at fit time.
            if not self._group_weights:
                raise ValueError(
                    f"target_distribution keys {sorted(map(str, self.target_distribution))} "
                    f"match none of the observed groups {sorted(map(str, group_counts.index))}"
                )
            uncovered = [g for g in group_counts.index if g not in self.target_distribution]
            if uncovered:
                warnings.warn(
                    "ReweightingTransformer(method='custom'): target_distribution has no "
                    f"entry for observed group(s) {uncovered}; their rows will get a NaN "
                    "sample weight (no target share was given), not 1.0."
                )
        else:
            raise ValueError(f"Unsupported reweighting method {self.method!r}")

        # Normalize weights to sum to n_samples over the observed data.
        # (The Kamiran & Calders weights already sum to n_samples exactly:
        # sum over (g, c) of n_gc * W(g, c) = sum n_g * n_c / n = n, so for
        # 'target_parity' this is a mathematical no-op kept for uniformity.)
        total_weight = sum(self._group_weights[k] * key_counts[k] for k in self._group_weights)
        for group in self._group_weights:
            self._group_weights[group] *= n_samples / total_weight

        # Surface grade g002 (2026-09-17): one observed group makes every
        # inverse-frequency weight exactly n / (1 * n) = 1.0, and 1.0 is the
        # "no reweighting needed" value. Measured on a 60-row single-group
        # frame: group_weights {'a': 1.0}, every sample weight 1.0,
        # fit_result.warnings == [] and no Python warning, byte-identical to a
        # frame that genuinely needed no correction. A single observed LABEL
        # does the same to the Kamiran & Calders weights (P(c) = 1, so
        # W(g, c) = 1 for every cell). The sibling InversePropensityWeighter in
        # data_balancing.py already refuses to grade a single-group frame; this
        # is the same disclosure. The weights themselves stay as computed: they
        # are the correct value of the formula, and replacing a computable
        # number with NaN would throw evidence away. What is added is the
        # third state, that no comparison was made.
        # A row whose protected value is MISSING belongs to no group, so no weight
        # was derived for it and get_sample_weights hands it back NaN. Counted on
        # fit_result too, because that is the serialised record.
        # BGL stage 4 (2026-09-27): this was hardcoded to 0 for method='custom', on
        # the ground that its uncovered groups have their own disclosure and would be
        # double counted by the n_samples - sum(key_counts) difference. But
        # 'uncovered' comes from value_counts(), which EXCLUDES NaN, so a row with a
        # missing protected value got no fit-side disclosure of any kind. Measured on
        # 60 rows with 10 missing 'race' values and target_distribution
        # {'a': 0.5, 'b': 0.5}: method='custom' published
        # n_rows_without_a_recorded_group 0 with fit_result.warnings == [] and no
        # Python warning, while method='inverse_frequency' on the SAME frame reported
        # 10 on both channels. The missing values are now counted directly, on the
        # key each method actually groups by, so 'custom' reports 10 as its sibling
        # does and an uncovered-but-present group is still counted only once, by its
        # own disclosure above.
        # BGL6 F02 (2026-09-29): for 'target_parity' the key is (group, label) and
        # groupby drops a row whose LABEL is missing as readily as one whose group
        # is, so the n_samples - sum(key_counts) difference counted both and the
        # disclosure named only the first. Measured on 60 rows whose 'race' column
        # has no missing value at all and whose y carries 10 NaNs:
        #   before -> n_rows_without_a_recorded_group 10 and, on both channels,
        #             "10 row(s) have a missing value in ['race'], so they belong to
        #             no known group". Those rows have a recorded group. The count
        #             of rows with no weight was right; the condition it named was
        #             not present in the data, which would send a reader to the
        #             protected attribute to fix a gap in the label column.
        #   after  -> n_rows_without_a_recorded_group 0,
        #             n_rows_without_a_recorded_label 10, and the disclosure names
        #             the missing LABEL. The two are counted separately, and a row
        #             missing BOTH is counted once, under the group.
        n_rows_without_label: Optional[int] = None
        if self.method == "target_parity":
            n_rows_without_group = int(group_series.isna().sum())
            n_rows_without_weight = int(n_samples - sum(key_counts.values()))
            n_rows_without_label = int(n_rows_without_weight - n_rows_without_group)
        else:
            n_rows_without_group = int(df[attr].isna().sum())

        fit_warnings: List[str] = []
        if n_rows_without_group:
            fit_warnings.append(
                f"{n_rows_without_group} row(s) have a missing value in {list(attrs)}, so they "
                "belong to no known group: no weight was derived for them and "
                "get_sample_weights returns NaN for those rows rather than a weight "
                "computed against a group inferred from the missingness itself."
            )
            warnings.warn(f"ReweightingTransformer: {fit_warnings[-1]}")
        if n_rows_without_label:
            fit_warnings.append(
                f"{n_rows_without_label} row(s) carry a MISSING LABEL. The Kamiran & "
                "Calders weights are conditioned on (group, label), so a row with no "
                "label belongs to no (group, label) cell: no weight was derived for it "
                "and get_sample_weights returns NaN for those rows. Those rows DO have a "
                "recorded group; they are counted under n_rows_without_a_recorded_label, "
                "not under n_rows_without_a_recorded_group, and the label rates below "
                "are rates over the rows that carry a label."
            )
            warnings.warn(f"ReweightingTransformer: {fit_warnings[-1]}")
        if n_groups < 2:
            fit_warnings.append(
                f"{group_key_desc} has {n_groups} observed group(s) in these rows, fewer than the "
                "2 a representation imbalance needs: nothing was compared, so the weights "
                "below are 1.0 by construction rather than because the groups were already "
                "balanced. This is a could-not-check, not a balanced sample."
            )
            warnings.warn(f"ReweightingTransformer: {fit_warnings[-1]}")
        if self.method == "target_parity" and (self._n_labels_observed or 0) < 2:
            fit_warnings.append(
                f"y carries {self._n_labels_observed} observed label(s), fewer than the 2 a "
                "positive-rate difference needs: P(y=c) is 1 for the only class, so every "
                "Kamiran & Calders weight is 1.0 by construction and no label imbalance "
                "was measured. This is a could-not-check, not demographic parity."
            )
            warnings.warn(f"ReweightingTransformer: {fit_warnings[-1]}")

        self.fit_result = TransformationResult(
            method=f"reweighting_{self.method}",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "group_weights": {str(k): v for k, v in self._group_weights.items()},
                "n_groups_observed": n_groups,
                "n_labels_observed": self._n_labels_observed,
                "between_group_balance_measured": bool(n_groups >= 2),
                "group_key_counted": group_key_desc,
                "n_rows_without_a_recorded_group": n_rows_without_group,
                # None, not 0, for the methods that never look at the label: 0
                # would read as "every row carries one", a measurement nothing
                # made. Only 'target_parity' conditions on the label.
                "n_rows_without_a_recorded_label": n_rows_without_label,
            },
            warnings=fit_warnings,
        )

        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform returns the features unchanged.

        Use get_sample_weights() to get the computed weights.

        Args:
            X: Feature matrix

        Returns:
            Feature matrix (unchanged)
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()

        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        feature_cols = self._extract_feature_columns(df, self.protected_attributes)

        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    def get_sample_weights(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """
        Get sample weights for the data.

        Args:
            X: Feature matrix
            y: Target labels. Required for 'target_parity', whose Kamiran &
               Calders weights are conditioned on (group, label); unused by
               the other methods.

        Returns:
            Array of sample weights. Every value is 1.0 when the fit had nothing
            to compare (a single observed group, or a single observed label for
            'target_parity'); that case is a could-not-check and carries a
            UserWarning, because the array itself cannot distinguish it from an
            already balanced sample.
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()

        # Surface grade g002 (2026-09-17): the returned ndarray is the surface a
        # caller actually uses, and an all-1.0 array reads as "no reweighting
        # needed". The fit-time refusal has to reach it, or the caller has to
        # read fit_result to learn that nothing was compared.
        if self._n_groups_observed is not None and self._n_groups_observed < 2:
            warnings.warn(
                "ReweightingTransformer: the fit observed "
                f"{self._n_groups_observed} protected group(s), so no representation "
                "imbalance was measured and these weights are 1.0 by construction, not "
                "because the sample was balanced. See .fit_result.warnings."
            )
        elif self.method == "target_parity" and (self._n_labels_observed or 0) < 2:
            warnings.warn(
                "ReweightingTransformer: the fit observed "
                f"{self._n_labels_observed} label(s), so no positive-rate difference was "
                "measured and these weights are 1.0 by construction. See "
                ".fit_result.warnings."
            )

        df = self._to_dataframe(X, self.feature_names_in_)

        if self.method == "target_parity":
            if y is None:
                raise ValueError(
                    "method='target_parity' weights are conditioned on the "
                    "label (Kamiran & Calders, 2012): pass y."
                )
            y_arr = np.asarray(y)
            if len(y_arr) != len(df):
                raise ValueError(f"y has length {len(y_arr)}, expected {len(df)} to match X")
            group_keys = self._group_key_series(df).to_numpy()
            return self._lookup_weights(
                [(g, c) for g, c in zip(group_keys, y_arr)], "(group, label) cell(s)"
            )

        attr = self.protected_attributes[0]
        return self._lookup_weights(list(df[attr]), f"group(s) of {attr!r}")

    def _lookup_weights(self, keys: List[Any], key_kind: str) -> np.ndarray:
        """Per-row weights for ``keys``; a key with no fitted weight is NaN.

        audit-6 lane 2 (2026-09-09): `.get(key, 1.0)` handed every row of a
        group (or (group, label) cell) unseen at fit the weight 1.0, the "no
        reweighting needed" value, silently. Measured: fit on {A, B}, weights
        for [A, B, C, C] came back [0.75, 1.5, 1.0, 1.0] with no warning. A
        weight that was never derived is NaN (could not check): sklearn refuses
        NaN sample weights at fit and np.average returns NaN, so the gap cannot
        train silently, while the measured weights of every other row are kept.
        A hard raise would collapse "could not check" for a few rows into
        "failed" for the whole call. The warning names the keys so the caller
        can drop those rows or refit on data that covers them.
        """
        weights = np.full(len(keys), np.nan)
        unseen: List[Any] = []
        for i, key in enumerate(keys):
            fitted = self._group_weights.get(key)
            if fitted is None:
                if key not in unseen:
                    unseen.append(key)
            else:
                weights[i] = fitted
        if unseen:
            warnings.warn(
                f"ReweightingTransformer: no weight was fitted for {key_kind} {unseen[:8]}"
                f"{' ...' if len(unseen) > 8 else ''}; {int(np.isnan(weights).sum())} row(s) "
                "get a NaN sample weight (not 1.0). Drop them or refit on data that covers them."
            )
        return weights


class DisparateImpactRemover(BaseFeatureTransformer):
    """
    Removes disparate impact by repairing the conditional distribution of
    numeric features across protected groups (Feldman et al., 2015).

    For each numeric feature, the within-group rank (empirical quantile) of
    every value is preserved while the value itself is mapped toward a common
    target distribution -- the per-quantile median across all groups. This
    makes the feature's marginal distribution (approximately) independent of
    the protected attribute while preserving within-group ordering, which
    protects predictive utility.

    ``repair_level`` (lambda in the paper) interpolates between the original
    feature (0.0, no change) and the fully repaired feature (1.0).

    Non-numeric features are left unchanged. The protected attribute columns
    themselves are never modified (and are excluded from the output, matching
    the other transformers; the consumer re-adds them for downstream stages).

    With several protected attributes, the repair is applied across their
    *intersection* (e.g. race x gender), making each feature independent of the
    joint group rather than only the first attribute.

    Args:
        protected_attributes: Names of protected attribute columns.
        repair_level: Repair strength in [0, 1]. 0 = no change, 1 = full repair.
        n_quantiles: Number of quantile knots used to model each distribution.

    References:
        - Feldman, Friedler, Moeller, Scheidegger, Venkatasubramanian (2015):
          "Certifying and Removing Disparate Impact." KDD 2015.

    Example:
        >>> remover = DisparateImpactRemover(
        ...     protected_attributes=['race'], repair_level=1.0
        ... )
        >>> X_fair = remover.fit_transform(X)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: disparate_impact_removal. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        repair_level: float = 1.0,
        n_quantiles: int = 100,
        min_group_size: int = 10,
    ):
        super().__init__(protected_attributes)
        if not 0.0 <= float(repair_level) <= 1.0:
            raise ValueError(f"repair_level must be in [0, 1], got {repair_level}")
        if int(min_group_size) < 2:
            raise ValueError(
                f"min_group_size must be at least 2, got {min_group_size}: a single "
                "observation has no distribution to estimate."
            )
        self.repair_level = float(repair_level)
        self.n_quantiles = int(n_quantiles)
        self.min_group_size = int(min_group_size)
        # feature -> {'quantiles', 'per_group_sorted', 'target_q'}
        self._repair_maps: Dict[str, Dict[str, Any]] = {}
        # Populated by transform(): what the LAST transform() call could not repair.
        self.transform_result: Dict[str, Any] = {}

    def _group_key_series(self, df: pd.DataFrame) -> pd.Series:
        """Intersectional group key joined across ALL protected attributes, so the
        repair makes features independent of the *joint* group (e.g. race x gender),
        not only the first attribute. With one attribute this is just that column.

        BGL stage 2b (2026-09-17): ``astype(str)`` turned a missing protected
        value into the literal level ``"nan"``, which ``dropna()`` cannot drop
        once it is a string. A row whose group is UNKNOWN was therefore fitted a
        quantile curve of its own and published as a group (measured: 6 NaN rows
        gave n_groups=3, groups ['a', 'b', 'nan'], warnings == []). The key stays
        missing here, so those rows match no group and are disclosed as
        unrepaired instead of repaired against a group that does not exist.
        """
        attrs = self.protected_attributes
        assert attrs is not None  # set in fit()
        sub = df[list(attrs)]
        missing = sub.isna().any(axis=1)
        if len(attrs) == 1:
            keys = sub[attrs[0]].astype(str)
        else:
            keys = sub.astype(str).agg("|".join, axis=1)
        return keys.mask(missing.to_numpy())

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "DisparateImpactRemover":
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()

        group_keys = self._group_key_series(df)
        groups = list(pd.unique(group_keys.dropna()))
        quantiles = np.linspace(0.0, 1.0, self.n_quantiles + 1)
        numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

        warnings_list: List[str] = []
        n_rows_without_group = int(group_keys.isna().sum())
        if n_rows_without_group:
            warnings_list.append(
                f"{n_rows_without_group} row(s) have a missing value in {attrs}, so they "
                "belong to no known group: they are excluded from every fitted quantile "
                "curve and transform() returns them UNREPAIRED rather than repairing "
                "them against a group inferred from the missingness itself."
            )
            warnings.warn(f"DisparateImpactRemover: {warnings_list[-1]}")
        if len(groups) < 2:
            warnings_list.append(
                f"Protected attribute(s) {attrs} yield fewer than 2 groups; no repair applied."
            )

        self._repair_maps = {}
        n_per_group: Dict[str, Dict[str, int]] = {}
        undersized: Dict[str, Dict[str, int]] = {}
        for feat in numeric_features:
            per_group_sorted: Dict[Any, np.ndarray] = {}
            per_group_quantiles: List[np.ndarray] = []
            n_per_group[feat] = {}
            for g in groups:
                vals = df.loc[(group_keys == g).values, feat].dropna().astype(float).values
                n_per_group[feat][str(g)] = int(vals.size)
                if vals.size == 0:
                    continue
                # BGL stage 2 (2026-09-16): `vals.size == 0` was the ONLY guard,
                # and a one-row group passes it. np.quantile of a single value
                # returns that value at all 101 knots, so the group contributes a
                # flat, entirely fabricated "distribution" to the shared target
                # curve. With two groups np.median of two curves is their mean,
                # so one observation moved the repair target for every row:
                # measured 59 group-a rows dragged from mean -0.2294 to +2.4010
                # by a single group-b row, with warnings == []. The group is
                # still used (dropping it would silently change the repair), but
                # it is no longer silent.
                if vals.size < self.min_group_size:
                    undersized.setdefault(feat, {})[str(g)] = int(vals.size)
                per_group_sorted[g] = np.sort(vals)
                per_group_quantiles.append(np.quantile(vals, quantiles))
            if len(per_group_sorted) < 2:
                continue
            # Target distribution: per-quantile median across groups.
            target_q = np.median(np.vstack(per_group_quantiles), axis=0)
            self._repair_maps[feat] = {
                "quantiles": quantiles,
                "per_group_sorted": per_group_sorted,
                "target_q": target_q,
            }

        unreliable = sorted(f for f in undersized if f in self._repair_maps)
        if unreliable:
            detail = "; ".join(
                f"{f}: " + ", ".join(f"{g} n={n}" for g, n in sorted(undersized[f].items()))
                for f in unreliable[:4]
            )
            warnings_list.append(
                f"repair target rests on an unreliable quantile estimate for "
                f"{len(unreliable)} feature(s): {detail}"
                + (" ..." if len(unreliable) > 4 else "")
                + f". A group with fewer than min_group_size={self.min_group_size} rows "
                "has no distribution to estimate, and its flat quantile curve moves the "
                "shared target for EVERY row of that feature. Treat those features as "
                "could-not-check rather than repaired."
            )
            warnings.warn(warnings_list[-1])

        skipped = [c for c in feature_cols if c not in numeric_features]
        if skipped:
            warnings_list.append(
                f"Non-numeric features left unchanged: {skipped[:8]}"
                + (" ..." if len(skipped) > 8 else "")
            )

        self.fit_result = TransformationResult(
            method="disparate_impact_removal",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            features_modified=list(self._repair_maps.keys()),
            fit_metrics={
                "repair_level": self.repair_level,
                "n_features_repaired": len(self._repair_maps),
                "n_groups": len(groups),
                "groups": [str(g) for g in groups],
                "min_group_size": self.min_group_size,
                "n_per_group": n_per_group,
                "features_repaired_on_unreliable_estimate": unreliable,
                "n_rows_without_a_recorded_group": n_rows_without_group,
            },
            warnings=warnings_list,
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        self._check_is_fitted()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        group_keys = self._group_key_series(df)
        out = df.copy()

        # BGL stage 2 (2026-09-16): the loop below iterates only the groups SEEN
        # at fit. A row whose group was never fitted matches no mask, so the raw
        # copy survives into the output and is returned in the same column, in
        # the same frame, indistinguishable from a repaired value. Measured: fit
        # on {a, b}, transform of ['a','b','c','c'] returned the two group-c rows
        # byte-identical to their raw input with warnings == []. Repaired and
        # un-repaired values sharing one column is the fabrication; it is counted
        # and named here. Mapping unseen groups onto a fitted group's curve would
        # fabricate a different thing, so they are left raw and DISCLOSED.
        # BGL stage 2b (2026-09-17): `unmatched` was a single mask ANDed across
        # ALL features, so a group fitted for f1 but NOT for f2 (its f2 column was
        # all-NaN at fit) had its raw f2 value returned inside the REPAIRED f2
        # column while transform_result reported n_rows_unrepaired 0, unseen_groups
        # [] and zero warnings: the original defect with an affirmative clean
        # certificate on top. The mask is per feature now and the per-feature
        # counts are published; a row unrepaired for ANY feature counts as
        # unrepaired.
        unmatched = np.zeros(len(df), dtype=bool)
        unrepaired_by_feature: Dict[str, int] = {}
        unseen_by_feature: Dict[str, List[str]] = {}
        any_feature_repaired = False

        for feat, m in self._repair_maps.items():
            if feat not in out.columns:
                continue
            any_feature_repaired = True
            feat_unmatched = np.ones(len(df), dtype=bool)
            quantiles = m["quantiles"]
            target_q = m["target_q"]
            repaired = out[feat].astype(float).values.copy()
            for g, sorted_vals in m["per_group_sorted"].items():
                mask = (group_keys == g).values
                feat_unmatched &= ~mask
                if not mask.any():
                    continue
                orig_vals = df.loc[mask, feat].astype(float).values
                n = len(sorted_vals)
                # Empirical-CDF rank in (0, 1], robust to ties.
                q_pos = np.searchsorted(sorted_vals, orig_vals, side="right") / n
                q_pos = np.clip(q_pos, 0.0, 1.0)
                # Map rank -> value in the shared target distribution.
                full_repaired = np.interp(q_pos, quantiles, target_q)
                repaired[mask] = (
                    1.0 - self.repair_level
                ) * orig_vals + self.repair_level * full_repaired
            out[feat] = repaired
            unrepaired_by_feature[feat] = int(feat_unmatched.sum())
            unseen_by_feature[feat] = sorted(
                {str(k) for k in group_keys[feat_unmatched].dropna().unique()}
            )
            unmatched |= feat_unmatched

        # BGL stage 2b (2026-09-17): this used to zero the mask, publishing
        # "0 of N rows unrepaired" for a transform in which NOTHING was repaired
        # at all (a single-group fit builds no repair map), including for rows of
        # a group that was never seen. Nothing repaired is a could-not-check, not
        # a clean repair: every row is unrepaired and the reason says why.
        reason: Optional[str] = None
        if not any_feature_repaired:
            unmatched = np.ones(len(df), dtype=bool)
            reason = (
                "no repair map was fitted (fewer than 2 groups carried data for any "
                "numeric feature), so NO row was repaired; the frame is returned "
                "unchanged and this is a could-not-check, not a completed repair."
            )
        unseen_keys = sorted({str(k) for k in group_keys[unmatched].dropna().unique()})
        n_missing_key = int(group_keys[unmatched].isna().sum())
        self.transform_result = {
            "n_rows": int(len(df)),
            "n_rows_unrepaired": int(unmatched.sum()),
            "unseen_groups": unseen_keys,
            "unrepaired_row_positions": np.flatnonzero(unmatched).tolist(),
            "unrepaired_by_feature": unrepaired_by_feature,
            "unseen_groups_by_feature": unseen_by_feature,
            "n_rows_without_a_recorded_group": n_missing_key,
            "n_features_repaired": len(unrepaired_by_feature),
            "reason": reason,
        }
        if unmatched.any():
            warnings.warn(
                f"DisparateImpactRemover: {int(unmatched.sum())} of {len(df)} row(s) "
                + (
                    reason
                    if reason is not None
                    else (
                        "belong to group(s) never seen at fit for at least one repaired "
                        f"feature {unseen_keys[:8]}"
                        + (" ..." if len(unseen_keys) > 8 else "")
                        + (
                            f" (and {n_missing_key} row(s) carry no group at all)"
                            if n_missing_key
                            else ""
                        )
                        + f"; per feature {unrepaired_by_feature}. They are returned "
                        "UNREPAIRED in the same columns as the repaired rows. See "
                        ".transform_result for the row positions."
                    )
                )
            )

        assert self.protected_attributes is not None  # set in fit()
        feature_cols = self._extract_feature_columns(out, self.protected_attributes)
        if was_dataframe:
            return out[feature_cols].copy()
        return out[feature_cols].values


class LabelMassager(BaseFeatureTransformer):
    """
    Corrects historical bias in the training labels by relabelling a minimal
    set of borderline instances ("massaging", Kamiran & Calders, 2012).

    A ranker estimates P(y=1 | x) for every instance. The instances closest to
    the decision boundary are then relabelled to remove the difference in
    positive rates between the favored and the deprived group:

      - the M deprived-group negatives with the *highest* scores are promoted
        (0 -> 1), and
      - the M favored-group positives with the *lowest* scores are demoted
        (1 -> 0),

    where M is the smallest count that equalizes the group positive rates:

        M = (n_dep * pos_fav - n_fav * pos_dep) / (n_dep + n_fav)

    Only labels change -- features and the number of rows are preserved. Because
    the handler's ``transform(X) -> X'`` contract returns features and re-uses
    the original target, the relabelled target is exposed through the additive
    :meth:`get_massaged_labels` hook (mirroring
    :meth:`ReweightingTransformer.get_sample_weights`).

    Args:
        protected_attributes: Names of protected attribute columns.
        ranker: Probability model used to rank instances. Only ``'logistic'``
            is currently supported (scikit-learn ``LogisticRegression``).
        max_flip_fraction: Safety cap on the share of rows that may be
            relabelled, in [0, 0.5]. The actual number of flips is
            ``min(M, max_flip_fraction * n_samples)``.

    References:
        - Kamiran & Calders (2012): "Data preprocessing techniques for
          classification without discrimination." KAIS 33(1).

    Example:
        >>> massager = LabelMassager(protected_attributes=['gender'])
        >>> massager.fit(X, y)
        >>> y_fair = massager.get_massaged_labels(X, y)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: label_massaging. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        ranker: Literal["logistic"] = "logistic",
        max_flip_fraction: float = 0.5,
    ):
        super().__init__(protected_attributes)
        if ranker != "logistic":
            raise ValueError(f"Unsupported ranker '{ranker}'; only 'logistic' is available.")
        if not 0.0 <= float(max_flip_fraction) <= 0.5:
            raise ValueError(f"max_flip_fraction must be in [0, 0.5], got {max_flip_fraction}")
        self.ranker = ranker
        self.max_flip_fraction = float(max_flip_fraction)
        # Fitted sklearn LogisticRegression (untyped third-party -> Any), or None
        # when the ranker could not be trained.
        self._model: Optional[Any] = None
        self._ranker_features: List[str] = []
        self.deprived_group_: Any = None
        self.favored_group_: Any = None
        # Measured by the LAST get_massaged_labels() call (or by fit(), on the
        # fit data). n_labels_flipped_ = n_promoted_ + n_demoted_, the number of
        # labels that ACTUALLY changed, not a per-side plan.
        self.n_promoted_: int = 0
        self.n_demoted_: int = 0
        self.n_labels_flipped_: int = 0
        # BGL stage 2b (2026-09-17): these two are three-state. ``None`` means the
        # LAST call never scored a frame at all (it short-circuited), which is a
        # could-not-check; a 0 means a frame WAS scored and needed nothing. They
        # used to be plain ints that no call reset, so a short-circuiting call
        # reported the PREVIOUS call's substitution count (measured: 7 reported,
        # 0 actual) against a docstring that says it describes the last call.
        self.n_features_imputed_: Optional[int] = None
        self.n_flips_tie_broken_: Optional[int] = None
        # Groups observed by the LAST fit, i.e. how many the deprived/favored
        # choice above rests on. None means no fit has run. Kept separately from
        # ``deprived_group_ == favored_group_``, which is ALSO true when two groups
        # happen to tie on the positive rate, and that case is a real measurement.
        self._n_groups_fitted: Optional[int] = None

    def _numeric_feature_frame(self, df: pd.DataFrame, attrs: List[str]) -> pd.DataFrame:
        """Numeric, non-protected feature columns used to train the ranker."""
        feature_cols = self._extract_feature_columns(df, attrs)
        numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        return df[numeric].astype(float).fillna(0.0)

    def _group_key_series(self, df: pd.DataFrame) -> pd.Series:
        """Intersectional group key (joined across ALL protected attributes).

        With one protected attribute this is just that column; with several it is
        the intersection (e.g. ``"race|gender"``), so massaging equalizes the most
        and least favored *intersectional* subgroups rather than only the first
        attribute.

        BGL stage 2b (2026-09-17): ``astype(str)`` turned a missing protected
        value into the literal group ``"nan"``, and ``dropna()`` could not drop it
        because it was a string by then. Measured: 10 NaN values published
        n_groups=3 with warnings == [] and no Python warning, and that phantom
        group was eligible to be picked as the deprived or the favored one, i.e.
        the whole relabelling could be aimed at a group that does not exist. The
        key stays missing, so those rows join no group.
        """
        attrs = self.protected_attributes
        assert attrs is not None  # set in fit()
        sub = df[list(attrs)]
        missing = sub.isna().any(axis=1)
        if len(attrs) == 1:
            keys = sub[attrs[0]].astype(str)
        else:
            keys = sub.astype(str).agg("|".join, axis=1)
        return keys.mask(missing.to_numpy())

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "LabelMassager":
        from sklearn.linear_model import LogisticRegression

        if y is None:
            raise ValueError("LabelMassager.fit requires the target y.")
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs

        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()

        y_arr = np.asarray(y).astype(int)
        group_keys = self._group_key_series(df)
        groups = list(pd.unique(group_keys.dropna()))
        self._n_groups_fitted = len(groups)

        # Identify deprived (lowest positive rate) vs favored (highest) intersectional group.
        warnings_list: List[str] = []
        n_rows_without_group = int(group_keys.isna().sum())
        if n_rows_without_group:
            warnings_list.append(
                f"{n_rows_without_group} row(s) have a missing value in {list(attrs)}, so "
                "they belong to no known group: they are excluded from the positive-rate "
                "comparison and can be neither promoted nor demoted."
            )
            warnings.warn(f"LabelMassager: {warnings_list[-1]}")
        if len(groups) < 2:
            warnings_list.append(
                f"Protected attribute(s) {attrs} yield fewer than 2 groups; "
                "no relabelling will be applied."
            )
            self.deprived_group_ = self.favored_group_ = groups[0] if groups else None
        else:
            pos_rates = {g: float(np.mean(y_arr[(group_keys == g).values])) for g in groups}
            self.deprived_group_ = min(pos_rates, key=lambda g: pos_rates[g])
            self.favored_group_ = max(pos_rates, key=lambda g: pos_rates[g])

        # Train the ranker on numeric features only.
        Xr = self._numeric_feature_frame(df, attrs)
        self._ranker_features = list(Xr.columns)
        self._model = LogisticRegression(max_iter=1000, solver="lbfgs")
        if Xr.shape[1] > 0 and len(np.unique(y_arr)) > 1:
            self._model.fit(Xr.values, y_arr)
        else:
            self._model = None
            warnings_list.append(
                "Ranker could not be trained (no numeric features or single "
                "class); get_massaged_labels will return labels unchanged."
            )

        # BGL stage 2 (2026-09-16): the column-count guard above passes for
        # columns that are PRESENT and empty, because _numeric_feature_frame
        # turns every NaN into 0.0 and the frame keeps its shape. Measured on
        # all-NaN features: the ranker trained, produced exactly 1 distinct
        # score across 60 rows, and the record still said ranker='logistic' with
        # warnings == [] - byte-identical to the real-features run. A ranker with
        # one distinct score has no decision boundary, so "the instances closest
        # to the boundary" is argsort tie-breaking on row order: on a shuffled
        # copy of the same rows it re-picked only 7 of its own 16 records, where
        # the healthy fit picks 16 of 16. Tested with np.unique on the RAW
        # scores, never a spread compared against 0.0.
        if self._model is not None:
            fit_scores = self._model.predict_proba(Xr.values)[:, 1]
            if np.unique(fit_scores).size < 2:
                self._model = None
                warnings_list.append(
                    "Ranker trained but produced a single distinct score for every row "
                    "(features carry no usable variation), so no borderline ordering "
                    "exists; get_massaged_labels will return labels unchanged."
                )
                warnings.warn(f"LabelMassager: {warnings_list[-1]}")

        # BGL stage 2 (2026-09-16): n_labels_flipped used to be the PLANNED
        # per-side count M. get_massaged_labels promotes M deprived negatives AND
        # demotes M favored positives, so 2M labels change: the report was
        # exactly half the rows the transformer rewrites, every time, and always
        # in the reassuring direction (seed 0: reported 21, actual 42). It is the
        # measured total now, taken from the actual index arrays.
        planned_per_side = self._planned_flip_count(df, y_arr)
        _, n_promoted, n_demoted = self._massage(df, y_arr)
        self.n_promoted_, self.n_demoted_ = n_promoted, n_demoted
        self.n_labels_flipped_ = n_promoted + n_demoted
        # BGL stage 2b (2026-09-17): the fit-time substitutions and the tie-break
        # count reached a Python warning and an instance attribute, never the
        # RESULT. Measured: 150 substituted zeros produced fit_result.warnings ==
        # [] and no fit_metrics key, so the serialised record of a fit that rested
        # on invented feature values was byte-identical to a clean one.
        if self.n_features_imputed_:
            warnings_list.append(
                f"{self.n_features_imputed_} missing ranker feature value(s) were filled "
                "with 0.0 before fitting the ranker; the flip decisions for those rows "
                "rest on a substituted value, not on an observed one."
            )
        if self.n_flips_tie_broken_:
            warnings_list.append(
                f"{self.n_flips_tie_broken_} of the {n_promoted + n_demoted} flips were "
                "decided by a tie-break, not by proximity to the decision boundary: the "
                "boundary score is shared by more candidate rows than there were flips "
                "to make. Ties are broken by a row-content digest, so two rows that "
                "DIFFER are picked the same way under any row order; rows that are "
                "identical in every feature the ranker sees cannot be told apart at all, "
                "and for those the choice of WHICH row carries the flip is arbitrary. "
                "Either way it is not a measurement of which row is closest to the "
                "boundary."
            )
            warnings.warn(f"LabelMassager: {warnings_list[-1]}")
        self.fit_result = TransformationResult(
            method="label_massaging",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "protected_attributes": list(attrs),
                "deprived_group": str(self.deprived_group_),
                "favored_group": str(self.favored_group_),
                "n_groups": len(groups),
                "n_labels_flipped": int(n_promoted + n_demoted),
                "n_promoted": int(n_promoted),
                "n_demoted": int(n_demoted),
                "planned_flips_per_side": int(planned_per_side),
                "max_flip_fraction": self.max_flip_fraction,
                "ranker": self.ranker if self._model is not None else None,
                "n_features_imputed": self.n_features_imputed_,
                "n_flips_tie_broken": self.n_flips_tie_broken_,
                "n_rows_without_a_recorded_group": n_rows_without_group,
            },
            warnings=warnings_list,
        )
        self.is_fitted = True
        return self

    def _planned_flip_count(self, df: pd.DataFrame, y_arr: np.ndarray) -> int:
        """Smallest M that equalizes the deprived/favored group positive rates,
        capped by the safety cap. Groups are intersectional across all attrs."""
        if self.deprived_group_ is None or self.favored_group_ is None:
            return 0
        if self.deprived_group_ == self.favored_group_:
            return 0
        group_keys = self._group_key_series(df)
        dep_mask = (group_keys == self.deprived_group_).values
        fav_mask = (group_keys == self.favored_group_).values
        n_dep, n_fav = int(dep_mask.sum()), int(fav_mask.sum())
        if n_dep == 0 or n_fav == 0:
            return 0
        pos_dep = int(y_arr[dep_mask].sum())
        pos_fav = int(y_arr[fav_mask].sum())
        m = (n_dep * pos_fav - n_fav * pos_dep) / (n_dep + n_fav)
        m = max(0, int(np.floor(m)))
        cap = int(np.floor(self.max_flip_fraction * len(y_arr)))
        # Cannot promote more deprived negatives than exist, nor demote more
        # favored positives than exist.
        m = min(m, n_dep - pos_dep, pos_fav, cap)
        return max(0, m)

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features unchanged. Use get_massaged_labels() for the labels."""
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        feature_cols = self._extract_feature_columns(df, self.protected_attributes)
        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    def _ranker_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The ranker's feature columns, taken from ``df``.

        BGL stage 2 (2026-09-16): this used to be a
        ``df.reindex(columns=..., fill_value=0.0)``, so a column absent from X at
        call time was INVENTED as an all-zero column and the ranker was asked to
        score rows whose features were never supplied. Measured: dropping the
        dominant predictor (coef 2.077) still rewrote 14 labels and changed 26
        record-level decisions, with no error and no warning. A missing fitted
        feature is refused; a NaN is filled with 0.0 exactly as at fit time (so
        the ranker sees what it was trained on) and COUNTED.
        """
        self._require_ranker_columns(df)
        Xr = df[self._ranker_features].astype(float) if self._ranker_features else df.iloc[:, :0]
        self.n_features_imputed_ = int(Xr.isna().to_numpy().sum()) if len(Xr.columns) else 0
        if self.n_features_imputed_:
            warnings.warn(
                f"LabelMassager: {self.n_features_imputed_} missing ranker feature "
                "value(s) were filled with 0.0 (the same fill used at fit); the "
                "relabelling decisions for those rows rest on a substituted value."
            )
        return Xr.fillna(0.0)

    def _require_ranker_columns(self, df: pd.DataFrame) -> None:
        """Refuse a frame missing a column the fitted ranker needs.

        BGL stage 2b (2026-09-17): this check lived inside ``_ranker_frame``,
        below ``_massage``'s ``n_flips <= 0 or self._model is None`` short-circuit,
        so in exactly the could-not-check state it never ran: a frame missing the
        dominant predictor came back with no error and no warning. It is checked
        before any decision, degenerate or not.
        """
        missing = [c for c in self._ranker_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"LabelMassager: column(s) required by the fitted ranker are absent "
                f"from X: {missing}. Relabelling decisions cannot be made from features "
                "that were never supplied; pass the frame the ranker was fitted on."
            )

    def _massage(
        self,
        df: pd.DataFrame,
        y: ArrayLike,
    ) -> Tuple[np.ndarray, int, int]:
        """``(y_new, n_promoted, n_demoted)``: the MEASURED effect, not the plan."""
        y_new = np.asarray(y).astype(int).copy()
        # Reset first: a call that short-circuits below must not leave the
        # PREVIOUS call's substitution count standing as if it described this one.
        self.n_features_imputed_ = None
        self.n_flips_tie_broken_ = None
        self._require_ranker_columns(df)

        n_flips = self._planned_flip_count(df, y_new)
        if n_flips <= 0 or self._model is None:
            return y_new, 0, 0

        # Rank borderline instances by the ranker's positive-class probability.
        Xr = self._ranker_frame(df)
        scores = self._model.predict_proba(Xr.values)[:, 1]

        # A score vector with one distinct value has no boundary to be "close
        # to": argsort then returns row order, and the selection is an artefact
        # of how the frame happened to be sorted. Refuse rather than relabel.
        if np.unique(scores).size < 2:
            warnings.warn(
                "LabelMassager: the ranker gives every row the same score, so no "
                "borderline ordering exists and no labels were flipped. Any selection "
                "here would be argsort tie-breaking on row order, not proximity to a "
                "decision boundary."
            )
            return y_new, 0, 0

        group_keys = self._group_key_series(df)
        dep_mask = (group_keys == self.deprived_group_).values
        fav_mask = (group_keys == self.favored_group_).values

        # BGL stage 2b (2026-09-17): `np.unique(scores).size < 2` only catches the
        # ALL-tied case. The selection is equally arbitrary whenever the flip
        # BOUNDARY falls inside a tied block, which an ordinary single binary
        # feature produces (2 distinct scores, 24 flips, only 14 of the 24
        # selected rows surviving a row permutation, warnings == []). Ties are
        # broken by a row-content digest so the same ROWS are picked under any
        # permutation, and the number that had to be tie-broken is disclosed.
        keys = _row_content_keys(Xr)
        n_promoted = n_demoted = 0
        tie_broken = 0
        # Promote the deprived-group negatives closest to the boundary (highest score).
        dep_neg_idx = np.where(dep_mask & (y_new == 0))[0]
        if dep_neg_idx.size:
            promote, n_tied, _ = _select_boundary_rows(
                dep_neg_idx, scores, keys, n_flips, take_highest=True
            )
            y_new[promote] = 1
            n_promoted = int(promote.size)
            tie_broken += n_tied
        # Demote the favored-group positives closest to the boundary (lowest score).
        fav_pos_idx = np.where(fav_mask & (y_new == 1))[0]
        if fav_pos_idx.size:
            demote, n_tied, _ = _select_boundary_rows(
                fav_pos_idx, scores, keys, n_flips, take_highest=False
            )
            y_new[demote] = 0
            n_demoted = int(demote.size)
            tie_broken += n_tied
        self.n_flips_tie_broken_ = tie_broken
        return y_new, n_promoted, n_demoted

    def get_massaged_labels(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: ArrayLike,
    ) -> np.ndarray:
        """
        Return a relabelled copy of ``y`` with borderline labels flipped to
        equalize group positive rates. ``X`` must contain the protected
        attribute column(s) and every numeric feature the ranker was fitted on;
        rows are preserved in order.

        The number of labels this actually changed is recorded on
        ``n_promoted_``, ``n_demoted_`` and ``n_labels_flipped_``.

        Raises:
            ValueError: if a column the ranker was fitted on is absent from ``X``.
        """
        self._check_is_fitted()
        df = self._to_dataframe(X, self.feature_names_in_)
        # BGL stage 2b (2026-09-17): this path was silent when the ranker could
        # not be trained, while the constant-score path warned. A caller who did
        # not run fit() in the same scope got an unchanged y back with nothing
        # saying that no ranking existed to make a decision from.
        if self._model is None:
            warnings.warn(
                "LabelMassager: no usable ranker was fitted (no numeric features, a "
                "single class, or a single distinct score), so the labels are returned "
                "UNCHANGED. This is a could-not-check, not a frame that needed no "
                "relabelling. See .fit_result.warnings for which of those it was."
            )
        elif (self._n_groups_fitted or 0) < 2:
            # BGL stage 3 (2026-09-27): this path was silent. With one observed
            # protected group the ranker trains perfectly well, deprived == favored,
            # M is 0, and the array comes back identical to y. Measured on 60 rows
            # of a single group with two usable features: 0 labels changed,
            # n_labels_flipped_ 0, n_promoted_ 0, n_demoted_ 0 and NO warning from
            # either this method or fit() at call time, byte-identical to a frame
            # whose two groups genuinely had equal positive rates. The sibling
            # ReweightingTransformer.get_sample_weights already discloses exactly
            # this state on the array it hands back.
            warnings.warn(
                "LabelMassager: the fit observed "
                f"{self._n_groups_fitted} protected group(s), fewer than the 2 a "
                "positive-rate difference needs, so nothing was compared and the labels "
                "are returned UNCHANGED. n_labels_flipped_ == 0 here is a "
                "could-not-check, not a frame whose groups were already equal. See "
                ".fit_result.warnings."
            )
        else:
            # BGL stage 4 (2026-09-27): both guards above read the FIT, and the
            # defect arrives at the CALL. _planned_flip_count returns 0 as soon as
            # `n_dep == 0 or n_fav == 0` on THIS frame, so a frame carrying neither
            # of the two groups the fit compared came back with y unchanged and
            # nothing said. Measured after a healthy fit on the 200-row two-group
            # frame (n_groups 2, deprived 'a', favored 'b', 60 flips), then called
            # on 40 rows: gender 'c' (a level the fit never saw) -> identical True,
            # n_labels_flipped_ 0, warnings []; all 'b' (the favored group only) ->
            # the same; all missing -> the same. Now each of those three warns with
            # the two counts. The guard reads how many rows of the DEPRIVED and the
            # FAVORED group this frame carries, not deprived == favored, because two
            # groups that tie on the positive rate make those equal and that is a
            # real measurement: the control asserts the tie stays silent, and the
            # fitted frame still flips its 60 labels without a warning.
            call_keys = self._group_key_series(df)
            n_dep = (
                int((call_keys == self.deprived_group_).sum())
                if self.deprived_group_ is not None
                else 0
            )
            n_fav = (
                int((call_keys == self.favored_group_).sum())
                if self.favored_group_ is not None
                else 0
            )
            # THE INELIGIBLE ROWS ARE DISCLOSED WHETHER OR NOT A GROUP IS ABSENT
            # (BGL6 F02, 2026-09-28). n_no_group was computed only INSIDE the
            # total-absence branch below, so a call frame of 10 'a' + 10 'b' + 20
            # rows with no protected value flipped 6 labels and said nothing about
            # the half of the frame that could be neither promoted nor demoted.
            # fit() already publishes exactly this on both channels, as
            # n_rows_without_a_recorded_group, so the two disagreed about the same
            # frame. A rate equalised over the rows that HAVE the attribute is not
            # the rate over the frame.
            n_no_group = int(call_keys.isna().sum())
            self.n_call_rows_without_a_recorded_group_ = n_no_group
            if n_no_group and n_dep and n_fav:
                warnings.warn(
                    f"LabelMassager: {n_no_group} of {len(df)} row(s) in this frame "
                    f"carry no value for {list(self._fitted_protected_attributes())}, "
                    f"so they are in neither group and can be neither promoted nor "
                    f"demoted. The labels that ARE flipped equalise the two groups "
                    f"over the rows that carry the attribute, which is not the same "
                    f"as over the frame. Read "
                    f".n_call_rows_without_a_recorded_group_.",
                    UserWarning,
                    stacklevel=2,
                )
            if n_dep == 0 or n_fav == 0:
                warnings.warn(
                    f"LabelMassager: this frame carries {n_dep} row(s) of the deprived "
                    f"group {self.deprived_group_!r} and {n_fav} row(s) of the favored "
                    f"group {self.favored_group_!r}, so the positive rates the fit "
                    "compared cannot both be formed here and no label was flipped "
                    f"({n_no_group} of {len(df)} row(s) carry no protected value at "
                    "all). n_labels_flipped_ == 0 is a could-not-check for THIS frame, "
                    "not a frame whose groups were already equal. The labels are "
                    "returned UNCHANGED."
                )
        y_new, n_promoted, n_demoted = self._massage(df, y)
        self.n_promoted_, self.n_demoted_ = n_promoted, n_demoted
        self.n_labels_flipped_ = n_promoted + n_demoted
        # A GAP SMALLER THAN ONE LABEL IS A COULD-NOT-ACT, NOT A TIE (BGL6 F02,
        # 2026-09-28). _planned_flip_count computes the smallest equalizing count as
        # (n_dep*pos_fav - n_fav*pos_dep) / (n_dep + n_fav) and FLOORS it, so a real
        # disparity worth less than one whole label produces zero flips. Measured on
        # 10 rows of 'a' at a 0.5 positive rate against 10 rows of 'b' at 0.6:
        # M = 0.5 -> 0 flips, the array returned byte-identical,
        # n_labels_flipped_ == 0 and warnings == [], which is byte-identical to the
        # genuine 0.5 / 0.5 tie. A caller cannot tell a mitigation that had nothing
        # to do from one that could not act on a live gap, and a neutered mitigation
        # reports SUCCESS rather than failure.
        #
        # The existing guard covers a call frame missing one of the two groups; this
        # is the case where both are present and the arithmetic floors away.
        self.unequalized_rate_gap_: Optional[float] = None
        if self.n_labels_flipped_ == 0:
            keys = self._group_key_series(df)
            dep = (keys == self.deprived_group_).values
            fav = (keys == self.favored_group_).values
            if dep.any() and fav.any():
                rate_dep = float(y_new[dep].mean())
                rate_fav = float(y_new[fav].mean())
                if rate_dep != rate_fav:
                    self.unequalized_rate_gap_ = rate_fav - rate_dep
                    warnings.warn(
                        f"LabelMassager: NOTHING WAS FLIPPED and the groups are NOT "
                        f"equal. {self.deprived_group_!r} has a positive rate of "
                        f"{rate_dep:.4f} against {rate_fav:.4f} for "
                        f"{self.favored_group_!r}, a gap of "
                        f"{rate_fav - rate_dep:+.4f}, and the smallest equalizing "
                        f"count is less than one whole label at this sample size, so "
                        f"it floors to zero. The labels are returned UNCHANGED. "
                        f"n_labels_flipped_ == 0 is a could-not-act for THIS frame, "
                        f"not a frame whose groups were already equal; read "
                        f".unequalized_rate_gap_, which is None only when they were.",
                        UserWarning,
                        stacklevel=2,
                    )

        if self.n_flips_tie_broken_:
            warnings.warn(
                f"LabelMassager: {self.n_flips_tie_broken_} of the "
                f"{self.n_labels_flipped_} flips were decided by a tie-break at the "
                "decision boundary, not by proximity to it. See "
                ".n_flips_tie_broken_."
            )
        return y_new


class Resampler(BaseFeatureTransformer):
    """
    Balances group representation by random over- or under-sampling
    (Kamiran & Calders, 2012). Row counts change; features and labels of the
    sampled rows are preserved.

    Cells are defined either per protected group (``balance_by='group'``) or
    per group x label combination (``balance_by='group_label'``, which also
    balances the class distribution within each group). Oversampling draws with
    replacement up to the largest cell; undersampling draws without replacement
    down to the smallest cell.

    Because resampling changes the row count it breaks the handler's
    ``transform(X) -> X'`` (same-rows) contract, so the resampled training set
    is exposed through the additive :meth:`get_resampled_data` hook.

    Args:
        protected_attributes: Names of protected attribute columns.
        strategy: ``'oversample'`` (default) or ``'undersample'``.
        balance_by: ``'group'`` or ``'group_label'``.
        random_state: Seed for reproducible sampling.

    References:
        - Kamiran & Calders (2012): "Data preprocessing techniques for
          classification without discrimination." KAIS 33(1).

    Example:
        >>> resampler = Resampler(protected_attributes=['race'])
        >>> resampler.fit(X, y)
        >>> X_bal, y_bal = resampler.get_resampled_data(X, y)

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: resampling. See docs/BETA_GO_LIVE_PLAN.md for the batch definitions.
    (end Beta Go-Live proof status)
    """

    # Above this many copies per distinct source row, a cell's extra rows carry
    # no extra information: the row count grew and the effective sample size
    # did not. Documented threshold, not a magic number in a branch.
    _REPLICATION_WARN_FACTOR = 5.0

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        strategy: Literal["oversample", "undersample"] = "oversample",
        balance_by: Literal["group", "group_label"] = "group_label",
        random_state: int = 42,
    ):
        super().__init__(protected_attributes)
        if strategy not in ("oversample", "undersample"):
            raise ValueError(f"strategy must be 'oversample' or 'undersample', got {strategy}")
        if balance_by not in ("group", "group_label"):
            raise ValueError(f"balance_by must be 'group' or 'group_label', got {balance_by}")
        self.strategy = strategy
        self.balance_by = balance_by
        self.random_state = int(random_state)
        # Set by get_resampled_data(): what the LAST resample actually did.
        self.resample_result: Dict[str, Any] = {}
        self._disclosed: List[str] = []

    def _group_key_series(self, df: pd.DataFrame) -> pd.Series:
        """Intersectional group key across ALL protected attributes, missing where
        any protected value is.

        BGL stage 4 (2026-09-27): the key was built with ``astype(str)``, so a
        MISSING protected value became the literal group ``"nan"``: it was
        oversampled to the target like any other cell, and because
        ``observed_group_keys`` is derived from the same values the single-group
        refusal "between-group balance was NOT measured: only one protected group"
        could not fire on the one frame it was written for. Measured on 60 rows
        whose 'gender' held ONE observed level 'a' plus 10 missing values:
        cell_sizes_before {'(nan, label=1)': 10, '(a, label=1)': 20,
        '(a, label=0)': 30}, all three cells at 30 after,
        group_totals_after {'nan': 30, 'a': 60}, n_rows_out 90 of which 30 carried
        a missing gender, could_not_check None, and no single-group disclosure. The
        three siblings in this file that key on a group (ReweightingTransformer,
        DisparateImpactRemover, LabelMassager) already mask the key for exactly
        this reason.
        """
        attrs = self.protected_attributes
        assert attrs is not None  # set in fit()
        sub = df[list(attrs)]
        missing = sub.isna().any(axis=1)
        if len(attrs) == 1:
            keys = sub[attrs[0]].astype(str)
        else:
            keys = sub.astype(str).agg("|".join, axis=1)
        return keys.mask(missing.to_numpy())

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "Resampler":
        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs
        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self.feature_names_out_ = feature_cols.copy()

        self.fit_result = TransformationResult(
            method="resampling",
            n_features_original=len(feature_cols),
            n_features_transformed=len(feature_cols),
            n_samples=len(df),
            fit_metrics={
                "strategy": self.strategy,
                "balance_by": self.balance_by,
            },
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Return features unchanged. Use get_resampled_data() for balanced rows."""
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)
        feature_cols = self._extract_feature_columns(df, self.protected_attributes)
        if was_dataframe:
            return df[feature_cols].copy()
        return df[feature_cols].values

    def get_resampled_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: ArrayLike,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Return a balanced ``(X_resampled, y_resampled)``. ``X`` keeps all of its
        input columns (features and protected attributes); ``y`` is aligned.
        """
        self._check_is_fitted()
        assert self.protected_attributes is not None  # set in fit()
        df = self._to_dataframe(X, self.feature_names_in_).reset_index(drop=True)
        y_arr = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        # Intersectional group key across ALL protected attributes, so multi-attribute
        # selections balance subgroups like (race x gender), not just the first
        # attribute. Missing where the protected value is (see _group_key_series):
        # a row whose group is unknown cannot be assigned to a cell, so it is left
        # out of the balancing and counted instead of being coined into a group
        # called "nan" and oversampled.
        key_series = self._group_key_series(df)
        groupless = np.flatnonzero(key_series.isna().to_numpy())
        groupless_set = set(groupless.tolist())
        group_vals = list(key_series.values)

        # Build the cell key for each row.
        if self.balance_by == "group_label":
            keys = list(zip(group_vals, y_arr))
        else:
            keys = group_vals
        cell_index_lists: Dict[Any, List[int]] = {}
        for i, k in enumerate(keys):
            if i in groupless_set:
                continue
            cell_index_lists.setdefault(k, []).append(i)
        cell_indices: Dict[Any, np.ndarray] = {
            k: np.asarray(v) for k, v in cell_index_lists.items()
        }
        if not cell_indices:
            self._record_no_rows(len(df), n_without_group=len(groupless))
            return df, y_arr

        sizes = [len(v) for v in cell_indices.values()]
        target = max(sizes) if self.strategy == "oversample" else min(sizes)

        selected: List[int] = []
        for idx in cell_indices.values():
            if self.strategy == "oversample":
                # Keep originals, then draw extras with replacement.
                extra = target - len(idx)
                chosen = idx.tolist()
                if extra > 0:
                    chosen += rng.choice(idx, size=extra, replace=True).tolist()
            else:
                chosen = rng.choice(idx, size=target, replace=False).tolist()
            selected.extend(chosen)

        # The rows with no recorded group are returned, once each and unbalanced,
        # rather than dropped: deleting observations would be a silent loss, and the
        # siblings in this file (DisparateImpactRemover.transform, LabelMassager)
        # return their un-intervened rows too and count them. They are excluded from
        # every cell count and from group_totals_after, and reported under
        # n_rows_without_a_recorded_group.
        selected.extend(groupless.tolist())
        rng.shuffle(selected)
        selected_arr = np.asarray(selected)
        self._record_resample(
            cell_indices,
            selected_arr,
            group_vals,
            y_arr,
            target,
            n_without_group=len(groupless),
        )
        return df.iloc[selected_arr].reset_index(drop=True), y_arr[selected_arr]

    def _record_no_rows(self, n_rows: int, n_without_group: int = 0) -> None:
        """A call with no cell to balance is a could-not-check, not a repeat.

        BGL stage 2b (2026-09-17): the ``if not cell_indices`` early return
        skipped :meth:`_record_resample` entirely, so the object kept reporting
        the PREVIOUS call's numbers. Measured: after a 114-row resample, a
        0-row call left ``n_rows_out=114``,
        ``group_totals_after={'a': 76, 'b': 38}`` and three warnings standing on
        ``fit_result``, every one of them about a frame that was not the frame
        just handed in. The state is cleared and the refusal recorded instead.
        """
        if self.fit_result is None:
            raise RuntimeError("get_resampled_data() requires fit() to have run first")
        if n_without_group and n_without_group == n_rows:
            # BGL stage 4 (2026-09-27): reachable only since a missing protected
            # value stopped being the group "nan". Before, a frame in which EVERY
            # protected value is missing was balanced into one phantom group and
            # reported as a resample; the row count is not 0 here, so the message
            # above would have been false about the input.
            message = (
                f"nothing was resampled: all {n_rows} row(s) have a missing value in "
                f"{list(self.protected_attributes or [])}, so no row belongs to a known "
                "group, there is no (group, label) cell to balance and no balance to "
                "measure. The frame is returned unchanged."
            )
        else:
            message = (
                f"nothing was resampled: {n_rows} row(s) were passed in, so there is no "
                "(group, label) cell to balance and no balance to measure. The frame is "
                "returned unchanged."
            )
        self.resample_result = {
            "strategy": self.strategy,
            "balance_by": self.balance_by,
            "cell_sizes_before": {},
            "cell_sizes_after": {},
            "n_distinct_source_rows": {},
            "n_rows_out": int(n_rows),
            "n_duplicated_rows": 0,
            "n_effective_rows": int(n_rows),
            # None, never 1.0 or 0.0: nothing was replicated because nothing was
            # read, and a neutral number here reads as a measured perfect run.
            "max_replication_factor": None,
            "group_totals_after": {},
            "unreachable_cells": None,
            "n_rows_without_a_recorded_group": int(n_without_group),
            "could_not_check": message,
        }
        self.fit_result.fit_metrics.update(self.resample_result)
        self.fit_result.warnings = [
            w for w in self.fit_result.warnings if w not in self._disclosed
        ] + [message]
        self._disclosed = [message]
        warnings.warn(f"Resampler: {message}")

    def _record_resample(
        self,
        cell_indices: Dict[Any, np.ndarray],
        selected_arr: np.ndarray,
        group_vals: List[str],
        y_arr: np.ndarray,
        target: int,
        n_without_group: int = 0,
    ) -> None:
        """Disclose what the resample actually did, on ``fit_result``.

        BGL stage 2 (2026-09-16): ``get_resampled_data`` returned a bare
        ``(DataFrame, ndarray)`` and ``fit_result.warnings == []``, an empty list
        that reads as "the balance succeeded, nothing to report". Measured on 59
        'a' rows and 1 'b' row: the frame came back with group b at 30 rows drawn
        from exactly 1 distinct observation (and, balancing by group only, group
        totals {'a': 76, 'b': 38}, not balanced at all). The row count counted
        copies as observations and no channel carried either fact. The sibling
        SyntheticResampler already discloses the absent-cell half on the
        identical frame, so this is the same disclosure, plus a replication count
        the synthetic path does not need.
        """
        if self.fit_result is None:
            raise RuntimeError("get_resampled_data() requires fit() to have run first")

        # BGL stage 2b (2026-09-17): the disclosure was keyed by the RENDERED
        # label, and two cells can render alike: a missing label is a distinct
        # dict key per row (nan is not equal to itself), so every one of them
        # printed as "(a, label=nan)" and the dicts overwrote each other.
        # Measured on 60 input rows with two missing labels: cell_sizes_before
        # summed to 59, cell_sizes_after credited one rendered cell with 112 rows,
        # and unreachable_cells listed "(b, label=nan)" twice. The label is made
        # unique per CELL IDENTITY, so the counts add up to the rows that went in.
        def _render(key: Any) -> str:
            if self.balance_by == "group_label":
                g, lab = key
                return f"({g}, label={lab})"
            return str(key)

        label_of: Dict[Any, str] = {}
        used: Dict[str, int] = {}
        n_missing_label = 0
        for k in cell_indices:
            base = _render(k)
            seen = used.get(base, 0)
            used[base] = seen + 1
            label_of[k] = base if seen == 0 else f"{base} #{seen + 1}"
            if self.balance_by == "group_label":
                lab = k[1]
                if isinstance(lab, float) and np.isnan(lab):
                    n_missing_label += len(cell_indices[k])

        def _label(key: Any) -> str:
            return label_of[key]

        sizes_before = {_label(k): int(len(v)) for k, v in cell_indices.items()}
        source_of = {i: k for k, idx in cell_indices.items() for i in idx.tolist()}
        sizes_after: Dict[str, int] = {}
        distinct_after: Dict[str, set] = {}
        for pos in selected_arr.tolist():
            # A row with no recorded group belongs to no cell, so it is in the
            # returned frame but in none of the cell counts (see get_resampled_data).
            if pos not in source_of:
                continue
            key = _label(source_of[pos])
            sizes_after[key] = sizes_after.get(key, 0) + 1
            distinct_after.setdefault(key, set()).add(pos)

        replication = {
            k: (sizes_after[k] / len(distinct_after[k])) for k in sizes_after if distinct_after[k]
        }
        max_replication = max(replication.values()) if replication else 1.0
        worst_cell = max(replication, key=lambda k: replication[k]) if replication else None
        # The groupless rows are observations, each returned once, so they count as
        # neither copies nor cell members: without this they read as duplicates.
        n_distinct_selected = int(sum(len(v) for v in distinct_after.values()) + n_without_group)
        n_duplicated = int(len(selected_arr) - n_distinct_selected)

        group_after: Dict[str, int] = {}
        for pos in selected_arr.tolist():
            if pos not in source_of:
                continue
            g = str(group_vals[pos])
            group_after[g] = group_after.get(g, 0) + 1

        # A (group, label) cell that does not exist cannot be resampled from, so
        # the requested balance is unreachable. Reported, not repaired: inventing
        # rows for it would fabricate the very thing the caller wants measured.
        absent: List[str] = []
        if self.balance_by == "group_label":
            observed_groups = {str(g) for g, _ in cell_indices}
            # A missing label is not a class, so it is not one of the labels every
            # group is expected to carry; without this it entered the product and
            # produced a duplicated "(g, label=nan)" entry per NaN-labelled row.
            observed_labels = {
                lab for _, lab in cell_indices if not (isinstance(lab, float) and np.isnan(lab))
            }
            present = {(str(g), lab) for g, lab in cell_indices}
            absent = sorted(
                f"({g}, label={lab})"
                for g in observed_groups
                for lab in observed_labels
                if (g, lab) not in present
            )

        disclosures: List[str] = []
        if n_without_group:
            disclosures.append(
                f"{n_without_group} row(s) have a missing value in "
                f"{list(self.protected_attributes or [])}, so they belong to no known "
                "group: they cannot be assigned to a (group, label) cell, they were "
                "excluded from the balancing and from the cell and group counts below, "
                "and they are returned UNBALANCED in the same frame as the balanced "
                "rows. They are counted under n_rows_without_a_recorded_group, which is "
                "what n_rows_out has that group_totals_after does not."
            )
        if n_missing_label:
            disclosures.append(
                f"{n_missing_label} row(s) carry a MISSING label. A missing label is not "
                "a class, so each such row formed a single-row cell of its own and was "
                "replicated up to the target: those rows are copies of one observation, "
                "not a balanced class, and the group totals below count them."
            )
        # One observed protected group is not a balanced one: there is no second
        # group to balance against, so between-group balance was not measured at
        # all. `unreachable_cells == []` is true and says nothing about that.
        # Only the rows that HAVE a group: a missing protected value is not a second
        # group to balance against, and while it was counted as one this refusal
        # could not fire on the frame it was written for (see _group_key_series).
        observed_group_keys = sorted({str(g) for g in group_vals if not pd.isna(g)})
        if len(observed_group_keys) < 2:
            disclosures.append(
                "between-group balance was NOT measured: only one protected group "
                f"({observed_group_keys}) appears in these rows, so there is no second "
                "group to balance against. Whatever the cell sizes are now, this run "
                "neither equalised nor could equalise group representation."
            )
        if absent:
            disclosures.append(
                f"not group-balanced: {len(absent)} (group, label) cell(s) have no "
                "examples to resample from, so they stay empty and the group totals "
                f"remain uneven: {', '.join(absent[:8])}"
                + (" ..." if len(absent) > 8 else "")
                + f". Every cell that does exist was balanced to {target} rows."
            )
        if len({v for v in group_after.values()}) > 1:
            disclosures.append(
                f"group totals after resampling are uneven: {group_after}. The requested "
                f"balance_by={self.balance_by!r} balance was not reached."
            )
        if max_replication > self._REPLICATION_WARN_FACTOR:
            disclosures.append(
                f"{n_duplicated} of {len(selected_arr)} returned row(s) are copies, not "
                f"observations: cell {worst_cell} was expanded {max_replication:.1f}x from "
                f"{len(distinct_after[str(worst_cell)])} distinct row(s). Above "
                f"{self._REPLICATION_WARN_FACTOR}x that cell is a could-not-check, not a "
                "balanced cell: the effective sample size did not grow with the row count."
            )
        elif n_duplicated:
            disclosures.append(
                f"{n_duplicated} of {len(selected_arr)} returned row(s) are copies drawn "
                "with replacement, not new observations."
            )
        if self.strategy == "undersample":
            # Counted against the rows that were IN a cell: the groupless rows are
            # in selected_arr but were never candidates to discard, and including
            # them made this difference read low (negative, for an all-missing frame).
            dropped = int(
                sum(len(v) for v in cell_indices.values()) - (len(selected_arr) - n_without_group)
            )
            if dropped:
                disclosures.append(
                    f"undersampling discarded {dropped} observed row(s) to reach {target} per cell."
                )

        self.resample_result = {
            "strategy": self.strategy,
            "balance_by": self.balance_by,
            "cell_sizes_before": sizes_before,
            "cell_sizes_after": sizes_after,
            "n_distinct_source_rows": {k: len(v) for k, v in distinct_after.items()},
            "n_rows_out": int(len(selected_arr)),
            "n_duplicated_rows": n_duplicated,
            "n_effective_rows": n_distinct_selected,
            "max_replication_factor": float(max_replication),
            "group_totals_after": group_after,
            "unreachable_cells": absent,
            "n_rows_without_a_recorded_group": int(n_without_group),
            # BGL6 F02 (2026-09-29): WRITTEN, as None, rather than left out. The
            # metrics go onto fit_result through .update(), so the key a refused
            # call wrote in _record_no_rows survived into every later call that did
            # not write it. Measured: an all-missing 40-row frame followed by a
            # fully measured resample of a healthy two-group frame published all
            # four cells at 15, n_rows_out 60,
            # n_rows_without_a_recorded_group 0, warnings == [] AND a
            # could_not_check about "all 40 row(s)" nobody had just passed in. A
            # could-not-check reported for a measurement that WAS made is this
            # campaign's defect running backwards. None is the third state, not the
            # absence of the key: a reader can tell "this run checked" from "this
            # field was never populated".
            "could_not_check": None,
        }
        self.fit_result.fit_metrics.update(self.resample_result)
        # Replace the previous call's disclosures rather than stacking them.
        self.fit_result.warnings = [
            w for w in self.fit_result.warnings if w not in self._disclosed
        ] + disclosures
        self._disclosed = disclosures
        for message in disclosures:
            warnings.warn(f"Resampler: {message}")


class FairRepresentationTransformer(BaseFeatureTransformer):
    """
    Learns a latent feature representation that preserves task-relevant signal
    while being (approximately) invariant to the protected attribute, following
    the fair-representation-learning family (Zemel et al. 2013; Louizos et al.
    2016 VFAE; Madras et al. 2018 LAFTR).

    A small autoencoder encodes the (standardized) numeric features into a
    ``representation_dim`` latent space, trained with a reconstruction loss plus
    an adversary that tries to predict the protected group from the latent code.
    A gradient-reversal layer flips the adversary's gradient into the encoder, so
    the encoder is pushed to *hide* group membership while still reconstructing
    the inputs. ``transform`` then returns the latent code (columns ``rep_0..``)
    as the new feature set; protected attributes are excluded (the consumer
    re-adds them). Multiple protected attributes are handled jointly via their
    intersectional group code.

    Requires torch (optional dependency); ``fit`` raises a clear error if torch
    is unavailable. Non-numeric features are dropped from the representation.

    Args:
        protected_attributes: Names of protected attribute columns.
        representation_dim: Size of the latent fair representation.
        lambda_fairness: Strength of the adversarial (invariance) penalty.
        alpha_reconstruction: Weight of the reconstruction loss.
        epochs: Training epochs.
        lr: Adam learning rate.
        random_state: Seed for reproducibility.

    References:
        - Zemel, Wu, Swersky, Pitassi, Dwork (2013): Learning Fair Representations.
        - Louizos et al. (2016): The Variational Fair Autoencoder.
        - Madras et al. (2018): Learning Adversarially Fair and Transferable Reps.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: fair_representation. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(
        self,
        protected_attributes: Optional[List[str]] = None,
        representation_dim: int = 8,
        lambda_fairness: float = 1.0,
        alpha_reconstruction: float = 1.0,
        epochs: int = 200,
        lr: float = 0.01,
        random_state: int = 42,
    ):
        super().__init__(protected_attributes)
        self.representation_dim = int(representation_dim)
        self.lambda_fairness = float(lambda_fairness)
        self.alpha_reconstruction = float(alpha_reconstruction)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.random_state = int(random_state)
        # Fitted torch encoder module (untyped third-party -> Any), or None.
        self._encoder: Optional[Any] = None
        self._feature_mu: Optional[np.ndarray] = None
        self._feature_sd: Optional[np.ndarray] = None
        self._numeric_features: List[str] = []
        self._rep_dim_eff = 0
        # Set by transform(): how many feature values it had to substitute.
        self.transform_result: Dict[str, Any] = {}

    def _group_codes(self, df: pd.DataFrame) -> np.ndarray:
        """Intersectional group code per row, and ``-1`` where the protected value
        is MISSING.

        BGL stage 3 (2026-09-27): ``astype(str)`` turned a missing protected value
        into the literal level ``"nan"``, and np.unique then counted it as a group.
        Measured on 60 rows whose 'gender' held ONE observed level plus 10 missing
        values: ``n_groups`` published 2 with ``fit_result.warnings == []`` and no
        Python warning, byte-identical to a genuine two-group run, and the
        adversary was trained to hide value-present from value-missing, which is
        not a protected group. The two siblings in this file that key on a group,
        DisparateImpactRemover and LabelMassager, already mask the key for exactly
        this reason. A row with no group carries no group membership to hide, so it
        is excluded from the adversary's target and disclosed by fit().
        """
        attrs = self.protected_attributes
        assert attrs is not None  # set in fit()
        sub = df[list(attrs)]
        missing = sub.isna().any(axis=1).to_numpy()
        if len(attrs) == 1:
            keys = sub[attrs[0]].astype(str).to_numpy()
        else:
            keys = sub.astype(str).agg("|".join, axis=1).to_numpy()
        codes = np.full(len(df), -1, dtype=np.int64)
        if (~missing).any():
            _, inverse = np.unique(keys[~missing], return_inverse=True)
            codes[~missing] = inverse
        return codes

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[ArrayLike] = None,
        protected_attributes: Optional[List[str]] = None,
    ) -> "FairRepresentationTransformer":
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise ImportError(
                "FairRepresentationTransformer requires torch. Install with "
                "`pip install torch` to learn fair representations."
            ) from exc

        attrs = self._validate_protected_attributes(X, protected_attributes)
        self.protected_attributes = attrs
        df = self._to_dataframe(X)
        self.feature_names_in_ = list(df.columns)
        feature_cols = self._extract_feature_columns(df, attrs)
        self._numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        n_feat = len(self._numeric_features)
        if n_feat == 0:
            raise ValueError("FairRepresentationTransformer needs at least one numeric feature.")
        # BGL stage 2 (2026-09-16): fit() on a 0-row frame returned normally with
        # is_fitted=True and fit_metrics byte-identical to a real run, after
        # observing zero rows and zero group levels. numpy's own "Mean of empty
        # slice" was the only signal, and the encoder it produced then returned a
        # 100% NaN frame for 60 REAL rows with warnings == []. An encoder fitted
        # on nothing is not a weak measurement, it is no measurement.
        if len(df) == 0:
            raise ValueError(
                "FairRepresentationTransformer.fit needs at least one row; got 0. An "
                "encoder fitted on zero rows has NaN standardisation statistics and "
                "returns an all-NaN representation."
            )

        rep_dim = max(1, min(self.representation_dim, n_feat))
        self._rep_dim_eff = rep_dim
        self.feature_names_out_ = [f"rep_{i}" for i in range(rep_dim)]

        # BGL stage 4 (2026-09-27): this was
        # ``df[...].astype(float).fillna(0.0).values``, so a never-observed feature
        # value entered the encoder as 0.0 AND entered the standardisation
        # statistics, uncounted, and transform() then imputed every later missing
        # value with that corrupted mean while transform_result labelled it the
        # "fit-time mean". Measured on 60 rows with 30 of the 'f1' values missing:
        # _feature_mu [50.08643804, 0.13323775] and _feature_sd [50.17814817,
        # 1.00618775] where the mean of the 30 OBSERVED f1 values is
        # 100.17287608734327, with fit_result.warnings == [] and no Python warning,
        # byte-identical in shape to a complete-data fit. The statistics are taken
        # over the OBSERVED values now (nanmean/nanstd) and the substituted cells
        # are filled with that observed mean, which standardises to 0, the centre,
        # exactly as this class's own transform() already does for the same input;
        # the substitution is counted on fit_metrics and disclosed on both channels.
        # A column with NO observed value has no mean to impute with and is refused
        # rather than silently standardised around an invented 0.0. A cell counts as
        # unobserved when it is not FINITE, so an inf is substituted and counted as
        # well: fillna() left it in place and one inf made the whole column's mean
        # inf, which is no more a fit-time mean than the 50.086438 above.
        Xnum_raw = df[self._numeric_features].astype(float).to_numpy(dtype=float)
        na_cells = ~np.isfinite(Xnum_raw)
        n_values_substituted = int(na_cells.sum())
        substituted_per_feature = {
            c: int(na_cells[:, j].sum()) for j, c in enumerate(self._numeric_features)
        }
        all_missing = [
            c for j, c in enumerate(self._numeric_features) if bool(na_cells[:, j].all())
        ]
        if all_missing:
            raise ValueError(
                f"FairRepresentationTransformer.fit: feature(s) {all_missing} have no "
                "finite value in these rows, so there is no observed mean to standardise "
                "or to impute with. Filling them with 0.0 would put an invented value in "
                "the encoder and in the fit-time statistics every later transform() "
                "imputes with; drop the column or supply values for it."
            )
        observed_only = np.where(na_cells, np.nan, Xnum_raw)
        observed_mu = np.nanmean(observed_only, axis=0)
        observed_sd = np.nanstd(observed_only, axis=0)
        Xnum = np.where(na_cells, np.broadcast_to(observed_mu, Xnum_raw.shape), Xnum_raw)
        self._feature_mu = observed_mu
        self._feature_sd = observed_sd
        # np.unique on the RAW column, not a std compared against 0.0: an
        # accumulated std of a constant column is exactly 0.0 only at some n.
        constant_cols = np.array(
            [np.unique(Xnum[:, j]).size < 2 for j in range(Xnum.shape[1])], dtype=bool
        )
        self._feature_sd[constant_cols] = 1.0
        self._feature_sd[~np.isfinite(self._feature_sd)] = 1.0
        Xs = (Xnum - self._feature_mu) / self._feature_sd
        codes = self._group_codes(df)
        # BGL stage 2 (2026-09-16): `max(2, ...)` was BOTH the adversary's output
        # width and the number reported as a measurement, so a column with ONE
        # observed level published n_groups=2 with warnings == [], byte-identical
        # to a genuine 2-group run. The floor can only fire when the true count is
        # below 2, so the substituted 2 marked exactly the case it concealed. The
        # architectural width and the measurement are separate values now.
        # -1 marks a row with no known group (see _group_codes): it is not a level,
        # so it is neither counted nor used as an adversary target.
        has_group = codes >= 0
        n_rows_without_group = int((~has_group).sum())
        n_groups_observed = int(np.unique(codes[has_group]).size) if has_group.any() else 0
        adversary_out_dim = max(2, n_groups_observed)

        torch.manual_seed(self.random_state)
        hidden = max(rep_dim, min(32, max(8, n_feat)))

        class _GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lambd):
                ctx.lambd = lambd
                return x.view_as(x)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output.neg() * ctx.lambd, None

        encoder = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, rep_dim))
        decoder = nn.Sequential(nn.Linear(rep_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_feat))
        adversary = nn.Sequential(
            nn.Linear(rep_dim, hidden), nn.ReLU(), nn.Linear(hidden, adversary_out_dim)
        )

        Xt = torch.tensor(Xs, dtype=torch.float32)
        # The adversary is trained only on rows that HAVE a group. grp_rows is None
        # when every row does, so the common path indexes nothing and its training
        # trajectory is unchanged by this guard.
        grp_positions = np.flatnonzero(has_group)
        grp_rows = (
            None if grp_positions.size == len(df) else torch.tensor(grp_positions, dtype=torch.long)
        )
        gt = torch.tensor(codes[grp_positions], dtype=torch.long)
        params = (
            list(encoder.parameters()) + list(decoder.parameters()) + list(adversary.parameters())
        )
        opt = torch.optim.Adam(params, lr=self.lr)
        mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()

        encoder.train()
        decoder.train()
        adversary.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            z = encoder(Xt)
            recon = decoder(z)
            loss = self.alpha_reconstruction * mse(recon, Xt)
            if gt.numel():
                # Gradient reversal: adversary minimizes its own loss, but the
                # reversed gradient pushes the encoder to make the group
                # UN-predictable. With no row carrying a group there is no group
                # membership to hide, so the term is absent rather than computed
                # against a target invented from the missingness.
                z_adv = z if grp_rows is None else z[grp_rows]
                adv_logits = adversary(_GradReverse.apply(z_adv, self.lambda_fairness))
                loss = loss + ce(adv_logits, gt)
            loss.backward()
            opt.step()

        encoder.eval()
        self._encoder = encoder

        fit_warnings: List[str] = []
        dropped = [c for c in feature_cols if c not in self._numeric_features]
        if dropped:
            fit_warnings.append(
                f"Non-numeric features dropped from the representation: {dropped[:8]}"
            )
        if n_values_substituted:
            # The sibling LabelMassager.fit publishes n_features_imputed on
            # fit_metrics AND in fit_result.warnings for exactly this input, and this
            # class's own transform() counts its substitutions in transform_result;
            # only this fit was silent about the ones it makes.
            named = {c: n for c, n in substituted_per_feature.items() if n}
            fit_warnings.append(
                f"{n_values_substituted} non-finite feature value(s) {named} were "
                "substituted with the OBSERVED mean of their own column before fitting "
                "the encoder; the representation of those rows rests on a substituted "
                "value, not on an observed one. The standardisation statistics reported "
                "as the fit-time mean are computed over the observed values only, so "
                "they are not themselves built on the substitution."
            )
            warnings.warn(f"FairRepresentationTransformer: {fit_warnings[-1]}")
        if n_rows_without_group:
            fit_warnings.append(
                f"{n_rows_without_group} of {len(df)} row(s) have a missing value in "
                f"{list(attrs)}, so they belong to no known group: they are excluded from "
                "the adversary's target and the invariance penalty says nothing about "
                "them. They are still encoded and still reconstructed."
            )
            warnings.warn(f"FairRepresentationTransformer: {fit_warnings[-1]}")
        if n_groups_observed < 2:
            fit_warnings.append(
                f"Protected attribute(s) {list(attrs)} yield {n_groups_observed} observed "
                "group(s), fewer than the 2 an adversary needs: there is no group "
                "membership to hide, so NO fairness constraint was applied and the "
                "representation is an ordinary autoencoding. The adversary head was "
                f"still built with {adversary_out_dim} outputs for architectural reasons; "
                "that width is reported separately and is not a group count."
            )
            warnings.warn(f"FairRepresentationTransformer: {fit_warnings[-1]}")

        self.fit_result = TransformationResult(
            method="fair_representation",
            n_features_original=len(feature_cols),
            n_features_transformed=rep_dim,
            n_samples=len(df),
            features_modified=self._numeric_features,
            fit_metrics={
                "protected_attributes": list(attrs),
                "representation_dim": rep_dim,
                "lambda_fairness": self.lambda_fairness,
                "alpha_reconstruction": self.alpha_reconstruction,
                "n_groups": n_groups_observed,
                "adversary_output_dim": int(adversary_out_dim),
                "n_rows_without_a_recorded_group": n_rows_without_group,
                "n_rows_in_the_adversary_target": int(len(df) - n_rows_without_group),
                "n_feature_values_substituted": n_values_substituted,
                "substituted_per_feature": substituted_per_feature,
                "feature_statistics": "observed values only (missing cells excluded)",
            },
            warnings=fit_warnings,
        )
        self.is_fitted = True
        return self

    def transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> Union[pd.DataFrame, np.ndarray]:
        self._check_is_fitted()
        import torch

        # Encoder and standardization stats are populated by fit().
        assert self._encoder is not None
        assert self._feature_mu is not None and self._feature_sd is not None
        was_dataframe = isinstance(X, pd.DataFrame)
        df = self._to_dataframe(X, self.feature_names_in_)

        # BGL stage 2 (2026-09-16): `reindex(..., fill_value=0.0)` made a DROPPED
        # column and a present-but-NaN column indistinguishable, and the
        # substituted 0.0 was not even the fit-time mean. Measured: fit-time
        # income mu/sd 51278.5 / 11463.2, so a never-observed value entered the
        # encoder at -4.4733 sd, and a frame carrying NONE of the three fitted
        # features still returned a confident [0.185392, 0.136378]. A missing
        # fitted feature is refused; a NaN is imputed with the fit-time mean
        # (which standardises to 0, the centre rather than the tail) and counted.
        missing = [c for c in self._numeric_features if c not in df.columns]
        if missing:
            raise ValueError(
                f"FairRepresentationTransformer: column(s) required by fit are absent: "
                f"{missing}. The encoder was fitted on {self._numeric_features}; a "
                "representation cannot be computed from features that were never supplied."
            )
        Xnum = df[self._numeric_features].astype(float)
        na_mask = Xnum.isna().to_numpy()
        n_imputed = int(na_mask.sum())
        Xnum = Xnum.to_numpy(dtype=float)
        if n_imputed:
            mu_row = np.asarray(self._feature_mu, dtype=float)
            Xnum = np.where(na_mask, np.broadcast_to(mu_row, Xnum.shape), Xnum)
        self.transform_result = {
            "n_rows": int(len(df)),
            "n_values_imputed": n_imputed,
            "imputed_per_feature": {
                c: int(na_mask[:, j].sum()) for j, c in enumerate(self._numeric_features)
            },
            "imputation": "fit-time mean",
        }
        if n_imputed:
            warnings.warn(
                f"FairRepresentationTransformer: {n_imputed} missing feature value(s) "
                "were imputed with the fit-time mean before encoding; the representation "
                "for those rows rests on a substituted value. See .transform_result."
            )
        Xs = (Xnum - self._feature_mu) / self._feature_sd
        with torch.no_grad():
            z = self._encoder(torch.tensor(Xs, dtype=torch.float32)).numpy()
        rep = pd.DataFrame(z, columns=self.feature_names_out_, index=df.index)
        if was_dataframe:
            return rep
        return rep.values
