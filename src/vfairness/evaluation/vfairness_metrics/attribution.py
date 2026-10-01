"""
Feature Attribution Explainer (per-decision XAI).

Answers the GDPR Art. 22 / EU AI Act Art. 13 "right to explanation" question:
*which input features drove this model's decision, and in which direction?*

Two scopes:
  * GLOBAL  -- which features matter most across a dataset (permutation
    importance: shuffle a column, measure how much the model's predictions /
    score degrade).
  * LOCAL   -- for ONE individual, which features pushed the decision up or
    down relative to a baseline (occlusion: replace each feature with the
    population baseline and measure the prediction shift). A model-agnostic,
    dependency-light approximation of a Shapley/SHAP value.

Design principles (matches the library's minimal-deps rule):
  * Works on ANY callable `predict(X) -> array` -- a fitted sklearn estimator,
    a wrapped API client, or a plain function. No model-format assumptions.
  * `scikit-learn` is OPTIONAL, used only to speed up global permutation
    importance with a proper scorer; a NumPy fallback runs when it is absent.
  * `shap` is OPTIONAL; if installed and requested, exact SHAP values can be
    used for local attribution. Otherwise the occlusion approximation is used.
  * No torch / pandas hard requirement at import time.

This module is intentionally NOT imported from the package __init__ so that
`import vfairness` never pulls in sklearn. Import it explicitly:

    from vfairness.evaluation.vfairness_metrics.attribution import (
        FeatureAttributionExplainer,
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from vfairness._not_assessed import NOT_ASSESSED
from vfairness._triage import is_measured

logger = logging.getLogger(__name__)

# Result containers


@dataclass
class FeatureContribution:
    """One feature's attribution."""

    feature: str
    # Magnitude (always >= 0), or NaN when the attribution could NOT be
    # measured. NaN is a third state and must not be read as 0.0: see
    # `direction == 'not_assessed'` and the sibling note on AttributionResult.
    importance: float
    direction: str  # 'increase' | 'decrease' | 'neutral' | 'not_assessed'
    signed_value: Optional[
        float
    ]  # signed contribution / signed score drop; None when the path has no sign (label-free numpy permutation)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionResult:
    """Container for a global or local attribution run."""

    scope: str  # 'global' | 'local'
    method: str  # 'permutation' | 'permutation_numpy' | 'occlusion' | 'shap'
    contributions: List[FeatureContribution] = field(default_factory=list)
    base_value: Optional[float] = None  # local: baseline prediction
    prediction: Optional[float] = None  # local: this individual's prediction
    notes: List[str] = field(default_factory=list)

    def top(self, k: int = 5) -> List[FeatureContribution]:
        """The ``k`` features with the largest MEASURED importance, strongest first.

        A feature whose importance is not a measurement can never occupy a top
        slot, and its exclusion is said out loud.

        BGL grade-1 G01, 2026-09-30. NaN is a first-class state of
        ``importance`` here (see :class:`FeatureContribution` and
        :func:`_all_not_assessed`): a constant column is never actually
        permuted, a permuted R^2 can come back non-finite, a Shapley background
        can be unusable. ``sorted(key=lambda c: c.importance)`` puts such a
        feature WHEREVER THE INPUT ORDER LEFT IT, because every comparison
        against NaN is False and Timsort therefore never moves it. The features
        arrive in column order, so column 0 stays at position 0. Measured on
        this repo through the public entry point, a 60-row dataset whose first
        column is constant and whose real driver is column 1::

            global_importance(X, y=y, n_repeats=5).top(1)
              -> [('constant_col', nan, 'not_assessed')]

        i.e. the one feature the library explicitly REFUSED to measure was
        published as the single strongest driver of the model, ahead of a
        measured importance of 1.0091. ``notes`` said exactly what had happened
        and ``top()`` contradicted it. The whole purpose of this method is to
        name the drivers a reader should act on, so a could-not-check in slot 1
        is the could-not-check-as-measurement defect on the surface where it
        costs the most.

        Keyed on :func:`vfairness._triage.is_measured` of the IMPORTANCE, never
        on ``direction``: ``direction == 'not_assessed'`` is also produced by the
        label-free numpy path, where the magnitude IS measured and only the SIGN
        is not (pinned in ``tests/test_bgl_stage2b_s2b09.py``), so reading the
        direction would throw away real evidence. ``is_measured`` also refuses
        the infinities and a bool, and accepts np.float32/np.float64.

        Unmeasured contributions are NOT deleted: they stay in
        ``contributions`` and in :meth:`to_dict`, which is where a reader who
        needs the full list looks. Returning fewer than ``k`` entries, or none
        at all, is the honest answer to "which features drive this model" when
        no importance was measured.
        """
        ranked = [c for c in self.contributions if is_measured(c.importance)]
        excluded = [c.feature for c in self.contributions if not is_measured(c.importance)]
        if excluded:
            warnings.warn(
                f"AttributionResult.top: {len(excluded)} of {len(self.contributions)} "
                f"feature(s) have an importance that is not a measurement and were left "
                f"out of the ranking ({', '.join(str(f) for f in excluded)}). They are "
                f"COULD NOT CHECK, not a measured importance of any size, so they have "
                f"no rank; see notes for why each one could not be measured. This "
                f"ranking covers the remaining {len(ranked)} feature(s) only.",
                UserWarning,
                stacklevel=2,
            )
        return sorted(ranked, key=lambda c: float(c.importance), reverse=True)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "method": self.method,
            "base_value": self.base_value,
            "prediction": self.prediction,
            "notes": list(self.notes),
            "contributions": [c.to_dict() for c in self.contributions],
        }


# Helpers


def _as_2d_array(X: Any) -> "np.ndarray":
    """Coerce a DataFrame / list-of-rows / ndarray to a 2D ndarray."""
    if hasattr(X, "to_numpy"):  # pandas DataFrame
        return X.to_numpy()
    return np.asarray(X)


def _feature_names(X: Any, n_cols: int, provided: Optional[Sequence[str]]) -> List[str]:
    if provided is not None:
        return list(provided)
    if hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    return [f"feature_{i}" for i in range(n_cols)]


def _direction(delta: float, eps: float = 1e-9) -> str:
    """Which way the feature moved the score, or that nobody could tell.

    BGL-S2B (2026-09-17). A NaN delta failed both comparisons and fell through
    to "neutral", which is a VERDICT ("this feature has no directional effect")
    on the same scale as increase and decrease. Verified at the helper:
    ``_direction(float("nan"))`` returned 'neutral'. A delta that is not a
    number was not measured, and the dataclass already names the third state.
    """
    if not np.isfinite(delta):
        return NOT_ASSESSED
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "neutral"


def _column_rearranged(before: "np.ndarray", after: "np.ndarray") -> bool:
    """Did the shuffle actually put this column in a DIFFERENT arrangement?

    BGL5 A-evaluation-4, 2026-09-27. Permutation importance measures what
    happens when a column's link to the rows is broken. A shuffle that lands on
    the ORIGINAL order breaks nothing, so its "score drop" is exactly 0.0 by
    construction and is not a measurement of anything. Two inputs make that the
    normal case rather than a fluke, and the row-count guard above catches
    neither:

    * n = 2. A two-row column has exactly TWO arrangements, one of them the
      identity, so ``rng.shuffle`` lands on the identity about half the time.
    * a CONSTANT column, at any n. Every arrangement of it is value-identical to
      the original, so no shuffle can ever move it.

    Compared by VALUE, not by index permutation, because that is what makes the
    constant column fall out of the same test: the model cannot tell the two
    apart either. ``equal_nan=True`` so a column carrying NaN is not read as
    "moved" every single time (``nan != nan``); the fallback covers dtypes on
    which numpy refuses that keyword.
    """
    try:
        return not np.array_equal(before, after, equal_nan=True)
    except TypeError:  # pragma: no cover - dtype-dependent numpy behaviour
        return not np.array_equal(before, after)


def _never_permuted_reason(
    unmeasured: Sequence[str], n_features: int, n_repeats: int, n_rows: int
) -> str:
    """The sentence for features whose column no shuffle ever rearranged.

    BGL5 A-evaluation-4, 2026-09-27. Measured on this repo, model
    y = 2*x0 + 0.5*x1 + 0.1*x2 and X = rng(7).normal(0, 1, (200, 3)):

        global_importance(X[:2], n_repeats=1, random_state=0)
            before -> f0 0.0, f1 0.0, f2 0.0, 0 warnings
            after  -> all three NaN, one warning naming 0 of 3 measured
        global_importance(X[:2], n_repeats=3, random_state=0)
            before -> f0 0.0 (the DOMINANT feature, weight 2.0),
                      f1 0.2511, f2 0.0239, 0 warnings
            after  -> f0 NaN, f1 and f2 keep a measured number, one warning
        a constant column f1 held at 4.2 over all 200 rows, n_repeats=3
            before -> f1 0.0, silent
            after  -> f1 NaN, one warning

    The same rule was fixed on the same day in llm/counterfactual.py ("a shuffle
    that lands on the ORIGINAL order tests nothing"). n >= 2 was the old guard,
    and a threshold on n cannot make a permutation exist.
    """
    return (
        f"{len(unmeasured)} of {n_features} feature(s) were never actually "
        f"permuted: every one of the {n_repeats} shuffle(s) of "
        f"{', '.join(unmeasured)} returned the column to its original "
        f"arrangement, so nothing was broken and no drop was measured. A column "
        f"of {n_rows} row(s) has few arrangements (at n=2 a shuffle lands on the "
        f"identity about half the time) and a CONSTANT column has only one at any "
        f"n. Their importance is NaN (could not check), not 0.0, which would state "
        f"that the feature does not drive the model."
    )


def _all_not_assessed(
    names: Sequence[str],
    scope: str,
    method: str,
    reason: str,
    base_value: Optional[float] = None,
    prediction: Optional[float] = None,
) -> "AttributionResult":
    """Every feature as could-not-check: NaN magnitude, ``not_assessed``, no sign."""
    return AttributionResult(
        scope=scope,
        method=method,
        contributions=[
            FeatureContribution(
                feature=name,
                importance=float("nan"),
                direction=NOT_ASSESSED,
                signed_value=None,
            )
            for name in names
        ],
        base_value=base_value,
        prediction=prediction,
        notes=[reason],
    )


def _baseline_columns_without_a_value(bg: "np.ndarray") -> Optional[List[int]]:
    """Which columns of ``background`` cannot produce a baseline value.

    BGL5 A-evaluation-4, 2026-09-27. ``_occlusion_local`` derives EVERYTHING it
    reports from ``np.median(background, axis=0)`` and never checked that the
    median exists. The refusal that was pinned for an all-NaN background came
    from the FIXTURE's model propagating NaN through a matrix multiply, not from
    this class, so any predict that fills missing values (SimpleImputer plus
    LinearRegression, HistGradientBoosting, a wrapped API client: the module
    docstring names "a fitted sklearn estimator" as supported) answered an
    unusable background with confident, signed, directional attributions.

    Measured before this guard, model y = 2*x0 + 0.5*x1 + 0.1*x2 with
    NaN-imputing predict, X = rng(7).normal(0, 1, (200, 3)):

        explain_decision(X[0], np.full((50, 3), np.nan))
            before -> f0 0.5031 increase, f1 0.1899 increase, f2 0.0204
                      decrease, base_value -0.5481, notes [], SILENT under
                      warnings.simplefilter("error")
            the same call over the REAL 200-row background
                   -> f0 0.4985 increase, f1 0.1746 increase, f2 0.0201
                      decrease, base_value -0.5285
        The two differ by under 2 percent, so nothing in the result told a
        reader that one baseline came from 200 rows and the other from none.
        An empty (0, 3) background produced the identical output, disclosed by
        nothing but numpy's "Mean of empty slice" RuntimeWarning.

        after -> every affected contribution NaN with direction
                 'not_assessed', base_value NaN, a note, and a UserWarning.

    Returns the offending column indices, or ``None`` when the question could
    not be asked at all (a dtype numpy cannot take a median of), in which case
    the caller must leave the behaviour exactly as it was rather than invent a
    refusal: ``_occlusion_local`` raises on such a background anyway.
    """
    if bg.shape[0] == 0:
        return list(range(bg.shape[1]))
    try:
        baseline = np.median(bg, axis=0)
        finite = np.isfinite(np.asarray(baseline, dtype=float))
    except (TypeError, ValueError):
        # LOGGED, never discarded: the answer here is "the question could not be
        # asked", and nobody can tell that from "the background is fine" without
        # this line. The caller leaves the behaviour unchanged on this path.
        logger.debug(
            "explain_decision could not take a median of the background (dtype %r), so "
            "whether it can form a baseline is UNKNOWN, not yes",
            getattr(bg, "dtype", None),
            exc_info=True,
        )
        return None
    return [int(j) for j in np.flatnonzero(~finite)]


def _scores_from_predict(predict: Callable, X: "np.ndarray") -> "np.ndarray":
    """Get a 1D numeric score per row. Accepts predict_proba-style 2D output."""
    out = np.asarray(predict(X))
    if out.ndim == 2:
        out = out[:, -1]  # probability matrix -> positive/last column
    return out.astype(float).ravel()


# Explainer


class FeatureAttributionExplainer:
    """Model-agnostic feature attribution.

    Parameters
    ----------
    predict : callable
        ``predict(X_2d) -> array``. May return labels, scores, or a 2D
        probability matrix (the last column is used).
    feature_names : sequence of str, optional
        Names for columns; inferred from a DataFrame when omitted.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: feature_attribution_explainer. See docs/BETA_GO_LIVE_PLAN.md for the
    batch definitions.
    (end Beta Go-Live proof status)
    """

    def __init__(self, predict: Callable, feature_names: Optional[Sequence[str]] = None):
        if not callable(predict):
            raise TypeError("predict must be callable: predict(X) -> array")
        self._predict = predict
        self._feature_names = list(feature_names) if feature_names is not None else None

    # GLOBAL

    def global_importance(
        self,
        X: Any,
        y: Optional[Sequence] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> AttributionResult:
        """Permutation importance across the dataset.

        When `y` is provided, computes a label-scored permutation importance: it
        shuffles each feature in turn and measures the drop in R^2 (through
        ``_r2_or_not_measured``, the one R^2 predicate this library has). When
        `y` is absent, falls back to a label-free NumPy permutation that measures
        the change in the model's predictions. This is a hand-rolled permutation
        importance, not a call to sklearn.inspection.permutation_importance.

        Fewer than two rows is refused on BOTH paths: with one row (or none) a
        column cannot be shuffled into any other arrangement, so no importance is
        measured for any feature. Importances are NaN with
        ``direction='not_assessed'``, never 0.0.
        """
        Xarr = _as_2d_array(X)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        names = _feature_names(X, Xarr.shape[1], self._feature_names)

        # THE GUARD SITS ABOVE THE DISPATCH. Permutation importance needs at
        # least two rows to have a permutation at all: shuffling a length-1
        # column is the identity, so every permuted score equals the base score
        # and every drop is exactly 0.0. The scored path below happens to refuse
        # a single row already (one target value is a constant target), so only
        # the label-free NumPy path was exposed, and that path is the DEFAULT
        # call. Measured 2026-09-27 on global_importance(X[:1]) against
        # y = 2*x0 + 0.5*x1 + 0.1*x2: f0/f1/f2 all importance 0.0, i.e. "no
        # feature drove this model", for the model's dominant feature. Zero rows
        # returned NaN by accident, through numpy's mean-of-empty-slice
        # RuntimeWarning rather than a statement about what was not measured.
        # Guarding here rather than inside _numpy_permutation keeps the two paths
        # from disagreeing about the same input again.
        if Xarr.shape[0] < 2:
            reason = (
                f"permutation importance needs at least 2 rows to permute a "
                f"column; X has {Xarr.shape[0]}. A column that short has only one "
                f"arrangement, so shuffling changes nothing and no importance was "
                f"measured for any feature. Importances are NaN (could not "
                f"check), not 0.0."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=2)
            return AttributionResult(
                scope="global",
                method="permutation" if y is not None else "permutation_numpy",
                contributions=[
                    FeatureContribution(
                        feature=name,
                        importance=float("nan"),
                        direction=NOT_ASSESSED,
                        signed_value=None,
                    )
                    for name in names
                ],
                notes=[reason],
            )

        # BGL5 A-evaluation-4, 2026-09-27. ALSO ABOVE THE DISPATCH: a repeat
        # budget below 1 makes the loop the importance is AVERAGED OVER empty, so
        # nothing is permuted and nothing is measured. Measured before this guard
        # on 200 rows against y = 2*x0 + 0.5*x1 + 0.1*x2, n_repeats=0:
        # every importance nan with notes == [] and the only disclosure numpy's
        # own "Mean of empty slice" RuntimeWarning plus "invalid value encountered
        # in scalar divide", i.e. the value was right by accident and said nothing
        # about what had not been measured. After: the same NaN, with the reason
        # in notes[0] and as a UserWarning, on BOTH paths.
        if int(n_repeats) < 1:
            reason = (
                f"permutation importance is the MEAN score drop over n_repeats "
                f"shuffles; n_repeats is {n_repeats}, so not one shuffle was "
                f"performed and no importance was measured for any feature. "
                f"Importances are NaN (could not check), not 0.0."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=2)
            return _all_not_assessed(
                names,
                "global",
                "permutation" if y is not None else "permutation_numpy",
                reason,
            )

        if y is not None:
            return self._scored_permutation(
                Xarr, np.asarray(y, dtype=float), names, n_repeats, random_state
            )

        return self._numpy_permutation(Xarr, names, n_repeats, random_state)

    def _scored_permutation(self, Xarr, y, names, n_repeats, random_state) -> AttributionResult:
        """Label-scored permutation importance: drop in R^2 when a column is
        shuffled. Higher drop => more important. Direction is the sign of the
        score drop (positive = permuting hurts the model = the feature helps).

        Computed directly against the predict callable so it needs no sklearn
        estimator protocol. R^2 comes from ``_r2_or_not_measured``, the predicate
        the library's other R^2 sites use, so the same target cannot get one
        answer here and a different one there.
        """
        # ONE R^2 PREDICATE, NOT A DISPATCH BETWEEN TWO THAT DISAGREED.
        # B4 tier-1 audit, 2026-09-30. This function used to take
        # sklearn.metrics.r2_score when it imported and fall back to a local
        # NumPy R^2 whose denominator was
        # ``float(np.sum((yt - np.mean(yt)) ** 2)) or 1e-12``. ``0.0 or 1e-12``
        # is 1e-12, so a target whose total sum of squares is genuinely zero got
        # a denominator of one part in a trillion and the quotient was published
        # as a measurement. Nothing warned, and the two arms did not even agree
        # about it.
        #
        # Measured on this repo, 200 rows, X = default_rng(7).normal(0, 1),
        # model weights [2.0, 0.5, 0.1], y = np.linspace(1e-200, 2e-200, 200):
        # 200 DISTINCT target values, so the constant-target guard below
        # correctly lets it through, but every squared deviation underflows and
        # ss_tot is exactly 0.0.
        #   fallback arm: f0 8.79e+12, f1 2.56e+13, f2 2.14e+12,
        #                 direction 'increase', notes [], ZERO warnings
        #   sklearn arm:  f0 0.0, f1 0.0, f2 0.0, direction 'neutral',
        #                 notes [], ZERO warnings
        # One arm fabricated a magnitude and the other fabricated a neutral
        # value, from one input, and which one a reader got depended on whether
        # sklearn happened to import.
        #
        # The predicate is IMPORTED rather than copied, for the same reason
        # _warn_dropped_groups is imported in regression.py: a third copy of the
        # "does this target have a denominator" test is a third answer. It is
        # imported INSIDE the function because regression.py reaches pandas
        # through ._statistics and this module's docstring promises no pandas at
        # import time.
        #
        # On healthy input it is sklearn's number to the bit. Measured on the
        # fixtures above: 0.9974384549178099 against r2_score's
        # 0.9974384549178099, and -5.162589530967523 against -5.162589530967523
        # for a genuinely bad model, so no measured importance moves.
        from .regression import _r2_or_not_measured

        # BGL-S2 (2026-09-16). R^2 is UNDEFINED on a constant target: its
        # denominator is the variance of y. sklearn.metrics.r2_score substitutes
        # 0.0 for that case and, verified by execution, raises no warning, so
        # base_score and every permuted_score came back 0.0 and every mean_drop
        # was exactly 0.0. Measured at this public entry against a model whose
        # true weights are [2.0, 0.5, 0.1]: y = np.ones(200) returned f0/f1/f2
        # all importance 0.0, direction 'neutral', notes [] -- three features
        # declared exactly neutral for a model that is dominated by the first of
        # them, with the one field that exists to carry a caveat left empty.
        #
        # Tested with np.unique on the RAW finite values, never a variance == 0
        # test: an accumulated variance is not reliably exactly zero.
        #
        # A TARGET WITH NON-FINITE ROWS IS NOT A SHORTER TARGET.
        # B4 tier-1 audit, 2026-09-30. ``y_finite`` exists because y can carry
        # NaN, and the guard below only ever asked whether two finite values were
        # LEFT; the full y was then handed to R^2, where one NaN makes every sum
        # NaN. Measured 2026-09-30 with 100 of 200 target rows set to NaN: the
        # sklearn arm raised ``ValueError: Input contains NaN`` out of sklearn's
        # own validator, and the NumPy arm returned NaN silently, which gives
        # every feature a NaN importance with an EMPTY notes list. Refused here
        # instead, with the count, so the could-not-check is the same three-state
        # disclosure as every other branch of this function and does not depend
        # on a dependency's input validation.
        n_not_finite = int(np.count_nonzero(~np.isfinite(y)))
        if n_not_finite:
            reason = (
                f"{n_not_finite} of {y.size} target value(s) are not finite "
                f"(NaN or infinite). R^2 is a sum over every row, so one such row "
                f"makes the score NaN and no permutation importance was measured "
                f"for any feature. Importances are NaN (could not check), not "
                f"0.0, and those rows were NOT silently dropped: permutation "
                f"importance over a subset of the rows is a different measurement "
                f"from the one that was asked for."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            return _all_not_assessed(names, "global", "permutation", reason)

        y_finite = y[np.isfinite(y)]
        n_distinct = int(np.unique(y_finite).size)
        # THE DENOMINATOR TEST IS NOT np.unique ALONE.
        # B4 tier-1 audit, 2026-09-30. np.unique is EXACT, so a target that is
        # constant only to the resolution of the arithmetic that produced it has
        # many distinct values and walked straight past this guard. Measured on
        # 200 rows with y = (base + 100.1) - base for
        # base = np.linspace(1e6, 1e8, 200), i.e. 100.1 carrying 7.45e-09 of
        # rounding inherited from 1e8 arithmetic: 5 distinct values, ss_tot
        # 2.28e-15, and the published importances were f0 3.86e+15, f1 1.13e+16,
        # f2 9.42e+14, direction 'increase', notes [], ZERO warnings.
        #
        # The predicate asked here is the one both R^2 sites in regression.py
        # were moved onto the same day. Handing the target in as its OWN
        # prediction makes it a perfect fit, so the value cannot fail the
        # magnitude band and the only thing that can come back is the reason the
        # target has no denominator: constant to within the resolution of its own
        # magnitude, or a sum of squares that underflowed to zero.
        #
        # It does NOT catch the fixture above, whose relative spread of 7.4e-11
        # clears the 1e-12 line, and that is deliberate rather than a gap: a
        # target of 100.1 +/- 7e-09 recorded exactly is a real target with a real
        # R^2, and nothing in y alone distinguishes it from one whose spread is
        # inherited rounding. Tightening this line to sqrt(eps) would refuse the
        # honest one, which is the trade regression.py measured and rejected. The
        # fixture is caught one step down instead, by the magnitude of the base
        # score it actually produces.
        #
        # WHAT REMOVING THIS GUARD COSTS, measured by sabotage 2026-09-30 rather
        # than assumed: the VALUES stay refused, because the base-score guard
        # below asks the same predicate about the same target and its spread test
        # does not depend on the predictions. What is lost is the MESSAGE. This
        # branch is the one that reports "N distinct finite target value(s) in M
        # rows" and the words "constant target" that
        # tests/test_bgl_stage2_s2g01.py reads, and it refuses BEFORE the model is
        # called n_features * n_repeats times. So it is a disclosure-quality and
        # cost guard with a value guard behind it, not the only thing standing
        # between a degenerate target and a published number.
        _probe, target_reason = _r2_or_not_measured(y_finite, y_finite)
        if y_finite.size < 2 or n_distinct < 2 or target_reason is not None:
            detail = f"; {target_reason}" if target_reason is not None else ""
            reason = (
                f"R^2 is undefined on a constant target "
                f"({n_distinct} distinct finite target value(s) in "
                f"{y.size} rows{detail}), so permutation importance was NOT "
                f"measured. Importances are NaN (could not check), not 0.0."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            return AttributionResult(
                scope="global",
                method="permutation",
                contributions=[
                    FeatureContribution(
                        feature=name,
                        importance=float("nan"),
                        direction="not_assessed",
                        signed_value=None,
                    )
                    for name in names
                ],
                notes=[reason],
            )

        rng = np.random.default_rng(random_state)
        base_pred = _scores_from_predict(self._predict, Xarr)
        base_score, base_reason = _r2_or_not_measured(y, base_pred)
        # THE BASE SCORE IS THE SUBTRAHEND OF EVERY IMPORTANCE, SO ITS GUARD SITS
        # ABOVE THE PER-FEATURE LOOP.
        # B4 tier-1 audit, 2026-09-30. Every importance below is
        # ``base_score - permuted_score``. A base score that is not a measurement
        # makes all of them not measurements, and a guard inside the loop would
        # have to repeat itself per feature and still could not say that.
        #
        # This is where the near-constant target described above is caught: its
        # own spread clears the resolution test, and then the R^2 it produces
        # against the model's predictions is -8.90e+20, i.e. the squared error is
        # that many times the target's entire variation, which is not a score of
        # anything. ``_r2_or_not_measured`` returns a NON-FINITE R^2 unchanged and
        # without a reason, deliberately, because its other callers three-state it
        # at a grading surface further down; this caller has no such surface, so
        # finiteness is checked here as well as the reason.
        if base_reason is not None or not np.isfinite(base_score):
            detail = base_reason if base_reason is not None else f"it is {base_score!r}"
            reason = (
                f"the model's R^2 on the unpermuted data is not a measurement "
                f"({detail}), and every importance is a DROP from that score, so "
                f"nothing was measured for any feature. Importances are NaN "
                f"(could not check), not 0.0."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            return _all_not_assessed(names, "global", "permutation", reason)

        contribs: List[FeatureContribution] = []
        never_permuted: List[str] = []
        # feature -> why at least one of its shuffles produced no measurement.
        score_not_measured: Dict[str, str] = {}
        for j in range(Xarr.shape[1]):
            drops = []
            for _ in range(n_repeats):
                Xp = Xarr.copy()
                rng.shuffle(Xp[:, j])
                # BGL5 A-evaluation-4: a shuffle that returns the column to its
                # original arrangement measures nothing, and its drop of exactly
                # 0.0 must not be averaged in as evidence. See
                # _column_rearranged.
                if not _column_rearranged(Xarr[:, j], Xp[:, j]):
                    continue
                permuted_score, permuted_reason = _r2_or_not_measured(
                    y, _scores_from_predict(self._predict, Xp)
                )
                # A PERMUTED SCORE THAT IS NOT A MEASUREMENT IS NOT EVIDENCE.
                # B4 tier-1 audit, 2026-09-30. The same rule as
                # _column_rearranged one line up, applied to the SCORE rather
                # than to the shuffle. A model that is sensitive to row ORDER
                # (the one thing permutation importance perturbs) can return a
                # non-finite prediction for a permuted arrangement: the drop was
                # then ``base_score - -inf`` = inf, and ``float(abs(inf))`` was
                # published as the importance with direction 'not_assessed' and
                # notes EMPTY, so the one field that exists to say why said
                # nothing. Discarded shuffles are counted and named below rather
                # than averaged in.
                if permuted_reason is not None or not np.isfinite(permuted_score):
                    score_not_measured.setdefault(
                        names[j],
                        permuted_reason
                        if permuted_reason is not None
                        else f"a permuted R^2 came back {permuted_score!r}",
                    )
                    continue
                drops.append(base_score - permuted_score)  # >0 => feature mattered
            if not drops:
                if names[j] not in score_not_measured:
                    # Only a feature whose shuffles all landed on the original
                    # arrangement belongs in that sentence. One whose scores could
                    # not be measured is disclosed by its own note below, because
                    # telling a reader the wrong reason is worse than telling them
                    # none.
                    never_permuted.append(names[j])
                contribs.append(
                    FeatureContribution(
                        feature=names[j],
                        importance=float("nan"),
                        direction=NOT_ASSESSED,
                        signed_value=None,
                    )
                )
                continue
            mean_drop = float(np.mean(drops))
            contribs.append(
                FeatureContribution(
                    feature=names[j],
                    importance=float(abs(mean_drop)),
                    direction=_direction(mean_drop),
                    signed_value=mean_drop,
                )
            )
        notes: List[str] = []
        if never_permuted:
            reason = _never_permuted_reason(
                never_permuted, len(names), int(n_repeats), int(Xarr.shape[0])
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            notes.append(reason)
        if score_not_measured:
            detail = "; ".join(f"{nm}: {why}" for nm, why in sorted(score_not_measured.items()))
            reason = (
                f"{len(score_not_measured)} of {len(names)} feature(s) had at "
                f"least one shuffle whose R^2 was not a measurement, and those "
                f"shuffles were NOT averaged in as evidence ({detail}). A feature "
                f"left with no measurable shuffle has importance NaN (could not "
                f"check), not 0.0."
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            notes.append(reason)
        return AttributionResult(
            scope="global", method="permutation", contributions=contribs, notes=notes
        )

    def _numpy_permutation(self, Xarr, names, n_repeats, random_state) -> AttributionResult:
        rng = np.random.default_rng(random_state)
        base = _scores_from_predict(self._predict, Xarr)
        contribs: List[FeatureContribution] = []
        never_permuted: List[str] = []
        for j in range(Xarr.shape[1]):
            drops = []
            for _ in range(n_repeats):
                Xp = Xarr.copy()
                rng.shuffle(Xp[:, j])
                # BGL5 A-evaluation-4: skip the shuffles that did not rearrange
                # anything. This is the DEFAULT call, so it was the exposed one:
                # at n=2 the identity arrangement turns up about half the time
                # and its 0.0 drop was averaged in as a measurement.
                if not _column_rearranged(Xarr[:, j], Xp[:, j]):
                    continue
                permuted = _scores_from_predict(self._predict, Xp)
                drops.append(float(np.mean(np.abs(permuted - base))))
            if drops:
                imp = float(np.mean(drops))
            else:
                never_permuted.append(names[j])
                imp = float("nan")
            contribs.append(
                FeatureContribution(
                    feature=names[j],
                    importance=imp,
                    # BGL-S2B (2026-09-17). Not "neutral". This path permutes a
                    # column and measures the mean ABSOLUTE score change, so it
                    # never computes a sign at all: the direction is not
                    # measured here, and "neutral" states that the feature has
                    # no directional effect. Measured at the public entry
                    # global_importance(X) (the default, label-free call), a
                    # model y = 2*x0 + 0.5*x1 reported feature_0 with
                    # importance 0.734 and direction 'neutral'.
                    direction=NOT_ASSESSED,
                    signed_value=None,
                )
            )
        notes = [
            "NumPy permutation: magnitude only (label-free). The DIRECTION of each "
            "feature's effect was not measured on this path: direction is "
            f"'{NOT_ASSESSED}' and signed_value is null for every feature. Pass y "
            "(and install scikit-learn) for a signed permutation importance."
        ]
        if never_permuted:
            reason = _never_permuted_reason(
                never_permuted, len(names), int(n_repeats), int(Xarr.shape[0])
            )
            warnings.warn(f"global_importance: {reason}", UserWarning, stacklevel=3)
            notes.insert(0, reason)
        return AttributionResult(
            scope="global",
            method="permutation_numpy",
            contributions=contribs,
            notes=notes,
        )

    # PER-INSTANCE SHAPLEY MATRIX

    def shapley_matrix(
        self,
        X: Any,
        background: Any = None,
        n_permutations: int = 25,
        random_state: int = 42,
        max_rows: Optional[int] = None,
        baseline: str = "mean",
    ) -> Dict[str, Any]:
        """Per-instance Shapley attribution matrix via Monte-Carlo permutation
        sampling (model-agnostic; no ``shap`` dependency).

        For every row it estimates each feature's SIGNED contribution relative to
        a single baseline reference (the per-column mean or median of
        ``background``). Permutation sampling preserves the SHAP completeness /
        local-accuracy axiom EXACTLY for every sampled ordering (each ordering
        telescopes to ``predict(x) - predict(baseline)``), so the per-row
        contributions sum to ``prediction - base_value`` to numerical precision
        regardless of ``n_permutations``. That exact additivity is what lets the
        fairness-disparity decomposition close to within 1e-6, and it is what the
        SP-SHAP / beeswarm views need (a full per-instance matrix, not one global
        vector).

        Returns a dict with ``values`` (n_rows x n_features signed contributions,
        as nested lists), ``base_value``, ``predictions``, ``feature_names``,
        ``n_rows``, ``n_permutations``, ``method`` and ``notes``. Cost: roughly
        ``n_permutations * n_features`` batched predict calls.

        Refused, with everything NaN and the reason in ``notes``, when
        ``n_permutations`` is below 1 (no ordering is sampled, so there is no
        average to take) or when the background cannot produce the baseline
        reference the contributions are measured against.
        """
        Xarr = _as_2d_array(X).astype(float)
        if Xarr.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if Xarr.shape[0] == 0:
            raise ValueError("X must be non-empty")
        if max_rows is not None and Xarr.shape[0] > max_rows:
            Xarr = Xarr[:max_rows]
        n_rows, n_feat = Xarr.shape
        names = _feature_names(X, n_feat, self._feature_names)

        bg = _as_2d_array(background).astype(float) if background is not None else Xarr

        # BGL5 A-evaluation-4, 2026-09-27. TWO out-of-reach inputs published a
        # contribution matrix nobody measured, and the completeness axiom in the
        # docstring above held only because both sides of it were built from the
        # same zeros.
        #
        # 1. A PERMUTATION BUDGET BELOW 1. ``range(int(-1))`` is empty, so the
        #    sampling loop never runs and ``phi /= -1.0`` leaves -0.0 in every
        #    cell. Measured with predict = X @ [2.0, 0.5, 0.1] and
        #    X = rng(7).normal(0, 1, (200, 3)):
        #        shapley_matrix(X[:2], background=X, n_permutations=-1)
        #        before -> values [[-0.0, -0.0, -0.0], [-0.0, -0.0, -0.0]],
        #                  base_value -0.5481, "predictions"
        #                  [-0.5480792970402042, -0.5480792970402042] (the base
        #                  value twice, while the model's real predictions for
        #                  those rows are 0.1244 and -2.1077), ZERO warnings
        #                  under warnings.simplefilter("error").
        #        n_permutations=0 -> all-NaN values, disclosed by nothing but
        #                  numpy's "invalid value encountered in divide".
        # 2. A REFERENCE THAT DOES NOT EXIST. With an imputing predict,
        #    shapley_matrix(X[:2], background=np.empty((0, 3)), n_permutations=5)
        #        before -> base_value 0.0 and finite values
        #                  [[0.0025, 0.1494, -0.0274], [-1.7812, -0.2273,
        #                  -0.0992]], so the all-NaN refusal that was pinned for
        #                  an empty background was the fixture model's NaN
        #                  arithmetic, not a check in this method.
        #
        # After: NaN everywhere (values, base_value and predictions alike), a
        # UserWarning, and the reason in a "notes" list on the returned dict.
        # The reference is derived only AFTER the two cheap refusals, so an empty
        # background no longer raises numpy's own "Mean of empty slice" on the way
        # to a refusal that then states the reason properly.
        n_perm = int(n_permutations)
        ref = None
        ref_missing: List[int] = []
        if n_perm >= 1 and bg.shape[0] > 0:
            ref = np.median(bg, axis=0) if baseline == "median" else np.mean(bg, axis=0)
            ref_missing = [
                int(j) for j in np.flatnonzero(~np.isfinite(np.asarray(ref, dtype=float)))
            ]
        refusal: Optional[str] = None
        if n_perm < 1:
            refusal = (
                f"a Shapley value is the AVERAGE marginal contribution over sampled "
                f"feature orderings, and n_permutations is {n_permutations}: not one "
                f"ordering was sampled, so nothing was measured. Every value is NaN "
                f"(could not check), and so are base_value and predictions. A matrix "
                f"of zeros would state that no feature moved the prediction, and its "
                f"completeness against a prediction rebuilt from the same zeros is "
                f"not evidence of anything."
            )
        elif bg.shape[0] == 0 or len(ref_missing) == n_feat:
            refusal = (
                f"every contribution here is measured RELATIVE TO one baseline "
                f"reference, and that reference does not exist AT ALL: the background "
                f"holds {bg.shape[0]} row(s) and its per-column {baseline} is not a "
                f"finite number for {len(ref_missing) or n_feat} of {n_feat} "
                f"feature(s). Every value is NaN (could not check), and so are "
                f"base_value and predictions."
            )
        if refusal is None and ref_missing:
            # A PARTIAL reference is DISCLOSED, not refused. Withdrawing the whole
            # matrix over one unusable column is an over-correction, and a test in
            # the suite caught it: tests/test_bgl_stage2_s2g01.py
            # ::test_a_constant_and_an_all_nan_feature_are_not_cleared_of_being_proxies
            # decomposes 400 rows whose 4th feature is all-NaN with a model that
            # never reads it, and the other three features' contributions (the
            # real proxy at > 0.9 among them) are genuine measurements that its
            # consumer already three-states per feature. So the columns are named
            # and the numbers stand.
            warnings.warn(
                f"shapley_matrix: the background's per-column {baseline} is not a "
                f"finite number for {len(ref_missing)} of {n_feat} feature(s) "
                f"({', '.join(names[j] for j in ref_missing)}), so those feature(s) "
                f"have no baseline to be measured against and their contribution is "
                f"whatever the model does with a missing value there, not a shift "
                f"from the population. The other {n_feat - len(ref_missing)} "
                f"feature(s) are unaffected.",
                UserWarning,
                stacklevel=2,
            )
        if refusal is not None:
            warnings.warn(f"shapley_matrix: {refusal}", UserWarning, stacklevel=2)
            nan_matrix = np.full((n_rows, n_feat), float("nan"))
            return {
                "values": nan_matrix.tolist(),
                "base_value": float("nan"),
                "predictions": [float("nan")] * n_rows,
                "feature_names": names,
                "n_rows": int(n_rows),
                "n_permutations": n_perm,
                "method": "permutation_shapley",
                "notes": [refusal],
            }

        # Reached only when neither refusal fired, i.e. n_permutations >= 1 and the
        # background has rows, which is exactly when `ref` was computed above.
        if ref is None:
            raise RuntimeError("shapley_matrix: no reference, yet no refusal was issued")
        base_value = float(_scores_from_predict(self._predict, ref.reshape(1, -1))[0])

        rng = np.random.default_rng(random_state)
        phi = np.zeros((n_rows, n_feat), dtype=float)

        for _ in range(int(n_permutations)):
            order = rng.permutation(n_feat)
            # Coalition starts at the baseline for every row, then reveals one
            # feature at a time in this ordering. The ordering is shared across
            # rows so each reveal is a single batched predict call.
            coal = np.tile(ref, (n_rows, 1))
            prev = np.full(n_rows, base_value, dtype=float)
            for j in order:
                coal[:, j] = Xarr[:, j]
                cur = _scores_from_predict(self._predict, coal)
                phi[:, j] += cur - prev
                prev = cur
        phi /= float(n_permutations)

        predictions = base_value + phi.sum(axis=1)
        return {
            "values": phi.tolist(),
            "base_value": base_value,
            "predictions": predictions.tolist(),
            "feature_names": names,
            "n_rows": int(n_rows),
            "n_permutations": int(n_permutations),
            "method": "permutation_shapley",
            # Present on every return, empty when there is nothing to disclose,
            # so a consumer can read it without asking whether the key exists.
            "notes": [],
        }

    # LOCAL

    def explain_decision(
        self,
        x_row: Any,
        background: Any,
        use_shap: bool = False,
    ) -> AttributionResult:
        """Explain a single prediction.

        Parameters
        ----------
        x_row : 1D feature vector (or single-row DataFrame) for the individual.
        background : 2D dataset used to compute the baseline (column medians)
            and, if ``use_shap``, as the SHAP background distribution.
        use_shap : if True and ``shap`` is importable, use exact SHAP values;
            otherwise use the occlusion approximation.

        A background that cannot produce a baseline is refused HERE, above the
        dispatch, so neither the occlusion path nor the shap path can answer it:
        every contribution derived from a missing baseline is NaN with
        ``direction='not_assessed'`` and ``base_value`` is NaN. See
        :func:`_baseline_columns_without_a_value` for the measured before and
        after. A background whose median happens to EQUAL the row is not that
        case: it yields a genuine 0.0 / 'neutral', which is a measurement.
        """
        bg = _as_2d_array(background)
        if bg.ndim != 2:
            raise ValueError("background must be 2D")
        row = _as_2d_array(x_row)
        row = row.reshape(1, -1) if row.ndim == 1 else row[:1]
        names = _feature_names(background, bg.shape[1], self._feature_names)

        # THE GUARD SITS ABOVE THE DISPATCH, for the same reason
        # global_importance's does: the two arms would otherwise disagree about
        # the same background and only one of them would ever be fixed.
        unusable = _baseline_columns_without_a_value(bg)
        if unusable:
            reason = (
                f"the baseline this explanation is measured AGAINST does not exist "
                f"for {len(unusable)} of {len(names)} feature(s) "
                f"({', '.join(names[j] for j in unusable)}): the background holds "
                f"{bg.shape[0]} row(s) and its per-column median is not a finite "
                f"number there. A contribution is the shift from that baseline, so "
                f"it was NOT measured for those feature(s): they are NaN (could not "
                f"check) with direction '{NOT_ASSESSED}', never 0.0 or 'neutral', "
                f"and base_value is NaN."
            )
            warnings.warn(f"explain_decision: {reason}", UserWarning, stacklevel=2)
            if len(unusable) == len(names):
                # Nothing to dispatch to: no column has a baseline. The
                # individual's own prediction does not depend on the background,
                # so it is still reported rather than thrown away with the rest.
                return _all_not_assessed(
                    names,
                    "local",
                    "shap" if use_shap else "occlusion",
                    reason,
                    base_value=float("nan"),
                    prediction=float(_scores_from_predict(self._predict, row)[0]),
                )

        if use_shap:
            try:
                res = self._shap_local(row, bg, names)
            except Exception as exc:
                res = self._occlusion_local(row, bg, names)
                res.notes.append(f"shap unavailable ({type(exc).__name__}); used occlusion.")
        else:
            res = self._occlusion_local(row, bg, names)

        if unusable:
            # A PARTIAL refusal: the columns that do have a baseline keep their
            # measured contribution, and only the ones that do not are withdrawn.
            # base_value is the prediction AT the baseline row, and that row does
            # not exist, so it goes with them.
            res.base_value = float("nan")
            res.notes.append(reason)
            for j in unusable:
                res.contributions[j] = FeatureContribution(
                    feature=names[j],
                    importance=float("nan"),
                    direction=NOT_ASSESSED,
                    signed_value=None,
                )
        return res

    def _occlusion_local(self, row, bg, names) -> AttributionResult:
        baseline = np.median(bg, axis=0)
        pred = float(_scores_from_predict(self._predict, row)[0])
        base_pred = float(_scores_from_predict(self._predict, baseline.reshape(1, -1))[0])

        contribs: List[FeatureContribution] = []
        for j in range(row.shape[1]):
            occluded = row.copy()
            occluded[0, j] = baseline[j]
            occ_pred = float(_scores_from_predict(self._predict, occluded)[0])
            # Contribution = how much restoring this feature's actual value
            # (vs baseline) moves the prediction. Positive => pushed prediction UP.
            contribution = pred - occ_pred
            contribs.append(
                FeatureContribution(
                    feature=names[j],
                    importance=float(abs(contribution)),
                    direction=_direction(contribution),
                    signed_value=float(contribution),
                )
            )
        return AttributionResult(
            scope="local",
            method="occlusion",
            contributions=contribs,
            base_value=base_pred,
            prediction=pred,
        )

    def _shap_local(self, row, bg, names) -> AttributionResult:
        import shap  # optional

        explainer = shap.KernelExplainer(
            lambda X: _scores_from_predict(self._predict, X),
            shap.sample(bg, min(50, len(bg))),
        )
        values = np.asarray(explainer.shap_values(row, silent=True)).reshape(-1)
        pred = float(_scores_from_predict(self._predict, row)[0])
        contribs = [
            FeatureContribution(
                feature=names[j],
                importance=float(abs(values[j])),
                direction=_direction(values[j]),
                signed_value=float(values[j]),
            )
            for j in range(len(names))
        ]
        return AttributionResult(
            scope="local",
            method="shap",
            contributions=contribs,
            base_value=float(explainer.expected_value),
            prediction=pred,
        )
