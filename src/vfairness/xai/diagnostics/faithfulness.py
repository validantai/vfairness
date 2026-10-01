"""
Faithfulness diagnostics. References: Yeh et al. (2019) NeurIPS;
Hooker, Erhan, Kindermans & Kim (2019) NeurIPS (ROAR / removal curve).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vfairness._not_assessed import warn_not_assessed

# np.trapezoid arrived in numpy 2.0 (where np.trapz was removed); the
# declared floor is numpy>=1.21, so resolve whichever name exists. The
# import-time guard below guarantees this is a callable, never None, at
# call sites -- annotate it as such so mypy does not treat it as None.
_trapezoid_or_none: Callable[..., Any] | None = getattr(np, "trapezoid", None) or getattr(
    np, "trapz", None
)
if _trapezoid_or_none is None:  # pragma: no cover -- one of the two always exists
    raise ImportError("numpy provides neither trapezoid nor trapz")
# Bind the narrowed, guaranteed-non-None callable so call sites in the
# functions below see a plain callable rather than an Optional.
_trapezoid: Callable[..., Any] = _trapezoid_or_none


def removal_curve_auc(
    *,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    attributions: np.ndarray,
    background_mean: np.ndarray,
    n_steps: int | None = None,
) -> float:
    """ROAR-style removal curve AUC.

    Iteratively mask the top-k features (replace with the background
    mean) and measure how fast the prediction degrades. A faithful
    explanation degrades the prediction quickly; AUC is normalised to
    [0, 1] where higher is better.

    READINESS-6, 2026-09-10. This is the SECOND implementation of
    ``removal_curve_auc``. The other lives in
    ``evaluation.vfairness_metrics.explanation_diagnostics``, both are exported,
    and only that one received the READINESS-6 fixes. Measured on identical
    inputs, a model with large offsetting weights so the base prediction is
    small and the removal drops are large:

    ==========================  ========  ==========
    weights                     canonical this twin
    ==========================  ========  ==========
    ``[10, -9.9, 0]``           1.0000    33.83
    ``[100, -99.99, 0]``        1.0000    3333.83
    ``[1000, -999.999, 0]``     1.0000    333333.83
    ==========================  ========  ==========

    The docstring above promised "[0, 1]" and this function returned an
    unbounded ratio that GROWS as the base prediction shrinks, which is the
    fabricated-magnitude shape: the number looks more emphatic the less
    prediction there was to explain. It also lacked the non-finite guard, so an
    empty background could hand a caller a score for a curve nobody computed.

    Delegated rather than patched, for the reason the six disagreeing
    ``is_measured`` implementations exist: two copies of a formula answer the
    same question differently the moment one of them is fixed.
    """
    from vfairness.evaluation.vfairness_metrics.explanation_diagnostics import (
        removal_curve_auc as _canonical,
    )

    return _canonical(predict_fn, x, attributions, background_mean, n_steps)


def local_r_squared(
    *,
    surrogate_predictions: np.ndarray,
    true_predictions: np.ndarray,
) -> float:
    """Local R^2 for surrogate-fit explainers (LIME).

    Returns the coefficient of determination of the surrogate against
    the true model output on the local sample. The LIME adapter records
    this on every returned Explanation.fidelity.

    Three states: a coefficient, or ``nan`` with a warning when the local
    sample cannot carry one (fewer than two observations, or a non-finite
    prediction on either side).
    """
    sur = np.asarray(surrogate_predictions, dtype=float)
    true = np.asarray(true_predictions, dtype=float)
    n = int(true.size)
    n_finite = int(np.count_nonzero(np.isfinite(sur) & np.isfinite(true)))
    # BGL3 xai-1, 2026-09-27. This guard sits AHEAD of the sums because with
    # fewer than two observations the total sum of squares is 0.0 for the
    # arithmetic's own reason, not a degenerate-target one, and the branch below
    # then read a zero residual as PERFECT local fidelity. Measured before the
    # guard, beside sklearn.metrics.r2_score on the identical input:
    #
    #     n=0 (both arrays empty)     ours 1.0   sklearn ValueError
    #     n=1, surrogate == true      ours 1.0   sklearn nan + UndefinedMetricWarning
    #     n=1, surrogate != true      ours 0.0   sklearn nan + UndefinedMetricWarning
    #     n=3, every true NaN         ours nan, silently
    #     n=5, healthy                ours 0.992 sklearn 0.992  (unchanged)
    #
    # 1.0 is the top of this scale and it is recorded on
    # ``Explanation.fidelity``, so a surrogate fitted on a single point, or on
    # none at all, was published as explaining the model perfectly. R^2 is
    # variance EXPLAINED and one observation has no variance to explain. The
    # n>=2 constant-target case below is NOT this case and is unchanged: it has
    # variance to explain, there simply is none, which sklearn also scores.
    if n < 2:
        warn_not_assessed(
            "local_r_squared",
            measured=n_finite,
            total=n,
            unit="paired local observation(s) were usable",
            requirement="R^2 needs at least 2, because one point has no variance to explain",
            reporting="nan (could not check)",
            instead_of="1.0 for a zero-residual fit, the top of the fidelity scale",
        )
        return float("nan")
    if n_finite < n:
        warn_not_assessed(
            "local_r_squared",
            measured=n_finite,
            total=n,
            unit="paired local prediction(s) were finite",
            requirement="R^2 is undefined once a residual is not a number",
            reporting="nan (could not check)",
            instead_of="a coefficient computed from a NaN sum of squares",
        )
        return float("nan")
    ss_res = float(np.sum((true - sur) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot == 0:
        # Degenerate: the true output is constant on the local sample, so R^2
        # is undefined. Match sklearn.metrics.r2_score -- a zero-residual
        # (perfect) surrogate scores 1.0, otherwise 0.0. An arbitrary surrogate
        # over a constant target is NOT perfect fidelity and must not report 1.0.
        return 1.0 if ss_res <= 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)
