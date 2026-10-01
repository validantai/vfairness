"""
vfairness.xai.decomposition.shap_fairness
=========================================

Lundberg fairness decomposition: turn a group-fairness metric on the
model output into a sum of per-feature contributions, by computing the
metric on each feature's SHAP-value distribution.

Reference: Lundberg, S. M. (2020). *Explaining Quantitative Measures of
Fairness.* Fair & Responsible AI Workshop @ CHI 2020. The additive
identity follows from SHAP local accuracy and the linearity of
expectation under the group conditional.

This is the spine the platform's XAI view rests on; the 1e-6 identity
is asserted before writing any row. A violation almost always means the
SHAP values were computed in a different unit (probability vs log-odds)
than the metric uses; convert before decomposing.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np

from ..schemas import FairnessDecomposition, FairnessMetric

DEFAULT_PROXY_THRESHOLD = 0.15


def lundberg_fairness_decomposition(
    *,
    shap_values: np.ndarray,
    group_labels: np.ndarray,
    feature_names: list[str],
    metric: FairnessMetric,
    protected_attribute: str,
    subject_id: str,
    audit_artifact_id: str,
    decomposition_id: str = "",
    proxy_threshold: float = DEFAULT_PROXY_THRESHOLD,
) -> FairnessDecomposition:
    """Decompose ``metric`` on ``protected_attribute`` per feature.

    Parameters
    ----------
    shap_values
        ``(n_samples, n_features)`` matrix of SHAP values in the SAME
        UNITS as the model output the metric is computed on. Mixing
        log-odds SHAP with a probability metric will break the 1e-6
        identity and the function will fail loudly.
    group_labels
        ``(n_samples,)`` integer / boolean / string labels for the
        protected attribute. Must define two classes for the metric to
        be a meaningful binary disparity.
    feature_names
        Feature names in the same order as the columns of ``shap_values``.
    metric
        One of the literals in ``vfairness.xai.schemas.FairnessMetric``.
    protected_attribute, subject_id, audit_artifact_id, decomposition_id
        Pass-through identifiers for the returned object.
    proxy_threshold
        Flagging threshold for ``proxy_scores`` (default 0.15 per
        Part 7.4 of the XAI implementation plan).

    Returns
    -------
    FairnessDecomposition
        With the 1e-6 identity already asserted.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: lundberg_decomposition. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """

    if shap_values.shape[1] != len(feature_names):
        raise ValueError("feature_names length must equal shap_values.shape[1].")
    if shap_values.shape[0] != group_labels.shape[0]:
        raise ValueError("group_labels must have one entry per row of shap_values.")

    # This decomposition is a between-group MEAN DIFFERENCE of SHAP values,
    # which is the demographic-parity disparity in model-output space. Other
    # metrics (equal opportunity, equalized odds, ...) need conditioning on
    # y_true, which this function does not receive: silently computing the
    # mean difference for them mislabels the result. Fail loudly instead.
    _supported = {"demographic_parity", "mean_difference"}
    if str(metric) not in _supported:
        raise NotImplementedError(
            f"lundberg_fairness_decomposition currently implements only "
            f"{sorted(_supported)} (a between-group mean difference of SHAP "
            f"values); got metric={metric!r}. Label-conditioned metrics need "
            f"y_true and are not yet supported."
        )

    unique_groups = list(_unique_preserve_order(group_labels))
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need >= 2 protected-attribute groups for {metric}; saw only {unique_groups!r}."
        )
    if len(unique_groups) > 2:
        # The decomposition is a BINARY disparity; silently dropping the
        # remaining groups would mislabel a multi-group attribute's result.
        warnings.warn(
            f"Protected attribute {protected_attribute!r} has "
            f"{len(unique_groups)} groups {unique_groups!r}; this binary "
            f"decomposition compares only the first two in order of "
            f"appearance ({unique_groups[0]!r} vs {unique_groups[1]!r}) and "
            f"ignores the rest. Recode the labels or run pairwise if you "
            f"need the other groups.",
            UserWarning,
            stacklevel=2,
        )

    # A SHAP column that is not finite cannot be decomposed, and the failure
    # is SILENT if it is allowed through: the group means of that column are
    # NaN, which poisons total_disparity and -- because proxy_score divides by
    # sum_k |disparity(phi_k)| -- EVERY feature's proxy score, one bad column
    # or not. `score >= proxy_threshold` is False for NaN, so flagged_proxies
    # came back [] ("no proxy features detected") while a real proxy sat in the
    # data, measured and unflagged, with the identical disparity it carries in
    # the healthy run. assert_identity() could not catch it either: the
    # residual is NaN and `NaN > tolerance` is False, so the corrupt row was
    # written. This is a could-not-check, never a clean result, so refuse
    # loudly and name the columns. The guard goes ABOVE the per-feature loop
    # and the proxy dispatch: fixing only flagged_proxies would leave the same
    # fabrication in total_disparity and in the row the worker writes.
    _finite_col = np.isfinite(np.asarray(shap_values, dtype=float)).all(axis=0)
    if not _finite_col.all():
        _n_rows = shap_values.shape[0]
        _bad = {
            feature_names[j]: int((~np.isfinite(np.asarray(shap_values[:, j], dtype=float))).sum())
            for j in np.flatnonzero(~_finite_col)
        }
        raise ValueError(
            "shap_values contain non-finite entries, so the fairness "
            "decomposition COULD NOT BE MEASURED for: "
            + ", ".join(f"{name} ({cnt} of {_n_rows} rows)" for name, cnt in _bad.items())
            + ". Refusing rather than returning a decomposition whose "
            "flagged_proxies list would read as 'no proxy features detected'. "
            "Impute or drop those rows (and re-run the explainer) first."
        )

    # We compute the binary disparity on each feature's SHAP-value
    # distribution. For demographic_parity, "disparity on feature j" is
    # the difference of the mean SHAP value of feature j between the
    # most- and least-served groups. The sum across features equals the
    # disparity of the model output (= sum of SHAP values + base value;
    # the base value cancels under the difference).
    a, b = unique_groups[0], unique_groups[1]
    mask_a = group_labels == a
    mask_b = group_labels == b

    per_feature: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = shap_values[:, j]
        per_feature[name] = float(col[mask_a].mean() - col[mask_b].mean())

    # Total disparity is the model-level metric on the sum of SHAP
    # values (= prediction minus the base value, which cancels under
    # the difference between groups).
    total_disparity = float(
        shap_values[mask_a].sum(axis=1).mean() - shap_values[mask_b].sum(axis=1).mean()
    )

    # proxy_score takes the per-feature disparities alone: it normalises by
    # sum_k |disparity(phi_k)|, not by total_disparity (R6-3, see below).
    # It now returns None for any feature whose own disparity is not finite,
    # which is could-not-check; `None >= threshold` raises rather than
    # quietly reading as "below threshold" the way `nan >= threshold` did.
    maybe_scores = proxy_score(per_feature)
    unscored = sorted(name for name, score in maybe_scores.items() if score is None)
    if unscored:
        # REACHABLE, despite the finite guard above (the comment here used to
        # claim otherwise, and that was wrong). `_finite_col` checks the SHAP
        # CELLS; this checks the between-group MEAN DIFFERENCE of a column,
        # which can overflow out of float range even when every cell in it is
        # finite. Measured: a column holding +1e308 for group A and -1e308 for
        # group B is all-finite cell by cell, and its disparity is +inf, so
        # proxy_score has no share to report for it. A silent None in
        # proxy_scores is exactly the shape this function exists to refuse,
        # and `assert` is stripped by `python -O`, so this stays a real guard.
        raise ValueError(
            "proxy_score could not be computed for: "
            + ", ".join(unscored)
            + ". Refusing rather than writing a decomposition whose flagged_proxies "
            "list would read as 'no proxy features detected'."
        )
    proxy_scores_dict: dict[str, float] = {
        name: score for name, score in maybe_scores.items() if score is not None
    }
    flagged = [name for name, score in proxy_scores_dict.items() if score >= proxy_threshold]

    decomposition = FairnessDecomposition(
        id=decomposition_id,
        subject_id=subject_id,
        metric=metric,
        protected_attribute=protected_attribute,
        total_disparity=total_disparity,
        per_feature=per_feature,
        proxy_scores=proxy_scores_dict,
        flagged_proxies=flagged,
        audit_artifact_id=audit_artifact_id,
    )
    # Loud failure on unit mismatch.
    decomposition.assert_identity()
    return decomposition


def proxy_score(per_feature: dict[str, float]) -> dict[str, float | None]:
    """Per-feature proxy score (Part 7.4 of the XAI implementation plan).

    ``proxy_score(j) = |disparity(phi_j)| / sum_k |disparity(phi_k)|``

    A score >= 0.15 (default threshold) flags the feature as a likely
    proxy or redundant encoding of the protected attribute.

    There is no ``total_disparity`` argument, and there was never a use for
    one. R6-3 (2026-09-10): it was REQUIRED and positional, and did nothing.
    Measured: ``0.55``, ``-0.55``, ``0.0``, ``nan`` and ``1e9`` all returned
    the identical dict; the name appeared exactly once in the function, in
    its own signature. Worse than dead weight, it asserted a formula this is
    not: normalising by the total would give ``|phi_j| / |sum_k phi_k|``,
    which differs from the documented denominator whenever the per-feature
    disparities carry opposite signs and cancel (and is undefined when they
    cancel exactly). It was REMOVED rather than commented as unused, so a
    caller who passes one gets a loud TypeError instead of a wrong model of
    the maths. The identity between the parts and the whole is checked where
    it belongs, in ``FairnessDecomposition.assert_identity()``.

    THREE STATES (Stage 2, s2g07, 2026-09-16). A feature whose own disparity
    is not finite has no share to report, and NaN was the worst possible way
    to say so: the caller's `score >= threshold` test is False for NaN, so an
    unmeasurable feature read as "below the proxy threshold" and a measured
    proxy sitting beside it in the same dict was silenced too, because one
    non-finite value poisons the shared denominator. Such features now come
    back as ``None``, and the denominator is summed over the FINITE values
    only, so a feature that WAS measured keeps its measured share.

    The zero denominator is NOT the same case and is deliberately left as
    0.0. ``denom`` is a sum of absolute values, so a finite ``denom == 0``
    means every per-feature disparity was measured and every one of them is
    exactly 0.0: "no feature is a proxy" is then the right answer, and
    refusing there would delete a real finding. Only when not one value is
    finite is there nothing to divide and nothing measured, and then every
    feature is ``None``.

    Beta Go-Live proof status (2026-10-01): BGL-A PROVEN. Executed on healthy input
    and on the degenerate inputs where nothing it claims to measure exists; in each
    it either refused or returned a value that was checked to be a true measurement,
    and a test in the suite names it beside a refusal assertion. NOT INDEPENDENTLY
    CHECKED: one examiner reached that judgement and nobody has yet tried to refute
    it. Roughly one grade in three has been overturned when somebody did, so treat
    this as a claim with evidence behind it rather than a settled one. This does NOT
    establish that the pin has been sabotage-checked, so it is not known whether the
    pin can fail at all. This does NOT establish that its statistics are accurate,
    nor that the pin covers every scenario.

    Ledger row: proxy_feature_score. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    finite = {name: v for name, v in per_feature.items() if math.isfinite(v)}
    unmeasured = [name for name in per_feature if name not in finite]
    if unmeasured:
        warnings.warn(
            "proxy_score: the disparity of "
            + ", ".join(sorted(unmeasured))
            + " is not finite, so its share of the total disparity COULD NOT BE "
            "MEASURED. Returning None for it, not nan and not 0.0: `nan >= threshold` "
            "is False, which reads as 'not a proxy'. The remaining features are "
            "normalised over the finite disparities only.",
            UserWarning,
            stacklevel=2,
        )
    if not finite:
        if not per_feature:
            # Nothing was ASKED about, which is not the same as nothing being
            # measurable. An empty request gets an empty answer and no warning;
            # warning here told a caller who passed {} that its disparities were
            # unmeasurable, which is a claim about data that does not exist.
            return {}
        warnings.warn(
            "proxy_score: not one per-feature disparity is finite, so no share could "
            "be computed for any feature. Every score is None (could not check), which "
            "is not a finding that no feature is a proxy.",
            UserWarning,
            stacklevel=2,
        )
        return {name: None for name in per_feature}

    # Scale by the largest absolute disparity BEFORE summing. Summing the raw
    # absolute values overflows to inf whenever the disparities are near the
    # float ceiling, and `abs(v) / inf` is 0.0 for EVERY feature: measured
    # {"f1": 1e308, "f2": 1e308, "f3": 0.5} came back {0.0, 0.0, 0.0} with no
    # warning, so two features carrying a 1e308 disparity were published as
    # "0 percent of the total disparity" and flagged_proxies read []. The share
    # is a RATIO, so dividing numerator and denominator by the same constant
    # leaves it unchanged while keeping both operands inside float range; the
    # scaled denominator lies in [1, len(finite)] and cannot overflow. This is
    # a measurement, not a refusal: the same input now returns 0.5 / 0.5 / ~0.
    _scale = max(abs(v) for v in finite.values())
    if _scale == 0:
        # Every disparity was measured and every one is exactly 0.0.
        return {name: (0.0 if name in finite else None) for name in per_feature}
    denom = sum(abs(v) / _scale for v in finite.values())
    if not math.isfinite(denom) or denom == 0:
        # Not reachable through the scaling above for any finite input, and
        # kept as a real guard rather than an assert (which `python -O` strips)
        # because a share divided by a denominator nobody could compute is
        # exactly the fabrication this function exists to refuse.
        warnings.warn(
            "proxy_score: the total disparity that each share is measured against "
            f"COULD NOT BE COMPUTED (denominator={denom!r}), so no feature has a "
            "share to report. Every score is None (could not check), which is not a "
            "finding that no feature is a proxy.",
            UserWarning,
            stacklevel=2,
        )
        return {name: None for name in per_feature}

    scores: dict[str, float | None] = {}
    _unstable: list[str] = []
    for name in per_feature:
        if name not in finite:
            scores[name] = None
            continue
        share = (abs(finite[name]) / _scale) / denom
        if not math.isfinite(share):
            # Belt and braces: a share that is not a real number is a
            # could-not-check, never a 0.0 and never a nan that reads as
            # "below the proxy threshold".
            scores[name] = None
            _unstable.append(name)
        else:
            scores[name] = share
    if _unstable:
        warnings.warn(
            "proxy_score: the share of "
            + ", ".join(sorted(_unstable))
            + " did not evaluate to a real number, so it COULD NOT BE MEASURED. "
            "Returning None for it, not nan and not 0.0.",
            UserWarning,
            stacklevel=2,
        )
    return scores


def _unique_preserve_order(arr: Iterable):
    """Like ``set(arr)`` but ordered by first appearance."""
    seen = []
    for x in arr:
        if x not in seen:
            seen.append(x)
    return seen
