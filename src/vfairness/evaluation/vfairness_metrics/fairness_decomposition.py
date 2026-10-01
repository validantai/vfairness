"""Additive decomposition of a group-fairness disparity (Lundberg-style).

Attributes a demographic-parity-style group disparity in the model's mean score
to individual input features, and does so EXACTLY. Because the per-instance
Shapley contributions sum to ``prediction - base_value`` (the completeness
axiom), the group-mean difference of those contributions sums to the group-mean
prediction difference, which is the disparity. The base value cancels, so

    sum_j per_feature[j]  ==  total_disparity      (to within ~1e-6)

This makes the fairness disparity and the feature attributions ONE object: it
shows WHICH features carry the disparity, and, cross-referenced with each
feature's correlation to the protected attribute, which carry it via a proxy.

Model-agnostic (any ``predict(X) -> scores`` callable). No ``shap`` / ``dowhy``
dependency: it uses the permutation-Shapley matrix from
``FeatureAttributionExplainer`` (which is exact-additive by construction).

Not imported from the package __init__ (keeps ``import vfairness`` light):

    from vfairness.evaluation.vfairness_metrics.fairness_decomposition import (
        fairness_decomposition,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .attribution import FeatureAttributionExplainer, _as_2d_array, _feature_names


@dataclass
class FairnessDecompositionResult:
    metric: str  # e.g. 'demographic_parity'
    protected_attribute: str
    group_advantaged: str  # the higher-mean-prediction group
    group_disadvantaged: str
    total_disparity: float  # mean_adv(pred) - mean_dis(pred)
    base_value: float
    per_feature: Dict[str, float]  # signed contribution of each feature to the disparity
    # |corr(feature, protected indicator)|, 0..1, or None when that correlation
    # could NOT be measured (too few rows, a constant column, non-finite values).
    # None is a third state: such a feature is neither flagged as a proxy nor
    # cleared of being one, and `notes` names it with the reason. Until
    # 2026-09-16 the unmeasurable cases returned 0.0, "definitely not a proxy".
    proxy_scores: Dict[str, Optional[float]]
    flagged_proxies: List[
        str
    ]  # features that carry disparity AND correlate with the protected attr
    residual: float  # total_disparity - sum(per_feature); ~0 by construction
    n: int
    severity: str
    interpretation: str
    notes: List[str] = field(default_factory=list)
    # READINESS-6, 2026-09-10. The rows this verdict was actually measured on.
    # `n` alone cannot say whether it is the whole dataset or a budgeted sample,
    # and the answer changes the verdict; see `fairness_decomposition`.
    n_rows_supplied: int = 0
    subsampled: bool = False
    # Rows behind the compared pair. A 'critical' verdict off 1 row per group is
    # not the same evidence as one off 500, and `n` alone cannot say so.
    n_advantaged: int = 0
    n_disadvantaged: int = 0
    # Whether the additivity identity in `notes` was CHECKED, and not merely
    # asserted. None is could-not-check (a non-finite residual), never a pass.
    additivity_verified: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grade(total_disparity: float) -> str:
    mag = abs(total_disparity)
    # NaN COMPARES FALSE AGAINST EVERY BOUND, so an unmeasurable disparity fell
    # through all four bands to "info", the band that means "no material
    # disparity". Measured 2026-09-10 with a model whose predictions are all
    # NaN: severity 'info' and the interpretation "No material demographic
    # parity disparity". Not measured is not the all-clear.
    if not np.isfinite(mag):
        return "not_assessable"
    if mag >= 0.20:
        return "critical"
    if mag >= 0.10:
        return "high"
    if mag >= 0.05:
        return "medium"
    if mag > 1e-9:
        return "low"
    return "info"


# Minimum rows in the compared pair before a |correlation| means anything.
# A Pearson correlation through 2 points is +-1 BY CONSTRUCTION (a line through
# two points is exact) and has no degrees of freedom until n = 3; 8 is the
# stated floor below which this module will not publish a proxy score.
_MIN_PROXY_CORR_ROWS = 8


def _abs_corr(feature_col: "np.ndarray", indicator: "np.ndarray") -> Optional[float]:
    """|Pearson correlation| between a feature and the protected indicator.

    Point-biserial when the feature is continuous and the indicator binary; a
    plain correlation otherwise.

    Returns ``None`` -- COULD NOT CHECK -- when the correlation is not
    measurable. BGL-S2 (2026-09-16), two fabrications lived here at once and
    they ran in opposite directions:

      * NO SAMPLE-SIZE FLOOR. At n = 2 every feature scores exactly 1.0, an
        arithmetic artefact, and 1.0 clears the 0.30 proxy threshold. Executed
        on 2 rows: proxy_scores {'f0': 1.0, 'f1': 1.0, 'f2': 1.0}, f0 entered
        flagged_proxies and was written into prose a human reads as "Proxy
        features carrying it: f0" -- a named accusation against a feature whose
        correlation with the protected attribute was never measurable.
      * THE 0.0 FALLBACKS. A constant feature and an all-NaN feature both
        returned 0.0 on perfectly healthy 400-row data, i.e. "measured, and
        definitely not a proxy", for a correlation whose denominator is zero.

    None keeps both out of ``flagged_proxies`` (as 0.0 did) while saying which
    features were not checked, so "no proxy found" cannot be read off a feature
    nobody could look at. Variation is tested with np.unique on the RAW column,
    never a std/variance comparison against zero.
    """
    f = np.asarray(feature_col, dtype=float)
    g = np.asarray(indicator, dtype=float)
    if f.size < _MIN_PROXY_CORR_ROWS:
        return None
    if not (np.isfinite(f).all() and np.isfinite(g).all()):
        return None
    if np.unique(f).size < 2 or np.unique(g).size < 2:
        return None
    c = float(np.corrcoef(f, g)[0, 1])
    if not np.isfinite(c):
        return None
    return abs(c)


def _abs_corr_reason(feature_col: "np.ndarray", indicator: "np.ndarray") -> str:
    """Why ``_abs_corr`` could not measure this column. Reported, not guessed."""
    f = np.asarray(feature_col, dtype=float)
    if f.size < _MIN_PROXY_CORR_ROWS:
        return f"only {f.size} rows in the compared pair, below the floor of {_MIN_PROXY_CORR_ROWS}"
    if not np.isfinite(f).all():
        return f"{int((~np.isfinite(f)).sum())} of {f.size} values are not finite"
    if np.unique(f).size < 2:
        return "the column has a single distinct value, so a correlation is undefined"
    return "the correlation did not evaluate to a finite number"


def fairness_decomposition(
    predict,
    X: Any,
    protected: Sequence,
    feature_names: Optional[Sequence[str]] = None,
    metric: str = "demographic_parity",
    advantaged_group: Optional[Any] = None,
    n_permutations: int = 25,
    max_rows: int = 200,
    random_state: int = 42,
    proxy_corr_threshold: float = 0.30,
    proxy_contribution_share: float = 0.10,
    baseline: str = "mean",
) -> FairnessDecompositionResult:
    """Decompose a group-fairness disparity into per-feature contributions.

    Parameters
    ----------
    predict : callable ``predict(X_2d) -> scores`` (probabilities preferred).
    X : 2D feature matrix (DataFrame or array).
    protected : per-row protected-attribute values (aligned with X rows).
    metric : the disparity being decomposed. ``'demographic_parity'`` (default)
        decomposes the difference in mean predicted score between groups.
    advantaged_group : which protected value is the higher-scoring reference. If
        omitted, the two groups with the most extreme mean predictions are used
        (advantaged = higher mean), so the reported disparity is non-negative.
    proxy_corr_threshold / proxy_contribution_share : a feature is flagged as a
        proxy when it both correlates with the protected attribute (|corr| >=
        threshold) and carries a meaningful share of the disparity.
    max_rows : attribution budget. When ``X`` has more rows than this, the
        decomposition is measured on a SEEDED STRATIFIED SAMPLE of ``max_rows``
        rows (proportional within each protected value, at least one row per
        value), a warning is raised, and the result records it in ``notes``,
        ``n_rows_supplied`` and ``subsampled``. It was a head slice until
        2026-09-10; see the comment at the sampling site.

    Raises
    ------
    ValueError
        When no two groups can be ranked, including when a group's mean
        prediction is not a finite number. Refusing is deliberate: there is no
        honest advantaged/disadvantaged pair to report, and reporting one would
        be a fabricated verdict.

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

    Ledger row: fairness_decomposition. See docs/BETA_GO_LIVE_PLAN.md for the batch
    definitions.
    (end Beta Go-Live proof status)
    """
    Xarr = _as_2d_array(X).astype(float)
    if Xarr.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    prot = np.asarray(list(protected))
    if prot.shape[0] != Xarr.shape[0]:
        raise ValueError(f"protected length ({prot.shape[0]}) must match X rows ({Xarr.shape[0]}).")
    names = list(_feature_names(X, Xarr.shape[1], feature_names))

    # Subsample BEFORE attribution so X and protected stay aligned.
    #
    # READINESS-6, 2026-09-10. THIS WAS `Xarr[:max_rows]`, A HEAD SLICE, and a
    # head slice is not a sample. Row order carries meaning in real data (sorted
    # by date, by branch, by score, by the order an export ran), so the first 200
    # rows are a population, not a draw from one. Measured this day on 2000 rows
    # whose first 200 happened to be disparity-free: the verdict read
    # severity='low', disparity +0.0002, flagged_proxies=[], where the data the
    # caller passed gives severity='critical', disparity +0.7200,
    # flagged_proxies=['proxy']. 1800 of 2000 rows were discarded with no
    # warning, no note, and no field recording that anything had been dropped.
    #
    # Fixed as a SEEDED STRATIFIED sample: proportional within each protected
    # value, at least one row per value, so the budget cannot silently delete a
    # group and turn a disparity into a missing comparison. Deterministic for a
    # given ``random_state``, and the truncation is now disclosed three ways: a
    # warning, a note on the result, and the ``n_rows_supplied`` /
    # ``subsampled`` fields.
    n_supplied = int(Xarr.shape[0])
    subsampled = n_supplied > max_rows
    if subsampled:
        sample_rng = np.random.default_rng(random_state)
        by_value: Dict[Any, List[int]] = {}
        for i, v in enumerate(prot.tolist()):
            by_value.setdefault(v, []).append(i)
        picked: List[np.ndarray] = []
        for idx_list in by_value.values():
            idx_arr = np.asarray(idx_list)
            take = max(1, int(np.floor(max_rows * idx_arr.size / n_supplied)))
            take = min(take, idx_arr.size)
            picked.append(sample_rng.choice(idx_arr, size=take, replace=False))
        sel = np.concatenate(picked) if picked else np.arange(0)
        if sel.size < max_rows:
            # Proportional allocation floors, so top the budget back up from the
            # rows not already drawn rather than under-using it.
            rest = np.setdiff1d(np.arange(n_supplied), sel)
            extra = min(max_rows - sel.size, rest.size)
            if extra:
                sel = np.concatenate([sel, sample_rng.choice(rest, size=extra, replace=False)])
        sel = np.sort(sel)
        Xarr = Xarr[sel]
        prot = prot[sel]
        warnings.warn(
            f"fairness_decomposition measured {Xarr.shape[0]} of {n_supplied} rows: "
            f"max_rows={max_rows} caps the attribution budget. The rows are a seeded "
            f"stratified sample (random_state={random_state}), not the first "
            f"{max_rows}, but the disparity reported is a SAMPLE ESTIMATE. Raise "
            "max_rows to decompose every row.",
            UserWarning,
            stacklevel=2,
        )

    explainer = FeatureAttributionExplainer(predict, feature_names=names)
    mat = explainer.shapley_matrix(
        Xarr,
        background=Xarr,
        n_permutations=n_permutations,
        random_state=random_state,
        baseline=baseline,
    )
    values = np.asarray(mat["values"], dtype=float)  # (n_rows, n_feat)
    base_value = float(mat["base_value"])
    preds = np.asarray(mat["predictions"], dtype=float)

    # Resolve the two groups to compare.
    uniques = list(dict.fromkeys(prot.tolist()))

    # A GROUP WHOSE MEAN PREDICTION IS NOT A NUMBER CANNOT BE RANKED, and
    # `max`/`min` over NaN do not say so: they return whichever key they saw
    # first. Measured 2026-09-10 with a model returning all-NaN scores,
    # group_advantaged and group_disadvantaged both came back 'A' and the
    # interpretation read "No material demographic parity disparity between
    # groups 'A' and 'A'". Refused here rather than ranked, because there is no
    # honest advantaged/disadvantaged pair to report and no disparity was
    # measured: this is COULD NOT CHECK, not a finding of parity.
    unrankable = [
        g
        for g in uniques
        if not np.isfinite(float(np.mean(preds[prot == g])) if np.any(prot == g) else np.nan)
    ]
    if unrankable:
        raise ValueError(
            "mean prediction is not a finite number for group(s) "
            f"{', '.join(repr(g) for g in unrankable)}, so no advantaged/disadvantaged "
            "pair could be resolved and no disparity was measured. This is COULD NOT "
            "CHECK, not a finding of parity: check that `predict` returns finite scores "
            "for every row."
        )

    if advantaged_group is not None:
        adv = advantaged_group
        others = [u for u in uniques if u != adv]
        if not others:
            raise ValueError("advantaged_group leaves no comparison group.")
        # Disadvantaged = the remaining group with the lowest mean prediction.
        dis = min(others, key=lambda g: float(preds[prot == g].mean()))
    else:
        if len(uniques) < 2:
            raise ValueError("protected attribute must have at least two groups.")
        group_means = {g: float(preds[prot == g].mean()) for g in uniques}
        adv = max(group_means, key=group_means.__getitem__)
        dis = min(group_means, key=group_means.__getitem__)

    mask_a = prot == adv
    mask_d = prot == dis
    if mask_a.sum() == 0 or mask_d.sum() == 0:
        raise ValueError("both groups must be present in the sample.")

    total_disparity = float(preds[mask_a].mean() - preds[mask_d].mean())

    per_feature: Dict[str, float] = {}
    for j, name in enumerate(names):
        contrib = float(values[mask_a, j].mean() - values[mask_d, j].mean())
        per_feature[name] = contrib

    residual = float(total_disparity - sum(per_feature.values()))

    # Proxy diagnosis: correlation of each feature with the advantaged-group
    # indicator, restricted to the two compared groups.
    two_group = mask_a | mask_d
    indicator = mask_a.astype(float)[two_group]
    proxy_scores: Dict[str, Optional[float]] = {}
    unmeasured_proxies: Dict[str, str] = {}
    for j, name in enumerate(names):
        score = _abs_corr(Xarr[two_group, j], indicator)
        proxy_scores[name] = score
        if score is None:
            unmeasured_proxies[name] = _abs_corr_reason(Xarr[two_group, j], indicator)

    # A feature whose correlation could NOT be measured is neither flagged nor
    # cleared: it is excluded here and named in `notes` and the interpretation
    # below, so "no proxy found" can never be read off a feature nobody looked
    # at. `None >= threshold` would also raise in Python 3, so the check is
    # explicit rather than incidental.
    flagged_proxies: List[str] = []
    if abs(total_disparity) > 1e-9:
        for name in names:
            score = proxy_scores[name]
            if score is None:
                continue
            carries = abs(per_feature[name]) >= proxy_contribution_share * abs(total_disparity)
            correlated = score >= proxy_corr_threshold
            if carries and correlated:
                flagged_proxies.append(name)

    severity = _grade(total_disparity)
    top = sorted(per_feature.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    top_str = ", ".join(f"{n} ({v:+.3f})" for n, v in top) if top else "no feature"
    if severity == "not_assessable":
        # Defence in depth: the guard above refuses an unrankable group, so this
        # is only reachable if the disparity itself is unmeasurable for some
        # other reason. It must still never read as the all-clear.
        interpretation = (
            f"COULD NOT CHECK the {metric.replace('_', ' ')} disparity between groups "
            f"'{adv}' and '{dis}': it evaluated to {total_disparity}. This is not a "
            "finding of parity."
        )
    elif severity == "info":
        interpretation = (
            f"No material {metric.replace('_', ' ')} disparity between groups "
            f"'{adv}' and '{dis}' (disparity {total_disparity:+.3f})."
        )
    else:
        # The "diffuse" sentence is an affirmative all-clear about proxies, so it
        # may only be said about features that were actually measured. Anything
        # unmeasurable is named instead of being absorbed into the clearance.
        if flagged_proxies:
            proxy_note = f" Proxy features carrying it: {', '.join(flagged_proxies)}."
        elif len(unmeasured_proxies) == len(names):
            proxy_note = (
                " NO feature's correlation with the protected attribute could be measured "
                f"({'; '.join(f'{k}: {v}' for k, v in unmeasured_proxies.items())}), so "
                "whether this disparity runs through a proxy is UNKNOWN, not ruled out."
            )
        elif unmeasured_proxies:
            proxy_note = (
                " No MEASURABLE feature both correlates with the protected attribute and"
                f" carries a large share; {len(unmeasured_proxies)} feature(s) could not be"
                f" checked ({', '.join(unmeasured_proxies)}), so a proxy there is not ruled"
                " out."
            )
        else:
            proxy_note = (
                " No single feature both correlates with the protected attribute and"
                " carries a large share, so the disparity is diffuse."
            )
        interpretation = (
            f"A {total_disparity:+.3f} {metric.replace('_', ' ')} disparity favours "
            f"group '{adv}' over '{dis}'. It decomposes additively across features; "
            f"the largest contributors are {top_str}.{proxy_note}"
        )

    # THE ADDITIVITY GUARANTEE IS NOW STATED ONLY WHERE IT WAS CHECKED.
    # The first note asserted "per-feature contributions sum to the total
    # disparity" unconditionally, and the only thing that could contradict it was
    # `abs(residual) > 1e-6`, which is False for a NaN residual. Measured
    # 2026-09-10 on an unmeasurable decomposition: residual=nan, and the result
    # carried the guarantee as its sole note with nothing having been verified.
    # Three states: verified / exceeded / could-not-check.
    notes: List[str] = []
    additivity_verified: Optional[bool] = None
    if not np.isfinite(residual):
        notes.append(
            f"COULD NOT CHECK the additive identity: the decomposition residual is "
            f"{residual}, so it is unknown whether the per-feature contributions sum "
            "to the total disparity. This is not a verified decomposition."
        )
    elif abs(residual) > 1e-6:
        additivity_verified = False
        notes.append(
            "Additive Shapley decomposition: per-feature contributions sum to the "
            "total disparity (the base value cancels across groups)."
        )
        notes.append(
            f"Decomposition residual {residual:.2e} exceeds 1e-6; increase "
            "n_permutations for a tighter identity."
        )
    else:
        additivity_verified = True
        notes.append(
            "Additive Shapley decomposition: per-feature contributions sum to the "
            f"total disparity (the base value cancels across groups); verified, "
            f"residual {residual:.2e} is within 1e-6."
        )
    if unmeasured_proxies:
        notes.append(
            "COULD NOT CHECK the proxy correlation for "
            f"{', '.join(f'{k} ({v})' for k, v in unmeasured_proxies.items())}. "
            "Their proxy_scores are None, not 0.0: they are neither flagged as proxies "
            "nor cleared of being one."
        )
    if subsampled:
        notes.append(
            f"Measured on a seeded stratified sample of {int(Xarr.shape[0])} of the "
            f"{n_supplied} rows supplied (max_rows={max_rows}); the disparity and every "
            "per-feature contribution are sample estimates, not the whole dataset."
        )
    if len(uniques) > 2:
        notes.append(
            f"Protected attribute has {len(uniques)} groups; compared the "
            f"highest-mean ('{adv}') against the lowest-mean ('{dis}')."
        )

    return FairnessDecompositionResult(
        metric=metric,
        protected_attribute=str(getattr(protected, "name", "protected")),
        group_advantaged=str(adv),
        group_disadvantaged=str(dis),
        total_disparity=total_disparity,
        base_value=base_value,
        per_feature=per_feature,
        proxy_scores=proxy_scores,
        flagged_proxies=flagged_proxies,
        residual=residual,
        n=int(Xarr.shape[0]),
        severity=severity,
        interpretation=interpretation,
        notes=notes,
        n_rows_supplied=n_supplied,
        subsampled=subsampled,
        n_advantaged=int(mask_a.sum()),
        n_disadvantaged=int(mask_d.sum()),
        additivity_verified=additivity_verified,
    )
