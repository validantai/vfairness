"""
Anomaly / drift attribution via DoWhy GCM.

Given a fitted SCM and a baseline + current data window, rank upstream nodes by
their share of the explained shift in the outcome's distribution. Used by the
monitoring module to answer "which input caused the base-rate-gap shift?".

The attributed quantity is the change in the outcome's MARGINAL DISTRIBUTION
(DoWhy's Shapley decomposition of a divergence between the two windows), not the
change in its mean; see :func:`_divergence`.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..._triage import is_measured

#: Significance level for "did the outcome's distribution change at all". The
#: shares below apportion that change; if it cannot be distinguished from no
#: change, there is nothing to apportion and normalising the residual noise to
#: 1.0 invents the ranking. See :func:`_change_test`.
CHANGE_ALPHA = 0.05

#: Rows each window needs before a two-window comparison is attempted. Same
#: floor the sibling mediation op applies (``len(frame) < 5``), reused rather
#: than invented, and it is also DoWhy GCM's own ``n_splits=5``.
MIN_ROWS = 5

#: A numeric column with at most this many distinct integral values is treated
#: Share of a window's outcome values that may be unreadable before the window
#: is refused rather than compared on the rest. The same ceiling the sibling
#: mediation op applies to the column whose coefficient it reports
#: (``mediate.MAX_UNREADABLE_SHARE``), reused rather than invented, so partial
#: absence cannot pass here while total absence is refused.
MAX_UNREADABLE_SHARE = 0.20

#: A numeric column with at most this many distinct integral values is treated
#: as CATEGORICAL by :func:`_divergence`. See the measurement in its docstring:
#: DoWhy's ``auto_estimate_kl_divergence`` sends such a column down its
#: continuous k-NN path, where every neighbour distance is a tie and the answer
#: is 0 whatever the data says.
MAX_DISCRETE_LEVELS = 20


@dataclass
class NodeContribution:
    node: str
    share: float
    #: The node's UNNORMALISED contribution, in the units of the divergence that
    #: ``outcome_change`` reports. ``share`` alone cannot say whether the change
    #: being apportioned was large or vanishing, because it is normalised by the
    #: contributions' own total: see :func:`attribute_distribution_change`.
    contribution: Optional[float] = None


@dataclass
class AttributionResult:
    outcome: str
    contributions: List[NodeContribution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: The measured change in the outcome's marginal distribution between the
    #: two windows, in the divergence's own units. ``None`` when it could not be
    #: measured.
    outcome_change: Optional[float] = None
    #: p-value of the two-sample test for "the outcome's distribution changed",
    #: and which test produced it. ``None`` when the test could not run.
    outcome_change_p_value: Optional[float] = None
    outcome_change_test: Optional[str] = None
    #: Why no share was published. EMPTY means the shares below are a
    #: measurement; non-empty with an empty ``contributions`` means NOTHING was
    #: attributed, which is not the same answer as "no node contributed".
    not_assessable: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "contributions": [asdict(c) for c in self.contributions],
            "warnings": list(self.warnings),
            "outcome_change": self.outcome_change,
            "outcome_change_p_value": self.outcome_change_p_value,
            "outcome_change_test": self.outcome_change_test,
            "not_assessable": list(self.not_assessable),
        }


def _import_gcm():
    try:
        import networkx as nx  # noqa: F401
        from dowhy import gcm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vfairness.operations.causal.attribute needs dowhy's gcm module "
            "(plus networkx). gcm requires dowhy>=0.11, which currently supports "
            "Python <=3.13 only; on Python 3.14 the available dowhy (0.8) has a gcm "
            "that is incompatible with numpy>=1.25. Run this op on a Python 3.13 "
            "worker, or use distribution-shift diagnostics that need no gcm. "
            f"(underlying error: {exc})"
        ) from exc
    return gcm


def _looks_discrete(values: Any) -> bool:
    """Is this column a set of labels rather than a quantity?

    Strings, booleans and objects are labels outright. A numeric column counts
    as labels when every finite value is integral and it carries at most
    ``MAX_DISCRETE_LEVELS`` distinct values, which is what a 0/1 decision, a
    coded group or a small ordinal scale looks like after an ordinary CSV read.
    """
    arr = np.asarray(values).ravel()
    if arr.dtype.kind in "OUSb":
        return True
    if arr.dtype.kind not in "fiu":
        return False
    if arr.dtype.kind == "f":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return False
        if not np.all(finite == np.rint(finite)):
            return False
    else:
        finite = arr
        if finite.size == 0:
            return False
    return int(np.unique(finite).size) <= MAX_DISCRETE_LEVELS


def _divergence(x: Any, y: Any) -> float:
    """Divergence between two samples of the outcome, label-aware.

    BGL-G02 (2026-09-30). DoWhy's default ``auto_estimate_kl_divergence``
    RETURNS 0 FOR AN INTEGER-CODED BINARY COLUMN, whatever the data says: the
    column is not ``is_categorical`` (int dtype), is not a probability matrix,
    so it falls through to the continuous k-NN estimator, where every
    nearest-neighbour distance between 0s and 1s is a tie and the estimate is
    exactly 0.0. Measured 2026-09-30 on a hiring DGP whose outcome base rate
    went 0.330 -> 0.657 because the upstream cause A went 0.2 -> 0.8:

        auto_estimate_kl_divergence(Y_baseline, Y_current)          -> 0
        the same two columns cast to str                            -> 0.2209
        attribute_distribution_change(...)  contributions           -> A 0.0,
                                                                       B 0.0,
                                                                       Y 0.0
                                            warnings                -> []

    Every node scored exactly 0.0, with nothing said, for the one question this
    module's own docstring says it exists to answer ("which input caused the
    base-rate-gap shift?"). A reader sees "no input caused the shift". With the
    dispatch below the same call answers A 0.971, B 0.028, Y 0.001.

    Both sides have to look like labels before the categorical estimator is
    used, because the samples this receives during attribution are GENERATED by
    the two fitted mechanisms, not the raw columns.
    """
    from dowhy.gcm.divergence import (
        auto_estimate_kl_divergence,
        estimate_kl_divergence_categorical,
    )

    xa = np.asarray(x)
    ya = np.asarray(y)
    if _looks_discrete(xa) and _looks_discrete(ya):
        return float(estimate_kl_divergence_categorical(xa, ya))
    return float(auto_estimate_kl_divergence(xa, ya))


def _change_test(baseline: pd.Series, current: pd.Series) -> Tuple[Optional[float], str]:
    """Did the outcome's distribution change at all? ``(p_value, test_name)``.

    A calibrated two-sample test, NOT a threshold on the divergence itself. The
    k-NN divergence estimator is far too noisy to compare against a permutation
    null of itself: measured 2026-09-30 on a genuine one-sigma mean shift at
    n=300, the estimate was 0.290 while the largest of 25 label permutations
    reached 0.352, so a permutation gate REFUSED a real drift. Kolmogorov-Smirnov
    on the same pair answers p = 2.4e-17.

    Labels get a chi-square test of the two windows' level counts; quantities get
    Kolmogorov-Smirnov. ``(None, ...)`` when neither could run, which is a
    could-not-check and never a "no change".
    """
    from scipy import stats

    b = baseline.to_numpy()
    c = current.to_numpy()
    if _looks_discrete(b) or _looks_discrete(c):
        bs = pd.Series(b).astype(str)
        cs = pd.Series(c).astype(str)
        levels = sorted(set(bs.unique()) | set(cs.unique()))
        table = np.array(
            [[int((bs == lev).sum()) for lev in levels], [int((cs == lev).sum()) for lev in levels]]
        )
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:
            return None, "single level"
        try:
            return float(stats.chi2_contingency(table).pvalue), "chi-square on level counts"
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return None, f"chi-square failed ({exc})"
    try:
        return float(stats.ks_2samp(b, c).pvalue), "two-sample Kolmogorov-Smirnov"
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return None, f"Kolmogorov-Smirnov failed ({exc})"


def _structural_reasons(
    baseline: Any,
    current: Any,
    outcome: str,
) -> List[str]:
    """By-construction impossibilities, before any model is fitted."""
    reasons: List[str] = []
    for label, frame in (("baseline", baseline), ("current", current)):
        if not isinstance(frame, pd.DataFrame):
            reasons.append(
                f"the {label} window is a {type(frame).__name__}, not a DataFrame, so no "
                f"distribution could be read from it"
            )
            continue
        if frame.empty:
            reasons.append(
                f"the {label} window holds no rows, so it carries no distribution to compare"
            )
            continue
        if outcome not in frame.columns:
            reasons.append(
                f"the outcome '{outcome}' is not a column of the {label} window "
                f"(columns: {sorted(map(str, frame.columns))[:8]}), so its change cannot "
                f"be measured, let alone attributed"
            )
            continue
        readable = int(frame[outcome].notna().sum())
        if readable < MIN_ROWS:
            reasons.append(
                f"the {label} window carries {readable} readable value(s) of '{outcome}', "
                f"below the {MIN_ROWS} this module treats as a minimum, so its "
                f"distribution is not established"
            )
            continue
        # PARTIAL absence, refused on the same ceiling the sibling mediation op
        # applies to the column whose coefficient it reports. Total absence was
        # already refused above; a window four fifths of which cannot be read is
        # not a measurement of that window either, and whether the unreadable
        # cells differ from the readable ones cannot be established from the
        # readable ones.
        share_unreadable = 1.0 - (readable / len(frame))
        if share_unreadable > MAX_UNREADABLE_SHARE:
            reasons.append(
                f"{len(frame) - readable} of {len(frame)} value(s) of '{outcome}' in the "
                f"{label} window cannot be read ({share_unreadable:.1%}), above the "
                f"{MAX_UNREADABLE_SHARE:.0%} ceiling this module family applies. The change "
                f"is NOT measured on the remainder: whether the unreadable rows differ from "
                f"the readable ones cannot be established from the readable ones"
            )
    return reasons


def attribute_distribution_change(
    gml: str,
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    outcome: str,
) -> AttributionResult:
    """
    Returns each node's normalized share of the explained change in `outcome`'s
    marginal distribution, and the size of the change being apportioned.

    A share is only published when the change itself was measured. BGL-G02
    (2026-09-30): ``share`` is normalised by the contributions' OWN total
    (``sum(abs(v)) or 1.0``), so the shares sum to 1 in absolute value for any
    input whatsoever, including one where nothing moved. Measured on the
    IDENTICAL frame passed as both windows, i.e. a drift of exactly zero::

        contributions: A +0.500, B -0.279, Y -0.221
        warnings:      []

    "Node A explains half of the shift" for a shift that does not exist, and the
    normalisation is what hides it: the denominator is the noise. Two fresh draws
    from one unchanged DGP gave B -0.500, A +0.250, Y +0.250, also silently.
    ``not_assessable`` now carries the reason, ``contributions`` is empty, and
    ``outcome_change`` / ``outcome_change_p_value`` publish what the change
    actually measured so a reader can check the ranking is worth reading.
    """
    gcm = _import_gcm()
    import networkx as nx

    out = AttributionResult(outcome=outcome)

    # ABOVE the fit, not inside it: every branch below shares the precondition
    # "there are two windows and an outcome column in both", and fitting an SCM
    # costs tens of seconds before failing on it.
    reasons = _structural_reasons(baseline, current, outcome)
    if reasons:
        out.not_assessable = reasons
        out.warnings.extend(reasons)
        _warnings.warn(
            "attribute_distribution_change could not attribute anything: "
            + "; ".join(reasons)
            + ". An empty contribution list here means nothing was attributed, not that "
            "no node contributed to the change.",
            UserWarning,
            stacklevel=2,
        )
        return out

    p_value, test_name = _change_test(baseline[outcome].dropna(), current[outcome].dropna())
    out.outcome_change_p_value = p_value
    out.outcome_change_test = test_name
    try:
        change = _divergence(
            baseline[outcome].dropna().to_numpy(), current[outcome].dropna().to_numpy()
        )
        out.outcome_change = change if is_measured(change) else None
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        out.warnings.append(f"Could not measure the size of the outcome's change: {exc}")

    if p_value is None or p_value >= CHANGE_ALPHA:
        detail = (
            f"p = {p_value:.4g} by {test_name}, at or above the {CHANGE_ALPHA} level"
            if p_value is not None
            else f"the test could not run ({test_name})"
        )
        reason = (
            f"the outcome '{outcome}' shows no change between the two windows that can be "
            f"distinguished from no change ({detail}), so there is nothing to attribute. No "
            f"share is reported: every share is normalised by the contributions' own total, "
            f"so apportioning an unmeasurable change would rank the residual noise and print "
            f"it as percentages summing to 1."
        )
        out.not_assessable = [reason]
        out.warnings.append(reason)
        _warnings.warn(
            "attribute_distribution_change did not attribute anything: " + reason,
            UserWarning,
            stacklevel=2,
        )
        return out

    try:
        graph = nx.parse_gml(gml)
        scm = gcm.StructuralCausalModel(graph)
        gcm.auto.assign_causal_mechanisms(scm, baseline)
        gcm.fit(scm, baseline)

        contributions = gcm.distribution_change(
            scm,
            baseline,
            current,
            outcome,
            difference_estimation_func=_divergence,
        )
    except Exception as exc:
        reason = f"Attribution raised: {exc}"
        out.warnings.append(reason)
        out.not_assessable = [reason]
        return out

    unreadable = [str(node) for node, value in contributions.items() if not is_measured(value)]
    readable = {
        str(node): float(value) for node, value in contributions.items() if is_measured(value)
    }
    if unreadable:
        out.warnings.append(
            "No contribution could be read for "
            + ", ".join(sorted(unreadable))
            + "; those node(s) are left out of the shares and of their denominator."
        )

    total = sum(abs(v) for v in readable.values())
    if not readable or not is_measured(total) or total <= 0.0:
        reason = (
            f"the outcome '{outcome}' DID change (p = {p_value:.4g} by {test_name}), but the "
            f"attribution returned no usable contribution for any node, so the change could "
            f"not be apportioned. This is a could-not-check: it is NOT a finding that no node "
            f"contributed."
        )
        out.not_assessable = [reason]
        out.warnings.append(reason)
        _warnings.warn(
            "attribute_distribution_change did not attribute anything: " + reason,
            UserWarning,
            stacklevel=2,
        )
        return out

    for node, value in readable.items():
        out.contributions.append(
            NodeContribution(node=node, share=value / total, contribution=value)
        )
    out.contributions.sort(key=lambda c: abs(c.share), reverse=True)

    return out
