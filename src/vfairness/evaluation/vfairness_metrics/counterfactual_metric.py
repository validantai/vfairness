"""
Counterfactual-fairness metric (CF-001), Workstream F.

A perturbation-sensitivity / individual-consistency measure: given the model's
predictions on the FACTUAL data and on a COUNTERFACTUAL version supplied by the
caller, it quantifies how much the model's output moves. It scores exactly the
counterfactual predictions it is handed.

IMPORTANT, this is NOT the full structural-causal-model counterfactual of
Kusner et al. (2017). It does not re-derive the variables downstream of the
protected attribute; it only compares the two prediction vectors it receives.
If the caller's counterfactual holds mediators and proxies fixed (as a simple
"flip the attribute and re-query" flow does), bias laundered through a proxy
will NOT be detected here and will look fair. The metric is closest in spirit
to "fairness through unawareness sensitivity". For a true SCM counterfactual
that propagates downstream effects, use the DoWhy-backed `compute_counterfactual`
op (`vfairness_causal_counterfactual`) instead.

Reported:
  * flip_rate     -- fraction of individuals whose *decision* changes (for
    classification / thresholded scores). The headline number.
  * mean_abs_diff -- mean |score_factual - score_counterfactual| (for scores).
  * max_abs_diff  -- worst individual gap.
  * severity + interpretation.

This needs no DoWhy: the caller supplies the counterfactual predictions (e.g.
via the `vfairness_counterfactual_predict` handler, which flips the attribute
and re-queries the model).

Lives outside the package __init__'s heavy imports; expose via the metrics
module. Pure numpy.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vfairness._not_assessed import NOT_ASSESSED


@dataclass
class CounterfactualFairnessResult:
    # Rows actually ASSESSED: both the factual and the counterfactual score were
    # finite, so a decision could be read on each side. NOT the length of the
    # input arrays; rows that could not be scored are excluded and named in
    # `notes`. Until 2026-09-16 this was the input length, which asserted full
    # support for a measurement taken on fewer rows (or on none at all).
    n: int
    # Fraction of ASSESSED decisions that change, or None when no row was
    # scoreable. None is a third state (could not check) and must never be read
    # as 0.0: `severity` says 'not_assessed' in that case, and `notes` says why.
    flip_rate: Optional[float]
    mean_abs_diff: float  # mean |score change| over assessed rows (NaN if none)
    max_abs_diff: float
    threshold: Optional[float]  # decision threshold used (None for label inputs)
    severity: str
    interpretation: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _grade(flip_rate: float, mean_abs_diff: float):
    # Flip rate is the primary signal; thresholds align with the platform's
    # disparity ladder (0.10 borderline, 0.20 high).
    if flip_rate >= 0.20:
        sev = "critical"
    elif flip_rate >= 0.10:
        sev = "high"
    elif flip_rate >= 0.05:
        sev = "medium"
    elif flip_rate > 0.0:
        sev = "low"
    else:
        sev = "info"
    # An infinite score has a decidable side of the threshold but no readable
    # DISTANCE from it, so mean_abs_diff can be NaN while flip_rate is measured.
    # "mean score change nan" reads as a broken number; say which it is.
    mad_text = f"{mean_abs_diff:.4f}" if np.isfinite(mean_abs_diff) else "NOT MEASURED"
    if sev == "info":
        interp = (
            "Robust to the perturbation: no decision changed when only the "
            "protected attribute was flipped in the supplied counterfactuals "
            f"(mean score change {mad_text})."
        )
    else:
        interp = (
            f"{flip_rate:.1%} of decisions change when only the protected "
            f"attribute is flipped in the supplied counterfactuals (mean score "
            f"change {mad_text}). This perturbation-sensitivity / "
            "individual-consistency signal shows the model output depends on the "
            "flipped attribute. It equals counterfactual fairness in the sense of "
            "Kusner et al. (2017) only when the counterfactuals are derived from a "
            "structural causal model (flipping the attribute and everything "
            "causally downstream of it); for a naive single-attribute flip it "
            "measures direct sensitivity, not full counterfactual fairness. "
            "Relevant to GDPR Art. 22 and EU AI Act high-risk requirements."
        )
    return sev, interp


def counterfactual_fairness(
    y_pred_factual: Sequence[float],
    y_pred_counterfactual: Sequence[float],
    threshold: Optional[float] = 0.5,
) -> CounterfactualFairnessResult:
    """Compute counterfactual fairness from factual vs counterfactual predictions.

    Parameters
    ----------
    y_pred_factual : predictions on the original data.
    y_pred_counterfactual : predictions after flipping the protected attribute
        (same individuals, same order).
    threshold : if the predictions are continuous scores, the decision
        threshold used to compute the flip rate. If the inputs are already
        hard 0/1 labels, pass ``threshold=None`` and the flip rate is computed
        on label changes directly.

    Beta Go-Live proof status (2026-10-01): BGL-B SEMI-PROVEN. Executed and judged
    honest, but nothing in the suite holds it there. The behaviour is observed, not
    protected: a refactor can reintroduce the defect with every gate staying green.

    Ledger row: counterfactual_fairness_metric. See docs/BETA_GO_LIVE_PLAN.md for
    the batch definitions.
    (end Beta Go-Live proof status)
    """
    f = np.asarray(y_pred_factual, dtype=float).ravel()
    c = np.asarray(y_pred_counterfactual, dtype=float).ravel()
    if f.shape != c.shape:
        raise ValueError(
            f"factual ({f.shape}) and counterfactual ({c.shape}) predictions "
            "must have the same shape."
        )
    if f.size == 0:
        raise ValueError("predictions must be non-empty.")

    # BGL-S2B (2026-09-17). The guard goes ABOVE the decision dispatch, because
    # both branches below share this precondition. Measured at this entry on
    # data with a genuine 0.30 flip rate, counterfactual_fairness(f, c,
    # threshold=nan) returned flip_rate 0.0, severity 'info', notes [], no
    # warning and the interpretation "Robust to the perturbation": every
    # comparison against NaN is False, so both sides became decision 0 and
    # nothing ever flipped.
    #
    # NaN only, deliberately not `not np.isfinite`. A threshold of +inf
    # ("accept nobody") or -inf ("accept everybody") does decide every row, and
    # the flip rate that follows IS a measurement of that rule; refusing it
    # would delete a real finding. It is disclosed in `notes` instead.
    if threshold is not None and np.isnan(threshold):
        raise ValueError(
            "threshold=nan cannot decide any row: every comparison against NaN is "
            "False, so both the factual and the counterfactual side collapse to the "
            "same decision and the flip rate comes back 0.0, which reads as "
            "'robust to the perturbation' rather than as a measurement that was "
            "never taken. Pass a real threshold, or threshold=None for label inputs."
        )

    # BGL-S2 (2026-09-16). A NaN score compares False against ANY threshold, so
    # an unreadable row silently became decision 0 on BOTH sides and was counted
    # as "this individual's decision did not flip". Measured at this entry:
    #   * 200 rows of pure NaN returned flip_rate=0.0, severity='info', n=200 and
    #     the interpretation "Robust to the perturbation: no decision changed",
    #     an affirmative claim of robustness over 200 rows where no decision was
    #     decidable at all.
    #   * Half NaN, with 60 of the 100 readable rows genuinely flipping (a true
    #     rate of 0.60 on scoreable data), reported flip_rate=0.30 with n=200:
    #     the unreadable rows acted as evidence of NON-flipping and halved the
    #     rate, while n still claimed full support.
    # With threshold=None (label inputs) the same rows ran the other way: NaN !=
    # NaN is True, so every unreadable row counted as a FLIP.
    # Both directions are fixed by scoring only rows readable on BOTH sides.
    #
    # BGL-S2B (2026-09-17). "Readable" for a DECISION is not-NaN, not finite.
    # `np.isfinite` also excluded +/-inf, and an infinite score sits decidably
    # on one side of a finite threshold. Measured at this entry: 100 clean rows
    # plus 100 rows whose factual 0.2 became a counterfactual +inf (a reject
    # that turns into an accept) read flip_rate 0.0, severity 'info', "Robust
    # to the perturbation", where the code before the isfinite mask read 0.5
    # and CRITICAL. Excluding those rows deleted a real finding, which is a
    # worse outcome than the fabrication the mask was added to stop.
    #
    # So there are two masks, because the two statistics need different things:
    #   decidable: the flip rate only needs a SIDE, which +/-inf has.
    #   finite:    mean/max |difference| need a DISTANCE, which +/-inf poisons.
    decidable = ~np.isnan(f) & ~np.isnan(c)
    finite = np.isfinite(f) & np.isfinite(c)
    n_assessed = int(decidable.sum())
    n_measurable = int(finite.sum())
    n_input = int(f.size)
    notes: List[str] = []

    if n_assessed < n_input:
        warnings.warn(
            f"counterfactual_fairness: {n_input - n_assessed} of {n_input} rows had a "
            f"factual or counterfactual score that was NaN and were excluded. Figures "
            f"are over the {n_assessed} assessed rows.",
            UserWarning,
            stacklevel=2,
        )
        notes.append(
            f"{n_input - n_assessed} of {n_input} rows could not be scored (a factual or "
            f"counterfactual prediction was NaN) and were EXCLUDED from "
            f"every figure below. n reports the {n_assessed} assessed rows, not the "
            f"{n_input} supplied."
        )
    if n_measurable < n_assessed:
        warnings.warn(
            f"counterfactual_fairness: {n_assessed - n_measurable} of the {n_assessed} "
            f"assessed rows carry an infinite score. They are DECIDED (an infinity is "
            f"on a definite side of the threshold) and counted in flip_rate, but they "
            f"have no readable distance, so mean_abs_diff and max_abs_diff are measured "
            f"over {n_measurable} row(s) only.",
            UserWarning,
            stacklevel=2,
        )
        notes.append(
            f"{n_assessed - n_measurable} of the {n_assessed} assessed rows carry an "
            f"infinite score: counted in flip_rate (an infinity has a definite side of "
            f"the threshold), excluded from mean_abs_diff and max_abs_diff (an infinity "
            f"has no readable distance). Those two figures cover "
            f"{n_measurable} row(s)."
        )

    if n_assessed == 0:
        notes.append(
            "No row was scoreable, so counterfactual sensitivity was NOT measured. "
            "flip_rate is None (could not check), not 0.0."
        )
        return CounterfactualFairnessResult(
            n=0,
            flip_rate=None,
            mean_abs_diff=float("nan"),
            max_abs_diff=float("nan"),
            threshold=threshold,
            severity=NOT_ASSESSED,
            interpretation=(
                "Counterfactual sensitivity could NOT be assessed: none of the "
                f"{n_input} supplied rows carried a readable score on both the factual "
                "and the counterfactual side, so no decision could be compared. This is "
                "not evidence of robustness."
            ),
            notes=notes,
        )

    if n_measurable:
        abs_diff = np.abs(f[finite] - c[finite])
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))
    else:
        # Every assessed row is infinite on one side: the decisions are still
        # readable, the distances are not. NaN, not 0.0.
        mean_abs_diff = float("nan")
        max_abs_diff = float("nan")

    fa = f[decidable]
    ca = c[decidable]
    if threshold is None:
        decisions_f = fa
        decisions_c = ca
    else:
        decisions_f = (fa >= threshold).astype(int)
        decisions_c = (ca >= threshold).astype(int)
    flip_rate = float(np.mean(decisions_f != decisions_c))

    severity, interpretation = _grade(flip_rate, mean_abs_diff)
    if threshold is not None and np.isinf(threshold):
        notes.append(
            f"threshold={threshold} puts every finite score on one side of the "
            f"decision, so a flip rate of {flip_rate:.4f} describes that degenerate "
            f"rule and not the model's own operating point."
        )
    if n_assessed < n_input:
        # The verdict a reader sees must carry its own coverage. Until now the
        # severity and this string were the unqualified _grade() text, and only
        # `notes` said the sample had been narrowed: an assessed 1 of 200 read
        # exactly like an assessed 200 of 200 (BGL-S2B, 2026-09-17).
        interpretation += (
            f" COVERAGE: this verdict rests on the {n_assessed} of {n_input} supplied "
            f"row(s) that carried a readable score on both sides. The other "
            f"{n_input - n_assessed} were not assessed, and nothing here is a "
            f"statement about them."
        )
    return CounterfactualFairnessResult(
        n=n_assessed,
        flip_rate=flip_rate,
        mean_abs_diff=mean_abs_diff,
        max_abs_diff=max_abs_diff,
        threshold=threshold,
        severity=severity,
        interpretation=interpretation,
        notes=notes,
    )
