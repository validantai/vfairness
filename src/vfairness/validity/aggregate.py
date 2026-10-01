"""VA-50: aggregate per-record groundedness into the VG-* family summary.

Fail-closed: only records with a real measurement (``available and value is not
None``) are aggregated. If none are measured, the aggregate reports
``available=False`` and no numbers, so an unmeasured batch can never read as a
clean grade. Confidence intervals come from
``vfairness.evaluation.vfairness_metrics._statistics`` when available; if that
import or call is unavailable the CI is simply omitted (never faked).

VA-74 EMISSION ALLOWLIST: this aggregate may only emit metric ids in
``EMITTABLE_METRICS``. Fail-closed applies to NAMES, not only to numbers. A metric
whose scorer is blocked must be structurally incapable of reaching an aggregate,
because the moment a rung returns its key it would otherwise be published as a
number WITH a bootstrap CI, past every downstream gate, having never been validated.
This is not hypothetical: VG-002 shipped for days as a literal duplicate of VG-001
(the judge returns ``faithfulness = groundedness``) while the platform contract
marked it ``blocked-no-engine``, so one judge call was presented as two independent
checks. VG-003 is plumbed the same way and would repeat it exactly on the day any
rung returns ``context_precision``.

Withholding is REPORTED, never silent: anything computed but not emitted is listed
under ``withheld`` with a reason, so a rung that gains a real capability is visible
rather than swallowed. Widening the allowlist is a deliberate act that must move
together with the consuming platform's frozen validity metric contract.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional

from .._triage import is_measured
from .groundedness import (
    VG_CONTEXT_PRECISION,
    VG_FAITHFULNESS,
    VG_GROUNDEDNESS,
    VG_HALLUCINATION_RATE,
    GroundednessResult,
)

#: The ONLY metric ids ``aggregate_validity`` may emit. Mirrors the ids the
#: consuming platform's validity metric contract marks ``scorerStatus ===
#: 'available'``. KEEP THE TWO IN SYNC: widening this set without widening the contract publishes a
#: number the platform will refuse, and widening the contract without widening this
#: set silently starves a metric the platform now expects.
#:
#: Deliberately a POSITIVE list. A denylist would admit every future metric by
#: default, which is the wrong direction to fail.
EMITTABLE_METRICS = frozenset({VG_GROUNDEDNESS, VG_HALLUCINATION_RATE})

#: Why each currently-withheld id is withheld. Reported to the caller so a rung that
#: starts returning a key is noticed rather than silently ignored.
_WITHHELD_REASON = {
    VG_FAITHFULNESS: (
        "scorer blocked: the interim judge returns faithfulness identical to "
        "groundedness, so emitting it would present one measurement as two"
    ),
    VG_CONTEXT_PRECISION: (
        "scorer blocked: no rung computes context precision yet; the field is "
        "plumbed but unvalidated"
    ),
}


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _bootstrap_ci(xs: List[float]) -> Optional[List[float]]:
    if len(xs) < 2:
        return None
    try:
        import numpy as np

        from vfairness.evaluation.vfairness_metrics._statistics import bootstrap_ci

        # bootstrap_ci(data, statistic, ...) -> StatisticalResult (NOT a tuple). The
        # statistic arg is required and the result exposes .lower_bound/.upper_bound;
        # calling it tuple-style (or without a statistic) threw on every call, so the
        # CI was silently None for every validity run. Use the real contract.
        # Pin the resampling seed so a sealed CI is reproducible on re-verification
        # (a grade's confidence band should not shift run-to-run for the same inputs).
        res = bootstrap_ci(np.asarray(xs, dtype=float), np.mean, random_state=0)
        if not math.isfinite(res.lower_bound) or not math.isfinite(res.upper_bound):
            return None
        return [float(res.lower_bound), float(res.upper_bound)]
    except Exception:
        # Any signature drift or missing extra: omit the CI, never fabricate one.
        return None


def aggregate_validity(results: List[GroundednessResult]) -> Dict[str, Any]:
    """Summarise a batch of GroundednessResult into VG-* aggregates (fail-closed)."""
    n = len(results)
    # `is_measured`, NOT `r.value is not None`. G13, 2026-09-30: a NaN is not
    # None, an infinity is not None, and `True` is not None, so all three passed
    # this gate and were aggregated as measurements. Executed on ten records
    # carrying `available=True` with:
    #
    #   value=nan   -> available True, n_measured 10, coverage 1.0, note None, no
    #                  warning, VG-001 mean `nan`, and VG-005 mean **0.0**
    #                  against its 0.1 maximum -- a CLEAN PASS, the strongest
    #                  all-clear this metric has, over ten records that measured
    #                  nothing. `nan < 1.0` is False, so the NaN suppressed the
    #                  hallucination finding for every record in the batch;
    #   value=inf   -> VG-001 mean `inf`, which beats any higher-is-better
    #                  threshold, and the same 0.0 VG-005 pass;
    #   value=True  -> VG-001 mean **1.0**, a perfect score. This is verbatim the
    #                  case `_triage`'s own docstring records ("a rung answering
    #                  True instead of a score clamps to a perfect 1.0.
    #                  validity/groundedness.py already rejects booleans first
    #                  for exactly this reason").
    #
    # One NaN among nine real 0.95s also destroyed the mean for the other nine
    # (VG-001 `nan`) while coverage still read 1.0.
    #
    # The module docstring's rule is unchanged; what changed is the predicate
    # that enforces it, to the one this repo keeps for the purpose. Such a record
    # now counts as UNMEASURED, so it reaches the coverage shortfall below, and a
    # batch of nothing but these reaches the fail-closed branch.
    measured = [r for r in results if r.available and is_measured(r.value)]
    if not measured:
        return {
            "available": False,
            "n": n,
            "n_measured": 0,
            "note": "No measured validity results; nothing feeds the grade (fail-closed).",
        }

    grounded = [float(r.value) for r in measured]
    faith = [float(r.faithfulness) for r in measured if is_measured(r.faithfulness)]
    ctx_prec = [float(r.context_precision) for r in measured if is_measured(r.context_precision)]
    mean_grounded = _mean(grounded)

    # VG-005 = "Share of answers with unsupported or fabricated content" (the frozen
    # platform validity metric contract). This is an ANSWER-INCIDENCE
    # rate, NOT the claim-weighted 1 - mean(groundedness): a batch where every answer holds
    # a small unsupported fraction has a low claim-weighted mean (which can clear the 0.1
    # gate) yet a 100% share of answers that contain unsupported content. Publishing the
    # claim-weighted mean under this id let an all-hallucinated batch PASS. An answer counts
    # as hallucinated if its groundedness is below a perfect 1.0 (>=1 unsupported claim) or
    # it carries any unsupported span.
    # Every record in `measured` now carries a real finite value, so the `< 1.0`
    # comparison cannot be silently False for a value nobody measured.
    n_hallucinated = sum(1 for r in measured if float(r.value) < 1.0 or r.unsupported_spans)
    hallucination_share = n_hallucinated / len(measured)

    # READINESS-6, 2026-09-10. COVERAGE, stated rather than left to be derived.
    #
    # `n` and `n_measured` were both in the envelope, so a careful reader COULD
    # divide them. Nothing said it, no warning fired, and `available` is the
    # field a gate actually reads. Measured before this change: one measured
    # result out of a hundred returned available=True with VG-001 mean 0.95
    # against a 0.8 threshold, a clean pass, silently computed from 1% of the
    # batch. A mean over one answer and a mean over a hundred are different
    # claims and they were rendered identically.
    #
    # No coverage THRESHOLD is invented here: what counts as enough is a product
    # decision and this module does not get to make it. What it can do is refuse
    # to let the shortfall pass unremarked.
    coverage = len(measured) / n if n else None
    if len(measured) < n:
        # A record that CLAIMED `available=True` and then carried a NaN, an
        # infinity or a bool is a different failure from a record that said
        # `available=False`: the first is a rung reporting a score it does not
        # have, and it used to be aggregated. Counted separately so it is visible
        # rather than folded into the ordinary unmeasured tail.
        claimed = sum(1 for r in results if r.available and not is_measured(r.value))
        warnings.warn(
            f"aggregate_validity: {len(measured)} of {n} result(s) were measured "
            f"({coverage:.1%} coverage), so every VG-* aggregate below describes "
            f"only that subset. The {n - len(measured)} unmeasured result(s) are "
            f"NOT evidence of validity and are not counted as clean."
            + (
                f" {claimed} of them reported available=True while carrying no real "
                f"finite number (a NaN, an infinity or a bool), which is a scorer "
                f"reporting a score it does not have."
                if claimed
                else ""
            ),
            UserWarning,
            stacklevel=2,
        )

    out: Dict[str, Any] = {
        "available": True,
        "n": n,
        "n_measured": len(measured),
        "coverage": coverage,
        "note": (
            None
            if len(measured) == n
            else (
                f"Partial coverage: {len(measured)} of {n} result(s) were measured. "
                f"Every aggregate here describes that subset only."
            )
        ),
        VG_GROUNDEDNESS: {
            "mean": mean_grounded,
            "ci": _bootstrap_ci(grounded),
            "direction": "higher_better",
            "threshold": 0.8,
            "positive_class": "unsupported",
        },
        VG_HALLUCINATION_RATE: {
            "mean": hallucination_share,
            "direction": "lower_better",
            "threshold": 0.1,
        },
    }
    if faith:
        out[VG_FAITHFULNESS] = {
            "mean": _mean(faith),
            "ci": _bootstrap_ci(faith),
            "direction": "higher_better",
            "threshold": 0.8,
        }
    if ctx_prec:
        out[VG_CONTEXT_PRECISION] = {
            "mean": _mean(ctx_prec),
            "ci": _bootstrap_ci(ctx_prec),
            "direction": "higher_better",
            "threshold": 0.7,
        }
    return _apply_emission_allowlist(out)


def _apply_emission_allowlist(out: Dict[str, Any]) -> Dict[str, Any]:
    """Drop every VG-* key not in ``EMITTABLE_METRICS`` and report what was dropped.

    Envelope keys (``available``, ``n``, ``n_measured``, ``coverage``, ``note``)
    are never touched; only metric ids are filtered, so a caller reading the
    envelope is unaffected.
    """
    withheld = []
    for key in [k for k in out if k.startswith("VG-") and k not in EMITTABLE_METRICS]:
        del out[key]
        withheld.append(
            {
                "metric": key,
                "reason": _WITHHELD_REASON.get(key, "not in the emission allowlist"),
            }
        )
    if withheld:
        # Sorted so the envelope is stable for hashing and for run-to-run comparison.
        out["withheld"] = sorted(withheld, key=lambda w: w["metric"])
    return out
