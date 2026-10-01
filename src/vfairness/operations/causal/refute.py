"""
Robustness suite via DoWhy refuters.

Runs: placebo treatment, random common cause, data subset, dummy outcome.
A passed test does not prove correctness; it only rules out trivial drivers.
We expose this as a single 'robustness_score' (fraction of tests passed) with
per-test detail so the UI can render a robustness panel without exposing the
underlying library names.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pandas as pd

from ..._triage import is_measured
from .mediate import _constant_treatment_reason

#: Rows the dataset needs before any refuter is run. The same floor the sibling
#: mediation op applies (``len(frame) < 5``), reused rather than invented.
#:
#: BGL-G02 (2026-09-30). There was NO floor here. Measured on THREE rows of a
#: three-variable DGP (T, Y and one confounder, so the design has as many
#: parameters as rows and the fit is exact by construction)::
#:
#:     base_estimate 0.1563, robustness_score 0.75, warnings []
#:     Placebo treatment PASS, Random common cause PASS, Data subset PASS,
#:     Dummy outcome FAIL
#:
#: A robustness score published for a dataset that cannot support the regression
#: it is scoring, and the data-subset refuter had nothing to subset.
MIN_ROWS_FOR_REFUTATION = 5


@dataclass
class RefutationOutcome:
    test_name: str
    #: ``None`` when the refuter RAN but its result could not be read as a
    #: comparable effect. That is not a failed refutation and it does not enter
    #: the robustness_score denominator, exactly as a crashed refuter does not.
    #: READINESS-6, 2026-09-10.
    passed: Optional[bool]
    observed_effect: Optional[float] = None
    refuted_effect: Optional[float] = None
    p_value: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RefutationResult:
    treatment: str
    outcome: str
    base_estimate: Optional[float] = None
    outcomes: List[RefutationOutcome] = field(default_factory=list)
    robustness_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "base_estimate": self.base_estimate,
            "outcomes": [asdict(o) for o in self.outcomes],
            "robustness_score": self.robustness_score,
            "warnings": list(self.warnings),
        }


def _import_dowhy():
    try:
        from dowhy import CausalModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vfairness.operations.causal.refute requires 'dowhy'. "
            "Install with: pip install dowhy networkx"
        ) from exc
    return CausalModel


_REFUTERS = [
    ("Placebo treatment", "placebo_treatment_refuter"),
    ("Random common cause", "random_common_cause"),
    ("Data subset", "data_subset_refuter"),
    ("Dummy outcome", "dummy_outcome_refuter"),
]

# A refuter "passes" when the refuted estimate is close to the original
# null hypothesis for that test. Convention follows DoWhy's docs:
#   placebo / dummy outcome: refuted estimate should be near 0
#   random common cause / subset: refuted estimate should be near the original


#: The two families differ in what their null says, so they get separate
#: tolerances. Both are ``max(relative, ABSOLUTE FLOOR)``, and the floor is what
#: makes them unfailable on small effects: see :func:`_tolerance`.
_NULL_IS_ZERO = {"Placebo treatment", "Dummy outcome"}


def _tolerance(test_name: str, observed: float) -> float:
    """How far the refuted estimate may sit from its null and still pass."""
    if test_name in _NULL_IS_ZERO:
        return max(0.05, 0.1 * abs(observed) + 1e-9)
    return max(0.05 * abs(observed) + 0.05, 0.1)


def _passed(test_name: str, observed: float, refuted: float) -> Optional[bool]:
    """Did this refuter fail to refute the estimate? None when it could not tell.

    READINESS-6, 2026-09-10. THIS SUITE COULD NOT FAIL AT FAIRNESS SCALE, and it
    reported ``robustness_score`` 1.0 for saying so.

    Both tolerances carry an ABSOLUTE floor (0.05, and 0.1 for the near-original
    family). Fairness effects live at 0.01 to 0.05, so the floor routinely
    exceeded the whole effect under test, and a tolerance wider than the effect
    admits every possible outcome. Measured on this repo before the fix, with an
    observed effect of 0.03::

        placebo returns the ENTIRE original effect (0.03)  -> PASS
        data subset makes the effect VANISH (0.00)         -> PASS
        data subset REVERSES the sign (-0.03)              -> PASS

    Each of those is the most complete refutation its test can produce. At an
    observed effect of 0.30 all three correctly FAIL, which is why this was
    invisible: the suite works on large effects and silently stops working on
    exactly the small ones this library exists to measure.

    This is the degenerate-bound defect that :func:`check_threshold` already
    guards for thresholds, in a third costume: **a bound that cannot be breached
    grades nothing.** The answer is the same one, three states rather than two.
    A refuter whose tolerance is wider than the effect gives None, is disclosed,
    and is excluded from the ``robustness_score`` denominator, exactly as a
    refuter that crashed or returned an unreadable result already is. It is not
    a pass, because nothing was tested; and it is not a failure, because the
    refuter did nothing wrong.

    The comparison is ``tolerance > abs(observed)``, strictly. At equality the
    test still grades: it detects a total vanishing or a total reproduction and
    nothing weaker, which is thin but real, and calling it unmeasurable there
    would discard a genuine refutation. That is the over-correction direction and
    it is pinned in the tests.
    """
    if not is_measured(observed) or not is_measured(refuted):
        return None

    tolerance = _tolerance(test_name, observed)

    # The most extreme outcome each family can produce sits |observed| away from
    # its null: the placebo reproducing the whole effect, or the perturbation
    # destroying it. If the tolerance admits even that, no outcome can fail.
    if tolerance > abs(observed):
        return None

    if test_name in _NULL_IS_ZERO:
        return abs(refuted) < tolerance
    return abs(refuted - observed) < tolerance


def run_refutation_suite(
    gml: str,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> RefutationResult:
    CausalModel = _import_dowhy()
    result = RefutationResult(treatment=treatment, outcome=outcome)

    if data is None or data.empty:
        result.warnings.append("Refutation requires a non-empty dataset.")
        return result

    # ABOVE the base estimate and therefore above all four refuters: they share
    # the precondition "the estimate being refuted is a measurement". A refuter
    # can only ask whether the estimate survives a perturbation, so an estimate
    # that was never identified cannot be rescued by four of them passing.
    if len(data) < MIN_ROWS_FOR_REFUTATION:
        result.warnings.append(
            f"Refused: {len(data)} row(s) is below the {MIN_ROWS_FOR_REFUTATION} this "
            f"module treats as a minimum, so neither the estimate nor its robustness was "
            f"established. The subset refuter has nothing to subset and the regression has "
            f"no residual degrees of freedom to speak of. robustness_score is unavailable."
        )
        return result

    # The PREDICATE is the sibling mediation op's, so the two ops cannot drift
    # apart on what "no contrast" means; the sentence is this op's own, because
    # a refutation reader has no proportion_mediated to read about.
    if _constant_treatment_reason(data, treatment, "treatment") is not None:
        result.warnings.append(
            f"Refused: the treatment '{treatment}' takes a single value in every row, so the "
            f"data holds no contrast and the fitted coefficient is the minimum-norm solution "
            f"of a rank-deficient design rather than an effect. No base estimate is reported, "
            f"and nothing was refuted: every refuter compares a perturbed estimate against "
            f"this one. robustness_score is unavailable."
        )
        return result

    try:
        model = CausalModel(data=data, treatment=treatment, outcome=outcome, graph=gml)
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        base = model.estimate_effect(
            estimand,
            method_name="backdoor.linear_regression",
        )
        observed = float(getattr(base, "value", float("nan")))
    except Exception as exc:
        result.warnings.append(f"Could not establish base estimate: {exc}")
        return result

    if not is_measured(observed):
        # Not published as base_estimate: a NaN there reads as an estimate in
        # every consumer that formats it, and it also makes the envelope
        # json.dumps writes non-compliant JSON.
        result.warnings.append(
            f"Could not establish base estimate: the estimator returned {observed!r}, "
            f"which is not a finite number, so there is nothing to refute and "
            f"robustness_score is unavailable."
        )
        return result
    result.base_estimate = observed

    passes = 0
    attempted = 0
    for label, method in _REFUTERS:
        # Starts at None, not False. READINESS-6: the crash path below and the
        # unreadable-result path both leave this at its initial value, and both
        # are already excluded from the robustness_score denominator on the
        # stated grounds that a refuter which gave no evidence is not evidence
        # against the estimate. `passed=False` said the opposite to every
        # consumer reading the outcome itself.
        out = RefutationOutcome(test_name=label, passed=None)
        try:
            ref = model.refute_estimate(estimand, base, method_name=method)
            out.observed_effect = observed

            # READINESS-6, 2026-09-10. `refute_estimate` returns a LIST for some
            # refuters, one CausalRefutation per simulation, and a list has no
            # `new_effect`, so `getattr(ref, "new_effect", nan)` returned the
            # DEFAULT. The refuter ran fine and its answer was discarded.
            #
            # Measured that day on 600 rows, dowhy 0.14, the dummy-outcome
            # refuter: the list held one CausalRefutation with
            # new_effect=0.00124 and p_value=0.94, which PASSES this suite's own
            # rule against an observed effect of -0.019. It was recorded
            # passed=False, and because `attempted` had already been
            # incremented, that permanent failure sat in the denominator:
            # robustness_score 0.75 where the three readable refuters all
            # passed. This test could never have passed, on any data.
            #
            # A refuter whose result cannot be read is now treated exactly like
            # one that CRASHED, which this function already handles correctly:
            # unavailable, disclosed, and out of the denominator. Counting it as
            # a failed refutation would be evidence against the estimate that
            # nobody gathered.
            candidates = list(ref) if isinstance(ref, (list, tuple)) else [ref]
            effects = []
            p_values = []
            for item in candidates:
                raw = getattr(item, "new_effect", None)
                try:
                    value = float(raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    effects.append(value)
                outcome_dict = getattr(item, "refutation_result", None)
                if isinstance(outcome_dict, dict) and "p_value" in outcome_dict:
                    try:
                        p_values.append(float(outcome_dict["p_value"]))
                    except (TypeError, ValueError):
                        pass

            if not effects:
                out.notes.append(
                    f"Ran, but no comparable effect could be read from its result "
                    f"(type {type(ref).__name__})."
                )
                out.notes.append(
                    "Unavailable: excluded from the robustness_score denominator. "
                    "This is NOT a failed refutation; nothing was measured."
                )
                result.outcomes.append(out)
                continue

            refuted = sum(effects) / len(effects)
            out.refuted_effect = refuted
            out.p_value = p_values[0] if p_values else None
            if len(effects) > 1:
                out.notes.append(
                    f"Averaged over {len(effects)} simulations returned by this refuter."
                )

            out.passed = _passed(label, observed, refuted)
            if out.passed is None:
                # READINESS-6: the refuter RAN and produced a readable number,
                # but its tolerance is wider than the effect under test, so no
                # outcome it could have produced would have failed. Same
                # disposition as a crash or an unreadable result: disclosed, and
                # out of the denominator. `attempted` is deliberately NOT
                # incremented, because incrementing it before the verdict is
                # what let a test that could never pass sit in the denominator.
                tolerance = _tolerance(label, observed)
                out.notes.append(
                    f"Could not check: this test's tolerance ({tolerance:.4g}) is wider "
                    f"than the effect under test ({abs(observed):.4g}), so no outcome "
                    f"would have failed it. Not a pass; nothing was graded."
                )
                out.notes.append("Unavailable: excluded from the robustness_score denominator.")
                result.outcomes.append(out)
                continue

            attempted += 1
            if out.passed:
                passes += 1
        except Exception as exc:
            # A refuter that CRASHES gave no evidence against the estimate;
            # counting it as a failed refutation would penalize the score
            # for an environment/library problem. It is disclosed as
            # unavailable and excluded from the denominator instead.
            out.notes.append(f"Refuter raised: {exc}")
            out.notes.append(
                "Unavailable: this refuter could not run, so it is "
                "excluded from the robustness_score denominator (it is "
                "not evidence against the estimate)."
            )
        result.outcomes.append(out)

    if attempted:
        result.robustness_score = passes / attempted
        if attempted < len(result.outcomes):
            result.warnings.append(
                f"{len(result.outcomes) - attempted} of "
                f"{len(result.outcomes)} refuters could not run and were "
                "excluded from the robustness_score denominator."
            )
    else:
        result.warnings.append("No refuter could run; robustness_score is unavailable.")

    return result
