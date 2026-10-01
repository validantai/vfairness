"""
vfairness.operations.causal - Causal Inference Engine (DoWhy-backed)

Phased capability surface:

Phase 1 (identification):
    - identify_paths(gml, treatments, outcomes) -> IdentificationResult
      Backdoor / instrument / frontdoor adjustment sets per (treatment, outcome).

Phase 2 (effect decomposition + robustness):
    - decompose_mediation(gml, data, treatments, outcomes, mediators)
        -> MediationResult
      Splits total effect into NDE (direct) and NIE (indirect via mediator).
    - run_refutation_suite(gml, data, treatment, outcome) -> RefutationResult
      Placebo / random common cause / data subset / dummy outcome.

Phase 3 (individual-level + attribution):
    - compute_counterfactual(gml, data, treatment, outcome, factual,
                             intervention_value)
        -> CounterfactualResult
      What would Y have been for this individual under the alternative treatment.
    - attribute_distribution_change(gml, baseline, current, outcome)
        -> AttributionResult
      Ranks upstream nodes by share of explained drift in the outcome.

DoWhy is an OPTIONAL dependency. ImportError is deferred until a function is
actually called, so vfairness still imports in environments that lack dowhy.
"""

from .attribute import AttributionResult, NodeContribution, attribute_distribution_change
from .counterfactual import CounterfactualResult, compute_counterfactual
from .identify import IdentificationResult, PathIdentification, identify_paths
from .mediate import MediationDecomposition, MediationResult, decompose_mediation
from .refute import RefutationOutcome, RefutationResult, run_refutation_suite
from .task_handlers import (
    handle_attribute,
    handle_counterfactual,
    handle_identify,
    handle_mediate,
    handle_refute,
)

__all__ = [
    # Task handlers (consumer entry points)
    "handle_identify",
    "handle_mediate",
    "handle_refute",
    "handle_counterfactual",
    "handle_attribute",
    # Phase 1
    "IdentificationResult",
    "PathIdentification",
    "identify_paths",
    # Phase 2
    "MediationDecomposition",
    "MediationResult",
    "decompose_mediation",
    "RefutationOutcome",
    "RefutationResult",
    "run_refutation_suite",
    # Phase 3
    "CounterfactualResult",
    "compute_counterfactual",
    "AttributionResult",
    "NodeContribution",
    "attribute_distribution_change",
]
