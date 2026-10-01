"""
vfairness.multi_agent: Multi-Agent System Fairness Testing

Tests for fairness failure modes that only manifest when multiple
agents interact:
- Compositionality of bias (component vs system bias)
- Emergent amplification (system bias exceeds any component)
- Groupthink / echo-chamber detection
- Adversarial collusion (bias amplified by inter-agent interaction)
- Role-based delegation routing (orchestrator routes by demographic)
- Turn-by-turn negotiation fairness drift

A framework-agnostic ``MultiAgentRunHarness`` is provided to capture
traces from real agent runs (autogen, crewai, langgraph, raw
function-calling) and adapt them to the analyzer input shapes.
"""

from .collusion import AdversarialCollusionDetector, CollusionResult
from .compositionality import CompositionalityAnalyzer, CompositionalityResult
from .delegation import DelegationResult, DelegationRoutingAuditor
from .emergent import EmergentBiasDetector, EmergentBiasResult
from .groupthink import GroupthinkDetector, GroupthinkResult
from .harness import HarnessTrace, MultiAgentRunHarness
from .negotiation import NegotiationFairnessTracker, NegotiationResult

__all__ = [
    "AdversarialCollusionDetector",
    "CollusionResult",
    "CompositionalityAnalyzer",
    "CompositionalityResult",
    "DelegationResult",
    "DelegationRoutingAuditor",
    "EmergentBiasDetector",
    "EmergentBiasResult",
    "GroupthinkDetector",
    "GroupthinkResult",
    "HarnessTrace",
    "MultiAgentRunHarness",
    "NegotiationFairnessTracker",
    "NegotiationResult",
]
