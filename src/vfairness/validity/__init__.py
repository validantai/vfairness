"""The validity (groundedness) axis for generative systems.

A fail-closed groundedness scorer (VA-10 interim judge, VA-21 owned detector) and
its VG-* aggregation, dispatched to the platform via the
``vfairness_validity_run`` task type. See docs/audits/groundedness-scorer-build-plan.md.
"""

from .aggregate import EMITTABLE_METRICS, aggregate_validity
from .groundedness import (
    GOLD_REFUSAL_NOTE,
    VG_ANSWER_CORRECTNESS,
    VG_CITATION_ACCURACY,
    VG_CONTEXT_PRECISION,
    VG_CONTEXT_RECALL,
    VG_FAITHFULNESS,
    VG_GROUNDEDNESS,
    VG_HALLUCINATION_RATE,
    GoldUnsupportedWarning,
    GroundednessJudge,
    GroundednessResult,
    GroundednessScorer,
    ValidityScorerUnavailableWarning,
    groundedness_scorer_status,
)
from .judge import PROMPT_VERSION, LlmGroundednessJudge

__all__ = [
    "GroundednessScorer",
    "GroundednessResult",
    "GroundednessJudge",
    "ValidityScorerUnavailableWarning",
    # A refusal a caller cannot import is a refusal they cannot act on: this is
    # the warning score() raises for the unimplemented `gold` parameter, and the
    # note it stamps on every result so the refusal survives into serialized
    # output. Exported so `simplefilter("error", GoldUnsupportedWarning)` is
    # reachable from the package root, as its sibling above already is.
    "GoldUnsupportedWarning",
    "GOLD_REFUSAL_NOTE",
    "groundedness_scorer_status",
    "aggregate_validity",
    "EMITTABLE_METRICS",
    "LlmGroundednessJudge",
    "PROMPT_VERSION",
    "VG_GROUNDEDNESS",
    "VG_FAITHFULNESS",
    "VG_CONTEXT_PRECISION",
    "VG_CONTEXT_RECALL",
    "VG_HALLUCINATION_RATE",
    "VG_CITATION_ACCURACY",
    "VG_ANSWER_CORRECTNESS",
]
