"""Pulse -- quick assessment operation.

Pulse is the non-expert entry to the platform. The browser dispatches a
single ``vfairness_pulse_run`` task with the captured inputs + a signed URL
to the encrypted artifact. The consumer downloads, decrypts, runs this
pipeline, and returns the canonical PulseResult shape the React UI renders.

Public entry points are exposed for the consumer to register:

    from vfairness.operations.pulse import handle_pulse_run

See ``CONSUMER_REGISTRATION.md`` for the full payload contract and the two
registration patterns (subprocess CLI / in-process bridge).
"""

from .agent_probe import agent_probe_pulse
from .causal_skeleton import build_causal_skeleton, domain_dag_template
from .llm_probe import llm_probe_pulse
from .recommend import recommend_fairness_definition, recommend_interventions
from .task_handlers import handle_pulse_run, main

# G-10: the package-level run_pulse is the trace-ingestion front door.
# It flattens raw OTel/Langfuse span exports into the per-episode trace
# table BEFORE the orchestrator's column detection, then delegates to
# orchestrator.run_pulse unchanged (which stays the single source of
# truth); it is a transparent pass-through for every other frame.
from .traces import ingest_and_run_pulse as run_pulse
from .vision_probe import vision_probe_pulse

__all__ = [
    "handle_pulse_run",
    "main",
    "run_pulse",
    "recommend_fairness_definition",
    "recommend_interventions",
    "build_causal_skeleton",
    "domain_dag_template",
    "llm_probe_pulse",
    "agent_probe_pulse",
    "vision_probe_pulse",
]
